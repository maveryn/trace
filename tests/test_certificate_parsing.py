"""Strict parsing, envelope/element failure semantics, F, A guards and scorer robustness.

The scorer is exercised end to end through a small synthetic adapter (``ToySumAdapter``) so
these tests do not depend on the four task adapters (DESIGN 2.1, 2.4, 5, 9).
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Mapping

import pytest

from trace_tasks.core.reward_contracts import resolve_reward_contract
from trace_tasks.core.reward_scoring import score_trace_response
from trace_tasks.experimental.certificates import registry
from trace_tasks.experimental.certificates.adapters.base import Adapter, PublicContext
from trace_tasks.experimental.certificates.checks import (
    bind_elements,
    bound_exactly_once,
    counted_truth,
    gated_check,
    indices_bijective,
    local_check,
    partial_credit,
    read_int,
)
from trace_tasks.experimental.certificates.identity import CandidateSet
from trace_tasks.experimental.certificates.parsing import (
    INDEX,
    INT,
    JSONObject,
    StrictJSONError,
    duplicate_key_paths,
    find_brace_spans,
    locate_scored_object,
    strict_loads,
)
from trace_tasks.experimental.certificates.scoring import reward_arms, score_certificate_response
from trace_tasks.experimental.certificates.types import (
    CertificateExtension,
    CertificateVersionError,
    Corruption,
    ExtensionMismatchError,
    certificate_extension_id,
    require_rfc8785,
)

IMAGE_SIZE = (640, 480)
BLOCKS = [
    {"bbox": [100.0, 280.0, 158.0, 332.0], "value": 3, "selected": True},
    {"bbox": [170.0, 280.0, 228.0, 332.0], "value": 5, "selected": True},
    {"bbox": [240.0, 280.0, 298.0, 332.0], "value": 2, "selected": True},
    {"bbox": [400.0, 280.0, 458.0, 332.0], "value": 7, "selected": False},
]
TOTAL = 10
TOY_TASK = "task_toy__blocks__sum"


class ToySumAdapter(Adapter):
    """Sum of the values printed on the selected blocks (bbox_set annotation)."""

    _allow_foreign_module = True
    adapter_id = "toy_sum.blocks"
    kind = "toy_sum"
    version = "1.0.0"
    schema_id = "cert.toy_sum.v1"
    verifier_version = "1.0.0"
    supported = {TOY_TASK: ("single",)}
    annotation_type = "bbox_set"
    list_key = "terms"
    list_cap = 8
    cert_fields = {"total": INT}
    element_fields = {"annotation_index": INDEX, "value": INT}
    m_families = ("toy.items.truth", "toy.value.truth", "toy.total.truth")
    family_stages = {"toy.items.truth": "grounding", "toy.value.truth": "operand", "toy.total.truth": "computation"}
    local_check_ids = ("toy.ref.local", "toy.answer.local")

    def build_oracle(self, record):  # pragma: no cover - not used by these tests
        return {"blocks": BLOCKS, "total": TOTAL}

    def split_cell(self, record):  # pragma: no cover
        return "toy"

    def semantic_key(self, record):  # pragma: no cover
        return "toy"

    def sft_certificate(self, oracle, rng_seed):
        selected = [b for b in oracle["blocks"] if b["selected"]]
        terms = [{"annotation_index": i, "value": b["value"]} for i, b in enumerate(selected)]
        return [b["bbox"] for b in selected], {"kind": self.kind, "terms": terms, "total": oracle["total"]}, oracle["total"]

    def candidates(self, oracle):
        blocks = oracle["blocks"]
        boxes = tuple(tuple(b["bbox"]) for b in blocks)
        return {"": CandidateSet(boxes, tuple(i for i, b in enumerate(blocks) if b["selected"]))}

    def checks(self, cert, annotation, oracle, ctx: PublicContext):
        candidates = self.candidates(oracle)
        resolutions = self.resolve_role(annotation, "", oracle, ctx, candidates)
        bindings = bind_elements(cert.elements, resolutions)
        selected = candidates[""].selected
        once = bound_exactly_once(bindings, selected)
        n, k = cert.n_claimed, len(selected)
        items = counted_truth("toy.items.truth", len(once), n, k)
        hits = sum(1 for target, pos in once.items() if cert.elements[pos].get("value") == oracle["blocks"][target]["value"])
        value = counted_truth("toy.value.truth", hits, n, k)
        total_claim = cert.fields.get("total")
        values = [element.get("value") for element in cert.elements if element.valid]
        total_ok = total_claim is not None and len(values) == n and total_claim == sum(values)
        total = gated_check("toy.total.truth", [items, value], total_ok)
        return [
            items,
            value,
            total,
            local_check("toy.ref.local", indices_bijective(cert.elements, annotation.size())),
            local_check("toy.answer.local", cert.answer is not None and cert.answer == total_claim),
        ]

    def corruptions(self, oracle, sft_obj, rng_seed):
        return [Corruption("noop", dict(sft_obj), {"M": "pass"})]

    def certificate_line(self):
        return 'Set "certificate" to {"kind": "toy_sum", "terms": [...], "total": t}.'

    def example_object(self):
        return {"annotation": [[1, 2, 3, 4]], "certificate": {"kind": "toy_sum", "terms": [], "total": 0}, "answer": 0}


ADAPTER = ToySumAdapter()


def make_extension(adapter: Adapter = ADAPTER) -> CertificateExtension:
    base = {
        "instance_id": "blake3:" + "1" * 64,
        "trace_record_hash": "blake3:" + "2" * 64,
        "task": TOY_TASK,
        "query_id": "single",
        "instance_seed": 17,
    }
    identity = adapter.identity()
    ext_id = certificate_extension_id(
        instance_id=base["instance_id"],
        trace_record_hash=base["trace_record_hash"],
        adapter_id=identity["id"],
        adapter_version=identity["version"],
        schema_id=identity["schema_id"],
    )
    selected = [b["bbox"] for b in BLOCKS if b["selected"]]
    return CertificateExtension(
        certificate_extension_id=ext_id,
        base=base,
        adapter=identity,
        labels={
            "answer_gt": {"type": "integer", "value": TOTAL},
            "annotation_gt": {"type": "bbox_set", "value": selected},
            "reward_contract": resolve_reward_contract(answer_type="integer", annotation_type="bbox_set").to_dict(),
        },
        public={"image_size": list(IMAGE_SIZE)},
        oracle={"blocks": BLOCKS, "total": TOTAL},
    )


EXT = make_extension()
BASE = {"instance_id": EXT.instance_id, "cert_extension_id": EXT.certificate_extension_id}


def good_obj() -> Dict[str, Any]:
    annotation, certificate, answer = ADAPTER.sft_certificate(EXT.oracle, 0)
    return {"annotation": annotation, "certificate": certificate, "answer": answer}


def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


def score(response: str, **kwargs: Any):
    frame = kwargs.pop("coord_frame", "px_source")
    return score_certificate_response(response, BASE, EXT, coord_frame=frame, adapter=ADAPTER, **kwargs)


def check(result, check_id: str):
    return next(c for c in result.checks if c.check_id == check_id)


def assert_bounded(result) -> None:
    for name, value in (("A", result.answer), ("G", result.grounding), ("M", result.semantics), ("F", result.format)):
        assert math.isfinite(value) and 0.0 <= value <= 1.0, name
    for arm, value in result.overall.items():
        assert math.isfinite(value) and 0.0 <= value <= 1.0, arm
    assert result.joint in (0.0, 1.0)


# ------------------------------------------------------------------------------------------
# strict JSON


def test_rfc8785_is_in_use() -> None:
    require_rfc8785()


def test_strict_loads_marks_duplicate_keys_and_keeps_last() -> None:
    value = strict_loads('{"a": 1, "b": {"c": 1, "c": 2}, "a": 3}')
    assert isinstance(value, JSONObject)
    assert value["a"] == 3 and value.duplicate_keys == ("a",)
    assert value["b"]["c"] == 2
    assert duplicate_key_paths(value) == ["$.a", "$.b.c"]


@pytest.mark.parametrize(
    "text",
    ['{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}', '{"a": 1e999}', '{"a": -1e400}'],
)
def test_strict_loads_rejects_non_finite(text: str) -> None:
    with pytest.raises(StrictJSONError) as info:
        strict_loads(text)
    assert info.value.code == "non_finite"


@pytest.mark.parametrize(
    "text",
    ["{'a': 1}", '{"a": True}', '{"a": None}', '{"a": 1,}', "{a: 1}", '{"a": [1, 2,]}', "", "[1, 2"],
)
def test_strict_loads_rejects_python_literals_and_bad_json(text: str) -> None:
    with pytest.raises(StrictJSONError) as info:
        strict_loads(text)
    assert info.value.code == "invalid_json"


def test_strict_loads_size_cap() -> None:
    small = '{"a": "' + "x" * 100 + '"}'
    assert strict_loads(small)["a"] == "x" * 100
    big = '{"a": "' + "x" * (16 * 1024) + '"}'
    with pytest.raises(StrictJSONError) as info:
        strict_loads(big)
    assert info.value.code == "budget"
    # multi-byte characters count in bytes, not characters
    wide = '{"a": "' + "é" * (9 * 1024) + '"}'
    with pytest.raises(StrictJSONError):
        strict_loads(wide)


def test_read_int_rejects_bool_and_flags_integral_float() -> None:
    assert read_int(True).value is None and read_int(False).value is None
    assert read_int(3).value == 3 and not read_int(3).integral_float
    assert read_int(15.0).value == 15 and read_int(15.0).integral_float
    for bad in (15.5, "15", None, float("nan"), float("inf"), [1]):
        assert read_int(bad).value is None


def test_partial_credit_denominator() -> None:
    assert partial_credit(0, 0, 0) == 1.0
    assert partial_credit(3, 3, 3) == 1.0
    assert partial_credit(3, 5, 3) == pytest.approx(0.6)
    assert partial_credit(2, 2, 3) == pytest.approx(2 / 3)
    assert partial_credit(0, 4, 0) == 0.0


def test_brace_spans_match_legacy_scanner() -> None:
    from trace_tasks.core.reward_scoring import _extract_brace_object_spans

    texts = ['a {"x": "}"} b {"y": {"z": 1}} c', '{"a": "\\"}"}', "}{", '{"unterminated": "x}', "plain"]
    for text in texts:
        assert find_brace_spans(text) == [(s, e) for s, e, _ in _extract_brace_object_spans(text)]


# ------------------------------------------------------------------------------------------
# location of the final and scored objects


def test_locator_final_object_and_trailing_prose() -> None:
    text = dumps(good_obj())
    located = locate_scored_object("Let me think.\n" + text)
    assert located.scored is not None and located.scored_is_final and located.final_found
    trailing = locate_scored_object(text + "\nThat is the answer.")
    assert trailing.scored == located.scored and not trailing.scored_is_final
    fenced = locate_scored_object("Reasoning\n```json\n" + text + "\n```")
    assert fenced.scored_is_final


def test_locator_never_scores_a_draft_when_a_final_object_exists() -> None:
    """Review CONF-08 (DESIGN 2.1): a final object is the claim; when it lacks a certificate or
    is invalid, the response has an envelope failure and an earlier draft is not scored."""
    draft = good_obj()
    draft["answer"] = 99
    text = "draft: " + dumps(draft) + "\nfinal: " + dumps({"answer": 10})
    located = locate_scored_object(text)
    assert located.scored is None and located.final_found
    assert located.final_error == located.envelope_error == "missing_certificate"
    final = dumps(good_obj())
    nan_final = "draft: " + dumps(draft) + "\nfinal: " + final[: final.rindex(":") + 1] + "NaN}"
    located = locate_scored_object(nan_final)
    assert located.scored is None and located.envelope_error == "non_finite"
    result = score(text)
    assert result.semantics == 0.0 and result.diagnostics["envelope_error"] == "missing_certificate"
    # no final object (trailing prose): the last certificate object is scored, F = 0 only
    prose = "draft: " + dumps(draft) + "\nThat is my answer."
    located = locate_scored_object(prose)
    assert located.scored is not None and located.scored["answer"] == 99 and not located.scored_is_final


def test_locator_envelope_errors() -> None:
    assert locate_scored_object("no json").envelope_error == "no_object"
    assert locate_scored_object("").envelope_error == "no_object"
    nan = dumps(good_obj()).replace('"total":10', '"total":NaN')
    assert locate_scored_object(nan).envelope_error == "non_finite"
    literal = str(good_obj())
    assert locate_scored_object(literal).envelope_error == "invalid_json"
    huge = good_obj()
    huge["certificate"]["note"] = "x" * 20000
    assert locate_scored_object(dumps(huge)).envelope_error == "budget"


# ------------------------------------------------------------------------------------------
# end-to-end F / M / G semantics


def test_oracle_certificate_scores_one() -> None:
    result = score(dumps(good_obj()))
    assert (result.answer, result.grounding, result.semantics, result.format) == (1.0, 1.0, 1.0, 1.0)
    assert result.joint == 1.0 and result.error_class == "correct"
    assert result.diagnostics["G_id"] == 1.0
    assert check(result, "toy.answer.local").score == 1.0
    assert_bounded(result)


def test_trailing_prose_costs_f_only() -> None:
    clean = score(dumps(good_obj()))
    trailing = score(dumps(good_obj()) + "\nI hope this helps!")
    assert trailing.format == 0.0 and "not_final" in trailing.diagnostics["format_reasons"]
    assert (trailing.answer, trailing.grounding, trailing.semantics) == (clean.answer, clean.grounding, clean.semantics)
    assert [c.to_dict() for c in trailing.checks] == [c.to_dict() for c in clean.checks]


def test_key_order_is_never_scored() -> None:
    obj = good_obj()
    certificate = dict(reversed(list(obj["certificate"].items())))
    reordered = {"answer": obj["answer"], "certificate": certificate, "annotation": obj["annotation"]}
    a, b = score(dumps(obj)), score(dumps(reordered))
    assert (a.answer, a.grounding, a.semantics, a.format) == (b.answer, b.grounding, b.semantics, b.format)


def test_top_level_duplicate_key_is_envelope_failure() -> None:
    text = dumps(good_obj())[:-1] + ',"certificate":{"kind":"toy_sum","terms":[],"total":0}}'
    result = score(text)
    assert result.semantics == 0.0 and result.format == 0.0
    assert result.error_class == "envelope" and result.diagnostics["envelope_error"] == "duplicate_key"
    assert all(not c.valid_input for c in result.checks)


def test_conflicting_duplicate_answer_scores_zero_but_logs_legacy() -> None:
    text = dumps(good_obj())[:-1] + ',"answer":10}'
    same = score(text)
    assert same.answer == 1.0  # duplicate with the same value is recoverable
    conflict = score(dumps(good_obj())[:-1].replace('"answer":10', '"answer":11') + ',"answer":10}')
    assert conflict.answer == 0.0 and conflict.diagnostics["A_legacy"] == 1.0
    assert conflict.diagnostics["answer_guard"] == "duplicate_answer"


def test_element_duplicate_key_is_element_failure() -> None:
    text = dumps(good_obj()).replace('{"annotation_index":0,"value":3}', '{"annotation_index":0,"value":3,"value":3}')
    result = score(text)
    assert result.format == 0.0 and "duplicate_key" in result.diagnostics["format_reasons"]
    assert result.diagnostics["envelope_error"] is None
    assert check(result, "toy.items.truth").score == pytest.approx(2 / 3)
    assert check(result, "toy.total.truth").score == 0.0


def test_bool_answer_and_bool_values() -> None:
    obj = good_obj()
    obj["answer"] = True
    ext = make_extension()
    ext.labels["answer_gt"] = {"type": "integer", "value": 1}
    result = score_certificate_response(dumps(obj), BASE, ext, coord_frame="px_source", adapter=ADAPTER)
    assert result.answer == 0.0 and result.diagnostics["A_legacy"] == 1.0
    assert result.diagnostics["answer_guard"] == "bool_answer" and result.format == 0.0
    obj = good_obj()
    obj["certificate"]["terms"][0]["value"] = True
    result = score(dumps(obj))
    assert check(result, "toy.value.truth").score == pytest.approx(2 / 3)
    assert result.format == 0.0


def test_nan_and_python_literal_void_m_but_not_legacy_g() -> None:
    nan = dumps(good_obj()).replace('"total":10', '"total":NaN')
    for text in (nan, str(good_obj())):
        result = score(text)
        legacy = score_trace_response(
            response=text,
            answer_gt=EXT.labels["answer_gt"],
            annotation_gt=EXT.labels["annotation_gt"],
            reward_contract=EXT.labels["reward_contract"],
        )
        assert result.semantics == 0.0 and result.format == 0.0 and result.error_class == "envelope"
        assert result.grounding == legacy["annotation_reward"]
        assert result.diagnostics["annotation_source"] == "legacy_extraction"


def test_integral_float_is_scored_by_value_but_f_zero() -> None:
    obj = good_obj()
    obj["certificate"]["total"] = 10.0
    obj["certificate"]["terms"][1]["value"] = 5.0
    result = score(dumps(obj))
    assert result.semantics == 1.0 and result.format == 0.0
    assert "integral_float" in result.diagnostics["format_reasons"]
    obj = good_obj()
    obj["answer"] = 10.0
    result = score(dumps(obj))
    assert result.answer == 1.0 and result.format == 0.0 and "answer_type" in result.diagnostics["format_reasons"]


def test_unknown_certificate_key_costs_f_only() -> None:
    obj = good_obj()
    obj["certificate"]["note"] = "extra"
    result = score(dumps(obj))
    assert result.semantics == 1.0 and result.format == 0.0 and "extra_key" in result.diagnostics["format_reasons"]


def test_extra_top_level_key_costs_f_only() -> None:
    obj = good_obj()
    obj["reasoning"] = "because"
    result = score(dumps(obj))
    assert result.semantics == 1.0 and result.format == 0.0 and "top_level_keys" in result.diagnostics["format_reasons"]


@pytest.mark.parametrize("index", [-1, 3, 100])
def test_out_of_range_index_never_binds(index: int) -> None:
    obj = good_obj()
    obj["certificate"]["terms"][2]["annotation_index"] = index
    result = score(dumps(obj))
    assert check(result, "toy.items.truth").score == pytest.approx(2 / 3)
    assert result.format == 0.0 and "element_invalid" in result.diagnostics["format_reasons"]


def test_missing_and_extra_element_keys() -> None:
    obj = good_obj()
    del obj["certificate"]["terms"][0]["value"]
    obj["certificate"]["terms"][1]["side"] = "left"
    result = score(dumps(obj))
    assert check(result, "toy.items.truth").score == pytest.approx(1 / 3)
    assert result.format == 0.0


def test_wrong_kind_missing_list_and_cap_are_envelope_failures() -> None:
    cases = {
        "wrong_kind": lambda o: o["certificate"].update(kind="weighted_sum"),
        "missing_list": lambda o: o["certificate"].pop("terms"),
        "budget": lambda o: o["certificate"].update(terms=o["certificate"]["terms"] * 3),
        "certificate_not_object": lambda o: o.update(certificate=[1, 2]),
    }
    for code, mutate in cases.items():
        obj = good_obj()
        mutate(obj)
        result = score(dumps(obj))
        assert result.diagnostics["envelope_error"] == code, code
        assert result.semantics == 0.0 and result.format == 0.0 and result.error_class == "envelope"
        assert all(not c.valid_input for c in result.checks)
        assert result.grounding == 1.0  # G reads the same located object


def test_missing_certificate_falls_back_to_legacy_g() -> None:
    obj = good_obj()
    del obj["certificate"]
    result = score(dumps(obj))
    assert result.diagnostics["envelope_error"] == "missing_certificate"
    assert result.grounding == 1.0 and result.answer == 1.0 and result.semantics == 0.0


def test_empty_claims_never_raise_credit() -> None:
    obj = good_obj()
    obj["certificate"]["terms"] = []
    obj["certificate"]["total"] = 0
    result = score(dumps(obj))
    assert check(result, "toy.items.truth").score == 0.0
    obj = good_obj()
    obj["certificate"]["terms"].append({"annotation_index": 0, "value": 3})
    result = score(dumps(obj))
    assert check(result, "toy.items.truth").score == pytest.approx(2 / 4)


def test_one_megabyte_inputs() -> None:
    text = dumps(good_obj())
    prose = ("lorem ipsum " * 90000)[: 1 << 20]
    result = score(prose + text)
    assert result.semantics == 1.0 and result.format == 1.0
    huge = good_obj()
    huge["certificate"]["note"] = "x" * (1 << 20)
    result = score(dumps(huge))
    assert result.diagnostics["envelope_error"] == "budget" and result.semantics == 0.0
    assert_bounded(score("{" * (1 << 20)))
    assert_bounded(score("[" * (1 << 20)))
    assert_bounded(score('{"a":' * 200000 + "1" + "}" * 200000))


def _mutations(rng: random.Random, base: str) -> str:
    choice = rng.randrange(16)
    if choice == 0:
        return base[: rng.randrange(len(base) + 1)]
    if choice == 1:
        position = rng.randrange(len(base) + 1)
        return base[:position] + rng.choice(['"', "{", "}", "[", "]", ",", ":", "\\", "\x00", " "]) + base[position:]
    if choice == 2:
        position = rng.randrange(len(base))
        return base[:position] + base[position + 1 :]
    if choice == 3:
        bad = rng.choice(["NaN", "Infinity", "true", "null", "1e999", "-0", "1.5", '"3"'])
        return base.replace(str(rng.choice([3, 5, 2, 10])), bad, 1)
    if choice == 4:
        return base[:-1] + rng.choice([',"answer":11}', ',"certificate":null}', ',"annotation":[]}', ',"x":1}'])
    if choice == 5:
        return base.replace('"', "'")
    if choice == 6:
        return base.replace("true", "True").replace("null", "None") + rng.choice(["", " ", "\n```"])
    if choice == 7:
        prefix = rng.choice(["Here: ", "```json\n", "<answer>", ""])
        return prefix + base + rng.choice(["", "\n```", "</answer>", " done", "}"])
    if choice == 8:
        return "".join(rng.choice('{}[]",:0123456789-.eE truefalsnul\\') for _ in range(rng.randrange(1, 200)))
    if choice == 9:
        bad = rng.choice(["-1", "1.0", "true", "999", '"1"', "null"])
        return base.replace('"annotation_index":1', '"annotation_index":' + bad)
    if choice == 10:
        return base.replace('"kind":"toy_sum"', '"kind":' + rng.choice(['"TOY_SUM"', "1", "null", '"path"', "[]"]))
    if choice == 11:
        return base.replace("[[", rng.choice(["[[[", "[{", "[", "[[true,"]), 1)
    if choice == 12:
        depth = rng.randrange(1, 3000)
        return '{"annotation":' + "[" * depth + "]" * depth + "}"
    if choice == 13:
        return base * rng.randrange(2, 5)
    if choice == 14:
        return base.replace('"terms":[', '"terms":[' + '{"annotation_index":0,"value":3},' * rng.randrange(1, 12), 1)
    return rng.choice(["", " ", "\n", "{}", "[]", "null", "10", "true", '"10"', "{" * 50, '{"certificate":{}}'])


@pytest.mark.filterwarnings("ignore::SyntaxWarning")
def test_fuzz_malformed_strings_never_raise() -> None:
    rng = random.Random(20261018)
    base = dumps(good_obj())
    seen = set()
    for _ in range(1000):
        text = _mutations(rng, base)
        if rng.random() < 0.3:
            text = _mutations(rng, text) if text else text
        seen.add(text)
        result = score(text)
        assert_bounded(result)
        assert result.coverage_status == "supported"
        assert result.error_class in ("envelope", "grounding", "operand", "computation", "answer_inconsistent", "correct")
        try:
            legacy = score_trace_response(
                response=text,
                answer_gt=EXT.labels["answer_gt"],
                annotation_gt=EXT.labels["annotation_gt"],
                reward_contract=EXT.labels["reward_contract"],
            )
        except RecursionError:  # the legacy canonicalizer cannot score pathological nesting
            assert result.diagnostics["answer_guard"] == "unscorable_answer"
            continue
        assert result.diagnostics["A_legacy"] == legacy["accuracy"]
        if result.diagnostics["answer_guard"] is None:
            assert result.answer == legacy["accuracy"]
        if result.diagnostics["annotation_source"] == "legacy_extraction":
            assert result.grounding == legacy["annotation_reward"]
    assert len(seen) > 500


def test_scores_are_deterministic_and_wall_clock_free() -> None:
    texts = [dumps(good_obj()), dumps(good_obj()) + " ok", str(good_obj()), "nothing"]
    first = [score(t).to_dict() for t in texts]
    second = [score(t).to_dict() for t in texts]
    for a, b in zip(first, second):
        a["diagnostics"].pop("verifier_ms"), b["diagnostics"].pop("verifier_ms")
        a["diagnostics"].pop("slow_row"), b["diagnostics"].pop("slow_row")
        assert a == b


# ------------------------------------------------------------------------------------------
# frames, arms, error classes, extension integrity


def test_rel1000_frame_scores_like_px_source() -> None:
    obj = good_obj()
    width, height = IMAGE_SIZE
    obj["annotation"] = [
        [x0 * 1000 / width, y0 * 1000 / height, x1 * 1000 / width, y1 * 1000 / height]
        for x0, y0, x1, y1 in obj["annotation"]
    ]
    result = score(dumps(obj), coord_frame="rel1000")
    assert result.semantics == 1.0 and result.format == 1.0
    assert result.grounding == pytest.approx(1.0, abs=1e-6)
    wrong = score(dumps(obj), coord_frame="px_source")
    assert wrong.semantics < 1.0 and wrong.grounding < 0.5


def test_shown_frame_diagnostics() -> None:
    result = score(dumps(good_obj()), shown_frame_diagnostics=True)
    assert result.diagnostics["shown_size"] == [644, 476]
    assert 0.0 <= result.diagnostics["G_shown"] <= 1.0 and result.diagnostics["M_shown"] == 1.0
    assert "G_shown" not in score(dumps(good_obj())).diagnostics


def test_reward_arm_formulas() -> None:
    arms = reward_arms(1.0, 0.5, 0.25, 1.0)
    assert arms["A"] == pytest.approx(0.95 * 1.0 + 0.05)
    assert arms["G"] == pytest.approx(0.95 * (0.8 + 0.2 * 0.5) + 0.05)
    assert arms["P"] == pytest.approx(0.95 * (0.8 + 0.2 * 0.375) + 0.05)
    assert arms["P_gated"] == pytest.approx(0.95 * (0.8 + 0.2 * 0.375) + 0.05)
    zero = reward_arms(0.0, 1.0, 1.0, 1.0)
    assert zero["P"] == pytest.approx(0.95 * 0.2 + 0.05) and zero["P_gated"] == pytest.approx(0.05)
    assert zero["G_gated"] == pytest.approx(0.05) and zero["A"] == pytest.approx(0.05)
    result = score(dumps(good_obj()))
    assert result.overall == reward_arms(1.0, 1.0, 1.0, 1.0)


def test_error_class_order() -> None:
    obj = good_obj()
    obj["certificate"]["terms"][0]["value"] = 4  # operand misread, arithmetic consistent
    obj["certificate"]["total"] = 11
    obj["answer"] = 11
    result = score(dumps(obj))
    assert result.error_class == "operand" and check(result, "toy.total.truth").error_code == "gate_closed"
    obj = good_obj()
    obj["certificate"]["total"] = 12
    assert score(dumps(obj)).error_class == "computation"
    obj = good_obj()
    obj["answer"] = 12
    assert score(dumps(obj)).error_class == "answer_inconsistent"
    obj = good_obj()
    obj["annotation"][0] = [400.0, 280.0, 458.0, 332.0]  # distractor block
    assert score(dumps(obj)).error_class == "grounding"
    assert score("no json at all").error_class == "envelope"


def test_extension_integrity_raises() -> None:
    with pytest.raises(CertificateVersionError):
        score_certificate_response("{}", BASE, None, coord_frame="px_source", adapter=ADAPTER)  # type: ignore[arg-type]
    with pytest.raises(ExtensionMismatchError):
        other = {"instance_id": "blake3:" + "9" * 64}
        score_certificate_response("{}", other, EXT, coord_frame="px_source", adapter=ADAPTER)
    with pytest.raises(ExtensionMismatchError):
        wrong_id = {"instance_id": EXT.instance_id, "cert_extension_id": "blake3:x"}
        score_certificate_response("{}", wrong_id, EXT, coord_frame="px_source", adapter=ADAPTER)
    tampered = CertificateExtension.from_dict(EXT.to_dict())
    tampered.base["trace_record_hash"] = "blake3:" + "3" * 64
    with pytest.raises(ExtensionMismatchError):
        score_certificate_response("{}", BASE, tampered, coord_frame="px_source", adapter=ADAPTER)
    stale = CertificateExtension.from_dict(EXT.to_dict())
    stale.adapter["verifier_version"] = "0.9.0"
    with pytest.raises(CertificateVersionError):
        score_certificate_response("{}", BASE, stale, coord_frame="px_source", adapter=ADAPTER)
    with pytest.raises(CertificateVersionError):
        CertificateExtension.from_dict({"certificate_extension_id": "x", "base": {}})
    with pytest.raises(ValueError):
        score_certificate_response("{}", BASE, EXT, coord_frame="pixels", adapter=ADAPTER)


def test_registry_resolves_by_extension_and_reports_unsupported() -> None:
    registry.register_adapter(ADAPTER)
    try:
        assert registry.resolve(TOY_TASK, "single") is ADAPTER
        assert registry.adapter_for_extension(EXT.to_dict()) is ADAPTER
        result = score_certificate_response(dumps(good_obj()), BASE, EXT.to_dict(), coord_frame="px_source")
        assert result.semantics == 1.0
    finally:
        registry.unregister_adapter(ADAPTER.adapter_id)
    unsupported = registry.resolve("task_charts__bar__nothing", "single")
    assert isinstance(unsupported, registry.Unsupported) and unsupported.reason == "no_adapter"
    assert registry.resolve("task_charts__table__threshold_count", "single").reason.startswith("selected-set")
    with pytest.raises(CertificateVersionError):
        registry.adapter_by_id("nonexistent.adapter")


def test_adapter_definition_is_validated() -> None:
    class Broken(ToySumAdapter):
        family_stages = {"toy.items.truth": "grounding"}

    with pytest.raises(TypeError):
        Broken.validate_definition()
    ToySumAdapter.validate_definition()
    assert ADAPTER.element_schema_ok(good_obj()["certificate"], 3) == [True, True, True]
    assert ADAPTER.element_schema_ok(good_obj()["certificate"], 2) == [True, True, False]
    assert ADAPTER.example_json().startswith('{"annotation":')
    contract = ADAPTER.public_contract(IMAGE_SIZE, "rel1000").to_dict()
    assert contract["kind"] == "toy_sum" and contract["image_size"] == [640, 480]
