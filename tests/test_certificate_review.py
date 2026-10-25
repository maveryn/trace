"""Regression tests of the M1 evaluation-layer review (finding ids in ``<!-- review: -->`` notes).

Each test pins one finding of the adversarial review (lenses exploit EX-, leakage LEAK-,
conformance CONF-, data DATA-, contract CONTRACT-) on the frozen adapter fixtures of
``tests/fixtures/certificates`` through the real adapters, so a regression fails here even
when the frozen data under ``trace-cert-runs`` is not reachable.
"""

from __future__ import annotations

import copy
import decimal
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Tuple

import pytest

from tests.test_certificate_prompts import validate
from tests.test_certificate_roundtrip import CASES, CASE_IDS, extension_of, oracle_object, public_prompts, score
from trace_tasks.experimental.certificates import evaluate, export, prompts, registry, reward
from trace_tasks.experimental.certificates.adapters.path import Grid, steps_for
from trace_tasks.experimental.certificates.geometry import round_half_up
from trace_tasks.experimental.certificates.parsing import final_json_object, locate_scored_object
from trace_tasks.experimental.certificates.scoring import score_certificate_response
from trace_tasks.experimental.certificates.types import (
    CertificateExtension,
    CertificateVersionError,
    ExtensionMismatchError,
)

BIG = 10**400  # valid RFC 8259 JSON; float() raises OverflowError on it
FIRST_CASES = [case for case in CASES if case.index < 2]
FIRST_IDS = [str(case) for case in FIRST_CASES]
SELECTION = {"selected_set": "ss.set.truth", "filtered_mean": "fm.rows.truth", "weighted_sum": "ws.terms.truth"}


def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


def bounded(result) -> None:
    for value in (result.answer, result.grounding, result.semantics, result.format, *result.overall.values()):
        assert math.isfinite(value) and 0.0 <= value <= 1.0


def geometry_leaves(obj: Any, path: Tuple[Any, ...] = ()) -> Iterator[Tuple[Any, ...]]:
    """Paths of every number in the annotation and in path ``steps[*].at``."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from geometry_leaves(value, path + (key,))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from geometry_leaves(value, path + (index,))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if path[0] == "annotation" or (len(path) >= 4 and path[1] == "steps" and path[3] == "at"):
            yield path


def replaced(obj: Any, path: Tuple[Any, ...], value: Any) -> Any:
    out = copy.deepcopy(obj)
    node = out
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return out


def check(result, check_id: str):
    return next(c for c in result.checks if c.check_id == check_id)


# ======================================================================================
# EX-1, CONF-01, CONTRACT-1/2/3: model output never crashes the scorer or the reward batch


@pytest.mark.parametrize("case", FIRST_CASES, ids=FIRST_IDS)
def test_numbers_too_large_for_a_float_in_every_geometry_slot_score_without_raising(case) -> None:
    ext = extension_of(case)
    obj = oracle_object(ext)
    leaves = list(geometry_leaves(obj))
    assert leaves
    for path in leaves:
        for big in (BIG, -BIG):
            text = dumps(replaced(obj, path, big))
            results = []
            for frame, shown in (("px_source", False), ("px_source", True), ("rel1000", False)):
                result = score_certificate_response(
                    text, {"instance_id": ext.instance_id}, ext, coord_frame=frame, shown_frame_diagnostics=shown
                )
                bounded(result)
                assert result.format == 0.0, path  # an unusable number is never a valid coordinate
                results.append(result)
            plain, shown = results[0], results[1]
            # the shown-frame diagnostic never changes whether or how a row scores (CONTRACT-2)
            assert (plain.answer, plain.grounding, plain.semantics, plain.format) == (
                shown.answer, shown.grounding, shown.semantics, shown.format)
            if path[0] == "certificate":  # a path step point: that step is an invalid element
                assert check(plain, "pa.legal.truth").error_code == "invalid_step"


@pytest.mark.parametrize("literal", ["{1j}", "{b'x'}", "{1j, 2}"])
def test_python_literal_answers_the_legacy_helper_cannot_serialize_score_zero(literal: str) -> None:
    ext = extension_of(CASES[0])
    for text in (f"The answer is below.\n{{'answer': {literal}}}", f"{{'annotation': [], 'answer': {literal}}}"):
        result = score(text, ext)
        bounded(result)
        assert result.answer == 0.0 and result.diagnostics["answer_guard"] == "unscorable_answer"


def _store(tmp_path: Path, cases) -> Tuple[str, str, List[CertificateExtension]]:
    extensions = [extension_of(case) for case in cases]
    path = tmp_path / "store.jsonl.zst"
    export.write_store(path, [ext.to_dict() for ext in extensions])
    return str(path), export.file_sha256(path), extensions


def test_one_bad_rollout_never_aborts_the_reward_batch(tmp_path: Path) -> None:
    cases = [next(c for c in CASES if c.adapter_name == name) for name in export.ADAPTER_ORDER]
    path, sha, extensions = _store(tmp_path, cases)
    rows = []
    for ext in extensions:
        obj = oracle_object(ext)
        texts = [dumps(obj)] + [dumps(replaced(obj, leaf, BIG)) for leaf in list(geometry_leaves(obj))[:6]]
        texts += ["x\n{'answer': {1j}}", "x\n{'answer': {b'x'}}", "{" * 3000, "```json\n" + dumps(obj)]
        for text in texts:
            rows.append({"response": text, "ground_truth": {"cert_extension_id": ext.certificate_extension_id},
                         "instance_id": ext.instance_id})
    for arm, gated in (("A", False), ("G", False), ("P", False), ("P", True)):
        out = reward.compute_score(
            rows, store_path=path, store_sha256=sha, arm=arm, gated=gated, details_dir=str(tmp_path / "details"),
            split_phases={"fixtures": "train"},
        )
        assert len(out) == len(rows)
        for metrics in out:
            assert all(math.isfinite(v) and v >= 0.0 for v in metrics.values())


# ======================================================================================
# VER-1: integers with no decimal form under sys.int_max_str_digits (4300) never raise.
# Hex, octal and binary Python literals parse without the digit limit, but repr/str/ast.dump
# and json.dumps of the value raise ValueError; a strict-JSON int has at most 4300 digits, but a
# sum a check logs (two 4300-digit path costs) has 4301.

HUGE_LITERALS = {  # each value exceeds 10**4300; about 450 Qwen tokens for the hex forms
    "hex": "0x" + "f" * 3572,
    "neg_hex": "-0x" + "f" * 3572,
    "upper_hex": "0X" + "F" * 3572,
    "octal": "0o" + "7" * 4763,
    "binary": "0b" + "1" * 14285,
}
HEX = HUGE_LITERALS["hex"]
D4300 = int("9" * 4300)  # the largest decimal int json.loads accepts


def _cert_int_path(obj: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Path of the first integer certificate value (not a path step point)."""
    def walk(node: Any, path: Tuple[Any, ...]):
        if isinstance(node, dict):
            for key, value in node.items():
                if key != "at":
                    yield from walk(value, path + (key,))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                yield from walk(value, path + (index,))
        elif isinstance(node, int) and not isinstance(node, bool):
            yield path

    return next(walk(obj["certificate"], ("certificate",)))


def _python_literal(obj: Mapping[str, Any], path: Tuple[Any, ...], literal: str) -> str:
    """``obj`` as a Python literal (legacy ``ast.literal_eval`` fallback) with ``literal`` at ``path``."""
    return repr(replaced(obj, path, "@@BIG@@")).replace("'@@BIG@@'", literal)


def huge_int_texts(ext: CertificateExtension) -> Dict[str, str]:
    """Responses holding an int with no decimal form (or a strict-JSON int at the 4300-digit
    limit), in the answer, duplicated answer keys, annotation boxes and map keys, and in
    certificate fields, as Python literals and as JSON."""
    obj = oracle_object(ext)
    gt = ext.labels["answer_gt"]["value"]
    leaf = _cert_int_path(obj)
    texts: Dict[str, str] = {}
    for name, lit in HUGE_LITERALS.items():
        texts[f"{name}:bare"] = f"{{'answer': {lit}}}"
        texts[f"{name}:dup_then_gt"] = f"{{'answer': {lit}, 'answer': {gt}}}"
        texts[f"{name}:cert_literal"] = _python_literal(obj, leaf, lit)
    texts.update({
        "hex:prose": f"Let me think.\n{{'answer': {HEX}, 'annotation': []}}",
        "hex:fenced": f"```json\n{{'answer': {HEX}}}\n```",
        "hex:in_list": f"{{'answer': [{HEX}]}}",
        "hex:dup_gt_then": f"{{'answer': {gt}, 'answer': {HEX}}}",
        "hex:dup_same": f"{{'answer': {HEX}, 'answer': {HEX}}}",
        "hex:dup_same_list": f"{{'answer': [{HEX}], 'answer': [{HEX}]}}",
        "hex:annotation_box": f"{{'answer': {gt}, 'annotation': [[{HEX}, 1, 2, 3]]}}",
        "hex:annotation_map_key": f"{{'answer': {gt}, 'annotation': {{{HEX}: [1, 2, 3, 4]}}}}",
        "hex:annotation_tuple_key": f"{{'answer': {gt}, 'annotation': {{({HEX}, 1): [1, 2, 3, 4]}}}}",
        "hex:annotation_answer_literal": _python_literal(obj, ("annotation",), f"[[{HEX}, 1, 2, 3]]"),
        "json:answer_4301_digits": '{"answer": ' + "9" * 4301 + "}",
        "json:answer_4300_digits": dumps(dict(obj, answer=D4300)),
        "json:cert_4300_digits": dumps(replaced(obj, leaf, D4300)),
    })
    if ext.adapter_id.startswith("path."):  # valid JSON: the logged claimed_sum has 4301 digits
        big = copy.deepcopy(obj)
        for step in big["certificate"]["steps"][-2:]:
            step["cost"] = D4300
        big["certificate"]["total"] = D4300
        texts["json:path_costs_sum_4301_digits"] = dumps(big)
    return texts


@pytest.mark.parametrize("name", export.ADAPTER_ORDER)
def test_ints_with_no_decimal_form_never_raise_in_the_scorer(name: str) -> None:
    from trace_tasks.experimental.certificates import parsing

    ext = extension_of(next(c for c in CASES if c.adapter_name == name))
    for label, text in huge_int_texts(ext).items():
        assert isinstance(parsing.text_answer_conflict(text), bool), label
        for frame, shown in (("px_source", False), ("px_source", True), ("rel1000", False)):
            result = score_certificate_response(
                text, {"instance_id": ext.instance_id}, ext, coord_frame=frame, shown_frame_diagnostics=shown
            )
            bounded(result)
            json.dumps(result.to_dict(), allow_nan=False)  # every logged record serializes
            if label.endswith(":dup_then_gt") or label == "hex:dup_gt_then":
                # the duplicate-answer guard holds although one value has no repr (EX-7, VER-1)
                assert result.answer == 0.0 and result.diagnostics["answer_guard"] == "duplicate_answer", label
            if label in ("hex:dup_same", "hex:dup_same_list"):
                assert result.diagnostics["answer_guard"] != "duplicate_answer", label
            if label == "json:path_costs_sum_4301_digits" and frame == "px_source":
                total = next(c for c in result.to_dict()["checks"] if c["check_id"] == "pa.total.truth")
                assert total["detail"]["claimed_sum"].startswith("<int: "), total


def test_duplicate_answer_comparison_is_total() -> None:
    from trace_tasks.experimental.certificates import parsing

    for lit in HUGE_LITERALS.values():
        assert parsing.text_answer_conflict(f"{{'answer': {lit}, 'answer': {lit}}}") is False
        assert parsing.text_answer_conflict(f"{{'answer': 3, 'answer': {lit}}}") is True
        assert parsing.text_answer_conflict(f"{{'answer': ({lit}, 1), 'answer': ({lit}, 2)}}") is True
    assert parsing.text_answer_conflict("{'answer': 3, 'answer': 3.0}") is True  # repr semantics kept
    assert parsing.text_answer_conflict("{'answer': x, 'answer': x}") is False  # ast.dump semantics kept
    assert parsing.text_answer_conflict("{'answer': x, 'answer': y}") is True
    nested = parsing.strict_loads('{"answer": [[1]], "answer": [[2]]}')
    assert parsing.answer_duplicate_conflict(nested) is True


def test_json_safe_replaces_only_ints_with_no_decimal_form() -> None:
    from trace_tasks.experimental.certificates.types import json_safe

    plain = {"a": [1, 2.5, None, True, "x", (3, 4)], "b": {"c": D4300, 7: -(1 << 70)}}
    assert json_safe(plain) is plain  # unchanged records are the same object (byte-identical logs)
    big = 16 ** 3572
    out = json_safe({"a": [1, big, (big, 2)], big: {"k": -big}})
    marker = f"<int: {big.bit_length()} bits>"
    assert out == {"a": [1, marker, (marker, 2)], marker: {"k": marker}}
    json.dumps(out, allow_nan=False)


def test_ints_with_no_decimal_form_never_abort_the_reward_batch(tmp_path: Path) -> None:
    cases = [next(c for c in CASES if c.adapter_name == name) for name in export.ADAPTER_ORDER]
    path, sha, extensions = _store(tmp_path, cases)
    rows = [
        {"response": text, "ground_truth": {"cert_extension_id": ext.certificate_extension_id},
         "instance_id": ext.instance_id}
        for ext in extensions for text in [dumps(oracle_object(ext)), *huge_int_texts(ext).values()]
    ]
    for arm in ("A", "G", "P"):
        for details in (None, tmp_path / f"details_{arm}"):
            out = reward.compute_score(
                rows, store_path=path, store_sha256=sha, arm=arm, details_dir=None if details is None else str(details),
                split_phases={"fixtures": "train"},
            )
            assert len(out) == len(rows)
            assert all(math.isfinite(v) and v >= 0.0 for metrics in out for v in metrics.values())
            if details is not None:
                logged = [json.loads(line) for f in sorted(details.iterdir()) for line in f.read_text().splitlines()]
                assert len(logged) == len(rows)


@pytest.fixture(scope="module")
def tiny_root() -> Path:
    from tests.test_certificate_export import build_tiny_root

    return build_tiny_root()


def test_ints_with_no_decimal_form_never_raise_in_the_evaluate_cli(tiny_root: Path, tmp_path: Path) -> None:
    from tests.test_certificate_export import store_rows, write_run
    from trace_tasks.experimental.certificates.__main__ import main as cli_main

    exts = {row["base"]["instance_id"]: CertificateExtension.from_dict(row) for row in store_rows(tiny_root, "fixtures")}
    texts = {iid: huge_int_texts(ext) for iid, ext in exts.items()}
    shapes = ("hex:bare", "hex:dup_then_gt", "octal:cert_literal", "hex:annotation_box", "hex:annotation_map_key",
              "json:cert_4300_digits", "json:path_costs_sum_4301_digits")
    conditions = (
        ("cert_px", "Qwen/Qwen2.5-VL-3B-Instruct", []),
        ("cert_rel1000", "Qwen/Qwen3.5-9B", []),
        ("ans", "Qwen/Qwen3.5-4B", ["--allow-settings-mismatch"]),  # trace-tool records TRACE's system text
        ("ann", "Qwen/Qwen2.5-VL-7B-Instruct", []),
    )
    for cond, model, extra in conditions:
        for shape in shapes:
            responses = {iid: t.get(shape, t["json:cert_4300_digits"]) for iid, t in texts.items()}
            run = write_run(tiny_root, "fixtures", cond, tmp_path / cond / shape.replace(":", "_"), responses, model=model)
            out = tmp_path / "out" / cond / shape.replace(":", "_")
            assert cli_main([
                "score", "--responses", str(run), "--store", str(tiny_root / "private" / export.STORE_NAME),
                "--receipt", str(tiny_root / "receipts" / "fixtures.json"), "--out", str(out), *extra,
            ]) == 0, (cond, shape)
            scored = [json.loads(line) for line in (out / "cert_scores.jsonl").read_text().splitlines()]
            assert len(scored) == len(exts), (cond, shape)
            assert all(0.0 <= row["A"] <= 1.0 for row in scored)
            if shape == "hex:dup_then_gt":
                assert all(row["A"] == 0.0 and row["answer_guard"] == "duplicate_answer" for row in scored), cond


# ======================================================================================
# EX-2, EX-3, CONF-03, LEAK-03: no image-blind floor and no cost farming on paths


def _mean_m(responses: Mapping[str, str], name: str) -> float:
    cases = [c for c in CASES if c.adapter_name == name]
    values = [score(responses[name] if isinstance(responses[name], str) else dumps(responses[name]), extension_of(c)).semantics for c in cases]
    return sum(values) / len(values)


@pytest.mark.parametrize("name", export.ADAPTER_ORDER)
def test_copying_the_prompt_example_earns_no_semantic_credit(name: str) -> None:
    example = registry.load_adapter(name).example_object()
    assert _mean_m({name: dumps(example)}, name) < 0.05


def test_best_fixed_image_blind_walk_earns_no_semantic_credit() -> None:
    cases = [c for c in CASES if c.adapter_name == "path"]
    tile = Grid.from_oracle(extension_of(cases[0]).oracle).tile_px
    best = 0.0
    for row in range(0, 8, 2):
        for col in range(0, 8, 2):
            for d_row, d_col in ((0, 1), (1, 0)):
                steps = [{"at": [tile * (col + d_col * i) + tile // 2, tile * (row + d_row * i) + tile // 2],
                          "cost": 0 if i == 0 else 1} for i in range(5)]
                cert = {"kind": "path", "reachable": True, "steps": steps, "total": 4}
                first, last = steps[0]["at"], steps[-1]["at"]
                annotation = {"player_cell": [first[0] - 40, first[1] - 40, first[0] + 40, first[1] + 40],
                              "target_cell": [last[0] - 40, last[1] - 40, last[0] + 40, last[1] + 40]}
                text = dumps({"annotation": annotation, "certificate": cert, "answer": 4})
                best = max(best, sum(score(text, extension_of(c)).semantics for c in cases) / len(cases))
    assert best < 0.05


@pytest.mark.parametrize("case", [c for c in CASES if c.adapter_name == "path"], ids=lambda c: str(c))
def test_stutter_and_backtrack_routes_cannot_farm_cost_credit(case) -> None:
    ext = extension_of(case)
    oracle, obj = ext.oracle, oracle_object(ext)
    grid = Grid.from_oracle(oracle)
    start, target = grid.index(*oracle["start"]), grid.index(*oracle["target"])
    m = int(oracle["min_tiles"])
    if m <= 2:
        pytest.skip("start and target are adjacent: [start, target] is the real optimal route")
    stutter_tiles = [start] * (m - 1) + [target]
    stutter = dict(obj, certificate={"kind": "path", "reachable": True, "total": 0,
                                     "steps": steps_for(grid, stutter_tiles, [0] + [grid.entry[t] for t in stutter_tiles[1:]])})
    result = score(stutter, ext)
    assert check(result, "pa.cost.truth").score <= 1.0 / m + 1e-12
    honest = copy.deepcopy(obj)  # the optimal route with one wrong read
    honest["certificate"]["steps"][1]["cost"] = honest["certificate"]["steps"][1]["cost"] % 3 + 1
    assert check(result, "pa.cost.truth").score < check(score(honest, ext), "pa.cost.truth").score
    neighbour = next((v for v in grid.neighbors(start) if grid.passable(v)), None)
    walk = [start if i % 2 == 0 else neighbour for i in range(max(m, 6))]
    back = dict(obj, certificate={"kind": "path", "reachable": True, "total": 0,
                                  "steps": steps_for(grid, walk, [0] + [grid.entry[t] for t in walk[1:]])})
    assert check(score(back, ext), "pa.cost.truth").score <= 2.0 / len(walk) + 1e-12


def test_reversed_route_and_detached_walks_get_no_legality_or_cost() -> None:
    case = next(c for c in CASES if c.adapter_name == "path")
    ext = extension_of(case)
    obj = oracle_object(ext)
    reversed_obj = copy.deepcopy(obj)
    reversed_obj["certificate"]["steps"] = list(reversed(obj["certificate"]["steps"]))
    result = score(reversed_obj, ext)
    assert check(result, "pa.legal.truth").score == 0.0 and check(result, "pa.cost.truth").score == 0.0
    assert check(result, "pa.legal.truth").error_code == "wrong_start"


# ======================================================================================
# EX-4, EX-7, CONF-05, CONF-08: one object for A, G and M; the guard follows it


@pytest.mark.parametrize("case", FIRST_CASES, ids=FIRST_IDS)
def test_an_earlier_fence_never_hides_the_final_object(case) -> None:
    ext = extension_of(case)
    good = dumps(oracle_object(ext))
    example = case.adapter.example_json()
    wrong = copy.deepcopy(oracle_object(ext))
    wrong["answer"] = wrong["answer"] + 7
    for text in (
        f"Format:\n```json\n{example}\n```\nMy answer:\n```json\n{good}\n```",
        f"Draft:\n```json\n{dumps(wrong)}\n```\nCorrected:\n```json\n{good}\n```",
        f"```python\nprint(1)\n```\n```json\n{good}\n```",
    ):
        assert final_json_object(text) == (good, True)
        result = score(text, ext)
        assert (result.answer, result.semantics, result.format) == (1.0, 1.0, 1.0), text[:40]
    assert final_json_object(f"```json\n{good}\n```\n```python\nx = 1\n```") == (None, False)


@pytest.mark.parametrize("case", FIRST_CASES, ids=FIRST_IDS)
def test_duplicate_answer_guard_does_not_depend_on_position(case) -> None:
    ext = extension_of(case)
    obj = oracle_object(ext)
    gt = obj["answer"]
    text = dumps(obj)[:-1].rsplit('"answer"', 1)[0] + f'"answer":{gt + 7},"answer":{gt}}}'
    for response in (text, text + " Done.", "Reasoning.\n" + text + "\nDone.", f"```json\n{text}\n```"):
        result = score(response, ext)
        assert result.answer == 0.0 and result.diagnostics["answer_guard"] == "duplicate_answer", response[-12:]


def test_a_broken_final_object_is_an_envelope_failure_not_a_draft() -> None:
    ext = extension_of(CASES[0])
    good = dumps(oracle_object(ext))
    for final in (good[: good.rindex(":") + 1] + "NaN}", dumps({"answer": oracle_object(ext)["answer"]})):
        result = score(f"draft: {good}\nfinal: {final}", ext)
        assert result.semantics == 0.0 and result.diagnostics["envelope_error"] in ("non_finite", "missing_certificate")
        assert locate_scored_object(f"draft: {good}\nfinal: {final}").scored is None


# ======================================================================================
# EX-5: a box nobody references is still a claim


@pytest.mark.parametrize("case", [c for c in CASES if c.adapter_name in SELECTION], ids=lambda c: str(c))
def test_unreferenced_annotation_boxes_lower_the_selection_family(case) -> None:
    ext = extension_of(case)
    obj = copy.deepcopy(oracle_object(ext))  # the cached extension must not be mutated
    candidates = case.adapter.candidates(ext.oracle)
    if case.adapter_name == "filtered_mean":
        role = "filter_cells"
        chosen = set(candidates[role].selected)
        extra = next(i for i in range(len(candidates[role].boxes)) if i not in chosen)
        obj["annotation"][role].append(list(candidates[role].boxes[extra]))
    else:
        chosen = set(candidates[""].selected)
        extra = next((i for i in range(len(candidates[""].boxes)) if i not in chosen), None)
        if extra is None:
            pytest.skip("every object of this scene is selected")
        obj["annotation"].append(list(candidates[""].boxes[extra]))
    result = score(obj, ext)
    assert check(result, SELECTION[case.adapter_name]).score < 1.0
    assert result.semantics < 1.0 and result.joint == 0.0 and result.error_class != "correct"


def test_boxing_every_piece_on_an_empty_scene_is_not_correct() -> None:
    for case in (c for c in CASES if c.adapter_name == "selected_set"):
        ext = extension_of(case)
        if ext.oracle["selected"]:
            continue
        boxes = [list(piece["bbox"]) for piece in ext.oracle["pieces"]]
        obj = {"annotation": boxes, "certificate": {"kind": "selected_set", "members": [], "count": 0}, "answer": 0}
        result = score(obj, ext)
        assert result.semantics < 1.0 and result.joint == 0.0 and result.error_class != "correct"


# ======================================================================================
# CONF-02: F agrees with the public schema on the annotation


def _annotation_mutations(obj: Mapping[str, Any], name: str) -> Dict[str, Any]:
    ann = obj["annotation"]
    out: Dict[str, Any] = {"python_literal_string": repr(ann), "json_string": json.dumps(ann), "null": None}
    if isinstance(ann, dict):
        keys = list(ann)
        out.update({
            "empty_map": {},
            "missing_key": {keys[0]: ann[keys[0]]},
            "renamed_keys": {f"x_{k}": v for k, v in ann.items()},
            "extra_key": dict(ann, extra=ann[keys[0]]),
        })
    else:
        out.update({"bboxes_wrapper": {"bboxes": ann}, "empty_list_string": "[]"})
        if ann:
            out["bare_box"] = ann[0]
    return out


@pytest.mark.parametrize("case", FIRST_CASES, ids=FIRST_IDS)
def test_f_never_accepts_an_annotation_the_public_schema_rejects(case) -> None:
    ext = extension_of(case)
    obj = oracle_object(ext)
    schema = prompts.public_schema(case.adapter.kind)
    assert validate(json.loads(dumps(obj)), schema) == [] and score(obj, ext).format == 1.0
    for label, annotation in _annotation_mutations(obj, case.adapter_name).items():
        mutated = dict(obj, annotation=annotation)
        result = score(mutated, ext)
        if result.format == 1.0:
            assert validate(json.loads(dumps(mutated)), schema) == [], label
        assert result.format == 0.0, label


# ======================================================================================
# CONTRACT-4, CONTRACT-5, CONTRACT-6


def test_mistyped_reward_kwargs_raise(tmp_path: Path) -> None:
    path, sha, extensions = _store(tmp_path, CASES[:1])
    ext = extensions[0]
    rows = [{"response": dumps(oracle_object(ext)), "ground_truth": {"cert_extension_id": ext.certificate_extension_id},
             "instance_id": ext.instance_id}]
    kwargs = dict(store_path=path, store_sha256=sha, split_phases={"fixtures": "train"})
    for bad in ({"gated": "false"}, {"gated": "no"}, {"gated": 0}, {"alpha": True}, {"format_weight": False},
                {"alpha": "0.8"}, {"arm": 1}):
        with pytest.raises(ValueError):
            reward.compute_score(rows, **{"arm": "P", **kwargs, **bad})
    assert reward.compute_score(rows, arm="P", gated=False, **kwargs)


def test_an_over_cap_annotation_is_a_budget_failure_and_boxes_resolve_once(monkeypatch) -> None:
    from trace_tasks.experimental.certificates.adapters import base as adapter_base

    case = next(c for c in CASES if c.adapter_name == "selected_set" and c.record)
    ext = extension_of(case)
    obj = oracle_object(ext)
    cap = case.adapter.annotation_cap()
    degenerate = dict(obj, annotation=[[1, 1, 9, 9]] * (cap + 1))  # one box over the cap, far under 16 KiB
    assert len(dumps(degenerate)) < 16 * 1024
    result = score(degenerate, ext)
    assert result.grounding == score(dict(degenerate), ext).grounding  # legacy G still scores it
    assert result.semantics == 0.0 and result.diagnostics["envelope_error"] == "budget"
    assert result.diagnostics["identity_budget"] is True and result.diagnostics["G_id"] is None
    assert "annotation_budget" in result.diagnostics["format_reasons"]
    calls: List[int] = []
    original = adapter_base.resolve_boxes

    def counting(boxes, candidates, params):
        calls.append(len(boxes))
        return original(boxes, candidates, params)

    monkeypatch.setattr(adapter_base, "resolve_boxes", counting)
    score_certificate_response(dumps(obj), {"instance_id": ext.instance_id}, ext, coord_frame="px_source",
                               shown_frame_diagnostics=True)
    assert len(calls) == 2  # once in the contract frame (checks, diagnostics, identity), once shown


def test_sft_rounding_is_hermetic_to_the_decimal_context() -> None:
    with decimal.localcontext(decimal.Context(prec=2)) as context:
        context.clear_flags()
        assert round_half_up(278.5) == 279 and round_half_up(12345.5) == 12346
        assert not any(context.flags.values())


# ======================================================================================
# EX-6, LEAK-02, CONF-09: frame and prompt integrity; plan API defaults


def test_frame_is_read_from_the_row_and_a_mismatch_raises() -> None:
    case = next(c for c in CASES if public_prompts(c) is not None)
    record, adapter = case.record, case.adapter
    ext = export.build_extension(record.train, record.trace, adapter)  # the plan's three-argument call
    prompts_by_cond = public_prompts(case)
    assert set(ext.public["prompt_sha256"]) == {"px_source", "rel1000", "ans_v1", "ann_v1"}
    good = dumps(oracle_object(ext))
    base_px = {"instance_id": ext.instance_id, "prompt_answer": prompts_by_cond["cert_px"]}
    result = score_certificate_response(good, base_px, ext)  # the plan's three-argument call
    assert result.diagnostics["coord_frame"] == "px_source" and result.semantics == 1.0
    base_rel = {"instance_id": ext.instance_id, "prompt_cert": prompts_by_cond["cert_rel1000"]}
    assert score_certificate_response(good, base_rel, ext).diagnostics["coord_frame"] == "rel1000"
    with pytest.raises(CertificateVersionError):
        score_certificate_response(good, base_rel, ext, coord_frame="px_source")
    with pytest.raises(ValueError, match="coord_frame is required"):
        score_certificate_response(good, {"instance_id": ext.instance_id}, ext)
    for key in ("payload_sha256", "cert_code_sha256", "trace_core_sha256", "env_lock_sha256", "trace_tasks_commit"):
        assert key in ext.provenance
    ext.verify()


def test_plan_api_accepts_traces_own_base_row_and_never_guesses_the_frame() -> None:
    """Review VER-2: the plan passes TRACE's own base row (the TrainInstance given to
    ``build_extension``) to ``score_certificate_response``. Its ``prompt`` is TRACE's legacy
    prompt, not a certificate prompt, so it states no frame: the frame comes from an explicit
    argument, the row's ``coord_frame`` field or an extension that froze a single frame."""
    import hashlib

    case = next(c for c in CASES if public_prompts(c) is not None)
    record, adapter = case.record, case.adapter
    ext = export.build_extension(record.train, record.trace, adapter)  # freezes both frames
    base = dict(record.train)
    assert isinstance(base.get("prompt"), str)
    assert hashlib.sha256(base["prompt"].encode()).hexdigest() not in ext.public["prompt_sha256"].values()
    good = dumps(oracle_object(ext))
    result = score_certificate_response(good, base, ext, coord_frame="px_source")
    assert result.diagnostics["coord_frame"] == "px_source"
    assert (result.answer, result.semantics, result.format) == (1.0, 1.0, 1.0)
    assert score_certificate_response(good, base, ext, coord_frame="rel1000").diagnostics["coord_frame"] == "rel1000"
    assert score_certificate_response(good, dict(base, coord_frame="rel1000"), ext).diagnostics["coord_frame"] == "rel1000"
    with pytest.raises(ValueError, match="coord_frame is required"):  # two frames frozen: never guessed
        score_certificate_response(good, base, ext)
    single = export.build_extension(
        record.train, record.trace, adapter, prompt_sha256={"px_source": ext.public["prompt_sha256"]["px_source"]}
    )
    assert score_certificate_response(good, base, single).diagnostics["coord_frame"] == "px_source"
    # A `prompt` that is one of the extension's frozen prompts is a public prompt: read and checked.
    prompts_by_cond = public_prompts(case)
    rel_base = dict(base, prompt=prompts_by_cond["cert_rel1000"])
    assert score_certificate_response(good, rel_base, ext).diagnostics["coord_frame"] == "rel1000"
    with pytest.raises(CertificateVersionError):
        score_certificate_response(good, rel_base, ext, coord_frame="px_source")
    with pytest.raises(CertificateVersionError):  # the answer-only prompt is no certificate prompt
        score_certificate_response(good, dict(base, prompt=prompts_by_cond["ans"]), ext)
    # The public rows' prompt columns stay strict: a prompt that is not frozen raises.
    for key in ("prompt_answer", "prompt_cert"):
        with pytest.raises(CertificateVersionError):
            score_certificate_response(good, dict(base, **{key: base["prompt"]}), ext, coord_frame="px_source")


def test_reward_refuses_test_splits_and_frames_the_prompt_did_not_declare(tmp_path: Path) -> None:
    case = next(c for c in CASES if public_prompts(c) is not None)
    record, adapter = case.record, case.adapter
    ext = export.build_extension(record.train, record.trace, adapter, image_size=record.source_size(), split="e2_test")
    path = tmp_path / "store.jsonl.zst"
    export.write_store(path, [ext.to_dict()])
    sha = export.file_sha256(path)
    row = {"response": dumps(oracle_object(ext)), "ground_truth": {"cert_extension_id": ext.certificate_extension_id},
           "instance_id": ext.instance_id, "prompt_cert": public_prompts(case)["cert_px"]}
    with pytest.raises(CertificateVersionError, match="held-out oracles never reach RL"):
        reward.compute_score([row], store_path=str(path), store_sha256=sha, arm="P")
    phases = {"e2_test": "val"}
    assert reward.compute_score([row], store_path=str(path), store_sha256=sha, arm="P", split_phases=phases)
    with pytest.raises(CertificateVersionError, match="rel1000 prompt"):
        reward.compute_score([row], store_path=str(path), store_sha256=sha, arm="P", coord_frame="rel1000",
                             split_phases=phases)


# ======================================================================================
# LEAK-04: store integrity


def test_an_edited_store_row_is_detected_and_explicit_shas_add_to_the_receipt(tmp_path: Path) -> None:
    ext = extension_of(CASES[0])
    tampered = CertificateExtension.from_dict(copy.deepcopy(ext.to_dict()))
    tampered.labels["answer_gt"] = dict(tampered.labels["answer_gt"], value=tampered.labels["answer_gt"]["value"] + 1)
    tampered.verify_id()  # the id does not cover labels ...
    with pytest.raises(ExtensionMismatchError, match="payload"):
        tampered.verify()  # ... the payload hash does
    path = tmp_path / "tampered.jsonl.zst"
    export.write_store(path, [tampered.to_dict()])
    with pytest.raises(ExtensionMismatchError):
        evaluate.load_store(path)
    good = tmp_path / "good.jsonl.zst"
    export.write_store(good, [ext.to_dict()])
    store = evaluate.load_store(good)
    receipt = {"split": "fixtures", "outputs": {f"private/fixtures.{export.STORE_NAME}": {"sha256": "0" * 64}}}
    with pytest.raises(CertificateVersionError, match="does not contain"):
        evaluate.verify_store(store, receipt, store_sha256=store.sha256)
    receipt["outputs"][f"private/fixtures.{export.STORE_NAME}"]["sha256"] = store.sha256
    assert evaluate.verify_store(store, receipt, store_sha256=store.sha256) == "explicit_sha256+split_store"


# ======================================================================================
# CONF-04, CONF-06, CONF-07, LEAK-05, CONF-14: offline evaluation integrity


def test_server_rejections_are_infrastructure_exclusions(tmp_path: Path) -> None:
    path = tmp_path / "responses.jsonl"
    lines = [
        {"row_index": 0, "instance_id": "a", "stop_reason": "answer", "final_response": "x", "turns": 1},
        {"row_index": 1, "instance_id": "b", "stop_reason": "rejected", "turns": 0, "completion_tokens": 0},
        {"row_index": 2, "instance_id": "c", "stop_reason": "error"},
        {"row_index": 3, "instance_id": "d", "stop_reason": "rejected", "turns": 0, "completion_tokens": 0},
        {"row_index": 3, "instance_id": "d", "stop_reason": "answer", "final_response": "y", "turns": 1},
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    records, failures, _ = evaluate.read_responses(path)
    assert [r["instance_id"] for r in records] == ["a", "d"]
    assert {r["instance_id"]: evaluate.infrastructure_failure(r) for r in failures} == {"b": "rejected", "c": "endpoint_error"}


def test_pooled_summaries_carry_every_run_denominator(tmp_path: Path) -> None:
    runs = []
    for seed in (42, 43):
        run = tmp_path / f"seed{seed}"
        run.mkdir()
        (run / "cert_scores.jsonl").write_text(json.dumps({"model": "m", "cond": "cert_px", "seed": seed, "adapter_id": "x.y",
                                                           "split": "e1_dev", "A": 1.0}) + "\n")
        summary = {"run": {"model": "m", "cond": "cert_px", "split": "e1_dev"},
                   "denominators": {"x.y": {"supported": 16, "scored": 2, "missing": 13, "endpoint_errors": 1,
                                            "rejected": 0, "quarantined": 0}, "unsupported_adapters": {}}}
        (run / "summary.json").write_text(json.dumps(summary))
        runs.append(run / "cert_scores.jsonl")
    pooled = evaluate.summarize_files(runs, tmp_path / "pooled.json")
    totals = pooled["denominators"]["m|cert_px|e1_dev"]
    assert totals["runs"] == 2 and totals["by_adapter"]["x.y"]["missing"] == 26 and totals["by_adapter"]["x.y"]["supported"] == 32
    (runs[0].parent / "summary.json").unlink()
    with pytest.raises(CertificateVersionError, match="summary.json"):
        evaluate.summarize_files(runs, tmp_path / "pooled2.json")


def _run_dir(tmp_path: Path, config: Mapping[str, Any] | None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (tmp_path / "run_config.json").write_text(json.dumps(config))
    (tmp_path / "responses.jsonl").write_text("")
    return tmp_path / "responses.jsonl"


def _settings(cond: str, **override: Any) -> Dict[str, Any]:
    settings = {"system_prompt_sha256": prompts.expected_system_prompt_sha256(export.PUBLIC_CONDITIONS[cond]),
                "oracle_crop": False, "tools": [], "decoding": "trace"}
    settings.update(override)
    return settings


def test_unfrozen_datasets_and_wrong_settings_are_refused_unless_overridden(tmp_path: Path) -> None:
    receipt = {"split": "e1_dev", "outputs": {"public/e1_dev.cert_px.parquet": {
        "kind": "public_parquet", "sha256": "a" * 64, "condition": "cert_px", "split": "e1_dev"}}}
    frozen = {"dataset": {"sha256": "a" * 64}, "settings": _settings("cert_px"), "model_id": "Qwen/Qwen2.5-VL-3B-Instruct"}
    info = evaluate.resolve_run(_run_dir(tmp_path / "ok", frozen), receipt)
    assert info.cond == "cert_px" and info.dataset_frozen and not info.settings_problems
    unknown = dict(frozen, dataset={"sha256": "0" * 64})
    for config in (unknown, None):
        responses = _run_dir(tmp_path / f"u{config is None}", config)
        with pytest.raises(CertificateVersionError, match="not a frozen public parquet"):
            evaluate.resolve_run(responses, receipt, cond="cert_px")
    info = evaluate.resolve_run(_run_dir(tmp_path / "u2", unknown), receipt, cond="cert_px", allow_unfrozen=True)
    assert info.checks()["dataset_frozen"] is False and info.checks()["allow_unfrozen"] is True
    for override in ({"system_prompt_sha256": prompts.TRACE_SYSTEM_PROMPT_SHA256}, {"oracle_crop": True},
                     {"tools": ["crop"]}, {"decoding": "greedy"}):
        bad = dict(frozen, settings=_settings("cert_px", **override))
        responses = _run_dir(tmp_path / f"s{len(json.dumps(override))}", bad)
        with pytest.raises(CertificateVersionError, match="settings"):
            evaluate.resolve_run(responses, receipt)
        info = evaluate.resolve_run(responses, receipt, allow_settings_mismatch=True)
        assert info.checks()["settings_problems"]
    assert prompts.expected_system_prompt_sha256("ans_v1") == prompts.TRACE_SYSTEM_PROMPT_SHA256
    assert prompts.expected_system_prompt_sha256("ann_v1") == prompts.system_prompt_sha256()


def test_transcript_prompt_is_rebuilt_like_the_public_parquet() -> None:
    record = {"transcript": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "image", "index": 0, "size": [9, 9]}, {"type": "text", "text": "Q?\nmore"}]},
    ]}
    assert evaluate.transcript_prompt(record) == ("sys", "<image>Q?\nmore")
    assert evaluate.transcript_prompt({}) == (None, None)


# ======================================================================================
# CONF-11, CONF-13, DATA-1, DATA-4, DATA-5, CONF-15: export and coverage


def test_lever_semantic_key_ignores_the_render_variant() -> None:
    case = next(c for c in CASES if c.adapter_name == "weighted_sum")
    record = case.record
    key = case.adapter.semantic_key(record)
    for variant in ("center_fulcrum", "offset_fulcrum", "textured_beam"):
        trace = copy.deepcopy(dict(record.trace))
        for block in ("render_spec", "execution_trace"):
            if isinstance(trace.get(block), dict) and "scene_variant" in trace[block]:
                trace[block]["scene_variant"] = variant
        assert case.adapter.semantic_key(type(record)(train=record.train, trace=trace, image_size=record.image_size)) == key


def test_test_splits_drop_semantic_duplicates_within_the_split(monkeypatch) -> None:
    adapter = registry.load_adapter("weighted_sum")
    strata = export.quota_plan(adapter, "e1_test", 3)
    stratum = strata[0].name

    def fake_chunk(fill, split, task, seed, start, end, stage_root, runner):
        keys = ["k1", "k1", "k2", "k1", "k3", "k4"]
        return [{"seed_index": start + i, "status": "ok", "semantic_key": keys[i % len(keys)], "stratum": stratum,
                 "cell": "center_fulcrum", "instance_id": f"i{start + i}", "roundtrip": {"ok": True}}
                for i in range(end - start)]

    monkeypatch.setattr(export, "_load_or_run_chunk", fake_chunk)
    fill = export._AdapterFill(adapter=adapter, name="weighted_sum", strata=strata)
    export._fill_adapter(fill, "e1_test", 7401, Path("."), None, 6, set(), dedupe_within=True)
    assert [row["semantic_key"] for row in fill.accepted] == ["k1", "k2", "k3"]
    assert fill.counts["dedupe_within_dropped"] == 2 and fill.counts["duplicate_semantic_keys_within_split"] == 0


def test_an_oracle_round_trip_failure_is_quarantined(monkeypatch, tmp_path: Path) -> None:
    adapter = registry.load_adapter("path")
    task = next(iter(adapter.supported))
    strata = export.quota_plan(adapter, "fixtures", 1)
    job = export.CandidateJob(adapter.adapter_id, task, "fixtures", 7001, 0, str(tmp_path), strata, {})
    monkeypatch.setattr(export, "oracle_roundtrip", lambda ext, adapter: {"oracle": [0.0] * 4, "sft": [0.0] * 4, "ok": False})
    outcome = export.run_candidate(job)
    assert outcome["status"] == "quarantined" and outcome["reason"] == "roundtrip_failed"


def test_chunk_cache_key_covers_trace_generator_code(monkeypatch, tmp_path: Path) -> None:
    core = tmp_path / "trace_tasks"
    (core / "tasks").mkdir(parents=True)
    (core / "experimental").mkdir()
    (core / "tasks" / "gen.py").write_text("A = 1\n")
    (core / "experimental" / "cert.py").write_text("ignored\n")
    monkeypatch.setattr(export, "_core_root", lambda: core)
    export.trace_core_sha256.cache_clear()
    try:
        first = export.trace_core_sha256()
        (core / "experimental" / "cert.py").write_text("still ignored\n")
        export.trace_core_sha256.cache_clear()
        assert export.trace_core_sha256() == first
        (core / "tasks" / "gen.py").write_text("A = 2\n")
        export.trace_core_sha256.cache_clear()
        assert export.trace_core_sha256() != first
    finally:
        export.trace_core_sha256.cache_clear()
    adapter = registry.load_adapter("path")
    fill = export._AdapterFill(adapter=adapter, name="path", strata=export.quota_plan(adapter, "fixtures", 1))
    header = export._chunk_header(fill, "fixtures", "t", 7001, 0, 8)
    assert header["trace_core_sha256"] == export.code_provenance()["trace_core_sha256"]


def test_env_lock_does_not_depend_on_pythonpath_or_the_checkout(monkeypatch) -> None:
    text, _ = export.env_lock_text()
    monkeypatch.setenv("PYTHONPATH", str(Path(export.__file__).parents[3]))
    assert export.env_lock_text()[0] == text
    assert not any("trace_tasks" in line.lower().replace("-", "_") for line in text.splitlines())


def test_design10_path_like_tasks_carry_their_reason() -> None:
    from trace_tasks.experimental.certificates.__main__ import build_coverage

    tasks = ["task_graph__node_link__longest_path_length", "task_puzzles__maze__exit_reachability_label",
             "task_puzzles__cell_board__reachable_region_size"]
    for entry in build_coverage(tasks=tasks)["tasks"]:
        assert entry["status"] == "unsupported" and "path-like" in entry["reason"], entry


def test_cert_gate_requires_the_external_roots() -> None:
    """Review CONF-12: under ``CERT_GATE=1`` (the documented M1 gate) the leakage test 5 and the
    release-reward pin must run, so both roots must be set."""
    if os.environ.get("CERT_GATE") != "1":
        pytest.skip("not the M1 gate run (CERT_GATE unset)")
    assert os.environ.get("TRACE_TOOL_ROOT") and os.environ.get("TRACE_RLVR_ROOT")
