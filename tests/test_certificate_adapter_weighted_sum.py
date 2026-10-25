"""weighted_sum adapter (lever side torque, DESIGN 3.3, 9): oracle, round trip, corruptions,
equivalence, the wrong-operand property and edge cases.

Fixtures are 16 lever instances built with ``trace_tasks.build_dataset`` (sampling_seed 7001,
code_hash ``cert-v1``): the full persisted TrainInstance row plus the trace-record views the
adapter reads (``scene_ir``, ``execution_trace``, a slim ``render_spec``; images, query spec,
render map and label copies stripped, image size kept), together with the oracle and SFT
object the adapter produced at generation time.

Regenerate the fixtures (only after an intentional adapter or upstream change):
    python -m tests.test_certificate_adapter_weighted_sum --regen <scratch_dir>
Sweep round trip and corruptions over freshly generated instances:
    python -m tests.test_certificate_adapter_weighted_sum --sweep <scratch_dir> [count] [sampling_seed]
or over already built datasets (every lever record under the directory):
    TRACE_CERT_SWEEP_DIR=<dir> pytest tests/test_certificate_adapter_weighted_sum.py -k sweep
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Tuple

import pytest

from trace_tasks.experimental.certificates import registry
from trace_tasks.experimental.certificates.adapters import weighted_sum
from trace_tasks.experimental.certificates.geometry import round_box_half_up
from trace_tasks.experimental.certificates.registry import Unsupported
from trace_tasks.experimental.certificates.scoring import score_certificate_response
from trace_tasks.experimental.certificates.types import (
    CertificateExtension,
    QuarantineError,
    ScoreResult,
    TraceRecord,
    certificate_extension_id,
    read_trace_records,
)

TASK = weighted_sum.TASK_ID
FIXTURE = Path(__file__).parent / "fixtures" / "certificates" / "weighted_sum" / "lever_side_torque_7001.json"
SAMPLING_SEED = 7001
FIXTURE_COUNT = 16
CODE_HASH = "cert-v1"
TRACE_KEEP = ("instance_id", "scene_ir", "execution_trace", "render_spec")  # what the adapter reads
RENDER_SPEC_KEEP = ("scene_variant", "canvas_width", "canvas_height", "domain", "scene_id", "task_id", "query_id")
DESIGN9_CORRUPTIONS = {
    "wrong_weight_consistent", "wrong_distance_consistent", "wrong_product_answer_kept", "omit_term",
    "duplicate_term", "wrong_side_extra_term", "index_out_of_range", "negative_index", "boxes_shifted_off_target",
}

ADAPTER = registry.adapter_by_id("weighted_sum.lever_side_torque")
SIGNALS = ("A", "G", "M", "F")


# ------------------------------------------------------------------------------------------
# helpers


def record_of(item: Mapping[str, Any]) -> TraceRecord:
    return TraceRecord(train=item["train"], trace=item["trace"], image_size=tuple(item["image_size"]))


def make_extension(record: TraceRecord, oracle: Mapping[str, Any]) -> CertificateExtension:
    identity = ADAPTER.identity()
    base = {
        "instance_id": record.instance_id,
        "trace_record_hash": record.trace_record_hash,
        "task": record.task,
        "query_id": record.query_id,
        "instance_seed": record.instance_seed,
        "instance_version": record.instance_version,
    }
    ext_id = certificate_extension_id(
        instance_id=base["instance_id"],
        trace_record_hash=base["trace_record_hash"],
        adapter_id=identity["id"],
        adapter_version=identity["version"],
        schema_id=identity["schema_id"],
    )
    return CertificateExtension(
        certificate_extension_id=ext_id,
        base=base,
        adapter=identity,
        labels={
            "answer_gt": record.answer_gt,
            "annotation_gt": record.annotation_gt,
            "reward_contract": record.reward_contract,
        },
        public={"image_size": list(record.source_size())},
        oracle=dict(oracle),
    )


def score(response: Any, ext: CertificateExtension, *, frame: str = "px_source", **kwargs: Any) -> ScoreResult:
    text = response if isinstance(response, str) else json.dumps(response)
    return score_certificate_response(text, {"instance_id": ext.instance_id}, ext, coord_frame=frame, **kwargs)


def value(result: ScoreResult, key: str) -> float:
    if key in SIGNALS:
        return {"A": result.answer, "G": result.grounding, "M": result.semantics, "F": result.format}[key]
    return next(c.score for c in result.checks if c.check_id == key)


def families(result: ScoreResult) -> Dict[str, float]:
    ids = ADAPTER.m_families + ADAPTER.local_check_ids
    return {c.check_id: c.score for c in result.checks if c.check_id in ids}


def matches(expected: Any, actual: float, baseline: float) -> bool:
    if expected == "pass":
        return actual == 1.0
    if expected == "fail":
        return actual < 1.0
    if expected == "unchanged":
        return abs(actual - baseline) <= 0.01
    return math.isclose(actual, float(expected), abs_tol=1e-9)


def sft_obj(oracle: Mapping[str, Any], seed: int = 0) -> Dict[str, Any]:
    annotation, certificate, answer = ADAPTER.sft_certificate(oracle, seed)
    return {"annotation": annotation, "certificate": certificate, "answer": answer}


def corruption_failures(record: TraceRecord, oracle: Mapping[str, Any], ext: CertificateExtension) -> List[str]:
    """Every (corruption, key) whose score does not match its expectation."""
    obj = sft_obj(oracle, record.instance_seed)
    baseline = score(obj, ext)
    failures = []
    for corruption in ADAPTER.corruptions(oracle, obj, record.instance_seed):
        result = score(corruption.response_obj, ext)
        for key, expected in corruption.expected.items():
            if not matches(expected, value(result, key), value(baseline, key)):
                failures.append(f"{corruption.name}:{key}={value(result, key):.3f} expected {expected}")
    return failures


def slim(train: Mapping[str, Any], trace: Mapping[str, Any], image_size: Tuple[int, int]) -> Dict[str, Any]:
    kept = {key: copy.deepcopy(trace[key]) for key in TRACE_KEEP if key in trace}
    kept["render_spec"] = {k: v for k, v in trace.get("render_spec", {}).items() if k in RENDER_SPEC_KEEP}
    return {"train": copy.deepcopy(dict(train)), "trace": kept, "image_size": list(image_size)}


# synthetic scenes for edge cases (canvas 1280x640, fulcrum at x=640, 66 px slots as rendered)


def block_box(side: str, distance: int, cx: float = 640.0) -> List[float]:
    x = cx + (distance if side == "right" else -distance) * 66.0
    return [x - 29.0, 278.5, x + 29.0, 330.5]


def synthetic(query_side: str, specs: List[Tuple[str, int, int]]) -> Tuple[Dict[str, Any], CertificateExtension]:
    """Oracle + extension for blocks ``(side, distance, weight)``; blocks on ``query_side`` are relevant."""
    blocks = [
        {"oid": oid, "bbox": block_box(side, d), "side": side, "weight": w, "distance": d,
         "relevant": side == query_side}
        for oid, (side, d, w) in enumerate(specs)
    ]
    oracle = {"query_side": query_side, "blocks": blocks,
              "total": sum(b["weight"] * b["distance"] for b in blocks if b["relevant"])}
    identity = ADAPTER.identity()
    base = {"instance_id": "blake3:" + "a" * 64, "trace_record_hash": "blake3:" + "b" * 64, "task": TASK,
            "query_id": "single", "instance_seed": 5, "instance_version": "v0"}
    ext = CertificateExtension(
        certificate_extension_id=certificate_extension_id(
            instance_id=base["instance_id"], trace_record_hash=base["trace_record_hash"],
            adapter_id=identity["id"], adapter_version=identity["version"], schema_id=identity["schema_id"],
        ),
        base=base,
        adapter=identity,
        labels={
            "answer_gt": {"type": "integer", "value": oracle["total"]},
            "annotation_gt": {"type": "bbox_set", "value": [b["bbox"] for b in blocks if b["relevant"]]},
            "reward_contract": {
                "reward_contract_version": "v0",
                "answer": {"id": "answer_exact_match_v0", "type": "integer"},
                "annotation": {"id": "bbox_set_soft_iou_v0", "type": "bbox_set"},
            },
        },
        public={"image_size": [1280, 640]},
        oracle=oracle,
    )
    return oracle, ext


# ------------------------------------------------------------------------------------------
# fixtures


@pytest.fixture(scope="module")
def items() -> List[Dict[str, Any]]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["generator"]["sampling_seed"] == SAMPLING_SEED
    assert len(payload["records"]) == FIXTURE_COUNT
    return payload["records"]


@pytest.fixture(scope="module")
def cases(items: List[Dict[str, Any]]) -> List[Tuple[TraceRecord, Dict[str, Any], CertificateExtension]]:
    out = []
    for item in items:
        record = record_of(item)
        oracle = ADAPTER.build_oracle(record)
        out.append((record, oracle, make_extension(record, oracle)))
    return out


# ------------------------------------------------------------------------------------------
# definition, registry, oracle


def test_adapter_is_registered_and_valid() -> None:
    type(ADAPTER).validate_definition()
    assert registry.resolve(TASK, "single", "v0") is ADAPTER
    assert isinstance(registry.resolve(TASK, "other", "v0"), Unsupported)
    assert isinstance(registry.resolve(TASK, "single", "v1"), Unsupported)
    assert ADAPTER.identity() == {
        "id": "weighted_sum.lever_side_torque",
        "version": "1.0.0",
        "schema_id": "cert.weighted_sum.v1",
        "verifier_version": "1.1.0",
    }
    assert ADAPTER.m_families == ("ws.terms.truth", "ws.weight.truth", "ws.distance.truth", "ws.product.truth")
    assert ADAPTER.list_key == "terms" and ADAPTER.list_cap == 8 and not ADAPTER.cert_fields
    assert set(ADAPTER.element_fields) == {"annotation_index", "weight", "distance", "product"}


def test_fixture_oracles_and_sft_targets_are_reproduced(items, cases) -> None:
    for item, (record, oracle, _) in zip(items, cases):
        assert oracle == item["oracle"]
        assert sft_obj(oracle, record.instance_seed) == item["sft"]
        assert "lever_weight" not in json.dumps(oracle)  # entity ids stay out of the oracle
        relevant = [b for b in oracle["blocks"] if b["relevant"]]
        assert 1 <= len(relevant) <= 3 and all(b["side"] == oracle["query_side"] for b in relevant)
        assert oracle["total"] == record.answer_gt["value"]
        xs = [box[0] for box in item["sft"]["annotation"]]
        assert xs == sorted(xs)  # reading order
        assert list(item["sft"]["certificate"]) == ["kind", "terms"]


def test_sft_target_is_independent_of_rng_seed(cases) -> None:
    for record, oracle, _ in cases:
        assert sft_obj(oracle, 0) == sft_obj(oracle, record.instance_seed) == sft_obj(oracle, 12345)


# ------------------------------------------------------------------------------------------
# oracle round trip (DESIGN 9)


def test_oracle_round_trip_unrounded(cases) -> None:
    for record, oracle, ext in cases:
        result = score(sft_obj(oracle, record.instance_seed), ext, shown_frame_diagnostics=True)
        assert (result.answer, result.grounding, result.semantics, result.format) == (1.0, 1.0, 1.0, 1.0)
        assert result.joint == 1.0 and result.error_class == "correct"
        assert set(families(result).values()) == {1.0}
        assert result.diagnostics["G_id"] == 1.0
        assert result.diagnostics["M_shown"] == 1.0
        assert result.diagnostics["adapter"]["ws.operand_read_acc"] == 1.0
        assert result.diagnostics["adapter"]["ws.wrong_side_reads"]["terms"] == 0


def test_rounded_sft_target_resolves_every_box(cases) -> None:
    for record, oracle, ext in cases:
        obj = sft_obj(oracle, record.instance_seed)
        obj["annotation"] = [round_box_half_up(box) for box in obj["annotation"]]
        result = score(json.dumps(obj, separators=(",", ":")), ext)
        assert (result.answer, result.semantics, result.format) == (1.0, 1.0, 1.0)
        statuses = [r["status"] for r in result.diagnostics["identity"][""]]
        assert statuses and set(statuses) == {"resolved"}
        assert 0.9 < result.grounding <= 1.0  # logged; a 0.5 px edge shift costs a little IoU


def test_rel1000_round_trip(cases) -> None:
    for record, oracle, ext in cases:
        width, height = record.source_size()
        obj = sft_obj(oracle, record.instance_seed)
        obj["annotation"] = [
            [b[0] * 1000 / width, b[1] * 1000 / height, b[2] * 1000 / width, b[3] * 1000 / height]
            for b in obj["annotation"]
        ]
        result = score(obj, ext, frame="rel1000")
        assert result.answer == 1.0 and result.semantics == 1.0 and result.format == 1.0
        assert result.grounding > 0.999


# ------------------------------------------------------------------------------------------
# counterexamples (DESIGN 9)


def test_every_corruption_fails_exactly_its_checks(cases) -> None:
    seen: Counter = Counter()
    complete = set(ADAPTER.m_families + ADAPTER.local_check_ids + SIGNALS)
    for record, oracle, ext in cases:
        corruptions = ADAPTER.corruptions(oracle, sft_obj(oracle, record.instance_seed), record.instance_seed)
        for corruption in corruptions:
            assert set(corruption.expected) == complete, corruption.name
            seen[corruption.name] += 1
        assert corruption_failures(record, oracle, ext) == []
    assert DESIGN9_CORRUPTIONS <= set(seen)
    assert seen["swap_annotation_index"] > 0 and seen["wrong_side_swap"] > 0


def test_corruptions_are_deterministic_and_leave_the_input_alone(cases) -> None:
    record, oracle, _ = cases[0]
    obj = sft_obj(oracle)
    before = copy.deepcopy(obj)
    first = ADAPTER.corruptions(oracle, obj, 11)
    second = ADAPTER.corruptions(oracle, obj, 11)
    assert obj == before
    def flat(items: List[Any]) -> List[Any]:
        return [(c.name, c.response_obj, c.expected) for c in items]

    assert flat(first) == flat(second)


def test_operand_corruptions_keep_g_and_lower_m(cases) -> None:
    """Go/no-go 1b: at least three corruption types leave G unchanged while lowering M."""
    for record, oracle, ext in cases:
        obj = sft_obj(oracle, record.instance_seed)
        kinds = set()
        for corruption in ADAPTER.corruptions(oracle, obj, record.instance_seed):
            result = score(corruption.response_obj, ext)
            if abs(result.grounding - 1.0) < 0.01 and result.semantics < 1.0:
                kinds.add(corruption.name)
        assert len(kinds) >= 3, kinds


def test_answer_sum_slip_is_answer_inconsistent(cases) -> None:
    record, oracle, ext = cases[0]
    obj = sft_obj(oracle)
    obj["answer"] += 1
    result = score(obj, ext)
    assert result.semantics == 1.0 and result.answer == 0.0 and result.error_class == "answer_inconsistent"
    assert value(result, "ws.answer.local") == 0.0


def test_error_classes_of_operand_and_grounding_failures(cases) -> None:
    record, oracle, ext = cases[0]
    by_name = {c.name: c for c in ADAPTER.corruptions(oracle, sft_obj(oracle), 3)}
    assert score(by_name["wrong_weight_consistent"].response_obj, ext).error_class == "operand"
    assert score(by_name["wrong_product_answer_kept"].response_obj, ext).error_class == "computation"
    assert score(by_name["boxes_shifted_off_target"].response_obj, ext).error_class == "grounding"
    assert score(by_name["wrong_side_extra_term"].response_obj, ext).error_class == "grounding"


# ------------------------------------------------------------------------------------------
# equivalence (DESIGN 9)


def _reorder_keys(value: Any, rng: random.Random) -> Any:
    if isinstance(value, dict):
        keys = list(value)
        rng.shuffle(keys)
        return {key: _reorder_keys(value[key], rng) for key in keys}
    if isinstance(value, list):
        return [_reorder_keys(item, rng) for item in value]
    return value


def _same_scores(a: ScoreResult, b: ScoreResult, *, with_format: bool = True) -> None:
    assert (a.answer, a.grounding, a.semantics) == (b.answer, b.grounding, b.semantics)
    assert families(a) == families(b)
    if with_format:
        assert a.format == b.format


def test_permuted_annotation_with_rebound_indices(cases) -> None:
    rng = random.Random(7)
    for record, oracle, ext in cases:
        obj = sft_obj(oracle)
        reference = score(obj, ext)
        for _ in range(4):
            order = list(range(len(obj["annotation"])))
            rng.shuffle(order)
            position = {old: new for new, old in enumerate(order)}
            permuted = copy.deepcopy(obj)
            permuted["annotation"] = [obj["annotation"][old] for old in order]
            for term in permuted["certificate"]["terms"]:
                term["annotation_index"] = position[term["annotation_index"]]
            _same_scores(score(permuted, ext), reference)


def test_term_order_and_key_order_are_never_scored(cases) -> None:
    rng = random.Random(8)
    for record, oracle, ext in cases:
        obj = sft_obj(oracle)
        reference = score(obj, ext)
        shuffled = copy.deepcopy(obj)
        rng.shuffle(shuffled["certificate"]["terms"])
        _same_scores(score(shuffled, ext), reference)
        _same_scores(score(_reorder_keys(obj, rng), ext), reference)


def test_prose_and_fences_change_only_f(cases) -> None:
    record, oracle, ext = cases[0]
    text = json.dumps(sft_obj(oracle))
    reference = score(text, ext)
    _same_scores(score("Reading the blocks left to right.\n" + text, ext), reference)
    _same_scores(score("```json\n" + text + "\n```", ext), reference)
    trailing = score(text + "\nThat is the torque.", ext)
    _same_scores(trailing, reference, with_format=False)
    assert trailing.format == 0.0 and reference.format == 1.0


# ------------------------------------------------------------------------------------------
# property: wrong operands with self-consistent arithmetic never reach M = 1


def test_wrong_operands_with_consistent_arithmetic_give_m_below_one(cases) -> None:
    rng = random.Random(2026)
    trials = 0
    for record, oracle, ext in cases:
        obj = sft_obj(oracle)
        terms = obj["certificate"]["terms"]
        for _ in range(12):
            corrupted = copy.deepcopy(obj)
            chosen = rng.sample(range(len(terms)), rng.randint(1, len(terms)))
            for index in chosen:
                term = corrupted["certificate"]["terms"][index]
                field = rng.choice(("weight", "distance", "both"))
                if field in ("weight", "both"):
                    term["weight"] = rng.choice([v for v in range(1, 10) if v != term["weight"]])
                if field in ("distance", "both"):
                    term["distance"] = rng.choice([v for v in range(1, 9) if v != term["distance"]])
                term["product"] = term["weight"] * term["distance"]
            corrupted["answer"] = sum(t["product"] for t in corrupted["certificate"]["terms"])
            result = score(corrupted, ext)
            assert result.semantics < 1.0 and result.grounding == 1.0 and result.format == 1.0
            assert value(result, "ws.product.local") == 1.0 and value(result, "ws.answer.local") == 1.0
            trials += 1
    assert trials == 12 * FIXTURE_COUNT


def test_swapped_operands_with_the_same_product_get_no_product_credit() -> None:
    """3x2 read as 2x3: product and answer stay right (A=1) but the reads are false (M<1)."""
    oracle, ext = synthetic("left", [("left", 2, 3), ("left", 5, 1), ("right", 1, 4)])
    obj = sft_obj(oracle)
    term = next(t for t in obj["certificate"]["terms"] if (t["weight"], t["distance"]) == (3, 2))
    term["weight"], term["distance"] = 2, 3
    result = score(obj, ext)
    assert result.answer == 1.0 and result.grounding == 1.0 and result.format == 1.0
    assert value(result, "ws.weight.truth") == 0.5 and value(result, "ws.distance.truth") == 0.5
    assert value(result, "ws.product.truth") == 0.5 and result.semantics < 1.0


# ------------------------------------------------------------------------------------------
# edge cases (DESIGN 2.1, 2.3, 3.3)


def test_mirror_twin_on_the_wrong_side_is_resolved_but_not_credited() -> None:
    """Identical-looking blocks across the fulcrum: binding is by geometry, never by label."""
    oracle, ext = synthetic("left", [("left", 1, 2), ("right", 1, 2), ("right", 4, 5)])
    twin = next(b for b in oracle["blocks"] if b["side"] == "right" and b["distance"] == 1)
    obj = {"annotation": [twin["bbox"]],
           "certificate": {"kind": "weighted_sum",
                           "terms": [{"annotation_index": 0, "weight": 2, "distance": 1, "product": 2}]},
           "answer": 2}
    result = score(obj, ext)
    assert result.answer == 1.0 and result.semantics == 0.0 and result.grounding == 0.0
    assert result.error_class == "grounding"
    assert result.diagnostics["identity"][""][0]["status"] == "resolved"
    reads = result.diagnostics["adapter"]["ws.wrong_side_reads"]
    assert reads == {"terms": 1, "weight_correct": 1, "distance_correct": 1}


def test_digit_tight_box_resolves_and_straddling_box_does_not() -> None:
    oracle, ext = synthetic("right", [("right", 2, 4), ("right", 3, 1), ("left", 2, 6)])
    a, b = (blk["bbox"] for blk in oracle["blocks"][:2])
    tight = [a[0] + 19.0, a[1] + 13.0, a[2] - 19.0, a[3] - 13.0]  # around the digit, inside the block
    straddle = [(a[0] + a[2]) / 2.0, a[1], (b[0] + b[2]) / 2.0, a[3]]  # half of each neighbour
    terms = [{"annotation_index": 0, "weight": 4, "distance": 2, "product": 8},
             {"annotation_index": 1, "weight": 1, "distance": 3, "product": 3}]
    result = score({"annotation": [tight, straddle],
                    "certificate": {"kind": "weighted_sum", "terms": terms}, "answer": 11}, ext)
    statuses = [r["status"] for r in result.diagnostics["identity"][""]]
    assert statuses == ["resolved", "unresolved"]
    assert value(result, "ws.terms.truth") == 0.5 and result.semantics == 0.5


def test_shift_tolerance_matches_the_identity_rule() -> None:
    oracle, ext = synthetic("left", [("left", 3, 2)])
    box = oracle["blocks"][0]["bbox"]
    term = [{"annotation_index": 0, "weight": 2, "distance": 3, "product": 6}]
    for dx, resolved in ((19.0, True), (20.0, False), (-19.0, True), (-20.0, False)):
        shifted = [box[0] + dx, box[1], box[2] + dx, box[3]]
        result = score({"annotation": [shifted], "certificate": {"kind": "weighted_sum", "terms": term},
                        "answer": 6}, ext)
        assert (result.semantics == 1.0) is resolved, dx


@pytest.mark.parametrize("index", [-1, -3, 1, 99])
def test_out_of_range_index_never_binds(index: int) -> None:
    oracle, ext = synthetic("left", [("left", 1, 7)])
    obj = sft_obj(oracle)
    obj["certificate"]["terms"][0]["annotation_index"] = index
    result = score(obj, ext)
    assert result.semantics == 0.0 and result.format == 0.0 and result.answer == 1.0
    assert value(result, "ws.ref.local") == 0.0
    assert result.checks[0].detail["terms"][0]["status"] == "invalid_element"


def test_term_cap_is_a_deterministic_budget_envelope(cases) -> None:
    record, oracle, ext = cases[0]
    obj = sft_obj(oracle)
    obj["certificate"]["terms"] = [dict(obj["certificate"]["terms"][0]) for _ in range(9)]
    result = score(obj, ext)
    assert result.diagnostics["envelope_error"] == "budget" and result.error_class == "envelope"
    assert result.semantics == 0.0 and result.format == 0.0
    assert all(not c.valid_input for c in result.checks)


TERM = {"annotation_index": 0, "weight": 1, "distance": 1, "product": 1}


@pytest.mark.parametrize(
    "certificate, annotation, fmt",
    [
        ({"kind": "weighted_sum", "terms": []}, None, 1.0),  # schema-valid empty claim
        ({"kind": "weighted_sum", "terms": [{"annotation_index": 0}]}, None, 0.0),
        ({"kind": "weighted_sum", "terms": [dict(TERM, weight="3")]}, None, 0.0),
        ({"kind": "weighted_sum", "terms": [dict(TERM, side="left")]}, None, 0.0),
        ({"kind": "weighted_sum", "terms": [7, None, "x"]}, None, 0.0),
        ({"kind": "weighted_sum", "terms": [TERM]}, {"bboxes": "x"}, 0.0),
        ({"kind": "weighted_sum", "terms": [dict(TERM, weight=0, distance=-2, product=0)]}, [], 0.0),
        ({"kind": "selected_set", "terms": []}, None, 0.0),
        ({"kind": "weighted_sum"}, None, 0.0),
        ({"kind": "weighted_sum", "terms": {}}, None, 0.0),
    ],
)
def test_malformed_certificates_score_without_raising(cases, certificate, annotation, fmt) -> None:
    record, oracle, ext = cases[0]
    obj = sft_obj(oracle)
    obj["certificate"] = certificate
    if annotation is not None:
        obj["annotation"] = annotation
    result = score(obj, ext)
    assert result.semantics == 0.0 and result.format == fmt and result.joint == 0.0
    for name in SIGNALS:
        assert 0.0 <= value(result, name) <= 1.0


def test_fuzzed_term_values_never_raise(cases) -> None:
    rng = random.Random(99)
    pool: List[Any] = [0, 1, 2, -1, 9, 10, 72, 2**70, 1.0, 2.5, True, False, None, "3", [1], {"a": 1}]
    record, oracle, ext = cases[0]
    for _ in range(300):
        obj = sft_obj(oracle)
        for term in obj["certificate"]["terms"]:
            for key in rng.sample(list(term), rng.randint(0, len(term))):
                term[key] = rng.choice(pool)
            if rng.random() < 0.2:
                term.pop(rng.choice(list(term)))
        if rng.random() < 0.3:
            obj["answer"] = rng.choice(pool)
        result = score(obj, ext)
        assert len(families(result)) == 7
        for name in SIGNALS:
            assert 0.0 <= value(result, name) <= 1.0


# ------------------------------------------------------------------------------------------
# export invariants, split cells, semantic keys


def _mutated(record: TraceRecord, mutate: Callable[[Dict[str, Any], Dict[str, Any]], None]) -> TraceRecord:
    train, trace = copy.deepcopy(dict(record.train)), copy.deepcopy(dict(record.trace))
    mutate(train, trace)
    return TraceRecord(train=train, trace=trace, image_size=record.image_size)


def _weights(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [e for e in trace["scene_ir"]["entities"] if e["entity_type"] == "physics_lever_weight"]


def _first(trace: Dict[str, Any], relevant: bool) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    index = next(i for i, e in enumerate(_weights(trace)) if e["meta"]["relevant_to_task"] is relevant)
    return index, _weights(trace)[index], trace["execution_trace"]["weight_specs"][index]


def _set_both(trace: Dict[str, Any], relevant: bool, **changes: Any) -> None:
    _, entity, spec = _first(trace, relevant)
    for key, val in changes.items():
        entity["meta"][key] = val
        spec["relevant_to_query" if key == "relevant_to_task" else key] = val


def _flip_distractor(trace: Dict[str, Any]) -> None:
    _set_both(trace, False, relevant_to_task=True)


def _overlap(trace: Dict[str, Any]) -> None:
    box = _first(trace, True)[1]["bbox_px"]
    _first(trace, False)[1]["bbox_px"] = [box[0] + 10.0, box[1], box[2] + 10.0, box[3]]


def _bump(mapping: Dict[str, Any], key: str) -> None:
    mapping[key] += 1


def _flip_query_side(trace: Dict[str, Any]) -> None:
    execution = trace["execution_trace"]
    execution["query_side"] = "left" if execution["query_side"] == "right" else "right"


def _drop_weights(trace: Dict[str, Any]) -> None:
    entities = trace["scene_ir"]["entities"]
    trace["scene_ir"]["entities"] = [e for e in entities if e["entity_type"] != "physics_lever_weight"]


def _spec_value(trace: Dict[str, Any]) -> None:
    spec = _first(trace, False)[2]
    spec["value"] = spec["value"] % 9 + 1


def _same_slot(trace: Dict[str, Any]) -> None:
    relevant = [e for e in _weights(trace) if e["meta"]["relevant_to_task"]]
    distance = relevant[0]["meta"]["distance_units"]
    index = _weights(trace).index(relevant[1])
    relevant[1]["meta"]["distance_units"] = distance
    trace["execution_trace"]["weight_specs"][index]["distance_units"] = distance


QUARANTINE_CASES: Dict[str, Tuple[str, Callable[[Dict[str, Any], Dict[str, Any]], None]]] = {
    "specs_reordered": ("weight_specs_misaligned", lambda tr, t: t["execution_trace"]["weight_specs"].reverse()),
    "spec_dropped": ("weight_specs_misaligned", lambda tr, t: t["execution_trace"]["weight_specs"].pop()),
    "spec_value": ("weight_specs_misaligned", lambda tr, t: _spec_value(t)),
    "answer_gt": ("total_mismatch", lambda tr, t: _bump(tr["answer_gt"], "value")),
    "target_answer": ("total_mismatch", lambda tr, t: _bump(t["execution_trace"], "target_answer")),
    "relevant_flag": ("relevant_not_queried_side", lambda tr, t: _flip_distractor(t)),
    "query_side": ("side_mismatch", lambda tr, t: _flip_query_side(t)),
    "missing_value": ("missing_field", lambda tr, t: _weights(t)[0]["meta"].pop("value")),
    "missing_meta": ("missing_field", lambda tr, t: _weights(t)[0].pop("meta")),
    "placeholder": ("missing_weight", lambda tr, t: _set_both(t, False, missing=True)),
    "weight_zero": ("operand_out_of_range", lambda tr, t: _set_both(t, False, value=0)),
    "distance_nine": ("operand_out_of_range", lambda tr, t: _set_both(t, False, distance_units=9)),
    "bad_bbox": ("bad_bbox", lambda tr, t: _weights(t)[0].update(bbox_px=list(reversed(_weights(t)[0]["bbox_px"])))),
    "overlap": ("overlapping_blocks", lambda tr, t: _overlap(t)),
    "same_slot": ("duplicate_slot", lambda tr, t: _same_slot(t)),
    "annotation_gt": ("annotation_gt_mismatch", lambda tr, t: tr["annotation_gt"]["value"].pop()),
    "no_weights": ("missing_weights", lambda tr, t: _drop_weights(t)),
    "query_id": ("unsupported_task", lambda tr, t: tr.update(query_id="other")),
    "task": ("unsupported_task", lambda tr, t: tr.update(task="task_physics__lever__missing_weight_balance_value")),
}


@pytest.mark.parametrize("name", sorted(QUARANTINE_CASES))
def test_export_invariants_quarantine(cases, name: str) -> None:
    reason, mutate = QUARANTINE_CASES[name]
    record = next(r for r, o, _ in cases if sum(b["relevant"] for b in o["blocks"]) >= 2
                  and any(not b["relevant"] for b in o["blocks"]))
    with pytest.raises(QuarantineError) as excinfo:
        ADAPTER.build_oracle(_mutated(record, mutate))
    assert excinfo.value.reason == reason


def test_split_cells_and_semantic_keys(cases) -> None:
    cells = Counter()
    keys = set()
    for record, oracle, _ in cases:
        cell = ADAPTER.split_cell(record)
        assert cell == record.execution_trace["scene_variant"]
        cells[cell] += 1
        key = ADAPTER.semantic_key(record)
        assert key.startswith("blake3:") and key == ADAPTER.semantic_key(record)
        shuffled = _mutated(record, lambda tr, t: t["scene_ir"]["entities"].reverse())
        assert ADAPTER.semantic_key(shuffled) == key  # sorted blocks: entity order is irrelevant
        keys.add(key)
    assert len(keys) == len(cases)
    assert set(cells) <= set(weighted_sum.TRAIN_CELLS) | set(weighted_sum.HELDOUT_CELLS)
    assert set(weighted_sum.TRAIN_CELLS).isdisjoint(weighted_sum.HELDOUT_CELLS)
    assert weighted_sum.HELDOUT_CELLS == {"textured_beam": "render"}
    record = cases[0][0]
    for bad, reason in (("bent_beam", "unknown_scene_variant"), (None, "missing_field")):
        mutated = _mutated(record, lambda tr, t: (t["execution_trace"].update(scene_variant=bad),
                                                  t["scene_ir"]["relations"].pop("scene_variant"),
                                                  t["render_spec"].pop("scene_variant")))
        with pytest.raises(QuarantineError) as excinfo:
            ADAPTER.split_cell(mutated)
        assert excinfo.value.reason == reason
    mismatch = _mutated(record, lambda tr, t: t["render_spec"].update(scene_variant="textured_beam"
                                                                       if ADAPTER.split_cell(record) != "textured_beam"
                                                                       else "center_fulcrum"))
    with pytest.raises(QuarantineError):
        ADAPTER.split_cell(mismatch)


# ------------------------------------------------------------------------------------------
# prompt pieces (DESIGN 4)


def test_prompt_pieces_are_fixed_and_self_consistent(cases) -> None:
    assert ADAPTER.example_json() == (
        '{"annotation":[[192,282,250,334],[326,282,384,334]],"certificate":{"kind":"weighted_sum","terms":'
        '[{"annotation_index":0,"weight":3,"distance":2,"product":6},'
        '{"annotation_index":1,"weight":2,"distance":3,"product":6}]},"answer":12}'
    )
    example = ADAPTER.example_object()
    terms = example["certificate"]["terms"]
    assert all(t["product"] == t["weight"] * t["distance"] for t in terms)
    assert example["answer"] == sum(t["product"] for t in terms)
    assert sorted(t["annotation_index"] for t in terms) == list(range(len(example["annotation"])))
    assert all(0 <= v <= 1000 for box in example["annotation"] for v in box)
    line = ADAPTER.certificate_line()
    for token in ('"weighted_sum"', '"terms"', '"annotation_index"', '"weight"', '"distance"', '"product"',
                  "number printed on the block", "distance mark below the block", "p = w × d"):
        assert token in line
    for token in ("lever_weight", "left", "right", "fulcrum", "side"):
        assert token not in line  # the side is public only through the question
    for _, oracle, _ in cases:
        assert ADAPTER.certificate_line() == line and str(oracle["total"]) not in line


def test_prompt_pieces_and_public_schema_match_the_bundle() -> None:
    """One source of truth: the versioned bundle that prompts.py renders, and the public schema."""
    from trace_tasks.experimental.certificates import prompts

    assert ADAPTER.certificate_line() == prompts.certificate_line("weighted_sum")
    assert ADAPTER.example_json() == prompts.example_json("weighted_sum")
    schema = prompts.public_schema("weighted_sum")
    certificate = schema["properties"]["certificate"]
    assert certificate["properties"]["kind"] == {"const": ADAPTER.kind}
    assert set(certificate["required"]) == {"kind", ADAPTER.list_key} and not certificate["additionalProperties"]
    terms = certificate["properties"][ADAPTER.list_key]
    assert terms["maxItems"] == ADAPTER.list_cap
    assert set(terms["items"]["required"]) == set(ADAPTER.element_fields) == set(terms["items"]["properties"])
    assert terms["items"]["additionalProperties"] is False


# ------------------------------------------------------------------------------------------
# maintenance: fixture regeneration and the larger sweep


def generate_records(work_dir: Path, count: int, sampling_seed: int, workers: int = 1) -> List[TraceRecord]:
    from trace_tasks import BuildConfig, BuildTaskConfig, build_dataset

    config = BuildConfig(
        output_root=str(work_dir),
        dataset_name=f"cert_weighted_sum_{sampling_seed}",
        instance_version="v0",
        image_format="png",
        tasks=[BuildTaskConfig(task_id=TASK, count=count)],
        max_attempts_per_instance=100,
        sampling_seed=sampling_seed,
        workers=workers,
    )
    return read_trace_records(build_dataset(config, code_hash=CODE_HASH))


def build_fixture_payload(work_dir: Path) -> Dict[str, Any]:
    records = generate_records(work_dir, FIXTURE_COUNT, SAMPLING_SEED)
    out = []
    for record in records:
        oracle = ADAPTER.build_oracle(record)
        item = slim(record.train, record.trace, record.source_size())
        item["oracle"] = oracle
        item["sft"] = sft_obj(oracle, record.instance_seed)
        out.append(item)
    return {
        "generator": {
            "task": TASK,
            "sampling_seed": SAMPLING_SEED,
            "count": FIXTURE_COUNT,
            "code_hash": CODE_HASH,
            "adapter": ADAPTER.identity(),
            "trace_keys_kept": list(TRACE_KEEP),
            "render_spec_keys_kept": list(RENDER_SPEC_KEEP),
            "command": "python -m tests.test_certificate_adapter_weighted_sum --regen <scratch_dir>",
        },
        "records": out,
    }


def run_sweep(records: List[TraceRecord]) -> Dict[str, Any]:
    """Round trip (unrounded and rounded SFT target) and every corruption on each lever record."""
    stats: Counter = Counter()
    failures: Dict[str, List[str]] = {}
    quarantined: Counter = Counter()
    cells: Counter = Counter()
    for record in records:
        try:
            oracle = ADAPTER.build_oracle(record)
        except QuarantineError as exc:
            quarantined[exc.reason] += 1
            continue
        cells[ADAPTER.split_cell(record)] += 1
        ext = make_extension(record, oracle)
        obj = sft_obj(oracle, record.instance_seed)
        base = score(obj, ext, shown_frame_diagnostics=True)
        rounded = copy.deepcopy(obj)
        rounded["annotation"] = [round_box_half_up(box) for box in rounded["annotation"]]
        result = score(rounded, ext, shown_frame_diagnostics=True)
        resolved = all(r["status"] == "resolved" for r in result.diagnostics["identity"][""])
        outcomes = {
            "round_trip_unrounded": (base.answer, base.grounding, base.semantics, base.format) == (1, 1, 1, 1),
            "round_trip_rounded": (result.answer, result.semantics, result.format) == (1, 1, 1) and resolved,
            "m_shown_one": result.diagnostics["M_shown"] == 1.0,
        }
        for corruption in ADAPTER.corruptions(oracle, obj, record.instance_seed):
            corrupted = score(corruption.response_obj, ext)
            outcomes[f"corruption:{corruption.name}"] = all(
                matches(expected, value(corrupted, key), value(base, key))
                for key, expected in corruption.expected.items()
            )
        for name, ok in outcomes.items():
            stats[f"{name}:n"] += 1
            stats[f"{name}:ok"] += bool(ok)
            if not ok:
                failures.setdefault(name, []).append(record.instance_id)
    return {"records": len(records), "cells": dict(cells), "quarantined": dict(quarantined),
            "stats": dict(sorted(stats.items())), "failures": failures}


def _sweep_records(root: Path) -> List[TraceRecord]:
    dirs = sorted({path.parent for path in root.rglob("train_instances.jsonl")})
    return [record for path in dirs for record in read_trace_records(path) if record.task == TASK]


@pytest.mark.skipif(not os.environ.get("TRACE_CERT_SWEEP_DIR"), reason="set TRACE_CERT_SWEEP_DIR to run the sweep")
def test_sweep_built_datasets() -> None:
    records = _sweep_records(Path(os.environ["TRACE_CERT_SWEEP_DIR"]))
    if not records:
        pytest.skip("no lever side-torque records under TRACE_CERT_SWEEP_DIR")
    summary = run_sweep(records)
    print(json.dumps(summary["stats"], indent=1))
    assert not summary["failures"] and not summary["quarantined"]


if __name__ == "__main__":  # pragma: no cover - maintenance entry point
    if len(sys.argv) >= 3 and sys.argv[1] == "--regen":
        payload = build_fixture_payload(Path(sys.argv[2]))
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes, {len(payload['records'])} records)")
    elif len(sys.argv) >= 3 and sys.argv[1] == "--sweep":
        count = int(sys.argv[3]) if len(sys.argv) > 3 else 320
        seed = int(sys.argv[4]) if len(sys.argv) > 4 else 7091
        records = generate_records(Path(sys.argv[2]), count, seed, workers=8)
        print(json.dumps(run_sweep(records), indent=1))
    else:
        raise SystemExit(
            "usage: python -m tests.test_certificate_adapter_weighted_sum "
            "(--regen <scratch_dir> | --sweep <scratch_dir> [count] [sampling_seed])"
        )
