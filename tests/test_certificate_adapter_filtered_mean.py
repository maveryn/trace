"""``filtered_mean.table_filtered_column_mean`` adapter (DESIGN 3.2, 6, 9).

Fixtures: 16 ``task_charts__table__filtered_column_mean`` records built by TRACE's builder at
sampling_seed 7001 (code_hash ``cert-v1``), kept small: no images (the image size is kept), the
persisted trace reduced to the fields the adapter reads, ``table_cell`` entities stored as a
compact grid that expands back to the persisted dicts exactly, and ``prompt`` restored from
``prompt_variants``. Each record also carries the oracle, SFT object, split cell and semantic key
the adapter produced when the fixture was frozen.

Regenerate (only after an intentional change, and review the diff):
    python -m tests.test_certificate_adapter_filtered_mean --regen <scratch_dir>
Larger in-memory sweep (round trip + corruptions + property test; prints pass rates):
    python -m tests.test_certificate_adapter_filtered_mean --sweep <n> [<sampling_seed>]
"""

from __future__ import annotations

import copy
import functools
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pytest

from trace_tasks.core.reward_contracts import resolve_reward_contract
from trace_tasks.experimental.certificates import registry
from trace_tasks.experimental.certificates.adapters import filtered_mean as fm
from trace_tasks.experimental.certificates.adapters.base import Adapter, PublicContext
from trace_tasks.experimental.certificates.geometry import annotation_view, round_box_half_up, source_frame
from trace_tasks.experimental.certificates.identity import IdentityParams
from trace_tasks.experimental.certificates.parsing import validate_certificate
from trace_tasks.experimental.certificates.registry import Unsupported
from trace_tasks.experimental.certificates.scoring import score_certificate_response
from trace_tasks.experimental.certificates.types import (
    CertificateExtension,
    QuarantineError,
    ScoreResult,
    TraceRecord,
    canonical_digest,
    certificate_extension_id,
)

FIXTURE = Path(__file__).parent / "fixtures" / "certificates" / "filtered_mean" / "records.json"
TASK = fm.TASK_ID
QUERIES = tuple(fm.QUERY_FILTERS)
FIXTURE_SAMPLING_SEED = 7001
FIXTURE_COUNT = 16
CODE_HASH = "cert-v1"
# sampling_seed 7002, seed_index 3, interval query: degenerate interval [30, 30] (found offline).
DEGENERATE_INTERVAL = (7002, 3)
SWEEP_SAMPLING_SEED = 7901
ADAPTER = registry.load_adapter("filtered_mean")

REQUIRED_CORRUPTIONS = {  # DESIGN 9 plus ADAPTER_INTERFACE 3.5 for this adapter
    "wrong_target_value_consistent",
    "wrong_filter_value_consistent",
    "wrong_sum",
    "omit_row",
    "duplicate_row_same_index",
    "duplicate_row_new_box",
    "swap_filter_target",
    "wrong_pairing",
    "extra_row",
    "boundary_value_row",
    "wrong_count",
    "index_out_of_range",
    "negative_index",
    "box_shift_row",
    "box_straddle",
}


# ------------------------------------------------------------------------------------------
# helpers


def load_fixture() -> Dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# The fixture keeps only the persisted trace fields the adapter reads. ``table_cell`` entities
# are stored as a grid ``[role, row_index, column_index, x0, y0, x1, y1, value]`` that expands
# back to the persisted entity dicts exactly (minus ``text_bbox_px``); regen asserts this.
_ROLES = {"h": "header", "l": "row_label", "v": "value"}
_RELATION_KEYS = (
    "query_id", "scene_variant", "filter_variant", "filter_column", "filter_column_index", "target_column",
    "target_column_index", "selected_row_indices", "row_count", "numeric_column_count", "threshold_value",
    "interval_min", "interval_max",
)


def compact_entities(entities: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    headers: Dict[int, str] = {}
    labels: Dict[int, str] = {}
    cells = []
    for entity in entities:
        attrs = entity["attrs"]
        role = {name: code for code, name in _ROLES.items()}[attrs["cell_role"]]
        headers[attrs["column_index"]] = attrs["column_header"]
        if attrs["row_label"] is not None:
            labels[attrs["row_index"]] = attrs["row_label"]
        cells.append([role, attrs["row_index"], attrs["column_index"], *entity["bbox_px"], attrs["value"]])
    return {
        "headers": [headers[c] for c in range(len(headers))],
        "labels": [labels.get(r) for r in range(max(labels) + 1)],
        "cells": cells,
    }


def expand_entities(compact: Mapping[str, Any]) -> List[Dict[str, Any]]:
    entities = []
    for role, r, c, x0, y0, x1, y1, value in compact["cells"]:
        header, label = compact["headers"][c], compact["labels"][r]
        text = {"h": header, "l": label, "v": None if value is None else str(value)}[role]
        attrs = {"cell_role": _ROLES[role], "row_index": r, "column_index": c, "row_label": label,
                 "column_header": header, "text": text, "value": value}
        entities.append({"entity_id": f"cell_r{r}_c{c}", "entity_type": "table_cell", "bbox_px": [x0, y0, x1, y1],
                         "attrs": attrs})
    return entities


def fixture_record(entry: Mapping[str, Any]) -> TraceRecord:
    train = dict(entry["train"])
    train["prompt"] = train["prompt_variants"]["answer_and_annotation"]
    trace = copy.deepcopy(dict(entry["trace"]))
    trace["scene_ir"]["entities"] = expand_entities(trace["scene_ir"].pop("table_cells"))
    return TraceRecord(train=train, trace=trace, image_size=tuple(entry["image_size"]))


FIXTURES: List[Dict[str, Any]] = load_fixture()["records"] if FIXTURE.exists() else []
IDS = [f"{i:02d}-{entry['split_cell']}" for i, entry in enumerate(FIXTURES)]
INDICES = range(len(FIXTURES))


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
        instance_id=record.instance_id,
        trace_record_hash=record.trace_record_hash,
        adapter_id=identity["id"],
        adapter_version=identity["version"],
        schema_id=identity["schema_id"],
    )
    labels = {"answer_gt": record.answer_gt, "annotation_gt": record.annotation_gt, "reward_contract": record.reward_contract}
    return CertificateExtension(ext_id, base, identity, labels, {"image_size": list(record.source_size())}, dict(oracle))


class Case:
    """One record with its oracle, extension and SFT object (unrounded source px)."""

    def __init__(self, record: TraceRecord) -> None:
        self.record = record
        self.oracle = ADAPTER.build_oracle(record)
        self.ext = make_extension(record, self.oracle)
        annotation, certificate, answer = ADAPTER.sft_certificate(self.oracle, record.instance_seed)
        self.sft = {"annotation": annotation, "certificate": certificate, "answer": answer}

    def score(self, response: Any, coord_frame: str = "px_source", **kwargs: Any) -> ScoreResult:
        text = response if isinstance(response, str) else json.dumps(response, separators=(",", ":"))
        base = {"instance_id": self.record.instance_id, "cert_extension_id": self.ext.certificate_extension_id}
        return score_certificate_response(text, base, self.ext, coord_frame=coord_frame, **kwargs)

    @property
    def k(self) -> int:
        return int(self.oracle["count"])


@functools.lru_cache(maxsize=None)
def fixture_records() -> List[TraceRecord]:
    return [fixture_record(entry) for entry in FIXTURES]


@functools.lru_cache(maxsize=None)
def fixture_case(index: int) -> Case:
    return Case(fixture_records()[index])


def signals(result: ScoreResult) -> Dict[str, float]:
    values = {check.check_id: check.score for check in result.checks}
    values.update(A=result.answer, G=result.grounding, M=result.semantics, F=result.format)
    return values


def expectation_failures(result: ScoreResult, reference: ScoreResult, expected: Mapping[str, Any]) -> List[str]:
    """Every ``expected`` entry that does not hold (see ``types.Corruption``)."""
    got, ref = signals(result), signals(reference)
    failures = []
    for key, want in expected.items():
        value = got[key]
        if want == "pass":
            ok = value == 1.0
        elif want == "fail":
            ok = value < 1.0
        elif want == "unchanged":
            ok = abs(value - ref[key]) <= 0.01
        else:
            ok = abs(value - float(want)) <= 1e-9
        if not ok:
            failures.append(f"{key}: expected {want}, got {value}")
    return failures


def rounded(obj: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(dict(obj))
    out["annotation"] = {role: [round_box_half_up(box) for box in boxes] for role, boxes in obj["annotation"].items()}
    return out


def to_rel1000(obj: Mapping[str, Any], size: Sequence[int]) -> Dict[str, Any]:
    width, height = size
    out = copy.deepcopy(dict(obj))
    out["annotation"] = {
        role: [[b[0] * 1000 / width, b[1] * 1000 / height, b[2] * 1000 / width, b[3] * 1000 / height] for b in boxes]
        for role, boxes in obj["annotation"].items()
    }
    return out


def identity_statuses(result: ScoreResult) -> List[str]:
    return [item["status"] for items in result.diagnostics["identity"].values() for item in items]


def generate_record(sampling_seed: int, seed_index: int, query_id: Optional[str] = None) -> TraceRecord:
    """Generate one instance in memory as the builder would persist it (sanitized trace, JSON
    round trip). Only the builder-injected taxonomy keys are missing; the adapter ignores them."""
    from trace_tasks.core.annotation_sanitization import sanitize_trace_payload_for_public_annotation
    from trace_tasks.core.seed import hash64
    from trace_tasks.tasks import create_task

    instance_seed = hash64(sampling_seed, f"{TASK}:instance_seed", seed_index)
    params = {"query_id": query_id} if query_id else {}
    generated = create_task(TASK).generate(int(instance_seed), params=params, max_attempts=100)
    payload = sanitize_trace_payload_for_public_annotation(generated.trace_payload, annotation_gt=generated.annotation_gt)
    trace = json.loads(json.dumps(payload))
    answer_gt, annotation_gt = generated.answer_gt.to_dict(), generated.annotation_gt.to_dict()
    trace.update(answer_gt=answer_gt, annotation_gt=annotation_gt)
    instance_id = canonical_digest({"sampling_seed": sampling_seed, "seed_index": seed_index, "params": params})
    trace["instance_id"] = instance_id
    train = {
        "task": TASK,
        "query_id": generated.query_id,
        "instance_id": instance_id,
        "instance_seed": int(instance_seed),
        "instance_version": "v0",
        "answer_gt": answer_gt,
        "annotation_gt": annotation_gt,
        "reward_contract": resolve_reward_contract(
            answer_type=answer_gt["type"], annotation_type=annotation_gt["type"]
        ).to_dict(),
        "prompt_variants": dict(generated.prompt_variants or {}),
        "trace_ref": {"trace_record_hash": canonical_digest(trace)},
    }
    return TraceRecord(train=train, trace=trace, image_size=tuple(generated.image.size))


def wrong_operands(case: Case, rng: random.Random, field: str) -> Optional[Dict[str, Any]]:
    """True operands replaced by wrong ones; count, sum, mean and answer kept self-consistent."""
    obj = copy.deepcopy(case.sft)
    rows = obj["certificate"]["rows"]
    k = len(rows)
    chosen = rng.sample(range(k), rng.randint(1, k))
    predicate = case.oracle["predicate"]
    for position in chosen:
        truth = rows[position][field]
        if field == "filter_value":
            options = [v for v in range(0, 41) if v != truth and fm.predicate_holds(predicate, v)]
            if not options:
                return None
            rows[position][field] = rng.choice(options)
        else:
            rows[position][field] = rng.choice([v for v in range(1, 41) if v != truth])
    if field == "target_value":
        last = chosen[-1]
        remainder = sum(row["target_value"] for row in rows) % k
        if remainder:
            rows[last]["target_value"] += k - remainder
        if rows[last]["target_value"] == case.sft["certificate"]["rows"][last]["target_value"]:
            rows[last]["target_value"] += k
    cert = obj["certificate"]
    cert["count"] = k
    cert["sum"] = sum(row["target_value"] for row in rows)
    cert["mean"] = cert["sum"] // k
    obj["answer"] = cert["mean"]
    return obj


# ------------------------------------------------------------------------------------------
# definition, registry, fixtures


def test_definition_and_registry() -> None:
    from trace_tasks.tasks.charts.table import filtered_column_mean as task_module

    assert fm.QUERY_FILTERS == task_module.QUERY_FILTERS
    assert QUERIES == task_module.SUPPORTED_QUERY_IDS
    assert isinstance(ADAPTER, fm.FilteredMeanAdapter)
    type(ADAPTER).validate_definition()
    for query_id in QUERIES:
        assert registry.resolve(TASK, query_id, "v0") is ADAPTER
    assert isinstance(registry.resolve(TASK, "single", "v0"), Unsupported)
    assert isinstance(registry.resolve(TASK, QUERIES[0], "v1"), Unsupported)
    assert registry.adapter_by_id(ADAPTER.adapter_id) is ADAPTER
    assert ADAPTER.identity() == {
        "id": "filtered_mean.table_filtered_column_mean",
        "version": "1.0.0",
        "schema_id": "cert.filtered_mean.v1",
        "verifier_version": "1.1.0",
    }
    contract = ADAPTER.public_contract((940, 900), "rel1000")
    assert (contract.kind, contract.annotation_type, contract.list_key, contract.list_cap) == (
        "filtered_mean", "bbox_set_map", "rows", 16
    )


def test_fixtures_cover_every_query_and_are_persisted_records() -> None:
    generator = load_fixture()["generator"]
    assert generator["sampling_seed"] == FIXTURE_SAMPLING_SEED and len(FIXTURES) == FIXTURE_COUNT
    records = fixture_records()
    assert {record.query_id for record in records} == set(QUERIES)
    for record in records:
        assert record.task == TASK and record.instance_version == "v0"
        assert record.source_size() == (940, 900)
        assert record.trace["instance_id"] == record.instance_id
        assert record.train["versions"]["code_hash"] == CODE_HASH


@pytest.mark.parametrize("index", INDICES, ids=IDS)
def test_oracle_split_key_and_sft_are_frozen(index: int) -> None:
    entry, record = FIXTURES[index], fixture_records()[index]
    oracle = ADAPTER.build_oracle(record)
    assert oracle == entry["oracle"]
    assert json.loads(json.dumps(oracle)) == oracle  # JSON-safe
    assert ADAPTER.split_cell(record) == entry["split_cell"]
    assert ADAPTER.semantic_key(record) == entry["semantic_key"]
    annotation, certificate, answer = ADAPTER.sft_certificate(oracle, record.instance_seed)
    assert {"annotation": annotation, "certificate": certificate, "answer": answer} == entry["sft"]
    assert list(certificate) == ["kind", "rows", "sum", "count", "mean"]


@pytest.mark.parametrize("index", INDICES, ids=IDS)
def test_oracle_matches_the_scene(index: int) -> None:
    case = fixture_case(index)
    oracle, record = case.oracle, case.record
    relations = record.scene_ir["relations"]
    chosen = [row for row in oracle["rows"] if row["selected"]]
    assert [row["r"] for row in chosen] == sorted(relations["selected_row_indices"])
    assert oracle["count"] == len(chosen) >= 1
    assert oracle["sum"] == sum(row["target_value"] for row in chosen) == oracle["count"] * oracle["mean"]
    assert oracle["mean"] == record.answer_gt["value"]
    assert len(oracle["rows"]) == oracle["n_rows"] == relations["row_count"]
    values = record.execution_trace["values_by_row"]
    for row in oracle["rows"]:
        assert row["filter_value"] == values[row["label"]][oracle["filter_column"]]
        assert row["target_value"] == values[row["label"]][oracle["target_column"]]
        assert row["selected"] == fm.predicate_holds(oracle["predicate"], row["filter_value"])
        if oracle["predicate"]["op"] != "in" and row["filter_value"] == oracle["predicate"]["t"]:
            assert not row["selected"]  # boundary rows are never selected (strict > / <)
    annotation = record.annotation_gt["value"]
    assert annotation["filter_cells"] == [row["filter_cell"] for row in chosen]
    assert annotation["target_cells"] == [row["target_cell"] for row in chosen]


def test_split_cells_and_holdouts() -> None:
    assert set(fm.TRAIN_CELLS) == {f"{q}|{v}" for q in ("above", "below") for v in ("spreadsheet", "zebra", "card_table")}
    assert fm.HELD_OUT_CELLS == {
        "above|ledger": "render",
        "below|ledger": "render",
        "interval|spreadsheet": "program",
        "interval|zebra": "program",
        "interval|card_table": "program",
        "interval|ledger": "both",
    }
    for case in map(fixture_case, INDICES):
        cell = ADAPTER.split_cell(case.record)
        query, variant = cell.split("|")
        assert query == fm.QUERY_SHORT[case.record.query_id]
        assert variant == case.record.scene_ir["relations"]["scene_variant"]
        expected = None if query != "interval" and variant != "ledger" else fm.HELD_OUT_CELLS[cell]
        assert fm.holdout_type(cell) == expected
    with pytest.raises(ValueError):
        fm.holdout_type("sideways|zebra")


def test_semantic_key_is_stable_and_content_sensitive() -> None:
    record = fixture_records()[0]
    key = ADAPTER.semantic_key(record)
    assert key.startswith("filtered_mean:blake3:")
    assert ADAPTER.semantic_key(copy.deepcopy(record)) == key
    trace = copy.deepcopy(dict(record.trace))
    label = next(iter(trace["execution_trace"]["values_by_row"]))
    header = trace["execution_trace"]["column_headers"][0]
    trace["execution_trace"]["values_by_row"][label][header] += 1
    assert ADAPTER.semantic_key(TraceRecord(record.train, trace, record.image_size)) != key
    assert len({ADAPTER.semantic_key(r) for r in fixture_records()}) == len(FIXTURES)


# ------------------------------------------------------------------------------------------
# oracle round trip (DESIGN 9)


@pytest.mark.parametrize("index", INDICES, ids=IDS)
def test_round_trip_unrounded_sft_scores_one(index: int) -> None:
    case = fixture_case(index)
    assert case.sft["annotation"] == case.record.annotation_gt["value"]  # unrounded annotation_gt boxes
    result = case.score(case.sft, shown_frame_diagnostics=True)
    assert (result.answer, result.grounding, result.semantics, result.format) == (1.0, 1.0, 1.0, 1.0)
    assert result.joint == 1.0 and result.error_class == "correct"
    assert result.diagnostics["G_id"] == 1.0 and set(identity_statuses(result)) == {"resolved"}
    assert all(check.score == 1.0 for check in result.checks)
    assert result.diagnostics["M_shown"] == 1.0  # DESIGN 2.2: table cells survive the 28-px frame shift
    adapter_diag = result.diagnostics["adapter"]
    assert adapter_diag["fm.binding_status"] == {"bound": case.k} and adapter_diag["fm.extra_rows"] == 0
    assert adapter_diag["fm.filter_read_acc"] == adapter_diag["fm.target_read_acc"] == 1.0


@pytest.mark.parametrize("index", INDICES, ids=IDS)
def test_round_trip_rounded_sft_target(index: int) -> None:
    case = fixture_case(index)
    target = rounded(case.sft)
    assert all(isinstance(v, int) for boxes in target["annotation"].values() for box in boxes for v in box)
    result = case.score(json.dumps(target, separators=(",", ":")))
    assert (result.answer, result.semantics, result.format) == (1.0, 1.0, 1.0)
    assert set(identity_statuses(result)) == {"resolved"}
    assert 0.95 < result.grounding <= 1.0  # logged: half-pixel edges cost a little IoU


@pytest.mark.parametrize("index", INDICES, ids=IDS)
def test_round_trip_rel1000_frame(index: int) -> None:
    case = fixture_case(index)
    result = case.score(to_rel1000(case.sft, case.record.source_size()), coord_frame="rel1000")
    assert (result.answer, result.semantics, result.format) == (1.0, 1.0, 1.0)
    assert result.grounding > 0.999


def test_round_trip_live_generation_degenerate_interval() -> None:
    record = generate_record(*DEGENERATE_INTERVAL, query_id="interval_filtered_mean")
    case = Case(record)
    predicate = case.oracle["predicate"]
    assert predicate["op"] == "in" and predicate["lo"] == predicate["hi"]
    result = case.score(case.sft)
    assert (result.answer, result.grounding, result.semantics, result.format) == (1.0, 1.0, 1.0, 1.0)
    for corruption in ADAPTER.corruptions(case.oracle, case.sft, record.instance_seed):
        failures = expectation_failures(case.score(corruption.response_obj), result, corruption.expected)
        assert not failures, (corruption.name, failures)
    names = {c.name: c for c in ADAPTER.corruptions(case.oracle, case.sft, 0)}
    # No other integer passes a one-value interval, so the misread also breaks the predicate.
    assert names["wrong_filter_value_consistent"].expected["fm.pred.local"] == "fail"


# ------------------------------------------------------------------------------------------
# counterexamples (DESIGN 9)


@pytest.mark.parametrize("index", INDICES, ids=IDS)
def test_corruptions_fail_exactly_their_checks(index: int) -> None:
    case = fixture_case(index)
    reference = case.score(case.sft)
    corruptions = ADAPTER.corruptions(case.oracle, case.sft, case.record.instance_seed)
    assert REQUIRED_CORRUPTIONS <= {c.name for c in corruptions}
    expected_keys = set(ADAPTER.m_families) | set(ADAPTER.local_check_ids) | {"A", "G", "M", "F"}
    for corruption in corruptions:
        assert set(corruption.expected) == expected_keys, corruption.name
        result = case.score(corruption.response_obj)
        failures = expectation_failures(result, reference, corruption.expected)
        assert not failures, (corruption.name, failures)
    # The corruption list is deterministic in its seed and never mutates the SFT object.
    again = ADAPTER.corruptions(case.oracle, case.sft, case.record.instance_seed)
    assert [(c.name, c.response_obj) for c in again] == [(c.name, c.response_obj) for c in corruptions]
    assert case.sft == Case(case.record).sft


def test_corruptions_separate_m_from_g() -> None:
    """Go/no-go (b): >= 3 corruption types leave G unchanged (|dG| < 0.01) and lower M."""
    case = fixture_case(0)
    reference = case.score(case.sft)
    separating = set()
    for corruption in ADAPTER.corruptions(case.oracle, case.sft, 0):
        result = case.score(corruption.response_obj)
        if abs(result.grounding - reference.grounding) < 0.01 and result.semantics < reference.semantics:
            separating.add(corruption.name)
    assert {"wrong_target_value_consistent", "wrong_sum", "wrong_pairing"} <= separating
    assert len(separating) >= 3


def test_wrong_pairing_is_invisible_to_g_but_not_to_m() -> None:
    case = fixture_case(1)
    obj = copy.deepcopy(case.sft)
    targets = obj["annotation"]["target_cells"]
    targets[0], targets[-1] = targets[-1], targets[0]
    result = case.score(obj)
    assert result.grounding == 1.0 and result.answer == 1.0
    assert result.semantics < 1.0 and result.error_class == "grounding"
    status = result.diagnostics["adapter"]["fm.binding_status"]
    assert status == {"bound": case.k - 2, "wrong_pairing": 2}


def test_negative_index_never_binds_the_last_box() -> None:
    case = fixture_case(2)
    obj = copy.deepcopy(case.sft)
    obj["certificate"]["rows"][0]["annotation_index"] = -1
    result = case.score(obj)
    rows = next(check for check in result.checks if check.check_id == "fm.rows.truth")
    assert rows.detail["elements"][0] == ["invalid_element", None]
    assert rows.score == pytest.approx((case.k - 1) / case.k)
    assert result.format == 0.0 and "element_invalid" in result.diagnostics["format_reasons"]


def test_wrong_column_is_logged() -> None:
    case = fixture_case(3)
    oracle = case.oracle
    other = next(c for c in range(oracle["n_cols"]) if c not in (oracle["filter_col"], oracle["target_col"]))
    first = next(row for row in oracle["rows"] if row["selected"])
    obj = copy.deepcopy(case.sft)
    obj["annotation"]["target_cells"][0] = oracle["cells"][first["r"]][other]
    result = case.score(obj)
    assert result.diagnostics["adapter"]["fm.binding_status"] == {"bound": case.k - 1, "wrong_column": 1}
    assert result.error_class == "grounding"


@pytest.mark.parametrize("index", INDICES, ids=IDS)
def test_property_wrong_operands_with_consistent_arithmetic(index: int) -> None:
    case = fixture_case(index)
    rng = random.Random(case.record.instance_seed)
    for trial in range(12):
        field = ("target_value", "filter_value")[trial % 2]
        obj = wrong_operands(case, rng, field)
        if obj is None:
            continue
        result = case.score(obj)
        assert result.semantics < 1.0, (field, obj["certificate"])
        operand = "fm.target_value.truth" if field == "target_value" else "fm.filter_value.truth"
        values = signals(result)
        assert values[operand] < 1.0 and values["fm.rows.truth"] == 1.0
        for local in ("fm.ref.local", "fm.count.local", "fm.sum.local", "fm.mean.local", "fm.answer.local"):
            assert values[local] == 1.0, (local, obj["certificate"])


def test_empty_claim_and_model_count_zero() -> None:
    case = fixture_case(0)
    obj = {"annotation": {"filter_cells": [], "target_cells": []},
           "certificate": {"kind": "filtered_mean", "rows": [], "sum": 0, "count": 0, "mean": 0}, "answer": 0}
    values = signals(case.score(obj))
    assert values["M"] == 0.0 and values["fm.mean.local"] == 0.0 and values["fm.count.local"] == 1.0
    assert values["fm.rows.truth"] == values["fm.filter_value.truth"] == values["fm.target_value.truth"] == 0.0


# ------------------------------------------------------------------------------------------
# equivalence (DESIGN 9)


@pytest.mark.parametrize("index", INDICES, ids=IDS)
def test_equivalent_forms_score_identically(index: int) -> None:
    case = fixture_case(index)
    reference = signals(case.score(case.sft))
    rng = random.Random(case.record.instance_seed)
    k = case.k

    permuted = copy.deepcopy(case.sft)  # annotation pairs permuted, indices re-bound
    order = rng.sample(range(k), k)
    for role in ("filter_cells", "target_cells"):
        permuted["annotation"][role] = [case.sft["annotation"][role][i] for i in order]
    for row in permuted["certificate"]["rows"]:
        row["annotation_index"] = order.index(row["annotation_index"])
    assert permuted["annotation"] != case.sft["annotation"]

    shuffled = copy.deepcopy(case.sft)  # certificate rows in another order
    rng.shuffle(shuffled["certificate"]["rows"])

    def reverse_keys(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: reverse_keys(value[key]) for key in reversed(list(value))}
        if isinstance(value, list):
            return [reverse_keys(item) for item in value]
        return value

    for variant in (permuted, shuffled, reverse_keys(case.sft), reverse_keys(permuted)):
        assert signals(case.score(variant)) == reference

    trailing = case.score(json.dumps(case.sft) + "\nSo the mean is as stated above.")
    values = signals(trailing)
    assert values["M"] == reference["M"] and values["G"] == reference["G"] and values["F"] == 0.0
    assert "not_final" in trailing.diagnostics["format_reasons"]


@pytest.mark.parametrize("params", [IdentityParams(), IdentityParams(0.3, 0.8), IdentityParams(0.7, 0.9)])
def test_identity_prefilter_equals_the_full_scan(params: IdentityParams) -> None:
    """The adapter's intersecting-cells prefilter returns exactly the shared full-scan result."""
    rng = random.Random(7)
    statuses = set()
    for case in map(fixture_case, range(min(6, len(FIXTURES)))):
        boxes: List[Any] = []
        for row in case.oracle["cells"]:
            for x0, y0, x1, y1 in row:
                w, h = x1 - x0, y1 - y0
                for _ in range(4):
                    dx, dy = rng.uniform(-0.8, 0.8) * w, rng.uniform(-0.8, 0.8) * h
                    sx, sy = rng.uniform(0.1, 1.4), rng.uniform(0.1, 1.4)
                    cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
                    boxes.append([cx - sx * w / 2, cy - sy * h / 2, cx + sx * w / 2, cy + sy * h / 2])
        boxes.append(list(case.oracle["cells"][0][0]))  # duplicate of the first cell (dedupe)
        boxes += [[1, 1, 30, 30], [900, 880, 939, 899], "not a box", [5, 5, 5, 9]]  # off-table, malformed
        ctx = PublicContext(
            TASK, case.record.query_id, (940, 900), "px_source", source_frame("px_source", (940, 900)), params
        )
        view = annotation_view({"filter_cells": boxes, "target_cells": boxes[::7]}, "bbox_set_map", ctx.frame)
        for role in ("filter_cells", "target_cells"):
            fast = ADAPTER.resolve_role(view, role, case.oracle, ctx)
            assert fast == Adapter.resolve_role(ADAPTER, view, role, case.oracle, ctx)
            statuses.update(r.status for r in fast)
    assert {"resolved", "unresolved", "duplicate", "invalid"} <= statuses


# ------------------------------------------------------------------------------------------
# quarantine (export invariants)


def _doctored(record: TraceRecord, mutate: Callable[[Dict[str, Any], Dict[str, Any]], None]) -> TraceRecord:
    train, trace = copy.deepcopy(dict(record.train)), copy.deepcopy(dict(record.trace))
    mutate(train, trace)
    return TraceRecord(train, trace, record.image_size)


def _value_entity(trace: Dict[str, Any], r: int, c: int) -> Dict[str, Any]:
    return next(
        e for e in trace["scene_ir"]["entities"]
        if e["attrs"]["cell_role"] == "value" and (e["attrs"]["row_index"], e["attrs"]["column_index"]) == (r + 1, c + 1)
    )


def _bump_target(train: Dict[str, Any], trace: Dict[str, Any]) -> None:
    relations = trace["scene_ir"]["relations"]
    r, c = relations["selected_row_indices"][0], relations["target_column_index"]
    entity = _value_entity(trace, r, c)
    entity["attrs"]["value"] += 1
    entity["attrs"]["text"] = str(entity["attrs"]["value"])
    trace["execution_trace"]["values_by_row"][entity["attrs"]["row_label"]][relations["target_column"]] += 1


def _empty_selection(train: Dict[str, Any], trace: Dict[str, Any]) -> None:
    relations = trace["scene_ir"]["relations"]
    relations["threshold_value"] = 99 if relations["filter_variant"] == "above_threshold" else -99
    relations["selected_row_indices"] = []


def _overlap(train: Dict[str, Any], trace: Dict[str, Any]) -> None:
    relations = trace["scene_ir"]["relations"]
    other = next(
        c for c in range(relations["numeric_column_count"])
        if c not in (relations["filter_column_index"], relations["target_column_index"])
    )
    box = _value_entity(trace, 0, other)["bbox_px"]
    box[2] += 5.0


QUARANTINE_CASES = {
    "empty_selection": _empty_selection,
    "sum_not_divisible": _bump_target,
    "mean_mismatch": lambda train, trace: train["answer_gt"].update(value=train["answer_gt"]["value"] + 1),
    "predicate_mismatch": lambda train, trace: trace["scene_ir"]["relations"]["selected_row_indices"].pop(),
    "annotation_mismatch": lambda train, trace: train["annotation_gt"]["value"]["target_cells"][0].__setitem__(0, 1.0),
    "values_by_row_mismatch": lambda train, trace: trace["execution_trace"]["column_headers"].reverse(),
    "text_value_mismatch": lambda train, trace: trace["scene_ir"]["entities"][-1]["attrs"].update(text="x"),
    "missing_field": lambda train, trace: trace["scene_ir"]["relations"].pop("target_column"),
    "query_variant_mismatch": lambda train, trace: trace["scene_ir"]["relations"].update(filter_variant="in_interval"),
    "unsupported_query": lambda train, trace: train.update(query_id="single"),
    "column_header_mismatch": lambda train, trace: trace["scene_ir"]["relations"].update(filter_column="NOPE"),
    "incomplete_grid": lambda train, trace: trace["scene_ir"]["entities"].pop(),
    "cells_overlap": _overlap,
    "answer_type": lambda train, trace: train["answer_gt"].update(type="number"),
}


@pytest.mark.parametrize("reason", sorted(QUARANTINE_CASES))
def test_export_invariants_quarantine(reason: str) -> None:
    record = next(r for r in fixture_records() if r.query_id == "above_threshold_filtered_mean")
    with pytest.raises(QuarantineError) as raised:
        ADAPTER.build_oracle(_doctored(record, QUARANTINE_CASES[reason]))
    assert raised.value.reason == reason


# ------------------------------------------------------------------------------------------
# prompt pieces


def test_prompt_pieces_are_fixed_and_self_consistent() -> None:
    line = ADAPTER.certificate_line()
    for token in ('"filter_cells"', '"target_cells"', '"annotation_index"', '"filter_value"', '"target_value"', "m = s / k"):
        assert token in line
    example = json.loads(ADAPTER.example_json())
    assert list(example) == ["annotation", "certificate", "answer"]
    cert = example["certificate"]
    view = validate_certificate(cert, ADAPTER.schema(cert), 2)
    assert view.schema_ok and list(cert)[0] == "kind"
    targets = [row["target_value"] for row in cert["rows"]]
    assert cert["sum"] == sum(targets) and cert["count"] == len(targets) == len(example["annotation"]["filter_cells"])
    assert cert["count"] * cert["mean"] == cert["sum"] and example["answer"] == cert["mean"]
    coords = [v for boxes in example["annotation"].values() for box in boxes for v in box]
    assert all(isinstance(v, int) and 0 <= v <= 1000 for v in coords)
    for case in map(fixture_case, INDICES):  # instance independence: no oracle header or label in the fixed text
        texts = (line, ADAPTER.example_json())
        for name in (case.oracle["filter_column"], case.oracle["target_column"], case.oracle["rows"][0]["label"]):
            assert all(f'"{name}"' not in text for text in texts)


# ------------------------------------------------------------------------------------------
# maintenance entry points


def _strip_record(record: TraceRecord) -> Dict[str, Any]:
    """The persisted record reduced to what the adapter reads; lossless for those fields."""
    trace = record.trace
    entities = []
    for entity in trace["scene_ir"]["entities"]:
        entity = copy.deepcopy(dict(entity))
        entity["attrs"].pop("text_bbox_px", None)
        entities.append(entity)
    relations = trace["scene_ir"]["relations"]
    execution = trace["execution_trace"]
    spec = trace.get("render_spec") or {}
    stripped_trace = {
        "instance_id": trace["instance_id"],
        "scene_ir": {
            "relations": {key: relations[key] for key in _RELATION_KEYS if key in relations},
            "table_cells": compact_entities(entities),
        },
        "execution_trace": {key: execution[key] for key in ("scene_variant", "column_headers", "values_by_row")},
        "render_spec": {key: spec[key] for key in ("canvas_width", "canvas_height", "coord_space", "scene_variant") if key in spec},
    }
    train = {key: value for key, value in record.train.items() if key != "prompt"}
    entry = {"train": train, "trace": stripped_trace, "image_size": list(record.source_size())}
    restored = fixture_record(entry)
    assert restored.scene_ir["entities"] == entities, "table_cells must expand to the persisted entities"
    assert dict(restored.train) == dict(record.train), "prompt must equal prompt_variants.answer_and_annotation"
    return entry


def build_fixture(work_dir: Path) -> Dict[str, Any]:
    from trace_tasks import BuildConfig, BuildTaskConfig, build_dataset
    from trace_tasks.experimental.certificates.types import read_trace_records

    work_dir.mkdir(parents=True, exist_ok=True)
    config = BuildConfig(
        output_root=str(work_dir),
        dataset_name="cert_fixtures_filtered_mean",
        instance_version="v0",
        image_format="png",
        tasks=[BuildTaskConfig(task_id=TASK, count=FIXTURE_COUNT)],
        max_attempts_per_instance=100,
        sampling_seed=FIXTURE_SAMPLING_SEED,
        workers=1,
    )
    dataset_dir = build_dataset(config, code_hash=CODE_HASH)
    entries = []
    for record in read_trace_records(dataset_dir):
        oracle = ADAPTER.build_oracle(record)
        annotation, certificate, answer = ADAPTER.sft_certificate(oracle, record.instance_seed)
        entries.append(
            {
                **_strip_record(record),
                "split_cell": ADAPTER.split_cell(record),
                "semantic_key": ADAPTER.semantic_key(record),
                "oracle": oracle,
                "sft": {"annotation": annotation, "certificate": certificate, "answer": answer},
            }
        )
    return {
        "generator": {
            "task": TASK,
            "sampling_seed": FIXTURE_SAMPLING_SEED,
            "count": FIXTURE_COUNT,
            "code_hash": CODE_HASH,
            "adapter": ADAPTER.identity(),
            "trace_fields": "scene_ir.relations (keys read), scene_ir.entities as table_cells (no text_bbox_px), "
            "execution_trace.{scene_variant,column_headers,values_by_row}, render_spec canvas keys; "
            "train without prompt (= prompt_variants.answer_and_annotation)",
            "command": "python -m tests.test_certificate_adapter_filtered_mean --regen <scratch_dir>",
        },
        "records": entries,
    }


def run_sweep(n: int, sampling_seed: int = SWEEP_SAMPLING_SEED) -> Dict[str, Any]:
    """Round trip, corruptions and the property test on ``n`` in-memory instances."""
    counts: Dict[str, List[int]] = {}
    quarantined: Dict[str, int] = {}
    separating: Dict[str, int] = {}
    cells: Dict[str, int] = {}
    times: List[float] = []
    g_rounded: List[float] = []

    def tally(name: str, ok: bool) -> None:
        bucket = counts.setdefault(name, [0, 0])
        bucket[0] += int(ok)
        bucket[1] += 1

    for index in range(n):
        record = generate_record(sampling_seed, index, QUERIES[index % len(QUERIES)])
        try:
            case = Case(record)
        except QuarantineError as exc:
            quarantined[exc.reason] = quarantined.get(exc.reason, 0) + 1
            continue
        cell = ADAPTER.split_cell(record)
        cells[cell] = cells.get(cell, 0) + 1
        reference = case.score(case.sft, shown_frame_diagnostics=True)
        times.append(reference.diagnostics["verifier_ms"])
        tally("roundtrip_unrounded", (reference.answer, reference.grounding, reference.semantics, reference.format)
              == (1.0, 1.0, 1.0, 1.0))
        tally("roundtrip_M_shown", reference.diagnostics["M_shown"] == 1.0)
        result = case.score(rounded(case.sft))
        g_rounded.append(result.grounding)
        tally("roundtrip_rounded", (result.answer, result.semantics, result.format) == (1.0, 1.0, 1.0)
              and set(identity_statuses(result)) == {"resolved"})
        result = case.score(to_rel1000(case.sft, record.source_size()), coord_frame="rel1000")
        tally("roundtrip_rel1000", (result.answer, result.semantics, result.format) == (1.0, 1.0, 1.0))
        for corruption in ADAPTER.corruptions(case.oracle, case.sft, record.instance_seed):
            result = case.score(corruption.response_obj)
            tally(f"corruption:{corruption.name}", not expectation_failures(result, reference, corruption.expected))
            if abs(result.grounding - reference.grounding) < 0.01 and result.semantics < 1.0:
                separating[corruption.name] = separating.get(corruption.name, 0) + 1
        rng = random.Random(record.instance_seed)
        for trial in range(4):
            obj = wrong_operands(case, rng, ("target_value", "filter_value")[trial % 2])
            if obj is not None:
                tally("property_wrong_operands_M_below_1", case.score(obj).semantics < 1.0)
    return {
        "n_generated": n,
        "sampling_seed": sampling_seed,
        "quarantined": quarantined,
        "cells": dict(sorted(cells.items())),
        "pass_rates": {name: f"{ok}/{total}" for name, (ok, total) in sorted(counts.items())},
        "all_pass": all(ok == total for ok, total in counts.values()),
        "g_unchanged_m_lowered": dict(sorted(separating.items())),
        "rounded_G": {"min": min(g_rounded), "mean": statistics.fmean(g_rounded)} if g_rounded else {},
        "verifier_ms_with_shown": {"median": statistics.median(times), "max": max(times)} if times else {},
    }


if __name__ == "__main__":  # pragma: no cover - maintenance entry points
    if len(sys.argv) == 3 and sys.argv[1] == "--regen":
        payload = build_fixture(Path(sys.argv[2]))
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        lines = ",\n".join(json.dumps(entry, separators=(",", ":")) for entry in payload["records"])
        FIXTURE.write_text(
            '{"generator":' + json.dumps(payload["generator"], separators=(",", ":")) + ',\n"records":[\n' + lines + "\n]}\n",
            encoding="utf-8",
        )
        print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes, {len(payload['records'])} records)")
    elif len(sys.argv) in (3, 4) and sys.argv[1] == "--sweep":
        seed = int(sys.argv[3]) if len(sys.argv) == 4 else SWEEP_SAMPLING_SEED
        print(json.dumps(run_sweep(int(sys.argv[2]), seed), indent=1))
    else:
        raise SystemExit(
            "usage: python -m tests.test_certificate_adapter_filtered_mean --regen <scratch_dir> | --sweep <n> [<seed>]"
        )
