"""Path certificate adapter (``path.rpg_movement_cost``, DESIGN 3.4 and 9).

Fixtures are 16 persisted rpg ``movement_cost_value`` records built by ``trace_tasks`` with
sampling_seed 7001, each with its oracle, split cell, semantic key, canonical SFT object and
every enumerated optimal route. To keep the file small, images are stripped (image size kept),
the unused ``query_spec`` and ``taxonomy`` blocks are dropped, and ``scene_ir.tiles`` is stored
as the marker ``"terrain_rows"``: :func:`decode_trace` rebuilds the tile list from
``execution_trace.renderer.terrain_rows``, and regeneration asserts the rebuilt list equals
the persisted one exactly. Use :func:`load_fixture_entries` to read the file.

Maintenance entry points (write only under the given scratch directory):
    python -m tests.test_certificate_adapter_path --regen <scratch_dir>
    python -m tests.test_certificate_adapter_path --sweep <scratch_dir> [count] [sampling_seed]
"""

from __future__ import annotations

import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest

from trace_tasks.core.reward_contracts import resolve_reward_contract
from trace_tasks.experimental.certificates import registry
from trace_tasks.experimental.certificates.adapters.path import (
    FAM_COST,
    FAM_ENDPOINTS,
    FAM_LEGAL,
    FAM_OPTIMAL,
    FAM_TOTAL,
    BLOCKED_TERRAINS,
    PUBLIC_TERRAIN_COSTS,
    TASK_ID,
    Grid,
    PathAdapter,
    grid_oracle,
    oracle_paths,
    steps_for,
)
from trace_tasks.experimental.certificates.geometry import round_box_half_up
from trace_tasks.experimental.certificates.identity import resolve_points
from trace_tasks.experimental.certificates.registry import Unsupported
from trace_tasks.experimental.certificates.scoring import score_certificate_response
from trace_tasks.experimental.certificates.types import (
    CertificateExtension,
    QuarantineError,
    ScoreResult,
    TraceRecord,
    certificate_extension_id,
)

FIXTURE = Path(__file__).parent / "fixtures" / "certificates" / "path" / "fixtures_7001.json"
FIXTURE_SAMPLING_SEED = 7001
FIXTURE_COUNT = 16
DROPPED_TRACE_KEYS = ("query_spec", "taxonomy")
TILES_MARKER = "terrain_rows"
SIGNALS = ("A", "G", "M", "F")
ALWAYS_PRESENT = {
    "wrong_cost_consistent", "wrong_total", "answer_inconsistent", "truncated_path", "duplicate_step",
    "reversed_endpoints", "claims_unreachable", "teleport_start_start", "annotation_shifted",
    "boundary_points", "point_off_canvas", "bool_cost", "float_costs", "total_missing",
}
DESIGN9 = ALWAYS_PRESENT | {
    "skipped_tile", "extra_step_after_target", "suboptimal_path", "water_tile", "all_costs_one",
    "teleport_start_target",
    # review EX-2/EX-3/CONF-03/LEAK-03: cost credit only on the legal prefix from the start tile
    "stutter_on_start", "backtrack", "detached_walk",
}

ADAPTER = registry.load_adapter("path")


# ------------------------------------------------------------------------------------------
# helpers


def record_of(entry: Mapping[str, Any]) -> TraceRecord:
    return TraceRecord(train=entry["train"], trace=entry["trace"], image_size=tuple(entry["image_size"]))


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


def score(obj: Any, record: TraceRecord, extension: CertificateExtension, **kwargs: Any) -> ScoreResult:
    text = obj if isinstance(obj, str) else json.dumps(obj, separators=(",", ":"))
    kwargs.setdefault("coord_frame", "px_source")
    return score_certificate_response(text, {"instance_id": record.instance_id}, extension, adapter=ADAPTER, **kwargs)


def check(result: ScoreResult, check_id: str):
    return next(c for c in result.checks if c.check_id == check_id)


def signal(result: ScoreResult, key: str) -> float:
    return {"A": result.answer, "G": result.grounding, "M": result.semantics, "F": result.format}[key]


def family_scores(result: ScoreResult) -> Dict[str, float]:
    return {c.check_id: c.score for c in result.checks}


def matches(expected: Any, actual: float, baseline: float) -> bool:
    if expected == "pass":
        return actual == 1.0
    if expected == "fail":
        return actual < 1.0
    if expected == "unchanged":
        return abs(actual - baseline) <= 0.01
    return abs(actual - float(expected)) <= 1e-9


def corruption_failures(obj_score: ScoreResult, corrupted: ScoreResult, expected: Mapping[str, Any]) -> List[str]:
    """Every deviation from ``expected``; checks not listed must pass (``fails exactly``)."""
    problems = []
    by_id = {c.check_id: c for c in corrupted.checks}
    for key, want in expected.items():
        if key in SIGNALS:
            actual, baseline = signal(corrupted, key), signal(obj_score, key)
        else:
            actual, baseline = by_id[key].score, 1.0
        if not matches(want, actual, baseline):
            problems.append(f"{key}: expected {want!r}, got {actual!r}")
    for check_id in ADAPTER.m_families + ADAPTER.local_check_ids:
        result = by_id[check_id]
        if check_id not in expected and result.applicable and result.score < 1.0:
            problems.append(f"{check_id}: unexpected failure {result.score!r}")
    return problems


def sft_obj(entry: Mapping[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(entry["sft"])


def path_obj(entry: Mapping[str, Any], route: Sequence[Sequence[int]]) -> Dict[str, Any]:
    oracle = entry["oracle"]
    grid = Grid.from_oracle(oracle)
    tiles = [grid.index(r, c) for r, c in route]
    cert = {"kind": "path", "reachable": True, "steps": steps_for(grid, tiles), "total": oracle["opt"]}
    return {"annotation": entry["sft"]["annotation"], "certificate": cert, "answer": oracle["opt"]}


def _tiles_from_terrain(terrain_rows: Sequence[Sequence[str]], tile_px: int) -> List[Dict[str, Any]]:
    size = float(tile_px)
    return [
        {
            "tile_id": f"r{row:02d}_c{col:02d}",
            "row": row,
            "col": col,
            "terrain": name,
            "movement_cost": PUBLIC_TERRAIN_COSTS.get(name),
            "passable": name not in BLOCKED_TERRAINS,
            "bbox": [col * size, row * size, (col + 1) * size, (row + 1) * size],
            "point": [(col + 0.5) * size, (row + 0.5) * size],
            "metadata": {"tile_px": tile_px},
        }
        for row, names in enumerate(terrain_rows)
        for col, name in enumerate(names)
    ]


def encode_trace(trace: Mapping[str, Any]) -> Dict[str, Any]:
    """Fixture form of a persisted trace (lossless for these records; checked on regen)."""
    encoded = {key: copy.deepcopy(value) for key, value in trace.items() if key not in DROPPED_TRACE_KEYS}
    encoded["scene_ir"]["tiles"] = TILES_MARKER
    rebuilt, original = decode_trace(encoded)["scene_ir"]["tiles"], trace["scene_ir"]["tiles"]
    if json.dumps(rebuilt) != json.dumps(original):  # byte-exact: types and key order included
        raise ValueError("tile list is not reproducible from renderer.terrain_rows")
    return encoded


def decode_trace(encoded: Mapping[str, Any]) -> Dict[str, Any]:
    trace = copy.deepcopy(dict(encoded))
    if trace["scene_ir"].get("tiles") == TILES_MARKER:
        renderer = trace["execution_trace"]["renderer"]
        trace["scene_ir"]["tiles"] = _tiles_from_terrain(renderer["terrain_rows"], renderer["tile_px"])
    return trace


def load_fixture_entries(path: Path = FIXTURE) -> Dict[str, Any]:
    """The fixture file with every trace decoded to its persisted form."""
    if not path.exists():  # only while regenerating; test_fixture_header then fails loudly
        return {"generator": {}, "adapter": {}, "records": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    for entry in payload["records"]:
        entry["trace"] = decode_trace(entry["trace"])
    return payload


FIXTURES = load_fixture_entries()
ENTRIES = FIXTURES["records"]
IDS = [f"r{i:02d}" for i in range(len(ENTRIES))]


@pytest.fixture(scope="module")
def prepared() -> List[Dict[str, Any]]:
    """Each fixture with its record, extension and the score of its unrounded SFT object."""
    out = []
    for entry in ENTRIES:
        record = record_of(entry)
        extension = make_extension(record, entry["oracle"])
        out.append({"entry": entry, "record": record, "ext": extension, "sft_score": score(entry["sft"], record, extension)})
    return out


# ------------------------------------------------------------------------------------------
# registry, oracle extraction and fixtures


def test_registry_resolves_path_adapter() -> None:
    adapter = registry.resolve(TASK_ID, "single", "v0")
    assert isinstance(adapter, PathAdapter) and adapter is ADAPTER
    assert registry.adapter_by_id("path.rpg_movement_cost") is ADAPTER
    assert isinstance(registry.resolve(TASK_ID, "other", "v0"), Unsupported)
    assert isinstance(registry.resolve(TASK_ID, "single", "v9"), Unsupported)
    PathAdapter.validate_definition()
    assert ADAPTER.identity() == {
        "id": "path.rpg_movement_cost", "version": "1.0.0", "schema_id": "cert.path.v1", "verifier_version": "1.1.0"
    }
    assert ADAPTER.family_stages == {
        FAM_ENDPOINTS: "grounding", FAM_LEGAL: "grounding", FAM_COST: "operand",
        FAM_TOTAL: "computation", FAM_OPTIMAL: "computation",
    }


def test_fixture_header() -> None:
    generator = FIXTURES["generator"]
    assert generator["sampling_seed"] == FIXTURE_SAMPLING_SEED and len(ENTRIES) == FIXTURE_COUNT
    assert FIXTURES["adapter"] == ADAPTER.identity()
    assert all(key not in entry["trace"] for entry in ENTRIES for key in DROPPED_TRACE_KEYS)


@pytest.mark.parametrize("index", range(len(ENTRIES)), ids=IDS)
def test_oracle_extraction_reproduces_fixture(index: int) -> None:
    entry = ENTRIES[index]
    record = record_of(entry)
    oracle = ADAPTER.build_oracle(record)
    assert oracle == entry["oracle"]
    assert ADAPTER.split_cell(record) == entry["split_cell"]
    assert ADAPTER.semantic_key(record) == entry["semantic_key"]
    annotation, certificate, answer = ADAPTER.sft_certificate(oracle, record.instance_seed)
    assert {"annotation": annotation, "certificate": certificate, "answer": answer} == entry["sft"]
    assert list(certificate)[0] == "kind" and type(answer) is int and answer == record.answer_gt["value"]
    assert annotation == record.annotation_gt["value"]
    routes = oracle_paths(oracle)
    assert routes == entry["optimal_paths"] and len(routes) == oracle["n_optimal"] >= 1
    positions = {t["tile_id"]: [t["row"], t["col"]] for t in record.scene_ir["tiles"]}
    stored = [positions[t] for t in record.scene_ir["relations"]["shortest_path_tile_ids"]]
    assert stored in routes  # the stored tie-break witness is one optimum, not the only one
    assert all(len(route) >= oracle["min_tiles"] for route in routes)


def test_fixtures_cover_every_profile_and_nonunique_optima() -> None:
    cells = Counter(entry["split_cell"] for entry in ENTRIES)
    assert set(cells) == {"landscape", "portrait", "square"}
    assert set(ADAPTER.train_cells) | set(ADAPTER.heldout_cells) == set(cells)
    assert "square" in ADAPTER.heldout_cells and "square" not in ADAPTER.train_cells
    assert sum(entry["oracle"]["n_optimal"] > 1 for entry in ENTRIES) >= 4
    assert len({entry["semantic_key"] for entry in ENTRIES}) == len(ENTRIES)


# ------------------------------------------------------------------------------------------
# oracle round trip (DESIGN 9 class 1)


def _assert_perfect(result: ScoreResult) -> None:
    assert (result.answer, result.grounding, result.semantics, result.format) == (1.0, 1.0, 1.0, 1.0)
    assert result.joint == 1.0 and result.error_class == "correct"
    assert all(c.score == 1.0 for c in result.checks if c.applicable), [c for c in result.checks if c.score < 1]


@pytest.mark.parametrize("index", range(len(ENTRIES)), ids=IDS)
def test_unrounded_sft_target_scores_one(prepared, index: int) -> None:
    item = prepared[index]
    result = score(item["entry"]["sft"], item["record"], item["ext"], shown_frame_diagnostics=True)
    _assert_perfect(result)
    assert result.diagnostics["M_shown"] == 1.0  # tile points survive Qwen2.5-VL's 28-px resize
    assert result.diagnostics["adapter"]["legal_move_frac"] == 1.0
    assert result.diagnostics["adapter"]["path_cost"] == item["entry"]["oracle"]["opt"]


@pytest.mark.parametrize("index", range(len(ENTRIES)), ids=IDS)
def test_rounded_sft_target_resolves_every_box(prepared, index: int) -> None:
    item = prepared[index]
    obj = sft_obj(item["entry"])
    obj["annotation"] = {role: round_box_half_up(box) for role, box in obj["annotation"].items()}
    text = json.dumps(obj, separators=(",", ":"))
    assert ".0" not in text  # the serialized SFT target is integers only
    result = score(text, item["record"], item["ext"])
    assert (result.answer, result.semantics, result.format) == (1.0, 1.0, 1.0)
    assert 0.0 <= result.grounding <= 1.0  # logged; tile boxes are integral, so it stays 1 here
    identity = result.diagnostics["identity"]
    assert {r["status"] for role in ("player_cell", "target_cell") for r in identity[role]} == {"resolved"}
    assert check(result, FAM_LEGAL).detail["resolved_steps"] == len(obj["certificate"]["steps"])


@pytest.mark.parametrize("index", range(len(ENTRIES)), ids=IDS)
def test_every_optimal_path_scores_identically(prepared, index: int) -> None:
    item = prepared[index]
    reference = family_scores(item["sft_score"])
    for route in item["entry"]["optimal_paths"]:
        result = score(path_obj(item["entry"], route), item["record"], item["ext"])
        _assert_perfect(result)
        assert family_scores(result) == reference


@pytest.mark.parametrize("index", range(len(ENTRIES)), ids=IDS)
def test_rel1000_frame_round_trip(prepared, index: int) -> None:
    item = prepared[index]
    width, height = item["record"].source_size()
    obj = sft_obj(item["entry"])
    obj["annotation"] = {
        role: [b[0] * 1000 / width, b[1] * 1000 / height, b[2] * 1000 / width, b[3] * 1000 / height]
        for role, b in obj["annotation"].items()
    }
    for step in obj["certificate"]["steps"]:
        step["at"] = [step["at"][0] * 1000 / width, step["at"][1] * 1000 / height]
    result = score(obj, item["record"], item["ext"], coord_frame="rel1000")
    assert (result.answer, result.semantics, result.format) == (1.0, 1.0, 1.0)
    assert result.grounding >= 0.999


# ------------------------------------------------------------------------------------------
# counterexamples (DESIGN 9 class 2)


@pytest.mark.parametrize("index", range(len(ENTRIES)), ids=IDS)
def test_every_corruption_fails_exactly_its_checks(prepared, index: int) -> None:
    item = prepared[index]
    oracle = item["entry"]["oracle"]
    corruptions = ADAPTER.corruptions(oracle, item["entry"]["sft"], item["record"].instance_seed)
    assert ALWAYS_PRESENT <= {c.name for c in corruptions}
    for corruption in corruptions:
        assert set(ADAPTER.m_families) | set(SIGNALS) <= set(corruption.expected), corruption.name
        result = score(corruption.response_obj, item["record"], item["ext"])
        problems = corruption_failures(item["sft_score"], result, corruption.expected)
        assert not problems, (corruption.name, problems)
        assert result.semantics < 1.0 or corruption.name in ("answer_inconsistent", "annotation_shifted", "float_costs")


def test_corruptions_cover_design9_and_are_deterministic(prepared) -> None:
    names = set()
    for item in prepared:
        oracle, seed = item["entry"]["oracle"], item["record"].instance_seed
        first = ADAPTER.corruptions(oracle, item["entry"]["sft"], seed)
        again = ADAPTER.corruptions(oracle, item["entry"]["sft"], seed)
        assert [(c.name, c.response_obj, c.expected) for c in first] == [(c.name, c.response_obj, c.expected) for c in again]
        names |= {c.name for c in first}
    assert names == DESIGN9


def test_corruptions_keep_g_unchanged_while_lowering_m(prepared) -> None:
    """Go/no-go (b): at least 3 corruption types leave G unchanged while lowering M."""
    separating = set()
    for item in prepared:
        for corruption in ADAPTER.corruptions(item["entry"]["oracle"], item["entry"]["sft"], 0):
            result = score(corruption.response_obj, item["record"], item["ext"])
            if abs(result.grounding - item["sft_score"].grounding) < 0.01 and result.semantics < 1.0:
                separating.add(corruption.name)
    assert len(separating) >= 3 and {"wrong_cost_consistent", "suboptimal_path", "all_costs_one"} <= separating


def test_teleports_and_suboptimal_paths(prepared) -> None:
    for item in prepared:
        oracle = item["entry"]["oracle"]
        m = oracle["min_tiles"]
        by_name = {c.name: c for c in ADAPTER.corruptions(oracle, item["entry"]["sft"], 1)}
        for name in ("teleport_start_target", "teleport_start_start"):
            if name in by_name:
                result = score(by_name[name].response_obj, item["record"], item["ext"])
                assert check(result, FAM_COST).score <= 2.0 / m + 1e-12
                assert check(result, FAM_LEGAL).score == 0.0 and check(result, FAM_OPTIMAL).score == 0.0
        if "suboptimal_path" in by_name:
            result = score(by_name["suboptimal_path"].response_obj, item["record"], item["ext"])
            kept = [check(result, f).score for f in (FAM_ENDPOINTS, FAM_LEGAL, FAM_COST, FAM_TOTAL)]
            assert kept == [1.0] * 4 and check(result, FAM_OPTIMAL).score == 0.0
            assert result.error_class == "computation"
            assert result.diagnostics["adapter"]["cost_ratio"] < 1.0


def test_error_classes_follow_the_first_failing_stage(prepared) -> None:
    item = prepared[0]
    by_name = {c.name: c for c in ADAPTER.corruptions(item["entry"]["oracle"], item["entry"]["sft"], 0)}
    classes = {name: score(by_name[name].response_obj, item["record"], item["ext"]).error_class for name in by_name}
    assert classes["reversed_endpoints"] == "grounding"
    assert classes["wrong_cost_consistent"] == "operand"
    assert classes["wrong_total"] == "computation"
    assert classes["answer_inconsistent"] == "answer_inconsistent"
    assert classes["claims_unreachable"] == "grounding"


# ------------------------------------------------------------------------------------------
# property test: wrong operands with self-consistent arithmetic never reach M = 1


@pytest.mark.parametrize("index", range(len(ENTRIES)), ids=IDS)
def test_wrong_operands_with_consistent_arithmetic(prepared, index: int) -> None:
    item = prepared[index]
    rng = random.Random(9000 + index)
    for _ in range(12):
        obj = sft_obj(item["entry"])
        steps = obj["certificate"]["steps"]
        chosen = rng.sample(range(len(steps)), rng.randint(1, len(steps)))
        for position in chosen:
            steps[position]["cost"] = rng.choice([v for v in range(0, 5) if v != steps[position]["cost"]])
        obj["certificate"]["total"] = obj["answer"] = sum(step["cost"] for step in steps)
        result = score(obj, item["record"], item["ext"])
        assert result.semantics < 1.0
        assert check(result, FAM_COST).score < 1.0 and check(result, FAM_TOTAL).score == 0.0
        assert check(result, "pa.total.local").score == 1.0 and check(result, "pa.answer.local").score == 1.0


# ------------------------------------------------------------------------------------------
# equivalence (DESIGN 9 class 3)


def _reorder(value: Any, rng: random.Random) -> Any:
    if isinstance(value, dict):
        keys = list(value)
        rng.shuffle(keys)
        return {key: _reorder(value[key], rng) for key in keys}
    if isinstance(value, list):
        return [_reorder(item, rng) for item in value]
    return value


@pytest.mark.parametrize("index", range(0, len(ENTRIES), 3), ids=IDS[::3])
def test_key_order_and_point_position_never_matter(prepared, index: int) -> None:
    item = prepared[index]
    reference = item["sft_score"]
    rng = random.Random(index)
    for _ in range(4):
        result = score(_reorder(item["entry"]["sft"], rng), item["record"], item["ext"])
        assert family_scores(result) == family_scores(reference) and result.format == 1.0
    obj = sft_obj(item["entry"])
    for step in obj["certificate"]["steps"]:  # any interior point of the tile, not only its centre
        step["at"] = [step["at"][0] + rng.uniform(-39.0, 39.0), step["at"][1] + rng.uniform(-39.0, 39.0)]
    assert family_scores(score(obj, item["record"], item["ext"])) == family_scores(reference)


def test_trailing_prose_costs_only_f(prepared) -> None:
    item = prepared[1]
    text = "Route found.\n" + json.dumps(item["entry"]["sft"]) + "\nThat is the cheapest route."
    result = score(text, item["record"], item["ext"])
    assert (result.answer, result.grounding, result.semantics) == (1.0, 1.0, 1.0) and result.format == 0.0
    assert family_scores(result) == family_scores(item["sft_score"])


def test_scores_are_deterministic(prepared) -> None:
    item = prepared[2]
    first, second = (score(item["entry"]["sft"], item["record"], item["ext"]).to_dict() for _ in range(2))
    for payload in (first, second):
        payload["diagnostics"].pop("verifier_ms")
        payload["diagnostics"].pop("slow_row")
    assert first == second


# ------------------------------------------------------------------------------------------
# edge cases


def _with_steps(entry: Mapping[str, Any], steps: List[Any], total: Optional[int] = None) -> Dict[str, Any]:
    obj = sft_obj(entry)
    obj["certificate"]["steps"] = steps
    obj["certificate"]["total"] = obj["answer"] = total if total is not None else 0
    return obj


def test_empty_single_and_over_cap_routes(prepared) -> None:
    item = prepared[0]
    empty = score(_with_steps(item["entry"], []), item["record"], item["ext"])
    assert empty.semantics == 0.0 and empty.error_class == "grounding"
    assert check(empty, FAM_LEGAL).error_code == "too_short"
    first = item["entry"]["sft"]["certificate"]["steps"][0]
    single = score(_with_steps(item["entry"], [first]), item["record"], item["ext"])
    assert check(single, FAM_ENDPOINTS).score == 0.5 and check(single, FAM_LEGAL).score == 0.0
    assert check(single, FAM_COST).score == pytest.approx(1 / item["entry"]["oracle"]["min_tiles"])
    over = score(_with_steps(item["entry"], [first] * 33), item["record"], item["ext"])
    assert over.semantics == 0.0 and over.diagnostics["envelope_error"] == "budget"
    assert all(not c.valid_input for c in over.checks)


def test_malformed_steps_are_invalid_claims(prepared) -> None:
    item = prepared[3]
    obj = sft_obj(item["entry"])
    steps = obj["certificate"]["steps"]
    steps[1] = {"at": steps[1]["at"], "cost": steps[1]["cost"], "terrain": "grass"}  # extra key
    steps[2] = [steps[2]["at"], steps[2]["cost"]]  # not an object
    result = score(obj, item["record"], item["ext"])
    assert result.format == 0.0 and check(result, FAM_LEGAL).error_code == "invalid_step"
    n = len(steps)
    # cost credit ends at the first invalid step (only step 0 is on the legal prefix)
    assert check(result, FAM_COST).score == pytest.approx(1 / max(n, item["entry"]["oracle"]["min_tiles"]))


def test_boundary_point_is_unresolved(prepared) -> None:
    item = prepared[4]
    obj = sft_obj(item["entry"])
    grid = Grid.from_oracle(item["entry"]["oracle"])
    x, y = obj["certificate"]["steps"][0]["at"]
    obj["certificate"]["steps"][0]["at"] = [x + grid.tile_px / 2, y]  # on the shared vertical edge
    result = score(obj, item["record"], item["ext"])
    assert check(result, FAM_ENDPOINTS).score == 0.5
    assert check(result, FAM_LEGAL).error_code == "unresolved_step"


def test_reachable_false_and_missing_steps(prepared) -> None:
    item = prepared[5]
    obj = sft_obj(item["entry"])
    obj["certificate"] = {"kind": "path", "reachable": False}
    result = score(obj, item["record"], item["ext"])
    assert result.semantics == 0.0 and result.format == 1.0 and result.answer == 1.0
    assert all(not c.applicable for c in result.checks if c.kind == "local")
    obj["certificate"] = {"kind": "path", "reachable": False, "steps": [], "total": 0}
    assert score(obj, item["record"], item["ext"]).format == 0.0  # extra keys on the unreachable form
    obj["certificate"] = {"kind": "path", "reachable": True, "total": 3}
    missing = score(obj, item["record"], item["ext"])
    assert missing.diagnostics["envelope_error"] == "missing_list" and missing.semantics == 0.0
    obj["certificate"] = {"kind": "path", "reachable": "true", "steps": item["entry"]["sft"]["certificate"]["steps"],
                          "total": item["entry"]["oracle"]["opt"]}
    loose = score(obj, item["record"], item["ext"])
    assert loose.semantics == 1.0 and loose.format == 0.0  # a mistyped flag costs F only


FUZZ_VALUES: List[Any] = [
    None, True, False, -1, 0, 1, 3, 2.5, 7.0, 1e308, -1e308, "1", [], {}, [1], [1, 2, 3],
    [True, 1], [400, 300], [-40, 40], [0, 0], {"at": [1, 1]},
]


def test_fuzzed_certificates_never_raise(prepared) -> None:
    rng = random.Random(4242)
    for round_index in range(300):
        item = prepared[round_index % len(prepared)]
        obj = sft_obj(item["entry"])
        cert = obj["certificate"]
        for _ in range(rng.randint(1, 4)):
            where = rng.choice(("cert", "step", "steps", "annotation", "answer"))
            if where == "cert":
                cert[rng.choice(("reachable", "total", "steps", "extra"))] = rng.choice(FUZZ_VALUES)
            elif where == "step" and isinstance(cert.get("steps"), list) and cert["steps"]:
                step = rng.choice(cert["steps"])
                if isinstance(step, dict):
                    step[rng.choice(("at", "cost", "x"))] = rng.choice(FUZZ_VALUES)
            elif where == "steps" and isinstance(cert.get("steps"), list):
                cert["steps"].insert(rng.randint(0, len(cert["steps"])), rng.choice(FUZZ_VALUES))
            elif where == "annotation":
                obj["annotation"][rng.choice(("player_cell", "target_cell", "other"))] = rng.choice(FUZZ_VALUES)
            else:
                obj["answer"] = rng.choice(FUZZ_VALUES)
        for frame in ("px_source", "rel1000"):
            result = score(obj, item["record"], item["ext"], coord_frame=frame, shown_frame_diagnostics=True)
            for value in (result.answer, result.grounding, result.semantics, result.format):
                assert 0.0 <= value <= 1.0
            assert result.joint in (0.0, 1.0)


def _walled_oracle() -> Dict[str, Any]:
    """5x5 map whose target (2, 4) is enclosed by water (unreachable)."""
    cost: List[List[Optional[int]]] = [[1] * 5 for _ in range(5)]
    for r, c in ((1, 4), (3, 4), (2, 3)):
        cost[r][c] = None
    return grid_oracle(rows=5, cols=5, tile_px=80, profile="square", cost=cost, start=(2, 0), target=(2, 4))


def test_unreachable_form_on_an_unreachable_oracle() -> None:
    oracle = _walled_oracle()
    assert oracle["reachable"] is False and oracle["opt"] is None and oracle["n_optimal"] == 0
    view = ADAPTER.validate_certificate({"kind": "path", "reachable": False}, 0)
    from trace_tasks.experimental.certificates.adapters.base import PublicContext
    from trace_tasks.experimental.certificates.geometry import annotation_view, source_frame

    frame = source_frame("px_source", (400, 400))
    ctx = PublicContext(TASK_ID, "single", (400, 400), "px_source", frame)
    annotation = annotation_view({"player_cell": [0, 160, 80, 240], "target_cell": [320, 160, 400, 240]}, "bbox_map", frame)
    right = ADAPTER.checks(view, annotation, oracle, ctx)
    assert [c.score for c in right if c.kind == "truth"] == [1.0] * 5
    route = ADAPTER.validate_certificate(
        {"kind": "path", "reachable": True, "steps": [{"at": [40, 200], "cost": 0}], "total": 0}, 0
    )
    assert [c.score for c in ADAPTER.checks(route, annotation, oracle, ctx) if c.kind == "truth"] == [0.0] * 5
    with pytest.raises(QuarantineError):
        ADAPTER.sft_certificate(oracle, 0)


# ------------------------------------------------------------------------------------------
# export invariants -> QuarantineError


def _mutated(index: int, mutate) -> TraceRecord:
    entry = copy.deepcopy(ENTRIES[index])
    mutate(entry)
    return record_of(entry)


def _tile(entry: Dict[str, Any], tile_id: str) -> Dict[str, Any]:
    return next(t for t in entry["trace"]["scene_ir"]["tiles"] if t["tile_id"] == tile_id)


def _relations(entry: Dict[str, Any]) -> Dict[str, Any]:
    return entry["trace"]["scene_ir"]["relations"]


def _wall_target(entry: Dict[str, Any]) -> None:
    target = _tile(entry, _relations(entry)["target_tile_id"])
    for tile in entry["trace"]["scene_ir"]["tiles"]:
        if abs(tile["row"] - target["row"]) + abs(tile["col"] - target["col"]) == 1:
            tile.update(terrain="water", passable=False, movement_cost=None)
    entry["trace"]["execution_trace"]["renderer"].pop("terrain_rows", None)


def _first_path_tile(entry: Dict[str, Any]) -> Dict[str, Any]:
    return _tile(entry, _relations(entry)["shortest_path_tile_ids"][1])


QUARANTINE_CASES = {
    "answer_mismatch": lambda e: e["train"]["answer_gt"].update(value=e["train"]["answer_gt"]["value"] + 1),
    "tile_cost_mismatch": lambda e: _first_path_tile(e).update(movement_cost=7),
    "tile_geometry_mismatch": lambda e: e["trace"]["scene_ir"]["tiles"][0]["bbox"].__setitem__(2, 81.0),
    "unknown_terrain": lambda e: _first_path_tile(e).update(terrain="lava"),
    "terrain_grid_mismatch": lambda e: e["trace"]["execution_trace"]["renderer"]["terrain_rows"][0].__setitem__(0, "lava"),
    "terrain_costs_mismatch": lambda e: _relations(e)["terrain_movement_costs"].update(forest=4),
    "annotation_mismatch": lambda e: e["train"]["annotation_gt"]["value"]["target_cell"].__setitem__(0, -1.0),
    "unit_not_on_start": lambda e: e["trace"]["scene_ir"]["units"][0].update(tile_id="nowhere"),
    "unknown_endpoint": lambda e: _relations(e).update(target_tile_id="r99_c99"),
    "degenerate_query": lambda e: _relations(e).update(target_tile_id=_relations(e)["start_tile_id"]),
    "canvas_mismatch": lambda e: e.update(image_size=[e["image_size"][0] + 1, e["image_size"][1]]),
    "unknown_profile": lambda e: e["trace"]["execution_trace"]["renderer"].update(canvas_profile="panorama"),
    "missing_field": lambda e: e["trace"]["execution_trace"].pop("renderer"),
    "unsupported_row": lambda e: e["train"].update(query_id="other"),
    "stored_path_inconsistent": lambda e: _relations(e)["shortest_path_tile_ids"].reverse(),
    "dist_mismatch": lambda e: e["trace"]["render_map"]["movement_costs_by_tile_id"].update(
        {_relations(e)["target_tile_id"]: 99}
    ),
    "unreachable_target": _wall_target,
}


@pytest.mark.parametrize("reason", sorted(QUARANTINE_CASES))
def test_export_invariants_quarantine(reason: str) -> None:
    index = 1  # landscape map with several optima
    record = _mutated(index, QUARANTINE_CASES[reason])
    with pytest.raises(QuarantineError) as raised:
        ADAPTER.build_oracle(record)
    assert raised.value.reason == reason


def test_profile_mismatch_is_quarantined() -> None:
    entry = next(e for e in ENTRIES if e["split_cell"] == "landscape")
    mutated = copy.deepcopy(entry)
    mutated["trace"]["execution_trace"]["renderer"]["canvas_profile"] = "portrait"
    with pytest.raises(QuarantineError) as raised:
        ADAPTER.split_cell(record_of(mutated))
    assert raised.value.reason == "profile_mismatch"


# ------------------------------------------------------------------------------------------
# graph search and SFT sampling


def _brute_force_optima(oracle: Mapping[str, Any]) -> List[List[int]]:
    grid = Grid.from_oracle(oracle)
    start, target = grid.index(*oracle["start"]), grid.index(*oracle["target"])
    best: List[List[int]] = []
    best_cost = [None]

    def walk(route: List[int], cost: int) -> None:
        if best_cost[0] is not None and cost > best_cost[0]:
            return
        if route[-1] == target:
            if best_cost[0] is None or cost < best_cost[0]:
                best_cost[0], best[:] = cost, []
            best.append(list(route))
            return
        for v in grid.neighbors(route[-1]):
            if grid.passable(v) and v not in route:
                walk(route + [v], cost + grid.entry[v])

    walk([start], 0)
    return sorted(best)


def test_dijkstra_labels_match_brute_force() -> None:
    cost: List[List[Optional[int]]] = [[1, 2, 1, 1], [1, None, 3, 1], [1, 1, 1, 2], [2, 1, None, 1]]
    oracle = grid_oracle(rows=4, cols=4, tile_px=80, profile="square", cost=cost, start=(0, 0), target=(3, 3))
    grid = Grid.from_oracle(oracle)
    brute = _brute_force_optima(oracle)
    enumerated = sorted([grid.index(r, c) for r, c in route] for route in oracle_paths(oracle))
    assert enumerated == brute and oracle["n_optimal"] == len(brute)
    assert oracle["opt"] == sum(grid.entry[t] for t in brute[0][1:])
    assert oracle["min_tiles"] == 7


def test_sft_sampling_is_seeded_and_capped() -> None:
    open_map = grid_oracle(
        rows=8, cols=8, tile_px=80, profile="square", cost=[[1] * 8 for _ in range(8)], start=(0, 0), target=(5, 5)
    )
    assert open_map["n_optimal"] == 252 and len(oracle_paths(open_map, cap=16)) == 16
    first16 = oracle_paths(open_map, cap=16)
    chosen = []
    for seed in range(64):
        _, cert, _ = ADAPTER.sft_certificate(open_map, seed)
        assert ADAPTER.sft_certificate(open_map, seed)[1] == cert
        grid = Grid.from_oracle(open_map)
        route = [list(grid.rc(r.candidate)) for r in resolve_points([s["at"] for s in cert["steps"]], grid.boxes)]
        assert route in first16
        chosen.append(json.dumps(route))
    assert len(set(chosen)) >= 8


def test_sft_sampling_covers_alternative_optima() -> None:
    entry = max(ENTRIES, key=lambda e: e["oracle"]["n_optimal"])
    assert 1 < entry["oracle"]["n_optimal"] <= 16
    routes = {json.dumps(path_obj(entry, r)["certificate"]["steps"]) for r in entry["optimal_paths"]}
    drawn = {json.dumps(ADAPTER.sft_certificate(entry["oracle"], seed)[1]["steps"]) for seed in range(200)}
    assert drawn == routes  # every optimum is a possible SFT witness


# ------------------------------------------------------------------------------------------
# prompt pieces (public, instance-independent)


def test_prompt_pieces_are_public_constants() -> None:
    line = ADAPTER.certificate_line()
    example = json.loads(ADAPTER.example_json())
    assert list(example) == ["annotation", "certificate", "answer"]
    for token in ('"kind": "path"', '"reachable": false', "at most 32 steps", "0 for the starting tile"):
        assert token in line.lower()
    for text in (line, ADAPTER.example_json()):
        for leak in ("r0", "_c0", "tile_id", "shortest_path", "movement_costs_by", "dist"):
            assert leak not in text
    cert = example["certificate"]
    assert list(cert)[0] == "kind" and cert["total"] == sum(s["cost"] for s in cert["steps"]) == example["answer"]
    coords = [v for s in cert["steps"] for v in s["at"]] + [v for b in example["annotation"].values() for v in b]
    assert all(type(v) is int and 0 <= v <= 1000 for v in coords)


def test_example_is_a_valid_certificate_on_a_matching_map() -> None:
    """The fixed example is self-consistent: on a map built to match it, it scores A=G=M=F=1."""
    cost: List[List[Optional[int]]] = [[3] * 6 for _ in range(6)]
    cost[5][2], cost[5][3], cost[5][4], cost[4][4], cost[4][3] = 1, 2, 1, 3, None
    oracle = grid_oracle(rows=6, cols=6, tile_px=80, profile="square", cost=cost, start=(4, 2), target=(4, 4))
    assert oracle["opt"] == 7 and oracle["n_optimal"] == 1
    example = ADAPTER.example_object()
    record_id = "blake3:" + "e" * 64
    identity = ADAPTER.identity()
    ext = CertificateExtension(
        certificate_extension_id(instance_id=record_id, trace_record_hash=record_id, adapter_id=identity["id"],
                                 adapter_version=identity["version"], schema_id=identity["schema_id"]),
        {"instance_id": record_id, "trace_record_hash": record_id, "task": TASK_ID, "query_id": "single", "instance_seed": 0},
        identity,
        {"answer_gt": {"type": "integer", "value": 7},
         "annotation_gt": {"type": "bbox_map", "value": example["annotation"]},
         "reward_contract": resolve_reward_contract(answer_type="integer", annotation_type="bbox_map").to_dict()},
        {"image_size": [480, 480]},
        oracle,
    )
    result = score_certificate_response(ADAPTER.example_json(), {"instance_id": record_id}, ext,
                                        coord_frame="px_source", adapter=ADAPTER)
    _assert_perfect(result)


# ------------------------------------------------------------------------------------------
# maintenance: fixture regeneration and the large sweep


def _build(work_dir: Path, count: int, sampling_seed: int, workers: int = 1) -> List[TraceRecord]:
    from trace_tasks import BuildConfig, BuildTaskConfig, build_dataset
    from trace_tasks.experimental.certificates.types import read_trace_records

    work_dir.mkdir(parents=True, exist_ok=True)
    config = BuildConfig(
        output_root=str(work_dir),
        dataset_name=f"cert_path_{sampling_seed}",
        instance_version="v0",
        image_format="png",
        tasks=[BuildTaskConfig(task_id=TASK_ID, count=count)],
        max_attempts_per_instance=100,
        sampling_seed=sampling_seed,
        workers=workers,
    )
    build_dataset(config, code_hash="cert-v1")
    datasets = sorted((work_dir / "datasets").iterdir(), key=lambda p: p.stat().st_mtime)
    return read_trace_records(datasets[-1])


def build_fixtures(work_dir: Path) -> Dict[str, Any]:
    records = _build(work_dir, FIXTURE_COUNT, FIXTURE_SAMPLING_SEED)
    entries = []
    for record in records:
        oracle = ADAPTER.build_oracle(record)
        annotation, certificate, answer = ADAPTER.sft_certificate(oracle, record.instance_seed)
        entries.append(
            {
                "train": dict(record.train),
                "trace": encode_trace(record.trace),
                "image_size": list(record.source_size()),
                "split_cell": ADAPTER.split_cell(record),
                "semantic_key": ADAPTER.semantic_key(record),
                "oracle": oracle,
                "sft": {"annotation": annotation, "certificate": certificate, "answer": answer},
                "optimal_paths": oracle_paths(oracle),
            }
        )
    return {
        "generator": {
            "task": TASK_ID,
            "sampling_seed": FIXTURE_SAMPLING_SEED,
            "count": FIXTURE_COUNT,
            "code_hash": "cert-v1",
            "dropped_trace_keys": list(DROPPED_TRACE_KEYS),
            "tiles_encoding": "scene_ir.tiles == 'terrain_rows': rebuild with decode_trace()",
            "command": "python -m tests.test_certificate_adapter_path --regen <scratch_dir>",
        },
        "adapter": ADAPTER.identity(),
        "records": entries,
    }


def sweep(work_dir: Path, count: int, sampling_seed: int) -> Dict[str, Any]:
    """Round trip + corruptions on freshly generated records (CPU only, scratch output)."""
    import time

    records = _build(work_dir, count, sampling_seed)
    stats: Dict[str, Counter] = {"roundtrip": Counter(), "corruption": Counter(), "present": Counter()}
    quarantined: Counter = Counter()
    failures: List[Any] = []
    cells: Counter = Counter()
    timings: List[float] = []
    started = time.perf_counter()
    for record in records:
        try:
            oracle = ADAPTER.build_oracle(record)
        except QuarantineError as exc:
            quarantined[exc.reason] += 1
            continue
        cells[ADAPTER.split_cell(record)] += 1
        extension = make_extension(record, oracle)
        annotation, certificate, answer = ADAPTER.sft_certificate(oracle, record.instance_seed)
        obj = {"annotation": annotation, "certificate": certificate, "answer": answer}
        base = score(obj, record, extension, shown_frame_diagnostics=True)
        timings.append(base.diagnostics["verifier_ms"])
        checks = {
            "unrounded": (base.answer, base.grounding, base.semantics, base.format, base.diagnostics["M_shown"]) == (1, 1, 1, 1, 1),
        }
        rounded = dict(obj, annotation={role: round_box_half_up(box) for role, box in annotation.items()})
        result = score(rounded, record, extension)
        checks["rounded"] = (result.answer, result.semantics, result.format) == (1, 1, 1)
        routes = oracle_paths(oracle, cap=64)
        entry = {"oracle": oracle, "sft": obj}
        checks["all_optima"] = all(
            score(path_obj(entry, route), record, extension).semantics == 1.0 for route in routes
        )
        for name, ok in checks.items():
            stats["roundtrip"][f"{name}_pass" if ok else f"{name}_fail"] += 1
            if not ok:
                failures.append((record.instance_id, name))
        for corruption in ADAPTER.corruptions(oracle, obj, record.instance_seed):
            stats["present"][corruption.name] += 1
            problems = corruption_failures(base, score(corruption.response_obj, record, extension), corruption.expected)
            stats["corruption"][corruption.name + ("" if not problems else ":FAIL")] += 1
            if problems:
                failures.append((record.instance_id, corruption.name, problems))
    timings.sort()
    return {
        "records": len(records),
        "quarantined": dict(quarantined),
        "cells": dict(cells),
        "roundtrip": dict(stats["roundtrip"]),
        "corruptions_present": dict(sorted(stats["present"].items())),
        "corruption_results": dict(sorted(stats["corruption"].items())),
        "failures": failures[:20],
        "n_failures": len(failures),
        "verifier_ms_p50_p99_max": [timings[len(timings) // 2], timings[int(len(timings) * 0.99)], timings[-1]] if timings else [],
        "wall_s": time.perf_counter() - started,
    }


if __name__ == "__main__":  # pragma: no cover - maintenance entry points
    if len(sys.argv) == 3 and sys.argv[1] == "--regen":
        payload = build_fixtures(Path(sys.argv[2]))
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
        print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes, {len(payload['records'])} records)")
    elif len(sys.argv) >= 3 and sys.argv[1] == "--sweep":
        count = int(sys.argv[3]) if len(sys.argv) > 3 else 400
        seed = int(sys.argv[4]) if len(sys.argv) > 4 else 7091
        print(json.dumps(sweep(Path(sys.argv[2]), count, seed), indent=1, default=str))
    else:
        raise SystemExit(
            "usage: python -m tests.test_certificate_adapter_path --regen <scratch_dir>\n"
            "       python -m tests.test_certificate_adapter_path --sweep <scratch_dir> [count] [sampling_seed]"
        )
