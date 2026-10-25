"""Oracle round trip of every certificate adapter through the plan API (DESIGN 9).

For each adapter's 16 frozen fixture records (and every enumerated optimal path of the rpg
fixtures), the private extension is built with ``export.build_extension`` and:

- the oracle object with **unrounded** ``annotation_gt`` boxes scores A = G = M = F = 1;
- the serialized SFT target (``make_sft_target``: boxes rounded half-up) scores A = M = F = 1
  with every annotation box resolved (its G is only logged, DESIGN 9);
- the SFT object satisfies the public schema and the extension the private schema;
- export -> store -> reload reproduces the extension id, the scores and the public prompt sha.

The per-adapter files (``test_certificate_adapter_*.py``) pin the fixtures themselves; this file
checks the shared plan API on all of them. ``fixture_cases`` / ``extension_of`` / ``score`` are
reused by the counterexample and equivalence tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pytest

from tests.test_certificate_prompts import validate
from trace_tasks.experimental.certificates import export, prompts, registry
from trace_tasks.experimental.certificates.adapters.base import Adapter
from trace_tasks.experimental.certificates.adapters.path import Grid, oracle_paths, steps_for
from trace_tasks.experimental.certificates.scoring import score_certificate_response
from trace_tasks.experimental.certificates.types import CertificateExtension, ScoreResult, TraceRecord

ADAPTERS = export.ADAPTER_ORDER


@lru_cache(maxsize=None)
def fixture_records(name: str) -> Tuple[TraceRecord, ...]:
    """The persisted fixture records of one adapter, decoded by that adapter's own test file."""
    if name == "selected_set":
        from tests.test_certificate_adapter_selected_set import FIXTURE, record_of

        entries = json.loads(FIXTURE.read_text(encoding="utf-8"))["records"]
        return tuple(record_of(entry) for entry in entries)
    if name == "filtered_mean":
        from tests.test_certificate_adapter_filtered_mean import FIXTURES, fixture_record

        return tuple(fixture_record(entry) for entry in FIXTURES)
    if name == "weighted_sum":
        from tests.test_certificate_adapter_weighted_sum import FIXTURE, record_of

        entries = json.loads(FIXTURE.read_text(encoding="utf-8"))["records"]
        return tuple(record_of(entry) for entry in entries)
    if name == "path":
        from tests.test_certificate_adapter_path import ENTRIES, record_of

        return tuple(record_of(entry) for entry in ENTRIES)
    raise KeyError(name)


@dataclass(frozen=True)
class FixtureCase:
    adapter_name: str
    index: int

    @property
    def adapter(self) -> Adapter:
        return registry.load_adapter(self.adapter_name)

    @property
    def record(self) -> TraceRecord:
        return fixture_records(self.adapter_name)[self.index]

    def __str__(self) -> str:
        return f"{self.adapter_name}-{self.index:02d}"


def fixture_cases() -> List[FixtureCase]:
    return [FixtureCase(name, i) for name in ADAPTERS for i in range(len(fixture_records(name)))]


CASES = fixture_cases()
CASE_IDS = [str(case) for case in CASES]


def public_prompts(case: FixtureCase) -> Optional[Dict[str, str]]:
    """All public prompts of a fixture row (None when the trimmed fixture has no prompt variants)."""
    if not isinstance(case.record.train.get("prompt_variants"), Mapping):
        return None
    return export.render_prompts(case.record.train, case.adapter, case.record.source_size())


@lru_cache(maxsize=None)
def _extension(case: FixtureCase) -> CertificateExtension:
    record, adapter = case.record, case.adapter
    rendered = public_prompts(case)
    shas = None
    if rendered is not None:
        shas = {frame: export.prompt_sha256(rendered[export.FRAME_CONDITION[frame]]) for frame in export.FRAME_CONDITION}
    cell = adapter.split_cell(record)
    return export.build_extension(
        record, None, adapter, split="fixtures", sampling_seed=7001, seed_index=case.index, cell=cell,
        holdout=adapter.holdout_type(cell), prompt_sha256=shas,
    )


def extension_of(case: FixtureCase) -> CertificateExtension:
    """The fixture's private extension (cached; callers must not mutate it)."""
    return _extension(case)


def oracle_object(ext: CertificateExtension) -> Dict[str, Any]:
    """The unrounded oracle response object (the extension's SFT target before rounding)."""
    return {key: ext.sft_target[key] for key in ("annotation", "certificate", "answer")}


def score(response: Any, ext: CertificateExtension, *, adapter: Optional[Adapter] = None,
          coord_frame: str = "px_source") -> ScoreResult:
    text = response if isinstance(response, str) else json.dumps(response)
    return score_certificate_response(text, {"instance_id": ext.instance_id}, ext,
                                      coord_frame=coord_frame, adapter=adapter)


def signals(result: ScoreResult) -> Tuple[float, float, float, float]:
    return result.answer, result.grounding, result.semantics, result.format


def score_vector(result: ScoreResult) -> Tuple[Any, ...]:
    """Everything a score reports except wall-clock diagnostics."""
    checks = tuple((c.check_id, c.kind, round(c.score, 12), c.applicable, c.valid_input, c.error_code)
                   for c in result.checks)
    return (*signals(result), result.joint, result.error_class, tuple(sorted(result.overall.items())), checks)


def resolutions(result: ScoreResult) -> List[Mapping[str, Any]]:
    return [item for items in result.diagnostics["identity"].values() for item in items]


# ------------------------------------------------------------------------------------------


def test_every_adapter_has_sixteen_fixtures() -> None:
    assert {name: len(fixture_records(name)) for name in ADAPTERS} == {name: 16 for name in ADAPTERS}


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_extension_is_deterministic_and_reloads(case: FixtureCase) -> None:
    ext = extension_of(case)
    ext.verify_id()
    _extension.cache_clear()
    again = extension_of(case)
    assert again.to_dict() == ext.to_dict()
    reloaded = CertificateExtension.from_dict(json.loads(json.dumps(ext.to_dict(), allow_nan=False)))
    reloaded.verify_id()
    assert reloaded.to_dict() == ext.to_dict()
    assert registry.adapter_for_extension(reloaded) is case.adapter
    assert ext.base["split"] == "fixtures" and ext.base["cell"] == case.adapter.split_cell(case.record)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_unrounded_oracle_object_scores_one(case: FixtureCase) -> None:
    ext = extension_of(case)
    result = score(oracle_object(ext), ext)
    assert signals(result) == (1.0, 1.0, 1.0, 1.0)
    assert result.joint == 1.0 and result.error_class == "correct"
    assert all(item["status"] == "resolved" for item in resolutions(result))
    assert score_vector(score(oracle_object(ext), ext, adapter=case.adapter)) == score_vector(result)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_rounded_sft_target_scores_one_except_logged_g(case: FixtureCase) -> None:
    ext = extension_of(case)
    target = export.make_sft_target(case.record.train, ext)
    assert target == json.dumps(json.loads(target), separators=(",", ":"), ensure_ascii=False)
    assert list(json.loads(target)) == ["annotation", "certificate", "answer"]
    assert list(json.loads(target)["certificate"])[0] == "kind"
    result = score(target, ext)
    assert (result.answer, result.semantics, result.format) == (1.0, 1.0, 1.0)
    assert all(item["status"] == "resolved" for item in resolutions(result))
    assert result.grounding > 0.9  # logged; a <= 0.5 px edge shift costs a little IoU (DESIGN 9)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_rel1000_frame_round_trip(case: FixtureCase) -> None:
    ext = extension_of(case)
    width, height = ext.image_size
    obj = json.loads(json.dumps(oracle_object(ext)))

    def scale(node: Any) -> Any:
        if isinstance(node, list) and len(node) in (2, 4) and all(isinstance(v, (int, float)) for v in node):
            return [v * 1000.0 / (width if i % 2 == 0 else height) for i, v in enumerate(node)]
        if isinstance(node, list):
            return [scale(item) for item in node]
        if isinstance(node, dict):
            return {key: scale(item) for key, item in node.items()}
        return node

    obj["annotation"] = scale(obj["annotation"])
    for step in obj["certificate"].get("steps", []):
        step["at"] = scale(step["at"])
    result = score(obj, ext, coord_frame="rel1000")
    assert (result.answer, result.semantics) == (1.0, 1.0) and result.grounding > 0.999


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_sft_object_and_extension_satisfy_the_versioned_schemas(case: FixtureCase) -> None:
    ext = extension_of(case)
    assert validate(export.sft_target_object(ext), prompts.public_schema(case.adapter.kind)) == []
    assert validate(oracle_object(ext), prompts.public_schema(case.adapter.kind)) == []
    assert validate(json.loads(json.dumps(ext.to_dict())), prompts.private_schema()) == []


@pytest.mark.parametrize("case", [c for c in CASES if c.adapter_name == "path"], ids=lambda c: str(c))
def test_every_optimal_path_scores_one(case: FixtureCase) -> None:
    ext = extension_of(case)
    grid = Grid.from_oracle(ext.oracle)
    routes = oracle_paths(ext.oracle)
    assert len(routes) == ext.oracle["n_optimal"] >= 1
    for route in routes:
        cert = {"kind": "path", "reachable": True,
                "steps": steps_for(grid, [grid.index(r, c) for r, c in route]), "total": ext.oracle["opt"]}
        obj = {"annotation": ext.sft_target["annotation"], "certificate": cert, "answer": ext.oracle["opt"]}
        assert signals(score(obj, ext)) == (1.0, 1.0, 1.0, 1.0), route


def test_store_round_trip_reproduces_ids_scores_and_prompts(tmp_path) -> None:
    extensions = [extension_of(case) for case in CASES]
    path = tmp_path / "private" / export.STORE_NAME
    assert export.write_store(path, [ext.to_dict() for ext in extensions]) == len(CASES)
    rows = export.read_store(path)
    assert [row["certificate_extension_id"] for row in rows] == [ext.certificate_extension_id for ext in extensions]
    rendered = 0
    for case, ext, row in zip(CASES, extensions, rows):
        reloaded = CertificateExtension.from_dict(row)
        reloaded.verify_id()
        assert score_vector(score(oracle_object(reloaded), reloaded)) == score_vector(score(oracle_object(ext), ext))
        prompts_now = public_prompts(case)
        if prompts_now is None:
            assert reloaded.public["prompt_sha256"] == {}
            continue
        rendered += 1
        for frame, condition in export.FRAME_CONDITION.items():
            assert reloaded.public["prompt_sha256"][frame] == export.prompt_sha256(prompts_now[condition])
    assert rendered >= 48  # the trimmed selected_set fixtures carry no prompt variants
