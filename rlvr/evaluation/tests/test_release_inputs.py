from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = EVALUATION_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import build_release_results  # noqa: E402
import validate_release_inputs as validator  # noqa: E402


def _load(name: str) -> dict:
    return json.loads(
        (EVALUATION_ROOT / "trace_eval" / name).read_text(encoding="utf-8")
    )


def _dataset_receipt() -> dict:
    return json.loads(
        (REPO_ROOT / "rlvr" / "dataset_equivalence.v1.json").read_text(encoding="utf-8")
    )


def test_frozen_release_inputs_validate_offline() -> None:
    report = validator.validate_release_inputs()

    assert report["status"] == "ok"
    assert report["suite"] == {
        "benchmarks": 24,
        "categories": 6,
        "rows_per_model_seed": 32_805,
    }
    assert report["provenance"]["benchmarks"] == 24
    assert report["results"]["models"] == 8
    assert report["results"]["seeds"] == 3
    assert report["results"]["benchmark_scores"] == 576
    assert report["release_receipts"]["source_archives"] == 3
    assert report["dataset_release"] == {
        "status": "ok",
        "historical_training_revision": validator.HISTORICAL_DATASET_REVISION,
        "compared_revision": validator.COMPARED_DATASET_REVISION,
        "reproduction_dataset_revision": validator.REPRODUCTION_DATASET_REVISION,
        "release_tag": "dataset-v1",
        "receipt_sha256": validator.DATASET_EQUIVALENCE_SHA256,
        "parquet_files": 17,
        "rows": 66_000,
    }
    assert report["trace_validation"] == {
        "models": 8,
        "phase": "release",
        "results_sha256": "1e2b78d46dbbd606210653b584ed77dc436d16bbdc0d329c5aaf55718dc37e11",
        "rows_per_model": 2000,
        "source_revision": "cf0d14aed86db2661d397ce8b68b36171873478d",
        "status": "ok",
    }
    assert report["source_map"] == {
        "files": 128,
        "exact_copy": 68,
        "public_adaptation": 60,
        "public_authored": 11,
    }


def test_dataset_equivalence_receipt_is_hash_and_contract_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _dataset_receipt()
    receipt["aggregate"]["row_order_equal"] = False
    changed = tmp_path / "dataset_equivalence.v1.json"
    changed.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="receipt hash mismatch"):
        validator.validate_dataset_equivalence(changed)

    monkeypatch.setattr(
        validator,
        "DATASET_EQUIVALENCE_SHA256",
        hashlib.sha256(changed.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="aggregate did not pass"):
        validator.validate_dataset_equivalence(changed)


def test_dataset_equivalence_rejects_unsafe_inventory_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _dataset_receipt()
    receipt["compared_current_inventory"]["files"][0]["path"] = "../outside"
    changed = tmp_path / "dataset_equivalence.v1.json"
    changed.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        validator,
        "DATASET_EQUIVALENCE_SHA256",
        hashlib.sha256(changed.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="unsafe path"):
        validator.validate_dataset_equivalence(changed)


def test_canonical_results_remain_byte_frozen() -> None:
    path = EVALUATION_ROOT / "trace_eval" / "results.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "569ea2463ace4c6b505d1fb168957e558667641810e90a8d9ede80f555823c98"
    )
    results = _load("results.json")
    assert results["generation"] == {
        "builder": "scripts/build_trace_eval_release_results.py",
        "historical_final_answer_only_manifest_is_input": False,
        "policy": "recomputed_from_immutable_sanitized_score_metadata",
    }
    assert build_release_results.CANONICAL_SUITE_PROVENANCE_PATH == (
        "evaluation/trace_eval/suite.v1.json"
    )
    assert build_release_results.CANONICAL_BUILDER_PROVENANCE_PATH == (
        "scripts/build_trace_eval_release_results.py"
    )


def test_every_benchmark_has_complete_provenance() -> None:
    suite = _load("suite.v1.json")
    provenance = _load("benchmark_provenance.v1.json")
    rows = provenance["benchmarks"]

    assert [row["benchmark_id"] for row in rows] == [
        row["key"] for row in suite["benchmarks"]
    ]
    for row in rows:
        assert row["source"]["revision"]
        assert row["source"]["split"]
        assert row["license_or_terms"]["status"]
        assert row["license_or_terms"]["evidence"].startswith("http")
        assert row["citations"]


def test_provenance_rejects_a_missing_split() -> None:
    suite = _load("suite.v1.json")
    provenance = _load("benchmark_provenance.v1.json")
    broken = copy.deepcopy(provenance)
    del broken["benchmarks"][0]["source"]["split"]

    with pytest.raises(ValueError, match="source.split"):
        validator.validate_provenance(suite, broken)


def test_results_reject_an_incomplete_identity_matrix() -> None:
    suite = _load("suite.v1.json")
    results = _load("results.json")
    broken = copy.deepcopy(results)
    broken["scores"]["benchmark_scores"].pop()

    with pytest.raises(ValueError, match="8 x 3 x 24"):
        validator.validate_results(suite, broken)


def test_receipts_keep_producer_and_post_run_revision_distinct() -> None:
    results = _load("results.json")
    receipts = _load("release_receipts.v1.json")
    patches = _load("post_run_patches.v1.json")

    broken_receipts = copy.deepcopy(receipts)
    broken_receipts["code_revisions"][
        "producer"
    ] = "b7e4bcf2bae88684a442834419d41d74c58e3eac"
    with pytest.raises(ValueError, match="code revisions"):
        validator.validate_release_receipts(results, broken_receipts)

    broken_patches = copy.deepcopy(patches)
    broken_patches["producer_revision"] = "b7e4bcf2bae88684a442834419d41d74c58e3eac"
    with pytest.raises(ValueError, match="producer revision"):
        validator.validate_post_run_patches(broken_patches)


def test_historical_final25_manifest_is_not_a_release_input() -> None:
    receipts = _load("release_receipts.v1.json")
    exclusion = receipts["historical_exclusions"][0]

    assert exclusion["path"] == "rlvr/experiments/final_answer_only_manifest.json"
    assert exclusion["is_release_input"] is False
    assert not (REPO_ROOT / exclusion["path"]).exists()


def test_reviewed_source_map_is_default_deny_and_runtime_closed() -> None:
    source_map = json.loads(
        (REPO_ROOT / "rlvr" / "source_map.v1.json").read_text(encoding="utf-8")
    )

    report = validator.validate_source_map(source_map)

    assert report["files"] == 128
    assert source_map["review_policy"]["default"] == "deny"
    destinations = {row["destination_path"] for row in source_map["files"]}
    assert (
        "rlvr/evaluation/vlmevalkit_extensions/batched_vlmevalkit_qwen3vl.py"
        in destinations
    )
    assert validator.EXPECTED_TRACE_VALIDATION_MAPPINGS <= destinations
    assert source_map["release_inputs"]["trace_dataset"] == (
        validator.EXPECTED_DATASET_RELEASE
    )


def test_source_map_rejects_reproduction_dataset_drift() -> None:
    source_map = json.loads(
        (REPO_ROOT / "rlvr" / "source_map.v1.json").read_text(encoding="utf-8")
    )
    source_map["release_inputs"]["trace_dataset"]["reproduction_dataset_revision"] = (
        "0" * 40
    )

    with pytest.raises(ValueError, match="dataset release input is stale"):
        validator.validate_source_map(source_map)


def test_trace_validation_review_receipt_is_required() -> None:
    source_map = json.loads(
        (REPO_ROOT / "rlvr" / "source_map.v1.json").read_text(encoding="utf-8")
    )
    source_map["trace_validation_review_receipt"]["reviewed_destination_count"] -= 1

    with pytest.raises(ValueError, match="validation review receipt is stale"):
        validator.validate_source_map(source_map)
