from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rlvr.evaluation.trace_validation import verify

ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT / "rlvr" / "evaluation" / "trace_validation"


def test_canonical_release_files_remain_byte_frozen() -> None:
    expected = {
        "suite.v1.json": (
            "f9cccdcdddb6135c16d3a9d434f985b51e4105c07ff0c74a54a71a4dfe7c85c7"
        ),
        "results.v1.json": (
            "1e2b78d46dbbd606210653b584ed77dc436d16bbdc0d329c5aaf55718dc37e11"
        ),
        "release_receipt.v1.json": (
            "865c7901f70578b21f3ce8c0c74621b30fc9bfe64f957ba4bd38c8e830aa971f"
        ),
    }

    assert {
        name: hashlib.sha256((RELEASE_ROOT / name).read_bytes()).hexdigest()
        for name in expected
    } == expected


def test_release_metadata_matches_immutable_hf_receipt() -> None:
    report = verify.verify_release_metadata()

    assert report == {
        "phase": "release",
        "status": "ok",
        "source_revision": "cf0d14aed86db2661d397ce8b68b36171873478d",
        "models": 8,
        "rows_per_model": 2000,
        "results_sha256": "1e2b78d46dbbd606210653b584ed77dc436d16bbdc0d329c5aaf55718dc37e11",
    }


def test_release_scores_are_the_frozen_eight_model_values() -> None:
    document = json.loads(
        (RELEASE_ROOT / "results.v1.json").read_text(encoding="utf-8")
    )

    assert {row["model_id"]: row["score"] for row in document["results"]} == {
        "qwen2.5-vl-3b-base": 24.45,
        "trace-qwen2.5-vl-3b": 41.05,
        "qwen2.5-vl-7b-base": 34.25,
        "trace-qwen2.5-vl-7b": 51.55,
        "game-rl-qwen2.5-vl-7b": 35.55,
        "sphinx-qwen2.5-vl-7b": 33.5,
        "pcgrpo-qwen2.5-vl-7b": 34.1,
        "vero-qwen2.5-vl-7b": 38.4,
    }


def test_release_validator_rejects_result_drift(tmp_path: Path) -> None:
    results = json.loads((RELEASE_ROOT / "results.v1.json").read_text(encoding="utf-8"))
    results["results"][0]["score"] = 99.0
    changed_results = tmp_path / "results.v1.json"
    changed_results.write_text(json.dumps(results), encoding="utf-8")

    with pytest.raises(verify.VerificationError, match="results hash mismatch"):
        verify.verify_release_metadata(results_path=changed_results)


def test_release_validator_rejects_receipt_drift(tmp_path: Path) -> None:
    receipt = json.loads(
        (RELEASE_ROOT / "release_receipt.v1.json").read_text(encoding="utf-8")
    )
    receipt["campaign_verification"]["unresolved_rows"] += 1
    changed_receipt = tmp_path / "release_receipt.v1.json"
    changed_receipt.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(verify.VerificationError, match="receipt SHA-256 mismatch"):
        verify.verify_release_metadata(receipt_path=changed_receipt)
