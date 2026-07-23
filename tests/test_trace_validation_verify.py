from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

from rlvr.evaluation.trace_validation import generate
from rlvr.evaluation.trace_validation import prepare_dataset as prepare
from rlvr.evaluation.trace_validation import score
from rlvr.evaluation.trace_validation import verify
from rlvr.evaluation.scripts import prepare_trace_eval_models as model_prep


def test_verify_direct_script_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "rlvr/evaluation/trace_validation/verify.py"),
            "--help",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (
        "Verify the pinned TRACE validation generation and scoring campaign."
        in completed.stdout
    )


class FakeResponse:
    status_code = 200

    def __init__(self, payload: dict):
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0

    def post(self, _url, *, headers, json, timeout):
        del headers, timeout
        text = "".join(
            item.get("text", "")
            for item in json["messages"][1]["content"]
            if item.get("type") == "text"
        )
        match = re.search(r"question (\d+)", text)
        assert match
        row_index = int(match.group(1))
        if row_index == 1:
            content = "After considering the image, I settle on value 1."
        else:
            content = json_module.dumps({"answer": row_index})
        with self._lock:
            call_index = self.calls
            self.calls += 1
        return FakeResponse(
            {
                "id": f"response-{json['model']}-{call_index}",
                "model": json["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            }
        )


json_module = json


@dataclass
class Campaign:
    suite_path: Path
    manifest_path: Path
    parquet_path: Path
    equivalence_receipt: Path
    equivalence_provenance: dict
    generation_root: Path
    score_dir: Path
    judge_results: Path
    policy: verify.VerificationPolicy


def _png_bytes(index: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (9 + index, 7), (index * 40, 10, 20)).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_equivalence_receipt(
    path: Path,
    *,
    dataset_file: str,
    rows: int,
    historical_revision: str,
    historical_sha256: str,
    historical_size: int,
    compared_revision: str,
    reproduction_sha256: str,
    reproduction_size: int,
) -> None:
    columns = list(prepare._EQUIVALENCE_COMPARED_COLUMNS)
    truth = {
        "current_differs_only_by_added_trace_supervision_mode": True,
        "existing_columns_value_equal": True,
        "existing_field_schemas_equal": True,
        "image_bytes_equal": True,
        "instance_ids_equal_and_unique": True,
        "row_counts_equal": True,
        "row_order_equal": True,
        "schema_metadata_only_adds_trace_supervision_mode": True,
        "task_distributions_equal": True,
        "train_validation_instance_ids_disjoint": True,
    }
    receipt = {
        "added_advisory_column": "trace_supervision_mode",
        "aggregate": {
            **truth,
            "outcome": "pass",
            "parquet_file_count": 1,
            "row_count": rows,
        },
        "compared_columns": columns,
        "compared_current_inventory": {
            "files": [
                {
                    "path": dataset_file,
                    "lfs_sha256": reproduction_sha256,
                    "size": reproduction_size,
                }
            ],
            "revision": compared_revision,
        },
        "compared_current_revision": compared_revision,
        "comparison_program": {"name": "fixture", "sha256": "3" * 64},
        "comparison_runtime": {"python": "fixture"},
        "current_schema": [
            {"name": name} for name in [*columns, "trace_supervision_mode"]
        ],
        "original_schema": [{"name": name} for name in columns],
        "original_training_revision": historical_revision,
        "parquet_files": [
            {
                "column_equality": {name: True for name in columns},
                "compared_columns_equal_in_row_order": True,
                "current": {
                    "bytes": reproduction_size,
                    "lfs_oid_matches_sha256": True,
                    "lfs_sha256": reproduction_sha256,
                    "sha256": reproduction_sha256,
                },
                "existing_field_schemas_equal": True,
                "image_bytes_equal_in_row_order": True,
                "original": {
                    "bytes": historical_size,
                    "lfs_oid_matches_sha256": True,
                    "lfs_sha256": historical_sha256,
                    "sha256": historical_sha256,
                },
                "path": dataset_file,
                "rows": rows,
                "schema_metadata_only_adds_trace_supervision_mode": True,
                "split": "validation",
                "task_distribution_equal": True,
                "trace_supervision_mode_counts": {"answer": rows},
            }
        ],
        "receipt_version": "trace-dataset-equivalence-v1",
        "repository": "fixture/trace",
        "splits": {
            "validation": {
                "current_task_distribution_sha256": "4" * 64,
                "instance_ids_equal": True,
                "original_task_distribution_sha256": "4" * 64,
                "rows": rows,
                "rows_per_task": 1,
                "task_count": rows,
                "task_distribution_equal": True,
                "trace_supervision_mode_counts": {"answer": rows},
                "unique_instance_ids": rows,
            }
        },
    }
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")


def _build_campaign(tmp_path: Path) -> Campaign:
    dataset_root = tmp_path / "prepared"
    parquet_path = tmp_path / "fixture.parquet"
    parquet_path.write_bytes(b"pinned-fixture-parquet")
    parquet_sha = prepare.sha256_file(parquet_path)
    historical_revision = "f" * 40
    historical_parquet_sha = "1" * 64
    historical_parquet_size = 456
    historical_manifest_sha = "2" * 64
    reproduction_revision = "a" * 40
    compared_revision = "e" * 40
    dataset_file = "data/fixture.parquet"
    rows = []
    for index in range(3):
        payload = _png_bytes(index)
        digest = prepare.sha256_bytes(payload)
        relative_path = f"media/sha256/{digest[:2]}/{digest}.png"
        image_path = dataset_root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(payload)
        rows.append(
            {
                "row_index": index,
                "instance_id": f"instance-{index}",
                "task": f"task_fixture__scene__objective_{index}",
                "domain": "fixture",
                "answer_type": "integer",
                "prompt_answer": f"<image>fixture question {index}",
                "answer_gt": {"type": "integer", "value": index},
                "images": [
                    {
                        "image_index": 0,
                        "sha256": digest,
                        "size_bytes": len(payload),
                        "width": 9 + index,
                        "height": 7,
                        "format": "PNG",
                        "mime_type": "image/png",
                        "relative_path": relative_path,
                    }
                ],
            }
        )
    manifest = {
        "schema_version": prepare.MANIFEST_SCHEMA,
        "dataset": {
            "repo_id": "fixture/trace",
            "revision": reproduction_revision,
            "config": "default",
            "split": "validation",
            "file": dataset_file,
            "file_sha256": parquet_sha,
            "file_size_bytes": parquet_path.stat().st_size,
            "row_count": 3,
        },
        "media": {
            "storage": prepare.MEDIA_STORAGE,
            "paths_relative_to": prepare.MANIFEST_NAME,
            "reencoded": False,
            "resized": False,
        },
        "rows": rows,
    }
    manifest_path = dataset_root / "manifest.json"
    prepare.atomic_write_json(manifest_path, manifest)

    model_pins = (
        verify.ModelPin("fixture-model-a", "fixture/model-a", "b" * 40),
        verify.ModelPin("fixture-model-b", "fixture/model-b", "c" * 40),
    )
    suite = {
        "schema_version": "trace-validation-suite-v1",
        "suite_id": "fixture-iid-suite",
        "dataset": {
            "repo_id": "fixture/trace",
            "revision": historical_revision,
            "split": "validation",
            "file": dataset_file,
            "sha256": historical_parquet_sha,
            "rows": 3,
            "tasks": 3,
            "samples_per_task": 1,
            "distribution": "same_task_programs_nonoverlapping_samples",
            "prompt_key": "prompt_answer",
            "image_key": "images",
        },
        "prompt": {
            "system_prompt_file": "rlvr/examples/prompts/trace_vero_json_system_prompt_answer.txt",
            "system_prompt_sha256": verify.EXPECTED_SYSTEM_PROMPT_FILE_SHA256,
            "chat_template": "native_checkpoint_template",
            "add_generation_prompt": True,
        },
        "generation": {
            "seed": 42,
            "responses_per_prompt": 1,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": -1,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "max_tokens": 2048,
            "min_image_pixels": 262144,
            "max_image_pixels": 4194304,
            "server_generation_config": "vllm",
        },
        "scoring": {
            "strict_contract": "trace-answer-exact-json-v1",
            "deterministic_extraction": verify.ANSWER_EXTRACTION_VERSION,
            "semantic_match": "trace-answer-exact-match-v0",
            "judge_only_on": ["missing", "ambiguous"],
            "unresolved_rows_score": 0,
            "drop_failed_rows": False,
        },
        "judge": {
            "repo_id": "Qwen/Qwen3-32B",
            "revision": "d" * 40,
            "served_model_name": "fixture-judge",
            "temperature": 0.0,
            "top_p": 1.0,
            "thinking": False,
            "max_token_retries": [128, 256, 512],
            "input_fields": ["raw_response", "answer_type"],
            "forbidden_fields": ["image", "question", "choices", "answer_gt_value"],
        },
        "models": [
            {
                "slug": pin.slug,
                "label": pin.slug.upper(),
                "repo_id": pin.repo_id,
                "revision": pin.revision,
            }
            for pin in model_pins
        ],
    }
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    equivalence_receipt = tmp_path / "dataset_equivalence.v1.json"
    _write_equivalence_receipt(
        equivalence_receipt,
        dataset_file=dataset_file,
        rows=3,
        historical_revision=historical_revision,
        historical_sha256=historical_parquet_sha,
        historical_size=historical_parquet_size,
        compared_revision=compared_revision,
        reproduction_sha256=parquet_sha,
        reproduction_size=parquet_path.stat().st_size,
    )
    policy = verify.VerificationPolicy(
        suite_sha256=prepare.sha256_file(suite_path),
        suite_id="fixture-iid-suite",
        historical_dataset=verify.DatasetPin(
            repo_id="fixture/trace",
            revision=historical_revision,
            file=dataset_file,
            file_sha256=historical_parquet_sha,
            file_size_bytes=historical_parquet_size,
            manifest_sha256=historical_manifest_sha,
        ),
        reproduction_dataset=verify.DatasetPin(
            repo_id="fixture/trace",
            revision=reproduction_revision,
            file=dataset_file,
            file_sha256=parquet_sha,
            file_size_bytes=parquet_path.stat().st_size,
            manifest_sha256=prepare.sha256_file(manifest_path),
        ),
        equivalence_compared_revision=compared_revision,
        equivalence_receipt_sha256=prepare.sha256_file(equivalence_receipt),
        rows=3,
        tasks=3,
        samples_per_task=1,
        models=model_pins,
    )
    equivalence_provenance = prepare.validate_dataset_equivalence_bridge(
        equivalence_receipt,
        historical_identity=policy.historical_dataset.normalized_identity(rows=3),
        reproduction_identity=policy.reproduction_dataset.normalized_identity(rows=3),
        historical_manifest_sha256=historical_manifest_sha,
        reproduction_manifest_sha256=policy.reproduction_dataset.manifest_sha256,
        expected_compared_revision=compared_revision,
        expected_receipt_sha256=policy.equivalence_receipt_sha256,
    )

    generation_root = tmp_path / "generation"
    session = FakeSession()
    for pin in model_pins:
        model_path = tmp_path / "models" / pin.slug
        model_path.mkdir(parents=True)
        (model_path / "config.json").write_text("{}\n", encoding="utf-8")
        (model_path / "model.safetensors").write_bytes(
            f"fixture weights for {pin.slug}".encode("utf-8")
        )
        model_hashes = model_prep._relative_hashes(
            model_path, model_prep._validation_snapshot_files(model_path)
        )
        (model_path / ".trace_model_revision.json").write_text(
            json.dumps(
                {
                    "schema_version": "trace-model-revision-v1",
                    "slug": pin.slug,
                    "source": pin.repo_id,
                    "source_revision": pin.revision,
                    "resolved_commit": pin.revision,
                    "immutable_revision": pin.revision,
                    "inference_revision": pin.revision,
                    "content_revision": model_prep._content_revision(model_hashes),
                    "model_origin": "public_download",
                    "file_count": len(model_hashes),
                    "file_sha256": model_hashes,
                }
            ),
            encoding="utf-8",
        )
        with mock.patch.object(generate, "_http_session", return_value=session):
            metadata = generate.run_generation(
                manifest_path=manifest_path,
                output_dir=generation_root / pin.slug,
                endpoint_url="http://127.0.0.1:9000/v1",
                served_model=pin.slug,
                model_slug=pin.slug,
                model_path=model_path,
                model_revision=pin.revision,
                media_transport="data-url",
                concurrency=2,
                retry_backoff_seconds=0,
                progress_every=0,
                expected_rows=3,
                require_pinned_manifest=False,
            )
        assert metadata["status"] == "complete"
    return Campaign(
        suite_path=suite_path,
        manifest_path=manifest_path,
        parquet_path=parquet_path,
        equivalence_receipt=equivalence_receipt,
        equivalence_provenance=equivalence_provenance,
        generation_root=generation_root,
        score_dir=tmp_path / "scoring" / "final",
        judge_results=tmp_path / "judge" / "judge_results.jsonl",
        policy=policy,
    )


@pytest.fixture
def campaign(tmp_path: Path) -> Campaign:
    return _build_campaign(tmp_path)


def _verify_generation(campaign: Campaign):
    return verify.verify_generation_phase(
        suite_path=campaign.suite_path,
        dataset_manifest=campaign.manifest_path,
        dataset_parquet=campaign.parquet_path,
        generation_root=campaign.generation_root,
        dataset_equivalence_receipt=campaign.equivalence_receipt,
        policy=campaign.policy,
    )


def test_generation_only_verifies_all_receipts_and_hashes(campaign: Campaign):
    suite, dataset, generation = _verify_generation(campaign)

    assert suite["suite_id"] == "fixture-iid-suite"
    assert suite["dataset"]["revision"] == "f" * 40
    assert dataset.manifest["dataset"]["revision"] == "a" * 40
    assert dataset.equivalence == campaign.equivalence_provenance
    assert len(dataset.rows) == 3
    assert len(generation.records) == 6
    assert set(generation.response_paths) == {"fixture-model-a", "fixture-model-b"}


def test_generation_only_rejects_equivalence_receipt_drift(campaign: Campaign):
    receipt = json.loads(campaign.equivalence_receipt.read_text(encoding="utf-8"))
    receipt["aggregate"]["row_order_equal"] = False
    campaign.equivalence_receipt.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(
        verify.VerificationError,
        match="dataset-equivalence receipt SHA-256 mismatch",
    ):
        _verify_generation(campaign)


def test_generation_only_rejects_decoding_drift(campaign: Campaign):
    metadata_path = campaign.generation_root / "fixture-model-a" / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["decoding"]["top_k"] = 0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(verify.VerificationError, match="decoding contract mismatch"):
        _verify_generation(campaign)


def test_generation_only_rejects_response_content_drift(campaign: Campaign):
    model_dir = campaign.generation_root / "fixture-model-a"
    responses_path = model_dir / "responses.jsonl"
    rows = [json.loads(line) for line in responses_path.read_text().splitlines()]
    rows[0]["raw_response"] = "tampered"
    _write_jsonl(responses_path, rows)
    metadata_path = model_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["responses_sha256"] = prepare.sha256_file(responses_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(verify.VerificationError, match="raw response hash mismatch"):
        _verify_generation(campaign)


def _score_with_fixture_loader(
    campaign: Campaign,
    *,
    output_dir: Path,
    judge_results: Path | None,
) -> None:
    manifest = json.loads(campaign.manifest_path.read_text())
    generation_paths = [
        campaign.generation_root / pin.slug / "responses.jsonl"
        for pin in campaign.policy.models
    ]
    args = argparse.Namespace(
        dataset_manifest=campaign.manifest_path,
        dataset_equivalence_receipt=campaign.equivalence_receipt,
        suite=campaign.suite_path,
        generation_jsonl=generation_paths,
        judge_results=judge_results,
        output_dir=output_dir,
    )
    suite_document = json.loads(campaign.suite_path.read_text())
    suite_models = {model["slug"]: model for model in suite_document["models"]}
    with (
        mock.patch.object(
            score,
            "_load_dataset_manifest",
            return_value=(manifest, manifest["rows"]),
        ),
        mock.patch.object(
            score,
            "_load_suite",
            return_value=(
                suite_document,
                suite_models,
                campaign.equivalence_provenance,
            ),
        ),
    ):
        score.run(args)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["provenance"]["rows_per_model"] = campaign.policy.rows
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _prepare_full_artifacts(campaign: Campaign) -> None:
    preliminary = campaign.score_dir.parent / "pre_judge"
    _score_with_fixture_loader(campaign, output_dir=preliminary, judge_results=None)
    pending = [
        json.loads(line)
        for line in (preliminary / "judge_pending.jsonl").read_text().splitlines()
    ]
    assert len(pending) == 2
    judge_rows = []
    for pending_row in pending:
        rendered_prompt_sha256 = hashlib.sha256(
            f"prompt:{pending_row['model_slug']}:{pending_row['row_index']}".encode()
        ).hexdigest()
        tokenizer_template_sha256 = hashlib.sha256(b"template").hexdigest()
        request_contract = {
            "contract_version": verify.JUDGE_CONTRACT_VERSION,
            "model_slug": pending_row["model_slug"],
            "row_index": pending_row["row_index"],
            "instance_id": pending_row["instance_id"],
            "raw_response_sha256": pending_row["raw_response_sha256"],
            "answer_type": pending_row["answer_type"],
            "deterministic_status": pending_row["deterministic_status"],
            "deterministic_extraction_version": verify.ANSWER_EXTRACTION_VERSION,
            "judge_model": "fixture-judge",
            "judge_revision": "d" * 40,
            "tokenizer_model": "Qwen/Qwen3-32B",
            "system_prompt_sha256": verify.EXPECTED_JUDGE_SYSTEM_PROMPT_SHA256,
            "rendered_prompt_sha256": rendered_prompt_sha256,
            "tokenizer_template_sha256": tokenizer_template_sha256,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "retry_token_limits": [128, 256, 512],
        }
        request_hash = hashlib.sha256(
            json.dumps(
                request_contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        raw_output = '{"status":"ok","answer":1,"evidence":"value 1"}'
        result = {
            "schema_version": "trace-validation-judge-receipt-v1",
            "contract_version": verify.JUDGE_CONTRACT_VERSION,
            "request_hash": request_hash,
            "request_contract": request_contract,
            "raw_response_sha256": pending_row["raw_response_sha256"],
            "rendered_prompt_sha256": rendered_prompt_sha256,
            "system_prompt_sha256": verify.EXPECTED_JUDGE_SYSTEM_PROMPT_SHA256,
            "tokenizer_template_sha256": tokenizer_template_sha256,
            "tokenizer_model": "Qwen/Qwen3-32B",
            "judge_revision": "d" * 40,
            "judge_model": "fixture-judge",
            "model_slug": pending_row["model_slug"],
            "row_index": pending_row["row_index"],
            "instance_id": pending_row["instance_id"],
            "answer_type": pending_row["answer_type"],
            "deterministic_status": pending_row["deterministic_status"],
            "deterministic_extraction_version": verify.ANSWER_EXTRACTION_VERSION,
            "judge_status": "ok",
            "answer": 1,
            "evidence": "value 1",
            "attempts": [
                {
                    "retry_number": 0,
                    "max_tokens": 128,
                    "endpoint": "http://127.0.0.1:9100/v1",
                    "raw_output": raw_output,
                    "finish_reason": "stop",
                    "elapsed_seconds": 0.1,
                    "validation_error": None,
                }
            ],
        }
        receipt_path = (
            campaign.judge_results.parent / "receipts" / f"{request_hash}.json"
        )
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(result), encoding="utf-8")
        judge_rows.append(result)
    judge_rows.sort(key=lambda row: (row["model_slug"], row["row_index"]))
    _write_jsonl(campaign.judge_results, judge_rows)
    _score_with_fixture_loader(
        campaign,
        output_dir=campaign.score_dir,
        judge_results=campaign.judge_results,
    )


def _verify_full(campaign: Campaign):
    return verify.verify_full_phase(
        suite_path=campaign.suite_path,
        dataset_manifest=campaign.manifest_path,
        dataset_parquet=campaign.parquet_path,
        generation_root=campaign.generation_root,
        score_dir=campaign.score_dir,
        judge_results=campaign.judge_results,
        dataset_equivalence_receipt=campaign.equivalence_receipt,
        policy=campaign.policy,
    )


def test_full_phase_verifies_scoring_judge_and_summary(campaign: Campaign):
    _prepare_full_artifacts(campaign)

    report = _verify_full(campaign)

    assert report["status"] == "ok"
    assert report["generation_rows"] == 6
    assert report["scored_rows"] == 6
    assert report["judge_pending_rows"] == 2
    assert report["judge_result_rows"] == 2
    assert report["unresolved_rows"] == 0


def test_full_phase_rejects_dataset_equivalence_provenance_drift(
    campaign: Campaign,
):
    _prepare_full_artifacts(campaign)
    summary_path = campaign.score_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["provenance"]["dataset_equivalence"]["receipt_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(
        verify.VerificationError,
        match="dataset-equivalence provenance mismatch",
    ):
        _verify_full(campaign)


def test_full_phase_rejects_missing_judge_result(campaign: Campaign):
    _prepare_full_artifacts(campaign)
    rows = campaign.judge_results.read_text().splitlines()
    campaign.judge_results.write_text(rows[0] + "\n", encoding="utf-8")

    with pytest.raises(verify.VerificationError, match="one-to-one with pending"):
        _verify_full(campaign)


def test_full_phase_recomputes_historical_scores(campaign: Campaign):
    _prepare_full_artifacts(campaign)
    scored_path = campaign.score_dir / "scored_rows.jsonl"
    rows = [json.loads(line) for line in scored_path.read_text().splitlines()]
    rows[0]["historical_answer_correct"] = 1 - rows[0]["historical_answer_correct"]
    rows[0]["historical_reward"] = (
        0.95 * rows[0]["historical_answer_correct"]
        + 0.05 * rows[0]["historical_format_correct"]
    )
    _write_jsonl(scored_path, rows)

    with pytest.raises(
        verify.VerificationError, match="historical_answer_correct mismatch"
    ):
        _verify_full(campaign)


def test_full_phase_rejects_deterministic_score_on_fallback(campaign: Campaign):
    _prepare_full_artifacts(campaign)
    scored_path = campaign.score_dir / "scored_rows.jsonl"
    rows = [json.loads(line) for line in scored_path.read_text().splitlines()]
    fallback = next(row for row in rows if row["deterministic_found"] == 0)
    fallback["deterministic_semantic_correct"] = 1
    _write_jsonl(scored_path, rows)

    with pytest.raises(
        verify.VerificationError,
        match="fallback row has a deterministic score",
    ):
        _verify_full(campaign)


def test_full_phase_rejects_stale_failed_fenced_attempt(campaign: Campaign):
    _prepare_full_artifacts(campaign)
    judge_rows = [
        json.loads(line) for line in campaign.judge_results.read_text().splitlines()
    ]
    stale = judge_rows[0]
    stale["judge_status"] = "failed"
    stale["answer"] = None
    stale["evidence"] = ""
    fenced_output = (
        "```json\n" '{"status":"missing","answer":null,"evidence":""}' "\n```"
    )
    stale["attempts"] = [
        {
            "retry_number": retry_number,
            "max_tokens": max_tokens,
            "endpoint": "http://127.0.0.1:9100/v1",
            "raw_output": fenced_output,
            "finish_reason": "stop",
            "elapsed_seconds": 0.1,
            "validation_error": "JudgeOutputError: judge output: invalid JSON",
        }
        for retry_number, max_tokens in enumerate((128, 256, 512))
    ]
    _write_jsonl(campaign.judge_results, judge_rows)
    receipt_path = (
        campaign.judge_results.parent / "receipts" / f"{stale['request_hash']}.json"
    )
    receipt_path.write_text(json.dumps(stale), encoding="utf-8")

    with pytest.raises(
        verify.VerificationError, match="valid attempt is marked as failed"
    ):
        _verify_full(campaign)


def test_full_phase_rejects_inconsistent_summary(campaign: Campaign):
    _prepare_full_artifacts(campaign)
    summary_path = campaign.score_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["overall"][0]["combined_semantic_accuracy"] = 0.123
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(
        verify.VerificationError, match="overall summary is inconsistent"
    ):
        _verify_full(campaign)
