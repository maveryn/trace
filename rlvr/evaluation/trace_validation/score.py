#!/usr/bin/env python3
"""Score TRACE validation generations and prepare ground-truth-blind fallbacks."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from .answer_extraction import (
    ANSWER_EXTRACTION_CONTRACT_VERSION,
    extract_answer,
)
from . import prepare_dataset as dataset_prep
from .judge_extract import (
    CONTRACT_VERSION as JUDGE_CONTRACT_VERSION,
    RECEIPT_SCHEMA_VERSION as JUDGE_RECEIPT_SCHEMA_VERSION,
    SYSTEM_PROMPT as JUDGE_SYSTEM_PROMPT,
    JudgeOutputError,
    _validate_receipt as _validate_judge_receipt,
    validate_judge_output,
)
from trace_tasks.core.reward_scoring import (
    _canonical_jsonable,
    _score_trace_answer,
    evaluate_trace_response_format,
    extract_trace_prediction,
)

SCORING_CONTRACT_VERSION = "trace-validation-answer-scoring-v2"
_ANSWER_TYPES = frozenset({"integer", "number", "option_letter", "string"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class _DuplicateKeyError(ValueError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_hash(value: Any) -> str:
    # Generation artifacts use prepare_dataset.canonical_json_bytes, whose
    # canonical representation includes one terminal newline.
    return hashlib.sha256((_canonical_json(value) + "\n").encode("utf-8")).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_json_loads(text: str, *, source: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except Exception as exc:
        raise ValueError(f"{source}: invalid strict JSON: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, _canonical_json(value) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(_canonical_json(row) + "\n" for row in rows))


def _load_json(path: Path) -> Any:
    return _strict_json_loads(path.read_text(encoding="utf-8"), source=str(path))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = _strict_json_loads(line, source=f"{path}:{line_number}")
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _load_dataset_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _load_json(path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != (
        "trace-validation-dataset-manifest-v1"
    ):
        raise ValueError(f"unsupported dataset manifest: {path}")
    # Reuse the preparer's full row/media-metadata validation while avoiding a
    # second 2,000-image content scan during scoring.
    validated = dataset_prep.load_manifest(
        path,
        expected_rows=dataset_prep.EXPECTED_ROWS,
        require_pinned=True,
        verify_media=False,
    )
    if validated != manifest:
        raise ValueError(f"dataset manifest changed while being loaded: {path}")
    rows = manifest.get("rows")
    expected = int(manifest.get("dataset", {}).get("row_count", -1))
    if not isinstance(rows, list) or len(rows) != expected or expected != 2000:
        raise ValueError(f"dataset manifest must contain exactly 2,000 rows: {path}")
    for index, row in enumerate(rows):
        answer_type = row.get("answer_type")
        if answer_type not in _ANSWER_TYPES:
            raise ValueError(f"dataset row {index} has unsupported answer_type")
        answer_gt = row.get("answer_gt")
        if not isinstance(answer_gt, dict) or answer_gt.get("type") != answer_type:
            raise ValueError(f"dataset row {index} has inconsistent answer_gt")
        # The same GT-blind type coercer used for predictions validates the
        # manifest's typed reference without changing its value.
        encoded = _canonical_json({"answer": answer_gt.get("value")})
        validation = extract_answer(encoded, answer_type=answer_type)
        if validation.status != "found":
            raise ValueError(f"dataset row {index} has invalid typed answer_gt")
    return manifest, rows


def _load_suite(
    path: Path,
    *,
    dataset_identity: Mapping[str, Any],
    dataset_manifest_sha256: str,
    dataset_equivalence_receipt: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    suite_sha256 = _sha256_file(path)
    if suite_sha256 != dataset_prep.HISTORICAL_SUITE_SHA256:
        raise ValueError(
            "TRACE validation suite SHA-256 mismatch: "
            f"{suite_sha256} != {dataset_prep.HISTORICAL_SUITE_SHA256}"
        )
    suite = _load_json(path)
    if not isinstance(suite, dict) or suite.get("schema_version") != (
        "trace-validation-suite-v1"
    ):
        raise ValueError(f"unsupported TRACE validation suite: {path}")
    models = suite.get("models")
    if not isinstance(models, list) or len(models) != 8:
        raise ValueError("TRACE validation suite must pin exactly eight models")
    by_slug: dict[str, dict[str, Any]] = {}
    for position, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"suite model {position} is not an object")
        slug = model.get("slug")
        if not isinstance(slug, str) or not slug or slug in by_slug:
            raise ValueError(f"suite model {position} has invalid or duplicate slug")
        if not isinstance(model.get("repo_id"), str) or not model["repo_id"]:
            raise ValueError(f"suite model {slug} has no repo_id")
        revision = model.get("revision")
        if (
            not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        ):
            raise ValueError(f"suite model {slug} has no immutable source revision")
        runtime_revision = model.get("runtime_view_revision", revision)
        if (
            not isinstance(runtime_revision, str)
            or re.fullmatch(
                r"(?:[0-9a-f]{40}|sha256set:[0-9a-f]{64})", runtime_revision
            )
            is None
        ):
            raise ValueError(f"suite model {slug} has invalid runtime revision")
        by_slug[slug] = model

    suite_dataset = suite.get("dataset")
    if not isinstance(suite_dataset, dict):
        raise ValueError("suite has no dataset identity")
    equivalence = dataset_prep.validate_dataset_equivalence_bridge(
        dataset_equivalence_receipt,
        historical_identity=dataset_prep.normalized_suite_dataset_identity(
            suite_dataset,
            file_size_bytes=dataset_prep.HISTORICAL_DATASET_FILE_SIZE_BYTES,
        ),
        reproduction_identity=dataset_prep.normalized_manifest_dataset_identity(
            dataset_identity
        ),
        historical_manifest_sha256=(dataset_prep.HISTORICAL_DATASET_MANIFEST_SHA256),
        reproduction_manifest_sha256=dataset_manifest_sha256,
    )
    scoring = suite.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError("suite has no scoring contract")
    expected_scoring = {
        "deterministic_extraction": ANSWER_EXTRACTION_CONTRACT_VERSION,
        "judge_only_on": ["missing", "ambiguous"],
        "unresolved_rows_score": 0,
        "drop_failed_rows": False,
    }
    for key, expected_value in expected_scoring.items():
        if scoring.get(key) != expected_value:
            raise ValueError(f"suite scoring contract mismatch for {key}")
    judge = suite.get("judge")
    if not isinstance(judge, dict):
        raise ValueError("suite has no judge contract")
    if not isinstance(judge.get("repo_id"), str) or not judge["repo_id"]:
        raise ValueError("suite judge has no repo_id")
    if re.fullmatch(r"[0-9a-f]{40}", str(judge.get("revision", ""))) is None:
        raise ValueError("suite judge has no immutable revision")
    if (
        not isinstance(judge.get("served_model_name"), str)
        or not judge["served_model_name"]
    ):
        raise ValueError("suite judge has no served_model_name")
    return suite, by_slug, equivalence


def _raw_response(row: dict[str, Any]) -> str:
    for key in ("raw_response", "response", "text", "prediction"):
        if key in row:
            value = row[key]
            if not isinstance(value, str) or not value:
                raise ValueError(f"generation row has invalid {key}")
            return value
    raise ValueError("generation row has no raw_response field")


def _validate_generation_metadata(
    path: Path,
    *,
    metadata: dict[str, Any],
    model: dict[str, Any],
    dataset_identity: Mapping[str, Any],
    dataset_manifest_sha256: str,
    suite: Mapping[str, Any],
    file_rows: int,
) -> None:
    if metadata.get("schema_version") != "trace-validation-generation-run-v1":
        raise ValueError(f"{path}: unsupported generation run metadata")
    if metadata.get("status") != "complete" or metadata.get("error_rows") != 0:
        raise ValueError(f"{path}: generation run is not complete")
    if metadata.get("model_slug") != model["slug"]:
        raise ValueError(f"{path}: run metadata model_slug mismatch")
    expected_revision = model.get("runtime_view_revision", model["revision"])
    if metadata.get("model_revision") != expected_revision:
        raise ValueError(f"{path}: run metadata model revision mismatch")
    if metadata.get("manifest_sha256") != dataset_manifest_sha256:
        raise ValueError(f"{path}: run metadata manifest hash mismatch")
    identity_fields = {
        "dataset_repo_id": "repo_id",
        "dataset_revision": "revision",
        "dataset_file": "file",
        "dataset_file_sha256": "file_sha256",
        "dataset_rows": "row_count",
    }
    for metadata_key, dataset_key in identity_fields.items():
        if metadata.get(metadata_key) != dataset_identity.get(dataset_key):
            raise ValueError(f"{path}: run metadata {metadata_key} mismatch")
    if metadata.get("responses_sha256") != _sha256_file(path):
        raise ValueError(f"{path}: responses SHA-256 mismatch")
    if metadata.get("selected_rows") != file_rows:
        raise ValueError(f"{path}: selected row count mismatch")
    run_contract = metadata.get("run_contract")
    run_identity = metadata.get("run_identity_sha256")
    if not isinstance(run_contract, dict) or not isinstance(run_identity, str):
        raise ValueError(f"{path}: missing generation run contract")
    if _canonical_hash(run_contract) != run_identity:
        raise ValueError(f"{path}: generation run identity hash mismatch")
    expected_contract = {
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_repo_id": dataset_identity["repo_id"],
        "dataset_revision": dataset_identity["revision"],
        "dataset_file_sha256": dataset_identity["file_sha256"],
        "model_slug": model["slug"],
        "model_revision": expected_revision,
    }
    for key, expected_value in expected_contract.items():
        if run_contract.get(key) != expected_value:
            raise ValueError(f"{path}: generation run contract {key} mismatch")

    prompt_contract = suite.get("prompt")
    generation_contract = suite.get("generation")
    if not isinstance(prompt_contract, dict) or not isinstance(
        generation_contract, dict
    ):
        raise ValueError("suite prompt/generation contract is invalid")
    if metadata.get("system_prompt_file_sha256") != prompt_contract.get(
        "system_prompt_sha256"
    ):
        raise ValueError(f"{path}: system prompt provenance mismatch")
    expected_decoding = {
        "seed": generation_contract.get("seed"),
        "temperature": generation_contract.get("temperature"),
        "top_p": generation_contract.get("top_p"),
        "top_k": generation_contract.get("top_k"),
        "presence_penalty": generation_contract.get("presence_penalty"),
        "frequency_penalty": generation_contract.get("frequency_penalty"),
        "repetition_penalty": generation_contract.get("repetition_penalty"),
        "max_tokens": generation_contract.get("max_tokens"),
        "n": generation_contract.get("responses_per_prompt"),
        "stream": False,
    }
    if metadata.get("decoding") != expected_decoding:
        raise ValueError(f"{path}: generation decoding contract mismatch")
    expected_mm = {
        "min_pixels": generation_contract.get("min_image_pixels"),
        "max_pixels": generation_contract.get("max_image_pixels"),
    }
    if metadata.get("mm_processor_kwargs") != expected_mm:
        raise ValueError(f"{path}: multimodal processor contract mismatch")


def _validate_generation_row(
    row: dict[str, Any],
    *,
    path: Path,
    dataset_row: dict[str, Any],
    model_slug: str,
    run_identity_sha256: str,
) -> str:
    row_index = dataset_row["row_index"]
    expected = {
        "schema_version": "trace-validation-generation-receipt-v1",
        "status": "complete",
        "run_identity_sha256": run_identity_sha256,
        "model_slug": model_slug,
        "row_index": row_index,
        "instance_id": dataset_row["instance_id"],
        "task": dataset_row["task"],
        "domain": dataset_row["domain"],
        "answer_type": dataset_row["answer_type"],
    }
    for key, expected_value in expected.items():
        if row.get(key) != expected_value:
            raise ValueError(f"{path}: generation row {row_index} {key} mismatch")
    response = _raw_response(row)
    if (
        row.get("raw_response_sha256")
        != hashlib.sha256(response.encode("utf-8")).hexdigest()
    ):
        raise ValueError(
            f"{path}: generation row {row_index} raw response hash mismatch"
        )
    for key in ("request_hash", "response_sha256"):
        value = row.get(key)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"{path}: generation row {row_index} has invalid {key}")
    api_response = row.get("api_response")
    if (
        not isinstance(api_response, dict)
        or _canonical_hash(api_response) != row["response_sha256"]
    ):
        raise ValueError(
            f"{path}: generation row {row_index} API response hash mismatch"
        )
    request = row.get("request")
    expected_request = {
        "prompt_sha256": hashlib.sha256(
            dataset_row["prompt_answer"].encode("utf-8")
        ).hexdigest(),
        "ordered_image_sha256": [image["sha256"] for image in dataset_row["images"]],
    }
    if request != expected_request:
        raise ValueError(
            f"{path}: generation row {row_index} request provenance mismatch"
        )
    finish_reason = row.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise ValueError(f"{path}: generation row {row_index} invalid finish_reason")
    return response


def _load_generations(
    paths: list[Path],
    dataset_rows: list[dict[str, Any]],
    *,
    suite: dict[str, Any],
    suite_models: dict[str, dict[str, Any]],
    dataset_identity: Mapping[str, Any],
    dataset_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    combined: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    keys: set[tuple[str, int]] = set()
    model_indices: dict[str, set[int]] = collections.defaultdict(set)
    for path in paths:
        file_rows = _load_jsonl(path)
        if not file_rows:
            raise ValueError(f"{path}: generation file is empty")
        file_slugs = {str(row.get("model_slug", "")) for row in file_rows}
        if len(file_slugs) != 1:
            raise ValueError(f"{path}: generation file must contain exactly one model")
        file_slug = next(iter(file_slugs))
        model = suite_models.get(file_slug)
        if model is None:
            raise ValueError(f"{path}: model {file_slug!r} is not in the frozen suite")
        metadata_path = path.parent / "run_metadata.json"
        metadata = _load_json(metadata_path)
        if not isinstance(metadata, dict):
            raise ValueError(f"{metadata_path}: expected object")
        _validate_generation_metadata(
            path,
            metadata=metadata,
            model=model,
            dataset_identity=dataset_identity,
            dataset_manifest_sha256=dataset_manifest_sha256,
            suite=suite,
            file_rows=len(file_rows),
        )
        run_identity = str(metadata["run_identity_sha256"])
        provenance.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "run_metadata": str(metadata_path.resolve()),
                "run_metadata_sha256": _sha256_file(metadata_path),
                "run_identity_sha256": run_identity,
                "model_slug": file_slug,
                "model_revision": metadata["model_revision"],
                "shard_index": metadata.get("shard_index"),
                "shard_count": metadata.get("shard_count"),
                "rows": len(file_rows),
            }
        )
        for row in file_rows:
            model_slug = str(row.get("model_slug", ""))
            try:
                row_index = int(row["row_index"])
            except Exception as exc:
                raise ValueError(
                    f"{path}: generation row has invalid row_index"
                ) from exc
            if not 0 <= row_index < len(dataset_rows):
                raise ValueError(f"{path}: row_index out of range: {row_index}")
            response = _validate_generation_row(
                row,
                path=path,
                dataset_row=dataset_rows[row_index],
                model_slug=model_slug,
                run_identity_sha256=run_identity,
            )
            key = (model_slug, row_index)
            if key in keys:
                raise ValueError(f"duplicate generation key: {key}")
            keys.add(key)
            model_indices[model_slug].add(row_index)
            row = dict(row)
            row["raw_response"] = response
            row["generation_file"] = str(path.resolve())
            combined.append(row)
    expected_indices = set(range(len(dataset_rows)))
    for model_slug, indices in model_indices.items():
        if indices != expected_indices:
            missing = sorted(expected_indices - indices)
            extra = sorted(indices - expected_indices)
            raise ValueError(
                f"model {model_slug} does not have the exact 2,000 rows: "
                f"missing={missing[:10]} extra={extra[:10]}"
            )
    if not model_indices:
        raise ValueError("no generation rows found")
    if set(model_indices) != set(suite_models):
        raise ValueError(
            "generation model coverage mismatch: "
            f"missing={sorted(set(suite_models) - set(model_indices))} "
            f"extra={sorted(set(model_indices) - set(suite_models))}"
        )
    return (
        sorted(
            combined, key=lambda row: (str(row["model_slug"]), int(row["row_index"]))
        ),
        sorted(
            provenance,
            key=lambda row: (str(row["model_slug"]), int(row["shard_index"] or 0)),
        ),
    )


def _historical_scores(response: str, answer_gt: dict[str, Any]) -> dict[str, Any]:
    answer_value, _, structured_found = extract_trace_prediction(response)
    candidate = response if answer_value is None else answer_value
    accuracy, _ = _score_trace_answer(
        candidate, answer_gt, trace_answer_scoring="exact_json"
    )
    format_details = evaluate_trace_response_format(
        response, trace_reward_mode="answer"
    )
    format_score = float(format_details["format"])
    strict_rfc_json = False
    answer_block = format_details.get("answer_block")
    if isinstance(answer_block, str):
        try:
            payload = json.loads(answer_block)
            strict_rfc_json = isinstance(payload, dict) and set(payload) == {"answer"}
        except Exception:
            strict_rfc_json = False
    return {
        "historical_answer_correct": int(accuracy == 1.0),
        "historical_format_correct": int(format_score == 1.0),
        "historical_reward": float(0.95 * accuracy + 0.05 * format_score),
        "structured_answer_found": bool(structured_found),
        "terminal_rfc_json": strict_rfc_json,
    }


def _semantic_correct(candidate: Any, answer_gt: dict[str, Any]) -> int:
    return int(
        _canonical_jsonable(candidate) == _canonical_jsonable(answer_gt["value"])
    )


def _load_judge_results(path: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if path is None:
        return {}
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _load_jsonl(path):
        model_slug = row.get("model_slug")
        row_index = row.get("row_index")
        if not isinstance(model_slug, str) or not model_slug:
            raise ValueError("judge result has invalid model_slug")
        if (
            isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or row_index < 0
        ):
            raise ValueError("judge result has invalid row_index")
        key = (model_slug, row_index)
        if key in result:
            raise ValueError(f"duplicate judge result: {key}")
        result[key] = row
    return result


def _validate_judge_result(
    result: dict[str, Any],
    *,
    generation: dict[str, Any],
    dataset_row: dict[str, Any],
    deterministic_status: str,
    judge_contract: Mapping[str, Any],
) -> None:
    response = generation["raw_response"]
    raw_sha256 = hashlib.sha256(response.encode("utf-8")).hexdigest()
    expected = {
        "schema_version": JUDGE_RECEIPT_SCHEMA_VERSION,
        "contract_version": JUDGE_CONTRACT_VERSION,
        "model_slug": generation["model_slug"],
        "row_index": dataset_row["row_index"],
        "instance_id": dataset_row["instance_id"],
        "answer_type": dataset_row["answer_type"],
        "raw_response_sha256": raw_sha256,
        "deterministic_status": deterministic_status,
        "deterministic_extraction_version": ANSWER_EXTRACTION_CONTRACT_VERSION,
        "judge_model": judge_contract["served_model_name"],
        "judge_revision": judge_contract["revision"],
        "tokenizer_model": judge_contract["repo_id"],
        "system_prompt_sha256": hashlib.sha256(
            JUDGE_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    }
    for key, expected_value in expected.items():
        if result.get(key) != expected_value:
            raise ValueError(
                f"judge result {(generation['model_slug'], dataset_row['row_index'])} "
                f"{key} mismatch"
            )
    for key in (
        "request_hash",
        "rendered_prompt_sha256",
        "tokenizer_template_sha256",
    ):
        value = result.get(key)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"judge result has invalid {key}")
    request_contract = result.get("request_contract")
    if not isinstance(request_contract, dict):
        raise ValueError("judge result has no request contract")
    if hashlib.sha256(
        _canonical_json(request_contract).encode("utf-8")
    ).hexdigest() != (result["request_hash"]):
        raise ValueError("judge result request contract hash mismatch")
    expected_request_contract = {
        "contract_version": JUDGE_CONTRACT_VERSION,
        "model_slug": generation["model_slug"],
        "row_index": dataset_row["row_index"],
        "instance_id": dataset_row["instance_id"],
        "raw_response_sha256": raw_sha256,
        "answer_type": dataset_row["answer_type"],
        "deterministic_status": deterministic_status,
        "deterministic_extraction_version": ANSWER_EXTRACTION_CONTRACT_VERSION,
        "judge_model": judge_contract["served_model_name"],
        "judge_revision": judge_contract["revision"],
        "tokenizer_model": judge_contract["repo_id"],
        "system_prompt_sha256": expected["system_prompt_sha256"],
        "rendered_prompt_sha256": result["rendered_prompt_sha256"],
        "tokenizer_template_sha256": result["tokenizer_template_sha256"],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "retry_token_limits": judge_contract["max_token_retries"],
    }
    if request_contract != expected_request_contract:
        raise ValueError("judge result request contract fields mismatch")
    attempts = result.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("judge result has no attempt provenance")
    pending_identity = {
        "request_hash": result["request_hash"],
        "request_contract": request_contract,
        "raw_response_sha256": raw_sha256,
        "rendered_prompt_sha256": result["rendered_prompt_sha256"],
    }
    pending_row = {
        "model_slug": generation["model_slug"],
        "row_index": dataset_row["row_index"],
        "instance_id": dataset_row["instance_id"],
        "answer_type": dataset_row["answer_type"],
        "raw_response": response,
        "deterministic_status": deterministic_status,
        "deterministic_extraction_version": ANSWER_EXTRACTION_CONTRACT_VERSION,
    }
    try:
        _validate_judge_receipt(
            result,
            row=pending_row,
            identity=pending_identity,
            api_model=judge_contract["served_model_name"],
            judge_revision=judge_contract["revision"],
            tokenizer_model=judge_contract["repo_id"],
            tokenizer_template_sha256=result["tokenizer_template_sha256"],
            allow_failed=True,
        )
    except (JudgeOutputError, ValueError, TypeError) as exc:
        raise ValueError("judge receipt provenance validation failed") from exc
    status = result.get("judge_status")
    if status == "failed":
        if result.get("answer") is not None or result.get("evidence") != "":
            raise ValueError(
                "failed judge result must have null answer and empty evidence"
            )
        return
    if status not in {"ok", "missing", "ambiguous"}:
        raise ValueError("judge result has invalid status")
    try:
        validated = validate_judge_output(
            _canonical_json(
                {
                    "status": status,
                    "answer": result.get("answer"),
                    "evidence": result.get("evidence"),
                }
            ),
            raw_response=response,
            answer_type=dataset_row["answer_type"],
        )
    except (JudgeOutputError, ValueError, TypeError) as exc:
        raise ValueError("judge result answer/evidence validation failed") from exc
    if validated != {
        "status": status,
        "answer": result.get("answer"),
        "evidence": result.get("evidence"),
    }:
        raise ValueError("judge result answer is not canonical")


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(sum(float(row[field]) for row in rows) / len(rows)) if rows else 0.0


def _aggregate(
    rows: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(
        groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        record = {field: value for field, value in zip(group_fields, key)}
        route_counts = collections.Counter(
            str(row["final_extraction_route"]) for row in members
        )
        judge_requested_rows = sum(int(row["judge_requested"]) for row in members)
        judge_resolved_rows = sum(int(row["judge_resolved"]) for row in members)
        unresolved_rows = sum(int(row["unresolved"]) for row in members)
        record.update(
            {
                "rows": len(members),
                "accuracy_denominator_rows": len(members),
                "historical_answer_accuracy": _mean(
                    members, "historical_answer_correct"
                ),
                "historical_format_rate": _mean(members, "historical_format_correct"),
                "terminal_rfc_json_rate": _mean(members, "terminal_rfc_json"),
                "historical_mean_reward": _mean(members, "historical_reward"),
                "deterministic_semantic_accuracy": _mean(
                    members, "deterministic_semantic_correct"
                ),
                "combined_semantic_accuracy": _mean(
                    members, "combined_semantic_correct"
                ),
                "deterministic_found_rate": _mean(members, "deterministic_found"),
                "judge_fallback_rate": _mean(members, "judge_requested"),
                "judge_requested_rows": judge_requested_rows,
                "judge_resolved_rows": judge_resolved_rows,
                "judge_resolved_overall_rate": _mean(members, "judge_resolved"),
                "judge_resolution_rate": (
                    judge_resolved_rows / judge_requested_rows
                    if judge_requested_rows
                    else 0.0
                ),
                "unresolved_rows": unresolved_rows,
                "unresolved_rate": _mean(members, "unresolved"),
                "route_counts": dict(sorted(route_counts.items())),
            }
        )
        output.append(record)
    return output


def _summary_markdown(overall: list[dict[str, Any]], labels: dict[str, str]) -> str:
    lines = [
        "# TRACE validation results",
        "",
        "The split contains 2,000 unseen validation instances from the same 1,000 task programs used for training.",
        "",
        "| Model | Strict answer | Terminal format | Deterministic semantic | Combined semantic | Judge fallback | Unresolved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        label = labels.get(str(row["model_slug"]), str(row["model_slug"]))
        lines.append(
            "| {label} | {ha:.2%} | {fmt:.2%} | {det:.2%} | {comb:.2%} | {judge:.2%} | {unresolved:.2%} |".format(
                label=label,
                ha=row["historical_answer_accuracy"],
                fmt=row["historical_format_rate"],
                det=row["deterministic_semantic_accuracy"],
                comb=row["combined_semantic_accuracy"],
                judge=row["judge_fallback_rate"],
                unresolved=row["unresolved_rate"],
            )
        )
    lines.extend(
        [
            "",
            "Strict answer accuracy uses the training reward scorer. Combined semantic accuracy applies the same deterministic and Qwen3 extraction policy to every model; unresolved rows remain incorrect and in the denominator.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    dataset_manifest, dataset_rows = _load_dataset_manifest(args.dataset_manifest)
    dataset_manifest_sha256 = _sha256_file(args.dataset_manifest)
    suite, suite_models, dataset_equivalence = _load_suite(
        args.suite,
        dataset_identity=dataset_manifest["dataset"],
        dataset_manifest_sha256=dataset_manifest_sha256,
        dataset_equivalence_receipt=args.dataset_equivalence_receipt,
    )
    generations, generation_provenance = _load_generations(
        args.generation_jsonl,
        dataset_rows,
        suite=suite,
        suite_models=suite_models,
        dataset_identity=dataset_manifest["dataset"],
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    labels = {
        str(model["slug"]): str(model.get("label", model["slug"]))
        for model in suite["models"]
    }
    judge_results = _load_judge_results(args.judge_results)
    judge_contract = suite["judge"]

    scored: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    expected_judge_keys: set[tuple[str, int]] = set()
    for generation in generations:
        row_index = int(generation["row_index"])
        dataset_row = dataset_rows[row_index]
        model_slug = str(generation["model_slug"])
        response = str(generation["raw_response"])
        answer_gt = dataset_row["answer_gt"]
        answer_type = str(dataset_row["answer_type"])
        historical = _historical_scores(response, answer_gt)
        extraction = extract_answer(response, answer_type=answer_type)
        deterministic_found = extraction.status == "found"
        deterministic_correct = (
            _semantic_correct(extraction.typed_candidate, answer_gt)
            if deterministic_found
            else 0
        )
        key = (model_slug, row_index)
        judge_requested = not deterministic_found
        judge_result = judge_results.get(key)
        if judge_requested:
            expected_judge_keys.add(key)
            pending.append(
                {
                    "model_slug": model_slug,
                    "row_index": row_index,
                    "instance_id": dataset_row["instance_id"],
                    "answer_type": answer_type,
                    "raw_response": response,
                    "raw_response_sha256": hashlib.sha256(
                        response.encode("utf-8")
                    ).hexdigest(),
                    "deterministic_status": extraction.status,
                    "deterministic_extraction_version": (
                        ANSWER_EXTRACTION_CONTRACT_VERSION
                    ),
                }
            )

        if judge_result is not None and judge_requested:
            _validate_judge_result(
                judge_result,
                generation=generation,
                dataset_row=dataset_row,
                deterministic_status=extraction.status,
                judge_contract=judge_contract,
            )

        judge_resolved = bool(
            judge_requested
            and judge_result is not None
            and judge_result.get("judge_status") == "ok"
        )
        combined_candidate = (
            extraction.typed_candidate
            if deterministic_found
            else judge_result.get("answer") if judge_resolved else None
        )
        combined_correct = (
            _semantic_correct(combined_candidate, answer_gt)
            if deterministic_found or judge_resolved
            else 0
        )
        unresolved = int(judge_requested and not judge_resolved)
        if deterministic_found:
            final_route = f"deterministic:{extraction.route}"
        elif judge_resolved:
            final_route = "judge"
        elif judge_result is not None:
            final_route = f"judge:{judge_result.get('judge_status', 'failed')}"
        else:
            final_route = f"pending:{extraction.status}"

        scored.append(
            {
                "schema_version": "trace-validation-scored-row-v1",
                "scoring_contract_version": SCORING_CONTRACT_VERSION,
                "deterministic_extraction_version": ANSWER_EXTRACTION_CONTRACT_VERSION,
                "model_slug": model_slug,
                "row_index": row_index,
                "instance_id": dataset_row["instance_id"],
                "task": dataset_row["task"],
                "domain": dataset_row["domain"],
                "answer_type": answer_type,
                "answer_gt": answer_gt,
                "raw_response_sha256": hashlib.sha256(
                    response.encode("utf-8")
                ).hexdigest(),
                **historical,
                "deterministic_extraction": extraction.as_dict(),
                "deterministic_found": int(deterministic_found),
                "deterministic_semantic_correct": int(deterministic_correct),
                "judge_requested": int(judge_requested),
                "judge_status": (
                    judge_result.get("judge_status") if judge_result else None
                ),
                "judge_answer": judge_result.get("answer") if judge_result else None,
                "judge_evidence": (
                    judge_result.get("evidence", "") if judge_result else ""
                ),
                "judge_resolved": int(judge_resolved),
                "combined_semantic_correct": int(combined_correct),
                "unresolved": unresolved,
                "final_extraction_route": final_route,
                "generation_finish_reason": generation.get("finish_reason"),
                "generation_request_hash": generation.get("request_hash"),
            }
        )

    if args.judge_results is not None:
        actual_keys = set(judge_results)
        if actual_keys != expected_judge_keys:
            missing = sorted(expected_judge_keys - actual_keys)
            extra = sorted(actual_keys - expected_judge_keys)
            raise ValueError(
                f"judge coverage mismatch: missing={missing[:10]} extra={extra[:10]}"
            )

    overall = _aggregate(scored, ("model_slug",))
    by_domain = _aggregate(scored, ("model_slug", "domain"))
    by_answer_type = _aggregate(scored, ("model_slug", "answer_type"))
    by_task = _aggregate(scored, ("model_slug", "task"))
    expected_total_rows = len(suite_models) * len(dataset_rows)
    if len(scored) != expected_total_rows or len(overall) != len(suite_models):
        raise RuntimeError(
            "finalized coverage mismatch: "
            f"rows={len(scored)}/{expected_total_rows} "
            f"models={len(overall)}/{len(suite_models)}"
        )
    provenance = {
        "schema_version": "trace-validation-score-provenance-v1",
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "dataset_manifest": str(args.dataset_manifest.resolve()),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_identity": dataset_manifest["dataset"],
        "dataset_equivalence": dataset_equivalence,
        "suite": str(args.suite.resolve()),
        "suite_sha256": _sha256_file(args.suite),
        "generation_files": generation_provenance,
        "judge_results": (
            {
                "path": str(args.judge_results.resolve()),
                "sha256": _sha256_file(args.judge_results),
            }
            if args.judge_results is not None
            else None
        ),
        "code_sha256": {
            "score": _sha256_file(Path(__file__)),
            "answer_extraction": _sha256_file(
                Path(__file__).with_name("answer_extraction.py")
            ),
            "judge_extraction": _sha256_file(
                Path(__file__).with_name("judge_extract.py")
            ),
            "reward_scoring": _sha256_file(
                Path(__file__).parents[3] / "src/trace_tasks/core/reward_scoring.py"
            ),
            "dataset_preparation": _sha256_file(
                Path(__file__).with_name("prepare_dataset.py")
            ),
        },
        "deterministic_extraction_contract_version": (
            ANSWER_EXTRACTION_CONTRACT_VERSION
        ),
        "judge_extraction_contract_version": JUDGE_CONTRACT_VERSION,
        "judge_model": {
            "repo_id": judge_contract["repo_id"],
            "revision": judge_contract["revision"],
            "served_model_name": judge_contract["served_model_name"],
        },
        "model_slugs": sorted(suite_models),
        "models": len(overall),
        "rows_per_model": 2000,
        "total_rows": len(scored),
        "accuracy_denominator_rows": len(scored),
        "judge_requested_rows": len(expected_judge_keys),
        "judge_pending_rows": len(expected_judge_keys),
        "judge_result_rows": len(judge_results),
        "judge_resolved_rows": sum(int(row["judge_resolved"]) for row in scored),
        "unresolved_rows": sum(int(row["unresolved"]) for row in scored),
        "denominator_policy": {
            "accuracy": "all_generation_rows",
            "judge_fallback_rate": "all_generation_rows",
            "judge_resolution_rate": "judge_requested_rows",
            "unresolved_rows_score": 0,
            "drop_failed_rows": False,
        },
        "judge_finalized": args.judge_results is not None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "scored_rows.jsonl", scored)
    _write_jsonl(args.output_dir / "judge_pending.jsonl", pending)
    _write_json(
        args.output_dir / "summary.json",
        {
            "provenance": provenance,
            "overall": overall,
            "by_domain": by_domain,
            "by_answer_type": by_answer_type,
        },
    )
    _write_jsonl(args.output_dir / "by_task.jsonl", by_task)
    _atomic_text(args.output_dir / "summary.md", _summary_markdown(overall, labels))
    print(
        _canonical_json(
            {
                "models": len(overall),
                "rows": len(scored),
                "judge_pending": len(expected_judge_keys),
                "judge_finalized": args.judge_results is not None,
                "output_dir": str(args.output_dir.resolve()),
            }
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset-equivalence-receipt",
        type=Path,
        default=dataset_prep.DATASET_EQUIVALENCE_RECEIPT,
    )
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--generation-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--judge-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
