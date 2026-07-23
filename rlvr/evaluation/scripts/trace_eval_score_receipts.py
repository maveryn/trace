#!/usr/bin/env python3
"""Validate fail-closed trace_eval_v1 score campaign receipts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark_queue_lib import score_path, spec_by_key
from trace_eval_code_provenance import trace_eval_code_manifest
from trace_eval_evaluator_provenance import sha256_file
from trace_eval_suite import TraceEvalSuite


CONTRACT_VERSION = "trace-eval-score-campaign-v1"
ERROR_FIELDS = ("error", "errors", "failure", "failures", "failed", "incomplete")
SUCCESS_STATUSES = frozenset({"complete", "completed", "ok", "ready", "success", "succeeded"})


class ScoreReceiptError(RuntimeError):
    """Raised when a score campaign receipt cannot prove exact completion."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ScoreReceiptError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScoreReceiptError(f"invalid {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ScoreReceiptError(f"{label} must be a JSON object: {path}")
    return value


def _nonempty_failure(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _reject_incomplete_state(value: Mapping[str, Any], *, label: str) -> None:
    for field in ERROR_FIELDS:
        if field in value and _nonempty_failure(value[field]):
            raise ScoreReceiptError(f"{label}.{field} records an incomplete/error state")
    if "status" in value:
        status = str(value.get("status") or "").strip().lower()
        if status not in SUCCESS_STATUSES:
            raise ScoreReceiptError(f"{label}.status is not complete: {value.get('status')!r}")


def _require_sha(value: Any, *, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ScoreReceiptError(f"{label} must be a lowercase SHA-256")
    return text


def _require_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ScoreReceiptError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ScoreReceiptError(f"{label} must be an integer") from error
    if isinstance(value, float) and not value.is_integer():
        raise ScoreReceiptError(f"{label} must be an integer")
    return result


def _require_exact_keys(value: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ScoreReceiptError(f"{label} keys mismatch: expected={sorted(expected)} actual={actual}")
    return value


def _validate_scoring_implementation(
    contract: Mapping[str, Any], repo_root: Path, evaluator_sha256: str
) -> None:
    implementation = contract.get("scoring_implementation")
    if not isinstance(implementation, dict) or not implementation:
        raise ScoreReceiptError("contract.scoring_implementation must be nonempty")
    current_manifest = trace_eval_code_manifest(
        repo_root=repo_root,
        evaluator_sha256=evaluator_sha256,
    )
    recorded_manifest = contract.get("trace_eval_code_provenance")
    expected_record = {
        "schema_version": current_manifest["schema_version"],
        "sha256": current_manifest["sha256"],
        "evaluator_provenance_sha256": current_manifest["evaluator_provenance_sha256"],
    }
    if recorded_manifest != expected_record:
        raise ScoreReceiptError(
            "contract trace_eval code provenance does not match the active scoring surface"
        )
    if implementation != current_manifest["files"]:
        raise ScoreReceiptError("contract.scoring_implementation file map is incomplete or stale")


def _validate_generation_inputs(
    contract: Mapping[str, Any],
    *,
    model_slugs: Sequence[str],
    suite: TraceEvalSuite,
) -> None:
    generation_inputs = _require_exact_keys(
        contract.get("generation_inputs"),
        set(model_slugs),
        label="contract.generation_inputs",
    )
    benchmark_keys = set(suite.benchmark_keys)
    for model_slug in model_slugs:
        entries = _require_exact_keys(
            generation_inputs[model_slug],
            benchmark_keys,
            label=f"contract.generation_inputs[{model_slug!r}]",
        )
        for benchmark in suite.benchmarks:
            entry = entries[benchmark.key]
            if not isinstance(entry, dict):
                raise ScoreReceiptError(
                    f"generation input must be an object: {model_slug}/{benchmark.key}"
                )
            _reject_incomplete_state(entry, label=f"generation_inputs.{model_slug}.{benchmark.key}")
            if _require_int(
                entry.get("rows"),
                label=f"generation_inputs.{model_slug}.{benchmark.key}.rows",
            ) != benchmark.rows:
                raise ScoreReceiptError(
                    f"generation input row mismatch: {model_slug}/{benchmark.key}"
                )
            summary_path = Path(str(entry.get("generation_summary") or "")).expanduser().resolve()
            expected_sha = _require_sha(
                entry.get("generation_summary_sha256"),
                label=f"generation_inputs.{model_slug}.{benchmark.key}.generation_summary_sha256",
            )
            if not summary_path.is_file() or sha256_file(summary_path) != expected_sha:
                raise ScoreReceiptError(
                    f"generation summary hash mismatch: {model_slug}/{benchmark.key}: {summary_path}"
                )
            snapshot = _require_sha(
                entry.get("dataset_snapshot_sha256"),
                label=f"generation_inputs.{model_slug}.{benchmark.key}.dataset_snapshot_sha256",
            )
            selection_snapshot = str(
                ((contract.get("dataset_manifest") or {}).get("view_snapshot_sha256")) or ""
            )
            if snapshot != selection_snapshot:
                raise ScoreReceiptError(
                    f"generation dataset snapshot mismatch: {model_slug}/{benchmark.key}"
                )


def validate_score_campaign_receipts(
    *,
    score_root: Path,
    seed: int,
    model_slugs: Sequence[str],
    suite: TraceEvalSuite,
    evaluator_sha256: str,
    repo_root: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Validate one seed receipt and return its exact verified slice map."""

    score_root = score_root.expanduser().resolve()
    repo_root = repo_root.expanduser().resolve()
    requested_models = tuple(dict.fromkeys(model_slugs))
    if not requested_models or len(requested_models) != len(model_slugs):
        raise ScoreReceiptError("requested model slugs must be nonempty and unique")
    evaluator_hash = _require_sha(evaluator_sha256, label="current evaluator provenance")

    manifest_path = score_root / f"score_campaign_manifest_seed_{seed}.json"
    completion_path = score_root / f"score_campaign_completion_seed_{seed}.json"
    manifest = _load_object(manifest_path, label="score campaign manifest")
    completion = _load_object(completion_path, label="score campaign completion")
    _reject_incomplete_state(manifest, label="manifest")
    _reject_incomplete_state(completion, label="completion")

    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise ScoreReceiptError("manifest.contract must be an object")
    _reject_incomplete_state(contract, label="contract")
    contract_sha = _require_sha(manifest.get("contract_sha256"), label="manifest.contract_sha256")
    recomputed_sha = _sha256_json(contract)
    if contract_sha != recomputed_sha:
        raise ScoreReceiptError(
            f"manifest contract SHA mismatch: {contract_sha} != {recomputed_sha}"
        )
    if completion.get("contract_sha256") != contract_sha:
        raise ScoreReceiptError("completion contract SHA does not match the manifest")

    if contract.get("contract_version") != CONTRACT_VERSION:
        raise ScoreReceiptError("score campaign contract version mismatch")
    if contract.get("suite") != suite.suite_id or contract.get("dataset_view") != suite.dataset_manifest_view:
        raise ScoreReceiptError("score campaign suite/dataset view mismatch")
    if _require_int(contract.get("seed"), label="contract.seed") != seed:
        raise ScoreReceiptError("score campaign seed mismatch")

    selection = contract.get("selection")
    if not isinstance(selection, dict):
        raise ScoreReceiptError("contract.selection must be an object")
    if selection.get("sha256") != suite.manifest_sha256:
        raise ScoreReceiptError("score campaign suite selection SHA mismatch")
    if selection.get("benchmarks") != list(suite.benchmark_keys):
        raise ScoreReceiptError("score campaign suite selection/order mismatch")

    expected_routes = {
        "official_vlmevalkit": list(suite.routes["official_vlmevalkit"]),
        "direct": list(suite.routes["direct_score"]),
        "mme_reasoning": list(suite.routes["dedicated_score"]),
    }
    if contract.get("routes") != expected_routes:
        raise ScoreReceiptError("score campaign route partition/order mismatch")

    campaigns = contract.get("campaigns")
    if not isinstance(campaigns, list) or len(campaigns) != len(requested_models):
        raise ScoreReceiptError("score campaign model list length mismatch")
    campaign_by_slug: dict[str, dict[str, Any]] = {}
    for campaign in campaigns:
        if not isinstance(campaign, dict):
            raise ScoreReceiptError("contract.campaigns entries must be objects")
        slug = str(campaign.get("model_slug") or "")
        if not slug or slug in campaign_by_slug or not str(campaign.get("model") or ""):
            raise ScoreReceiptError("contract.campaigns has an invalid or duplicate model")
        campaign_by_slug[slug] = campaign
    if set(campaign_by_slug) != set(requested_models):
        raise ScoreReceiptError("score campaign models do not exactly match the requested models")

    provenance = contract.get("evaluator_provenance")
    if not isinstance(provenance, dict):
        raise ScoreReceiptError("contract.evaluator_provenance must be an object")
    if provenance.get("sha256") != evaluator_hash:
        raise ScoreReceiptError("score campaign evaluator provenance does not match the current worktree")
    _validate_scoring_implementation(contract, repo_root, evaluator_hash)
    _validate_generation_inputs(contract, model_slugs=requested_models, suite=suite)

    expected_identities = {
        (model_slug, benchmark_key)
        for model_slug in requested_models
        for benchmark_key in suite.benchmark_keys
    }
    expected_count = len(expected_identities)
    if _require_int(
        completion.get("expected_slices"), label="completion.expected_slices"
    ) != expected_count:
        raise ScoreReceiptError("completion expected_slices mismatch")
    if _require_int(
        completion.get("completed_slices"), label="completion.completed_slices"
    ) != expected_count:
        raise ScoreReceiptError("completion completed_slices mismatch")
    slices = completion.get("slices")
    if not isinstance(slices, list) or len(slices) != expected_count:
        raise ScoreReceiptError("completion slices length mismatch")

    verified: dict[tuple[str, str], dict[str, Any]] = {}
    for position, receipt in enumerate(slices):
        if not isinstance(receipt, dict):
            raise ScoreReceiptError(f"completion.slices[{position}] must be an object")
        _reject_incomplete_state(receipt, label=f"completion.slices[{position}]")
        identity = (
            str(receipt.get("model_slug") or ""),
            str(receipt.get("benchmark_key") or ""),
        )
        if identity not in expected_identities:
            raise ScoreReceiptError(f"unexpected completion slice identity: {identity}")
        if identity in verified:
            raise ScoreReceiptError(f"duplicate completion slice identity: {identity}")
        model_slug, benchmark_key = identity
        campaign = campaign_by_slug[model_slug]
        if receipt.get("model") != campaign.get("model"):
            raise ScoreReceiptError(f"completion slice model mismatch: {identity}")
        expected_rows = suite.rows_by_benchmark[benchmark_key]
        if _require_int(
            receipt.get("rows"), label=f"completion.slices[{position}].rows"
        ) != expected_rows:
            raise ScoreReceiptError(f"completion slice row mismatch: {identity}")
        try:
            receipt_score = float(receipt["score"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ScoreReceiptError(f"completion slice score is invalid: {identity}") from error
        if not math.isfinite(receipt_score):
            raise ScoreReceiptError(f"completion slice score is non-finite: {identity}")

        expected_path = score_path(
            spec_by_key(benchmark_key),
            model_slug,
            score_root / f"seed_{seed}" / "benchmark",
        ).resolve()
        recorded_path = Path(str(receipt.get("scores_path") or "")).expanduser().resolve()
        if recorded_path != expected_path:
            raise ScoreReceiptError(
                f"completion slice score path mismatch: {identity}: {recorded_path} != {expected_path}"
            )
        score_sha = _require_sha(
            receipt.get("scores_sha256"),
            label=f"completion.slices[{position}].scores_sha256",
        )
        score_payload = _load_object(expected_path, label=f"score artifact {model_slug}/{benchmark_key}")
        _reject_incomplete_state(score_payload, label=f"scores.{model_slug}.{benchmark_key}")
        for field, expected in (
            ("benchmark_key", benchmark_key),
            ("model_slug", model_slug),
            ("model", campaign.get("model")),
        ):
            if score_payload.get(field) is not None and score_payload.get(field) != expected:
                raise ScoreReceiptError(f"score artifact {field} mismatch: {identity}")
        actual_sha = sha256_file(expected_path)
        if actual_sha != score_sha:
            raise ScoreReceiptError(f"score artifact SHA mismatch: {identity}")
        metric_key = "accuracy" if benchmark_key == "mme_reasoning" else "score"
        try:
            actual_score_float = float(score_payload[metric_key])
            actual_rows = _require_int(
                score_payload.get("rows"), label=f"scores.{model_slug}.{benchmark_key}.rows"
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ScoreReceiptError(f"score artifact metric is invalid: {identity}") from error
        if actual_rows != expected_rows:
            raise ScoreReceiptError(f"score artifact row mismatch: {identity}")
        if not math.isfinite(actual_score_float) or actual_score_float != receipt_score:
            raise ScoreReceiptError(
                f"score artifact value mismatch: {identity}: {actual_score_float} != {receipt_score}"
            )
        verified[identity] = {
            "score": receipt_score,
            "rows": expected_rows,
            "path": str(expected_path),
            "sha256": actual_sha,
        }

    if set(verified) != expected_identities:
        missing = sorted(expected_identities - set(verified))
        raise ScoreReceiptError(f"completion is missing exact slice identities: {missing}")
    return verified
