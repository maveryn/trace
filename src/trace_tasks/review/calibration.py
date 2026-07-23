"""Portable answer calibration against caller-managed HTTP endpoints."""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from trace_tasks import __version__ as trace_tasks_version
from trace_tasks.core.reward_scoring import score_trace_response
from trace_tasks.review.models import RECIPE_ID

from .app.index import (
    build_review_index,
    is_path_within,
    load_sample_payload,
)

DEFAULT_API_KEY_ENV = "TRACE_REVIEW_API_KEY"
MAX_DISCOVERED_REPORT_BYTES = 64 * 1024 * 1024
MAX_DISCOVERED_REPORTS = 128
MAX_DISCOVERED_TOTAL_BYTES = 128 * 1024 * 1024
CalibrationTransport = Callable[
    [str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]
]


@dataclass(frozen=True)
class CalibrationConfig:
    """Caller-controlled OpenAI-compatible chat-completions settings."""

    endpoint: str
    model: str
    api_key_env: str = DEFAULT_API_KEY_ENV
    timeout_seconds: float = 60.0
    max_tokens: int = 512
    temperature: float = 0.0
    rollouts_per_sample: int = 1
    max_retries: int = 2
    diagnostic_accuracy_threshold: float | None = None

    def validate(self) -> None:
        _chat_completions_url(self.endpoint)
        if not str(self.model).strip():
            raise ValueError("calibration model must not be empty")
        if not str(self.api_key_env).strip():
            raise ValueError("api_key_env must not be empty")
        if float(self.timeout_seconds) <= 0:
            raise ValueError("timeout_seconds must be positive")
        if int(self.max_tokens) <= 0:
            raise ValueError("max_tokens must be positive")
        if float(self.temperature) < 0:
            raise ValueError("temperature must be non-negative")
        if int(self.rollouts_per_sample) <= 0:
            raise ValueError("rollouts_per_sample must be positive")
        if int(self.max_retries) < 0:
            raise ValueError("max_retries must be non-negative")
        if self.diagnostic_accuracy_threshold is not None and not (
            0.0 <= float(self.diagnostic_accuracy_threshold) <= 1.0
        ):
            raise ValueError("diagnostic_accuracy_threshold must be in [0, 1]")


def discover_calibration_summaries(
    review_root: Path | str,
    *,
    calibration_root: Path | str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load allowlisted summaries from ignored local calibration reports.

    Raw responses, headers, endpoint URLs, and arbitrary report fields are
    deliberately excluded from the returned records.
    """

    review_path = Path(review_root).expanduser().resolve()
    workspace = (
        review_path.parent if review_path.name == "task-reviews" else review_path
    )
    root_candidate = (
        Path(calibration_root).expanduser()
        if calibration_root is not None
        else (workspace / "calibration")
    )
    if root_candidate.is_symlink():
        return [], ["calibration directory must not be a symlink"]
    root = root_candidate.resolve()
    summaries: list[dict[str, Any]] = []
    errors: list[str] = []
    if not root.is_dir():
        return summaries, errors
    paths = sorted(root.glob("*.json"))
    if len(paths) > MAX_DISCOVERED_REPORTS:
        errors.append(
            f"calibration report count exceeds discovery limit of {MAX_DISCOVERED_REPORTS}"
        )
        paths = paths[:MAX_DISCOVERED_REPORTS]
    discovered_bytes = 0
    for path in paths:
        if not path.is_file() or path.is_symlink() or not is_path_within(root, path):
            errors.append(
                f"{path.name}: calibration report is not a contained regular file"
            )
            continue
        try:
            size = path.stat().st_size
            if size > MAX_DISCOVERED_REPORT_BYTES:
                raise ValueError(
                    f"report exceeds {MAX_DISCOVERED_REPORT_BYTES} byte discovery limit"
                )
            if discovered_bytes + size > MAX_DISCOVERED_TOTAL_BYTES:
                errors.append(
                    "calibration reports exceed total discovery byte limit of "
                    f"{MAX_DISCOVERED_TOTAL_BYTES}"
                )
                break
            discovered_bytes += size
            raw = json.loads(path.read_text(encoding="utf-8"))
            summaries.append(_calibration_summary(raw, source_name=path.name))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: {_portable_error(exc)}")
    summaries.sort(key=lambda item: (item["created_at"], item["source"]))
    return summaries, errors


def calibrate_review_root(
    *,
    review_root: Path | str,
    config: CalibrationConfig,
    output_path: Path | str,
    task_ids: Sequence[str] = (),
    limit: int | None = None,
    transport: CalibrationTransport | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Score materialized answer prompts and write an ignored local report.

    Credentials are read only from ``config.api_key_env``. The report never
    includes the resolved credential or authorization headers.
    """

    config.validate()
    if limit is not None and int(limit) <= 0:
        raise ValueError("limit must be positive when supplied")
    root = Path(review_root).expanduser().resolve()
    destination = _calibration_output_path(root, output_path)
    index = build_review_index(root)
    selected_tasks = {str(task_id) for task_id in task_ids if str(task_id).strip()}
    unknown_tasks = selected_tasks - set(index.tasks)
    if unknown_tasks:
        raise ValueError(f"unknown task ids: {sorted(unknown_tasks)!r}")
    samples = [
        sample
        for sample in index.samples.values()
        if not selected_tasks or sample.task_id in selected_tasks
    ]
    samples.sort(
        key=lambda sample: (
            sample.domain,
            sample.scene_id,
            sample.task_id,
            sample.query_id,
            sample.ordinal,
        )
    )
    materialized_task_ids = {sample.task_id for sample in samples}
    missing_selected = selected_tasks - materialized_task_ids
    if missing_selected:
        raise ValueError(
            "selected tasks have no materialized samples: "
            f"{sorted(missing_selected)!r}"
        )
    if not samples:
        raise ValueError("review root has no materialized samples to calibrate")
    recipe_ids = {sample.recipe_id for sample in samples if sample.recipe_id}
    recipe_digests = {
        sample.recipe_digest for sample in samples if sample.recipe_digest
    }
    if recipe_ids != {RECIPE_ID} or len(recipe_digests) != 1:
        raise ValueError(
            "selected samples must share one canonical recipe id and digest"
        )
    if limit is not None:
        samples = samples[: int(limit)]

    source = os.environ if environ is None else environ
    api_key = str(source.get(config.api_key_env, "")).strip()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request_url = _chat_completions_url(config.endpoint)
    sender = transport or _http_transport
    results: list[dict[str, Any]] = []
    for sample in samples:
        try:
            payload = load_sample_payload(index, sample)
            prompt = _answer_only_prompt(payload)
            image_path = index.media.get(sample.media_id)
            if (
                image_path is None
                or not image_path.is_file()
                or not is_path_within(root, image_path)
            ):
                raise ValueError("sample image is not materialized")
            image_url = _image_data_url(image_path)
        except (OSError, ValueError) as exc:
            for rollout in range(int(config.rollouts_per_sample)):
                results.append(
                    _error_result(
                        sample,
                        rollout=rollout,
                        reason=f"sample_unavailable: {_portable_error(exc)}",
                    )
                )
            continue

        for rollout in range(int(config.rollouts_per_sample)):
            request_body = {
                "model": str(config.model),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                "max_tokens": int(config.max_tokens),
                "temperature": float(config.temperature),
            }
            started = time.monotonic()
            response_payload: Mapping[str, Any] | None = None
            error = ""
            attempts = 0
            for attempt in range(int(config.max_retries) + 1):
                attempts = attempt + 1
                try:
                    response_payload = sender(
                        request_url,
                        headers,
                        request_body,
                        float(config.timeout_seconds),
                    )
                    break
                except Exception as exc:  # transport is deliberately injectable.
                    error = _portable_error(exc, include_type=True)
                    if api_key:
                        error = error.replace(api_key, "<redacted>")
                    if attempt >= int(config.max_retries):
                        break
            elapsed = round(time.monotonic() - started, 6)
            if response_payload is None:
                result = _error_result(sample, rollout=rollout, reason=error)
                result.update({"attempts": attempts, "elapsed_seconds": elapsed})
                results.append(result)
                continue
            try:
                response_text = _response_text(response_payload)
                metrics = _score_answer(response_text, payload)
            except ValueError as exc:
                result = _error_result(
                    sample, rollout=rollout, reason=f"invalid_response: {exc}"
                )
                result.update({"attempts": attempts, "elapsed_seconds": elapsed})
                results.append(result)
                continue
            results.append(
                {
                    "status": "ok",
                    "sample_uid": sample.uid,
                    "domain": sample.domain,
                    "scene_id": sample.scene_id,
                    "task_id": sample.task_id,
                    "query_id": sample.query_id,
                    "ordinal": sample.ordinal,
                    "rollout": rollout,
                    "response": response_text,
                    "answer_reward": float(metrics["answer_reward"]),
                    "format": float(metrics["format"]),
                    "answer_parse_ok": bool(metrics["answer_parse_ok"]),
                    "finish_reason": _finish_reason(response_payload),
                    "response_cap_reached": _finish_reason(response_payload)
                    == "length",
                    "usage": _numeric_usage(response_payload),
                    "recipe_digest": str(payload.get("recipe_digest", "")),
                    "attempts": attempts,
                    "elapsed_seconds": elapsed,
                }
            )

    successful = [result for result in results if result["status"] == "ok"]
    mean_answer_reward = (
        sum(float(result["answer_reward"]) for result in successful) / len(successful)
        if successful
        else None
    )
    threshold = config.diagnostic_accuracy_threshold
    report = {
        "schema": "trace-review-calibration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": str(config.model),
        "endpoint": _endpoint_origin(request_url),
        "api_key_env": str(config.api_key_env),
        "provenance": {
            "recipe_ids": sorted(recipe_ids),
            "recipe_digests": sorted(recipe_digests),
            "trace_tasks_version": trace_tasks_version,
            "scorer": "trace_tasks.core.reward_scoring.score_trace_response",
            "trace_reward_mode": "answer",
            "trace_answer_scoring": "exact_json",
        },
        "settings": {
            "timeout_seconds": float(config.timeout_seconds),
            "max_tokens": int(config.max_tokens),
            "temperature": float(config.temperature),
            "rollouts_per_sample": int(config.rollouts_per_sample),
            "max_retries": int(config.max_retries),
            "diagnostic_accuracy_threshold": (
                float(threshold) if threshold is not None else None
            ),
        },
        "selection": {
            "task_ids": sorted(selected_tasks),
            "sample_count": len(samples),
            "sample_uids": [sample.uid for sample in samples],
            "limit": int(limit) if limit is not None else None,
        },
        "summary": {
            "rollout_count": len(results),
            "successful_count": len(successful),
            "error_count": len(results) - len(successful),
            "mean_answer_reward": mean_answer_reward,
            "response_cap_count": sum(
                bool(result.get("response_cap_reached")) for result in successful
            ),
            "diagnostic_threshold_met": (
                None
                if threshold is None or mean_answer_reward is None
                else bool(mean_answer_reward >= float(threshold))
            ),
        },
        "results": results,
    }
    _atomic_json_write(destination, report)
    return report


def _score_answer(response: str, payload: Mapping[str, Any]) -> dict[str, float]:
    answer_gt = payload.get("answer_gt")
    if not isinstance(answer_gt, Mapping):
        raise ValueError("sample is missing answer_gt")
    annotation_gt = payload.get("annotation_gt")
    reward_contract = payload.get("reward_contract")
    return score_trace_response(
        response=response,
        answer_gt=dict(answer_gt),
        annotation_gt=dict(annotation_gt) if isinstance(annotation_gt, Mapping) else {},
        reward_contract=(
            dict(reward_contract) if isinstance(reward_contract, Mapping) else {}
        ),
        answer_weight=1.0,
        annotation_weight=0.0,
        trace_reward_mode="answer",
        trace_answer_scoring="exact_json",
    )


def _calibration_summary(raw: Any, *, source_name: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("calibration report must be a JSON object")
    if raw.get("schema") != "trace-review-calibration-v1":
        raise ValueError("unsupported calibration report schema")
    created_at = str(raw.get("created_at", "")).strip()
    model = str(raw.get("model", "")).strip()
    if not created_at or not model:
        raise ValueError("calibration report requires created_at and model")
    summary = raw.get("summary")
    selection = raw.get("selection")
    settings = raw.get("settings")
    provenance = raw.get("provenance")
    if not isinstance(summary, Mapping) or not isinstance(selection, Mapping):
        raise ValueError("calibration report requires summary and selection objects")
    task_ids = {
        str(task_id)
        for task_id in selection.get("task_ids", [])
        if isinstance(task_id, str) and task_id.strip()
    }
    results = raw.get("results")
    if isinstance(results, list):
        task_ids.update(
            str(result.get("task_id"))
            for result in results
            if isinstance(result, Mapping)
            and isinstance(result.get("task_id"), str)
            and str(result.get("task_id")).strip()
        )
    allowed_summary = {
        key: summary.get(key)
        for key in (
            "rollout_count",
            "successful_count",
            "error_count",
            "mean_answer_reward",
            "response_cap_count",
            "diagnostic_threshold_met",
        )
        if key in summary and _safe_summary_value(summary.get(key))
    }
    allowed_settings = {}
    if isinstance(settings, Mapping):
        allowed_settings = {
            key: settings.get(key)
            for key in (
                "timeout_seconds",
                "max_tokens",
                "temperature",
                "rollouts_per_sample",
                "max_retries",
                "diagnostic_accuracy_threshold",
            )
            if key in settings and _safe_summary_value(settings.get(key))
        }
    allowed_provenance: dict[str, Any] = {}
    if isinstance(provenance, Mapping):
        for key in (
            "recipe_ids",
            "recipe_digests",
            "trace_tasks_version",
            "scorer",
            "trace_reward_mode",
            "trace_answer_scoring",
        ):
            value = provenance.get(key)
            if isinstance(value, str):
                allowed_provenance[key] = value
            elif isinstance(value, list) and all(
                isinstance(item, str) for item in value
            ):
                allowed_provenance[key] = list(value)
    return {
        "source": f"calibration/{source_name}",
        "created_at": created_at,
        "model": model,
        "task_ids": sorted(task_ids),
        "sample_count": (
            selection.get("sample_count")
            if isinstance(selection.get("sample_count"), int)
            else None
        ),
        "summary": allowed_summary,
        "settings": allowed_settings,
        "provenance": allowed_provenance,
    }


def _safe_summary_value(value: Any) -> bool:
    return value is None or (
        isinstance(value, (bool, int, float))
        and (not isinstance(value, float) or math.isfinite(value))
    )


def _answer_only_prompt(payload: Mapping[str, Any]) -> str:
    variants = payload.get("prompt_variants")
    if isinstance(variants, Mapping):
        prompt = str(variants.get("answer_only", "")).strip()
        if prompt:
            return prompt
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("sample has no prompt")
    return prompt


def _response_text(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], Mapping)
    ):
        raise ValueError("response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise ValueError("response choice has no message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, Mapping)
            and part.get("type") in {None, "text", "output_text"}
        ]
        text = "".join(text_parts)
        if text:
            return text
    raise ValueError("response message has no text content")


def _finish_reason(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], Mapping)
    ):
        return ""
    return str(choices[0].get("finish_reason", ""))


def _numeric_usage(payload: Mapping[str, Any]) -> dict[str, int | float]:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return {}
    return {
        str(key): value
        for key, value in usage.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _chat_completions_url(endpoint: str) -> str:
    supplied = str(endpoint)
    raw = supplied.strip()
    if supplied != raw:
        raise ValueError("endpoint URL must not contain surrounding whitespace")
    if not raw or any(
        ord(character) <= 32 or ord(character) == 127 for character in raw
    ):
        raise ValueError(
            "endpoint URL must not contain whitespace or control characters"
        )
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("endpoint must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("endpoint URLs must not embed credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("endpoint URLs must not contain query strings or fragments")
    hostname = parsed.hostname or ""
    try:
        normalized_host = hostname.encode("idna").decode("ascii").lower()
        port = parsed.port
    except (UnicodeError, ValueError) as exc:
        raise ValueError("endpoint URL has an invalid host or port") from exc
    if not normalized_host or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789.-:"
        for character in normalized_host
    ):
        raise ValueError("endpoint URL has an invalid host")
    display_host = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    netloc = f"{display_host}:{port}" if port is not None else display_host
    path = parsed.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path = f"{path}/chat/completions"
    return urlunsplit((parsed.scheme, netloc, path, "", ""))


def _endpoint_origin(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _http_transport(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout: float,
) -> Mapping[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = Request(url, data=body, headers=dict(headers), method="POST")
    opener = build_opener(_NoRedirectHandler())
    try:
        with opener.open(
            request, timeout=float(timeout)
        ) as response:  # noqa: S310 - caller supplies endpoint.
            decoded = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"endpoint returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError("endpoint request failed") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("endpoint returned invalid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise RuntimeError("endpoint returned a non-object JSON payload")
    return decoded


def _error_result(sample: Any, *, rollout: int, reason: str) -> dict[str, Any]:
    return {
        "status": "error",
        "sample_uid": sample.uid,
        "domain": sample.domain,
        "scene_id": sample.scene_id,
        "task_id": sample.task_id,
        "query_id": sample.query_id,
        "ordinal": sample.ordinal,
        "rollout": int(rollout),
        "error": _sanitize_error_text(reason),
    }


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=resolved.parent,
        prefix=f".{resolved.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    try:
        temporary.replace(resolved)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _calibration_output_path(review_root: Path, output_path: Path | str) -> Path:
    workspace = (
        review_root.parent if review_root.name == "task-reviews" else review_root
    )
    calibration_root = workspace / "calibration"
    if calibration_root.is_symlink():
        raise ValueError("calibration output directory must not be a symlink")
    root = calibration_root.resolve()
    destination_input = Path(output_path).expanduser()
    if destination_input.suffix.lower() != ".json":
        raise ValueError("calibration output must use a .json filename")
    if destination_input.is_symlink():
        raise ValueError("calibration output must not be a symlink")
    destination = destination_input.resolve()
    if not is_path_within(root, destination):
        raise ValueError("calibration output must stay under review/calibration")
    current = destination.parent
    while current != root:
        if current.is_symlink():
            raise ValueError("calibration output path must not contain symlinks")
        if root not in current.parents:
            raise ValueError("calibration output must stay under review/calibration")
        current = current.parent
    return destination


_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:/[A-Za-z0-9_.~:@%+\-]+)+|"
    r"(?<![A-Za-z0-9_.-])[A-Za-z]:\\[^\s]+"
)


def _sanitize_error_text(value: Any) -> str:
    text = str(value).strip().replace("\r", " ").replace("\n", " ")
    text = _ABSOLUTE_PATH_RE.sub("<path>", text)
    return text[:500] or "operation failed"


def _portable_error(exc: BaseException, *, include_type: bool = False) -> str:
    if isinstance(exc, OSError):
        detail = "I/O operation failed"
    elif isinstance(exc, json.JSONDecodeError):
        detail = f"invalid JSON at line {exc.lineno}, column {exc.colno}"
    else:
        detail = _sanitize_error_text(exc)
    if include_type:
        return f"{type(exc).__name__}: {detail}"
    return detail


__all__ = [
    "CalibrationConfig",
    "CalibrationTransport",
    "DEFAULT_API_KEY_ENV",
    "MAX_DISCOVERED_REPORT_BYTES",
    "MAX_DISCOVERED_REPORTS",
    "MAX_DISCOVERED_TOTAL_BYTES",
    "calibrate_review_root",
    "discover_calibration_summaries",
]
