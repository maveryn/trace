#!/usr/bin/env python3
"""Generate TRACE validation responses through one OpenAI-compatible endpoint.

Each invocation addresses exactly one vLLM endpoint/model.  Successful rows
are durably persisted before a canonical ``responses.jsonl`` is assembled, so
an interrupted run can resume without repeating completed requests.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import ipaddress
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import requests

try:
    from . import prepare_dataset as dataset_prep
except ImportError:  # Direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import prepare_dataset as dataset_prep  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[3]
SYSTEM_PROMPT_PATH = (
    REPO_ROOT
    / "rlvr"
    / "examples"
    / "prompts"
    / "trace_vero_json_system_prompt_answer.txt"
)
SYSTEM_PROMPT_FILE_SHA256 = (
    "f394927d9abcfb7a1e43ef48a30c29b8c70e6facdbda314d7b27c59d8c3ae900"
)

RECEIPT_SCHEMA = "trace-validation-generation-receipt-v1"
RUN_METADATA_SCHEMA = "trace-validation-generation-run-v1"

DECODING_CONTRACT: dict[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": -1,
    "max_tokens": 2_048,
    "seed": 42,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "n": 1,
    "stream": False,
}
MM_PROCESSOR_CONTRACT: dict[str, int] = {
    "min_pixels": 262_144,
    "max_pixels": 4_194_304,
}

_IMMUTABLE_REVISION_RE = re.compile(
    r"(?:[0-9a-f]{40}|[0-9a-f]{64}|sha256set:[0-9a-f]{64})"
)
_HTTP_LOCAL = threading.local()


class EndpointFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.response_body = response_body


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(dataset_prep.canonical_json_bytes(value)).hexdigest()


def _load_system_prompt() -> tuple[str, str]:
    raw = SYSTEM_PROMPT_PATH.read_bytes()
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if file_sha256 != SYSTEM_PROMPT_FILE_SHA256:
        raise RuntimeError(
            "shared TRACE response system prompt changed: "
            f"{file_sha256} != {SYSTEM_PROMPT_FILE_SHA256}"
        )
    prompt = raw.decode("utf-8").strip()
    if not prompt:
        raise RuntimeError(
            f"shared TRACE response system prompt is empty: {SYSTEM_PROMPT_PATH}"
        )
    return prompt, file_sha256


def _completion_url(endpoint_url: str) -> str:
    text = endpoint_url.strip().rstrip("/")
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"invalid OpenAI-compatible endpoint URL: {endpoint_url!r}")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "endpoint URL must not contain credentials, query, or fragment"
        )
    if text.endswith("/chat/completions"):
        return text
    if text.endswith("/v1"):
        return text + "/chat/completions"
    return text + "/v1/chat/completions"


def _validate_endpoint_auth(completion_url: str, api_key: str | None) -> None:
    """Reject bearer credentials over cleartext except on explicit loopback hosts."""

    if not api_key or urlparse(completion_url).scheme == "https":
        return
    parsed = urlparse(completion_url)
    hostname = (parsed.hostname or "").rstrip(".").lower()
    is_loopback = hostname == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise ValueError(
            "refusing to send an API key over plain HTTP to a non-loopback endpoint"
        )


def _api_key_from_environment(variable_name: str | None) -> str | None:
    """Resolve endpoint authentication only after an explicit CLI opt-in."""

    if variable_name is None:
        return None
    variable_name = variable_name.strip()
    if not variable_name:
        raise ValueError("--api-key-env must name a non-empty environment variable")
    value = os.environ.get(variable_name, "").strip()
    if not value:
        raise ValueError(
            f"--api-key-env requested {variable_name!r}, but it is unset or empty"
        )
    return value


def _validate_model_revision(revision: str) -> str:
    revision = revision.strip()
    if not _IMMUTABLE_REVISION_RE.fullmatch(revision):
        raise ValueError(
            "model revision must be an immutable 40/64-hex commit or "
            "sha256set:<64-hex> identity"
        )
    return revision


def _media_path(manifest_path: Path, image_record: Mapping[str, Any]) -> Path:
    relative = image_record.get("relative_path")
    if not isinstance(relative, str):
        raise RuntimeError("prepared image has no relative path")
    root = manifest_path.parent.resolve()
    path = root.joinpath(*Path(relative).parts).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise RuntimeError(f"unsafe or missing prepared image path: {relative!r}")
    return path


def _image_url(
    manifest_path: Path,
    image_record: Mapping[str, Any],
    *,
    media_transport: str,
    allowed_local_media_root: Path | None,
) -> str:
    path = _media_path(manifest_path, image_record)
    if media_transport == "file-url":
        if allowed_local_media_root is None:
            raise ValueError("file-url transport requires --allowed-local-media-root")
        allowed_root = allowed_local_media_root.expanduser().resolve()
        if not allowed_root.is_dir():
            raise ValueError(
                f"allowed local media root is not a directory: {allowed_root}"
            )
        if not path.is_relative_to(allowed_root):
            raise ValueError(
                f"prepared image {path} is outside allowed media root {allowed_root}"
            )
        return path.as_uri()
    if media_transport != "data-url":
        raise ValueError(f"unsupported media transport: {media_transport!r}")
    payload = path.read_bytes()
    current_sha256 = hashlib.sha256(payload).hexdigest()
    if current_sha256 != image_record.get("sha256"):
        raise RuntimeError(
            f"prepared image changed after manifest verification: {path}"
        )
    mime_type = image_record.get("mime_type")
    if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
        raise RuntimeError(f"prepared image has invalid MIME type: {mime_type!r}")
    return f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}"


def _user_content(
    row: Mapping[str, Any],
    manifest_path: Path,
    *,
    media_transport: str,
    allowed_local_media_root: Path | None,
) -> list[dict[str, Any]]:
    prompt = row.get("prompt_answer")
    images = row.get("images")
    if not isinstance(prompt, str) or not isinstance(images, list) or not images:
        raise RuntimeError(f"invalid prepared row {row.get('row_index')!r}")
    marker_count = prompt.count("<image>")
    if marker_count != len(images):
        raise RuntimeError(
            f"row {row.get('row_index')} has {marker_count} <image> markers for "
            f"{len(images)} images"
        )
    image_urls = [
        _image_url(
            manifest_path,
            image,
            media_transport=media_transport,
            allowed_local_media_root=allowed_local_media_root,
        )
        for image in images
    ]
    content: list[dict[str, Any]] = []
    segments = prompt.split("<image>")
    for index, segment in enumerate(segments):
        if segment:
            content.append({"type": "text", "text": segment})
        if index < len(image_urls):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_urls[index]},
                }
            )
    if not content:
        raise RuntimeError(f"row {row.get('row_index')} produced empty user content")
    return content


def build_request_payload(
    row: Mapping[str, Any],
    manifest_path: Path,
    *,
    served_model: str,
    system_prompt: str,
    media_transport: str = "data-url",
    allowed_local_media_root: Path | None = None,
) -> dict[str, Any]:
    """Build the exact native-chat request for one row."""

    payload: dict[str, Any] = {
        "model": served_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": _user_content(
                    row,
                    manifest_path,
                    media_transport=media_transport,
                    allowed_local_media_root=allowed_local_media_root,
                ),
            },
        ],
        "mm_processor_kwargs": dict(MM_PROCESSOR_CONTRACT),
        **DECODING_CONTRACT,
    }
    # Ground truth is available only in the preparation manifest.  Keep this
    # structural assertion next to request construction to prevent future leaks.
    forbidden_keys = {"answer_gt", "annotation_gt", "reward_contract"}
    stack: list[Any] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            leaked = forbidden_keys.intersection(value)
            if leaked:
                raise RuntimeError(
                    f"ground-truth fields leaked into request: {sorted(leaked)}"
                )
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return payload


def _http_session() -> requests.Session:
    session = getattr(_HTTP_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _HTTP_LOCAL.session = session
    return session


def _post_once(
    *,
    completion_url: str,
    payload: Mapping[str, Any],
    api_key: str | None,
    request_timeout_seconds: float,
) -> tuple[dict[str, Any], str, str | None]:
    _validate_endpoint_auth(completion_url, api_key)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = _http_session().post(
            completion_url,
            headers=headers,
            json=payload,
            timeout=request_timeout_seconds,
        )
    except requests.RequestException as exc:
        raise EndpointFailure(
            f"endpoint request failed: {exc}",
            retryable=True,
        ) from exc
    if response.status_code != 200:
        body = response.text
        retryable = (
            response.status_code in {408, 409, 425, 429} or response.status_code >= 500
        )
        raise EndpointFailure(
            f"endpoint returned HTTP {response.status_code}",
            retryable=retryable,
            status_code=response.status_code,
            response_body=body,
        )
    try:
        response_payload = response.json()
    except ValueError as exc:
        raise EndpointFailure(
            "endpoint returned non-JSON success response",
            retryable=True,
            status_code=response.status_code,
            response_body=response.text,
        ) from exc
    if not isinstance(response_payload, dict):
        raise EndpointFailure("endpoint response is not a JSON object", retryable=True)
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise EndpointFailure(
            "endpoint response must contain exactly one choice",
            retryable=True,
        )
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise EndpointFailure("endpoint choice has no message", retryable=True)
    raw_response = choice["message"].get("content")
    if not isinstance(raw_response, str) or not raw_response:
        raise EndpointFailure(
            "endpoint choice has empty non-string content", retryable=True
        )
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise EndpointFailure("endpoint finish_reason is not a string", retryable=True)
    return response_payload, raw_response, finish_reason


def _artifact_root(output_dir: Path, *, shard_index: int, shard_count: int) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if shard_count == 1 and shard_index == 0:
        return output_dir
    return output_dir / "shards" / f"{shard_index:05d}-of-{shard_count:05d}"


def _receipt_path(artifact_root: Path, row_index: int) -> Path:
    return artifact_root / "receipts" / f"{row_index:06d}.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load durable generation artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"generation artifact is not a JSON object: {path}")
    return value


def _validate_existing_receipt(
    receipt: Mapping[str, Any],
    *,
    path: Path,
    row: Mapping[str, Any],
    model_slug: str,
    run_identity_sha256: str,
    request_hash: str,
) -> bool:
    expected = {
        "schema_version": RECEIPT_SCHEMA,
        "run_identity_sha256": run_identity_sha256,
        "row_index": row["row_index"],
        "instance_id": row["instance_id"],
        "model_slug": model_slug,
        "request_hash": request_hash,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(
                f"existing receipt identity mismatch at {path}: "
                f"{key}={receipt.get(key)!r} expected {value!r}"
            )
    if receipt.get("status") == "error":
        return False
    if receipt.get("status") != "complete":
        raise RuntimeError(f"existing receipt has invalid status at {path}")
    raw_response = receipt.get("raw_response")
    api_response = receipt.get("api_response")
    if not isinstance(raw_response, str) or not isinstance(api_response, dict):
        raise RuntimeError(f"existing completed receipt is incomplete at {path}")
    if hashlib.sha256(raw_response.encode("utf-8")).hexdigest() != receipt.get(
        "raw_response_sha256"
    ):
        raise RuntimeError(f"existing receipt raw response hash mismatch at {path}")
    if _canonical_hash(api_response) != receipt.get("response_sha256"):
        raise RuntimeError(f"existing receipt API response hash mismatch at {path}")
    return True


def _base_receipt(
    *,
    row: Mapping[str, Any],
    model_slug: str,
    run_identity_sha256: str,
    request_hash: str,
    prompt_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "run_identity_sha256": run_identity_sha256,
        "row_index": row["row_index"],
        "instance_id": row["instance_id"],
        "task": row["task"],
        "domain": row["domain"],
        "answer_type": row["answer_type"],
        "model_slug": model_slug,
        "request_hash": request_hash,
        "request": {
            "prompt_sha256": prompt_sha256,
            "ordered_image_sha256": [image["sha256"] for image in row["images"]],
        },
    }


def _generate_row(
    *,
    row: Mapping[str, Any],
    manifest_path: Path,
    artifact_root: Path,
    model_slug: str,
    served_model: str,
    system_prompt: str,
    completion_url: str,
    run_identity_sha256: str,
    media_transport: str,
    allowed_local_media_root: Path | None,
    api_key: str | None,
    max_attempts: int,
    retry_backoff_seconds: float,
    request_timeout_seconds: float,
) -> dict[str, Any]:
    payload = build_request_payload(
        row,
        manifest_path,
        served_model=served_model,
        system_prompt=system_prompt,
        media_transport=media_transport,
        allowed_local_media_root=allowed_local_media_root,
    )
    request_hash = _canonical_hash(payload)
    path = _receipt_path(artifact_root, int(row["row_index"]))
    if path.exists():
        receipt = _load_json(path)
        if _validate_existing_receipt(
            receipt,
            path=path,
            row=row,
            model_slug=model_slug,
            run_identity_sha256=run_identity_sha256,
            request_hash=request_hash,
        ):
            return receipt

    base = _base_receipt(
        row=row,
        model_slug=model_slug,
        run_identity_sha256=run_identity_sha256,
        request_hash=request_hash,
        prompt_sha256=hashlib.sha256(row["prompt_answer"].encode("utf-8")).hexdigest(),
    )
    attempt_history: list[dict[str, Any]] = []
    started_at = _utc_now()
    for attempt in range(1, max_attempts + 1):
        try:
            api_response, raw_response, finish_reason = _post_once(
                completion_url=completion_url,
                payload=payload,
                api_key=api_key,
                request_timeout_seconds=request_timeout_seconds,
            )
        except EndpointFailure as exc:
            attempt_record: dict[str, Any] = {
                "attempt": attempt,
                "error": str(exc),
                "retryable": exc.retryable,
                "status_code": exc.status_code,
            }
            if exc.response_body is not None:
                attempt_record["response_body_sha256"] = hashlib.sha256(
                    exc.response_body.encode("utf-8", errors="replace")
                ).hexdigest()
                attempt_record["response_body_excerpt"] = exc.response_body[:2_000]
            attempt_history.append(attempt_record)
            if not exc.retryable or attempt == max_attempts:
                receipt = {
                    **base,
                    "status": "error",
                    "raw_response": None,
                    "raw_response_sha256": None,
                    "finish_reason": None,
                    "response_sha256": None,
                    "api_response": None,
                    "attempt_count": attempt,
                    "attempt_history": attempt_history,
                    "error": str(exc),
                    "started_at": started_at,
                    "completed_at": _utc_now(),
                }
                dataset_prep.atomic_write_json(path, receipt)
                return receipt
            delay = min(30.0, retry_backoff_seconds * (2 ** (attempt - 1)))
            if delay > 0:
                time.sleep(delay)
            continue

        receipt = {
            **base,
            "status": "complete",
            "raw_response": raw_response,
            "raw_response_sha256": hashlib.sha256(
                raw_response.encode("utf-8")
            ).hexdigest(),
            "finish_reason": finish_reason,
            "response_sha256": _canonical_hash(api_response),
            "api_response": api_response,
            "attempt_count": attempt,
            "attempt_history": attempt_history,
            "error": None,
            "started_at": started_at,
            "completed_at": _utc_now(),
        }
        dataset_prep.atomic_write_json(path, receipt)
        return receipt

    raise AssertionError("unreachable generation retry state")


def _write_responses_jsonl(path: Path, receipts: list[dict[str, Any]]) -> None:
    encoded = b"".join(
        dataset_prep.canonical_json_bytes(receipt) for receipt in receipts
    )
    dataset_prep.atomic_write_bytes(path, encoded)


def _validate_response_set(
    responses_path: Path,
    *,
    expected_rows: list[Mapping[str, Any]],
    require_complete: bool,
) -> list[dict[str, Any]]:
    try:
        lines = responses_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(
            f"cannot read generated responses: {responses_path}"
        ) from exc
    if len(lines) != len(expected_rows):
        raise RuntimeError(
            f"response row count mismatch: {len(lines)} != {len(expected_rows)}"
        )
    records: list[dict[str, Any]] = []
    seen_instance_ids: set[str] = set()
    for expected, line in zip(expected_rows, lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid response JSONL row for {expected['row_index']}"
            ) from exc
        if not isinstance(record, dict):
            raise RuntimeError(
                f"response JSONL row is not an object: {expected['row_index']}"
            )
        if record.get("row_index") != expected["row_index"]:
            raise RuntimeError(
                f"response JSONL row order changed at {expected['row_index']}"
            )
        if record.get("instance_id") != expected["instance_id"]:
            raise RuntimeError(
                f"response JSONL identity changed at {expected['row_index']}"
            )
        if record["instance_id"] in seen_instance_ids:
            raise RuntimeError(
                f"duplicate response instance_id: {record['instance_id']}"
            )
        seen_instance_ids.add(record["instance_id"])
        required = {
            "model_slug",
            "raw_response",
            "finish_reason",
            "request_hash",
            "response_sha256",
        }
        if not required.issubset(record):
            raise RuntimeError(
                f"response JSONL row lacks scorer fields: {expected['row_index']}"
            )
        if require_complete and record.get("status") != "complete":
            raise RuntimeError(
                f"response JSONL contains failed row: {expected['row_index']}"
            )
        records.append(record)
    if len(seen_instance_ids) != len(expected_rows):
        raise RuntimeError(
            "response JSONL does not contain the exact unique selected rows"
        )
    return records


def run_generation(
    *,
    manifest_path: Path,
    output_dir: Path,
    endpoint_url: str,
    served_model: str,
    model_slug: str,
    model_path: Path,
    model_revision: str,
    media_transport: str = "data-url",
    allowed_local_media_root: Path | None = None,
    api_key: str | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    concurrency: int = 16,
    max_attempts: int = 4,
    retry_backoff_seconds: float = 1.0,
    request_timeout_seconds: float = 600.0,
    progress_every: int = 50,
    expected_rows: int = dataset_prep.EXPECTED_ROWS,
    require_pinned_manifest: bool = True,
) -> dict[str, Any]:
    """Run or resume one model/shard and return its final run metadata."""

    if not served_model.strip() or not model_slug.strip():
        raise ValueError("served model and model slug must be non-empty")
    model_revision = _validate_model_revision(model_revision)
    model_path = model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise ValueError(f"model path is not a local directory: {model_path}")
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"invalid shard {shard_index} of {shard_count}")
    if concurrency <= 0 or max_attempts <= 0:
        raise ValueError("concurrency and max attempts must be positive")
    if retry_backoff_seconds < 0 or request_timeout_seconds <= 0:
        raise ValueError(
            "retry backoff must be nonnegative and timeout must be positive"
        )
    if media_transport not in {"data-url", "file-url"}:
        raise ValueError(f"unsupported media transport: {media_transport!r}")
    if media_transport == "file-url" and allowed_local_media_root is None:
        raise ValueError("file-url transport requires --allowed-local-media-root")

    completion_url = _completion_url(endpoint_url)
    _validate_endpoint_auth(completion_url, api_key)

    manifest_path = manifest_path.expanduser().resolve()
    manifest = dataset_prep.load_manifest(
        manifest_path,
        expected_rows=expected_rows,
        require_pinned=require_pinned_manifest,
        verify_media=True,
    )
    manifest_sha256 = dataset_prep.sha256_file(manifest_path)
    system_prompt, system_prompt_file_sha256 = _load_system_prompt()
    selected_rows = [
        row for row in manifest["rows"] if row["row_index"] % shard_count == shard_index
    ]
    expected_selected_count = len(range(shard_index, expected_rows, shard_count))
    if len(selected_rows) != expected_selected_count:
        raise RuntimeError(
            f"selected shard row count mismatch: {len(selected_rows)} != {expected_selected_count}"
        )

    resolved_allowed_root = (
        allowed_local_media_root.expanduser().resolve()
        if allowed_local_media_root is not None
        else None
    )
    run_contract: dict[str, Any] = {
        "dataset_manifest_sha256": manifest_sha256,
        "dataset_repo_id": manifest["dataset"]["repo_id"],
        "dataset_revision": manifest["dataset"]["revision"],
        "dataset_file_sha256": manifest["dataset"]["file_sha256"],
        "system_prompt_file_sha256": system_prompt_file_sha256,
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "endpoint_url": completion_url,
        "served_model": served_model,
        "model_slug": model_slug,
        "model_path": str(model_path),
        "model_revision": model_revision,
        "media_transport": media_transport,
        "allowed_local_media_root": (
            str(resolved_allowed_root) if resolved_allowed_root is not None else None
        ),
        "decoding": DECODING_CONTRACT,
        "mm_processor_kwargs": MM_PROCESSOR_CONTRACT,
        "shard_index": shard_index,
        "shard_count": shard_count,
    }
    run_identity_sha256 = _canonical_hash(run_contract)
    artifact_root = _artifact_root(
        output_dir, shard_index=shard_index, shard_count=shard_count
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    metadata_path = artifact_root / "run_metadata.json"
    responses_path = artifact_root / "responses.jsonl"

    created_at = _utc_now()
    if metadata_path.exists():
        existing_metadata = _load_json(metadata_path)
        if existing_metadata.get("run_identity_sha256") != run_identity_sha256:
            raise RuntimeError(
                f"output directory belongs to a different immutable run: {metadata_path}"
            )
        created_at = str(existing_metadata.get("created_at") or created_at)
    running_metadata: dict[str, Any] = {
        "schema_version": RUN_METADATA_SCHEMA,
        "status": "running",
        "run_identity_sha256": run_identity_sha256,
        "created_at": created_at,
        "updated_at": _utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "dataset_repo_id": manifest["dataset"]["repo_id"],
        "dataset_revision": manifest["dataset"]["revision"],
        "dataset_file": manifest["dataset"]["file"],
        "dataset_file_sha256": manifest["dataset"]["file_sha256"],
        "dataset_rows": expected_rows,
        "model_slug": model_slug,
        "served_model": served_model,
        "model_path": str(model_path),
        "model_revision": model_revision,
        "endpoint_url": completion_url,
        "system_prompt_path": str(SYSTEM_PROMPT_PATH),
        "system_prompt_file_sha256": system_prompt_file_sha256,
        "decoding": dict(DECODING_CONTRACT),
        "mm_processor_kwargs": dict(MM_PROCESSOR_CONTRACT),
        "media_transport": media_transport,
        "allowed_local_media_root": (
            str(resolved_allowed_root) if resolved_allowed_root is not None else None
        ),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected_rows": len(selected_rows),
        "responses_path": str(responses_path),
        "receipts_path": str(artifact_root / "receipts"),
        "completed_rows": 0,
        "error_rows": 0,
        "run_contract": run_contract,
    }
    dataset_prep.atomic_write_json(metadata_path, running_metadata)

    receipts_by_index: dict[int, dict[str, Any]] = {}
    completed_futures = 0
    with ThreadPoolExecutor(
        max_workers=min(concurrency, max(1, len(selected_rows)))
    ) as executor:
        futures = {
            executor.submit(
                _generate_row,
                row=row,
                manifest_path=manifest_path,
                artifact_root=artifact_root,
                model_slug=model_slug,
                served_model=served_model,
                system_prompt=system_prompt,
                completion_url=completion_url,
                run_identity_sha256=run_identity_sha256,
                media_transport=media_transport,
                allowed_local_media_root=resolved_allowed_root,
                api_key=api_key,
                max_attempts=max_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
                request_timeout_seconds=request_timeout_seconds,
            ): row
            for row in selected_rows
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                receipt = future.result()
            except BaseException as exc:
                # Construction/storage failures are not safe to convert into a
                # normal endpoint receipt.  Cancel pending work and fail loudly.
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(
                    f"generation worker failed before durable receipt for row "
                    f"{row['row_index']}: {exc}"
                ) from exc
            receipts_by_index[int(row["row_index"])] = receipt
            completed_futures += 1
            if progress_every > 0 and (
                completed_futures % progress_every == 0
                or completed_futures == len(selected_rows)
            ):
                print(
                    f"[trace-validation] {model_slug} shard {shard_index}/{shard_count}: "
                    f"{completed_futures}/{len(selected_rows)} durable rows",
                    flush=True,
                )

    ordered_receipts = [receipts_by_index[row["row_index"]] for row in selected_rows]
    if len(ordered_receipts) != len(selected_rows):
        raise RuntimeError("generation lost rows before response assembly")
    _write_responses_jsonl(responses_path, ordered_receipts)
    _validate_response_set(
        responses_path,
        expected_rows=selected_rows,
        require_complete=False,
    )
    completed_rows = sum(
        receipt.get("status") == "complete" for receipt in ordered_receipts
    )
    error_rows = len(ordered_receipts) - completed_rows
    final_metadata = {
        **running_metadata,
        "status": "complete" if error_rows == 0 else "incomplete",
        "updated_at": _utc_now(),
        "completed_at": _utc_now(),
        "completed_rows": completed_rows,
        "error_rows": error_rows,
        "responses_sha256": dataset_prep.sha256_file(responses_path),
    }
    dataset_prep.atomic_write_json(metadata_path, final_metadata)
    if error_rows == 0:
        _validate_response_set(
            responses_path,
            expected_rows=selected_rows,
            require_complete=True,
        )
    return final_metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint-url", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--media-transport",
        choices=("data-url", "file-url"),
        default="data-url",
        help="data-url is safest; file-url requires a matching vLLM allowed-local-media-path",
    )
    parser.add_argument(
        "--allowed-local-media-root",
        type=Path,
        help="required containment root when --media-transport=file-url",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        metavar="ENV_VAR",
        help=(
            "opt in to bearer authentication using the named environment variable; "
            "the default sends no Authorization header"
        ),
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    api_key = _api_key_from_environment(args.api_key_env)
    metadata = run_generation(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        endpoint_url=args.endpoint_url,
        served_model=args.served_model,
        model_slug=args.model_slug,
        model_path=args.model_path,
        model_revision=args.model_revision,
        media_transport=args.media_transport,
        allowed_local_media_root=args.allowed_local_media_root,
        api_key=api_key,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        concurrency=args.concurrency,
        max_attempts=args.max_attempts,
        retry_backoff_seconds=args.retry_backoff_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "model_slug": metadata["model_slug"],
                "responses_path": metadata["responses_path"],
                "responses_sha256": metadata["responses_sha256"],
                "completed_rows": metadata["completed_rows"],
                "error_rows": metadata["error_rows"],
            },
            sort_keys=True,
        )
    )
    return 0 if metadata["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
