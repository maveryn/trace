#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd
import requests
from tqdm import tqdm

try:
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from benchmark_queue_lib import (  # noqa: E402
    BENCHMARK_RUN_SETS,
    DEFAULT_QUEUE_ROOT,
    REPO_ROOT as LIB_REPO_ROOT,
    BenchmarkSpec,
    benchmark_specs_for_run_set,
    build_vlmeval_dataset,
    filter_benchmark_specs,
    json_default,
    run_dir,
    write_json,
)
from trace_eval_media_contract import (  # noqa: E402
    DATASET_MANIFEST_SCHEMA,
    GENERATION_CONTRACT_VERSION,
    MEDIA_CONTRACT_VERSION,
    MEDIA_TRANSPORT,
    QWEN_MAX_IMAGE_PIXELS,
    QWEN_MIN_IMAGE_PIXELS,
    load_dataset_manifest,
    media_set_sha256,
    row_lookup_key,
    source_record_sha256,
    source_row_sha256,
)


def _import_vlmeval_runner():
    scripts_root = LIB_REPO_ROOT / "external" / "VLMEvalKit" / "scripts"
    for path in (scripts_root.parent, scripts_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import batched_vlmevalkit_qwen3vl as runner

    return runner


def _row_hash(row: Any) -> str:
    """Compatibility name for the immutable producer source-row hash."""

    return source_row_sha256(row)


@dataclass(frozen=True)
class RowJob:
    spec: BenchmarkSpec
    row: dict[str, Any]
    rank: int
    row_key: str
    output_dir: Path
    result_path: Path
    source_ordinal: int | None = None
    mirror_result_path: Path | None = None
    attempted_endpoints: tuple[str, ...] = ()
    attempt_count: int = 0


@dataclass(frozen=True)
class PreparedJob:
    job: RowJob
    messages: list[dict[str, Any]]
    request_hash: str | None = None
    media_provenance: dict[str, Any] | None = None


@dataclass(frozen=True)
class PersistenceJob:
    job: RowJob
    payload: dict[str, Any]
    error: dict[str, Any] | None = None


class PipelineMetrics:
    """Small thread-safe timing and queue telemetry for generation stages."""

    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self._stages: dict[str, dict[str, float | int]] = {}
        self._queue_high_watermarks: dict[str, int] = {}
        self._counters: Counter[str] = Counter()
        self._lock = threading.Lock()

    def observe_stage(self, stage: str, seconds: float) -> None:
        value = max(0.0, float(seconds))
        with self._lock:
            stats = self._stages.setdefault(
                stage, {"count": 0, "total_seconds": 0.0, "max_seconds": 0.0}
            )
            stats["count"] = int(stats["count"]) + 1
            stats["total_seconds"] = float(stats["total_seconds"]) + value
            stats["max_seconds"] = max(float(stats["max_seconds"]), value)

    def observe_queue(self, name: str, value: int) -> None:
        with self._lock:
            self._queue_high_watermarks[name] = max(
                self._queue_high_watermarks.get(name, 0), int(value)
            )

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += int(value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stages = {
                name: {
                    **stats,
                    "mean_seconds": (
                        float(stats["total_seconds"]) / int(stats["count"])
                        if int(stats["count"])
                        else 0.0
                    ),
                }
                for name, stats in self._stages.items()
            }
            queues = dict(self._queue_high_watermarks)
            counters = dict(self._counters)
        return {
            "elapsed_seconds": max(0.0, time.monotonic() - self.started_at),
            "stages": stages,
            "queue_high_watermarks": queues,
            "counters": counters,
        }


class PermanentAPIError(RuntimeError):
    """A request error caused by the row payload rather than endpoint health."""

    def __init__(self, endpoint: str, status_code: int, detail: str):
        self.endpoint = endpoint
        self.status_code = int(status_code)
        self.detail = detail
        super().__init__(f"{endpoint} returned HTTP {status_code}: {detail}")


class RetryableAPIError(RuntimeError):
    """A transport or server error that may succeed on another endpoint."""

    def __init__(self, endpoint: str, detail: str, *, affects_health: bool = True):
        self.endpoint = endpoint
        self.detail = detail
        self.affects_health = bool(affects_health)
        super().__init__(f"{endpoint} failed: {detail}")


class EndpointHealth:
    """Track endpoint failures and quarantine unhealthy workers for this run."""

    def __init__(self, endpoints: list[str], failure_threshold: int):
        self.endpoints = tuple(endpoints)
        self.failure_threshold = max(1, int(failure_threshold))
        self.failure_streak = {endpoint: 0 for endpoint in self.endpoints}
        self.failure_count = {endpoint: 0 for endpoint in self.endpoints}
        self.disabled: set[str] = set()
        self.lock = threading.Lock()

    def is_disabled(self, endpoint: str) -> bool:
        with self.lock:
            return endpoint in self.disabled

    def should_defer(self, endpoint: str, attempted_endpoints: tuple[str, ...]) -> bool:
        attempted = set(attempted_endpoints)
        if endpoint not in attempted:
            return False
        with self.lock:
            return any(
                candidate not in self.disabled and candidate not in attempted
                for candidate in self.endpoints
            )

    def record_success(self, endpoint: str) -> None:
        with self.lock:
            self.failure_streak[endpoint] = 0

    def record_failure(self, endpoint: str) -> bool:
        """Record a retriable failure and return whether the endpoint was newly disabled."""

        with self.lock:
            self.failure_count[endpoint] += 1
            self.failure_streak[endpoint] += 1
            if self.failure_streak[endpoint] < self.failure_threshold:
                return False
            # Keep one endpoint available so queued rows terminate with explicit
            # errors instead of leaving queue.join() blocked when the whole pool
            # is unavailable.
            if (
                endpoint in self.disabled
                or len(self.disabled) >= len(self.endpoints) - 1
            ):
                return False
            self.disabled.add(endpoint)
            return True

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "failure_threshold": self.failure_threshold,
                "disabled_endpoints": sorted(self.disabled),
                "failure_count": dict(self.failure_count),
            }


class EndpointRouter:
    """Bound concurrent calls per endpoint while allowing prepared-job failover."""

    def __init__(
        self,
        endpoints: list[str],
        parallelism_per_endpoint: int,
        health: EndpointHealth,
    ):
        self.endpoints = tuple(endpoints)
        self.health = health
        self._slots = {
            endpoint: threading.BoundedSemaphore(max(1, int(parallelism_per_endpoint)))
            for endpoint in self.endpoints
        }
        self._cursor = 0
        self._lock = threading.Lock()

    def acquire(
        self, attempted_endpoints: tuple[str, ...], timeout: float = 0.25
    ) -> str | None:
        attempted = set(attempted_endpoints)
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                start = self._cursor
                self._cursor = (self._cursor + 1) % len(self.endpoints)
            ordered = [
                self.endpoints[(start + offset) % len(self.endpoints)]
                for offset in range(len(self.endpoints))
            ]
            available = [
                endpoint
                for endpoint in ordered
                if not self.health.is_disabled(endpoint)
            ]
            preferred = [
                endpoint for endpoint in available if endpoint not in attempted
            ]
            for endpoint in preferred or available:
                if self._slots[endpoint].acquire(blocking=False):
                    return endpoint
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.002)

    def release(self, endpoint: str) -> None:
        self._slots[endpoint].release()


class DatasetHandle:
    def __init__(
        self,
        spec: BenchmarkSpec,
        dataset: Any = None,
        media_rows: dict[tuple[int, str, str], dict[str, Any]] | None = None,
        dataset_snapshot_sha256: str | None = None,
        source_ordinals: list[int] | None = None,
    ):
        self.spec = spec
        self.dataset = dataset
        self.media_rows = media_rows
        self.dataset_snapshot_sha256 = dataset_snapshot_sha256
        self.source_ordinals = source_ordinals
        self.lock = threading.Lock()


COMPACT_MEDIA_COLUMNS = frozenset(
    {
        "image",
        "images",
        "image_path",
        "image_paths",
    }
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_result_name(index: Any) -> str:
    text = str(index)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:80]
    return f"{safe}.{digest}.json"


def _row_key(row: dict[str, Any], rank: int) -> str:
    row_hash = _row_hash(row)[:16]
    return f"{int(rank):08d}:{row.get('index')}:{row_hash}"


def _safe_result_name_for_row(row: dict[str, Any], rank: int) -> str:
    index = str(row.get("index"))
    safe_index = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in index)[
        :64
    ]
    digest = hashlib.sha256(_row_key(row, rank).encode("utf-8")).hexdigest()[:16]
    return f"{int(rank):08d}.{safe_index}.{digest}.json"


def _row_identity_from_key(index: Any, row_key: Any) -> str | None:
    """Return an order-independent row identity from a persisted row key."""

    text = str(row_key or "")
    if not text or ":" not in text:
        return None
    row_hash = text.rsplit(":", 1)[-1]
    if not re.fullmatch(r"[0-9a-f]{16}", row_hash):
        return None
    return f"identity:{index}:{row_hash}"


def _row_identity(row: dict[str, Any]) -> str:
    return f"identity:{row.get('index')}:{_row_hash(row)[:16]}"


def _compact_prediction_record(record: dict[str, Any]) -> dict[str, Any]:
    """Drop duplicated media payloads while retaining scoring metadata."""

    return {
        key: value
        for key, value in record.items()
        if str(key).lower() not in COMPACT_MEDIA_COLUMNS
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
    tmp.write_bytes(payload)
    tmp.replace(path)


def _write_job_result(job: RowJob, payload: dict[str, Any]) -> None:
    """Atomically write the canonical result and its optional persistent mirror."""

    encoded = (
        json.dumps(payload, ensure_ascii=False, default=json_default) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(job.result_path, encoded)
    if job.mirror_result_path is not None:
        _atomic_write_bytes(job.mirror_result_path, encoded)


def _persist_worker_result(
    job: RowJob,
    payload: dict[str, Any],
    errors: list[dict[str, Any]],
    error_lock: threading.Lock,
) -> bool:
    """Persist a final row result and surface storage failures in the suite result."""

    try:
        _write_job_result(job, payload)
        return True
    except OSError as exc:
        error = {
            "benchmark_key": job.spec.key,
            "index": str(job.row.get("index")),
            "row_key": job.row_key,
            "error": f"result persistence failed: {exc!r}",
            "result_path": str(job.result_path),
            "mirror_result_path": (
                str(job.mirror_result_path)
                if job.mirror_result_path is not None
                else None
            ),
        }
        with error_lock:
            errors.append(error)
        print(f"[api-generate:persistence-error] row_key={job.row_key} error={exc!r}")
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_media_path(value: Any, allowed_root: Path | None = None) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(
            "file-url media transport requires a materialized local path; "
            f"received {type(value).__name__}"
        )
    text = str(value)
    parsed = urlparse(text)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
    elif parsed.scheme:
        raise ValueError(
            f"file-url media transport rejects non-local URL: {text[:160]}"
        )
    else:
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing prompt media: {path}")
    if allowed_root is not None:
        root = allowed_root.expanduser().resolve()
        if not path.is_relative_to(root):
            raise ValueError(
                f"prompt media {path} is outside allowed local media path {root}"
            )
    return path


def _image_to_file_url(value: Any, *, allowed_root: Path | None = None) -> str:
    """Return an exact local source URL without decoding or re-encoding pixels."""

    return _local_media_path(value, allowed_root).as_uri()


def _image_transport_url(
    args: argparse.Namespace,
    value: Any,
    *,
    metrics: PipelineMetrics | None = None,
) -> str:
    del metrics
    if getattr(args, "media_transport", MEDIA_TRANSPORT) != MEDIA_TRANSPORT:
        raise ValueError("trace_eval_v1 requires file-url media transport")
    return _image_to_file_url(
        value,
        allowed_root=getattr(args, "allowed_local_media_path", None),
    )


def _load_generation_dataset_manifest(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    cached = getattr(args, "_loaded_dataset_manifest", None)
    if cached is not None:
        return cached
    path = getattr(args, "dataset_manifest", None)
    if path is None:
        return None
    manifest = load_dataset_manifest(Path(path).expanduser().resolve())
    view = str(
        getattr(args, "dataset_manifest_view", "trace_eval_v1") or "trace_eval_v1"
    )
    view_snapshots = manifest.get("view_snapshot_sha256") or {}
    if view not in view_snapshots:
        raise RuntimeError(
            f"TRACE evaluation dataset manifest has no complete snapshot for view {view!r}"
        )
    snapshot = str(view_snapshots[view])
    configured_revision = os.environ.get("TRACE_EVAL_DATASET_REVISION", "").strip()
    expected_revision = f"{DATASET_MANIFEST_SCHEMA}:{snapshot}"
    if configured_revision and configured_revision != expected_revision:
        raise RuntimeError(
            "TRACE_EVAL_DATASET_REVISION does not match the prepared media snapshot: "
            f"{configured_revision!r} != {expected_revision!r}"
        )
    args._loaded_dataset_manifest = manifest
    args.dataset_snapshot_sha256 = snapshot
    return manifest


def _media_rows_for_spec(
    manifest: dict[str, Any] | None,
    spec: BenchmarkSpec,
) -> tuple[dict[tuple[int, str, str], dict[str, Any]] | None, str | None]:
    if manifest is None:
        return None, None
    receipt = (manifest.get("datasets") or {}).get(spec.key)
    if not isinstance(receipt, dict) or receipt.get("status") != "ready":
        raise RuntimeError(
            f"TRACE evaluation dataset manifest has no ready receipt for {spec.key}"
        )
    rows = receipt.get("row_media")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(
            f"TRACE evaluation dataset manifest has no row media records for {spec.key}"
        )
    index: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in rows:
        key = row_lookup_key(
            int(row.get("ordinal")),
            row.get("index"),
            str(row.get("source_row_hash") or ""),
        )
        if key in index:
            raise RuntimeError(
                f"duplicate TRACE evaluation media row identity for {spec.key}: {key}"
            )
        index[key] = row
    return index, str(receipt.get("dataset_snapshot_sha256") or "")


def _image_paths_from_messages(messages: list[dict[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            image_url = item.get("image_url") or {}
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if not isinstance(url, str) or not url.startswith("file://"):
                raise RuntimeError(
                    "official TRACE evaluation media requests must contain local file:// image URLs"
                )
            paths.append(_local_media_path(url))
    return paths


def _media_provenance(
    args: argparse.Namespace,
    handle: DatasetHandle,
    row: dict[str, Any],
    messages: list[dict[str, Any]],
    source_ordinal: int,
) -> dict[str, Any]:
    source_row_hash = _row_hash(row)
    paths = _image_paths_from_messages(messages)
    if handle.media_rows is not None:
        manifest_row = handle.media_rows.get(
            row_lookup_key(source_ordinal, row.get("index"), source_row_hash)
        )
        if manifest_row is None:
            raise RuntimeError(
                f"prepared media manifest has no current row for {handle.spec.key}/{row.get('index')}"
            )
        expected_media = list(manifest_row.get("media") or [])
        if len(expected_media) != len(paths):
            raise RuntimeError(
                f"prepared media count changed for {handle.spec.key}/{row.get('index')}: "
                f"{len(paths)} != {len(expected_media)}"
            )
        ordered_media = []
        for path, expected in zip(paths, expected_media):
            expected_path = Path(str(expected.get("path") or "")).resolve()
            if path != expected_path:
                raise RuntimeError(
                    f"prepared media path changed for {handle.spec.key}/{row.get('index')}: "
                    f"{path} != {expected_path}"
                )
            size = path.stat().st_size
            if size != int(expected.get("size_bytes") or -1):
                raise RuntimeError(
                    f"prepared media size changed; rerun dataset verification: {path}"
                )
            current_sha256 = _sha256_file(path)
            if current_sha256 != str(expected.get("sha256") or ""):
                raise RuntimeError(
                    f"prepared media content changed; rerun dataset verification: {path}"
                )
            ordered_media.append(
                {
                    "type": str(expected.get("type") or "image"),
                    "sha256": current_sha256,
                }
            )
        media_hash = media_set_sha256(ordered_media)
        if media_hash != manifest_row.get("media_set_sha256"):
            raise RuntimeError(
                f"prepared media hash list is corrupt for {handle.spec.key}/{row.get('index')}"
            )
        record_hash = source_record_sha256(source_row_hash, media_hash)
        if record_hash != manifest_row.get("source_record_sha256"):
            raise RuntimeError(
                f"prepared source record is stale for {handle.spec.key}/{row.get('index')}"
            )
    else:
        ordered_media = [
            {"type": "image", "sha256": _sha256_file(path)} for path in paths
        ]
        media_hash = media_set_sha256(ordered_media)
        record_hash = source_record_sha256(source_row_hash, media_hash)
    return {
        "source_row_hash": source_row_hash,
        "ordered_media_sha256": [item["sha256"] for item in ordered_media],
        "media_set_sha256": media_hash,
        "source_record_sha256": record_hash,
        "dataset_snapshot_sha256": (
            str(getattr(args, "dataset_snapshot_sha256", "") or "")
            or str(handle.dataset_snapshot_sha256 or "")
        ),
    }


def _vlmeval_messages(
    args: argparse.Namespace, handle: DatasetHandle, row: dict[str, Any]
) -> list[dict[str, Any]]:
    runner = _import_vlmeval_runner()
    row_series = pd.Series(row)
    # Some VLMEvalKit datasets materialize cached images in build_prompt. Keep that
    # per-dataset call serialized while allowing image encoding/API calls to run in parallel.
    with handle.lock:
        struct = runner.build_prompt_for_runner(handle.dataset, row_series)
    content: list[dict[str, Any]] = []
    for item in struct:
        typ = item.get("type")
        value = item.get("value")
        if typ == "image":
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_transport_url(
                            args,
                            value,
                            metrics=getattr(args, "_pipeline_metrics", None),
                        )
                    },
                }
            )
        elif typ not in {"image", "text"}:
            raise RuntimeError(
                f"trace_eval_v1 prompt contains unsupported media type {typ!r}: "
                f"{handle.spec.key}"
            )
        else:
            text = "" if value is None else str(value)
            if text:
                content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content or [{"type": "text", "text": ""}]}]


def _completion_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1"):
        return endpoint + "/chat/completions"
    return endpoint + "/v1/chat/completions"


def _resolve_image_pixel_defaults(args: argparse.Namespace) -> None:
    if getattr(args, "media_transport", MEDIA_TRANSPORT) != MEDIA_TRANSPORT:
        raise ValueError("trace_eval_v1 requires file-url media transport")
    if getattr(args, "min_image_pixels", None) is None:
        args.min_image_pixels = QWEN_MIN_IMAGE_PIXELS
    if getattr(args, "max_image_pixels", None) is None:
        args.max_image_pixels = QWEN_MAX_IMAGE_PIXELS


_HTTP_LOCAL = threading.local()


def _http_session() -> requests.Session:
    session = getattr(_HTTP_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _HTTP_LOCAL.session = session
    return session


def _close_http_session() -> None:
    session = getattr(_HTTP_LOCAL, "session", None)
    if session is not None:
        session.close()
        delattr(_HTTP_LOCAL, "session")


def _api_payload(
    args: argparse.Namespace, messages: list[dict[str, Any]]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.api_model,
        "messages": messages,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_tokens": int(args.max_tokens),
        "presence_penalty": float(args.presence_penalty),
        "stream": False,
    }
    if getattr(args, "media_transport", MEDIA_TRANSPORT) != MEDIA_TRANSPORT:
        raise ValueError("trace_eval_v1 requires file-url media transport")
    min_pixels = int(getattr(args, "min_image_pixels", QWEN_MIN_IMAGE_PIXELS))
    max_pixels = int(getattr(args, "max_image_pixels", QWEN_MAX_IMAGE_PIXELS))
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError(
            f"invalid Qwen image processor bounds: "
            f"min_pixels={min_pixels}, max_pixels={max_pixels}"
        )
    payload["mm_processor_kwargs"] = {
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }
    if args.top_k is not None:
        payload["top_k"] = int(args.top_k)
    if args.repetition_penalty is not None:
        payload["repetition_penalty"] = float(args.repetition_penalty)
    if args.seed is not None:
        payload["seed"] = int(args.seed)
    return payload


def _request_hash(args: argparse.Namespace, messages: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        _api_payload(args, messages),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _configured_model_revision(args: argparse.Namespace) -> str:
    from trace_eval_archive_hooks import resolve_model_revision

    return resolve_model_revision(str(args.model_slug), str(args.model))


def _selection_contract(args: argparse.Namespace) -> dict[str, Any]:
    cached = getattr(args, "_selection_contract", None)
    if isinstance(cached, dict):
        return cached

    material = {
        "mode": "full",
        "limit": None,
        "sample_seed": None,
        "subset_manifest_sha256": None,
    }
    args._selection_contract = material
    return material


def _generation_contract_hash(
    args: argparse.Namespace,
    *,
    model_revision: str | None = None,
    dataset_revision: str | None = None,
    final25_code_hash: str | None = None,
) -> str:
    """Hash every non-row input that can change a generation response."""

    media_transport = str(getattr(args, "media_transport", MEDIA_TRANSPORT))
    material = {
        "contract_version": GENERATION_CONTRACT_VERSION,
        "model": str(args.model),
        "model_slug": str(args.model_slug),
        "model_revision": model_revision or _configured_model_revision(args),
        "api_model": str(args.api_model),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k) if args.top_k is not None else None,
        "presence_penalty": float(args.presence_penalty),
        "repetition_penalty": (
            float(args.repetition_penalty)
            if args.repetition_penalty is not None
            else None
        ),
        "max_tokens": int(args.max_tokens),
        "seed": int(args.seed) if args.seed is not None else None,
        "media_contract_version": MEDIA_CONTRACT_VERSION,
        "media_transport": media_transport,
        "min_image_pixels": int(
            getattr(args, "min_image_pixels", QWEN_MIN_IMAGE_PIXELS)
        ),
        "max_image_pixels": int(args.max_image_pixels),
        "max_image_side": None,
        "image_jpeg_quality": None,
        "dataset_snapshot_sha256": str(
            getattr(args, "dataset_snapshot_sha256", "") or ""
        ),
        "dataset_revision": (
            dataset_revision
            if dataset_revision is not None
            else os.environ.get("TRACE_EVAL_DATASET_REVISION", "unknown")
        ),
        # Immutable producer field name retained so canonical archived
        # generation receipts can be read and verified byte-for-byte.
        "final25_code_hash": (
            final25_code_hash
            if final25_code_hash is not None
            else os.environ.get("TRACE_EVAL_CODE_HASH", "unknown")
        ),
        "selection": _selection_contract(args),
    }
    return hashlib.sha256(
        json.dumps(
            material, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _text_prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    """Persist prompt text for provenance without retaining encoded media."""

    chunks: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            if content:
                chunks.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text") or "")
                if text:
                    chunks.append(text)
    return "\n".join(chunks)


def _call_endpoint(
    args: argparse.Namespace, endpoint: str, messages: list[dict[str, Any]]
) -> dict[str, Any]:
    payload = _api_payload(args, messages)
    headers = {"Authorization": f"Bearer {args.api_key}"}
    try:
        response = _http_session().post(
            _completion_url(endpoint),
            json=payload,
            headers=headers,
            timeout=float(args.api_timeout),
        )
    except requests.RequestException as exc:
        raise RetryableAPIError(endpoint, repr(exc)) from exc

    detail = response.text[:2000].strip()
    if 400 <= response.status_code < 500 and response.status_code != 429:
        raise PermanentAPIError(endpoint, response.status_code, detail)
    if response.status_code >= 400:
        raise RetryableAPIError(
            endpoint,
            f"HTTP {response.status_code}: {detail}",
            affects_health=response.status_code != 429,
        )
    try:
        return response.json()
    except ValueError as exc:
        raise RetryableAPIError(endpoint, f"invalid JSON response: {detail}") from exc


def _result_from_response(
    job: RowJob,
    response: dict[str, Any],
    endpoint: str,
    request_hash: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    generation_contract_hash: str | None = None,
    model_revision: str | None = None,
    media_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    choice = response.get("choices", [{}])[0]
    message = choice.get("message") or {}
    usage = response.get("usage") or {}
    text = message.get("content") or ""
    provenance = media_provenance or {}
    return {
        "index": str(job.row["index"]),
        "row_key": job.row_key,
        "prediction": text,
        "finish_reason": choice.get("finish_reason"),
        "output_token_count": usage.get("completion_tokens"),
        "prompt_token_count": usage.get("prompt_tokens"),
        "api_endpoint": endpoint,
        "rank": int(job.rank),
        "source_ordinal": int(
            job.source_ordinal if job.source_ordinal is not None else job.rank
        ),
        "source_row_hash": provenance.get("source_row_hash") or _row_hash(job.row),
        "source_record_sha256": provenance.get("source_record_sha256"),
        "media_set_sha256": provenance.get("media_set_sha256"),
        "ordered_media_sha256": provenance.get("ordered_media_sha256") or [],
        "dataset_snapshot_sha256": provenance.get("dataset_snapshot_sha256"),
        "benchmark_key": job.spec.key,
        "request_hash": request_hash,
        "prompt": _text_prompt_from_messages(messages or []),
        "usage": usage,
        "generation_contract_hash": generation_contract_hash,
        "generation_contract_version": GENERATION_CONTRACT_VERSION,
        "media_contract_version": MEDIA_CONTRACT_VERSION,
        "model_revision": model_revision,
    }


def _load_existing_result_paths(output_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in (output_dir / "api_row_results").glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = row.get("row_key")
        if key is not None and not row.get("error"):
            out[str(key)] = path
    return out


def _manifest_row_provenance(
    handle: DatasetHandle,
    row: dict[str, Any],
    source_ordinal: int,
) -> dict[str, Any] | None:
    if handle.media_rows is None:
        return None
    source_row_hash = _row_hash(row)
    record = handle.media_rows.get(
        row_lookup_key(source_ordinal, row.get("index"), source_row_hash)
    )
    if record is None:
        raise RuntimeError(
            f"prepared media manifest has no current row for {handle.spec.key}/{row.get('index')}"
        )
    ordered_media = []
    for media in record.get("media") or []:
        path = Path(str(media.get("path") or "")).resolve()
        if not path.is_file() or path.stat().st_size != int(
            media.get("size_bytes") or -1
        ):
            raise RuntimeError(f"prepared media file is missing or changed: {path}")
        current_sha256 = _sha256_file(path)
        if current_sha256 != str(media.get("sha256") or ""):
            raise RuntimeError(
                f"prepared media content changed; rerun dataset verification: {path}"
            )
        ordered_media.append(
            {"type": str(media.get("type") or "image"), "sha256": current_sha256}
        )
    media_hash = media_set_sha256(ordered_media)
    if media_hash != str(record.get("media_set_sha256") or ""):
        raise RuntimeError(
            f"prepared media hash list is corrupt for {handle.spec.key}/{row.get('index')}"
        )
    record_hash = source_record_sha256(source_row_hash, media_hash)
    if record_hash != str(record.get("source_record_sha256") or ""):
        raise RuntimeError(
            f"prepared source record is stale for {handle.spec.key}/{row.get('index')}"
        )
    return {
        "source_row_hash": source_row_hash,
        "source_record_sha256": record_hash,
        "media_set_sha256": media_hash,
        "ordered_media_sha256": [item["sha256"] for item in ordered_media],
    }


def _matches_int(value: Any, expected: int) -> bool:
    try:
        return int(value) == int(expected)
    except (TypeError, ValueError):
        return False


def _prune_stale_row_results(
    args: argparse.Namespace,
    output_dir: Path,
    row_records: list[dict[str, Any]],
    handle: DatasetHandle | None = None,
) -> int:
    """Remove successful/error artifacts that no longer match current source rows."""

    current_rows: dict[str, tuple[dict[str, Any], int]] = {}
    for rank, row in enumerate(row_records):
        source_ordinal = (
            int(handle.source_ordinals[rank])
            if handle is not None and handle.source_ordinals is not None
            else rank
        )
        current_rows[_row_key(row, rank)] = (row, source_ordinal)
    expected_contract_hash = _generation_contract_hash(args)
    expected_model_revision = _configured_model_revision(args)
    removed = 0
    for path in (output_dir / "api_row_results").glob("*.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        row_key = str(result.get("row_key") or "")
        source_entry = current_rows.get(row_key)
        source_row = source_entry[0] if source_entry is not None else None
        source_ordinal = source_entry[1] if source_entry is not None else -1
        expected_provenance = (
            _manifest_row_provenance(handle, source_row, source_ordinal)
            if handle is not None and source_row is not None
            else None
        )
        keep = bool(
            source_row is not None
            and not result.get("error")
            and _matches_int(result.get("source_ordinal"), source_ordinal)
            and str(result.get("source_row_hash") or "") == _row_hash(source_row)
            and str(result.get("request_hash") or "").strip()
            and str(result.get("generation_contract_version") or "")
            == GENERATION_CONTRACT_VERSION
            and str(result.get("media_contract_version") or "")
            == MEDIA_CONTRACT_VERSION
            and str(result.get("generation_contract_hash") or "")
            == expected_contract_hash
            and str(result.get("model_revision") or "") == expected_model_revision
            and (
                expected_provenance is None
                or (
                    str(result.get("source_record_sha256") or "")
                    == expected_provenance["source_record_sha256"]
                    and str(result.get("media_set_sha256") or "")
                    == expected_provenance["media_set_sha256"]
                    and list(result.get("ordered_media_sha256") or [])
                    == expected_provenance["ordered_media_sha256"]
                )
            )
            and (
                not str(getattr(args, "dataset_snapshot_sha256", "") or "")
                or str(result.get("dataset_snapshot_sha256") or "")
                == str(args.dataset_snapshot_sha256)
            )
        )
        if not keep:
            path.unlink()
            removed += 1
    return removed


def _restore_mirrored_row_results(
    row_result_dir: Path, mirror_row_result_dir: Path | None
) -> None:
    """Restore missing canonical row files from a persistent result mirror."""

    if mirror_row_result_dir is None or not mirror_row_result_dir.exists():
        return
    row_result_dir.mkdir(parents=True, exist_ok=True)
    restored = 0
    for source in mirror_row_result_dir.glob("*.json"):
        target = row_result_dir / source.name
        if target.exists():
            continue
        shutil.copy2(source, target)
        restored += 1
    if restored:
        print(f"[api-generate:restore] rows={restored} source={mirror_row_result_dir}")


def _prepare_handles_and_jobs(
    args: argparse.Namespace, specs: list[BenchmarkSpec]
) -> tuple[dict[str, DatasetHandle], list[RowJob]]:
    dataset_manifest = _load_generation_dataset_manifest(args)

    handles: dict[str, DatasetHandle] = {}
    jobs_by_spec: list[list[RowJob]] = []
    for spec in specs:
        media_rows, dataset_snapshot = _media_rows_for_spec(dataset_manifest, spec)
        output_dir = run_dir(spec, args.model_slug, args.run_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        row_result_dir = output_dir / "api_row_results"
        mirror_output_dir = (
            run_dir(spec, args.model_slug, args.result_mirror_root)
            if args.result_mirror_root is not None
            else None
        )
        mirror_row_result_dir = (
            mirror_output_dir / "api_row_results"
            if mirror_output_dir is not None
            else None
        )
        if args.no_resume and row_result_dir.exists():
            shutil.rmtree(row_result_dir)
        if (
            args.no_resume
            and mirror_row_result_dir is not None
            and mirror_row_result_dir.exists()
        ):
            shutil.rmtree(mirror_row_result_dir)
        row_result_dir.mkdir(parents=True, exist_ok=True)
        _restore_mirrored_row_results(row_result_dir, mirror_row_result_dir)

        dataset = build_vlmeval_dataset(spec)
        dataset.data = dataset.data.copy()
        source_ordinals = list(range(len(dataset.data)))
        handle = DatasetHandle(
            spec=spec,
            dataset=dataset,
            media_rows=media_rows,
            dataset_snapshot_sha256=dataset_snapshot,
            source_ordinals=source_ordinals,
        )
        row_records = [row.to_dict() for _, row in dataset.data.iterrows()]
        handles[spec.key] = handle

        pruned = _prune_stale_row_results(args, output_dir, row_records, handle)
        existing = {} if args.no_resume else _load_existing_result_paths(output_dir)
        pending = 0
        spec_jobs: list[RowJob] = []
        for rank, row in enumerate(row_records):
            row_key = _row_key(row, rank)
            if row_key in existing:
                continue
            spec_jobs.append(
                RowJob(
                    spec=spec,
                    row=row,
                    rank=rank,
                    row_key=row_key,
                    output_dir=output_dir,
                    result_path=row_result_dir / _safe_result_name_for_row(row, rank),
                    source_ordinal=source_ordinals[rank],
                    mirror_result_path=(
                        mirror_row_result_dir / _safe_result_name_for_row(row, rank)
                        if mirror_row_result_dir is not None
                        else None
                    ),
                )
            )
            pending += 1
        jobs_by_spec.append(spec_jobs)
        print(
            "[api-generate:prepare] "
            f"{spec.key} rows={len(row_records)} existing={len(set(existing.values()))} "
            f"pending={pending} pruned={pruned} output={output_dir}"
        )
    jobs: list[RowJob] = []
    max_len = max((len(spec_jobs) for spec_jobs in jobs_by_spec), default=0)
    for offset in range(max_len):
        for spec_jobs in jobs_by_spec:
            if offset < len(spec_jobs):
                jobs.append(spec_jobs[offset])
    return handles, jobs


def _row_error(
    job: RowJob, error: BaseException, endpoint: str | None = None
) -> dict[str, Any]:
    return {
        "benchmark_key": job.spec.key,
        "index": str(job.row.get("index")),
        "row_key": job.row_key,
        "error": repr(error),
        "endpoint": endpoint,
        "result_path": str(job.result_path),
    }


def _put_persistence_job(
    persistence_jobs: "queue.Queue[PersistenceJob]",
    item: PersistenceJob,
    metrics: PipelineMetrics,
) -> None:
    persistence_jobs.put(item)
    metrics.observe_queue("persistence", persistence_jobs.qsize())


def _preparation_worker_loop(
    *,
    args: argparse.Namespace,
    handles: dict[str, DatasetHandle],
    jobs: "queue.Queue[RowJob]",
    prepared_jobs: "queue.Queue[PreparedJob]",
    persistence_jobs: "queue.Queue[PersistenceJob]",
    metrics: PipelineMetrics,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            job = jobs.get(timeout=0.25)
        except queue.Empty:
            continue
        started = time.monotonic()
        try:
            handle = handles[job.spec.key]
            messages = _vlmeval_messages(args, handle, job.row)
            provenance = _media_provenance(
                args,
                handle,
                job.row,
                messages,
                int(job.source_ordinal if job.source_ordinal is not None else job.rank),
            )
            prepared_jobs.put(
                PreparedJob(
                    job=job,
                    messages=messages,
                    request_hash=_request_hash(args, messages),
                    media_provenance=provenance,
                )
            )
            metrics.observe_queue("prepared", prepared_jobs.qsize())
        except Exception as exc:
            error = _row_error(job, exc)
            _put_persistence_job(
                persistence_jobs,
                PersistenceJob(
                    job=job,
                    payload={**error, "prediction": "", "finish_reason": "error"},
                    error=error,
                ),
                metrics,
            )
            metrics.increment("preparation_errors")
        finally:
            metrics.observe_stage("preparation", time.monotonic() - started)
            jobs.task_done()


def _request_worker_loop(
    *,
    args: argparse.Namespace,
    prepared_jobs: "queue.Queue[PreparedJob]",
    persistence_jobs: "queue.Queue[PersistenceJob]",
    endpoint_router: EndpointRouter,
    endpoint_health: EndpointHealth,
    metrics: PipelineMetrics,
    stop_event: threading.Event,
) -> None:
    try:
        while not stop_event.is_set():
            try:
                prepared = prepared_jobs.get(timeout=0.25)
            except queue.Empty:
                continue
            job = prepared.job
            try:
                while True:
                    endpoint = endpoint_router.acquire(job.attempted_endpoints)
                    if endpoint is None:
                        if stop_event.is_set():
                            break
                        continue
                    started = time.monotonic()
                    try:
                        response = _call_endpoint(args, endpoint, prepared.messages)
                        endpoint_health.record_success(endpoint)
                        result = _result_from_response(
                            job,
                            response,
                            endpoint,
                            prepared.request_hash,
                            prepared.messages,
                            _generation_contract_hash(args),
                            _configured_model_revision(args),
                            prepared.media_provenance,
                        )
                        _put_persistence_job(
                            persistence_jobs,
                            PersistenceJob(job=job, payload=result),
                            metrics,
                        )
                        metrics.increment("request_successes")
                        break
                    except PermanentAPIError as exc:
                        error = _row_error(job, exc, endpoint)
                        _put_persistence_job(
                            persistence_jobs,
                            PersistenceJob(
                                job=job,
                                payload={
                                    **error,
                                    "prediction": "",
                                    "finish_reason": "error",
                                },
                                error=error,
                            ),
                            metrics,
                        )
                        metrics.increment("permanent_api_errors")
                        break
                    except RetryableAPIError as exc:
                        disabled = (
                            endpoint_health.record_failure(endpoint)
                            if exc.affects_health
                            else False
                        )
                        if disabled:
                            print(
                                f"[api-generate:endpoint-disabled] endpoint={endpoint} reason={exc.detail}"
                            )
                        job = replace(
                            job,
                            attempted_endpoints=tuple(
                                dict.fromkeys((*job.attempted_endpoints, endpoint))
                            ),
                            attempt_count=job.attempt_count + 1,
                        )
                        metrics.increment("request_retries")
                        if job.attempt_count >= int(args.api_max_retries):
                            error = _row_error(job, exc, endpoint)
                            _put_persistence_job(
                                persistence_jobs,
                                PersistenceJob(
                                    job=job,
                                    payload={
                                        **error,
                                        "prediction": "",
                                        "finish_reason": "error",
                                    },
                                    error=error,
                                ),
                                metrics,
                            )
                            metrics.increment("retry_exhausted_errors")
                            break
                        time.sleep(min(5.0, 0.25 * job.attempt_count))
                    except Exception as exc:
                        error = _row_error(job, exc, endpoint)
                        _put_persistence_job(
                            persistence_jobs,
                            PersistenceJob(
                                job=job,
                                payload={
                                    **error,
                                    "prediction": "",
                                    "finish_reason": "error",
                                },
                                error=error,
                            ),
                            metrics,
                        )
                        metrics.increment("unexpected_request_errors")
                        break
                    finally:
                        metrics.observe_stage("request", time.monotonic() - started)
                        endpoint_router.release(endpoint)
            finally:
                prepared_jobs.task_done()
    finally:
        _close_http_session()


def _persistence_worker_loop(
    *,
    jobs: "queue.Queue[PersistenceJob]",
    errors: list[dict[str, Any]],
    error_lock: threading.Lock,
    progress: tqdm,
    metrics: PipelineMetrics,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            item = jobs.get(timeout=0.25)
        except queue.Empty:
            continue
        started = time.monotonic()
        try:
            persisted = _persist_worker_result(
                item.job, item.payload, errors, error_lock
            )
            if item.error is not None:
                with error_lock:
                    errors.append(item.error)
            metrics.increment("persisted_rows" if persisted else "persistence_errors")
        except Exception as exc:
            error = _row_error(
                item.job, RuntimeError(f"result persistence failed: {exc!r}")
            )
            with error_lock:
                errors.append(error)
            metrics.increment("persistence_errors")
            print(
                f"[api-generate:persistence-error] row_key={item.job.row_key} error={exc!r}"
            )
        finally:
            metrics.observe_stage("persistence", time.monotonic() - started)
            progress.update(1)
            jobs.task_done()


def _prediction_map_from_row_results(output_dir: Path) -> dict[str, dict[str, Any]]:
    pred_map: dict[str, dict[str, Any]] = {}
    for path in sorted((output_dir / "api_row_results").glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = row.get("row_key")
        if key is None and row.get("index") is not None:
            key = str(row["index"])
        if key is not None and not row.get("error"):
            pred_map[str(key)] = row
            identity = _row_identity_from_key(row.get("index"), row.get("row_key"))
            if identity is not None:
                pred_map[identity] = row
    return pred_map


def _prediction_for_row(
    pred_map: dict[str, dict[str, Any]], row: dict[str, Any], rank: int
) -> dict[str, Any]:
    return pred_map.get(
        _row_key(row, rank),
        pred_map.get(_row_identity(row), pred_map.get(str(row["index"]), {})),
    )


def _current_prediction_results(
    pred_map: dict[str, dict[str, Any]], row_records: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Select at most one persisted result for each row in the active dataset."""

    current: dict[str, dict[str, Any]] = {}
    for rank, row in enumerate(row_records):
        prediction = _prediction_for_row(pred_map, row, rank)
        if prediction:
            current[_row_key(row, rank)] = prediction
    return current


def _token_stats(pred_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    vals = sorted(int(v.get("output_token_count") or 0) for v in pred_map.values())
    if not vals:
        return {"mean": 0, "median": 0, "max": 0, "length_cap_fraction": 0}
    finish = Counter(str(v.get("finish_reason")) for v in pred_map.values())
    n = len(vals)
    median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    return {
        "mean": sum(vals) / n,
        "median": median,
        "max": max(vals),
        "length_cap_fraction": finish.get("length", 0) / n,
    }


def _archive_generation_slice(
    args: argparse.Namespace,
    handle: DatasetHandle,
    row_records: list[dict[str, Any]],
    pred_map: dict[str, dict[str, Any]],
    summary: dict[str, Any],
) -> Path | None:
    if not os.environ.get("TRACE_EVAL_HF_SPOOL_ROOT", "").strip():
        return None
    from trace_eval_archive_hooks import emit_generation_slice, resolve_model_source

    spec = handle.spec
    sampling = {
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k) if args.top_k is not None else None,
        "presence_penalty": float(args.presence_penalty),
        "repetition_penalty": (
            float(args.repetition_penalty)
            if args.repetition_penalty is not None
            else None
        ),
        "max_tokens": int(args.max_tokens),
        "seed": int(args.seed) if args.seed is not None else None,
    }
    records: list[dict[str, Any]] = []
    for rank, row in enumerate(row_records):
        result = _prediction_for_row(pred_map, row, rank)
        request_hash = str(result.get("request_hash") or "").strip()
        if not request_hash:
            raise RuntimeError(
                f"generation row {spec.key}/{row.get('index', rank)} has no request_hash; "
                "stale row artifacts must be regenerated"
            )
        options = {
            key: value
            for key, value in row.items()
            if isinstance(key, str)
            and len(key) == 1
            and "A" <= key <= "Z"
            and value is not None
        }
        records.append(
            {
                "source_index": str(row.get("index", rank)),
                "source_ordinal": int(result.get("source_ordinal", rank)),
                "source_row_hash": str(result.get("source_row_hash") or _row_hash(row)),
                "request_hash": request_hash,
                "question": row.get("question"),
                "ground_truth": row.get("answer"),
                "options": options,
                "source_row": row,
                "prompt": result.get("prompt") or row.get("question") or "",
                "model_response": result.get("prediction", ""),
                "sampling": sampling,
                "finish_reason": result.get("finish_reason"),
                "usage": result.get("usage")
                or {
                    "prompt_tokens": result.get("prompt_token_count"),
                    "completion_tokens": result.get("output_token_count"),
                },
                "metadata": {
                    "source_record_sha256": result.get("source_record_sha256"),
                    "media_hash": result.get("media_set_sha256"),
                    "image_hash": result.get("ordered_media_sha256") or [],
                    "image_count": len(result.get("ordered_media_sha256") or []),
                    "dataset_snapshot_sha256": result.get("dataset_snapshot_sha256"),
                    "transport_contract_version": result.get("media_contract_version"),
                },
            }
        )
    return emit_generation_slice(
        records=records,
        model=resolve_model_source(str(args.model_slug), str(args.model)),
        model_slug=str(args.model_slug),
        model_revision=_configured_model_revision(args),
        seed=int(args.seed if args.seed is not None else 0),
        benchmark=spec.key,
        dataset_alias=spec.alias,
        dataset_split=spec.split or "default",
        dataset_revision=os.environ.get(
            "TRACE_EVAL_DATASET_REVISION",
            os.environ.get("TRACE_VLMEVALKIT_GIT_COMMIT", "unknown"),
        ),
        contract_version=GENERATION_CONTRACT_VERSION,
        aggregate=summary,
    )


def _finalize_spec(args: argparse.Namespace, handle: DatasetHandle) -> dict[str, Any]:
    spec = handle.spec
    output_dir = run_dir(spec, args.model_slug, args.run_root)
    pred_map = _prediction_map_from_row_results(output_dir)
    row_records = [row.to_dict() for _, row in handle.dataset.data.iterrows()]
    unique_predictions = _current_prediction_results(pred_map, row_records)
    pred_jsonl = output_dir / "predictions.jsonl"
    records = []
    with pred_jsonl.open("w", encoding="utf-8") as handle_jsonl:
        for rank, row_dict in enumerate(row_records):
            pred = _prediction_for_row(pred_map, row_dict, rank)
            record = {
                "index": str(row_dict["index"]),
                "prediction": pred.get("prediction", ""),
                "finish_reason": pred.get("finish_reason", ""),
                "output_token_count": pred.get("output_token_count"),
                "prompt_token_count": pred.get("prompt_token_count"),
                "request_hash": pred.get("request_hash"),
                "source_ordinal": int(pred.get("source_ordinal", rank)),
                "source_row_hash": pred.get("source_row_hash") or _row_hash(row_dict),
                "prompt": pred.get("prompt", ""),
            }
            persisted_row = {**row_dict, **record}
            if getattr(args, "compact_prediction_tables", False):
                persisted_row = _compact_prediction_record(persisted_row)
            records.append(persisted_row)
            handle_jsonl.write(
                json.dumps(record, ensure_ascii=False, default=json_default) + "\n"
            )
    frame = pd.DataFrame(records)
    eval_file = output_dir / f"{spec.alias}_predictions.xlsx"
    frame.to_excel(eval_file, index=False)
    table_jsonl = output_dir / f"{spec.alias}_predictions_table.jsonl"
    with table_jsonl.open("w", encoding="utf-8") as handle_jsonl:
        for record in records:
            handle_jsonl.write(
                json.dumps(record, ensure_ascii=False, default=json_default) + "\n"
            )
    expected_rows = len(handle.dataset.data)

    finish_reasons = Counter(
        str(v.get("finish_reason")) for v in unique_predictions.values()
    )
    summary = {
        "dataset": spec.alias,
        "display": spec.display,
        "model": args.model,
        "model_slug": args.model_slug,
        "rows": len(unique_predictions),
        "expected_rows": expected_rows,
        "generation": {
            "contract_version": GENERATION_CONTRACT_VERSION,
            "contract_hash": _generation_contract_hash(args),
            "media_contract_version": MEDIA_CONTRACT_VERSION,
            "media_transport": MEDIA_TRANSPORT,
            "dataset_snapshot_sha256": str(
                getattr(args, "dataset_snapshot_sha256", "") or ""
            ),
            "dataset_revision": os.environ.get(
                "TRACE_EVAL_DATASET_REVISION", "unknown"
            ),
            "model_revision": _configured_model_revision(args),
            "selection": _selection_contract(args),
            "backend": "openai_compatible_vllm_endpoint_pool",
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k) if args.top_k is not None else None,
            "presence_penalty": float(args.presence_penalty),
            "repetition_penalty": (
                float(args.repetition_penalty)
                if args.repetition_penalty is not None
                else None
            ),
            "max_tokens": int(args.max_tokens),
            "seed": int(args.seed) if args.seed is not None else None,
            "api_model": args.api_model,
            "endpoint_count": len(args.api_bases),
            "parallelism_per_endpoint": int(args.parallelism_per_endpoint),
            "min_image_pixels": int(
                getattr(args, "min_image_pixels", QWEN_MIN_IMAGE_PIXELS)
            ),
            "max_image_pixels": int(args.max_image_pixels),
            "max_image_side": None,
            "image_jpeg_quality": None,
            "compact_prediction_tables": bool(
                getattr(args, "compact_prediction_tables", False)
            ),
            "preparation_workers": int(getattr(args, "preparation_workers", 32)),
            "queue_capacity": int(getattr(args, "queue_capacity", 256)),
            "persistence_workers": int(getattr(args, "persistence_workers", 16)),
            "finalization_workers": int(getattr(args, "finalization_workers", 4)),
        },
        "finish_reason": dict(finish_reasons),
        "output_token_stats": _token_stats(unique_predictions),
        "artifacts": {
            "predictions_jsonl": str(pred_jsonl),
            "eval_file": str(eval_file),
        },
    }
    if len(unique_predictions) != expected_rows:
        summary["warning"] = (
            f"missing {expected_rows - len(unique_predictions)} predictions"
        )
    write_json(output_dir / "generation_summary.json", summary)
    if len(unique_predictions) == expected_rows:
        archive_path = _archive_generation_slice(
            args, handle, row_records, pred_map, summary
        )
        if archive_path is not None:
            summary["archive_descriptor"] = str(archive_path)
            write_json(output_dir / "generation_summary.json", summary)
    print(
        "[api-generate:finalize] "
        f"{spec.key} rows={len(unique_predictions)}/{expected_rows} "
        f"mean_tokens={summary['output_token_stats']['mean']:.1f} "
        f"cap_hit={summary['output_token_stats']['length_cap_fraction']:.3f}"
    )
    return summary


def run(args: argparse.Namespace) -> None:
    _resolve_image_pixel_defaults(args)
    specs = benchmark_specs_for_run_set(args.run_set, model_slug=args.model_slug)
    if args.exact_only and args.only:
        keep = {str(key) for key in args.only}
        specs = [spec for spec in specs if spec.key in keep]
        specs = filter_benchmark_specs(specs, exclude=args.exclude)
    else:
        specs = filter_benchmark_specs(specs, only=args.only, exclude=args.exclude)
    if not specs:
        raise ValueError("No benchmark specs selected")
    if not getattr(args, "finalize_only", False) and not args.api_bases:
        raise ValueError("At least one API endpoint is required")
    if int(args.parallelism_per_endpoint) < 1:
        raise ValueError("--parallelism-per-endpoint must be at least 1")
    if getattr(args, "allowed_local_media_path", None) is None:
        raise ValueError(
            "--allowed-local-media-path is required for file-url media transport"
        )
    if int(getattr(args, "min_image_pixels", 0)) <= 0:
        raise ValueError("--min-image-pixels must be positive")
    if int(args.max_image_pixels) < int(args.min_image_pixels):
        raise ValueError("--max-image-pixels must be at least --min-image-pixels")
    handles, pending = _prepare_handles_and_jobs(args, specs)
    metrics = PipelineMetrics()
    args._pipeline_metrics = metrics
    queue_capacity = max(1, int(getattr(args, "queue_capacity", 256)))
    preparation_workers = max(1, int(getattr(args, "preparation_workers", 32)))
    persistence_workers = max(1, int(getattr(args, "persistence_workers", 16)))
    finalization_workers = max(1, int(getattr(args, "finalization_workers", 4)))
    if getattr(args, "finalize_only", False):
        if pending:
            raise RuntimeError(
                f"cannot finalize {args.model_slug}: {len(pending)} generation rows are missing or stale"
            )

        def finalize_existing(handle: DatasetHandle) -> dict[str, Any]:
            started = time.monotonic()
            try:
                return _finalize_spec(args, handle)
            finally:
                metrics.observe_stage("finalization", time.monotonic() - started)

        with ThreadPoolExecutor(
            max_workers=min(finalization_workers, len(specs))
        ) as executor:
            summaries = [
                future.result()
                for future in [
                    executor.submit(finalize_existing, handles[spec.key])
                    for spec in specs
                ]
            ]
        pipeline_metrics = metrics.snapshot()
        write_json(
            args.run_root / f"{args.model_slug}_api_generation_suite_summary.json",
            {
                "model": args.model,
                "model_slug": args.model_slug,
                "run_root": str(args.run_root),
                "benchmarks": summaries,
                "errors": [],
                "error_count": 0,
                "endpoint_health": {},
                "pipeline_metrics": pipeline_metrics,
                "mode": "finalize_only",
            },
        )
        print(
            f"[api-generate:finalize-only] slug={args.model_slug} benchmarks={len(summaries)} "
            f"elapsed={pipeline_metrics['elapsed_seconds']:.1f}s"
        )
        return
    input_jobs: queue.Queue[RowJob] = queue.Queue(maxsize=queue_capacity)
    prepared_jobs: queue.Queue[PreparedJob] = queue.Queue(maxsize=queue_capacity)
    persistence_jobs: queue.Queue[PersistenceJob] = queue.Queue(maxsize=queue_capacity)
    errors: list[dict[str, Any]] = []
    error_lock = threading.Lock()
    endpoint_health = EndpointHealth(args.api_bases, args.endpoint_failure_threshold)
    total_workers = len(args.api_bases) * int(args.parallelism_per_endpoint)
    endpoint_router = EndpointRouter(
        args.api_bases, args.parallelism_per_endpoint, endpoint_health
    )
    print(
        "[api-generate:start] "
        f"model={args.model} slug={args.model_slug} specs={len(specs)} pending_rows={len(pending)} "
        f"endpoints={len(args.api_bases)} request_workers={total_workers} prep_workers={preparation_workers} "
        f"write_workers={persistence_workers} queue_capacity={queue_capacity} run_root={args.run_root}"
    )
    with tqdm(total=len(pending), desc=f"{args.model_slug} api generate") as progress:
        preparation_stop = threading.Event()
        request_stop = threading.Event()
        persistence_stop = threading.Event()
        preparation_threads = [
            threading.Thread(
                target=_preparation_worker_loop,
                kwargs={
                    "args": args,
                    "handles": handles,
                    "jobs": input_jobs,
                    "prepared_jobs": prepared_jobs,
                    "persistence_jobs": persistence_jobs,
                    "metrics": metrics,
                    "stop_event": preparation_stop,
                },
                name=f"api-prepare-{worker}",
                daemon=True,
            )
            for worker in range(preparation_workers)
        ]
        request_threads = [
            threading.Thread(
                target=_request_worker_loop,
                kwargs={
                    "args": args,
                    "prepared_jobs": prepared_jobs,
                    "persistence_jobs": persistence_jobs,
                    "endpoint_router": endpoint_router,
                    "endpoint_health": endpoint_health,
                    "metrics": metrics,
                    "stop_event": request_stop,
                },
                name=f"api-request-{worker}",
                daemon=True,
            )
            for worker in range(total_workers)
        ]
        persistence_threads = [
            threading.Thread(
                target=_persistence_worker_loop,
                kwargs={
                    "jobs": persistence_jobs,
                    "errors": errors,
                    "error_lock": error_lock,
                    "progress": progress,
                    "metrics": metrics,
                    "stop_event": persistence_stop,
                },
                name=f"api-persist-{worker}",
                daemon=True,
            )
            for worker in range(persistence_workers)
        ]
        for thread in (*persistence_threads, *request_threads, *preparation_threads):
            thread.start()
        for job in pending:
            input_jobs.put(job)
            metrics.observe_queue("input", input_jobs.qsize())

        input_jobs.join()
        preparation_stop.set()
        for thread in preparation_threads:
            thread.join(timeout=5)
        prepared_jobs.join()
        request_stop.set()
        for thread in request_threads:
            thread.join(timeout=5)
        persistence_jobs.join()
        persistence_stop.set()
        for thread in persistence_threads:
            thread.join(timeout=5)

    if getattr(args, "defer_finalization", False):
        pipeline_metrics = metrics.snapshot()
        suite_summary = {
            "model": args.model,
            "model_slug": args.model_slug,
            "run_root": str(args.run_root),
            "benchmarks": [],
            "errors": errors[:100],
            "error_count": len(errors),
            "endpoint_health": endpoint_health.snapshot(),
            "pipeline_metrics": pipeline_metrics,
            "mode": "finalization_deferred",
        }
        write_json(
            args.run_root / f"{args.model_slug}_api_generation_suite_summary.json",
            suite_summary,
        )
        if errors:
            raise RuntimeError(
                f"{len(errors)} row generation errors; first error: {errors[0]}"
            )
        print(
            f"[api-generate:deferred] slug={args.model_slug} durable_rows={len(pending)} "
            f"elapsed={pipeline_metrics['elapsed_seconds']:.1f}s"
        )
        return

    def finalize(handle: DatasetHandle) -> dict[str, Any]:
        started = time.monotonic()
        try:
            return _finalize_spec(args, handle)
        finally:
            metrics.observe_stage("finalization", time.monotonic() - started)

    with ThreadPoolExecutor(
        max_workers=min(finalization_workers, len(specs))
    ) as executor:
        futures = [executor.submit(finalize, handles[spec.key]) for spec in specs]
        summaries = [future.result() for future in futures]
    pipeline_metrics = metrics.snapshot()
    suite_summary = {
        "model": args.model,
        "model_slug": args.model_slug,
        "run_root": str(args.run_root),
        "benchmarks": summaries,
        "errors": errors[:100],
        "error_count": len(errors),
        "endpoint_health": endpoint_health.snapshot(),
        "pipeline_metrics": pipeline_metrics,
    }
    write_json(
        args.run_root / f"{args.model_slug}_api_generation_suite_summary.json",
        suite_summary,
    )
    print(
        "[api-generate:metrics] "
        f"elapsed={pipeline_metrics['elapsed_seconds']:.1f}s "
        f"queues={pipeline_metrics['queue_high_watermarks']} counters={pipeline_metrics['counters']}"
    )
    if errors:
        raise RuntimeError(
            f"{len(errors)} row generation errors; first error: {errors[0]}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--api-model", required=True)
    parser.add_argument("--api-base", action="append", dest="api_bases", default=[])
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--api-timeout", type=float, default=300.0)
    parser.add_argument(
        "--api-max-retries",
        type=int,
        default=5,
        help="Maximum total attempts for a row across the endpoint pool; permanent HTTP 4xx errors are not retried.",
    )
    parser.add_argument(
        "--endpoint-failure-threshold",
        type=int,
        default=2,
        help="Consecutive transport/server failures before an endpoint is quarantined for the remainder of the run.",
    )
    parser.add_argument("--parallelism-per-endpoint", type=int, default=4)
    parser.add_argument(
        "--preparation-workers",
        type=int,
        default=32,
        help="Threads that build prompts and verify media ahead of endpoint requests.",
    )
    parser.add_argument(
        "--queue-capacity",
        type=int,
        default=256,
        help="Maximum buffered rows between each preparation, request, and persistence stage.",
    )
    parser.add_argument(
        "--persistence-workers",
        type=int,
        default=16,
        help="Threads dedicated to atomic row-result and mirror writes.",
    )
    parser.add_argument(
        "--finalization-workers",
        type=int,
        default=4,
        help="Benchmarks finalized concurrently after all row results are durable.",
    )
    parser.add_argument(
        "--media-transport",
        choices=(MEDIA_TRANSPORT,),
        default=MEDIA_TRANSPORT,
        help="Use exact local file URLs.",
    )
    parser.add_argument(
        "--allowed-local-media-path",
        type=Path,
        default=None,
        help="Root shared with vLLM --allowed-local-media-path for file-url requests.",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=None,
        help="Prepared trace_eval_v1 dataset manifest containing ordered media hashes.",
    )
    parser.add_argument(
        "--dataset-manifest-view",
        choices=("trace_eval_v1",),
        default="trace_eval_v1",
    )
    parser.add_argument(
        "--run-set",
        choices=BENCHMARK_RUN_SETS,
        default="trace_eval_v1",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=LIB_REPO_ROOT / "rlvr" / "evaluation" / ".work" / "runs",
    )
    parser.add_argument(
        "--result-mirror-root",
        type=Path,
        default=None,
        help="Optional persistent run root that mirrors every atomic per-row result and restores missing rows on resume.",
    )
    parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="Interpret --only as exact BenchmarkSpec.key values, not aliases or aggregate groups.",
    )
    parser.add_argument("--min-image-pixels", type=int, default=QWEN_MIN_IMAGE_PIXELS)
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=None,
        help="Defaults to the canonical Qwen image-processor bound of 12,845,056.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--compact-prediction-tables",
        action="store_true",
        help="Exclude duplicated image payload columns from finalized prediction tables.",
    )
    parser.add_argument("--no-resume", action="store_true")
    finalization_mode = parser.add_mutually_exclusive_group()
    finalization_mode.add_argument(
        "--defer-finalization",
        action="store_true",
        help="Return after row results are durable so CPU finalization can run in a separate process.",
    )
    finalization_mode.add_argument(
        "--finalize-only",
        action="store_true",
        help="Finalize and archive complete row results without contacting a model endpoint.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _resolve_image_pixel_defaults(args)
    run(args)


if __name__ == "__main__":
    main()
