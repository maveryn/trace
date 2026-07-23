#!/usr/bin/env python3
"""Content-addressed, evaluation-independent trace_eval_v1 archival.

Evaluation workers only call :func:`emit_slice_ready`.  That function snapshots
sanitized records locally and atomically publishes a small descriptor.  A
separate :class:`ArchiveDaemon` turns descriptors into immutable Parquet slices,
uploads them, and records verified remote state in a SQLite ledger.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import signal
import sqlite3
import stat
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import RepositoryNotFoundError


SCHEMA_VERSION = "trace-final25-hf-archive-v1"
DEFAULT_REPO_ID = "maveryn/trace-eval-runs"
DEFAULT_REVISION = "main"
ALLOWED_STAGES = frozenset({"generation", "extraction", "score"})
REQUIRED_PROVENANCE = frozenset(
    {"trace_git_commit", "vlmevalkit_git_commit", "contract_version", "campaign_config_hash"}
)
COMMON_RECORD_KEYS = frozenset(
    {"source_index", "source_ordinal", "source_row_hash", "request_hash", "question", "ground_truth"}
)
STAGE_RECORD_KEYS = {
    "generation": frozenset(
        {"source_row", "prompt", "model_response", "sampling", "finish_reason", "usage"}
    ),
    "extraction": frozenset(
        {
            "model_response",
            "judge_prompt",
            "judge_response",
            "normalized_extraction",
            "retries",
        }
    ),
    "score": frozenset({"prediction", "score", "scorer", "excluded"}),
}
OPAQUE_TEXT_RECORD_KEYS = {
    "generation": frozenset(
        {"question", "ground_truth", "prompt", "model_response"}
    ),
    "extraction": frozenset(
        {
            "question",
            "ground_truth",
            "model_response",
            "judge_prompt",
            "judge_response",
        }
    ),
    "score": frozenset({"question", "ground_truth", "prediction"}),
}
OPAQUE_TEXT_RECORD_PATHS = {
    "generation": frozenset(),
    "extraction": frozenset(
        {
            ("retries", "events", "*", "prompt"),
            ("retries", "events", "*", "response"),
        }
    ),
    "score": frozenset(),
}
JSON_COLUMNS = (
    "ground_truth",
    "options",
    "source_row",
    "metadata",
    "sampling",
    "usage",
    "normalized_extraction",
    "retries",
    "prediction",
    "score",
    "excluded",
)
TEXT_COLUMNS = (
    "question",
    "prompt",
    "model_response",
    "finish_reason",
    "judge_prompt",
    "judge_response",
    "scorer",
)
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("descriptor_id", pa.string(), nullable=False),
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("stage", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("model", pa.string(), nullable=False),
        pa.field("model_slug", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("benchmark", pa.string(), nullable=False),
        pa.field("dataset_alias", pa.string(), nullable=False),
        pa.field("dataset_split", pa.string(), nullable=False),
        pa.field("dataset_revision", pa.string(), nullable=False),
        pa.field("source_index", pa.string(), nullable=False),
        pa.field("source_ordinal", pa.int64(), nullable=False),
        pa.field("source_row_hash", pa.string(), nullable=False),
        pa.field("request_hash", pa.string(), nullable=False),
        pa.field("provenance_json", pa.string(), nullable=False),
        pa.field("record_json", pa.string(), nullable=False),
        *[pa.field(key, pa.string()) for key in TEXT_COLUMNS],
        *[pa.field(f"{key}_json", pa.string()) for key in JSON_COLUMNS],
    ],
    metadata={b"trace_schema_version": SCHEMA_VERSION.encode("ascii")},
)
MEDIA_SUFFIXES = frozenset(
    {
        ".apng",
        ".avi",
        ".bmp",
        ".flac",
        ".gif",
        ".jpeg",
        ".jpg",
        ".m4a",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".ogg",
        ".png",
        ".tif",
        ".tiff",
        ".wav",
        ".webm",
        ".webp",
    }
)
SAFE_MEDIA_METADATA_KEYS = frozenset(
    {
        "frame_count",
        "image_count",
        "image_hash",
        "image_height",
        "image_jpeg_quality",
        "image_size",
        "image_width",
        "max_image_pixels",
        "max_image_side",
        "media_count",
        "media_contract_version",
        "media_hash",
        "media_transport",
        "min_image_pixels",
        "video_count",
        "video_hash",
    }
)
SECRET_KEY_PARTS = frozenset(
    {
        "access_token",
        "accesstoken",
        "api_key",
        "apikey",
        "auth_token",
        "authtoken",
        "authorization",
        "bearer_token",
        "bearertoken",
        "client_secret",
        "clientsecret",
        "credential",
        "credentials",
        "gh_token",
        "ghtoken",
        "github_token",
        "githubtoken",
        "hf_token",
        "hftoken",
        "id_token",
        "idtoken",
        "password",
        "private_key",
        "privatekey",
        "refresh_token",
        "refreshtoken",
        "secret",
        "token",
        "token_file",
        "tokenfile",
        "wandb_token",
        "wandbtoken",
    }
)


class ArchiveError(RuntimeError):
    """Base archive exception."""


class ArchiveValidationError(ArchiveError):
    """A producer supplied an unsafe or incomplete archive slice."""


class ArchiveIntegrityError(ArchiveError):
    """Immutable local or remote content did not match its recorded digest."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def redact_secret(message: Any, secret: str | None) -> str:
    text = str(message)
    return text.replace(secret, "<redacted>") if secret else text


def _slug(value: str, label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-.").lower()
    if not cleaned:
        raise ArchiveValidationError(f"{label} must contain a path-safe character")
    return cleaned


def _key_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _is_media_key(key: str) -> bool:
    if key in SAFE_MEDIA_METADATA_KEYS:
        return False
    tokens = set(key.split("_"))
    if tokens & {
        "audio",
        "image",
        "images",
        "media",
        "photo",
        "photos",
        "picture",
        "pictures",
        "screenshot",
        "screenshots",
        "thumbnail",
        "thumbnails",
        "video",
        "videos",
    }:
        return True
    return key.endswith(("_image_path", "_video_path", "_media_path"))


def _is_secret_key(key: str) -> bool:
    if key in SECRET_KEY_PARTS:
        return True
    return key.endswith(
        (
            "_access_token",
            "_api_key",
            "_auth_token",
            "_bearer_token",
            "_credential",
            "_credentials",
            "_password",
            "_private_key",
            "_secret",
            "_token",
            "_token_file",
        )
    )


def _is_media_path(value: str) -> bool:
    stripped = value.strip()
    lowered = stripped.lower()
    if re.search(r"data\s*:\s*(?:image|video|audio|application/octet-stream)[^,]*,", lowered):
        return True
    if lowered.startswith("file://"):
        return True
    without_query = lowered.split("?", 1)[0].split("#", 1)[0]
    suffix = Path(without_query).suffix
    if suffix not in MEDIA_SUFFIXES:
        return False
    return bool(
        stripped.startswith(("/", "~/", "./", "../", "\\\\"))
        or "/" in stripped
        or "\\" in stripped
        or re.fullmatch(r"[^\s/\\]+\.[A-Za-z0-9]{2,5}", stripped)
    )


def sanitize_archive_value(
    value: Any,
    *,
    path: str = "record",
    opaque_text_keys: frozenset[str] = frozenset(),
    opaque_text_paths: frozenset[tuple[str, ...]] = frozenset(),
) -> Any:
    """Return a JSON-safe value while rejecting media, paths, bytes, and secrets."""

    if isinstance(value, (bytes, bytearray, memoryview, Path)):
        raise ArchiveValidationError(f"binary/path value is forbidden at {path}")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        if _is_media_path(value):
            raise ArchiveValidationError(f"media path or data URL is forbidden at {path}")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = _key_name(key)
            if _is_secret_key(normalized):
                raise ArchiveValidationError(f"credential field is forbidden at {path}.{key}")
            if _is_media_key(normalized):
                # A null setting carries neither media nor a local path. This
                # lets fixed-shape configuration summaries retain disabled
                # media settings without weakening checks on populated fields.
                if child is None:
                    result[key] = None
                    continue
                raise ArchiveValidationError(f"media field is forbidden at {path}.{key}")
            child_opaque_paths = frozenset(
                candidate[1:]
                for candidate in opaque_text_paths
                if candidate and candidate[0] == normalized
            )
            if (
                normalized in opaque_text_keys
                or () in child_opaque_paths
            ) and isinstance(child, str):
                # Prompt and answer text may legitimately mention a local media
                # path or contain a data URL emitted by the model. Only an
                # explicitly declared scalar field receives this exemption;
                # media-named fields and nested/binary payloads remain strict.
                result[key] = child
                continue
            result[key] = sanitize_archive_value(
                child,
                path=f"{path}.{key}",
                opaque_text_paths=child_opaque_paths,
            )
        return result
    if isinstance(value, (list, tuple)):
        child_opaque_paths = frozenset(
            candidate[1:]
            for candidate in opaque_text_paths
            if candidate and candidate[0] == "*"
        )
        return [
            sanitize_archive_value(
                child,
                path=f"{path}[{index}]",
                opaque_text_paths=child_opaque_paths,
            )
            for index, child in enumerate(value)
        ]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    if value.__class__.__name__ == "NAType":
        return None
    if hasattr(value, "item"):
        try:
            return sanitize_archive_value(value.item(), path=path)
        except (TypeError, ValueError):
            pass
    raise ArchiveValidationError(f"unsupported value type at {path}: {type(value).__name__}")


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    _ensure_private_directory(path.parent)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def _require_nonempty(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ArchiveValidationError(f"{label} is required")
    return text


def stable_record_id(
    *,
    run_id: str,
    model_slug: str,
    model_revision: str,
    seed: int,
    benchmark: str,
    dataset_alias: str,
    dataset_split: str,
    dataset_revision: str,
    source_index: Any,
    source_ordinal: int,
    source_row_hash: str,
) -> str:
    identity = {
        "benchmark": benchmark,
        "dataset_alias": dataset_alias,
        "dataset_revision": dataset_revision,
        "dataset_split": dataset_split,
        "model_slug": model_slug,
        "model_revision": model_revision,
        "run_id": run_id,
        "seed": seed,
        "source_index": str(source_index),
        "source_ordinal": source_ordinal,
        "source_row_hash": source_row_hash,
    }
    return f"rec-{sha256_bytes(canonical_json(identity).encode('utf-8'))}"


def emit_slice_ready(
    spool_root: Path | str,
    *,
    stage: str,
    run_id: str,
    model: str,
    model_revision: str,
    seed: int,
    benchmark: str,
    dataset_alias: str,
    dataset_split: str,
    dataset_revision: str,
    records: Iterable[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    aggregate: Mapping[str, Any] | None = None,
    model_slug: str | None = None,
) -> Path:
    """Snapshot one completed model/seed/benchmark stage and publish a descriptor.

    The returned ``*.ready.json`` is the only synchronization primitive needed
    by evaluation code.  It appears only after the immutable JSONL payload has
    been flushed and renamed into place.
    """

    if stage not in ALLOWED_STAGES:
        raise ArchiveValidationError(f"unsupported archive stage: {stage!r}")
    run_id = _require_nonempty(run_id, "run_id")
    model = _require_nonempty(model, "model")
    model_revision = _require_nonempty(model_revision, "model_revision")
    benchmark = _require_nonempty(benchmark, "benchmark")
    dataset_alias = _require_nonempty(dataset_alias, "dataset_alias")
    dataset_split = _require_nonempty(dataset_split, "dataset_split")
    dataset_revision = _require_nonempty(dataset_revision, "dataset_revision")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ArchiveValidationError("seed must be an integer")
    model_slug = _slug(model_slug or model, "model_slug")

    safe_provenance = sanitize_archive_value(provenance, path="provenance")
    missing_provenance = sorted(REQUIRED_PROVENANCE - set(safe_provenance))
    if missing_provenance:
        raise ArchiveValidationError(
            "provenance is missing required keys: " + ", ".join(missing_provenance)
        )
    for key in REQUIRED_PROVENANCE:
        _require_nonempty(safe_provenance[key], f"provenance.{key}")
    safe_aggregate = sanitize_archive_value(aggregate or {}, path="aggregate")

    identity = {
        "benchmark": benchmark,
        "benchmark_slug": _slug(benchmark, "benchmark"),
        "dataset_alias": dataset_alias,
        "dataset_revision": dataset_revision,
        "dataset_split": dataset_split,
        "model": model,
        "model_revision": model_revision,
        "model_slug": model_slug,
        "run_id": run_id,
        "run_slug": _slug(run_id, "run_id"),
        "seed": seed,
        "stage": stage,
    }
    required_keys = COMMON_RECORD_KEYS | STAGE_RECORD_KEYS[stage]
    normalized_records: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    for row_number, raw_record in enumerate(records):
        if not isinstance(raw_record, Mapping):
            raise ArchiveValidationError(f"record {row_number} is not a mapping")
        safe_record = sanitize_archive_value(
            raw_record,
            path=f"records[{row_number}]",
            opaque_text_keys=OPAQUE_TEXT_RECORD_KEYS[stage],
            opaque_text_paths=OPAQUE_TEXT_RECORD_PATHS[stage],
        )
        missing = sorted(required_keys - set(safe_record))
        if missing:
            raise ArchiveValidationError(
                f"record {row_number} is missing required keys: {', '.join(missing)}"
            )
        source_ordinal = safe_record["source_ordinal"]
        if isinstance(source_ordinal, bool) or not isinstance(source_ordinal, int) or source_ordinal < 0:
            raise ArchiveValidationError(f"record {row_number} source_ordinal must be >= 0")
        source_row_hash = _require_nonempty(safe_record["source_row_hash"], "source_row_hash")
        request_hash = _require_nonempty(safe_record["request_hash"], "request_hash")
        record_id = stable_record_id(
            run_id=run_id,
            model_slug=model_slug,
            model_revision=model_revision,
            seed=seed,
            benchmark=benchmark,
            dataset_alias=dataset_alias,
            dataset_split=dataset_split,
            dataset_revision=dataset_revision,
            source_index=safe_record["source_index"],
            source_ordinal=source_ordinal,
            source_row_hash=source_row_hash,
        )
        supplied_record_id = safe_record.get("record_id")
        if supplied_record_id is not None and supplied_record_id != record_id:
            raise ArchiveValidationError(
                f"record {row_number} record_id does not match its provenance identity"
            )
        if record_id in seen_record_ids:
            raise ArchiveValidationError(
                f"duplicate record identity at row {row_number}; provide distinct source_ordinal values"
            )
        seen_record_ids.add(record_id)
        safe_record["record_id"] = record_id
        safe_record["source_row_hash"] = source_row_hash
        safe_record["request_hash"] = request_hash
        normalized_records.append(safe_record)
    if not normalized_records:
        raise ArchiveValidationError("an archive slice must contain at least one record")

    payload_bytes = b"".join(
        canonical_json(record).encode("utf-8") + b"\n" for record in normalized_records
    )
    payload_sha = sha256_bytes(payload_bytes)
    descriptor_material = {
        "aggregate": safe_aggregate,
        "identity": identity,
        "payload_sha256": payload_sha,
        "provenance": safe_provenance,
        "rows": len(normalized_records),
        "schema_version": SCHEMA_VERSION,
    }
    descriptor_id = sha256_bytes(canonical_json(descriptor_material).encode("utf-8"))
    descriptor = {
        **descriptor_material,
        "created_at": utc_now(),
        "descriptor_id": descriptor_id,
        "payload_path": f"payloads/{descriptor_id}.jsonl",
    }

    root = Path(spool_root)
    _ensure_private_directory(root)
    payload_path = root / descriptor["payload_path"]
    descriptor_path = root / "ready" / f"{descriptor_id}.ready.json"
    _ensure_private_directory(payload_path.parent)
    _ensure_private_directory(descriptor_path.parent)
    if payload_path.exists():
        if sha256_file(payload_path) != payload_sha:
            raise ArchiveIntegrityError(f"existing payload is corrupt: {payload_path}")
        os.chmod(payload_path, 0o600)
    else:
        _atomic_write_bytes(payload_path, payload_bytes)
    if descriptor_path.exists():
        existing = load_descriptor(descriptor_path, spool_root=root)
        if existing["descriptor_id"] != descriptor_id:
            raise ArchiveIntegrityError(f"existing descriptor is corrupt: {descriptor_path}")
        os.chmod(descriptor_path, 0o600)
    else:
        _atomic_write_bytes(
            descriptor_path,
            (json.dumps(descriptor, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    return descriptor_path


def load_descriptor(path: Path, *, spool_root: Path | None = None) -> dict[str, Any]:
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    descriptor_id = descriptor.get("descriptor_id")
    material = {
        "aggregate": descriptor.get("aggregate"),
        "identity": descriptor.get("identity"),
        "payload_sha256": descriptor.get("payload_sha256"),
        "provenance": descriptor.get("provenance"),
        "rows": descriptor.get("rows"),
        "schema_version": descriptor.get("schema_version"),
    }
    expected = sha256_bytes(canonical_json(material).encode("utf-8"))
    if descriptor_id != expected:
        raise ArchiveIntegrityError(f"descriptor digest mismatch: {path}")
    if descriptor.get("schema_version") != SCHEMA_VERSION:
        raise ArchiveValidationError(f"unsupported descriptor schema: {descriptor.get('schema_version')}")
    if spool_root is not None:
        payload_path = (spool_root / descriptor["payload_path"]).resolve()
        root = spool_root.resolve()
        if root != payload_path and root not in payload_path.parents:
            raise ArchiveIntegrityError(f"payload escapes spool root: {payload_path}")
        if not payload_path.is_file() or sha256_file(payload_path) != descriptor["payload_sha256"]:
            raise ArchiveIntegrityError(f"payload digest mismatch: {payload_path}")
    return descriptor


class ArchiveLedger:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        _ensure_private_directory(self.path.parent)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        os.chmod(self.path, 0o600)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS slices (
                    descriptor_id TEXT PRIMARY KEY,
                    descriptor_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    parquet_path TEXT,
                    manifest_path TEXT,
                    remote_parquet_path TEXT,
                    remote_manifest_path TEXT,
                    parquet_sha256 TEXT,
                    manifest_sha256 TEXT,
                    row_count INTEGER,
                    upload_attempts INTEGER NOT NULL DEFAULT 0,
                    uploaded_commit TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def register(self, descriptor_id: str, descriptor_path: Path) -> None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO slices(descriptor_id, descriptor_path, status, created_at, updated_at)
                VALUES (?, ?, 'ready', ?, ?)
                ON CONFLICT(descriptor_id) DO UPDATE SET
                    descriptor_path=excluded.descriptor_path,
                    updated_at=excluded.updated_at
                """,
                (descriptor_id, str(descriptor_path), now, now),
            )

    def mark_built(self, descriptor_id: str, built: Mapping[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE slices SET status='built', parquet_path=?, manifest_path=?,
                    remote_parquet_path=?, remote_manifest_path=?, parquet_sha256=?,
                    manifest_sha256=?, row_count=?, last_error=NULL, updated_at=?
                WHERE descriptor_id=?
                """,
                (
                    built["parquet_path"],
                    built["manifest_path"],
                    built["remote_parquet_path"],
                    built["remote_manifest_path"],
                    built["parquet_sha256"],
                    built["manifest_sha256"],
                    built["row_count"],
                    utc_now(),
                    descriptor_id,
                ),
            )

    def mark_error(self, descriptor_id: str, message: str, *, permanent: bool = False) -> None:
        status = "failed" if permanent else "built"
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE slices SET status=?, upload_attempts=upload_attempts+1,
                    last_error=?, updated_at=? WHERE descriptor_id=?
                """,
                (status, message, utc_now(), descriptor_id),
            )

    def mark_uploaded(self, descriptor_ids: Sequence[str], commit: str) -> None:
        if not descriptor_ids:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                UPDATE slices SET status='uploaded', uploaded_commit=?, last_error=NULL,
                    updated_at=? WHERE descriptor_id=?
                """,
                [(commit, utc_now(), descriptor_id) for descriptor_id in descriptor_ids],
            )

    def rows(self, statuses: Sequence[str] | None = None) -> list[sqlite3.Row]:
        with self._connect() as connection:
            if not statuses:
                return list(connection.execute("SELECT * FROM slices ORDER BY created_at, descriptor_id"))
            placeholders = ",".join("?" for _ in statuses)
            return list(
                connection.execute(
                    f"SELECT * FROM slices WHERE status IN ({placeholders}) ORDER BY created_at, descriptor_id",
                    tuple(statuses),
                )
            )


@dataclass(frozen=True)
class FlushReport:
    discovered: int = 0
    built: int = 0
    uploaded: int = 0
    already_remote: int = 0
    failed: int = 0


def _json_or_none(value: Any) -> str | None:
    return None if value is None else canonical_json(value)


def _parquet_row(descriptor: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    identity = descriptor["identity"]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "descriptor_id": descriptor["descriptor_id"],
        "record_id": record["record_id"],
        "stage": identity["stage"],
        "run_id": identity["run_id"],
        "model": identity["model"],
        "model_slug": identity["model_slug"],
        "model_revision": identity["model_revision"],
        "seed": identity["seed"],
        "benchmark": identity["benchmark"],
        "dataset_alias": identity["dataset_alias"],
        "dataset_split": identity["dataset_split"],
        "dataset_revision": identity["dataset_revision"],
        "source_index": str(record["source_index"]),
        "source_ordinal": int(record["source_ordinal"]),
        "source_row_hash": record["source_row_hash"],
        "request_hash": record["request_hash"],
        "provenance_json": canonical_json(descriptor["provenance"]),
        "record_json": canonical_json(record),
    }
    for key in TEXT_COLUMNS:
        value = record.get(key)
        result[key] = None if value is None else str(value)
    for key in JSON_COLUMNS:
        result[f"{key}_json"] = _json_or_none(record.get(key))
    return result


class ArchiveDaemon:
    """Build and upload ready archive slices without blocking evaluation."""

    def __init__(
        self,
        *,
        spool_root: Path | str,
        repo_id: str = DEFAULT_REPO_ID,
        revision: str = DEFAULT_REVISION,
        token: str | None = None,
        api: Any | None = None,
        batch_size: int = 48,
        upload_threads: int = 8,
        max_retries: int = 6,
        retry_base_seconds: float = 5.0,
        retry_cap_seconds: float = 300.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.spool_root = Path(spool_root)
        _ensure_private_directory(self.spool_root)
        self.repo_id = repo_id
        self.revision = revision
        self.token = token
        self.api = api if api is not None else HfApi(token=token)
        self.batch_size = batch_size
        self.upload_threads = max(1, upload_threads)
        self.max_retries = max(1, max_retries)
        self.retry_base_seconds = max(0.0, retry_base_seconds)
        self.retry_cap_seconds = max(0.0, retry_cap_seconds)
        self.sleep = sleep
        self.ledger = ArchiveLedger(self.spool_root / "ledger.sqlite3")

    def _api_call(self, name: str, /, **kwargs: Any) -> Any:
        kwargs.setdefault("token", self.token)
        return getattr(self.api, name)(**kwargs)

    def assert_private_repo(self, *, files_metadata: bool = True) -> Any:
        info = self._api_call(
            "repo_info",
            repo_id=self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            files_metadata=files_metadata,
        )
        if not bool(getattr(info, "private", False)):
            raise ArchiveValidationError(f"refusing to archive into non-private dataset repo {self.repo_id}")
        return info

    def initialize_repo(self) -> None:
        try:
            info = self.assert_private_repo(files_metadata=False)
        except RepositoryNotFoundError:
            self._api_call(
                "create_repo",
                repo_id=self.repo_id,
                repo_type="dataset",
                private=True,
                exist_ok=True,
            )
            info = self.assert_private_repo(files_metadata=False)
        remote_names = {sibling.rfilename for sibling in getattr(info, "siblings", [])}
        if "README.md" in remote_names:
            return
        readme = self.spool_root / "control" / "README.md"
        _atomic_write_bytes(
            readme,
            (
                "---\nlicense: other\npretty_name: TRACE Evaluation Runs\n---\n\n"
                "Private, content-addressed evaluation responses, extractions, and scores.\n"
                "Benchmark media is intentionally excluded.\n"
            ).encode("utf-8"),
        )
        self._api_call(
            "create_commit",
            repo_id=self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            operations=[CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(readme))],
            commit_message="Initialize TRACE evaluation archive",
            num_threads=1,
        )

    def discover(self) -> list[Path]:
        descriptors: list[Path] = []
        for path in sorted((self.spool_root / "ready").glob("*.ready.json")):
            try:
                descriptor = load_descriptor(path, spool_root=self.spool_root)
                self.ledger.register(descriptor["descriptor_id"], path)
            except Exception as error:
                safe_error = redact_secret(error, self.token)
                print(
                    f"[archive:invalid-descriptor] path={path} "
                    f"error={type(error).__name__}: {safe_error}",
                    file=sys.stderr,
                )
                continue
            descriptors.append(path)
        return descriptors

    def _load_payload(self, descriptor: Mapping[str, Any]) -> list[dict[str, Any]]:
        path = self.spool_root / descriptor["payload_path"]
        if sha256_file(path) != descriptor["payload_sha256"]:
            raise ArchiveIntegrityError(f"payload digest mismatch: {path}")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(records) != descriptor["rows"]:
            raise ArchiveIntegrityError(f"payload row count mismatch: {path}")
        return records

    def build_descriptor(self, descriptor_path: Path) -> dict[str, Any]:
        descriptor = load_descriptor(descriptor_path, spool_root=self.spool_root)
        records = self._load_payload(descriptor)
        table = pa.Table.from_pylist(
            [_parquet_row(descriptor, record) for record in records], schema=PARQUET_SCHEMA
        )
        identity = descriptor["identity"]
        partition = (
            Path("data")
            / identity["stage"]
            / f"run={identity['run_slug']}"
            / f"model={identity['model_slug']}"
            / f"seed={identity['seed']}"
            / f"benchmark={identity['benchmark_slug']}"
        )
        build_dir = self.spool_root / "building"
        _ensure_private_directory(build_dir)
        temporary = build_dir / f"{descriptor['descriptor_id']}.parquet.tmp"
        pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
        os.chmod(temporary, 0o600)
        parquet_sha = sha256_file(temporary)
        remote_parquet = partition / f"part-{parquet_sha}.parquet"
        parquet_path = self.spool_root / "staged" / remote_parquet
        _ensure_private_directory(parquet_path.parent)
        if parquet_path.exists():
            if sha256_file(parquet_path) != parquet_sha:
                raise ArchiveIntegrityError(f"existing Parquet is corrupt: {parquet_path}")
            temporary.unlink(missing_ok=True)
        else:
            os.replace(temporary, parquet_path)
        os.chmod(parquet_path, 0o600)

        manifest = {
            "aggregate": descriptor["aggregate"],
            "descriptor_id": descriptor["descriptor_id"],
            "identity": identity,
            "parquet_path": remote_parquet.as_posix(),
            "parquet_sha256": parquet_sha,
            "payload_sha256": descriptor["payload_sha256"],
            "provenance": descriptor["provenance"],
            "rows": descriptor["rows"],
            "schema_version": SCHEMA_VERSION,
        }
        manifest["manifest_sha256"] = sha256_bytes(
            canonical_json(manifest).encode("utf-8")
        )
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        manifest_sha = sha256_bytes(manifest_bytes)
        remote_manifest = (
            Path("metadata")
            / "slices"
            / partition.relative_to("data")
            / f"part-{parquet_sha}.manifest.json"
        )
        manifest_path = self.spool_root / "staged" / remote_manifest
        if manifest_path.exists():
            if sha256_file(manifest_path) != manifest_sha:
                raise ArchiveIntegrityError(f"existing manifest is corrupt: {manifest_path}")
        else:
            _atomic_write_bytes(manifest_path, manifest_bytes)
        os.chmod(manifest_path, 0o600)

        built = {
            "descriptor_id": descriptor["descriptor_id"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "parquet_path": str(parquet_path),
            "parquet_sha256": parquet_sha,
            "remote_manifest_path": remote_manifest.as_posix(),
            "remote_parquet_path": remote_parquet.as_posix(),
            "row_count": descriptor["rows"],
        }
        built_path = self.spool_root / "built" / f"{descriptor['descriptor_id']}.built.json"
        _atomic_write_bytes(
            built_path, (json.dumps(built, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        self.ledger.mark_built(descriptor["descriptor_id"], built)
        return built

    def build_ready(self) -> int:
        count = 0
        self.discover()
        for row in self.ledger.rows(("ready",)):
            try:
                self.build_descriptor(Path(row["descriptor_path"]))
                count += 1
            except Exception as error:
                self.ledger.mark_error(row["descriptor_id"], str(error), permanent=True)
        return count

    def _remote_digests(self) -> tuple[dict[str, str | None], Any]:
        info = self.assert_private_repo()
        result: dict[str, str | None] = {}
        for sibling in getattr(info, "siblings", []):
            lfs = getattr(sibling, "lfs", None)
            if isinstance(lfs, Mapping):
                digest = lfs.get("sha256")
            else:
                digest = getattr(lfs, "sha256", None) if lfs is not None else None
            result[sibling.rfilename] = digest
        return result, info

    def _download_remote_sha(self, remote_path: str) -> str:
        with tempfile.TemporaryDirectory(prefix="trace-eval-hf-verify-") as temporary:
            downloaded = self._api_call(
                "hf_hub_download",
                repo_id=self.repo_id,
                repo_type="dataset",
                revision=self.revision,
                filename=remote_path,
                local_dir=temporary,
                force_download=True,
            )
            return sha256_file(Path(downloaded))

    def _verify_remote_file(
        self, remote_path: str, expected_sha: str, remote_digests: Mapping[str, str | None]
    ) -> bool:
        if remote_path not in remote_digests:
            return False
        actual = remote_digests[remote_path] or self._download_remote_sha(remote_path)
        if actual != expected_sha:
            raise ArchiveIntegrityError(
                f"immutable remote path has digest {actual}, expected {expected_sha}: {remote_path}"
            )
        return True

    def _upload_rows(self, rows: Sequence[sqlite3.Row]) -> tuple[int, int]:
        remote_digests, _ = self._remote_digests()
        operations: list[CommitOperationAdd] = []
        ready_to_mark: list[str] = []
        upload_ids: list[str] = []
        for row in rows:
            parquet_exists = self._verify_remote_file(
                row["remote_parquet_path"], row["parquet_sha256"], remote_digests
            )
            manifest_exists = self._verify_remote_file(
                row["remote_manifest_path"], row["manifest_sha256"], remote_digests
            )
            if parquet_exists and manifest_exists:
                ready_to_mark.append(row["descriptor_id"])
                continue
            if not parquet_exists:
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=row["remote_parquet_path"], path_or_fileobj=row["parquet_path"]
                    )
                )
            if not manifest_exists:
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=row["remote_manifest_path"], path_or_fileobj=row["manifest_path"]
                    )
                )
            upload_ids.append(row["descriptor_id"])

        if ready_to_mark:
            self.ledger.mark_uploaded(ready_to_mark, "already-remote")
        if not operations:
            return 0, len(ready_to_mark)

        error: Exception | None = None
        commit: Any | None = None
        for attempt in range(self.max_retries):
            try:
                commit = self._api_call(
                    "create_commit",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    revision=self.revision,
                    operations=operations,
                    commit_message=f"Archive {len(upload_ids)} completed TRACE evaluation slices",
                    commit_description="Content-addressed response, extraction, and score slices; no media.",
                    num_threads=self.upload_threads,
                )
                error = None
                break
            except Exception as caught:
                error = caught
                if attempt + 1 < self.max_retries:
                    delay = min(self.retry_cap_seconds, self.retry_base_seconds * (2**attempt))
                    self.sleep(delay)
        if error is not None:
            for descriptor_id in upload_ids:
                self.ledger.mark_error(descriptor_id, redact_secret(error, self.token))
            raise error

        remote_digests, _ = self._remote_digests()
        for row in rows:
            if row["descriptor_id"] not in upload_ids:
                continue
            self._verify_remote_file(
                row["remote_parquet_path"], row["parquet_sha256"], remote_digests
            )
            self._verify_remote_file(
                row["remote_manifest_path"], row["manifest_sha256"], remote_digests
            )
        commit_id = str(getattr(commit, "oid", "unknown"))
        self.ledger.mark_uploaded(upload_ids, commit_id)
        return len(upload_ids), len(ready_to_mark)

    def flush(self) -> FlushReport:
        descriptors = self.discover()
        built = self.build_ready()
        rows = self.ledger.rows(("built",))[: self.batch_size]
        if not rows:
            failed = len(self.ledger.rows(("failed",)))
            return FlushReport(discovered=len(descriptors), built=built, failed=failed)
        try:
            uploaded, already_remote = self._upload_rows(rows)
        except Exception:
            failed = len(self.ledger.rows(("failed",))) + len(rows)
            return FlushReport(
                discovered=len(descriptors), built=built, failed=failed
            )
        return FlushReport(
            discovered=len(descriptors),
            built=built,
            uploaded=uploaded,
            already_remote=already_remote,
            failed=len(self.ledger.rows(("failed",))),
        )

    def flush_all(self) -> FlushReport:
        total = FlushReport()
        while True:
            report = self.flush()
            total = FlushReport(
                discovered=max(total.discovered, report.discovered),
                built=total.built + report.built,
                uploaded=total.uploaded + report.uploaded,
                already_remote=total.already_remote + report.already_remote,
                failed=max(total.failed, report.failed),
            )
            if not self.ledger.rows(("ready", "built")):
                return total
            if report.uploaded == 0 and report.already_remote == 0 and report.built == 0:
                return total

    def verify(self) -> dict[str, int]:
        self.discover()
        all_rows = self.ledger.rows()
        incomplete = [row for row in all_rows if row["status"] != "uploaded"]
        if incomplete:
            raise ArchiveIntegrityError(f"{len(incomplete)} local slices are not uploaded")
        remote_digests, _ = self._remote_digests()
        for row in all_rows:
            self._verify_remote_file(
                row["remote_parquet_path"], row["parquet_sha256"], remote_digests
            )
            self._verify_remote_file(
                row["remote_manifest_path"], row["manifest_sha256"], remote_digests
            )
        return {"slices": len(all_rows), "rows": sum(int(row["row_count"]) for row in all_rows)}

    def run(self, *, poll_seconds: float = 30.0, stop_event: threading.Event | None = None) -> None:
        stop_event = stop_event or threading.Event()
        while not stop_event.is_set():
            self.flush_all()
            stop_event.wait(max(0.1, poll_seconds))
        self.flush_all()


def verify_expected_slice_coverage(
    spool_root: Path | str,
    *,
    run_id: str,
    model_slugs: Sequence[str],
    seeds: Sequence[int],
    benchmarks: Sequence[str],
    stages: Sequence[str] = ("generation", "extraction", "score"),
    campaign_config_hash: str | None = None,
    dataset_revision: str | None = None,
) -> dict[str, int]:
    """Require exactly one matching descriptor for every campaign identity."""

    invalid_stages = sorted(set(stages) - ALLOWED_STAGES)
    if invalid_stages:
        raise ArchiveValidationError(f"unsupported expected stages: {invalid_stages}")
    expected = {
        (str(run_id), _slug(str(model_slug), "model_slug"), int(seed), str(benchmark), str(stage))
        for model_slug in model_slugs
        for seed in seeds
        for benchmark in benchmarks
        for stage in stages
    }
    root = Path(spool_root)
    required_campaign_hash = (
        _require_nonempty(campaign_config_hash, "campaign_config_hash")
        if campaign_config_hash is not None
        else None
    )
    required_dataset_revision = (
        _require_nonempty(dataset_revision, "dataset_revision")
        if dataset_revision is not None
        else None
    )
    found: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = {}
    for path in sorted((root / "ready").glob("*.ready.json")):
        descriptor = load_descriptor(path, spool_root=root)
        identity = descriptor["identity"]
        key = (
            str(identity["run_id"]),
            str(identity["model_slug"]),
            int(identity["seed"]),
            str(identity["benchmark"]),
            str(identity["stage"]),
        )
        if key in expected:
            found.setdefault(key, []).append(descriptor)
    missing = sorted(expected - set(found))
    if missing:
        preview = "; ".join(
            f"model={model} seed={seed} benchmark={benchmark} stage={stage}"
            for _run, model, seed, benchmark, stage in missing[:10]
        )
        raise ArchiveIntegrityError(
            f"archive is missing {len(missing)} expected slices for run {run_id}: {preview}"
        )
    duplicates = sorted(key for key in expected if len(found.get(key, ())) != 1)
    if duplicates:
        preview = "; ".join(
            f"model={model} seed={seed} benchmark={benchmark} stage={stage} "
            f"versions={len(found[(logical_run, model, seed, benchmark, stage)])}"
            for logical_run, model, seed, benchmark, stage in duplicates[:10]
        )
        raise ArchiveIntegrityError(
            f"archive has {len(duplicates)} duplicate logical slices for run {run_id}: {preview}"
        )
    mismatches: list[str] = []
    for key in sorted(expected):
        descriptor = found[key][0]
        identity = descriptor["identity"]
        provenance = descriptor["provenance"]
        if required_campaign_hash is not None and provenance.get("campaign_config_hash") != required_campaign_hash:
            mismatches.append(
                f"model={key[1]} seed={key[2]} benchmark={key[3]} stage={key[4]} "
                f"campaign_config_hash={provenance.get('campaign_config_hash')!r}"
            )
        if required_dataset_revision is not None and identity.get("dataset_revision") != required_dataset_revision:
            mismatches.append(
                f"model={key[1]} seed={key[2]} benchmark={key[3]} stage={key[4]} "
                f"dataset_revision={identity.get('dataset_revision')!r}"
            )
    if mismatches:
        raise ArchiveIntegrityError(
            f"archive has {len(mismatches)} provenance mismatches for run {run_id}: "
            + "; ".join(mismatches[:10])
        )
    return {
        "expected_identities": len(expected),
        "covered_identities": len(expected),
        "descriptor_versions": len(expected),
    }


def reconstruct_remote(
    *,
    api: Any,
    repo_id: str,
    revision: str,
    token: str | None,
    output: Path | str,
    stage: str | None = None,
    run_id: str | None = None,
    model_slug: str | None = None,
    seed: int | None = None,
    benchmark: str | None = None,
) -> int:
    """Download matching immutable slices and reconstruct one long-form Parquet."""

    segments = ["data"]
    if stage is not None:
        if stage not in ALLOWED_STAGES:
            raise ArchiveValidationError(f"unsupported stage: {stage}")
        segments.append(stage)
    prefix = "/".join(segments) + "/"
    filters = {
        "run": _slug(run_id, "run_id") if run_id else None,
        "model": _slug(model_slug, "model_slug") if model_slug else None,
        "seed": str(seed) if seed is not None else None,
        "benchmark": _slug(benchmark, "benchmark") if benchmark else None,
    }
    files = set(
        api.list_repo_files(
        repo_id=repo_id, repo_type="dataset", revision=revision, token=token
        )
    )
    selected: list[str] = []
    for name in sorted(files):
        if not name.startswith(prefix) or not name.endswith(".parquet"):
            continue
        if all(value is None or f"/{key}={value}/" in f"/{name}" for key, value in filters.items()):
            selected.append(name)
    if not selected:
        raise ArchiveValidationError("no remote Parquet slices matched the requested filters")

    tables: list[pa.Table] = []
    logical_slices: dict[tuple[str, str, str, int, str], str] = {}
    with tempfile.TemporaryDirectory(prefix="trace-eval-reconstruct-") as temporary:
        for remote_path in sorted(selected):
            filename_match = re.fullmatch(r"part-([0-9a-f]{64})\.parquet", Path(remote_path).name)
            if filename_match is None:
                raise ArchiveIntegrityError(
                    f"remote Parquet path is not content-addressed: {remote_path}"
                )
            expected_parquet_sha = filename_match.group(1)
            relative = Path(remote_path).relative_to("data")
            remote_manifest = (
                Path("metadata")
                / "slices"
                / relative.parent
                / f"part-{expected_parquet_sha}.manifest.json"
            ).as_posix()
            if remote_manifest not in files:
                raise ArchiveIntegrityError(
                    f"remote Parquet is missing its manifest: {remote_path}"
                )
            downloaded = api.hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                filename=remote_path,
                local_dir=temporary,
                force_download=True,
                token=token,
            )
            downloaded_path = Path(downloaded)
            actual_parquet_sha = sha256_file(downloaded_path)
            if actual_parquet_sha != expected_parquet_sha:
                raise ArchiveIntegrityError(
                    f"remote Parquet digest mismatch for {remote_path}: "
                    f"expected {expected_parquet_sha}, found {actual_parquet_sha}"
                )
            downloaded_manifest = api.hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                filename=remote_manifest,
                local_dir=temporary,
                force_download=True,
                token=token,
            )
            try:
                manifest = json.loads(Path(downloaded_manifest).read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ArchiveIntegrityError(
                    f"remote manifest is malformed: {remote_manifest}"
                ) from error
            if not isinstance(manifest, dict):
                raise ArchiveIntegrityError(f"remote manifest is not an object: {remote_manifest}")
            supplied_manifest_sha = manifest.get("manifest_sha256")
            manifest_material = dict(manifest)
            manifest_material.pop("manifest_sha256", None)
            try:
                actual_manifest_sha = sha256_bytes(
                    canonical_json(manifest_material).encode("utf-8")
                )
            except (TypeError, ValueError) as error:
                raise ArchiveIntegrityError(
                    f"remote manifest is not canonical JSON: {remote_manifest}"
                ) from error
            if supplied_manifest_sha != actual_manifest_sha:
                raise ArchiveIntegrityError(
                    f"remote manifest digest mismatch for {remote_manifest}: "
                    f"expected {supplied_manifest_sha!r}, found {actual_manifest_sha}"
                )
            if manifest.get("schema_version") != SCHEMA_VERSION:
                raise ArchiveIntegrityError(
                    f"remote manifest has unsupported schema {manifest.get('schema_version')!r}: "
                    f"{remote_manifest}"
                )
            if manifest.get("parquet_path") != remote_path:
                raise ArchiveIntegrityError(
                    f"remote manifest Parquet path mismatch: {remote_manifest}"
                )
            if manifest.get("parquet_sha256") != expected_parquet_sha:
                raise ArchiveIntegrityError(
                    f"remote manifest Parquet digest mismatch: {remote_manifest}"
                )
            descriptor_material = {
                "aggregate": manifest.get("aggregate"),
                "identity": manifest.get("identity"),
                "payload_sha256": manifest.get("payload_sha256"),
                "provenance": manifest.get("provenance"),
                "rows": manifest.get("rows"),
                "schema_version": manifest.get("schema_version"),
            }
            expected_descriptor_id = sha256_bytes(
                canonical_json(descriptor_material).encode("utf-8")
            )
            if manifest.get("descriptor_id") != expected_descriptor_id:
                raise ArchiveIntegrityError(
                    f"remote manifest descriptor digest mismatch: {remote_manifest}"
                )

            parquet_file = pq.ParquetFile(downloaded_path)
            table = parquet_file.read()
            if not table.schema.equals(PARQUET_SCHEMA, check_metadata=True):
                raise ArchiveIntegrityError(f"remote Parquet schema mismatch: {remote_path}")
            try:
                expected_rows = int(manifest["rows"])
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise ArchiveIntegrityError(
                    f"remote manifest row count is invalid: {remote_manifest}"
                ) from error
            if expected_rows < 1 or table.num_rows != expected_rows:
                raise ArchiveIntegrityError(
                    f"remote row count mismatch for {remote_path}: "
                    f"manifest={expected_rows}, parquet={table.num_rows}"
                )
            identity = manifest.get("identity")
            if not isinstance(identity, Mapping):
                raise ArchiveIntegrityError(f"remote manifest identity is invalid: {remote_manifest}")
            expected_partition = (
                Path("data")
                / str(identity.get("stage"))
                / f"run={identity.get('run_slug')}"
                / f"model={identity.get('model_slug')}"
                / f"seed={identity.get('seed')}"
                / f"benchmark={identity.get('benchmark_slug')}"
                / f"part-{expected_parquet_sha}.parquet"
            ).as_posix()
            if expected_partition != remote_path:
                raise ArchiveIntegrityError(
                    f"remote partition does not match manifest identity: {remote_path}"
                )
            identity_columns = (
                "stage",
                "run_id",
                "model",
                "model_slug",
                "model_revision",
                "seed",
                "benchmark",
                "dataset_alias",
                "dataset_split",
                "dataset_revision",
            )
            for column in identity_columns:
                values = pc.unique(table[column]).to_pylist()
                if values != [identity.get(column)]:
                    raise ArchiveIntegrityError(
                        f"remote Parquet {column} does not match manifest: {remote_path}"
                    )
            descriptor_ids = pc.unique(table["descriptor_id"]).to_pylist()
            if descriptor_ids != [manifest.get("descriptor_id")]:
                raise ArchiveIntegrityError(
                    f"remote Parquet descriptor_id does not match manifest: {remote_path}"
                )
            provenance_values = pc.unique(table["provenance_json"]).to_pylist()
            if provenance_values != [canonical_json(manifest.get("provenance"))]:
                raise ArchiveIntegrityError(
                    f"remote Parquet provenance does not match manifest: {remote_path}"
                )
            logical_key = (
                str(identity.get("stage")),
                str(identity.get("run_id")),
                str(identity.get("model_slug")),
                int(identity.get("seed")),
                str(identity.get("benchmark")),
            )
            previous = logical_slices.get(logical_key)
            if previous is not None:
                raise ArchiveIntegrityError(
                    f"duplicate logical remote slice for {logical_key}: {previous}, {remote_path}"
                )
            logical_slices[logical_key] = remote_path
            tables.append(table)
    table = pa.concat_tables(tables, promote_options="default")
    exact_filters = {
        "stage": stage,
        "run_id": run_id,
        "model_slug": _slug(model_slug, "model_slug") if model_slug else None,
        "seed": seed,
        "benchmark": benchmark,
    }
    for column, value in exact_filters.items():
        if value is not None:
            table = table.filter(pc.equal(table[column], pa.scalar(value)))
    if table.num_rows == 0:
        raise ArchiveValidationError("partition matches did not contain the requested exact identities")
    table = table.sort_by(
        [
            ("stage", "ascending"),
            ("benchmark", "ascending"),
            ("model_slug", "ascending"),
            ("seed", "ascending"),
            ("source_ordinal", "ascending"),
            ("source_index", "ascending"),
        ]
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    pq.write_table(table, temporary_output, compression="zstd", use_dictionary=True)
    os.chmod(temporary_output, 0o600)
    os.replace(temporary_output, output_path)
    return table.num_rows


def install_stop_signals(stop_event: threading.Event) -> None:
    def stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
