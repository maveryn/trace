"""Shared media identity contract for the trace_eval_v1 generation pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

# These three values are immutable producer-era wire identifiers. Canonical
# run artifacts already contain them, so readers and reproducibility reruns
# must retain the exact strings even though the public suite is trace_eval_v1.
# They do not enable any other suite or benchmark route.
DATASET_MANIFEST_SCHEMA = "trace-final25-datasets-v2"
MEDIA_CONTRACT_VERSION = "trace-final25-media-v2"
GENERATION_CONTRACT_VERSION = "trace-final25-generation-v7"
MEDIA_TRANSPORT = "file-url"
QWEN_MIN_IMAGE_PIXELS = 4 * 28 * 28
QWEN_MAX_IMAGE_PIXELS = 16384 * 28 * 28


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def media_set_sha256(media: Iterable[Mapping[str, Any]]) -> str:
    ordered = [
        {"type": str(item["type"]), "sha256": str(item["sha256"])} for item in media
    ]
    return canonical_sha256(ordered)


def source_record_sha256(source_row_hash: str, media_hash: str) -> str:
    return canonical_sha256(
        {
            "source_row_hash": str(source_row_hash),
            "media_set_sha256": str(media_hash),
        }
    )


_SOURCE_ROW_MEDIA_KEYS = {
    "image",
    "images",
    "image_path",
    "image_paths",
    "img",
    "picture",
    # Producer hashing also excluded these media column names. Retaining the
    # names here is necessary to verify canonical archived row hashes; the
    # public runtime does not construct or transport this media type.
    "video",
    "videos",
    "video_path",
    "video_paths",
}


def source_row_sha256(row: Any) -> str:
    """Hash non-media source fields using the canonical producer contract."""

    raw = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    material = {
        str(key): value
        for key, value in raw.items()
        if str(key) not in _SOURCE_ROW_MEDIA_KEYS
        and not str(key).startswith("__trace_")
    }
    try:
        import numpy as np

        def json_default(value: Any) -> Any:
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.bool_):
                return bool(value)
            return str(value)

    except ImportError:
        json_default = str
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_snapshot_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    material = [
        {
            "ordinal": int(row["ordinal"]),
            "index": str(row["index"]),
            "source_row_hash": str(row["source_row_hash"]),
            "media_set_sha256": str(row["media_set_sha256"]),
            "source_record_sha256": str(row["source_record_sha256"]),
        }
        for row in rows
    ]
    return canonical_sha256(material)


def manifest_snapshot_sha256(
    *,
    suite_sha256: str,
    vlmevalkit_commit: str,
    datasets: Mapping[str, Mapping[str, Any]],
    keys: Iterable[str] | None = None,
) -> str:
    selected = set(str(key) for key in keys) if keys is not None else None
    ready = {
        str(key): str(receipt["dataset_snapshot_sha256"])
        for key, receipt in sorted(datasets.items())
        if selected is None or str(key) in selected
        if receipt.get("status") == "ready" and receipt.get("dataset_snapshot_sha256")
    }
    if selected is not None and set(ready) != selected:
        missing = sorted(selected - set(ready))
        raise RuntimeError(
            f"dataset manifest is missing ready snapshot receipts for: {missing}"
        )
    return canonical_sha256(
        {
            "schema_version": DATASET_MANIFEST_SCHEMA,
            "suite_sha256": str(suite_sha256),
            "vlmevalkit_commit": str(vlmevalkit_commit),
            "datasets": ready,
        }
    )


def row_lookup_key(
    ordinal: int, index: Any, source_row_hash: str
) -> tuple[int, str, str]:
    return int(ordinal), str(index), str(source_row_hash)


def load_dataset_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DATASET_MANIFEST_SCHEMA:
        raise RuntimeError(
            f"unsupported TRACE evaluation dataset manifest schema in {path}: "
            f"{payload.get('schema_version')!r}"
        )
    for key, receipt in (payload.get("datasets") or {}).items():
        if receipt.get("status") != "ready":
            continue
        rows = receipt.get("row_media")
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(
                f"TRACE evaluation dataset {key!r} has no row media records in {path}"
            )
        for row in rows:
            media_hash = media_set_sha256(row.get("media") or [])
            if row.get("media_set_sha256") != media_hash:
                raise RuntimeError(
                    f"TRACE evaluation dataset {key!r} has a corrupt row media hash in {path}"
                )
            record_hash = source_record_sha256(
                str(row.get("source_row_hash") or ""), media_hash
            )
            if row.get("source_record_sha256") != record_hash:
                raise RuntimeError(
                    f"TRACE evaluation dataset {key!r} has a corrupt source record hash in {path}"
                )
        receipt_snapshot = dataset_snapshot_sha256(rows)
        if receipt.get("dataset_snapshot_sha256") != receipt_snapshot:
            raise RuntimeError(
                f"TRACE evaluation dataset {key!r} snapshot hash mismatch in {path}"
            )
    expected = manifest_snapshot_sha256(
        suite_sha256=str(payload.get("suite_sha256") or ""),
        vlmevalkit_commit=str(payload.get("vlmevalkit_commit") or ""),
        datasets=payload.get("datasets") or {},
    )
    if payload.get("dataset_snapshot_sha256") != expected:
        raise RuntimeError(f"TRACE evaluation dataset snapshot hash mismatch in {path}")
    for view, actual in (payload.get("view_snapshot_sha256") or {}).items():
        keys = (payload.get("dataset_views") or {}).get(view)
        if not isinstance(keys, list):
            raise RuntimeError(
                f"TRACE evaluation dataset manifest has no key list for view {view!r}"
            )
        view_expected = manifest_snapshot_sha256(
            suite_sha256=str(payload.get("suite_sha256") or ""),
            vlmevalkit_commit=str(payload.get("vlmevalkit_commit") or ""),
            datasets=payload.get("datasets") or {},
            keys=keys,
        )
        if actual != view_expected:
            raise RuntimeError(
                f"TRACE evaluation dataset {view!r} snapshot hash mismatch in {path}"
            )
    return payload
