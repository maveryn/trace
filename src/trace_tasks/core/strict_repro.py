"""Strict reproducibility comparison utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .canonical import canonical_json_bytes
from .hash_utils import blake3_file
from .trace_store import read_trace_shard


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL records from a file, returning an empty list if absent."""
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def _first_diff_path(left: Any, right: Any, prefix: str = "") -> str:
    """Return the first field-path where two nested structures differ."""
    if type(left) is not type(right):
        return prefix or "<root>"
    if isinstance(left, dict):
        left_keys = sorted(left.keys())
        right_keys = sorted(right.keys())
        if left_keys != right_keys:
            return (prefix + ".keys") if prefix else "keys"
        for key in left_keys:
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            path = _first_diff_path(left[key], right[key], child_prefix)
            if path:
                return path
        return ""
    if isinstance(left, list):
        if len(left) != len(right):
            return (prefix + ".length") if prefix else "length"
        for index, (l_item, r_item) in enumerate(zip(left, right)):
            child_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            path = _first_diff_path(l_item, r_item, child_prefix)
            if path:
                return path
        return ""
    if left != right:
        return prefix or "<root>"
    return ""


def _list_files(root: Path) -> List[Path]:
    """List all regular files under `root` using deterministic ordering."""
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def compare_staging_dirs(primary_root: Path, repro_root: Path) -> Dict[str, Any] | None:
    """Compare two staging directories and return first mismatch summary or None."""
    primary_train = _read_jsonl(primary_root / "train_instances.jsonl")
    repro_train = _read_jsonl(repro_root / "train_instances.jsonl")

    if len(primary_train) != len(repro_train):
        return {
            "section": "train_instances",
            "reason": "count_mismatch",
            "primary_count": len(primary_train),
            "repro_count": len(repro_train),
        }

    for index, (left_record, right_record) in enumerate(zip(primary_train, repro_train)):
        left_bytes = canonical_json_bytes(left_record)
        right_bytes = canonical_json_bytes(right_record)
        if left_bytes != right_bytes:
            return {
                "section": "train_instances",
                "reason": "record_mismatch",
                "line_index": index,
                "instance_id": left_record.get("instance_id"),
                "field_path": _first_diff_path(left_record, right_record),
            }

    primary_trace_files = sorted((primary_root / "traces").glob("*.jsonl.zst"))
    repro_trace_files = sorted((repro_root / "traces").glob("*.jsonl.zst"))
    primary_trace_names = [path.name for path in primary_trace_files]
    repro_trace_names = [path.name for path in repro_trace_files]
    if primary_trace_names != repro_trace_names:
        return {
            "section": "traces",
            "reason": "shard_set_mismatch",
            "primary_shards": primary_trace_names,
            "repro_shards": repro_trace_names,
        }

    for name in primary_trace_names:
        left_records = read_trace_shard(primary_root / "traces" / name)
        right_records = read_trace_shard(repro_root / "traces" / name)
        if len(left_records) != len(right_records):
            return {
                "section": "traces",
                "reason": "record_count_mismatch",
                "shard_id": name,
                "primary_count": len(left_records),
                "repro_count": len(right_records),
            }
        for index, (left_record, right_record) in enumerate(zip(left_records, right_records)):
            if canonical_json_bytes(left_record) != canonical_json_bytes(right_record):
                return {
                    "section": "traces",
                    "reason": "record_mismatch",
                    "shard_id": name,
                    "line_index": index,
                    "instance_id": left_record.get("instance_id"),
                    "field_path": _first_diff_path(left_record, right_record),
                }

    primary_images = _list_files(primary_root / "images")
    repro_images = _list_files(repro_root / "images")
    primary_rel = [path.relative_to(primary_root).as_posix() for path in primary_images]
    repro_rel = [path.relative_to(repro_root).as_posix() for path in repro_images]
    if primary_rel != repro_rel:
        return {
            "section": "images",
            "reason": "path_set_mismatch",
            "primary_paths": primary_rel,
            "repro_paths": repro_rel,
        }

    for rel_path in primary_rel:
        left_path = primary_root / rel_path
        right_path = repro_root / rel_path
        left_hash = blake3_file(left_path)
        right_hash = blake3_file(right_path)
        if left_hash != right_hash:
            return {
                "section": "images",
                "reason": "hash_mismatch",
                "image_path": rel_path,
                "primary_image_hash": left_hash,
                "repro_image_hash": right_hash,
            }

    return None
