#!/usr/bin/env python3
"""Dataset materialization helpers for the canonical trace_eval_v1 manifest."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from trace_eval_media_contract import (  # noqa: E402
    dataset_snapshot_sha256,
    media_set_sha256,
    source_record_sha256,
    source_row_sha256,
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(_canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _configure_environment(lmu_root: Path, hf_home: Path, token: str | None) -> None:
    lmu_root.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["LMUData"] = str(lmu_root.resolve())
    os.environ["HF_HOME"] = str(hf_home.resolve())
    os.environ["HF_DATASETS_CACHE"] = str((hf_home / "datasets").resolve())
    os.environ["HUGGINGFACE_HUB_CACHE"] = str((hf_home / "hub").resolve())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def _install_import_paths(vlmeval_root: Path) -> None:
    for path in (SCRIPTS_ROOT, vlmeval_root, vlmeval_root / "scripts"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _flatten_media_value(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_media_value(item))
        return flattened
    if isinstance(value, os.PathLike):
        return [os.fspath(value)]
    if isinstance(value, str):
        return [value]
    raise TypeError(f"unsupported prompt media value type: {type(value).__name__}")


def _resolve_media_path(value: str, working_directory: Path) -> Path:
    parsed = urlparse(value)
    if parsed.scheme and parsed.scheme != "file":
        raise RuntimeError(f"prompt media was not materialized locally: {value[:160]}")
    path = Path(parsed.path) if parsed.scheme == "file" else Path(value).expanduser()
    if not path.is_absolute():
        path = working_directory / path
    return path.resolve()


def _prompt_media(dataset: Any, row: Any, working_directory: Path) -> list[Path]:
    prompt = dataset.build_prompt(row)
    if not isinstance(prompt, list):
        raise TypeError(f"build_prompt returned {type(prompt).__name__}, expected list")
    media: list[Path] = []
    for item in prompt:
        if not isinstance(item, dict):
            continue
        media_type = str(item.get("type", "")).lower()
        if media_type == "video":
            raise RuntimeError("trace_eval_v1 contains image benchmarks only")
        if media_type != "image":
            continue
        for value in _flatten_media_value(item.get("value")):
            media.append(_resolve_media_path(value, working_directory))
    return media


def _verify_image(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing image: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError(f"empty image: {path}")

    from PIL import Image

    with Image.open(path) as image:
        image.load()
        width, height = image.size
        image_format = image.format
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid image dimensions for {path}: {width}x{height}")
    return {
        "path": str(path),
        "type": "image",
        "size_bytes": size,
        "sha256": _sha256_file(path),
        "width": width,
        "height": height,
        "format": image_format,
    }


def _verify_images(
    paths: Iterable[Path], workers: int
) -> tuple[list[dict[str, Any]], list[tuple[Path, Exception]]]:
    records: list[dict[str, Any]] = []
    failures: list[tuple[Path, Exception]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending = [(path, pool.submit(_verify_image, path)) for path in paths]
        for path, future in pending:
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append((path, exc))
    return records, failures


def _metadata_files(dataset: Any, alias: str, lmu_root: Path) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    pending = [dataset]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        data_path = getattr(current, "data_path", None)
        if data_path:
            candidates.append(Path(str(data_path)))
        dataset_map = getattr(current, "dataset_map", None)
        if isinstance(dataset_map, dict):
            pending.extend(dataset_map.values())
    candidates.extend((lmu_root / f"{alias}.tsv", lmu_root / f"{alias}_local.tsv"))

    files: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        files.append(
            {
                "path": str(candidate),
                "size_bytes": candidate.stat().st_size,
                "sha256": _sha256_file(candidate),
            }
        )
    if not files:
        raise RuntimeError(
            f"no persistent metadata file found for alias {alias} under {lmu_root}"
        )
    return files


def _materialize_dataset(
    *,
    key: str,
    alias: str,
    expected_rows: int,
    lmu_root: Path,
    workers: int,
    token: str | None,
) -> dict[str, Any]:
    """Materialize one exact trace_eval_v1 dataset and fail on stale media."""

    del token
    from benchmark_queue_lib import build_vlmeval_dataset, spec_by_key
    from trace_eval_suite import load_trace_eval_suite

    spec = spec_by_key(key)
    suite_benchmark = {item.key: item for item in load_trace_eval_suite().benchmarks}[
        key
    ]
    if alias != spec.alias or alias != suite_benchmark.official_alias:
        raise ValueError(f"non-canonical dataset alias for {key}: {alias!r}")
    if expected_rows != suite_benchmark.rows:
        raise ValueError(
            f"non-canonical row count for {key}: {expected_rows} != "
            f"{suite_benchmark.rows}"
        )

    started = _utc_now()
    dataset = build_vlmeval_dataset(spec)
    actual_rows = len(dataset.data)
    if actual_rows != expected_rows:
        raise RuntimeError(
            f"{key} row-count mismatch: downloaded {actual_rows}, "
            f"expected {expected_rows}"
        )

    working_directory = REPO_ROOT.resolve()
    unique_images: dict[str, Path] = {}
    row_image_paths: list[tuple[dict[str, Any], list[Path]]] = []
    media_references = 0
    rows_without_media: list[str] = []
    progress_every = max(100, min(500, actual_rows // 5 or 100))
    for ordinal, (_, row) in enumerate(dataset.data.iterrows(), start=1):
        row_images = _prompt_media(dataset, row, working_directory)
        row_mapping = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        row_image_paths.append((row_mapping, row_images))
        if not row_images:
            rows_without_media.append(str(row.get("index", ordinal - 1)))
        for path in row_images:
            unique_images[str(path)] = path
        media_references += len(row_images)
        if ordinal % progress_every == 0 or ordinal == actual_rows:
            print(
                f"[trace-eval-dataset:prompts] key={key} "
                f"rows={ordinal}/{actual_rows} unique_images={len(unique_images)}",
                flush=True,
            )
    if rows_without_media:
        raise RuntimeError(
            f"{key} has {len(rows_without_media)} rows without prompt images; "
            f"first indices={rows_without_media[:10]}"
        )

    media_files, media_failures = _verify_images(unique_images.values(), workers)
    if media_failures:
        details = "; ".join(
            f"{path}: {type(exc).__name__}: {exc}" for path, exc in media_failures[:10]
        )
        raise RuntimeError(f"{key} has {len(media_failures)} invalid images: {details}")
    media_files.sort(key=lambda item: item["path"])
    media_by_path = {
        str(Path(str(item["path"])).resolve()): item for item in media_files
    }

    row_media: list[dict[str, Any]] = []
    for ordinal, (row, prompt_images) in enumerate(row_image_paths):
        ordered_media = []
        for path in prompt_images:
            media = media_by_path[str(path.resolve())]
            ordered_media.append(
                {
                    "type": "image",
                    "path": str(path.resolve()),
                    "size_bytes": int(media["size_bytes"]),
                    "sha256": str(media["sha256"]),
                }
            )
        source_hash = source_row_sha256(row)
        media_hash = media_set_sha256(ordered_media)
        row_media.append(
            {
                "ordinal": ordinal,
                "index": str(row.get("index", ordinal)),
                "source_row_hash": source_hash,
                "media": ordered_media,
                "media_set_sha256": media_hash,
                "source_record_sha256": source_record_sha256(source_hash, media_hash),
            }
        )

    metadata_files = _metadata_files(dataset, alias, lmu_root)
    return {
        "status": "ready",
        "key": key,
        "alias": alias,
        "expected_rows": expected_rows,
        "rows": actual_rows,
        "media_references": media_references,
        "rows_without_media": 0,
        "unique_media": len(media_files),
        "media_bytes": sum(int(item["size_bytes"]) for item in media_files),
        "metadata_files": metadata_files,
        "media_files": media_files,
        "row_media": row_media,
        "dataset_snapshot_sha256": dataset_snapshot_sha256(row_media),
        "trace_normalization": getattr(dataset, "trace_normalization", {}),
        "started_at": started,
        "completed_at": _utc_now(),
    }


def _receipt_is_complete(
    receipt: dict[str, Any], *, alias: str, expected_rows: int
) -> bool:
    if not (
        receipt.get("status") == "ready"
        and receipt.get("alias") == alias
        and receipt.get("rows") == expected_rows
        and receipt.get("expected_rows") == expected_rows
    ):
        return False
    metadata_files = receipt.get("metadata_files")
    media_files = receipt.get("media_files")
    row_media = receipt.get("row_media")
    if (
        not isinstance(metadata_files, list)
        or not isinstance(media_files, list)
        or not media_files
        or not isinstance(row_media, list)
        or len(row_media) != expected_rows
        or not receipt.get("dataset_snapshot_sha256")
    ):
        return False
    for item in [*metadata_files, *media_files]:
        try:
            path = Path(str(item["path"]))
            size = int(item["size_bytes"])
        except (KeyError, TypeError, ValueError):
            return False
        if not path.is_file() or path.stat().st_size != size or size <= 0:
            return False
        expected_sha256 = str(item.get("sha256") or "")
        if len(expected_sha256) != 64 or _sha256_file(path) != expected_sha256:
            return False
    return dataset_snapshot_sha256(row_media) == receipt.get("dataset_snapshot_sha256")


def main() -> None:
    """Run the sole public trace_eval_v1 preparation interface."""

    from prepare_trace_eval_manifest import main as prepare_manifest

    prepare_manifest()


if __name__ == "__main__":
    main()
