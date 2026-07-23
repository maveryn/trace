#!/usr/bin/env python3
"""Prepare the pinned 2,000-row TRACE validation set for inference.

The source parquet embeds encoded image bytes.  This preparer writes those
bytes unchanged to content-addressed files and records a deterministic
manifest.  It intentionally does not decode/re-encode or resize the images.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
WORK_ROOT = REPO_ROOT / "rlvr" / "evaluation" / ".work" / "trace_validation"
DEFAULT_PARQUET = (
    WORK_ROOT
    / "inputs"
    / "data"
    / "validation"
    / "trace_rlvr_validation_iid_2000_all1000_seed1042.parquet"
)

DATASET_REPO_ID = "maveryn/trace"
DATASET_REVISION = "4e5b54361360296a855542b40cfd8b7f81b355fe"
DATASET_FILE = "data/validation/trace_rlvr_validation_iid_2000_all1000_seed1042.parquet"
DATASET_FILE_SHA256 = "051d1441b3f65f291841384962f16d6b5063f236072805f6c330f49afc02c4d1"
DATASET_FILE_SIZE_BYTES = 238_174_013
DATASET_MANIFEST_SHA256 = (
    "ff483e38c2dd2f618e6467950c0c2bab5048cca0d6bffa5b3d1fd45fcb4b0b69"
)
EXPECTED_ROWS = 2_000

HISTORICAL_DATASET_REVISION = "e317b746b258630682367cc6a9d87dedd195113c"
HISTORICAL_DATASET_FILE_SHA256 = (
    "0cb46bcf858ae3e9f39b88f60a24549a5de133976b9e8b74a45b4e6e4d699470"
)
HISTORICAL_DATASET_FILE_SIZE_BYTES = 238_177_343
HISTORICAL_DATASET_MANIFEST_SHA256 = (
    "825215fe98d1af3178c07449603d653cccb30a4eef63e4b9dc1cd45c3e43ce36"
)
HISTORICAL_SUITE_SHA256 = (
    "f9cccdcdddb6135c16d3a9d434f985b51e4105c07ff0c74a54a71a4dfe7c85c7"
)
DATASET_EQUIVALENCE_COMPARED_REVISION = "78f09b5482abc8e447a0a722cdf39e7d32f483c8"
DATASET_EQUIVALENCE_RECEIPT = REPO_ROOT / "rlvr" / "dataset_equivalence.v1.json"
DATASET_EQUIVALENCE_RECEIPT_SHA256 = (
    "40afcb59e0c67d7b7f47b78bb673a6c152320eb0656a990910ff52c634d43ae7"
)

MANIFEST_SCHEMA = "trace-validation-dataset-manifest-v1"
MANIFEST_NAME = "manifest.json"
MEDIA_STORAGE = "content-addressed-original-encoded-bytes"

_REQUIRED_COLUMNS = {
    "images",
    "image_sizes",
    "prompt_answer",
    "answer_gt",
    "instance_id",
    "domain",
    "task",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IMMUTABLE_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_EQUIVALENCE_COMPARED_COLUMNS = (
    "images",
    "image_sizes",
    "prompt_answer",
    "prompt_answer_and_annotation",
    "answer_gt",
    "annotation_gt",
    "reward_contract",
    "instance_id",
    "domain",
    "task",
    "scene_id",
    "query_id",
    "scene_variant",
    "trace_ref",
)
_EQUIVALENCE_ADVISORY_COLUMN = "trace_supervision_mode"
_EQUIVALENCE_TOP_LEVEL_KEYS = {
    "added_advisory_column",
    "aggregate",
    "compared_columns",
    "compared_current_inventory",
    "compared_current_revision",
    "comparison_program",
    "comparison_runtime",
    "current_schema",
    "original_schema",
    "original_training_revision",
    "parquet_files",
    "receipt_version",
    "repository",
    "splits",
}
_IMAGE_FORMATS = {
    "BMP": ("bmp", "image/bmp"),
    "GIF": ("gif", "image/gif"),
    "JPEG": ("jpg", "image/jpeg"),
    "PNG": ("png", "image/png"),
    "TIFF": ("tiff", "image/tiff"),
    "WEBP": ("webp", "image/webp"),
}


class DatasetEquivalenceError(RuntimeError):
    """Raised when the historical-to-reproduction bridge is not exact."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DatasetEquivalenceError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise DatasetEquivalenceError(f"non-finite JSON constant: {value}")


def _load_strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetEquivalenceError(
            f"cannot load dataset-equivalence receipt: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise DatasetEquivalenceError("dataset-equivalence receipt is not an object")
    return value


def _require_equivalence(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetEquivalenceError(message)


def _validate_normalized_identity(
    identity: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    expected_keys = {
        "repo_id",
        "revision",
        "split",
        "file",
        "file_sha256",
        "file_size_bytes",
        "row_count",
    }
    _require_equivalence(
        set(identity) == expected_keys,
        f"{label} dataset identity has unexpected fields",
    )
    normalized = dict(identity)
    _require_equivalence(
        isinstance(normalized["repo_id"], str) and bool(normalized["repo_id"]),
        f"{label} dataset repository is invalid",
    )
    _require_equivalence(
        isinstance(normalized["revision"], str)
        and _IMMUTABLE_REVISION_RE.fullmatch(normalized["revision"]) is not None,
        f"{label} dataset revision is not immutable",
    )
    _require_equivalence(
        normalized["split"] == "validation", f"{label} dataset split changed"
    )
    _require_equivalence(
        isinstance(normalized["file"], str) and bool(normalized["file"]),
        f"{label} dataset file is invalid",
    )
    _require_equivalence(
        isinstance(normalized["file_sha256"], str)
        and _SHA256_RE.fullmatch(normalized["file_sha256"]) is not None,
        f"{label} dataset file SHA-256 is invalid",
    )
    _require_equivalence(
        isinstance(normalized["file_size_bytes"], int)
        and not isinstance(normalized["file_size_bytes"], bool)
        and normalized["file_size_bytes"] > 0,
        f"{label} dataset file size is invalid",
    )
    _require_equivalence(
        isinstance(normalized["row_count"], int)
        and not isinstance(normalized["row_count"], bool)
        and normalized["row_count"] > 0,
        f"{label} dataset row count is invalid",
    )
    return normalized


def historical_dataset_identity() -> dict[str, Any]:
    """Return the normalized dataset identity frozen in ``suite.v1.json``."""

    return {
        "repo_id": DATASET_REPO_ID,
        "revision": HISTORICAL_DATASET_REVISION,
        "split": "validation",
        "file": DATASET_FILE,
        "file_sha256": HISTORICAL_DATASET_FILE_SHA256,
        "file_size_bytes": HISTORICAL_DATASET_FILE_SIZE_BYTES,
        "row_count": EXPECTED_ROWS,
    }


def reproduction_dataset_identity() -> dict[str, Any]:
    """Return the normalized post-squash dataset identity used for new runs."""

    return {
        "repo_id": DATASET_REPO_ID,
        "revision": DATASET_REVISION,
        "split": "validation",
        "file": DATASET_FILE,
        "file_sha256": DATASET_FILE_SHA256,
        "file_size_bytes": DATASET_FILE_SIZE_BYTES,
        "row_count": EXPECTED_ROWS,
    }


def normalized_suite_dataset_identity(
    dataset: Mapping[str, Any], *, file_size_bytes: int
) -> dict[str, Any]:
    """Project a suite dataset block into the bridge's common identity form."""

    return {
        "repo_id": dataset.get("repo_id"),
        "revision": dataset.get("revision"),
        "split": dataset.get("split"),
        "file": dataset.get("file"),
        "file_sha256": dataset.get("sha256"),
        "file_size_bytes": file_size_bytes,
        "row_count": dataset.get("rows"),
    }


def normalized_manifest_dataset_identity(
    dataset: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a prepared-manifest dataset block into the common identity form."""

    return {
        "repo_id": dataset.get("repo_id"),
        "revision": dataset.get("revision"),
        "split": dataset.get("split"),
        "file": dataset.get("file"),
        "file_sha256": dataset.get("file_sha256"),
        "file_size_bytes": dataset.get("file_size_bytes"),
        "row_count": dataset.get("row_count"),
    }


def validate_dataset_equivalence_bridge(
    receipt_path: Path,
    *,
    historical_identity: Mapping[str, Any],
    reproduction_identity: Mapping[str, Any],
    historical_manifest_sha256: str,
    reproduction_manifest_sha256: str,
    expected_compared_revision: str = DATASET_EQUIVALENCE_COMPARED_REVISION,
    expected_receipt_sha256: str | None = DATASET_EQUIVALENCE_RECEIPT_SHA256,
) -> dict[str, Any]:
    """Validate and return the historical-to-reproduction identity bridge.

    The reviewed receipt compares the historical dataset with the pre-squash
    current tree.  New campaigns use the immutable squash revision.  Equality
    is transitive because the compared and squash revisions are independently
    pinned here to the same validation Parquet hash and byte size.
    """

    receipt_path = receipt_path.expanduser().resolve()
    _require_equivalence(receipt_path.is_file(), f"missing receipt: {receipt_path}")
    receipt_sha256 = sha256_file(receipt_path)
    if expected_receipt_sha256 is not None:
        _require_equivalence(
            receipt_sha256 == expected_receipt_sha256,
            "dataset-equivalence receipt SHA-256 mismatch",
        )
    historical = _validate_normalized_identity(historical_identity, label="historical")
    reproduction = _validate_normalized_identity(
        reproduction_identity, label="reproduction"
    )
    for key in ("repo_id", "split", "file", "row_count"):
        _require_equivalence(
            historical[key] == reproduction[key],
            f"historical/reproduction dataset {key} mismatch",
        )
    _require_equivalence(
        _SHA256_RE.fullmatch(historical_manifest_sha256) is not None,
        "historical prepared-manifest SHA-256 is invalid",
    )
    _require_equivalence(
        _SHA256_RE.fullmatch(reproduction_manifest_sha256) is not None,
        "reproduction prepared-manifest SHA-256 is invalid",
    )
    _require_equivalence(
        _IMMUTABLE_REVISION_RE.fullmatch(expected_compared_revision) is not None,
        "compared dataset revision is not immutable",
    )

    receipt = _load_strict_json(receipt_path)
    _require_equivalence(
        set(receipt) == _EQUIVALENCE_TOP_LEVEL_KEYS,
        "dataset-equivalence receipt has unexpected top-level fields",
    )
    _require_equivalence(
        receipt.get("receipt_version") == "trace-dataset-equivalence-v1",
        "unsupported dataset-equivalence receipt version",
    )
    _require_equivalence(
        receipt.get("repository") == historical["repo_id"],
        "receipt repository does not match the suite dataset",
    )
    _require_equivalence(
        receipt.get("original_training_revision") == historical["revision"],
        "receipt historical revision does not match the suite",
    )
    _require_equivalence(
        receipt.get("compared_current_revision") == expected_compared_revision,
        "receipt compared revision mismatch",
    )
    _require_equivalence(
        receipt.get("added_advisory_column") == _EQUIVALENCE_ADVISORY_COLUMN,
        "receipt advisory-column contract changed",
    )
    _require_equivalence(
        receipt.get("compared_columns") == list(_EQUIVALENCE_COMPARED_COLUMNS),
        "receipt compared-column contract changed",
    )

    aggregate = receipt.get("aggregate")
    _require_equivalence(isinstance(aggregate, dict), "receipt aggregate is missing")
    _require_equivalence(aggregate.get("outcome") == "pass", "receipt did not pass")
    for key in (
        "current_differs_only_by_added_trace_supervision_mode",
        "existing_columns_value_equal",
        "existing_field_schemas_equal",
        "image_bytes_equal",
        "instance_ids_equal_and_unique",
        "row_counts_equal",
        "row_order_equal",
        "schema_metadata_only_adds_trace_supervision_mode",
        "task_distributions_equal",
        "train_validation_instance_ids_disjoint",
    ):
        _require_equivalence(
            aggregate.get(key) is True, f"receipt aggregate check failed: {key}"
        )

    original_schema = receipt.get("original_schema")
    current_schema = receipt.get("current_schema")
    _require_equivalence(
        isinstance(original_schema, list) and isinstance(current_schema, list),
        "receipt schemas are invalid",
    )
    _require_equivalence(
        [item.get("name") for item in original_schema if isinstance(item, dict)]
        == list(_EQUIVALENCE_COMPARED_COLUMNS),
        "receipt historical schema changed",
    )
    _require_equivalence(
        [item.get("name") for item in current_schema if isinstance(item, dict)]
        == [*_EQUIVALENCE_COMPARED_COLUMNS, _EQUIVALENCE_ADVISORY_COLUMN],
        "receipt current schema changed beyond the advisory column",
    )

    parquet_files = receipt.get("parquet_files")
    _require_equivalence(isinstance(parquet_files, list), "receipt files are invalid")
    matching_files = [
        row
        for row in parquet_files
        if isinstance(row, dict) and row.get("path") == historical["file"]
    ]
    _require_equivalence(
        len(matching_files) == 1, "receipt does not identify one validation Parquet"
    )
    validation_file = matching_files[0]
    _require_equivalence(
        validation_file.get("split") == "validation"
        and validation_file.get("rows") == historical["row_count"],
        "receipt validation-file shape changed",
    )
    for key in (
        "compared_columns_equal_in_row_order",
        "existing_field_schemas_equal",
        "image_bytes_equal_in_row_order",
        "schema_metadata_only_adds_trace_supervision_mode",
        "task_distribution_equal",
    ):
        _require_equivalence(
            validation_file.get(key) is True,
            f"receipt validation-file check failed: {key}",
        )
    _require_equivalence(
        validation_file.get("column_equality")
        == {key: True for key in _EQUIVALENCE_COMPARED_COLUMNS},
        "receipt validation columns are not all equal",
    )
    for label, identity in (("original", historical), ("current", reproduction)):
        file_identity = validation_file.get(label)
        _require_equivalence(
            isinstance(file_identity, dict), f"receipt {label} file identity is missing"
        )
        _require_equivalence(
            file_identity.get("bytes") == identity["file_size_bytes"]
            and file_identity.get("sha256") == identity["file_sha256"]
            and file_identity.get("lfs_sha256") == identity["file_sha256"]
            and file_identity.get("lfs_oid_matches_sha256") is True,
            f"receipt {label} validation-file identity mismatch",
        )

    inventory = receipt.get("compared_current_inventory")
    _require_equivalence(isinstance(inventory, dict), "receipt inventory is missing")
    _require_equivalence(
        inventory.get("revision") == expected_compared_revision,
        "receipt inventory revision mismatch",
    )
    inventory_files = inventory.get("files")
    _require_equivalence(isinstance(inventory_files, list), "receipt inventory invalid")
    inventory_matches = [
        row
        for row in inventory_files
        if isinstance(row, dict) and row.get("path") == reproduction["file"]
    ]
    _require_equivalence(
        len(inventory_matches) == 1,
        "receipt inventory does not identify one validation Parquet",
    )
    inventory_file = inventory_matches[0]
    _require_equivalence(
        inventory_file.get("size") == reproduction["file_size_bytes"]
        and inventory_file.get("lfs_sha256") == reproduction["file_sha256"],
        "receipt inventory validation-file identity mismatch",
    )

    splits = receipt.get("splits")
    validation_split = splits.get("validation") if isinstance(splits, dict) else None
    _require_equivalence(
        isinstance(validation_split, dict), "receipt validation split is missing"
    )
    _require_equivalence(
        validation_split.get("rows") == historical["row_count"]
        and validation_split.get("unique_instance_ids") == historical["row_count"]
        and validation_split.get("instance_ids_equal") is True
        and validation_split.get("task_distribution_equal") is True
        and validation_split.get("current_task_distribution_sha256")
        == validation_split.get("original_task_distribution_sha256"),
        "receipt validation split is not equivalent",
    )

    return {
        "schema_version": "trace-validation-dataset-equivalence-provenance-v1",
        "receipt_sha256": receipt_sha256,
        "historical_dataset": historical,
        "historical_prepared_manifest_sha256": historical_manifest_sha256,
        "compared_current_revision": expected_compared_revision,
        "reproduction_dataset": reproduction,
        "reproduction_prepared_manifest_sha256": reproduction_manifest_sha256,
        "equivalence": "exact_existing_columns_in_row_order",
        "added_advisory_column": _EQUIVALENCE_ADVISORY_COLUMN,
    }


def canonical_json_bytes(value: object) -> bytes:
    """Return the stable JSON encoding used by all validation artifacts."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Atomically replace ``path`` after syncing bytes and its directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.tmp.",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes(path, canonical_json_bytes(payload))


def _inspect_image(payload: bytes) -> dict[str, Any]:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image_format = str(image.format or "").upper()
            width, height = image.size
            image.verify()
    except Exception as exc:
        raise RuntimeError("embedded image bytes are not a valid image") from exc
    if image_format not in _IMAGE_FORMATS:
        raise RuntimeError(f"unsupported embedded image format: {image_format!r}")
    extension, mime_type = _IMAGE_FORMATS[image_format]
    return {
        "format": image_format,
        "extension": extension,
        "mime_type": mime_type,
        "width": int(width),
        "height": int(height),
    }


def _materialize_image(output_root: Path, payload: bytes) -> dict[str, Any]:
    inspection = _inspect_image(payload)
    digest = sha256_bytes(payload)
    relative_path = PurePosixPath(
        "media",
        "sha256",
        digest[:2],
        f"{digest}.{inspection['extension']}",
    )
    target = output_root.joinpath(*relative_path.parts)
    if target.exists():
        if target.stat().st_size != len(payload) or sha256_file(target) != digest:
            raise RuntimeError(
                f"corrupt content-addressed image already exists: {target}"
            )
    else:
        atomic_write_bytes(target, payload)
    return {
        "sha256": digest,
        "size_bytes": len(payload),
        "width": inspection["width"],
        "height": inspection["height"],
        "format": inspection["format"],
        "mime_type": inspection["mime_type"],
        "relative_path": relative_path.as_posix(),
    }


def _parse_answer_gt(raw_answer: Any, *, row_index: int) -> dict[str, Any]:
    if not isinstance(raw_answer, str):
        raise RuntimeError(f"row {row_index} answer_gt is not encoded JSON")
    try:
        answer = json.loads(raw_answer)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"row {row_index} answer_gt is invalid JSON") from exc
    if not isinstance(answer, dict):
        raise RuntimeError(f"row {row_index} answer_gt must be a JSON object")
    if not isinstance(answer.get("type"), str) or not answer["type"]:
        raise RuntimeError(f"row {row_index} answer_gt has no answer type")
    if "value" not in answer:
        raise RuntimeError(f"row {row_index} answer_gt has no value")
    canonical_json_bytes(answer)
    return answer


def prepare_dataset(
    parquet_path: Path,
    output_root: Path,
    *,
    dataset_repo_id: str = DATASET_REPO_ID,
    dataset_revision: str = DATASET_REVISION,
    dataset_file: str = DATASET_FILE,
    expected_file_sha256: str = DATASET_FILE_SHA256,
    expected_rows: int = EXPECTED_ROWS,
) -> Path:
    """Prepare a parquet and return the resulting manifest path.

    Non-default identity/count arguments exist for focused tests.  The CLI is
    deliberately pinned to the production constants above.
    """

    parquet_path = parquet_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not parquet_path.is_file():
        raise FileNotFoundError(f"missing TRACE validation parquet: {parquet_path}")
    source_sha256 = sha256_file(parquet_path)
    if source_sha256 != expected_file_sha256:
        raise RuntimeError(
            "TRACE validation parquet SHA-256 mismatch: "
            f"{source_sha256} != {expected_file_sha256}"
        )

    try:
        import pyarrow.parquet as pq
    except (
        ImportError
    ) as exc:  # pragma: no cover - dependency error is environment-specific.
        raise RuntimeError("pyarrow is required to prepare TRACE validation") from exc

    parquet = pq.ParquetFile(parquet_path)
    missing_columns = _REQUIRED_COLUMNS - set(parquet.schema_arrow.names)
    if missing_columns:
        raise RuntimeError(
            f"TRACE validation parquet is missing columns: {sorted(missing_columns)}"
        )
    if parquet.metadata.num_rows != expected_rows:
        raise RuntimeError(
            "TRACE validation row count mismatch: "
            f"{parquet.metadata.num_rows} != {expected_rows}"
        )

    rows: list[dict[str, Any]] = []
    seen_instance_ids: set[str] = set()
    columns = [
        "images",
        "image_sizes",
        "prompt_answer",
        "answer_gt",
        "instance_id",
        "domain",
        "task",
    ]
    for batch in parquet.iter_batches(columns=columns, batch_size=128):
        for source_row in batch.to_pylist():
            row_index = len(rows)
            instance_id = source_row.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                raise RuntimeError(f"row {row_index} has no instance_id")
            if instance_id in seen_instance_ids:
                raise RuntimeError(f"duplicate TRACE instance_id: {instance_id}")
            seen_instance_ids.add(instance_id)

            prompt_answer = source_row.get("prompt_answer")
            if not isinstance(prompt_answer, str) or not prompt_answer:
                raise RuntimeError(f"row {row_index} has no prompt_answer")
            domain = source_row.get("domain")
            task = source_row.get("task")
            if not isinstance(domain, str) or not domain:
                raise RuntimeError(f"row {row_index} has no domain")
            if not isinstance(task, str) or not task:
                raise RuntimeError(f"row {row_index} has no task")

            answer_gt = _parse_answer_gt(
                source_row.get("answer_gt"), row_index=row_index
            )
            embedded_images = source_row.get("images")
            declared_sizes = source_row.get("image_sizes")
            if not isinstance(embedded_images, list) or not embedded_images:
                raise RuntimeError(f"row {row_index} has no embedded images")
            if not isinstance(declared_sizes, list) or len(declared_sizes) != len(
                embedded_images
            ):
                raise RuntimeError(f"row {row_index} image_sizes do not match images")

            images: list[dict[str, Any]] = []
            for image_index, (embedded, declared_size) in enumerate(
                zip(embedded_images, declared_sizes)
            ):
                if not isinstance(embedded, dict):
                    raise RuntimeError(
                        f"row {row_index} image {image_index} has invalid storage"
                    )
                payload = embedded.get("bytes")
                if not isinstance(payload, bytes) or not payload:
                    raise RuntimeError(
                        f"row {row_index} image {image_index} has no embedded bytes"
                    )
                image_record = _materialize_image(output_root, payload)
                if not isinstance(declared_size, dict):
                    raise RuntimeError(
                        f"row {row_index} image {image_index} has invalid declared size"
                    )
                expected_size = (
                    int(declared_size.get("width", -1)),
                    int(declared_size.get("height", -1)),
                )
                actual_size = (image_record["width"], image_record["height"])
                if expected_size != actual_size:
                    raise RuntimeError(
                        f"row {row_index} image {image_index} size mismatch: "
                        f"{actual_size} != {expected_size}"
                    )
                image_record["image_index"] = image_index
                images.append(image_record)

            rows.append(
                {
                    "row_index": row_index,
                    "instance_id": instance_id,
                    "task": task,
                    "domain": domain,
                    "answer_type": answer_gt["type"],
                    "prompt_answer": prompt_answer,
                    "answer_gt": answer_gt,
                    "images": images,
                }
            )

    if len(rows) != expected_rows or len(seen_instance_ids) != expected_rows:
        raise RuntimeError(
            "TRACE validation preparation did not produce the exact unique row set: "
            f"rows={len(rows)} unique_instance_ids={len(seen_instance_ids)} "
            f"expected={expected_rows}"
        )

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset": {
            "repo_id": dataset_repo_id,
            "revision": dataset_revision,
            "config": "default",
            "split": "validation",
            "file": dataset_file,
            "file_sha256": source_sha256,
            "file_size_bytes": parquet_path.stat().st_size,
            "row_count": expected_rows,
        },
        "media": {
            "storage": MEDIA_STORAGE,
            "paths_relative_to": MANIFEST_NAME,
            "reencoded": False,
            "resized": False,
        },
        "rows": rows,
    }
    manifest_path = output_root / MANIFEST_NAME
    atomic_write_json(manifest_path, manifest)
    load_manifest(
        manifest_path,
        expected_rows=expected_rows,
        require_pinned=(
            dataset_repo_id == DATASET_REPO_ID
            and dataset_revision == DATASET_REVISION
            and dataset_file == DATASET_FILE
            and expected_file_sha256 == DATASET_FILE_SHA256
        ),
        verify_media=True,
    )
    return manifest_path


def _safe_media_path(manifest_path: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise RuntimeError("manifest image has no relative_path")
    pure_path = PurePosixPath(relative_path)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise RuntimeError(f"unsafe manifest media path: {relative_path!r}")
    root = manifest_path.parent.resolve()
    path = root.joinpath(*pure_path.parts).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError(f"manifest media escapes its root: {relative_path!r}")
    return path


def load_manifest(
    manifest_path: Path,
    *,
    expected_rows: int = EXPECTED_ROWS,
    require_pinned: bool = True,
    verify_media: bool = True,
) -> dict[str, Any]:
    """Load and fully validate a prepared TRACE validation manifest."""

    manifest_path = manifest_path.expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot load TRACE validation manifest: {manifest_path}"
        ) from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA
    ):
        raise RuntimeError(f"unsupported TRACE validation manifest: {manifest_path}")

    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise RuntimeError("TRACE validation manifest has no dataset identity")
    if require_pinned:
        expected_identity = {
            "repo_id": DATASET_REPO_ID,
            "revision": DATASET_REVISION,
            "config": "default",
            "split": "validation",
            "file": DATASET_FILE,
            "file_sha256": DATASET_FILE_SHA256,
            "file_size_bytes": DATASET_FILE_SIZE_BYTES,
            "row_count": EXPECTED_ROWS,
        }
        for key, expected in expected_identity.items():
            if dataset.get(key) != expected:
                raise RuntimeError(
                    f"TRACE validation manifest dataset {key} mismatch: "
                    f"{dataset.get(key)!r} != {expected!r}"
                )
        manifest_sha256 = sha256_file(manifest_path)
        if manifest_sha256 != DATASET_MANIFEST_SHA256:
            raise RuntimeError(
                "TRACE validation prepared-manifest SHA-256 mismatch: "
                f"{manifest_sha256} != {DATASET_MANIFEST_SHA256}"
            )
    if dataset.get("row_count") != expected_rows:
        raise RuntimeError(
            f"TRACE validation manifest row_count is not {expected_rows}: "
            f"{dataset.get('row_count')!r}"
        )

    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise RuntimeError(
            f"TRACE validation manifest must contain exactly {expected_rows} rows"
        )
    seen_instance_ids: set[str] = set()
    verified_media: dict[str, dict[str, Any]] = {}
    for expected_index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("row_index") != expected_index:
            raise RuntimeError(
                f"TRACE validation manifest row index mismatch at {expected_index}"
            )
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise RuntimeError(f"manifest row {expected_index} has no instance_id")
        if instance_id in seen_instance_ids:
            raise RuntimeError(f"duplicate manifest instance_id: {instance_id}")
        seen_instance_ids.add(instance_id)
        if not isinstance(row.get("prompt_answer"), str) or not row["prompt_answer"]:
            raise RuntimeError(f"manifest row {expected_index} has no prompt_answer")
        if not isinstance(row.get("task"), str) or not row["task"]:
            raise RuntimeError(f"manifest row {expected_index} has no task")
        if not isinstance(row.get("domain"), str) or not row["domain"]:
            raise RuntimeError(f"manifest row {expected_index} has no domain")
        answer_gt = row.get("answer_gt")
        if (
            not isinstance(answer_gt, dict)
            or "type" not in answer_gt
            or "value" not in answer_gt
        ):
            raise RuntimeError(f"manifest row {expected_index} has invalid answer_gt")
        if row.get("answer_type") != answer_gt.get("type"):
            raise RuntimeError(
                f"manifest row {expected_index} answer_type is inconsistent"
            )

        images = row.get("images")
        if not isinstance(images, list) or not images:
            raise RuntimeError(f"manifest row {expected_index} has no images")
        for expected_image_index, image_record in enumerate(images):
            if not isinstance(image_record, dict):
                raise RuntimeError(
                    f"manifest row {expected_index} has invalid image metadata"
                )
            if image_record.get("image_index") != expected_image_index:
                raise RuntimeError(
                    f"manifest row {expected_index} image index mismatch at "
                    f"{expected_image_index}"
                )
            digest = image_record.get("sha256")
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise RuntimeError(
                    f"manifest row {expected_index} has invalid image SHA-256"
                )
            if (
                not isinstance(image_record.get("size_bytes"), int)
                or image_record["size_bytes"] <= 0
            ):
                raise RuntimeError(
                    f"manifest row {expected_index} has invalid image byte size"
                )
            if (
                not isinstance(image_record.get("width"), int)
                or image_record["width"] <= 0
            ):
                raise RuntimeError(
                    f"manifest row {expected_index} has invalid image width"
                )
            if (
                not isinstance(image_record.get("height"), int)
                or image_record["height"] <= 0
            ):
                raise RuntimeError(
                    f"manifest row {expected_index} has invalid image height"
                )
            media_path = _safe_media_path(
                manifest_path, image_record.get("relative_path")
            )
            if not verify_media:
                continue
            if digest not in verified_media:
                if not media_path.is_file():
                    raise RuntimeError(f"missing prepared TRACE image: {media_path}")
                if media_path.stat().st_size != image_record["size_bytes"]:
                    raise RuntimeError(
                        f"prepared TRACE image size changed: {media_path}"
                    )
                if sha256_file(media_path) != digest:
                    raise RuntimeError(
                        f"prepared TRACE image content changed: {media_path}"
                    )
                inspection = _inspect_image(media_path.read_bytes())
                for key in ("width", "height", "format", "mime_type"):
                    if inspection[key] != image_record.get(key):
                        raise RuntimeError(
                            f"prepared TRACE image {key} changed: {media_path}"
                        )
                verified_media[digest] = image_record
            else:
                canonical = verified_media[digest]
                for key in (
                    "size_bytes",
                    "width",
                    "height",
                    "format",
                    "mime_type",
                    "relative_path",
                ):
                    if image_record.get(key) != canonical.get(key):
                        raise RuntimeError(
                            f"inconsistent metadata for prepared TRACE image {digest}"
                        )

    if len(seen_instance_ids) != expected_rows:
        raise RuntimeError(
            "TRACE validation manifest does not contain the exact unique instance set"
        )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARQUET,
        help="local copy of the pinned validation parquet",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="directory for manifest.json and original encoded media bytes",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an existing output-root/manifest.json without reading parquet",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.verify_only:
        manifest_path = args.output_root.expanduser().resolve() / MANIFEST_NAME
        manifest = load_manifest(manifest_path)
    else:
        manifest_path = prepare_dataset(args.parquet, args.output_root)
        manifest = load_manifest(manifest_path)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "rows": len(manifest["rows"]),
                "dataset_revision": manifest["dataset"]["revision"],
                "dataset_file_sha256": manifest["dataset"]["file_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
