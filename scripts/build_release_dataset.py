#!/usr/bin/env python3
"""Build or verify the TRACE training and IID validation dataset.

The tool writes a local artifact tree and performs no upload. Builds require a
clean, committed source tree. After a failure, partial output and work
directories remain available for inspection; reruns do not resume them
automatically.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import io
import json
import math
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, UnidentifiedImageError

# Keep the source-checkout script runnable before an editable installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from trace_tasks import __version__
from trace_tasks.core.build_presets import build_equal_split_all_tasks_config
from trace_tasks.core.builder import BuildError, build_dataset
from trace_tasks.core.canonical import canonical_json_bytes
from trace_tasks.core.hash_utils import blake3_file, blake3_hex
from trace_tasks.core.identity import compute_instance_id
from trace_tasks.core.reward_contracts import validate_reward_contract_payload
from trace_tasks.core.rlvr_export import (
    ExportedImageInfo,
    _build_prompt_columns,
    _scale_annotation_gt_for_export,
    export_trace_dataset_to_rlvr,
)
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.core.type_registry import DEFAULT_REGISTRY_PATH, load_type_registry
from trace_tasks.tasks.registry import list_default_task_ids

RECIPE_ID = "trace_rlvr_all1000_iid_v1"
RECIPE_SCHEMA_VERSION = "trace-release-dataset-recipe-v1"
SPLIT_MANIFEST_SCHEMA_VERSION = "trace-release-dataset-split-v1"
RELEASE_MANIFEST_SCHEMA_VERSION = "trace-release-dataset-manifest-v1"
EXPECTED_TASK_COUNT = 1_000
EXPECTED_TASK_IDS_SHA256 = (
    "1ef0419e23368309a961dd596d0cbc7212cdee64625d98d78cf1f6e654e47a98"
)
MAX_EMBEDDED_IMAGE_PIXELS = 1_280_000
ROW_ORDER_SEED = 20_260_711
DEFAULT_OUTPUT_DIR = Path("release-dataset")
DEFAULT_WORK_DIR = Path("release-dataset-work")
RELEASE_MANIFEST_NAME = "release_manifest.json"

HISTORICAL_DATASET_REPOSITORY = "maveryn/trace"
HISTORICAL_DATASET_REVISION = "e317b746b258630682367cc6a9d87dedd195113c"
REBUILD_RELATION = (
    "fresh_public_semantic_rebuild_of_frozen_recipe; "
    "not_byte_or_instance_id_identical_to_historical_paper_training_artifact"
)
IMAGE_INTEGRITY_THREAT_MODEL = (
    "The ordered semantic digest commits to decoded exported pixels. "
    "Receipted export provenance binds those pixels and export geometry "
    "to each train-record image id and original BLAKE3 content hash; "
    "original image bytes are intentionally omitted, so an artifact-only "
    "verifier cannot independently replay the resize transform."
)

SOURCE_INPUT_PATHS = (
    "src/trace_tasks",
    "scripts/build_release_dataset.py",
    "docs/task_catalog/catalog.v1.json",
    "pyproject.toml",
    "constraints/release.txt",
)
CONSTRAINT_PATHS = ("pyproject.toml", "constraints/release.txt")
CRITICAL_DISTRIBUTIONS = (
    "Pillow",
    "CairoSVG",
    "PyYAML",
    "numpy",
    "scipy",
    "networkx",
    "rfc8785",
    "blake3",
    "zstandard",
    "datasets",
    "pyarrow",
)
SUPPORTED_PYTHON_MINORS = ((3, 10), (3, 11), (3, 12))

SIDECAR_SCHEMA_VERSION = "trace-release-sidecars-v1"
EXPORT_PROVENANCE_RECORD_SCHEMA_VERSION = "trace-release-export-provenance-v1"
SIDECAR_REQUIRED_BASENAMES = (
    "build_report.json",
    "curriculum_index.jsonl.zst",
    "export_provenance.jsonl.zst",
    "train_instances.jsonl.zst",
    "validation_report.json",
)

TASK_CATALOG_PATH = REPO_ROOT / "docs" / "task_catalog" / "catalog.v1.json"
FROZEN_ANSWER_TYPES = {"integer", "number", "option_letter", "string"}
FROZEN_ANNOTATION_TYPES = {
    "bbox",
    "bbox_map",
    "bbox_sequence",
    "bbox_set",
    "bbox_set_map",
    "point",
    "point_map",
    "point_sequence",
    "point_set",
    "point_set_map",
    "segment",
    "segment_set",
}

_MACHINE_PATH_PREFIXES = (
    "/" + "home/",
    "/" + "root/",
    "/" + "workspace/",
    "/" + "tmp/",
    "/" + "Users/",
    "/" + "dev/shm",
)
_MACHINE_PATH_RE = re.compile(
    "(?:"
    + "|".join(re.escape(prefix) for prefix in _MACHINE_PATH_PREFIXES)
    + r"|[A-Za-z]:[\\\\/](?:Users|Documents and Settings)[\\\\/])"
)
_HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_BLAKE3_RE = re.compile(r"^blake3:[0-9a-f]{64}$")

VIEWER_COLUMNS = [
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
]

DROPPED_EXPORT_COLUMNS = [
    "uid",
    "prompt",
    "prompt_active",
    "prompt_answer_only",
    "prompt_mode",
    "difficulty_bin",
    "bucket_id_str",
    "image_sizes_original",
    "image_sizes_exported",
]


@dataclass(frozen=True)
class SplitRecipe:
    """One immutable split in the paper-release dataset recipe."""

    role: str
    dataset_name: str
    seed: int
    samples_per_task: int
    shard_count: int
    data_dir: str
    metadata_dir: str

    @property
    def rows(self) -> int:
        return EXPECTED_TASK_COUNT * self.samples_per_task

    @property
    def parquet_name(self) -> str:
        return f"{self.dataset_name}.parquet"

    @property
    def rows_per_shard(self) -> int:
        if self.rows % self.shard_count:
            raise ReleaseDatasetError(
                f"{self.role} rows are not divisible by its shard count"
            )
        return self.rows // self.shard_count


@dataclass(frozen=True)
class SplitInspection:
    """Semantic inventory recovered from one published split."""

    ordered_ids: tuple[str, ...]
    task_counts: Mapping[str, int]
    row_metadata: Mapping[str, tuple[str, str, str, str]]
    trace_refs: Mapping[str, Mapping[str, Any]]
    row_contracts: Mapping[
        str,
        tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    ]
    row_prompts: Mapping[str, tuple[str, str]]
    row_images: Mapping[str, tuple[Mapping[str, Any], ...]]

    def summary(self) -> dict[str, Any]:
        return {
            "row_count": len(self.ordered_ids),
            "unique_instance_count": len(set(self.ordered_ids)),
            "task_count": len(self.task_counts),
            "instance_id_order_sha256": _ordered_id_digest(self.ordered_ids),
            "viewer_image_semantic_sha256": _viewer_image_semantic_digest(
                self.ordered_ids,
                self.row_images,
            ),
        }


SPLIT_RECIPES = (
    SplitRecipe(
        role="train",
        dataset_name="trace_rlvr_train_64000_all1000_seed42",
        seed=42,
        samples_per_task=64,
        shard_count=16,
        data_dir="data/train",
        metadata_dir="metadata/train",
    ),
    SplitRecipe(
        role="validation_iid",
        dataset_name="trace_rlvr_validation_iid_2000_all1000_seed1042",
        seed=1_042,
        samples_per_task=2,
        shard_count=1,
        data_dir="data/validation",
        metadata_dir="metadata/validation_iid",
    ),
)


class ReleaseDatasetError(RuntimeError):
    """Raised when release construction or verification fails."""


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseDatasetError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseDatasetError(f"expected a JSON object in {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_contained_regular_file(path: Path, *, root: Path, context: str) -> None:
    root_resolved = root.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ReleaseDatasetError(f"{context} is outside {root}: {path}") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ReleaseDatasetError(f"{context} must not use a symlink: {current}")
    if not path.is_file():
        raise ReleaseDatasetError(f"{context} is not a regular file: {path}")
    resolved = path.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ReleaseDatasetError(f"{context} escapes {root}: {path}")


def _require_artifact_tree_safe(output_root: Path) -> None:
    if output_root.is_symlink():
        raise ReleaseDatasetError(
            f"release output directory must not be a symlink: {output_root}"
        )
    if not output_root.is_dir():
        raise ReleaseDatasetError(f"release output directory is missing: {output_root}")
    for path in output_root.rglob("*"):
        if path.is_symlink():
            raise ReleaseDatasetError(
                "release artifact tree must not contain symlinks: "
                f"{path.relative_to(output_root)}"
            )
        if path.is_file():
            _require_contained_regular_file(
                path,
                root=output_root,
                context=path.relative_to(output_root).as_posix(),
            )
            continue
        if not path.is_dir():
            raise ReleaseDatasetError(
                "release artifact tree contains a special filesystem node: "
                f"{path.relative_to(output_root)}"
            )


def _ordered_id_digest(instance_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for instance_id in instance_ids:
        digest.update(str(instance_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _viewer_image_semantic_digest(
    ordered_ids: Iterable[str],
    row_images: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    """Hash exported pixel content in published row order.

    The digest intentionally excludes PNG container metadata.  Byte hashes are
    retained separately in export provenance; this digest is the stable
    release-side commitment to the decoded RGBA pixels and geometry viewers
    actually consume.
    """

    digest = hashlib.sha256()
    for instance_id in ordered_ids:
        images = row_images.get(instance_id)
        if images is None:
            raise ReleaseDatasetError(
                f"missing viewer-image inventory for {instance_id}"
            )
        digest.update(instance_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical_json_bytes(list(images)))
        digest.update(b"\n")
    return digest.hexdigest()


def _relative_file_receipt(path: Path, *, root: Path) -> dict[str, Any]:
    _require_contained_regular_file(path, root=root, context="release receipt")
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _dependency_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _is_ignored_source_cache(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}


def _source_input_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for relative in SOURCE_INPUT_PATHS:
        candidate = repo_root / relative
        if candidate.is_symlink():
            raise ReleaseDatasetError(
                f"release source input must not be a symlink: {relative}"
            )
        if candidate.is_file():
            files.append(candidate)
            continue
        if candidate.is_dir():
            for path in candidate.rglob("*"):
                if path.is_symlink():
                    raise ReleaseDatasetError(
                        "release source input must not be a symlink: "
                        f"{path.relative_to(repo_root)}"
                    )
                if path.is_file() and not _is_ignored_source_cache(
                    path.relative_to(repo_root)
                ):
                    files.append(path)
            continue
        raise ReleaseDatasetError(f"release source input is missing: {relative}")
    return sorted(set(files))


def _source_tree_digest(paths: Sequence[Path], *, repo_root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(repo_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _constraint_receipts(repo_root: Path) -> dict[str, str]:
    receipts: dict[str, str] = {}
    for relative in CONSTRAINT_PATHS:
        path = repo_root / relative
        if not path.is_file() or path.is_symlink():
            raise ReleaseDatasetError(
                f"release constraint must be a regular file: {relative}"
            )
        receipts[relative] = _sha256_file(path)
    return receipts


def _source_provenance(
    repo_root: Path,
    *,
    require_clean: bool = False,
) -> dict[str, Any]:
    """Return content-addressed source provenance for generation inputs."""

    provenance: dict[str, Any] = {
        "trace_tasks_version": __version__,
        "source_input_paths": list(SOURCE_INPUT_PATHS),
    }
    try:
        top_level = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--",
                *SOURCE_INPUT_PATHS,
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
        tracked_raw = subprocess.run(
            ["git", "ls-files", "-z", "--", *SOURCE_INPUT_PATHS],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        if require_clean:
            raise ReleaseDatasetError(
                "canonical semantic rebuild requires an available Git checkout"
            ) from exc
        provenance["git_source_clean"] = False
        provenance["source_error"] = "git provenance unavailable"
        return provenance

    if Path(top_level).resolve() != repo_root.resolve():
        raise ReleaseDatasetError(
            f"release script repository root mismatch: {top_level} != {repo_root}"
        )
    if not _HEX_40_RE.fullmatch(revision):
        raise ReleaseDatasetError(
            f"invalid Git revision for release source: {revision!r}"
        )

    files = _source_input_files(repo_root)
    actual_relatives = {path.relative_to(repo_root).as_posix() for path in files}
    tracked_relatives = {
        item.decode("utf-8") for item in tracked_raw.split(b"\0") if item
    }
    untracked_inputs = sorted(actual_relatives - tracked_relatives)
    missing_tracked_inputs = sorted(tracked_relatives - actual_relatives)
    clean = not status and not untracked_inputs and not missing_tracked_inputs
    provenance.update(
        {
            "git_revision": revision,
            "git_source_clean": clean,
            "source_tree_sha256": _source_tree_digest(files, repo_root=repo_root),
            "source_file_count": len(files),
            "constraint_sha256": _constraint_receipts(repo_root),
        }
    )
    if not clean:
        provenance["dirty_entry_count"] = len(
            [item for item in status.split(b"\0") if item]
        )
        provenance["untracked_source_inputs"] = untracked_inputs
        provenance["missing_tracked_source_inputs"] = missing_tracked_inputs
    if require_clean and not clean:
        details = []
        if status:
            details.append("tracked or untracked Git changes")
        if untracked_inputs:
            details.append(f"untracked inputs={untracked_inputs[:5]}")
        if missing_tracked_inputs:
            details.append(f"missing inputs={missing_tracked_inputs[:5]}")
        raise ReleaseDatasetError(
            "canonical semantic rebuild requires clean committed source inputs: "
            + "; ".join(details)
        )
    return provenance


def _expected_dependency_pins(repo_root: Path) -> dict[str, str]:
    constraint_path = repo_root / "constraints" / "release.txt"
    try:
        lines = constraint_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReleaseDatasetError(
            f"cannot read release constraints: {constraint_path}"
        ) from exc
    parsed: dict[str, str] = {}
    critical_by_normalized = {
        name.lower().replace("_", "-"): name for name in CRITICAL_DISTRIBUTIONS
    }
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        name, raw_version = line.split("==", 1)
        normalized = name.strip().lower().replace("_", "-")
        canonical = critical_by_normalized.get(normalized)
        if canonical is None:
            continue
        version = raw_version.split(";", 1)[0].strip()
        if not version:
            raise ReleaseDatasetError(
                f"empty critical dependency pin in {constraint_path}: {raw_line}"
            )
        parsed[canonical] = version
    missing = sorted(set(CRITICAL_DISTRIBUTIONS) - set(parsed))
    if missing:
        raise ReleaseDatasetError(
            "release constraints do not freeze critical dependencies: "
            + ", ".join(missing)
        )
    return {name: parsed[name] for name in CRITICAL_DISTRIBUTIONS}


def _python_runtime_supported(*, implementation: str, version: str) -> bool:
    try:
        major, minor, *_rest = (int(part) for part in version.split("."))
    except ValueError:
        return False
    return implementation == "CPython" and (major, minor) in SUPPORTED_PYTHON_MINORS


def _environment_receipt(
    *,
    repo_root: Path,
    require_match: bool,
) -> dict[str, Any]:
    expected = _expected_dependency_pins(repo_root)
    distributions = {
        distribution: _dependency_version(distribution)
        for distribution in CRITICAL_DISTRIBUTIONS
    }
    missing = sorted(name for name, version in distributions.items() if not version)
    mismatches = {
        name: {"expected": expected[name], "actual": distributions.get(name)}
        for name in CRITICAL_DISTRIBUTIONS
        if distributions.get(name) != expected[name]
    }
    implementation = platform.python_implementation()
    python_version = platform.python_version()
    python_supported = _python_runtime_supported(
        implementation=implementation,
        version=python_version,
    )
    matches = not missing and not mismatches and python_supported
    if require_match and not matches:
        details: list[str] = []
        if not python_supported:
            details.append(
                f"unsupported runtime {implementation} {python_version}; "
                "expected CPython 3.10-3.12"
            )
        if mismatches:
            details.append(
                "dependency mismatches="
                + json.dumps(mismatches, sort_keys=True, separators=(",", ":"))
            )
        raise ReleaseDatasetError(
            "canonical semantic rebuild environment does not match release "
            "constraints: " + "; ".join(details)
        )
    return {
        "python_implementation": implementation,
        "python_version": python_version,
        "python_supported": python_supported,
        "distributions": distributions,
        "expected_distributions": expected,
        "dependency_mismatches": mismatches,
        "complete": not missing,
        "matches_release_constraints": matches,
    }


def _rebuild_contract() -> dict[str, Any]:
    return {
        "kind": "fresh_public_semantic_rebuild",
        "relation_to_historical_artifact": REBUILD_RELATION,
        "historical_paper_training_input": {
            "repository_id": HISTORICAL_DATASET_REPOSITORY,
            "revision": HISTORICAL_DATASET_REVISION,
        },
        "byte_identity_expected": False,
        "instance_id_identity_expected": False,
    }


def _assert_no_machine_paths(value: Any, *, context: str) -> None:
    """Reject machine-local absolute path strings from release payloads."""

    if isinstance(value, str):
        if _MACHINE_PATH_RE.search(value):
            raise ReleaseDatasetError(
                f"{context} contains a machine-local path: {value[:160]!r}"
            )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_no_machine_paths(child, context=f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_machine_paths(child, context=f"{context}[{index}]")


def _resolved_plan(
    *,
    output_dir: Path,
    work_dir: Path,
    workers: int,
    max_in_flight: int,
    parquet_cpu_count: int,
    task_ids: Sequence[str],
    source: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RECIPE_SCHEMA_VERSION,
        "recipe": RECIPE_ID,
        "rebuild_contract": _rebuild_contract(),
        "source": dict(source),
        "environment": dict(environment),
        "output_dir": str(output_dir),
        "work_dir": str(work_dir),
        "task_count": len(task_ids),
        "task_ids_sha256": _ordered_id_digest(task_ids),
        "expected_task_ids_sha256": EXPECTED_TASK_IDS_SHA256,
        "task_selection": "all public default tasks in sorted registry order",
        "task_sampling_policy": "equal_exact_count_per_task",
        "instance_version": "v0",
        "image_format": "png",
        "strict_repro": False,
        "max_attempts_per_instance": 100,
        "prompt_export_variant": "answer_and_annotation",
        "viewer_schema_profile": "trace_rlvr_viewer_v1",
        "viewer_columns": VIEWER_COLUMNS,
        "dropped_export_columns": DROPPED_EXPORT_COLUMNS,
        "image_storage_mode": "embedded_bytes",
        "max_embedded_image_pixels": MAX_EMBEDDED_IMAGE_PIXELS,
        "row_order": "deterministic_shuffle",
        "row_order_seed": ROW_ORDER_SEED,
        "automatic_resume": False,
        "failure_policy": (
            "partial output and work directories are retained for inspection; "
            "they are not release artifacts"
        ),
        "workers": workers,
        "max_in_flight": max_in_flight,
        "parquet_cpu_count": parquet_cpu_count,
        "splits": [
            {
                "role": recipe.role,
                "dataset_name": recipe.dataset_name,
                "rows": recipe.rows,
                "task_count": EXPECTED_TASK_COUNT,
                "samples_per_task": recipe.samples_per_task,
                "generation_seed": recipe.seed,
                "shard_count": recipe.shard_count,
                "rows_per_shard": recipe.rows_per_shard,
                "shard_paths": [
                    _shard_path(output_dir, recipe, shard_index)
                    .relative_to(output_dir)
                    .as_posix()
                    for shard_index in range(recipe.shard_count)
                ],
                "required_sidecars": list(SIDECAR_REQUIRED_BASENAMES),
            }
            for recipe in SPLIT_RECIPES
        ],
    }


def _require_canonical_task_registry(task_ids: Sequence[str]) -> None:
    if len(task_ids) != EXPECTED_TASK_COUNT:
        raise ReleaseDatasetError(
            "canonical release requires exactly "
            f"{EXPECTED_TASK_COUNT} default tasks, found {len(task_ids)}"
        )
    if list(task_ids) != sorted(task_ids):
        raise ReleaseDatasetError("default task registry must return sorted task ids")
    if len(set(task_ids)) != len(task_ids):
        raise ReleaseDatasetError("default task registry contains duplicate task ids")
    digest = _ordered_id_digest(task_ids)
    if digest != EXPECTED_TASK_IDS_SHA256:
        raise ReleaseDatasetError(
            "canonical task registry digest mismatch: "
            f"{digest} != {EXPECTED_TASK_IDS_SHA256}"
        )


def _require_new_path(path: Path, *, label: str) -> None:
    if path.exists():
        raise ReleaseDatasetError(
            f"{label} already exists: {path}; choose a new path to avoid overwriting data"
        )


def _validate_parallelism(
    *, workers: int, max_in_flight: int, parquet_cpu_count: int
) -> None:
    if workers < 0 or max_in_flight < 0 or parquet_cpu_count < 0:
        raise ReleaseDatasetError(
            "workers, max-in-flight, and parquet-cpu-count must be non-negative"
        )


def _require_separate_paths(output_dir: Path, work_dir: Path) -> None:
    output = output_dir.expanduser().resolve()
    work = work_dir.expanduser().resolve()
    if output == work or output in work.parents or work in output.parents:
        raise ReleaseDatasetError(
            "output directory and work directory must not overlap"
        )


def _sanitize_build_report(source: Path, destination: Path) -> None:
    report = _read_json(source)
    type_registry = report.get("type_registry")
    if isinstance(type_registry, dict) and "path" in type_registry:
        type_registry["path"] = "trace_tasks/configs/type_registry_v0.json"
    _assert_no_machine_paths(report, context="build_report")
    _write_json(destination, report)


def _sanitize_validation_report(source: Path, destination: Path) -> None:
    report = _read_json(source)
    build_context = report.get("build_context")
    if isinstance(build_context, dict):
        build_context["dataset_root"] = "."
        build_context.pop("temp_path", None)
        build_context.pop("timestamp", None)
    _assert_no_machine_paths(report, context="validation_report")
    _write_json(destination, report)


def _compress_zstd(source: Path, destination: Path) -> None:
    try:
        import zstandard as zstd
    except ImportError as exc:  # pragma: no cover - required project dependency.
        raise ReleaseDatasetError("zstandard is required to package sidecars") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstd.ZstdCompressor(level=6, threads=0)
    with source.open("rb") as input_handle, destination.open("wb") as output_handle:
        with compressor.stream_writer(output_handle) as writer:
            shutil.copyfileobj(input_handle, writer)


def _iter_plain_jsonl(path: Path, *, context: str) -> Iterable[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ReleaseDatasetError(f"cannot open {context}: {path}") from exc
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReleaseDatasetError(
                    f"{context} has invalid JSON on line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ReleaseDatasetError(
                    f"{context} line {line_number} is not a JSON object"
                )
            yield value


def _positive_image_sizes(
    value: Any,
    *,
    instance_id: str,
    field: str,
) -> list[tuple[int, int]]:
    if not isinstance(value, list) or not value:
        raise ReleaseDatasetError(f"{instance_id} has no {field}")
    sizes: list[tuple[int, int]] = []
    for raw_size in value:
        if not isinstance(raw_size, Mapping) or set(raw_size) != {
            "width",
            "height",
        }:
            raise ReleaseDatasetError(f"{instance_id} has an invalid {field} entry")
        width = raw_size.get("width")
        height = raw_size.get("height")
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            or not isinstance(height, int)
            or isinstance(height, bool)
            or height <= 0
        ):
            raise ReleaseDatasetError(f"{instance_id} has an invalid {field} entry")
        sizes.append((width, height))
    return sizes


def _write_export_provenance(
    *,
    raw_parquet: Path,
    dataset_root: Path,
    destination: Path,
) -> None:
    """Bind original image identities to exported bytes, pixels, and geometry."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - export dependency preflight.
        raise ReleaseDatasetError("pyarrow is required for export provenance") from exc

    train_records: dict[str, Mapping[str, Any]] = {}
    for record in _iter_plain_jsonl(
        dataset_root / "train_instances.jsonl",
        context="generated train instances",
    ):
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str) or instance_id in train_records:
            raise ReleaseDatasetError(
                "generated train instances have an invalid or duplicate instance id"
            )
        train_records[instance_id] = record

    schema_names = set(pq.read_schema(raw_parquet).names)
    required_columns = {
        "instance_id",
        "images",
        "image_sizes_original",
        "image_sizes_exported",
    }
    if not required_columns <= schema_names:
        raise ReleaseDatasetError(
            "raw RLVR export lacks columns required for image provenance: "
            f"{sorted(required_columns - schema_names)}"
        )

    seen_ids: set[str] = set()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output_handle:
        parquet = pq.ParquetFile(raw_parquet)
        for batch in parquet.iter_batches(
            columns=sorted(required_columns),
            batch_size=256,
        ):
            for row in batch.to_pylist():
                instance_id = row.get("instance_id")
                if (
                    not isinstance(instance_id, str)
                    or instance_id in seen_ids
                    or instance_id not in train_records
                ):
                    raise ReleaseDatasetError(
                        "raw RLVR export has an invalid, duplicate, or unknown instance id"
                    )
                seen_ids.add(instance_id)
                exported_receipts = _validate_images(
                    row.get("images"),
                    row.get("image_sizes_exported"),
                    instance_id=instance_id,
                )
                original_sizes = _positive_image_sizes(
                    row.get("image_sizes_original"),
                    instance_id=instance_id,
                    field="original image sizes",
                )
                train_images = train_records[instance_id].get("images")
                if (
                    not isinstance(train_images, list)
                    or len(train_images) != len(exported_receipts)
                    or len(original_sizes) != len(exported_receipts)
                ):
                    raise ReleaseDatasetError(
                        f"{instance_id} image cardinality differs across train and export"
                    )
                provenance_images: list[dict[str, Any]] = []
                for index, (train_image, original_size, exported_receipt) in enumerate(
                    zip(train_images, original_sizes, exported_receipts)
                ):
                    if not isinstance(train_image, Mapping):
                        raise ReleaseDatasetError(
                            f"{instance_id} train image {index} is invalid"
                        )
                    image_id = train_image.get("image_id")
                    image_format = train_image.get("format")
                    image_hash = train_image.get("image_hash")
                    raw_path = train_image.get("path")
                    if (
                        not isinstance(image_id, str)
                        or not image_id
                        or image_format != "png"
                        or not isinstance(image_hash, str)
                        or not _BLAKE3_RE.fullmatch(image_hash)
                        or not isinstance(raw_path, str)
                        or not raw_path
                        or Path(raw_path).is_absolute()
                    ):
                        raise ReleaseDatasetError(
                            f"{instance_id} train image {index} has invalid identity metadata"
                        )
                    source_image = dataset_root / raw_path
                    _require_contained_regular_file(
                        source_image,
                        root=dataset_root,
                        context=f"{instance_id} source image {index}",
                    )
                    if blake3_file(source_image) != image_hash:
                        raise ReleaseDatasetError(
                            f"{instance_id} source image {index} hash differs from train identity"
                        )
                    try:
                        with Image.open(source_image) as opened:
                            opened.load()
                            source_size = (int(opened.width), int(opened.height))
                            source_format = opened.format
                    except (OSError, UnidentifiedImageError) as exc:
                        raise ReleaseDatasetError(
                            f"{instance_id} source image {index} is invalid"
                        ) from exc
                    if source_format != "PNG" or source_size != original_size:
                        raise ReleaseDatasetError(
                            f"{instance_id} original image geometry or format differs from export"
                        )
                    provenance_images.append(
                        {
                            "index": index,
                            "image_id": image_id,
                            "format": image_format,
                            "source_image_hash": image_hash,
                            "original_width": original_size[0],
                            "original_height": original_size[1],
                            "exported_width": exported_receipt["width"],
                            "exported_height": exported_receipt["height"],
                            "exported_png_bytes_sha256": exported_receipt[
                                "png_bytes_sha256"
                            ],
                            "exported_rgba_pixels_sha256": exported_receipt[
                                "rgba_pixels_sha256"
                            ],
                        }
                    )
                record = {
                    "schema_version": EXPORT_PROVENANCE_RECORD_SCHEMA_VERSION,
                    "instance_id": instance_id,
                    "resize_policy": "pillow_lanczos_max_pixel_cap_v1",
                    "max_embedded_image_pixels": MAX_EMBEDDED_IMAGE_PIXELS,
                    "images": provenance_images,
                }
                output_handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
    if seen_ids != set(train_records):
        raise ReleaseDatasetError(
            "raw RLVR export membership differs from generated train instances"
        )


def _package_sidecars(
    dataset_root: Path,
    destination: Path,
    *,
    output_root: Path,
    export_provenance_path: Path,
) -> list[str]:
    destination.mkdir(parents=True, exist_ok=False)
    required = (
        "build_report.json",
        "validation_report.json",
        "curriculum_index.jsonl",
        "train_instances.jsonl",
    )
    missing = [name for name in required if not (dataset_root / name).is_file()]
    trace_files = sorted((dataset_root / "traces").glob("*.jsonl.zst"))
    if missing or not trace_files or not export_provenance_path.is_file():
        details = ", ".join(missing) if missing else "trace shards"
        if not export_provenance_path.is_file():
            details = "export provenance"
        raise ReleaseDatasetError(
            f"canonical dataset sidecars are incomplete under {dataset_root}: {details}"
        )

    _sanitize_build_report(
        dataset_root / "build_report.json", destination / "build_report.json"
    )
    _sanitize_validation_report(
        dataset_root / "validation_report.json",
        destination / "validation_report.json",
    )
    _compress_zstd(
        dataset_root / "curriculum_index.jsonl",
        destination / "curriculum_index.jsonl.zst",
    )
    _compress_zstd(
        export_provenance_path,
        destination / "export_provenance.jsonl.zst",
    )
    _compress_zstd(
        dataset_root / "train_instances.jsonl",
        destination / "train_instances.jsonl.zst",
    )
    traces_destination = destination / "traces"
    traces_destination.mkdir()
    for source in trace_files:
        shutil.copy2(source, traces_destination / source.name)

    packaged_files = sorted(path for path in destination.rglob("*") if path.is_file())
    included = [
        _relative_file_receipt(path, root=output_root) for path in packaged_files
    ]
    sidecar_manifest = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "included": included,
        "excluded": [
            {
                "path": "images/",
                "reason": "release parquet rows embed the exported image bytes",
            }
        ],
    }
    manifest_path = destination / "sidecar_manifest.json"
    _write_json(manifest_path, sidecar_manifest)
    return [item["path"] for item in included]


def _prepare_viewer_dataset(raw_parquet: Path, *, cache_dir: Path):
    try:
        import pyarrow.parquet as pq
        from datasets import load_dataset
    except ImportError as exc:
        raise ReleaseDatasetError(
            "dataset construction requires the export dependencies; "
            "install trace-tasks[export]"
        ) from exc

    source_columns = set(pq.read_schema(raw_parquet).names)
    required = [column for column in VIEWER_COLUMNS if column != "image_sizes"]
    missing = sorted(set(required) - source_columns)
    if missing:
        raise ReleaseDatasetError(
            f"raw RLVR export {raw_parquet} is missing columns: {missing}"
        )
    if not {"image_sizes", "image_sizes_exported"} & source_columns:
        raise ReleaseDatasetError(
            f"raw RLVR export {raw_parquet} has no exported image-size column"
        )

    dataset = load_dataset(
        "parquet",
        data_files=str(raw_parquet),
        split="train",
        cache_dir=str(cache_dir),
    )
    if "image_sizes" not in dataset.column_names:
        dataset = dataset.rename_column("image_sizes_exported", "image_sizes")
    return dataset.select_columns(VIEWER_COLUMNS).shuffle(seed=ROW_ORDER_SEED)


def _shard_path(output_root: Path, recipe: SplitRecipe, shard_index: int) -> Path:
    data_dir = output_root / recipe.data_dir
    if recipe.shard_count == 1:
        return data_dir / recipe.parquet_name
    stem = recipe.parquet_name.removesuffix(".parquet")
    return data_dir / (f"{stem}-{shard_index:05d}-of-{recipe.shard_count:05d}.parquet")


def _write_viewer_shards(
    dataset: Any,
    *,
    output_root: Path,
    recipe: SplitRecipe,
) -> list[Path]:
    paths: list[Path] = []
    for shard_index in range(recipe.shard_count):
        path = _shard_path(output_root, recipe, shard_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        shard = (
            dataset
            if recipe.shard_count == 1
            else dataset.shard(
                num_shards=recipe.shard_count,
                index=shard_index,
                contiguous=True,
            )
        )
        shard.to_parquet(str(path), batch_size=512)
        paths.append(path)
    return paths


def _validate_viewer_arrow_schema(path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ReleaseDatasetError(
            "dataset verification requires pyarrow; install trace-tasks[export]"
        ) from exc

    schema = pq.read_schema(path)
    if list(schema.names) != VIEWER_COLUMNS:
        raise ReleaseDatasetError(f"{path} has unexpected columns: {schema.names}")

    images = schema.field("images").type
    if not pa.types.is_list(images) or not pa.types.is_struct(images.value_type):
        raise ReleaseDatasetError(
            f"{path} images must be list<struct<bytes: binary, path: string>>"
        )
    image_struct = images.value_type
    if [field.name for field in image_struct] != ["bytes", "path"]:
        raise ReleaseDatasetError(f"{path} has an invalid embedded-image struct")
    if not pa.types.is_binary(
        image_struct.field("bytes").type
    ) or not pa.types.is_string(image_struct.field("path").type):
        raise ReleaseDatasetError(f"{path} has an invalid embedded-image struct")

    image_sizes = schema.field("image_sizes").type
    if not pa.types.is_list(image_sizes) or not pa.types.is_struct(
        image_sizes.value_type
    ):
        raise ReleaseDatasetError(
            f"{path} image_sizes must be list<struct<width: int64, height: int64>>"
        )
    size_struct = image_sizes.value_type
    if [field.name for field in size_struct] != ["width", "height"]:
        raise ReleaseDatasetError(f"{path} has an invalid image-size struct")
    if (
        size_struct.field("width").type != pa.int64()
        or size_struct.field("height").type != pa.int64()
    ):
        raise ReleaseDatasetError(f"{path} has an invalid image-size struct")

    for column in VIEWER_COLUMNS[2:]:
        if not pa.types.is_string(schema.field(column).type):
            raise ReleaseDatasetError(f"{path} column {column} must be a string")


def _iter_parquet_inventory(paths: Sequence[Path]) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ReleaseDatasetError(
            "dataset verification requires pyarrow; install trace-tasks[export]"
        ) from exc

    for path in paths:
        _validate_viewer_arrow_schema(path)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=VIEWER_COLUMNS, batch_size=256):
            yield from batch.to_pylist()


def _parse_json_object(value: Any, *, field: str, instance_id: str) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        raise ReleaseDatasetError(f"{instance_id} has an empty {field}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ReleaseDatasetError(f"{instance_id} has invalid JSON in {field}") from exc
    if not isinstance(parsed, dict):
        raise ReleaseDatasetError(f"{instance_id} {field} must be a JSON object")
    _assert_no_machine_paths(parsed, context=f"{instance_id}.{field}")
    return parsed


def _frozen_task_contracts(
    expected_task_ids: set[str],
) -> dict[str, tuple[str, str]]:
    """Load the reviewed per-task output contracts for the frozen recipe."""

    catalog = _read_json(TASK_CATALOG_PATH)
    if catalog.get("schema_version") != "trace_task_catalog_v1":
        raise ReleaseDatasetError("frozen task catalog has the wrong schema version")
    raw_tasks = catalog.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ReleaseDatasetError("frozen task catalog has no task entries")
    contracts: dict[str, tuple[str, str]] = {}
    for raw_task in raw_tasks:
        if not isinstance(raw_task, Mapping):
            raise ReleaseDatasetError("frozen task catalog has an invalid task entry")
        task_id = raw_task.get("task_id")
        answer_type = raw_task.get("answer_type")
        annotation_type = raw_task.get("annotation_type")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in contracts
            or answer_type not in FROZEN_ANSWER_TYPES
            or annotation_type not in FROZEN_ANNOTATION_TYPES
        ):
            raise ReleaseDatasetError(
                "frozen task catalog has an invalid or duplicate output contract"
            )
        contracts[task_id] = (str(answer_type), str(annotation_type))
    if set(contracts) != expected_task_ids:
        missing = sorted(expected_task_ids - set(contracts))
        extra = sorted(set(contracts) - expected_task_ids)
        raise ReleaseDatasetError(
            "frozen task catalog differs from the release task allowlist: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    return contracts


def _require_finite_number(value: Any, *, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ReleaseDatasetError(f"{context} must be a finite JSON number")
    return float(value)


def _validate_point_value(value: Any, *, context: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ReleaseDatasetError(f"{context} must be a two-coordinate point")
    for index, coordinate in enumerate(value):
        _require_finite_number(coordinate, context=f"{context}[{index}]")


def _validate_bbox_value(value: Any, *, context: str) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise ReleaseDatasetError(f"{context} must be a four-coordinate bbox")
    coordinates = [
        _require_finite_number(coordinate, context=f"{context}[{index}]")
        for index, coordinate in enumerate(value)
    ]
    # Scoring canonicalizes reversed endpoints, so order is intentionally not
    # constrained.  Both dimensions must nevertheless have non-zero extent.
    if coordinates[0] == coordinates[2] or coordinates[1] == coordinates[3]:
        raise ReleaseDatasetError(f"{context} must have non-zero bbox area")


def _validate_segment_value(value: Any, *, context: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ReleaseDatasetError(f"{context} must contain exactly two points")
    for index, point in enumerate(value):
        _validate_point_value(point, context=f"{context}[{index}]")


def _validate_value_list(
    value: Any,
    *,
    context: str,
    item_validator: Any,
) -> None:
    if not isinstance(value, list):
        raise ReleaseDatasetError(f"{context} must be a list")
    for index, item in enumerate(value):
        item_validator(item, context=f"{context}[{index}]")


def _validate_value_map(
    value: Any,
    *,
    context: str,
    item_validator: Any,
) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ReleaseDatasetError(f"{context} must be a non-empty object")
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ReleaseDatasetError(f"{context} keys must be non-empty strings")
        item_validator(item, context=f"{context}.{key}")


def _validate_answer_value(type_id: str, value: Any, *, context: str) -> None:
    if type_id == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ReleaseDatasetError(f"{context} must be an integer")
        return
    if type_id == "number":
        _require_finite_number(value, context=context)
        return
    if type_id == "string":
        if not isinstance(value, str):
            raise ReleaseDatasetError(f"{context} must be a string")
        return
    if type_id == "option_letter":
        if not isinstance(value, str) or re.fullmatch(r"[A-Z]", value) is None:
            raise ReleaseDatasetError(f"{context} must be one uppercase option letter")
        return
    raise ReleaseDatasetError(
        f"{context} uses answer type {type_id!r} outside the frozen recipe"
    )


def _validate_annotation_value(type_id: str, value: Any, *, context: str) -> None:
    primitive_validators = {
        "point": _validate_point_value,
        "bbox": _validate_bbox_value,
        "segment": _validate_segment_value,
    }
    primitive = primitive_validators.get(type_id)
    if primitive is not None:
        primitive(value, context=context)
        return

    list_types = {
        "point_set": _validate_point_value,
        "point_sequence": _validate_point_value,
        "bbox_set": _validate_bbox_value,
        "bbox_sequence": _validate_bbox_value,
        "segment_set": _validate_segment_value,
    }
    item_validator = list_types.get(type_id)
    if item_validator is not None:
        _validate_value_list(
            value,
            context=context,
            item_validator=item_validator,
        )
        return

    map_types = {
        "point_map": _validate_point_value,
        "bbox_map": _validate_bbox_value,
    }
    item_validator = map_types.get(type_id)
    if item_validator is not None:
        _validate_value_map(
            value,
            context=context,
            item_validator=item_validator,
        )
        return

    set_map_types = {
        "point_set_map": _validate_point_value,
        "bbox_set_map": _validate_bbox_value,
    }
    item_validator = set_map_types.get(type_id)
    if item_validator is not None:

        def validate_items(items: Any, *, context: str) -> None:
            _validate_value_list(
                items,
                context=context,
                item_validator=item_validator,
            )

        _validate_value_map(
            value,
            context=context,
            item_validator=validate_items,
        )
        return

    raise ReleaseDatasetError(
        f"{context} uses annotation type {type_id!r} outside the frozen recipe"
    )


def _validate_typed_value(
    payload: Mapping[str, Any],
    *,
    field: str,
    instance_id: str,
    registered_types: set[str],
) -> str:
    if set(payload) != {"type", "value"}:
        raise ReleaseDatasetError(
            f"{instance_id} {field} must contain exactly type and value"
        )
    type_id = payload.get("type")
    if not isinstance(type_id, str) or type_id not in registered_types:
        raise ReleaseDatasetError(
            f"{instance_id} {field} has unregistered type {type_id!r}"
        )
    context = f"{instance_id}.{field}.value"
    if field == "answer_gt":
        _validate_answer_value(type_id, payload.get("value"), context=context)
    elif field == "annotation_gt":
        _validate_annotation_value(type_id, payload.get("value"), context=context)
    else:  # pragma: no cover - private caller contract.
        raise ReleaseDatasetError(f"unsupported typed release field {field!r}")
    return type_id


def _validate_images(
    images: Any,
    image_sizes: Any,
    *,
    instance_id: str,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(images, list) or not images:
        raise ReleaseDatasetError(f"{instance_id} has no embedded images")
    if not isinstance(image_sizes, list) or not image_sizes:
        raise ReleaseDatasetError(f"{instance_id} has no exported image sizes")
    if len(images) != len(image_sizes):
        raise ReleaseDatasetError(
            f"{instance_id} image and image-size cardinalities differ"
        )
    receipts: list[dict[str, Any]] = []
    for image_index, (image, image_size) in enumerate(zip(images, image_sizes)):
        if not isinstance(image, Mapping):
            raise ReleaseDatasetError(f"{instance_id} has an invalid image record")
        image_bytes = image.get("bytes")
        if (
            not isinstance(image_bytes, (bytes, bytearray, memoryview))
            or not image_bytes
        ):
            raise ReleaseDatasetError(
                f"{instance_id} image {image_index} has no embedded bytes"
            )
        if image.get("path") not in {None, ""}:
            raise ReleaseDatasetError(
                f"{instance_id} image {image_index} contains a path instead of bytes"
            )
        if not isinstance(image_size, Mapping):
            raise ReleaseDatasetError(f"{instance_id} has an invalid image-size record")
        width = image_size.get("width")
        height = image_size.get("height")
        if (
            set(image_size) != {"width", "height"}
            or not isinstance(width, int)
            or isinstance(width, bool)
            or not isinstance(height, int)
            or isinstance(height, bool)
        ):
            raise ReleaseDatasetError(f"{instance_id} has an invalid image-size record")
        if width <= 0 or height <= 0:
            raise ReleaseDatasetError(
                f"{instance_id} has non-positive image dimensions"
            )
        try:
            payload = bytes(image_bytes)
            with Image.open(io.BytesIO(payload)) as opened:
                if opened.format != "PNG":
                    raise ReleaseDatasetError(
                        f"{instance_id} image {image_index} is not PNG"
                    )
                actual_size = (int(opened.width), int(opened.height))
                if actual_size != (width, height):
                    raise ReleaseDatasetError(
                        f"{instance_id} image {image_index} dimensions differ from image_sizes"
                    )
                if actual_size[0] * actual_size[1] > MAX_EMBEDDED_IMAGE_PIXELS:
                    raise ReleaseDatasetError(
                        f"{instance_id} exceeds the "
                        f"{MAX_EMBEDDED_IMAGE_PIXELS}-pixel cap"
                    )
                opened.verify()
            with Image.open(io.BytesIO(payload)) as loaded:
                loaded.load()
                rgba = loaded.convert("RGBA")
                semantic_digest = hashlib.sha256()
                semantic_digest.update(f"RGBA:{width}x{height}\0".encode("ascii"))
                semantic_digest.update(rgba.tobytes())
                receipts.append(
                    {
                        "index": image_index,
                        "width": width,
                        "height": height,
                        "png_bytes_sha256": hashlib.sha256(payload).hexdigest(),
                        "rgba_pixels_sha256": semantic_digest.hexdigest(),
                    }
                )
        except ReleaseDatasetError:
            raise
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise ReleaseDatasetError(
                f"{instance_id} image {image_index} is not a valid complete PNG"
            ) from exc
    return tuple(receipts)


def _validate_viewer_row(
    row: Mapping[str, Any],
    *,
    type_registry: Any,
) -> tuple[
    str,
    str,
    tuple[str, str, str, str],
    dict[str, Any],
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    tuple[str, str],
    tuple[dict[str, Any], ...],
]:
    raw_instance_id = row.get("instance_id")
    if not isinstance(raw_instance_id, str) or not _BLAKE3_RE.fullmatch(
        raw_instance_id
    ):
        raise ReleaseDatasetError(
            f"viewer row has invalid instance id {raw_instance_id!r}"
        )
    instance_id = raw_instance_id
    task_id = row.get("task")
    domain = row.get("domain")
    scene_id = row.get("scene_id")
    query_id = row.get("query_id")
    scene_variant = row.get("scene_variant")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (task_id, domain, scene_id, query_id)
    ):
        raise ReleaseDatasetError(
            f"{instance_id} has incomplete domain/task/scene/query taxonomy"
        )
    task_id = str(task_id)
    domain = str(domain)
    scene_id = str(scene_id)
    query_id = str(query_id)
    if not isinstance(scene_variant, str):
        raise ReleaseDatasetError(f"{instance_id} has an invalid scene_variant")
    for field_name, field_value in (
        ("domain", domain),
        ("task", task_id),
        ("scene_id", scene_id),
        ("query_id", query_id),
        ("scene_variant", scene_variant),
    ):
        _assert_no_machine_paths(
            field_value,
            context=f"{instance_id}.{field_name}",
        )
    taxonomy = resolve_task_taxonomy(
        task_id,
        source_domain=domain,
        source_scene_id=scene_id,
    )
    if taxonomy.domain != domain or taxonomy.scene_id != scene_id:
        raise ReleaseDatasetError(
            f"{instance_id} taxonomy does not match task {task_id}"
        )

    image_receipts = _validate_images(
        row.get("images"),
        row.get("image_sizes"),
        instance_id=instance_id,
    )
    image_count = len(image_receipts)
    prompts: list[str] = []
    for prompt_key in ("prompt_answer", "prompt_answer_and_annotation"):
        prompt = row.get(prompt_key)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ReleaseDatasetError(f"{instance_id} has an empty {prompt_key}")
        if prompt.count("<image>") != image_count:
            raise ReleaseDatasetError(
                f"{instance_id} {prompt_key} has the wrong image-marker count"
            )
        _assert_no_machine_paths(prompt, context=f"{instance_id}.{prompt_key}")
        prompts.append(prompt)

    answer_gt = _parse_json_object(
        row.get("answer_gt"), field="answer_gt", instance_id=instance_id
    )
    annotation_gt = _parse_json_object(
        row.get("annotation_gt"), field="annotation_gt", instance_id=instance_id
    )
    reward_contract = _parse_json_object(
        row.get("reward_contract"), field="reward_contract", instance_id=instance_id
    )
    trace_ref = _parse_json_object(
        row.get("trace_ref"), field="trace_ref", instance_id=instance_id
    )
    answer_type = _validate_typed_value(
        answer_gt,
        field="answer_gt",
        instance_id=instance_id,
        registered_types=set(type_registry.answer_types),
    )
    annotation_type = _validate_typed_value(
        annotation_gt,
        field="annotation_gt",
        instance_id=instance_id,
        registered_types=set(type_registry.annotation_types),
    )
    reward_error = validate_reward_contract_payload(
        reward_contract,
        answer_type=answer_type,
        annotation_type=annotation_type,
    )
    if reward_error:
        raise ReleaseDatasetError(
            f"{instance_id} has an invalid reward contract: {reward_error}"
        )
    if set(trace_ref) != {"shard_id", "line_index", "trace_record_hash"}:
        raise ReleaseDatasetError(f"{instance_id} has an invalid trace_ref shape")
    shard_id = trace_ref.get("shard_id")
    line_index = trace_ref.get("line_index")
    trace_hash = trace_ref.get("trace_record_hash")
    if (
        not isinstance(shard_id, str)
        or not shard_id
        or Path(shard_id).name != shard_id
        or not isinstance(line_index, int)
        or isinstance(line_index, bool)
        or line_index < 0
        or not isinstance(trace_hash, str)
        or not _BLAKE3_RE.fullmatch(trace_hash)
    ):
        raise ReleaseDatasetError(f"{instance_id} has an invalid trace_ref")
    return (
        instance_id,
        task_id,
        (task_id, domain, scene_id, query_id),
        trace_ref,
        (answer_gt, annotation_gt, reward_contract),
        (prompts[0], prompts[1]),
        image_receipts,
    )


def _inspect_split_rows(
    paths: Sequence[Path],
    *,
    recipe: SplitRecipe,
    expected_task_ids: set[str],
) -> SplitInspection:
    ordered_ids: list[str] = []
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    row_metadata: dict[str, tuple[str, str, str, str]] = {}
    trace_refs: dict[str, Mapping[str, Any]] = {}
    row_contracts: dict[
        str,
        tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    ] = {}
    row_prompts: dict[str, tuple[str, str]] = {}
    row_images: dict[str, tuple[Mapping[str, Any], ...]] = {}
    type_registry = load_type_registry()
    task_contracts = _frozen_task_contracts(expected_task_ids)
    for row in _iter_parquet_inventory(paths):
        (
            instance_id,
            task_id,
            metadata,
            trace_ref,
            contracts,
            prompts,
            images,
        ) = _validate_viewer_row(row, type_registry=type_registry)
        if instance_id in seen_ids:
            raise ReleaseDatasetError(
                f"{recipe.role} contains duplicate instance id {instance_id}"
            )
        seen_ids.add(instance_id)
        ordered_ids.append(instance_id)
        counts[task_id] += 1
        row_metadata[instance_id] = metadata
        trace_refs[instance_id] = trace_ref
        row_contracts[instance_id] = contracts
        row_prompts[instance_id] = prompts
        row_images[instance_id] = images
        if task_id not in task_contracts:
            raise ReleaseDatasetError(
                f"{instance_id} task is outside the frozen release catalog"
            )
        expected_answer_type, expected_annotation_type = task_contracts[task_id]
        if contracts[0].get("type") != expected_answer_type:
            raise ReleaseDatasetError(
                f"{instance_id} answer type differs from the frozen task catalog"
            )
        if contracts[1].get("type") != expected_annotation_type:
            raise ReleaseDatasetError(
                f"{instance_id} annotation type differs from the frozen task catalog"
            )

    if len(ordered_ids) != recipe.rows:
        raise ReleaseDatasetError(
            f"{recipe.role} has {len(ordered_ids)} rows, expected {recipe.rows}"
        )
    if set(counts) != expected_task_ids:
        missing = sorted(expected_task_ids - set(counts))
        extra = sorted(set(counts) - expected_task_ids)
        raise ReleaseDatasetError(
            f"{recipe.role} task coverage mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    wrong_counts = {
        task_id: count
        for task_id, count in counts.items()
        if count != recipe.samples_per_task
    }
    if wrong_counts:
        sample = sorted(wrong_counts.items())[:5]
        raise ReleaseDatasetError(
            f"{recipe.role} does not have {recipe.samples_per_task} rows per task: {sample}"
        )
    return SplitInspection(
        ordered_ids=tuple(ordered_ids),
        task_counts=dict(counts),
        row_metadata=row_metadata,
        trace_refs=trace_refs,
        row_contracts=row_contracts,
        row_prompts=row_prompts,
        row_images=row_images,
    )


def _write_split_manifest(
    *,
    output_root: Path,
    recipe: SplitRecipe,
    shard_paths: Sequence[Path],
    sidecar_paths: Sequence[str],
    expected_task_ids: set[str],
) -> Path:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - guarded during materialization.
        raise ReleaseDatasetError(
            "pyarrow is required to inspect release shards"
        ) from exc

    if len(shard_paths) != recipe.shard_count:
        raise ReleaseDatasetError(
            f"{recipe.role} produced {len(shard_paths)} shards, "
            f"expected {recipe.shard_count}"
        )
    expected_shard_paths = [
        _shard_path(output_root, recipe, shard_index)
        for shard_index in range(recipe.shard_count)
    ]
    if list(shard_paths) != expected_shard_paths:
        raise ReleaseDatasetError(f"{recipe.role} produced noncanonical shard paths")
    for path in shard_paths:
        _validate_viewer_arrow_schema(path)
        actual_rows = int(pq.ParquetFile(path).metadata.num_rows)
        if actual_rows != recipe.rows_per_shard:
            raise ReleaseDatasetError(
                f"{path} has {actual_rows} rows, expected {recipe.rows_per_shard}"
            )

    row_summary = _inspect_split_rows(
        shard_paths,
        recipe=recipe,
        expected_task_ids=expected_task_ids,
    )
    summary = row_summary.summary()
    shards = []
    for path in shard_paths:
        receipt = _relative_file_receipt(path, root=output_root)
        receipt["rows"] = int(pq.ParquetFile(path).metadata.num_rows)
        shards.append(receipt)
    manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "recipe": RECIPE_ID,
        "dataset_name": recipe.dataset_name,
        "split_role": recipe.role,
        "task_count": EXPECTED_TASK_COUNT,
        "rows": recipe.rows,
        "samples_per_task": recipe.samples_per_task,
        "generation_seed": recipe.seed,
        "schema_profile": "trace_rlvr_viewer_v1",
        "columns": VIEWER_COLUMNS,
        "dropped_columns": DROPPED_EXPORT_COLUMNS,
        "prompt_storage": (
            "RLVR export stores prompt_answer and prompt_answer_and_annotation."
        ),
        "image_storage_mode": "embedded_bytes",
        "max_embedded_image_pixels": MAX_EMBEDDED_IMAGE_PIXELS,
        "row_order": "deterministic_shuffle",
        "row_order_seed": ROW_ORDER_SEED,
        "instance_id_order_sha256": summary["instance_id_order_sha256"],
        "viewer_image_semantic_sha256": summary["viewer_image_semantic_sha256"],
        "image_integrity_threat_model": IMAGE_INTEGRITY_THREAT_MODEL,
        "shard_count": recipe.shard_count,
        "shards": shards,
        "sidecars": list(sidecar_paths),
    }
    manifest_path = (
        output_root / recipe.metadata_dir / f"{recipe.parquet_name}.manifest.json"
    )
    _write_json(manifest_path, manifest)
    return manifest_path


def _materialize_split(
    *,
    recipe: SplitRecipe,
    output_root: Path,
    work_root: Path,
    workers: int,
    max_in_flight: int,
    parquet_cpu_count: int,
    code_revision: str,
    expected_task_ids: set[str],
) -> Path:
    split_work_root = work_root / "builds" / recipe.role
    config = build_equal_split_all_tasks_config(
        output_root=str(split_work_root),
        dataset_name=recipe.dataset_name,
        num_instances=recipe.rows,
        instance_version="v0",
        image_format="png",
        strict_repro=False,
        max_attempts_per_instance=100,
        sampling_seed=recipe.seed,
        workers=workers,
        max_in_flight=max_in_flight,
    )
    if {task.task_id for task in config.tasks} != expected_task_ids:
        raise ReleaseDatasetError(
            f"{recipe.role} build config does not cover the canonical task registry"
        )
    try:
        dataset_root = build_dataset(config, code_hash=code_revision)
    except BuildError as exc:
        raise ReleaseDatasetError(f"{recipe.role} generation failed: {exc}") from exc

    raw_parquet = work_root / "exports" / recipe.parquet_name
    export = export_trace_dataset_to_rlvr(
        dataset_root,
        raw_parquet,
        output_format="parquet",
        prompt_variant="answer_and_annotation",
        image_path_mode="relative",
        image_storage_mode="embedded_bytes",
        parquet_cpu_count=parquet_cpu_count,
        max_embedded_image_pixels=MAX_EMBEDDED_IMAGE_PIXELS,
    )
    if export.row_count != recipe.rows:
        raise ReleaseDatasetError(
            f"{recipe.role} export has {export.row_count} rows, expected {recipe.rows}"
        )

    export_provenance_path = (
        work_root / "provenance" / recipe.role / "export_provenance.jsonl"
    )
    _write_export_provenance(
        raw_parquet=raw_parquet,
        dataset_root=dataset_root,
        destination=export_provenance_path,
    )
    sidecar_paths = _package_sidecars(
        dataset_root,
        output_root / "sidecars" / recipe.role,
        output_root=output_root,
        export_provenance_path=export_provenance_path,
    )
    viewer_dataset = _prepare_viewer_dataset(
        raw_parquet,
        cache_dir=work_root / "cache" / recipe.role,
    )
    if int(viewer_dataset.num_rows) != recipe.rows:
        raise ReleaseDatasetError(
            f"{recipe.role} viewer dataset has {viewer_dataset.num_rows} rows, "
            f"expected {recipe.rows}"
        )
    shard_paths = _write_viewer_shards(
        viewer_dataset,
        output_root=output_root,
        recipe=recipe,
    )
    return _write_split_manifest(
        output_root=output_root,
        recipe=recipe,
        shard_paths=shard_paths,
        sidecar_paths=sidecar_paths,
        expected_task_ids=expected_task_ids,
    )


def _release_file_receipts(output_root: Path) -> list[dict[str, Any]]:
    return [
        _relative_file_receipt(path, root=output_root)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != RELEASE_MANIFEST_NAME
    ]


def _canonical_release_artifact_paths(output_root: Path) -> set[str]:
    """Derive the complete artifact allowlist from the frozen recipe."""

    paths: set[str] = set()
    for recipe in SPLIT_RECIPES:
        paths.add(f"{recipe.metadata_dir}/{recipe.parquet_name}.manifest.json")
        paths.update(
            _shard_path(output_root, recipe, shard_index)
            .relative_to(output_root)
            .as_posix()
            for shard_index in range(recipe.shard_count)
        )
        paths.add(f"sidecars/{recipe.role}/sidecar_manifest.json")
        paths.update(
            f"sidecars/{recipe.role}/{basename}"
            for basename in SIDECAR_REQUIRED_BASENAMES
        )
        paths.add(f"sidecars/{recipe.role}/traces/trace_shard_0001.jsonl.zst")
    return paths


def build_release_dataset(
    *,
    output_dir: Path,
    work_dir: Path,
    workers: int,
    max_in_flight: int,
    parquet_cpu_count: int,
    keep_work_dir: bool,
    repo_root: Path,
) -> dict[str, Any]:
    """Build the two immutable release splits and return the verified manifest."""

    _validate_parallelism(
        workers=workers,
        max_in_flight=max_in_flight,
        parquet_cpu_count=parquet_cpu_count,
    )
    _require_separate_paths(output_dir, work_dir)
    task_ids = list_default_task_ids()
    _require_canonical_task_registry(task_ids)
    _require_new_path(output_dir, label="output directory")
    _require_new_path(work_dir, label="work directory")
    source = _source_provenance(repo_root, require_clean=True)
    environment = _environment_receipt(repo_root=repo_root, require_match=True)
    output_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True)

    code_revision = str(source["git_revision"])
    split_manifests: list[str] = []
    success = False
    try:
        for recipe in SPLIT_RECIPES:
            split_manifest = _materialize_split(
                recipe=recipe,
                output_root=output_dir,
                work_root=work_dir,
                workers=workers,
                max_in_flight=max_in_flight,
                parquet_cpu_count=parquet_cpu_count,
                code_revision=code_revision,
                expected_task_ids=set(task_ids),
            )
            split_manifests.append(split_manifest.relative_to(output_dir).as_posix())

        release_manifest = {
            "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
            "recipe": RECIPE_ID,
            "rebuild_contract": _rebuild_contract(),
            "source": source,
            "runtime": {
                "python_implementation": environment["python_implementation"],
                "python_version": environment["python_version"],
            },
            "dependencies": environment["distributions"],
            "task_count": EXPECTED_TASK_COUNT,
            "task_ids_sha256": _ordered_id_digest(task_ids),
            "task_ids": task_ids,
            "split_manifests": split_manifests,
            "files": _release_file_receipts(output_dir),
        }
        _write_json(output_dir / RELEASE_MANIFEST_NAME, release_manifest)
        verify_release_dataset(output_dir, expected_task_ids=set(task_ids))
        success = True
        return _read_json(output_dir / RELEASE_MANIFEST_NAME)
    finally:
        if success and not keep_work_dir:
            shutil.rmtree(work_dir)


def _require_manifest_value(
    manifest: Mapping[str, Any],
    key: str,
    expected: Any,
    *,
    context: str,
) -> None:
    actual = manifest.get(key)
    if actual != expected:
        raise ReleaseDatasetError(
            f"{context} has {key}={actual!r}, expected {expected!r}"
        )


def _verify_file_receipts(
    receipts: Any,
    *,
    output_root: Path,
    context: str,
) -> None:
    if not isinstance(receipts, list) or not receipts:
        raise ReleaseDatasetError(f"{context} has no file receipts")
    seen: set[str] = set()
    for raw_receipt in receipts:
        if not isinstance(raw_receipt, Mapping):
            raise ReleaseDatasetError(f"{context} contains an invalid file receipt")
        relative_path = str(raw_receipt.get("path", ""))
        path = Path(relative_path)
        if not relative_path or path.is_absolute() or ".." in path.parts:
            raise ReleaseDatasetError(
                f"{context} contains unsafe artifact path {relative_path!r}"
            )
        if relative_path in seen:
            raise ReleaseDatasetError(
                f"{context} contains duplicate artifact path {relative_path}"
            )
        seen.add(relative_path)
        artifact = output_root / path
        _require_contained_regular_file(
            artifact,
            root=output_root,
            context=f"{context}:{relative_path}",
        )
        expected_bytes = int(raw_receipt.get("bytes", -1))
        if artifact.stat().st_size != expected_bytes:
            raise ReleaseDatasetError(
                f"size mismatch for {relative_path}: "
                f"{artifact.stat().st_size} != {expected_bytes}"
            )
        expected_sha256 = str(raw_receipt.get("sha256", ""))
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ReleaseDatasetError(
                f"invalid SHA-256 receipt for {relative_path}: {expected_sha256!r}"
            )
        actual_sha256 = _sha256_file(artifact)
        if actual_sha256 != expected_sha256:
            raise ReleaseDatasetError(
                f"SHA-256 mismatch for {relative_path}: "
                f"{actual_sha256} != {expected_sha256}"
            )


def _verify_release_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_manifest_value(
        manifest,
        "rebuild_contract",
        _rebuild_contract(),
        context=RELEASE_MANIFEST_NAME,
    )
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ReleaseDatasetError(f"{RELEASE_MANIFEST_NAME} has no source receipt")
    if source.get("trace_tasks_version") != __version__:
        raise ReleaseDatasetError(
            f"{RELEASE_MANIFEST_NAME} has the wrong trace_tasks version"
        )
    revision = source.get("git_revision")
    if not isinstance(revision, str) or not _HEX_40_RE.fullmatch(revision):
        raise ReleaseDatasetError(
            f"{RELEASE_MANIFEST_NAME} has an invalid source revision"
        )
    if source.get("git_source_clean") is not True:
        raise ReleaseDatasetError(
            f"{RELEASE_MANIFEST_NAME} was not built from clean source inputs"
        )
    tree_hash = source.get("source_tree_sha256")
    if not isinstance(tree_hash, str) or not _HEX_64_RE.fullmatch(tree_hash):
        raise ReleaseDatasetError(
            f"{RELEASE_MANIFEST_NAME} has an invalid source-tree receipt"
        )
    if (
        not isinstance(source.get("source_file_count"), int)
        or int(source["source_file_count"]) <= 0
    ):
        raise ReleaseDatasetError(
            f"{RELEASE_MANIFEST_NAME} has an invalid source file count"
        )
    if source.get("source_input_paths") != list(SOURCE_INPUT_PATHS):
        raise ReleaseDatasetError(
            f"{RELEASE_MANIFEST_NAME} has the wrong source-input boundary"
        )
    constraints = source.get("constraint_sha256")
    if not isinstance(constraints, Mapping) or set(constraints) != set(
        CONSTRAINT_PATHS
    ):
        raise ReleaseDatasetError(
            f"{RELEASE_MANIFEST_NAME} has incomplete constraint receipts"
        )
    for path, digest in constraints.items():
        if not isinstance(digest, str) or not _HEX_64_RE.fullmatch(digest):
            raise ReleaseDatasetError(
                f"{RELEASE_MANIFEST_NAME} has an invalid constraint receipt for {path}"
            )
    expected_constraint_receipts = _constraint_receipts(REPO_ROOT)
    if dict(constraints) != expected_constraint_receipts:
        raise ReleaseDatasetError(
            f"{RELEASE_MANIFEST_NAME} constraint receipts do not match the frozen release files"
        )

    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ReleaseDatasetError(f"{RELEASE_MANIFEST_NAME} has no runtime receipt")
    for key in ("python_implementation", "python_version"):
        if not isinstance(runtime.get(key), str) or not str(runtime[key]).strip():
            raise ReleaseDatasetError(
                f"{RELEASE_MANIFEST_NAME} has an invalid runtime {key}"
            )
    if not _python_runtime_supported(
        implementation=str(runtime["python_implementation"]),
        version=str(runtime["python_version"]),
    ):
        raise ReleaseDatasetError(
            f"{RELEASE_MANIFEST_NAME} records an unsupported Python runtime"
        )
    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        CRITICAL_DISTRIBUTIONS
    ):
        raise ReleaseDatasetError(
            f"{RELEASE_MANIFEST_NAME} has incomplete dependency receipts"
        )
    expected_dependencies = _expected_dependency_pins(REPO_ROOT)
    for distribution, version in dependencies.items():
        if (
            not isinstance(version, str)
            or version != expected_dependencies[distribution]
        ):
            raise ReleaseDatasetError(
                f"{RELEASE_MANIFEST_NAME} has the wrong frozen version for {distribution}"
            )
    _assert_no_machine_paths(manifest, context=RELEASE_MANIFEST_NAME)
    return dict(source)


def _iter_zstd_jsonl(path: Path, *, context: str) -> Iterable[dict[str, Any]]:
    try:
        import zstandard as zstd
    except ImportError as exc:  # pragma: no cover - required project dependency.
        raise ReleaseDatasetError("zstandard is required to verify sidecars") from exc

    with path.open("rb") as compressed:
        with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
                for line_number, raw_line in enumerate(text_stream, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ReleaseDatasetError(
                            f"{context} has invalid JSON on line {line_number}"
                        ) from exc
                    if not isinstance(value, dict):
                        raise ReleaseDatasetError(
                            f"{context} line {line_number} is not a JSON object"
                        )
                    _assert_no_machine_paths(
                        value,
                        context=f"{context}[{line_number}]",
                    )
                    yield value


def _require_exact_task_counts(
    raw_counts: Any,
    *,
    recipe: SplitRecipe,
    expected_task_ids: set[str],
    context: str,
) -> None:
    if not isinstance(raw_counts, Mapping):
        raise ReleaseDatasetError(f"{context} has no authoritative task counts")
    counts: dict[str, int] = {}
    for task_id, count in raw_counts.items():
        if not isinstance(count, int) or isinstance(count, bool):
            raise ReleaseDatasetError(f"{context} has a non-integer task count")
        counts[str(task_id)] = count
    if set(counts) != expected_task_ids or any(
        count != recipe.samples_per_task for count in counts.values()
    ):
        raise ReleaseDatasetError(
            f"{context} does not contain the canonical per-task counts"
        )


def _verify_sidecars(
    *,
    output_root: Path,
    recipe: SplitRecipe,
    inspection: SplitInspection,
    expected_task_ids: set[str],
    source_revision: str,
    split_sidecars: Any,
) -> None:
    sidecar_root = output_root / "sidecars" / recipe.role
    build_report_path = sidecar_root / "build_report.json"
    validation_report_path = sidecar_root / "validation_report.json"
    sidecar_manifest_path = sidecar_root / "sidecar_manifest.json"
    for path in (build_report_path, validation_report_path, sidecar_manifest_path):
        _require_contained_regular_file(
            path,
            root=output_root,
            context=path.relative_to(output_root).as_posix(),
        )

    build_report = _read_json(build_report_path)
    validation_report = _read_json(validation_report_path)
    _assert_no_machine_paths(build_report, context=f"{recipe.role}.build_report")
    _assert_no_machine_paths(
        validation_report,
        context=f"{recipe.role}.validation_report",
    )
    _require_exact_task_counts(
        build_report.get("accepted_counts_by_task"),
        recipe=recipe,
        expected_task_ids=expected_task_ids,
        context=f"{recipe.role}.build_report",
    )
    if int(validation_report.get("total_errors", -1)) != 0:
        raise ReleaseDatasetError(
            f"{recipe.role} validation report is missing or has errors"
        )
    build_context = validation_report.get("build_context")
    if (
        not isinstance(build_context, Mapping)
        or build_context.get("dataset_root") != "."
    ):
        raise ReleaseDatasetError(f"{recipe.role} validation report is not portable")
    code_provenance = build_report.get("code_provenance")
    if (
        not isinstance(code_provenance, Mapping)
        or code_provenance.get("code_hash") != source_revision
        or code_provenance.get("identity_input") is not True
    ):
        raise ReleaseDatasetError(
            f"{recipe.role} build-report source or identity provenance mismatch"
        )
    type_registry = build_report.get("type_registry")
    current_type_registry = load_type_registry(DEFAULT_REGISTRY_PATH)
    expected_type_registry = {
        "type_registry_version": current_type_registry.version,
        "path": "trace_tasks/configs/type_registry_v0.json",
        "hash": blake3_file(DEFAULT_REGISTRY_PATH),
    }
    if type_registry != expected_type_registry:
        raise ReleaseDatasetError(
            f"{recipe.role} build report has the wrong frozen type registry"
        )

    canonical_trace_shard = {
        "shard_id": "trace_shard_0001.jsonl.zst",
        "path": "traces/trace_shard_0001.jsonl.zst",
        "record_count": recipe.rows,
    }
    trace_manifest = build_report.get("trace_shard_manifest")
    if trace_manifest != {"shards": [canonical_trace_shard]}:
        raise ReleaseDatasetError(
            f"{recipe.role} trace-shard manifest is not the canonical single shard"
        )
    trace_shards = [("trace_shard_0001.jsonl.zst", recipe.rows)]

    expected_included = sorted(
        [
            f"sidecars/{recipe.role}/{basename}"
            for basename in SIDECAR_REQUIRED_BASENAMES
        ]
        + [f"sidecars/{recipe.role}/traces/{shard_id}" for shard_id, _ in trace_shards]
    )
    sidecar_manifest = _read_json(sidecar_manifest_path)
    _require_manifest_value(
        sidecar_manifest,
        "schema_version",
        SIDECAR_SCHEMA_VERSION,
        context=sidecar_manifest_path.relative_to(output_root).as_posix(),
    )
    _require_manifest_value(
        sidecar_manifest,
        "excluded",
        [
            {
                "path": "images/",
                "reason": "release parquet rows embed the exported image bytes",
            }
        ],
        context=sidecar_manifest_path.relative_to(output_root).as_posix(),
    )
    included = sidecar_manifest.get("included")
    _verify_file_receipts(
        included,
        output_root=output_root,
        context=sidecar_manifest_path.relative_to(output_root).as_posix(),
    )
    included_paths = [str(item["path"]) for item in included]
    if included_paths != expected_included:
        raise ReleaseDatasetError(
            f"{recipe.role} sidecar inventory is not the canonical complete set"
        )
    if split_sidecars != expected_included:
        raise ReleaseDatasetError(
            f"{recipe.role} split manifest sidecars differ from the complete set"
        )

    parquet_ids = set(inspection.ordered_ids)
    provenance_path = sidecar_root / "export_provenance.jsonl.zst"
    export_provenance: dict[str, tuple[Mapping[str, Any], ...]] = {}
    expected_provenance_image_keys = {
        "index",
        "image_id",
        "format",
        "source_image_hash",
        "original_width",
        "original_height",
        "exported_width",
        "exported_height",
        "exported_png_bytes_sha256",
        "exported_rgba_pixels_sha256",
    }
    for record in _iter_zstd_jsonl(
        provenance_path,
        context=f"{recipe.role}.export_provenance",
    ):
        if set(record) != {
            "schema_version",
            "instance_id",
            "resize_policy",
            "max_embedded_image_pixels",
            "images",
        }:
            raise ReleaseDatasetError(
                f"{recipe.role} export provenance has an invalid record shape"
            )
        instance_id = record.get("instance_id")
        raw_images = record.get("images")
        if (
            record.get("schema_version") != EXPORT_PROVENANCE_RECORD_SCHEMA_VERSION
            or record.get("resize_policy") != "pillow_lanczos_max_pixel_cap_v1"
            or record.get("max_embedded_image_pixels") != MAX_EMBEDDED_IMAGE_PIXELS
            or not isinstance(instance_id, str)
            or instance_id in export_provenance
            or not isinstance(raw_images, list)
            or not raw_images
        ):
            raise ReleaseDatasetError(
                f"{recipe.role} export provenance has invalid release metadata"
            )
        viewer_images = inspection.row_images.get(instance_id)
        if viewer_images is None or len(raw_images) != len(viewer_images):
            raise ReleaseDatasetError(
                f"{recipe.role} export provenance image membership differs for "
                f"{instance_id}"
            )
        validated_images: list[Mapping[str, Any]] = []
        for index, (raw_image, viewer_image) in enumerate(
            zip(raw_images, viewer_images)
        ):
            if not isinstance(raw_image, Mapping) or set(raw_image) != (
                expected_provenance_image_keys
            ):
                raise ReleaseDatasetError(
                    f"{recipe.role} export provenance image {index} is invalid"
                )
            integer_fields = (
                "index",
                "original_width",
                "original_height",
                "exported_width",
                "exported_height",
            )
            if any(
                not isinstance(raw_image.get(field), int)
                or isinstance(raw_image.get(field), bool)
                for field in integer_fields
            ):
                raise ReleaseDatasetError(
                    f"{recipe.role} export provenance image geometry is invalid"
                )
            if (
                raw_image.get("index") != index
                or int(raw_image["original_width"]) <= 0
                or int(raw_image["original_height"]) <= 0
                or int(raw_image["exported_width"]) <= 0
                or int(raw_image["exported_height"]) <= 0
                or raw_image.get("format") != "png"
                or not isinstance(raw_image.get("image_id"), str)
                or not str(raw_image["image_id"])
                or not isinstance(raw_image.get("source_image_hash"), str)
                or not _BLAKE3_RE.fullmatch(str(raw_image["source_image_hash"]))
                or not isinstance(raw_image.get("exported_png_bytes_sha256"), str)
                or not _HEX_64_RE.fullmatch(str(raw_image["exported_png_bytes_sha256"]))
                or not isinstance(raw_image.get("exported_rgba_pixels_sha256"), str)
                or not _HEX_64_RE.fullmatch(
                    str(raw_image["exported_rgba_pixels_sha256"])
                )
            ):
                raise ReleaseDatasetError(
                    f"{recipe.role} export provenance image identity is invalid"
                )
            if (
                raw_image["exported_width"] != viewer_image["width"]
                or raw_image["exported_height"] != viewer_image["height"]
                or raw_image["exported_png_bytes_sha256"]
                != viewer_image["png_bytes_sha256"]
                or raw_image["exported_rgba_pixels_sha256"]
                != viewer_image["rgba_pixels_sha256"]
            ):
                raise ReleaseDatasetError(
                    f"{recipe.role} embedded image differs from export provenance "
                    f"for {instance_id}"
                )
            validated_images.append(dict(raw_image))
        export_provenance[instance_id] = tuple(validated_images)
    if set(export_provenance) != parquet_ids:
        raise ReleaseDatasetError(
            f"{recipe.role} export provenance membership differs from parquet"
        )

    train_path = sidecar_root / "train_instances.jsonl.zst"
    train_ids: set[str] = set()
    train_task_counts: Counter[str] = Counter()
    train_annotations: dict[str, Mapping[str, Any]] = {}
    type_registry_for_sidecars = load_type_registry()
    for record in _iter_zstd_jsonl(
        train_path,
        context=f"{recipe.role}.train_instances",
    ):
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str) or instance_id in train_ids:
            raise ReleaseDatasetError(
                f"{recipe.role} train sidecar has an invalid or duplicate instance id"
            )
        try:
            recomputed_instance_id = compute_instance_id(dict(record))
        except Exception as exc:
            raise ReleaseDatasetError(
                f"{recipe.role} train sidecar identity cannot be recomputed for "
                f"{instance_id}"
            ) from exc
        if recomputed_instance_id != instance_id:
            raise ReleaseDatasetError(
                f"{recipe.role} train sidecar instance id differs from canonical "
                f"identity for {instance_id}"
            )
        train_ids.add(instance_id)
        task_id = str(record.get("task", ""))
        train_task_counts[task_id] += 1
        expected_metadata = inspection.row_metadata.get(instance_id)
        actual_metadata = (
            task_id,
            str(record.get("domain", "")),
            str(record.get("scene_id", "")),
            str(record.get("query_id", "")),
        )
        if expected_metadata != actual_metadata:
            raise ReleaseDatasetError(
                f"{recipe.role} train sidecar metadata differs for {instance_id}"
            )
        if record.get("trace_ref") != inspection.trace_refs.get(instance_id):
            raise ReleaseDatasetError(
                f"{recipe.role} train sidecar trace_ref differs for {instance_id}"
            )
        parquet_answer, parquet_annotation, parquet_reward = inspection.row_contracts[
            instance_id
        ]
        if record.get("answer_gt") != parquet_answer:
            raise ReleaseDatasetError(
                f"{recipe.role} answer_gt differs between parquet and train sidecar "
                f"for {instance_id}"
            )
        if record.get("reward_contract") != parquet_reward:
            raise ReleaseDatasetError(
                f"{recipe.role} reward contract differs between parquet and train "
                f"sidecar for {instance_id}"
            )
        train_annotation = record.get("annotation_gt")
        if not isinstance(train_annotation, Mapping):
            raise ReleaseDatasetError(
                f"{recipe.role} train sidecar annotation is invalid for {instance_id}"
            )
        _validate_typed_value(
            train_annotation,
            field="annotation_gt",
            instance_id=instance_id,
            registered_types=set(type_registry_for_sidecars.annotation_types),
        )
        train_images = record.get("images")
        provenance_images = export_provenance[instance_id]
        if not isinstance(train_images, list) or len(train_images) != len(
            provenance_images
        ):
            raise ReleaseDatasetError(
                f"{recipe.role} train images differ from export provenance for "
                f"{instance_id}"
            )
        for index, (train_image, provenance_image) in enumerate(
            zip(train_images, provenance_images)
        ):
            if not isinstance(train_image, Mapping) or any(
                train_image.get(train_key) != provenance_image.get(provenance_key)
                for train_key, provenance_key in (
                    ("image_id", "image_id"),
                    ("format", "format"),
                    ("image_hash", "source_image_hash"),
                )
            ):
                raise ReleaseDatasetError(
                    f"{recipe.role} train image {index} identity differs from export "
                    f"provenance for {instance_id}"
                )
        try:
            expected_prompts = _build_prompt_columns(
                record,
                image_count=len(train_images),
            )
        except (TypeError, ValueError) as exc:
            raise ReleaseDatasetError(
                f"{recipe.role} prompts cannot be reconstructed for {instance_id}"
            ) from exc
        actual_prompts = inspection.row_prompts[instance_id]
        if actual_prompts != (
            expected_prompts["prompt_answer"],
            expected_prompts["prompt_answer_and_annotation"],
        ):
            raise ReleaseDatasetError(
                f"{recipe.role} exported prompts differ from deterministic export "
                f"for {instance_id}"
            )
        image_infos = [
            ExportedImageInfo(
                original_width=int(image["original_width"]),
                original_height=int(image["original_height"]),
                exported_width=int(image["exported_width"]),
                exported_height=int(image["exported_height"]),
            )
            for image in provenance_images
        ]
        try:
            expected_exported_annotation = _scale_annotation_gt_for_export(
                train_annotation,
                image_infos=image_infos,
            )
        except (TypeError, ValueError) as exc:
            raise ReleaseDatasetError(
                f"{recipe.role} annotation cannot be exported for {instance_id}"
            ) from exc
        if parquet_annotation != expected_exported_annotation:
            raise ReleaseDatasetError(
                f"{recipe.role} annotation_gt differs from deterministic export for "
                f"{instance_id}"
            )
        train_annotations[instance_id] = dict(train_annotation)
        versions = record.get("versions")
        if not isinstance(versions, Mapping) or versions.get("code_hash") != (
            source_revision
        ):
            raise ReleaseDatasetError(
                f"{recipe.role} train sidecar source differs for {instance_id}"
            )
    if train_ids != parquet_ids or len(train_ids) != recipe.rows:
        raise ReleaseDatasetError(
            f"{recipe.role} train sidecar membership differs from parquet"
        )
    _require_exact_task_counts(
        train_task_counts,
        recipe=recipe,
        expected_task_ids=expected_task_ids,
        context=f"{recipe.role}.train_instances",
    )

    curriculum_path = sidecar_root / "curriculum_index.jsonl.zst"
    curriculum_ids: set[str] = set()
    for record in _iter_zstd_jsonl(
        curriculum_path,
        context=f"{recipe.role}.curriculum_index",
    ):
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str) or instance_id in curriculum_ids:
            raise ReleaseDatasetError(
                f"{recipe.role} curriculum has an invalid or duplicate instance id"
            )
        curriculum_ids.add(instance_id)
        expected_metadata = inspection.row_metadata.get(instance_id)
        actual_metadata = (
            str(record.get("task", "")),
            str(record.get("domain", "")),
            str(record.get("scene_id", "")),
            str(record.get("query_id", "")),
        )
        if expected_metadata != actual_metadata:
            raise ReleaseDatasetError(
                f"{recipe.role} curriculum metadata differs for {instance_id}"
            )
    if curriculum_ids != parquet_ids or len(curriculum_ids) != recipe.rows:
        raise ReleaseDatasetError(
            f"{recipe.role} curriculum membership differs from parquet"
        )

    refs_by_shard: dict[str, dict[int, tuple[str, str]]] = {
        shard_id: {} for shard_id, _ in trace_shards
    }
    for instance_id, trace_ref in inspection.trace_refs.items():
        shard_id = str(trace_ref["shard_id"])
        if shard_id not in refs_by_shard:
            raise ReleaseDatasetError(
                f"{recipe.role} references unmanifested trace shard {shard_id}"
            )
        line_index = int(trace_ref["line_index"])
        if line_index in refs_by_shard[shard_id]:
            raise ReleaseDatasetError(
                f"{recipe.role} has duplicate trace reference {shard_id}:{line_index}"
            )
        refs_by_shard[shard_id][line_index] = (
            instance_id,
            str(trace_ref["trace_record_hash"]),
        )
    traced_ids: set[str] = set()
    for shard_id, expected_count in trace_shards:
        trace_path = sidecar_root / "traces" / shard_id
        actual_count = 0
        for line_index, record in enumerate(
            _iter_zstd_jsonl(
                trace_path,
                context=f"{recipe.role}.traces.{shard_id}",
            )
        ):
            expected_ref = refs_by_shard[shard_id].get(line_index)
            if expected_ref is None:
                raise ReleaseDatasetError(
                    f"{recipe.role} has an unreferenced trace record {shard_id}:{line_index}"
                )
            expected_id, expected_hash = expected_ref
            if record.get("instance_id") != expected_id:
                raise ReleaseDatasetError(
                    f"{recipe.role} trace instance mismatch at {shard_id}:{line_index}"
                )
            actual_hash = blake3_hex(canonical_json_bytes(record))
            if actual_hash != expected_hash:
                raise ReleaseDatasetError(
                    f"{recipe.role} trace hash mismatch at {shard_id}:{line_index}"
                )
            task_id, domain, scene_id, query_id = inspection.row_metadata[expected_id]
            taxonomy = record.get("taxonomy")
            public_taxonomy = (
                taxonomy.get("public") if isinstance(taxonomy, Mapping) else None
            )
            if not isinstance(public_taxonomy, Mapping) or any(
                public_taxonomy.get(key) != expected
                for key, expected in (
                    ("task_id", task_id),
                    ("domain", domain),
                    ("scene_id", scene_id),
                    ("query_id", query_id),
                )
            ):
                raise ReleaseDatasetError(
                    f"{recipe.role} trace taxonomy differs for {expected_id}"
                )
            parquet_answer, _parquet_annotation, parquet_reward = (
                inspection.row_contracts[expected_id]
            )
            if record.get("answer_gt") != parquet_answer:
                raise ReleaseDatasetError(
                    f"{recipe.role} trace answer_gt differs for {expected_id}"
                )
            if record.get("reward_contract") != parquet_reward:
                raise ReleaseDatasetError(
                    f"{recipe.role} trace reward contract differs for {expected_id}"
                )
            trace_annotation = record.get("annotation_gt")
            if trace_annotation != train_annotations[expected_id]:
                raise ReleaseDatasetError(
                    f"{recipe.role} trace annotation differs from train sidecar for "
                    f"{expected_id}"
                )
            traced_ids.add(expected_id)
            actual_count += 1
        if actual_count != expected_count or len(refs_by_shard[shard_id]) != (
            expected_count
        ):
            raise ReleaseDatasetError(
                f"{recipe.role} trace record count mismatch for {shard_id}"
            )
    if traced_ids != parquet_ids:
        raise ReleaseDatasetError(
            f"{recipe.role} trace membership differs from parquet"
        )


def verify_release_dataset(
    output_dir: Path,
    *,
    expected_task_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Verify release manifests, hashes, schema, coverage, and image limits."""

    _require_artifact_tree_safe(output_dir)
    manifest_path = output_dir / RELEASE_MANIFEST_NAME
    release_manifest = _read_json(manifest_path)
    _require_manifest_value(
        release_manifest,
        "schema_version",
        RELEASE_MANIFEST_SCHEMA_VERSION,
        context=RELEASE_MANIFEST_NAME,
    )
    source = _verify_release_provenance(release_manifest)
    _require_manifest_value(
        release_manifest,
        "recipe",
        RECIPE_ID,
        context=RELEASE_MANIFEST_NAME,
    )
    _require_manifest_value(
        release_manifest,
        "task_count",
        EXPECTED_TASK_COUNT,
        context=RELEASE_MANIFEST_NAME,
    )
    _verify_file_receipts(
        release_manifest.get("files"),
        output_root=output_dir,
        context=RELEASE_MANIFEST_NAME,
    )

    release_receipts = release_manifest["files"]
    receipted_paths = {str(item["path"]) for item in release_receipts}
    actual_paths = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != RELEASE_MANIFEST_NAME
    }
    canonical_paths = _canonical_release_artifact_paths(output_dir)
    if receipted_paths != canonical_paths or actual_paths != canonical_paths:
        unreceipted = sorted(actual_paths - receipted_paths)
        missing = sorted(canonical_paths - actual_paths)
        forbidden = sorted(actual_paths - canonical_paths)
        noncanonical_receipts = sorted(receipted_paths - canonical_paths)
        raise ReleaseDatasetError(
            "release file inventory differs from the canonical artifact allowlist: "
            f"unreceipted={unreceipted[:5]} missing={missing[:5]} "
            f"forbidden={forbidden[:5]} "
            f"noncanonical_receipts={noncanonical_receipts[:5]}"
        )

    task_ids = expected_task_ids
    if task_ids is None:
        registry_ids = list_default_task_ids()
        _require_canonical_task_registry(registry_ids)
        task_ids = set(registry_ids)
    if len(task_ids) != EXPECTED_TASK_COUNT:
        raise ReleaseDatasetError(
            f"verification requires exactly {EXPECTED_TASK_COUNT} frozen task ids"
        )
    expected_task_id_digest = _ordered_id_digest(sorted(task_ids))
    if expected_task_id_digest != EXPECTED_TASK_IDS_SHA256:
        raise ReleaseDatasetError(
            "verification task allowlist does not match the frozen recipe"
        )
    _require_manifest_value(
        release_manifest,
        "task_ids_sha256",
        expected_task_id_digest,
        context=RELEASE_MANIFEST_NAME,
    )
    _require_manifest_value(
        release_manifest,
        "task_ids",
        sorted(task_ids),
        context=RELEASE_MANIFEST_NAME,
    )

    expected_split_manifests = [
        f"{recipe.metadata_dir}/{recipe.parquet_name}.manifest.json"
        for recipe in SPLIT_RECIPES
    ]
    _require_manifest_value(
        release_manifest,
        "split_manifests",
        expected_split_manifests,
        context=RELEASE_MANIFEST_NAME,
    )
    split_summaries: list[dict[str, Any]] = []
    all_split_ids: set[str] = set()
    for recipe, relative_manifest_path in zip(SPLIT_RECIPES, expected_split_manifests):
        split_manifest = _read_json(output_dir / relative_manifest_path)
        context = relative_manifest_path
        for key, expected in (
            ("schema_version", SPLIT_MANIFEST_SCHEMA_VERSION),
            ("recipe", RECIPE_ID),
            ("dataset_name", recipe.dataset_name),
            ("split_role", recipe.role),
            ("task_count", EXPECTED_TASK_COUNT),
            ("rows", recipe.rows),
            ("samples_per_task", recipe.samples_per_task),
            ("generation_seed", recipe.seed),
            ("shard_count", recipe.shard_count),
            ("schema_profile", "trace_rlvr_viewer_v1"),
            ("columns", VIEWER_COLUMNS),
            ("dropped_columns", DROPPED_EXPORT_COLUMNS),
            (
                "prompt_storage",
                "RLVR export stores prompt_answer and prompt_answer_and_annotation.",
            ),
            ("image_storage_mode", "embedded_bytes"),
            ("max_embedded_image_pixels", MAX_EMBEDDED_IMAGE_PIXELS),
            ("row_order", "deterministic_shuffle"),
            ("row_order_seed", ROW_ORDER_SEED),
            ("image_integrity_threat_model", IMAGE_INTEGRITY_THREAT_MODEL),
        ):
            _require_manifest_value(split_manifest, key, expected, context=context)

        shards = split_manifest.get("shards")
        if not isinstance(shards, list) or len(shards) != recipe.shard_count:
            raise ReleaseDatasetError(
                f"{context} must list exactly {recipe.shard_count} shards"
            )
        expected_shard_paths = [
            _shard_path(output_dir, recipe, shard_index)
            .relative_to(output_dir)
            .as_posix()
            for shard_index in range(recipe.shard_count)
        ]
        actual_shard_paths = [
            str(receipt.get("path", ""))
            for receipt in shards
            if isinstance(receipt, Mapping)
        ]
        if actual_shard_paths != expected_shard_paths:
            raise ReleaseDatasetError(f"{context} lists noncanonical shard paths")
        _verify_file_receipts(shards, output_root=output_dir, context=context)
        shard_paths = [output_dir / str(item["path"]) for item in shards]
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - checked by row inspection.
            raise ReleaseDatasetError(
                "dataset verification requires pyarrow; install trace-tasks[export]"
            ) from exc
        for receipt, shard_path in zip(shards, shard_paths):
            _validate_viewer_arrow_schema(shard_path)
            actual_rows = int(pq.ParquetFile(shard_path).metadata.num_rows)
            if actual_rows != recipe.rows_per_shard:
                raise ReleaseDatasetError(
                    f"{receipt['path']} has {actual_rows} rows, "
                    f"expected {recipe.rows_per_shard}"
                )
            if int(receipt.get("rows", -1)) != recipe.rows_per_shard:
                raise ReleaseDatasetError(
                    f"row receipt mismatch for {receipt['path']}: "
                    f"{receipt.get('rows')!r} != {recipe.rows_per_shard}"
                )
        inspection = _inspect_split_rows(
            shard_paths,
            recipe=recipe,
            expected_task_ids=task_ids,
        )
        overlap = all_split_ids & set(inspection.ordered_ids)
        if overlap:
            raise ReleaseDatasetError(
                f"{recipe.role} overlaps another split: {sorted(overlap)[:5]}"
            )
        all_split_ids.update(inspection.ordered_ids)
        summary = inspection.summary()
        _require_manifest_value(
            split_manifest,
            "instance_id_order_sha256",
            summary["instance_id_order_sha256"],
            context=context,
        )
        _require_manifest_value(
            split_manifest,
            "viewer_image_semantic_sha256",
            summary["viewer_image_semantic_sha256"],
            context=context,
        )
        split_summaries.append({"role": recipe.role, **summary})
        _verify_sidecars(
            output_root=output_dir,
            recipe=recipe,
            inspection=inspection,
            expected_task_ids=task_ids,
            source_revision=str(source["git_revision"]),
            split_sidecars=split_manifest.get("sidecars"),
        )

    return {
        "status": "ok",
        "recipe": RECIPE_ID,
        "output_dir": str(output_dir),
        "splits": split_summaries,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"release artifact directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help=f"temporary build directory (default: {DEFAULT_WORK_DIR})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="generation workers (0 uses all visible CPUs)",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help="maximum queued generation attempts (0 uses twice the worker count)",
    )
    parser.add_argument(
        "--parquet-cpu-count",
        type=int,
        default=0,
        help="parquet preparation CPU count (0 uses all visible CPUs)",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="retain canonical generation and unsharded export intermediates",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="print the immutable recipe and resolved paths without writing files",
    )
    mode.add_argument(
        "--verify",
        "--check",
        dest="verify",
        action="store_true",
        help="verify an existing output directory without changing it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _validate_parallelism(
            workers=args.workers,
            max_in_flight=args.max_in_flight,
            parquet_cpu_count=args.parquet_cpu_count,
        )
        if args.verify:
            result = verify_release_dataset(args.output_dir)
        else:
            task_ids = list_default_task_ids()
            _require_canonical_task_registry(task_ids)
            if args.dry_run:
                source = _source_provenance(REPO_ROOT, require_clean=False)
                environment = _environment_receipt(
                    repo_root=REPO_ROOT,
                    require_match=False,
                )
                result = _resolved_plan(
                    output_dir=args.output_dir,
                    work_dir=args.work_dir,
                    workers=args.workers,
                    max_in_flight=args.max_in_flight,
                    parquet_cpu_count=args.parquet_cpu_count,
                    task_ids=task_ids,
                    source=source,
                    environment=environment,
                )
            else:
                result = build_release_dataset(
                    output_dir=args.output_dir,
                    work_dir=args.work_dir,
                    workers=args.workers,
                    max_in_flight=args.max_in_flight,
                    parquet_cpu_count=args.parquet_cpu_count,
                    keep_work_dir=args.keep_work_dir,
                    repo_root=REPO_ROOT,
                )
    except (OSError, ReleaseDatasetError, ValueError) as exc:
        print(f"release dataset error: {exc}", file=sys.stderr)
        if not args.dry_run and not args.verify:
            print(
                "partial output/work directories, if created, are retained for "
                "inspection only; automatic resume is not supported, and partial "
                "trees must not be published",
                file=sys.stderr,
            )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
