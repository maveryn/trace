from __future__ import annotations

import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
from unittest.mock import ANY

import pytest
from PIL import Image
import zstandard as zstd

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_release_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("build_release_dataset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
release_dataset = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release_dataset
SPEC.loader.exec_module(release_dataset)


def _task_ids(count: int) -> list[str]:
    return [f"task_test__scene__objective_{index:04d}" for index in range(count)]


def test_dry_run_reports_the_frozen_recipe_without_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    task_ids = _task_ids(release_dataset.EXPECTED_TASK_COUNT)
    monkeypatch.setattr(release_dataset, "list_default_task_ids", lambda: task_ids)
    monkeypatch.setattr(
        release_dataset,
        "EXPECTED_TASK_IDS_SHA256",
        release_dataset._ordered_id_digest(task_ids),
    )
    monkeypatch.setattr(
        release_dataset,
        "_frozen_task_contracts",
        lambda expected: {task_id: ("integer", "point") for task_id in expected},
    )
    output_dir = tmp_path / "release"
    work_dir = tmp_path / "work"

    assert (
        release_dataset.main(
            [
                "--output-dir",
                str(output_dir),
                "--work-dir",
                str(work_dir),
                "--dry-run",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["recipe"] == "trace_rlvr_all1000_iid_v1"
    assert result["task_count"] == 1_000
    assert result["task_ids_sha256"] == release_dataset._ordered_id_digest(task_ids)
    assert result["max_embedded_image_pixels"] == 1_280_000
    assert result["row_order_seed"] == 20_260_711
    assert result["rebuild_contract"]["historical_paper_training_input"] == {
        "repository_id": "maveryn/trace",
        "revision": "e317b746b258630682367cc6a9d87dedd195113c",
    }
    assert result["rebuild_contract"]["byte_identity_expected"] is False
    assert result["prompt_export_variant"] == "answer_and_annotation"
    assert result["viewer_columns"] == release_dataset.VIEWER_COLUMNS
    assert result["splits"][0]["rows"] == 64_000
    assert result["splits"][0]["rows_per_shard"] == 4_000
    assert len(result["splits"][0]["shard_paths"]) == 16
    assert result["splits"][1]["generation_seed"] == 1_042
    assert not output_dir.exists()
    assert not work_dir.exists()


def test_defaults_and_source_are_machine_neutral() -> None:
    assert not release_dataset.DEFAULT_OUTPUT_DIR.is_absolute()
    assert not release_dataset.DEFAULT_WORK_DIR.is_absolute()
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "/" + "home/",
        "/" + "dev/shm",
        "/" + "workspace/",
        "HfApi",
        "upload_folder",
    ):
        assert forbidden not in source


def test_overlapping_output_and_work_paths_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="must not overlap",
    ):
        release_dataset._require_separate_paths(
            tmp_path / "release",
            tmp_path / "release" / "work",
        )


def test_sidecar_packaging_removes_machine_paths_and_writes_hashes(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "generated"
    dataset_root.mkdir()
    (dataset_root / "build_report.json").write_text(
        json.dumps(
            {
                "dataset_id": "abc",
                "type_registry": {
                    "path": "/" + "home/alice/trace/type_registry_v0.json"
                },
            }
        ),
        encoding="utf-8",
    )
    (dataset_root / "validation_report.json").write_text(
        json.dumps(
            {
                "total_errors": 0,
                "build_context": {
                    "temp_path": "/" + "dev/shm/trace-build",
                    "timestamp": "2026-07-11T00:00:00Z",
                },
            }
        ),
        encoding="utf-8",
    )
    (dataset_root / "curriculum_index.jsonl").write_text("{}\n", encoding="utf-8")
    (dataset_root / "train_instances.jsonl").write_text("{}\n", encoding="utf-8")
    export_provenance = dataset_root / "export_provenance.jsonl"
    export_provenance.write_text("{}\n", encoding="utf-8")
    traces = dataset_root / "traces"
    traces.mkdir()
    (traces / "trace_shard_0001.jsonl.zst").write_bytes(b"trace")

    output_root = tmp_path / "release"
    destination = output_root / "sidecars" / "train"
    paths = release_dataset._package_sidecars(
        dataset_root,
        destination,
        output_root=output_root,
        export_provenance_path=export_provenance,
    )

    build_report = json.loads(
        (destination / "build_report.json").read_text(encoding="utf-8")
    )
    validation_report = json.loads(
        (destination / "validation_report.json").read_text(encoding="utf-8")
    )
    assert build_report["type_registry"]["path"] == (
        "trace_tasks/configs/type_registry_v0.json"
    )
    assert validation_report["build_context"] == {"dataset_root": "."}
    assert paths == [
        "sidecars/train/build_report.json",
        "sidecars/train/curriculum_index.jsonl.zst",
        "sidecars/train/export_provenance.jsonl.zst",
        "sidecars/train/traces/trace_shard_0001.jsonl.zst",
        "sidecars/train/train_instances.jsonl.zst",
        "sidecars/train/validation_report.json",
    ]
    manifest = json.loads(
        (destination / "sidecar_manifest.json").read_text(encoding="utf-8")
    )
    for receipt in manifest["included"]:
        artifact = output_root / receipt["path"]
        assert receipt["bytes"] == artifact.stat().st_size
        assert receipt["sha256"] == release_dataset._sha256_file(artifact)


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (16, 12), color=(20, 40, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def _different_png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (16, 12), color=(200, 10, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


def _reward_contract() -> dict[str, Any]:
    return {
        "reward_contract_version": "v0",
        "answer": {"id": "answer_exact_match_v0", "type": "integer"},
        "annotation": {"id": "point_soft_distance_v0", "type": "point"},
    }


def _tiny_rows(
    task_ids: list[str],
    samples_per_task: int,
    role: str,
    *,
    source_revision: str,
    overlap: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    train_records: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    image_bytes = _png_bytes()
    for task_id in task_ids:
        for sample_index in range(samples_per_task):
            identity_role = "shared" if overlap else role
            answer_gt = {"type": "integer", "value": sample_index}
            annotation_gt = {"type": "point", "value": [1, 2]}
            train_record = {
                "instance_version": "v0",
                "instance_seed": sample_index,
                "domain": "test",
                "task": task_id,
                "scene_id": "scene",
                "query_id": "default",
                "prompt": f"annotation prompt {identity_role} {sample_index}",
                "prompt_variants": {
                    "answer_only": f"answer prompt {identity_role} {sample_index}",
                    "answer_and_annotation": (
                        f"annotation prompt {identity_role} {sample_index}"
                    ),
                },
                "images": [
                    {
                        "image_id": f"{identity_role}:{task_id}:{sample_index}",
                        "format": "png",
                        "image_hash": release_dataset.blake3_hex(image_bytes),
                        "path": f"images/test/{task_id}/{sample_index:06d}.png",
                    }
                ],
                "answer_gt": answer_gt,
                "annotation_gt": annotation_gt,
                "reward_contract": _reward_contract(),
                "versions": {
                    "seed_derivation_version": "v0",
                    "code_hash": source_revision,
                },
            }
            instance_id = release_dataset.compute_instance_id(train_record)
            train_record["instance_id"] = instance_id
            trace_record = {
                "instance_id": instance_id,
                "scene_ir": {},
                "query_spec": {},
                "render_spec": {},
                "render_map": {},
                "execution_trace": {},
                "witness_symbolic": {},
                "projected_annotation": {},
                "taxonomy": {
                    "public": {
                        "task_id": task_id,
                        "domain": "test",
                        "scene_id": "scene",
                        "query_id": "default",
                    }
                },
                "answer_gt": answer_gt,
                "annotation_gt": annotation_gt,
                "reward_contract": _reward_contract(),
            }
            trace_ref = {
                "shard_id": "trace_shard_0001.jsonl.zst",
                "line_index": len(trace_records),
                "trace_record_hash": release_dataset.blake3_hex(
                    release_dataset.canonical_json_bytes(trace_record)
                ),
            }
            train_record["trace_ref"] = trace_ref
            prompts = release_dataset._build_prompt_columns(
                train_record,
                image_count=1,
            )
            row = {
                "images": [{"bytes": image_bytes, "path": None}],
                "image_sizes": [{"width": 16, "height": 12}],
                "prompt_answer": prompts["prompt_answer"],
                "prompt_answer_and_annotation": prompts["prompt_answer_and_annotation"],
                "answer_gt": json.dumps(answer_gt, sort_keys=True),
                "annotation_gt": json.dumps(annotation_gt, sort_keys=True),
                "reward_contract": json.dumps(_reward_contract(), sort_keys=True),
                "instance_id": instance_id,
                "domain": "test",
                "task": task_id,
                "scene_id": "scene",
                "query_id": "default",
                "scene_variant": "default",
                "trace_ref": json.dumps(trace_ref, sort_keys=True),
            }
            rows.append(row)
            train_records.append(train_record)
            trace_records.append(trace_record)
    return rows, train_records, trace_records


def _tiny_dataset(rows: list[dict[str, Any]]):
    datasets = pytest.importorskip("datasets")
    dataset = datasets.Dataset.from_dict(
        {
            column: [row[column] for row in rows]
            for column in release_dataset.VIEWER_COLUMNS
        }
    )
    features = dataset.features.copy()
    features["images"] = datasets.Sequence(datasets.Image())
    features["image_sizes"] = datasets.List(
        {
            "width": datasets.Value("int64"),
            "height": datasets.Value("int64"),
        }
    )
    return dataset.cast(features)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_zstd_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")
    path.write_bytes(zstd.ZstdCompressor(level=3).compress(raw))


def _export_provenance_records(
    rows: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    train_by_id = {record["instance_id"]: record for record in train_records}
    records: list[dict[str, Any]] = []
    for row in rows:
        instance_id = row["instance_id"]
        receipts = release_dataset._validate_images(
            row["images"],
            row["image_sizes"],
            instance_id=instance_id,
        )
        images = []
        for index, (train_image, receipt) in enumerate(
            zip(train_by_id[instance_id]["images"], receipts)
        ):
            images.append(
                {
                    "index": index,
                    "image_id": train_image["image_id"],
                    "format": train_image["format"],
                    "source_image_hash": train_image["image_hash"],
                    "original_width": receipt["width"],
                    "original_height": receipt["height"],
                    "exported_width": receipt["width"],
                    "exported_height": receipt["height"],
                    "exported_png_bytes_sha256": receipt["png_bytes_sha256"],
                    "exported_rgba_pixels_sha256": receipt["rgba_pixels_sha256"],
                }
            )
        records.append(
            {
                "schema_version": (
                    release_dataset.EXPORT_PROVENANCE_RECORD_SCHEMA_VERSION
                ),
                "instance_id": instance_id,
                "resize_policy": "pillow_lanczos_max_pixel_cap_v1",
                "max_embedded_image_pixels": (
                    release_dataset.MAX_EMBEDDED_IMAGE_PIXELS
                ),
                "images": images,
            }
        )
    return records


def _package_valid_sidecars(
    output_root: Path,
    recipe: Any,
    task_ids: list[str],
    train_records: list[dict[str, Any]],
    trace_records: list[dict[str, Any]],
    *,
    source_revision: str,
) -> list[str]:
    raw_root = output_root.parent / f"raw-{recipe.role}"
    raw_root.mkdir()
    counts = {task_id: recipe.samples_per_task for task_id in task_ids}
    release_dataset._write_json(
        raw_root / "build_report.json",
        {
            "dataset_id": f"dataset-{recipe.role}",
            "accepted_counts_by_task": counts,
            "trace_shard_manifest": {
                "shards": [
                    {
                        "shard_id": "trace_shard_0001.jsonl.zst",
                        "path": "traces/trace_shard_0001.jsonl.zst",
                        "record_count": len(trace_records),
                    }
                ]
            },
            "code_provenance": {
                "code_hash": source_revision,
                "identity_input": True,
            },
            "type_registry": {
                "type_registry_version": release_dataset.load_type_registry(
                    release_dataset.DEFAULT_REGISTRY_PATH
                ).version,
                "path": "/" + "home/reviewer/trace/type_registry_v0.json",
                "hash": release_dataset.blake3_file(
                    release_dataset.DEFAULT_REGISTRY_PATH
                ),
            },
        },
    )
    release_dataset._write_json(
        raw_root / "validation_report.json",
        {
            "total_errors": 0,
            "build_context": {
                "dataset_id": f"dataset-{recipe.role}",
                "temp_path": "/" + "dev/shm/trace-build",
                "timestamp": "2026-07-11T00:00:00Z",
            },
        },
    )
    curriculum = [
        {
            "instance_id": record["instance_id"],
            "domain": record["domain"],
            "task": record["task"],
            "scene_id": record["scene_id"],
            "query_id": record["query_id"],
        }
        for record in train_records
    ]
    _write_jsonl(raw_root / "train_instances.jsonl", train_records)
    _write_jsonl(raw_root / "curriculum_index.jsonl", curriculum)
    _write_zstd_jsonl(
        raw_root / "traces" / "trace_shard_0001.jsonl.zst",
        trace_records,
    )
    export_provenance_path = raw_root / "export_provenance.jsonl"
    rows_by_id = {
        record["instance_id"]: {
            "instance_id": record["instance_id"],
            "images": [{"bytes": _png_bytes(), "path": None}],
            "image_sizes": [{"width": 16, "height": 12}],
        }
        for record in train_records
    }
    _write_jsonl(
        export_provenance_path,
        _export_provenance_records(
            list(rows_by_id.values()),
            train_records,
        ),
    )
    return release_dataset._package_sidecars(
        raw_root,
        output_root / "sidecars" / recipe.role,
        output_root=output_root,
        export_provenance_path=export_provenance_path,
    )


def _source_receipt(source_revision: str) -> dict[str, Any]:
    return {
        "trace_tasks_version": release_dataset.__version__,
        "source_input_paths": list(release_dataset.SOURCE_INPUT_PATHS),
        "git_revision": source_revision,
        "git_source_clean": True,
        "source_tree_sha256": "1" * 64,
        "source_file_count": 1,
        "constraint_sha256": release_dataset._constraint_receipts(
            release_dataset.REPO_ROOT
        ),
    }


def _write_release_manifest(
    output_root: Path,
    *,
    task_ids: list[str],
    split_manifests: list[str],
    source_revision: str,
) -> None:
    release_dataset._write_json(
        output_root / release_dataset.RELEASE_MANIFEST_NAME,
        {
            "schema_version": release_dataset.RELEASE_MANIFEST_SCHEMA_VERSION,
            "recipe": release_dataset.RECIPE_ID,
            "rebuild_contract": release_dataset._rebuild_contract(),
            "source": _source_receipt(source_revision),
            "runtime": {
                "python_implementation": "CPython",
                "python_version": "3.11.0",
            },
            "dependencies": {
                name: version
                for name, version in release_dataset._expected_dependency_pins(
                    release_dataset.REPO_ROOT
                ).items()
            },
            "task_count": len(task_ids),
            "task_ids_sha256": release_dataset._ordered_id_digest(task_ids),
            "task_ids": task_ids,
            "split_manifests": split_manifests,
            "files": release_dataset._release_file_receipts(output_root),
        },
    )


def _build_tiny_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    overlap: bool = False,
) -> tuple[Path, list[Any], list[str]]:
    pytest.importorskip("pyarrow")
    task_ids = _task_ids(2)
    recipes = [
        release_dataset.SplitRecipe(
            role="train",
            dataset_name="tiny_train",
            seed=42,
            samples_per_task=2,
            shard_count=2,
            data_dir="data/train",
            metadata_dir="metadata/train",
        ),
        release_dataset.SplitRecipe(
            role="validation_iid",
            dataset_name="tiny_validation",
            seed=1_042,
            samples_per_task=1,
            shard_count=1,
            data_dir="data/validation",
            metadata_dir="metadata/validation_iid",
        ),
    ]
    monkeypatch.setattr(release_dataset, "EXPECTED_TASK_COUNT", 2)
    monkeypatch.setattr(release_dataset, "SPLIT_RECIPES", tuple(recipes))
    monkeypatch.setattr(
        release_dataset,
        "EXPECTED_TASK_IDS_SHA256",
        release_dataset._ordered_id_digest(task_ids),
    )
    monkeypatch.setattr(
        release_dataset,
        "_frozen_task_contracts",
        lambda expected: {task_id: ("integer", "point") for task_id in expected},
    )
    output_root = tmp_path / "release"
    output_root.mkdir()
    source_revision = "a" * 40
    split_manifests: list[str] = []
    for recipe in recipes:
        rows, train_records, trace_records = _tiny_rows(
            task_ids,
            recipe.samples_per_task,
            recipe.role,
            source_revision=source_revision,
            overlap=overlap,
        )
        dataset = _tiny_dataset(rows)
        shards = release_dataset._write_viewer_shards(
            dataset,
            output_root=output_root,
            recipe=recipe,
        )
        sidecars = _package_valid_sidecars(
            output_root,
            recipe,
            task_ids,
            train_records,
            trace_records,
            source_revision=source_revision,
        )
        manifest_path = release_dataset._write_split_manifest(
            output_root=output_root,
            recipe=recipe,
            shard_paths=shards,
            sidecar_paths=sidecars,
            expected_task_ids=set(task_ids),
        )
        split_manifests.append(manifest_path.relative_to(output_root).as_posix())

    _write_release_manifest(
        output_root,
        task_ids=task_ids,
        split_manifests=split_manifests,
        source_revision=source_revision,
    )
    return output_root, recipes, task_ids


def test_small_release_round_trip_and_tamper_detection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root, recipes, task_ids = _build_tiny_release(monkeypatch, tmp_path)

    result = release_dataset.verify_release_dataset(
        output_root,
        expected_task_ids=set(task_ids),
    )
    assert result["status"] == "ok"
    assert result["splits"] == [
        {
            "instance_id_order_sha256": ANY,
            "role": "train",
            "row_count": 4,
            "task_count": 2,
            "unique_instance_count": 4,
            "viewer_image_semantic_sha256": ANY,
        },
        {
            "instance_id_order_sha256": ANY,
            "role": "validation_iid",
            "row_count": 2,
            "task_count": 2,
            "unique_instance_count": 2,
            "viewer_image_semantic_sha256": ANY,
        },
    ]

    shard = release_dataset._shard_path(output_root, recipes[0], 0)
    with shard.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(release_dataset.ReleaseDatasetError, match="size mismatch"):
        release_dataset.verify_release_dataset(
            output_root,
            expected_task_ids=set(task_ids),
        )


def test_verify_rejects_unreceipted_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root, _recipes, task_ids = _build_tiny_release(monkeypatch, tmp_path)
    (output_root / "unexpected.txt").write_text("not receipted\n", encoding="utf-8")

    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="canonical artifact allowlist",
    ):
        release_dataset.verify_release_dataset(
            output_root,
            expected_task_ids=set(task_ids),
        )


def _refresh_release_files(output_root: Path) -> None:
    manifest_path = output_root / release_dataset.RELEASE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = release_dataset._release_file_receipts(output_root)
    release_dataset._write_json(manifest_path, manifest)


def _refresh_sidecar_receipt(
    output_root: Path,
    *,
    role: str,
    artifact: Path,
) -> None:
    manifest_path = output_root / "sidecars" / role / "sidecar_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = artifact.relative_to(output_root).as_posix()
    manifest["included"] = [
        (
            release_dataset._relative_file_receipt(artifact, root=output_root)
            if item["path"] == relative
            else item
        )
        for item in manifest["included"]
    ]
    release_dataset._write_json(manifest_path, manifest)
    _refresh_release_files(output_root)


def _read_zstd_jsonl_records(path: Path) -> list[dict[str, Any]]:
    return list(
        release_dataset._iter_zstd_jsonl(
            path,
            context=path.name,
        )
    )


def _refresh_parquet_receipt(
    output_root: Path,
    *,
    recipe: Any,
    shard_path: Path,
    task_ids: list[str],
    refresh_image_digest: bool = False,
) -> None:
    manifest_path = (
        output_root / recipe.metadata_dir / f"{recipe.parquet_name}.manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = shard_path.relative_to(output_root).as_posix()
    replacement = release_dataset._relative_file_receipt(
        shard_path,
        root=output_root,
    )
    replacement["rows"] = recipe.rows_per_shard
    manifest["shards"] = [
        replacement if item["path"] == relative else item for item in manifest["shards"]
    ]
    if refresh_image_digest:
        shard_paths = [
            release_dataset._shard_path(output_root, recipe, shard_index)
            for shard_index in range(recipe.shard_count)
        ]
        inspection = release_dataset._inspect_split_rows(
            shard_paths,
            recipe=recipe,
            expected_task_ids=set(task_ids),
        )
        manifest["viewer_image_semantic_sha256"] = inspection.summary()[
            "viewer_image_semantic_sha256"
        ]
    release_dataset._write_json(manifest_path, manifest)
    _refresh_release_files(output_root)


def test_task_registry_requires_the_frozen_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _task_ids(2)
    monkeypatch.setattr(release_dataset, "EXPECTED_TASK_COUNT", 2)
    monkeypatch.setattr(
        release_dataset,
        "EXPECTED_TASK_IDS_SHA256",
        release_dataset._ordered_id_digest(original),
    )
    replacement = [original[0], "task_test__scene__replacement"]
    assert replacement == sorted(replacement)
    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="digest mismatch",
    ):
        release_dataset._require_canonical_task_registry(replacement)


def test_clean_source_rejects_relevant_untracked_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "constraints").mkdir()
    (repo / "src" / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "script.py").write_text("pass\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (repo / "constraints" / "release.txt").write_text("x==1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "review@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Review"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)
    (repo / "src" / "untracked.py").write_text("VALUE = 2\n", encoding="utf-8")
    monkeypatch.setattr(
        release_dataset,
        "SOURCE_INPUT_PATHS",
        ("src", "script.py", "pyproject.toml", "constraints/release.txt"),
    )
    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="clean committed source inputs",
    ):
        release_dataset._source_provenance(repo, require_clean=True)
    receipt = release_dataset._source_provenance(repo, require_clean=False)
    assert receipt["git_source_clean"] is False
    assert receipt["untracked_source_inputs"] == ["src/untracked.py"]


def test_viewer_schema_and_rows_fail_closed(tmp_path: Path) -> None:
    datasets = pytest.importorskip("datasets")
    rows, _train, _traces = _tiny_rows(
        _task_ids(1),
        1,
        "train",
        source_revision="a" * 40,
        overlap=False,
    )
    malformed = dict(rows[0])
    malformed["images"] = []
    malformed_dataset = datasets.Dataset.from_dict(
        {column: [malformed[column]] for column in release_dataset.VIEWER_COLUMNS}
    )
    malformed_path = tmp_path / "malformed.parquet"
    malformed_dataset.to_parquet(malformed_path)
    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="images must be",
    ):
        release_dataset._validate_viewer_arrow_schema(malformed_path)
    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="no embedded images",
    ):
        release_dataset._validate_viewer_row(
            malformed,
            type_registry=release_dataset.load_type_registry(),
        )
    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="valid complete PNG",
    ):
        release_dataset._validate_images(
            [{"bytes": b"not-an-image", "path": None}],
            [{"width": 1, "height": 1}],
            instance_id="blake3:" + "0" * 64,
        )
    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="dimensions differ",
    ):
        release_dataset._validate_images(
            [{"bytes": _png_bytes(), "path": None}],
            [{"width": 15, "height": 12}],
            instance_id="blake3:" + "0" * 64,
        )


def test_environment_must_match_frozen_release_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = release_dataset._expected_dependency_pins(release_dataset.REPO_ROOT)
    assert expected["datasets"] == "4.8.4"
    assert expected["pyarrow"] == "24.0.0"
    monkeypatch.setattr(
        release_dataset,
        "_dependency_version",
        lambda name: "0.0.0" if name == "pyarrow" else expected[name],
    )
    receipt = release_dataset._environment_receipt(
        repo_root=release_dataset.REPO_ROOT,
        require_match=False,
    )
    assert receipt["matches_release_constraints"] is False
    assert receipt["dependency_mismatches"]["pyarrow"] == {
        "expected": "24.0.0",
        "actual": "0.0.0",
    }
    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="does not match release constraints",
    ):
        release_dataset._environment_receipt(
            repo_root=release_dataset.REPO_ROOT,
            require_match=True,
        )


@pytest.mark.parametrize("minor", (10, 11, 12))
def test_canonical_rebuild_accepts_only_frozen_cpython_minors(minor: int) -> None:
    assert release_dataset._python_runtime_supported(
        implementation="CPython",
        version=f"3.{minor}.0",
    )


@pytest.mark.parametrize(
    ("implementation", "version"),
    (
        ("CPython", "3.9.0"),
        ("CPython", "3.13.0"),
        ("CPython", "3.14.0"),
        ("PyPy", "3.11.0"),
        ("CPython", "not-a-version"),
    ),
)
def test_canonical_rebuild_rejects_nonfrozen_python_runtimes(
    implementation: str,
    version: str,
) -> None:
    assert not release_dataset._python_runtime_supported(
        implementation=implementation,
        version=version,
    )


def test_python_314_constraints_are_outside_canonical_rebuild_inputs() -> None:
    assert release_dataset.CONSTRAINT_PATHS == (
        "pyproject.toml",
        "constraints/release.txt",
    )
    assert "constraints/compat-py314.txt" not in release_dataset.SOURCE_INPUT_PATHS


def test_verify_rejects_cross_split_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root, _recipes, task_ids = _build_tiny_release(
        monkeypatch,
        tmp_path,
        overlap=True,
    )
    with pytest.raises(
        release_dataset.ReleaseDatasetError, match="overlaps another split"
    ):
        release_dataset.verify_release_dataset(
            output_root,
            expected_task_ids=set(task_ids),
        )


def test_verify_rejects_artifact_symlink_escape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root, recipes, task_ids = _build_tiny_release(monkeypatch, tmp_path)
    shard = release_dataset._shard_path(output_root, recipes[0], 0)
    outside = tmp_path / "outside.parquet"
    shutil.copy2(shard, outside)
    shard.unlink()
    shard.symlink_to(outside)
    with pytest.raises(release_dataset.ReleaseDatasetError, match="symlinks"):
        release_dataset.verify_release_dataset(
            output_root,
            expected_task_ids=set(task_ids),
        )


def test_artifact_tree_rejects_root_symlink_and_special_node(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked_root = tmp_path / "linked-release"
    linked_root.symlink_to(target, target_is_directory=True)
    with pytest.raises(
        release_dataset.ReleaseDatasetError, match="must not be a symlink"
    ):
        release_dataset._require_artifact_tree_safe(linked_root)

    release_root = tmp_path / "release"
    release_root.mkdir()
    os.mkfifo(release_root / "named-pipe")
    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="special filesystem node",
    ):
        release_dataset._require_artifact_tree_safe(release_root)


def test_verify_rejects_incomplete_sidecars_even_with_fresh_receipts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root, recipes, task_ids = _build_tiny_release(monkeypatch, tmp_path)
    recipe = recipes[0]
    relative_validation = f"sidecars/{recipe.role}/validation_report.json"
    validation_path = output_root / relative_validation
    validation_path.unlink()

    sidecar_manifest_path = (
        output_root / "sidecars" / recipe.role / "sidecar_manifest.json"
    )
    sidecar_manifest = json.loads(sidecar_manifest_path.read_text(encoding="utf-8"))
    sidecar_manifest["included"] = [
        item
        for item in sidecar_manifest["included"]
        if item["path"] != relative_validation
    ]
    release_dataset._write_json(sidecar_manifest_path, sidecar_manifest)

    split_manifest_path = (
        output_root / recipe.metadata_dir / f"{recipe.parquet_name}.manifest.json"
    )
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    split_manifest["sidecars"] = [
        path for path in split_manifest["sidecars"] if path != relative_validation
    ]
    release_dataset._write_json(split_manifest_path, split_manifest)
    _refresh_release_files(output_root)

    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="canonical artifact allowlist",
    ):
        release_dataset.verify_release_dataset(
            output_root,
            expected_task_ids=set(task_ids),
        )


def test_verify_rejects_machine_path_and_incomplete_dependency_receipts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root, recipes, task_ids = _build_tiny_release(monkeypatch, tmp_path)
    validation_path = (
        output_root / "sidecars" / recipes[0].role / "validation_report.json"
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["debug"] = "/" + "home/reviewer/private"
    release_dataset._write_json(validation_path, validation)
    _refresh_sidecar_receipt(
        output_root,
        role=recipes[0].role,
        artifact=validation_path,
    )
    with pytest.raises(release_dataset.ReleaseDatasetError, match="machine-local path"):
        release_dataset.verify_release_dataset(
            output_root,
            expected_task_ids=set(task_ids),
        )

    validation.pop("debug")
    release_dataset._write_json(validation_path, validation)
    _refresh_sidecar_receipt(
        output_root,
        role=recipes[0].role,
        artifact=validation_path,
    )
    release_manifest_path = output_root / release_dataset.RELEASE_MANIFEST_NAME
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    release_manifest["dependencies"].pop("pyarrow")
    release_dataset._write_json(release_manifest_path, release_manifest)
    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="incomplete dependency receipts",
    ):
        release_dataset.verify_release_dataset(
            output_root,
            expected_task_ids=set(task_ids),
        )


def test_verify_rejects_noncanonical_shard_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root, recipes, task_ids = _build_tiny_release(monkeypatch, tmp_path)
    recipe = recipes[0]
    original = release_dataset._shard_path(output_root, recipe, 0)
    renamed = original.with_name("renamed.parquet")
    original.rename(renamed)
    split_manifest_path = (
        output_root / recipe.metadata_dir / f"{recipe.parquet_name}.manifest.json"
    )
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    replacement = release_dataset._relative_file_receipt(
        renamed,
        root=output_root,
    )
    replacement["rows"] = recipe.rows_per_shard
    split_manifest["shards"][0] = replacement
    release_dataset._write_json(split_manifest_path, split_manifest)
    _refresh_release_files(output_root)
    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="canonical artifact allowlist",
    ):
        release_dataset.verify_release_dataset(
            output_root,
            expected_task_ids=set(task_ids),
        )


def test_verify_cross_links_parquet_contracts_to_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    output_root, recipes, task_ids = _build_tiny_release(monkeypatch, tmp_path)
    recipe = recipes[0]
    shard_path = release_dataset._shard_path(output_root, recipe, 0)
    table = pq.read_table(shard_path)
    answers = table.column("answer_gt").to_pylist()
    answers[0] = json.dumps({"type": "integer", "value": 999}, sort_keys=True)
    answer_index = table.schema.get_field_index("answer_gt")
    mutated = table.set_column(
        answer_index,
        table.schema.field("answer_gt"),
        pa.array(answers, type=pa.string()),
    )
    pq.write_table(mutated, shard_path)

    split_manifest_path = (
        output_root / recipe.metadata_dir / f"{recipe.parquet_name}.manifest.json"
    )
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    replacement = release_dataset._relative_file_receipt(
        shard_path,
        root=output_root,
    )
    replacement["rows"] = recipe.rows_per_shard
    split_manifest["shards"][0] = replacement
    release_dataset._write_json(split_manifest_path, split_manifest)
    _refresh_release_files(output_root)

    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="answer_gt differs between parquet and train sidecar",
    ):
        release_dataset.verify_release_dataset(
            output_root,
            expected_task_ids=set(task_ids),
        )


def test_typed_values_reject_malformed_integer_and_point() -> None:
    rows, _train, _traces = _tiny_rows(
        _task_ids(1),
        1,
        "train",
        source_revision="a" * 40,
        overlap=False,
    )
    malformed_integer = dict(rows[0])
    malformed_integer["answer_gt"] = json.dumps({"type": "integer", "value": 1.5})
    with pytest.raises(release_dataset.ReleaseDatasetError, match="must be an integer"):
        release_dataset._validate_viewer_row(
            malformed_integer,
            type_registry=release_dataset.load_type_registry(),
        )

    malformed_point = dict(rows[0])
    malformed_point["annotation_gt"] = json.dumps({"type": "point", "value": [1]})
    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="two-coordinate point",
    ):
        release_dataset._validate_viewer_row(
            malformed_point,
            type_registry=release_dataset.load_type_registry(),
        )


def test_verify_rejects_exported_prompt_mutation_with_fresh_receipts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    output_root, recipes, task_ids = _build_tiny_release(monkeypatch, tmp_path)
    recipe = recipes[0]
    shard_path = release_dataset._shard_path(output_root, recipe, 0)
    table = pq.read_table(shard_path)
    prompts = table.column("prompt_answer").to_pylist()
    prompts[0] = "<image>mutated but structurally valid prompt"
    column_index = table.schema.get_field_index("prompt_answer")
    pq.write_table(
        table.set_column(
            column_index,
            table.schema.field("prompt_answer"),
            pa.array(prompts, type=pa.string()),
        ),
        shard_path,
    )
    _refresh_parquet_receipt(
        output_root,
        recipe=recipe,
        shard_path=shard_path,
        task_ids=task_ids,
    )

    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="exported prompts differ from deterministic export",
    ):
        release_dataset.verify_release_dataset(
            output_root,
            expected_task_ids=set(task_ids),
        )


def test_verify_rejects_exported_annotation_value_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    output_root, recipes, task_ids = _build_tiny_release(monkeypatch, tmp_path)
    recipe = recipes[0]
    shard_path = release_dataset._shard_path(output_root, recipe, 0)
    table = pq.read_table(shard_path)
    annotations = table.column("annotation_gt").to_pylist()
    annotations[0] = json.dumps({"type": "point", "value": [5, 6]})
    column_index = table.schema.get_field_index("annotation_gt")
    pq.write_table(
        table.set_column(
            column_index,
            table.schema.field("annotation_gt"),
            pa.array(annotations, type=pa.string()),
        ),
        shard_path,
    )
    _refresh_parquet_receipt(
        output_root,
        recipe=recipe,
        shard_path=shard_path,
        task_ids=task_ids,
    )

    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="annotation_gt differs from deterministic export",
    ):
        release_dataset.verify_release_dataset(
            output_root,
            expected_task_ids=set(task_ids),
        )


def test_verify_rejects_same_size_different_png_even_if_digest_is_refreshed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    output_root, recipes, task_ids = _build_tiny_release(monkeypatch, tmp_path)
    recipe = recipes[0]
    shard_path = release_dataset._shard_path(output_root, recipe, 0)
    table = pq.read_table(shard_path)
    images = table.column("images").to_pylist()
    images[0][0]["bytes"] = _different_png_bytes()
    column_index = table.schema.get_field_index("images")
    pq.write_table(
        table.set_column(
            column_index,
            table.schema.field("images"),
            pa.array(images, type=table.schema.field("images").type),
        ),
        shard_path,
    )
    _refresh_parquet_receipt(
        output_root,
        recipe=recipe,
        shard_path=shard_path,
        task_ids=task_ids,
        refresh_image_digest=True,
    )

    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="embedded image differs from export provenance",
    ):
        release_dataset.verify_release_dataset(
            output_root,
            expected_task_ids=set(task_ids),
        )


def test_verify_rejects_extra_receipted_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root, _recipes, task_ids = _build_tiny_release(monkeypatch, tmp_path)
    (output_root / "secret.txt").write_text("receipted secret\n", encoding="utf-8")
    _refresh_release_files(output_root)

    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="canonical artifact allowlist",
    ):
        release_dataset.verify_release_dataset(
            output_root,
            expected_task_ids=set(task_ids),
        )


@pytest.mark.parametrize(
    "mutation",
    ["identity_flag", "registry_hash", "instance_id"],
)
def test_verify_rejects_false_identity_provenance_or_instance_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    output_root, recipes, task_ids = _build_tiny_release(monkeypatch, tmp_path)
    role = recipes[0].role
    if mutation in {"identity_flag", "registry_hash"}:
        build_report_path = output_root / "sidecars" / role / "build_report.json"
        build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
        if mutation == "identity_flag":
            build_report["code_provenance"]["identity_input"] = False
            error = "source or identity provenance mismatch"
        else:
            build_report["type_registry"]["hash"] = "blake3:" + "0" * 64
            error = "wrong frozen type registry"
        release_dataset._write_json(build_report_path, build_report)
        _refresh_sidecar_receipt(
            output_root,
            role=role,
            artifact=build_report_path,
        )
    else:
        train_path = output_root / "sidecars" / role / "train_instances.jsonl.zst"
        records = _read_zstd_jsonl_records(train_path)
        records[0]["instance_id"] = "blake3:" + "f" * 64
        _write_zstd_jsonl(train_path, records)
        _refresh_sidecar_receipt(
            output_root,
            role=role,
            artifact=train_path,
        )
        error = "instance id differs from canonical identity"

    with pytest.raises(release_dataset.ReleaseDatasetError, match=error):
        release_dataset.verify_release_dataset(
            output_root,
            expected_task_ids=set(task_ids),
        )


def test_verify_compares_original_train_and_trace_annotations_exactly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    output_root, recipes, task_ids = _build_tiny_release(monkeypatch, tmp_path)
    recipe = recipes[0]
    role = recipe.role
    shard_path = release_dataset._shard_path(output_root, recipe, 0)
    table = pq.read_table(shard_path)
    target_id = table.column("instance_id").to_pylist()[0]
    trace_refs = table.column("trace_ref").to_pylist()
    target_ref = json.loads(trace_refs[0])

    trace_path = output_root / "sidecars" / role / "traces" / target_ref["shard_id"]
    trace_records = _read_zstd_jsonl_records(trace_path)
    trace_records[target_ref["line_index"]]["annotation_gt"]["value"] = [8, 9]
    replacement_hash = release_dataset.blake3_hex(
        release_dataset.canonical_json_bytes(trace_records[target_ref["line_index"]])
    )
    _write_zstd_jsonl(trace_path, trace_records)

    train_path = output_root / "sidecars" / role / "train_instances.jsonl.zst"
    train_records = _read_zstd_jsonl_records(train_path)
    for record in train_records:
        if record["instance_id"] == target_id:
            record["trace_ref"]["trace_record_hash"] = replacement_hash
            break
    _write_zstd_jsonl(train_path, train_records)

    target_ref["trace_record_hash"] = replacement_hash
    trace_refs[0] = json.dumps(target_ref, sort_keys=True)
    column_index = table.schema.get_field_index("trace_ref")
    pq.write_table(
        table.set_column(
            column_index,
            table.schema.field("trace_ref"),
            pa.array(trace_refs, type=pa.string()),
        ),
        shard_path,
    )
    for artifact in (trace_path, train_path):
        _refresh_sidecar_receipt(
            output_root,
            role=role,
            artifact=artifact,
        )
    _refresh_parquet_receipt(
        output_root,
        recipe=recipe,
        shard_path=shard_path,
        task_ids=task_ids,
    )

    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="trace annotation differs from train sidecar",
    ):
        release_dataset.verify_release_dataset(
            output_root,
            expected_task_ids=set(task_ids),
        )


def test_build_orchestration_emits_a_verified_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_ids = _task_ids(2)
    recipes = (
        release_dataset.SplitRecipe(
            role="train",
            dataset_name="tiny_train",
            seed=42,
            samples_per_task=2,
            shard_count=2,
            data_dir="data/train",
            metadata_dir="metadata/train",
        ),
        release_dataset.SplitRecipe(
            role="validation_iid",
            dataset_name="tiny_validation",
            seed=1_042,
            samples_per_task=1,
            shard_count=1,
            data_dir="data/validation",
            metadata_dir="metadata/validation_iid",
        ),
    )
    source_revision = "a" * 40
    monkeypatch.setattr(release_dataset, "EXPECTED_TASK_COUNT", 2)
    monkeypatch.setattr(release_dataset, "SPLIT_RECIPES", recipes)
    monkeypatch.setattr(
        release_dataset,
        "EXPECTED_TASK_IDS_SHA256",
        release_dataset._ordered_id_digest(task_ids),
    )
    monkeypatch.setattr(release_dataset, "list_default_task_ids", lambda: task_ids)
    monkeypatch.setattr(
        release_dataset,
        "_frozen_task_contracts",
        lambda expected: {task_id: ("integer", "point") for task_id in expected},
    )
    monkeypatch.setattr(
        release_dataset,
        "_source_provenance",
        lambda _root, require_clean=False: _source_receipt(source_revision),
    )
    monkeypatch.setattr(
        release_dataset,
        "_environment_receipt",
        lambda *, repo_root, require_match: {
            "python_implementation": "CPython",
            "python_version": "3.11.0",
            "distributions": release_dataset._expected_dependency_pins(
                release_dataset.REPO_ROOT
            ),
            "complete": True,
        },
    )

    def materialize(**kwargs: Any) -> Path:
        recipe = kwargs["recipe"]
        output_root = kwargs["output_root"]
        rows, train_records, trace_records = _tiny_rows(
            task_ids,
            recipe.samples_per_task,
            recipe.role,
            source_revision=source_revision,
            overlap=False,
        )
        shards = release_dataset._write_viewer_shards(
            _tiny_dataset(rows),
            output_root=output_root,
            recipe=recipe,
        )
        sidecars = _package_valid_sidecars(
            output_root,
            recipe,
            task_ids,
            train_records,
            trace_records,
            source_revision=source_revision,
        )
        return release_dataset._write_split_manifest(
            output_root=output_root,
            recipe=recipe,
            shard_paths=shards,
            sidecar_paths=sidecars,
            expected_task_ids=set(task_ids),
        )

    monkeypatch.setattr(release_dataset, "_materialize_split", materialize)
    output_dir = tmp_path / "orchestrated-release"
    work_dir = tmp_path / "orchestrated-work"
    manifest = release_dataset.build_release_dataset(
        output_dir=output_dir,
        work_dir=work_dir,
        workers=0,
        max_in_flight=0,
        parquet_cpu_count=0,
        keep_work_dir=False,
        repo_root=tmp_path,
    )
    assert manifest["rebuild_contract"]["kind"] == ("fresh_public_semantic_rebuild")
    assert manifest["task_ids_sha256"] == release_dataset._ordered_id_digest(task_ids)
    assert output_dir.is_dir()
    assert not work_dir.exists()


def test_build_preflight_failure_creates_no_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_ids = _task_ids(2)
    monkeypatch.setattr(release_dataset, "EXPECTED_TASK_COUNT", 2)
    monkeypatch.setattr(
        release_dataset,
        "EXPECTED_TASK_IDS_SHA256",
        release_dataset._ordered_id_digest(task_ids),
    )
    monkeypatch.setattr(release_dataset, "list_default_task_ids", lambda: task_ids)
    monkeypatch.setattr(
        release_dataset,
        "_source_provenance",
        lambda _root, require_clean=False: _source_receipt("a" * 40),
    )

    def reject_environment(*, repo_root: Path, require_match: bool) -> dict[str, Any]:
        raise release_dataset.ReleaseDatasetError("environment mismatch")

    monkeypatch.setattr(
        release_dataset,
        "_environment_receipt",
        reject_environment,
    )
    output_dir = tmp_path / "release"
    work_dir = tmp_path / "work"
    with pytest.raises(
        release_dataset.ReleaseDatasetError,
        match="environment mismatch",
    ):
        release_dataset.build_release_dataset(
            output_dir=output_dir,
            work_dir=work_dir,
            workers=0,
            max_in_flight=0,
            parquet_cpu_count=0,
            keep_work_dir=False,
            repo_root=tmp_path,
        )
    assert not output_dir.exists()
    assert not work_dir.exists()
