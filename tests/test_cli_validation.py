from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from trace_tasks import BuildConfig, BuildTaskConfig, build_dataset
from trace_tasks.cli import validate_main
from trace_tasks.core import validation as validation_module
from trace_tasks.core.trace_store import read_trace_shard
from trace_tasks.core.validation import validate_candidate_instance

TASK_ID = "task_geometry__graph_paper__polygon_area_value"


@pytest.fixture(scope="module")
def canonical_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_root = tmp_path_factory.mktemp("cli-validation")
    return build_dataset(
        BuildConfig(
            output_root=str(output_root),
            dataset_name="cli-validation",
            instance_version="v0",
            image_format="png",
            tasks=[BuildTaskConfig(task_id=TASK_ID, count=2)],
            max_attempts_per_instance=40,
            sampling_seed=7,
            workers=1,
        ),
        code_hash="cli-validation-test",
    )


def _copy_dataset(source: Path, destination: Path) -> Path:
    return Path(shutil.copytree(source, destination))


def _delete_last_training_row(dataset_root: Path) -> None:
    records_path = dataset_root / "train_instances.jsonl"
    lines = [
        line
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records_path.write_text(f"{lines[0]}\n", encoding="utf-8")


def _read_training_rows(dataset_root: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (dataset_root / "train_instances.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def _write_training_rows(
    dataset_root: Path,
    rows: list[dict[str, object]],
) -> None:
    payload = "".join(
        f"{json.dumps(row, ensure_ascii=False, sort_keys=True)}\n" for row in rows
    )
    (dataset_root / "train_instances.jsonl").write_text(payload, encoding="utf-8")


def _trace_shard_path(dataset_root: Path) -> Path:
    return dataset_root / "traces" / "trace_shard_0001.jsonl.zst"


def _first_trace_record(dataset_root: Path) -> dict[str, object]:
    record = read_trace_shard(_trace_shard_path(dataset_root))[0]
    assert isinstance(record, dict)
    return record


def test_validate_accepts_canonical_dataset(
    canonical_dataset: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert validate_main([str(canonical_dataset), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["total_errors"] == 0


def test_validate_uses_canonical_build_report_counts(
    canonical_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_root = _copy_dataset(canonical_dataset, tmp_path / "dataset")
    _delete_last_training_row(dataset_root)

    assert validate_main([str(dataset_root), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["error_counts_by_code"]["count_per_task_shortfall"] == 1
    assert report["error_counts_by_code"]["trace_shard_manifest_mismatch"] == 1


def test_validate_detects_tampered_manifest_even_when_task_counts_match(
    canonical_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_root = _copy_dataset(canonical_dataset, tmp_path / "dataset")
    _delete_last_training_row(dataset_root)
    report_path = dataset_root / "build_report.json"
    build_report = json.loads(report_path.read_text(encoding="utf-8"))
    build_report["accepted_counts_by_task"][TASK_ID] = 1
    build_report["trace_shard_manifest"]["shards"][0]["record_count"] = 1
    report_path.write_text(json.dumps(build_report), encoding="utf-8")

    assert validate_main([str(dataset_root), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert "count_per_task_shortfall" not in report["error_counts_by_code"]
    assert report["error_counts_by_code"]["trace_shard_manifest_mismatch"] == 1


def test_validate_detects_deleted_trace_shard(
    canonical_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_root = _copy_dataset(canonical_dataset, tmp_path / "dataset")
    _trace_shard_path(dataset_root).unlink()

    assert validate_main([str(dataset_root), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["error_counts_by_code"]["trace_ref_not_found"] == 2
    assert report["error_counts_by_code"]["trace_shard_manifest_mismatch"] == 1


def test_validate_reports_corrupt_trace_shard_without_crashing(
    canonical_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_root = _copy_dataset(canonical_dataset, tmp_path / "dataset")
    _trace_shard_path(dataset_root).write_bytes(b"not a zstandard trace shard")

    assert validate_main([str(dataset_root), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["error_counts_by_code"]["trace_shard_read_failed"] == 1
    assert report["error_counts_by_code"]["trace_shard_manifest_mismatch"] == 1


def test_validate_detects_duplicate_trace_reference(
    canonical_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_root = _copy_dataset(canonical_dataset, tmp_path / "dataset")
    rows = _read_training_rows(dataset_root)
    rows[1]["trace_ref"] = dict(rows[0]["trace_ref"])  # type: ignore[arg-type]
    _write_training_rows(dataset_root, rows)

    assert validate_main([str(dataset_root), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["error_counts_by_code"]["trace_shard_manifest_mismatch"] == 1


def test_validate_detects_unmanifested_physical_trace_shard(
    canonical_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_root = _copy_dataset(canonical_dataset, tmp_path / "dataset")
    source_shard = dataset_root / "traces" / "trace_shard_0001.jsonl.zst"
    shutil.copy2(source_shard, dataset_root / "traces" / "unexpected.jsonl.zst")

    assert validate_main([str(dataset_root), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["error_counts_by_code"]["trace_shard_manifest_mismatch"] == 1


def test_validate_never_reads_escaping_standalone_trace_reference(
    canonical_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = _copy_dataset(canonical_dataset, tmp_path / "dataset")
    (dataset_root / "build_report.json").unlink()
    shutil.copy2(_trace_shard_path(dataset_root), dataset_root / "outside.zst")
    rows = _read_training_rows(dataset_root)
    trace_ref = dict(rows[0]["trace_ref"])  # type: ignore[arg-type]
    trace_ref["shard_id"] = "../outside.zst"
    rows[0]["trace_ref"] = trace_ref
    _write_training_rows(dataset_root, rows)

    traces_root = (dataset_root / "traces").resolve()
    original_reader = validation_module.read_trace_shard
    paths_read: list[Path] = []

    def guarded_reader(path: str | Path) -> list[dict[str, object]]:
        resolved = Path(path).resolve()
        assert resolved.parent == traces_root
        paths_read.append(resolved)
        return original_reader(path)

    monkeypatch.setattr(validation_module, "read_trace_shard", guarded_reader)

    assert validate_main([str(dataset_root / "train_instances.jsonl"), "--json"]) == 1
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["error_counts_by_code"]["trace_ref_not_found"] == 1
    assert paths_read == [_trace_shard_path(dataset_root).resolve()]
    assert "completeness cannot be checked" in captured.err


def test_validate_rejects_malformed_canonical_build_report(
    canonical_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_root = _copy_dataset(canonical_dataset, tmp_path / "dataset")
    report_path = dataset_root / "build_report.json"
    build_report = json.loads(report_path.read_text(encoding="utf-8"))
    del build_report["trace_shard_manifest"]
    report_path.write_text(json.dumps(build_report), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        validate_main([str(dataset_root), "--json"])
    assert exc_info.value.code == 2
    assert "must contain trace_shard_manifest" in capsys.readouterr().err


def test_validate_rejects_non_utf8_build_report_without_traceback(
    canonical_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_root = _copy_dataset(canonical_dataset, tmp_path / "dataset")
    (dataset_root / "build_report.json").write_bytes(b"\xff\xfe")

    with pytest.raises(SystemExit) as exc_info:
        validate_main([str(dataset_root), "--json"])
    assert exc_info.value.code == 2
    assert "could not read build_report.json" in capsys.readouterr().err


def test_validate_rejects_non_file_manifest_shard_id(
    canonical_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_root = _copy_dataset(canonical_dataset, tmp_path / "dataset")
    report_path = dataset_root / "build_report.json"
    build_report = json.loads(report_path.read_text(encoding="utf-8"))
    build_report["trace_shard_manifest"]["shards"][0].update(
        {"shard_id": ".", "path": "traces"}
    )
    report_path.write_text(json.dumps(build_report), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        validate_main([str(dataset_root), "--json"])
    assert exc_info.value.code == 2
    assert "traces/<shard_id> path" in capsys.readouterr().err


def test_validate_rejects_dataset_root_without_build_report(
    canonical_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_root = _copy_dataset(canonical_dataset, tmp_path / "dataset")
    (dataset_root / "build_report.json").unlink()

    with pytest.raises(SystemExit) as exc_info:
        validate_main([str(dataset_root), "--json"])
    assert exc_info.value.code == 2
    assert (
        "canonical dataset roots require build_report.json" in capsys.readouterr().err
    )


def test_validate_warns_for_standalone_jsonl_fallback(
    canonical_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_root = _copy_dataset(canonical_dataset, tmp_path / "dataset")
    (dataset_root / "build_report.json").unlink()

    assert validate_main([str(dataset_root / "train_instances.jsonl"), "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["total_errors"] == 0
    assert "completeness cannot be checked" in captured.err


def test_validate_reports_malformed_image_entry_without_crashing(
    canonical_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_root = _copy_dataset(canonical_dataset, tmp_path / "dataset")
    rows = _read_training_rows(dataset_root)
    rows[0]["images"] = ["not-an-image-record"]
    _write_training_rows(dataset_root, rows)

    assert validate_main([str(dataset_root), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["error_counts_by_code"]["schema_type_mismatch"] >= 1
    assert report["error_counts_by_code"]["schema_canonicalization_failed"] >= 1


def test_validate_reports_directory_image_path_without_crashing(
    canonical_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_root = _copy_dataset(canonical_dataset, tmp_path / "dataset")
    rows = _read_training_rows(dataset_root)
    image = dict(rows[0]["images"][0])  # type: ignore[index]
    image["path"] = "images"
    rows[0]["images"] = [image]
    _write_training_rows(dataset_root, rows)

    assert validate_main([str(dataset_root), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["error_counts_by_code"]["image_file_not_found"] == 1


@pytest.mark.parametrize("unsafe_kind", ("parent", "symlink"))
def test_validate_never_hashes_escaping_or_symlinked_image(
    canonical_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    unsafe_kind: str,
) -> None:
    dataset_root = _copy_dataset(canonical_dataset, tmp_path / "dataset")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside dataset")
    if unsafe_kind == "parent":
        unsafe_path = "../outside.png"
    else:
        link = dataset_root / "images" / "outside-link.png"
        link.symlink_to(outside)
        unsafe_path = "images/outside-link.png"

    rows = _read_training_rows(dataset_root)
    image = dict(rows[0]["images"][0])  # type: ignore[index]
    image["path"] = unsafe_path
    rows[0]["images"] = [image]
    _write_training_rows(dataset_root, rows)

    original_hasher = validation_module.blake3_file
    paths_hashed: list[Path] = []

    def guarded_hasher(path: str | Path) -> str:
        resolved = Path(path).resolve()
        assert dataset_root.resolve() in resolved.parents
        assert not Path(path).is_symlink()
        paths_hashed.append(resolved)
        return original_hasher(path)

    monkeypatch.setattr(validation_module, "blake3_file", guarded_hasher)

    assert validate_main([str(dataset_root), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["error_counts_by_code"]["image_path_not_relative"] == 1
    assert outside.resolve() not in paths_hashed


@pytest.mark.parametrize("unsafe_kind", ("parent", "symlink", "directory"))
def test_candidate_validation_uses_safe_regular_image_files(
    canonical_dataset: Path,
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    dataset_root = _copy_dataset(canonical_dataset, tmp_path / "dataset")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside dataset")
    if unsafe_kind == "parent":
        unsafe_path = "../outside.png"
        expected_code = "image_path_not_relative"
    elif unsafe_kind == "symlink":
        link = dataset_root / "images" / "outside-link.png"
        link.symlink_to(outside)
        unsafe_path = "images/outside-link.png"
        expected_code = "image_path_not_relative"
    else:
        unsafe_path = "images"
        expected_code = "image_file_not_found"

    row = _read_training_rows(dataset_root)[0]
    image = dict(row["images"][0])  # type: ignore[index]
    image["path"] = unsafe_path
    row["images"] = [image]
    report = validate_candidate_instance(
        row,
        _first_trace_record(dataset_root),
        staging_root=dataset_root,
        expected_instance_version="v0",
    )

    assert report["error_counts_by_code"][expected_code] == 1


def test_candidate_validation_rejects_boolean_trace_line_index(
    canonical_dataset: Path,
) -> None:
    row = _read_training_rows(canonical_dataset)[0]
    trace_ref = dict(row["trace_ref"])  # type: ignore[arg-type]
    trace_ref["line_index"] = True
    row["trace_ref"] = trace_ref

    report = validate_candidate_instance(
        row,
        _first_trace_record(canonical_dataset),
        staging_root=canonical_dataset,
        expected_instance_version="v0",
    )

    assert report["error_counts_by_code"]["trace_ref_index_out_of_range"] == 1


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        ("{not json}\n", "could not read validation source"),
        ("[]\n", "expected an object on line 1"),
    ),
)
def test_validate_routes_malformed_jsonl_through_argument_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    payload: str,
    message: str,
) -> None:
    source = tmp_path / "train_instances.jsonl"
    source.write_text(payload, encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        validate_main([str(source), "--json"])

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err
