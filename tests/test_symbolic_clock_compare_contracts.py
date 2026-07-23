"""Contract tests for clock-compare task."""

from __future__ import annotations

import json
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.symbolic.clock.time_extremum_label import SymbolicClockCompareTask
from trace_tasks.tasks.symbolic.clock.time_order_label import SymbolicClockTimeOrderLabelTask
from tests.helpers import read_jsonl


def test_symbolic_clock_compare_deterministic() -> None:
    task = SymbolicClockCompareTask()
    params = {
        "query_id": "earliest_time_label",
        "scene_variant": "minimal",
        "style_variant": "accented",
        "accent_color_name": "purple",
    }
    out_a = task.generate(20640, params=params, max_attempts=20)
    out_b = task.generate(20640, params=params, max_attempts=20)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_symbolic_clock_compare_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_symbolic__clock__time_extremum_label"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_symbolic__clock__time_extremum_label",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_symbolic__clock__time_extremum_label",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=20,
        sampling_seed=31,
    )
    final_path = build_dataset(config, code_hash="symbolic-clock-compare-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "symbolic" for record in train_records)
    assert all(record["task"] == "task_symbolic__clock__time_extremum_label" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_symbolic__clock__time_extremum_label"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0


def test_symbolic_clock_time_order_label_deterministic() -> None:
    task = SymbolicClockTimeOrderLabelTask()
    params = {
        "scene_variant": "outline",
        "style_variant": "marker",
        "accent_color_name": "cyan",
        "shown_total_minutes_by_label": {"A": 300, "B": 140, "C": 415, "D": 20},
        "answer_label": "5",
    }
    out_a = task.generate(20670, params=params, max_attempts=20)
    out_b = task.generate(20670, params=params, max_attempts=20)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_symbolic_clock_time_order_label_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_symbolic__clock__time_order_label"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_symbolic__clock__time_order_label",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_symbolic__clock__time_order_label",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=20,
        sampling_seed=37,
    )
    final_path = build_dataset(config, code_hash="symbolic-clock-time-order-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "symbolic" for record in train_records)
    assert all(record["task"] == "task_symbolic__clock__time_order_label" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_symbolic__clock__time_order_label"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
