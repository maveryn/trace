"""Contract tests for clock-readout task."""

from __future__ import annotations

import json
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.symbolic.clock.alarm_wait_time_value import SymbolicClockAlarmWaitTimeValueTask
from trace_tasks.tasks.symbolic.clock.full_time_readout import SymbolicClockFullTimeReadoutTask
from trace_tasks.tasks.symbolic.clock.hand_angle_value import SymbolicClockHandAngleValueTask
from trace_tasks.tasks.symbolic.clock.offset_readout import SymbolicClockOffsetReadoutTask
from tests.helpers import read_jsonl


def test_symbolic_clock_readout_deterministic() -> None:
    task = SymbolicClockOffsetReadoutTask()
    params = {
        "query_id": "minutes_after",
        "scene_variant": "minimal",
        "style_variant": "accented",
        "accent_color_name": "purple",
        "delta_minutes": 30,
    }
    out_a = task.generate(20440, params=params, max_attempts=20)
    out_b = task.generate(20440, params=params, max_attempts=20)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()
    assert out_a.answer_gt.type == "option_letter"
    assert out_a.annotation_gt.type == "bbox"
    execution = out_a.trace_payload["execution_trace"]
    assert execution["answer_label"] == out_a.answer_gt.value
    assert execution["option_text_by_label"][out_a.answer_gt.value] == execution["answer_value"]


def test_symbolic_clock_hand_angle_deterministic() -> None:
    task = SymbolicClockHandAngleValueTask()
    params = {
        "shown_hour": 4,
        "shown_minute": 0,
        "scene_variant": "minimal",
        "style_variant": "marker",
        "accent_color_name": "cyan",
    }
    out_a = task.generate(20441, params=params, max_attempts=20)
    out_b = task.generate(20441, params=params, max_attempts=20)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()
    assert out_a.answer_gt.type == "option_letter"
    assert out_a.annotation_gt.type == "bbox"
    execution = out_a.trace_payload["execution_trace"]
    assert execution["answer_label"] == out_a.answer_gt.value
    assert int(execution["option_values_by_label"][out_a.answer_gt.value]) == int(execution["answer_value"])


def test_symbolic_clock_full_time_readout_deterministic() -> None:
    task = SymbolicClockFullTimeReadoutTask()
    params = {
        "shown_hour": 3,
        "shown_minute": 25,
        "shown_second": 40,
        "scene_variant": "outline",
        "style_variant": "marker",
        "accent_color_name": "orange",
    }
    out_a = task.generate(20442, params=params, max_attempts=20)
    out_b = task.generate(20442, params=params, max_attempts=20)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()
    assert out_a.answer_gt.type == "option_letter"
    assert out_a.annotation_gt.type == "bbox"
    execution = out_a.trace_payload["execution_trace"]
    assert execution["answer_value"] == "03:25:40"
    assert execution["answer_label"] == out_a.answer_gt.value
    assert execution["option_text_by_label"][out_a.answer_gt.value] == "03:25:40"


def test_symbolic_clock_alarm_wait_time_deterministic() -> None:
    task = SymbolicClockAlarmWaitTimeValueTask()
    params = {
        "shown_hour": 3,
        "shown_minute": 25,
        "alarm_hour": 7,
        "scene_variant": "minimal",
        "style_variant": "accented",
        "accent_color_name": "blue",
    }
    out_a = task.generate(20443, params=params, max_attempts=20)
    out_b = task.generate(20443, params=params, max_attempts=20)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()
    assert out_a.answer_gt.type == "option_letter"
    assert out_a.annotation_gt.type == "bbox"
    execution = out_a.trace_payload["execution_trace"]
    assert int(execution["answer_value"]) == 215
    assert execution["answer_label"] == out_a.answer_gt.value
    assert int(execution["option_values_by_label"][out_a.answer_gt.value]) == 215


def test_symbolic_clock_offset_readout_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_symbolic__clock__offset_readout"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_symbolic__clock__offset_readout",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_symbolic__clock__offset_readout",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=20,
        sampling_seed=29,
    )
    final_path = build_dataset(config, code_hash="symbolic-clock-offset-readout-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "symbolic" for record in train_records)
    assert all(record["task"] == "task_symbolic__clock__offset_readout" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_symbolic__clock__offset_readout"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0


def test_symbolic_clock_alarm_wait_time_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_symbolic__clock__alarm_wait_time_value"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_symbolic__clock__alarm_wait_time_value",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_symbolic__clock__alarm_wait_time_value",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=20,
        sampling_seed=35,
    )
    final_path = build_dataset(config, code_hash="symbolic-clock-alarm-wait-time-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "symbolic" for record in train_records)
    assert all(record["task"] == "task_symbolic__clock__alarm_wait_time_value" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_symbolic__clock__alarm_wait_time_value"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0


def test_symbolic_clock_full_time_readout_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_symbolic__clock__full_time_readout"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_symbolic__clock__full_time_readout",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_symbolic__clock__full_time_readout",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=20,
        sampling_seed=33,
    )
    final_path = build_dataset(config, code_hash="symbolic-clock-full-time-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "symbolic" for record in train_records)
    assert all(record["task"] == "task_symbolic__clock__full_time_readout" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_symbolic__clock__full_time_readout"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0


def test_symbolic_clock_hand_angle_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_symbolic__clock__hand_angle_value"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_symbolic__clock__hand_angle_value",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_symbolic__clock__hand_angle_value",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=20,
        sampling_seed=31,
    )
    final_path = build_dataset(config, code_hash="symbolic-clock-hand-angle-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "symbolic" for record in train_records)
    assert all(record["task"] == "task_symbolic__clock__hand_angle_value" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_symbolic__clock__hand_angle_value"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
