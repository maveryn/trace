"""Contract tests for the month-view calendar task."""

from __future__ import annotations

import json
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.pages.calendar.marked_day_class_count import PagesCalendarMarkedDayClassCountTask
from trace_tasks.tasks.pages.calendar.workday_offset_date import PagesCalendarWorkdayOffsetDateTask
from tests.helpers import read_jsonl


def test_pages_calendar_month_view_deterministic() -> None:
    task = PagesCalendarMarkedDayClassCountTask()
    params = {
        "marked_day_class": "weekend",
        "scene_variant": "minimal",
        "information_scene_treatments": ["dark_report_card"],
    }
    out_a = task.generate(21920, params=params, max_attempts=20)
    out_b = task.generate(21920, params=params, max_attempts=20)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_pages_calendar_workday_offset_date_deterministic() -> None:
    task = PagesCalendarWorkdayOffsetDateTask()
    params = {
        "query_id": "workday_after_offset_date",
        "scene_variant": "classic",
        "information_scene_treatments": ["poster_explainer"],
    }
    out_a = task.generate(21924, params=params, max_attempts=20)
    out_b = task.generate(21924, params=params, max_attempts=20)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_pages_calendar_marked_day_class_count_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_pages__calendar__marked_day_class_count"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_pages__calendar__marked_day_class_count",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_pages__calendar__marked_day_class_count",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=20,
        sampling_seed=31,
    )
    final_path = build_dataset(config, code_hash="pages-calendar-marked-day-class-count-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "pages" for record in train_records)
    assert all(record["scene_id"] == "calendar" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_pages__calendar__marked_day_class_count"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0


def test_pages_calendar_workday_offset_date_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_pages__calendar__workday_offset_date"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_pages__calendar__workday_offset_date",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_pages__calendar__workday_offset_date",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=20,
        sampling_seed=43,
    )
    final_path = build_dataset(config, code_hash="pages-calendar-workday-offset-date-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "pages" for record in train_records)
    assert all(record["scene_id"] == "calendar" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_pages__calendar__workday_offset_date"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
