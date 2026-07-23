"""Contract tests for the milestone-timeline task."""

from __future__ import annotations

import json
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.pages.timeline.date_threshold_event_count import PagesTimelineDateThresholdEventCountTask
from trace_tasks.tasks.pages.timeline.interval_membership_count import PagesTimelineIntervalMembershipCountTask
from trace_tasks.tasks.pages.timeline.relative_position_event_label import PagesTimelineRelativePositionEventLabelTask
from tests.helpers import read_jsonl


def test_pages_timeline_milestones_deterministic() -> None:
    task = PagesTimelineIntervalMembershipCountTask()
    params = {
        "query_id": "between_reference_events_count",
        "scene_variant": "roadmap",
        "style_variant": "accented",
        "accent_color_name": "purple",
    }
    out_a = task.generate(22720, params=params, max_attempts=20)
    out_b = task.generate(22720, params=params, max_attempts=20)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_pages_timeline_date_threshold_event_count_deterministic() -> None:
    task = PagesTimelineDateThresholdEventCountTask()
    params = {
        "scene_variant": "classic",
        "style_variant": "marker",
        "accent_color_name": "cyan",
    }
    out_a = task.generate(22726, params=params, max_attempts=20)
    out_b = task.generate(22726, params=params, max_attempts=20)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_pages_timeline_relative_position_event_label_deterministic() -> None:
    task = PagesTimelineRelativePositionEventLabelTask()
    params = {
        "query_id": "event_before_dated_event_label",
        "scene_variant": "minimal",
        "style_variant": "accented",
        "accent_color_name": "green",
        "relative_offset": 3,
    }
    out_a = task.generate(22729, params=params, max_attempts=20)
    out_b = task.generate(22729, params=params, max_attempts=20)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_pages_timeline_interval_membership_count_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_pages__timeline__interval_membership_count"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_pages__timeline__interval_membership_count",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_pages__timeline__interval_membership_count",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=20,
        sampling_seed=37,
    )
    final_path = build_dataset(config, code_hash="pages-timeline-interval-membership-count-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "pages" for record in train_records)
    assert all(record["task"] == "task_pages__timeline__interval_membership_count" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_pages__timeline__interval_membership_count"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0


def test_pages_timeline_relative_position_event_label_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_pages__timeline__relative_position_event_label"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_pages__timeline__relative_position_event_label",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_pages__timeline__relative_position_event_label",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=20,
        sampling_seed=43,
    )
    final_path = build_dataset(config, code_hash="pages-timeline-relative-position-event-label-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "pages" for record in train_records)
    assert all(record["task"] == "task_pages__timeline__relative_position_event_label" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_pages__timeline__relative_position_event_label"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0


def test_pages_timeline_date_threshold_event_count_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_pages__timeline__date_threshold_event_count"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_pages__timeline__date_threshold_event_count",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_pages__timeline__date_threshold_event_count",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=20,
        sampling_seed=41,
    )
    final_path = build_dataset(config, code_hash="pages-timeline-date-threshold-event-count-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "pages" for record in train_records)
    assert all(record["task"] == "task_pages__timeline__date_threshold_event_count" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_pages__timeline__date_threshold_event_count"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
