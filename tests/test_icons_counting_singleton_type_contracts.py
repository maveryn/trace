"""Contract tests for icon singleton-type counting task."""

from __future__ import annotations

import json
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.icons.icon_field.frequency_extreme_type_label import IconsIconFieldFrequencyExtremeTypeLabelTask
from trace_tasks.tasks.icons.icon_field.most_frequent_type_count import IconsIconFieldMostFrequentTypeCountTask
from trace_tasks.tasks.icons.icon_field.singleton_type_count import IconsIconFieldSingletonTypeCountTask
from trace_tasks.tasks.icons.icon_grid.distinct_color_count import IconsIconGridDistinctColorCountTask
from trace_tasks.tasks.icons.icon_grid.distinct_type_count import IconsIconGridDistinctTypeCountTask
from tests.helpers import read_jsonl


def test_icons_counting_singleton_type_deterministic() -> None:
    task = IconsIconFieldSingletonTypeCountTask()
    params = {}
    out_a = task.generate(18320, params=params, max_attempts=200)
    out_b = task.generate(18320, params=params, max_attempts=200)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()
    assert sorted(out_a.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert out_a.prompt == out_a.prompt_variants["answer_and_annotation"]
    assert out_a.answer_gt.type == "integer"
    assert out_a.annotation_gt.type == "bbox_set"


def test_icons_counting_most_frequent_type_deterministic() -> None:
    task = IconsIconFieldMostFrequentTypeCountTask()
    params = {}
    out_a = task.generate(18321, params=params, max_attempts=200)
    out_b = task.generate(18321, params=params, max_attempts=200)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()
    assert sorted(out_a.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert out_a.prompt == out_a.prompt_variants["answer_and_annotation"]
    assert out_a.scene_id == "icon_field"
    assert out_a.query_id == "single"
    assert out_a.answer_gt.type == "integer"
    assert out_a.annotation_gt.type == "bbox_set"

    frequencies = out_a.trace_payload["execution_trace"]["type_frequencies"]
    max_frequency = max(int(value) for value in frequencies.values())
    assert int(out_a.answer_gt.value) == int(max_frequency)
    assert sum(1 for value in frequencies.values() if int(value) == int(max_frequency)) == 1
    assert len(out_a.annotation_gt.value) == int(out_a.answer_gt.value)


def test_icons_frequency_extreme_type_label_deterministic() -> None:
    task = IconsIconFieldFrequencyExtremeTypeLabelTask()
    params = {"query_id": "least_frequent_type_label", "option_count": 6}
    out_a = task.generate(18327, params=params, max_attempts=200)
    out_b = task.generate(18327, params=params, max_attempts=200)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()
    assert sorted(out_a.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert out_a.prompt == out_a.prompt_variants["answer_and_annotation"]
    assert out_a.scene_id == "icon_field"
    assert out_a.query_id == "least_frequent_type_label"
    assert out_a.answer_gt.type == "option_letter"
    assert out_a.annotation_gt.type == "bbox_set"


def test_icons_counting_distinct_type_deterministic() -> None:
    task = IconsIconGridDistinctTypeCountTask()
    out_a = task.generate(18322, params={}, max_attempts=200)
    out_b = task.generate(18322, params={}, max_attempts=200)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()
    assert out_a.scene_id == "icon_grid"
    assert out_a.query_id == "single"
    assert out_a.answer_gt.type == "integer"
    assert out_a.annotation_gt.type == "bbox_set"
    assert len(out_a.annotation_gt.value) == int(out_a.answer_gt.value)


def test_icons_counting_distinct_color_deterministic() -> None:
    task = IconsIconGridDistinctColorCountTask()
    out_a = task.generate(18323, params={}, max_attempts=200)
    out_b = task.generate(18323, params={}, max_attempts=200)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()
    assert out_a.scene_id == "icon_grid"
    assert out_a.query_id == "single"
    assert out_a.answer_gt.type == "integer"
    assert out_a.annotation_gt.type == "bbox_set"
    assert len(out_a.annotation_gt.value) == int(out_a.answer_gt.value)


def test_icons_counting_singleton_type_build_smoke(tmp_path: Path) -> None:
    singleton_task_id = "task_icons__icon_field__singleton_type_count"
    most_frequent_task_id = "task_icons__icon_field__most_frequent_type_count"
    frequency_extreme_task_id = "task_icons__icon_field__frequency_extreme_type_label"
    distinct_type_task_id = "task_icons__icon_grid__distinct_type_count"
    distinct_color_task_id = "task_icons__icon_grid__distinct_color_count"
    output_root = tmp_path / "task_icons__icon_field__frequency_split"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_icons__icon_field__frequency_split",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id=singleton_task_id,
                count=4,
                params={},
            ),
            BuildTaskConfig(
                task_id=most_frequent_task_id,
                count=4,
                params={},
            ),
            BuildTaskConfig(
                task_id=frequency_extreme_task_id,
                count=4,
                params={},
            ),
            BuildTaskConfig(
                task_id=distinct_type_task_id,
                count=4,
                params={},
            ),
            BuildTaskConfig(
                task_id=distinct_color_task_id,
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=200,
        sampling_seed=31,
    )
    final_path = build_dataset(config, code_hash="icons-counting-frequency-type-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 20
    assert all(record["domain"] == "icons" for record in train_records)
    assert {str(record["task"]).split("__")[1] for record in train_records} == {"icon_field", "icon_grid"}

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"][singleton_task_id]) == 4
    assert int(build_report["accepted_counts_by_task"][most_frequent_task_id]) == 4
    assert int(build_report["accepted_counts_by_task"][frequency_extreme_task_id]) == 4
    assert int(build_report["accepted_counts_by_task"][distinct_type_task_id]) == 4
    assert int(build_report["accepted_counts_by_task"][distinct_color_task_id]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
