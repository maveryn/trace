"""Contract tests for icon sequence-strip completion tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.icons.sequence_strip.count_progression_completion_label import (
    IconsSequenceStripCountProgressionCompletionTask,
)
from trace_tasks.tasks.icons.sequence_strip.rotation_progression_completion_label import (
    IconsSequenceStripRotationProgressionCompletionTask,
)
from trace_tasks.tasks.icons.sequence_strip.size_progression_completion_label import (
    IconsSequenceStripSizeProgressionCompletionTask,
)
from tests.helpers import read_jsonl


_TASK_CLASSES = (
    IconsSequenceStripCountProgressionCompletionTask,
    IconsSequenceStripRotationProgressionCompletionTask,
    IconsSequenceStripSizeProgressionCompletionTask,
)


@pytest.mark.parametrize("task_cls", _TASK_CLASSES)
def test_icons_sequence_completion_is_deterministic(task_cls) -> None:
    task = task_cls()
    out_a = task.generate(15120, params={}, max_attempts=200)
    out_b = task.generate(15120, params={}, max_attempts=200)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()
    assert out_a.answer_gt.type == "string"
    assert out_a.annotation_gt.type == "bbox"


@pytest.mark.parametrize("task_cls", _TASK_CLASSES)
def test_icons_sequence_completion_build_smoke(tmp_path: Path, task_cls) -> None:
    task_id = str(task_cls.task_id)
    output_root = tmp_path / task_id
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name=f"build_smoke_{task_id}",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id=task_id,
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=200,
        sampling_seed=31,
    )
    final_path = build_dataset(config, code_hash="icons-sequence-completion-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "icons" for record in train_records)
    assert all(record["scene_id"] == "sequence_strip" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"][task_id]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
