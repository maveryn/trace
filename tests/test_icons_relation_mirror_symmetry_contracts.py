"""Contract tests for the icon mirror-symmetry match-label task."""

from __future__ import annotations

import json
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.icons.mirror_grid.mirror_symmetry_match_label import (
    IconsMirrorGridMirrorSymmetryMatchLabelTask,
)
from tests.helpers import read_jsonl


TASK_ID = "task_icons__mirror_grid__mirror_symmetry_match_label"


def test_icons_relation_mirror_symmetry_match_deterministic() -> None:
    task = IconsMirrorGridMirrorSymmetryMatchLabelTask()
    out_a = task.generate(15120, params={}, max_attempts=200)
    out_b = task.generate(15120, params={}, max_attempts=200)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()
    assert sorted(out_a.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert out_a.prompt == out_a.prompt_variants["answer_and_annotation"]
    assert out_a.answer_gt.type == "option_letter"
    assert out_a.annotation_gt.type == "bbox_map"


def test_icons_relation_mirror_symmetry_match_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / TASK_ID
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name=f"build_smoke_{TASK_ID}",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id=TASK_ID,
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=200,
        sampling_seed=31,
    )
    final_path = build_dataset(config, code_hash="icons-relation-mirror-symmetry-match-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "icons" for record in train_records)
    assert all(record["scene_id"] == "mirror_grid" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"][TASK_ID]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
