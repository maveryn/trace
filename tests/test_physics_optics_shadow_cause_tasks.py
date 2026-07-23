"""Contract tests for the physics shadow-cause optics task."""

from __future__ import annotations

import json
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.physics.shadow_cause.light_source_label import (
    OPPOSITE_DIRECTION,
    PhysicsShadowCauseLightSourceLabelTask,
)
from tests.helpers import read_jsonl


def _assert_bbox_in_bounds(out) -> None:
    width, height = out.image.size
    assert out.annotation_gt.type == "bbox"
    bbox = out.annotation_gt.value
    assert 0 <= bbox[0] < bbox[2] <= width
    assert 0 <= bbox[1] < bbox[3] <= height


def test_physics_shadow_cause_contract_and_direction_logic() -> None:
    task = PhysicsShadowCauseLightSourceLabelTask()
    for index, shadow_direction in enumerate(OPPOSITE_DIRECTION):
        out = task.generate(
            31800 + index,
            params={
                "shadow_direction": shadow_direction,
                "correct_option_letter": "D",
                "object_shape": "sphere",
            },
            max_attempts=20,
        )
        render_map = out.trace_payload["render_map"]
        execution = out.trace_payload["execution_trace"]

        assert out.scene_id == "shadow_cause"
        assert out.query_id == "single"
        assert out.answer_gt.type == "option_letter"
        assert out.answer_gt.value == "D"
        _assert_bbox_in_bounds(out)
        assert execution["query_id"] == "single"
        assert execution["internal_query_id"] == "source_from_shadow_label"
        assert execution["shadow_direction"] == shadow_direction
        assert execution["source_direction"] == OPPOSITE_DIRECTION[shadow_direction]
        assert execution["candidate_directions"]["D"] == execution["source_direction"]
        assert render_map["candidate_directions"]["D"] == render_map["source_direction"]
        assert list(render_map["candidate_directions"].values()).count(render_map["source_direction"]) == 1
        assert out.annotation_gt.value == render_map["candidate_light_sources"]["D"]["option_bbox_px"]
        assert out.trace_payload["projected_annotation"]["bbox"] == out.annotation_gt.value


def test_physics_shadow_cause_is_deterministic() -> None:
    params = {
        "shadow_direction": "northwest",
        "correct_option_letter": "B",
        "object_shape": "cylinder",
    }
    task = PhysicsShadowCauseLightSourceLabelTask()
    out_a = task.generate(31901, params=params, max_attempts=20)
    out_b = task.generate(31901, params=params, max_attempts=20)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_physics_shadow_cause_prompt_bundle_supports_variants() -> None:
    bundle = json.loads(Path("src/trace_tasks/resources/prompts/physics/shadow_cause/physics_shadow_cause_v1.json").read_text(encoding="utf-8"))

    assert "shadow_cause_query" in bundle["templates"]["task"]
    assert len(bundle["templates"]["scene"]["shadow_cause_diagram"]) == 5
    assert len(bundle["templates"]["query"]["single"]) == 5
    assert "query:single" in bundle["static_slots_by_key"]


def test_physics_shadow_cause_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "physics_shadow_cause"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_physics_shadow_cause",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_physics__shadow_cause__light_source_label",
                count=2,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=20,
        sampling_seed=73,
    )
    final_path = build_dataset(config, code_hash="physics-shadow-cause-smoke")
    train_records = read_jsonl(final_path / "train_instances.jsonl")

    assert len(train_records) == 2
    assert all(record["domain"] == "physics" for record in train_records)
    assert all(record["scene_id"] == "shadow_cause" for record in train_records)
    assert {record["task"] for record in train_records} == {"task_physics__shadow_cause__light_source_label"}
    assert {record["query_id"] for record in train_records} == {"single"}

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
