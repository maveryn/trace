"""Contract tests for the split physics optics ray tasks."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.physics.ray_optics.ray_bounce_count import (
    PhysicsRayOpticsRayBounceCountTask,
)
from trace_tasks.tasks.physics.ray_optics.ray_target_hit_count import (
    PhysicsRayOpticsRayTargetHitCountTask,
)
from tests.helpers import read_jsonl


@pytest.mark.parametrize(
    ("task_cls", "params", "expected_query_id", "expected_answer"),
    (
        (PhysicsRayOpticsRayBounceCountTask, {"scene_variant": "quad_mirror", "target_answer": 4}, "bounce_count", 4),
        (PhysicsRayOpticsRayBounceCountTask, {"scene_variant": "five_mirror", "target_answer": 5}, "bounce_count", 5),
        (
            PhysicsRayOpticsRayTargetHitCountTask,
            {"scene_variant": "single_mirror", "target_answer": 2},
            "target_hit_count",
            2,
        ),
    ),
)
def test_physics_optics_ray_tasks_emit_expected_contract(
    task_cls: type,
    params: dict[str, int | str],
    expected_query_id: str,
    expected_answer: int,
) -> None:
    out = task_cls().generate(27001, params=params, max_attempts=60)
    trace = out.trace_payload
    execution = trace["execution_trace"]


    assert out.answer_gt.type == "integer"

    assert int(out.answer_gt.value) == int(expected_answer)

    assert out.annotation_gt.type == "point_set"

    assert out.query_id == "single"

    assert trace["query_spec"]["query_id"] == "single"

    assert trace["query_spec"]["params"]["query_id"] == "single"
    assert trace["query_spec"]["params"]["internal_query_id"] == expected_query_id

    assert execution["query_id"] == "single"
    assert execution["internal_query_id"] == expected_query_id
    assert execution["ray_event_kind"] == expected_query_id

    assert int(execution["target_answer"]) == int(expected_answer)

    assert trace["projected_annotation"]["type"] == "point_set"

    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value

    assert trace["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value

    assert len(trace["projected_annotation"]["pixel_point_map"]) == len(out.annotation_gt.value)

    assert execution["annotation_pixel_points"] == out.annotation_gt.value
    assert trace["render_map"]["annotation_point_set_px"] == out.annotation_gt.value
    assert trace["render_spec"]["font"]["selection_policy"]["pool"] == "global_approved_font_pool"
    assert trace["render_spec"]["layout_placement"]["mode"] == "whole_ray_optics_board_offset"
    width = int(trace["render_spec"]["canvas_width"])
    height = int(trace["render_spec"]["canvas_height"])
    board_bbox = trace["render_map"]["board_bbox_px"]
    assert float(board_bbox[2]) - float(board_bbox[0]) >= 500.0
    assert float(board_bbox[3]) - float(board_bbox[1]) >= 500.0
    assert ((float(board_bbox[2]) - float(board_bbox[0])) * (float(board_bbox[3]) - float(board_bbox[1]))) / float(width * height) >= 0.40
    for point in out.annotation_gt.value:

        assert len(point) == 2

        assert 0 <= float(point[0]) <= width

        assert 0 <= float(point[1]) <= height
    if expected_query_id == "bounce_count":

        assert len(execution["bounce_cells"]) == int(expected_answer)

        assert len(out.annotation_gt.value) == int(expected_answer)

        assert execution["target_specs"] == []
    else:
        hit_targets = [spec for spec in execution["target_specs"] if bool(spec["hit"])]

        assert len(hit_targets) == int(expected_answer)

        assert len(out.annotation_gt.value) == int(expected_answer)

        assert all({"col", "row"} <= set(spec.keys()) for spec in execution["target_specs"])

        assert trace["render_map"]["accent_color_name"] == execution["accent_color_name"]

        assert "ray_polyline_px" in trace["render_map"]

        assert "source_direction_px" in trace["render_map"]


def test_physics_optics_ray_target_hit_count_is_deterministic() -> None:
    params = {
        "scene_variant": "double_mirror",
        "target_answer": 2,
        "accent_color_name": "purple",
    }
    task = PhysicsRayOpticsRayTargetHitCountTask()
    out_a = task.generate(27021, params=params, max_attempts=60)
    out_b = task.generate(27021, params=params, max_attempts=60)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()

    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()

    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]

    assert out_a.prompt == out_b.prompt

    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_physics_optics_ray_tasks_reject_unknown_scene_variant() -> None:
    with pytest.raises(ValueError):
        PhysicsRayOpticsRayTargetHitCountTask().generate(
            27031,
            params={"scene_variant": "hex_mirror"},
            max_attempts=20,
        )


def test_physics_optics_ray_tasksseeded_sampler_decouples_scene_and_answer_support() -> None:
    scenes_by_query: dict[str, Counter[str]] = defaultdict(Counter)
    answers_by_scene_query: dict[tuple[str, str], set[int]] = defaultdict(set)

    bounce_task = PhysicsRayOpticsRayBounceCountTask()
    for index in range(30):
        out = bounce_task.generate(
            27100 + index,
            params={},
            max_attempts=60,
        )
        query_id = str(out.trace_payload["execution_trace"]["internal_query_id"])
        scene_variant = str(out.trace_payload["query_spec"]["params"]["scene_variant"])
        scenes_by_query[query_id][scene_variant] += 1
        answers_by_scene_query[(scene_variant, query_id)].add(int(out.answer_gt.value))

    target_task = PhysicsRayOpticsRayTargetHitCountTask()
    for index in range(96):
        out = target_task.generate(
            27200 + index,
            params={},
            max_attempts=60,
        )
        query_id = str(out.trace_payload["execution_trace"]["internal_query_id"])
        scene_variant = str(out.trace_payload["query_spec"]["params"]["scene_variant"])
        scenes_by_query[query_id][scene_variant] += 1
        answers_by_scene_query[(scene_variant, query_id)].add(int(out.answer_gt.value))


    assert set(scenes_by_query["bounce_count"].keys()) == {"five_mirror"}

    assert set(scenes_by_query["target_hit_count"].keys()) == {
        "single_mirror",
        "double_mirror",
        "triple_mirror",
    }

    assert answers_by_scene_query[("five_mirror", "bounce_count")] == {1, 2, 3, 4, 5}
    for scene_variant in ("single_mirror", "double_mirror", "triple_mirror"):

        assert answers_by_scene_query[(scene_variant, "target_hit_count")] == {1, 2, 3, 4, 5}


def test_physics_optics_ray_prompt_bundle_supports_variants() -> None:
    bundle = json.loads(Path("src/trace_tasks/resources/prompts/physics/ray_optics/physics_ray_optics_v1.json").read_text(encoding="utf-8"))

    assert len(bundle["templates"]["query"]["bounce_count"]) == 5

    assert len(bundle["templates"]["query"]["target_hit_count"]) == 5

    assert len(set(bundle["templates"]["output"]["answer_and_annotation"])) == 5


def test_physics_optics_ray_tasks_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "physics_optics_ray"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_physics_optics_ray",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_physics__ray_optics__ray_bounce_count",
                count=2,
                params={},
            ),
            BuildTaskConfig(
                task_id="task_physics__ray_optics__ray_target_hit_count",
                count=2,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=60,
        sampling_seed=71,
    )
    final_path = build_dataset(config, code_hash="physics-optics-ray-trace-smoke")

    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")

    assert len(train_records) == 4

    assert all(record["domain"] == "physics" for record in train_records)

    assert all(record["scene_id"] == "ray_optics" for record in train_records)

    assert {record["task"] for record in train_records} == {
        "task_physics__ray_optics__ray_bounce_count",
        "task_physics__ray_optics__ray_target_hit_count",
    }

    assert {record["query_id"] for record in train_records} == {"single"}

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))

    assert int(build_report["accepted_counts_by_task"]["task_physics__ray_optics__ray_bounce_count"]) == 2

    assert int(build_report["accepted_counts_by_task"]["task_physics__ray_optics__ray_target_hit_count"]) == 2

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))

    assert validation["total_errors"] == 0
