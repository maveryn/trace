"""Contract tests for migrated voxel-cube puzzle tasks."""

from __future__ import annotations

from trace_tasks.tasks import create_task

TASK_IDS = (
    "task_puzzles__voxel_cube__cube_count",
    "task_puzzles__voxel_cube__cube_structure_change_count",
    "task_puzzles__voxel_cube__cube_visible_projection_count",
    "task_puzzles__voxel_cube__cube_projection_match_label",
)


def test_voxel_cube_tasks_generate_with_migrated_scene_contracts() -> None:
    for index, task_id in enumerate(TASK_IDS):
        output = create_task(task_id).generate(
            51100 + index,
            params={},
            max_attempts=80,
        )
        assert output.scene_id == "voxel_cube"
        assert output.query_id == "single"
        assert output.trace_payload["query_spec"]["params"]["scene_id"] == "voxel_cube"
        assert output.trace_payload["execution_trace"]["task_id"] == task_id


def test_voxel_cube_annotation_schemas_match_public_contracts() -> None:
    expected = {
        "task_puzzles__voxel_cube__cube_count": "bbox",
        "task_puzzles__voxel_cube__cube_structure_change_count": "bbox_set",
        "task_puzzles__voxel_cube__cube_visible_projection_count": "bbox_set",
        "task_puzzles__voxel_cube__cube_projection_match_label": "bbox",
    }
    for index, task_id in enumerate(TASK_IDS):
        output = create_task(task_id).generate(
            51200 + index,
            params={},
            max_attempts=80,
        )
        assert output.annotation_gt.type == expected[task_id]
        if output.annotation_gt.type == "bbox":
            assert len(output.annotation_gt.value) == 4
        if output.annotation_gt.type == "bbox_set":
            assert all(len(bbox) == 4 for bbox in output.annotation_gt.value)


def test_voxel_cube_semantic_axes_can_be_pinned() -> None:
    cases = [
        (
            "task_puzzles__voxel_cube__cube_structure_change_count",
            {"change_type": "missing_to_complete"},
            "missing_to_complete",
        ),
        (
            "task_puzzles__voxel_cube__cube_structure_change_count",
            {"change_type": "removed"},
            "removed",
        ),
        (
            "task_puzzles__voxel_cube__cube_visible_projection_count",
            {"view_direction": "top"},
            "top",
        ),
        (
            "task_puzzles__voxel_cube__cube_projection_match_label",
            {"view_direction": "front"},
            "front",
        ),
    ]
    for index, (task_id, params, expected_value) in enumerate(cases):
        output = create_task(task_id).generate(
            51300 + index,
            params=dict(params),
            max_attempts=100,
        )
        execution = output.trace_payload["execution_trace"]
        assert expected_value in {
            str(value) for key, value in execution.items() if key in params
        }


def test_voxel_cube_projection_scenes_keep_reference_stack_separate_and_vary_palette() -> None:
    """Projection-option scenes should not overlap the reference stack and options."""

    projection_tasks = (
        "task_puzzles__voxel_cube__cube_projection_match_label",
    )
    palette_ids: set[str] = set()
    for index, task_id in enumerate(projection_tasks * 20):
        output = create_task(task_id).generate(
            51400 + index,
            params={},
            max_attempts=80,
        )
        render_map = output.trace_payload["render_map"]
        stack_bbox = [float(value) for value in render_map["stack_bbox_px"]]
        option_bboxes = [
            [float(value) for value in bbox]
            for bbox in render_map["option_panel_bboxes_px"].values()
        ]
        assert stack_bbox[2] + 14.0 < min(bbox[0] for bbox in option_bboxes)
        palette_ids.add(
            str(output.trace_payload["render_spec"]["voxel_palette"]["palette_id"])
        )
    assert len(palette_ids) >= 2
