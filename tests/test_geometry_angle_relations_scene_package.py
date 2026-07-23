"""source-layout contracts for geometry angle-relations tasks."""

from __future__ import annotations

from pathlib import Path

import pytest

from trace_tasks.core.taxonomy import TASK_TAXONOMY
from trace_tasks.tasks import create_task


ACTIVE_TASK_IDS = {
    "task_geometry__angle_relations__algebraic_angle_value",
    "task_geometry__angle_relations__parallel_algebraic_angle_value",
    "task_geometry__angle_relations__parallel_supplement_angle",
    "task_geometry__angle_relations__parallel_transversal_triangle_angle_value",
    "task_geometry__angle_relations__triangle_exterior_angle",
}
RETIRED_TASK_IDS = {
    "task_geometry__angle_relations__algebraic_angle_value_triangle_double_extension_expression",
    "task_geometry__angle_relations__algebraic_angle_value_triangle_single_extension_expression",
}


def test_angle_relations_registry_surface_is_merged() -> None:
    """The algebraic single/double extension cases are metadata, not tasks."""

    for task_id in ACTIVE_TASK_IDS:
        assert create_task(task_id).task_id == task_id
    for task_id in RETIRED_TASK_IDS:
        assert task_id not in TASK_TAXONOMY


def test_angle_relations_scene_has_no_wrapper_only_modules() -> None:
    """The first geometry v2 scene should not keep legacy wrapper/base files."""

    scene_dir = Path("src/trace_tasks/tasks/geometry/angle_relations")
    public_files = {
        path.name
        for path in scene_dir.glob("*.py")
        if path.name not in {"__init__.py", "_lifecycle.py"}
    }
    assert public_files == {
        "algebraic_angle_value.py",
        "parallel_algebraic_angle_value.py",
        "parallel_supplement_angle.py",
        "parallel_transversal_triangle_angle_value.py",
        "triangle_exterior_angle.py",
    }
    assert not (scene_dir / "shared" / "task_base.py").exists()
    assert not (scene_dir / "shared" / "cases.py").exists()


@pytest.mark.parametrize(
    ("task_id", "query_ids"),
    (
        ("task_geometry__angle_relations__algebraic_angle_value", ("target_angle_value", "variable_x_value")),
        ("task_geometry__angle_relations__parallel_algebraic_angle_value", ("single",)),
        ("task_geometry__angle_relations__parallel_supplement_angle", ("single",)),
        ("task_geometry__angle_relations__parallel_transversal_triangle_angle_value", ("single",)),
        ("task_geometry__angle_relations__triangle_exterior_angle", ("single",)),
    ),
)
def test_angle_relations_tasks_generate_keyed_angle_points(task_id: str, query_ids: tuple[str, ...]) -> None:
    task = create_task(task_id)
    assert tuple(task.supported_query_ids) == query_ids
    for query_id in query_ids:
        output = task.generate(20260610, params={"query_id": query_id, "case_index": 0}, max_attempts=20)
        assert output.scene_id == "angle_relations"
        assert output.query_id == query_id
        assert output.answer_gt.type == "integer"
        assert output.annotation_gt.type == "point_map"
        assert output.trace_payload["projected_annotation"]["type"] == "point_map"
        assert set(output.annotation_gt.value) == set(output.trace_payload["execution_trace"]["annotation_roles"])
        if task_id == "task_geometry__angle_relations__algebraic_angle_value":
            assert set(output.annotation_gt.value) == {"A", "B", "C", "D"}
            assert output.trace_payload["execution_trace"]["extension_case"] in {"single_extension", "double_extension"}
            assert "variable_x_value" in output.trace_payload["execution_trace"]
            assert "target_angle_value" in output.trace_payload["execution_trace"]
            if query_id == "target_angle_value":
                assert output.answer_gt.value == output.trace_payload["execution_trace"]["target_angle_value"]
            else:
                assert output.answer_gt.value == output.trace_payload["execution_trace"]["variable_x_value"]
        elif task_id == "task_geometry__angle_relations__parallel_algebraic_angle_value":
            assert set(output.annotation_gt.value) == {"BPQ", "DQP", "FRQ"}
            assert (
                output.trace_payload["scene_ir"]["relations"]["relation_id"]
                == "same_side_supplementary_expression_pair"
            )
            assert "variable_x_value" in output.trace_payload["execution_trace"]
            assert "target_angle_value" in output.trace_payload["execution_trace"]
            assert output.trace_payload["execution_trace"]["expression_angle_names"] == ["BPQ", "DQP"]
            assert output.trace_payload["execution_trace"]["target_angle_name"] == "FRQ"
            expression_values = output.trace_payload["execution_trace"]["expression_angle_values"]
            assert sum(int(value) for value in expression_values) == 180
            assert output.trace_payload["execution_trace"]["target_angle_value"] == int(expression_values[1])
            assert output.answer_gt.value == output.trace_payload["execution_trace"]["target_angle_value"]
            assert 'angle "FRQ"' in output.prompt
            assert "marked with the expression" not in output.prompt
        elif task_id == "task_geometry__angle_relations__parallel_transversal_triangle_angle_value":
            assert set(output.annotation_gt.value) == {"P", "Q", "R", "S", "T"}
            assert output.trace_payload["execution_trace"]["internal_query_id"] == "parallel_transversal_triangle_angle_value"
            exterior_angles = output.trace_payload["execution_trace"]["displayed_exterior_angles"]
            derived_base_angles = output.trace_payload["execution_trace"]["derived_lower_triangle_base_angles"]
            assert derived_base_angles == [180 - int(exterior_angles[0]), 180 - int(exterior_angles[1])]
            assert output.answer_gt.value == int(exterior_angles[0]) + int(exterior_angles[1]) - 180
        width, height = output.image.size
        for point in output.annotation_gt.value.values():
            assert len(point) == 2
            assert 0.0 <= float(point[0]) <= float(width)
            assert 0.0 <= float(point[1]) <= float(height)


def test_parallel_supplement_uses_aef_given_angle_and_cfe_target() -> None:
    task = create_task("task_geometry__angle_relations__parallel_supplement_angle")
    output = task.generate(
        20260612,
        params={"query_id": "single", "case_index": 0},
        max_attempts=20,
    )

    assert output.query_id == "single"
    assert output.trace_payload["execution_trace"]["internal_query_id"] == "parallel_supplement_angle"
    assert set(output.annotation_gt.value) == {"AEF", "CFE"}
    assert "given_angle_AEF" in output.trace_payload["execution_trace"]
    assert "given_angle_BEF" not in output.trace_payload["execution_trace"]
    assert "What is the measure of angle \"CFE\"?" in output.prompt
    assert "Lines \"AB\" and \"CD\" are parallel" not in output.prompt


@pytest.mark.parametrize("case_index", (90, 91, 120, 121, 178, 179))
def test_parallel_supplement_obtuse_cases_keep_annotation_in_canvas(case_index: int) -> None:
    """Obtuse supplement cases must preserve the transversal direction when laying out point F."""

    task = create_task("task_geometry__angle_relations__parallel_supplement_angle")
    output = task.generate(
        20260612 + int(case_index),
        params={"query_id": "single", "case_index": int(case_index)},
        max_attempts=20,
    )

    width, height = output.image.size
    assert output.answer_gt.value >= 90
    for point in output.annotation_gt.value.values():
        assert len(point) == 2
        assert 0.0 <= float(point[0]) <= float(width)
        assert 0.0 <= float(point[1]) <= float(height)


def test_parallel_supplement_samples_two_or_three_parallel_lines() -> None:
    task = create_task("task_geometry__angle_relations__parallel_supplement_angle")
    two_line = task.generate(20260612, params={"query_id": "single", "case_index": 0}, max_attempts=20)
    three_line = task.generate(20260612, params={"query_id": "single", "case_index": 1}, max_attempts=20)

    assert two_line.trace_payload["execution_trace"]["parallel_line_count"] == 2
    assert three_line.trace_payload["execution_trace"]["parallel_line_count"] == 3
