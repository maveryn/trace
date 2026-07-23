"""Solve a variable from marked equal-angle polygon expressions."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_polygon_equation_task
from .shared.sampling import sample_equal_angle_relation
from .shared.state import PolygonEquationCase

TASK_ID = "task_geometry__polygon_equation_diagram__equal_angle_variable_value"
SUPPORTED_QUERY_IDS: tuple[str, ...] = ("single",)


def _build_case(*, instance_seed: int, params: Mapping[str, Any], generation_defaults: Mapping[str, Any]):
    relation = sample_equal_angle_relation(
        instance_seed=int(instance_seed),
        params=params,
        namespace="polygon_equation.equal_angle.variable_value",
        include_distractors=False,
    )
    return PolygonEquationCase(
        side_count=int(relation["side_count"]),
        answer=int(relation["variable_value"]),
        target_name=str(relation["variable_name"]),
        variable_name=str(relation["variable_name"]),
        formula_schema="equal_angle_expression_variable_value",
        relation="equal_angle_marked_equation",
        output_role="variable_value",
        angle_labels=dict(relation["angle_labels"]),
        angle_mark_counts=dict(relation["angle_mark_counts"]),
        equal_angles=tuple(str(angle) for angle in relation["equal_angles"]),
        target_angle=str(relation["target_angle"]),
        witness=dict(relation["witness"]),
    )


@register_task
class GeometryPolygonEquationDiagramEqualAngleVariableValueTask:
    """Task-owned equal-angle variable objective."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_polygon_equation_task(
            task_id=TASK_ID,
            build_case=_build_case,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


__all__ = ["GeometryPolygonEquationDiagramEqualAngleVariableValueTask", "TASK_ID", "SUPPORTED_QUERY_IDS"]
