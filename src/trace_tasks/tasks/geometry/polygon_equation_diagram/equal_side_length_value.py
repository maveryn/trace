"""Solve a side length from marked equal-side polygon expressions."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_polygon_equation_task
from .shared.sampling import sample_equal_side_relation
from .shared.state import PolygonEquationCase

TASK_ID = "task_geometry__polygon_equation_diagram__equal_side_length_value"
SUPPORTED_QUERY_IDS: tuple[str, ...] = ("single",)


def _build_case(*, instance_seed: int, params: Mapping[str, Any], generation_defaults: Mapping[str, Any]):
    relation = sample_equal_side_relation(
        instance_seed=int(instance_seed),
        params=params,
        namespace="polygon_equation.equal_side.side_length_value",
        include_distractors=True,
    )
    return PolygonEquationCase(
        side_count=int(relation["side_count"]),
        answer=int(relation["side_value"]),
        target_name=f"side {relation['target_side']}",
        variable_name=str(relation["variable_name"]),
        formula_schema="equal_side_expression_side_length_value",
        relation="equal_side_marked_equation",
        output_role="side_length_value",
        side_labels=dict(relation["side_labels"]),
        side_mark_counts=dict(relation["side_mark_counts"]),
        equal_sides=tuple(str(side) for side in relation["equal_sides"]),
        target_side=str(relation["target_side"]),
        witness=dict(relation["witness"]),
    )


@register_task
class GeometryPolygonEquationDiagramEqualSideLengthValueTask:
    """Task-owned equal-side length objective."""

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


__all__ = ["GeometryPolygonEquationDiagramEqualSideLengthValueTask", "TASK_ID", "SUPPORTED_QUERY_IDS"]
