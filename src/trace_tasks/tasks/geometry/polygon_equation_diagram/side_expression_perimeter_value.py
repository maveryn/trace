"""Solve a polygon perimeter from side expressions after an equal-side equation."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_polygon_equation_task
from .shared.sampling import sample_equal_side_perimeter_relation
from .shared.state import PolygonEquationCase

TASK_ID = "task_geometry__polygon_equation_diagram__side_expression_perimeter_value"
SUPPORTED_QUERY_IDS: tuple[str, ...] = ("single",)


def _build_case(*, instance_seed: int, params: Mapping[str, Any], generation_defaults: Mapping[str, Any]):
    relation = sample_equal_side_perimeter_relation(
        instance_seed=int(instance_seed),
        params=params,
        namespace="polygon_equation.side_expression.perimeter_value",
        include_distractors=False,
    )
    return PolygonEquationCase(
        side_count=int(relation["side_count"]),
        answer=int(relation["perimeter_value"]),
        target_name="the perimeter",
        variable_name=str(relation["variable_name"]),
        formula_schema="equal_side_expression_perimeter_value",
        relation="equal_side_marked_equation_then_perimeter_sum",
        output_role="perimeter_value",
        side_labels=dict(relation["side_labels"]),
        side_mark_counts=dict(relation["side_mark_counts"]),
        equal_sides=tuple(str(side) for side in relation["equal_sides"]),
        target_side=str(relation["target_side"]),
        witness=dict(relation["witness"]),
    )


@register_task
class GeometryPolygonEquationDiagramSideExpressionPerimeterValueTask:
    """Task-owned perimeter objective after solving an equal-side relation."""

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


__all__ = ["GeometryPolygonEquationDiagramSideExpressionPerimeterValueTask", "TASK_ID", "SUPPORTED_QUERY_IDS"]
