"""Solve an angle measure from polygon interior-angle-sum expressions."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_polygon_equation_task
from .shared.algebra import angle_name
from .shared.sampling import sample_interior_angle_sum_relation
from .shared.state import PolygonEquationCase

TASK_ID = "task_geometry__polygon_equation_diagram__interior_angle_sum_angle_value"
SUPPORTED_QUERY_IDS: tuple[str, ...] = ("single",)


def _build_case(*, instance_seed: int, params: Mapping[str, Any], generation_defaults: Mapping[str, Any]):
    relation = sample_interior_angle_sum_relation(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        namespace="polygon_equation.angle_sum.angle_measure_value",
    )
    labels = tuple(chr(ord("A") + index) for index in range(int(relation["side_count"])))
    target_index = labels.index(str(relation["target_angle"]))
    return PolygonEquationCase(
        side_count=int(relation["side_count"]),
        answer=int(relation["target_angle_value"]),
        target_name=angle_name(labels, target_index),
        variable_name=str(relation["variable_name"]),
        formula_schema="interior_angle_sum_angle_measure_value",
        relation="polygon_interior_angle_sum_equation",
        output_role="angle_measure_value",
        angle_labels=dict(relation["angle_labels"]),
        target_angle=str(relation["target_angle"]),
        witness=dict(relation["witness"]),
    )


@register_task
class GeometryPolygonEquationDiagramInteriorAngleSumAngleValueTask:
    """Task-owned polygon angle-sum angle-measure objective."""

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


__all__ = ["GeometryPolygonEquationDiagramInteriorAngleSumAngleValueTask", "TASK_ID", "SUPPORTED_QUERY_IDS"]
