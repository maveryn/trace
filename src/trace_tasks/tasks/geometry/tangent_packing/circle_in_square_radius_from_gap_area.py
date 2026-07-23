"""Infer the circle radius from the shaded square-container gap area."""

from __future__ import annotations

import math

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_tangent_packing_public_entry
from .shared.algebra import (
    RadiusFromGapAreaSpec,
    circle_in_square_radius_trace_values,
    prepare_radius_from_gap_area,
)
from .shared.measurements import square_container_circle_gap
from .shared.rendering import render_circle_in_square_scene
from .shared.state import DOMAIN, TangentPackingProblem

TASK_ID = "task_geometry__tangent_packing__circle_in_square_radius_from_gap_area"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "circle_in_square_radius_from_gap_area_query"
FORMULA_FAMILY = "circle_in_square_radius_from_gap_area"
_OBJECTIVE_SPEC = RadiusFromGapAreaSpec(
    namespace_key=TASK_ID,
    family=FORMULA_FAMILY,
    construction="circle_in_square",
    coefficient=float(4.0 - math.pi),
    gap_area_fn=square_container_circle_gap,
    formula_text="radius = sqrt(shaded area / (4 - pi))",
    extra_trace_fn=circle_in_square_radius_trace_values,
    derived_radius_key="derived_radius_check",
)


def _prepare_circle_in_square_radius(
    *,
    instance_seed,
    params,
    selected_query,
    branch_probabilities,
) -> tuple[TangentPackingProblem, float | int, dict[str, float | int | str]]:
    """Bind square gap area to circle radius via shaded_area = (4 - pi) r^2."""

    return prepare_radius_from_gap_area(
        instance_seed=int(instance_seed),
        params=params,
        spec=_OBJECTIVE_SPEC,
    )


@register_task
class GeometryCircleInSquareRadiusFromGapAreaTask:
    """Infer the circle radius from the shaded square-container gap area."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SINGLE_QUERY_ID
    task_prompt_key = TASK_PROMPT_KEY
    render_scene = staticmethod(render_circle_in_square_scene)
    prepare_objective = staticmethod(_prepare_circle_in_square_radius)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate an inverse radius-from-gap task for the square container."""

        return run_tangent_packing_public_entry(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GeometryCircleInSquareRadiusFromGapAreaTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
