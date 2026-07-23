from __future__ import annotations

import math

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_tangent_packing_public_entry
from .shared.algebra import (
    RadiusFromGapAreaSpec,
    prepare_radius_from_gap_area,
    two_circles_rectangle_radius_trace_values,
)
from .shared.measurements import rectangle_equal_circles_gap
from .shared.rendering import render_two_circles_rectangle_scene
from .shared.state import DOMAIN, TangentPackingProblem

TASK_ID = "task_geometry__tangent_packing__two_circles_in_rectangle_radius_from_gap_area"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "two_circles_in_rectangle_radius_from_gap_area_query"
FORMULA_FAMILY = "two_circles_in_rectangle_radius_from_gap_area"
_OBJECTIVE_SPEC = RadiusFromGapAreaSpec(
    namespace_key=TASK_ID,
    family=FORMULA_FAMILY,
    construction="two_circles_in_rectangle",
    coefficient=float(8.0 - 2.0 * math.pi),
    gap_area_fn=rectangle_equal_circles_gap,
    formula_text="radius = sqrt(shaded area / (8 - 2*pi))",
    extra_trace_fn=two_circles_rectangle_radius_trace_values,
    derived_radius_key="radius_from_gap_area",
)


def _prepare_two_circle_radius_from_gap_area(
    *,
    instance_seed,
    params,
    selected_query,
    branch_probabilities,
) -> tuple[TangentPackingProblem, float | int, dict[str, float | int | str]]:
    # Inverse rectangle-packing hook: bind visible gap area, then solve back to the common circle radius.
    return prepare_radius_from_gap_area(
        instance_seed=int(instance_seed),
        params=params,
        spec=_OBJECTIVE_SPEC,
    )


@register_task
class GeometryTwoCirclesInRectangleRadiusFromGapAreaTask:
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SINGLE_QUERY_ID
    task_prompt_key = TASK_PROMPT_KEY
    render_scene = staticmethod(render_two_circles_rectangle_scene)
    prepare_objective = staticmethod(_prepare_two_circle_radius_from_gap_area)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_tangent_packing_public_entry(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GeometryTwoCirclesInRectangleRadiusFromGapAreaTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
