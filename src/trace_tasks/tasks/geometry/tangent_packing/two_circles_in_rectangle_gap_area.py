"""Compute the shaded gap area around two equal circles in a rectangle."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_tangent_packing_public_entry
from .shared.measurements import case_trace_values, fmt_measure, rectangle_equal_circles_gap
from .shared.rendering import render_two_circles_rectangle_scene
from .shared.sampling import answer_support_probabilities, choose_radius
from .shared.state import DOMAIN, TangentPackingProblem

TASK_ID = "task_geometry__tangent_packing__two_circles_in_rectangle_gap_area"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "two_circles_in_rectangle_gap_area_query"
FORMULA_FAMILY = "two_circles_in_rectangle_gap_area"


def _prepare_two_circle_rectangle_gap_area(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_query: str,
    branch_probabilities: Mapping[str, float],
) -> tuple[TangentPackingProblem, float, dict[str, Any]]:
    """Bind rectangle-minus-two-circles area while preserving the packed width witness."""

    case, radius_probabilities = choose_radius(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.radius",
    )
    answer = rectangle_equal_circles_gap(int(case.radius))
    support_probabilities = answer_support_probabilities(
        answer_fn=rectangle_equal_circles_gap,
        key_fn=fmt_measure,
    )
    problem = TangentPackingProblem(
        construction_kind="two_circles_in_rectangle",
        target_kind="shaded_area",
        support_kind="width",
        target_text="shaded area=?",
        support_text=f"width={case.packed_rectangle_width}",
        answer=float(answer),
        case=case,
        formula_family=FORMULA_FAMILY,
        formula_text="shaded area = rectangle area - areas of two equal circles",
        reasoning_steps=2,
        radius_probabilities=dict(radius_probabilities),
        answer_support_probabilities=dict(support_probabilities),
    )
    trace_values = {
        "formula_family": FORMULA_FAMILY,
        "construction_kind": "two_circles_in_rectangle",
        "target_kind": "shaded_area",
        "support_kind": "width",
        "radius_probabilities": dict(radius_probabilities),
        "target_support_probabilities": dict(support_probabilities),
        **case_trace_values(case),
    }
    return problem, float(answer), trace_values


@register_task
class GeometryTwoCirclesInRectangleGapAreaTask:
    """Compute the shaded gap area around two equal circles in a rectangle."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SINGLE_QUERY_ID
    task_prompt_key = TASK_PROMPT_KEY
    render_scene = staticmethod(render_two_circles_rectangle_scene)
    prepare_objective = staticmethod(_prepare_two_circle_rectangle_gap_area)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate a rectangle packing shaded-gap area task."""

        return run_tangent_packing_public_entry(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GeometryTwoCirclesInRectangleGapAreaTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
