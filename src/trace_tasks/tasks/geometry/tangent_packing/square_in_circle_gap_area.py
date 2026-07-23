"""Compute the shaded gap area between a circle and its inscribed square."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_tangent_packing_public_entry
from .shared.measurements import case_trace_values, circle_container_square_gap, fmt_measure
from .shared.rendering import render_square_in_circle_scene
from .shared.sampling import answer_support_probabilities, choose_radius
from .shared.state import DOMAIN, TangentPackingProblem

TASK_ID = "task_geometry__tangent_packing__square_in_circle_gap_area"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "square_in_circle_gap_area_query"
FORMULA_FAMILY = "square_in_circle_gap_area"


def _prepare_square_in_circle_gap_area(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_query: str,
    branch_probabilities: Mapping[str, float],
) -> tuple[TangentPackingProblem, float, dict[str, Any]]:
    """Bind the circle-minus-square gap-area formula for one sample."""

    case, radius_probabilities = choose_radius(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.radius",
    )
    answer = circle_container_square_gap(int(case.radius))
    support_probabilities = answer_support_probabilities(
        answer_fn=circle_container_square_gap,
        key_fn=fmt_measure,
    )
    problem = TangentPackingProblem(
        construction_kind="square_in_circle",
        target_kind="shaded_area",
        support_kind="radius",
        target_text="shaded area=?",
        support_text=f"r={case.radius}",
        answer=float(answer),
        case=case,
        formula_family=FORMULA_FAMILY,
        formula_text="shaded area = circle area - inscribed square area",
        reasoning_steps=2,
        radius_probabilities=dict(radius_probabilities),
        answer_support_probabilities=dict(support_probabilities),
    )
    trace_values = {
        "formula_family": FORMULA_FAMILY,
        "construction_kind": "square_in_circle",
        "target_kind": "shaded_area",
        "support_kind": "radius",
        "radius_probabilities": dict(radius_probabilities),
        "target_support_probabilities": dict(support_probabilities),
        **case_trace_values(case),
    }
    return problem, float(answer), trace_values


@register_task
class GeometrySquareInCircleGapAreaTask:
    """Compute the shaded gap area between a circle and its inscribed square."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SINGLE_QUERY_ID
    task_prompt_key = TASK_PROMPT_KEY
    render_scene = staticmethod(render_square_in_circle_scene)
    prepare_objective = staticmethod(_prepare_square_in_circle_gap_area)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate a circle-container shaded-gap area task instance."""

        return run_tangent_packing_public_entry(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GeometrySquareInCircleGapAreaTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
