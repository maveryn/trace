"""Compute the shaded gap area between a square and its inscribed circle."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_tangent_packing_public_entry
from .shared.measurements import case_trace_values, fmt_measure, square_container_circle_gap
from .shared.rendering import render_circle_in_square_scene
from .shared.sampling import answer_support_probabilities, choose_radius
from .shared.state import DOMAIN, SCENE_ID, TangentPackingProblem

TASK_ID = "task_geometry__tangent_packing__circle_in_square_gap_area"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "circle_in_square_gap_area_query"
FORMULA_FAMILY = "circle_in_square_gap_area"


def _prepare_circle_in_square_gap_area(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_query: str,
    branch_probabilities: Mapping[str, float],
) -> tuple[TangentPackingProblem, float, dict[str, Any]]:
    """Bind the square-minus-circle gap-area formula for one sample."""

    case, radius_probabilities = choose_radius(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.radius",
    )
    answer = square_container_circle_gap(int(case.radius))
    support_probabilities = answer_support_probabilities(
        answer_fn=square_container_circle_gap,
        key_fn=fmt_measure,
    )
    problem = TangentPackingProblem(
        construction_kind="circle_in_square",
        target_kind="shaded_area",
        support_kind="square_side",
        target_text="shaded area=?",
        support_text=f"side={case.square_side}",
        answer=float(answer),
        case=case,
        formula_family=FORMULA_FAMILY,
        formula_text="shaded area = square area - circle area",
        reasoning_steps=2,
        radius_probabilities=dict(radius_probabilities),
        answer_support_probabilities=dict(support_probabilities),
    )
    trace_values = {
        "formula_family": FORMULA_FAMILY,
        "construction_kind": "circle_in_square",
        "target_kind": "shaded_area",
        "support_kind": "square_side",
        "radius_probabilities": dict(radius_probabilities),
        "target_support_probabilities": dict(support_probabilities),
        **case_trace_values(case),
    }
    return problem, float(answer), trace_values


@register_task
class GeometryCircleInSquareGapAreaTask:
    """Compute the shaded gap area between a square and its inscribed circle."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SINGLE_QUERY_ID
    task_prompt_key = TASK_PROMPT_KEY
    render_scene = staticmethod(render_circle_in_square_scene)
    prepare_objective = staticmethod(_prepare_circle_in_square_gap_area)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate a square-container shaded-gap area task instance."""

        return run_tangent_packing_public_entry(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GeometryCircleInSquareGapAreaTask", "SCENE_ID", "SUPPORTED_QUERY_IDS", "TASK_ID"]
