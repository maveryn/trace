"""Infer the inscribed square side length from the shaded circle gap area."""

from __future__ import annotations

import math

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_tangent_packing_public_entry
from .shared.measurements import case_trace_values, circle_container_square_gap, fmt_measure, inscribed_square_side
from .shared.rendering import render_square_in_circle_scene
from .shared.sampling import choose_radius
from .shared.state import DOMAIN, TangentPackingProblem

TASK_ID = "task_geometry__tangent_packing__square_in_circle_side_from_gap_area"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "square_in_circle_side_from_gap_area_query"
FORMULA_FAMILY = "square_in_circle_side_from_gap_area"


def _prepare_square_side_from_gap_area(
    *,
    instance_seed,
    params,
    selected_query,
    branch_probabilities,
) -> tuple[TangentPackingProblem, float, dict[str, float | int | str]]:
    """Bind a circle-container gap area to the inscribed square side length."""

    case, radius_probabilities = choose_radius(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.radius",
    )
    answer = inscribed_square_side(case.radius)
    shaded_area = circle_container_square_gap(case.radius)
    circle_area = float(math.pi * case.radius * case.radius)
    inscribed_square_area = float(2 * case.radius * case.radius)
    side_from_radius = float(math.sqrt(inscribed_square_area))
    problem = TangentPackingProblem(
        construction_kind="square_in_circle",
        target_kind="square_side",
        support_kind="shaded_area",
        target_text="square side=?",
        support_text=f"shaded area={fmt_measure(shaded_area)}",
        answer=float(answer),
        case=case,
        formula_family=FORMULA_FAMILY,
        formula_text="inscribed square side = sqrt(2 * shaded area / (pi - 2))",
        reasoning_steps=1,
    )
    trace_values = case_trace_values(case)
    trace_values.update(
        {
            "formula_family": FORMULA_FAMILY,
            "construction_kind": "square_in_circle",
            "target_kind": "square_side",
            "support_kind": "shaded_area",
            "visible_shaded_area": float(shaded_area),
            "circle_area": circle_area,
            "inscribed_square_area": inscribed_square_area,
            "side_from_radius": side_from_radius,
        }
    )
    return problem, float(answer), trace_values


@register_task
class GeometrySquareInCircleSideFromGapAreaTask:
    """Infer the inscribed square side length from the shaded circle gap area."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SINGLE_QUERY_ID
    task_prompt_key = TASK_PROMPT_KEY
    render_scene = staticmethod(render_square_in_circle_scene)
    prepare_objective = staticmethod(_prepare_square_side_from_gap_area)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate an inverse side-length task for the circle container."""

        return run_tangent_packing_public_entry(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GeometrySquareInCircleSideFromGapAreaTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
