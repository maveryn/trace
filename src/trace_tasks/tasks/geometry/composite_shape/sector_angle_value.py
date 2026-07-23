"""Sector central-angle objective from arc length or sector area."""

from __future__ import annotations

import math

from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_composite_shape_public_entry
from .shared.construction import resolve_answer_balanced_sector_dimensions
from .shared.measurements import SECTOR_DIMENSION_CANDIDATES, round1, sector_arc_length, sector_area
from .shared.sampling import group_cases_by_answer
from .shared.state import CompositeShapeProblem

TASK_ID = "task_geometry__composite_shape__sector_angle_value"
QUERY_FROM_ARC_LENGTH = "from_arc_length"
QUERY_FROM_SECTOR_AREA = "from_sector_area"
SUPPORTED_QUERY_IDS = (QUERY_FROM_ARC_LENGTH, QUERY_FROM_SECTOR_AREA)


def _arc_answer_for_case(case: tuple[int, int]) -> float:
    theta_degrees, radius_units = case
    arc_length = round1(sector_arc_length(theta_degrees, radius_units))
    return round1((360.0 * float(arc_length)) / (2.0 * math.pi * float(radius_units)))


def _area_answer_for_case(case: tuple[int, int]) -> float:
    theta_degrees, radius_units = case
    sector_area_value = round1(sector_area(theta_degrees, radius_units))
    return round1((360.0 * float(sector_area_value)) / (math.pi * float(radius_units) ** 2))


_CASES_BY_QUERY = {
    QUERY_FROM_ARC_LENGTH: group_cases_by_answer(
        SECTOR_DIMENSION_CANDIDATES,
        answer_fn=_arc_answer_for_case,
    ),
    QUERY_FROM_SECTOR_AREA: group_cases_by_answer(
        SECTOR_DIMENSION_CANDIDATES,
        answer_fn=_area_answer_for_case,
    ),
}


def _resolve_problem(*, selected_query: str, instance_seed, params):
    """Bind a missing sector central angle from one visible support measure."""

    if selected_query == QUERY_FROM_ARC_LENGTH:
        metric_kind = "sector_from_arc"
        answer_fn = _arc_answer_for_case
        formula_family = "sector_angle_from_arc"
        prompt_slots_key = "arc_length"
    elif selected_query == QUERY_FROM_SECTOR_AREA:
        metric_kind = "sector_from_area"
        answer_fn = _area_answer_for_case
        formula_family = "sector_angle_from_area"
        prompt_slots_key = "sector_area"
    else:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_query}")

    theta_degrees, radius_units, answer_probabilities = (
        resolve_answer_balanced_sector_dimensions(
            instance_seed=int(instance_seed),
            params=params,
            namespace=f"{TASK_ID}.{selected_query}.values",
            answer_cases=_CASES_BY_QUERY[str(selected_query)],
            answer_fn=answer_fn,
        )
    )
    arc_length = round1(sector_arc_length(theta_degrees, radius_units))
    sector_area_value = round1(sector_area(theta_degrees, radius_units))
    answer = answer_fn((theta_degrees, radius_units))
    dimensions = {
        "theta_degrees": int(theta_degrees),
        "radius_units": int(radius_units),
        "arc_length": float(arc_length),
        "sector_area": float(sector_area_value),
        "answer_value": float(answer),
    }
    prompt_value = arc_length if prompt_slots_key == "arc_length" else sector_area_value
    return CompositeShapeProblem(
        prompt_key=str(selected_query),
        shape_family="sector",
        metric_kind=metric_kind,
        answer_value=float(answer),
        answer_type="number",
        reasoning_kind="sector_angle",
        scene_kind="geometry_curvilinear_composite_shape",
        witness_type="curvilinear_composite_formula",
        dimensions=dimensions,
        formula_family=formula_family,
        reasoning_steps=2,
        prompt_slots={prompt_slots_key: prompt_value},
        metadata_fields={
            "target_answer_support_probabilities": dict(answer_probabilities),
        },
        execution_fields={"angle_formula_source": str(selected_query), "answer_rounding": "one_decimal"},
    )


@register_task
class GeometrySectorAngleValueTask:
    """Compute a sector central angle from arc length or sector area."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        """Select the sector-angle branch and bind its formula inputs."""

        return run_composite_shape_public_entry(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_FROM_ARC_LENGTH,
            resolve_problem=_resolve_problem,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )
