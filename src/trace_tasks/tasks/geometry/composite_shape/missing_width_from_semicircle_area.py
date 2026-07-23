"""Missing width from rectangle plus/minus semicircle area objective."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_composite_shape_public_entry
from .shared.construction import resolve_answer_balanced_semicircle_dimensions
from .shared.measurements import round1, semicircle_arc_length, semicircle_area
from .shared.sampling import group_cases_by_answer
from .shared.state import CompositeShapeProblem

TASK_ID = "task_geometry__composite_shape__missing_width_from_semicircle_area"
QUERY_CAP_FROM_TOTAL_AREA = "cap_from_total_area"
QUERY_CUTOUT_FROM_TOTAL_AREA = "cutout_from_total_area"
SUPPORTED_QUERY_IDS = (QUERY_CAP_FROM_TOTAL_AREA, QUERY_CUTOUT_FROM_TOTAL_AREA)


def _answer_for_case(case: tuple[int, int, int]) -> float:
    width_units, _height_units, _radius_units = case
    return float(width_units)


_MISSING_WIDTH_CASES = tuple(
    (width, height, radius)
    for width in range(8, 73)
    for height in range(6, 27)
    for radius in range(3, min(12, height // 2) + 1)
)
_CASES_BY_ANSWER = group_cases_by_answer(
    _MISSING_WIDTH_CASES,
    answer_fn=_answer_for_case,
)


def _resolve_problem(*, selected_query: str, instance_seed, params):
    """Bind missing rectangle width from a total semicircle-composite area."""

    if selected_query == QUERY_CAP_FROM_TOTAL_AREA:
        shape_family = "semi_cap"
        formula_family = "width_from_area_plus_semicircle"
        area_operation = "add"
    elif selected_query == QUERY_CUTOUT_FROM_TOTAL_AREA:
        shape_family = "semi_cut"
        formula_family = "width_from_area_minus_semicircle"
        area_operation = "subtract"
    else:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_query}")

    width_units, height_units, radius_units, answer_probabilities = (
        resolve_answer_balanced_semicircle_dimensions(
            instance_seed=int(instance_seed),
            params=params,
            namespace=f"{TASK_ID}.{selected_query}.values",
            answer_cases=_CASES_BY_ANSWER,
            answer_fn=_answer_for_case,
        )
    )
    semi_area = semicircle_area(radius_units)
    if area_operation == "add":
        total_area = round1(float(width_units * height_units) + float(semi_area))
    else:
        total_area = round1(float(width_units * height_units) - float(semi_area))
    dimensions = {
        "width_units": int(width_units),
        "height_units": int(height_units),
        "radius_units": int(radius_units),
        "total_area": float(total_area),
        "semicircle_area": round1(semi_area),
        "arc_length": round1(semicircle_arc_length(radius_units)),
        "answer_value": float(width_units),
    }
    return CompositeShapeProblem(
        prompt_key=str(selected_query),
        shape_family=shape_family,
        metric_kind="missing_width",
        answer_value=float(width_units),
        answer_type="number",
        reasoning_kind="missing_width",
        scene_kind="geometry_curvilinear_composite_shape",
        witness_type="curvilinear_composite_formula",
        dimensions=dimensions,
        formula_family=formula_family,
        reasoning_steps=3,
        prompt_slots={"total_area": total_area},
        metadata_fields={
            "target_answer_support_probabilities": dict(answer_probabilities),
        },
        execution_fields={"area_operation": area_operation, "answer_rounding": "one_decimal"},
    )


@register_task
class GeometryMissingWidthFromSemicircleAreaTask:
    """Infer a missing rectangle width from cap or cutout total area."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        """Select the missing-width branch and bind its formula inputs."""

        return run_composite_shape_public_entry(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_CAP_FROM_TOTAL_AREA,
            resolve_problem=_resolve_problem,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )
