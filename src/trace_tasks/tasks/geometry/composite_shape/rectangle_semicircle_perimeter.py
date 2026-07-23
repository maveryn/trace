"""Rectangle plus/minus semicircle perimeter objective."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_composite_shape_public_entry
from .shared.construction import resolve_semicircle_side_remainder_perimeter_case
from .shared.measurements import SEMICIRCLE_DIMENSION_CANDIDATES, semicircle_side_remainder_perimeter
from .shared.sampling import group_cases_by_answer
from .shared.state import CompositeShapeProblem

TASK_ID = "task_geometry__composite_shape__rectangle_semicircle_perimeter"
QUERY_CAP_PERIMETER = "cap_perimeter"
QUERY_CUTOUT_PERIMETER = "cutout_perimeter"
SUPPORTED_QUERY_IDS = (QUERY_CAP_PERIMETER, QUERY_CUTOUT_PERIMETER)


def _answer_for_case(case: tuple[int, int, int]) -> float:
    width_units, height_units, radius_units = case
    return semicircle_side_remainder_perimeter(width_units, height_units, radius_units)


_PERIMETER_CASES = tuple(
    case
    for case in SEMICIRCLE_DIMENSION_CANDIDATES
    if int(case[1]) > (2 * int(case[2]))
)

_CASES_BY_ANSWER = group_cases_by_answer(
    _PERIMETER_CASES,
    answer_fn=_answer_for_case,
)


def _resolve_problem(*, selected_query: str, instance_seed, params):
    """Bind highlighted perimeter for a rectangle with a semicircle cap or cutout."""

    if selected_query == QUERY_CAP_PERIMETER:
        shape_family = "semi_cap"
        formula_family = "rectangle_semicircle_cap_boundary_with_side_remainders"
    elif selected_query == QUERY_CUTOUT_PERIMETER:
        shape_family = "semi_cut"
        formula_family = "rectangle_semicircle_cutout_boundary_with_side_remainders"
    else:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_query}")

    resolved = resolve_semicircle_side_remainder_perimeter_case(
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{TASK_ID}.{selected_query}.values",
        answer_cases=_CASES_BY_ANSWER,
        answer_fn=_answer_for_case,
    )
    return CompositeShapeProblem(
        prompt_key=str(selected_query),
        shape_family=shape_family,
        metric_kind="perimeter",
        answer_value=float(resolved.answer),
        answer_type="number",
        reasoning_kind="perimeter",
        scene_kind="geometry_curvilinear_composite_shape",
        witness_type="curvilinear_composite_formula",
        dimensions=resolved.dimensions,
        formula_family=formula_family,
        reasoning_steps=2,
        metadata_fields={
            "target_answer_support_probabilities": dict(resolved.answer_probabilities),
        },
        execution_fields=resolved.execution_fields,
    )


@register_task
class GeometryRectangleSemicirclePerimeterTask:
    """Compute the perimeter of a rectangle with a semicircle cap or cutout."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        """Select the semicircle perimeter branch and bind its formula inputs."""

        return run_composite_shape_public_entry(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_CAP_PERIMETER,
            resolve_problem=_resolve_problem,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )
