from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_composite_shape_public_entry
from .shared.construction import resolve_answer_balanced_semicircle_dimensions
from .shared.measurements import SEMICIRCLE_DIMENSION_CANDIDATES, round1, semicircle_arc_length, semicircle_area
from .shared.sampling import group_cases_by_answer
from .shared.state import CompositeShapeProblem

TASK_ID = "task_geometry__composite_shape__rectangle_semicircle_area"
QUERY_CAP_AREA = "cap_area"
QUERY_CUTOUT_AREA = "cutout_area"
SUPPORTED_QUERY_IDS = (QUERY_CAP_AREA, QUERY_CUTOUT_AREA)


def _cap_answer_for_case(case):
    width_units, height_units, radius_units = case
    return round1(width_units * height_units + semicircle_area(radius_units))


def _cutout_answer_for_case(case):
    width_units, height_units, radius_units = case
    return round1(width_units * height_units - semicircle_area(radius_units))


_CASES_BY_QUERY = {
    QUERY_CAP_AREA: group_cases_by_answer(
        SEMICIRCLE_DIMENSION_CANDIDATES,
        answer_fn=_cap_answer_for_case,
    ),
    QUERY_CUTOUT_AREA: group_cases_by_answer(
        SEMICIRCLE_DIMENSION_CANDIDATES,
        answer_fn=_cutout_answer_for_case,
    ),
}
_BRANCHES = {
    QUERY_CAP_AREA: (
        "semi_cap",
        _cap_answer_for_case,
        "rectangle_plus_semicircle",
        "width*height + 0.5*pi*r^2",
    ),
    QUERY_CUTOUT_AREA: (
        "semi_cut",
        _cutout_answer_for_case,
        "rectangle_minus_semicircle",
        "width*height - 0.5*pi*r^2",
    ),
}


def _resolve_problem(*, selected_query: str, instance_seed, params):
    # Select the add/remove semicircle formula while keeping one annotation schema.
    try:
        shape_family, answer_fn, formula_family, area_formula = _BRANCHES[str(selected_query)]
    except KeyError as exc:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_query}")

    width_units, height_units, radius_units, answer_probabilities = (
        resolve_answer_balanced_semicircle_dimensions(
            instance_seed=int(instance_seed),
            params=params,
            namespace=f"{TASK_ID}.{selected_query}.values",
            answer_cases=_CASES_BY_QUERY[str(selected_query)],
            answer_fn=answer_fn,
        )
    )
    semi_area = semicircle_area(radius_units)
    answer = answer_fn((width_units, height_units, radius_units))
    dimensions = {
        "width_units": width_units,
        "height_units": height_units,
        "radius_units": radius_units,
        "semicircle_area": round1(semi_area),
        "arc_length": round1(semicircle_arc_length(radius_units)),
        "answer_value": answer,
    }
    return CompositeShapeProblem(
        prompt_key=str(selected_query),
        shape_family=shape_family,
        metric_kind="area",
        answer_value=answer,
        answer_type="number",
        reasoning_kind="area",
        scene_kind="geometry_curvilinear_composite_shape",
        witness_type="curvilinear_composite_formula",
        dimensions=dimensions,
        formula_family=formula_family,
        reasoning_steps=2,
        metadata_fields={
            "target_answer_support_probabilities": dict(answer_probabilities),
        },
        execution_fields={"area_formula": area_formula, "answer_rounding": "one_decimal"},
    )


@register_task
class GeometryRectangleSemicircleAreaTask:
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_composite_shape_public_entry(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_CAP_AREA,
            resolve_problem=_resolve_problem,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )
