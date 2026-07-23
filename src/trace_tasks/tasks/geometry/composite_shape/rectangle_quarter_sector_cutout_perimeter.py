"""Rectangle minus quarter-sector perimeter objective."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_composite_shape_public_entry
from .shared.construction import resolve_answer_balanced_quarter_cut_dimensions
from .shared.measurements import QUARTER_CUT_DIMENSION_CANDIDATES, quarter_sector_values, round1
from .shared.sampling import group_cases_by_answer
from .shared.state import CompositeShapeProblem

TASK_ID = "task_geometry__composite_shape__rectangle_quarter_sector_cutout_perimeter"
QUERY_ID = "rectangle_quarter_sector_cutout_perimeter"
SUPPORTED_QUERY_IDS = (QUERY_ID,)


def _answer_for_case(case: tuple[int, int, int]) -> float:
    width_units, height_units, radius_units = case
    _sector_area, arc_length = quarter_sector_values(radius_units)
    straight_boundary = (
        (2.0 * float(width_units))
        + (2.0 * float(height_units))
        - (2.0 * float(radius_units))
    )
    return round1(float(straight_boundary) + float(arc_length))


_CASES_BY_ANSWER = group_cases_by_answer(
    QUARTER_CUT_DIMENSION_CANDIDATES,
    answer_fn=_answer_for_case,
)

def _resolve_problem(*, selected_query: str, instance_seed, params):
    """Bind highlighted boundary length for a quarter-sector cutout figure."""

    width_units, height_units, radius_units, answer_probabilities = (
        resolve_answer_balanced_quarter_cut_dimensions(
            instance_seed=int(instance_seed),
            params=params,
            namespace=f"{TASK_ID}.{QUERY_ID}.values",
            answer_cases=_CASES_BY_ANSWER,
            answer_fn=_answer_for_case,
        )
    )
    sector_area, arc_length = quarter_sector_values(radius_units)
    straight_boundary = (2.0 * float(width_units)) + (2.0 * float(height_units)) - (2.0 * float(radius_units))
    answer = _answer_for_case((width_units, height_units, radius_units))
    dimensions = {"width_units": width_units, "height_units": height_units, "radius_units": radius_units, "theta_degrees": 90, "sector_area": round1(sector_area), "arc_length": round1(arc_length), "straight_boundary_length": round1(straight_boundary), "answer_value": answer}
    return CompositeShapeProblem(prompt_key=QUERY_ID, shape_family="quarter_cut", metric_kind="perimeter", answer_value=float(answer), answer_type="number", reasoning_kind="perimeter", scene_kind="geometry_curvilinear_composite_shape", witness_type="curvilinear_composite_formula", dimensions=dimensions, formula_family="rectangle_quarter_sector_boundary", reasoning_steps=3, metadata_fields={"target_answer_support_probabilities": dict(answer_probabilities)})


@register_task
class GeometryRectangleQuarterSectorCutoutPerimeterTask:
    """Compute the perimeter of a rectangle with a quarter-sector cutout."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        """Select the sole perimeter branch and bind its formula inputs."""

        return run_composite_shape_public_entry(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            resolve_problem=_resolve_problem,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )
