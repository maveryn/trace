"""House-outline perimeter objective."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_composite_shape_public_entry
from .shared.sampling import group_cases_by_answer, select_answer_balanced_case
from .shared.state import CompositeShapeProblem

TASK_ID = "task_geometry__composite_shape__house_outline_perimeter"
QUERY_ID = "house_outline_perimeter"
SUPPORTED_QUERY_IDS = (QUERY_ID,)

_CASES = tuple(
    (width, wall_height, roof_side)
    for width in range(8, 25)
    for wall_height in range(5, 17)
    for roof_side in range((width // 2) + 2, (width // 2) + 11)
)


def _answer(case: tuple[int, int, int]) -> int:
    width, wall_height, roof_side = case
    return int(width) + (2 * int(wall_height)) + (2 * int(roof_side))


_CASES_BY_ANSWER = group_cases_by_answer(_CASES, answer_fn=_answer)


def _resolve_problem(*, selected_query: str, instance_seed, params):
    """Bind the pentagonal outline perimeter from wall and roof dimensions."""

    (
        width,
        wall_height,
        roof_side,
    ), answer_probabilities = select_answer_balanced_case(
        _CASES_BY_ANSWER,
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{TASK_ID}.{QUERY_ID}.case",
    )
    answer = _answer((width, wall_height, roof_side))
    return CompositeShapeProblem(
        prompt_key=QUERY_ID,
        shape_family="house",
        metric_kind="perimeter",
        answer_value=int(answer),
        answer_type="integer",
        reasoning_kind="composite_perimeter",
        scene_kind="geometry_rectilinear_composite_shape",
        witness_type="rectilinear_composite_perimeter_formula",
        dimensions={"width": width, "wall_height": wall_height, "roof_side": roof_side},
        formula_family="house_outline",
        reasoning_steps=2,
        metadata_fields={
            "target_answer_support_probabilities": dict(answer_probabilities),
        },
        execution_fields={"perimeter_formula": "base + 2*wall_height + 2*roof_side"},
    )


@register_task
class GeometryMeasurementCompositePerimeterValueTask:
    """Compute the outer perimeter of a house-outline composite shape."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        """Bind a house-outline case and construct the perimeter output."""

        return run_composite_shape_public_entry(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            resolve_problem=_resolve_problem,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )
