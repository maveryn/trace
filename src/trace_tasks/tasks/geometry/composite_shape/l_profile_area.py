from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_composite_shape_public_entry
from .shared.sampling import group_cases_by_answer, select_answer_balanced_case
from .shared.state import CompositeShapeProblem

TASK_ID = "task_geometry__composite_shape__l_profile_area"
PROMPT_QUERY_KEY = "l_profile_area"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)

_CASES = tuple(
    (width, height, cut_width, cut_height)
    for width in range(10, 35)
    for height in range(8, 27)
    for cut_width in range(3, width - 2)
    for cut_height in range(3, height - 2)
)


def _answer_for_case(case: tuple[int, int, int, int]) -> int:
    width, height, cut_width, cut_height = case
    return (int(width) * int(height)) - (int(cut_width) * int(cut_height))


_CASES_BY_ANSWER = group_cases_by_answer(
    _CASES,
    answer_fn=_answer_for_case,
)


def _resolve_problem(*, selected_query: str, instance_seed, params):
    """Bind an L-profile shaded-area problem."""

    (width, height, cut_width, cut_height), answer_probabilities = select_answer_balanced_case(
        _CASES_BY_ANSWER,
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{TASK_ID}.{PROMPT_QUERY_KEY}.case",
    )
    answer = _answer_for_case((width, height, cut_width, cut_height))
    return CompositeShapeProblem(
        prompt_key=PROMPT_QUERY_KEY,
        shape_family="l_profile",
        metric_kind="area",
        answer_value=int(answer),
        answer_type="integer",
        reasoning_kind="composite_area",
        scene_kind="geometry_rectilinear_composite_shape",
        witness_type="rectilinear_composite_area_formula",
        dimensions={"width": width, "height": height, "cut_width": cut_width, "cut_height": cut_height},
        formula_family="outer_rectangle_minus_corner_rectangle",
        reasoning_steps=3,
        metadata_fields={
            "area_case_family": "l_profile",
            "target_answer_support_probabilities": dict(answer_probabilities),
        },
        execution_fields={"area_formula": "width*height - cut_width*cut_height"},
    )


@register_task
class GeometryLProfileAreaTask:
    """Compute area after subtracting a corner rectangle from an L-profile."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        """Bind the single L-profile area program."""

        return run_composite_shape_public_entry(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            resolve_problem=_resolve_problem,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )
