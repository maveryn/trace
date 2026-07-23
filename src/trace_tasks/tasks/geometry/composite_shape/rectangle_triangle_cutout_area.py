"""Rectangle minus triangle area objective."""

from __future__ import annotations

from dataclasses import dataclass

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_composite_shape_public_entry
from .shared.sampling import group_cases_by_answer, select_answer_balanced_case
from .shared.state import CompositeShapeProblem

TASK_ID = "task_geometry__composite_shape__rectangle_triangle_cutout_area"
PROMPT_QUERY_KEY = "rectangle_triangle_cutout_area"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)


@dataclass(frozen=True)
class _TriangleCutoutCase:
    """Task-local dimensions for a rectangular region with a triangular cutout."""

    width: int
    height: int
    cut_base: int
    cut_height: int

    @property
    def shaded_area(self) -> int:
        return (self.width * self.height) - ((self.cut_base * self.cut_height) // 2)

    def dimensions(self) -> dict[str, int]:
        return {
            "width": self.width,
            "height": self.height,
            "cut_base": self.cut_base,
            "cut_height": self.cut_height,
        }


_CASES = tuple(
    _TriangleCutoutCase(width, height, cut_base, cut_height)
    for width in range(10, 33)
    for height in range(8, 25)
    for cut_base in range(3, width - 2)
    for cut_height in range(3, height - 2)
    if (cut_base * cut_height) % 2 == 0
)


def _answer_for_case(case: _TriangleCutoutCase) -> int:
    return int(case.shaded_area)


_CASES_BY_ANSWER = group_cases_by_answer(
    _CASES,
    answer_fn=_answer_for_case,
)


def _resolve_problem(*, selected_query: str, instance_seed, params):
    """Bind a rectangle-minus-triangle shaded-area problem."""

    case, answer_probabilities = select_answer_balanced_case(
        _CASES_BY_ANSWER,
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{TASK_ID}.{PROMPT_QUERY_KEY}.case",
    )
    return CompositeShapeProblem(
        prompt_key=PROMPT_QUERY_KEY,
        shape_family="rect_cut",
        metric_kind="area",
        answer_value=_answer_for_case(case),
        answer_type="integer",
        reasoning_kind="composite_area",
        scene_kind="geometry_rectilinear_composite_shape",
        witness_type="rectilinear_composite_area_formula",
        dimensions=case.dimensions(),
        formula_family="outer_rectangle_minus_triangle",
        reasoning_steps=3,
        metadata_fields={
            "area_case_family": "rect_cut",
            "target_answer_support_probabilities": dict(answer_probabilities),
        },
    )


@register_task
class GeometryRectangleTriangleCutoutAreaTask:
    """Compute area after subtracting a triangular cutout from a rectangle."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        """Bind the single rectangle-triangle area program."""

        return run_composite_shape_public_entry(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            resolve_problem=_resolve_problem,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )
