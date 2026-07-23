"""Tabbed rectilinear perimeter objective."""

from __future__ import annotations

from dataclasses import dataclass

from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_composite_shape_public_entry
from .shared.sampling import group_cases_by_answer, select_answer_balanced_case
from .shared.state import CompositeShapeProblem

TASK_ID = "task_geometry__composite_shape__tabbed_rectilinear_perimeter"
QUERY_ID = "tabbed_rectilinear_perimeter"
SUPPORTED_QUERY_IDS = (QUERY_ID,)


@dataclass(frozen=True)
class TabbedOutlineCase:
    """All visible edge lengths needed by the tabbed-outline perimeter objective."""

    total_width: int
    side_height: int
    tab_rise: int
    tab_top_width: int

    @property
    def shoulder_width(self) -> int:
        return (int(self.total_width) - int(self.tab_top_width)) // 2


_CASES = tuple(
    TabbedOutlineCase(
        total_width=width,
        side_height=height,
        tab_rise=tab_height,
        tab_top_width=tab_width,
    )
    for width in range(12, 47)
    for height in range(6, 35)
    for tab_height in range(2, 21)
    for tab_width in range(4, width - 3)
    if (width - tab_width) % 2 == 0
)


def _tabbed_outline_perimeter(case: TabbedOutlineCase) -> int:
    """Return the perimeter after paired shoulders and the tab top collapse algebraically."""

    return (
        (2 * int(case.total_width))
        + (2 * int(case.side_height))
        + (2 * int(case.tab_rise))
    )


def _visible_lengths(case: TabbedOutlineCase) -> dict[str, int]:
    """Expose the side labels drawn by the renderer for the tabbed polygon."""

    return {
        "width": int(case.total_width),
        "height": int(case.side_height),
        "tab_height": int(case.tab_rise),
        "tab_width": int(case.tab_top_width),
        "shoulder_width": int(case.shoulder_width),
    }


_CASES_BY_ANSWER = group_cases_by_answer(_CASES, answer_fn=_tabbed_outline_perimeter)


def _resolve_problem(*, selected_query: str, instance_seed, params):
    """Bind a rectilinear tab perimeter from total width, height, and tab height."""

    case, answer_probabilities = select_answer_balanced_case(
        _CASES_BY_ANSWER,
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{TASK_ID}.{QUERY_ID}.case",
    )
    answer = _tabbed_outline_perimeter(case)
    return CompositeShapeProblem(
        prompt_key=QUERY_ID,
        shape_family="tabbed",
        metric_kind="perimeter",
        answer_value=int(answer),
        answer_type="integer",
        reasoning_kind="composite_perimeter",
        scene_kind="geometry_rectilinear_composite_shape",
        witness_type="rectilinear_composite_perimeter_formula",
        dimensions=_visible_lengths(case),
        formula_family="tabbed_rectilinear_outline",
        reasoning_steps=2,
        metadata_fields={
            "target_answer_support_probabilities": dict(answer_probabilities),
        },
        execution_fields={
            "perimeter_formula": "AB + BC + CD + DE + EF + FG + GH + HA",
            "equivalent_formula": "2*width + 2*height + 2*tab_height",
        },
    )


@register_task
class GeometryCompositeShapeTabbedRectilinearPerimeterTask:
    """Compute the perimeter of a tabbed rectilinear composite shape."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        """Bind a tabbed-outline case and construct the perimeter output."""

        return run_composite_shape_public_entry(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            resolve_problem=_resolve_problem,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )
