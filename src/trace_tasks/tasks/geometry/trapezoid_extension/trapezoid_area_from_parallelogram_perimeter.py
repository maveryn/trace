from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_trapezoid_extension_public_entry
from .shared.measurements import (
    area_case_pool,
    case_trace_values,
    fmt_measure,
    trapezoid_area_from_parallelogram_perimeter,
)
from .shared.rendering import render_trapezoid_extension_scene
from .shared.sampling import choose_case_by_answer
from .shared.state import DOMAIN, LabelSpec, TrapezoidExtensionProblem


TASK_ID = "task_geometry__trapezoid_extension__trapezoid_area_from_parallelogram_perimeter"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "trapezoid_area_from_parallelogram_perimeter_query"

_CASE_POOL = area_case_pool()


def _prepare_area_from_parallelogram_perimeter(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_query: str,
    branch_probabilities: Mapping[str, float],
) -> tuple[TrapezoidExtensionProblem, float, dict[str, Any]]:
    """Bind the perimeter-derived parallelogram side before solving area.

    The task invariant is that the displayed parallelogram perimeter and side
    determine the missing trapezoid base used in the final area computation.
    """

    if str(selected_query) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported query branch for {TASK_ID}: {selected_query}")
    case, answer_probabilities = choose_case_by_answer(
        cases=_CASE_POOL,
        answer_fn=trapezoid_area_from_parallelogram_perimeter,
        params=params,
        instance_seed=int(instance_seed),
        namespace=TASK_ID,
    )
    answer = trapezoid_area_from_parallelogram_perimeter(case)
    trace_values = {
        "formula_family": "trapezoid_area_from_parallelogram_perimeter",
        "formula": "trapezoid area = h * (AB + DC) / 2, with DC = parallelogram perimeter / 2 - AD",
        "target_support_probabilities": dict(answer_probabilities),
        **case_trace_values(case),
    }
    problem = TrapezoidExtensionProblem(
        target_text="trapezoid area=?",
        target_position_key="target_area",
        answer=float(answer),
        case=case,
        formula_family="trapezoid_area_from_parallelogram_perimeter",
        formula_text="trapezoid area = h * (AB + DC) / 2, with DC = parallelogram perimeter / 2 - AD",
        reasoning_steps=2,
        support_labels=(
            LabelSpec(
                "parallelogram_perimeter",
                f"parallelogram perimeter={fmt_measure(case.parallelogram_perimeter)}",
                "parallelogram_perimeter",
            ),
            LabelSpec("side", f"AD={fmt_measure(case.side)}", "side"),
        ),
        target_support_probabilities=dict(answer_probabilities),
    )
    return problem, float(answer), trace_values


@register_task
class GeometryTrapezoidAreaFromParallelogramPerimeterTask:
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SINGLE_QUERY_ID
    task_prompt_key = TASK_PROMPT_KEY
    render_scene = staticmethod(render_trapezoid_extension_scene)
    prepare_objective = staticmethod(_prepare_area_from_parallelogram_perimeter)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_trapezoid_extension_public_entry(self, int(instance_seed), params=params, max_attempts=int(max_attempts))
