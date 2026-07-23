"""Infer the trapezoid extension length from completed parallelogram perimeter."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_trapezoid_extension_public_entry
from .shared.measurements import (
    case_trace_values,
    completion_length_from_perimeter,
    extension_case_pool,
    fmt_measure,
)
from .shared.rendering import render_trapezoid_extension_scene
from .shared.sampling import choose_case_by_answer
from .shared.state import DOMAIN, LabelSpec, TrapezoidExtensionProblem


TASK_ID = "task_geometry__trapezoid_extension__extension_from_parallelogram_perimeter"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "extension_from_parallelogram_perimeter_query"

_CASE_POOL = extension_case_pool()


def _prepare_extension_from_perimeter(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_query: str,
    branch_probabilities: Mapping[str, float],
) -> tuple[TrapezoidExtensionProblem, float, dict[str, Any]]:
    """Bind the parallelogram-perimeter formula for one extension-length sample."""

    if str(selected_query) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported query branch for {TASK_ID}: {selected_query}")
    case, answer_probabilities = choose_case_by_answer(
        cases=_CASE_POOL,
        answer_fn=completion_length_from_perimeter,
        params=params,
        instance_seed=int(instance_seed),
        namespace=TASK_ID,
    )
    answer = completion_length_from_perimeter(case)
    trace_values = {
        "formula_family": "extension_from_parallelogram_perimeter",
        "formula": "BE = parallelogram perimeter / 2 - AD - AB",
        "target_support_probabilities": dict(answer_probabilities),
        **case_trace_values(case),
    }
    problem = TrapezoidExtensionProblem(
        target_text="BE=?",
        target_position_key="target_extension",
        answer=float(answer),
        case=case,
        formula_family="extension_from_parallelogram_perimeter",
        formula_text="BE = parallelogram perimeter / 2 - AD - AB",
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
        include_height_in_support=False,
        annotation_mode="extension_segment",
    )
    return problem, float(answer), trace_values


@register_task
class GeometryTrapezoidExtensionFromParallelogramPerimeterTask:
    """Infer the extension length from completed parallelogram perimeter."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SINGLE_QUERY_ID
    task_prompt_key = TASK_PROMPT_KEY
    render_scene = staticmethod(render_trapezoid_extension_scene)
    prepare_objective = staticmethod(_prepare_extension_from_perimeter)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_trapezoid_extension_public_entry(self, int(instance_seed), params=params, max_attempts=int(max_attempts))


__all__ = ["GeometryTrapezoidExtensionFromParallelogramPerimeterTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
