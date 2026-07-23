"""Secant-secant length circle-theorem task."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_integer_circle_theorem_task
from .shared.construction import (
    _build_secant_secant_scene,
    _build_secant_secant_variable_scene,
)
from .shared.sampling import _feasible_secant_secant_variable_target_kinds
from .shared.state import (
    SECANT_SECANT_ANSWER_SUPPORT,
    VARIABLE_SECANT_ANSWER_SUPPORT,
    CircleTheoremProblem,
)

TASK_ID = "task_geometry__circle_theorem__secant_secant_length_value"
SUPPORTED_QUERY_IDS = ("secant_secant_length", "secant_secant_variable_segment_length")
ANSWER_SUPPORT_BY_QUERY = {
    "secant_secant_length": SECANT_SECANT_ANSWER_SUPPORT,
    "secant_secant_variable_segment_length": VARIABLE_SECANT_ANSWER_SUPPORT,
}


def _select_variable_target_kind(
    *, instance_seed: int, params: Mapping[str, Any], target_answer: int
) -> tuple[str, Dict[str, float]]:
    """Select which secant segment is hidden in the variable branch."""

    feasible = _feasible_secant_secant_variable_target_kinds(int(target_answer))
    if not feasible:
        raise ValueError(f"unsupported target answer for variable secant theorem: {target_answer}")
    explicit = params.get("secant_secant_variable_target_kind")
    selected = str(explicit) if explicit is not None else str(spawn_rng(int(instance_seed), f"{TASK_ID}.variable_target_kind").choice(feasible))
    if selected not in set(feasible):
        raise ValueError(f"unsupported variable secant target kind: {selected!r}")
    probability = 1.0 / float(len(feasible))
    return selected, {str(kind): float(probability) for kind in feasible}


def _bind_problem(
    instance_seed: int,
    params: Mapping[str, Any],
    selected_query: str,
    target_answer: int,
    target_probabilities: Mapping[str, float],
):
    """Bind either fixed or variable secant-secant power construction."""

    if str(selected_query) == "secant_secant_length":
        problem = CircleTheoremProblem(
            target_answer=int(target_answer),
            target_answer_probabilities=dict(target_probabilities),
        )
        return problem, {}, lambda rng, bound: _build_secant_secant_scene(rng, problem=bound)
    if str(selected_query) == "secant_secant_variable_segment_length":
        target_kind, target_kind_probabilities = _select_variable_target_kind(
            instance_seed=int(instance_seed),
            params=params,
            target_answer=int(target_answer),
        )
        problem = CircleTheoremProblem(
            target_answer=int(target_answer),
            target_answer_probabilities=dict(target_probabilities),
            secant_secant_variable_target_kind=str(target_kind),
            secant_secant_variable_target_kind_probabilities=dict(target_kind_probabilities),
        )
        return (
            problem,
            {
                "secant_secant_variable_target_kind": str(target_kind),
                "secant_secant_variable_target_kind_probabilities": dict(target_kind_probabilities),
            },
            lambda rng, bound: _build_secant_secant_variable_scene(rng, problem=bound),
        )
    raise ValueError(f"unsupported circle theorem query for this task: {selected_query}")


@register_task
class GeometryCircleSecantSecantLengthValueTask:
    """Solve a missing length from two secants."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(
        self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int
    ) -> TaskOutput:
        return run_integer_circle_theorem_task(
            task_id=TASK_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            answer_support=ANSWER_SUPPORT_BY_QUERY,
            bind_problem=_bind_problem,
        )


__all__ = ["GeometryCircleSecantSecantLengthValueTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
