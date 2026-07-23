"""Inscribed-angle from arc circle-theorem task."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_integer_circle_theorem_task
from .shared.construction import _build_inscribed_angle_from_arc_scene
from .shared.state import INSCRIBED_ANGLE_ANSWER_SUPPORT, CircleTheoremProblem

TASK_ID = 'task_geometry__circle_theorem__inscribed_angle_value_inscribed_angle_from_arc'
SUPPORTED_QUERY_IDS = ('inscribed_angle_from_arc',)
ANSWER_SUPPORT = INSCRIBED_ANGLE_ANSWER_SUPPORT


def _bind_problem(
    _instance_seed: int,
    _params: Mapping[str, Any],
    _selected_query: str,
    target_answer: int,
    target_probabilities: Mapping[str, float],
):
    """Bind the answer target to this objective's theorem construction."""

    problem = CircleTheoremProblem(
        target_answer=int(target_answer),
        target_answer_probabilities=dict(target_probabilities),
    )
    return problem, {}, lambda rng, bound: _build_inscribed_angle_from_arc_scene(rng, problem=bound)


@register_task
class GeometryCircleInscribedAngleFromArcTask:
    """Solve an inscribed angle from an arc measure."""

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
            answer_support=ANSWER_SUPPORT,
            bind_problem=_bind_problem,
        )


__all__ = ["GeometryCircleInscribedAngleFromArcTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
