"""Central/inscribed angle circle-theorem task."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_integer_circle_theorem_task
from .shared.construction import (
    _build_central_angle_from_inscribed_scene,
    _build_inscribed_angle_from_central_scene,
)
from .shared.state import (
    CENTRAL_ANGLE_ANSWER_SUPPORT,
    INSCRIBED_ANGLE_ANSWER_SUPPORT,
    CircleTheoremProblem,
)

TASK_ID = "task_geometry__circle_theorem__inscribed_central_angle_value"
SUPPORTED_QUERY_IDS = ("inscribed_angle_from_central", "central_angle_from_inscribed")
ANSWER_SUPPORT_BY_QUERY = {
    "inscribed_angle_from_central": INSCRIBED_ANGLE_ANSWER_SUPPORT,
    "central_angle_from_inscribed": CENTRAL_ANGLE_ANSWER_SUPPORT,
}


def _bind_problem(
    _instance_seed: int,
    _params: Mapping[str, Any],
    selected_query: str,
    target_answer: int,
    target_probabilities: Mapping[str, float],
):
    """Bind the selected direction of the central/inscribed angle theorem."""

    problem = CircleTheoremProblem(
        target_answer=int(target_answer),
        target_answer_probabilities=dict(target_probabilities),
    )
    if str(selected_query) == "inscribed_angle_from_central":
        return (
            problem,
            {},
            lambda rng, bound: _build_inscribed_angle_from_central_scene(
                rng,
                problem=bound,
            ),
        )
    if str(selected_query) == "central_angle_from_inscribed":
        return (
            problem,
            {},
            lambda rng, bound: _build_central_angle_from_inscribed_scene(
                rng,
                problem=bound,
            ),
        )
    raise ValueError(f"unsupported circle theorem query for this task: {selected_query}")


@register_task
class GeometryCircleInscribedCentralAngleValueTask:
    """Solve either central or inscribed angle from the paired circle angle."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
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


__all__ = [
    "GeometryCircleInscribedCentralAngleValueTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
