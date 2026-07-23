"""Diameter-perpendicular chord length circle-theorem task."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_integer_circle_theorem_task
from .shared.construction import _build_diameter_perpendicular_chord_scene
from .shared.state import DIAMETER_CHORD_ANSWER_SUPPORT, CircleTheoremProblem

TASK_ID = 'task_geometry__circle_theorem__diameter_perpendicular_chord_length_value'
SUPPORTED_QUERY_IDS = ('diameter_perpendicular_chord_length',)
ANSWER_SUPPORT = DIAMETER_CHORD_ANSWER_SUPPORT


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
    return problem, {}, lambda rng, bound: _build_diameter_perpendicular_chord_scene(rng, problem=bound)


@register_task
class GeometryCircleDiameterPerpendicularChordLengthValueTask:
    """Solve a missing segment from a diameter perpendicular to a chord."""

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


__all__ = ["GeometryCircleDiameterPerpendicularChordLengthValueTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
