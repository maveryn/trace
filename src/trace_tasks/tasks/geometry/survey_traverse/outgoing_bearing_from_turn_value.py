"""Compute an outgoing survey bearing from an incoming bearing and turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import SurveyRenderedAttempt, build_survey_task_output, render_survey_attempts
from .shared.annotations import area_scene_annotation
from .shared.defaults import load_survey_traverse_defaults
from .shared.measurements import normalize_bearing
from .shared.rendering import render_closed_traverse_scene
from .shared.sampling import BEARING_SUPPORT, choose_from_support, choose_station_labels3, choose_turn
from .shared.state import DOMAIN, SCENE_ID, BearingTurnCase

TASK_ID = "task_geometry__survey_traverse__outgoing_bearing_from_turn_value"
SINGLE_QUERY_ID = "single"
SUPPORTED_QUERY_IDS: tuple[str, ...] = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "outgoing_bearing_from_turn_value_query"

_GENERATION_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_survey_traverse_defaults()


@dataclass(frozen=True)
class ResolvedOutgoingBearingProblem:
    """Task-owned outgoing-bearing problem with formula inputs bound."""

    query_id: str
    answer: int
    base_bearing: int
    turn_angle: int
    turn_direction: str
    station_labels: tuple[str, str, str]
    query_probabilities: dict[str, float]
    bearing_probabilities: dict[str, float]
    turn_probabilities: dict[str, float]


def _resolve_problem(*, query_id: str, query_probabilities: Mapping[str, float], instance_seed: int, params: Mapping[str, Any]) -> ResolvedOutgoingBearingProblem:
    """Resolve the single traverse-turn objective into concrete bearing inputs."""

    labels = choose_station_labels3(params=params, instance_seed=int(instance_seed), namespace=f"{TASK_ID}.labels")
    base_bearing, bearing_probabilities = choose_from_support(
        values=BEARING_SUPPORT,
        params=params,
        explicit_key="base_bearing",
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.base_bearing",
    )
    turn_angle, turn_direction, turn_probabilities = choose_turn(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.turn",
    )
    answer = (
        normalize_bearing(int(base_bearing) - int(turn_angle))
        if str(turn_direction) == "left"
        else normalize_bearing(int(base_bearing) + int(turn_angle))
    )
    if int(answer) == 0:
        answer = 360
    if int(answer) >= 360:
        raise ValueError("closed traverse generated unsupported 360-degree answer")
    return ResolvedOutgoingBearingProblem(
        query_id=str(query_id),
        answer=int(answer),
        base_bearing=int(base_bearing),
        turn_angle=int(turn_angle),
        turn_direction=str(turn_direction),
        station_labels=labels,
        query_probabilities=dict(query_probabilities),
        bearing_probabilities=dict(bearing_probabilities),
        turn_probabilities=dict(turn_probabilities),
    )


def _render_attempt(
    *,
    problem: ResolvedOutgoingBearingProblem,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> SurveyRenderedAttempt:
    """Render the traverse-turn scene bound to this public task."""

    def render_scene(context, attempt_seed: int):
        return render_closed_traverse_scene(
            context,
            BearingTurnCase(
                answer=int(problem.answer),
                base_bearing=int(problem.base_bearing),
                turn_angle=int(problem.turn_angle),
                turn_direction=str(problem.turn_direction),
                station_labels=problem.station_labels,
                bearing_probabilities=dict(problem.bearing_probabilities),
                turn_probabilities=dict(problem.turn_probabilities),
            ),
            instance_seed=int(attempt_seed),
        )

    return render_survey_attempts(
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
        render_defaults=_RENDER_DEFAULTS,
        render_scene=render_scene,
        build_annotation=area_scene_annotation,
    )


@register_task
class GeometrySurveyTraverseOutgoingBearingFromTurnValueTask:
    """Compute an outgoing bearing from the shown traverse turn."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations', 'formula_evaluation')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int):
        """Bind the single traverse-turn formula, prompt, annotation, and output."""

        query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.query",
        )
        problem = _resolve_problem(
            query_id=str(query_id),
            query_probabilities=query_probabilities,
            instance_seed=int(instance_seed),
            params=task_params,
        )
        rendered_attempt = _render_attempt(
            problem=problem,
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
        )
        answer = int(problem.answer)
        return build_survey_task_output(
            task_identity=TASK_ID,
            task_prompt_key=TASK_PROMPT_KEY,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            prompt_branch_key=str(problem.query_id),
            formula_family="survey_outgoing_bearing_from_turn",
            rendered_attempt=rendered_attempt,
            answer_value=int(answer),
            query_probabilities=problem.query_probabilities,
            query_params_extra={
                "bearing_probabilities": dict(problem.bearing_probabilities),
                "turn_probabilities": dict(problem.turn_probabilities),
                **dict(rendered_attempt.rendered.witness),
            },
            execution_extra={
                "known_bearing": int(problem.base_bearing),
                "turn_angle": int(problem.turn_angle),
                "turn_direction": str(problem.turn_direction),
                "target_bearing": int(problem.answer),
                **dict(rendered_attempt.rendered.witness),
            },
            witness_extra=dict(rendered_attempt.rendered.witness),
        )


__all__ = [
    "GeometrySurveyTraverseOutgoingBearingFromTurnValueTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
