"""Compute a station elevation from survey leveling notes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import SurveyRenderedAttempt, build_survey_task_output, render_survey_attempts
from .shared.annotations import area_scene_annotation
from .shared.defaults import load_survey_traverse_defaults
from .shared.rendering import render_leveling_station_scene
from .shared.sampling import LEVELING_CASES, choose_from_support, choose_station_labels3
from .shared.state import (
    DOMAIN,
    ElevationLevelingCase,
)

TASK_ID = "task_geometry__survey_traverse__station_elevation_value"
LEVELING_BRANCH = "leveling_station_elevation"
SUPPORTED_QUERY_IDS: tuple[str, ...] = (LEVELING_BRANCH,)
TASK_PROMPT_KEY = "station_elevation_value_query"

_GENERATION_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_survey_traverse_defaults()


@dataclass(frozen=True)
class ResolvedElevationProblem:
    """Task-owned elevation problem with formula inputs bound."""

    branch_name: str
    answer: int
    reference_elevation: int
    target_elevation: int
    station_labels: tuple[str, str, str]
    backsight: int | None
    foresight: int | None
    height_of_instrument: int | None
    formula_family: str
    branch_probabilities: dict[str, float]
    case_probabilities: dict[str, float]


def _resolve_elevation_problem(*, branch_name: str, branch_probabilities: Mapping[str, float], instance_seed: int, params: Mapping[str, Any]) -> ResolvedElevationProblem:
    """Resolve one public elevation branch into concrete field-note arithmetic."""

    labels = choose_station_labels3(params=params, instance_seed=int(instance_seed), namespace=f"{TASK_ID}.labels")
    if str(branch_name) == LEVELING_BRANCH:
        case, case_probabilities = choose_from_support(
            values=LEVELING_CASES,
            params=params,
            explicit_key="elevation_case",
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.leveling_case",
        )
        reference_elevation, backsight, foresight = [int(value) for value in case]
        height_of_instrument = int(reference_elevation + backsight)
        target_elevation = int(height_of_instrument - foresight)
        return ResolvedElevationProblem(
            branch_name=str(branch_name),
            answer=int(target_elevation),
            reference_elevation=int(reference_elevation),
            target_elevation=int(target_elevation),
            station_labels=labels,
            backsight=int(backsight),
            foresight=int(foresight),
            height_of_instrument=int(height_of_instrument),
            formula_family="survey_leveling_station_elevation",
            branch_probabilities=dict(branch_probabilities),
            case_probabilities=dict(case_probabilities),
        )
    raise ValueError(f"unsupported station elevation query: {branch_name}")


def _render_elevation_attempt(
    *,
    problem: ResolvedElevationProblem,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> SurveyRenderedAttempt:
    """Render the elevation field-note case selected by this public task."""

    def render_scene(context, attempt_seed: int):
        return render_leveling_station_scene(
            context,
            ElevationLevelingCase(
                answer=int(problem.answer),
                reference_elevation=int(problem.reference_elevation),
                backsight=int(problem.backsight or 0),
                foresight=int(problem.foresight or 0),
                height_of_instrument=int(problem.height_of_instrument or 0),
                station_labels=problem.station_labels,
                case_probabilities=dict(problem.case_probabilities),
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


def _answer_value(problem: ResolvedElevationProblem) -> int:
    """Bind the integer answer from the selected elevation formula."""

    return int(problem.answer)


def _formula_fields(problem: ResolvedElevationProblem, rendered_attempt: SurveyRenderedAttempt) -> dict[str, Any]:
    """Return task-owned elevation formula fields for trace output."""

    return {
        "reference_elevation": int(problem.reference_elevation),
        "target_elevation": int(problem.target_elevation),
        "backsight": problem.backsight,
        "foresight": problem.foresight,
        "height_of_instrument": problem.height_of_instrument,
        **dict(rendered_attempt.rendered.witness),
    }


@register_task
class GeometrySurveyTraverseStationElevationValueTask:
    """Compute a missing station elevation from a leveling diagram and field note."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int):
        """Bind the selected elevation formula, prompt, annotation, and output."""

        branch_name, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=LEVELING_BRANCH,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.query",
        )
        problem = _resolve_elevation_problem(
            branch_name=str(branch_name),
            branch_probabilities=branch_probabilities,
            instance_seed=int(instance_seed),
            params=task_params,
        )
        rendered_attempt = _render_elevation_attempt(
            problem=problem,
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
        )
        answer = _answer_value(problem)
        formula_fields = _formula_fields(problem, rendered_attempt)
        return build_survey_task_output(
            task_identity=TASK_ID,
            task_prompt_key=TASK_PROMPT_KEY,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            prompt_branch_key=str(problem.branch_name),
            formula_family=str(problem.formula_family),
            rendered_attempt=rendered_attempt,
            answer_value=int(answer),
            query_probabilities=problem.branch_probabilities,
            query_params_extra={
                "case_probabilities": dict(problem.case_probabilities),
                **dict(rendered_attempt.rendered.witness),
            },
            execution_extra=formula_fields,
            witness_extra=dict(rendered_attempt.rendered.witness),
        )


__all__ = ["GeometrySurveyTraverseStationElevationValueTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
