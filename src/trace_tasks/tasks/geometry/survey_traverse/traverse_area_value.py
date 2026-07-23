"""Compute a survey traverse area from offset field notes."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import build_survey_task_output, render_survey_attempts
from .shared.annotations import area_scene_annotation
from .shared.defaults import load_survey_traverse_defaults
from .shared.measurements import offset_area_from_chainages
from .shared.rendering import render_offset_area_scene
from .shared.sampling import OFFSET_TRAPEZOID_CASES, choose_from_support, choose_station_labels4
from .shared.state import (
    DOMAIN,
    AreaOffsetCase,
)

TASK_ID = "task_geometry__survey_traverse__traverse_area_value"
OFFSET_AREA_BRANCH = "offset_trapezoid_area"
SUPPORTED_QUERY_IDS: tuple[str, ...] = (OFFSET_AREA_BRANCH,)
TASK_PROMPT_KEY = "traverse_area_value_query"

_GENERATION_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_survey_traverse_defaults()


def _offset_area_scene_inputs(*, instance_seed: int, params: Mapping[str, Any]) -> tuple[AreaOffsetCase, dict[str, float]]:
    """Choose offset survey notes and bind their trapezoid-rule area."""

    labels = choose_station_labels4(params=params, instance_seed=int(instance_seed), namespace=f"{TASK_ID}.labels")
    case, case_probabilities = choose_from_support(
        values=OFFSET_TRAPEZOID_CASES,
        params=params,
        explicit_key="area_case",
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.offset_case",
    )
    chainages = (0, int(case[0]), int(case[1]), int(case[2]))
    offsets = (int(case[3]), int(case[4]), int(case[5]), int(case[6]))
    return (
        AreaOffsetCase(
            answer=int(offset_area_from_chainages(chainages, offsets)),
            station_labels=labels,
            chainages=chainages,
            offsets=offsets,
            case_probabilities=dict(case_probabilities),
        ),
        dict(case_probabilities),
    )


@register_task
class GeometrySurveyTraverseTraverseAreaValueTask:
    """Compute a survey traverse area from offset field notes."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int):
        """Bind offset field notes to an integer traverse area."""

        branch_name, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=OFFSET_AREA_BRANCH,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.query",
        )

        area_case, case_probabilities = _offset_area_scene_inputs(instance_seed=int(instance_seed), params=task_params)
        formula_family = "survey_offset_trapezoid_area"

        def render_area_scene(context, attempt_seed: int):
            return render_offset_area_scene(context, area_case, instance_seed=int(attempt_seed))

        area_fields = {
            "chainages": [int(value) for value in area_case.chainages],
            "offsets": [int(value) for value in area_case.offsets],
        }

        rendered_attempt = render_survey_attempts(
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            render_defaults=_RENDER_DEFAULTS,
            render_scene=render_area_scene,
            build_annotation=area_scene_annotation,
        )
        answer = int(area_case.answer)
        return build_survey_task_output(
            task_identity=TASK_ID,
            task_prompt_key=TASK_PROMPT_KEY,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            prompt_branch_key=str(branch_name),
            formula_family=str(formula_family),
            rendered_attempt=rendered_attempt,
            answer_value=int(answer),
            query_probabilities=branch_probabilities,
            query_params_extra={
                "case_probabilities": dict(case_probabilities),
                **dict(rendered_attempt.rendered.witness),
            },
            execution_extra={
                **area_fields,
                **dict(rendered_attempt.rendered.witness),
            },
            witness_extra=dict(rendered_attempt.rendered.witness),
        )


__all__ = ["GeometrySurveyTraverseTraverseAreaValueTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
