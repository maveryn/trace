"""Compute triangle perimeter from incircle tangent segment labels."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_incircle_lifecycle
from .shared.defaults import SCENE_ID
from .shared.measurements import incircle_spec_from_case, perimeter_answer_for_case
from .shared.sampling import all_tangent_triangle_cases, explicit_tangent_case, group_cases_by_answer, select_answer_balanced_case, support_for_answer
from .shared.state import IncircleDiagramSpec

TASK_ID = "task_geometry__incircle_tangents__incircle_tangent_perimeter_value"
SUPPORTED_QUERY_IDS: tuple[str, ...] = ("single",)
INTERNAL_QUERY_ID = "triangle_perimeter_from_tangent_segments"
ANNOTATION_ROLES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")

_CASES_BY_ANSWER = group_cases_by_answer(
    cases=all_tangent_triangle_cases(),
    answer_fn=perimeter_answer_for_case,
)


def _build_perimeter_spec(*, instance_seed: int, params: Mapping[str, Any]) -> tuple[IncircleDiagramSpec, int, dict[str, float]]:
    """Select a perimeter answer first, then bind one triangle construction."""

    explicit_case = explicit_tangent_case(params)
    if explicit_case is None:
        case, case_index, answer_probabilities = select_answer_balanced_case(
            answer_cases=_CASES_BY_ANSWER,
            instance_seed=int(instance_seed),
            params=params,
            namespace=f"{TASK_ID}.{INTERNAL_QUERY_ID}",
        )
    else:
        case = explicit_case
        case_index = -1
        answer_probabilities = support_for_answer(
            answer_cases=_CASES_BY_ANSWER,
            answer=perimeter_answer_for_case(case),
        )
    answer = perimeter_answer_for_case(case)
    return (
        incircle_spec_from_case(
            case=case,
            answer=int(answer),
            answer_type="integer",
            answer_rounding="exact_integer",
            unknown_measure="perimeter_measure",
            formula_family=INTERNAL_QUERY_ID,
            unknown_label="P=?",
            show_area_label=False,
            show_radius_segment=False,
            annotation_roles=ANNOTATION_ROLES,
        ),
        int(case_index),
        dict(answer_probabilities),
    )


@register_task
class GeometryIncircleTangentPerimeterValueTask:
    """Task-owned perimeter objective for the incircle-tangents scene."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_incircle_lifecycle(
            task_id=TASK_ID,
            internal_query_id=INTERNAL_QUERY_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            answer_type="integer",
            reasoning_steps=1,
            build_spec=_build_perimeter_spec,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )
