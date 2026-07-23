"""Compute incircle radius from tangent segment labels."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_incircle_lifecycle
from .shared.measurements import incircle_spec_from_case, radius_answer_for_case
from .shared.sampling import all_tangent_triangle_cases, explicit_tangent_case, format_number_answer_key, group_cases_by_answer, select_answer_balanced_case, support_for_answer
from .shared.state import IncircleDiagramSpec

TASK_ID = "task_geometry__incircle_tangents__incircle_radius_from_area_value"
SUPPORTED_QUERY_IDS: tuple[str, ...] = ("single",)
INTERNAL_QUERY_ID = "inradius_from_tangent_segments"
ANNOTATION_ROLES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "O")

_CASES_BY_ANSWER = group_cases_by_answer(
    cases=all_tangent_triangle_cases(),
    answer_fn=radius_answer_for_case,
)


def _build_radius_spec(*, instance_seed: int, params: Mapping[str, Any]) -> tuple[IncircleDiagramSpec, int, dict[str, float]]:
    """Select a one-decimal radius answer first, then bind a triangle construction."""

    explicit_case = explicit_tangent_case(params)
    if explicit_case is None:
        case, case_index, answer_probabilities = select_answer_balanced_case(
            answer_cases=_CASES_BY_ANSWER,
            instance_seed=int(instance_seed),
            params=params,
            namespace=f"{TASK_ID}.{INTERNAL_QUERY_ID}",
            key_fn=format_number_answer_key,
        )
    else:
        case = explicit_case
        case_index = -1
        answer_probabilities = support_for_answer(
            answer_cases=_CASES_BY_ANSWER,
            answer=radius_answer_for_case(case),
            key_fn=format_number_answer_key,
        )
    answer = radius_answer_for_case(case)
    return (
        incircle_spec_from_case(
            case=case,
            answer=float(answer),
            answer_type="number",
            answer_rounding="one_decimal",
            unknown_measure="radius_length",
            formula_family=INTERNAL_QUERY_ID,
            unknown_label="r=?",
            show_area_label=False,
            show_radius_segment=True,
            annotation_roles=ANNOTATION_ROLES,
        ),
        int(case_index),
        dict(answer_probabilities),
    )


@register_task
class GeometryIncircleRadiusFromAreaValueTask:
    """Task-owned inradius objective for the incircle-tangents scene."""

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
            answer_type="number",
            reasoning_steps=2,
            build_spec=_build_radius_spec,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )
