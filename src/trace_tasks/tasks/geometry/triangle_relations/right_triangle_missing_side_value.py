"""Infer a requested side length in a right triangle from an angle and side measure."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_triangle_relations_public_entry
from .shared.annotations import triangle_relations_annotation_mode
from .shared.construction import (
    case_trace_values,
    ground_from_angle_height_cases,
    ground_from_angle_hypotenuse_cases,
    height_from_angle_ground_cases,
    height_from_angle_hypotenuse_cases,
    hypotenuse_from_angle_height_cases,
)
from .shared.rendering import render_triangle_relations_scene
from .shared.sampling import choose_case_by_answer
from .shared.state import DOMAIN, SCENE_ID, TriangleRelationsCase, TriangleRelationsProblem

TASK_ID = "task_geometry__triangle_relations__right_triangle_missing_side_value"
TASK_PROMPT_KEY = "right_triangle_missing_side_value_query"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
_CASE_POOL = (
    height_from_angle_ground_cases()
    + ground_from_angle_height_cases()
    + hypotenuse_from_angle_height_cases()
    + height_from_angle_hypotenuse_cases()
    + ground_from_angle_hypotenuse_cases()
)


def _segment_name(case: TriangleRelationsCase):
    return "target side" if case.target_segment is None else "".join(case.target_segment)


def _prepare_right_triangle_missing_side(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
) -> tuple[TriangleRelationsProblem, int | float, dict[str, Any]]:
    """Bind one right-triangle side-length case."""

    if str(selected_branch) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported query branch for {TASK_ID}: {selected_branch}")
    case, answer_probs = choose_case_by_answer(
        cases=_CASE_POOL,
        answer_fn=lambda item: item.answer,
        params=params,
        instance_seed=int(instance_seed),
        namespace=TASK_ID,
    )
    trace_values = {
        "target_support_probabilities": dict(answer_probs),
        "target_segment_name": _segment_name(case),
        **case_trace_values(case),
    }
    problem = TriangleRelationsProblem(
        case=case,
        answer_support_probabilities=dict(answer_probs),
        prompt_target=_segment_name(case),
        annotation_mode=triangle_relations_annotation_mode(case),
    )
    return problem, case.answer, trace_values


@register_task
class GeometryRightTriangleMissingSideValueTask:
    """Infer a requested side length in a right triangle from an angle and side measure."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SINGLE_QUERY_ID
    task_prompt_key = TASK_PROMPT_KEY
    render_scene = staticmethod(render_triangle_relations_scene)
    prepare_objective = staticmethod(_prepare_right_triangle_missing_side)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_triangle_relations_public_entry(
            self,
            int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


__all__ = ["GeometryRightTriangleMissingSideValueTask"]
