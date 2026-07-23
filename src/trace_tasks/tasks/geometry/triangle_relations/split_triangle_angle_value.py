"""Find a target angle in a split-triangle diagram."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_triangle_relations_public_entry
from .shared.annotations import triangle_relations_annotation_mode
from .shared.construction import case_trace_values, split_triangle_angle_cases
from .shared.rendering import render_triangle_relations_scene
from .shared.sampling import choose_case_by_answer
from .shared.state import DOMAIN, SCENE_ID, TriangleRelationsProblem

TASK_ID = "task_geometry__triangle_relations__split_triangle_angle_value"
TASK_PROMPT_KEY = "split_triangle_angle_value_query"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
_CASE_POOL = split_triangle_angle_cases()


def _prepare_split_triangle_angle(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
) -> tuple[TriangleRelationsProblem, int | float, dict[str, Any]]:
    """Bind one split-triangle angle-chase construction."""

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
        **case_trace_values(case),
    }
    problem = TriangleRelationsProblem(
        case=case,
        answer_support_probabilities=dict(answer_probs),
        prompt_target="the marked angle",
        annotation_mode=triangle_relations_annotation_mode(case),
    )
    return problem, case.answer, trace_values


@register_task
class GeometryTriangleRelationsSplitTriangleAngleValueTask:
    """Find a target angle in a split-triangle diagram."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SINGLE_QUERY_ID
    task_prompt_key = TASK_PROMPT_KEY
    render_scene = staticmethod(render_triangle_relations_scene)
    prepare_objective = staticmethod(_prepare_split_triangle_angle)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_triangle_relations_public_entry(
            self,
            int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


__all__ = [
    "GeometryTriangleRelationsSplitTriangleAngleValueTask",
    "SCENE_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
