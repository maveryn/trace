"""Solve altitude or projection lengths from a right-triangle altitude construction."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_triangle_relations_public_entry
from .shared.annotations import triangle_relations_annotation_mode
from .shared.construction import (
    altitude_from_two_projections_cases,
    case_trace_values,
    projection_from_altitude_cases,
)
from .shared.rendering import render_triangle_relations_scene
from .shared.sampling import choose_case_by_answer
from .shared.state import DOMAIN, SCENE_ID, TriangleRelationsProblem

TASK_ID = "task_geometry__triangle_relations__altitude_to_hypotenuse_value"
TASK_PROMPT_KEY = "altitude_to_hypotenuse_value_query"
SUPPORTED_QUERY_IDS = (
    "altitude_from_split_hypotenuse",
    "missing_projection_from_altitude",
)
DEFAULT_QUERY_ID = "altitude_from_split_hypotenuse"


def _prepare_altitude_to_hypotenuse(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
) -> tuple[TriangleRelationsProblem, int | float, dict[str, Any]]:
    """Bind one altitude-to-hypotenuse theorem construction."""

    if str(selected_branch) == "altitude_from_split_hypotenuse":
        cases = altitude_from_two_projections_cases()
    elif str(selected_branch) == "missing_projection_from_altitude":
        cases = projection_from_altitude_cases()
    else:
        raise ValueError(f"unsupported query branch for {TASK_ID}: {selected_branch}")
    case, answer_probs = choose_case_by_answer(
        cases=cases,
        answer_fn=lambda item: item.answer,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.{selected_branch}",
    )
    trace_values = {
        "target_support_probabilities": dict(answer_probs),
        **case_trace_values(case),
    }
    problem = TriangleRelationsProblem(
        case=case,
        answer_support_probabilities=dict(answer_probs),
        prompt_target=str(case.trace_values.get("target_name", "target segment")),
        annotation_mode=triangle_relations_annotation_mode(case),
    )
    return problem, case.answer, trace_values


@register_task
class GeometryTriangleRelationsAltitudeToHypotenuseValueTask:
    """Solve altitude or projection lengths from the altitude-to-hypotenuse theorem."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    task_prompt_key = TASK_PROMPT_KEY
    render_scene = staticmethod(render_triangle_relations_scene)
    prepare_objective = staticmethod(_prepare_altitude_to_hypotenuse)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_triangle_relations_public_entry(
            self,
            int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


__all__ = [
    "GeometryTriangleRelationsAltitudeToHypotenuseValueTask",
    "SCENE_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
