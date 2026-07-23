"""Infer the angle between two transversals crossing parallel lines."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from ._lifecycle import build_integer_angle_relation_trace, render_angle_relation_runtime, select_angle_relation_case
from .shared.construction import make_parallel_transversal_triangle_case
from .shared.state import DOMAIN, SCENE_ID


TASK_ID = "task_geometry__angle_relations__parallel_transversal_triangle_angle_value"
PUBLIC_QUERY_ID = "single"
PROMPT_QUERY_KEY = "parallel_transversal_triangle_angle_value"
TASK_PROMPT_KEY = "parallel_transversal_triangle_angle_value"
SUPPORTED_QUERY_IDS = (PUBLIC_QUERY_ID,)

_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)

PARALLEL_TRANSVERSAL_TRIANGLE_CASES = tuple(
    make_parallel_transversal_triangle_case(left_angle, right_angle)
    for left_angle in range(35, 76)
    for right_angle in range(35, 76)
    if 45 <= (180 - left_angle - right_angle) <= 110
)
ANSWER_SUPPORT = tuple(sorted(set(case.answer for case in PARALLEL_TRANSVERSAL_TRIANGLE_CASES)))


def _answer_probabilities() -> dict[str, float]:
    """Return uniform answer-support probabilities for trace metadata."""

    if not ANSWER_SUPPORT:
        return {}
    weight = 1.0 / float(len(ANSWER_SUPPORT))
    return {str(int(value)): weight for value in ANSWER_SUPPORT}


@register_task
class GeometryAngleRelationsParallelTransversalTriangleAngleValueTask:
    """Return the target angle between transversals using parallel-line angle sums."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Sample one two-transversal case and bind answer plus annotation."""

        selected_query, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=PUBLIC_QUERY_ID,
            task_id=TASK_ID,
        )
        case, case_index = select_angle_relation_case(
            cases=PARALLEL_TRANSVERSAL_TRIANGLE_CASES,
            params=task_params,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.case",
        )
        runtime = render_angle_relation_runtime(
            case=case,
            case_index=int(case_index),
            prompt_query_key=PROMPT_QUERY_KEY,
            prompt_task_key=TASK_PROMPT_KEY,
            instance_seed=int(instance_seed),
            params=task_params,
            render_defaults=_RENDER_DEFAULTS,
            max_attempts=int(max_attempts),
        )
        answer_value = int(runtime.rendered_context.rendered_scene.answer)
        trace_payload = build_integer_angle_relation_trace(
            runtime=runtime,
            branch_name=str(selected_query),
            branch_probabilities=query_probabilities,
            answer_value=int(answer_value),
            query_params={
                "answer_support_probabilities": _answer_probabilities(),
            },
            scene_relation_fields={
                "relation_id": "parallel_transversal_opposite_triangle_supplement_angle_sum",
                "displayed_lower_triangle_base_angles": list(
                    runtime.rendered_context.rendered_scene.witness["displayed_lower_triangle_base_angles"]
                ),
            },
            execution_fields_extra={
                "internal_query_id": PROMPT_QUERY_KEY,
                "displayed_lower_triangle_base_angles": list(
                    runtime.rendered_context.rendered_scene.witness["displayed_lower_triangle_base_angles"]
                ),
                "answer_angle_PQR": int(answer_value),
            },
            witness_fields_extra={
                "internal_query_id": PROMPT_QUERY_KEY,
                "displayed_lower_triangle_base_angles": list(
                    runtime.rendered_context.rendered_scene.witness["displayed_lower_triangle_base_angles"]
                ),
                "answer_angle_PQR": int(answer_value),
            },
        )
        return TaskOutput(
            prompt=str(runtime.prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(answer_value)),
            annotation_gt=TypedValue(
                type=str(runtime.annotation_artifacts.annotation_type),
                value=runtime.annotation_artifacts.value,
            ),
            image=runtime.rendered_context.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_query),
            prompt_variants=dict(runtime.prompt_artifacts.prompt_variants),
        )


__all__ = [
    "GeometryAngleRelationsParallelTransversalTriangleAngleValueTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
