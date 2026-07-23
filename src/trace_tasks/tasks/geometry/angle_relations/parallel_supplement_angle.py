"""Solve a supplement angle created by a transversal crossing parallel lines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from ._lifecycle import build_integer_angle_relation_trace, render_angle_relation_runtime, select_angle_relation_case
from .shared.construction import make_parallel_supplement_case
from .shared.state import DOMAIN, SCENE_ID, AngleRelationCase


TASK_ID = "task_geometry__angle_relations__parallel_supplement_angle"
PUBLIC_QUERY_ID = "single"
PARALLEL_SUPPLEMENT_PROMPT_QUERY_KEY = "parallel_supplement_angle"
SUPPORTED_QUERY_IDS = (PUBLIC_QUERY_ID,)
PARALLEL_SUPPLEMENT_ANSWER_SUPPORT = tuple(range(38, 83)) + tuple(range(98, 143))
PARALLEL_SUPPLEMENT_CASES = tuple(
    make_parallel_supplement_case(180 - answer_value, parallel_line_count=line_count)
    for answer_value in PARALLEL_SUPPLEMENT_ANSWER_SUPPORT
    for line_count in (2, 3)
)
_RENDER_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)[1]


@dataclass(frozen=True)
class _ParallelSupplementSelection:
    """Task-owned selected construction and public query metadata."""

    branch_name: str
    branch_probabilities: Mapping[str, float]
    params: Mapping[str, Any]
    case: AngleRelationCase
    case_index: int


def _select_parallel_supplement(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
) -> _ParallelSupplementSelection:
    """Select the single semantic branch and one answer-first supplement case."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id=PUBLIC_QUERY_ID,
        task_id=TASK_ID,
    )
    case, case_index = select_angle_relation_case(
        cases=PARALLEL_SUPPLEMENT_CASES,
        params=task_params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.{selected_query}.case",
    )
    return _ParallelSupplementSelection(
        branch_name=str(selected_query),
        branch_probabilities=dict(query_probabilities),
        params=dict(task_params),
        case=case,
        case_index=int(case_index),
    )


@register_task
class GeometryAngleRelationsParallelSupplementAngleTask:
    """Return the unknown angle supplementary to the shown transversal angle."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Bind the supplement answer and annotation from one rendered trace."""

        selection = _select_parallel_supplement(
            instance_seed=int(instance_seed),
            params=params,
        )
        runtime = render_angle_relation_runtime(
            case=selection.case,
            case_index=int(selection.case_index),
            prompt_query_key=PARALLEL_SUPPLEMENT_PROMPT_QUERY_KEY,
            instance_seed=int(instance_seed),
            params=selection.params,
            render_defaults=_RENDER_DEFAULTS,
            max_attempts=int(max_attempts),
        )
        answer_value = int(runtime.rendered_context.rendered_scene.answer)
        trace_payload = build_integer_angle_relation_trace(
            runtime=runtime,
            branch_name=str(selection.branch_name),
            branch_probabilities=selection.branch_probabilities,
            answer_value=int(answer_value),
            query_params={
                "target_answer_support_probabilities": {
                    str(value): 1.0 / float(len(PARALLEL_SUPPLEMENT_ANSWER_SUPPORT))
                    for value in PARALLEL_SUPPLEMENT_ANSWER_SUPPORT
                },
            },
            execution_fields_extra={
                "internal_query_id": PARALLEL_SUPPLEMENT_PROMPT_QUERY_KEY,
            },
            witness_fields_extra={
                "internal_query_id": PARALLEL_SUPPLEMENT_PROMPT_QUERY_KEY,
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
            query_id=str(selection.branch_name),
            prompt_variants=dict(runtime.prompt_artifacts.prompt_variants),
        )


__all__ = ["GeometryAngleRelationsParallelSupplementAngleTask"]
