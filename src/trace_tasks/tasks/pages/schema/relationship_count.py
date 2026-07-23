"""Schema task for counting relationship lines."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from . import _lifecycle
from .shared.annotations import round_segment


TASK_ID = "task_pages__schema__relationship_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
TASK_NAMESPACE = "pages.schema.relationship_count"
PROMPT_QUERY_KEY = "total_relationship_count"


def _build_relationship_line_case(instance_seed: int, params: Mapping[str, Any]):
    target_total = 5 + (abs(int(instance_seed)) % 5)
    return _lifecycle.build_schema_case(
        instance_seed=int(instance_seed),
        params=params,
        namespace=TASK_NAMESPACE,
        table_count_min=8,
        table_count_max=8,
        relationship_count_min=5,
        relationship_count_max=9,
        target_relationship_total=int(target_total),
    )


def _bind_relationship_line_count(
    instance_seed: int,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    case: _lifecycle.SchemaCase,
    rendered: _lifecycle.RenderedSchema,
):
    """Bind all rendered relationship line segments as count witnesses."""

    del instance_seed, selected_branch, branch_probabilities
    relationship_ids = [str(relationship["relationship_id"]) for relationship in case.relationships]
    annotation_ids = [
        f"relationship_segment:{relationship_id}"
        for relationship_id in relationship_ids
    ]
    segments = [
        round_segment(rendered.render_map["relationship_point_pairs_px"][relationship_id])
        for relationship_id in relationship_ids
    ]
    return (
        _lifecycle.SchemaPromptBinding(
            prompt_query_key=PROMPT_QUERY_KEY,
            dynamic_slots={},
        ),
        _lifecycle.segment_set_answer_binding(
            answer_value=int(len(relationship_ids)),
            segments=segments,
            annotation_ids=annotation_ids,
            extra_params={
                "answer": int(len(relationship_ids)),
                "annotation_relationship_ids": list(relationship_ids),
            },
        ),
    )


@register_task
class PagesSchemaRelationshipCountTask:
    """Count visible relationship lines in the schema diagram."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'topology')
    domain = _lifecycle.DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Mapping[str, Any], max_attempts: int):
        del max_attempts
        selected_branch, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
        )
        case = _build_relationship_line_case(int(instance_seed), task_params)
        return _lifecycle.render_bound_schema(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            case=case,
            binding_factory=_bind_relationship_line_count,
            question_format="schema_relationship_count",
            source_query_name=PROMPT_QUERY_KEY,
        )


__all__ = [
    "PROMPT_QUERY_KEY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "TASK_NAMESPACE",
    "PagesSchemaRelationshipCountTask",
]
