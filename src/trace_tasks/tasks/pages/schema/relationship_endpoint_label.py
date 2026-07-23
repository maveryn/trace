"""Schema task for identifying the table at a relationship endpoint."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from . import _lifecycle


TASK_ID = "task_pages__schema__relationship_endpoint_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
TASK_NAMESPACE = "pages.schema.relationship_endpoint_label"
PROMPT_QUERY_KEY = "target_table_for_relationship_label"


def _build_endpoint_case(instance_seed: int, params: Mapping[str, Any]):
    return _lifecycle.build_schema_case(
        instance_seed=int(instance_seed),
        params=params,
        namespace=TASK_NAMESPACE,
        table_count_min=8,
        table_count_max=8,
        relationship_count_min=6,
        relationship_count_max=9,
    )


def _bind_endpoint_label(
    instance_seed: int,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    case: _lifecycle.SchemaCase,
    rendered: _lifecycle.RenderedSchema,
):
    """Bind source table, relationship label, and target table boxes."""

    del selected_branch, branch_probabilities
    return _lifecycle.bind_labeled_relation_endpoint(
        instance_seed=int(instance_seed),
        case=case,
        rendered=rendered,
        prompt_query_key=PROMPT_QUERY_KEY,
    )


@register_task
class PagesSchemaRelationshipEndpointLabelTask:
    """Identify the target table reached by a labeled relationship."""

    task_id = TASK_ID
    reasoning_operations = ('topology',)
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
        case = _build_endpoint_case(int(instance_seed), task_params)
        return _lifecycle.render_bound_schema(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            case=case,
            binding_factory=_bind_endpoint_label,
            question_format="schema_relationship_endpoint_label",
            source_query_name=PROMPT_QUERY_KEY,
        )


__all__ = [
    "PROMPT_QUERY_KEY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "TASK_NAMESPACE",
    "PagesSchemaRelationshipEndpointLabelTask",
]
