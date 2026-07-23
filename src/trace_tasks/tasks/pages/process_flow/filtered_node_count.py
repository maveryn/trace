"""Process-flow node count task filtered by visible node attributes."""

from __future__ import annotations

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.annotations import node_bbox_annotation
from .shared.defaults import DOMAIN
from .shared.sampling import choose_answer_bucket_candidate, node_filter_candidates


TASK_ID = "task_pages__process_flow__filtered_node_count"
SHAPE_NODE_COUNT_QUERY_ID = "shape_node_count"
STATUS_NODE_COUNT_QUERY_ID = "status_node_count"
ROLE_NODE_COUNT_QUERY_ID = "role_node_count"
SUPPORTED_QUERY_IDS = (
    SHAPE_NODE_COUNT_QUERY_ID,
    STATUS_NODE_COUNT_QUERY_ID,
    ROLE_NODE_COUNT_QUERY_ID,
)
FILTER_KIND_BY_BRANCH = {
    SHAPE_NODE_COUNT_QUERY_ID: "shape",
    STATUS_NODE_COUNT_QUERY_ID: "status",
    ROLE_NODE_COUNT_QUERY_ID: "role",
}


def _dynamic_slots(selected_branch: str, target_payload: dict) -> dict[str, str]:
    if str(selected_branch) == SHAPE_NODE_COUNT_QUERY_ID:
        return {
            "shape_filter_description": str(target_payload["shape_filter_description"]),
        }
    if str(selected_branch) == STATUS_NODE_COUNT_QUERY_ID:
        return {
            "status_filter_description": str(target_payload["status_filter_description"]),
        }
    return {
        "role_filter_description": str(target_payload["role_filter_description"]),
    }


def _bind_filtered_count(instance_seed, selected_branch, branch_probabilities, case, rendered):
    target_candidates = node_filter_candidates(
        nodes=case.nodes,
        filter_kind=str(FILTER_KIND_BY_BRANCH[str(selected_branch)]),
    )
    target_rng = spawn_rng(
        int(instance_seed),
        f"{TASK_ID}.{selected_branch}.target",
    )
    target_payload = choose_answer_bucket_candidate(
        target_candidates,
        rng=target_rng,
        min_answer=2,
    )
    annotation_value, projected, annotation_ids = node_bbox_annotation(
        rendered.render_map,
        [str(value) for value in target_payload["annotation_node_ids"]],
    )
    prompt_binding = _lifecycle.ProcessFlowPromptBinding(
        prompt_branch_key=str(selected_branch),
        dynamic_slots=_dynamic_slots(str(selected_branch), target_payload),
    )
    answer_binding = _lifecycle.integer_binding(
        annotation_kind="bbox_set",
        annotation_value=annotation_value,
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_value=int(target_payload["answer"]),
        target_payload={**dict(target_payload), "prompt_query_key": str(selected_branch)},
        projected_annotation=projected,
        annotation_ids=tuple(annotation_ids),
        question_format="process_flow_filtered_node_count",
    )
    return prompt_binding, answer_binding


@register_task
class PagesProcessFlowFilteredNodeCountTask:
    """Count process-flow nodes matching one visible shape, badge, or role filter."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        del max_attempts
        selected_branch, branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=SHAPE_NODE_COUNT_QUERY_ID,
            public_task=TASK_ID,
        )
        namespace = f"{TASK_ID}.{selected_branch}"
        return _lifecycle.build_process_flow_response(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            namespace=namespace,
            binding_factory=_bind_filtered_count,
        )


__all__ = [
    "FILTER_KIND_BY_BRANCH",
    "ROLE_NODE_COUNT_QUERY_ID",
    "SHAPE_NODE_COUNT_QUERY_ID",
    "STATUS_NODE_COUNT_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesProcessFlowFilteredNodeCountTask",
]
