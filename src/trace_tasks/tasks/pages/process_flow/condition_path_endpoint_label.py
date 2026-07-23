"""Process-flow decision-path endpoint label task."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.annotations import path_bbox_map_annotation
from .shared.defaults import DOMAIN
from .shared.sampling import condition_path_candidate


TASK_ID = "task_pages__process_flow__condition_path_endpoint_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "condition_path_endpoint_label"


def _bind_condition_path_endpoint(
    instance_seed,
    selected_branch,
    branch_probabilities,
    case,
    rendered,
):
    target_rng = spawn_rng(int(instance_seed), f"{TASK_ID}.path")
    target_payload = condition_path_candidate(
        rng=target_rng,
        nodes=case.nodes,
        edges=case.edges,
    )
    annotation_value, projected, annotation_ids, key_to_ref = path_bbox_map_annotation(
        rendered.render_map,
        list(target_payload["annotation_roles"]),
    )
    prompt_binding = _lifecycle.ProcessFlowPromptBinding(
        prompt_branch_key=PROMPT_QUERY_KEY,
        dynamic_slots={
            "start_label": str(target_payload["start_label"]),
            "condition_sequence_text": str(target_payload["condition_sequence_text"]),
        },
    )
    answer_binding = _lifecycle.string_binding(
        annotation_kind="bbox_map",
        annotation_value=annotation_value,
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_value=str(target_payload["answer"]),
        target_payload={**dict(target_payload), "prompt_query_key": PROMPT_QUERY_KEY},
        projected_annotation=projected,
        annotation_ids=tuple(annotation_ids),
        annotation_key_to_bbox_id=key_to_ref,
        question_format="process_flow_condition_path_endpoint_label",
    )
    return prompt_binding, answer_binding


@register_task
class PagesProcessFlowConditionPathEndpointLabelTask:
    """Follow two visible decision labels and return the reached step label."""

    task_id = TASK_ID
    reasoning_operations = ('topology',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        del max_attempts
        selected_branch, branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=SINGLE_QUERY_ID,
            public_task=TASK_ID,
        )
        namespace = f"{TASK_ID}.{PROMPT_QUERY_KEY}"
        return _lifecycle.build_process_flow_response(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            namespace=namespace,
            binding_factory=_bind_condition_path_endpoint,
        )


__all__ = [
    "PROMPT_QUERY_KEY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesProcessFlowConditionPathEndpointLabelTask",
]
