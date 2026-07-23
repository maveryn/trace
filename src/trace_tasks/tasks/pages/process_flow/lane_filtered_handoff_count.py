"""Process-flow task for counting cross-lane handoffs involving one lane."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.annotations import edge_segment_annotation
from .shared.defaults import DOMAIN
from .shared.sampling import handoff_candidates


TASK_ID = "task_pages__process_flow__lane_filtered_handoff_count"
LANE_OUTGOING_HANDOFF_COUNT_QUERY_ID = "lane_outgoing_handoff_count"
LANE_INVOLVED_HANDOFF_COUNT_QUERY_ID = "lane_involved_handoff_count"
SUPPORTED_QUERY_IDS = (
    LANE_OUTGOING_HANDOFF_COUNT_QUERY_ID,
    LANE_INVOLVED_HANDOFF_COUNT_QUERY_ID,
)
HANDOFF_SCOPE_BY_BRANCH = {
    LANE_OUTGOING_HANDOFF_COUNT_QUERY_ID: "lane_outgoing",
    LANE_INVOLVED_HANDOFF_COUNT_QUERY_ID: "lane_involved",
}
OUTGOING_ANSWER_WEIGHTS = {1: 1.5, 2: 0.8, 3: 1.5, 4: 0.35}
INVOLVED_ANSWER_WEIGHTS = {3: 0.9, 4: 1.2, 5: 0.85, 6: 1.9}


def _choose_weighted_answer_bucket(
    candidates: Sequence[Mapping[str, Any]],
    *,
    rng: Any,
    answer_weights: Mapping[int, float],
) -> dict:
    by_answer: dict[int, list[dict]] = {}
    for candidate in candidates:
        by_answer.setdefault(int(candidate["answer"]), []).append(dict(candidate))
    if not by_answer:
        raise ValueError("process-flow lane selector received no candidates")
    weighted_answers = [
        (answer, max(0.01, float(answer_weights.get(int(answer), 1.0))))
        for answer in sorted(by_answer)
    ]
    total = sum(weight for _, weight in weighted_answers)
    threshold = float(rng.random()) * total
    cumulative = 0.0
    selected_answer = int(weighted_answers[-1][0])
    for answer, weight in weighted_answers:
        cumulative += float(weight)
        if threshold <= cumulative:
            selected_answer = int(answer)
            break
    return dict(rng.choice(by_answer[selected_answer]))


def _choose_lane_target(instance_seed: int, selected_branch: str, case) -> dict:
    target_rng = spawn_rng(int(instance_seed), f"{TASK_ID}.{selected_branch}.target")
    candidates = handoff_candidates(
        nodes=case.nodes,
        edges=case.edges,
        handoff_scope=str(HANDOFF_SCOPE_BY_BRANCH[str(selected_branch)]),
    )
    if str(selected_branch) == LANE_OUTGOING_HANDOFF_COUNT_QUERY_ID:
        return _choose_weighted_answer_bucket(
            candidates,
            rng=target_rng,
            answer_weights=OUTGOING_ANSWER_WEIGHTS,
        )
    eligible_candidates = [
        candidate for candidate in candidates if int(candidate.get("answer", 0)) >= 3
    ]
    return _choose_weighted_answer_bucket(
        eligible_candidates or candidates,
        rng=target_rng,
        answer_weights=INVOLVED_ANSWER_WEIGHTS,
    )


def _bind_lane_filtered_handoff(
    instance_seed,
    selected_branch,
    branch_probabilities,
    case,
    rendered,
):
    target_payload = _choose_lane_target(int(instance_seed), str(selected_branch), case)
    annotation_value, projected, annotation_ids = edge_segment_annotation(
        rendered.render_map,
        [str(value) for value in target_payload["annotation_edge_ids"]],
    )
    prompt_binding = _lifecycle.ProcessFlowPromptBinding(
        prompt_branch_key=str(selected_branch),
        dynamic_slots={"lane_name": str(target_payload["lane_name"])},
    )
    answer_binding = _lifecycle.integer_binding(
        annotation_kind="segment_set",
        annotation_value=annotation_value,
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_value=int(target_payload["answer"]),
        target_payload={**dict(target_payload), "prompt_query_key": str(selected_branch)},
        projected_annotation=projected,
        annotation_ids=tuple(annotation_ids),
        question_format="process_flow_lane_filtered_handoff_count",
    )
    return prompt_binding, answer_binding


@register_task
class PagesProcessFlowLaneFilteredHandoffCountTask:
    """Count cross-lane handoff arrows for one named lane."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'topology')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        del max_attempts
        selected_branch, branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=LANE_OUTGOING_HANDOFF_COUNT_QUERY_ID,
            public_task=TASK_ID,
        )
        handoff_rng = spawn_rng(int(instance_seed), f"{TASK_ID}.{selected_branch}.handoff_chain")
        task_params = {
            **dict(task_params),
            "flow_family": "handoff_chain",
            "node_count_min": 9,
            "node_count_max": 9,
            "lane_count_min": 5,
            "lane_count_max": 5,
            "target_cross_count": int(handoff_rng.choice((4, 5, 6, 7, 8))),
        }
        namespace = f"{TASK_ID}.{selected_branch}"
        return _lifecycle.build_process_flow_response(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            namespace=namespace,
            binding_factory=_bind_lane_filtered_handoff,
        )


__all__ = [
    "HANDOFF_SCOPE_BY_BRANCH",
    "LANE_INVOLVED_HANDOFF_COUNT_QUERY_ID",
    "LANE_OUTGOING_HANDOFF_COUNT_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesProcessFlowLaneFilteredHandoffCountTask",
]
