"""Radial Sankey grouped transfer total task."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task

from ._lifecycle import build_radial_sankey_plan, run_radial_sankey_task
from .shared.sampling import (
    answer_value_bounds,
    join_quoted,
    link_dict,
    links_by_source,
    links_by_target,
    sample_frame,
    sample_group_size,
    sorted_by_source_label,
    sorted_by_target_label,
)
from .shared.state import DOMAIN, FlowLink, RadialSankeyDataset, RadialSankeyQuestion


TASK_ID = "task_charts__radial_sankey__transfer_total_value"
SOURCE_TO_TARGETS_QUERY_ID = "source_to_targets_total"
SOURCES_TO_TARGET_QUERY_ID = "sources_to_target_total"
SUPPORTED_QUERY_IDS = (SOURCE_TO_TARGETS_QUERY_ID, SOURCES_TO_TARGET_QUERY_ID)
DEFAULT_QUERY_ID = SOURCE_TO_TARGETS_QUERY_ID

TASK_PARAM_DEFAULTS: dict[str, Any] = {
    "radial_group_size_min": 2,
    "radial_group_size_max": 2,
    "radial_source_count_min": 4,
    "radial_source_count_max": 4,
    "radial_target_count_min": 4,
    "radial_target_count_max": 4,
    "radial_link_count_min": 5,
    "radial_link_count_max": 7,
}


def _source_to_targets_question(
    *,
    frame,
    params: dict[str, Any],
    instance_seed: int,
    branch_probabilities: dict[str, float],
) -> RadialSankeyQuestion:
    """Bind a source plus target subset whose printed value labels sum to the answer."""

    group_size = sample_group_size(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.source_group_size",
    )
    answer_min, answer_max = answer_value_bounds(params)
    eligible: list[list[FlowLink]] = []
    for group in links_by_source(frame.links).values():
        ordered = sorted_by_target_label(group)
        if len(ordered) < int(group_size):
            continue
        for start in range(0, len(ordered) - int(group_size) + 1):
            subset = ordered[start : start + int(group_size)]
            answer = sum(int(link.value) for link in subset)
            if int(answer_min) <= int(answer) <= int(answer_max):
                eligible.append(list(subset))
    if not eligible:
        raise ValueError("no eligible radial Sankey source-to-target grouped total")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.source_to_targets_selection")
    selected_links = eligible[int(rng.randrange(len(eligible)))]
    source_label = str(selected_links[0].source_label)
    target_labels = [str(link.target_label) for link in selected_links]
    answer = sum(int(link.value) for link in selected_links)
    link_refs = tuple(str(link.link_id) for link in selected_links)
    return RadialSankeyQuestion(
        branch_id=SOURCE_TO_TARGETS_QUERY_ID,
        branch_probabilities=dict(branch_probabilities),
        answer=int(answer),
        answer_type="integer",
        annotation_type="bbox_set",
        annotation_link_ids=tuple(link_refs),
        annotation_node_ids=tuple(),
        params={
            "program_code": "sum(value(flow) for flow in flows if flow.source == source_label and flow.target in target_label_set)",
            "source_label": str(source_label),
            "source_labels": [],
            "source_labels_joined": "",
            "target_label": "",
            "target_labels": list(target_labels),
            "target_labels_joined": join_quoted(target_labels),
            "query_link_ids": list(link_refs),
            "comparison_link_ids": [],
            "group_size": int(len(selected_links)),
            "expression": " + ".join(str(int(link.value)) for link in selected_links),
            "link_details": [link_dict(link) for link in selected_links],
        },
    )


def _sources_to_target_question(
    *,
    frame,
    params: dict[str, Any],
    instance_seed: int,
    branch_probabilities: dict[str, float],
) -> RadialSankeyQuestion:
    """Bind a target plus source subset whose printed value labels sum to the answer."""

    group_size = sample_group_size(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.target_group_size",
    )
    answer_min, answer_max = answer_value_bounds(params)
    eligible: list[list[FlowLink]] = []
    for group in links_by_target(frame.links).values():
        ordered = sorted_by_source_label(group)
        if len(ordered) < int(group_size):
            continue
        for start in range(0, len(ordered) - int(group_size) + 1):
            subset = ordered[start : start + int(group_size)]
            answer = sum(int(link.value) for link in subset)
            if int(answer_min) <= int(answer) <= int(answer_max):
                eligible.append(list(subset))
    if not eligible:
        raise ValueError("no eligible radial Sankey sources-to-target grouped total")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.sources_to_target_selection")
    selected_links = eligible[int(rng.randrange(len(eligible)))]
    target_label = str(selected_links[0].target_label)
    source_labels = [str(link.source_label) for link in selected_links]
    answer = sum(int(link.value) for link in selected_links)
    link_refs = tuple(str(link.link_id) for link in selected_links)
    return RadialSankeyQuestion(
        branch_id=SOURCES_TO_TARGET_QUERY_ID,
        branch_probabilities=dict(branch_probabilities),
        answer=int(answer),
        answer_type="integer",
        annotation_type="bbox_set",
        annotation_link_ids=tuple(link_refs),
        annotation_node_ids=tuple(),
        params={
            "program_code": "sum(value(flow) for flow in flows if flow.source in source_label_set and flow.target == target_label)",
            "source_label": "",
            "source_labels": list(source_labels),
            "source_labels_joined": join_quoted(source_labels),
            "target_label": str(target_label),
            "target_labels": [],
            "target_labels_joined": "",
            "query_link_ids": list(link_refs),
            "comparison_link_ids": [],
            "group_size": int(len(selected_links)),
            "expression": " + ".join(str(int(link.value)) for link in selected_links),
            "link_details": [link_dict(link) for link in selected_links],
        },
    )


def _build_plan(params: dict[str, Any], instance_seed: int, selected: str, probabilities: dict[str, float]):
    """Bind grouped source/target operands, answer total, and value-label witnesses."""

    frame = sample_frame(params, instance_seed=int(instance_seed))
    if str(selected) == SOURCE_TO_TARGETS_QUERY_ID:
        question = _source_to_targets_question(
            frame=frame,
            params=params,
            instance_seed=int(instance_seed),
            branch_probabilities=dict(probabilities),
        )
    elif str(selected) == SOURCES_TO_TARGET_QUERY_ID:
        question = _sources_to_target_question(
            frame=frame,
            params=params,
            instance_seed=int(instance_seed),
            branch_probabilities=dict(probabilities),
        )
    else:
        raise ValueError(f"unsupported radial Sankey transfer branch: {selected}")
    dataset = RadialSankeyDataset(frame=frame, question=question)
    return build_radial_sankey_plan(
        dataset=dataset,
        prompt_key=str(selected),
        question_format="radial_sankey_transfer_total_value",
        witness_type="radial_sankey_transfer_total_value_witness",
    )


@register_task
class ChartsRadialSankeyTransferTotalValueTask:
    """Return a grouped transfer total from a radial Sankey chart."""

    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'topology')
    domain = DOMAIN
    objective_contract = "transfer_total_value"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_radial_sankey_task(
            self,
            int(instance_seed),
            {**TASK_PARAM_DEFAULTS, **dict(params)},
            int(max_attempts),
        )


__all__ = [
    "ChartsRadialSankeyTransferTotalValueTask",
    "SUPPORTED_QUERY_IDS",
]
