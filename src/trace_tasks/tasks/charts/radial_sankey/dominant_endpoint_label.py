"""Radial Sankey dominant endpoint label task."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task

from ._lifecycle import build_radial_sankey_plan, run_radial_sankey_task
from .shared.sampling import link_dict, links_by_source, links_by_target, sample_frame
from .shared.state import DOMAIN, FlowLink, RadialSankeyDataset, RadialSankeyQuestion


TASK_ID = "task_charts__radial_sankey__dominant_endpoint_label"
LARGEST_TARGET_QUERY_ID = "largest_target_for_source"
LARGEST_SOURCE_QUERY_ID = "largest_source_for_target"
SUPPORTED_QUERY_IDS = (LARGEST_TARGET_QUERY_ID, LARGEST_SOURCE_QUERY_ID)
DEFAULT_QUERY_ID = LARGEST_TARGET_QUERY_ID


def _distinct_value_groups(groups: dict[str, list[FlowLink]], *, endpoint_role: str) -> list[list[FlowLink]]:
    eligible: list[list[FlowLink]] = []
    for group in groups.values():
        if str(endpoint_role) == "target":
            ordered = sorted(group, key=lambda item: (-int(item.value), str(item.target_label), str(item.link_id)))
        else:
            ordered = sorted(group, key=lambda item: (-int(item.value), str(item.source_label), str(item.link_id)))
        values = [int(item.value) for item in ordered]
        if len(ordered) >= 2 and len(set(values)) == len(values):
            eligible.append(list(ordered))
    return eligible


def _largest_target_question(
    *,
    frame,
    instance_seed: int,
    branch_probabilities: dict[str, float],
) -> RadialSankeyQuestion:
    eligible = _distinct_value_groups(links_by_source(frame.links), endpoint_role="target")
    if not eligible:
        raise ValueError("no eligible radial Sankey largest-target comparison")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.largest_target_selection")
    selected_group = eligible[int(rng.randrange(len(eligible)))]
    winner = selected_group[0]
    link_refs = tuple(str(link.link_id) for link in selected_group)
    return RadialSankeyQuestion(
        branch_id=LARGEST_TARGET_QUERY_ID,
        branch_probabilities=dict(branch_probabilities),
        answer=str(winner.target_label),
        answer_type="string",
        annotation_type="bbox",
        annotation_link_ids=tuple(),
        annotation_node_ids=(str(winner.target_id),),
        params={
            "program_code": "argmax_label(target(flow), value(flow), where source(flow) == source_label)",
            "source_label": str(winner.source_label),
            "source_labels": [],
            "source_labels_joined": "",
            "target_label": "",
            "target_labels": [],
            "target_labels_joined": "",
            "query_link_ids": list(link_refs),
            "comparison_link_ids": list(link_refs),
            "group_size": int(len(selected_group)),
            "expression": f"argmax target for {winner.source_label}",
            "link_details": [link_dict(link) for link in selected_group],
            "answer_node_id": str(winner.target_id),
        },
    )


def _largest_source_question(
    *,
    frame,
    instance_seed: int,
    branch_probabilities: dict[str, float],
) -> RadialSankeyQuestion:
    eligible = _distinct_value_groups(links_by_target(frame.links), endpoint_role="source")
    if not eligible:
        raise ValueError("no eligible radial Sankey largest-source comparison")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.largest_source_selection")
    selected_group = eligible[int(rng.randrange(len(eligible)))]
    winner = selected_group[0]
    link_refs = tuple(str(link.link_id) for link in selected_group)
    return RadialSankeyQuestion(
        branch_id=LARGEST_SOURCE_QUERY_ID,
        branch_probabilities=dict(branch_probabilities),
        answer=str(winner.source_label),
        answer_type="string",
        annotation_type="bbox",
        annotation_link_ids=tuple(),
        annotation_node_ids=(str(winner.source_id),),
        params={
            "program_code": "argmax_label(source(flow), value(flow), where target(flow) == target_label)",
            "source_label": "",
            "source_labels": [],
            "source_labels_joined": "",
            "target_label": str(winner.target_label),
            "target_labels": [],
            "target_labels_joined": "",
            "query_link_ids": list(link_refs),
            "comparison_link_ids": list(link_refs),
            "group_size": int(len(selected_group)),
            "expression": f"argmax source for {winner.target_label}",
            "link_details": [link_dict(link) for link in selected_group],
            "answer_node_id": str(winner.source_id),
        },
    )


def _build_plan(params: dict[str, Any], instance_seed: int, selected: str, probabilities: dict[str, float]):
    """Bind one endpoint comparison group and the selected endpoint-node witness."""

    frame = sample_frame(params, instance_seed=int(instance_seed))
    if str(selected) == LARGEST_TARGET_QUERY_ID:
        question = _largest_target_question(
            frame=frame,
            instance_seed=int(instance_seed),
            branch_probabilities=dict(probabilities),
        )
    elif str(selected) == LARGEST_SOURCE_QUERY_ID:
        question = _largest_source_question(
            frame=frame,
            instance_seed=int(instance_seed),
            branch_probabilities=dict(probabilities),
        )
    else:
        raise ValueError(f"unsupported radial Sankey dominant endpoint branch: {selected}")
    dataset = RadialSankeyDataset(frame=frame, question=question)
    return build_radial_sankey_plan(
        dataset=dataset,
        prompt_key=str(selected),
        question_format="radial_sankey_dominant_endpoint_label",
        witness_type="radial_sankey_dominant_endpoint_label_witness",
    )


@register_task
class ChartsRadialSankeyDominantEndpointLabelTask:
    """Return a dominant source or target endpoint from a radial Sankey chart."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'topology')
    domain = DOMAIN
    objective_contract = "dominant_endpoint_label"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_radial_sankey_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = [
    "ChartsRadialSankeyDominantEndpointLabelTask",
    "SUPPORTED_QUERY_IDS",
]
