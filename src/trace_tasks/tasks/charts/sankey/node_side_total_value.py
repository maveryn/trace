"""Return a one-sided total for a source or target Sankey node."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.registry import register_task

from ._lifecycle import build_sankey_plan, run_sankey_task
from .shared.sampling import (
    answer_value_bounds,
    path_dict,
    paths_by_source,
    paths_by_target,
    sample_frame,
    sorted_by_middle_target,
    sorted_by_source_middle,
)
from .shared.state import DOMAIN, SankeyDataset, SankeyQuestion


TASK_ID = "task_charts__sankey__node_side_total_value"
SOURCE_OUTGOING_QUERY_ID = "source_outgoing_total_flow"
TARGET_INCOMING_QUERY_ID = "target_incoming_total_flow"
SUPPORTED_QUERY_IDS = (SOURCE_OUTGOING_QUERY_ID, TARGET_INCOMING_QUERY_ID)
DEFAULT_QUERY_ID = SOURCE_OUTGOING_QUERY_ID

TASK_PARAM_DEFAULTS: dict[str, Any] = {}


def _source_outgoing_question(frame, *, params: dict[str, Any], probabilities: dict[str, float]) -> SankeyQuestion:
    """Choose one source node whose outgoing printed flow labels form the answer total."""

    answer_min, answer_max = answer_value_bounds(
        params,
        min_key="node_side_total_answer_min",
        max_key="node_side_total_answer_max",
        fallback_min=10,
        fallback_max=90,
        context="Sankey node-side total answer",
    )
    connected_min, connected_max = answer_value_bounds(
        params,
        min_key="node_side_total_connected_min",
        max_key="node_side_total_connected_max",
        fallback_min=2,
        fallback_max=3,
        context="Sankey node-side connected path count",
    )
    eligible: list[tuple[str, str, int, list[Any]]] = []
    for source_id, group in sorted(paths_by_source(frame.paths).items()):
        ordered_group = sorted_by_middle_target(group)
        if not (int(connected_min) <= len(ordered_group) <= int(connected_max)):
            continue
        answer = sum(int(path.first_value) for path in ordered_group)
        if int(answer_min) <= int(answer) <= int(answer_max):
            eligible.append((str(source_id), str(ordered_group[0].source_label), int(answer), ordered_group))
    if not eligible:
        raise ValueError("no eligible Sankey source outgoing total")
    selected_id, source_label, answer, selected_paths = eligible[0]
    segment_refs = tuple(f"{path.path_id}:source_middle" for path in selected_paths)
    return SankeyQuestion(
        branch_id=SOURCE_OUTGOING_QUERY_ID,
        branch_probabilities=dict(probabilities),
        answer=int(answer),
        answer_type="integer",
        annotation_type="point_set",
        annotation_segment_ids=tuple(segment_refs),
        params={
            "program_code": "sum(value(source_to_middle) for link in links if link.source == source_label)",
            "source_id": str(selected_id),
            "source_label": str(source_label),
            "middle_label": "",
            "target_label": "",
            "node_side": "source_outgoing",
            "connected_count": int(len(selected_paths)),
            "node_side_total": int(answer),
            "route_count": int(len(selected_paths)),
            "query_path_ids": [str(path.path_id) for path in selected_paths],
            "expression": " + ".join(str(int(path.first_value)) for path in selected_paths),
            "path_details": [path_dict(path) for path in selected_paths],
        },
    )


def _target_incoming_question(frame, *, params: dict[str, Any], probabilities: dict[str, float]) -> SankeyQuestion:
    """Choose one target node whose incoming printed flow labels form the answer total."""

    answer_min, answer_max = answer_value_bounds(
        params,
        min_key="node_side_total_answer_min",
        max_key="node_side_total_answer_max",
        fallback_min=10,
        fallback_max=90,
        context="Sankey node-side total answer",
    )
    connected_min, connected_max = answer_value_bounds(
        params,
        min_key="node_side_total_connected_min",
        max_key="node_side_total_connected_max",
        fallback_min=2,
        fallback_max=3,
        context="Sankey node-side connected path count",
    )
    eligible: list[tuple[str, str, int, list[Any]]] = []
    for target_id, group in sorted(paths_by_target(frame.paths).items()):
        ordered_group = sorted_by_source_middle(group)
        if not (int(connected_min) <= len(ordered_group) <= int(connected_max)):
            continue
        answer = sum(int(path.second_value) for path in ordered_group)
        if int(answer_min) <= int(answer) <= int(answer_max):
            eligible.append((str(target_id), str(ordered_group[0].target_label), int(answer), ordered_group))
    if not eligible:
        raise ValueError("no eligible Sankey target incoming total")
    selected_id, target_label, answer, selected_paths = eligible[0]
    segment_refs = tuple(f"{path.path_id}:middle_target" for path in selected_paths)
    return SankeyQuestion(
        branch_id=TARGET_INCOMING_QUERY_ID,
        branch_probabilities=dict(probabilities),
        answer=int(answer),
        answer_type="integer",
        annotation_type="point_set",
        annotation_segment_ids=tuple(segment_refs),
        params={
            "program_code": "sum(value(middle_to_target) for link in links if link.target == target_label)",
            "source_label": "",
            "middle_label": "",
            "target_id": str(selected_id),
            "target_label": str(target_label),
            "node_side": "target_incoming",
            "connected_count": int(len(selected_paths)),
            "node_side_total": int(answer),
            "route_count": int(len(selected_paths)),
            "query_path_ids": [str(path.path_id) for path in selected_paths],
            "expression": " + ".join(str(int(path.second_value)) for path in selected_paths),
            "path_details": [path_dict(path) for path in selected_paths],
        },
    )


def _build_plan(params: dict[str, Any], instance_seed: int, selected: str, probabilities: dict[str, float]):
    """Bind a source-outgoing or target-incoming node-side total."""

    frame = sample_frame(params, instance_seed=int(instance_seed))
    if str(selected) == SOURCE_OUTGOING_QUERY_ID:
        question = _source_outgoing_question(frame, params=params, probabilities=dict(probabilities))
    elif str(selected) == TARGET_INCOMING_QUERY_ID:
        question = _target_incoming_question(frame, params=params, probabilities=dict(probabilities))
    else:
        raise ValueError(f"unsupported Sankey node-side total branch: {selected}")
    return build_sankey_plan(
        dataset=SankeyDataset(frame=frame, question=question),
        prompt_key=str(selected),
        question_format="sankey_node_side_total_value",
        witness_type="sankey_node_side_total_value_witness",
    )


@register_task
class ChartsFlowSankeyNodeSideTotalValuePublicTask:
    """Return a one-sided total for a source or target Sankey node."""

    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'topology')
    domain = DOMAIN
    objective_contract = "node_side_total_value"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_sankey_task(
            self,
            int(instance_seed),
            {**TASK_PARAM_DEFAULTS, **dict(params)},
            int(max_attempts),
        )


__all__ = [
    "ChartsFlowSankeyNodeSideTotalValuePublicTask",
    "SOURCE_OUTGOING_QUERY_ID",
    "TARGET_INCOMING_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
