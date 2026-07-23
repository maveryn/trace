"""Return the total bottleneck flow across routes from one source to one target."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import build_sankey_plan, run_sankey_task
from .shared.sampling import (
    answer_value_bounds,
    bottleneck_segment_ref,
    path_dict,
    sample_frame,
    sample_route_count,
    sorted_by_middle_label,
)
from .shared.state import DOMAIN, SankeyDataset, SankeyQuestion


TASK_ID = "task_charts__sankey__source_to_target_total_flow"
QUERY_ID = SINGLE_QUERY_ID
SUPPORTED_QUERY_IDS = (QUERY_ID,)
DEFAULT_QUERY_ID = QUERY_ID
PROMPT_KEY = "source_to_target_total_flow"

TASK_PARAM_DEFAULTS: dict[str, Any] = {}


def _build_plan(params: dict[str, Any], instance_seed: int, selected: str, probabilities: dict[str, float]):
    """Bind a source-target route family and sum each route's bottleneck value."""

    if str(selected) != QUERY_ID:
        raise ValueError(f"unsupported Sankey source-target total branch: {selected}")
    route_count = sample_route_count(params, instance_seed=int(instance_seed))
    frame = sample_frame(params, instance_seed=int(instance_seed), reserved_route_count=int(route_count))
    source_id = str(frame.route_focus["source_id"])
    target_id = str(frame.route_focus["target_id"])
    selected_paths = sorted_by_middle_label(
        [
            path
            for path in frame.paths
            if str(path.source_id) == source_id and str(path.target_id) == target_id
        ]
    )
    if len(selected_paths) < 2:
        raise ValueError("Sankey source-target total requires at least two routes")
    answer = sum(int(path.bottleneck_value) for path in selected_paths)
    answer_min, answer_max = answer_value_bounds(
        params,
        min_key="source_target_answer_min",
        max_key="source_target_answer_max",
        fallback_min=10,
        fallback_max=90,
        context="Sankey source-target total answer",
    )
    if int(answer) < int(answer_min) or int(answer) > int(answer_max):
        raise ValueError("Sankey source-target total answer outside configured support")
    segment_refs = tuple(bottleneck_segment_ref(path) for path in selected_paths)
    question = SankeyQuestion(
        branch_id=QUERY_ID,
        branch_probabilities=dict(probabilities),
        answer=int(answer),
        answer_type="integer",
        annotation_type="point_set",
        annotation_segment_ids=tuple(segment_refs),
        params={
            "program_code": "sum(min(value(source_to_middle), value(middle_to_target)) for route in routes(source_label, target_label))",
            "source_label": str(frame.route_focus["source_label"]),
            "middle_label": "",
            "target_label": str(frame.route_focus["target_label"]),
            "route_count": int(len(selected_paths)),
            "query_path_ids": [str(path.path_id) for path in selected_paths],
            "expression": " + ".join(str(int(path.bottleneck_value)) for path in selected_paths),
            "path_details": [path_dict(path) for path in selected_paths],
        },
    )
    return build_sankey_plan(
        dataset=SankeyDataset(frame=frame, question=question),
        prompt_key=PROMPT_KEY,
        question_format="sankey_path_value",
        witness_type="sankey_source_to_target_total_flow_witness",
    )


@register_task
class ChartsFlowSankeySourceToTargetTotalFlowPublicTask:
    """Return the total bottleneck flow across routes from one source to one target."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'aggregation', 'topology')
    domain = DOMAIN
    objective_contract = "source_to_target_total_flow"
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
    "ChartsFlowSankeySourceToTargetTotalFlowPublicTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
