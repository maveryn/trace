"""Select the x-axis label for an extremum point in one scatter series."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.unanswerable import (
    UNANSWERABLE_ANSWER,
    absence_proof,
    should_use_unanswerable_branch,
)
from trace_tasks.tasks.charts.scatter_readout._lifecycle import (
    ScatterReadoutTaskPlan,
    run_scatter_readout_lifecycle,
    single_point_readout_binding,
    single_point_readout_plan,
)
from trace_tasks.tasks.charts.scatter_readout.shared.sampling import build_base_dataset, missing_series_label
from trace_tasks.tasks.charts.scatter_readout.shared.state import DOMAIN, QueryBinding
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__scatter_readout__series_x_extremum_label"
QUESTION_FORMAT = "scatter_series_readout_query"
PROGRAM_CODE = (
    "select_label(x_label(arg_extreme(point in series, y_value(point), direction))); "
    "output=string_label_or_unanswerable; annotation=point(target_mark)|empty_map; "
    "scene=scatter_readout; scope=series_x_extremum_label"
)
QUERY_IDS = ("series_highest_x_label", "series_lowest_x_label")
QUERY_ARGS = {
    "series_highest_x_label": {"extremum": "highest"},
    "series_lowest_x_label": {"extremum": "lowest"},
}
REASONING_LOAD = 0.56
DEFAULT_QUERY_ID = QUERY_IDS[0]
TASK_PARAM_DEFAULTS: dict[str, Any] = {"_enable_unanswerable": True}


def _build_extremum_plan(
    params: Mapping[str, Any],
    instance_seed: int,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
) -> ScatterReadoutTaskPlan:
    """Bind one series as a highest/lowest x-label selection objective."""

    if str(selected_query_id) not in QUERY_ARGS:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_query_id}")
    semantic_args = dict(QUERY_ARGS[str(selected_query_id)])
    dataset = build_base_dataset(params=params, instance_seed=int(instance_seed))
    target_series = uniform_choice(
        spawn_rng(int(instance_seed), f"{TASK_ID}.target"),
        tuple(dataset.series),
    )

    if should_use_unanswerable_branch(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.{selected_query_id}",
        enabled=bool(params.get("_enable_unanswerable", False)),
    ):
        missing_label = missing_series_label(
            visible_labels=[str(series.label) for series in dataset.series],
            instance_seed=int(instance_seed),
        )
        binding = QueryBinding(
            answer=UNANSWERABLE_ANSWER,
            answer_type="string",
            target_series_label=str(missing_label),
            target_point_id="",
            annotation_point_ids=(),
            annotation_x_label="",
            trace={
                "target_series_label": str(missing_label),
                "target_point_id": "",
                "target_x_label": "",
                "target_y_value": "",
                "annotation_point_ids": [],
                "annotation_x_label": "",
                "answer": UNANSWERABLE_ANSWER,
                "answer_type": "string",
                "answerability": "unanswerable",
                "absence_proof": absence_proof(
                    requested_item=str(missing_label),
                    visible_candidates=[str(series.label) for series in dataset.series],
                    checked_scope="scatter plot legend series labels",
                    absence_reason="requested series label is not visible in the legend",
                ),
                **dict(semantic_args),
            },
        )
    else:
        if semantic_args["extremum"] == "highest":
            target_point = max(target_series.points, key=lambda point: (int(point.y_value), str(point.x_label)))
        else:
            target_point = min(target_series.points, key=lambda point: (int(point.y_value), str(point.x_label)))
        binding = single_point_readout_binding(
            answer=str(target_point.x_label),
            answer_type="string",
            target_series_label=str(target_series.label),
            target_point_id=str(target_point.point_id),
            target_x_label=str(target_point.x_label),
            target_y_value=int(target_point.y_value),
            extra_trace={
                "answerability": "answerable",
                **dict(semantic_args),
            },
        )

    return single_point_readout_plan(
        dataset=dataset,
        binding=binding,
        params={**TASK_PARAM_DEFAULTS, **dict(params)},
        prompt_query_key=str(selected_query_id),
        question_format=QUESTION_FORMAT,
        program_code=PROGRAM_CODE,
        query_params={
            **dict(semantic_args),
            "query_id_probabilities": dict(query_probabilities),
        },
        reasoning_load=REASONING_LOAD,
        include_unanswerable_instruction=True,
    )


@register_task
class ChartsScatterSeriesExtremumXLabelTask:
    """Select the x-axis label for the highest or lowest point in one series."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    objective_contract = "series_x_extremum_label"
    supported_query_ids = QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_scatter_readout_lifecycle(
            task=self,
            instance_seed=int(instance_seed),
            params={**TASK_PARAM_DEFAULTS, **dict(params)},
            max_attempts=int(max_attempts),
            default_query_id=DEFAULT_QUERY_ID,
            build_plan=_build_extremum_plan,
        )


__all__ = ["ChartsScatterSeriesExtremumXLabelTask"]
