"""Select the x-axis label where one styled legend series is extremal."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.style_legend._lifecycle import (
    package_point_label_plan,
    run_style_legend_lifecycle,
)
from trace_tasks.tasks.charts.style_legend.shared.defaults import balanced_choice, gen_int
from trace_tasks.tasks.charts.style_legend.shared.prompts import (
    ANSWER_HINT_X_LABEL,
)
from trace_tasks.tasks.charts.style_legend.shared.sampling import (
    StyleLegendSampleContext,
    dataset_from_context,
    replace_series_values,
    sample_context,
)
from trace_tasks.tasks.charts.style_legend.shared.state import DOMAIN, SeriesSpec, point_id
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__style_legend__series_extremum_x_label"
SUPPORTED_QUERY_IDS = (
    "series_highest_x_label",
    "series_lowest_x_label",
)
DEFAULT_QUERY_ID = "series_highest_x_label"
TASK_PARAM_DEFAULTS: dict[str, Any] = {}
PROGRAM_CODE = "arg_extremum(x_position, value(series, x_position), direction={highest,lowest}); output=string_label; annotation=point; scene=style_legend; scope=series_extremum_x_label"
QUERY_DIRECTIONS = {
    "series_highest_x_label": "highest",
    "series_lowest_x_label": "lowest",
}


def _selected_direction(selected: str) -> str:
    try:
        return str(QUERY_DIRECTIONS[str(selected)])
    except KeyError as exc:
        raise ValueError(f"unsupported query_id: {selected}") from exc


def _sample_operands(context: StyleLegendSampleContext, params: Mapping[str, Any], *, seed: int) -> tuple[int, int]:
    answer_x_index = int(
        balanced_choice(
            tuple(range(int(context.x_count))),
            params,
            instance_seed=int(seed),
            namespace=f"{TASK_ID}.answer_x_index",
        )
    )
    target_series_index = int(
        balanced_choice(
            tuple(range(int(context.series_count))),
            params,
            instance_seed=int(seed),
            namespace=f"{TASK_ID}.target_series",
        )
    )
    return int(answer_x_index), int(target_series_index)


def _force_fixed_series_extremum(
    context: StyleLegendSampleContext,
    series: SeriesSpec,
    *,
    answer_x_index: int,
    direction: str,
    seed: int,
    params: Mapping[str, Any],
) -> SeriesSpec:
    rng = spawn_rng(int(seed), f"{TASK_ID}.force")
    target_value = int(rng.randint(74, 94)) if str(direction) == "highest" else int(rng.randint(6, 26))
    gap_min = max(3, int(gen_int(params, "style_legend_extremum_gap_min", 8)))
    gap_max = max(int(gap_min), int(gen_int(params, "style_legend_extremum_gap_max", 26)))
    forced_values: list[int] = []
    for x_index, old_value in enumerate(series.values):
        if int(x_index) == int(answer_x_index):
            forced_values.append(int(target_value))
        elif str(direction) == "highest":
            lowered = min(int(old_value), int(target_value) - int(rng.randint(int(gap_min), int(gap_max))))
            forced_values.append(max(int(context.value_min) + 3, int(lowered)))
        else:
            raised = max(int(old_value), int(target_value) + int(rng.randint(int(gap_min), int(gap_max))))
            forced_values.append(min(int(context.value_max) - 3, int(raised)))
    return replace_series_values(series, forced_values)


def _build_plan(params: Mapping[str, Any], seed: int, selected: str, probabilities: Mapping[str, float]):
    """Sample one fixed-series extremum objective and bind the answer marker."""

    task_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
    context = sample_context(task_params, instance_seed=int(seed))
    direction = _selected_direction(str(selected))
    answer_x_index, target_series_index = _sample_operands(context, task_params, seed=int(seed))
    target_series = context.series[int(target_series_index)]
    updated_target = _force_fixed_series_extremum(
        context,
        target_series,
        answer_x_index=int(answer_x_index),
        direction=str(direction),
        seed=int(seed),
        params=task_params,
    )
    updated = list(context.series)
    updated[int(target_series_index)] = updated_target
    answer_x_label = str(context.labels_x[int(answer_x_index)])
    dataset = dataset_from_context(context, series=updated, target_x_index=int(answer_x_index))
    return package_point_label_plan(
        dataset=dataset,
        params=task_params,
        answer_value=str(answer_x_label),
        annotation_marker_id=point_id(str(updated_target.series_id), int(answer_x_index)),
        prompt_key=str(selected),
        prompt_slots={
            "target_series_label": str(updated_target.label),
        },
        answer_hint=ANSWER_HINT_X_LABEL,
        json_example_key="x_label",
        program_code=PROGRAM_CODE,
        reasoning_load=0.58,
        objective_trace={
            "target_series_id": str(updated_target.series_id),
            "target_series_label": str(updated_target.label),
            "extremum_direction": str(direction),
            "answer_x_index": int(answer_x_index),
            "answer_x_label": str(answer_x_label),
            "answer_support": [str(label) for label in context.labels_x],
            "query_id_probabilities": dict(probabilities),
        },
    )


@register_task
class ChartsStyleLegendSeriesExtremumXLabelTask:
    """Select the x-axis label where one styled legend series is extremal."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    objective_contract = "series_extremum_x_label"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_style_legend_lifecycle(
            task=self,
            instance_seed=int(instance_seed),
            params={**TASK_PARAM_DEFAULTS, **dict(params)},
            max_attempts=int(max_attempts),
            default_query_id=self.default_query_id,
            build_plan=_build_plan,
        )


__all__ = ["ChartsStyleLegendSeriesExtremumXLabelTask"]
