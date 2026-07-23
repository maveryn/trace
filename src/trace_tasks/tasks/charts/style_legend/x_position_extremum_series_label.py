"""Select the styled legend series with an extremal value at one x position."""

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
    ANSWER_HINT_LABEL,
)
from trace_tasks.tasks.charts.style_legend.shared.sampling import (
    dataset_from_context,
    replace_series_value,
    sample_context,
)
from trace_tasks.tasks.charts.style_legend.shared.state import DOMAIN, point_id
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__style_legend__x_position_extremum_series_label"
SUPPORTED_QUERY_IDS = (
    "x_position_highest_series_label",
    "x_position_lowest_series_label",
)
DEFAULT_QUERY_ID = "x_position_highest_series_label"
TASK_PARAM_DEFAULTS: dict[str, Any] = {}
PROGRAM_CODE = "arg_extremum(series, value(series, x_position), direction={highest,lowest}); output=string_label; annotation=point; scene=style_legend; scope=x_position_extremum_series_label"


def _direction(selected: str) -> str:
    if str(selected) == "x_position_highest_series_label":
        return "highest"
    if str(selected) == "x_position_lowest_series_label":
        return "lowest"
    raise ValueError(f"unsupported query_id: {selected}")


def _build_plan(params: Mapping[str, Any], seed: int, selected: str, probabilities: Mapping[str, float]):
    """Sample one extremal marker objective and bind the selected point."""

    task_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
    context = sample_context(task_params, instance_seed=int(seed))
    x_index = int(
        balanced_choice(
            tuple(range(1, max(2, int(context.x_count) - 1))),
            task_params,
            instance_seed=int(seed),
            namespace=f"{TASK_ID}.x_index",
        )
    )
    answer_index = int(
        balanced_choice(
            tuple(range(int(context.series_count))),
            task_params,
            instance_seed=int(seed),
            namespace=f"{TASK_ID}.answer_series",
        )
    )
    rng = spawn_rng(int(seed), f"{TASK_ID}.force")
    direction = _direction(str(selected))
    target_value = int(rng.randint(72, 92)) if direction == "highest" else int(rng.randint(8, 28))
    gap_min = max(3, int(gen_int(task_params, "style_legend_extremum_gap_min", 8)))
    gap_max = max(int(gap_min), int(gen_int(task_params, "style_legend_extremum_gap_max", 26)))
    updated = []
    for index, item in enumerate(context.series):
        if int(index) == int(answer_index):
            value = int(target_value)
        elif direction == "highest":
            value = max(int(context.value_min) + 3, int(target_value) - int(rng.randint(int(gap_min), int(gap_max))))
        else:
            value = min(int(context.value_max) - 3, int(target_value) + int(rng.randint(int(gap_min), int(gap_max))))
        updated.append(replace_series_value(item, x_index=int(x_index), value=int(value)))
    answer_series = updated[int(answer_index)]
    dataset = dataset_from_context(context, series=updated, target_x_index=int(x_index))
    return package_point_label_plan(
        dataset=dataset,
        params=task_params,
        answer_value=str(answer_series.label),
        annotation_marker_id=point_id(str(answer_series.series_id), int(x_index)),
        prompt_key=str(selected),
        prompt_slots={
            "x_label": str(context.labels_x[int(x_index)]),
            "extremum_direction": str(direction),
        },
        answer_hint=ANSWER_HINT_LABEL,
        json_example_key="extremum_label",
        program_code=PROGRAM_CODE,
        reasoning_load=0.58,
        objective_trace={
            "x_label": str(context.labels_x[int(x_index)]),
            "extremum_direction": str(direction),
            "answer_series_id": str(answer_series.series_id),
            "answer_series_label": str(answer_series.label),
            "answer_support": [str(label) for label in context.labels_series],
            "query_id_probabilities": dict(probabilities),
        },
    )


@register_task
class ChartsStyleLegendXPositionExtremumSeriesLabelTask:
    """Select the styled legend series with an extremal value at one x position."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    objective_contract = "x_position_extremum_series_label"
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


__all__ = ["ChartsStyleLegendXPositionExtremumSeriesLabelTask"]
