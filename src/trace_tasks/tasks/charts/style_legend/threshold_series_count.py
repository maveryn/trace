"""Count styled legend series satisfying a threshold at one x position."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.style_legend._lifecycle import (
    package_style_legend_plan,
    run_style_legend_lifecycle,
)
from trace_tasks.tasks.charts.style_legend.shared.defaults import GEN_DEFAULTS, balanced_choice, gen_int
from trace_tasks.tasks.charts.style_legend.shared.prompts import (
    ANSWER_HINT_COUNT,
    ANSWER_ONLY_EXAMPLES,
    JSON_EXAMPLES,
    POINT_SET_HINT,
)
from trace_tasks.tasks.charts.style_legend.shared.sampling import (
    base_series,
    common_setup,
    package_dataset,
    replace_series_value,
)
from trace_tasks.tasks.charts.style_legend.shared.state import DOMAIN, point_id
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import group_default


TASK_ID = "task_charts__style_legend__threshold_series_count"
SUPPORTED_QUERY_IDS = (
    "above_threshold_series_count",
    "below_threshold_series_count",
)
DEFAULT_QUERY_ID = "above_threshold_series_count"
TASK_PARAM_DEFAULTS: dict[str, Any] = {}
PROGRAM_CODE = "count(filter(series, compare(value(series, x_position), threshold, comparator={above,below}))); output=integer_count; annotation=point_set; scene=style_legend; scope=threshold_series_count"


def _comparator(selected: str) -> str:
    if str(selected) == "above_threshold_series_count":
        return "above"
    if str(selected) == "below_threshold_series_count":
        return "below"
    raise ValueError(f"unsupported query_id: {selected}")


def _threshold_support(params: Mapping[str, Any]) -> tuple[int, ...]:
    raw = params.get("style_legend_threshold_values", group_default(GEN_DEFAULTS, "style_legend_threshold_values", (40, 50, 60)))
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple(int(value) for value in raw)
    return (40, 50, 60)


def _build_plan(params: Mapping[str, Any], seed: int, selected: str, probabilities: Mapping[str, float]):
    """Sample the threshold-count objective and bind every counted marker."""

    task_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
    answer_min = int(gen_int(task_params, "style_legend_threshold_answer_min", 0))
    answer_max = int(gen_int(task_params, "style_legend_threshold_answer_max", 5))
    answer_support = tuple(range(int(answer_min), int(answer_max) + 1))
    target_count = int(balanced_choice(answer_support, task_params, instance_seed=int(seed), namespace=f"{TASK_ID}.answer_count"))
    (
        x_count,
        series_count,
        labels_x,
        meta_x,
        labels_series,
        meta_series,
        palette_mode,
        palette_probs,
        legend_position,
        legend_probs,
        styles,
    ) = common_setup(task_params, instance_seed=int(seed), min_series_count=max(4, int(target_count)))
    if int(target_count) > int(series_count):
        raise ValueError("threshold target count exceeds series count")
    value_min = int(gen_int(task_params, "style_legend_value_min", 0))
    value_max = int(gen_int(task_params, "style_legend_value_max", 100))
    x_index = int(
        balanced_choice(
            tuple(range(1, max(2, int(x_count) - 1))),
            task_params,
            instance_seed=int(seed),
            namespace=f"{TASK_ID}.x_index",
        )
    )
    threshold = int(balanced_choice(_threshold_support(task_params), task_params, instance_seed=int(seed), namespace=f"{TASK_ID}.threshold_value"))
    comparator = _comparator(str(selected))
    selected_indices = set(range(int(target_count)))
    series = base_series(
        labels=labels_series,
        x_count=int(x_count),
        styles=styles,
        instance_seed=int(seed),
        value_min=int(value_min),
        value_max=int(value_max),
    )
    rng = spawn_rng(int(seed), f"{TASK_ID}.force")
    updated = []
    annotation_ids: list[str] = []
    for index, item in enumerate(series):
        if int(index) in selected_indices:
            if comparator == "above":
                value = int(rng.randint(int(threshold) + 8, min(int(value_max) - 3, int(threshold) + 34)))
            else:
                value = int(rng.randint(max(int(value_min) + 3, int(threshold) - 34), int(threshold) - 8))
            annotation_ids.append(point_id(str(item.series_id), int(x_index)))
        else:
            if comparator == "above":
                value = int(rng.randint(max(int(value_min) + 3, int(threshold) - 34), int(threshold) - 8))
            else:
                value = int(rng.randint(int(threshold) + 8, min(int(value_max) - 3, int(threshold) + 34)))
        updated.append(replace_series_value(item, x_index=int(x_index), value=int(value)))
    dataset = package_dataset(
        x_labels_value=labels_x,
        x_label_meta=meta_x,
        series=updated,
        series_label_meta=meta_series,
        target_x_index=int(x_index),
        threshold_value=int(threshold),
        palette_mode=str(palette_mode),
        palette_mode_probabilities=palette_probs,
        legend_position=str(legend_position),
        legend_position_probabilities=legend_probs,
    )
    return package_style_legend_plan(
        dataset=dataset,
        params=task_params,
        answer_value=int(target_count),
        answer_type="integer",
        annotation_type="point_set",
        annotation_marker_ids=tuple(annotation_ids),
        prompt_key=str(selected),
        prompt_slots={
            "x_label": str(labels_x[int(x_index)]),
            "threshold_value": str(threshold),
            "threshold_relation_phrase": str(comparator),
        },
        answer_hint=ANSWER_HINT_COUNT,
        annotation_hint=POINT_SET_HINT,
        json_example=str(JSON_EXAMPLES["threshold_count"]),
        json_example_answer_only=str(ANSWER_ONLY_EXAMPLES["threshold_count"]),
        program_code=PROGRAM_CODE,
        reasoning_load=0.62,
        objective_trace={
            "x_label": str(labels_x[int(x_index)]),
            "threshold_value": int(threshold),
            "threshold_comparator": str(comparator),
            "answer_support": [int(value) for value in answer_support],
            "counted_series_ids": [str(updated[index].series_id) for index in sorted(selected_indices)],
            "query_id_probabilities": dict(probabilities),
        },
    )


@register_task
class ChartsStyleLegendThresholdSeriesCountTask:
    """Count styled legend series satisfying a threshold at one x position."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = "threshold_series_count"
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


__all__ = ["ChartsStyleLegendThresholdSeriesCountTask"]
