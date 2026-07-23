"""Public task for `task_charts__combo_mark__series_threshold_crossing_label`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import hash64, spawn_rng
from trace_tasks.tasks.base import TaskOutput
from ._lifecycle import (
    ComboTaskPlan,
    combo_task_output_fields,
    make_combo_plan,
    run_combo_public_task,
)
from .shared.defaults import DOMAIN, GENERATION_DEFAULTS, SCENE_NAMESPACE, scene_default
from .shared.prompts import build_prompt_artifacts
from .shared.sampling import dataset_with_values, sample_base_dataset
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


PRIMARY_ABOVE_QUERY_ID = "primary_first_above_threshold_label"
PRIMARY_BELOW_QUERY_ID = "primary_first_below_threshold_label"
LINE_ABOVE_QUERY_ID = "line_first_above_threshold_label"
LINE_BELOW_QUERY_ID = "line_first_below_threshold_label"
TASK_PARAM_DEFAULTS = {
    "crossing_index_min": 2,
    "crossing_index_max": 8,
}


def _target_role_and_direction(selected_query_id: str) -> tuple[str, bool]:
    if str(selected_query_id) == PRIMARY_ABOVE_QUERY_ID:
        return "primary", True
    if str(selected_query_id) == PRIMARY_BELOW_QUERY_ID:
        return "primary", False
    if str(selected_query_id) == LINE_ABOVE_QUERY_ID:
        return "line", True
    if str(selected_query_id) == LINE_BELOW_QUERY_ID:
        return "line", False
    raise ValueError(f"unsupported threshold-crossing query: {selected_query_id}")


def _balanced_crossing_index(
    *,
    params: dict[str, Any],
    instance_seed: int,
    label_count: int,
    sampling_divisor: int,
) -> tuple[int, dict[int, float], tuple[int, int]]:
    low = int(params.get("crossing_index_min", scene_default(GENERATION_DEFAULTS, "crossing_index_min", 2)))
    high = int(
        params.get(
            "crossing_index_max",
            scene_default(GENERATION_DEFAULTS, "crossing_index_max", int(label_count) - 2),
        )
    )
    low = max(1, int(low))
    high = min(int(high), int(label_count) - 2)
    if int(low) > int(high):
        raise ValueError("no feasible combo threshold-crossing index support")
    support = tuple(range(int(low), int(high) + 1))
    probabilities = {int(value): 1.0 / float(len(support)) for value in support}
    selected = int(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.crossing_index:{int(label_count)}"),
            support,
            sort_keys=True,
        )
    )
    return int(selected), probabilities, (int(low), int(high))


def _crossing_series_values(
    *,
    target_role: str,
    above: bool,
    label_count: int,
    value_min: int,
    value_max: int,
    params: dict[str, Any],
    instance_seed: int,
    target_sampling_divisor: int,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Construct one series where the first threshold crossing is controlled."""

    if str(target_role) not in {"primary", "line"}:
        raise ValueError(f"unsupported crossing target role: {target_role}")
    if int(value_min) + 2 > int(value_max):
        raise ValueError("combo threshold crossing requires at least three possible values")
    crossing_index, index_probabilities, index_range = _balanced_crossing_index(
        params=params,
        instance_seed=int(instance_seed),
        label_count=int(label_count),
        sampling_divisor=int(target_sampling_divisor),
    )
    rng_seed = hash64(int(instance_seed), f"{SCENE_NAMESPACE}.threshold_crossing:{target_role}:{'above' if above else 'below'}")
    rng = spawn_rng(int(rng_seed), "values")
    crossing_threshold = int(rng.randint(int(value_min) + 1, int(value_max) - 1))
    values: list[int] = []
    for idx in range(int(label_count)):
        if int(idx) < int(crossing_index):
            value = rng.randint(int(value_min), int(crossing_threshold)) if bool(above) else rng.randint(int(crossing_threshold), int(value_max))
        elif int(idx) == int(crossing_index):
            value = rng.randint(int(crossing_threshold) + 1, int(value_max)) if bool(above) else rng.randint(int(value_min), int(crossing_threshold) - 1)
        else:
            value = rng.randint(int(value_min), int(value_max))
        values.append(int(value))
    return tuple(int(value) for value in values), {
        "crossing_threshold_value": int(crossing_threshold),
        "threshold_value": int(crossing_threshold),
        "crossing_index": int(crossing_index),
        "crossing_index_range": [int(index_range[0]), int(index_range[1])],
        "crossing_index_probabilities": {str(key): float(value) for key, value in index_probabilities.items()},
        "crossing_direction": "above" if bool(above) else "below",
        "comparison_phrase": "above" if bool(above) else "below",
        "target_series_role": str(target_role),
    }


@register_task
class ChartsComboSeriesThresholdCrossingLabelTask:
    """Return the first category where one combo series crosses a threshold."""

    task_id = "task_charts__combo_mark__series_threshold_crossing_label"
    reasoning_operations = ('filtering', 'comparison', 'ranking')
    domain = DOMAIN
    objective_contract = "series_threshold_crossing_label"
    supported_query_ids = (
        PRIMARY_ABOVE_QUERY_ID,
        PRIMARY_BELOW_QUERY_ID,
        LINE_ABOVE_QUERY_ID,
        LINE_BELOW_QUERY_ID,
    )
    default_dataset_enabled = True
    default_crossing_dataset_enabled = True

    def _build_crossing_threshold_crossing_plan(
        self,
        instance_seed: int,
        params: dict[str, Any],
        selected_query_id: str,
    ) -> ComboTaskPlan:
        """Bind the first threshold-crossing label and answer-mark witness."""

        effective_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
        target_role, above = _target_role_and_direction(str(selected_query_id))
        crossing_dataset, controlled_trace_base = sample_base_dataset(
            params=effective_params,
            instance_seed=int(instance_seed),
            scene_sampling_divisor=len(self.supported_query_ids),
        )
        value_min, value_max = [int(value) for value in controlled_trace_base["value_range"]]
        controlled_values, controlled_trace = _crossing_series_values(
            target_role=str(target_role),
            above=bool(above),
            label_count=len(crossing_dataset.labels),
            value_min=int(value_min),
            value_max=int(value_max),
            params=effective_params,
            instance_seed=int(instance_seed),
            target_sampling_divisor=len(self.supported_query_ids),
        )
        crossing_dataset = (
            dataset_with_values(crossing_dataset, primary_values=controlled_values)
            if str(target_role) == "primary"
            else dataset_with_values(crossing_dataset, line_values=controlled_values)
        )
        controlled_series_values = crossing_dataset.primary_values if str(target_role) == "primary" else crossing_dataset.line_values
        crossing_answer_index = int(controlled_trace["crossing_index"])
        crossing_threshold = int(controlled_trace["crossing_threshold_value"])
        for idx, value in enumerate(controlled_series_values):
            satisfies = int(value) > int(crossing_threshold) if bool(above) else int(value) < int(crossing_threshold)
            if int(idx) < int(crossing_answer_index) and satisfies:
                raise ValueError("pre-crossing value unexpectedly satisfies crossing_threshold rule")
            if int(idx) == int(crossing_answer_index) and not satisfies:
                raise ValueError("crossing value does not satisfy crossing_threshold rule")
        target_series_name = str(crossing_dataset.primary_name if str(target_role) == "primary" else crossing_dataset.line_name)
        crossing_prompt = build_prompt_artifacts(
            scene_variant=crossing_dataset.scene_variant,
            prompt_query_key=str(selected_query_id),
            dynamic_slots={
                "primary_name": f'"{crossing_dataset.primary_name}"',
                "line_name": f'"{crossing_dataset.line_name}"',
                "target_series_name": f'"{target_series_name}"',
                "threshold_value": str(crossing_threshold),
                "comparison_phrase": str(controlled_trace["comparison_phrase"]),
            },
            instance_seed=int(instance_seed),
        )
        crossing_answer_label = str(crossing_dataset.labels[int(crossing_answer_index)])
        return make_combo_plan(
            dataset=crossing_dataset,
            dataset_trace=controlled_trace_base,
            answer_type="string",
            answer_value=str(crossing_answer_label),
            question_format="series_threshold_crossing_label_query",
            annotation_indices=(int(crossing_answer_index),),
            annotation_mode="single_mark_point",
            annotation_mark_role=str(target_role),
            relations={
                **dict(controlled_trace),
                "target_series_name": str(target_series_name),
                "target_series_values": [int(value) for value in controlled_series_values],
                "answer_index": int(crossing_answer_index),
                "crossing_answer_index": int(crossing_answer_index),
                "crossing_answer_label": str(crossing_answer_label),
                "ordered_annotation_labels": [str(crossing_dataset.labels[int(idx)]) for idx in range(0, int(crossing_answer_index) + 1)],
            },
            prompt_artifacts=crossing_prompt,
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, crossing_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=PRIMARY_ABOVE_QUERY_ID,
            task_id=self.task_id,
        )
        crossing_materialized = run_combo_public_task(
            instance_seed=int(instance_seed),
            params={**TASK_PARAM_DEFAULTS, **dict(crossing_params)},
            max_attempts=int(max_attempts),
            selected_query_id=str(selected_query_id),
            failure_label=self.task_id,
            build_plan=self._build_crossing_threshold_crossing_plan,
        )
        crossing_fields = combo_task_output_fields(crossing_materialized)
        return TaskOutput(**crossing_fields)


__all__ = ["ChartsComboSeriesThresholdCrossingLabelTask"]
