"""Public task for `task_charts__combo_mark__interval_threshold_condition_count`."""

from __future__ import annotations

from typing import Any, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.base import TaskOutput
from ._lifecycle import (
    ComboTaskPlan,
    combo_task_output_fields,
    make_combo_plan,
    run_combo_public_task,
)
from .shared.defaults import DOMAIN, SCENE_NAMESPACE
from .shared.prompts import build_prompt_artifacts
from .shared.sampling import (
    balanced_count_from_bounds,
    sample_base_dataset,
    threshold_candidates_for,
)
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


PRIMARY_INTERVAL_LINE_ABOVE_QUERY_ID = "primary_between_and_line_above"
LINE_INTERVAL_PRIMARY_ABOVE_QUERY_ID = "line_between_and_primary_above"


def _interval_role(selected_query_id: str) -> str:
    if str(selected_query_id) == PRIMARY_INTERVAL_LINE_ABOVE_QUERY_ID:
        return "primary"
    if str(selected_query_id) == LINE_INTERVAL_PRIMARY_ABOVE_QUERY_ID:
        return "line"
    raise ValueError(f"unsupported interval-threshold query: {selected_query_id}")


def _primary_interval_line_threshold(
    *,
    primary: Sequence[int],
    line: Sequence[int],
    interval_target_count: int,
    rng: Any,
) -> tuple[list[int], dict[str, int]]:
    """Choose a primary-value interval plus line lower threshold."""

    candidates: list[tuple[list[int], dict[str, int]]] = []
    primary_values = sorted(set(int(value) for value in primary))
    line_interval_thresholds = threshold_candidates_for(line, above=True)
    for low_index, low in enumerate(primary_values):
        for high in primary_values[low_index:]:
            for line_threshold in line_interval_thresholds:
                interval_matches = [
                    idx
                    for idx, (a, b) in enumerate(zip(primary, line))
                    if int(low) <= int(a) <= int(high) and int(b) > int(line_threshold)
                ]
                if len(interval_matches) == int(interval_target_count):
                    candidates.append(
                        (
                            [int(idx) for idx in interval_matches],
                            {
                                "lower_threshold_value": int(low),
                                "upper_threshold_value": int(high),
                                "line_threshold_value": int(line_threshold),
                            },
                        )
                    )
    if not candidates:
        raise ValueError("no primary-interval threshold pair")
    return candidates[int(rng.randrange(0, len(candidates)))]


def _line_interval_primary_threshold(
    *,
    primary: Sequence[int],
    line: Sequence[int],
    interval_target_count: int,
    rng: Any,
) -> tuple[list[int], dict[str, int]]:
    """Choose a line-value interval plus primary lower threshold."""

    candidates: list[tuple[list[int], dict[str, int]]] = []
    line_values = sorted(set(int(value) for value in line))
    primary_interval_thresholds = threshold_candidates_for(primary, above=True)
    for low_index, low in enumerate(line_values):
        for high in line_values[low_index:]:
            for primary_threshold in primary_interval_thresholds:
                interval_matches = [
                    idx
                    for idx, (a, b) in enumerate(zip(primary, line))
                    if int(low) <= int(b) <= int(high) and int(a) > int(primary_threshold)
                ]
                if len(interval_matches) == int(interval_target_count):
                    candidates.append(
                        (
                            [int(idx) for idx in interval_matches],
                            {
                                "lower_threshold_value": int(low),
                                "upper_threshold_value": int(high),
                                "primary_threshold_value": int(primary_threshold),
                            },
                        )
                    )
    if not candidates:
        raise ValueError("no line-interval threshold pair")
    return candidates[int(rng.randrange(0, len(candidates)))]


@register_task
class ChartsComboIntervalThresholdConditionCountTask:
    """Count categories satisfying one interval condition and one threshold condition."""

    task_id = "task_charts__combo_mark__interval_threshold_condition_count"
    reasoning_operations = ('filtering', 'counting', 'comparison', 'logical_composition')
    domain = DOMAIN
    objective_contract = "interval_threshold_condition_count"
    supported_query_ids = (PRIMARY_INTERVAL_LINE_ABOVE_QUERY_ID, LINE_INTERVAL_PRIMARY_ABOVE_QUERY_ID)
    default_dataset_enabled = True
    default_interval_count_dataset_enabled = True

    def _build_interval_threshold_plan(
        self,
        instance_seed: int,
        params: dict[str, Any],
        selected_query_id: str,
    ) -> ComboTaskPlan:
        """Bind the interval predicate, threshold predicate, and witnesses."""

        interval_count_dataset, interval_count_trace = sample_base_dataset(
            params=params,
            instance_seed=int(instance_seed),
            scene_sampling_divisor=len(self.supported_query_ids),
        )
        interval_target_count, interval_target_probabilities, interval_target_range = balanced_count_from_bounds(
            params=params,
            instance_seed=int(instance_seed),
            low_key="dual_condition_interval_target_count_min",
            high_key="dual_condition_interval_target_count_max",
            fallback=(1, 5),
            high_cap=max(1, len(interval_count_dataset.labels) - 1),
            sampling_divisor=len(self.supported_query_ids),
            namespace=f"{SCENE_NAMESPACE}.dual_condition_interval_target_count",
        )
        rng = spawn_rng(int(instance_seed), f"{self.task_id}.selection")
        if _interval_role(str(selected_query_id)) == "primary":
            interval_matches, interval_thresholds = _primary_interval_line_threshold(
                primary=interval_count_dataset.primary_values,
                line=interval_count_dataset.line_values,
                interval_target_count=int(interval_target_count),
                rng=rng,
            )
        else:
            interval_matches, interval_thresholds = _line_interval_primary_threshold(
                primary=interval_count_dataset.primary_values,
                line=interval_count_dataset.line_values,
                interval_target_count=int(interval_target_count),
                rng=rng,
            )
        interval_slots = {
            "primary_name": f'"{interval_count_dataset.primary_name}"',
            "line_name": f'"{interval_count_dataset.line_name}"',
            "lower_threshold_value": str(interval_thresholds["lower_threshold_value"]),
            "upper_threshold_value": str(interval_thresholds["upper_threshold_value"]),
        }
        if str(selected_query_id) == PRIMARY_INTERVAL_LINE_ABOVE_QUERY_ID:
            interval_slots["line_threshold_value"] = str(interval_thresholds["line_threshold_value"])
        else:
            interval_slots["primary_threshold_value"] = str(interval_thresholds["primary_threshold_value"])
        interval_count_prompt = build_prompt_artifacts(
            scene_variant=interval_count_dataset.scene_variant,
            prompt_query_key=str(selected_query_id),
            dynamic_slots=interval_slots,
            instance_seed=int(instance_seed),
        )
        return make_combo_plan(
            dataset=interval_count_dataset,
            dataset_trace=interval_count_trace,
            answer_type="integer",
            answer_value=int(len(interval_matches)),
            question_format="dual_condition_count_query",
            annotation_indices=tuple(int(idx) for idx in interval_matches),
            annotation_mode="mark_pair_set",
            relations={
                **interval_thresholds,
                "interval_target_count": int(interval_target_count),
                "interval_target_count_range": [int(interval_target_range[0]), int(interval_target_range[1])],
                "interval_target_count_probabilities": {str(key): float(value) for key, value in interval_target_probabilities.items()},
                "interval_matches_indices": [int(idx) for idx in interval_matches],
            },
            prompt_artifacts=interval_count_prompt,
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, interval_count_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=PRIMARY_INTERVAL_LINE_ABOVE_QUERY_ID,
            task_id=self.task_id,
        )
        interval_count_materialized = run_combo_public_task(
            instance_seed=int(instance_seed),
            params=interval_count_params,
            max_attempts=int(max_attempts),
            selected_query_id=str(selected_query_id),
            failure_label=self.task_id,
            build_plan=self._build_interval_threshold_plan,
        )
        interval_count_fields = combo_task_output_fields(interval_count_materialized)
        return TaskOutput(**interval_count_fields)


__all__ = ["ChartsComboIntervalThresholdConditionCountTask"]
