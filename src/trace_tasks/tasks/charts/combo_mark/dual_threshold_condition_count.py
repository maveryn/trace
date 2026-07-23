"""Public task for `task_charts__combo_mark__dual_threshold_condition_count`."""

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


PRIMARY_ABOVE_LINE_ABOVE_QUERY_ID = "primary_above_and_line_above"
PRIMARY_ABOVE_LINE_BELOW_QUERY_ID = "primary_above_and_line_below"
PRIMARY_BELOW_LINE_ABOVE_QUERY_ID = "primary_below_and_line_above"


def _threshold_relations(selected_query_id: str) -> tuple[str, str]:
    if str(selected_query_id) == PRIMARY_ABOVE_LINE_ABOVE_QUERY_ID:
        return "above", "above"
    if str(selected_query_id) == PRIMARY_ABOVE_LINE_BELOW_QUERY_ID:
        return "above", "below"
    if str(selected_query_id) == PRIMARY_BELOW_LINE_ABOVE_QUERY_ID:
        return "below", "above"
    raise ValueError(f"unsupported dual-threshold query: {selected_query_id}")


def _dual_threshold_matches(
    *,
    primary: Sequence[int],
    line: Sequence[int],
    dual_target_count: int,
    primary_relation: str,
    line_relation: str,
    rng: Any,
) -> tuple[list[int], dict[str, int]]:
    """Choose dual_thresholds that produce the target number of dual_matches categories."""

    primary_above = str(primary_relation) == "above"
    line_above = str(line_relation) == "above"
    primary_dual_thresholds = threshold_candidates_for(primary, above=primary_above)
    line_dual_thresholds = threshold_candidates_for(line, above=line_above)
    candidates: list[tuple[list[int], dict[str, int]]] = []
    for primary_threshold in primary_dual_thresholds:
        for line_threshold in line_dual_thresholds:
            dual_matches = [
                idx
                for idx, (a, b) in enumerate(zip(primary, line))
                if (int(a) > int(primary_threshold) if primary_above else int(a) < int(primary_threshold))
                and (int(b) > int(line_threshold) if line_above else int(b) < int(line_threshold))
            ]
            if len(dual_matches) == int(dual_target_count):
                candidates.append(
                    (
                        [int(idx) for idx in dual_matches],
                        {
                            "primary_threshold_value": int(primary_threshold),
                            "line_threshold_value": int(line_threshold),
                        },
                    )
                )
    if not candidates:
        raise ValueError("no targeted dual-condition threshold pair")
    return candidates[int(rng.randrange(0, len(candidates)))]


@register_task
class ChartsComboDualThresholdConditionCountTask:
    """Count categories satisfying two one-bound threshold conditions."""

    task_id = "task_charts__combo_mark__dual_threshold_condition_count"
    reasoning_operations = ('filtering', 'counting', 'comparison', 'logical_composition')
    domain = DOMAIN
    objective_contract = "dual_threshold_condition_count"
    supported_query_ids = (
        PRIMARY_ABOVE_LINE_ABOVE_QUERY_ID,
        PRIMARY_ABOVE_LINE_BELOW_QUERY_ID,
        PRIMARY_BELOW_LINE_ABOVE_QUERY_ID,
    )
    default_dataset_enabled = True
    default_dual_count_dataset_enabled = True

    def _build_dual_threshold_plan(
        self,
        instance_seed: int,
        params: dict[str, Any],
        selected_query_id: str,
    ) -> ComboTaskPlan:
        """Bind two threshold predicates and all dual_matches category witnesses."""

        dual_count_dataset, dual_count_trace = sample_base_dataset(
            params=params,
            instance_seed=int(instance_seed),
            scene_sampling_divisor=len(self.supported_query_ids),
        )
        dual_target_count, dual_target_probabilities, dual_target_range = balanced_count_from_bounds(
            params=params,
            instance_seed=int(instance_seed),
            low_key="dual_condition_dual_target_count_min",
            high_key="dual_condition_dual_target_count_max",
            fallback=(1, 5),
            high_cap=max(1, len(dual_count_dataset.labels) - 1),
            sampling_divisor=len(self.supported_query_ids),
            namespace=f"{SCENE_NAMESPACE}.dual_condition_dual_target_count",
        )
        primary_relation, line_relation = _threshold_relations(str(selected_query_id))
        rng = spawn_rng(int(instance_seed), f"{self.task_id}.selection")
        dual_matches, dual_thresholds = _dual_threshold_matches(
            primary=dual_count_dataset.primary_values,
            line=dual_count_dataset.line_values,
            dual_target_count=int(dual_target_count),
            primary_relation=str(primary_relation),
            line_relation=str(line_relation),
            rng=rng,
        )
        dual_count_prompt = build_prompt_artifacts(
            scene_variant=dual_count_dataset.scene_variant,
            prompt_query_key=str(selected_query_id),
            dynamic_slots={
                "primary_name": f'"{dual_count_dataset.primary_name}"',
                "line_name": f'"{dual_count_dataset.line_name}"',
                "primary_threshold_value": str(dual_thresholds["primary_threshold_value"]),
                "line_threshold_value": str(dual_thresholds["line_threshold_value"]),
            },
            instance_seed=int(instance_seed),
        )
        return make_combo_plan(
            dataset=dual_count_dataset,
            dataset_trace=dual_count_trace,
            answer_type="integer",
            answer_value=int(len(dual_matches)),
            question_format="dual_condition_count_query",
            annotation_indices=tuple(int(idx) for idx in dual_matches),
            annotation_mode="mark_pair_set",
            relations={
                **dual_thresholds,
                "dual_target_count": int(dual_target_count),
                "dual_target_count_range": [int(dual_target_range[0]), int(dual_target_range[1])],
                "dual_target_count_probabilities": {str(key): float(value) for key, value in dual_target_probabilities.items()},
                "dual_matches_indices": [int(idx) for idx in dual_matches],
            },
            prompt_artifacts=dual_count_prompt,
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, dual_count_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=PRIMARY_ABOVE_LINE_ABOVE_QUERY_ID,
            task_id=self.task_id,
        )
        dual_count_materialized = run_combo_public_task(
            instance_seed=int(instance_seed),
            params=dual_count_params,
            max_attempts=int(max_attempts),
            selected_query_id=str(selected_query_id),
            failure_label=self.task_id,
            build_plan=self._build_dual_threshold_plan,
        )
        dual_count_fields = combo_task_output_fields(dual_count_materialized)
        return TaskOutput(**dual_count_fields)


__all__ = ["ChartsComboDualThresholdConditionCountTask"]
