"""Public task for `task_charts__combo_mark__cross_mark_difference_value`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.base import TaskOutput
from ._lifecycle import (
    ComboTaskPlan,
    combo_task_output_fields,
    make_combo_plan,
    run_combo_public_task,
)
from .shared.defaults import DOMAIN
from .shared.prompts import build_prompt_artifacts
from .shared.sampling import sample_base_dataset
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


PRIMARY_MINUS_LINE_QUERY_ID = "primary_minus_line_at_label"
LINE_MINUS_PRIMARY_QUERY_ID = "line_minus_primary_at_label"
TASK_PARAM_DEFAULTS = {
    "label_count_min": 9,
    "label_count_max": 12,
}
ALLOWED_SCENE_VARIANTS = ("bar_line_shared_axis", "stacked_bar_line")


def _difference_mode(selected_query_id: str) -> str:
    if str(selected_query_id) == PRIMARY_MINUS_LINE_QUERY_ID:
        return "primary_minus_line"
    if str(selected_query_id) == LINE_MINUS_PRIMARY_QUERY_ID:
        return "line_minus_primary"
    raise ValueError(f"unsupported combo difference query: {selected_query_id}")


@register_task
class ChartsComboCrossMarkDifferenceValueTask:
    """Compute the signed difference between primary and overlaid line marks at one label."""

    task_id = "task_charts__combo_mark__cross_mark_difference_value"
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    objective_contract = "cross_mark_difference_value"
    supported_query_ids = (PRIMARY_MINUS_LINE_QUERY_ID, LINE_MINUS_PRIMARY_QUERY_ID)
    default_dataset_enabled = True
    default_difference_dataset_enabled = True

    def _build_cross_difference_plan(
        self,
        instance_seed: int,
        params: dict[str, Any],
        selected_query_id: str,
    ) -> ComboTaskPlan:
        """Bind the target label and signed subtraction before rendering."""

        effective_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
        difference_dataset, difference_trace = sample_base_dataset(
            params=effective_params,
            instance_seed=int(instance_seed),
            allowed_scene_variants=ALLOWED_SCENE_VARIANTS,
            label_count_bounds=(9, 12),
            scene_sampling_divisor=len(self.supported_query_ids),
        )
        usable = [
            idx
            for idx, (primary_value, line_value) in enumerate(zip(difference_dataset.primary_values, difference_dataset.line_values))
            if int(primary_value) != int(line_value)
        ]
        if not usable:
            raise ValueError("cross-mark gap needs unequal values")
        rng = spawn_rng(int(instance_seed), f"{self.task_id}.selection")
        difference_index = int(usable[int(rng.randrange(0, len(usable)))])
        if _difference_mode(str(selected_query_id)) == "primary_minus_line":
            answer = int(difference_dataset.primary_values[difference_index]) - int(difference_dataset.line_values[difference_index])
        else:
            answer = int(difference_dataset.line_values[difference_index]) - int(difference_dataset.primary_values[difference_index])
        difference_label = str(difference_dataset.labels[int(difference_index)])
        difference_prompt = build_prompt_artifacts(
            scene_variant=difference_dataset.scene_variant,
            prompt_query_key=str(selected_query_id),
            dynamic_slots={
                "primary_name": f'"{difference_dataset.primary_name}"',
                "line_name": f'"{difference_dataset.line_name}"',
                "target_label": f'"{difference_label}"',
            },
            instance_seed=int(instance_seed),
        )
        return make_combo_plan(
            dataset=difference_dataset,
            dataset_trace=difference_trace,
            answer_type="integer",
            answer_value=int(answer),
            question_format="cross_mark_difference_query",
            annotation_indices=(int(difference_index),),
            annotation_mode="paired_mark_map",
            relations={
                "difference_label": str(difference_label),
                "difference_index": int(difference_index),
            },
            prompt_artifacts=difference_prompt,
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, difference_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=PRIMARY_MINUS_LINE_QUERY_ID,
            task_id=self.task_id,
        )
        difference_materialized = run_combo_public_task(
            instance_seed=int(instance_seed),
            params={**TASK_PARAM_DEFAULTS, **dict(difference_params)},
            max_attempts=int(max_attempts),
            selected_query_id=str(selected_query_id),
            failure_label=self.task_id,
            build_plan=self._build_cross_difference_plan,
        )
        difference_fields = combo_task_output_fields(difference_materialized)
        return TaskOutput(**difference_fields)


__all__ = ["ChartsComboCrossMarkDifferenceValueTask"]
