"""Public task for `task_charts__curve_panels__global_value_extremum_panel_label`."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.curve_panels._lifecycle import (
    CurvePanelTaskPlan,
    build_curve_panel_plan_from_query,
    build_curve_panel_query_record,
    run_curve_panel_task_lifecycle,
)
from trace_tasks.tasks.charts.curve_panels.shared.defaults import SCENE_NAMESPACE
from trace_tasks.tasks.charts.curve_panels.shared.sampling import (
    base_curve_panel_sample,
    choose_method_label,
    point_id,
)
from trace_tasks.tasks.registry import register_task

MAX_QUERY_ID = "overall_maximum_value_panel_label"
MIN_QUERY_ID = "overall_minimum_value_panel_label"
QUERY_DIRECTIONS = {MAX_QUERY_ID: "maximum", MIN_QUERY_ID: "minimum"}
TASK_PARAM_DEFAULTS: dict[str, Any] = {
    "method_count_min": 3,
    "method_count_max": 5,
    "x_tick_count_min": 5,
    "x_tick_count_max": 8,
}


@register_task
class ChartsScientificGlobalValueExtremumPanelLabelTask:
    """Select the subplot containing the overall maximum or minimum marker."""

    task_id = "task_charts__curve_panels__global_value_extremum_panel_label"
    reasoning_operations = ('ranking',)
    domain = "charts"
    objective_contract = "global_value_extremum_panel_label"
    supported_query_ids = (MAX_QUERY_ID, MIN_QUERY_ID)
    default_dataset_enabled = True

    def _build_global_value_extremum_plan(
        self, instance_seed: int, params: Mapping[str, Any], selected_query_id: str
    ) -> CurvePanelTaskPlan:
        """Build the task-owned semantic sample before shared rendering."""

        effective_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
        try:
            direction = QUERY_DIRECTIONS[str(selected_query_id)]
        except KeyError as exc:
            raise ValueError(
                f"unsupported global value extremum query: {selected_query_id}"
            ) from exc

        sample = base_curve_panel_sample(
            params=effective_params,
            instance_seed=int(instance_seed),
            min_x_tick_count=5,
            min_panel_count=4,
            min_method_count=3,
            namespace=f"{SCENE_NAMESPACE}.global_value_extremum",
        )
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.global_value_extremum")
        answer_method = choose_method_label(
            method_labels=sample.method_labels,
            params=sample.non_answer_params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.global_value_extremum.method",
        )
        x_index = 1 + int(rng.randint(0, max(1, len(sample.x_values) - 3)))
        x_value = int(sample.x_values[int(x_index)])

        if str(direction) == "maximum":
            target_value = int(rng.randint(88, 96))
            cap = int(target_value) - 10
            for panel in sample.panel_labels:
                for method in sample.method_labels:
                    sample.values[str(panel)][str(method)] = [
                        min(int(value), int(cap))
                        for value in sample.values[str(panel)][str(method)]
                    ]
        else:
            target_value = int(rng.randint(4, 12))
            floor = int(target_value) + 10
            for panel in sample.panel_labels:
                for method in sample.method_labels:
                    sample.values[str(panel)][str(method)] = [
                        max(int(value), int(floor))
                        for value in sample.values[str(panel)][str(method)]
                    ]
        sample.values[sample.answer_panel][answer_method][int(x_index)] = int(
            target_value
        )

        panel_extrema = {}
        for panel in sample.panel_labels:
            panel_values = [
                int(value)
                for method in sample.method_labels
                for value in sample.values[str(panel)][str(method)]
            ]
            panel_extrema[str(panel)] = (
                max(panel_values) if str(direction) == "maximum" else min(panel_values)
            )
        selected_panel = (
            max(panel_extrema, key=lambda label: (panel_extrema[label], label))
            if str(direction) == "maximum"
            else min(panel_extrema, key=lambda label: (panel_extrema[label], label))
        )
        if str(selected_panel) != str(sample.answer_panel):
            raise RuntimeError("global value extremum construction lost unique target")

        annotation_id = point_id(sample.answer_panel, answer_method, int(x_value))
        query = build_curve_panel_query_record(
            prompt_key=selected_query_id,
            answer=sample.answer_panel,
            answer_type="string",
            panel_label=sample.answer_panel,
            method_label=answer_method,
            x_value=x_value,
            annotation_panel_labels=(sample.answer_panel,),
            annotation_point_ids=(annotation_id,),
            trace={
                "extremum_direction": str(direction),
                "winning_panel_label": str(sample.answer_panel),
                "winning_method_label": str(answer_method),
                "winning_x_value": int(x_value),
                "winning_value": int(target_value),
                "panel_extrema": dict(panel_extrema),
                "winning_point_id": str(annotation_id),
                **dict(sample.panel_label_meta),
            },
        )
        return build_curve_panel_plan_from_query(
            x_values=sample.x_values,
            y_min=sample.y_min,
            y_max=sample.y_max,
            panel_labels=sample.panel_labels,
            method_labels=sample.method_labels,
            colors=sample.colors,
            values_by_panel_method=sample.values,
            query=query,
            dynamic_slots={},
            instance_seed=int(instance_seed),
            annotation_type="point",
        )

    def generate(
        self, instance_seed: int, *, params: dict[str, Any], max_attempts: int
    ) -> TaskOutput:
        """Select the local query, then run neutral curve-panel lifecycle."""

        return run_curve_panel_task_lifecycle(
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            supported_query_ids=self.supported_query_ids,
            default_query_id=MAX_QUERY_ID,
            failure_label=self.task_id,
            build_plan=self._build_global_value_extremum_plan,
        )


__all__ = ["ChartsScientificGlobalValueExtremumPanelLabelTask"]
