"""Public task for `task_charts__curve_panels__cross_panel_threshold_earliest_label`."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.curve_panels._lifecycle import (
    CurvePanelTaskPlan,
    build_curve_panel_plan_from_query,
    build_curve_panel_query_record,
    run_curve_panel_task_lifecycle,
)
from trace_tasks.tasks.charts.curve_panels.shared.defaults import (
    SCENE_NAMESPACE,
)
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.charts.curve_panels.shared.sampling import (
    balanced_choice,
    common_axes,
    make_random_panels,
    make_single_threshold_crossing_curve,
    palette,
    panel_answer_index_support,
    threshold_crossing_points,
    threshold_for_x_values,
    without_sample_cursor,
)

UPWARD_QUERY_ID = "cross_panel_upward_threshold_earliest_label"
DOWNWARD_QUERY_ID = "cross_panel_downward_threshold_earliest_label"
QUERY_DIRECTIONS = {UPWARD_QUERY_ID: "upward", DOWNWARD_QUERY_ID: "downward"}
TASK_PARAM_DEFAULTS: dict[str, Any] = {
    "method_count_min": 3,
    "method_count_max": 5,
    "x_tick_count_min": 6,
    "x_tick_count_max": 9,
}


@register_task
class ChartsScientificCrossPanelThresholdEarliestLabelTask:
    """Select the panel where one method crosses a threshold earliest."""

    task_id = "task_charts__curve_panels__cross_panel_threshold_earliest_label"
    reasoning_operations = ('filtering', 'comparison', 'ranking')
    domain = "charts"
    objective_contract = "cross_panel_threshold_earliest_label"
    supported_query_ids = (UPWARD_QUERY_ID, DOWNWARD_QUERY_ID)
    default_dataset_enabled = True

    def _build_cross_panel_threshold_earliest_plan(
        self, instance_seed: int, params: Mapping[str, Any], selected_query_id: str
    ) -> CurvePanelTaskPlan:
        """Build the task-owned semantic sample before shared rendering."""

        effective_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
        try:
            direction = QUERY_DIRECTIONS[str(selected_query_id)]
        except KeyError as exc:
            raise ValueError(
                f"unsupported cross-panel threshold query: {selected_query_id}"
            ) from exc
        answer_panel_index = int(
            balanced_choice(
                panel_answer_index_support(effective_params),
                effective_params,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.cross_panel_threshold_earliest.answer",
            )
        )
        non_answer_params = without_sample_cursor(effective_params)
        min_panel_count = max(4, int(answer_panel_index) + 1)
        (
            sampled_x_values,
            y_min,
            y_max,
            _panel_total,
            panel_labels,
            method_labels,
            panel_label_meta,
        ) = common_axes(
            effective_params,
            instance_seed=int(instance_seed),
            min_x_tick_count=6,
            min_panel_count=int(min_panel_count),
        )
        answer_panel = str(panel_labels[int(answer_panel_index)])
        colors = palette(effective_params)
        rng = spawn_rng(
            int(instance_seed), f"{SCENE_NAMESPACE}.cross_panel_threshold_earliest"
        )
        method_label = str(
            balanced_choice(
                method_labels,
                non_answer_params,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.cross_panel_threshold_earliest.method",
            )
        )
        threshold = threshold_for_x_values(
            effective_params,
            instance_seed=int(instance_seed),
            x_axis_values=sampled_x_values,
        )
        values = make_random_panels(
            panel_labels=panel_labels,
            method_labels=method_labels,
            x_count=len(sampled_x_values),
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.cross_panel_threshold_earliest.values",
            value_min=y_min,
            value_max=y_max,
        )
        crossings = []
        crossing_x_by_panel: Dict[str, float] = {}
        answer_crossing_id = ""
        for panel_index, panel in enumerate(panel_labels):
            pivot = (
                1
                if str(panel) == str(answer_panel)
                else min(
                    len(sampled_x_values) - 1,
                    2 + (int(panel_index) % max(1, len(sampled_x_values) - 3)),
                )
            )
            values[str(panel)][str(method_label)] = (
                make_single_threshold_crossing_curve(
                    rng=rng,
                    x_count=len(sampled_x_values),
                    threshold=int(threshold),
                    direction=str(direction),
                    crossing_index=int(pivot),
                    y_min=int(y_min),
                    y_max=int(y_max),
                )
            )
            panel_crossings = threshold_crossing_points(
                panel_label=str(panel),
                method_label=str(method_label),
                x_axis_values=sampled_x_values,
                values=values[str(panel)][str(method_label)],
                threshold=int(threshold),
                direction=str(direction),
            )
            if len(panel_crossings) != 1:
                raise RuntimeError(
                    "cross-panel threshold construction drifted from one crossing per panel"
                )
            crossing = panel_crossings[0]
            crossings.append(crossing)
            crossing_x_by_panel[str(panel)] = round(float(crossing.x_value), 3)
            if str(panel) == str(answer_panel):
                answer_crossing_id = str(crossing.crossing_id)
        if min(
            crossing_x_by_panel, key=lambda label: (crossing_x_by_panel[label], label)
        ) != str(answer_panel):
            raise RuntimeError(
                "cross-panel threshold earliest construction lost unique target"
            )
        phrase = "upward through" if str(direction) == "upward" else "downward through"
        query = build_curve_panel_query_record(
            prompt_key=selected_query_id,
            answer=answer_panel,
            answer_type="string",
            panel_label=answer_panel,
            method_label=method_label,
            threshold_value=threshold,
            threshold_direction=direction,
            threshold_panel_labels=tuple(str(panel) for panel in panel_labels),
            annotation_panel_labels=(answer_panel,),
            annotation_threshold_crossing_ids=(answer_crossing_id,),
            trace={
                "method_label": str(method_label),
                "threshold_value": int(threshold),
                "threshold_crossing_direction": str(direction),
                "threshold_crossing_phrase": str(phrase),
                "threshold_crossing_x_by_panel": dict(crossing_x_by_panel),
                "winning_panel_label": str(answer_panel),
                **dict(panel_label_meta),
            },
        )
        return build_curve_panel_plan_from_query(
            x_values=tuple(sampled_x_values),
            y_min=int(y_min),
            y_max=int(y_max),
            panel_labels=tuple(panel_labels),
            method_labels=tuple(method_labels),
            colors=tuple(colors),
            values_by_panel_method=values,
            query=query,
            dynamic_slots={
                "method_label": f'"{query.method_label}"',
                "threshold_value": str(query.threshold_value),
                "threshold_crossing_phrase": str(phrase),
            },
            instance_seed=int(instance_seed),
            threshold_crossings=tuple(crossings),
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
            default_query_id=UPWARD_QUERY_ID,
            failure_label=self.task_id,
            build_plan=self._build_cross_panel_threshold_earliest_plan,
        )


__all__ = ["ChartsScientificCrossPanelThresholdEarliestLabelTask"]
