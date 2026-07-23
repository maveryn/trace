"""Public task for `task_charts__curve_panels__panel_curve_threshold_crossing_count`."""

from __future__ import annotations

from typing import Any, Mapping

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
    generation_int,
    make_single_threshold_crossing_curve,
    side_value,
    threshold_panel_context,
    threshold_crossing_points,
)

UPWARD_QUERY_ID = "panel_curve_upward_threshold_crossing_count"
DOWNWARD_QUERY_ID = "panel_curve_downward_threshold_crossing_count"
QUERY_DIRECTIONS = {UPWARD_QUERY_ID: "upward", DOWNWARD_QUERY_ID: "downward"}
TASK_PARAM_DEFAULTS: dict[str, Any] = {
    "method_count_min": 5,
    "method_count_max": 6,
    "x_tick_count_min": 5,
    "x_tick_count_max": 8,
    "curve_threshold_target_count_min": 1,
    "curve_threshold_target_count_max": 5,
}


@register_task
class ChartsScientificPanelCurveThresholdCrossingCountTask:
    """Count curves in one panel crossing a threshold in the requested direction."""

    task_id = "task_charts__curve_panels__panel_curve_threshold_crossing_count"
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = "charts"
    objective_contract = "panel_curve_threshold_crossing_count"
    supported_query_ids = (UPWARD_QUERY_ID, DOWNWARD_QUERY_ID)
    default_dataset_enabled = True

    def _build_panel_curve_threshold_crossing_plan(
        self, instance_seed: int, params: Mapping[str, Any], selected_query_id: str
    ) -> CurvePanelTaskPlan:
        """Build the task-owned semantic sample before shared rendering."""

        effective_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
        try:
            direction = QUERY_DIRECTIONS[str(selected_query_id)]
        except KeyError as exc:
            raise ValueError(
                f"unsupported curve threshold-crossing query: {selected_query_id}"
            ) from exc
        (
            sampled_x_values,
            y_min,
            y_max,
            panel_labels,
            method_labels,
            panel_label_meta,
            colors,
            rng,
            query_panel,
            threshold,
            values,
            _non_answer_params,
        ) = threshold_panel_context(
            effective_params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.panel_curve_threshold_crossing_count",
        )
        target_min = max(
            1,
            int(
                generation_int(effective_params, "curve_threshold_target_count_min", 1)
            ),
        )
        target_max = min(
            len(method_labels),
            int(
                generation_int(effective_params, "curve_threshold_target_count_max", 4)
            ),
        )
        if int(target_min) > int(target_max):
            target_min = int(target_max)
        target_count = int(
            balanced_choice(
                list(range(int(target_min), int(target_max) + 1)),
                effective_params,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.panel_curve_threshold_crossing_count.answer",
            )
        )
        shuffled_methods = list(method_labels)
        rng.shuffle(shuffled_methods)
        matching_methods = set(
            str(method) for method in shuffled_methods[: int(target_count)]
        )
        crossings = []
        nonmatching_side = "below" if str(direction) == "upward" else "above"
        for method_index, method in enumerate(method_labels):
            if str(method) in matching_methods:
                pivot = 1 + (
                    (
                        int(method_index)
                        + int(rng.randint(0, max(1, len(sampled_x_values) - 3)))
                    )
                    % max(1, len(sampled_x_values) - 2)
                )
                pivot = max(1, min(len(sampled_x_values) - 1, int(pivot)))
                values[str(query_panel)][str(method)] = (
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
                method_crossings = threshold_crossing_points(
                    panel_label=str(query_panel),
                    method_label=str(method),
                    x_axis_values=sampled_x_values,
                    values=values[str(query_panel)][str(method)],
                    threshold=int(threshold),
                    direction=str(direction),
                )
                if len(method_crossings) != 1:
                    raise RuntimeError(
                        "threshold crossing construction drifted from one crossing"
                    )
                crossings.extend(method_crossings)
            else:
                values[str(query_panel)][str(method)] = [
                    side_value(
                        rng=rng,
                        threshold=int(threshold),
                        side=str(nonmatching_side),
                        y_min=int(y_min),
                        y_max=int(y_max),
                    )
                    for _ in sampled_x_values
                ]
        if len(crossings) != int(target_count):
            raise RuntimeError(
                "curve threshold crossing construction lost target count"
            )
        annotation_crossing_ids = tuple(str(item.crossing_id) for item in crossings)
        phrase = "upward through" if str(direction) == "upward" else "downward through"
        query = build_curve_panel_query_record(
            prompt_key=selected_query_id,
            answer=target_count,
            answer_type="integer",
            panel_label=query_panel,
            threshold_value=threshold,
            threshold_direction=direction,
            threshold_panel_labels=(query_panel,),
            annotation_panel_labels=(query_panel,),
            annotation_threshold_crossing_ids=annotation_crossing_ids,
            trace={
                "query_panel_label": str(query_panel),
                "threshold_value": int(threshold),
                "threshold_crossing_direction": str(direction),
                "threshold_crossing_phrase": str(phrase),
                "matching_method_labels": [
                    str(method)
                    for method in method_labels
                    if str(method) in matching_methods
                ],
                "threshold_crossing_points": [
                    {
                        "crossing_id": str(item.crossing_id),
                        "method_label": str(item.method_label),
                        "x_value": round(float(item.x_value), 3),
                        "y_value": round(float(item.y_value), 3),
                        "direction": str(item.direction),
                    }
                    for item in crossings
                ],
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
                "panel_label": f'"{query.panel_label}"',
                "threshold_value": str(query.threshold_value),
                "threshold_crossing_phrase": str(phrase),
            },
            instance_seed=int(instance_seed),
            threshold_crossings=tuple(crossings),
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
            build_plan=self._build_panel_curve_threshold_crossing_plan,
        )


__all__ = ["ChartsScientificPanelCurveThresholdCrossingCountTask"]
