"""Public task for `task_charts__curve_panels__threshold_series_count`."""

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
from trace_tasks.tasks.charts.curve_panels.shared.defaults import (
    SCENE_NAMESPACE,
)
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.charts.curve_panels.shared.sampling import (
    balanced_choice,
    common_axes,
    make_random_panels,
    method_count_max,
    palette,
    point_id,
    threshold_for_x_values,
    without_sample_cursor,
)

ABOVE_QUERY_ID = "above_threshold_series_count"
BELOW_QUERY_ID = "below_threshold_series_count"
QUERY_TO_DIRECTION = {
    ABOVE_QUERY_ID: "above",
    BELOW_QUERY_ID: "below",
}
TASK_PARAM_DEFAULTS: dict[str, Any] = {}


def _force_threshold_methods(
    *,
    method_labels: tuple[str, ...],
    target_count: int,
    values: dict[str, dict[str, list[int]]],
    query_panel: str,
    x_index: int,
    threshold: int,
    threshold_direction: str,
    y_min: int,
    y_max: int,
    rng: Any,
) -> set[str]:
    """Choose target methods and force query-x values onto the requested side."""

    shuffled_methods = list(method_labels)
    rng.shuffle(shuffled_methods)
    matching_methods = set(
        str(method) for method in shuffled_methods[: int(target_count)]
    )
    direction = str(threshold_direction)
    if direction not in {"above", "below"}:
        raise ValueError(f"Unsupported threshold direction: {threshold_direction!r}")
    for method in method_labels:
        is_match = str(method) in matching_methods
        should_be_above = (direction == "above" and is_match) or (
            direction == "below" and not is_match
        )
        if should_be_above:
            values[str(query_panel)][str(method)][int(x_index)] = int(
                rng.randint(
                    int(threshold) + 8, min(int(y_max) - 5, int(threshold) + 34)
                )
            )
        else:
            values[str(query_panel)][str(method)][int(x_index)] = int(
                rng.randint(
                    max(int(y_min) + 5, int(threshold) - 34), int(threshold) - 4
                )
            )
    return matching_methods


@register_task
class ChartsScientificThresholdSeriesCountTask:
    """Count methods above or below a threshold at one x-position in one subplot."""

    task_id = "task_charts__curve_panels__threshold_series_count"
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = "charts"
    objective_contract = "threshold_series_count"
    supported_query_ids = (ABOVE_QUERY_ID, BELOW_QUERY_ID)
    default_dataset_enabled = True

    def _build_threshold_series_count_plan(
        self, instance_seed: int, params: Mapping[str, Any], selected_query_id: str
    ) -> CurvePanelTaskPlan:
        """Build the task-owned semantic sample before shared rendering."""

        direction = QUERY_TO_DIRECTION[str(selected_query_id)]
        effective_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
        non_answer_params = without_sample_cursor(effective_params)
        target_count = int(
            balanced_choice(
                list(range(1, method_count_max(effective_params) + 1)),
                effective_params,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.threshold_count.{direction}.answer",
            )
        )
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
            min_method_count=int(target_count),
        )
        colors = palette(effective_params)
        rng = spawn_rng(
            int(instance_seed), f"{SCENE_NAMESPACE}.threshold_count.{direction}"
        )
        query_panel = str(
            balanced_choice(
                panel_labels,
                non_answer_params,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.threshold_count.{direction}.panel",
            )
        )
        x_index = 1 + int(rng.randint(0, max(1, len(sampled_x_values) - 3)))
        x_value = int(sampled_x_values[int(x_index)])
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
            namespace=f"{SCENE_NAMESPACE}.threshold_count.{direction}.values",
            value_min=y_min,
            value_max=y_max,
        )
        matching_methods = _force_threshold_methods(
            method_labels=tuple(method_labels),
            target_count=int(target_count),
            values=values,
            query_panel=str(query_panel),
            x_index=int(x_index),
            threshold=int(threshold),
            threshold_direction=str(direction),
            y_min=int(y_min),
            y_max=int(y_max),
            rng=rng,
        )
        annotation_ids = tuple(
            point_id(str(query_panel), str(method), int(x_value))
            for method in method_labels
            if str(method) in matching_methods
        )
        query = build_curve_panel_query_record(
            prompt_key=selected_query_id,
            answer=target_count,
            answer_type="integer",
            panel_label=query_panel,
            x_value=x_value,
            threshold_value=threshold,
            threshold_direction=str(direction),
            threshold_panel_labels=(query_panel,),
            annotation_panel_labels=(query_panel,),
            annotation_point_ids=annotation_ids,
            trace={
                "query_panel_label": str(query_panel),
                "query_x_value": int(x_value),
                "threshold_value": int(threshold),
                "threshold_direction": str(direction),
                "threshold_direction_phrase": str(direction),
                "matching_method_labels": [
                    str(method)
                    for method in method_labels
                    if str(method) in matching_methods
                ],
                "values_at_query_x": {
                    str(method): int(
                        values[str(query_panel)][str(method)][int(x_index)]
                    )
                    for method in method_labels
                },
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
                "x_value": str(query.x_value),
                "threshold_value": str(query.threshold_value),
            },
            instance_seed=int(instance_seed),
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
            default_query_id=ABOVE_QUERY_ID,
            failure_label=self.task_id,
            build_plan=self._build_threshold_series_count_plan,
        )


__all__ = ["ChartsScientificThresholdSeriesCountTask"]
