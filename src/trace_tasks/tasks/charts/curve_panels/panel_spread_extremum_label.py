"""Public task for `task_charts__curve_panels__panel_spread_extremum_label`."""

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
    point_id,
)
from trace_tasks.tasks.registry import register_task

LARGEST_QUERY_ID = "largest_panel_spread_label"
SMALLEST_QUERY_ID = "smallest_panel_spread_label"
QUERY_DIRECTIONS = {LARGEST_QUERY_ID: "largest", SMALLEST_QUERY_ID: "smallest"}
TASK_PARAM_DEFAULTS: dict[str, Any] = {
    "method_count_min": 3,
    "method_count_max": 5,
    "x_tick_count_min": 5,
    "x_tick_count_max": 8,
}


def _panel_min_max_points(
    *,
    values: Mapping[str, Mapping[str, list[int]]],
    panel_label: str,
    method_labels: tuple[str, ...],
    x_values: tuple[int, ...],
) -> tuple[tuple[str, int, int], tuple[str, int, int]]:
    """Return (method, x, value) for min and max points in one panel."""

    min_point: tuple[str, int, int] | None = None
    max_point: tuple[str, int, int] | None = None
    for method in method_labels:
        for x_index, x_value in enumerate(x_values):
            value = int(values[str(panel_label)][str(method)][int(x_index)])
            candidate = (str(method), int(x_value), int(value))
            if min_point is None or value < min_point[2]:
                min_point = candidate
            if max_point is None or value > max_point[2]:
                max_point = candidate
    if min_point is None or max_point is None:
        raise RuntimeError("panel has no visible points")
    return min_point, max_point


@register_task
class ChartsScientificPanelSpreadExtremumLabelTask:
    """Select the subplot with the largest or smallest marker-value spread."""

    task_id = "task_charts__curve_panels__panel_spread_extremum_label"
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = "charts"
    objective_contract = "panel_spread_extremum_label"
    supported_query_ids = (LARGEST_QUERY_ID, SMALLEST_QUERY_ID)
    default_dataset_enabled = True

    def _build_panel_spread_extremum_plan(
        self, instance_seed: int, params: Mapping[str, Any], selected_query_id: str
    ) -> CurvePanelTaskPlan:
        """Build the task-owned semantic sample before shared rendering."""

        effective_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
        try:
            direction = QUERY_DIRECTIONS[str(selected_query_id)]
        except KeyError as exc:
            raise ValueError(
                f"unsupported panel spread extremum query: {selected_query_id}"
            ) from exc

        sample = base_curve_panel_sample(
            params=effective_params,
            instance_seed=int(instance_seed),
            min_x_tick_count=5,
            min_panel_count=4,
            min_method_count=3,
            namespace=f"{SCENE_NAMESPACE}.panel_spread",
        )
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.panel_spread")

        low_method = str(sample.method_labels[0])
        high_method = str(sample.method_labels[1])
        low_x_index = 1
        high_x_index = max(2, len(sample.x_values) - 2)
        if str(direction) == "largest":
            for panel in sample.panel_labels:
                for method in sample.method_labels:
                    sample.values[str(panel)][str(method)] = [
                        max(35, min(65, int(value)))
                        for value in sample.values[str(panel)][str(method)]
                    ]
            sample.values[sample.answer_panel][low_method][int(low_x_index)] = int(
                rng.randint(5, 12)
            )
            sample.values[sample.answer_panel][high_method][int(high_x_index)] = int(
                rng.randint(88, 96)
            )
        else:
            for panel in sample.panel_labels:
                if str(panel) == str(sample.answer_panel):
                    for method in sample.method_labels:
                        sample.values[str(panel)][str(method)] = [
                            int(50 + rng.randint(-3, 3))
                            for _ in sample.x_values
                        ]
                    sample.values[str(panel)][low_method][int(low_x_index)] = 46
                    sample.values[str(panel)][high_method][int(high_x_index)] = 54
                else:
                    for method in sample.method_labels:
                        sample.values[str(panel)][str(method)] = [
                            max(28, min(72, int(value)))
                            for value in sample.values[str(panel)][str(method)]
                        ]
                    sample.values[str(panel)][low_method][int(low_x_index)] = int(
                        rng.randint(8, 18)
                    )
                    sample.values[str(panel)][high_method][int(high_x_index)] = int(
                        rng.randint(82, 94)
                    )

        spreads = {}
        for panel in sample.panel_labels:
            all_values = [
                int(value)
                for method in sample.method_labels
                for value in sample.values[str(panel)][str(method)]
            ]
            spreads[str(panel)] = int(max(all_values) - min(all_values))
        selected_panel = (
            max(spreads, key=lambda label: (spreads[label], label))
            if str(direction) == "largest"
            else min(spreads, key=lambda label: (spreads[label], label))
        )
        if str(selected_panel) != str(sample.answer_panel):
            raise RuntimeError("panel spread construction lost unique target")

        min_point, max_point = _panel_min_max_points(
            values=sample.values,
            panel_label=sample.answer_panel,
            method_labels=sample.method_labels,
            x_values=sample.x_values,
        )
        query = build_curve_panel_query_record(
            prompt_key=selected_query_id,
            answer=sample.answer_panel,
            answer_type="string",
            panel_label=sample.answer_panel,
            annotation_panel_labels=(sample.answer_panel,),
            annotation_keyed_point_ids={
                "min_point": point_id(sample.answer_panel, min_point[0], min_point[1]),
                "max_point": point_id(sample.answer_panel, max_point[0], max_point[1]),
            },
            trace={
                "spread_direction": str(direction),
                "panel_spreads": dict(spreads),
                "winning_panel_label": str(sample.answer_panel),
                "min_point": {
                    "method_label": str(min_point[0]),
                    "x_value": int(min_point[1]),
                    "y_value": int(min_point[2]),
                },
                "max_point": {
                    "method_label": str(max_point[0]),
                    "x_value": int(max_point[1]),
                    "y_value": int(max_point[2]),
                },
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
            annotation_type="point_map",
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
            default_query_id=LARGEST_QUERY_ID,
            failure_label=self.task_id,
            build_plan=self._build_panel_spread_extremum_plan,
        )


__all__ = ["ChartsScientificPanelSpreadExtremumLabelTask"]
