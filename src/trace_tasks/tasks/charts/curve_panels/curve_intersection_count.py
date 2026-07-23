"""Public task for `task_charts__curve_panels__curve_intersection_count`."""

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
    build_intersection_curves,
    common_axes,
    intersection_points,
    make_random_panels,
    palette,
    without_sample_cursor,
)

QUERY_ID = "curve_intersection_count"
TASK_PARAM_DEFAULTS: dict[str, Any] = {}


@register_task
class ChartsScientificCurveIntersectionCountTask:
    """Count intersections between two methods in one subplot."""

    task_id = "task_charts__curve_panels__curve_intersection_count"
    reasoning_operations = ('counting', 'spatial_relations')
    domain = "charts"
    objective_contract = "curve_intersection_count"
    supported_query_ids = (QUERY_ID,)
    default_dataset_enabled = True

    def _build_curve_intersection_count_plan(
        self, instance_seed: int, params: Mapping[str, Any], selected_query_id: str
    ) -> CurvePanelTaskPlan:
        """Build the task-owned semantic sample before shared rendering."""

        effective_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
        (
            sampled_x_values,
            y_min,
            y_max,
            _panel_total,
            panel_labels,
            method_labels,
            panel_label_meta,
        ) = common_axes(
            effective_params, instance_seed=int(instance_seed), min_x_tick_count=5
        )
        colors = palette(effective_params)
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.intersection_count")
        query_panel = str(
            balanced_choice(
                panel_labels,
                without_sample_cursor(effective_params),
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.intersection_count.panel",
            )
        )
        method_a_label = str(method_labels[0])
        method_b_label = str(method_labels[1])
        target_count = int(
            balanced_choice(
                list(range(0, 5)),
                effective_params,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.intersection_count.answer",
            )
        )
        values = make_random_panels(
            panel_labels=panel_labels,
            method_labels=method_labels,
            x_count=len(sampled_x_values),
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.intersection_count.values",
            value_min=y_min,
            value_max=y_max,
        )
        method_a_values, method_b_values, _ = build_intersection_curves(
            x_axis_values=sampled_x_values,
            target_count=int(target_count),
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.intersection_count.curves",
        )
        values[str(query_panel)][str(method_a_label)] = list(method_a_values)
        values[str(query_panel)][str(method_b_label)] = list(method_b_values)
        intersections = intersection_points(
            panel_label=str(query_panel),
            method_a_label=str(method_a_label),
            method_b_label=str(method_b_label),
            x_axis_values=sampled_x_values,
            values_a=method_a_values,
            values_b=method_b_values,
        )
        if len(intersections) != int(target_count):
            raise RuntimeError("intersection construction drifted from target count")
        query = build_curve_panel_query_record(
            prompt_key=selected_query_id,
            answer=target_count,
            answer_type="integer",
            panel_label=query_panel,
            method_a_label=method_a_label,
            method_b_label=method_b_label,
            annotation_panel_labels=(query_panel,),
            annotation_intersection_ids=tuple(
                str(item.intersection_id) for item in intersections
            ),
            trace={
                "query_panel_label": str(query_panel),
                "method_a_label": str(method_a_label),
                "method_b_label": str(method_b_label),
                "intersection_count": int(target_count),
                "intersection_points": [
                    {
                        "x_value": round(float(item.x_value), 3),
                        "y_value": round(float(item.y_value), 3),
                    }
                    for item in intersections
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
                "method_a_label": f'"{query.method_a_label}"',
                "method_b_label": f'"{query.method_b_label}"',
            },
            instance_seed=int(instance_seed),
            intersections=tuple(intersections),
            allow_empty_annotation=True,
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
            default_query_id=QUERY_ID,
            failure_label=self.task_id,
            build_plan=self._build_curve_intersection_count_plan,
        )


__all__ = ["ChartsScientificCurveIntersectionCountTask"]
