"""Public task for `task_charts__curve_panels__curve_at_x_extremum_label`."""

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
    generation_int,
    make_random_panels,
    method_count_max,
    method_labels_for_seed,
    palette,
    point_id,
    without_sample_cursor,
)

QUERY_ID = "curve_at_x_extremum_label"
TASK_PARAM_DEFAULTS: dict[str, Any] = {
    "method_count_min": 6,
    "method_count_max": 6,
    "x_tick_count_min": 8,
    "x_tick_count_max": 10,
    "curve_at_x_winner_min": 62,
    "curve_at_x_winner_max": 88,
    "curve_at_x_gap_min": 4,
    "curve_at_x_gap_max": 18,
}


@register_task
class ChartsScientificCurveAtXExtremumLabelTask:
    """Return the method label with the highest value at one x-position in one subplot."""

    task_id = "task_charts__curve_panels__curve_at_x_extremum_label"
    reasoning_operations = ('ranking',)
    domain = "charts"
    objective_contract = "curve_at_x_extremum_label"
    supported_query_ids = (QUERY_ID,)
    default_dataset_enabled = True

    def _build_curve_at_x_extremum_plan(
        self, instance_seed: int, params: Mapping[str, Any], selected_query_id: str
    ) -> CurvePanelTaskPlan:
        """Build the task-owned semantic sample before shared rendering."""

        effective_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
        non_answer_params = without_sample_cursor(effective_params)
        method_answer_support = method_labels_for_seed(
            count=method_count_max(effective_params), instance_seed=int(instance_seed)
        )
        answer_method = str(
            balanced_choice(
                method_answer_support,
                effective_params,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.curve_at_x.answer",
            )
        )
        min_method_count = max(
            3, int(method_answer_support.index(str(answer_method))) + 1
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
            min_method_count=int(min_method_count),
        )
        colors = palette(effective_params)
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.curve_at_x")
        query_panel = str(
            balanced_choice(
                panel_labels,
                non_answer_params,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.curve_at_x.panel",
            )
        )
        x_index = 1 + int(rng.randint(0, max(1, len(sampled_x_values) - 3)))
        x_value = int(sampled_x_values[int(x_index)])
        values = make_random_panels(
            panel_labels=panel_labels,
            method_labels=method_labels,
            x_count=len(sampled_x_values),
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.curve_at_x.values",
            value_min=y_min,
            value_max=y_max,
        )
        winner_min = int(generation_int(effective_params, "curve_at_x_winner_min", 82))
        winner_max = int(generation_int(effective_params, "curve_at_x_winner_max", 92))
        gap_min = max(
            1, int(generation_int(effective_params, "curve_at_x_gap_min", 24))
        )
        gap_max = max(
            int(gap_min),
            int(generation_int(effective_params, "curve_at_x_gap_max", 56)),
        )
        winner_min = max(
            int(y_min) + int(gap_min) + 5, min(int(y_max) - 5, int(winner_min))
        )
        winner_max = max(int(winner_min), min(int(y_max) - 5, int(winner_max)))
        winning_value = int(rng.randint(int(winner_min), int(winner_max)))
        for method in method_labels:
            if str(method) == str(answer_method):
                values[str(query_panel)][str(method)][int(x_index)] = int(winning_value)
            else:
                gap = int(rng.randint(int(gap_min), int(gap_max)))
                values[str(query_panel)][str(method)][int(x_index)] = max(
                    int(y_min) + 5, min(int(y_max) - 5, int(winning_value) - int(gap))
                )
        annotation_ids = (
            point_id(str(query_panel), str(answer_method), int(x_value)),
        )
        query = build_curve_panel_query_record(
            prompt_key=selected_query_id,
            answer=answer_method,
            answer_type="string",
            panel_label=query_panel,
            method_label=answer_method,
            x_value=x_value,
            annotation_panel_labels=(query_panel,),
            annotation_point_ids=annotation_ids,
            trace={
                "query_panel_label": str(query_panel),
                "query_x_value": int(x_value),
                "compared_method_labels": list(method_labels),
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
            },
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
            default_query_id=QUERY_ID,
            failure_label=self.task_id,
            build_plan=self._build_curve_at_x_extremum_plan,
        )


__all__ = ["ChartsScientificCurveAtXExtremumLabelTask"]
