"""Public task for `task_charts__curve_panels__earliest_maximum_panel_label`."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

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
    palette,
    panel_answer_index_support,
    point_id,
    without_sample_cursor,
)

QUERY_ID = "earliest_maximum_panel_label"
TASK_PARAM_DEFAULTS: dict[str, Any] = {}


@register_task
class ChartsScientificEarliestMaximumPanelLabelTask:
    """Select the panel where one method reaches its maximum earliest."""

    task_id = "task_charts__curve_panels__earliest_maximum_panel_label"
    reasoning_operations = ('ranking',)
    domain = "charts"
    objective_contract = "earliest_maximum_panel_label"
    supported_query_ids = (QUERY_ID,)
    default_dataset_enabled = True

    def _build_earliest_maximum_panel_plan(
        self, instance_seed: int, params: Mapping[str, Any], selected_query_id: str
    ) -> CurvePanelTaskPlan:
        """Build the task-owned semantic sample before shared rendering."""

        effective_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
        answer_panel_index = int(
            balanced_choice(
                panel_answer_index_support(effective_params),
                effective_params,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.earliest_max.answer",
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
            min_panel_count=int(min_panel_count),
        )
        answer_panel = str(panel_labels[int(answer_panel_index)])
        colors = palette(effective_params)
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.earliest_max")
        method_label = str(
            balanced_choice(
                method_labels,
                non_answer_params,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.earliest_max.method",
            )
        )
        values = make_random_panels(
            panel_labels=panel_labels,
            method_labels=method_labels,
            x_count=len(sampled_x_values),
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.earliest_max.values",
            value_min=y_min,
            value_max=y_max,
        )
        peak_indices: Dict[str, int] = {}
        for panel_index, panel in enumerate(panel_labels):
            peak_index = (
                1
                if str(panel) == str(answer_panel)
                else min(
                    len(sampled_x_values) - 2,
                    2 + (int(panel_index) % max(1, len(sampled_x_values) - 3)),
                )
            )
            peak_indices[str(panel)] = int(peak_index)
            peak_value = int(82 + rng.randint(0, 12))
            curve: List[int] = []
            for index in range(len(sampled_x_values)):
                distance = abs(int(index) - int(peak_index))
                value = (
                    int(peak_value)
                    - (int(distance) * int(12 + rng.randint(0, 4)))
                    - int(rng.randint(0, 5))
                )
                curve.append(max(int(y_min) + 5, min(int(y_max) - 5, int(value))))
            curve[int(peak_index)] = int(peak_value)
            values[str(panel)][str(method_label)] = list(curve)
        peak_x_by_panel = {
            str(panel): int(sampled_x_values[int(peak_indices[str(panel)])])
            for panel in panel_labels
        }
        if min(
            peak_x_by_panel, key=lambda label: (peak_x_by_panel[label], label)
        ) != str(answer_panel):
            raise RuntimeError("earliest maximum construction lost unique target")
        query = build_curve_panel_query_record(
            prompt_key=selected_query_id,
            answer=answer_panel,
            answer_type="string",
            panel_label=answer_panel,
            method_label=method_label,
            annotation_panel_labels=(answer_panel,),
            annotation_point_ids=(
                point_id(
                    answer_panel,
                    method_label,
                    int(peak_x_by_panel[str(answer_panel)]),
                ),
            ),
            trace={
                "method_label": str(method_label),
                "peak_x_by_panel": dict(peak_x_by_panel),
                "peak_indices_by_panel": {
                    str(panel): int(index) for panel, index in peak_indices.items()
                },
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
            dynamic_slots={"method_label": f'"{query.method_label}"'},
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
            build_plan=self._build_earliest_maximum_panel_plan,
        )


__all__ = ["ChartsScientificEarliestMaximumPanelLabelTask"]
