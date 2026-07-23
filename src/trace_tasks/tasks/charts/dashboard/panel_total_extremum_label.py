"""Public task for `task_charts__dashboard__panel_total_extremum_label`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.dashboard._lifecycle import (
    DashboardTaskPlan,
    MaterializedDashboardTask,
    dashboard_task_output_fields,
    dashboard_total_extremum_plan_from_sample,
    run_dashboard_public_task,
)
from trace_tasks.tasks.charts.dashboard.shared.defaults import generation_default
from trace_tasks.tasks.charts.dashboard.shared.sampling import DashboardTotalExtremumSample, build_dashboard_base_sample, make_panel_total_extremum_sample
from trace_tasks.tasks.charts.dashboard.shared.state import DOMAIN, SCENE_ID, Panel
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


LARGEST_PANEL_TOTAL_QUERY_ID = "largest_panel_total_label"
SMALLEST_PANEL_TOTAL_QUERY_ID = "smallest_panel_total_label"
TOTAL_DIRECTION_BY_QUERY_ID = {
    LARGEST_PANEL_TOTAL_QUERY_ID: "largest",
    SMALLEST_PANEL_TOTAL_QUERY_ID: "smallest",
}


def _build_task_output(materialized: MaterializedDashboardTask) -> TaskOutput:
    return TaskOutput(**dashboard_task_output_fields(materialized))


def _validate_panel_total_sample(*, panels: tuple[Panel, ...], sample: DashboardTotalExtremumSample, direction: str) -> None:
    """Check that the selected panel is the unique requested total extremum."""

    totals_by_panel_id: dict[str, int] = {}
    for panel in panels:
        panel_id = str(panel.panel_id)
        totals_by_panel_id[panel_id] = sum(int(value) for value in panel.values_by_category_id.values())
    if totals_by_panel_id != {str(key): int(value) for key, value in sample.totals_by_id.items()}:
        raise ValueError("panel total sample metadata does not match generated values")
    selected_total = int(totals_by_panel_id[str(sample.answer_id)])
    other_totals = [int(value) for panel_id, value in totals_by_panel_id.items() if str(panel_id) != str(sample.answer_id)]
    if str(direction) == "largest" and not all(int(selected_total) > int(value) for value in other_totals):
        raise ValueError("selected panel is not the unique largest total")
    if str(direction) == "smallest" and not all(int(selected_total) < int(value) for value in other_totals):
        raise ValueError("selected panel is not the unique smallest total")


@register_task
class ChartsDashboardPanelTotalExtremumLabelTask:
    """Find which dashboard panel has the largest or smallest category total."""

    task_id = "task_charts__dashboard__panel_total_extremum_label"
    reasoning_operations = ('ranking', 'aggregation')
    domain = DOMAIN
    objective_contract = "panel_total_extremum_label"
    supported_query_ids = (LARGEST_PANEL_TOTAL_QUERY_ID, SMALLEST_PANEL_TOTAL_QUERY_ID)
    default_dataset_enabled = True

    def _construct_panel_total_extremum_plan(self, instance_seed: int, params: dict[str, Any], selected_query_id: str) -> DashboardTaskPlan:
        """Choose the answer panel by a unique sum across all categories."""

        rng = spawn_rng(int(instance_seed), f"{SCENE_ID}.{self.objective_contract}.selection")
        base_sample = build_dashboard_base_sample(params, instance_seed=int(instance_seed))
        direction = TOTAL_DIRECTION_BY_QUERY_ID[str(selected_query_id)]
        value_min = int(params.get("value_min", generation_default("value_min", 12)))
        value_max = int(params.get("value_max", generation_default("value_max", 92)))
        answer_panel = base_sample.panels[int(rng.randrange(len(base_sample.panels)))]
        total_sample = make_panel_total_extremum_sample(
            rng,
            panels=base_sample.panels,
            categories=base_sample.categories,
            answer_panel_id=str(answer_panel.panel_id),
            direction=str(direction),
            value_min=int(value_min),
            value_max=int(value_max),
        )
        _validate_panel_total_sample(panels=total_sample.panels, sample=total_sample, direction=str(direction))
        relations = {
            **dict(base_sample.common_params),
            "panel_total_extremum_direction": str(direction),
            "answer_panel_id": str(total_sample.answer_id),
            "answer_panel_name": str(total_sample.answer_label),
            "answer_panel_total": int(total_sample.answer_total),
            "panel_total_by_panel_id": dict(total_sample.totals_by_id),
            "selection_operation": "panel_total_extremum",
            "sum_operation": "sum_categories_within_panel",
        }
        return dashboard_total_extremum_plan_from_sample(
            categories=base_sample.categories,
            total_sample=total_sample,
            relations=relations,
            prompt_query_key=str(selected_query_id),
            instance_seed=int(instance_seed),
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=LARGEST_PANEL_TOTAL_QUERY_ID,
            task_id=self.task_id,
        )
        return run_dashboard_public_task(
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            selected_query_id=str(selected_query_id),
            build_plan=self._construct_panel_total_extremum_plan,
            build_output=_build_task_output,
        )


__all__ = ["ChartsDashboardPanelTotalExtremumLabelTask"]
