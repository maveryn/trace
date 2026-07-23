"""Public task for `task_charts__dashboard__category_extremum_panel_label`."""

from __future__ import annotations

from typing import Any, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.dashboard._lifecycle import DashboardTaskPlan, MaterializedDashboardTask, dashboard_task_output_fields, run_dashboard_public_task
from trace_tasks.tasks.charts.dashboard.shared.defaults import generation_default
from trace_tasks.tasks.charts.dashboard.shared.metrics import category_by_id, panel_by_id
from trace_tasks.tasks.charts.dashboard.shared.prompts import build_prompt_artifacts, build_prompt_slots
from trace_tasks.tasks.charts.dashboard.shared.sampling import build_dashboard_base_sample
from trace_tasks.tasks.charts.dashboard.shared.state import DOMAIN, SCENE_ID, SCENE_VARIANT, DashboardDataset, DashboardQuery, Panel
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


LARGEST_PANEL_QUERY_ID = "largest_category_panel_label"
SMALLEST_PANEL_QUERY_ID = "smallest_category_panel_label"
DIRECTION_BY_QUERY_ID = {
    LARGEST_PANEL_QUERY_ID: "largest",
    SMALLEST_PANEL_QUERY_ID: "smallest",
}


def _build_task_output(materialized: MaterializedDashboardTask) -> TaskOutput:
    return TaskOutput(**dashboard_task_output_fields(materialized))


def _replace_panel_values_for_category(*, panels: Sequence[Panel], category_id: str, values_by_panel_id: dict[str, int]) -> tuple[Panel, ...]:
    updated_panels: list[Panel] = []
    for panel in panels:
        values = dict(panel.values_by_category_id)
        values[str(category_id)] = int(values_by_panel_id[str(panel.panel_id)])
        updated_panels.append(Panel(panel_id=str(panel.panel_id), kind=str(panel.kind), name=str(panel.name), values_by_category_id=values))
    return tuple(updated_panels)


@register_task
class ChartsDashboardCategoryExtremumPanelLabelTask:
    """Find which dashboard panel has an extremal value for one category."""

    task_id = "task_charts__dashboard__category_extremum_panel_label"
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    objective_contract = "category_extremum_panel_label"
    supported_query_ids = (LARGEST_PANEL_QUERY_ID, SMALLEST_PANEL_QUERY_ID)
    default_dataset_enabled = True

    def _construct_category_extremum_panel_plan(self, instance_seed: int, params: dict[str, Any], selected_query_id: str) -> DashboardTaskPlan:
        """Bind one target category and make the requested answer panel unique."""

        rng = spawn_rng(int(instance_seed), f"{SCENE_ID}.{self.objective_contract}.selection")
        base_sample = build_dashboard_base_sample(params, instance_seed=int(instance_seed))
        direction = DIRECTION_BY_QUERY_ID[str(selected_query_id)]
        target_category = base_sample.categories[int(rng.randrange(len(base_sample.categories)))]
        answer_panel = base_sample.panels[int(rng.randrange(len(base_sample.panels)))]
        value_min = int(params.get("value_min", generation_default("value_min", 12)))
        value_max = int(params.get("value_max", generation_default("value_max", 92)))
        span = int(value_max) - int(value_min) + 1
        if span < len(base_sample.panels):
            raise ValueError("value range is too small to assign unique target-category panel values")
        values = sorted(int(value) for value in rng.sample(range(int(value_min), int(value_max) + 1), len(base_sample.panels)))
        if str(direction) == "largest":
            answer_value = int(values[-1])
            other_values = list(values[:-1])
        else:
            answer_value = int(values[0])
            other_values = list(values[1:])
        rng.shuffle(other_values)
        values_by_panel_id: dict[str, int] = {}
        other_index = 0
        for panel in base_sample.panels:
            panel_id = str(panel.panel_id)
            if panel_id == str(answer_panel.panel_id):
                values_by_panel_id[panel_id] = int(answer_value)
            else:
                values_by_panel_id[panel_id] = int(other_values[other_index])
                other_index += 1
        panels = _replace_panel_values_for_category(
            panels=base_sample.panels,
            category_id=str(target_category.category_id),
            values_by_panel_id=values_by_panel_id,
        )
        answer_panel_record = panel_by_id(panels, str(answer_panel.panel_id))
        target_category_record = category_by_id(base_sample.categories, str(target_category.category_id))
        answer_ref = (str(answer_panel_record.panel_id), str(target_category_record.category_id))
        relations = {
            **dict(base_sample.common_params),
            "target_category_id": str(target_category_record.category_id),
            "target_category_label": str(target_category_record.label),
            "extremum_direction": str(direction),
            "answer_panel_id": str(answer_panel_record.panel_id),
            "answer_panel_name": str(answer_panel_record.name),
            "answer_value": int(answer_value),
            "category_values_by_panel_id": {str(panel_id): int(value) for panel_id, value in values_by_panel_id.items()},
            "selection_operation": "category_value_extremum_panel",
        }
        dataset = DashboardDataset(
            scene_variant=SCENE_VARIANT,
            categories=base_sample.categories,
            panels=panels,
            query=DashboardQuery(answer=str(answer_panel_record.name), answer_type="string", annotation_refs=(answer_ref,), params=dict(relations)),
        )
        prompt_artifacts = build_prompt_artifacts(prompt_query_key=str(selected_query_id), dynamic_slots=build_prompt_slots(dataset=dataset), instance_seed=int(instance_seed))
        return DashboardTaskPlan(
            dataset=dataset,
            prompt_artifacts=prompt_artifacts,
            relations=relations,
            answer_gt=TypedValue(type="string", value=str(answer_panel_record.name)),
            annotation_refs=(answer_ref,),
            annotation_type="point",
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=LARGEST_PANEL_QUERY_ID,
            task_id=self.task_id,
        )
        return run_dashboard_public_task(
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            selected_query_id=str(selected_query_id),
            build_plan=self._construct_category_extremum_panel_plan,
            build_output=_build_task_output,
        )


__all__ = ["ChartsDashboardCategoryExtremumPanelLabelTask"]
