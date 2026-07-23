"""Public task for `task_charts__dashboard__global_value_extremum_category_label`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.dashboard._lifecycle import DashboardTaskPlan, MaterializedDashboardTask, dashboard_task_output_fields, run_dashboard_public_task
from trace_tasks.tasks.charts.dashboard.shared.defaults import generation_default
from trace_tasks.tasks.charts.dashboard.shared.metrics import category_by_id, panel_by_id
from trace_tasks.tasks.charts.dashboard.shared.prompts import build_prompt_artifacts, build_prompt_slots
from trace_tasks.tasks.charts.dashboard.shared.sampling import build_dashboard_base_sample, replace_panels_by_id
from trace_tasks.tasks.charts.dashboard.shared.state import DOMAIN, SCENE_ID, SCENE_VARIANT, DashboardDataset, DashboardQuery, Panel
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


GLOBAL_MAXIMUM_QUERY_ID = "global_maximum_value_category_label"
GLOBAL_MINIMUM_QUERY_ID = "global_minimum_value_category_label"
GLOBAL_DIRECTION_BY_QUERY_ID = {
    GLOBAL_MAXIMUM_QUERY_ID: "maximum",
    GLOBAL_MINIMUM_QUERY_ID: "minimum",
}


def _build_task_output(materialized: MaterializedDashboardTask) -> TaskOutput:
    return TaskOutput(**dashboard_task_output_fields(materialized))


@register_task
class ChartsDashboardGlobalValueExtremumCategoryLabelTask:
    """Find the category label of the single global value extremum in the dashboard."""

    task_id = "task_charts__dashboard__global_value_extremum_category_label"
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    objective_contract = "global_value_extremum_category_label"
    supported_query_ids = (GLOBAL_MAXIMUM_QUERY_ID, GLOBAL_MINIMUM_QUERY_ID)
    default_dataset_enabled = True

    def _construct_global_value_extremum_plan(self, instance_seed: int, params: dict[str, Any], selected_query_id: str) -> DashboardTaskPlan:
        """Force one unique global value extremum, then answer with its category label."""

        rng = spawn_rng(int(instance_seed), f"{SCENE_ID}.{self.objective_contract}.selection")
        base_sample = build_dashboard_base_sample(params, instance_seed=int(instance_seed))
        direction = GLOBAL_DIRECTION_BY_QUERY_ID[str(selected_query_id)]
        value_min = int(params.get("value_min", generation_default("value_min", 12)))
        value_max = int(params.get("value_max", generation_default("value_max", 92)))
        if int(value_max) - int(value_min) < 1:
            raise ValueError("value_min/value_max must leave room for a unique global extremum")

        answer_panel = base_sample.panels[int(rng.randrange(len(base_sample.panels)))]
        answer_category = base_sample.categories[int(rng.randrange(len(base_sample.categories)))]
        answer_value = int(value_max if str(direction) == "maximum" else value_min)
        updated_by_panel_id: dict[str, Panel] = {}
        for panel in base_sample.panels:
            values: dict[str, int] = {}
            for category in base_sample.categories:
                is_answer_mark = str(panel.panel_id) == str(answer_panel.panel_id) and str(category.category_id) == str(answer_category.category_id)
                if is_answer_mark:
                    values[str(category.category_id)] = int(answer_value)
                elif str(direction) == "maximum":
                    values[str(category.category_id)] = int(rng.randint(int(value_min), int(value_max) - 1))
                else:
                    values[str(category.category_id)] = int(rng.randint(int(value_min) + 1, int(value_max)))
            updated_by_panel_id[str(panel.panel_id)] = Panel(
                panel_id=str(panel.panel_id),
                kind=str(panel.kind),
                name=str(panel.name),
                values_by_category_id=values,
            )

        panels = replace_panels_by_id(base_sample.panels, updated_by_panel_id)
        answer_panel_record = panel_by_id(panels, str(answer_panel.panel_id))
        answer_category_record = category_by_id(base_sample.categories, str(answer_category.category_id))
        answer_ref = (str(answer_panel_record.panel_id), str(answer_category_record.category_id))
        relations = {
            **dict(base_sample.common_params),
            "global_extremum_direction": str(direction),
            "answer_panel_id": str(answer_panel_record.panel_id),
            "answer_panel_name": str(answer_panel_record.name),
            "answer_category_id": str(answer_category_record.category_id),
            "answer_category_label": str(answer_category_record.label),
            "answer_value": int(answer_value),
            "selection_operation": "global_value_extremum_category",
        }
        dataset = DashboardDataset(
            scene_variant=SCENE_VARIANT,
            categories=base_sample.categories,
            panels=panels,
            query=DashboardQuery(answer=str(answer_category_record.label), answer_type="string", annotation_refs=(answer_ref,), params=dict(relations)),
        )
        prompt_artifacts = build_prompt_artifacts(prompt_query_key=str(selected_query_id), dynamic_slots=build_prompt_slots(dataset=dataset), instance_seed=int(instance_seed))
        return DashboardTaskPlan(
            dataset=dataset,
            prompt_artifacts=prompt_artifacts,
            relations=relations,
            answer_gt=TypedValue(type="string", value=str(answer_category_record.label)),
            annotation_refs=(answer_ref,),
            annotation_type="point",
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=GLOBAL_MAXIMUM_QUERY_ID,
            task_id=self.task_id,
        )
        return run_dashboard_public_task(
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            selected_query_id=str(selected_query_id),
            build_plan=self._construct_global_value_extremum_plan,
            build_output=_build_task_output,
        )


__all__ = ["ChartsDashboardGlobalValueExtremumCategoryLabelTask"]
