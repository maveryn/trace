"""Public task for `task_charts__dashboard__panel_value_range_value`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.dashboard._lifecycle import DashboardTaskPlan, MaterializedDashboardTask, dashboard_task_output_fields, run_dashboard_public_task
from trace_tasks.tasks.charts.dashboard.shared.defaults import generation_default
from trace_tasks.tasks.charts.dashboard.shared.metrics import balanced_support_choice, category_by_id, panel_by_id, panel_value_range_support
from trace_tasks.tasks.charts.dashboard.shared.prompts import build_prompt_artifacts, build_prompt_slots
from trace_tasks.tasks.charts.dashboard.shared.sampling import build_dashboard_base_sample, make_panel_with_controlled_range, replace_panels_by_id
from trace_tasks.tasks.charts.dashboard.shared.state import DOMAIN, SCENE_ID, SCENE_VARIANT, DashboardDataset, DashboardQuery
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


PROMPT_QUERY_KEY = "panel_value_range_value"
def _build_task_output(materialized: MaterializedDashboardTask) -> TaskOutput:
    return TaskOutput(**dashboard_task_output_fields(materialized))


@register_task
class ChartsDashboardPanelValueRangeValueTask:
    """Compute the value range inside one named dashboard panel."""

    task_id = "task_charts__dashboard__panel_value_range_value"
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "panel_value_range_value"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_dataset_enabled = True

    def _construct_panel_value_range_plan(self, instance_seed: int, params: dict[str, Any], selected_query_id: str) -> DashboardTaskPlan:
        """Bind one named panel whose extrema are unique and whose range is controlled."""

        del selected_query_id
        rng = spawn_rng(int(instance_seed), f"{SCENE_ID}.{self.objective_contract}.selection")
        base_sample = build_dashboard_base_sample(params, instance_seed=int(instance_seed))
        value_min = int(params.get("value_min", generation_default("value_min", 12)))
        value_max = int(params.get("value_max", generation_default("value_max", 92)))
        support = panel_value_range_support(params, value_min=int(value_min), value_max=int(value_max))
        target_range = balanced_support_choice(params=params, instance_seed=int(instance_seed), namespace=f"{SCENE_ID}.panel_value_range_value.answer", support=support)
        selected_panel = base_sample.panels[int(rng.randrange(len(base_sample.panels)))]
        updated_panel, extrema = make_panel_with_controlled_range(
            rng,
            panel=selected_panel,
            categories=base_sample.categories,
            value_min=int(value_min),
            value_max=int(value_max),
            target_range=int(target_range),
        )
        panels = replace_panels_by_id(base_sample.panels, {str(updated_panel.panel_id): updated_panel})
        largest_ref = (str(updated_panel.panel_id), str(extrema["largest_category_id"]))
        smallest_ref = (str(updated_panel.panel_id), str(extrema["smallest_category_id"]))
        largest_category = category_by_id(base_sample.categories, str(extrema["largest_category_id"]))
        smallest_category = category_by_id(base_sample.categories, str(extrema["smallest_category_id"]))
        selected_panel_record = panel_by_id(panels, str(updated_panel.panel_id))
        relations = {
            **dict(base_sample.common_params),
            "selected_panel_id": str(updated_panel.panel_id),
            "selected_panel_name": str(selected_panel_record.name),
            "largest_category_id": str(largest_category.category_id),
            "largest_category_label": str(largest_category.label),
            "largest_value": int(extrema["largest_value"]),
            "smallest_category_id": str(smallest_category.category_id),
            "smallest_category_label": str(smallest_category.label),
            "smallest_value": int(extrema["smallest_value"]),
            "range_value": int(target_range),
            "target_range_support": list(support),
            "range_operation": "panel_max_minus_min",
        }
        refs = (largest_ref, smallest_ref)
        dataset = DashboardDataset(
            scene_variant=SCENE_VARIANT,
            categories=base_sample.categories,
            panels=panels,
            query=DashboardQuery(answer=int(target_range), answer_type="integer", annotation_refs=refs, params=dict(relations)),
        )
        prompt_artifacts = build_prompt_artifacts(prompt_query_key=PROMPT_QUERY_KEY, dynamic_slots=build_prompt_slots(dataset=dataset), instance_seed=int(instance_seed))
        return DashboardTaskPlan(
            dataset=dataset,
            prompt_artifacts=prompt_artifacts,
            relations=relations,
            answer_gt=TypedValue(type="integer", value=int(target_range)),
            annotation_refs=refs,
            annotation_roles={"largest_value": largest_ref, "smallest_value": smallest_ref},
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=SINGLE_QUERY_ID,
            task_id=self.task_id,
        )
        return run_dashboard_public_task(
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            selected_query_id=str(selected_query_id),
            build_plan=self._construct_panel_value_range_plan,
            build_output=_build_task_output,
        )


__all__ = ["ChartsDashboardPanelValueRangeValueTask"]
