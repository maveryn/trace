"""Public task for `task_charts__dashboard__category_panel_condition_count`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.dashboard._lifecycle import DashboardTaskPlan, MaterializedDashboardTask, dashboard_task_output_fields, run_dashboard_public_task
from trace_tasks.tasks.charts.dashboard.shared.prompts import build_prompt_artifacts, build_prompt_slots
from trace_tasks.tasks.charts.dashboard.shared.sampling import build_dashboard_base_sample
from trace_tasks.tasks.charts.dashboard.shared.state import DOMAIN, SCENE_ID, SCENE_VARIANT, DashboardDataset, DashboardQuery
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


def _build_task_output(materialized: MaterializedDashboardTask) -> TaskOutput:
    return TaskOutput(**dashboard_task_output_fields(materialized))
from trace_tasks.tasks.charts.dashboard.shared.defaults import generation_default
from trace_tasks.tasks.charts.dashboard.shared.metrics import balanced_support_choice, category_by_id, compare_condition, condition_phrase, panel_by_id, panel_condition_count_support


GREATER_THAN_QUERY_ID = "category_panel_greater_than_threshold_count"
LESS_THAN_QUERY_ID = "category_panel_less_than_threshold_count"
COMPARISON_BY_QUERY_ID = {
    GREATER_THAN_QUERY_ID: "greater_than",
    LESS_THAN_QUERY_ID: "less_than",
}


@dataclass(frozen=True)
class PanelConditionSelection:
    """Local selection record for a category scoped across dashboard panels."""

    category_id: str
    threshold: int
    matching_panel_ids: tuple[str, ...]


@register_task
class ChartsDashboardCategoryPanelConditionCountTask:
    """Count dashboard panels where one shared category satisfies a value threshold."""

    task_id = "task_charts__dashboard__category_panel_condition_count"
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = "category_panel_condition_count"
    supported_query_ids = (GREATER_THAN_QUERY_ID, LESS_THAN_QUERY_ID)
    default_dataset_enabled = True

    def _construct_category_panel_count_plan(self, instance_seed: int, params: dict[str, Any], selected_query_id: str) -> DashboardTaskPlan:
        """Bind a single category and threshold whose matching panel count is controlled."""
        rng = spawn_rng(int(instance_seed), f"{SCENE_ID}.{self.objective_contract}.selection")
        base_sample = build_dashboard_base_sample(params, instance_seed=int(instance_seed))
        comparison = COMPARISON_BY_QUERY_ID[str(selected_query_id)]
        support = panel_condition_count_support(params, len(base_sample.panels))
        target_count = balanced_support_choice(params=params, instance_seed=int(instance_seed), namespace=f"{SCENE_ID}.category_panel_condition_count.answer", support=support)
        value_min = int(params.get("value_min", generation_default("value_min", 12)))
        value_max = int(params.get("value_max", generation_default("value_max", 92)))
        candidates: list[tuple[str, int, tuple[str, ...]]] = []
        for category in base_sample.categories:
            category_id = str(category.category_id)
            for threshold in range(int(value_min) + 2, int(value_max) - 1):
                matches = tuple(str(panel.panel_id) for panel in base_sample.panels if compare_condition(int(panel.values_by_category_id[str(category_id)]), str(comparison), int(threshold)))
                if len(matches) == int(target_count):
                    candidates.append((str(category_id), int(threshold), matches))
        if not candidates:
            raise ValueError("no dashboard category/threshold realizes requested panel-condition count")
        category_id, threshold, matches = candidates[int(rng.randrange(len(candidates)))]
        category = category_by_id(base_sample.categories, category_id)
        panel_order = {str(panel.panel_id): index for index, panel in enumerate(base_sample.panels)}
        matching_panel_ids = tuple(sorted(matches, key=lambda panel_id: panel_order[str(panel_id)]))
        selection = PanelConditionSelection(
            category_id=str(category_id),
            threshold=int(threshold),
            matching_panel_ids=tuple(str(panel_id) for panel_id in matching_panel_ids),
        )
        refs = tuple((str(panel_id), str(selection.category_id)) for panel_id in selection.matching_panel_ids)
        relations = {
            **dict(base_sample.common_params),
            "condition_category_id": str(selection.category_id),
            "condition_category_label": str(category.label),
            "panel_condition_comparison": str(comparison),
            "panel_threshold": int(selection.threshold),
            "panel_condition_phrase": condition_phrase(str(comparison), int(selection.threshold)),
            "matching_panel_ids": list(selection.matching_panel_ids),
            "matching_panel_names": [str(panel_by_id(base_sample.panels, panel_id).name) for panel_id in selection.matching_panel_ids],
            "target_count_support": list(support),
            "count_value": int(len(selection.matching_panel_ids)),
            "predicate_scope": "one_category_across_panels",
        }
        dataset = DashboardDataset(scene_variant=SCENE_VARIANT, categories=base_sample.categories, panels=base_sample.panels, query=DashboardQuery(answer=int(len(selection.matching_panel_ids)), answer_type="integer", annotation_refs=refs, params=dict(relations)))
        prompt_artifacts = build_prompt_artifacts(prompt_query_key=str(selected_query_id), dynamic_slots=build_prompt_slots(dataset=dataset), instance_seed=int(instance_seed))
        return DashboardTaskPlan(dataset=dataset, prompt_artifacts=prompt_artifacts, relations=relations, answer_gt=TypedValue(type="integer", value=int(len(selection.matching_panel_ids))), annotation_refs=refs)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, task_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=self.supported_query_ids, default_query_id=GREATER_THAN_QUERY_ID, task_id=self.task_id)
        return run_dashboard_public_task(instance_seed=int(instance_seed), params=task_params, max_attempts=int(max_attempts), selected_query_id=str(selected_query_id), build_plan=self._construct_category_panel_count_plan, build_output=_build_task_output)


__all__ = ["ChartsDashboardCategoryPanelConditionCountTask"]
