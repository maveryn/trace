from __future__ import annotations

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.heatmap._lifecycle import package_heatmap_plan, run_heatmap_plan
from trace_tasks.tasks.charts.heatmap.shared.data import construct_colorbar_interval_sample
from trace_tasks.tasks.charts.heatmap.shared.defaults import DOMAIN
from trace_tasks.tasks.charts.heatmap.shared.output import continuous_colorbar_relation_params
from trace_tasks.tasks.registry import register_task


QUERY_IDS = ("single",)
DEFAULT_QUERY_ID = "single"


@register_task
class ChartsHeatmapColorbarIntervalCellCountTask:
    task_id = "task_charts__heatmap__colorbar_interval_cell_count"
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = "colorbar_interval_cell_count"
    supported_query_ids = QUERY_IDS
    default_dataset_enabled = True

    def _build_plan(self, instance_seed, *, params, selected_query_id):
        if str(selected_query_id) != DEFAULT_QUERY_ID:
            raise ValueError(f"unsupported query_id: {selected_query_id}")
        if params.get("scene_variant") is not None and str(params["scene_variant"]) != "continuous_colorbar_heatmap":
            raise ValueError("colorbar interval heatmap tasks require scene_variant='continuous_colorbar_heatmap'")
        dataset = construct_colorbar_interval_sample(
            prompt_key=self.objective_contract,
            params=dict(params),
            instance_seed=int(instance_seed),
        )
        answer_gt = TypedValue(type="integer", value=int(dataset["answer_value"]))
        relation_params = continuous_colorbar_relation_params(dataset=dataset)
        return package_heatmap_plan(
            dataset=dataset,
            params=dict(params),
            answer_gt=answer_gt,
            prompt_query_key=self.objective_contract,
            supports_unanswerable=False,
            relation_params=relation_params,
            instance_seed=int(instance_seed),
        )

    def generate(self, instance_seed, *, params, max_attempts):
        return run_heatmap_plan(
            task_id=self.task_id,
            supported_query_ids=self.supported_query_ids,
            default_query_id=DEFAULT_QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            build_plan=self._build_plan,
        )
