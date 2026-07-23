from __future__ import annotations

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.heatmap._lifecycle import package_heatmap_plan, run_heatmap_plan
from trace_tasks.tasks.charts.heatmap.shared.data import construct_condition_run_sample
from trace_tasks.tasks.charts.heatmap.shared.defaults import DOMAIN
from trace_tasks.tasks.charts.heatmap.shared.output import condition_relation_params
from trace_tasks.tasks.charts.heatmap.shared.sampling import require_discrete_heatmap, resolve_scene_condition_context
from trace_tasks.tasks.registry import register_task


QUERY_IDS = ("single",)
DEFAULT_QUERY_ID = "single"


@register_task
class ChartsHeatmapConditionRunExtremumLabelTask:
    task_id = "task_charts__heatmap__condition_run_extremum_label"
    reasoning_operations = ('filtering', 'ranking')
    domain = DOMAIN
    objective_contract = "condition_run_extremum_label"
    supported_query_ids = QUERY_IDS
    default_dataset_enabled = True

    def _build_plan(self, instance_seed, *, params, selected_query_id):
        if str(selected_query_id) != DEFAULT_QUERY_ID:
            raise ValueError(f"unsupported query_id: {selected_query_id}")
        seed = int(instance_seed)
        task_params = dict(params)
        require_discrete_heatmap(task_params, task_label="condition-run")
        scene_variant, scene_variant_probabilities, condition_kind, condition_probabilities = (
            resolve_scene_condition_context(task_params, instance_seed=seed)
        )
        dataset = construct_condition_run_sample(
            prompt_key=self.objective_contract,
            scene_variant=str(scene_variant),
            condition_kind=str(condition_kind),
            params=task_params,
            instance_seed=seed,
        )
        answer_gt = TypedValue(type=str(dataset["answer_type"]), value=str(dataset["answer_value"]))
        relation_params = condition_relation_params(
            dataset=dataset,
            scene_variant_probabilities=scene_variant_probabilities,
            condition_kind_probabilities=condition_probabilities,
        )
        return package_heatmap_plan(
            dataset=dataset,
            params=task_params,
            answer_gt=answer_gt,
            prompt_query_key=self.objective_contract,
            supports_unanswerable=False,
            relation_params=relation_params,
            instance_seed=seed,
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
