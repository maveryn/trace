from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.heatmap._lifecycle import package_heatmap_plan, run_heatmap_plan
from trace_tasks.tasks.charts.heatmap.shared.data import construct_axis_condition_sample
from trace_tasks.tasks.charts.heatmap.shared.defaults import DOMAIN
from trace_tasks.tasks.charts.heatmap.shared.output import condition_relation_params
from trace_tasks.tasks.charts.heatmap.shared.sampling import require_discrete_heatmap, resolve_scene_condition_context
from trace_tasks.tasks.registry import register_task


QUERY_AXIS_BY_ID = {
    "row_condition_extremum_label": "row",
    "column_condition_extremum_label": "column",
}


@register_task
class ChartsHeatmapAxisConditionExtremumLabelTask:
    task_id = "task_charts__heatmap__axis_condition_extremum_label"
    reasoning_operations = ('filtering', 'counting', 'ranking')
    domain = DOMAIN
    objective_contract = "axis_condition_extremum_label"
    supported_query_ids = tuple(QUERY_AXIS_BY_ID)
    default_dataset_enabled = True
    supports_unanswerable = False

    def _build_plan(self, instance_seed, *, params, selected_query_id):
        if str(selected_query_id) not in QUERY_AXIS_BY_ID:
            raise ValueError(f"unsupported query_id: {selected_query_id}")
        seed = int(instance_seed)
        task_params = dict(params)
        query_axis = str(QUERY_AXIS_BY_ID[str(selected_query_id)])
        require_discrete_heatmap(task_params, task_label="axis-condition")
        scene_variant, scene_variant_probabilities, condition_kind, condition_probabilities = (
            resolve_scene_condition_context(task_params, instance_seed=seed)
        )
        dataset = construct_axis_condition_sample(
            prompt_key=self.objective_contract,
            scene_variant=str(scene_variant),
            query_axis=str(query_axis),
            condition_kind=str(condition_kind),
            params=task_params,
            instance_seed=seed,
            allow_unanswerable=False,
        )
        answer_gt = TypedValue(type=str(dataset["answer_type"]), value=str(dataset["answer_value"]))
        relation_params = condition_relation_params(
            dataset=dataset,
            scene_variant_probabilities=scene_variant_probabilities,
            condition_kind_probabilities=condition_probabilities,
            query_axis_probabilities={str(query_axis): 1.0},
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
            default_query_id="row_condition_extremum_label",
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            build_plan=self._build_plan,
        )
