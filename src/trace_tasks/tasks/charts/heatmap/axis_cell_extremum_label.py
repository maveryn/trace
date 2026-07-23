from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.heatmap._lifecycle import package_heatmap_plan, run_heatmap_plan
from trace_tasks.tasks.charts.heatmap.shared.data import construct_axis_cell_sample
from trace_tasks.tasks.charts.heatmap.shared.defaults import DOMAIN
from trace_tasks.tasks.charts.heatmap.shared.output import axis_cell_relation_params
from trace_tasks.tasks.charts.heatmap.shared.sampling import _resolve_scene_variant, require_discrete_heatmap
from trace_tasks.tasks.registry import register_task


QUERY_BINDINGS = {
    "row_hottest_column_label": ("row", "hottest"),
    "row_coolest_column_label": ("row", "coolest"),
    "column_hottest_row_label": ("column", "hottest"),
    "column_coolest_row_label": ("column", "coolest"),
}


@register_task
class ChartsHeatmapAxisCellExtremumLabelTask:
    task_id = "task_charts__heatmap__axis_cell_extremum_label"
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    objective_contract = "axis_cell_extremum_label"
    supported_query_ids = tuple(QUERY_BINDINGS)
    default_dataset_enabled = True
    supports_unanswerable = False

    def _build_plan(self, instance_seed, *, params, selected_query_id):
        if str(selected_query_id) not in QUERY_BINDINGS:
            raise ValueError(f"unsupported query_id: {selected_query_id}")
        seed = int(instance_seed)
        task_params = dict(params)
        query_axis, extremum_direction = QUERY_BINDINGS[str(selected_query_id)]
        require_discrete_heatmap(task_params, task_label="axis-cell")
        scene_variant, scene_variant_probabilities = _resolve_scene_variant(task_params, instance_seed=seed)
        dataset = construct_axis_cell_sample(
            prompt_key=self.objective_contract,
            scene_variant=str(scene_variant),
            query_axis=str(query_axis),
            extremum_direction=str(extremum_direction),
            params=task_params,
            instance_seed=seed,
            allow_unanswerable=False,
        )
        answer_gt = TypedValue(type=str(dataset["answer_type"]), value=str(dataset["answer_value"]))
        relation_params = axis_cell_relation_params(
            dataset=dataset,
            scene_variant_probabilities=scene_variant_probabilities,
            query_axis=str(query_axis),
            extremum_direction=str(extremum_direction),
        )
        return package_heatmap_plan(
            dataset=dataset,
            params=task_params,
            answer_gt=answer_gt,
            annotation_type="bbox",
            prompt_query_key=self.objective_contract,
            supports_unanswerable=False,
            relation_params=relation_params,
            instance_seed=seed,
        )

    def generate(self, instance_seed, *, params, max_attempts):
        return run_heatmap_plan(
            task_id=self.task_id,
            supported_query_ids=self.supported_query_ids,
            default_query_id="row_hottest_column_label",
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            build_plan=self._build_plan,
        )
