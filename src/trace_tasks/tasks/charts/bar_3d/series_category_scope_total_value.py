"""Public task for `task_charts__bar_3d__series_category_scope_total_value`."""
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.bar_3d._lifecycle import bar_3d_task_output_fields, build_bar_3d_plan, run_bar_3d_public_task
from trace_tasks.tasks.charts.bar_3d.shared.state import DOMAIN, SCENE_ID
from trace_tasks.tasks.charts.bar_3d.shared.sampling import interval_scope_params, sample_dataset_with_selection, select_series_scoped_total
from trace_tasks.tasks.charts.bar_3d.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

def _build_task_output(materialized):
    return TaskOutput(**bar_3d_task_output_fields(materialized))
ALL_CATEGORIES_QUERY_ID = 'series_total_value'
INTERVAL_QUERY_ID = 'series_interval_total_value'

@register_task
class ChartsThreeDBarSeriesCategoryScopeTotalValueTask:
    """Compute a total for one series over all or a contiguous subset of categories."""
    task_id = 'task_charts__bar_3d__series_category_scope_total_value'
    reasoning_operations = ('aggregation',)
    domain = DOMAIN
    objective_contract = 'series_category_scope_total_value'
    supported_query_ids = (ALL_CATEGORIES_QUERY_ID, INTERVAL_QUERY_ID)
    default_dataset_enabled = True

    def _build_plan(self, instance_seed, params, selected_query_id):
        """Build one series-scope total plan for the selected all-category or interval branch."""
        resolved_params = interval_scope_params(params)
        dataset, ranges, selection_trace = sample_dataset_with_selection(params=resolved_params, condition_scope=False, pairwise_target_count=None, instance_seed=int(instance_seed), select=lambda x_labels, series_labels, values: select_series_scoped_total(interval_scope=str(selected_query_id) == INTERVAL_QUERY_ID, x_labels=x_labels, series_labels=series_labels, values=values, params=resolved_params, instance_seed=int(instance_seed)))
        dynamic_slots = {'series_label': f'''"{selection_trace['series_label']}"'''}
        if str(selected_query_id) == INTERVAL_QUERY_ID:
            dynamic_slots.update({'start_category_label': f'''"{selection_trace['start_category_label']}"''', 'end_category_label': f'''"{selection_trace['end_category_label']}"'''})
        prompt_artifacts = build_prompt_artifacts(prompt_query_key=str(selected_query_id), dynamic_slots=dynamic_slots, instance_seed=int(instance_seed))
        return build_bar_3d_plan(dataset=dataset, ranges=ranges, selection_trace=selection_trace, prompt_artifacts=prompt_artifacts)

    def generate(self, instance_seed, *, params, max_attempts):
        selected_query_id, _probabilities, task_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=self.supported_query_ids, default_query_id=ALL_CATEGORIES_QUERY_ID, task_id=self.task_id)
        return run_bar_3d_public_task(instance_seed=int(instance_seed), params=task_params, max_attempts=int(max_attempts), selected_query_id=str(selected_query_id), build_plan=self._build_plan, build_output=_build_task_output)
__all__ = ['ChartsThreeDBarSeriesCategoryScopeTotalValueTask']
