"""Public task for `task_charts__bar_3d__series_threshold_count`."""
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.bar_3d._lifecycle import bar_3d_task_output_fields, build_bar_3d_plan, run_bar_3d_public_task
from trace_tasks.tasks.charts.bar_3d.shared.state import DOMAIN, SCENE_ID
from trace_tasks.tasks.charts.bar_3d.shared.sampling import condition_axis_params, sample_dataset_with_selection, select_axis_threshold_count
from trace_tasks.tasks.charts.bar_3d.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

def _build_series_threshold_plan(instance_seed, params, selected_query_id):
    """Build one series-threshold count plan with a nontrivial target count."""
    resolved_params = condition_axis_params(params, axis='series')
    dataset, ranges, selection_trace = sample_dataset_with_selection(params=resolved_params, condition_scope=True, pairwise_target_count=None, instance_seed=int(instance_seed), select=lambda x_labels, series_labels, values: select_axis_threshold_count(axis='series', comparison_phrase='at least', want_at_least=True, x_labels=x_labels, series_labels=series_labels, values=values, params=resolved_params, instance_seed=int(instance_seed)))
    prompt_artifacts = build_prompt_artifacts(prompt_query_key='series_threshold_count', dynamic_slots={'series_label': f'''"{selection_trace['series_label']}"''', 'comparison_phrase': str(selection_trace['comparison_phrase']), 'threshold': str(selection_trace['threshold'])}, instance_seed=int(instance_seed))
    return build_bar_3d_plan(dataset=dataset, ranges=ranges, selection_trace=selection_trace, prompt_artifacts=prompt_artifacts)

def _build_task_output(materialized):
    return TaskOutput(**bar_3d_task_output_fields(materialized))

@register_task
class ChartsThreeDBarSeriesThresholdCountTask:
    """Count categories within one series whose bars satisfy a threshold."""
    task_id = 'task_charts__bar_3d__series_threshold_count'
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = 'series_threshold_count'
    supported_query_ids = ('series_threshold_count',)
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        selected_query_id, _probabilities, task_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=self.supported_query_ids, default_query_id='series_threshold_count', task_id=self.task_id)
        return run_bar_3d_public_task(instance_seed=int(instance_seed), params=task_params, max_attempts=int(max_attempts), selected_query_id=str(selected_query_id), build_plan=_build_series_threshold_plan, build_output=_build_task_output)
__all__ = ['ChartsThreeDBarSeriesThresholdCountTask']
