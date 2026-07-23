from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.bar_3d._lifecycle import bar_3d_task_output_fields, build_bar_3d_plan, run_bar_3d_public_task
from trace_tasks.tasks.charts.bar_3d.shared.state import DOMAIN, SCENE_ID
from trace_tasks.tasks.charts.bar_3d.shared.sampling import pairwise_target_max_category_count, sample_dataset_with_selection, sample_pairwise_target_count, select_pairwise_series_greater_count
from trace_tasks.tasks.charts.bar_3d.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

def _build_task_output(materialized):
    return TaskOutput(**bar_3d_task_output_fields(materialized))

@register_task
class ChartsThreeDBarPairwiseComparisonCountTask:
    task_id = 'task_charts__bar_3d__pairwise_comparison_count'
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    supported_query_ids = ('series_comparison_count',)
    default_dataset_enabled = True

    def _build_plan(self, instance_seed, params, selected_query_id):
        target_count = sample_pairwise_target_count(params, instance_seed=int(instance_seed), max_category_count=pairwise_target_max_category_count(params))
        dataset, ranges, selection_trace = sample_dataset_with_selection(params=params, condition_scope=True, pairwise_target_count=int(target_count), instance_seed=int(instance_seed), select=lambda x_labels, series_labels, values: select_pairwise_series_greater_count(x_labels=x_labels, series_labels=series_labels, values=values, target_count=int(target_count), instance_seed=int(instance_seed)))
        prompt_artifacts = build_prompt_artifacts(prompt_query_key='pairwise_comparison_count', dynamic_slots={'series_label_a': f'''"{selection_trace['series_label_a']}"''', 'series_label_b': f'''"{selection_trace['series_label_b']}"'''}, instance_seed=int(instance_seed))
        return build_bar_3d_plan(dataset=dataset, ranges=ranges, selection_trace=selection_trace, prompt_artifacts=prompt_artifacts)

    def generate(self, instance_seed, *, params, max_attempts):
        selected_query_id, _probabilities, task_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=self.supported_query_ids, default_query_id='series_comparison_count', task_id=self.task_id)
        return run_bar_3d_public_task(instance_seed=int(instance_seed), params=task_params, max_attempts=int(max_attempts), selected_query_id=str(selected_query_id), build_plan=self._build_plan, build_output=_build_task_output)
__all__ = ['ChartsThreeDBarPairwiseComparisonCountTask']
