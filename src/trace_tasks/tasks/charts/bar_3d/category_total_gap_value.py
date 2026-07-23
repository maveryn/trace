"""Public task for `task_charts__bar_3d__category_total_gap_value`."""
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.bar_3d._lifecycle import bar_3d_task_output_fields, build_bar_3d_plan, run_bar_3d_public_task
from trace_tasks.tasks.charts.bar_3d.shared.state import DOMAIN, SCENE_ID
from trace_tasks.tasks.charts.bar_3d.shared.sampling import sample_dataset_with_selection, select_axis_total_gap
from trace_tasks.tasks.charts.bar_3d.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

def _build_category_total_gap_plan(instance_seed, params, selected_query_id):
    """Build one category-total gap semantic plan bound to both selected categories."""
    dataset, ranges, selection_trace = sample_dataset_with_selection(params=params, condition_scope=False, pairwise_target_count=None, instance_seed=int(instance_seed), select=lambda x_labels, series_labels, values: select_axis_total_gap(axis='category', x_labels=x_labels, series_labels=series_labels, values=values, instance_seed=int(instance_seed)))
    prompt_artifacts = build_prompt_artifacts(prompt_query_key='category_total_gap_value', dynamic_slots={'category_label_a': f'''"{selection_trace['category_label_a']}"''', 'category_label_b': f'''"{selection_trace['category_label_b']}"'''}, instance_seed=int(instance_seed))
    return build_bar_3d_plan(dataset=dataset, ranges=ranges, selection_trace=selection_trace, prompt_artifacts=prompt_artifacts)

def _build_task_output(materialized):
    return TaskOutput(**bar_3d_task_output_fields(materialized))

@register_task
class ChartsThreeDBarCategoryTotalGapValueTask:
    """Compute the absolute gap between totals for two x-axis categories."""
    task_id = 'task_charts__bar_3d__category_total_gap_value'
    reasoning_operations = ('aggregation', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = 'category_total_gap_value'
    supported_query_ids = ('category_total_gap_value',)
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        selected_query_id, _probabilities, task_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=self.supported_query_ids, default_query_id='category_total_gap_value', task_id=self.task_id)
        return run_bar_3d_public_task(instance_seed=int(instance_seed), params=task_params, max_attempts=int(max_attempts), selected_query_id=str(selected_query_id), build_plan=_build_category_total_gap_plan, build_output=_build_task_output)
__all__ = ['ChartsThreeDBarCategoryTotalGapValueTask']
