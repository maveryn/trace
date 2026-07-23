"""Public task for `task_charts__bar_3d__category_extremum_gap_value`."""
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.bar_3d._lifecycle import bar_3d_task_output_fields, build_bar_3d_plan, run_bar_3d_public_task
from trace_tasks.tasks.charts.bar_3d.shared.state import DOMAIN, SCENE_ID
from trace_tasks.tasks.charts.bar_3d.shared.sampling import sample_dataset_with_selection, select_category_extremum_gap
from trace_tasks.tasks.charts.bar_3d.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

def _build_task_output(materialized):
    return TaskOutput(**bar_3d_task_output_fields(materialized))

@register_task
class ChartsThreeDBarCategoryExtremumGapValueTask:
    """Compute the highest-minus-lowest series-bar gap within one category."""
    task_id = 'task_charts__bar_3d__category_extremum_gap_value'
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = 'category_extremum_gap_value'
    supported_query_ids = ('category_extremum_gap_value',)
    default_dataset_enabled = True

    def _build_plan(self, instance_seed, params, selected_query_id):
        """Build one extrema-gap semantic plan bound to the selected category bars."""
        dataset, ranges, selection_trace = sample_dataset_with_selection(params=params, condition_scope=False, pairwise_target_count=None, instance_seed=int(instance_seed), select=lambda x_labels, series_labels, values: select_category_extremum_gap(x_labels=x_labels, series_labels=series_labels, values=values, instance_seed=int(instance_seed)))
        prompt_artifacts = build_prompt_artifacts(prompt_query_key=self.objective_contract, dynamic_slots={'category_label': f'''"{selection_trace['category_label']}"'''}, instance_seed=int(instance_seed))
        return build_bar_3d_plan(dataset=dataset, ranges=ranges, selection_trace=selection_trace, prompt_artifacts=prompt_artifacts)

    def generate(self, instance_seed, *, params, max_attempts):
        selected_query_id, _probabilities, task_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=self.supported_query_ids, default_query_id='category_extremum_gap_value', task_id=self.task_id)
        return run_bar_3d_public_task(instance_seed=int(instance_seed), params=task_params, max_attempts=int(max_attempts), selected_query_id=str(selected_query_id), build_plan=self._build_plan, build_output=_build_task_output)
__all__ = ['ChartsThreeDBarCategoryExtremumGapValueTask']
