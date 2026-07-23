"""Public task for `task_charts__annotated_series__callout_endpoint_change_value`."""
from __future__ import annotations
from typing import Any, Dict
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.annotated_series.shared.annotations import keyed_point_artifacts, keyed_points_for_labels
from trace_tasks.tasks.charts.annotated_series.shared.defaults import DOMAIN, SCENE_ID, generation_int
from trace_tasks.tasks.charts.annotated_series.shared.prompts import TASK_PROMPT_KEY, build_prompt_artifacts
from trace_tasks.tasks.charts.annotated_series.shared.rendering import draw_callout_markup, finish_rendered_image, render_base_series
from trace_tasks.tasks.charts.annotated_series.shared.sampling import build_series_sample, choose_semantic_branch
from trace_tasks.tasks.charts.annotated_series.shared.output import build_trace_payload_scaffold, mark_count_range_for_params
from trace_tasks.tasks.charts.shared.labeled_chart_values import balanced_choice_from_values
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
CALLOUT_ENDPOINT_QUERY_ID = 'callout_endpoint_change_value'
ENDPOINT_SIDES = ('first', 'last')

@register_task
class ChartsAnnotatedSeriesCalloutEndpointChangeValueTask:
    """Compute the absolute value change between a callout mark and one endpoint."""
    task_id = 'task_charts__annotated_series__callout_endpoint_change_value'
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    objective_contract = 'callout_endpoint_change_value'
    supported_query_ids = (CALLOUT_ENDPOINT_QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate the endpoint-change objective from one annotated chart.

        The public task owns endpoint/anchor selection, answer binding, and
        keyed annotation roles; shared helpers only build the series, render
        chart markup, and project already-selected labels.
        """
        del max_attempts
        selected_branch, _query_id_probabilities, task_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=self.supported_query_ids, default_query_id=CALLOUT_ENDPOINT_QUERY_ID, task_id=self.task_id)
        sample = build_series_sample(task_params, instance_seed=int(instance_seed))
        endpoint_side, endpoint_side_probabilities = choose_semantic_branch(task_params, support=ENDPOINT_SIDES, branch_name='endpoint_side', instance_seed=int(instance_seed))
        endpoint_index = 0 if endpoint_side == 'first' else len(sample.labels) - 1
        min_gap = max(2, generation_int(task_params, 'callout_gap_min', 3))
        max_gap = max(int(min_gap), generation_int(task_params, 'callout_gap_max', 9))
        feasible_indices = [int(index) for index in range(1, len(sample.labels) - 1) if int(index) != int(endpoint_index) and int(min_gap) <= abs(int(index) - int(endpoint_index)) <= int(max_gap)]
        if not feasible_indices:
            feasible_indices = [int(index) for index in range(1, len(sample.labels) - 1) if int(index) != int(endpoint_index)]
        anchor_index = balanced_choice_from_values(feasible_indices, params=task_params, instance_seed=int(instance_seed), namespace=f'{self.task_id}.anchor_index')
        anchor_label = str(sample.labels[int(anchor_index)])
        endpoint_label = str(sample.labels[int(endpoint_index)])
        answer_value = abs(int(sample.values[int(anchor_index)]) - int(sample.values[int(endpoint_index)]))
        base = render_base_series(sample, params=task_params, instance_seed=int(instance_seed))
        markup = draw_callout_markup(base=base, anchor_label=anchor_label, endpoint_label=endpoint_label, params=task_params, instance_seed=int(instance_seed))
        final = finish_rendered_image(markup, base=base, params=task_params, instance_seed=int(instance_seed))
        role_to_label = {'callout_mark': anchor_label, 'endpoint_mark': endpoint_label}
        keyed_points = keyed_points_for_labels(base.rendered_scene, role_to_label)
        annotation_gt, witness_symbolic, projected_annotation = keyed_point_artifacts(keyed_points, role_to_label)
        prompt_artifacts = build_prompt_artifacts(domain=self.domain, scene_variant=sample.scene_variant, prompt_task_key=TASK_PROMPT_KEY, prompt_query_key=str(selected_branch), dynamic_slots={'endpoint_label': f'"{endpoint_label}"'}, instance_seed=int(instance_seed))
        mark_count_range = mark_count_range_for_params(task_params)
        semantic_params = {'endpoint_side': str(endpoint_side), 'endpoint_side_probabilities': dict(endpoint_side_probabilities), 'anchor_label': str(anchor_label), 'endpoint_label': str(endpoint_label), 'anchor_index': int(anchor_index), 'endpoint_index': int(endpoint_index)}
        trace_payload, prompt_fields = build_trace_payload_scaffold(sample=sample, base=base, markup=markup, final=final, answer_value=int(answer_value), annotation_kind='callout', annotation_labels=(str(anchor_label), str(endpoint_label)), mark_count_range=mark_count_range, question_format='numeric_open', semantic_params=semantic_params, witness_symbolic=witness_symbolic, projected_annotation=projected_annotation)
        trace_payload['scene_ir']['relations']['query_id'] = str(selected_branch)
        trace_payload['query_spec'] = build_prompt_query_spec(prompt_artifacts=prompt_artifacts, query_id=str(selected_branch), params=prompt_fields)
        trace_payload['execution_trace']['query_id'] = str(selected_branch)
        return TaskOutput(prompt=str(prompt_artifacts.prompt), answer_gt=TypedValue(type='integer', value=int(answer_value)), annotation_gt=annotation_gt, image=final.image, image_id='img0', trace_payload=trace_payload, task_versions=default_task_versions(), scene_id=SCENE_ID, query_id=str(selected_branch), prompt_variants=dict(prompt_artifacts.prompt_variants))
__all__ = ['ChartsAnnotatedSeriesCalloutEndpointChangeValueTask']
