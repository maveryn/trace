from __future__ import annotations
from dataclasses import replace
from ...registry import register_task
from ._lifecycle import NodeLinkObjectivePlan, run_node_link_plan
from .shared.sampling import graph_label_sort_key
from .shared.sampling import sample_degree_count_graph
TASK_ID = 'task_graph__node_link__degree_after_removal_filter_count'
SCENE_ID = 'node_link'
SUPPORTED_QUERY_IDS = ('undirected_degree_one_filter_remaining_count', 'directed_in_degree_one_filter_remaining_count', 'directed_out_degree_one_filter_remaining_count')

def _sample_graph(rng, axes, attempts):
    directed = str(axes.query_id).startswith('directed')
    degree_mode = 'in_degree' if 'in_degree' in str(axes.query_id) else 'out_degree' if 'out_degree' in str(axes.query_id) else None
    remaining_count = int(axes.values['target_count'])
    excluded_count = max(1, min(4, 2 if remaining_count >= 3 else 5 - remaining_count))
    node_count = max(5, int(remaining_count) + int(excluded_count))
    excluded_count = int(node_count) - int(remaining_count)
    sample = sample_degree_count_graph(rng, query_id='directed_degree_count' if directed else 'degree_count', degree_mode=degree_mode, node_count=int(node_count), query_degree=1, target_count=int(excluded_count), max_degree=4, topology_profile=str(axes.topology_profile), label_variant=str(axes.label_variant), search_attempts=int(attempts))
    excluded_labels = {str(label) for label in sample.target_labels}
    remaining_labels = tuple(
        sorted(
            (str(label) for label in sample.node_labels if str(label) not in excluded_labels),
            key=graph_label_sort_key,
        )
    )
    return replace(sample, target_labels=remaining_labels)

def _build_objective_plan():
    return NodeLinkObjectivePlan(public_id=TASK_ID, class_name='GraphCountingDegreeAfterRemovalFilterCountTask', supported_query_ids=SUPPORTED_QUERY_IDS, sample_graph=_sample_graph, answer_type='integer', answer_field='target_count', annotation_type='point_set', annotation_kind='node_point_set', annotation_field='target_labels', prompt_query_key=lambda axes: str(axes.query_id), object_description_key=lambda axes: 'object_description_directed' if str(axes.query_id).startswith('directed') else 'object_description_undirected', annotation_hint_key=lambda axes: 'annotation_hint_' + str(axes.query_id), graph_directionality=lambda axes: 'directed' if str(axes.query_id).startswith('directed') else 'undirected', scene_kind='graph_degree_after_removal_filter_counting', question_format=lambda axes: str(axes.query_id), fixed_values={'query_degree': 1}, value_ranges={'target_count': (1, 5)}, prompt_bundle_id='graph_node_link_counting_v1', prompt_scene_key='single_graph_counting', prompt_task_key='node_count_after_degree_filter_query')

@register_task
class GraphCountingDegreeAfterRemovalFilterCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology', 'state_update')
    domain = 'graph'
    supported_query_ids = SUPPORTED_QUERY_IDS

    def _build_objective_plan(self):
        return _build_objective_plan()

    def generate(self, instance_seed, *, params, max_attempts):
        return run_node_link_plan(plan=self._build_objective_plan(), instance_seed=int(instance_seed), params=dict(params), max_attempts=int(max_attempts))
__all__ = ['GraphCountingDegreeAfterRemovalFilterCountTask', 'TASK_ID']
