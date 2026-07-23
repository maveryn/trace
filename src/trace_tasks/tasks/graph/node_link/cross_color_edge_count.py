from __future__ import annotations
from ...registry import register_task
from ...shared.named_colors import available_named_colors
from ._lifecycle import NodeLinkObjectivePlan, run_node_link_plan
from .shared.sampling import sample_cross_color_edge_count_graph
TASK_ID = 'task_graph__node_link__cross_color_edge_count'
SUPPORTED_QUERY_IDS = ('cross_color_edge_count', 'directed_cross_color_edge_count')
_COLOR_SUPPORT = tuple(str(name) for name, _rgb in available_named_colors())


def _resolve_color_pair(rng, axes):
    source = str(axes.values.get('source_color_name', '')).strip().lower()
    target = str(axes.values.get('target_color_name', '')).strip().lower()
    if source or target:
        if not source or not target:
            raise ValueError('source_color_name and target_color_name must be provided together')
        if source == target:
            raise ValueError('source_color_name and target_color_name must be distinct')
        if source not in _COLOR_SUPPORT or target not in _COLOR_SUPPORT:
            raise ValueError('source_color_name and target_color_name must be in the shared named-color palette')
        return source, target
    sampled = rng.sample(list(_COLOR_SUPPORT), 2)
    return str(sampled[0]), str(sampled[1])

def _sample_graph(rng, axes, attempts):
    directionality = 'directed' if str(axes.query_id).startswith('directed') else 'undirected'
    source_color_name, target_color_name = _resolve_color_pair(rng, axes)
    return sample_cross_color_edge_count_graph(rng, graph_directionality=directionality, node_count=max(int(axes.node_count), int(axes.values['target_count']) + 4), target_count=int(axes.values['target_count']), source_color_name=source_color_name, target_color_name=target_color_name, color_support=_COLOR_SUPPORT, topology_profile=str(axes.topology_profile), label_variant=str(axes.label_variant), max_degree=4)

def _build_objective_plan():
    return NodeLinkObjectivePlan(public_id=TASK_ID, class_name='GraphCountingCrossColorEdgeCountTask', supported_query_ids=SUPPORTED_QUERY_IDS, sample_graph=_sample_graph, answer_type='integer', answer_field='target_count', annotation_type='segment_set', annotation_kind='edge_segment_set', annotation_field='target_edges', prompt_query_key=lambda axes: str(axes.query_id), object_description_key=lambda axes: 'object_description_directed' if str(axes.query_id).startswith('directed') else 'object_description_undirected', annotation_hint_key=lambda axes: 'annotation_hint_' + str(axes.query_id), graph_directionality=lambda axes: 'directed' if str(axes.query_id).startswith('directed') else 'undirected', scene_kind='graph_cross_color_edge_counting', question_format=lambda axes: str(axes.query_id), value_ranges={'target_count': (1, 4)}, annotation_example=[[[180, 220], [310, 180]]])

@register_task
class GraphCountingCrossColorEdgeCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology')
    domain = 'graph'
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return run_node_link_plan(plan=_build_objective_plan(), instance_seed=int(instance_seed), params=dict(params), max_attempts=int(max_attempts))
__all__ = ['GraphCountingCrossColorEdgeCountTask', 'TASK_ID']
