from __future__ import annotations

from ...registry import register_task
from ._lifecycle import NodeLinkObjectivePlan, run_node_link_plan
from .shared.labels import build_edge_label_support_resolver
from .shared.sampling import sample_edge_attribute_label_graph
TASK_ID = 'task_graph__node_link__edge_between_nodes_label'
SUPPORTED_QUERY_IDS = ('edge_between_nodes_label', 'directed_edge_between_nodes_label')
EDGE_LABEL_SUPPORT_SIZE = 16
EDGE_LABEL_MIN_CHARS = 3
EDGE_LABEL_MAX_CHARS = 5
MAX_LABELED_EDGE_COUNT = 12


def _sample_graph(rng, axes, attempts):
    directionality = 'directed' if str(axes.query_id).startswith('directed') else 'undirected'
    return sample_edge_attribute_label_graph(
        rng,
        graph_directionality=directionality,
        node_count=int(axes.node_count),
        target_edge_label=str(axes.values.get('target_edge_label', '')),
        edge_label_support=(),
        topology_profile=str(axes.topology_profile),
        label_variant=str(axes.label_variant),
        max_degree=3,
        target_edge_label_index=int(axes.values['target_edge_label_index']),
        edge_label_support_resolver=build_edge_label_support_resolver(
            axes,
            default_support_size=EDGE_LABEL_SUPPORT_SIZE,
            default_min_chars=EDGE_LABEL_MIN_CHARS,
            default_max_chars=EDGE_LABEL_MAX_CHARS,
        ),
        max_labeled_edge_count=int(axes.values.get('max_labeled_edge_count', MAX_LABELED_EDGE_COUNT)),
    )

def _edge_label_plan():
    return NodeLinkObjectivePlan(public_id=TASK_ID, class_name='GraphRelationEdgeBetweenNodesLabelTask', supported_query_ids=SUPPORTED_QUERY_IDS, sample_graph=_sample_graph, answer_type='string', answer_field='target_edge_label', annotation_type='bbox', annotation_kind='edge_label_bbox', annotation_field='query_edge', prompt_query_key=lambda axes: str(axes.query_id), annotation_hint_key=lambda axes: 'annotation_hint_' + str(axes.query_id), graph_directionality=lambda axes: 'directed' if str(axes.query_id).startswith('directed') else 'undirected', scene_kind='graph_edge_label_lookup', question_format=lambda axes: str(axes.query_id), value_ranges={'target_edge_label_index': (0, EDGE_LABEL_SUPPORT_SIZE - 1)}, annotation_example=[180, 220, 230, 245], answer_example='alpha', strict_edge_label_placement=True)

@register_task
class GraphRelationEdgeBetweenNodesLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('topology',)
    domain = 'graph'
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        objective = _edge_label_plan()
        return run_node_link_plan(plan=objective, instance_seed=int(instance_seed), params=dict(params), max_attempts=int(max_attempts))
