from __future__ import annotations
from ...registry import register_task
from ._lifecycle import NodeLinkObjectivePlan, run_node_link_plan
from .shared.labels import build_edge_label_support_resolver
from .shared.sampling import sample_edge_text_label_count_graph
TASK_ID = 'task_graph__node_link__edge_text_count'
SUPPORTED_QUERY_IDS = ('single',)
EDGE_LABEL_SUPPORT_SIZE = 16
EDGE_LABEL_MIN_CHARS = 3
EDGE_LABEL_MAX_CHARS = 5
MAX_LABELED_EDGE_COUNT = 12


def _sample_graph(rng, axes, attempts):
    return sample_edge_text_label_count_graph(
        rng,
        graph_directionality='undirected',
        node_count=max(int(axes.node_count), int(axes.values['target_count']) + 3),
        target_count=int(axes.values['target_count']),
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

def _build_objective_plan() -> NodeLinkObjectivePlan:
    """Bind query ids, sampler, answer, and annotation for this objective."""
    return NodeLinkObjectivePlan(public_id=TASK_ID, class_name='GraphCountingEdgeTextLabelCountTask', supported_query_ids=SUPPORTED_QUERY_IDS, sample_graph=_sample_graph, answer_type='integer', answer_field='target_count', annotation_type='bbox_set', annotation_kind='edge_label_bbox_set', annotation_field='target_edges', prompt_query_key='edge_text_label_count', scene_kind='graph_edge_text_counting', question_format='edge_text_label_count', value_ranges={'target_count': (1, 5), 'target_edge_label_index': (0, EDGE_LABEL_SUPPORT_SIZE - 1)}, annotation_example=[[248, 190, 300, 214], [420, 238, 472, 262]], strict_edge_label_placement=True)

@register_task
class GraphCountingEdgeTextLabelCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = 'graph'
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return run_node_link_plan(plan=_build_objective_plan(), instance_seed=int(instance_seed), params=dict(params), max_attempts=int(max_attempts))
