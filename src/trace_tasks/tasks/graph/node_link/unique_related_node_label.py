"""Read the unique node satisfying a local relation."""
from __future__ import annotations
from typing import Any, Dict
from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import NodeLinkAxes, NodeLinkObjectivePlan, run_node_link_plan
from .shared.sampling import sample_unique_node_label_relation_graph
TASK_ID = 'task_graph__node_link__unique_related_node_label'
SCENE_ID = 'node_link'
SUPPORTED_QUERY_IDS = ('unique_neighbor_label', 'unique_successor_label', 'unique_predecessor_label')

def _sample_graph(rng: Any, axes: NodeLinkAxes, attempts: int) -> Any:
    """Sample a graph satisfying this public objective contract."""
    mode_by_query = {'unique_neighbor_label': 'undirected_unique_neighbor', 'unique_successor_label': 'directed_unique_successor', 'unique_predecessor_label': 'directed_unique_predecessor'}
    return sample_unique_node_label_relation_graph(rng, relation_mode=mode_by_query[str(axes.query_id)], node_count=int(axes.node_count), max_degree=4, topology_profile=str(axes.topology_profile), label_variant=str(axes.label_variant))

def _build_objective_plan() -> NodeLinkObjectivePlan:
    """Bind query ids, sampler, answer, and annotation for this objective."""
    return NodeLinkObjectivePlan(public_id=TASK_ID, class_name='GraphRelationUniqueNodeLabelTask', supported_query_ids=SUPPORTED_QUERY_IDS, sample_graph=_sample_graph, answer_type='string', answer_field='answer_label', annotation_type='point', annotation_kind='node_point', annotation_field='target_labels', prompt_query_key=lambda axes: str(axes.query_id), annotation_hint_key=lambda axes: 'annotation_hint_' + str(axes.query_id), graph_directionality=lambda axes: 'directed' if str(axes.query_id) != 'unique_neighbor_label' else 'undirected', scene_kind='graph_unique_related_node_label', question_format=lambda axes: str(axes.query_id), annotation_example=[303, 187], answer_example='B')

@register_task
class GraphRelationUniqueNodeLabelTask:
    """Public owner for the node-link unique-related-node label objective."""
    task_id = TASK_ID
    reasoning_operations = ('topology',)
    domain = 'graph'
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_objective_plan(self) -> NodeLinkObjectivePlan:
        """Return this task's local objective plan."""
        return _build_objective_plan()

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one task instance through neutral scene lifecycle plumbing."""
        return run_node_link_plan(plan=self._build_objective_plan(), instance_seed=int(instance_seed), params=dict(params), max_attempts=int(max_attempts))
__all__ = ['GraphRelationUniqueNodeLabelTask', 'TASK_ID']
