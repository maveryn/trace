"""Count nodes sharing a local relation to two queried nodes."""
from __future__ import annotations
from typing import Any, Dict
from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import NodeLinkAxes, NodeLinkObjectivePlan, run_node_link_plan
from .shared.sampling import sample_common_neighbor_count_graph
TASK_ID = 'task_graph__node_link__common_related_node_count'
SCENE_ID = 'node_link'
SUPPORTED_QUERY_IDS = ('undirected_common_neighbor_count', 'directed_common_successor_count', 'directed_common_predecessor_count')

def _sample_graph(rng: Any, axes: NodeLinkAxes, attempts: int) -> Any:
    """Sample a graph satisfying this public objective contract."""
    mode_by_query = {'undirected_common_neighbor_count': 'undirected_common_neighbor', 'directed_common_successor_count': 'directed_common_successor', 'directed_common_predecessor_count': 'directed_common_predecessor'}
    return sample_common_neighbor_count_graph(rng, common_neighbor_mode=mode_by_query[str(axes.query_id)], node_count=int(axes.node_count), target_count=int(axes.values['target_count']), max_degree=4, topology_profile=str(axes.topology_profile), label_variant=str(axes.label_variant), search_attempts=int(attempts))

def _build_objective_plan() -> NodeLinkObjectivePlan:
    """Bind query ids, sampler, answer, and annotation for this objective."""
    prompt_keys = {'undirected_common_neighbor_count': 'common_neighbor_count', 'directed_common_successor_count': 'common_successor_count', 'directed_common_predecessor_count': 'common_predecessor_count'}
    return NodeLinkObjectivePlan(public_id=TASK_ID, class_name='GraphRelationCommonNeighborCountTask', supported_query_ids=SUPPORTED_QUERY_IDS, sample_graph=_sample_graph, answer_type='integer', answer_field='target_count', annotation_type='point_set', annotation_kind='node_point_set', annotation_field='target_labels', prompt_query_key=lambda axes: prompt_keys[str(axes.query_id)], annotation_hint_key=lambda axes: 'annotation_hint_' + prompt_keys[str(axes.query_id)], graph_directionality=lambda axes: 'directed' if str(axes.query_id).startswith('directed') else 'undirected', scene_kind='graph_common_relation_counting', question_format=lambda axes: str(axes.query_id), value_ranges={'target_count': (1, 4)})

@register_task
class GraphRelationCommonNeighborCountTask:
    """Public owner for the node-link common-related-node count objective."""
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition', 'topology')
    domain = 'graph'
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_objective_plan(self) -> NodeLinkObjectivePlan:
        """Return this task's local objective plan."""
        return _build_objective_plan()

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one task instance through neutral scene lifecycle plumbing."""
        return run_node_link_plan(plan=self._build_objective_plan(), instance_seed=int(instance_seed), params=dict(params), max_attempts=int(max_attempts))
__all__ = ['GraphRelationCommonNeighborCountTask', 'TASK_ID']
