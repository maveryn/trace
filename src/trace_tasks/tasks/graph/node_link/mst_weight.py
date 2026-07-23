"""Find the minimum spanning tree total weight."""
from __future__ import annotations
from typing import Any, Dict
from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import NodeLinkAxes, NodeLinkObjectivePlan, run_node_link_plan
from .shared.sampling import sample_minimum_spanning_tree_weight_graph
TASK_ID = 'task_graph__node_link__mst_weight'
SCENE_ID = 'node_link'
SUPPORTED_QUERY_IDS = ('single',)

def _sample_graph(rng: Any, axes: NodeLinkAxes, attempts: int) -> Any:
    """Sample a graph satisfying this public objective contract."""
    return sample_minimum_spanning_tree_weight_graph(rng, node_count=int(axes.node_count), extra_edge_count=int(axes.values['extra_edge_count']), topology_profile=str(axes.topology_profile), label_variant=str(axes.label_variant), edge_weight_min=1, edge_weight_max=9)

def _build_objective_plan() -> NodeLinkObjectivePlan:
    """Bind query ids, sampler, answer, and annotation for this objective."""
    return NodeLinkObjectivePlan(public_id=TASK_ID, class_name='GraphOptimizationMinimumSpanningTreeWeightTask', supported_query_ids=SUPPORTED_QUERY_IDS, sample_graph=_sample_graph, answer_type='integer', answer_field='target_total_weight', annotation_type='segment_set', annotation_kind='edge_segment_set', annotation_field='target_edges', prompt_query_key='minimum_spanning_tree_weight', scene_kind='graph_mst_weight', question_format='minimum_spanning_tree_weight', value_ranges={'extra_edge_count': (1, 2)}, annotation_example=[[[180, 220], [310, 180]]])

@register_task
class GraphOptimizationMinimumSpanningTreeWeightTask:
    """Public owner for the node-link minimum-spanning-tree weight objective."""
    task_id = TASK_ID
    reasoning_operations = ('ranking', 'aggregation', 'topology')
    domain = 'graph'
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_objective_plan(self) -> NodeLinkObjectivePlan:
        """Return this task's local objective plan."""
        return _build_objective_plan()

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one task instance through neutral scene lifecycle plumbing."""
        return run_node_link_plan(plan=self._build_objective_plan(), instance_seed=int(instance_seed), params=dict(params), max_attempts=int(max_attempts))
__all__ = ['GraphOptimizationMinimumSpanningTreeWeightTask', 'TASK_ID']
