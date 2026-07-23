"""Count articulation-point nodes in a node-link graph."""
from __future__ import annotations
from typing import Any, Dict
from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import NodeLinkAxes, NodeLinkObjectivePlan, run_node_link_plan
from .shared.sampling import feasible_node_counts_for_articulation_point_count, sample_articulation_point_count_graph
TASK_ID = 'task_graph__node_link__articulation_point_count'
SCENE_ID = 'node_link'
SUPPORTED_QUERY_IDS = ('single',)

def _sample_graph(rng: Any, axes: NodeLinkAxes, attempts: int) -> Any:
    """Sample a graph satisfying this public objective contract."""
    target_count = int(axes.values['target_count'])
    feasible_nodes = feasible_node_counts_for_articulation_point_count(
        target_count=target_count,
        node_count_min=5,
        node_count_max=max(6, int(axes.node_count)),
    )
    if not feasible_nodes:
        raise ValueError('no feasible node count for articulation-point query')
    node_count = int(axes.node_count) if int(axes.node_count) in feasible_nodes else int(feasible_nodes[-1])
    return sample_articulation_point_count_graph(rng, node_count=node_count, target_count=target_count, topology_profile=str(axes.topology_profile), label_variant=str(axes.label_variant), attempts=int(attempts))

def _build_objective_plan() -> NodeLinkObjectivePlan:
    """Bind query ids, sampler, answer, and annotation for this objective."""
    return NodeLinkObjectivePlan(public_id=TASK_ID, class_name='GraphCountingArticulationPointCountTask', supported_query_ids=SUPPORTED_QUERY_IDS, sample_graph=_sample_graph, answer_type='integer', answer_field='target_count', annotation_type='point_set', annotation_kind='node_point_set', annotation_field='target_labels', prompt_query_key='articulation_point_count', scene_kind='graph_articulation_point_counting', question_format='count_articulation_points', value_ranges={'target_count': (1, 4)})

@register_task
class GraphCountingArticulationPointCountTask:
    """Public owner for the node-link articulation-point count objective."""
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology')
    domain = 'graph'
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_objective_plan(self) -> NodeLinkObjectivePlan:
        """Return this task's local objective plan."""
        return _build_objective_plan()

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one task instance through neutral scene lifecycle plumbing."""
        return run_node_link_plan(plan=self._build_objective_plan(), instance_seed=int(instance_seed), params=dict(params), max_attempts=int(max_attempts))
__all__ = ['GraphCountingArticulationPointCountTask', 'TASK_ID']
