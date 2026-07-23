"""Read a neighbor in the unique Hamiltonian cycle."""
from __future__ import annotations
from typing import Any, Dict
from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import NodeLinkAxes, NodeLinkObjectivePlan, run_node_link_plan
from .shared.sampling import sample_hamiltonian_cycle_neighbor_graph
TASK_ID = 'task_graph__node_link__hamiltonian_cycle_neighbor_label'
SCENE_ID = 'node_link'
SUPPORTED_QUERY_IDS = ('next_in_hamiltonian_cycle_label', 'previous_in_hamiltonian_cycle_label')

def _sample_graph(rng: Any, axes: NodeLinkAxes, attempts: int) -> Any:
    """Sample a graph satisfying this public objective contract."""
    return sample_hamiltonian_cycle_neighbor_graph(rng, query_id=str(axes.query_id), node_count=int(axes.node_count), topology_profile=str(axes.topology_profile), label_variant=str(axes.label_variant))

def _build_objective_plan() -> NodeLinkObjectivePlan:
    """Bind query ids, sampler, answer, and annotation for this objective."""
    return NodeLinkObjectivePlan(public_id=TASK_ID, class_name='GraphRelationHamiltonianCycleNeighborLabelTask', supported_query_ids=SUPPORTED_QUERY_IDS, sample_graph=_sample_graph, answer_type='string', answer_field='answer_label', annotation_type='point', annotation_kind='node_point', annotation_field='answer_label', prompt_query_key=lambda axes: str(axes.query_id), scene_kind='graph_hamiltonian_cycle_neighbor', question_format=lambda axes: str(axes.query_id), annotation_example=[310, 180], answer_example='B')

@register_task
class GraphRelationHamiltonianCycleNeighborLabelTask:
    """Public owner for the node-link Hamiltonian-cycle neighbor label objective."""
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
__all__ = ['GraphRelationHamiltonianCycleNeighborLabelTask', 'TASK_ID']
