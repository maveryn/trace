"""Find the largest connected component size."""
from __future__ import annotations
from typing import Any, Dict
from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import NodeLinkAxes, NodeLinkObjectivePlan, run_node_link_plan
from .shared.sampling import feasible_node_counts_for_unique_largest_component, sample_largest_component_size_graph
TASK_ID = 'task_graph__node_link__largest_component_size'
SCENE_ID = 'node_link'
SUPPORTED_QUERY_IDS = ('single',)

def _sample_graph(rng: Any, axes: NodeLinkAxes, attempts: int) -> Any:
    """Sample a graph satisfying this public objective contract."""
    target_size = int(axes.values['target_component_size'])
    component_count = 3
    feasible_nodes = feasible_node_counts_for_unique_largest_component(target_largest_component_size=target_size, component_count=component_count, node_count_min=3, node_count_max=max(3, int(axes.node_count), target_size + 2))
    if not feasible_nodes:
        raise ValueError('no feasible node count for largest-component query')
    node_count = int(feasible_nodes[-1])
    return sample_largest_component_size_graph(rng, node_count=node_count, target_largest_component_size=target_size, component_count=component_count, topology_profile=str(axes.topology_profile), label_variant=str(axes.label_variant))

def _build_objective_plan() -> NodeLinkObjectivePlan:
    """Bind query ids, sampler, answer, and annotation for this objective."""
    return NodeLinkObjectivePlan(public_id=TASK_ID, class_name='GraphComparisonLargestComponentSizeTask', supported_query_ids=SUPPORTED_QUERY_IDS, sample_graph=_sample_graph, answer_type='integer', answer_field='target_largest_component_size', annotation_type='point_set', annotation_kind='node_point_set', annotation_field='target_labels', prompt_query_key='largest_component_size', scene_kind='graph_largest_component_size', question_format='largest_component_size', value_ranges={'target_component_size': (2, 5)}, fixed_values={'component_count': 3})

@register_task
class GraphComparisonLargestComponentSizeTask:
    """Public owner for the node-link largest-component size objective."""
    task_id = TASK_ID
    reasoning_operations = ('counting', 'ranking', 'topology')
    domain = 'graph'
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_objective_plan(self) -> NodeLinkObjectivePlan:
        """Return this task's local objective plan."""
        return _build_objective_plan()

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one task instance through neutral scene lifecycle plumbing."""
        return run_node_link_plan(plan=self._build_objective_plan(), instance_seed=int(instance_seed), params=dict(params), max_attempts=int(max_attempts))
__all__ = ['GraphComparisonLargestComponentSizeTask', 'TASK_ID']
