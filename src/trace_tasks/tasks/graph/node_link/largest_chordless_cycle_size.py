"""Find the largest chordless cycle size."""
from __future__ import annotations
from typing import Any, Dict
from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import NodeLinkAxes, NodeLinkObjectivePlan, run_node_link_plan
from .shared.sampling import sample_largest_chordless_cycle_graph
TASK_ID = 'task_graph__node_link__largest_chordless_cycle_size'
SCENE_ID = 'node_link'
SUPPORTED_QUERY_IDS = ('single',)

def _sample_graph(rng: Any, axes: NodeLinkAxes, attempts: int) -> Any:
    """Sample a graph satisfying this public objective contract."""
    return sample_largest_chordless_cycle_graph(rng, node_count=int(axes.node_count), target_cycle_size=int(axes.values['target_cycle_size']), topology_profile=str(axes.topology_profile), label_variant=str(axes.label_variant))

def _build_objective_plan() -> NodeLinkObjectivePlan:
    """Bind query ids, sampler, answer, and annotation for this objective."""
    return NodeLinkObjectivePlan(public_id=TASK_ID, class_name='GraphRelationLargestChordlessCycleSizeTask', supported_query_ids=SUPPORTED_QUERY_IDS, sample_graph=_sample_graph, answer_type='integer', answer_field='target_cycle_size', annotation_type='point_set', annotation_kind='node_point_set', annotation_field='target_labels', prompt_query_key='largest_chordless_cycle_size', scene_kind='graph_largest_chordless_cycle', question_format='largest_chordless_cycle_size', value_ranges={'target_cycle_size': (3, 7)})

@register_task
class GraphRelationLargestChordlessCycleSizeTask:
    """Public owner for the node-link largest-chordless-cycle size objective."""
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'ranking', 'topology')
    domain = 'graph'
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_objective_plan(self) -> NodeLinkObjectivePlan:
        """Return this task's local objective plan."""
        return _build_objective_plan()

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one task instance through neutral scene lifecycle plumbing."""
        return run_node_link_plan(plan=self._build_objective_plan(), instance_seed=int(instance_seed), params=dict(params), max_attempts=int(max_attempts))
__all__ = ['GraphRelationLargestChordlessCycleSizeTask', 'TASK_ID']
