"""Find the length of a unique directed longest path."""
from __future__ import annotations
from typing import Any, Dict
from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import NodeLinkAxes, NodeLinkObjectivePlan, run_node_link_plan
from .shared.sampling import sample_longest_path_length_graph
TASK_ID = 'task_graph__node_link__longest_path_length'
SCENE_ID = 'node_link'
SUPPORTED_QUERY_IDS = ('single',)

def _sample_graph(rng: Any, axes: NodeLinkAxes, attempts: int) -> Any:
    """Sample a graph satisfying this public objective contract."""
    return sample_longest_path_length_graph(rng, node_count=int(axes.node_count), target_longest_path_length=int(axes.values['target_longest_path_length']), topology_profile=str(axes.topology_profile), label_variant=str(axes.label_variant))

def _build_objective_plan() -> NodeLinkObjectivePlan:
    """Bind query ids, sampler, answer, and annotation for this objective."""
    return NodeLinkObjectivePlan(public_id=TASK_ID, class_name='GraphPathLongestPathLengthTask', supported_query_ids=SUPPORTED_QUERY_IDS, sample_graph=_sample_graph, answer_type='integer', answer_field='target_longest_path_length', annotation_type='point_sequence', annotation_kind='node_point_sequence', annotation_field='target_labels', prompt_query_key='directed_longest_path_length', annotation_hint_key='annotation_hint_directed_longest_path_length', graph_directionality='directed', scene_kind='graph_longest_path_length', question_format='directed_longest_path_length', value_ranges={'target_longest_path_length': (2, 5)})

@register_task
class GraphPathLongestPathLengthTask:
    """Public owner for the node-link longest-path length objective."""
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
__all__ = ['GraphPathLongestPathLengthTask', 'TASK_ID']
