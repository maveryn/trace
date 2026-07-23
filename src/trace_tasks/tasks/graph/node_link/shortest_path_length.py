"""Find the length of a unique shortest path."""
from __future__ import annotations
from dataclasses import replace
from typing import Any, Dict
from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import NodeLinkAxes, NodeLinkObjectivePlan, run_node_link_plan
from .shared.sampling import sample_shortest_path_length_graph
TASK_ID = 'task_graph__node_link__shortest_path_length'
SCENE_ID = 'node_link'
SUPPORTED_QUERY_IDS = ('undirected_shortest_path_length', 'directed_shortest_path_length')

def _sample_graph(rng: Any, axes: NodeLinkAxes, attempts: int) -> Any:
    """Sample a graph satisfying this public objective contract."""
    directed = str(axes.query_id).startswith('directed')
    sample = sample_shortest_path_length_graph(rng, query_id='directed_shortest_path_length' if directed else 'shortest_path_length', node_count=int(axes.node_count), target_shortest_path_length=int(axes.values['target_shortest_path_length']), topology_profile=str(axes.topology_profile), label_variant=str(axes.label_variant))
    annotation_labels = tuple(str(label) for label in sample.target_labels[1:])
    if len(annotation_labels) != int(sample.target_shortest_path_length):
        raise ValueError("shortest-path annotation labels must exclude exactly the source node")
    return replace(sample, annotation_labels=annotation_labels)

def _build_objective_plan() -> NodeLinkObjectivePlan:
    """Bind query ids, sampler, answer, and annotation for this objective."""
    return NodeLinkObjectivePlan(public_id=TASK_ID, class_name='GraphPathShortestPathLengthTask', supported_query_ids=SUPPORTED_QUERY_IDS, sample_graph=_sample_graph, answer_type='integer', answer_field='target_shortest_path_length', annotation_type='point_sequence', annotation_kind='node_point_sequence', annotation_field='annotation_labels', prompt_query_key=lambda axes: 'directed_shortest_path_length' if str(axes.query_id).startswith('directed') else 'shortest_path_length', annotation_hint_key=lambda axes: 'annotation_hint_directed_shortest_path_length' if str(axes.query_id).startswith('directed') else 'annotation_hint_shortest_path_length', graph_directionality=lambda axes: 'directed' if str(axes.query_id).startswith('directed') else 'undirected', scene_kind='graph_shortest_path_length', question_format=lambda axes: str(axes.query_id), value_ranges={'target_shortest_path_length': (2, 5)})

@register_task
class GraphPathShortestPathLengthTask:
    """Public owner for the node-link shortest-path length objective."""
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
__all__ = ['GraphPathShortestPathLengthTask', 'TASK_ID']
