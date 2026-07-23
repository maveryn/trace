"""Count the edited connected component size for one queried node."""
from __future__ import annotations
from typing import Any, Dict
from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import NodeLinkAxes, NodeLinkObjectivePlan, run_node_link_plan
from .shared.sampling import sample_component_size_after_edge_edit_graph
TASK_ID = 'task_graph__node_link__component_size_after_edge_edit'
SCENE_ID = 'node_link'
SUPPORTED_QUERY_IDS = ('component_size_after_edge_removal', 'component_size_after_edge_addition')

def _sample_graph(rng: Any, axes: NodeLinkAxes, attempts: int) -> Any:
    """Sample a graph satisfying this public objective contract."""
    operation = 'edge_removal' if str(axes.query_id).endswith('removal') else 'edge_addition'
    target_size = max(2, int(axes.values['target_component_size']))
    return sample_component_size_after_edge_edit_graph(rng, edit_operation=operation, node_count=int(axes.node_count), target_component_size=target_size, topology_profile=str(axes.topology_profile), label_variant=str(axes.label_variant))

def _build_objective_plan() -> NodeLinkObjectivePlan:
    """Bind query ids, sampler, answer, and annotation for this objective."""
    return NodeLinkObjectivePlan(public_id=TASK_ID, class_name='GraphRelationComponentSizeAfterEdgeEditTask', supported_query_ids=SUPPORTED_QUERY_IDS, sample_graph=_sample_graph, answer_type='integer', answer_field='target_component_size', annotation_type='point_set', annotation_kind='node_point_set', annotation_field='target_labels', prompt_query_key=lambda axes: str(axes.query_id), annotation_hint_key=lambda axes: 'annotation_hint_' + str(axes.query_id), scene_kind='graph_component_size_after_edge_edit', question_format=lambda axes: str(axes.query_id), value_ranges={'target_component_size': (2, 6)})

@register_task
class GraphRelationComponentSizeAfterEdgeEditTask:
    """Public owner for the node-link component-size after edge-edit objective."""
    task_id = TASK_ID
    reasoning_operations = ('counting', 'topology', 'state_update')
    domain = 'graph'
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_objective_plan(self) -> NodeLinkObjectivePlan:
        """Return this task's local objective plan."""
        return _build_objective_plan()

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one task instance through neutral scene lifecycle plumbing."""
        return run_node_link_plan(plan=self._build_objective_plan(), instance_seed=int(instance_seed), params=dict(params), max_attempts=int(max_attempts))
__all__ = ['GraphRelationComponentSizeAfterEdgeEditTask', 'TASK_ID']
