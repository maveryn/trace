"""Find an endpoint node in the unique topological order."""
from __future__ import annotations
from dataclasses import replace
from typing import Any, Dict
from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import NodeLinkAxes, NodeLinkObjectivePlan, run_node_link_plan
from .shared.sampling import sample_topological_position_graph
TASK_ID = 'task_graph__node_link__topological_endpoint_node_label'
SCENE_ID = 'node_link'
SUPPORTED_QUERY_IDS = ('first_in_topological_order_label', 'last_in_topological_order_label')

def _sample_graph(rng: Any, axes: NodeLinkAxes, attempts: int) -> Any:
    """Sample a graph satisfying this public objective contract."""
    target_position = 1 if str(axes.query_id) == 'first_in_topological_order_label' else int(axes.node_count)
    sample = sample_topological_position_graph(
        rng,
        node_count=int(axes.node_count),
        target_position=int(target_position),
        topology_profile=str(axes.topology_profile),
        label_variant=str(axes.label_variant),
    )
    answer_label = str(sample.query_label)
    return replace(
        sample,
        answer_label=answer_label,
        annotation_labels=(answer_label,),
    )

def _build_objective_plan() -> NodeLinkObjectivePlan:
    """Bind query ids, sampler, answer, and annotation for this objective."""
    return NodeLinkObjectivePlan(public_id=TASK_ID, class_name='GraphOrderTopologicalEndpointNodeLabelTask', supported_query_ids=SUPPORTED_QUERY_IDS, sample_graph=_sample_graph, answer_type='string', answer_field='answer_label', annotation_type='point', annotation_kind='node_point', annotation_field='annotation_labels', prompt_query_key=lambda axes: str(axes.query_id), annotation_hint_key=lambda axes: 'annotation_hint_' + str(axes.query_id), graph_directionality='directed', scene_kind='graph_topological_endpoint_node_label', question_format=lambda axes: str(axes.query_id), value_ranges={}, annotation_example=[180, 160], answer_example='A')

@register_task
class GraphOrderTopologicalEndpointNodeLabelTask:
    """Public owner for the node-link topological-order endpoint objective."""
    task_id = TASK_ID
    reasoning_operations = ('ranking', 'topology')
    domain = 'graph'
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_objective_plan(self) -> NodeLinkObjectivePlan:
        """Return this task's local objective plan."""
        return _build_objective_plan()

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one task instance through neutral scene lifecycle plumbing."""
        return run_node_link_plan(plan=self._build_objective_plan(), instance_seed=int(instance_seed), params=dict(params), max_attempts=int(max_attempts))
__all__ = ['GraphOrderTopologicalEndpointNodeLabelTask', 'TASK_ID']
