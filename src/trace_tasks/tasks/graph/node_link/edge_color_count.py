"""Count edges of a queried color."""
from __future__ import annotations
from typing import Any, Dict
from ...base import TaskOutput
from ...registry import register_task
from ...shared.named_colors import available_named_colors
from ._lifecycle import NodeLinkAxes, NodeLinkObjectivePlan, run_node_link_plan
from .shared.sampling import sample_edge_color_count_graph
TASK_ID = 'task_graph__node_link__edge_color_count'
SCENE_ID = 'node_link'
SUPPORTED_QUERY_IDS = ('single',)
_COLOR_SUPPORT = tuple(str(name) for name, _rgb in available_named_colors())


def _resolve_target_color(rng: Any, axes: NodeLinkAxes) -> str:
    """Resolve the semantic edge color queried by this task."""

    target = str(axes.values.get('target_color_name', '')).strip().lower()
    if target:
        if target not in _COLOR_SUPPORT:
            raise ValueError('target_color_name must be in the shared named-color palette')
        return target
    return str(rng.choice(_COLOR_SUPPORT))

def _sample_graph(rng: Any, axes: NodeLinkAxes, attempts: int) -> Any:
    """Sample a graph satisfying this public objective contract."""
    target_color_name = _resolve_target_color(rng, axes)
    return sample_edge_color_count_graph(rng, graph_directionality='undirected', node_count=max(int(axes.node_count), int(axes.values['target_count']) + 3), target_count=int(axes.values['target_count']), target_color_name=target_color_name, color_support=_COLOR_SUPPORT, topology_profile=str(axes.topology_profile), label_variant=str(axes.label_variant), max_degree=4)

def _build_objective_plan() -> NodeLinkObjectivePlan:
    """Bind query ids, sampler, answer, and annotation for this objective."""
    return NodeLinkObjectivePlan(public_id=TASK_ID, class_name='GraphCountingEdgeColorCountTask', supported_query_ids=SUPPORTED_QUERY_IDS, sample_graph=_sample_graph, answer_type='integer', answer_field='target_count', annotation_type='segment_set', annotation_kind='edge_segment_set', annotation_field='target_edges', prompt_query_key='edge_color_count', scene_kind='graph_edge_color_counting', question_format='edge_color_count', value_ranges={'target_count': (1, 4)}, semantic_colors=_COLOR_SUPPORT, annotation_example=[[[180, 220], [310, 180]]])

@register_task
class GraphCountingEdgeColorCountTask:
    """Public owner for the node-link edge-color count objective."""
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = 'graph'
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_objective_plan(self) -> NodeLinkObjectivePlan:
        """Return this task's local objective plan."""
        return _build_objective_plan()

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one task instance through neutral scene lifecycle plumbing."""
        return run_node_link_plan(plan=self._build_objective_plan(), instance_seed=int(instance_seed), params=dict(params), max_attempts=int(max_attempts))
__all__ = ['GraphCountingEdgeColorCountTask', 'TASK_ID']
