"""Count nodes isolated after removing one node."""
from __future__ import annotations
from dataclasses import replace
from typing import Any, Dict
from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import NodeLinkAxes, NodeLinkObjectivePlan, run_node_link_plan
from .shared.sampling import feasible_node_counts_for_isolated_node_count_after_node_removal, sample_isolated_node_count_after_node_removal_graph
TASK_ID = 'task_graph__node_link__isolated_after_removal_count'
SCENE_ID = 'node_link'
SUPPORTED_QUERY_IDS = ('single',)
SUPPORTED_GRAPH_DIRECTIONALITIES = ('undirected', 'directed')

def _nearest_feasible_node_count(requested_node_count: int, feasible_nodes: tuple[int, ...]) -> int:
    """Return the nearest feasible node count at or above the requested count."""

    if not feasible_nodes:
        raise ValueError("no feasible node count for isolated-node-after-removal query")
    for node_count in feasible_nodes:
        if int(node_count) >= int(requested_node_count):
            return int(node_count)
    return int(feasible_nodes[-1])

def _resolve_graph_directionality(instance_seed: int, params: Dict[str, Any]) -> str:
    """Resolve this task's directed/undirected semantic axis."""

    raw_directionality = params.get('graph_directionality')
    if raw_directionality is not None and str(raw_directionality).strip():
        directionality = str(raw_directionality).strip().lower()
        if directionality not in set(SUPPORTED_GRAPH_DIRECTIONALITIES):
            raise ValueError(f"unsupported graph_directionality: {raw_directionality}")
        return directionality
    return str(
        uniform_choice(
            spawn_rng(int(instance_seed), f'{TASK_ID}:graph_directionality'),
            SUPPORTED_GRAPH_DIRECTIONALITIES,
        )
    )

def _sample_graph(rng: Any, axes: NodeLinkAxes, attempts: int) -> Any:
    """Sample a graph satisfying this public objective contract."""
    target_count = int(axes.values['target_count'])
    if int(target_count) < 0 or int(target_count) > 5:
        raise ValueError('target_count must be in [0, 5] for isolated-node-after-removal queries')
    directionality = str(axes.values.get('graph_directionality', 'undirected'))
    feasible_nodes = feasible_node_counts_for_isolated_node_count_after_node_removal(
        graph_directionality=directionality,
        target_count=target_count,
        node_count_min=5,
        node_count_max=max(10, int(axes.node_count)),
    )
    node_count = (
        int(axes.node_count)
        if int(axes.node_count) in feasible_nodes
        else _nearest_feasible_node_count(int(axes.node_count), feasible_nodes)
    )
    return sample_isolated_node_count_after_node_removal_graph(
        rng,
        graph_directionality=directionality,
        node_count=node_count,
        target_count=target_count,
        topology_profile=str(axes.topology_profile),
        label_variant=str(axes.label_variant),
    )

def _build_objective_plan() -> NodeLinkObjectivePlan:
    """Bind query ids, sampler, answer, and annotation for this objective."""
    return NodeLinkObjectivePlan(public_id=TASK_ID, class_name='GraphCountingIsolatedNodeCountAfterNodeRemovalTask', supported_query_ids=SUPPORTED_QUERY_IDS, sample_graph=_sample_graph, answer_type='integer', answer_field='target_count', annotation_type='point_set', annotation_kind='node_point_set', annotation_field='target_labels', prompt_query_key='isolated_node_count_after_node_removal', object_description_key=lambda axes: 'object_description_directed' if str(axes.values.get('graph_directionality')) == 'directed' else 'object_description_undirected', graph_directionality=lambda axes: str(axes.values.get('graph_directionality', 'undirected')), scene_kind='graph_isolated_after_removal_counting', question_format='isolated_node_count_after_node_removal', value_ranges={'target_count': (0, 5)})

@register_task
class GraphCountingIsolatedNodeCountAfterNodeRemovalTask:
    """Public owner for the node-link isolated-after-removal count objective."""
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology', 'state_update')
    domain = 'graph'
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_objective_plan(self) -> NodeLinkObjectivePlan:
        """Return this task's local objective plan."""
        return _build_objective_plan()

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one task instance through neutral scene lifecycle plumbing."""
        resolved_params = dict(params)
        directionality = _resolve_graph_directionality(int(instance_seed), resolved_params)
        plan = replace(
            self._build_objective_plan(),
            fixed_values={'graph_directionality': str(directionality)},
        )
        return run_node_link_plan(plan=plan, instance_seed=int(instance_seed), params=resolved_params, max_attempts=int(max_attempts))
__all__ = ['GraphCountingIsolatedNodeCountAfterNodeRemovalTask', 'TASK_ID']
