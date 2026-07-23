"""Return MST total weight from a weighted adjacency matrix."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from ...shared.fixed_query import select_task_query_id
from ._lifecycle import render_single_panel_artifacts, single_panel_render_kwargs, single_panel_task_output, single_panel_trace_payload
from .shared.annotations import mst_cell_bbox_artifacts
from .shared.output import label_node_query_params
from .shared.prompts import PROMPT_BUNDLE_ID as ADJACENCY_PROMPT_BUNDLE_ID
from .shared.prompts import build_adjacency_prompt_artifacts
from .shared.rendering import render_adjacency_matrix_panel
from .shared.sampling import resolve_adjacency_int_axis, resolve_adjacency_labels, resolve_label_node_axes, sample_weighted_matrix_mst_adjacency
from .shared.state import SCENE_ID
from ..shared.visual_defaults import load_graph_scene_background_defaults, load_graph_scene_noise_defaults
TASK_ID = 'task_graph__adjacency__mst_weight'
SCENE_ID = 'adjacency'
PUBLIC_QUERY_ID = 'single'
PROMPT_KEY = 'weighted_matrix_mst_weight'
SUPPORTED_ADJACENCY_MATRIX_MST_QUERY_IDS: Tuple[str, ...] = (PUBLIC_QUERY_ID,)

@dataclass(frozen=True)
class _TaskDefaults:
    """Stable fallback defaults for weighted adjacency-matrix MST tasks."""
    node_count_min: int = 4
    node_count_max: int = 7
    extra_edge_count_min: int = 1
    extra_edge_count_max: int = 3
    edge_weight_min: int = 1
    edge_weight_max: int = 12
    label_max_chars: int = 5
    label_variant: str = 'letters'
    canvas_width: int = 900
    canvas_height: int = 640
    label_font_size_px: int = 19

@dataclass(frozen=True)
class _ResolvedQuery:
    """Resolved support axes for one weighted adjacency-matrix MST instance."""
    query_id: str
    node_count: int
    extra_edge_count: int
    edge_weight_min: int
    edge_weight_max: int
    label_variant: str
    query_id_probabilities: Dict[str, float]
    node_count_probabilities: Dict[str, float]
    extra_edge_count_probabilities: Dict[str, float]
    label_variant_probabilities: Dict[str, float]
_DEFAULTS = _TaskDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults('graph', SCENE_ID, task_id=TASK_ID)
PROMPT_BUNDLE_ID = str(group_default(_PROMPT_DEFAULTS, 'bundle_id', ADJACENCY_PROMPT_BUNDLE_ID))
POST_IMAGE_BACKGROUND_DEFAULTS = load_graph_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_graph_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)

def _resolve_query(instance_seed: int, *, params: Mapping[str, Any]) -> _ResolvedQuery:
    """Resolve MST matrix axes while the public task owns the MST objective binding."""
    query_id, query_probs, task_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=SUPPORTED_ADJACENCY_MATRIX_MST_QUERY_IDS, default_query_id=SUPPORTED_ADJACENCY_MATRIX_MST_QUERY_IDS[0], task_id=TASK_ID, namespace='query_id')
    common_axes = resolve_label_node_axes(instance_seed=int(instance_seed), params=task_params, gen_defaults=_GEN_DEFAULTS, node_count_min=_DEFAULTS.node_count_min, node_count_max=_DEFAULTS.node_count_max, rng_namespace=TASK_ID)
    max_possible_extra = int(common_axes.node.value) * (int(common_axes.node.value) - 1) // 2 - (int(common_axes.node.value) - 1)
    extra_axis = resolve_adjacency_int_axis(instance_seed=int(instance_seed), params=task_params, gen_defaults=_GEN_DEFAULTS, axis_name='extra_edge_count', default_min=_DEFAULTS.extra_edge_count_min, default_max=_DEFAULTS.extra_edge_count_max, rng_namespace=TASK_ID, max_value=int(max_possible_extra))
    edge_weight_min, edge_weight_max = _resolve_edge_weight_bounds(task_params)
    return _ResolvedQuery(query_id=str(query_id), node_count=int(common_axes.node.value), extra_edge_count=int(extra_axis.value), edge_weight_min=int(edge_weight_min), edge_weight_max=int(edge_weight_max), label_variant=str(common_axes.label.value), query_id_probabilities=dict(query_probs), node_count_probabilities=dict(common_axes.node.probabilities), extra_edge_count_probabilities=dict(extra_axis.probabilities), label_variant_probabilities=dict(common_axes.label.probabilities))

def _resolve_edge_weight_bounds(task_params: Mapping[str, Any]) -> tuple[int, int]:
    """Resolve MST edge-weight bounds with room for distractor edges."""
    edge_weight_min = int(task_params.get('edge_weight_min', group_default(_GEN_DEFAULTS, 'edge_weight_min', _DEFAULTS.edge_weight_min)))
    edge_weight_max = int(task_params.get('edge_weight_max', group_default(_GEN_DEFAULTS, 'edge_weight_max', _DEFAULTS.edge_weight_max)))
    if int(edge_weight_max) < int(edge_weight_min) + 2:
        raise ValueError('edge_weight_max must leave room for non-MST distractor weights')
    return (int(edge_weight_min), int(edge_weight_max))

@register_task
class GraphOptimizationAdjacencyMatrixMSTWeightTask:
    """Return the unique minimum-spanning-tree weight from a weighted adjacency matrix."""
    task_id = 'task_graph__adjacency__mst_weight'
    reasoning_operations = ('ranking', 'aggregation', 'topology')
    domain = 'graph'
    supported_query_ids = SUPPORTED_ADJACENCY_MATRIX_MST_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate a weighted matrix and bind MST cells to the total-weight answer."""
        query = _resolve_query(int(instance_seed), params=params)
        labels = resolve_adjacency_labels(instance_seed=int(instance_seed), rng_namespace=TASK_ID, label_variant=str(query.label_variant), node_count=int(query.node_count), max_chars=int(group_default(_GEN_DEFAULTS, 'label_max_chars', _DEFAULTS.label_max_chars)))
        sample = sample_weighted_matrix_mst_adjacency(instance_seed=int(instance_seed), rng_namespace=TASK_ID, labels=labels.labels, extra_edge_count=int(query.extra_edge_count), edge_weight_min=int(query.edge_weight_min), edge_weight_max=int(query.edge_weight_max))
        render_artifacts = render_single_panel_artifacts(task_params=params, render_defaults=_RENDER_DEFAULTS, defaults=_DEFAULTS, background_defaults=POST_IMAGE_BACKGROUND_DEFAULTS, noise_defaults=POST_IMAGE_NOISE_DEFAULTS, instance_seed=int(instance_seed), render_panel=lambda base_image: render_adjacency_matrix_panel(sample=sample, base_image=base_image, title='Weighted Adjacency Matrix', subtitle='Undirected graph; blank cells mean no edge.', weighted=True, **single_panel_render_kwargs(params=params, render_defaults=_RENDER_DEFAULTS, defaults=_DEFAULTS, instance_seed=int(instance_seed))))
        rendered = render_artifacts.rendered
        annotation_artifacts, annotation_cell_edges = mst_cell_bbox_artifacts(sample, rendered)
        answer_gt = TypedValue(type='integer', value=int(sample.mst_weight))
        prompt_artifacts = build_adjacency_prompt_artifacts(domain=self.domain, bundle_id=PROMPT_BUNDLE_ID, prompt_key=PROMPT_KEY, dynamic_slots={'object_description': 'a connected undirected weighted graph as a weighted adjacency matrix'}, instance_seed=int(instance_seed))
        mst_edge_set = {tuple(edge) for edge in sample.mst_edges}
        node_entities = [{'entity_id': f'node_{label}', 'entity_kind': 'adjacency_matrix_label', 'label': str(label), 'row_bbox_xyxy': list(rendered.row_label_bboxes[str(label)]), 'column_bbox_xyxy': list(rendered.column_label_bboxes[str(label)])} for label in sample.labels]
        edge_entities = [{'entity_id': f'edge_{left}_{right}', 'entity_kind': 'weighted_adjacency_matrix_edge', 'node_u_label': str(left), 'node_v_label': str(right), 'weight': int(sample.weights[str(left), str(right)]), 'is_in_minimum_spanning_tree': bool((str(left), str(right)) in mst_edge_set)} for left, right in sample.edges]
        query_params = {'query_id': str(query.query_id), 'query_id_probabilities': dict(query.query_id_probabilities), **label_node_query_params(query, labels), 'edge_weight_min': int(query.edge_weight_min), 'edge_weight_max': int(query.edge_weight_max)}
        trace_payload = single_panel_trace_payload(task_id=TASK_ID, scene_id=SCENE_ID, query_id=str(query.query_id), prompt_bundle_id=PROMPT_BUNDLE_ID, prompt_artifacts=prompt_artifacts, render_artifacts=render_artifacts, entities=[*node_entities, *edge_entities], relations={'representation_variant': str(rendered.representation_variant), 'query_id': str(query.query_id), 'directed': False, 'weights': [{'edge': [str(left), str(right)], 'weight': int(weight)} for (left, right), weight in sorted(sample.weights.items())], 'minimum_spanning_tree_edges': [list(edge) for edge in sample.mst_edges], 'minimum_spanning_tree_total_weight': int(sample.mst_weight), 'adjacency': {str(key): list(values) for key, values in sample.adjacency.items()}}, query_params=query_params, execution_fields={'answer': int(sample.mst_weight), 'node_count': int(query.node_count), 'edge_count': int(len(sample.edges)), 'extra_edge_count': int(query.extra_edge_count), 'minimum_spanning_tree_edges': [list(edge) for edge in sample.mst_edges], 'annotation_cell_edges': [list(edge) for edge in annotation_cell_edges], 'label_variant': str(labels.label_variant)}, witness_symbolic={'type': 'weighted_matrix_cell_set', 'edges': [list(edge) for edge in annotation_cell_edges], 'mst_weight': int(sample.mst_weight)}, projected_annotation=dict(annotation_artifacts.projected_annotation))
        return single_panel_task_output(prompt_artifacts=prompt_artifacts, answer_gt=answer_gt, annotation_gt=annotation_artifacts.annotation_gt, render_artifacts=render_artifacts, trace_payload=trace_payload, scene_id=SCENE_ID, query_id=str(query.query_id))
__all__ = ['GraphOptimizationAdjacencyMatrixMSTWeightTask']
