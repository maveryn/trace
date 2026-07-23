"""Return the label at a position in BFS or DFS order from an adjacency list."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple
from ....core.sampling import uniform_choice
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from ...shared.fixed_query import select_task_query_id
from ._lifecycle import render_single_panel_artifacts, single_panel_render_kwargs, single_panel_task_output, single_panel_trace_payload
from .shared.algorithms import bfs_visit_order, dfs_visit_order
from .shared.annotations import row_label_bbox_sequence_value
from .shared.output import label_node_query_params
from .shared.prompts import PROMPT_BUNDLE_ID as ADJACENCY_PROMPT_BUNDLE_ID
from .shared.prompts import build_adjacency_prompt_artifacts
from .shared.rendering import render_adjacency_list_panel
from .shared.sampling import configured_axis_max, configured_axis_min, resolve_adjacency_int_axis, resolve_adjacency_labels, resolve_label_node_axes, sample_reachable_directed_adjacency
from .shared.state import SCENE_ID
from ..shared.task_support import format_graph_prompt_label
from ..shared.visual_defaults import load_graph_scene_background_defaults, load_graph_scene_noise_defaults
TASK_ID = 'task_graph__adjacency__traversal_kth_label'
SCENE_ID = 'adjacency'
SUPPORTED_ADJACENCY_TRAVERSAL_QUERY_IDS: Tuple[str, ...] = ('bfs_kth_visit_label', 'dfs_kth_visit_label')

@dataclass(frozen=True)
class _TaskDefaults:
    """Stable fallback defaults for adjacency-list traversal tasks."""
    node_count_min: int = 5
    node_count_max: int = 8
    traversal_position_min: int = 2
    traversal_position_max: int = 8
    extra_edge_count_min: int = 1
    extra_edge_count_max: int = 4
    label_max_chars: int = 5
    label_variant: str = 'letters'
    canvas_width: int = 900
    canvas_height: int = 640
    label_font_size_px: int = 20

@dataclass(frozen=True)
class _ResolvedQuery:
    """Resolved support axes for one adjacency traversal instance."""
    query_id: str
    node_count: int
    traversal_position: int
    extra_edge_count: int
    label_variant: str
    source_index: int
    query_id_probabilities: Dict[str, float]
    node_count_probabilities: Dict[str, float]
    traversal_position_probabilities: Dict[str, float]
    extra_edge_count_probabilities: Dict[str, float]
    label_variant_probabilities: Dict[str, float]
_DEFAULTS = _TaskDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults('graph', SCENE_ID, task_id=TASK_ID)
PROMPT_BUNDLE_ID = str(group_default(_PROMPT_DEFAULTS, 'bundle_id', ADJACENCY_PROMPT_BUNDLE_ID))
POST_IMAGE_BACKGROUND_DEFAULTS = load_graph_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_graph_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)

def _resolve_query(instance_seed: int, *, params: Mapping[str, Any]) -> _ResolvedQuery:
    """Resolve traversal axes while the public task owns BFS versus DFS semantics."""
    query_id, query_probs, task_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=SUPPORTED_ADJACENCY_TRAVERSAL_QUERY_IDS, default_query_id=SUPPORTED_ADJACENCY_TRAVERSAL_QUERY_IDS[0], task_id=TASK_ID, namespace='query_id')
    common_axes = resolve_label_node_axes(instance_seed=int(instance_seed), params=task_params, gen_defaults=_GEN_DEFAULTS, node_count_min=_DEFAULTS.node_count_min, node_count_max=_DEFAULTS.node_count_max, rng_namespace=TASK_ID)
    position_axis = _resolve_traversal_position_axis(instance_seed=int(instance_seed), task_params=task_params, node_count=int(common_axes.node.value))
    extra_axis = resolve_adjacency_int_axis(instance_seed=int(instance_seed), params=task_params, gen_defaults=_GEN_DEFAULTS, axis_name='extra_edge_count', default_min=_DEFAULTS.extra_edge_count_min, default_max=_DEFAULTS.extra_edge_count_max, rng_namespace=TASK_ID)
    source_index = _resolve_source_index(instance_seed=int(instance_seed), task_params=task_params, node_count=int(common_axes.node.value))
    return _ResolvedQuery(query_id=str(query_id), node_count=int(common_axes.node.value), traversal_position=int(position_axis.value), extra_edge_count=int(extra_axis.value), label_variant=str(common_axes.label.value), source_index=int(source_index), query_id_probabilities=dict(query_probs), node_count_probabilities=dict(common_axes.node.probabilities), traversal_position_probabilities=dict(position_axis.probabilities), extra_edge_count_probabilities=dict(extra_axis.probabilities), label_variant_probabilities=dict(common_axes.label.probabilities))

def _resolve_traversal_position_axis(*, instance_seed: int, task_params: Mapping[str, Any], node_count: int) -> Any:
    """Resolve the requested BFS/DFS visit position within the sampled node count."""
    position_max = min(int(node_count), configured_axis_max(task_params, _GEN_DEFAULTS, 'traversal_position', _DEFAULTS.traversal_position_max))
    return resolve_adjacency_int_axis(instance_seed=int(instance_seed), params=task_params, gen_defaults=_GEN_DEFAULTS, axis_name='traversal_position', default_min=_DEFAULTS.traversal_position_min, default_max=_DEFAULTS.traversal_position_max, rng_namespace=TASK_ID, support=tuple(range(configured_axis_min(task_params, _GEN_DEFAULTS, 'traversal_position', _DEFAULTS.traversal_position_min), int(position_max) + 1)))

def _resolve_source_index(*, instance_seed: int, task_params: Mapping[str, Any], node_count: int) -> int:
    """Resolve which row label is the traversal source."""
    explicit_source = task_params.get('source_index')
    if explicit_source is not None:
        source_index = int(explicit_source)
        if not 0 <= int(source_index) < int(node_count):
            raise ValueError('source_index is outside node support')
        return int(source_index)
    return int(
        uniform_choice(
            spawn_rng(int(instance_seed), f'{TASK_ID}:source_index'),
            tuple(range(int(node_count))),
        )
    )

def _ordered_visit_labels(sample: Any, source_label: str, query_id: str) -> tuple[str, ...]:
    """Return the traversal order selected by the task query."""
    if str(query_id) == 'bfs_kth_visit_label':
        return tuple(bfs_visit_order(sample.adjacency, str(source_label)))
    return tuple(dfs_visit_order(sample.adjacency, str(source_label)))

@register_task
class GraphOrderAdjacencyTraversalLabelTask:
    """Return the label at a requested BFS or DFS visit position."""
    task_id = 'task_graph__adjacency__traversal_kth_label'
    reasoning_operations = ('ranking', 'topology')
    domain = 'graph'
    supported_query_ids = SUPPORTED_ADJACENCY_TRAVERSAL_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate a reachable adjacency list and bind the visit-prefix annotation."""
        query = _resolve_query(int(instance_seed), params=params)
        labels = resolve_adjacency_labels(instance_seed=int(instance_seed), rng_namespace=TASK_ID, label_variant=str(query.label_variant), node_count=int(query.node_count), max_chars=int(group_default(_GEN_DEFAULTS, 'label_max_chars', _DEFAULTS.label_max_chars)))
        source_label = str(labels.labels[int(query.source_index)])
        sample = sample_reachable_directed_adjacency(instance_seed=int(instance_seed), rng_namespace=TASK_ID, labels=labels.labels, source_label=source_label, extra_edge_count=int(query.extra_edge_count))
        visit_order = _ordered_visit_labels(sample, source_label, str(query.query_id))
        if int(query.traversal_position) > len(visit_order):
            raise ValueError('traversal_position exceeds visited node count')
        answer_value = str(visit_order[int(query.traversal_position) - 1])
        annotation_labels = tuple((str(label) for label in visit_order[:int(query.traversal_position)]))
        render_artifacts = render_single_panel_artifacts(task_params=params, render_defaults=_RENDER_DEFAULTS, defaults=_DEFAULTS, background_defaults=POST_IMAGE_BACKGROUND_DEFAULTS, noise_defaults=POST_IMAGE_NOISE_DEFAULTS, instance_seed=int(instance_seed), render_panel=lambda base_image: render_adjacency_list_panel(sample=sample, base_image=base_image, title='Adjacency List', subtitle='Directed graph; read neighbors left to right.', **single_panel_render_kwargs(params=params, render_defaults=_RENDER_DEFAULTS, defaults=_DEFAULTS, instance_seed=int(instance_seed))))
        rendered = render_artifacts.rendered
        annotation_bboxes = row_label_bbox_sequence_value(rendered, annotation_labels)
        answer_gt = TypedValue(type='string', value=str(answer_value))
        annotation_gt = TypedValue(type='bbox_sequence', value=list(annotation_bboxes))
        prompt_artifacts = build_adjacency_prompt_artifacts(domain=self.domain, bundle_id=PROMPT_BUNDLE_ID, prompt_key=str(query.query_id), dynamic_slots={'object_description': 'a directed graph as an adjacency list', 'source_label': format_graph_prompt_label(source_label, label_variant=str(labels.label_variant)), 'traversal_position': int(query.traversal_position)}, instance_seed=int(instance_seed))
        node_entities = [{'entity_id': f'node_{label}', 'entity_kind': 'adjacency_row_label', 'label': str(label), 'neighbors': list(sample.adjacency.get(str(label), ())), 'bbox_xyxy': list(rendered.row_label_bboxes[str(label)]), 'is_source': bool(str(label) == source_label), 'is_answer': bool(str(label) == answer_value), 'is_in_annotation_prefix': bool(str(label) in set(annotation_labels))} for label in sample.labels]
        edge_entities = [{'entity_id': f'edge_{left}_{right}', 'entity_kind': 'adjacency_edge', 'source_label': str(left), 'target_label': str(right), 'directed': True} for left, right in sample.edges]
        query_params = {'query_id': str(query.query_id), 'query_id_probabilities': dict(query.query_id_probabilities), **label_node_query_params(query, labels), 'traversal_position': int(query.traversal_position), 'traversal_position_probabilities': dict(query.traversal_position_probabilities), 'source_label': str(source_label)}
        trace_payload = single_panel_trace_payload(task_id=TASK_ID, scene_id=SCENE_ID, query_id=str(query.query_id), prompt_bundle_id=PROMPT_BUNDLE_ID, prompt_artifacts=prompt_artifacts, render_artifacts=render_artifacts, entities=[*node_entities, *edge_entities], relations={'representation_variant': str(rendered.representation_variant), 'query_id': str(query.query_id), 'source_label': str(source_label), 'visit_order': list(visit_order), 'answer_label': str(answer_value), 'adjacency': {str(key): list(values) for key, values in sample.adjacency.items()}}, query_params=query_params, execution_fields={'source_label': str(source_label), 'answer': str(answer_value), 'annotation_labels': list(annotation_labels), 'visit_order': list(visit_order), 'node_count': int(query.node_count), 'edge_count': int(len(sample.edges)), 'label_variant': str(labels.label_variant)}, witness_symbolic={'type': 'adjacency_traversal_prefix', 'labels': list(annotation_labels), 'answer_label': str(answer_value)}, projected_annotation={'type': 'bbox_sequence', 'bbox_sequence': list(annotation_bboxes), 'pixel_bbox_sequence': list(annotation_bboxes)})
        return single_panel_task_output(prompt_artifacts=prompt_artifacts, answer_gt=answer_gt, annotation_gt=annotation_gt, render_artifacts=render_artifacts, trace_payload=trace_payload, scene_id=SCENE_ID, query_id=str(query.query_id))
__all__ = ['GraphOrderAdjacencyTraversalLabelTask']
