"""Count directed reciprocal-pair states from an adjacency matrix."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ....core.visual.background import make_background_canvas
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from ...shared.deterministic_sampling import uniform_probability_map
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from .shared.annotations import mirrored_pair_cell_point_pair_artifacts
from .shared.prompts import PROMPT_BUNDLE_ID as ADJACENCY_PROMPT_BUNDLE_ID
from .shared.prompts import build_adjacency_prompt_artifacts
from .shared.rendering import render_adjacency_matrix_panel
from .shared.sampling import edge_set_from_adjacency, resolve_adjacency_labels
from .shared.state import AdjacencyGraphSample, matrix_cell_key, SCENE_ID
from ..shared.graph_sample_types import SUPPORTED_NODE_LINK_LABEL_VARIANTS
from ..shared.task_support import graph_int_support, resolve_graph_named_variant
from ..shared.visual_defaults import load_graph_scene_background_defaults, load_graph_scene_noise_defaults
TASK_ID = 'task_graph__adjacency__directed_pair_reciprocity_count'
SCENE_ID = 'adjacency'
PUBLIC_QUERY_ID = 'single'
PROMPT_KEY = 'mutual_pair_count'
TARGET_PAIR_STATE = 'mutual'
SUPPORTED_ADJACENCY_PAIR_RECIPROCITY_QUERY_IDS: Tuple[str, ...] = (PUBLIC_QUERY_ID,)

@dataclass(frozen=True)
class _TaskDefaults:
    """Stable fallback defaults for directed adjacency-matrix reciprocity counts."""
    node_count_min: int = 5
    node_count_max: int = 7
    target_count_min: int = 0
    target_count_max: int = 6
    label_max_chars: int = 5
    label_variant: str = 'letters'
    canvas_width: int = 900
    canvas_height: int = 640
    label_font_size_px: int = 19

@dataclass(frozen=True)
class _ResolvedQuery:
    """Resolved support axes for one adjacency reciprocity-count instance."""
    query_id: str
    node_count: int
    target_count: int
    label_variant: str
    query_id_probabilities: Dict[str, float]
    node_count_probabilities: Dict[str, float]
    target_count_probabilities: Dict[str, float]
    label_variant_probabilities: Dict[str, float]

@dataclass(frozen=True)
class _PairState:
    """Directed state for one unordered off-diagonal node pair."""
    left: str
    right: str
    forward: bool
    reverse: bool
    state: str
_DEFAULTS = _TaskDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults('graph', SCENE_ID, task_id=TASK_ID)
PROMPT_BUNDLE_ID = str(group_default(_PROMPT_DEFAULTS, 'bundle_id', ADJACENCY_PROMPT_BUNDLE_ID))
POST_IMAGE_BACKGROUND_DEFAULTS = load_graph_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_graph_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)

def _query_state(query_id: str) -> str:
    if str(query_id) in {PUBLIC_QUERY_ID, PROMPT_KEY}:
        return TARGET_PAIR_STATE
    raise ValueError(f'unsupported reciprocity query_id: {query_id}')

def _resolve_query(instance_seed: int, *, params: Mapping[str, Any]) -> _ResolvedQuery:
    """Resolve reciprocity task axes while the public task owns the target pair state."""
    query_id, query_probs, task_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=SUPPORTED_ADJACENCY_PAIR_RECIPROCITY_QUERY_IDS, default_query_id=SUPPORTED_ADJACENCY_PAIR_RECIPROCITY_QUERY_IDS[0], task_id=TASK_ID, namespace='query_id')
    label_rng = spawn_rng(int(instance_seed), f'{TASK_ID}.label_variant')
    label_variant, label_probs = resolve_graph_named_variant(label_rng, params=task_params, gen_defaults=_GEN_DEFAULTS, explicit_key='label_variant', weights_key='label_variant_weights', balance_flag_key='balanced_label_variant_sampling', supported=SUPPORTED_NODE_LINK_LABEL_VARIANTS, instance_seed=int(instance_seed), task_id=TASK_ID, namespace='label_variant')
    node_support = graph_int_support(task_params, _GEN_DEFAULTS, 'node_count', _DEFAULTS.node_count_min, _DEFAULTS.node_count_max)
    explicit_node = task_params.get('node_count')
    if explicit_node is not None:
        node_count = int(explicit_node)
        if int(node_count) not in set(node_support):
            raise ValueError('node_count is outside configured support')
    else:
        node_rng = spawn_rng(int(instance_seed), f'{TASK_ID}.node_count')
        node_count = int(node_rng.choice(tuple((int(value) for value in node_support))))
    max_pairs = int(node_count) * (int(node_count) - 1) // 2
    configured_target_support = graph_int_support(task_params, _GEN_DEFAULTS, 'target_count', _DEFAULTS.target_count_min, _DEFAULTS.target_count_max)
    target_support = tuple((int(value) for value in configured_target_support if int(value) <= int(max_pairs)))
    if not target_support:
        raise ValueError('target_count support is empty for adjacency reciprocal-pair count')
    explicit_target = task_params.get('target_count')
    if explicit_target is not None:
        target_count = int(explicit_target)
        if int(target_count) not in set(target_support):
            raise ValueError('target_count is outside feasible support')
    else:
        target_rng = spawn_rng(int(instance_seed), f'{TASK_ID}.target_count')
        target_count = int(target_rng.choice(tuple((int(value) for value in target_support))))
    return _ResolvedQuery(query_id=str(query_id), node_count=int(node_count), target_count=int(target_count), label_variant=str(label_variant), query_id_probabilities=dict(query_probs), node_count_probabilities=uniform_probability_map(node_support, selected=int(node_count) if explicit_node is not None else None), target_count_probabilities=uniform_probability_map(target_support, selected=int(target_count) if explicit_target is not None else None), label_variant_probabilities=dict(label_probs))

def _sample_reciprocity_matrix(*, instance_seed: int, labels: Tuple[str, ...], query_id: str, target_count: int) -> Tuple[AdjacencyGraphSample, Tuple[_PairState, ...], Tuple[Tuple[str, str], ...]]:
    """Build a directed matrix with an exact number of task-selected pair states."""
    target_state = _query_state(str(query_id))
    rng = spawn_rng(int(instance_seed), f'{TASK_ID}.reciprocity_matrix.{target_state}.{int(target_count)}')
    label_order = tuple((str(label) for label in labels))
    pairs = [(label_order[i], label_order[j]) for i in range(len(label_order)) for j in range(i + 1, len(label_order))]
    rng.shuffle(pairs)
    target_pairs = set((tuple(pair) for pair in pairs[:int(target_count)]))
    edges: set[Tuple[str, str]] = set()
    states: List[_PairState] = []
    for pair in pairs:
        left, right = (str(pair[0]), str(pair[1]))
        if tuple(pair) in target_pairs:
            if target_state == 'mutual':
                forward = True
                reverse = True
                state = 'mutual'
            else:
                forward = bool(rng.randint(0, 1))
                reverse = not forward
                state = 'one_way'
        elif target_state == 'mutual':
            state = str(rng.choice(('one_way', 'absent')))
            if state == 'one_way':
                forward = bool(rng.randint(0, 1))
                reverse = not forward
            else:
                forward = False
                reverse = False
        else:
            state = str(rng.choice(('mutual', 'absent')))
            forward = state == 'mutual'
            reverse = state == 'mutual'
        if forward:
            edges.add((left, right))
        if reverse:
            edges.add((right, left))
        states.append(_PairState(left=left, right=right, forward=bool(forward), reverse=bool(reverse), state=str(state)))
    adjacency_lists: Dict[str, List[str]] = {label: [] for label in label_order}
    label_index = {label: index for index, label in enumerate(label_order)}
    for source, target in sorted(edges, key=lambda edge: (label_index[str(edge[0])], label_index[str(edge[1])])):
        adjacency_lists[str(source)].append(str(target))
    sample = AdjacencyGraphSample(labels=label_order, directed=True, adjacency={label: tuple(targets) for label, targets in adjacency_lists.items()}, edges=edge_set_from_adjacency(adjacency_lists, directed=True), weights={})
    ordered_states = tuple(sorted(states, key=lambda state: (label_index[str(state.left)], label_index[str(state.right)])))
    counted_pairs = tuple(((state.left, state.right) for state in ordered_states if state.state == target_state))
    return (sample, ordered_states, counted_pairs)

@register_task
class GraphCountingAdjacencyDirectedPairReciprocityCountTask:
    """Count mutual unordered node pairs in a directed adjacency matrix."""
    task_id = 'task_graph__adjacency__directed_pair_reciprocity_count'
    reasoning_operations = ('filtering', 'counting', 'logical_composition', 'topology')
    domain = 'graph'
    supported_query_ids = SUPPORTED_ADJACENCY_PAIR_RECIPROCITY_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate a mutual-pair count and bind one segment per counted pair."""
        query = _resolve_query(int(instance_seed), params=params)
        labels = resolve_adjacency_labels(instance_seed=int(instance_seed), rng_namespace=TASK_ID, label_variant=str(query.label_variant), node_count=int(query.node_count), max_chars=int(group_default(_GEN_DEFAULTS, 'label_max_chars', _DEFAULTS.label_max_chars)))
        sample, pair_states, counted_pairs = _sample_reciprocity_matrix(instance_seed=int(instance_seed), labels=tuple(labels.labels), query_id=str(query.query_id), target_count=int(query.target_count))
        canvas_width = int(params.get('canvas_width', group_default(_RENDER_DEFAULTS, 'canvas_width', _DEFAULTS.canvas_width)))
        canvas_height = int(params.get('canvas_height', group_default(_RENDER_DEFAULTS, 'canvas_height', _DEFAULTS.canvas_height)))
        base_image, background_meta = make_background_canvas(canvas_width=int(canvas_width), canvas_height=int(canvas_height), instance_seed=int(instance_seed), params=params, default_config=POST_IMAGE_BACKGROUND_DEFAULTS)
        rendered = render_adjacency_matrix_panel(sample=sample, base_image=base_image, title='Directed Adjacency Matrix', subtitle='Rows point to columns.', weighted=False, font_size_px=int(params.get('label_font_size_px', group_default(_RENDER_DEFAULTS, 'label_font_size_px', _DEFAULTS.label_font_size_px))), layout_seed=int(instance_seed), font_family=params.get('font_family'), context_text_probability=float(params.get('context_text_probability', group_default(_RENDER_DEFAULTS, 'context_text_probability', 0.35))))
        image, post_noise_meta = apply_post_image_noise(rendered.image, instance_seed=int(instance_seed), params=params, default_config=POST_IMAGE_NOISE_DEFAULTS)
        annotation_artifacts, annotation_cell_keys = mirrored_pair_cell_point_pair_artifacts(rendered, counted_pairs)
        answer_gt = TypedValue(type='integer', value=int(len(counted_pairs)))
        prompt_artifacts = build_adjacency_prompt_artifacts(domain=self.domain, bundle_id=PROMPT_BUNDLE_ID, prompt_key=PROMPT_KEY, dynamic_slots={'object_description': 'a directed graph as an adjacency matrix'}, instance_seed=int(instance_seed))
        node_entities = [{'entity_id': f'node_{label}', 'entity_kind': 'adjacency_row_label', 'label': str(label), 'neighbors': list(sample.adjacency.get(str(label), ())), 'bbox_xyxy': list(rendered.row_label_bboxes[str(label)])} for label in sample.labels]
        edge_entities = [{'entity_id': f'edge_{left}_{right}', 'entity_kind': 'adjacency_edge', 'source_label': str(left), 'target_label': str(right), 'directed': True} for left, right in sample.edges]
        pair_state_records = [{'left_label': str(state.left), 'right_label': str(state.right), 'forward_cell_key': matrix_cell_key(str(state.left), str(state.right)), 'reverse_cell_key': matrix_cell_key(str(state.right), str(state.left)), 'forward_edge_present': bool(state.forward), 'reverse_edge_present': bool(state.reverse), 'pair_state': str(state.state), 'is_counted': bool((str(state.left), str(state.right)) in set(counted_pairs))} for state in pair_states]
        trace_payload = {'scene_ir': {'task_id': TASK_ID, 'scene_id': SCENE_ID, 'scene_kind': 'adjacency', 'entities': [*node_entities, *edge_entities], 'relations': {'representation_variant': str(rendered.representation_variant), 'query_id': str(query.query_id), 'directed': True, 'adjacency': {str(key): list(values) for key, values in sample.adjacency.items()}, 'pair_states': list(pair_state_records), 'counted_pairs': [list(pair) for pair in counted_pairs]}, 'frames': {'pixel': {'origin': [0.0, 0.0], 'x_positive': 'right', 'y_positive': 'down'}, 'panels': dict(rendered.panel_geometry)}}, 'query_spec': {'task_id': TASK_ID, 'query_id': str(query.query_id), 'template_id': PROMPT_BUNDLE_ID, 'prompt_variant': dict(prompt_artifacts.prompt_variant), 'prompt_variant_active_key': str(prompt_artifacts.prompt_variant_active_key), 'prompt_variants': dict(prompt_artifacts.prompt_variants_for_trace), 'params': {'query_id': str(query.query_id), 'query_id_probabilities': dict(query.query_id_probabilities), 'node_count': int(query.node_count), 'node_count_probabilities': dict(query.node_count_probabilities), 'target_count': int(query.target_count), 'target_count_probabilities': dict(query.target_count_probabilities), 'label_variant': str(labels.label_variant), 'label_variant_probabilities': dict(query.label_variant_probabilities), 'label_source_kind': str(labels.label_source_kind), 'label_bucket': str(labels.label_bucket), 'label_manifest': str(labels.label_manifest), 'label_filter': dict(labels.label_filter), 'label_bucket_probabilities': dict(labels.label_bucket_probabilities)}}, 'render_spec': {'canvas_size': [int(canvas_width), int(canvas_height)], 'coord_space': 'pixel', 'panel_geometry': dict(rendered.panel_geometry), 'style': {'representation_variant': str(rendered.representation_variant), 'background_meta': dict(background_meta), 'post_image_noise_meta': dict(post_noise_meta), **dict(rendered.style_meta)}}, 'render_map': {'image_id': 'img0', 'anchors': {}}, 'execution_trace': {'task_id': TASK_ID, 'scene_id': SCENE_ID, 'query_id': str(query.query_id), 'representation_variant': str(rendered.representation_variant), 'answer': int(len(counted_pairs)), 'node_count': int(query.node_count), 'edge_count': int(len(sample.edges)), 'directed': True, 'label_variant': str(labels.label_variant), 'target_pair_state': _query_state(str(query.query_id)), 'counted_pairs': [list(pair) for pair in counted_pairs], 'annotation_cell_keys': list(annotation_cell_keys), 'pair_states': list(pair_state_records), 'adjacency': {str(key): list(values) for key, values in sample.adjacency.items()}}, 'witness_symbolic': {'type': 'directed_pair_reciprocity_set', 'target_pair_state': _query_state(str(query.query_id)), 'pairs': [list(pair) for pair in counted_pairs], 'cell_keys': list(annotation_cell_keys)}, 'projected_annotation': {**dict(annotation_artifacts.projected_annotation)}}
        return TaskOutput(prompt=str(prompt_artifacts.prompt), answer_gt=answer_gt, annotation_gt=annotation_artifacts.annotation_gt, image=image, image_id='img0', trace_payload=trace_payload, task_versions=default_task_versions(), scene_id=SCENE_ID, query_id=str(query.query_id), prompt_variants=dict(prompt_artifacts.prompt_variants))
__all__ = ['GraphCountingAdjacencyDirectedPairReciprocityCountTask']
