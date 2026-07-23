"""Simulate a deterministic state-transition diagram for a short input string."""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple
import networkx as nx
from ....core.sampling import uniform_choice
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
from ..shared.graph_sample_types import GraphTopologySample
from ..shared.graph_scene import RenderedGraphScene, projected_edge_label_bbox_annotation, projected_node_point_annotation, render_graph_scene
from ..shared.style import SUPPORTED_NODE_COLOR_NAMES
from ..shared.task_support import resolve_graph_named_variant, resolve_graph_render_params
from ..shared.visual_defaults import load_graph_scene_background_defaults, load_graph_scene_noise_defaults
from .shared.labels import sorted_state_label_tuple, state_labels
from .shared.prompts import PROMPT_BUNDLE_ID as AUTOMATON_PROMPT_BUNDLE_ID
from .shared.prompts import build_automaton_prompt_artifacts
from .shared.rendering import decorate_automaton_scene
from .shared.state import AUTOMATON_SYMBOLS, SUPPORTED_AUTOMATON_LAYOUT_VARIANTS
from .shared.topology import build_automaton_topology_sample
TASK_ID = 'task_graph__automaton__state_after_input_label'
SCENE_ID = 'automaton'
FINAL_STATE_QUERY_ID = 'final_state_label'
STEP_STATE_QUERY_ID = 'transition_step_state_label'
SUPPORTED_AUTOMATON_QUERY_IDS = (FINAL_STATE_QUERY_ID, STEP_STATE_QUERY_ID)
TASK_PROMPT_KEY = 'state_after_input_label_query'
OBJECT_DESCRIPTION = 'a deterministic state-transition diagram with a start arrow, double-ring accepting states, and transition labels'

@dataclass(frozen=True)
class _TaskDefaults:
    """Stable fallback defaults for automaton state-transition scenes."""
    state_count_min: int = 4
    state_count_max: int = 6
    input_length_min: int = 3
    input_length_max: int = 6
    transition_step_min: int = 2
    transition_step_max: int = 5
    distractor_edge_min: int = 2
    distractor_edge_max: int = 5
    canvas_width: int = 864
    canvas_height: int = 640
    outer_margin_px: int = 28
    panel_padding_px: int = 24
    panel_corner_radius_px: int = 20
    panel_title_font_size_px: int = 24
    node_shape_variant: str = 'circle'
    node_radius_min_px: int = 20
    node_radius_max_px: int = 25
    edge_width_px: int = 4
    arrow_length_px: int = 12
    arrow_width_px: int = 7
    node_border_width_px: int = 2
    label_font_size_px: int = 20
    node_color_name: str = 'blue'

@dataclass(frozen=True)
class _ResolvedQuery:
    """Resolved support and visual axes for one automaton simulation query."""
    query_id: str
    state_count: int
    input_length: int
    transition_step_count: int | None
    target_state_index: int
    distractor_edge_count: int
    layout_variant: str
    layout_transform_variant: str
    edge_routing_variant: str
    node_color_name: str
    query_id_probabilities: Dict[str, float]
    state_count_probabilities: Dict[str, float]
    input_length_probabilities: Dict[str, float]
    transition_step_count_probabilities: Dict[str, float]
    target_state_index_probabilities: Dict[str, float]
    distractor_edge_count_probabilities: Dict[str, float]
    layout_variant_probabilities: Dict[str, float]
    layout_transform_variant_probabilities: Dict[str, float]
    edge_routing_variant_probabilities: Dict[str, float]
    node_color_name_probabilities: Dict[str, float]

@dataclass(frozen=True)
class _AutomatonSample:
    """One trace-ready automaton and simulation path."""
    graph_sample: GraphTopologySample
    start_label: str
    accepting_labels: Tuple[str, ...]
    input_string: str
    query_step_count: int
    answer_label: str
    full_state_path_labels: Tuple[str, ...]
    annotation_state_labels: Tuple[str, ...]
    transition_labels_by_edge: Dict[Tuple[str, str], str]
    transition_function: Dict[str, Dict[str, str]]
    used_transition_edges: Tuple[Tuple[str, str], ...]
_DEFAULTS = _TaskDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults('graph', SCENE_ID, task_id=TASK_ID)
POST_IMAGE_BACKGROUND_DEFAULTS = load_graph_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_graph_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)
PROMPT_BUNDLE_ID = str(group_default(_PROMPT_DEFAULTS, 'bundle_id', AUTOMATON_PROMPT_BUNDLE_ID))

def _resolve_named_variant(instance_seed: int, *, params: Mapping[str, Any], explicit_key: str, weights_key: str, balance_flag_key: str, supported: Tuple[str, ...], namespace: str) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced automaton visual/query axis."""
    return resolve_graph_named_variant(spawn_rng(int(instance_seed), f'{TASK_ID}.{str(namespace)}'), params=params, gen_defaults=_GEN_DEFAULTS, explicit_key=str(explicit_key), weights_key=str(weights_key), balance_flag_key=str(balance_flag_key), supported=tuple((str(value) for value in supported)), instance_seed=int(instance_seed), task_id=TASK_ID, namespace=str(namespace))

def _uniform_probability(values: Sequence[int], *, selected: int | None=None) -> Dict[str, float]:
    """Return a uniform probability map over integer support."""
    return dict(uniform_probability_map(tuple((int(value) for value in values)), selected=selected))

def _resolve_query(instance_seed: int, *, params: Mapping[str, Any]) -> _ResolvedQuery:
    """Resolve query branch and scene axes for state simulation."""
    query_id, query_id_probabilities, task_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=SUPPORTED_AUTOMATON_QUERY_IDS, default_query_id=FINAL_STATE_QUERY_ID, task_id=TASK_ID, namespace='query_id')
    params = task_params
    state_count_min = int(params.get('state_count_min', group_default(_GEN_DEFAULTS, 'state_count_min', _DEFAULTS.state_count_min)))
    state_count_max = int(params.get('state_count_max', group_default(_GEN_DEFAULTS, 'state_count_max', _DEFAULTS.state_count_max)))
    state_count_support = tuple((int(value) for value in range(max(3, int(state_count_min)), int(state_count_max) + 1)))
    if not state_count_support:
        raise ValueError('no feasible state_count support exists for automaton simulation')
    explicit_state_count = params.get('state_count')
    if explicit_state_count is not None:
        state_count = int(explicit_state_count)
        if int(state_count) not in set(state_count_support):
            raise ValueError('state_count is outside feasible support')
    else:
        state_count = int(
            uniform_choice(
                spawn_rng(int(instance_seed), f'{TASK_ID}:state_count'),
                state_count_support,
            )
        )
    input_length_min = int(params.get('input_length_min', group_default(_GEN_DEFAULTS, 'input_length_min', _DEFAULTS.input_length_min)))
    input_length_max = int(params.get('input_length_max', group_default(_GEN_DEFAULTS, 'input_length_max', _DEFAULTS.input_length_max)))
    input_length_support = tuple((int(value) for value in range(max(1, int(input_length_min)), int(input_length_max) + 1)))
    if not input_length_support:
        raise ValueError('no feasible input_length support exists for automaton simulation')
    explicit_input_length = params.get('input_length')
    if explicit_input_length is not None:
        input_length = int(explicit_input_length)
        if int(input_length) not in set(input_length_support):
            raise ValueError('input_length is outside feasible support')
    else:
        input_length = int(
            uniform_choice(
                spawn_rng(int(instance_seed), f'{TASK_ID}:input_length'),
                input_length_support,
            )
        )
    transition_step_count: int | None = None
    transition_step_probabilities: Dict[str, float] = {}
    if str(query_id) == STEP_STATE_QUERY_ID:
        step_min = int(params.get('transition_step_min', group_default(_GEN_DEFAULTS, 'transition_step_min', _DEFAULTS.transition_step_min)))
        step_max = int(params.get('transition_step_max', group_default(_GEN_DEFAULTS, 'transition_step_max', _DEFAULTS.transition_step_max)))
        step_support = tuple((int(value) for value in range(max(1, int(step_min)), min(int(step_max), int(input_length)) + 1)))
        if not step_support:
            raise ValueError('no feasible transition_step support exists for automaton simulation')
        explicit_step = params.get('transition_step_count')
        if explicit_step is not None:
            transition_step_count = int(explicit_step)
            if int(transition_step_count) not in set(step_support):
                raise ValueError('transition_step_count is outside feasible support')
        else:
            transition_step_count = int(
                uniform_choice(
                    spawn_rng(int(instance_seed), f'{TASK_ID}:transition_step'),
                    step_support,
                )
            )
        transition_step_probabilities = _uniform_probability(tuple((int(value) for value in step_support)), selected=int(transition_step_count) if explicit_step is not None else None)
    target_support = tuple((int(value) for value in range(int(state_count))))
    explicit_target = params.get('target_state_index')
    if explicit_target is not None:
        target_state_index = int(explicit_target)
        if int(target_state_index) not in set(target_support):
            raise ValueError('target_state_index is outside feasible support')
    else:
        target_state_index = int(
            uniform_choice(
                spawn_rng(int(instance_seed), f'{TASK_ID}:target_state_index:{str(query_id)}'),
                target_support,
            )
        )
    distractor_min = int(params.get('distractor_edge_min', group_default(_GEN_DEFAULTS, 'distractor_edge_min', _DEFAULTS.distractor_edge_min)))
    distractor_max = int(params.get('distractor_edge_max', group_default(_GEN_DEFAULTS, 'distractor_edge_max', _DEFAULTS.distractor_edge_max)))
    distractor_support = tuple((int(value) for value in range(max(0, int(distractor_min)), max(int(distractor_min), int(distractor_max)) + 1)))
    explicit_distractors = params.get('distractor_edge_count')
    if explicit_distractors is not None:
        distractor_edge_count = int(explicit_distractors)
        if int(distractor_edge_count) not in set(distractor_support):
            raise ValueError('distractor_edge_count is outside feasible support')
    else:
        distractor_edge_count = int(
            uniform_choice(
                spawn_rng(int(instance_seed), f'{TASK_ID}:distractor_edge_count'),
                distractor_support,
            )
        )
    layout_variant, layout_variant_probabilities = _resolve_named_variant(int(instance_seed), params=params, explicit_key='layout_variant', weights_key='layout_variant_weights', balance_flag_key='balanced_layout_variant_sampling', supported=SUPPORTED_AUTOMATON_LAYOUT_VARIANTS, namespace='layout_variant')
    layout_transform_variant, layout_transform_variant_probabilities = _resolve_named_variant(int(instance_seed), params=params, explicit_key='layout_transform_variant', weights_key='layout_transform_variant_weights', balance_flag_key='balanced_layout_transform_variant_sampling', supported=('identity', 'mirror_left_right', 'mirror_up_down'), namespace='layout_transform_variant')
    edge_routing_variant, edge_routing_variant_probabilities = _resolve_named_variant(int(instance_seed), params=params, explicit_key='edge_routing_variant', weights_key='edge_routing_variant_weights', balance_flag_key='balanced_edge_routing_variant_sampling', supported=('straight', 'mixed_arc'), namespace='edge_routing_variant')
    node_color_name, node_color_name_probabilities = _resolve_named_variant(int(instance_seed), params=params, explicit_key='node_color_name', weights_key='node_color_name_weights', balance_flag_key='balanced_node_color_name_sampling', supported=SUPPORTED_NODE_COLOR_NAMES, namespace='node_color_name')
    return _ResolvedQuery(query_id=str(query_id), state_count=int(state_count), input_length=int(input_length), transition_step_count=int(transition_step_count) if transition_step_count is not None else None, target_state_index=int(target_state_index), distractor_edge_count=int(distractor_edge_count), layout_variant=str(layout_variant), layout_transform_variant=str(layout_transform_variant), edge_routing_variant=str(edge_routing_variant), node_color_name=str(node_color_name), query_id_probabilities=dict(query_id_probabilities), state_count_probabilities=_uniform_probability(state_count_support, selected=int(state_count) if explicit_state_count is not None else None), input_length_probabilities=_uniform_probability(input_length_support, selected=int(input_length) if explicit_input_length is not None else None), transition_step_count_probabilities=dict(transition_step_probabilities), target_state_index_probabilities=_uniform_probability(target_support, selected=int(target_state_index) if explicit_target is not None else None), distractor_edge_count_probabilities=_uniform_probability(distractor_support, selected=int(distractor_edge_count) if explicit_distractors is not None else None), layout_variant_probabilities=dict(layout_variant_probabilities), layout_transform_variant_probabilities=dict(layout_transform_variant_probabilities), edge_routing_variant_probabilities=dict(edge_routing_variant_probabilities), node_color_name_probabilities=dict(node_color_name_probabilities))

def _path_is_consistent(path: Sequence[int], input_symbols: Sequence[str]) -> bool:
    """Return whether a path can be represented by a deterministic transition table."""
    transition_by_key: Dict[Tuple[int, str], int] = {}
    target_by_state: Dict[int, set[int]] = {}
    for source, symbol, target in zip(path[:-1], input_symbols, path[1:]):
        source_i = int(source)
        target_i = int(target)
        if int(source_i) == int(target_i):
            return False
        key = (int(source_i), str(symbol))
        existing = transition_by_key.get(key)
        if existing is not None and int(existing) != int(target_i):
            return False
        target_set = target_by_state.setdefault(int(source_i), set())
        if int(target_i) in target_set and key not in transition_by_key:
            return False
        transition_by_key[key] = int(target_i)
        target_set.add(int(target_i))
    return True

def _sample_state_path(rng: random.Random, *, state_count: int, input_symbols: Sequence[str], answer_position: int, target_state_index: int) -> Tuple[int, ...]:
    """Sample a state path whose queried position is the requested answer state."""
    length = len(tuple(input_symbols))
    for _ in range(1000):
        path = [0]
        for position in range(1, int(length) + 1):
            if int(position) == int(answer_position):
                next_state = int(target_state_index)
            else:
                candidates = [int(value) for value in range(int(state_count)) if int(value) != int(path[-1])]
                next_state = int(rng.choice(candidates))
            path.append(int(next_state))
        if _path_is_consistent(path, input_symbols):
            return tuple((int(value) for value in path))
    raise ValueError('failed to sample a deterministic automaton path')

def _sample_transitions(rng: random.Random, *, state_count: int, input_symbols: Sequence[str], path: Sequence[int], distractor_edge_count: int) -> Dict[Tuple[int, str], int]:
    """Build required path transitions plus deterministic distractor transitions."""
    transitions: Dict[Tuple[int, str], int] = {}
    targets_by_state: Dict[int, set[int]] = {int(state): set() for state in range(int(state_count))}
    for source, symbol, target in zip(path[:-1], input_symbols, path[1:]):
        source_i = int(source)
        target_i = int(target)
        key = (int(source_i), str(symbol))
        existing = transitions.get(key)
        if existing is not None and int(existing) != int(target_i):
            raise ValueError('path transition conflict')
        transitions[key] = int(target_i)
        targets_by_state.setdefault(int(source_i), set()).add(int(target_i))
    candidates: list[Tuple[int, str]] = [(int(state), str(symbol)) for state in range(int(state_count)) for symbol in AUTOMATON_SYMBOLS if (int(state), str(symbol)) not in transitions]
    rng.shuffle(candidates)
    added = 0
    for state, symbol in candidates:
        if int(added) >= int(distractor_edge_count):
            break
        used_targets = targets_by_state.setdefault(int(state), set())
        target_candidates = [int(value) for value in range(int(state_count)) if int(value) != int(state) and int(value) not in used_targets]
        if not target_candidates:
            continue
        target = int(rng.choice(target_candidates))
        transitions[int(state), str(symbol)] = int(target)
        used_targets.add(int(target))
        added += 1
    return dict(transitions)

def _sample_automaton(rng: random.Random, *, query: _ResolvedQuery) -> _AutomatonSample:
    """Construct one deterministic state-transition diagram and simulation trace."""
    labels = state_labels(int(query.state_count))
    answer_position = int(query.input_length if str(query.query_id) == FINAL_STATE_QUERY_ID else int(query.transition_step_count or 1))
    input_symbols = tuple((str(rng.choice(AUTOMATON_SYMBOLS)) for _ in range(int(query.input_length))))
    path = _sample_state_path(rng, state_count=int(query.state_count), input_symbols=input_symbols, answer_position=int(answer_position), target_state_index=int(query.target_state_index))
    transitions = _sample_transitions(rng, state_count=int(query.state_count), input_symbols=input_symbols, path=path, distractor_edge_count=int(query.distractor_edge_count))
    graph = nx.DiGraph()
    graph.add_nodes_from(range(int(query.state_count)))
    for (source, _symbol), target in sorted(transitions.items()):
        graph.add_edge(int(source), int(target))
    transition_labels_by_edge: Dict[Tuple[str, str], str] = {}
    transition_function: Dict[str, Dict[str, str]] = {str(label): {} for label in labels}
    for (source, symbol), target in sorted(transitions.items()):
        source_label = str(labels[int(source)])
        target_label = str(labels[int(target)])
        transition_labels_by_edge[source_label, target_label] = str(symbol)
        transition_function[str(source_label)][str(symbol)] = str(target_label)
    accepting_count = int(rng.randint(1, min(2, int(query.state_count))))
    accepting_indices = set(rng.sample(range(int(query.state_count)), k=int(accepting_count)))
    if rng.random() < 0.5:
        accepting_indices.add(int(path[-1]))
    accepting_labels = sorted_state_label_tuple((str(labels[int(index)]) for index in accepting_indices))
    graph_sample = build_automaton_topology_sample(graph=graph, labels=labels, transition_labels_by_edge=transition_labels_by_edge)
    full_path_labels = tuple((str(labels[int(index)]) for index in path))
    annotation_labels = tuple((str(label) for label in full_path_labels[:int(answer_position) + 1]))
    used_edges = tuple(((str(labels[int(source)]), str(labels[int(target)])) for source, target in zip(path[:-1], path[1:])))
    return _AutomatonSample(graph_sample=graph_sample, start_label=str(labels[0]), accepting_labels=tuple((str(label) for label in accepting_labels)), input_string=''.join((str(symbol) for symbol in input_symbols)), query_step_count=int(answer_position), answer_label=str(full_path_labels[int(answer_position)]), full_state_path_labels=tuple((str(label) for label in full_path_labels)), annotation_state_labels=tuple((str(label) for label in annotation_labels)), transition_labels_by_edge={(str(left), str(right)): str(symbol) for (left, right), symbol in transition_labels_by_edge.items()}, transition_function={str(key): {str(symbol): str(target) for symbol, target in value.items()} for key, value in transition_function.items()}, used_transition_edges=tuple(((str(left), str(right)) for left, right in used_edges[:int(answer_position)])))

def _edge_label_bbox_entries(rendered_scene: RenderedGraphScene, edges: Sequence[Tuple[str, str]]) -> Tuple[list[list[int]], Dict[str, list[list[int]]]]:
    """Return transition-label bboxes for each used edge and all used-edge occurrences."""
    all_by_edge: Dict[str, list[list[int]]] = {}
    occurrence_bboxes: list[list[int]] = []
    for left, right in edges:
        projection = projected_edge_label_bbox_annotation(rendered_scene, (str(left), str(right)))
        boxes = [[int(round(float(value))) for value in bbox] for bbox in projection.get('bbox_set', [])]
        all_by_edge[f'{str(left)}->{str(right)}'] = list(boxes)
        if boxes:
            occurrence_bboxes.append(list(boxes[0]))
    return (occurrence_bboxes, dict(all_by_edge))

@register_task
class GraphRelationAutomatonStateSimulationLabelTask:
    """Simulate a state-transition diagram and answer the reached state label."""
    task_id = TASK_ID
    reasoning_operations = ('topology', 'state_update')
    domain = 'graph'
    supported_query_ids = SUPPORTED_AUTOMATON_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic automaton simulation instance."""
        query = _resolve_query(int(instance_seed), params=params)
        render_params = resolve_graph_render_params(params, instance_seed=int(instance_seed), task_id=TASK_ID, render_defaults=_RENDER_DEFAULTS, fallback_defaults=_DEFAULTS, node_color_name=str(query.node_color_name), node_shape_variant='circle', edge_routing_variant=str(query.edge_routing_variant))
        graph_rng = spawn_rng(int(instance_seed), f'{TASK_ID}.graph')
        last_error: Exception | None = None
        automaton = None
        rendered_scene = None
        image = None
        background_meta = {}
        post_noise_meta = {}
        transition_bbox_entries: list[list[int]] = []
        used_transition_bbox_by_edge: Dict[str, list[list[int]]] = {}
        for attempt in range(max(1, int(max_attempts))):
            try:
                automaton = _sample_automaton(graph_rng, query=query)
                background, background_meta = make_background_canvas(canvas_width=int(render_params.canvas_width), canvas_height=int(render_params.canvas_height), instance_seed=int(instance_seed), params=params, default_config=POST_IMAGE_BACKGROUND_DEFAULTS)
                node_style_by_label = {str(automaton.start_label): {'halo_rgb': tuple((int(value) for value in render_params.title_color_rgb)), 'halo_width_px': 3, 'halo_pad_px': 6}}
                rendered_scene = render_graph_scene(graph_sample=automaton.graph_sample, layout_variant=str(query.layout_variant), layout_transform_variant=str(query.layout_transform_variant), render_params=render_params, layout_seed=int(instance_seed + attempt), scene_title='State Transition Diagram', directed=True, base_image=background, edge_text_labels_by_label=automaton.transition_labels_by_edge, edge_text_label_font_size_px=max(15, int(render_params.label_font_size_px) - 3), node_style_by_label=node_style_by_label, layout_fallback_variants=('circular', 'shell', 'layered', 'spring'))
                decorate_automaton_scene(rendered_scene, start_label=str(automaton.start_label), accepting_labels=tuple((str(label) for label in automaton.accepting_labels)), render_params=render_params, layout_seed=int(instance_seed + attempt))
                transition_bbox_entries, used_transition_bbox_by_edge = _edge_label_bbox_entries(rendered_scene, automaton.used_transition_edges)
                if len(transition_bbox_entries) != len(automaton.used_transition_edges):
                    raise ValueError('not all used transition-label bboxes were rendered')
                image, post_noise_meta = apply_post_image_noise(rendered_scene.image, instance_seed=int(instance_seed), params=params, default_config=POST_IMAGE_NOISE_DEFAULTS)
                break
            except Exception as exc:
                last_error = exc
                continue
        else:
            raise RuntimeError('failed to generate automaton state-simulation instance') from last_error
        if automaton is None or rendered_scene is None or image is None:
            raise RuntimeError('failed to generate automaton state-simulation instance')
        prompt_artifacts = build_automaton_prompt_artifacts(domain=self.domain, bundle_id=PROMPT_BUNDLE_ID, task_key=TASK_PROMPT_KEY, prompt_query_key=str(query.query_id), dynamic_slots={'object_description': OBJECT_DESCRIPTION, 'input_string': str(automaton.input_string), 'transition_step_count': int(automaton.query_step_count)}, instance_seed=int(instance_seed))
        annotation_projection = projected_node_point_annotation(rendered_scene, automaton.annotation_state_labels)
        annotation_path = [[int(round(float(point[0]))), int(round(float(point[1])))] for point in annotation_projection['pixel_point_sequence']]
        if len(annotation_path) != len(automaton.annotation_state_labels):
            raise RuntimeError('automaton annotation path projection is incomplete')
        answer_gt = TypedValue(type='string', value=str(automaton.answer_label))
        annotation_gt = TypedValue(type='point_sequence', value=list(annotation_path))
        answer_label = str(automaton.answer_label)
        path_label_set = set((str(label) for label in automaton.annotation_state_labels))
        edge_usage_counts: Dict[str, int] = {}
        for left, right in automaton.used_transition_edges:
            key = f'{str(left)}->{str(right)}'
            edge_usage_counts[key] = int(edge_usage_counts.get(key, 0)) + 1
        node_entities = [{'entity_id': f'state_{node.label}', 'entity_kind': 'automaton_state', 'label': str(node.label), 'is_start_state': bool(str(node.label) == str(automaton.start_label)), 'is_accepting_state': bool(str(node.label) in set(automaton.accepting_labels)), 'is_answer_state': bool(str(node.label) == str(answer_label)), 'is_in_annotation_path': bool(str(node.label) in path_label_set), 'path_positions': [int(index) for index, label in enumerate(automaton.annotation_state_labels) if str(label) == str(node.label)], 'center_px': list(node.center_xy), 'bbox_xyxy': list(node.bbox_xyxy), 'successors': list(automaton.graph_sample.successors_by_label[str(node.label)]), 'predecessors': list(automaton.graph_sample.predecessors_by_label[str(node.label)])} for node in rendered_scene.nodes]
        edge_entities = [{'entity_id': str(edge.edge_id), 'entity_kind': 'automaton_transition', 'source_state_label': str(edge.node_u_label), 'target_state_label': str(edge.node_v_label), 'transition_symbol': str(automaton.transition_labels_by_edge[str(edge.node_u_label), str(edge.node_v_label)]), 'is_used_transition': bool(edge_usage_counts.get(f'{str(edge.node_u_label)}->{str(edge.node_v_label)}', 0) > 0), 'used_transition_count': int(edge_usage_counts.get(f'{str(edge.node_u_label)}->{str(edge.node_v_label)}', 0)), 'segment_px': [list(edge.segment_px[0]), list(edge.segment_px[1])], 'route_variant': str(edge.route_variant), 'control_px': list(edge.control_px) if edge.control_px is not None else None, 'edge_label_bbox_xyxy': list(edge.edge_label_bbox_xyxy) if edge.edge_label_bbox_xyxy is not None else None} for edge in rendered_scene.edges]
        transition_entries = [{'source_state': str(left), 'target_state': str(right), 'symbol': str(symbol)} for (left, right), symbol in automaton.transition_labels_by_edge.items()]
        trace_payload = {'scene_ir': {'scene_kind': 'automaton_state_transition_simulation', 'entities': [*node_entities, *edge_entities], 'relations': {'query_id': str(query.query_id), 'simulation_rule': 'start_at_start_state_and_follow_one_visible_transition_label_per_input_symbol', 'input_string': str(automaton.input_string), 'start_state_label': str(automaton.start_label), 'accepting_state_labels': list(automaton.accepting_labels), 'answer_state_label': str(automaton.answer_label), 'query_step_count': int(automaton.query_step_count), 'full_state_path_labels': list(automaton.full_state_path_labels), 'annotation_state_path_labels': list(automaton.annotation_state_labels), 'used_transition_edges': [list(edge) for edge in automaton.used_transition_edges], 'transition_function': dict(automaton.transition_function), 'transition_labels_by_edge': list(transition_entries), 'query_id_probabilities': dict(query.query_id_probabilities), 'state_count_probabilities': dict(query.state_count_probabilities), 'input_length_probabilities': dict(query.input_length_probabilities), 'transition_step_count_probabilities': dict(query.transition_step_count_probabilities), 'target_state_index_probabilities': dict(query.target_state_index_probabilities), 'distractor_edge_count_probabilities': dict(query.distractor_edge_count_probabilities)}, 'frames': {'pixel': {'origin': [0.0, 0.0], 'x_positive': 'right', 'y_positive': 'down'}, 'panels': dict(rendered_scene.panel_geometry)}}, 'query_spec': {'query_id': str(query.query_id), 'template_id': PROMPT_BUNDLE_ID, 'prompt_variant': dict(prompt_artifacts.prompt_variant), 'prompt_variant_active_key': str(prompt_artifacts.prompt_variant_active_key), 'prompt_variants': dict(prompt_artifacts.prompt_variants_for_trace), 'params': {'query_id': str(query.query_id), 'internal_query_id': str(query.query_id), 'query_id_probabilities': dict(query.query_id_probabilities), 'state_count': int(query.state_count), 'state_count_probabilities': dict(query.state_count_probabilities), 'input_length': int(query.input_length), 'input_length_probabilities': dict(query.input_length_probabilities), 'transition_step_count': int(automaton.query_step_count), 'transition_step_count_probabilities': dict(query.transition_step_count_probabilities), 'target_state_index': int(query.target_state_index), 'target_state_index_probabilities': dict(query.target_state_index_probabilities), 'distractor_edge_count': int(query.distractor_edge_count), 'distractor_edge_count_probabilities': dict(query.distractor_edge_count_probabilities), 'input_string': str(automaton.input_string), 'answer_state_label': str(automaton.answer_label), 'start_state_label': str(automaton.start_label), 'accepting_state_labels': list(automaton.accepting_labels), 'layout_variant': str(query.layout_variant), 'layout_variant_probabilities': dict(query.layout_variant_probabilities), 'layout_transform_variant': str(query.layout_transform_variant), 'layout_transform_variant_probabilities': dict(query.layout_transform_variant_probabilities), 'edge_routing_variant': str(query.edge_routing_variant), 'edge_routing_variant_probabilities': dict(query.edge_routing_variant_probabilities), 'node_color_name': str(query.node_color_name), 'node_color_name_probabilities': dict(query.node_color_name_probabilities)}}, 'render_spec': {'canvas_size': list(rendered_scene.panel_geometry['canvas_size']), 'coord_space': 'pixel', 'panel_geometry': dict(rendered_scene.panel_geometry), 'style': {'node_color_name': str(query.node_color_name), 'theme_tone': str(render_params.theme_tone), 'panel_style_variant': str(render_params.panel_style_variant), 'background_color_rgb': list(render_params.background_color_rgb), 'panel_fill_rgb': list(render_params.panel_fill_rgb), 'panel_border_rgb': list(render_params.panel_border_rgb), 'title_color_rgb': list(render_params.title_color_rgb), 'edge_color_rgb': list(render_params.edge_color_rgb), 'node_fill_rgb': list(render_params.node_fill_rgb), 'node_border_rgb': list(render_params.node_border_rgb), 'label_text_rgb': list(render_params.label_text_rgb), 'label_stroke_rgb': list(render_params.label_stroke_rgb), 'node_shape_variant': 'circle', 'node_radius_px': int(render_params.node_radius_px), 'edge_width_px': int(render_params.edge_width_px), 'edge_routing_variant': str(rendered_scene.edge_routing_variant), 'arrow_length_px': int(render_params.arrow_length_px), 'arrow_width_px': int(render_params.arrow_width_px), 'node_border_width_px': int(render_params.node_border_width_px), 'label_font_size_px': int(render_params.label_font_size_px), 'resolved_label_font_size_px': int(rendered_scene.resolved_label_font_size_px), 'label_stroke_width_px': int(rendered_scene.resolved_label_stroke_width_px), 'font_family': str(render_params.font_family or ''), 'font_asset': dict(render_params.font_asset) if isinstance(render_params.font_asset, Mapping) else {}, 'font_asset_version': str(render_params.font_asset_version or ''), 'font_exclusion_reason': str(render_params.font_exclusion_reason), 'context_text_elements': list(rendered_scene.panel_geometry.get('context_text_elements', [])), 'automaton_accepting_state_glyph': 'double_inner_ring', 'automaton_start_state_glyph': 'incoming_start_arrow', 'transition_labels_by_edge': list(transition_entries), 'background_meta': dict(background_meta), 'post_image_noise_meta': dict(post_noise_meta)}}, 'render_map': {'image_id': 'img0', 'anchors': {}}, 'execution_trace': {'query_id': str(query.query_id), 'internal_query_id': str(query.query_id), 'query_id_probabilities': dict(query.query_id_probabilities), 'scene_variant': str(rendered_scene.layout_variant), 'question_format': str(query.query_id), 'state_count': int(query.state_count), 'edge_count': int(automaton.graph_sample.edge_count), 'input_length': int(query.input_length), 'input_string': str(automaton.input_string), 'transition_step_count': int(automaton.query_step_count), 'answer': str(automaton.answer_label), 'answer_state_label': str(automaton.answer_label), 'start_state_label': str(automaton.start_label), 'accepting_state_labels': list(automaton.accepting_labels), 'full_state_path_labels': list(automaton.full_state_path_labels), 'annotation_state_path_labels': list(automaton.annotation_state_labels), 'used_transition_edges': [list(edge) for edge in automaton.used_transition_edges], 'transition_function': dict(automaton.transition_function), 'transition_labels_by_edge': list(transition_entries), 'used_transition_label_bboxes': list(transition_bbox_entries), 'used_transition_bbox_by_edge': dict(used_transition_bbox_by_edge), 'target_state_index': int(query.target_state_index), 'target_state_index_probabilities': dict(query.target_state_index_probabilities), 'state_count_probabilities': dict(query.state_count_probabilities), 'input_length_probabilities': dict(query.input_length_probabilities), 'transition_step_count_probabilities': dict(query.transition_step_count_probabilities), 'distractor_edge_count': int(query.distractor_edge_count), 'distractor_edge_count_probabilities': dict(query.distractor_edge_count_probabilities), 'layout_variant_requested': str(query.layout_variant), 'layout_variant_used': str(rendered_scene.layout_variant), 'layout_transform_variant': str(rendered_scene.layout_transform_variant), 'edge_routing_variant': str(rendered_scene.edge_routing_variant), 'node_color_name': str(query.node_color_name), 'crossing_count': int(rendered_scene.crossing_count)}, 'witness_symbolic': {'type': 'state_path', 'state_path_labels': list(automaton.annotation_state_labels), 'answer_state_label': str(automaton.answer_label)}, 'projected_annotation': {'type': 'point_sequence', 'point_sequence': list(annotation_path), 'pixel_point_sequence': list(annotation_path), 'pixel_bbox_set': list(annotation_projection['pixel_bbox_set']), 'used_transition_label_bboxes': list(transition_bbox_entries)}}
        return TaskOutput(prompt=str(prompt_artifacts.prompt), answer_gt=answer_gt, annotation_gt=annotation_gt, image=image, image_id='img0', trace_payload=trace_payload, task_versions=default_task_versions(), scene_id=SCENE_ID, query_id=str(query.query_id), prompt_variants=dict(prompt_artifacts.prompt_variants))
__all__ = ['GraphRelationAutomatonStateSimulationLabelTask']
