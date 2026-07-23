"""Count states with nondeterministic outgoing transitions in an automaton."""
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
from ...shared.output_metadata import default_task_versions
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
TASK_ID = 'task_graph__automaton__nondeterministic_state_count'
SCENE_ID = 'automaton'
QUERY_ID = 'nondeterministic_state_count'
EPSILON_SYMBOL = 'eps'
TASK_PROMPT_KEY = 'nondeterministic_state_count_query'
OBJECT_DESCRIPTION = 'a state-transition diagram with a start arrow, double-ring accepting states, and visible transition labels, including possible eps labels'

@dataclass(frozen=True)
class _TaskDefaults:
    """Stable fallback defaults for nondeterministic-state count scenes."""
    state_count_min: int = 4
    state_count_max: int = 6
    target_count_min: int = 0
    target_count_max: int = 5
    deterministic_edge_probability: float = 0.68
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
    """Resolved support and visual axes for one nondeterministic-state count query."""
    query_id: str
    state_count: int
    target_count: int
    deterministic_edge_probability: float
    layout_variant: str
    layout_transform_variant: str
    edge_routing_variant: str
    node_color_name: str
    query_id_probabilities: Dict[str, float]
    state_count_probabilities: Dict[str, float]
    target_count_probabilities: Dict[str, float]
    layout_variant_probabilities: Dict[str, float]
    layout_transform_variant_probabilities: Dict[str, float]
    edge_routing_variant_probabilities: Dict[str, float]
    node_color_name_probabilities: Dict[str, float]

@dataclass(frozen=True)
class _AutomatonNondeterminismSample:
    """One trace-ready automaton with nondeterministic-state witnesses."""
    graph_sample: Any
    start_label: str
    accepting_labels: Tuple[str, ...]
    transition_labels_by_edge: Dict[Tuple[str, str], str]
    transition_function: Dict[str, Dict[str, Tuple[str, ...]]]
    nondeterministic_state_labels: Tuple[str, ...]
    deterministic_state_labels: Tuple[str, ...]
    witness_edges_by_state: Dict[str, Tuple[Tuple[str, str, str], ...]]
    nondeterminism_reason_by_state: Dict[str, str]
_DEFAULTS = _TaskDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults('graph', SCENE_ID, task_id=TASK_ID)
POST_IMAGE_BACKGROUND_DEFAULTS = load_graph_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_graph_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)
PROMPT_BUNDLE_ID = str(group_default(_PROMPT_DEFAULTS, 'bundle_id', AUTOMATON_PROMPT_BUNDLE_ID))

def _resolve_named_variant(instance_seed: int, *, params: Mapping[str, Any], explicit_key: str, weights_key: str, balance_flag_key: str, supported: Tuple[str, ...], namespace: str) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced automaton visual/query axis."""
    return resolve_graph_named_variant(spawn_rng(int(instance_seed), f'{TASK_ID}.{str(namespace)}'), params=params, gen_defaults=_GEN_DEFAULTS, explicit_key=str(explicit_key), weights_key=str(weights_key), balance_flag_key=str(balance_flag_key), supported=tuple((str(value) for value in supported)), instance_seed=int(instance_seed), task_id=TASK_ID, namespace=str(namespace))

def _uniform_probability(values: Sequence[int | str], *, selected: int | str | None=None) -> Dict[str, float]:
    """Return a uniform probability map over support values."""
    return dict(uniform_probability_map(tuple(values), selected=selected))

def _resolve_query(instance_seed: int, *, params: Mapping[str, Any]) -> _ResolvedQuery:
    """Resolve sampling axes for the nondeterministic-state count objective."""
    query_id = QUERY_ID
    query_id_probabilities = {QUERY_ID: 1.0}
    state_count_min = int(params.get('state_count_min', group_default(_GEN_DEFAULTS, 'state_count_min', _DEFAULTS.state_count_min)))
    state_count_max = int(params.get('state_count_max', group_default(_GEN_DEFAULTS, 'state_count_max', _DEFAULTS.state_count_max)))
    state_count_support = tuple((int(value) for value in range(max(3, int(state_count_min)), int(state_count_max) + 1)))
    if not state_count_support:
        raise ValueError('no feasible state_count support exists for automaton nondeterministic-state count')
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
    target_count_min = int(params.get('target_count_min', group_default(_GEN_DEFAULTS, 'target_count_min', _DEFAULTS.target_count_min)))
    target_count_max = int(params.get('target_count_max', group_default(_GEN_DEFAULTS, 'target_count_max', _DEFAULTS.target_count_max)))
    target_support = tuple((int(value) for value in range(max(0, int(target_count_min)), min(int(target_count_max), int(state_count)) + 1)))
    if not target_support:
        raise ValueError('no feasible target_count support exists for automaton nondeterministic-state count')
    explicit_target_count = params.get('target_count', params.get('answer_count'))
    if explicit_target_count is not None:
        target_count = int(explicit_target_count)
        if int(target_count) not in set(target_support):
            raise ValueError('target_count is outside feasible support')
    else:
        target_count = int(
            uniform_choice(
                spawn_rng(int(instance_seed), f'{TASK_ID}:target_count'),
                target_support,
            )
        )
    layout_variant, layout_variant_probabilities = _resolve_named_variant(int(instance_seed), params=params, explicit_key='layout_variant', weights_key='layout_variant_weights', balance_flag_key='balanced_layout_variant_sampling', supported=SUPPORTED_AUTOMATON_LAYOUT_VARIANTS, namespace='layout_variant')
    layout_transform_variant, layout_transform_variant_probabilities = _resolve_named_variant(int(instance_seed), params=params, explicit_key='layout_transform_variant', weights_key='layout_transform_variant_weights', balance_flag_key='balanced_layout_transform_variant_sampling', supported=('identity', 'mirror_left_right', 'mirror_up_down'), namespace='layout_transform_variant')
    edge_routing_variant, edge_routing_variant_probabilities = _resolve_named_variant(int(instance_seed), params=params, explicit_key='edge_routing_variant', weights_key='edge_routing_variant_weights', balance_flag_key='balanced_edge_routing_variant_sampling', supported=('straight', 'mixed_arc'), namespace='edge_routing_variant')
    node_color_name, node_color_name_probabilities = _resolve_named_variant(int(instance_seed), params=params, explicit_key='node_color_name', weights_key='node_color_name_weights', balance_flag_key='balanced_node_color_name_sampling', supported=SUPPORTED_NODE_COLOR_NAMES, namespace='node_color_name')
    deterministic_edge_probability = float(params.get('deterministic_edge_probability', group_default(_GEN_DEFAULTS, 'deterministic_edge_probability', _DEFAULTS.deterministic_edge_probability)))
    return _ResolvedQuery(query_id=str(query_id), state_count=int(state_count), target_count=int(target_count), deterministic_edge_probability=max(0.0, min(1.0, float(deterministic_edge_probability))), layout_variant=str(layout_variant), layout_transform_variant=str(layout_transform_variant), edge_routing_variant=str(edge_routing_variant), node_color_name=str(node_color_name), query_id_probabilities=dict(query_id_probabilities), state_count_probabilities=_uniform_probability(state_count_support, selected=int(state_count) if explicit_state_count is not None else None), target_count_probabilities=_uniform_probability(target_support, selected=int(target_count) if explicit_target_count is not None else None), layout_variant_probabilities=dict(layout_variant_probabilities), layout_transform_variant_probabilities=dict(layout_transform_variant_probabilities), edge_routing_variant_probabilities=dict(edge_routing_variant_probabilities), node_color_name_probabilities=dict(node_color_name_probabilities))

def _add_transition(transitions: Dict[int, Dict[str, set[int]]], used_targets_by_state: Dict[int, set[int]], *, source: int, symbol: str, target: int) -> None:
    """Add one symbolic transition and reserve the source-target edge."""
    transitions.setdefault(int(source), {}).setdefault(str(symbol), set()).add(int(target))
    used_targets_by_state.setdefault(int(source), set()).add(int(target))

def _available_targets(*, state_count: int, source: int, used_targets: set[int]) -> list[int]:
    """Return targets that keep source-target edges visually distinct."""
    return [int(value) for value in range(int(state_count)) if int(value) != int(source) and int(value) not in set((int(item) for item in used_targets))]

def _compute_nondeterministic_state_indices(transitions: Mapping[int, Mapping[str, Sequence[int]]]) -> Tuple[int, ...]:
    """Return states with epsilon edges or repeated outgoing labels."""
    result: list[int] = []
    for source, per_symbol in sorted(transitions.items()):
        if any((str(symbol) == EPSILON_SYMBOL and len(tuple(targets)) > 0 for symbol, targets in per_symbol.items())):
            result.append(int(source))
            continue
        if any((len(tuple(targets)) > 1 for symbol, targets in per_symbol.items() if str(symbol) != EPSILON_SYMBOL)):
            result.append(int(source))
    return tuple((int(value) for value in result))

def _transition_label_map(*, transitions: Mapping[int, Mapping[str, Sequence[int]]], labels: Sequence[str]) -> Tuple[Dict[Tuple[str, str], str], Dict[str, Dict[str, Tuple[str, ...]]], nx.DiGraph]:
    """Convert symbolic transitions into renderable edge labels and transition table."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(tuple(labels))))
    symbols_by_edge: Dict[Tuple[int, int], list[str]] = {}
    transition_function: Dict[str, Dict[str, Tuple[str, ...]]] = {str(label): {} for label in labels}
    for source, per_symbol in sorted(transitions.items()):
        source_label = str(labels[int(source)])
        for symbol, targets in sorted(per_symbol.items()):
            target_labels: list[str] = []
            for target in sorted((int(value) for value in targets)):
                graph.add_edge(int(source), int(target))
                symbols_by_edge.setdefault((int(source), int(target)), []).append(str(symbol))
                target_labels.append(str(labels[int(target)]))
            transition_function[str(source_label)][str(symbol)] = tuple((str(label) for label in target_labels))
    transition_labels_by_edge: Dict[Tuple[str, str], str] = {}
    for (source, target), symbols in sorted(symbols_by_edge.items()):
        sorted_symbols = sorted(set((str(symbol) for symbol in symbols)), key=lambda value: (str(value) == EPSILON_SYMBOL, str(value)))
        edge_label = ','.join((str(symbol) for symbol in sorted_symbols))
        transition_labels_by_edge[str(labels[int(source)]), str(labels[int(target)])] = str(edge_label)
    return (dict(transition_labels_by_edge), {str(key): {str(symbol): tuple((str(target) for target in targets)) for symbol, targets in value.items()} for key, value in transition_function.items()}, graph)

def _sample_nondeterministic_automaton(rng: random.Random, *, query: _ResolvedQuery) -> _AutomatonNondeterminismSample:
    """Construct one automaton with exactly the requested number of nondeterministic states."""
    labels = state_labels(int(query.state_count))
    transitions: Dict[int, Dict[str, set[int]]] = {int(state): {} for state in range(int(query.state_count))}
    used_targets_by_state: Dict[int, set[int]] = {int(state): set() for state in range(int(query.state_count))}
    target_indices = set((int(value) for value in rng.sample(range(int(query.state_count)), k=int(query.target_count))))
    witness_edges_by_state_index: Dict[int, list[Tuple[str, str, str]]] = {int(state): [] for state in target_indices}
    reason_by_state_index: Dict[int, str] = {}
    for state in sorted(target_indices):
        used_targets = used_targets_by_state.setdefault(int(state), set())
        duplicate_possible = len(_available_targets(state_count=int(query.state_count), source=int(state), used_targets=used_targets)) >= 2
        mode = 'duplicate_label' if duplicate_possible and rng.random() < 0.58 else 'epsilon'
        if mode == 'duplicate_label':
            symbol = str(rng.choice(AUTOMATON_SYMBOLS))
            targets = rng.sample(_available_targets(state_count=int(query.state_count), source=int(state), used_targets=used_targets), k=2)
            for target in targets:
                _add_transition(transitions, used_targets_by_state, source=int(state), symbol=str(symbol), target=int(target))
                witness_edges_by_state_index[int(state)].append((str(labels[int(state)]), str(labels[int(target)]), str(symbol)))
            reason_by_state_index[int(state)] = f'duplicate outgoing label {str(symbol)}'
        else:
            candidates = _available_targets(state_count=int(query.state_count), source=int(state), used_targets=used_targets)
            if not candidates:
                raise ValueError('no epsilon-transition target available')
            target = int(rng.choice(candidates))
            _add_transition(transitions, used_targets_by_state, source=int(state), symbol=EPSILON_SYMBOL, target=int(target))
            witness_edges_by_state_index[int(state)].append((str(labels[int(state)]), str(labels[int(target)]), EPSILON_SYMBOL))
            reason_by_state_index[int(state)] = 'epsilon outgoing transition'
    for state in range(int(query.state_count)):
        for symbol in AUTOMATON_SYMBOLS:
            if str(symbol) in transitions[int(state)]:
                continue
            if rng.random() > float(query.deterministic_edge_probability):
                continue
            candidates = _available_targets(state_count=int(query.state_count), source=int(state), used_targets=used_targets_by_state.setdefault(int(state), set()))
            if not candidates:
                continue
            target = int(rng.choice(candidates))
            _add_transition(transitions, used_targets_by_state, source=int(state), symbol=str(symbol), target=int(target))
    for state in range(int(query.state_count)):
        if transitions[int(state)]:
            continue
        candidates = _available_targets(state_count=int(query.state_count), source=int(state), used_targets=used_targets_by_state.setdefault(int(state), set()))
        if not candidates:
            continue
        _add_transition(transitions, used_targets_by_state, source=int(state), symbol=str(rng.choice(AUTOMATON_SYMBOLS)), target=int(rng.choice(candidates)))
    normalized_transitions: Dict[int, Dict[str, Tuple[int, ...]]] = {int(source): {str(symbol): tuple(sorted((int(target) for target in targets))) for symbol, targets in per_symbol.items()} for source, per_symbol in transitions.items()}
    actual_nondeterministic = set(_compute_nondeterministic_state_indices(normalized_transitions))
    if actual_nondeterministic != target_indices:
        raise ValueError('sampled nondeterministic-state set does not match target')
    transition_labels_by_edge, transition_function, graph = _transition_label_map(transitions=normalized_transitions, labels=labels)
    graph_sample = build_automaton_topology_sample(graph=graph, labels=labels, transition_labels_by_edge=transition_labels_by_edge)
    accepting_count = int(rng.randint(1, min(2, int(query.state_count))))
    accepting_labels = sorted_state_label_tuple((str(labels[int(index)]) for index in rng.sample(range(int(query.state_count)), k=accepting_count)))
    nondeterministic_labels = sorted_state_label_tuple((str(labels[int(index)]) for index in sorted(actual_nondeterministic)))
    deterministic_labels = sorted_state_label_tuple((str(label) for label in labels if str(label) not in set(nondeterministic_labels)))
    witness_edges_by_state = {str(labels[int(state)]): tuple((tuple((str(item) for item in edge)) for edge in edges)) for state, edges in sorted(witness_edges_by_state_index.items())}
    reason_by_state = {str(labels[int(state)]): str(reason) for state, reason in sorted(reason_by_state_index.items())}
    return _AutomatonNondeterminismSample(graph_sample=graph_sample, start_label=str(labels[0]), accepting_labels=tuple((str(label) for label in accepting_labels)), transition_labels_by_edge={(str(left), str(right)): str(symbol) for (left, right), symbol in transition_labels_by_edge.items()}, transition_function={str(key): {str(symbol): tuple((str(target) for target in targets)) for symbol, targets in value.items()} for key, value in transition_function.items()}, nondeterministic_state_labels=tuple((str(label) for label in nondeterministic_labels)), deterministic_state_labels=tuple((str(label) for label in deterministic_labels)), witness_edges_by_state=dict(witness_edges_by_state), nondeterminism_reason_by_state=dict(reason_by_state))

def _witness_label_bboxes(rendered_scene: RenderedGraphScene, witness_edges_by_state: Mapping[str, Sequence[Sequence[str]]]) -> Dict[str, list[list[int]]]:
    """Return rendered transition-label bboxes for audit-only witness edges."""
    result: Dict[str, list[list[int]]] = {}
    for state_label, edges in witness_edges_by_state.items():
        boxes: list[list[int]] = []
        for edge in edges:
            if len(tuple(edge)) < 2:
                continue
            projection = projected_edge_label_bbox_annotation(rendered_scene, (str(edge[0]), str(edge[1])))
            for bbox in projection.get('bbox_set', []):
                boxes.append([int(round(float(value))) for value in bbox])
        result[str(state_label)] = list(boxes)
    return dict(result)

def _json_transition_function(transition_function: Mapping[str, Mapping[str, Sequence[str]]]) -> Dict[str, Dict[str, list[str]]]:
    """Return a canonical-JSON-safe transition function."""
    return {str(state_label): {str(symbol): [str(target) for target in targets] for symbol, targets in per_symbol.items()} for state_label, per_symbol in transition_function.items()}

@register_task
class GraphRelationAutomatonNondeterministicStateCountTask:
    """Count states with nondeterministic outgoing transitions."""
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition', 'topology')
    domain = 'graph'
    supported_query_ids = (QUERY_ID,)

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Own sampling, answer binding, annotation binding, and output assembly."""
        query = _resolve_query(int(instance_seed), params=params)
        render_params = resolve_graph_render_params(params, instance_seed=int(instance_seed), task_id=TASK_ID, render_defaults=_RENDER_DEFAULTS, fallback_defaults=_DEFAULTS, node_color_name=str(query.node_color_name), node_shape_variant='circle', edge_routing_variant=str(query.edge_routing_variant))
        graph_rng = spawn_rng(int(instance_seed), f'{TASK_ID}.graph')
        last_error: Exception | None = None
        sample = None
        rendered_scene = None
        image = None
        background_meta = {}
        post_noise_meta = {}
        witness_label_bboxes: Dict[str, list[list[int]]] = {}
        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_nondeterministic_automaton(graph_rng, query=query)
                background, background_meta = make_background_canvas(canvas_width=int(render_params.canvas_width), canvas_height=int(render_params.canvas_height), instance_seed=int(instance_seed), params=params, default_config=POST_IMAGE_BACKGROUND_DEFAULTS)
                node_style_by_label = {str(sample.start_label): {'halo_rgb': tuple((int(value) for value in render_params.title_color_rgb)), 'halo_width_px': 3, 'halo_pad_px': 6}}
                rendered_scene = render_graph_scene(graph_sample=sample.graph_sample, layout_variant=str(query.layout_variant), layout_transform_variant=str(query.layout_transform_variant), render_params=render_params, layout_seed=int(instance_seed + attempt), scene_title='State Transition Diagram', directed=True, base_image=background, edge_text_labels_by_label=sample.transition_labels_by_edge, edge_text_label_font_size_px=max(15, int(render_params.label_font_size_px) - 3), node_style_by_label=node_style_by_label, layout_fallback_variants=('circular', 'shell', 'layered', 'spring'))
                decorate_automaton_scene(rendered_scene, start_label=str(sample.start_label), accepting_labels=tuple((str(label) for label in sample.accepting_labels)), render_params=render_params, layout_seed=int(instance_seed + attempt))
                witness_label_bboxes = _witness_label_bboxes(rendered_scene, sample.witness_edges_by_state)
                image, post_noise_meta = apply_post_image_noise(rendered_scene.image, instance_seed=int(instance_seed), params=params, default_config=POST_IMAGE_NOISE_DEFAULTS)
                break
            except Exception as exc:
                last_error = exc
                continue
        else:
            raise RuntimeError('failed to generate automaton nondeterministic-state count instance') from last_error
        if sample is None or rendered_scene is None or image is None:
            raise RuntimeError('failed to generate automaton nondeterministic-state count instance')
        prompt_artifacts = build_automaton_prompt_artifacts(domain=self.domain, bundle_id=PROMPT_BUNDLE_ID, task_key=TASK_PROMPT_KEY, prompt_query_key=str(query.query_id), dynamic_slots={'object_description': OBJECT_DESCRIPTION}, instance_seed=int(instance_seed))
        annotation_projection = projected_node_point_annotation(rendered_scene, sample.nondeterministic_state_labels)
        annotation_points = [[int(round(float(point[0]))), int(round(float(point[1])))] for point in annotation_projection['pixel_point_set']]
        if len(annotation_points) != len(sample.nondeterministic_state_labels):
            raise RuntimeError('automaton nondeterministic-state annotation projection is incomplete')
        answer_gt = TypedValue(type='integer', value=int(len(sample.nondeterministic_state_labels)))
        annotation_gt = TypedValue(type='point_set', value=list(annotation_points))
        nondeterministic_set = set((str(label) for label in sample.nondeterministic_state_labels))
        node_entities = [{'entity_id': f'state_{node.label}', 'entity_kind': 'automaton_state', 'label': str(node.label), 'is_start_state': bool(str(node.label) == str(sample.start_label)), 'is_accepting_state': bool(str(node.label) in set(sample.accepting_labels)), 'is_nondeterministic_state': bool(str(node.label) in nondeterministic_set), 'nondeterminism_reason': str(sample.nondeterminism_reason_by_state.get(str(node.label), '')), 'center_px': list(node.center_xy), 'bbox_xyxy': list(node.bbox_xyxy), 'successors': list(sample.graph_sample.successors_by_label[str(node.label)]), 'predecessors': list(sample.graph_sample.predecessors_by_label[str(node.label)])} for node in rendered_scene.nodes]
        witness_edge_keys = {f'{str(edge[0])}->{str(edge[1])}' for edges in sample.witness_edges_by_state.values() for edge in edges}
        edge_entities = [{'entity_id': str(edge.edge_id), 'entity_kind': 'automaton_transition', 'source_state_label': str(edge.node_u_label), 'target_state_label': str(edge.node_v_label), 'transition_symbol': str(sample.transition_labels_by_edge[str(edge.node_u_label), str(edge.node_v_label)]), 'is_nondeterminism_witness': bool(f'{str(edge.node_u_label)}->{str(edge.node_v_label)}' in witness_edge_keys), 'segment_px': [list(edge.segment_px[0]), list(edge.segment_px[1])], 'route_variant': str(edge.route_variant), 'control_px': list(edge.control_px) if edge.control_px is not None else None, 'edge_label_bbox_xyxy': list(edge.edge_label_bbox_xyxy) if edge.edge_label_bbox_xyxy is not None else None} for edge in rendered_scene.edges]
        transition_entries = [{'source_state': str(left), 'target_state': str(right), 'symbol': str(symbol)} for (left, right), symbol in sample.transition_labels_by_edge.items()]
        transition_function_json = _json_transition_function(sample.transition_function)
        trace_payload = {'scene_ir': {'scene_kind': 'automaton_nondeterministic_state_count', 'entities': [*node_entities, *edge_entities], 'relations': {'query_id': str(query.query_id), 'nondeterministic_state_rule': 'a state is nondeterministic if it has an epsilon outgoing transition or two or more outgoing transitions with the same input label', 'missing_transition_policy': 'missing transitions do not count as nondeterministic', 'start_state_label': str(sample.start_label), 'accepting_state_labels': list(sample.accepting_labels), 'nondeterministic_state_labels': list(sample.nondeterministic_state_labels), 'deterministic_state_labels': list(sample.deterministic_state_labels), 'witness_edges_by_state': {str(key): [list(edge) for edge in edges] for key, edges in sample.witness_edges_by_state.items()}, 'nondeterminism_reason_by_state': dict(sample.nondeterminism_reason_by_state), 'transition_function': dict(transition_function_json), 'transition_labels_by_edge': list(transition_entries), 'query_id_probabilities': dict(query.query_id_probabilities), 'state_count_probabilities': dict(query.state_count_probabilities), 'target_count_probabilities': dict(query.target_count_probabilities)}, 'frames': {'pixel': {'origin': [0.0, 0.0], 'x_positive': 'right', 'y_positive': 'down'}, 'panels': dict(rendered_scene.panel_geometry)}}, 'query_spec': {'query_id': str(query.query_id), 'template_id': PROMPT_BUNDLE_ID, 'prompt_variant': dict(prompt_artifacts.prompt_variant), 'prompt_variant_active_key': str(prompt_artifacts.prompt_variant_active_key), 'prompt_variants': dict(prompt_artifacts.prompt_variants_for_trace), 'params': {'query_id': str(query.query_id), 'internal_query_id': str(query.query_id), 'query_id_probabilities': dict(query.query_id_probabilities), 'state_count': int(query.state_count), 'state_count_probabilities': dict(query.state_count_probabilities), 'target_count': int(query.target_count), 'target_count_probabilities': dict(query.target_count_probabilities), 'answer_count': int(answer_gt.value), 'start_state_label': str(sample.start_label), 'accepting_state_labels': list(sample.accepting_labels), 'nondeterministic_state_labels': list(sample.nondeterministic_state_labels), 'layout_variant': str(query.layout_variant), 'layout_variant_probabilities': dict(query.layout_variant_probabilities), 'layout_transform_variant': str(query.layout_transform_variant), 'layout_transform_variant_probabilities': dict(query.layout_transform_variant_probabilities), 'edge_routing_variant': str(query.edge_routing_variant), 'edge_routing_variant_probabilities': dict(query.edge_routing_variant_probabilities), 'node_color_name': str(query.node_color_name), 'node_color_name_probabilities': dict(query.node_color_name_probabilities)}}, 'render_spec': {'canvas_size': list(rendered_scene.panel_geometry['canvas_size']), 'coord_space': 'pixel', 'panel_geometry': dict(rendered_scene.panel_geometry), 'style': {'node_color_name': str(query.node_color_name), 'theme_tone': str(render_params.theme_tone), 'panel_style_variant': str(render_params.panel_style_variant), 'background_color_rgb': list(render_params.background_color_rgb), 'panel_fill_rgb': list(render_params.panel_fill_rgb), 'panel_border_rgb': list(render_params.panel_border_rgb), 'title_color_rgb': list(render_params.title_color_rgb), 'edge_color_rgb': list(render_params.edge_color_rgb), 'node_fill_rgb': list(render_params.node_fill_rgb), 'node_border_rgb': list(render_params.node_border_rgb), 'label_text_rgb': list(render_params.label_text_rgb), 'label_stroke_rgb': list(render_params.label_stroke_rgb), 'node_shape_variant': 'circle', 'node_radius_px': int(render_params.node_radius_px), 'edge_width_px': int(render_params.edge_width_px), 'edge_routing_variant': str(rendered_scene.edge_routing_variant), 'arrow_length_px': int(render_params.arrow_length_px), 'arrow_width_px': int(render_params.arrow_width_px), 'node_border_width_px': int(render_params.node_border_width_px), 'label_font_size_px': int(render_params.label_font_size_px), 'resolved_label_font_size_px': int(rendered_scene.resolved_label_font_size_px), 'label_stroke_width_px': int(rendered_scene.resolved_label_stroke_width_px), 'font_family': str(render_params.font_family or ''), 'font_asset': dict(render_params.font_asset) if isinstance(render_params.font_asset, Mapping) else {}, 'font_asset_version': str(render_params.font_asset_version or ''), 'font_exclusion_reason': str(render_params.font_exclusion_reason), 'context_text_elements': list(rendered_scene.panel_geometry.get('context_text_elements', [])), 'automaton_accepting_state_glyph': 'double_inner_ring', 'automaton_start_state_glyph': 'incoming_start_arrow', 'transition_labels_by_edge': list(transition_entries), 'background_meta': dict(background_meta), 'post_image_noise_meta': dict(post_noise_meta)}}, 'render_map': {'image_id': 'img0', 'anchors': {}}, 'execution_trace': {'query_id': str(query.query_id), 'internal_query_id': str(query.query_id), 'query_id_probabilities': dict(query.query_id_probabilities), 'scene_variant': str(rendered_scene.layout_variant), 'question_format': str(query.query_id), 'state_count': int(query.state_count), 'edge_count': int(sample.graph_sample.edge_count), 'answer': int(answer_gt.value), 'answer_count': int(answer_gt.value), 'target_count': int(query.target_count), 'start_state_label': str(sample.start_label), 'accepting_state_labels': list(sample.accepting_labels), 'nondeterministic_state_labels': list(sample.nondeterministic_state_labels), 'deterministic_state_labels': list(sample.deterministic_state_labels), 'witness_edges_by_state': {str(key): [list(edge) for edge in edges] for key, edges in sample.witness_edges_by_state.items()}, 'witness_transition_label_bboxes_by_state': dict(witness_label_bboxes), 'nondeterminism_reason_by_state': dict(sample.nondeterminism_reason_by_state), 'transition_function': dict(transition_function_json), 'transition_labels_by_edge': list(transition_entries), 'state_count_probabilities': dict(query.state_count_probabilities), 'target_count_probabilities': dict(query.target_count_probabilities), 'layout_variant_requested': str(query.layout_variant), 'layout_variant_used': str(rendered_scene.layout_variant), 'layout_transform_variant': str(rendered_scene.layout_transform_variant), 'edge_routing_variant': str(rendered_scene.edge_routing_variant), 'node_color_name': str(query.node_color_name), 'crossing_count': int(rendered_scene.crossing_count)}, 'witness_symbolic': {'type': 'nondeterministic_state_count', 'nondeterministic_state_labels': list(sample.nondeterministic_state_labels), 'witness_edges_by_state': {str(key): [list(edge) for edge in edges] for key, edges in sample.witness_edges_by_state.items()}, 'answer_count': int(answer_gt.value)}, 'projected_annotation': {'type': 'point_set', 'point_set': list(annotation_points), 'pixel_point_set': list(annotation_points), 'pixel_bbox_set': list(annotation_projection['pixel_bbox_set']), 'witness_transition_label_bboxes_by_state': dict(witness_label_bboxes)}}
        return TaskOutput(prompt=str(prompt_artifacts.prompt), answer_gt=answer_gt, annotation_gt=annotation_gt, image=image, image_id='img0', trace_payload=trace_payload, task_versions=default_task_versions(), scene_id=SCENE_ID, query_id=str(query.query_id), prompt_variants=dict(prompt_artifacts.prompt_variants))
__all__ = ['EPSILON_SYMBOL', 'GraphRelationAutomatonNondeterministicStateCountTask']
