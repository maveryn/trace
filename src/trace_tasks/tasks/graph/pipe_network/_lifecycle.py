"""Neutral lifecycle helpers for the pipe-network graph scene."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from ....core.scene_config import get_scene_defaults
from ....core.sampling import uniform_choice
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ....core.visual.background import make_background_canvas
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.config_defaults import group_default, required_group_defaults, split_scene_generation_rendering_prompt_defaults
from ...shared.deterministic_sampling import uniform_probability_map
from ...shared.fixed_query import force_query_id_params, select_task_query_id
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_scene_prompt_variants
from ..shared.style import SUPPORTED_NODE_COLOR_NAMES
from ..shared.task_support import format_graph_prompt_label, resolve_graph_named_variant, resolve_graph_render_params
from ..shared.visual_defaults import load_graph_scene_background_defaults, load_graph_scene_noise_defaults
from .shared.defaults import PipeNetworkTaskDefaults
from .shared.annotations import projected_pipe_node_point_annotation, projected_pipe_segment_annotation
from .shared.output import PipeBoundResult, PipePreparedInstance, PipeResolvedAxes, RenderedPipeAssets
from .shared.prompts import (
    ANSWER_HINT,
    JSON_OUTPUT_CONTRACT,
    JSON_OUTPUT_CONTRACT_ANSWER_ONLY,
    OBJECT_DESCRIPTION,
    PROMPT_BUNDLE_ID,
    SCENE_PROMPT_KEY,
    TASK_PROMPT_KEY,
    json_examples_for_annotation,
)
from .shared.rendering import render_pipe_network_scene
from .shared.algorithms import feasible_pipe_node_counts
from .shared.sampling import feasible_pipe_bridge_target_counts
from .shared.state import PIPE_VISUAL_STYLE_IDS, SCENE_ID, SUPPORTED_PIPE_GRID_SHAPE_VARIANTS, SUPPORTED_PIPE_LABEL_VARIANTS, PipeJunctionNetworkSample


FALLBACK_DEFAULTS = PipeNetworkTaskDefaults()
SCENE_TITLE = "Pipe Junction Board"

PipeAxesResolver = Callable[..., PipeResolvedAxes]
PipeSampler = Callable[[int, Mapping[str, Any], int, PipeResolvedAxes], PipeJunctionNetworkSample]
PipeBinder = Callable[[PipeJunctionNetworkSample, RenderedPipeAssets], PipeBoundResult]
PipeNetworkSamplerFn = Callable[..., PipeJunctionNetworkSample]


def load_pipe_defaults(owner_id: str) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load scene defaults for one public pipe-network task."""

    group_defaults = get_scene_defaults("graph", SCENE_ID)
    gen_defaults, render_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        group_defaults if isinstance(group_defaults, Mapping) else {},
        task_id=str(owner_id),
    )
    background_defaults = load_graph_scene_background_defaults(scene_id=SCENE_ID)
    noise_defaults = load_graph_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)
    return dict(gen_defaults), dict(render_defaults), dict(prompt_defaults), dict(background_defaults), dict(noise_defaults)


def integer_support(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    low_key: str,
    high_key: str,
    default_low: int,
    default_high: int,
) -> Tuple[int, ...]:
    """Resolve an inclusive integer support from params, scene config, and defaults."""

    low = int(params.get(str(low_key), group_default(gen_defaults, str(low_key), int(default_low))))
    high = int(params.get(str(high_key), group_default(gen_defaults, str(high_key), int(default_high))))
    if int(low) > int(high):
        raise ValueError(f"{low_key} must be <= {high_key}")
    return tuple(range(int(low), int(high) + 1))


def pipe_query_distance_support(*, params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> Tuple[int, ...]:
    """Resolve the scene-standard query-distance support."""

    return integer_support(
        params=params,
        gen_defaults=gen_defaults,
        low_key="query_distance_min",
        high_key="query_distance_max",
        default_low=FALLBACK_DEFAULTS.query_distance_min,
        default_high=FALLBACK_DEFAULTS.query_distance_max,
    )


def select_support_value(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    owner_id: str,
    support: Sequence[int],
    explicit_keys: Sequence[str],
    namespace_suffix: str,
) -> tuple[int, Dict[str, float]]:
    """Select one value from an already objective-filtered support."""

    support_tuple = tuple(int(value) for value in support)
    if not support_tuple:
        raise ValueError("empty pipe selection support")
    explicit_value = None
    for key in explicit_keys:
        if params.get(str(key)) is not None:
            explicit_value = int(params[str(key)])
            break
    if explicit_value is not None:
        if int(explicit_value) not in support_tuple:
            raise ValueError("requested pipe value is outside configured support")
        return int(explicit_value), dict(uniform_probability_map(support_tuple, selected=int(explicit_value)))
    selected = int(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{owner_id}:{namespace_suffix}"),
            support_tuple,
        )
    )
    return int(selected), dict(uniform_probability_map(support_tuple))


def resolve_pipe_style_axes(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    owner_id: str,
) -> tuple[str, Dict[str, float], str, Dict[str, float], str, Dict[str, float]]:
    """Resolve grid shape, label style, and node color axes."""

    grid_rng = spawn_rng(int(instance_seed), f"{owner_id}.grid_shape_variant")
    grid_shape_variant, grid_probs = resolve_graph_named_variant(
        grid_rng,
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="grid_shape_variant",
        weights_key="grid_shape_variant_weights",
        balance_flag_key="balanced_grid_shape_variant_sampling",
        supported=SUPPORTED_PIPE_GRID_SHAPE_VARIANTS,
        instance_seed=int(instance_seed),
        task_id=str(owner_id),
        namespace="grid_shape_variant",
    )
    label_rng = spawn_rng(int(instance_seed), f"{owner_id}.label_variant")
    label_variant, label_probs = resolve_graph_named_variant(
        label_rng,
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="label_variant",
        weights_key="label_variant_weights",
        balance_flag_key="balanced_label_variant_sampling",
        supported=SUPPORTED_PIPE_LABEL_VARIANTS,
        instance_seed=int(instance_seed),
        task_id=str(owner_id),
        namespace="label_variant",
    )
    color_rng = spawn_rng(int(instance_seed), f"{owner_id}.node_color_name")
    node_color_name, color_probs = resolve_graph_named_variant(
        color_rng,
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="node_color_name",
        weights_key="node_color_name_weights",
        balance_flag_key="balanced_node_color_name_sampling",
        supported=SUPPORTED_NODE_COLOR_NAMES,
        instance_seed=int(instance_seed),
        task_id=str(owner_id),
        namespace="node_color_name",
    )
    return str(grid_shape_variant), dict(grid_probs), str(label_variant), dict(label_probs), str(node_color_name), dict(color_probs)


def resolve_node_count(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    owner_id: str,
    feasible_nodes: Sequence[int],
) -> tuple[int, Dict[str, float]]:
    """Resolve one feasible node count for a public objective."""

    return select_support_value(
        params=params,
        instance_seed=int(instance_seed),
        owner_id=str(owner_id),
        support=tuple(int(value) for value in feasible_nodes),
        explicit_keys=("node_count",),
        namespace_suffix="node_count",
    )


def _node_count_bounds(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> tuple[int, int]:
    """Return scene-configured node-count bounds."""

    low = int(params.get("node_count_min", group_default(gen_defaults, "node_count_min", FALLBACK_DEFAULTS.node_count_min)))
    high = int(params.get("node_count_max", group_default(gen_defaults, "node_count_max", FALLBACK_DEFAULTS.node_count_max)))
    return int(low), int(high)


def resolve_pipe_target_axes(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    owner_id: str,
    target_low_key: str,
    target_high_key: str,
    default_target_low: int,
    default_target_high: int,
    explicit_target_keys: Sequence[str],
    target_namespace: str,
    minimum_nodes_for_target: Callable[[int, int], int],
    query_distance_support: Tuple[int, ...] = (),
) -> PipeResolvedAxes:
    """Resolve common target-count/path axes from semantic task parameters."""

    query_distance = 0
    distance_probs: Dict[str, float] = {}
    if query_distance_support:
        query_distance, distance_probs = select_support_value(
            params=params,
            instance_seed=int(instance_seed),
            owner_id=str(owner_id),
            support=query_distance_support,
            explicit_keys=("query_distance",),
            namespace_suffix="query_distance",
        )
    target_support = integer_support(
        params=params,
        gen_defaults=gen_defaults,
        low_key=str(target_low_key),
        high_key=str(target_high_key),
        default_low=int(default_target_low),
        default_high=int(default_target_high),
    )
    target_value, target_probs = select_support_value(
        params=params,
        instance_seed=int(instance_seed),
        owner_id=str(owner_id),
        support=target_support,
        explicit_keys=tuple(str(key) for key in explicit_target_keys),
        namespace_suffix=str(target_namespace),
    )
    grid_shape, grid_probs, label_variant, label_probs, color_name, color_probs = resolve_pipe_style_axes(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        owner_id=str(owner_id),
    )
    node_min, node_max = _node_count_bounds(params, gen_defaults)
    node_support = feasible_pipe_node_counts(
        node_count_min=int(node_min),
        node_count_max=int(node_max),
        grid_shape_variant=str(grid_shape),
    )
    min_nodes = int(minimum_nodes_for_target(int(target_value), int(query_distance)))
    node_support = tuple(int(value) for value in node_support if int(value) >= int(min_nodes))
    node_count, node_probs = resolve_node_count(
        params=params,
        instance_seed=int(instance_seed),
        owner_id=str(owner_id),
        feasible_nodes=node_support,
    )
    return PipeResolvedAxes(
        node_count=int(node_count),
        grid_shape_variant=str(grid_shape),
        label_variant=str(label_variant),
        node_color_name=str(color_name),
        target_value=int(target_value),
        query_distance=int(query_distance),
        node_count_probabilities=dict(node_probs),
        grid_shape_variant_probabilities=dict(grid_probs),
        label_variant_probabilities=dict(label_probs),
        node_color_name_probabilities=dict(color_probs),
        target_value_probabilities=dict(target_probs),
        query_distance_probabilities=dict(distance_probs),
    )


def resolve_pipe_bridge_axes(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    owner_id: str,
) -> PipeResolvedAxes:
    """Resolve bridge-count axes where feasible targets depend on node count."""

    grid_shape, grid_probs, label_variant, label_probs, color_name, color_probs = resolve_pipe_style_axes(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        owner_id=str(owner_id),
    )
    node_min, node_max = _node_count_bounds(params, gen_defaults)
    node_support = tuple(
        int(value)
        for value in feasible_pipe_node_counts(node_count_min=int(node_min), node_count_max=int(node_max), grid_shape_variant=str(grid_shape))
        if int(value) >= 4
    )
    node_count, node_probs = resolve_node_count(params=params, instance_seed=int(instance_seed), owner_id=str(owner_id), feasible_nodes=node_support)
    configured_targets = integer_support(
        params=params,
        gen_defaults=gen_defaults,
        low_key="target_count_min",
        high_key="target_count_max",
        default_low=FALLBACK_DEFAULTS.target_count_min,
        default_high=FALLBACK_DEFAULTS.target_count_max,
    )
    target_support = feasible_pipe_bridge_target_counts(int(node_count), tuple(configured_targets))
    target_count, target_probs = select_support_value(
        params=params,
        instance_seed=int(instance_seed),
        owner_id=f"{owner_id}.bridge_target",
        support=target_support,
        explicit_keys=("target_count",),
        namespace_suffix="target_count",
    )
    return PipeResolvedAxes(
        node_count=int(node_count),
        grid_shape_variant=str(grid_shape),
        label_variant=str(label_variant),
        node_color_name=str(color_name),
        target_value=int(target_count),
        node_count_probabilities=dict(node_probs),
        grid_shape_variant_probabilities=dict(grid_probs),
        label_variant_probabilities=dict(label_probs),
        node_color_name_probabilities=dict(color_probs),
        target_value_probabilities=dict(target_probs),
    )


def sample_pipe_target_network(
    *,
    owner_id: str,
    instance_seed: int,
    max_attempts: int,
    axes: PipeResolvedAxes,
    sampler: PipeNetworkSamplerFn,
    target_keyword: str,
    query_distance_keyword: str | None = None,
    attempt_floor: int = 200,
    attempt_multiplier: int = 1,
) -> PipeJunctionNetworkSample:
    """Call a pipe-network sampler with resolved scene axes and one semantic target."""

    kwargs: Dict[str, Any] = {
        "node_count": int(axes.node_count),
        str(target_keyword): int(axes.target_value),
        "grid_shape_variant": str(axes.grid_shape_variant),
        "label_variant": str(axes.label_variant),
        "max_attempts": max(int(attempt_floor), int(max_attempts) * int(attempt_multiplier)),
    }
    if query_distance_keyword:
        kwargs[str(query_distance_keyword)] = int(axes.query_distance)
    return sampler(spawn_rng(int(instance_seed), f"{owner_id}.pipe_network"), **kwargs)


def render_pipe_assets(
    *,
    owner_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    sample: PipeJunctionNetworkSample,
    axes: PipeResolvedAxes,
) -> RenderedPipeAssets:
    """Render a sampled pipe network and apply post-render visual effects."""

    render_params = resolve_graph_render_params(
        params,
        instance_seed=int(instance_seed),
        task_id=str(owner_id),
        render_defaults=render_defaults,
        fallback_defaults=FALLBACK_DEFAULTS,
        node_color_name=str(axes.node_color_name),
        node_shape_variant="circle",
    )
    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=params,
        default_config=background_defaults,
    )
    rendered_scene = render_pipe_network_scene(
        pipe_sample=sample,
        render_params=render_params,
        base_image=background,
        scene_title=SCENE_TITLE,
        layout_seed=int(instance_seed),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=noise_defaults,
    )
    return RenderedPipeAssets(
        image=image,
        rendered_scene=rendered_scene,
        render_params=render_params,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


def pipe_scene_entities(sample: PipeJunctionNetworkSample, rendered_scene: Any) -> list[Dict[str, Any]]:
    """Build node and pipe-segment entity records from the rendered pipe scene."""

    target_node_set = {str(label) for label in sample.target_labels}
    target_edge_set = {tuple(edge) for edge in sample.target_edges}
    node_entities = [
        {
            "entity_id": f"junction_{node.label}",
            "entity_kind": "pipe_junction",
            "label": str(node.label),
            "open_degree": int(node.open_degree),
            "grid_cell": list(node.grid_cell),
            "open_neighbors": list(node.open_neighbors),
            "center_px": list(node.center_xy),
            "bbox_xyxy": list(node.bbox_xyxy),
            "is_query_node": bool(str(node.label) == str(sample.query_label)),
            "is_source_node": bool(str(node.label) == str(sample.source_label)),
            "is_goal_node": bool(str(node.label) == str(sample.goal_label)),
            "is_witness_node": bool(str(node.label) in target_node_set),
        }
        for node in rendered_scene.nodes
    ]
    edge_entities = [
        {
            "entity_id": str(edge.edge_id),
            "entity_kind": "pipe_segment",
            "node_u_label": str(edge.node_u_label),
            "node_v_label": str(edge.node_v_label),
            "pipe_state": str(edge.pipe_state),
            "segment_px": [list(edge.segment_px[0]), list(edge.segment_px[1])],
            "is_open": bool(str(edge.pipe_state) == "open"),
            "is_blocked": bool(str(edge.pipe_state) == "blocked"),
            "is_witness_edge": bool((str(edge.node_u_label), str(edge.node_v_label)) in target_edge_set),
        }
        for edge in rendered_scene.edges
    ]
    return [*node_entities, *edge_entities]


def bind_pipe_point_set(
    sample: PipeJunctionNetworkSample,
    rendered: RenderedPipeAssets,
    *,
    labels: Sequence[str],
    answer_value: int,
    answer_key: str,
    relation_key: str,
    witness_extra: Mapping[str, Any] | None = None,
    prompt_slots: Mapping[str, Any] | None = None,
    trace_extra: Mapping[str, Any] | None = None,
) -> PipeBoundResult:
    """Bind unordered junction-center point witnesses to an integer answer."""

    label_tuple = tuple(str(label) for label in labels)
    projection = projected_pipe_node_point_annotation(rendered.rendered_scene, label_tuple)
    annotation = [[int(round(point[0])), int(round(point[1]))] for point in projection["pixel_point_set"]]
    witness = {"type": "junction_label_set", "labels": list(label_tuple)}
    if witness_extra:
        witness.update(dict(witness_extra))
    trace_params = {str(answer_key): int(answer_value), **dict(trace_extra or {})}
    return PipeBoundResult(
        answer_type="integer",
        answer_value=int(answer_value),
        annotation_type="point_set",
        annotation_value=list(annotation),
        prompt_slots=dict(prompt_slots or {}),
        trace_params=dict(trace_params),
        scene_relations={str(relation_key): list(label_tuple)},
        execution_trace={str(answer_key): int(answer_value), str(relation_key): list(label_tuple)},
        witness_symbolic=dict(witness),
        projected_annotation={"type": "point_set", **dict(projection), "point_set": list(annotation)},
    )


def bind_pipe_point_sequence(
    sample: PipeJunctionNetworkSample,
    rendered: RenderedPipeAssets,
    *,
    labels: Sequence[str],
    answer_value: int,
    answer_key: str,
    relation_key: str,
) -> PipeBoundResult:
    """Bind ordered junction-center point witnesses to an integer path answer."""

    label_tuple = tuple(str(label) for label in labels)
    projection = projected_pipe_node_point_annotation(rendered.rendered_scene, label_tuple)
    annotation = [[int(round(point[0])), int(round(point[1]))] for point in projection["pixel_point_sequence"]]
    return PipeBoundResult(
        answer_type="integer",
        answer_value=int(answer_value),
        annotation_type="point_sequence",
        annotation_value=list(annotation),
        prompt_slots={},
        trace_params={str(answer_key): int(answer_value)},
        scene_relations={str(relation_key): list(label_tuple)},
        execution_trace={str(answer_key): int(answer_value), str(relation_key): list(label_tuple)},
        witness_symbolic={"type": "junction_label_sequence", "labels": list(label_tuple)},
        projected_annotation={"type": "point_sequence", **dict(projection), "point_sequence": list(annotation)},
    )


def bind_pipe_segment_set(
    sample: PipeJunctionNetworkSample,
    rendered: RenderedPipeAssets,
    *,
    edges: Sequence[Sequence[str]],
    answer_value: int,
    answer_key: str,
    relation_key: str,
) -> PipeBoundResult:
    """Bind unordered pipe endpoint-center segments to an integer answer."""

    edge_tuple = tuple((str(left), str(right)) for left, right in edges)
    projection = projected_pipe_segment_annotation(rendered.rendered_scene, edge_tuple)
    annotation = [
        [[int(round(point[0])), int(round(point[1]))] for point in segment]
        for segment in projection["segment_set"]
    ]
    return PipeBoundResult(
        answer_type="integer",
        answer_value=int(answer_value),
        annotation_type="segment_set",
        annotation_value=list(annotation),
        prompt_slots={},
        trace_params={str(answer_key): int(answer_value)},
        scene_relations={str(relation_key): [list(edge) for edge in edge_tuple]},
        execution_trace={str(answer_key): int(answer_value), str(relation_key): [list(edge) for edge in edge_tuple]},
        witness_symbolic={"type": "pipe_segment_label_set", "edges": [list(edge) for edge in edge_tuple]},
        projected_annotation={"type": "segment_set", "segment_set": list(annotation), **dict(projection)},
    )


def _common_trace_fields(sample: PipeJunctionNetworkSample, axes: PipeResolvedAxes) -> Dict[str, Any]:
    """Return semantic fields common to all pipe-network objectives."""

    return {
        "node_count": int(axes.node_count),
        "open_edge_count": int(len(sample.open_edges)),
        "blocked_edge_count": int(len(sample.blocked_edges)),
        "query_label": str(sample.query_label),
        "source_label": str(sample.source_label),
        "goal_label": str(sample.goal_label),
        "query_distance": int(axes.query_distance),
        "matching_labels": list(sample.target_labels),
        "matching_edges": [list(edge) for edge in sample.target_edges],
        "open_adjacency_by_label": {str(key): list(values) for key, values in sample.adjacency_by_label.items()},
        "open_edge_labels": [list(edge) for edge in sample.open_edge_labels],
        "blocked_edge_labels": [list(edge) for edge in sample.blocked_edge_labels],
        "degrees_by_label": {str(key): int(value) for key, value in sample.degrees_by_label.items()},
        "grid_shape_variant": str(axes.grid_shape_variant),
        "label_variant": str(axes.label_variant),
        "node_color_name": str(axes.node_color_name),
    }


def prepare_pipe_instance(
    *,
    owner_id: str,
    branch_name: str,
    branch_probabilities: Mapping[str, float],
    prompt_query_key: str,
    prompt_annotation_key: str,
    instance_seed: int,
    prompt_defaults: Mapping[str, Any],
    sample: PipeJunctionNetworkSample,
    axes: PipeResolvedAxes,
    rendered: RenderedPipeAssets,
    bound: PipeBoundResult,
) -> PipePreparedInstance:
    """Compose the prompt and trace payload for an already-bound pipe objective."""

    prompt_defaults_required = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "object_description",
            str(prompt_annotation_key),
            "answer_hint",
            "json_example",
            "json_example_answer_only",
        ),
        context=f"prompt defaults for {owner_id}",
    )
    question_slots = {
        "source_label": format_graph_prompt_label(str(sample.source_label), label_variant=str(sample.label_variant)),
        "goal_label": format_graph_prompt_label(str(sample.goal_label), label_variant=str(sample.label_variant)),
        "query_label": format_graph_prompt_label(str(sample.query_label), label_variant=str(sample.label_variant)),
        "query_distance": int(sample.query_distance),
        **dict(bound.prompt_slots),
    }
    prompt_json_example, prompt_json_example_answer_only = json_examples_for_annotation(str(bound.annotation_type))
    annotation_hint = str(prompt_defaults_required[str(prompt_annotation_key)]).format(**question_slots)
    prompt_selection = render_scene_prompt_variants(
        domain="graph",
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults_required.get("bundle_id") or PROMPT_BUNDLE_ID),
        scene_key=str(prompt_defaults_required.get("scene_key") or SCENE_PROMPT_KEY),
        task_key=str(prompt_defaults_required.get("task_key") or TASK_PROMPT_KEY),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(prompt_defaults_required.get("object_description") or OBJECT_DESCRIPTION),
            **dict(question_slots),
            "json_output_contract": str(prompt_defaults_required.get("json_output_contract") or JSON_OUTPUT_CONTRACT),
            "json_output_contract_answer_only": str(prompt_defaults_required.get("json_output_contract_answer_only") or JSON_OUTPUT_CONTRACT_ANSWER_ONLY),
            "annotation_hint": str(annotation_hint),
            "answer_hint": str(prompt_defaults_required.get("answer_hint") or ANSWER_HINT),
            "json_example": str(prompt_json_example),
            "json_example_answer_only": str(prompt_json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
    common_trace = _common_trace_fields(sample, axes)
    params = {
        "query_id": str(branch_name),
        "scene_id": SCENE_ID,
        "node_count": int(axes.node_count),
        "node_count_probabilities": dict(axes.node_count_probabilities or {}),
        "grid_shape_variant": str(axes.grid_shape_variant),
        "grid_shape_variant_probabilities": dict(axes.grid_shape_variant_probabilities or {}),
        "label_variant": str(axes.label_variant),
        "label_variant_probabilities": dict(axes.label_variant_probabilities or {}),
        "node_color_name": str(axes.node_color_name),
        "node_color_name_probabilities": dict(axes.node_color_name_probabilities or {}),
        "target_value": int(axes.target_value),
        "target_value_probabilities": dict(axes.target_value_probabilities or {}),
        "query_distance": int(axes.query_distance),
        "query_distance_probabilities": dict(axes.query_distance_probabilities or {}),
        "query_id_probabilities": {str(key): float(value) for key, value in branch_probabilities.items()},
        **dict(bound.trace_params),
    }
    trace_payload = {
        "scene_ir": {
            "scene_kind": "pipe_network",
            "domain": "graph",
            "scene_id": SCENE_ID,
            "task_id": str(owner_id),
            "query_id": str(branch_name),
            "entities": list(bound.entities) if bound.entities is not None else pipe_scene_entities(sample, rendered.rendered_scene),
            "relations": {
                "graph_directionality": "undirected",
                "pipe_semantics": "only open pipes are traversable; blocked pipes are visible distractors",
                "query_id": str(branch_name),
                **dict(common_trace),
                **dict(bound.scene_relations),
            },
            "frames": {
                "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                "panels": dict(rendered.rendered_scene.panel_geometry),
            },
        },
        "query_spec": {
            "query_id": str(branch_name),
            "scene_id": SCENE_ID,
            "template_id": str(prompt_defaults_required.get("bundle_id") or PROMPT_BUNDLE_ID),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": dict(params),
        },
        "render_spec": {
            "coord_space": "pixel",
            "scene_id": SCENE_ID,
            "canvas_size": list(rendered.rendered_scene.panel_geometry["canvas_size"]),
            "panel_geometry": dict(rendered.rendered_scene.panel_geometry),
            "style": {
                "theme_tone": str(rendered.render_params.theme_tone),
                "panel_style_variant": str(rendered.render_params.panel_style_variant),
                "node_color_name": str(axes.node_color_name),
                "background_color_rgb": list(rendered.render_params.background_color_rgb),
                "panel_fill_rgb": list(rendered.render_params.panel_fill_rgb),
                "panel_border_rgb": list(rendered.render_params.panel_border_rgb),
                "title_color_rgb": list(rendered.render_params.title_color_rgb),
                "open_pipe_color_rgb": list(rendered.render_params.edge_color_rgb),
                "node_fill_rgb": list(rendered.render_params.node_fill_rgb),
                "node_border_rgb": list(rendered.render_params.node_border_rgb),
                "label_text_rgb": list(rendered.render_params.label_text_rgb),
                "label_stroke_rgb": list(rendered.render_params.label_stroke_rgb),
                "node_radius_px": int(rendered.render_params.node_radius_px),
                "open_pipe_width_px": int(rendered.rendered_scene.open_pipe_width_px),
                "blocked_pipe_width_px": int(rendered.rendered_scene.blocked_pipe_width_px),
                "label_font_size_px": int(rendered.render_params.label_font_size_px),
                "resolved_label_font_size_px": int(rendered.rendered_scene.resolved_label_font_size_px),
                "label_stroke_width_px": int(rendered.rendered_scene.resolved_label_stroke_width_px),
                "font_family": str(rendered.render_params.font_family or ""),
                "font_asset": dict(rendered.render_params.font_asset) if isinstance(rendered.render_params.font_asset, Mapping) else {},
                "font_asset_version": str(rendered.render_params.font_asset_version or ""),
                "font_exclusion_reason": str(rendered.render_params.font_exclusion_reason),
                "context_text_elements": list(rendered.rendered_scene.panel_geometry.get("context_text_elements", [])),
                "background_meta": dict(rendered.background_meta),
                "post_image_noise_meta": dict(rendered.post_noise_meta),
                "pipe_visual_style_ids": list(PIPE_VISUAL_STYLE_IDS),
            },
        },
        "render_map": {"image_id": "img0", "anchors": {}},
        "execution_trace": {
            "query_id": str(branch_name),
            "scene_variant": str(axes.grid_shape_variant),
            "scene_id": SCENE_ID,
            "question_format": str(prompt_query_key),
            "answer": bound.answer_value,
            **dict(common_trace),
            **dict(bound.execution_trace),
        },
        "witness_symbolic": dict(bound.witness_symbolic),
        "projected_annotation": dict(bound.projected_annotation),
        "task_versions": default_task_versions(),
    }
    return PipePreparedInstance(
        prompt=str(prompt_artifacts.prompt),
        image=rendered.image,
        trace_payload=dict(trace_payload),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


def run_pipe_objective(
    *,
    owner_id: str,
    prompt_query_key: str,
    prompt_annotation_key: str,
    supported_branch_names: Sequence[str],
    default_branch_name: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    resolve_axes: PipeAxesResolver,
    sample_network: PipeSampler,
    bind_result: PipeBinder,
) -> TaskOutput:
    """Run the neutral pipe lifecycle around task-owned sampling and binding hooks."""

    branch_name, branch_probs, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_branch_names),
        default_query_id=str(default_branch_name),
        task_id=str(owner_id),
        namespace=f"{owner_id}.query",
    )
    forced_params = force_query_id_params(task_params, query_id=str(branch_name))
    gen_defaults, render_defaults, prompt_defaults, background_defaults, noise_defaults = load_pipe_defaults(str(owner_id))
    axes = resolve_axes(instance_seed=int(instance_seed), params=forced_params, gen_defaults=gen_defaults)
    sample = sample_network(int(instance_seed), forced_params, int(max_attempts), axes)
    rendered = render_pipe_assets(
        owner_id=str(owner_id),
        instance_seed=int(instance_seed),
        params=forced_params,
        render_defaults=render_defaults,
        background_defaults=background_defaults,
        noise_defaults=noise_defaults,
        sample=sample,
        axes=axes,
    )
    bound = bind_result(sample, rendered)
    prepared = prepare_pipe_instance(
        owner_id=str(owner_id),
        branch_name=str(branch_name),
        branch_probabilities=branch_probs,
        prompt_query_key=str(prompt_query_key),
        prompt_annotation_key=str(prompt_annotation_key),
        instance_seed=int(instance_seed),
        prompt_defaults=prompt_defaults,
        sample=sample,
        axes=axes,
        rendered=rendered,
        bound=bound,
    )
    return TaskOutput(
        prompt=str(prepared.prompt),
        answer_gt=TypedValue(type=str(bound.answer_type), value=bound.answer_value),
        annotation_gt=TypedValue(type=str(bound.annotation_type), value=list(bound.annotation_value)),
        image=prepared.image,
        image_id="img0",
        trace_payload=dict(prepared.trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(branch_name),
        prompt_variants=dict(prepared.prompt_variants),
    )


__all__ = [
    "FALLBACK_DEFAULTS",
    "SCENE_ID",
    "bind_pipe_point_sequence",
    "bind_pipe_point_set",
    "bind_pipe_segment_set",
    "integer_support",
    "load_pipe_defaults",
    "pipe_scene_entities",
    "pipe_query_distance_support",
    "prepare_pipe_instance",
    "render_pipe_assets",
    "resolve_node_count",
    "resolve_pipe_bridge_axes",
    "resolve_pipe_style_axes",
    "resolve_pipe_target_axes",
    "sample_pipe_target_network",
    "select_support_value",
    "run_pipe_objective",
]
