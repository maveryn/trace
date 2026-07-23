"""Neutral lifecycle helpers for the metro graph scene."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image

from ....core.sampling import uniform_choice
from ....core.types import TypedValue
from ...base import TaskOutput
from ...shared.config_defaults import group_default, required_group_defaults, split_scene_generation_rendering_prompt_defaults
from ...shared.deterministic_sampling import uniform_probability_map
from ...shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_scene_prompt_variants
from ..shared.style import SUPPORTED_NODE_COLOR_NAMES
from ..shared.task_support import format_graph_prompt_label, resolve_graph_named_variant, resolve_graph_render_params
from ..shared.visual_defaults import load_graph_scene_background_defaults, load_graph_scene_noise_defaults
from ...shared.output_metadata import default_task_versions
from ....core.seed import spawn_rng
from ....core.scene_config import get_scene_defaults
from ....core.visual.background import make_background_canvas
from ....core.visual.noise import apply_post_image_noise
from .shared.annotations import projected_metro_station_point_annotation
from .shared.defaults import MetroRouteTaskDefaults
from .shared.output import MetroAnswerAnnotation, MetroPreparedAssets, MetroRouteResolvedAxes
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
from .shared.rendering import render_metro_scene
from .shared.state import MetroRouteNetworkSample, SCENE_ID, SUPPORTED_METRO_LABEL_VARIANTS


FALLBACK_DEFAULTS = MetroRouteTaskDefaults()
SCENE_TITLE = "Metro Route Graph"


def load_metro_defaults(owner_id: str) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load scene defaults for one public metro task."""

    group_defaults = get_scene_defaults("graph", SCENE_ID)
    gen_defaults, render_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        group_defaults if isinstance(group_defaults, Mapping) else {},
        task_id=str(owner_id),
    )
    background_defaults = load_graph_scene_background_defaults(scene_id=SCENE_ID)
    noise_defaults = load_graph_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)
    return dict(gen_defaults), dict(render_defaults), dict(prompt_defaults), dict(background_defaults), dict(noise_defaults)


def support_from_bounds(*, params: Mapping[str, Any], gen_defaults: Mapping[str, Any], low_key: str, high_key: str, default_low: int, default_high: int, feasible: Sequence[int]) -> Tuple[int, ...]:
    """Return a configured integer support intersected with feasible values."""

    low = int(params.get(low_key, group_default(gen_defaults, low_key, int(default_low))))
    high = int(params.get(high_key, group_default(gen_defaults, high_key, int(default_high))))
    if int(low) > int(high):
        raise ValueError("empty metro support")
    feasible_set = {int(value) for value in feasible}
    support = tuple(value for value in range(int(low), int(high) + 1) if int(value) in feasible_set)
    if not support:
        raise ValueError("no feasible metro support")
    return tuple(int(value) for value in support)


def select_support_value(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    owner_id: str,
    support: Sequence[int],
    explicit_keys: Sequence[str],
    namespace_suffix: str,
    balanced: bool = False,
) -> tuple[int, Dict[str, float]]:
    """Select one value from an already objective-filtered support."""

    support_tuple = tuple(int(value) for value in support)
    if not support_tuple:
        raise ValueError("empty metro selection support")
    explicit_value = None
    for key in explicit_keys:
        if params.get(str(key)) is not None:
            explicit_value = int(params[str(key)])
            break
    if explicit_value is not None:
        if int(explicit_value) not in support_tuple:
            raise ValueError("requested metro value is outside configured support")
        return int(explicit_value), dict(uniform_probability_map(support_tuple, selected=int(explicit_value)))
    rng_namespace = f"{owner_id}:{namespace_suffix}:balanced" if bool(balanced) else f"{owner_id}:{namespace_suffix}"
    selected = int(
        uniform_choice(
            spawn_rng(int(instance_seed), rng_namespace),
            support_tuple,
        )
    )
    return int(selected), dict(uniform_probability_map(support_tuple))


def route_count_support_for_target(*, params: Mapping[str, Any], gen_defaults: Mapping[str, Any], target_value: int, feasible_for_route_count: Any) -> Tuple[int, ...]:
    """Return route-count values that can realize one already-selected target."""

    low = int(params.get("route_count_min", group_default(gen_defaults, "route_count_min", FALLBACK_DEFAULTS.route_count_min)))
    high = int(params.get("route_count_max", group_default(gen_defaults, "route_count_max", FALLBACK_DEFAULTS.route_count_max)))
    if int(low) > int(high):
        raise ValueError("empty metro route-count support")
    support = tuple(
        route_count
        for route_count in range(int(low), int(high) + 1)
        if int(target_value) in {int(value) for value in feasible_for_route_count(route_count)}
    )
    if not support:
        raise ValueError("no feasible metro route-count support for selected target")
    return tuple(int(value) for value in support)


def resolve_target_route_style_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    owner_id: str,
    gen_defaults: Mapping[str, Any],
    feasible_values: Sequence[int],
    feasible_for_route_count: Any,
    low_key: str = "target_count_min",
    high_key: str = "target_count_max",
    explicit_target_keys: Sequence[str] = ("target_count",),
    default_low: int | None = None,
    default_high: int | None = None,
    query_distance: int = 0,
    target_namespace: str = "target_support_v0",
    route_namespace: str = "route_count_v0",
    balanced_target_selection: bool = False,
) -> MetroRouteResolvedAxes:
    """Resolve objective-filtered target count, route count, labels, and style axes."""

    target_support = support_from_bounds(
        params=params,
        gen_defaults=gen_defaults,
        low_key=str(low_key),
        high_key=str(high_key),
        default_low=FALLBACK_DEFAULTS.target_count_min if default_low is None else int(default_low),
        default_high=FALLBACK_DEFAULTS.target_count_max if default_high is None else int(default_high),
        feasible=tuple(int(value) for value in feasible_values),
    )
    target_count, target_probs = select_support_value(
        params=params,
        instance_seed=int(instance_seed),
        owner_id=str(owner_id),
        support=target_support,
        explicit_keys=tuple(str(value) for value in explicit_target_keys),
        namespace_suffix=str(target_namespace),
        balanced=bool(balanced_target_selection),
    )
    route_support = route_count_support_for_target(
        params=params,
        gen_defaults=gen_defaults,
        target_value=int(target_count),
        feasible_for_route_count=feasible_for_route_count,
    )
    route_count, route_probs = select_support_value(
        params=params,
        instance_seed=int(instance_seed),
        owner_id=str(owner_id),
        support=route_support,
        explicit_keys=("route_count",),
        namespace_suffix=str(route_namespace),
    )
    label_variant, label_probs, node_color_name, color_probs = resolve_style_axes(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        owner_id=str(owner_id),
    )
    return MetroRouteResolvedAxes(
        target_count=int(target_count),
        route_count=int(route_count),
        label_variant=str(label_variant),
        node_color_name=str(node_color_name),
        query_distance=int(query_distance),
        target_count_probabilities=dict(target_probs),
        route_count_probabilities=dict(route_probs),
        label_variant_probabilities=dict(label_probs),
        node_color_name_probabilities=dict(color_probs),
    )


def resolve_route_metric_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    owner_id: str,
    feasible_values_fn: Any,
    metric_kwargs: Mapping[str, Any] | None = None,
    low_key: str = "target_count_min",
    high_key: str = "target_count_max",
    explicit_target_keys: Sequence[str] = ("target_count",),
    query_distance: int = 0,
    balanced_target_selection: bool = False,
) -> MetroRouteResolvedAxes:
    """Resolve route-count and style axes for objectives backed by a route metric."""

    gen_defaults, *_ = load_metro_defaults(str(owner_id))
    route_low = int(params.get("route_count_min", group_default(gen_defaults, "route_count_min", FALLBACK_DEFAULTS.route_count_min)))
    route_high = int(params.get("route_count_max", group_default(gen_defaults, "route_count_max", FALLBACK_DEFAULTS.route_count_max)))
    kwargs = dict(metric_kwargs or {})
    return resolve_target_route_style_axes(
        instance_seed=int(instance_seed),
        params=params,
        owner_id=str(owner_id),
        gen_defaults=gen_defaults,
        feasible_values=feasible_values_fn(route_count_min=route_low, route_count_max=route_high, **kwargs),
        feasible_for_route_count=lambda route_count: feasible_values_fn(
            route_count_min=int(route_count),
            route_count_max=int(route_count),
            **kwargs,
        ),
        low_key=str(low_key),
        high_key=str(high_key),
        explicit_target_keys=tuple(str(value) for value in explicit_target_keys),
        query_distance=int(query_distance),
        balanced_target_selection=bool(balanced_target_selection),
    )


def resolve_style_axes(*, params: Mapping[str, Any], gen_defaults: Mapping[str, Any], instance_seed: int, owner_id: str) -> tuple[str, Dict[str, float], str, Dict[str, float]]:
    """Resolve label and station-color style axes for a metro instance."""

    label_rng = spawn_rng(int(instance_seed), f"{owner_id}.label_variant")
    label_variant, label_probs = resolve_graph_named_variant(
        label_rng,
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="label_variant",
        weights_key="label_variant_weights",
        balance_flag_key="balanced_label_variant_sampling",
        supported=SUPPORTED_METRO_LABEL_VARIANTS,
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
    return str(label_variant), dict(label_probs), str(node_color_name), dict(color_probs)


def finish_metro_result(*, assets: MetroPreparedAssets, branch_name: str) -> TaskOutput:
    """Package already-bound metro assets into the public task result."""

    return TaskOutput(
        prompt=str(assets.prompt),
        answer_gt=TypedValue(type="integer", value=int(assets.answer_annotation.answer_value)),
        annotation_gt=TypedValue(
            type=str(assets.answer_annotation.annotation_type),
            value=list(assets.answer_annotation.annotation_value),
        ),
        image=assets.image,
        image_id="img0",
        trace_payload=dict(assets.trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(branch_name),
        prompt_variants=dict(assets.prompt_variants),
    )


def with_query_id_probabilities(trace_payload: Mapping[str, Any], branch_probs: Mapping[str, float]) -> Dict[str, Any]:
    """Attach query selection probabilities to an already-built trace payload."""

    trace = dict(trace_payload)
    spec = dict(trace.get("query_spec") or {})
    params = dict(spec.get("params") or {})
    params["query_id_probabilities"] = {str(key): float(value) for key, value in branch_probs.items()}
    spec["params"] = params
    trace["query_spec"] = spec
    return trace


def station_annotation(sample: MetroRouteNetworkSample, rendered_scene: Any, *, labels: Sequence[str], ordered: bool, answer_value: int, witness_extra: Mapping[str, Any] | None = None) -> MetroAnswerAnnotation:
    """Project station labels and bind a point-set or point-sequence answer annotation."""

    projection = projected_metro_station_point_annotation(rendered_scene, tuple(str(label) for label in labels))
    if bool(ordered):
        annotation_type = "point_sequence"
        annotation = [list(point) for point in projection["pixel_point_sequence"]]
        witness_type = "station_label_sequence"
    else:
        annotation_type = "point_set"
        annotation = [list(point) for point in projection["pixel_point_set"]]
        witness_type = "station_label_set"
    witness = {"type": witness_type, "labels": [str(label) for label in labels]}
    if witness_extra:
        witness.update(dict(witness_extra))
    return MetroAnswerAnnotation(
        answer_value=int(answer_value),
        annotation_type=str(annotation_type),
        annotation_value=list(annotation),
        witness_symbolic=dict(witness),
        projected_annotation={"type": str(annotation_type), **dict(projection)},
    )


def _station_entities(sample: MetroRouteNetworkSample, rendered_scene: Any, branch_name: str) -> list[Dict[str, Any]]:
    target_label_set = {str(label) for label in (sample.target_labels if sample.target_labels else sample.transfer_labels)}
    return [
        {
            "entity_id": f"station_{station.label}",
            "entity_kind": "metro_station",
            "label": str(station.label),
            "grid_point": list(station.grid_point),
            "route_ids": list(station.route_ids),
            "route_count": int(len(station.route_ids)),
            "is_transfer": bool(station.is_transfer),
            "is_witness_node": bool(str(station.label) in target_label_set),
            "is_query_node": bool(str(station.label) == str(sample.query_label)),
            "is_source_node": bool(str(station.label) == str(sample.source_label)),
            "is_goal_node": bool(str(station.label) == str(sample.goal_label)),
            "center_px": list(station.center_xy),
            "bbox_xyxy": list(station.bbox_xyxy),
        }
        for station in rendered_scene.stations
    ]


def _route_entities(rendered_scene: Any) -> list[Dict[str, Any]]:
    return [
        {
            "entity_id": f"route_{route.route_id}",
            "entity_kind": "metro_route",
            "route_id": str(route.route_id),
            "route_name": str(route.route_name),
            "color_rgb": list(route.color_rgb),
            "station_labels": list(route.station_labels),
            "polyline_px": [list(point) for point in route.polyline_px],
        }
        for route in rendered_scene.routes
    ]


def _common_trace_fields(sample: MetroRouteNetworkSample, axes: MetroRouteResolvedAxes) -> Dict[str, Any]:
    route_by_id = {str(route.route_id): route for route in sample.route_templates}
    query_route_ids = tuple(str(route_id) for route_id in getattr(sample, "query_route_ids", ()) or ())
    query_route_names = tuple(str(route_by_id[route_id].route_name) for route_id in query_route_ids if route_id in route_by_id)
    return {
        "target_count": int(axes.target_count),
        "route_count": int(axes.route_count),
        "station_count": int(sample.station_count),
        "query_distance": int(axes.query_distance),
        "query_label": str(sample.query_label),
        "query_route_ids": list(query_route_ids),
        "query_route_names": list(query_route_names),
        "source_label": str(sample.source_label),
        "goal_label": str(sample.goal_label),
        "matching_labels": list(sample.target_labels or sample.transfer_labels),
        "transfer_station_labels": list(sample.transfer_labels),
        "single_route_station_labels": [str(label) for label, route_ids in sample.station_route_ids_by_label.items() if len(route_ids) == 1],
        "terminal_station_labels": list(sample.terminal_labels),
        "target_single_route_count": int(sample.target_single_route_count),
        "target_exact_distance_count": int(sample.target_exact_distance_count),
        "target_shortest_path_length": int(sample.target_shortest_path_length),
        "route_station_labels": {str(key): list(values) for key, values in sample.route_station_labels.items()},
        "station_route_ids_by_label": {str(key): list(values) for key, values in sample.station_route_ids_by_label.items()},
        "adjacency_by_label": {str(key): list(values) for key, values in sample.adjacency_by_label.items()},
        "edge_labels": [list(edge) for edge in sample.edge_labels],
        "label_variant": str(axes.label_variant),
        "node_color_name": str(axes.node_color_name),
    }


def prepare_metro_assets(
    *,
    owner_id: str,
    branch_name: str,
    prompt_query_key: str,
    prompt_annotation_key: str,
    instance_seed: int,
    params: Mapping[str, Any],
    sample: MetroRouteNetworkSample,
    axes: MetroRouteResolvedAxes,
    answer_value: int,
    annotation_labels: Sequence[str],
    ordered_annotation: bool,
    witness_extra: Mapping[str, Any] | None = None,
    json_example_key: str | None = None,
) -> MetroPreparedAssets:
    """Render a metro sample, compose the prompt, and build common trace metadata."""

    gen_defaults, render_defaults, prompt_defaults, background_defaults, noise_defaults = load_metro_defaults(str(owner_id))
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
    rendered_scene = render_metro_scene(
        metro_sample=sample,
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
    answer_annotation = station_annotation(
        sample,
        rendered_scene,
        labels=tuple(str(label) for label in annotation_labels),
        ordered=bool(ordered_annotation),
        answer_value=int(answer_value),
        witness_extra=dict(witness_extra or {}),
    )

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
            *((str(json_example_key),) if json_example_key else ()),
        ),
        context=f"prompt defaults for {owner_id}",
    )
    question_slots = {
        "query_label": format_graph_prompt_label(str(sample.query_label), label_variant=str(sample.label_variant)),
        "source_label": format_graph_prompt_label(str(sample.source_label), label_variant=str(sample.label_variant)),
        "via_label": format_graph_prompt_label(str(sample.query_label), label_variant=str(sample.label_variant)),
        "goal_label": format_graph_prompt_label(str(sample.goal_label), label_variant=str(sample.label_variant)),
        "query_distance": int(axes.query_distance),
    }
    query_route_names = list(_common_trace_fields(sample, axes).get("query_route_names") or [])
    question_slots["route_name"] = str(query_route_names[0]) if query_route_names else ""
    prompt_json_example, prompt_json_example_answer_only = json_examples_for_annotation(str(answer_annotation.annotation_type))
    if json_example_key:
        prompt_json_example = str(prompt_defaults_required[str(json_example_key)])
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
    common_trace["annotation_labels"] = [str(label) for label in annotation_labels]
    query_params = {
        "query_id": str(branch_name),
        "scene_id": SCENE_ID,
        "target_count": int(axes.target_count),
        "target_count_probabilities": dict(axes.target_count_probabilities or {}),
        "query_distance": int(axes.query_distance),
        "route_count": int(axes.route_count),
        "route_count_probabilities": dict(axes.route_count_probabilities or {}),
        "station_count": int(sample.station_count),
        "label_variant": str(axes.label_variant),
        "label_variant_probabilities": dict(axes.label_variant_probabilities or {}),
        "node_color_name": str(axes.node_color_name),
        "node_color_name_probabilities": dict(axes.node_color_name_probabilities or {}),
    }
    trace_payload = {
        "scene_ir": {
            "scene_kind": "metro",
            "domain": "graph",
            "scene_id": SCENE_ID,
            "task_id": str(owner_id),
            "query_id": str(branch_name),
            "entities": [*_station_entities(sample, rendered_scene, str(branch_name)), *_route_entities(rendered_scene)],
            "relations": {
                "graph_directionality": "undirected",
                "route_semantics": "a transfer station is served by two or more colored routes",
                "query_id": str(branch_name),
                **dict(common_trace),
            },
            "frames": {
                "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                "panels": dict(rendered_scene.panel_geometry),
            },
        },
        "query_spec": {
            "query_id": str(branch_name),
            "scene_id": SCENE_ID,
            "template_id": str(prompt_defaults_required.get("bundle_id") or PROMPT_BUNDLE_ID),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": dict(query_params),
        },
        "render_spec": {
            "coord_space": "pixel",
            "scene_id": SCENE_ID,
            "canvas_size": list(rendered_scene.panel_geometry["canvas_size"]),
            "panel_geometry": dict(rendered_scene.panel_geometry),
            "style": {
                "theme_tone": str(render_params.theme_tone),
                "panel_style_variant": str(render_params.panel_style_variant),
                "background_color_rgb": list(render_params.background_color_rgb),
                "panel_fill_rgb": list(render_params.panel_fill_rgb),
                "panel_border_rgb": list(render_params.panel_border_rgb),
                "title_color_rgb": list(render_params.title_color_rgb),
                "route_line_width_px": int(rendered_scene.route_line_width_px),
                "station_radius_px": int(rendered_scene.station_radius_px),
                "transfer_station_radius_px": int(rendered_scene.transfer_station_radius_px),
                "label_font_size_px": int(render_params.label_font_size_px),
                "resolved_label_font_size_px": int(rendered_scene.resolved_label_font_size_px),
                "font_family": str(render_params.font_family or ""),
                "font_asset": dict(render_params.font_asset) if isinstance(render_params.font_asset, Mapping) else {},
                "font_asset_version": str(render_params.font_asset_version or ""),
                "font_exclusion_reason": str(render_params.font_exclusion_reason),
                "context_text_elements": list(rendered_scene.panel_geometry.get("context_text_elements", [])),
                "background_meta": dict(background_meta),
                "post_image_noise_meta": dict(post_noise_meta),
            },
        },
        "render_map": {"image_id": "img0", "anchors": {}},
        "execution_trace": {
            "query_id": str(branch_name),
            "scene_variant": "metro",
            "scene_id": SCENE_ID,
            "question_format": str(branch_name),
            "answer": int(answer_annotation.answer_value),
            **dict(common_trace),
        },
        "witness_symbolic": dict(answer_annotation.witness_symbolic),
        "projected_annotation": dict(answer_annotation.projected_annotation),
        "task_versions": default_task_versions(),
    }
    return MetroPreparedAssets(
        prompt=str(prompt_artifacts.prompt),
        image=image,
        answer_annotation=answer_annotation,
        trace_payload=dict(trace_payload),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = [
    "FALLBACK_DEFAULTS",
    "finish_metro_result",
    "prepare_metro_assets",
    "load_metro_defaults",
    "resolve_target_route_style_axes",
    "resolve_route_metric_axes",
    "resolve_style_axes",
    "route_count_support_for_target",
    "select_support_value",
    "station_annotation",
    "support_from_bounds",
    "with_query_id_probabilities",
]
