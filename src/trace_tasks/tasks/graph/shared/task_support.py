"""Shared graph-task support for balanced variants and render defaults."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, Mapping, Tuple

from ....core.sampling import normalize_positive_weights, weighted_choice
from ....core.seed import spawn_rng
from ...shared.color_format import rgb_to_hex
from ...shared.config_defaults import group_default
from ...shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from ...shared.named_colors import named_color
from ...shared.text_legibility import resolve_readable_text_style, text_legibility_summary_from_records
from ...shared.variant_sampling import (
    apply_balanced_variant_sampling,
    has_non_null_param,
    is_uniform_probability_map,
    resolve_variant,
)
from .graph_scene import GraphRenderParams
from .information_style import (
    graph_surface_roles_from_information_style,
    infer_graph_scene_id,
    resolve_graph_information_style,
)
from .label_assets import default_graph_label_bucket_weights, resolve_graph_edge_label_support
from .style import (
    SUPPORTED_GRAPH_PANEL_STYLE_VARIANTS,
    SUPPORTED_GRAPH_THEME_TONES,
    apply_graph_panel_style,
    build_graph_named_color_theme,
)


def format_graph_prompt_label(label: str, *, label_variant: str) -> str:
    """Return a prompt-facing node label, quoting named text labels."""

    text = str(label)
    if str(label_variant) == "named":
        return f'"{text}"'
    return text


def resolve_graph_named_variant(
    rng,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Tuple[str, ...],
    instance_seed: int,
    task_id: str,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced named graph-query id axis."""

    selected_variant, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=supported,
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    variant = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        selected_variant=str(selected_variant),
        variant_probabilities=probabilities,
        supported_variants=supported,
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=f"{str(task_id)}:{str(namespace)}",
    )
    return str(variant), {str(key): float(value) for key, value in sorted(probabilities.items())}


def resolve_graph_static_node_color_name(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    fallback: str,
    supported: Sequence[str],
    key: str = "node_color_name",
) -> Tuple[str, Dict[str, float]]:
    """Resolve one explicitly configured graph theme color and probability map."""

    raw = params.get(str(key), group_default(gen_defaults, str(key), str(fallback)))
    color_name = str(raw).strip().lower()
    supported_names = tuple(str(name) for name in supported)
    if color_name not in set(supported_names):
        raise ValueError(f"unsupported {key}: {raw}")
    return str(color_name), {
        str(name): (1.0 if str(name) == str(color_name) else 0.0)
        for name in supported_names
    }


def resolve_graph_balanced_node_color_name(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    task_id: str,
    supported: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    """Resolve the common balanced non-semantic graph node-color axis."""

    color_rng = spawn_rng(int(instance_seed), f"{str(task_id)}.node_color_name")
    return resolve_graph_named_variant(
        color_rng,
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="node_color_name",
        weights_key="node_color_name_weights",
        balance_flag_key="balanced_node_color_name_sampling",
        supported=tuple(str(item) for item in supported),
        instance_seed=int(instance_seed),
        task_id=str(task_id),
        namespace="node_color_name",
    )


def resolve_graph_edge_label_support_from_params(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    task_id: str,
    default_support_size: int = 6,
    default_min_chars: int = 3,
    default_max_chars: int = 12,
    exclude_labels: Sequence[str] = (),
) -> Tuple[Tuple[str, ...], Dict[str, Any]]:
    """Resolve visible edge-text label support from explicit params or shared label assets."""

    excluded = {
        str(label).strip().lower()
        for label in exclude_labels
        if str(label).strip()
    }
    raw_support = params.get("edge_label_support", group_default(gen_defaults, "edge_label_support", None))
    if raw_support is not None:
        if isinstance(raw_support, str):
            support = tuple(str(item).strip().lower() for item in raw_support.split(",") if str(item).strip())
        else:
            support = tuple(str(item).strip().lower() for item in raw_support if str(item).strip())
        if excluded:
            support = tuple(str(label) for label in support if str(label).strip().lower() not in excluded)
        if len(set(support)) != len(support) or len(support) < 2:
            raise ValueError("edge_label_support must contain at least two unique labels")
        return tuple(str(label) for label in support), {
            "edge_label_source_kind": "explicit_support",
            "edge_label_bucket": "",
            "edge_label_manifest": "",
            "edge_label_filter": {"excluded_node_labels": sorted(excluded)} if excluded else {},
            "edge_label_bucket_probabilities": {},
        }

    support_size = int(
        params.get(
            "edge_label_support_size",
            group_default(gen_defaults, "edge_label_support_size", int(default_support_size)),
        )
    )
    min_chars = int(
        params.get(
            "edge_label_min_chars",
            group_default(gen_defaults, "edge_label_min_chars", int(default_min_chars)),
        )
    )
    max_chars = int(
        params.get(
            "edge_label_max_chars",
            group_default(gen_defaults, "edge_label_max_chars", int(default_max_chars)),
        )
    )
    raw_bucket_weights = params.get(
        "edge_label_bucket_weights",
        group_default(gen_defaults, "edge_label_bucket_weights", default_graph_label_bucket_weights()),
    )
    bucket_weights = dict(raw_bucket_weights) if isinstance(raw_bucket_weights, Mapping) else default_graph_label_bucket_weights()
    resolved = resolve_graph_edge_label_support(
        spawn_rng(int(instance_seed), f"{str(task_id)}.edge_label_bucket"),
        support_size=int(support_size),
        min_chars=int(min_chars),
        max_chars=int(max_chars),
        bucket_weights=bucket_weights,
        exclude_labels=tuple(sorted(excluded)),
    )
    label_filter = dict(resolved.label_filter)
    if excluded:
        label_filter["excluded_node_labels"] = sorted(excluded)
    return tuple(str(label) for label in resolved.labels), {
        "edge_label_source_kind": str(resolved.label_source_kind),
        "edge_label_bucket": str(resolved.label_bucket),
        "edge_label_manifest": str(resolved.label_manifest),
        "edge_label_filter": dict(label_filter),
        "edge_label_bucket_probabilities": dict(resolved.label_bucket_probabilities),
    }


def graph_int_support(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback_min: int,
    fallback_max: int,
) -> Tuple[int, ...]:
    """Resolve an inclusive integer support from graph task params/defaults."""

    lower = int(params.get(f"{key}_min", group_default(defaults, f"{key}_min", fallback_min)))
    upper = int(params.get(f"{key}_max", group_default(defaults, f"{key}_max", fallback_max)))
    if upper < lower:
        raise ValueError(f"{key} support is empty")
    return tuple(range(int(lower), int(upper) + 1))


def graph_balanced_axis_count(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    probabilities: Mapping[str, float],
    balance_flag_key: str,
    explicit_keys: Sequence[str],
    weights_key: str,
) -> int:
    """Return the active balanced count for one already-resolved graph axis."""

    enabled = bool(params.get(str(balance_flag_key), group_default(gen_defaults, str(balance_flag_key), True)))
    if not bool(enabled):
        return 1
    if any(has_non_null_param(params, key) for key in (*tuple(str(key) for key in explicit_keys), str(weights_key))):
        return 1
    if not is_uniform_probability_map(probabilities):
        return 1
    active_count = sum(1 for value in probabilities.values() if float(value) > 0.0)
    return max(1, int(active_count))


def graph_uniform_label_probability_map(values: Sequence[str], *, selected: str | None = None) -> Dict[str, float]:
    """Return a deterministic uniform probability map over graph text labels."""

    support = tuple(str(value) for value in values)
    if not support:
        return {}
    if selected is not None:
        return {str(selected): 1.0}
    probability = 1.0 / float(len(support))
    return {str(value): float(probability) for value in support}


def graph_edge_label_entries(edge_attribute_labels_by_label: Mapping[Tuple[str, str], str]) -> Tuple[Dict[str, Any], ...]:
    """Return JSON-friendly visible edge-label metadata entries."""

    return tuple(
        {
            "edge": [str(left), str(right)],
            "edge_label": str(edge_label),
        }
        for (left, right), edge_label in edge_attribute_labels_by_label.items()
    )


def graph_palette_rgb_by_name(names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Return trace metadata for named graph colors."""

    return {
        str(name): {
            "rgb": list(named_color(str(name))),
            "hex": str(rgb_to_hex(named_color(str(name)))),
        }
        for name in names
    }


def graph_semantic_node_style_by_label(
    *,
    node_color_names_by_label: Mapping[str, str],
    theme_tone: str,
) -> Dict[str, Dict[str, Any]]:
    """Resolve per-node render styles from shared named graph colors."""

    style_by_color: Dict[str, Dict[str, Any]] = {}
    for color_name in sorted(set(str(value) for value in node_color_names_by_label.values())):
        color_theme = build_graph_named_color_theme(str(color_name), theme_tone=str(theme_tone))
        style_by_color[str(color_name)] = {
            "color_name": str(color_name),
            "fill_rgb": tuple(int(value) for value in color_theme.node_fill_rgb),
            "border_rgb": tuple(int(value) for value in color_theme.node_border_rgb),
            "label_text_rgb": tuple(int(value) for value in color_theme.label_text_rgb),
            "label_stroke_rgb": tuple(int(value) for value in color_theme.label_stroke_rgb),
        }
    return {
        str(label): dict(style_by_color[str(color_name)])
        for label, color_name in node_color_names_by_label.items()
    }


def resolve_graph_render_params(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    task_id: str,
    render_defaults: Mapping[str, Any],
    fallback_defaults: Any,
    node_color_name: str,
    node_shape_variant: str,
    edge_routing_variant: str = "straight",
) -> GraphRenderParams:
    """Resolve one concrete graph render-parameter set for a graph task."""

    radius_min = int(
        params.get(
            "node_radius_min_px",
            group_default(render_defaults, "node_radius_min_px", int(getattr(fallback_defaults, "node_radius_min_px"))),
        )
    )
    radius_max = int(
        params.get(
            "node_radius_max_px",
            group_default(render_defaults, "node_radius_max_px", int(getattr(fallback_defaults, "node_radius_max_px"))),
        )
    )
    render_rng = spawn_rng(int(instance_seed), f"{str(task_id)}.render")
    node_radius = int(render_rng.randint(int(radius_min), int(max(radius_min, radius_max))))
    explicit_tone = str(params.get("theme_tone", "")).strip().lower()
    if explicit_tone in SUPPORTED_GRAPH_THEME_TONES:
        theme_tone = str(explicit_tone)
    else:
        raw_tone_weights = params.get(
            "theme_tone_weights",
            group_default(render_defaults, "theme_tone_weights", {str(key): 1.0 for key in SUPPORTED_GRAPH_THEME_TONES}),
        )
        tone_weights = dict(raw_tone_weights) if isinstance(raw_tone_weights, Mapping) else {}
        tone_probabilities = normalize_positive_weights(
            {str(key): float(tone_weights.get(str(key), 0.0)) for key in SUPPORTED_GRAPH_THEME_TONES},
            default_keys=SUPPORTED_GRAPH_THEME_TONES,
        )
        theme_tone = weighted_choice(
            spawn_rng(int(instance_seed), f"{str(task_id)}.render.theme_tone"),
            tone_probabilities,
            sort_keys=True,
        )
    try:
        color_theme = build_graph_named_color_theme(str(node_color_name), theme_tone=str(theme_tone))
    except Exception:
        color_theme = None

    explicit_panel_style = str(params.get("panel_style_variant", "")).strip().lower()
    if explicit_panel_style in SUPPORTED_GRAPH_PANEL_STYLE_VARIANTS:
        panel_style_variant = str(explicit_panel_style)
    else:
        raw_panel_style_weights = params.get(
            "panel_style_variant_weights",
            group_default(
                render_defaults,
                "panel_style_variant_weights",
                {str(key): 1.0 for key in SUPPORTED_GRAPH_PANEL_STYLE_VARIANTS},
            ),
        )
        panel_style_weights = dict(raw_panel_style_weights) if isinstance(raw_panel_style_weights, Mapping) else {}
        panel_style_probabilities = normalize_positive_weights(
            {str(key): float(panel_style_weights.get(str(key), 0.0)) for key in SUPPORTED_GRAPH_PANEL_STYLE_VARIANTS},
            default_keys=SUPPORTED_GRAPH_PANEL_STYLE_VARIANTS,
        )
        panel_style_variant = weighted_choice(
            spawn_rng(int(instance_seed), f"{str(task_id)}.render.panel_style_variant"),
            panel_style_probabilities,
            sort_keys=True,
        )
    if color_theme is not None:
        color_theme = apply_graph_panel_style(color_theme, panel_style_variant=str(panel_style_variant))
    graph_scene_id = infer_graph_scene_id(str(task_id))
    information_style_params: Dict[str, Any] = dict(render_defaults)
    information_style_params.update(dict(params))
    information_style, information_style_meta = resolve_graph_information_style(
        instance_seed=int(instance_seed),
        params=information_style_params,
        scene_id=str(graph_scene_id),
        protected_colors=(tuple(int(value) for value in named_color(str(node_color_name))),),
        allow_dark=False,
    )
    information_roles = graph_surface_roles_from_information_style(information_style)

    font_params: Dict[str, Any] = dict(render_defaults)
    font_params.update(dict(params))
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{str(task_id)}.render.font_family",
        params=font_params,
    )
    font_record = get_font_family_record(str(font_family))

    def _render_value(key: str, fallback: Any | None = None) -> Any:
        fallback_value = getattr(fallback_defaults, key, fallback)
        return params.get(key, group_default(render_defaults, key, fallback_value))

    raw_context_block_position_weights = _render_value("context_block_position_weights", {})
    context_block_position_weights = (
        dict(raw_context_block_position_weights)
        if isinstance(raw_context_block_position_weights, Mapping)
        else {}
    )
    raw_context_block_clutter_level_weights = _render_value("context_block_clutter_level_weights", {})
    context_block_clutter_level_weights = (
        dict(raw_context_block_clutter_level_weights)
        if isinstance(raw_context_block_clutter_level_weights, Mapping)
        else {}
    )
    node_fill_rgb = tuple(
        int(value) for value in (color_theme.node_fill_rgb if color_theme is not None else _render_value("node_fill_rgb"))
    )
    node_border_rgb = tuple(
        int(value)
        for value in (color_theme.node_border_rgb if color_theme is not None else _render_value("node_border_rgb"))
    )
    node_label_style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace=f"{str(task_id)}.render.node_label_text",
        role="graph_node_label_text",
        surface_rgbs=(node_fill_rgb,),
        preferred_rgbs=(
            tuple(int(value) for value in information_roles["label_text_rgb"]),
            (255, 255, 255),
            (10, 14, 22),
        ),
        min_contrast_ratio=4.0,
        min_lab_distance=24.0,
    )
    information_records = []
    if isinstance(information_style_meta, Mapping):
        information_legibility = information_style_meta.get("text_legibility")
        if isinstance(information_legibility, Mapping) and isinstance(information_legibility.get("records"), list):
            information_records = [
                dict(record)
                for record in information_legibility["records"]
                if isinstance(record, Mapping)
            ]
    text_legibility = text_legibility_summary_from_records(
        [*information_records, node_label_style.metadata()]
    )

    return GraphRenderParams(
        canvas_width=int(_render_value("canvas_width")),
        canvas_height=int(_render_value("canvas_height")),
        outer_margin_px=int(_render_value("outer_margin_px")),
        panel_padding_px=int(_render_value("panel_padding_px")),
        panel_corner_radius_px=int(_render_value("panel_corner_radius_px")),
        panel_title_font_size_px=int(_render_value("panel_title_font_size_px")),
        node_shape_variant=str(node_shape_variant),
        edge_routing_variant=str(edge_routing_variant),
        node_radius_px=int(node_radius),
        edge_width_px=int(_render_value("edge_width_px")),
        arrow_length_px=int(_render_value("arrow_length_px")),
        arrow_width_px=int(_render_value("arrow_width_px")),
        node_border_width_px=int(_render_value("node_border_width_px")),
        label_font_size_px=int(_render_value("label_font_size_px")),
        theme_tone=str(theme_tone),
        panel_style_variant=str(panel_style_variant),
        background_color_rgb=tuple(int(value) for value in information_roles["background_color_rgb"]),
        panel_fill_rgb=tuple(int(value) for value in information_roles["panel_fill_rgb"]),
        panel_border_rgb=tuple(int(value) for value in information_roles["panel_border_rgb"]),
        title_color_rgb=tuple(int(value) for value in information_roles["title_color_rgb"]),
        edge_color_rgb=tuple(int(value) for value in information_roles["edge_color_rgb"]),
        node_fill_rgb=tuple(int(value) for value in node_fill_rgb),
        node_border_rgb=tuple(int(value) for value in node_border_rgb),
        label_text_rgb=tuple(int(value) for value in node_label_style.fill_rgb),
        label_stroke_rgb=tuple(int(value) for value in node_label_style.stroke_rgb),
        information_scene_style=dict(information_style_meta),
        text_legibility=dict(text_legibility),
        font_family=str(font_family),
        font_asset=dict(font_record.to_trace()),
        font_asset_version=str(font_asset_version()),
        content_jitter_max_px=int(_render_value("content_jitter_max_px", 0)),
        context_text_probability=float(_render_value("context_text_probability", 0.0)),
        context_text_max_elements=int(_render_value("context_text_max_elements", 0)),
        context_block_probability=float(_render_value("context_block_probability", 0.0)),
        context_block_max_elements=int(_render_value("context_block_max_elements", 0)),
        context_block_position_weights=dict(context_block_position_weights),
        context_block_clutter_level_weights=dict(context_block_clutter_level_weights),
    )


__all__ = [
    "format_graph_prompt_label",
    "graph_balanced_axis_count",
    "graph_edge_label_entries",
    "graph_int_support",
    "graph_palette_rgb_by_name",
    "graph_semantic_node_style_by_label",
    "graph_uniform_label_probability_map",
    "resolve_graph_balanced_node_color_name",
    "resolve_graph_edge_label_support_from_params",
    "resolve_graph_named_variant",
    "resolve_graph_render_params",
    "resolve_graph_static_node_color_name",
]
