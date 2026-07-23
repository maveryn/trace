"""Render-style resolution for the named-path icons scene."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ....shared.config_defaults import group_default
from ....shared.text_legibility import resolve_readable_text_style, text_legibility_summary_from_records
from ...shared.icon_task_rendering import resolve_icon_render_params, resolve_icon_rgb_param

from .defaults import NamedPathDefaults, SCENE_ID


def resolve_named_path_render_params(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    fallback_defaults: NamedPathDefaults,
    instance_seed: int,
) -> Dict[str, Any]:
    """Resolve named-path render params and readable label styles."""

    render_params = resolve_icon_render_params(
        params=params,
        render_defaults=render_defaults,
        fallback_defaults=fallback_defaults,
        instance_seed=int(instance_seed),
    )
    for key in (
        "path_stroke_width_px",
        "path_line_alpha",
        "path_stop_radius_px",
        "path_horizontal_margin_px",
        "path_vertical_margin_px",
        "path_amplitude_min_px",
        "path_amplitude_max_px",
        "icon_collision_gap_px",
        "candidate_label_font_size_px",
        "candidate_label_padding_px",
        "candidate_label_gap_px",
        "endpoint_label_font_size_px",
    ):
        render_params[key] = int(
            params.get(key, group_default(render_defaults, key, getattr(fallback_defaults, key)))
        )
    for key in (
        "candidate_label_color_rgb",
        "candidate_label_background_rgb",
        "candidate_label_border_rgb",
        "path_color_rgb",
        "stop_fill_rgb",
        "stop_outline_rgb",
        "endpoint_label_color_rgb",
        "endpoint_label_background_rgb",
    ):
        render_params[key] = resolve_icon_rgb_param(
            params=params,
            render_defaults=render_defaults,
            key=key,
            fallback=getattr(fallback_defaults, key),
            instance_seed=int(instance_seed),
        )

    previous_records = []
    previous_legibility = render_params.get("text_legibility")
    if isinstance(previous_legibility, Mapping) and isinstance(previous_legibility.get("records"), list):
        previous_records = [dict(record) for record in previous_legibility["records"] if isinstance(record, Mapping)]

    candidate_label_style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_ID}:candidate_label_text",
        role="named_path_candidate_label_text",
        surface_rgbs=(
            tuple(int(value) for value in render_params["candidate_label_background_rgb"]),
        ),
        preferred_rgbs=(tuple(int(value) for value in render_params["candidate_label_color_rgb"]),),
    )
    render_params["candidate_label_color_rgb"] = tuple(int(value) for value in candidate_label_style.fill_rgb)
    render_params["candidate_label_stroke_rgb"] = tuple(
        int(value) for value in render_params["candidate_label_background_rgb"]
    )
    candidate_label_record = candidate_label_style.metadata()
    candidate_label_record["stroke_rgb"] = list(render_params["candidate_label_stroke_rgb"])

    endpoint_label_style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_ID}:endpoint_label_text",
        role="named_path_endpoint_label_text",
        surface_rgbs=(
            tuple(int(value) for value in render_params["endpoint_label_background_rgb"]),
        ),
        preferred_rgbs=(tuple(int(value) for value in render_params["endpoint_label_color_rgb"]),),
    )
    render_params["endpoint_label_color_rgb"] = tuple(int(value) for value in endpoint_label_style.fill_rgb)
    render_params["endpoint_label_stroke_rgb"] = tuple(
        int(value) for value in render_params["endpoint_label_background_rgb"]
    )
    endpoint_label_record = endpoint_label_style.metadata()
    endpoint_label_record["stroke_rgb"] = list(render_params["endpoint_label_stroke_rgb"])
    render_params["text_legibility"] = text_legibility_summary_from_records(
        [*previous_records, candidate_label_record, endpoint_label_record]
    )
    return render_params


def named_path_style_trace(render_params: Mapping[str, Any]) -> Dict[str, Any]:
    """Return named-path-specific style metadata."""

    return {
        "path_stroke_width_px": int(render_params["path_stroke_width_px"]),
        "path_color_rgb": [int(value) for value in render_params["path_color_rgb"]],
        "path_line_alpha": int(render_params["path_line_alpha"]),
        "path_stop_radius_px": int(render_params["path_stop_radius_px"]),
        "candidate_label_font_size_px": int(render_params["candidate_label_font_size_px"]),
        "candidate_label_color_rgb": [int(value) for value in render_params["candidate_label_color_rgb"]],
        "candidate_label_stroke_rgb": [int(value) for value in render_params["candidate_label_stroke_rgb"]],
        "candidate_label_background_rgb": [int(value) for value in render_params["candidate_label_background_rgb"]],
        "candidate_label_border_rgb": [int(value) for value in render_params["candidate_label_border_rgb"]],
        "candidate_label_padding_px": int(render_params["candidate_label_padding_px"]),
        "candidate_label_gap_px": int(render_params["candidate_label_gap_px"]),
        "endpoint_label_font_size_px": int(render_params["endpoint_label_font_size_px"]),
        "endpoint_label_color_rgb": [int(value) for value in render_params["endpoint_label_color_rgb"]],
        "endpoint_label_stroke_rgb": [int(value) for value in render_params["endpoint_label_stroke_rgb"]],
        "endpoint_label_background_rgb": [int(value) for value in render_params["endpoint_label_background_rgb"]],
        "stop_fill_rgb": [int(value) for value in render_params["stop_fill_rgb"]],
        "stop_outline_rgb": [int(value) for value in render_params["stop_outline_rgb"]],
    }
