"""Style resolution for named-ring icon scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ....shared.config_defaults import group_default
from ....shared.text_legibility import resolve_readable_text_style, text_legibility_summary_from_records
from ...shared.icon_task_rendering import resolve_icon_render_params, resolve_icon_rgb_param

from .defaults import NamedRingDefaults


def _previous_text_legibility_records(render_params: Mapping[str, Any]) -> list[dict[str, Any]]:
    previous_legibility = render_params.get("text_legibility")
    if not isinstance(previous_legibility, Mapping) or not isinstance(previous_legibility.get("records"), list):
        return []
    return [dict(record) for record in previous_legibility["records"] if isinstance(record, Mapping)]


def _resolve_named_ring_rgb(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
    instance_seed: int,
) -> Tuple[int, int, int]:
    return resolve_icon_rgb_param(
        params=params,
        render_defaults=render_defaults,
        key=str(key),
        fallback=tuple(int(value) for value in fallback),
        instance_seed=int(instance_seed),
    )


def resolve_named_ring_render_params(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    fallback_defaults: NamedRingDefaults,
    instance_seed: int,
) -> Dict[str, Any]:
    """Resolve named-ring render params and readable marker-label text."""

    render_params = resolve_icon_render_params(
        params=params,
        render_defaults=render_defaults,
        fallback_defaults=fallback_defaults,
        instance_seed=int(instance_seed),
    )
    for key in (
        "ring_margin_px",
        "ring_stroke_width_px",
        "ring_stop_radius_px",
        "marker_label_font_size_px",
        "marker_label_radius_px",
        "marker_label_gap_px",
    ):
        render_params[str(key)] = int(params.get(str(key), group_default(render_defaults, str(key), getattr(fallback_defaults, str(key)))))
    for key in (
        "ring_outline_rgb",
        "ring_stop_fill_rgb",
        "ring_stop_outline_rgb",
        "marker_label_background_rgb",
        "marker_label_border_rgb",
        "marker_label_color_rgb",
    ):
        render_params[str(key)] = _resolve_named_ring_rgb(
            params=params,
            render_defaults=render_defaults,
            key=str(key),
            fallback=getattr(fallback_defaults, str(key)),
            instance_seed=int(instance_seed),
        )

    marker_label_style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace="icons.named_ring.marker_label_text",
        role="named_ring_marker_label_text",
        surface_rgbs=(tuple(int(value) for value in render_params["marker_label_background_rgb"]),),
        preferred_rgbs=(tuple(int(value) for value in render_params["marker_label_color_rgb"]),),
    )
    render_params["marker_label_color_rgb"] = tuple(int(value) for value in marker_label_style.fill_rgb)
    render_params["marker_label_stroke_rgb"] = tuple(int(value) for value in marker_label_style.stroke_rgb)
    marker_label_record = marker_label_style.metadata()
    marker_label_record["stroke_rgb"] = list(render_params["marker_label_stroke_rgb"])
    render_params["text_legibility"] = text_legibility_summary_from_records(
        [*_previous_text_legibility_records(render_params), marker_label_record]
    )
    return render_params


def named_ring_style_trace(render_params: Mapping[str, Any]) -> Dict[str, Any]:
    """Return named-ring-specific render style metadata."""

    return {
        "ring_margin_px": int(render_params["ring_margin_px"]),
        "ring_stroke_width_px": int(render_params["ring_stroke_width_px"]),
        "ring_outline_rgb": [int(value) for value in render_params["ring_outline_rgb"]],
        "ring_stop_radius_px": int(render_params["ring_stop_radius_px"]),
        "ring_stop_fill_rgb": [int(value) for value in render_params["ring_stop_fill_rgb"]],
        "ring_stop_outline_rgb": [int(value) for value in render_params["ring_stop_outline_rgb"]],
        "marker_label_font_size_px": int(render_params["marker_label_font_size_px"]),
        "marker_label_radius_px": int(render_params["marker_label_radius_px"]),
        "marker_label_gap_px": int(render_params["marker_label_gap_px"]),
        "marker_label_background_rgb": [int(value) for value in render_params["marker_label_background_rgb"]],
        "marker_label_border_rgb": [int(value) for value in render_params["marker_label_border_rgb"]],
        "marker_label_color_rgb": [int(value) for value in render_params["marker_label_color_rgb"]],
        "marker_label_stroke_rgb": [int(value) for value in render_params["marker_label_stroke_rgb"]],
    }
