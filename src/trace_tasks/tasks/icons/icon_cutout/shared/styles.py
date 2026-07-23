"""Style and render-parameter helpers for icon-cutout scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ....shared.config_defaults import group_default
from ...shared.icon_style import icon_palette_meets_distance_constraints, sample_icon_palette
from ...shared.icon_task_rendering import (
    icon_render_style_trace,
    resolve_icon_cell_render_params,
    resolve_icon_rgb_param,
)

from .defaults import FRAGMENT_WINDOW_STYLES, IconCutoutDefaults


def _resolve_range(raw: Any, fallback: Sequence[float], *, key: str) -> Tuple[float, float]:
    """Resolve a two-value numeric range."""

    value = raw if raw is not None else fallback
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"{key} must contain two numeric bounds")
    lo = float(value[0])
    hi = float(value[1])
    return min(lo, hi), max(lo, hi)


def resolve_icon_cutout_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    defaults: IconCutoutDefaults,
    instance_seed: int,
) -> Dict[str, Any]:
    """Resolve render params for an icon-cutout scene."""

    render_params = resolve_icon_cell_render_params(
        params=params,
        render_defaults=render_defaults,
        fallback_defaults=defaults,
        instance_seed=int(instance_seed),
    )
    render_params["fragment_frame_rgb"] = resolve_icon_rgb_param(
        params=params,
        render_defaults=render_defaults,
        key="fragment_frame_rgb",
        fallback=defaults.fragment_frame_rgb,
        instance_seed=int(instance_seed),
    )
    render_params["fragment_frame_width_px"] = int(
        params.get(
            "fragment_frame_width_px",
            group_default(render_defaults, "fragment_frame_width_px", defaults.fragment_frame_width_px),
        )
    )
    raw_styles = params.get(
        "fragment_window_styles",
        group_default(render_defaults, "fragment_window_styles", list(defaults.fragment_window_styles)),
    )
    if not isinstance(raw_styles, (list, tuple)):
        raise ValueError("fragment_window_styles must be a sequence")
    styles = tuple(str(value) for value in raw_styles if str(value) in set(FRAGMENT_WINDOW_STYLES))
    if not styles:
        raise ValueError("fragment_window_styles resolved no supported styles")
    render_params["fragment_window_styles"] = styles
    render_params["fragment_window_width_fraction_range"] = _resolve_range(
        params.get(
            "fragment_window_width_fraction_range",
            group_default(
                render_defaults,
                "fragment_window_width_fraction_range",
                list(defaults.fragment_window_width_fraction_range),
            ),
        ),
        defaults.fragment_window_width_fraction_range,
        key="fragment_window_width_fraction_range",
    )
    render_params["fragment_window_height_fraction_range"] = _resolve_range(
        params.get(
            "fragment_window_height_fraction_range",
            group_default(
                render_defaults,
                "fragment_window_height_fraction_range",
                list(defaults.fragment_window_height_fraction_range),
            ),
        ),
        defaults.fragment_window_height_fraction_range,
        key="fragment_window_height_fraction_range",
    )
    render_params["fragment_visible_alpha_ratio_range"] = _resolve_range(
        params.get(
            "fragment_visible_alpha_ratio_range",
            group_default(
                render_defaults,
                "fragment_visible_alpha_ratio_range",
                list(defaults.fragment_visible_alpha_ratio_range),
            ),
        ),
        defaults.fragment_visible_alpha_ratio_range,
        key="fragment_visible_alpha_ratio_range",
    )
    render_params["fragment_alpha_density_min"] = float(
        params.get(
            "fragment_alpha_density_min",
            group_default(render_defaults, "fragment_alpha_density_min", defaults.fragment_alpha_density_min),
        )
    )
    render_params["fragment_sampling_attempts"] = int(
        params.get(
            "fragment_sampling_attempts",
            group_default(render_defaults, "fragment_sampling_attempts", defaults.fragment_sampling_attempts),
        )
    )
    render_params["reference_content_padding_px"] = int(
        params.get(
            "reference_content_padding_px",
            group_default(render_defaults, "reference_content_padding_px", defaults.reference_content_padding_px),
        )
    )
    render_params["rotation_candidates_degrees"] = tuple(
        int(value) % 360
        for value in params.get(
            "rotation_candidates_degrees",
            group_default(render_defaults, "rotation_candidates_degrees", list(defaults.rotation_candidates_degrees)),
        )
    )
    if not render_params["rotation_candidates_degrees"]:
        raise ValueError("rotation_candidates_degrees resolved no values")
    return render_params


def sample_icon_cutout_palette(rng, render_params: Mapping[str, Any]) -> Tuple[Tuple[int, int, int], ...]:
    """Sample a palette separated from icon chrome colors."""

    palette_size = int(rng.randint(int(render_params["palette_size_min"]), int(render_params["palette_size_max"])))
    anchor_colors = (
        tuple(int(v) for v in render_params["background_color_rgb"]),
        tuple(int(v) for v in render_params["panel_fill_rgb"]),
        tuple(int(v) for v in render_params["panel_border_rgb"]),
        tuple(int(v) for v in render_params["header_text_rgb"]),
        tuple(int(v) for v in render_params["fragment_frame_rgb"]),
    )
    palette = tuple(
        tuple(int(channel) for channel in color)
        for color in sample_icon_palette(
            rng,
            palette_size=int(palette_size),
            channel_min=int(render_params["color_channel_min"]),
            channel_max=int(render_params["color_channel_max"]),
            anchor_colors=anchor_colors,
            min_color_distance=float(render_params["min_color_distance"]),
            distance_space=str(render_params["color_distance_space"]),
        )
    )
    if not icon_palette_meets_distance_constraints(
        palette=palette,
        anchor_colors=anchor_colors,
        min_color_distance=float(render_params["min_color_distance"]),
        distance_space=str(render_params["color_distance_space"]),
    ):
        raise ValueError("sampled icon-cutout palette did not satisfy distance constraints")
    return palette


def icon_cutout_style_trace(
    *,
    render_params: Mapping[str, Any],
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...],
) -> Dict[str, Any]:
    """Return trace metadata for rendered icon-cutout style choices."""

    style = icon_render_style_trace(
        render_params=render_params,
        sampled_palette_rgb=sampled_palette_rgb,
    )
    style.update(
        {
            "cell_padding_px": int(render_params["cell_padding_px"]),
            "cell_border_rgb": list(render_params["cell_border_rgb"]),
            "cell_label_color_rgb": list(render_params["cell_label_color_rgb"]),
            "cell_label_stroke_rgb": list(render_params.get("cell_label_stroke_rgb", render_params["panel_fill_rgb"])),
            "cell_label_font_size_px": int(render_params["cell_label_font_size_px"]),
            "fragment_frame_rgb": list(render_params["fragment_frame_rgb"]),
            "fragment_frame_width_px": int(render_params["fragment_frame_width_px"]),
            "fragment_window_styles": [str(value) for value in render_params["fragment_window_styles"]],
            "fragment_window_width_fraction_range": [
                float(render_params["fragment_window_width_fraction_range"][0]),
                float(render_params["fragment_window_width_fraction_range"][1]),
            ],
            "fragment_window_height_fraction_range": [
                float(render_params["fragment_window_height_fraction_range"][0]),
                float(render_params["fragment_window_height_fraction_range"][1]),
            ],
            "fragment_visible_alpha_ratio_range": [
                float(render_params["fragment_visible_alpha_ratio_range"][0]),
                float(render_params["fragment_visible_alpha_ratio_range"][1]),
            ],
            "fragment_alpha_density_min": float(render_params["fragment_alpha_density_min"]),
            "fragment_sampling_attempts": int(render_params["fragment_sampling_attempts"]),
        }
    )
    return style


__all__ = [
    "icon_cutout_style_trace",
    "resolve_icon_cutout_render_params",
    "sample_icon_cutout_palette",
]
