"""Trace output helpers for overlap-grid icon scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple


def overlap_grid_style_trace(
    *,
    render_params: Mapping[str, Any],
    sampled_palette_rgb: Sequence[Tuple[int, int, int]],
) -> dict[str, Any]:
    """Return the canonical render-style trace block for overlap-grid scenes."""

    return {
        "background_color_rgb": list(render_params["background_color_rgb"]),
        "panel_fill_rgb": list(render_params["panel_fill_rgb"]),
        "panel_border_rgb": list(render_params["panel_border_rgb"]),
        "header_text_rgb": list(render_params["header_text_rgb"]),
        "header_text_stroke_rgb": list(render_params.get("header_text_stroke_rgb", render_params["panel_fill_rgb"])),
        "text_color_policy": str(
            render_params.get("text_color_policy", "read_required_text_uses_random_nonsemantic_readable_ink")
        ),
        "text_legibility": dict(render_params.get("text_legibility", {})),
        "icon_canvas_style": dict(render_params.get("icon_canvas_style", {"enabled": False})),
        "sampled_palette_rgb": [list(color) for color in sampled_palette_rgb],
        "color_channel_min": int(render_params["color_channel_min"]),
        "color_channel_max": int(render_params["color_channel_max"]),
        "min_color_distance": float(render_params["min_color_distance"]),
        "pair_min_color_distance": float(render_params["pair_min_color_distance"]),
        "color_distance_space": str(render_params["color_distance_space"]),
        "overlap_ratio_range": [
            float(render_params["overlap_ratio_range"][0]),
            float(render_params["overlap_ratio_range"][1]),
        ],
        "cell_padding_px": int(render_params["cell_padding_px"]),
        "cell_border_rgb": list(render_params["cell_border_rgb"]),
        "cell_label_color_rgb": list(render_params["cell_label_color_rgb"]),
        "cell_label_stroke_rgb": list(render_params.get("cell_label_stroke_rgb", render_params["panel_fill_rgb"])),
        "cell_label_font_size_px": int(render_params["cell_label_font_size_px"]),
        "icon_noise_edit_types": [str(value) for value in render_params["icon_noise_edit_types"]],
        "icon_noise_edit_count_range": [
            int(render_params["icon_noise_edit_count_range"][0]),
            int(render_params["icon_noise_edit_count_range"][1]),
        ],
        "icon_noise_value_ranges": {
            str(edit_type): {
                str(param): [float(bounds[0]), float(bounds[1])]
                for param, bounds in params.items()
            }
            for edit_type, params in render_params["icon_noise_value_ranges"].items()
        },
    }


__all__ = ["overlap_grid_style_trace"]
