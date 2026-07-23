"""Style resolution helpers for the pair-grid icons scene."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ....shared.config_defaults import group_default
from ....shared.text_legibility import resolve_readable_text_style, text_legibility_summary_from_records
from ...shared.icon_task_rendering import resolve_icon_render_params, resolve_icon_rgb_param


def resolve_pair_grid_render_params(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    fallback_defaults: Any,
    instance_seed: int,
) -> Dict[str, Any]:
    """Resolve neutral pair-grid rendering knobs and readable cell-label style."""

    render_params = resolve_icon_render_params(
        params=params,
        render_defaults=render_defaults,
        fallback_defaults=fallback_defaults,
        instance_seed=int(instance_seed),
    )
    render_params["cell_padding_px"] = int(
        params.get("cell_padding_px", group_default(render_defaults, "cell_padding_px", fallback_defaults.cell_padding_px))
    )
    render_params["pair_arrow_stroke_px"] = int(
        params.get(
            "pair_arrow_stroke_px",
            group_default(render_defaults, "pair_arrow_stroke_px", fallback_defaults.pair_arrow_stroke_px),
        )
    )
    render_params["cell_label_font_size_px"] = int(
        params.get(
            "cell_label_font_size_px",
            group_default(render_defaults, "cell_label_font_size_px", fallback_defaults.cell_label_font_size_px),
        )
    )
    render_params["cell_border_rgb"] = resolve_icon_rgb_param(
        params=params,
        render_defaults=render_defaults,
        key="cell_border_rgb",
        fallback=fallback_defaults.cell_border_rgb,
        instance_seed=int(instance_seed),
    )
    render_params["cell_label_color_rgb"] = resolve_icon_rgb_param(
        params=params,
        render_defaults=render_defaults,
        key="cell_label_color_rgb",
        fallback=fallback_defaults.cell_label_color_rgb,
        instance_seed=int(instance_seed),
    )
    cell_label_style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace="icons.pair_grid.cell_label_text",
        role="icon_cell_label_text",
        surface_rgbs=(
            tuple(int(value) for value in render_params["panel_fill_rgb"]),
            tuple(int(value) for value in render_params["background_color_rgb"]),
        ),
        preferred_rgbs=(tuple(int(value) for value in render_params["cell_label_color_rgb"]),),
        required=False,
    )
    render_params["cell_label_color_rgb"] = tuple(int(value) for value in cell_label_style.fill_rgb)
    render_params["cell_label_stroke_rgb"] = tuple(int(value) for value in render_params["panel_fill_rgb"])
    cell_label_record = cell_label_style.metadata()
    cell_label_record["stroke_rgb"] = list(render_params["cell_label_stroke_rgb"])
    previous_legibility = render_params.get("text_legibility")
    previous_records = []
    if isinstance(previous_legibility, Mapping) and isinstance(previous_legibility.get("records"), list):
        previous_records = [dict(record) for record in previous_legibility["records"] if isinstance(record, Mapping)]
    render_params["text_legibility"] = text_legibility_summary_from_records([*previous_records, cell_label_record])
    render_params["arrow_color_rgb"] = resolve_icon_rgb_param(
        params=params,
        render_defaults=render_defaults,
        key="arrow_color_rgb",
        fallback=fallback_defaults.arrow_color_rgb,
        instance_seed=int(instance_seed),
    )
    return render_params


def pair_grid_style_trace(
    *,
    render_params: Mapping[str, Any],
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...],
    size_scale_small: float | None = None,
    size_scale_large: float | None = None,
) -> Dict[str, Any]:
    """Return the canonical render-style trace block for pair-grid scenes."""

    style = {
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
        "color_distance_space": str(render_params["color_distance_space"]),
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
        "cell_padding_px": int(render_params["cell_padding_px"]),
        "pair_arrow_stroke_px": int(render_params["pair_arrow_stroke_px"]),
        "cell_label_font_size_px": int(render_params["cell_label_font_size_px"]),
        "cell_border_rgb": list(render_params["cell_border_rgb"]),
        "cell_label_color_rgb": list(render_params["cell_label_color_rgb"]),
        "cell_label_stroke_rgb": list(render_params.get("cell_label_stroke_rgb", render_params["panel_fill_rgb"])),
        "arrow_color_rgb": list(render_params["arrow_color_rgb"]),
    }
    if size_scale_small is not None:
        style["size_scale_small"] = float(size_scale_small)
    if size_scale_large is not None:
        style["size_scale_large"] = float(size_scale_large)
    return style
