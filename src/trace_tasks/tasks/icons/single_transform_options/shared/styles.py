"""Style resolution helpers for single-transform option icon scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ....shared.config_defaults import group_default
from ....shared.text_legibility import resolve_readable_text_style, text_legibility_summary_from_records
from ...shared.icon_task_rendering import (
    icon_render_style_trace,
    resolve_icon_cell_render_params,
    resolve_icon_rgb_param,
)


def resolve_single_transform_render_params(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    fallback_defaults: Any,
    instance_seed: int,
) -> Dict[str, Any]:
    """Resolve neutral rendering knobs and readable cue text style."""

    render_params = resolve_icon_cell_render_params(
        params=params,
        render_defaults=render_defaults,
        fallback_defaults=fallback_defaults,
        instance_seed=int(instance_seed),
    )
    render_params["operation_cue_font_size_px"] = int(
        params.get(
            "operation_cue_font_size_px",
            group_default(render_defaults, "operation_cue_font_size_px", fallback_defaults.operation_cue_font_size_px),
        )
    )
    render_params["operation_cue_color_rgb"] = resolve_icon_rgb_param(
        params=params,
        render_defaults=render_defaults,
        key="operation_cue_color_rgb",
        fallback=fallback_defaults.operation_cue_color_rgb,
        instance_seed=int(instance_seed),
    )
    render_params["reference_content_padding_px"] = int(
        params.get(
            "reference_content_padding_px",
            group_default(render_defaults, "reference_content_padding_px", fallback_defaults.reference_content_padding_px),
        )
    )
    cue_style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace="icons.single_transform_options.operation_cue_text",
        role="icon_operation_cue_text",
        surface_rgbs=(
            tuple(int(value) for value in render_params["panel_fill_rgb"]),
            tuple(int(value) for value in render_params["background_color_rgb"]),
        ),
        preferred_rgbs=(tuple(int(value) for value in render_params["operation_cue_color_rgb"]),),
    )
    render_params["operation_cue_color_rgb"] = tuple(int(value) for value in cue_style.fill_rgb)
    render_params["operation_cue_stroke_rgb"] = tuple(int(value) for value in render_params["panel_fill_rgb"])
    cue_record = cue_style.metadata()
    cue_record["stroke_rgb"] = list(render_params["operation_cue_stroke_rgb"])
    previous_legibility = render_params.get("text_legibility")
    previous_records = []
    if isinstance(previous_legibility, Mapping) and isinstance(previous_legibility.get("records"), list):
        previous_records = [dict(record) for record in previous_legibility["records"] if isinstance(record, Mapping)]
    render_params["text_legibility"] = text_legibility_summary_from_records([*previous_records, cue_record])
    return render_params


def single_transform_style_trace(
    *,
    render_params: Mapping[str, Any],
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...],
) -> Dict[str, Any]:
    """Return the canonical render-style trace block for this scene."""

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
            "operation_cue_font_size_px": int(render_params["operation_cue_font_size_px"]),
            "operation_cue_color_rgb": list(render_params["operation_cue_color_rgb"]),
            "operation_cue_stroke_rgb": list(render_params.get("operation_cue_stroke_rgb", render_params["panel_fill_rgb"])),
        }
    )
    return style


__all__ = ["resolve_single_transform_render_params", "single_transform_style_trace"]
