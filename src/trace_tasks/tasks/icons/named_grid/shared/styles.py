"""Render-style resolution for the named-grid icons scene."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....shared.config_defaults import group_default
from ....shared.text_legibility import resolve_readable_text_style, text_legibility_summary_from_records
from ...shared.icon_task_rendering import resolve_icon_render_params, resolve_icon_rgb_param

from .defaults import NamedGridDefaults


def render_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    value = params.get(str(key), fallback)
    return int(value)


def render_rgb(params: Mapping[str, Any], key: str, fallback: Sequence[int]) -> Tuple[int, int, int]:
    raw = params.get(str(key), fallback)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 3:
        raw = fallback
    return tuple(int(value) for value in list(raw)[:3])


def _previous_text_legibility_records(render_params: Mapping[str, Any]) -> List[Dict[str, Any]]:
    previous_legibility = render_params.get("text_legibility")
    if isinstance(previous_legibility, Mapping):
        records = previous_legibility.get("records")
        if isinstance(records, list):
            return [dict(record) for record in records if isinstance(record, Mapping)]
    return []


def resolve_named_grid_rgb(
    params: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
    *,
    render_defaults: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[int, int, int]:
    return resolve_icon_rgb_param(
        params=params,
        key=str(key),
        fallback=fallback,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
    )


def resolve_named_grid_render_params(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    fallback_defaults: NamedGridDefaults,
    instance_seed: int,
) -> Dict[str, Any]:
    """Resolve shared named-grid render params and readable axis-label text."""

    render_params = resolve_icon_render_params(
        params=params,
        render_defaults=render_defaults,
        fallback_defaults=fallback_defaults,
        instance_seed=int(instance_seed),
    )
    for key in (
        "row_label_band_width_px",
        "column_label_band_height_px",
        "grid_label_gap_px",
        "grid_cell_max_size_px",
        "grid_cell_padding_px",
        "grid_line_width_px",
        "grid_border_width_px",
        "axis_label_font_size_px",
    ):
        render_params[key] = int(
            params.get(
                key,
                group_default(render_defaults, key, getattr(fallback_defaults, key)),
            )
        )
    for key in ("grid_line_rgb", "cell_fill_rgb", "alternate_cell_fill_rgb", "axis_label_rgb"):
        render_params[key] = resolve_named_grid_rgb(
            params,
            key,
            getattr(fallback_defaults, key),
            render_defaults=render_defaults,
            instance_seed=int(instance_seed),
        )
    axis_label_style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace="icons.named_grid_axis_number_text",
        role="named_grid_axis_number_text",
        surface_rgbs=(
            tuple(int(value) for value in render_params["panel_fill_rgb"]),
            tuple(int(value) for value in render_params["background_color_rgb"]),
        ),
        preferred_rgbs=(tuple(int(value) for value in render_params["axis_label_rgb"]),),
    )
    render_params["axis_label_rgb"] = tuple(int(value) for value in axis_label_style.fill_rgb)
    render_params["axis_label_stroke_rgb"] = tuple(int(value) for value in render_params["panel_fill_rgb"])
    axis_label_record = axis_label_style.metadata()
    axis_label_record["stroke_rgb"] = list(render_params["axis_label_stroke_rgb"])
    render_params["text_legibility"] = text_legibility_summary_from_records(
        [*_previous_text_legibility_records(render_params), axis_label_record]
    )
    return render_params


def named_grid_style_trace(render_params: Mapping[str, Any]) -> Dict[str, Any]:
    """Return render-style metadata shared by named-grid tasks."""

    return {
        "row_label_band_width_px": int(render_params["row_label_band_width_px"]),
        "column_label_band_height_px": int(render_params["column_label_band_height_px"]),
        "grid_label_gap_px": int(render_params["grid_label_gap_px"]),
        "grid_cell_max_size_px": int(render_params["grid_cell_max_size_px"]),
        "grid_cell_padding_px": int(render_params["grid_cell_padding_px"]),
        "grid_line_width_px": int(render_params["grid_line_width_px"]),
        "grid_border_width_px": int(render_params["grid_border_width_px"]),
        "axis_label_font_size_px": int(render_params["axis_label_font_size_px"]),
        "grid_line_rgb": [int(value) for value in render_params["grid_line_rgb"]],
        "cell_fill_rgb": [int(value) for value in render_params["cell_fill_rgb"]],
        "alternate_cell_fill_rgb": [int(value) for value in render_params["alternate_cell_fill_rgb"]],
        "axis_label_rgb": [int(value) for value in render_params["axis_label_rgb"]],
        "axis_label_stroke_rgb": [int(value) for value in render_params["axis_label_stroke_rgb"]],
        "text_legibility": dict(render_params.get("text_legibility", {})),
    }
