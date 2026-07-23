"""Rendering primitives for waveform-panel diagrams."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.shared.drawing import draw_centered_text
from trace_tasks.tasks.shared.render_variation import resolve_render_int
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .state import (
    PANEL_LABELS,
    RenderedWaveformPanel,
    RenderedWaveformPanelScene,
    WaveformAxes,
    WaveformPanelDefaults,
    WaveformPanelSemanticSpec,
)


def bbox(values: Sequence[float]) -> List[float]:
    """Round one bbox to final image coordinates."""

    return [round(float(value), 3) for value in values[:4]]


def resolve_waveform_render_defaults(
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    *,
    fallback_defaults: WaveformPanelDefaults,
    instance_seed: int,
    namespace: str,
) -> Dict[str, int]:
    """Resolve integer rendering defaults for a waveform-panel scene."""

    keys = (
        "sheet_left_px",
        "sheet_top_px",
        "sheet_right_margin_px",
        "sheet_bottom_margin_px",
        "stack_left_px",
        "stack_top_px",
        "stack_width_px",
        "stack_height_px",
        "panel_gap_px",
        "label_font_size_px",
        "title_font_size_px",
        "wave_line_width_px",
        "grid_line_width_px",
        "midline_width_px",
    )
    return {
        key: resolve_render_int(
            params,
            rendering_defaults,
            key,
            int(getattr(fallback_defaults, key)),
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
        for key in keys
    }


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    *,
    panel_box: Sequence[float],
    scene_variant: str,
    style: Any,
    grid_width: int,
    midline_width: int,
) -> None:
    """Draw optional guide grid and required waveform midline."""

    left, top, right, bottom = [float(value) for value in panel_box]
    grid_rgb = tuple(int(value) for value in style.grid_minor_rgb)
    guide_rgb = tuple(int(value) for value in style.guide_rgb)
    mid_y = (top + bottom) * 0.5
    if str(scene_variant) in {"grid_stack", "lab_sheet"}:
        for idx in range(1, 10):
            x = left + ((right - left) * (float(idx) / 10.0))
            draw.line((x, top + 4.0, x, bottom - 4.0), fill=grid_rgb, width=max(1, int(grid_width)))
        for frac in (0.25, 0.75):
            y = top + ((bottom - top) * frac)
            draw.line((left + 4.0, y, right - 4.0, y), fill=grid_rgb, width=max(1, int(grid_width)))
    draw.line((left + 6.0, mid_y, right - 6.0, mid_y), fill=guide_rgb, width=max(1, int(midline_width)))
    if str(scene_variant) == "lab_sheet":
        tick_rgb = tuple(int(value) for value in style.axis_rgb)
        for idx in range(0, 11):
            x = left + ((right - left) * (float(idx) / 10.0))
            draw.line((x, bottom - 13.0, x, bottom - 4.0), fill=tick_rgb, width=1)


def _draw_waveform(
    draw: ImageDraw.ImageDraw,
    *,
    panel_box: Sequence[float],
    amplitude_px: float,
    cycle_count: int,
    style: Any,
    line_width: int,
) -> List[float]:
    """Draw one sinusoidal waveform and return its visual bbox."""

    left, top, right, bottom = [float(value) for value in panel_box]
    wave_left = left + 82.0
    wave_right = right - 28.0
    mid_y = (top + bottom) * 0.5
    points: List[Tuple[float, float]] = []
    sample_count = 260
    for idx in range(sample_count + 1):
        t = float(idx) / float(sample_count)
        x = wave_left + (t * (wave_right - wave_left))
        y = mid_y - (float(amplitude_px) * math.sin(2.0 * math.pi * float(cycle_count) * t))
        points.append((float(x), float(y)))
    shadow = tuple(int(value) for value in style.stroke_rgb)
    wave_rgb = tuple(int(value) for value in style.secondary_accent_rgb)
    draw.line(points, fill=shadow, width=max(1, int(line_width) + 2), joint="curve")
    draw.line(points, fill=wave_rgb, width=max(1, int(line_width)), joint="curve")
    return bbox((wave_left, mid_y - float(amplitude_px), wave_right, mid_y + float(amplitude_px)))


def _draw_panel_label(
    draw: ImageDraw.ImageDraw,
    *,
    panel_box: Sequence[float],
    label: str,
    font: Any,
    fill: Tuple[int, int, int],
) -> List[float]:
    """Draw one option letter near the panel's top-left corner."""

    left, top, _right, _bottom = [float(value) for value in panel_box]
    stroke_fill = resolve_text_stroke_fill(fill)
    stroke_width = 1
    x = left + 16.0
    y = top + 10.0
    label_bbox = draw.textbbox(
        (x, y),
        str(label),
        font=font,
        stroke_width=int(stroke_width),
    )
    pad_x = 7.0
    pad_y = 4.0
    draw.rounded_rectangle(
        (
            float(label_bbox[0]) - pad_x,
            float(label_bbox[1]) - pad_y,
            float(label_bbox[2]) + pad_x,
            float(label_bbox[3]) + pad_y,
        ),
        radius=6,
        fill=tuple(int(value) for value in fill),
    )
    draw.text(
        (x, y),
        str(label),
        font=font,
        fill=stroke_fill,
        stroke_width=0,
    )
    return bbox(label_bbox)


def _layout_panel_specs(
    *,
    axes: WaveformAxes,
    semantic_panels: Sequence[WaveformPanelSemanticSpec],
    render_defaults: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Attach pixel layout and amplitude size to semantic panel specs."""

    panel_count = int(axes.panel_count)
    stack_top = float(render_defaults["stack_top_px"])
    stack_left = float(render_defaults["stack_left_px"])
    stack_width = float(render_defaults["stack_width_px"])
    stack_height = float(render_defaults["stack_height_px"])
    panel_gap = float(render_defaults["panel_gap_px"])
    panel_height = (stack_height - (panel_gap * float(panel_count - 1))) / float(panel_count)
    min_amp = max(12.0, panel_height * 0.18)
    max_amp = max(min_amp + 6.0, panel_height * 0.39)
    rank_span = max(1, panel_count - 1)

    panels: List[Dict[str, Any]] = []
    for index, semantic in enumerate(semantic_panels):
        top = stack_top + (float(index) * (panel_height + panel_gap))
        bottom = top + panel_height
        rank = int(semantic.amplitude_rank)
        amplitude_px = min_amp + ((float(rank - 1) / float(rank_span)) * (max_amp - min_amp))
        panels.append(
            {
                "label": str(semantic.label),
                "amplitude_rank": int(rank),
                "cycle_count": int(semantic.cycle_count),
                "amplitude_px": round(float(amplitude_px), 3),
                "bbox_px": bbox((stack_left, top, stack_left + stack_width, bottom)),
                "is_correct": bool(semantic.is_correct),
            }
        )

    summary = {
        "panel_height_px": round(float(panel_height), 3),
        "amplitude_min_px": round(float(min_amp), 3),
        "amplitude_max_px": round(float(max_amp), 3),
    }
    return panels, summary


def render_waveform_panel_scene(
    *,
    image: Image.Image,
    axes: WaveformAxes,
    semantic_panels: Sequence[WaveformPanelSemanticSpec],
    query_property: str,
    query_extremum: str,
    render_defaults: Mapping[str, Any],
    font_family: str,
    style: Any,
) -> RenderedWaveformPanelScene:
    """Render the waveform-panel diagram and selected-panel witness."""

    draw = ImageDraw.Draw(image)
    canvas_width, canvas_height = image.size
    title_font = load_font(int(render_defaults["title_font_size_px"]), bold=True, font_family=font_family)
    label_font = load_font(int(render_defaults["label_font_size_px"]), bold=True, font_family=font_family)
    text_rgb = tuple(int(value) for value in style.label_rgb)
    panel_fill = tuple(int(value) for value in style.panel_fill_rgb)
    panel_alt = tuple(int(value) for value in style.panel_alt_fill_rgb)
    panel_border = tuple(int(value) for value in style.panel_border_rgb)

    sheet_box = (
        float(render_defaults["sheet_left_px"]),
        float(render_defaults["sheet_top_px"]),
        float(canvas_width - int(render_defaults["sheet_right_margin_px"])),
        float(canvas_height - int(render_defaults["sheet_bottom_margin_px"])),
    )
    draw.rounded_rectangle(sheet_box, radius=20, fill=panel_fill, outline=panel_border, width=3)
    draw_centered_text(
        draw,
        text="Waveform comparison",
        center=((sheet_box[0] + sheet_box[2]) * 0.5, sheet_box[1] + 34.0),
        font=title_font,
        fill=text_rgb,
        stroke_fill=resolve_text_stroke_fill(text_rgb),
        stroke_width=1,
    )

    panel_dicts, geometry_summary = _layout_panel_specs(
        axes=axes,
        semantic_panels=semantic_panels,
        render_defaults=render_defaults,
    )
    scene_entities: List[Dict[str, Any]] = []
    rendered_panels: List[RenderedWaveformPanel] = []
    for index, panel in enumerate(panel_dicts):
        box = list(panel["bbox_px"])
        fill = panel_alt if index % 2 else panel_fill
        draw.rounded_rectangle(tuple(box), radius=10, fill=fill, outline=panel_border, width=2)
        _draw_grid(
            draw,
            panel_box=box,
            scene_variant=str(axes.scene_variant),
            style=style,
            grid_width=int(render_defaults["grid_line_width_px"]),
            midline_width=int(render_defaults["midline_width_px"]),
        )
        label_bbox = _draw_panel_label(
            draw,
            panel_box=box,
            label=str(panel["label"]),
            font=label_font,
            fill=text_rgb,
        )
        wave_bbox = _draw_waveform(
            draw,
            panel_box=box,
            amplitude_px=float(panel["amplitude_px"]),
            cycle_count=int(panel["cycle_count"]),
            style=style,
            line_width=int(render_defaults["wave_line_width_px"]),
        )
        spec = RenderedWaveformPanel(
            label=str(panel["label"]),
            amplitude_rank=int(panel["amplitude_rank"]),
            cycle_count=int(panel["cycle_count"]),
            amplitude_px=float(panel["amplitude_px"]),
            bbox_px=list(box),
            wave_bbox_px=list(wave_bbox),
            label_bbox_px=list(label_bbox),
            is_correct=bool(panel["is_correct"]),
        )
        rendered_panels.append(spec)
        scene_entities.append(
            {
                "entity_id": f"panel_{spec.label}",
                "entity_type": "waveform_panel",
                "bbox_px": list(spec.bbox_px),
                "meta": {
                    "label": str(spec.label),
                    "amplitude_rank": int(spec.amplitude_rank),
                    "cycle_count": int(spec.cycle_count),
                    "is_correct": bool(spec.is_correct),
                },
            }
        )

    selected = next(panel for panel in rendered_panels if panel.is_correct)
    render_map = {
        "scene_variant": str(axes.scene_variant),
        "query_property": str(query_property),
        "query_extremum": str(query_extremum),
        "panel_count": int(axes.panel_count),
        "active_option_labels": list(PANEL_LABELS[: int(axes.panel_count)]),
        "correct_option_letter": str(axes.correct_option_letter),
        "selected_panel_bbox_px": list(selected.bbox_px),
        "panel_geometry": dict(geometry_summary),
        "panels": [
            {
                "label": str(panel.label),
                "amplitude_rank": int(panel.amplitude_rank),
                "cycle_count": int(panel.cycle_count),
                "wavelength_relative": round(1.0 / float(panel.cycle_count), 6),
                "amplitude_px": round(float(panel.amplitude_px), 3),
                "bbox_px": list(panel.bbox_px),
                "wave_bbox_px": list(panel.wave_bbox_px),
                "label_bbox_px": list(panel.label_bbox_px),
                "is_correct": bool(panel.is_correct),
            }
            for panel in rendered_panels
        ],
    }
    return RenderedWaveformPanelScene(
        image=image,
        selected_panel_bbox=list(selected.bbox_px),
        annotation_entity_ids=[f"panel_{str(axes.correct_option_letter)}"],
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
    )


__all__ = [
    "bbox",
    "render_waveform_panel_scene",
    "resolve_waveform_render_defaults",
]
