"""Rendering and layout helpers for pressure-volume diagrams."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from PIL import Image, ImageDraw

from trace_tasks.tasks.physics.shared.style import build_physics_pv_diagram_theme
from trace_tasks.tasks.shared.bbox_projection import bbox_union_many as _bbox_union
from trace_tasks.tasks.shared.drawing import draw_arrow, draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.shared.render_variation import resolve_layout_jitter
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .sampling import sign_for_work
from .state import (
    PVDiagramSceneSpec,
    PVDiagramTaskDefaults,
    PVWorkScenario,
    RenderedPVDiagramScene,
    SCENE_NAMESPACE,
)


RENDER_DEFAULT_KEYS: tuple[str, ...] = (
    "canvas_width",
    "canvas_height",
    "plot_left_px",
    "plot_top_px",
    "plot_width_px",
    "plot_height_px",
    "mini_plot_left_px",
    "mini_plot_top_px",
    "mini_cell_width_px",
    "mini_cell_height_px",
    "mini_cell_gap_x_px",
    "mini_cell_gap_y_px",
    "axis_width_px",
    "grid_line_width_px",
    "bold_grid_line_width_px",
    "process_line_width_px",
    "cycle_line_width_px",
    "arrow_head_length_px",
    "arrow_head_width_px",
    "label_font_size_px",
    "tick_font_size_px",
    "state_font_size_px",
    "option_font_size_px",
    "note_font_size_px",
    "label_stroke_width_px",
    "pressure_max_kpa",
    "volume_max_l",
)


DEFAULTS = PVDiagramTaskDefaults()

def _arrow_bbox(start: tuple[float, float], end: tuple[float, float], *, padding_px: float) -> list[float]:
    """Return a conservative bbox for one arrow."""

    return [
        round(float(min(float(start[0]), float(end[0])) - float(padding_px)), 3),
        round(float(min(float(start[1]), float(end[1])) - float(padding_px)), 3),
        round(float(max(float(start[0]), float(end[0])) + float(padding_px)), 3),
        round(float(max(float(start[1]), float(end[1])) + float(padding_px)), 3),
    ]


def _plot_bbox(render_defaults: Mapping[str, Any]) -> list[float]:
    """Return the main plot bbox."""

    offset_x = float(render_defaults.get("layout_offset_x_px", 0))
    offset_y = float(render_defaults.get("layout_offset_y_px", 0))
    return [
        float(render_defaults["plot_left_px"]) + float(offset_x),
        float(render_defaults["plot_top_px"]) + float(offset_y),
        float(render_defaults["plot_left_px"]) + float(render_defaults["plot_width_px"]) + float(offset_x),
        float(render_defaults["plot_top_px"]) + float(render_defaults["plot_height_px"]) + float(offset_y),
    ]


def _plot_xy(
    *,
    bbox: Sequence[float],
    volume_l: float,
    pressure_kpa: float,
    volume_max_l: int,
    pressure_max_kpa: int,
) -> tuple[float, float]:
    """Map PV coordinates to screen coordinates."""

    left, top, right, bottom = [float(value) for value in bbox[:4]]
    x = float(left + (float(volume_l) / float(volume_max_l)) * (right - left))
    y = float(bottom - (float(pressure_kpa) / float(pressure_max_kpa)) * (bottom - top))
    return float(x), float(y)


def _draw_text_tag(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: tuple[float, float],
    font,
    fill_rgb: Tuple[int, int, int],
    outline_rgb: Tuple[int, int, int],
    text_rgb: Tuple[int, int, int],
    stroke_width_px: int,
) -> list[float]:
    """Draw one rounded text tag and return its outer bbox."""

    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width_px)))
    text_width = float(text_bbox[2] - text_bbox[0])
    text_height = float(text_bbox[3] - text_bbox[1])
    pad_x = 12.0
    pad_y = 7.0
    center_x, center_y = float(center[0]), float(center[1])
    tag_bbox = [
        round(float(center_x - (0.5 * text_width) - pad_x), 3),
        round(float(center_y - (0.5 * text_height) - pad_y), 3),
        round(float(center_x + (0.5 * text_width) + pad_x), 3),
        round(float(center_y + (0.5 * text_height) + pad_y), 3),
    ]
    draw_rounded_rect(
        draw,
        tuple(float(value) for value in tag_bbox),
        radius=9,
        fill=tuple(int(value) for value in fill_rgb),
        outline=tuple(int(value) for value in outline_rgb),
        width=max(1, int(stroke_width_px)),
    )
    text_draw_bbox = draw_centered_text(
        draw,
        text=str(text),
        center=(float(center_x), float(center_y)),
        font=font,
        fill=tuple(int(value) for value in text_rgb),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(text_rgb)),
        stroke_width=1,
    )
    return _bbox_union(tag_bbox, text_draw_bbox)


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    render_defaults: Mapping[str, Any],
    scene_variant: str,
    theme,
    tick_font,
    label_font,
    include_tick_numbers: bool,
) -> dict[str, Any]:
    """Draw PV axes and return bbox metadata."""

    left, top, right, bottom = [float(value) for value in bbox[:4]]
    volume_max = int(render_defaults["volume_max_l"])
    pressure_max = int(render_defaults["pressure_max_kpa"])
    grid_width = int(render_defaults["bold_grid_line_width_px"]) if str(scene_variant) == "bold_grid" else int(render_defaults["grid_line_width_px"])
    plot_fill = tuple(int(value) for value in theme.paper_plot_fill_rgb) if str(scene_variant) == "paper_grid" else tuple(int(value) for value in theme.plot_fill_rgb)
    draw_rounded_rect(
        draw,
        tuple(float(value) for value in bbox),
        radius=0,
        fill=plot_fill,
        outline=tuple(int(value) for value in theme.plot_outline_rgb),
        width=3,
    )
    tick_bboxes: List[list[float]] = []
    for volume in range(0, int(volume_max) + 1):
        x, _ = _plot_xy(
            bbox=bbox,
            volume_l=float(volume),
            pressure_kpa=0.0,
            volume_max_l=volume_max,
            pressure_max_kpa=pressure_max,
        )
        draw.line([(float(x), float(top)), (float(x), float(bottom))], fill=tuple(int(value) for value in theme.grid_rgb), width=grid_width)
        if include_tick_numbers:
            label_bbox = draw_centered_text(
                draw,
                text=str(volume),
                center=(float(x), float(bottom + 22.0)),
                font=tick_font,
                fill=tuple(int(value) for value in theme.axis_text_rgb),
                stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.axis_text_rgb)),
                stroke_width=1,
            )
            tick_bboxes.append(list(label_bbox))
    for pressure in range(0, int(pressure_max) + 1):
        _, y = _plot_xy(
            bbox=bbox,
            volume_l=0.0,
            pressure_kpa=float(pressure),
            volume_max_l=volume_max,
            pressure_max_kpa=pressure_max,
        )
        draw.line([(float(left), float(y)), (float(right), float(y))], fill=tuple(int(value) for value in theme.grid_rgb), width=grid_width)
        if include_tick_numbers:
            label_bbox = draw_centered_text(
                draw,
                text=str(pressure),
                center=(float(left - 24.0), float(y)),
                font=tick_font,
                fill=tuple(int(value) for value in theme.axis_text_rgb),
                stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.axis_text_rgb)),
                stroke_width=1,
            )
            tick_bboxes.append(list(label_bbox))
    draw_arrow(
        draw,
        start=(float(left), float(bottom)),
        end=(float(right + 34.0), float(bottom)),
        fill=tuple(int(value) for value in theme.axis_rgb),
        width=int(render_defaults["axis_width_px"]),
        head_length_px=24.0,
        head_width_px=20.0,
    )
    draw_arrow(
        draw,
        start=(float(left), float(bottom)),
        end=(float(left), float(top - 34.0)),
        fill=tuple(int(value) for value in theme.axis_rgb),
        width=int(render_defaults["axis_width_px"]),
        head_length_px=24.0,
        head_width_px=20.0,
    )
    x_label_bbox = draw_centered_text(
        draw,
        text="V (L)",
        center=(float((left + right) / 2.0), float(bottom + 58.0)),
        font=label_font,
        fill=tuple(int(value) for value in theme.axis_text_rgb),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.axis_text_rgb)),
        stroke_width=1,
    )
    y_label_bbox = draw_centered_text(
        draw,
        text="P (kPa)",
        center=(float(left - 62.0), float(top - 30.0)),
        font=label_font,
        fill=tuple(int(value) for value in theme.axis_text_rgb),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.axis_text_rgb)),
        stroke_width=1,
    )
    return {
        "plot_bbox_px": [round(float(value), 3) for value in bbox],
        "tick_label_bboxes_px": [list(bbox_value) for bbox_value in tick_bboxes],
        "axis_label_bboxes_px": [list(x_label_bbox), list(y_label_bbox)],
        "axis_bbox_px": _bbox_union(bbox, x_label_bbox, y_label_bbox, *tick_bboxes),
    }


def _draw_state_label(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: tuple[float, float],
    font,
    theme,
) -> list[float]:
    """Draw a compact state/option label."""

    return _draw_text_tag(
        draw,
        text=str(text),
        center=(float(center[0]), float(center[1])),
        font=font,
        fill_rgb=tuple(int(value) for value in theme.label_fill_rgb),
        outline_rgb=tuple(int(value) for value in theme.label_outline_rgb),
        text_rgb=tuple(int(value) for value in theme.label_text_rgb),
        stroke_width_px=2,
    )


def _draw_single_process(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    render_defaults: Mapping[str, Any],
    scenario: PVWorkScenario,
    theme,
    state_font,
    note_font,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draw one horizontal isobaric process and return entity/render metadata."""

    if scenario.pressure is None or scenario.volume_start is None or scenario.volume_end is None:
        raise ValueError("single-process scenario is missing pressure or volume")
    volume_max = int(render_defaults["volume_max_l"])
    pressure_max = int(render_defaults["pressure_max_kpa"])
    start = _plot_xy(
        bbox=bbox,
        volume_l=float(scenario.volume_start),
        pressure_kpa=float(scenario.pressure),
        volume_max_l=volume_max,
        pressure_max_kpa=pressure_max,
    )
    end = _plot_xy(
        bbox=bbox,
        volume_l=float(scenario.volume_end),
        pressure_kpa=float(scenario.pressure),
        volume_max_l=volume_max,
        pressure_max_kpa=pressure_max,
    )
    baseline_start = _plot_xy(
        bbox=bbox,
        volume_l=float(scenario.volume_start),
        pressure_kpa=0.0,
        volume_max_l=volume_max,
        pressure_max_kpa=pressure_max,
    )
    baseline_end = _plot_xy(
        bbox=bbox,
        volume_l=float(scenario.volume_end),
        pressure_kpa=0.0,
        volume_max_l=volume_max,
        pressure_max_kpa=pressure_max,
    )
    shade_poly = [baseline_start, start, end, baseline_end]
    draw.polygon(
        [(float(x), float(y)) for x, y in shade_poly],
        fill=tuple(int(value) for value in theme.work_fill_rgb),
    )
    draw.line([baseline_start, start], fill=tuple(int(value) for value in theme.guide_rgb), width=3)
    draw.line([baseline_end, end], fill=tuple(int(value) for value in theme.guide_rgb), width=3)
    draw_arrow(
        draw,
        start=start,
        end=end,
        fill=tuple(int(value) for value in theme.process_rgb),
        width=int(render_defaults["process_line_width_px"]),
        head_length_px=float(render_defaults["arrow_head_length_px"]),
        head_width_px=float(render_defaults["arrow_head_width_px"]),
    )
    process_bbox = _arrow_bbox(start, end, padding_px=float(render_defaults["arrow_head_width_px"]) + 8.0)
    shade_bbox = _bbox_union(
        [baseline_start[0], baseline_start[1], baseline_end[0], baseline_end[1]],
        [start[0], start[1], end[0], end[1]],
    )
    a_label = _draw_state_label(
        draw,
        text="A",
        center=(float(start[0]), float(start[1] - 34.0)),
        font=state_font,
        theme=theme,
    )
    b_label = _draw_state_label(
        draw,
        text="B",
        center=(float(end[0]), float(end[1] - 34.0)),
        font=state_font,
        theme=theme,
    )
    sign_label = "expansion" if int(scenario.work_value) > 0 else "compression"
    mid_x = float((start[0] + end[0]) / 2.0)
    mid_y = float(min(start[1], end[1]) - 74.0)
    if float(mid_y) < float(bbox[1] + 28.0):
        mid_y = float(max(start[1], end[1]) + 54.0)
    note_bbox = _draw_text_tag(
        draw,
        text=str(sign_label),
        center=(mid_x, mid_y),
        font=note_font,
        fill_rgb=tuple(int(value) for value in theme.label_fill_rgb),
        outline_rgb=tuple(int(value) for value in theme.label_outline_rgb),
        text_rgb=tuple(int(value) for value in theme.label_text_rgb),
        stroke_width_px=2,
    )
    witness_bbox = _bbox_union(process_bbox, shade_bbox, a_label, b_label)
    entities = [
        {
            "entity_id": "work_process_arrow",
            "entity_type": "pv_process_arrow",
            "bbox_px": list(process_bbox),
            "meta": {
                "pressure_kpa": int(scenario.pressure),
                "volume_start_l": int(scenario.volume_start),
                "volume_end_l": int(scenario.volume_end),
                "delta_volume_l": int(scenario.volume_end - scenario.volume_start),
                "work_j": int(scenario.work_value),
            },
        },
        {
            "entity_id": "work_area",
            "entity_type": "pv_work_area",
            "bbox_px": list(shade_bbox),
            "meta": {"work_j": int(scenario.work_value), "sign": sign_for_work(int(scenario.work_value))},
        },
        {
            "entity_id": "work_witness_region",
            "entity_type": "pv_work_witness",
            "bbox_px": list(witness_bbox),
            "meta": {"members": ["work_process_arrow", "work_area"], "work_j": int(scenario.work_value)},
        },
    ]
    render_map = {
        "process_start_px": [round(float(start[0]), 3), round(float(start[1]), 3)],
        "process_end_px": [round(float(end[0]), 3), round(float(end[1]), 3)],
        "process_bbox_px": list(process_bbox),
        "work_area_bbox_px": list(shade_bbox),
        "work_witness_region_bbox_px": list(witness_bbox),
        "state_label_bboxes_px": {"A": list(a_label), "B": list(b_label)},
        "process_note_bbox_px": list(note_bbox),
    }
    return entities, render_map


def _cycle_points(
    *,
    bbox: Sequence[float],
    render_defaults: Mapping[str, Any],
    scenario: PVWorkScenario,
) -> dict[str, tuple[float, float]]:
    """Return screen points for the rectangular cycle states."""

    if (
        scenario.pressure_low is None
        or scenario.pressure_high is None
        or scenario.volume_left is None
        or scenario.volume_right is None
    ):
        raise ValueError("cycle scenario is missing rectangle bounds")
    volume_max = int(render_defaults["volume_max_l"])
    pressure_max = int(render_defaults["pressure_max_kpa"])
    return {
        "A": _plot_xy(
            bbox=bbox,
            volume_l=float(scenario.volume_left),
            pressure_kpa=float(scenario.pressure_low),
            volume_max_l=volume_max,
            pressure_max_kpa=pressure_max,
        ),
        "B": _plot_xy(
            bbox=bbox,
            volume_l=float(scenario.volume_left),
            pressure_kpa=float(scenario.pressure_high),
            volume_max_l=volume_max,
            pressure_max_kpa=pressure_max,
        ),
        "C": _plot_xy(
            bbox=bbox,
            volume_l=float(scenario.volume_right),
            pressure_kpa=float(scenario.pressure_high),
            volume_max_l=volume_max,
            pressure_max_kpa=pressure_max,
        ),
        "D": _plot_xy(
            bbox=bbox,
            volume_l=float(scenario.volume_right),
            pressure_kpa=float(scenario.pressure_low),
            volume_max_l=volume_max,
            pressure_max_kpa=pressure_max,
        ),
    }


def _draw_rectangular_cycle(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    render_defaults: Mapping[str, Any],
    scenario: PVWorkScenario,
    theme,
    state_font,
    note_font,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draw one rectangular PV cycle and return entity/render metadata."""

    points = _cycle_points(bbox=bbox, render_defaults=render_defaults, scenario=scenario)
    draw.polygon(
        [points["A"], points["B"], points["C"], points["D"]],
        fill=tuple(int(value) for value in theme.work_fill_rgb),
    )
    sequence = ("A", "B", "C", "D", "A") if str(scenario.cycle_direction) == "clockwise" else ("A", "D", "C", "B", "A")
    segment_bboxes: List[list[float]] = []
    for start_label, end_label in zip(sequence[:-1], sequence[1:]):
        start = points[str(start_label)]
        end = points[str(end_label)]
        draw_arrow(
            draw,
            start=start,
            end=end,
            fill=tuple(int(value) for value in theme.process_rgb),
            width=int(render_defaults["cycle_line_width_px"]),
            head_length_px=float(render_defaults["arrow_head_length_px"]),
            head_width_px=float(render_defaults["arrow_head_width_px"]),
        )
        segment_bboxes.append(_arrow_bbox(start, end, padding_px=float(render_defaults["arrow_head_width_px"]) + 8.0))
    label_offsets = {
        "A": (-28.0, 28.0),
        "B": (-28.0, -28.0),
        "C": (28.0, -28.0),
        "D": (28.0, 28.0),
    }
    state_label_bboxes: Dict[str, list[float]] = {}
    for label, point in points.items():
        dx, dy = label_offsets[str(label)]
        state_label_bboxes[str(label)] = _draw_state_label(
            draw,
            text=str(label),
            center=(float(point[0] + dx), float(point[1] + dy)),
            font=state_font,
            theme=theme,
        )
    direction_text = "clockwise" if int(scenario.work_value) > 0 else "counterclockwise"
    center_x = float((points["A"][0] + points["C"][0]) / 2.0)
    center_y = float((points["A"][1] + points["C"][1]) / 2.0)
    note_bbox = _draw_text_tag(
        draw,
        text=str(direction_text),
        center=(center_x, center_y),
        font=note_font,
        fill_rgb=tuple(int(value) for value in theme.label_fill_rgb),
        outline_rgb=tuple(int(value) for value in theme.label_outline_rgb),
        text_rgb=tuple(int(value) for value in theme.label_text_rgb),
        stroke_width_px=2,
    )
    area_bbox = _bbox_union(points["A"] + points["C"])
    witness_bbox = _bbox_union(area_bbox, *segment_bboxes, *state_label_bboxes.values())
    entities = [
        {
            "entity_id": "work_cycle_path",
            "entity_type": "pv_cycle_path",
            "bbox_px": _bbox_union(*segment_bboxes),
            "meta": {
                "cycle_direction": str(scenario.cycle_direction),
                "work_j": int(scenario.work_value),
                "pressure_low_kpa": int(scenario.pressure_low or 0),
                "pressure_high_kpa": int(scenario.pressure_high or 0),
                "volume_left_l": int(scenario.volume_left or 0),
                "volume_right_l": int(scenario.volume_right or 0),
            },
        },
        {
            "entity_id": "work_area",
            "entity_type": "pv_net_work_area",
            "bbox_px": list(area_bbox),
            "meta": {"work_j": int(scenario.work_value), "sign": sign_for_work(int(scenario.work_value))},
        },
        {
            "entity_id": "work_witness_region",
            "entity_type": "pv_work_witness",
            "bbox_px": list(witness_bbox),
            "meta": {"members": ["work_cycle_path", "work_area"], "work_j": int(scenario.work_value)},
        },
    ]
    render_map = {
        "cycle_state_points_px": {key: [round(float(point[0]), 3), round(float(point[1]), 3)] for key, point in points.items()},
        "cycle_segment_bboxes_px": [list(bbox_value) for bbox_value in segment_bboxes],
        "work_area_bbox_px": list(area_bbox),
        "work_witness_region_bbox_px": list(witness_bbox),
        "state_label_bboxes_px": {key: list(value) for key, value in state_label_bboxes.items()},
        "cycle_note_bbox_px": list(note_bbox),
    }
    return entities, render_map


def _draw_mini_axes(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    render_defaults: Mapping[str, Any],
    scene_variant: str,
    theme,
    tick_font,
) -> dict[str, Any]:
    """Draw one compact PV-axis frame for a process option."""

    left, top, right, bottom = [float(value) for value in bbox[:4]]
    draw_rounded_rect(
        draw,
        tuple(float(value) for value in bbox),
        radius=8,
        fill=tuple(int(value) for value in theme.plot_fill_rgb),
        outline=tuple(int(value) for value in theme.plot_outline_rgb),
        width=2,
    )
    for volume in (0, 4, 8, 12):
        x, _ = _plot_xy(
            bbox=bbox,
            volume_l=float(volume),
            pressure_kpa=0.0,
            volume_max_l=int(render_defaults["volume_max_l"]),
            pressure_max_kpa=int(render_defaults["pressure_max_kpa"]),
        )
        draw.line([(float(x), float(top)), (float(x), float(bottom))], fill=tuple(int(value) for value in theme.grid_rgb), width=1)
    for pressure in (0, 5, 10):
        _, y = _plot_xy(
            bbox=bbox,
            volume_l=0.0,
            pressure_kpa=float(pressure),
            volume_max_l=int(render_defaults["volume_max_l"]),
            pressure_max_kpa=int(render_defaults["pressure_max_kpa"]),
        )
        draw.line([(float(left), float(y)), (float(right), float(y))], fill=tuple(int(value) for value in theme.grid_rgb), width=1)
    draw_arrow(
        draw,
        start=(float(left + 14.0), float(bottom - 12.0)),
        end=(float(right - 12.0), float(bottom - 12.0)),
        fill=tuple(int(value) for value in theme.axis_rgb),
        width=3,
        head_length_px=13.0,
        head_width_px=11.0,
    )
    draw_arrow(
        draw,
        start=(float(left + 14.0), float(bottom - 12.0)),
        end=(float(left + 14.0), float(top + 12.0)),
        fill=tuple(int(value) for value in theme.axis_rgb),
        width=3,
        head_length_px=13.0,
        head_width_px=11.0,
    )
    v_label = draw_centered_text(
        draw,
        text="V",
        center=(float(right - 14.0), float(bottom + 12.0)),
        font=tick_font,
        fill=tuple(int(value) for value in theme.axis_text_rgb),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.axis_text_rgb)),
        stroke_width=1,
    )
    p_label = draw_centered_text(
        draw,
        text="P",
        center=(float(left - 10.0), float(top + 12.0)),
        font=tick_font,
        fill=tuple(int(value) for value in theme.axis_text_rgb),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.axis_text_rgb)),
        stroke_width=1,
    )
    return {
        "mini_plot_bbox_px": [round(float(value), 3) for value in bbox],
        "mini_axis_bbox_px": _bbox_union(bbox, v_label, p_label),
        "scene_variant": str(scene_variant),
    }


def _draw_sign_choice_scene(
    draw: ImageDraw.ImageDraw,
    *,
    render_defaults: Mapping[str, Any],
    scene_spec: PVDiagramSceneSpec,
    theme,
    tick_font,
    label_font,
    option_font,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draw eight labeled mini-process PV diagrams."""

    entities: list[dict[str, Any]] = []
    option_bboxes: Dict[str, list[float]] = {}
    option_process_bboxes: Dict[str, list[float]] = {}
    option_signs: Dict[str, str] = {}
    offset_x = float(render_defaults.get("layout_offset_x_px", 0))
    offset_y = float(render_defaults.get("layout_offset_y_px", 0))
    for index, candidate in enumerate(scene_spec.process_candidates):
        col = int(index % 4)
        row = int(index // 4)
        cell_left = float(render_defaults["mini_plot_left_px"]) + float(offset_x) + (float(render_defaults["mini_cell_width_px"]) + float(render_defaults["mini_cell_gap_x_px"])) * float(col)
        cell_top = float(render_defaults["mini_plot_top_px"]) + float(offset_y) + (float(render_defaults["mini_cell_height_px"]) + float(render_defaults["mini_cell_gap_y_px"])) * float(row)
        cell_bbox = [
            float(cell_left),
            float(cell_top),
            float(cell_left + float(render_defaults["mini_cell_width_px"])),
            float(cell_top + float(render_defaults["mini_cell_height_px"])),
        ]
        mini_bbox = [
            float(cell_bbox[0] + 58.0),
            float(cell_bbox[1] + 34.0),
            float(cell_bbox[2] - 18.0),
            float(cell_bbox[3] - 24.0),
        ]
        label_bbox = _draw_state_label(
            draw,
            text=str(candidate.letter),
            center=(float(cell_bbox[0] + 24.0), float(cell_bbox[1] + 16.0)),
            font=option_font,
            theme=theme,
        )
        axes_meta = _draw_mini_axes(
            draw,
            bbox=mini_bbox,
            render_defaults=render_defaults,
            scene_variant=str(scene_spec.scene_variant),
            theme=theme,
            tick_font=tick_font,
        )
        start = _plot_xy(
            bbox=mini_bbox,
            volume_l=float(candidate.volume_start),
            pressure_kpa=float(candidate.pressure_start),
            volume_max_l=int(render_defaults["volume_max_l"]),
            pressure_max_kpa=int(render_defaults["pressure_max_kpa"]),
        )
        end = _plot_xy(
            bbox=mini_bbox,
            volume_l=float(candidate.volume_end),
            pressure_kpa=float(candidate.pressure_end),
            volume_max_l=int(render_defaults["volume_max_l"]),
            pressure_max_kpa=int(render_defaults["pressure_max_kpa"]),
        )
        draw_arrow(
            draw,
            start=start,
            end=end,
            fill=tuple(int(value) for value in theme.process_rgb),
            width=max(5, int(render_defaults["process_line_width_px"]) - 2),
            head_length_px=20.0,
            head_width_px=18.0,
        )
        process_bbox = _arrow_bbox(start, end, padding_px=24.0)
        option_bbox = _bbox_union(cell_bbox, label_bbox, axes_meta["mini_axis_bbox_px"], process_bbox)
        option_bboxes[str(candidate.letter)] = list(option_bbox)
        option_process_bboxes[str(candidate.letter)] = list(process_bbox)
        option_signs[str(candidate.letter)] = str(candidate.sign)
        entities.append(
            {
                "entity_id": f"option_{str(candidate.letter)}",
                "entity_type": "candidate_pv_process",
                "bbox_px": list(option_bbox),
                "meta": {
                    "option_letter": str(candidate.letter),
                    "sign": str(candidate.sign),
                    "is_correct": str(candidate.letter) == str(scene_spec.correct_option_letter),
                    "pressure_start_kpa": int(candidate.pressure_start),
                    "pressure_end_kpa": int(candidate.pressure_end),
                    "volume_start_l": int(candidate.volume_start),
                    "volume_end_l": int(candidate.volume_end),
                    "delta_volume_l": int(candidate.volume_end - candidate.volume_start),
                },
            }
        )
        entities.append(
            {
                "entity_id": f"option_{str(candidate.letter)}_process",
                "entity_type": "candidate_pv_process_arrow",
                "bbox_px": list(process_bbox),
                "meta": {
                    "option_letter": str(candidate.letter),
                    "sign": str(candidate.sign),
                    "is_correct": str(candidate.letter) == str(scene_spec.correct_option_letter),
                    "pressure_start_kpa": int(candidate.pressure_start),
                    "pressure_end_kpa": int(candidate.pressure_end),
                    "volume_start_l": int(candidate.volume_start),
                    "volume_end_l": int(candidate.volume_end),
                    "delta_volume_l": int(candidate.volume_end - candidate.volume_start),
                },
            }
        )

    target_text = f"target: {str(scene_spec.target_sign)} work"
    target_bbox = _draw_text_tag(
        draw,
        text=target_text,
        center=(float(render_defaults["mini_plot_left_px"] + 540.0 + offset_x), float(render_defaults["canvas_height"] - 42.0 + offset_y)),
        font=label_font,
        fill_rgb=tuple(int(value) for value in theme.label_fill_rgb),
        outline_rgb=tuple(int(value) for value in theme.label_outline_rgb),
        text_rgb=tuple(int(value) for value in theme.label_text_rgb),
        stroke_width_px=2,
    )
    entities.append(
        {
            "entity_id": "target_sign_label",
            "entity_type": "target_sign_label",
            "bbox_px": list(target_bbox),
            "meta": {"target_sign": str(scene_spec.target_sign)},
        }
    )
    return entities, {
        "option_bboxes_px": {key: list(value) for key, value in option_bboxes.items()},
        "option_process_bboxes_px": {key: list(value) for key, value in option_process_bboxes.items()},
        "option_signs": dict(option_signs),
        "correct_option_letter": str(scene_spec.correct_option_letter),
        "target_sign": str(scene_spec.target_sign),
        "target_sign_label_bbox_px": list(target_bbox),
    }


def render_pv_diagram_scene(
    *,
    background: Image.Image,
    render_defaults: Mapping[str, Any],
    accent_color_name: str,
    scene_spec: PVDiagramSceneSpec,
    diagram_style: Any | None = None,
    font_family: str | None = None,
) -> RenderedPVDiagramScene:
    """Render one PV diagram and return trace metadata."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    theme = build_physics_pv_diagram_theme(str(accent_color_name), diagram_style=diagram_style)
    label_font = load_font(int(render_defaults["label_font_size_px"]), bold=True, font_family=font_family)
    tick_font = load_font(int(render_defaults["tick_font_size_px"]), bold=False, font_family=font_family)
    state_font = load_font(int(render_defaults["state_font_size_px"]), bold=True, font_family=font_family)
    option_font = load_font(int(render_defaults["option_font_size_px"]), bold=True, font_family=font_family)
    note_font = load_font(int(render_defaults["note_font_size_px"]), bold=True, font_family=font_family)

    scene_entities: list[dict[str, Any]] = []
    render_map: dict[str, Any] = {
        "accent_color_name": str(accent_color_name),
        "technical_diagram_frame_mode": str(getattr(diagram_style, "frame_mode", "none")),
        "scene_variant": str(scene_spec.scene_variant),
            }

    if scene_spec.work_scenario is not None:
        plot = _plot_bbox(render_defaults)
        axes_meta = _draw_axes(
            draw,
            bbox=plot,
            render_defaults=render_defaults,
            scene_variant=str(scene_spec.scene_variant),
            theme=theme,
            tick_font=tick_font,
            label_font=label_font,
            include_tick_numbers=True,
        )
        scene_entities.append(
            {
                "entity_id": "pv_axes",
                "entity_type": "pv_axes",
                "bbox_px": list(axes_meta["axis_bbox_px"]),
                "meta": {
                    "pressure_units": "kPa",
                    "volume_units": "L",
                    "work_unit_equivalence": "1 kPa*L = 1 J",
                },
            }
        )
        note_bbox = _draw_text_tag(
            draw,
            text="1 kPa*L = 1 J",
            center=(float(plot[2] - 120.0), float(plot[1] - 36.0)),
            font=note_font,
            fill_rgb=tuple(int(value) for value in theme.label_fill_rgb),
            outline_rgb=tuple(int(value) for value in theme.label_outline_rgb),
            text_rgb=tuple(int(value) for value in theme.label_text_rgb),
            stroke_width_px=2,
        )
        scene_entities.append(
            {
                "entity_id": "unit_equivalence_label",
                "entity_type": "unit_equivalence_label",
                "bbox_px": list(note_bbox),
                "meta": {"text": "1 kPa*L = 1 J"},
            }
        )
        if scene_spec.work_scenario is None:
            raise ValueError("work_value render requires a work scenario")
        if str(scene_spec.work_scenario.work_mode) == "single_process":
            entities, scenario_render_map = _draw_single_process(
                draw,
                bbox=plot,
                render_defaults=render_defaults,
                scenario=scene_spec.work_scenario,
                theme=theme,
                state_font=state_font,
                note_font=note_font,
            )
        else:
            entities, scenario_render_map = _draw_rectangular_cycle(
                draw,
                bbox=plot,
                render_defaults=render_defaults,
                scenario=scene_spec.work_scenario,
                theme=theme,
                state_font=state_font,
                note_font=note_font,
            )
        scene_entities.extend(entities)
        render_map.update(axes_meta)
        render_map.update(scenario_render_map)
    else:
        entities, sign_render_map = _draw_sign_choice_scene(
            draw,
            render_defaults=render_defaults,
            scene_spec=scene_spec,
            theme=theme,
            tick_font=tick_font,
            label_font=label_font,
            option_font=option_font,
        )
        scene_entities.extend(entities)
        render_map.update(sign_render_map)

    entity_bbox_map = {
        str(entity["entity_id"]): list(entity["bbox_px"])
        for entity in scene_entities
        if entity.get("bbox_px") is not None
    }
    annotation_bboxes = [
        list(entity_bbox_map[entity_id])
        for entity_id in scene_spec.annotation_entity_ids
        if str(entity_id) in entity_bbox_map
    ]
    render_map["annotation_entity_ids"] = list(scene_spec.annotation_entity_ids)
    render_map["annotation_bboxes_px"] = [list(bbox) for bbox in annotation_bboxes]
    return RenderedPVDiagramScene(
        image=image,
        annotation_bboxes=[list(bbox) for bbox in annotation_bboxes],
        annotation_entity_ids=list(scene_spec.annotation_entity_ids),
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
    )


def _pv_content_bbox(
    *,
    render_defaults: Mapping[str, Any],
    scene_spec: PVDiagramSceneSpec,
) -> list[float]:
    """Return a conservative bbox for the whole PV content before layout offset."""

    if scene_spec.work_scenario is not None:
        left = float(render_defaults["plot_left_px"]) - 92.0
        top = float(render_defaults["plot_top_px"]) - 72.0
        right = float(render_defaults["plot_left_px"]) + float(render_defaults["plot_width_px"]) + 82.0
        bottom = float(render_defaults["plot_top_px"]) + float(render_defaults["plot_height_px"]) + 88.0
    else:
        column_count = 4
        row_count = 2
        left = float(render_defaults["mini_plot_left_px"]) - 14.0
        top = float(render_defaults["mini_plot_top_px"]) - 26.0
        right = (
            float(render_defaults["mini_plot_left_px"])
            + (float(render_defaults["mini_cell_width_px"]) * float(column_count))
            + (float(render_defaults["mini_cell_gap_x_px"]) * float(column_count - 1))
        )
        bottom = max(
            float(render_defaults["mini_plot_top_px"])
            + (float(render_defaults["mini_cell_height_px"]) * float(row_count))
            + (float(render_defaults["mini_cell_gap_y_px"]) * float(row_count - 1))
            + 12.0,
            float(render_defaults["canvas_height"]) - 16.0,
        )
    return [
        round(float(left), 3),
        round(float(top), 3),
        round(float(right), 3),
        round(float(bottom), 3),
    ]


def resolve_pv_layout_placement(
    *,
    render_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    scene_spec: PVDiagramSceneSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve whole-PV-diagram placement before rendering and annotation projection."""

    canvas_width = int(render_defaults["canvas_width"])
    canvas_height = int(render_defaults["canvas_height"])
    content_bbox = _pv_content_bbox(render_defaults=render_defaults, scene_spec=scene_spec)
    content_left, content_top, content_right, content_bottom = [float(value) for value in content_bbox]
    jitter = resolve_layout_jitter(
        params,
        rendering_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.pv_layout",
    )
    min_margin = int(jitter.get("min_margin_px", 8))
    requested_dx = int(jitter.get("requested_dx_px", 0))
    requested_dy = int(jitter.get("requested_dy_px", 0))
    min_dx = int(math.ceil(float(min_margin) - float(content_left)))
    max_dx = int(math.floor(float(canvas_width) - float(min_margin) - float(content_right)))
    min_dy = int(math.ceil(float(min_margin) - float(content_top)))
    max_dy = int(math.floor(float(canvas_height) - float(min_margin) - float(content_bottom)))
    if int(min_dx) > int(max_dx):
        min_dx = 0
        max_dx = 0
    if int(min_dy) > int(max_dy):
        min_dy = 0
        max_dy = 0
    if not bool(jitter.get("enabled", False)):
        requested_dx = 0
        requested_dy = 0
    dx = max(int(min_dx), min(int(max_dx), int(requested_dx)))
    dy = max(int(min_dy), min(int(max_dy), int(requested_dy)))

    adjusted = dict(render_defaults)
    adjusted["layout_offset_x_px"] = int(dx)
    adjusted["layout_offset_y_px"] = int(dy)

    content_width = round(float(content_right) - float(content_left), 3)
    content_height = round(float(content_bottom) - float(content_top), 3)
    final_bbox = [
        round(float(content_left) + float(dx), 3),
        round(float(content_top) + float(dy), 3),
        round(float(content_right) + float(dx), 3),
        round(float(content_bottom) + float(dy), 3),
    ]
    placement = dict(jitter)
    placement.update(
        {
            "mode": "whole_pv_diagram_offset",
            "content_bbox_px": list(content_bbox),
            "content_size_px": [float(content_width), float(content_height)],
            "final_content_bbox_px": list(final_bbox),
            "canvas_size_px": [int(canvas_width), int(canvas_height)],
            "free_space_px": [
                round(float(canvas_width) - float(content_width), 3),
                round(float(canvas_height) - float(content_height), 3),
            ],
            "available_offset_x_px": [int(min_dx), int(max_dx)],
            "available_offset_y_px": [int(min_dy), int(max_dy)],
            "sampled_offset_px": [int(requested_dx), int(requested_dy)],
            "final_offset_px": [int(dx), int(dy)],
            "default_origin_px": [round(float(content_left), 3), round(float(content_top), 3)],
            "final_origin_px": [round(float(content_left) + float(dx), 3), round(float(content_top) + float(dy), 3)],
            "dx_px": int(dx),
            "dy_px": int(dy),
        }
    )
    return adjusted, placement




__all__ = ["RENDER_DEFAULT_KEYS", "render_pv_diagram_scene", "resolve_pv_layout_placement"]
