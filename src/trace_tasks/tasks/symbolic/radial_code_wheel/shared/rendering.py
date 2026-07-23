"""Renderer for symbolic radial code-wheel scenes."""

from __future__ import annotations

import math
from typing import Sequence

from PIL import Image, ImageDraw

from ....shared.text_legibility import contrast_ratio
from ....shared.text_rendering import load_font
from ...shared.drawing import draw_centered_text, draw_rounded_rect
from ...shared.scene_style import SymbolicSceneStyle

from .state import (
    CODE_SYMBOLS,
    RadialOptionSpec,
    RadialReferenceSpec,
    RadialRenderParams,
    RadialTerminalSpec,
    RenderedRadialScene,
)

_DARK_TEXT_RGB = (10, 14, 22)
_LIGHT_TEXT_RGB = (250, 252, 255)


def _rounded_bbox(values: Sequence[float]) -> list[float]:
    return [round(float(value), 3) for value in values]


def _rounded_point(values: Sequence[float]) -> list[float]:
    return [round(float(values[0]), 3), round(float(values[1]), 3)]


def _blend(left: Sequence[int], right: Sequence[int], alpha: float) -> tuple[int, int, int]:
    """Blend two RGB colors by alpha toward the second color."""

    return tuple(
        max(0, min(255, int(round((float(a) * (1.0 - float(alpha))) + (float(b) * float(alpha))))))
        for a, b in zip(tuple(left)[:3], tuple(right)[:3])
    )


def _readable_text_rgb(surface_rgb: Sequence[int]) -> tuple[int, int, int]:
    """Choose crisp dark/light ink for one local text surface."""

    surface = tuple(int(value) for value in tuple(surface_rgb)[:3])
    dark_contrast = float(contrast_ratio(_DARK_TEXT_RGB, surface))
    light_contrast = float(contrast_ratio(_LIGHT_TEXT_RGB, surface))
    return _DARK_TEXT_RGB if dark_contrast >= light_contrast else _LIGHT_TEXT_RGB


def _point(cx: float, cy: float, radius: float, angle_degrees: float) -> tuple[float, float]:
    """Return a point for a clockwise angle measured from 12 o'clock."""

    radians = math.radians(float(angle_degrees))
    return (
        float(cx) + (float(radius) * math.sin(radians)),
        float(cy) - (float(radius) * math.cos(radians)),
    )


def _code_digit_indices(code: str) -> tuple[int, int, int]:
    normalized = str(code).strip().upper()
    if len(normalized) != 3:
        raise ValueError("radial code path points require a three-symbol code")
    indices: list[int] = []
    for symbol in normalized:
        if symbol not in CODE_SYMBOLS:
            raise ValueError(f"unsupported radial code symbol for path points: {symbol}")
        indices.append(CODE_SYMBOLS.index(symbol))
    return int(indices[0]), int(indices[1]), int(indices[2])


def _code_path_points(*, code: str, params: RadialRenderParams) -> dict[str, list[float]]:
    """Return keyed centers for the inner/middle/outer symbols followed by a code."""

    first, second, third = _code_digit_indices(code)
    sector_indices = (
        int(first),
        int((first * len(CODE_SYMBOLS)) + second),
        int((first * len(CODE_SYMBOLS) * len(CODE_SYMBOLS)) + (second * len(CODE_SYMBOLS)) + third),
    )
    keys = ("inner_ring_symbol", "middle_ring_symbol", "outer_ring_symbol")
    sector_counts = (4, 16, 64)
    cx = float(params.wheel_center_x_px)
    cy = float(params.wheel_center_y_px)
    rotation = -45.0
    out: dict[str, list[float]] = {}
    for ring_index, (key, sector_count, sector_index) in enumerate(zip(keys, sector_counts, sector_indices)):
        radius0 = float(params.wheel_inner_radius_px) + (ring_index * float(params.ring_width_px))
        radius1 = radius0 + float(params.ring_width_px)
        label_radius = 0.5 * (radius0 + radius1)
        angle = rotation + (360.0 * (float(sector_index) + 0.5) / float(sector_count))
        out[str(key)] = _rounded_point(_point(cx, cy, label_radius, angle))
    return out


def _sector_polygon(
    *,
    cx: float,
    cy: float,
    inner_radius: float,
    outer_radius: float,
    start_degrees: float,
    end_degrees: float,
    steps: int,
) -> list[tuple[float, float]]:
    outer = [
        _point(cx, cy, outer_radius, start_degrees + ((end_degrees - start_degrees) * index / max(1, steps)))
        for index in range(max(1, steps) + 1)
    ]
    inner = [
        _point(cx, cy, inner_radius, end_degrees - ((end_degrees - start_degrees) * index / max(1, steps)))
        for index in range(max(1, steps) + 1)
    ]
    return outer + inner


def _draw_text_centered(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: tuple[float, float],
    font_size: int,
    fill: Sequence[int],
    stroke_fill: Sequence[int],
    stroke_width: int = 1,
    bold: bool = False,
) -> list[float]:
    font = load_font(int(font_size), bold=bool(bold))
    bbox = draw.textbbox((float(center[0]), float(center[1])), str(text), font=font, anchor="mm", stroke_width=int(stroke_width))
    draw.text(
        (float(center[0]), float(center[1])),
        str(text),
        font=font,
        anchor="mm",
        fill=tuple(int(value) for value in fill),
        stroke_fill=tuple(int(value) for value in stroke_fill),
        stroke_width=int(stroke_width),
    )
    return _rounded_bbox(bbox)


def _draw_card(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    params: RadialRenderParams,
    style: SymbolicSceneStyle,
) -> list[float]:
    card_bbox = _rounded_bbox(bbox)
    draw_rounded_rect(
        draw,
        tuple(float(value) for value in card_bbox),
        radius=int(params.card_corner_radius_px),
        fill=tuple(int(value) for value in style.panel_fill_rgb),
        outline=tuple(int(value) for value in style.panel_border_rgb),
        width=int(params.card_border_width_px),
    )
    return card_bbox


def _draw_reference_card(
    draw: ImageDraw.ImageDraw,
    *,
    reference: RadialReferenceSpec,
    bbox: Sequence[float],
    params: RadialRenderParams,
    style: SymbolicSceneStyle,
) -> tuple[list[float], dict[str, object]]:
    card_bbox = _draw_card(draw, bbox=bbox, params=params, style=style)
    text_rgb = _readable_text_rgb(style.panel_fill_rgb)
    draw_centered_text(
        draw,
        text=str(reference.title),
        center=(0.5 * (card_bbox[0] + card_bbox[2]), card_bbox[1] + 27),
        font=load_font(int(params.source_title_font_size_px), bold=False),
        fill=text_rgb,
        stroke_fill=text_rgb,
        stroke_width=1,
    )
    draw_centered_text(
        draw,
        text=str(reference.value),
        center=(0.5 * (card_bbox[0] + card_bbox[2]), card_bbox[1] + 72),
        font=load_font(int(params.source_value_font_size_px), bold=False),
        fill=text_rgb,
        stroke_fill=text_rgb,
        stroke_width=1,
    )
    return card_bbox, {
        "item_id": str(reference.item_id),
        "entity_type": "radial_reference_card",
        "role": str(reference.role),
        "title": str(reference.title),
        "value": str(reference.value),
        "bbox_px": list(card_bbox),
    }


def _draw_option_card(
    draw: ImageDraw.ImageDraw,
    *,
    option: RadialOptionSpec,
    bbox: Sequence[float],
    params: RadialRenderParams,
    style: SymbolicSceneStyle,
) -> tuple[list[float], dict[str, object]]:
    card_bbox = _draw_card(draw, bbox=bbox, params=params, style=style)
    text_rgb = _readable_text_rgb(style.panel_fill_rgb)
    draw_centered_text(
        draw,
        text=str(option.label),
        center=(card_bbox[0] + 22, card_bbox[1] + 24),
        font=load_font(int(params.option_label_font_size_px), bold=False),
        fill=text_rgb,
        stroke_fill=text_rgb,
        stroke_width=1,
    )
    draw_centered_text(
        draw,
        text=str(option.value),
        center=(0.5 * (card_bbox[0] + card_bbox[2]) + 10, 0.5 * (card_bbox[1] + card_bbox[3]) + 8),
        font=load_font(int(params.option_value_font_size_px), bold=False),
        fill=text_rgb,
        stroke_fill=text_rgb,
        stroke_width=1,
    )
    return card_bbox, {
        "item_id": str(option.item_id),
        "entity_type": "radial_option_card",
        "role": str(option.role),
        "label": str(option.label),
        "value": str(option.value),
        "bbox_px": list(card_bbox),
    }


def _draw_wheel(
    draw: ImageDraw.ImageDraw,
    *,
    terminals: Sequence[RadialTerminalSpec],
    params: RadialRenderParams,
    style: SymbolicSceneStyle,
) -> tuple[dict[str, list[float]], list[dict[str, object]]]:
    """Draw the 4/16/64 radial wheel and return terminal label bboxes."""

    cx = float(params.wheel_center_x_px)
    cy = float(params.wheel_center_y_px)
    inner_radius = float(params.wheel_inner_radius_px)
    ring_width = float(params.ring_width_px)
    rotation = -45.0
    line_rgb = tuple(int(value) for value in style.grid_rgb)
    item_bboxes: dict[str, list[float]] = {}
    entities: list[dict[str, object]] = []
    fill_sources = tuple(style.state_colors) or (style.option_fill_rgb,)

    for ring_index, sector_count in enumerate((4, 16, 64)):
        radius0 = inner_radius + (ring_index * ring_width)
        radius1 = radius0 + ring_width
        for sector_index in range(sector_count):
            start = rotation + (360.0 * sector_index / sector_count)
            end = rotation + (360.0 * (sector_index + 1) / sector_count)
            base = fill_sources[sector_index % len(fill_sources)]
            fill = _blend(base, style.panel_fill_rgb, 0.68 + (0.06 * ring_index))
            polygon = _sector_polygon(
                cx=cx,
                cy=cy,
                inner_radius=radius0,
                outer_radius=radius1,
                start_degrees=start,
                end_degrees=end,
                steps=5 if sector_count < 64 else 2,
            )
            draw.polygon(polygon, fill=fill, outline=line_rgb)

    outer_radius = inner_radius + (3 * ring_width)
    draw.ellipse(
        (cx - outer_radius, cy - outer_radius, cx + outer_radius, cy + outer_radius),
        outline=tuple(int(value) for value in style.panel_border_rgb),
        width=int(params.wheel_outline_width_px),
    )
    draw.ellipse(
        (cx - inner_radius, cy - inner_radius, cx + inner_radius, cy + inner_radius),
        fill=tuple(int(value) for value in style.panel_fill_rgb),
        outline=tuple(int(value) for value in style.panel_border_rgb),
        width=int(params.ring_line_width_px),
    )
    start_text_rgb = _readable_text_rgb(style.panel_fill_rgb)
    _draw_text_centered(
        draw,
        text="START",
        center=(cx, cy),
        font_size=16,
        fill=start_text_rgb,
        stroke_fill=start_text_rgb,
        stroke_width=1,
    )

    for ring_index, sector_count in enumerate((4, 16, 64)):
        radius0 = inner_radius + (ring_index * ring_width)
        radius1 = radius0 + ring_width
        label_radius = 0.5 * (radius0 + radius1)
        font_size = (
            int(params.inner_ring_symbol_font_size_px)
            if ring_index == 0
            else int(params.ring_symbol_font_size_px if ring_index == 1 else params.outer_ring_symbol_font_size_px)
        )
        for sector_index in range(sector_count):
            symbol = CODE_SYMBOLS[sector_index % len(CODE_SYMBOLS)]
            angle = rotation + (360.0 * (sector_index + 0.5) / sector_count)
            center = _point(cx, cy, label_radius, angle)
            base = fill_sources[sector_index % len(fill_sources)]
            sector_fill = _blend(base, style.panel_fill_rgb, 0.68 + (0.06 * ring_index))
            label_rgb = _readable_text_rgb(sector_fill)
            _draw_text_centered(
                draw,
                text=str(symbol),
                center=center,
                font_size=font_size,
                fill=label_rgb,
                stroke_fill=label_rgb,
                stroke_width=1,
                bold=False,
            )

    for terminal in terminals:
        angle = rotation + (360.0 * (int(terminal.terminal_index) + 0.5) / 64.0)
        center = _point(cx, cy, float(params.terminal_label_radius_px), angle)
        terminal_rgb = _readable_text_rgb(style.background_rgb)
        bbox = _draw_text_centered(
            draw,
            text=str(terminal.output_label),
            center=center,
            font_size=int(params.terminal_label_font_size_px),
            fill=terminal_rgb,
            stroke_fill=terminal_rgb,
            stroke_width=1,
            bold=False,
        )
        item_bboxes[str(terminal.item_id)] = bbox
        entities.append(
            {
                "item_id": str(terminal.item_id),
                "entity_type": "radial_terminal_label",
                "role": "terminal_output_label",
                "code": str(terminal.code),
                "output_label": str(terminal.output_label),
                "terminal_index": int(terminal.terminal_index),
                "bbox_px": list(bbox),
            }
        )
    return item_bboxes, entities


def render_radial_missing_symbol_scene(
    image: Image.Image,
    *,
    params: RadialRenderParams,
    style: SymbolicSceneStyle,
    target_output_label: str,
    partial_code: str,
    terminal_specs: Sequence[RadialTerminalSpec],
    target_code: str,
) -> RenderedRadialScene:
    """Render one radial wheel with a target output and incomplete code."""

    draw = ImageDraw.Draw(image)
    item_bboxes, entities = _draw_wheel(draw, terminals=terminal_specs, params=params, style=style)

    target_card = RadialReferenceSpec(item_id="target_output", title="Output", value=str(target_output_label), role="target_output")
    target_bbox = [
        float(params.source_card_left_px),
        float(params.source_card_top_px),
        float(params.source_card_left_px + params.source_card_width_px),
        float(params.source_card_top_px + params.source_card_height_px),
    ]
    bbox, entity = _draw_reference_card(draw, reference=target_card, bbox=target_bbox, params=params, style=style)
    item_bboxes[str(target_card.item_id)] = bbox
    entities.append(entity)

    partial_card = RadialReferenceSpec(item_id="partial_code", title="Code", value=str(partial_code), role="partial_code")
    partial_bbox = [
        float(params.source_card_left_px),
        float(params.missing_code_card_top_px),
        float(params.source_card_left_px + params.source_card_width_px),
        float(params.missing_code_card_top_px + params.source_card_height_px),
    ]
    bbox, entity = _draw_reference_card(draw, reference=partial_card, bbox=partial_bbox, params=params, style=style)
    item_bboxes[str(partial_card.item_id)] = bbox
    entities.append(entity)

    scene_bbox = [
        28.0,
        28.0,
        float(params.canvas_width - 28),
        float(params.canvas_height - 28),
    ]
    return RenderedRadialScene(
        image=image,
        entities=tuple(entities),
        item_bboxes={str(key): list(value) for key, value in item_bboxes.items()},
        item_points=_code_path_points(code=str(target_code), params=params),
        scene_bbox_px=_rounded_bbox(scene_bbox),
        style_metadata={
            "code_symbols": list(CODE_SYMBOLS),
            "ring_sector_counts": [4, 16, 64],
            "terminal_label_count": int(len(tuple(terminal_specs))),
            "layout": "left_wheel_right_target_partial_code",
        },
    )


def render_radial_choice_scene(
    image: Image.Image,
    *,
    params: RadialRenderParams,
    style: SymbolicSceneStyle,
    reference: RadialReferenceSpec,
    options: Sequence[RadialOptionSpec],
    terminal_specs: Sequence[RadialTerminalSpec],
    target_code: str,
) -> RenderedRadialScene:
    """Render one radial code-wheel choice scene."""

    draw = ImageDraw.Draw(image)
    item_bboxes, entities = _draw_wheel(draw, terminals=terminal_specs, params=params, style=style)

    reference_bbox = [
        float(params.source_card_left_px),
        float(params.source_card_top_px),
        float(params.source_card_left_px + params.source_card_width_px),
        float(params.source_card_top_px + params.source_card_height_px),
    ]
    bbox, entity = _draw_reference_card(draw, reference=reference, bbox=reference_bbox, params=params, style=style)
    item_bboxes[str(reference.item_id)] = bbox
    entities.append(entity)

    for option_index, option in enumerate(options):
        col = int(option_index % 2)
        row = int(option_index // 2)
        left = float(params.option_grid_left_px + col * (params.option_card_width_px + params.option_grid_col_gap_px))
        top = float(params.option_grid_top_px + row * (params.option_card_height_px + params.option_grid_row_gap_px))
        option_bbox = [
            left,
            top,
            left + float(params.option_card_width_px),
            top + float(params.option_card_height_px),
        ]
        bbox, entity = _draw_option_card(draw, option=option, bbox=option_bbox, params=params, style=style)
        item_bboxes[str(option.item_id)] = bbox
        entities.append(entity)

    scene_bbox = [
        28.0,
        28.0,
        float(params.canvas_width - 28),
        float(params.canvas_height - 28),
    ]
    return RenderedRadialScene(
        image=image,
        entities=tuple(entities),
        item_bboxes={str(key): list(value) for key, value in item_bboxes.items()},
        item_points=_code_path_points(code=str(target_code), params=params),
        scene_bbox_px=_rounded_bbox(scene_bbox),
        style_metadata={
            "code_symbols": list(CODE_SYMBOLS),
            "ring_sector_counts": [4, 16, 64],
            "terminal_label_count": int(len(tuple(terminal_specs))),
            "layout": "left_wheel_right_options",
        },
    )


__all__ = ["render_radial_choice_scene", "render_radial_missing_symbol_scene"]
