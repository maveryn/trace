"""Rendering helpers for pipe-flow repair puzzles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.puzzles.shared.scene_style import (
    make_puzzle_scene_background,
    resolve_puzzle_scene_style,
)
from trace_tasks.tasks.shared.drawing import draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.shared.text_rendering import load_font

from .rules import local_cells, normalize_openings
from .state import (
    BBox,
    Cell,
    Color,
    Openings,
    OptionSpec,
    PipeFlowDataset,
    PipeFlowMisrotatedDataset,
    RenderParams,
    RenderedPipeFlowScene,
)


@dataclass(frozen=True)
class PipeFlowVisualContext:
    """Resolved render params, background image, and style metadata."""

    render_params: RenderParams
    background: Image.Image
    background_meta: Mapping[str, Any]
    scene_style_meta: Mapping[str, Any]


def resolve_pipe_flow_visual_context(
    *,
    render_params: RenderParams,
    instance_seed: int,
    namespace: str,
) -> PipeFlowVisualContext:
    """Apply puzzle styling and create a coordinate-preserving background."""

    scene_style, scene_style_meta = resolve_puzzle_scene_style(
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    styled_params = replace(
        render_params,
        panel_fill_rgb=tuple(int(value) for value in scene_style.panel_fill_rgb),
        cell_fill_rgb=tuple(int(value) for value in scene_style.option_fill_rgb),
        grid_line_rgb=tuple(int(value) for value in scene_style.grid_rgb),
        pipe_rgb=tuple(int(value) for value in scene_style.mark_rgb),
        pipe_shadow_rgb=tuple(int(value) for value in scene_style.notebook_line_rgb),
        label_fill_rgb=tuple(int(value) for value in scene_style.panel_border_rgb),
        label_text_rgb=tuple(int(value) for value in scene_style.text_stroke_rgb),
        text_stroke_rgb=tuple(int(value) for value in scene_style.text_stroke_rgb),
    )
    background, background_meta = make_puzzle_scene_background(
        canvas_width=int(styled_params.canvas_width),
        canvas_height=int(styled_params.canvas_height),
        style=scene_style,
    )
    return PipeFlowVisualContext(
        render_params=styled_params,
        background=background,
        background_meta=background_meta,
        scene_style_meta=scene_style_meta,
    )


def tile_bbox(
    *,
    row: int,
    col: int,
    grid_left: int,
    grid_top: int,
    cell_size: int,
    gap: int,
) -> tuple[int, int, int, int]:
    """Project one grid coordinate to an image-pixel tile box."""

    x0 = int(grid_left + (int(col) * (int(cell_size) + int(gap))))
    y0 = int(grid_top + (int(row) * (int(cell_size) + int(gap))))
    return (x0, y0, int(x0 + cell_size), int(y0 + cell_size))


def draw_pipe(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: tuple[int, int, int, int],
    openings: Openings,
    pipe_rgb: Color,
    shadow_rgb: Color,
    pipe_width: int,
    scene_variant: str,
) -> None:
    """Draw a pipe/conduit segment inside one tile."""

    if not openings:
        return
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    edge_points = {
        "N": (cx, bbox[1]),
        "E": (bbox[2], cy),
        "S": (cx, bbox[3]),
        "W": (bbox[0], cy),
    }
    shadow_width = max(1, int(pipe_width) + 6)
    for direction in openings:
        draw.line(
            [(cx, cy), edge_points[str(direction)]],
            fill=tuple(shadow_rgb),
            width=shadow_width,
        )
    for direction in openings:
        draw.line(
            [(cx, cy), edge_points[str(direction)]],
            fill=tuple(pipe_rgb),
            width=int(pipe_width),
        )
    radius = max(5, int(pipe_width // 2))
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=tuple(pipe_rgb),
        outline=tuple(shadow_rgb),
        width=2,
    )
    if str(scene_variant) == "circuit_trace":
        dot_radius = max(3, int(pipe_width // 4))
        draw.ellipse(
            (cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius),
            fill=(255, 255, 255),
        )


def draw_label_chip(
    draw: ImageDraw.ImageDraw,
    *,
    label: str,
    bbox: tuple[int, int, int, int],
    render_params: RenderParams,
) -> None:
    """Draw the option label chip in the upper-left corner of a panel."""

    if not label:
        return
    chip = max(26, int(render_params.tile_label_font_size_px) + 12)
    chip_bbox = (
        int(bbox[0] + 7),
        int(bbox[1] + 7),
        int(bbox[0] + 7 + chip),
        int(bbox[1] + 7 + chip),
    )
    draw.rounded_rectangle(
        chip_bbox,
        radius=8,
        fill=tuple(render_params.label_fill_rgb),
        outline=tuple(render_params.text_stroke_rgb),
        width=1,
    )
    font = load_font(int(render_params.tile_label_font_size_px), bold=True)
    draw_centered_text(
        draw,
        text=str(label),
        center=(
            (chip_bbox[0] + chip_bbox[2]) / 2.0,
            (chip_bbox[1] + chip_bbox[3]) / 2.0,
        ),
        font=font,
        fill=tuple(render_params.label_text_rgb),
        stroke_fill=tuple(render_params.label_fill_rgb),
        stroke_width=0,
    )


def draw_start_marker(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: tuple[int, int, int, int],
    direction: str,
    render_params: RenderParams,
) -> tuple[int, int, int, int]:
    """Draw a compact visual start marker without text."""

    cx = int((bbox[0] + bbox[2]) / 2)
    cy = int((bbox[1] + bbox[3]) / 2)
    cell_size = int(min(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    radius = max(11, int(cell_size * 0.32))
    marker_bbox = (
        int(cx - radius),
        int(cy - radius),
        int(cx + radius),
        int(cy + radius),
    )
    draw.ellipse(
        marker_bbox,
        fill=tuple(render_params.source_fill_rgb),
        outline=tuple(render_params.source_outline_rgb),
        width=max(2, int(render_params.cell_border_width_px + 1)),
    )
    resolved_direction = str(direction) if str(direction) in {"N", "E", "S", "W"} else "E"
    if resolved_direction == "N":
        triangle = [
            (int(cx), int(cy - radius * 0.58)),
            (int(cx - radius * 0.54), int(cy + radius * 0.30)),
            (int(cx + radius * 0.54), int(cy + radius * 0.30)),
        ]
    elif resolved_direction == "S":
        triangle = [
            (int(cx), int(cy + radius * 0.58)),
            (int(cx - radius * 0.54), int(cy - radius * 0.30)),
            (int(cx + radius * 0.54), int(cy - radius * 0.30)),
        ]
    elif resolved_direction == "W":
        triangle = [
            (int(cx - radius * 0.58), int(cy)),
            (int(cx + radius * 0.30), int(cy - radius * 0.54)),
            (int(cx + radius * 0.30), int(cy + radius * 0.54)),
        ]
    else:
        triangle = [
            (int(cx + radius * 0.58), int(cy)),
            (int(cx - radius * 0.30), int(cy - radius * 0.54)),
            (int(cx - radius * 0.30), int(cy + radius * 0.54)),
        ]
    draw.polygon(triangle, fill=tuple(render_params.source_outline_rgb))
    return marker_bbox


def draw_finish_flag(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: tuple[int, int, int, int],
    render_params: RenderParams,
) -> tuple[int, int, int, int]:
    """Draw a compact red triangular destination flag without text."""

    cell_size = int(min(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    cx = int((bbox[0] + bbox[2]) / 2)
    cy = int((bbox[1] + bbox[3]) / 2)
    pole_h = max(26, int(cell_size * 0.74))
    pole_x = int(cx - cell_size * 0.22)
    pole_top = int(cy - pole_h * 0.50)
    pole_bottom = int(cy + pole_h * 0.50)
    flag_w = max(20, int(cell_size * 0.58))
    flag_h = max(16, int(cell_size * 0.42))
    marker_bbox = (
        int(pole_x - 4),
        int(pole_top - 4),
        int(pole_x + flag_w + 5),
        int(pole_bottom + 4),
    )
    flag_fill = (220, 57, 57)
    flag_outline = (132, 32, 32)
    draw.line(
        [(pole_x, pole_top), (pole_x, pole_bottom)],
        fill=flag_outline,
        width=max(3, int(cell_size * 0.08)),
    )
    triangle = [
        (int(pole_x), int(pole_top + 2)),
        (int(pole_x + flag_w), int(pole_top + flag_h * 0.48)),
        (int(pole_x), int(pole_top + flag_h)),
    ]
    draw.polygon(triangle, fill=flag_fill)
    draw.line([triangle[0], triangle[1], triangle[2], triangle[0]], fill=flag_outline, width=2)
    draw.ellipse((pole_x - 4, pole_bottom - 4, pole_x + 4, pole_bottom + 4), fill=flag_outline)
    return marker_bbox


def draw_tile_label_badge(
    draw: ImageDraw.ImageDraw,
    *,
    label: str,
    tile_box: tuple[int, int, int, int],
    render_params: RenderParams,
) -> tuple[int, int, int, int]:
    """Draw a compact candidate label inside one tile."""

    if not label:
        return tuple(int(value) for value in tile_box)
    cell_size = int(min(tile_box[2] - tile_box[0], tile_box[3] - tile_box[1]))
    font_size = max(
        12,
        min(
            int(render_params.tile_label_font_size_px),
            max(12, int(cell_size * 0.36)),
        ),
    )
    label_size = max(20, min(int(cell_size * 0.50), int(font_size + 16)))
    label_bbox = (
        int(tile_box[0] + 4),
        int(tile_box[1] + 4),
        int(tile_box[0] + 4 + label_size),
        int(tile_box[1] + 4 + label_size),
    )
    font = load_font(int(font_size), bold=True)
    draw.rounded_rectangle(
        label_bbox,
        radius=7,
        fill=tuple(render_params.label_fill_rgb),
        outline=tuple(render_params.text_stroke_rgb),
        width=1,
    )
    draw_centered_text(
        draw,
        text=str(label),
        center=(
            (label_bbox[0] + label_bbox[2]) / 2.0,
            (label_bbox[1] + label_bbox[3]) / 2.0,
        ),
        font=font,
        fill=tuple(render_params.label_text_rgb),
        stroke_fill=tuple(render_params.label_fill_rgb),
        stroke_width=0,
    )
    return label_bbox


def option_opening_map(option: OptionSpec) -> dict[Cell, Openings]:
    """Return local option openings keyed by local cell."""

    return {
        (int(row), int(col)): normalize_openings(openings)
        for row, col, openings in option.local_openings
    }


def draw_option_panel(
    draw: ImageDraw.ImageDraw,
    *,
    option: OptionSpec,
    panel_bbox: tuple[int, int, int, int],
    render_params: RenderParams,
    scene_variant: str,
    gap_size: int,
) -> tuple[dict[str, Any], dict[str, BBox]]:
    """Draw one labeled replacement-piece option panel."""

    draw_rounded_rect(
        draw,
        panel_bbox,
        radius=max(10, int(render_params.panel_corner_radius_px * 0.55)),
        fill=tuple(render_params.panel_fill_rgb),
        outline=tuple(render_params.grid_line_rgb),
        width=max(1, int(render_params.panel_border_width_px)),
    )
    draw_label_chip(
        draw,
        label=str(option.label),
        bbox=panel_bbox,
        render_params=render_params,
    )
    inner_left = int(panel_bbox[0] + 24)
    inner_top = int(panel_bbox[1] + 34)
    inner_right = int(panel_bbox[2] - 18)
    inner_bottom = int(panel_bbox[3] - 14)
    cell_gap = max(2, int(render_params.cell_gap_px))
    grid_size = max(1, int(gap_size))
    cell_size = int(
        min(
            (inner_right - inner_left - ((grid_size - 1) * cell_gap)) / grid_size,
            (inner_bottom - inner_top - ((grid_size - 1) * cell_gap)) / grid_size,
        )
    )
    grid_extent = int(grid_size * cell_size + (grid_size - 1) * cell_gap)
    grid_left = int((panel_bbox[0] + panel_bbox[2] - grid_extent) / 2)
    grid_top = int(inner_top + max(0, (inner_bottom - inner_top - grid_extent) / 2))
    option_map = option_opening_map(option)
    cell_bboxes: dict[str, BBox] = {}
    for row, col in local_cells(gap_size=int(gap_size)):
        bbox = tile_bbox(
            row=row,
            col=col,
            grid_left=grid_left,
            grid_top=grid_top,
            cell_size=cell_size,
            gap=cell_gap,
        )
        draw.rounded_rectangle(
            bbox,
            radius=max(5, int(cell_size * 0.16)),
            fill=tuple(render_params.cell_fill_rgb),
            outline=tuple(render_params.grid_line_rgb),
            width=max(1, int(render_params.cell_border_width_px)),
        )
        draw_pipe(
            draw,
            bbox=bbox,
            openings=option_map.get((row, col), tuple()),
            pipe_rgb=tuple(render_params.pipe_rgb),
            shadow_rgb=tuple(render_params.pipe_shadow_rgb),
            pipe_width=max(5, min(int(render_params.pipe_width_px), int(cell_size * 0.36))),
            scene_variant=str(scene_variant),
        )
        cell_bboxes[f"{option.option_id}_cell_{row}_{col}"] = tuple(float(value) for value in bbox)
    return (
        {
            "id": str(option.option_id),
            "type": "pipe_flow_option_panel",
            "label": str(option.label),
            "bbox_px": [int(value) for value in panel_bbox],
            "is_correct": bool(option.is_correct),
            "connects_in_place": bool(option.connects_in_place),
            "gap_size": int(gap_size),
            "local_openings": [
                {"row": int(row), "col": int(col), "openings": list(openings)}
                for row, col, openings in option.local_openings
            ],
        },
        {str(option.option_id): tuple(float(value) for value in panel_bbox), **cell_bboxes},
    )


def render_pipe_flow_scene(
    *,
    background: Image.Image,
    dataset: PipeFlowDataset,
    render_params: RenderParams,
) -> RenderedPipeFlowScene:
    """Render the board, missing gap, and option panels for review."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    option_panel_width = 112
    option_panel_height = 112
    option_gap = 14
    option_cols = 2
    option_rows = int((len(dataset.options) + option_cols - 1) // option_cols)
    option_width = int(option_cols * option_panel_width + (option_cols - 1) * option_gap)
    option_height = int(option_rows * option_panel_height + (option_rows - 1) * option_gap)
    board_option_gap = 48
    usable_width = int(render_params.canvas_width - (2 * render_params.scene_margin_px))
    usable_height = int(
        render_params.canvas_height
        - (2 * render_params.scene_margin_px)
        - option_height
        - board_option_gap
    )
    cell_size = int(
        min(
            (usable_width - ((dataset.cols - 1) * render_params.cell_gap_px)) / dataset.cols,
            (usable_height - ((dataset.rows - 1) * render_params.cell_gap_px)) / dataset.rows,
        )
    )
    cell_size = max(
        int(render_params.cell_size_min_px),
        min(int(render_params.cell_size_max_px), cell_size),
    )
    grid_width = int(dataset.cols * cell_size + (dataset.cols - 1) * render_params.cell_gap_px)
    grid_height = int(dataset.rows * cell_size + (dataset.rows - 1) * render_params.cell_gap_px)
    content_width = int(max(grid_width, option_width))
    content_height = int(grid_height + board_option_gap + option_height)
    content_left = int((render_params.canvas_width - content_width) / 2)
    grid_left = int(content_left + max(0, (content_width - grid_width) / 2))
    options_left = int(content_left + max(0, (content_width - option_width) / 2))
    content_top = int((render_params.canvas_height - content_height) / 2)
    grid_top = int(content_top)
    options_top = int(grid_top + grid_height + board_option_gap)
    panel_bbox = (
        int(grid_left - render_params.panel_padding_px),
        int(grid_top - render_params.panel_padding_px),
        int(grid_left + grid_width + render_params.panel_padding_px),
        int(grid_top + grid_height + render_params.panel_padding_px),
    )
    options_panel_bbox = (
        int(options_left - render_params.panel_padding_px),
        int(options_top - render_params.panel_padding_px),
        int(options_left + option_width + render_params.panel_padding_px),
        int(options_top + option_height + render_params.panel_padding_px),
    )
    draw_rounded_rect(
        draw,
        panel_bbox,
        radius=int(render_params.panel_corner_radius_px),
        fill=tuple(render_params.panel_fill_rgb),
        outline=tuple(render_params.grid_line_rgb),
        width=int(render_params.panel_border_width_px),
    )
    draw_rounded_rect(
        draw,
        options_panel_bbox,
        radius=int(render_params.panel_corner_radius_px),
        fill=tuple(render_params.panel_fill_rgb),
        outline=tuple(render_params.grid_line_rgb),
        width=int(render_params.panel_border_width_px),
    )

    tile_bbox_map: Dict[str, BBox] = {}
    item_bbox_map: Dict[str, BBox] = {}
    entities: list[dict[str, Any]] = [
        {
            "id": "pipe_flow_panel",
            "type": "pipe_flow_panel",
            "bbox_px": [int(value) for value in panel_bbox],
            "rows": int(dataset.rows),
            "cols": int(dataset.cols),
            "gap_size_variant": str(dataset.gap_size_variant),
            "gap_size": int(dataset.gap_size),
            "scene_variant": str(dataset.scene_variant),
            "missing_region_id": str(dataset.missing_region_id),
        }
    ]
    by_cell = {(tile.row, tile.col): tile for tile in dataset.tiles}
    missing_cells = set(dataset.missing_cells)
    missing_bboxes: list[tuple[int, int, int, int]] = []

    for row in range(dataset.rows):
        for col in range(dataset.cols):
            bbox = tile_bbox(
                row=row,
                col=col,
                grid_left=grid_left,
                grid_top=grid_top,
                cell_size=cell_size,
                gap=int(render_params.cell_gap_px),
            )
            if (row, col) in missing_cells:
                missing_bboxes.append(tuple(bbox))
                draw.rounded_rectangle(
                    bbox,
                    radius=max(4, int(cell_size * 0.14)),
                    fill=(18, 20, 24),
                    outline=(18, 20, 24),
                    width=1,
                )
                continue
            draw.rounded_rectangle(
                bbox,
                radius=max(4, int(cell_size * 0.14)),
                fill=tuple(render_params.cell_fill_rgb),
                outline=tuple(render_params.grid_line_rgb),
                width=int(render_params.cell_border_width_px),
            )
            tile = by_cell.get((row, col))
            if tile is None:
                continue
            pipe_width = int(render_params.pipe_width_px)
            if dataset.scene_variant == "circuit_trace":
                pipe_width = max(7, pipe_width - 4)
            draw_pipe(
                draw,
                bbox=bbox,
                openings=tuple(tile.current_openings),
                pipe_rgb=tuple(render_params.pipe_rgb),
                shadow_rgb=tuple(render_params.pipe_shadow_rgb),
                pipe_width=max(5, min(pipe_width, int(cell_size * 0.38))),
                scene_variant=str(dataset.scene_variant),
            )
            tile_bbox_map[str(tile.tile_id)] = tuple(float(value) for value in bbox)
            item_bbox_map[str(tile.tile_id)] = tuple(float(value) for value in bbox)
            entities.append(
                {
                    "id": str(tile.tile_id),
                    "type": "pipe_flow_tile",
                    "label": str(tile.label),
                    "row_index": int(tile.row),
                    "col_index": int(tile.col),
                    "bbox_px": [int(value) for value in bbox],
                    "current_openings": list(tile.current_openings),
                    "required_openings": list(tile.required_openings),
                    "is_path": bool(tile.is_path),
                    "is_branch": bool(tile.is_branch),
                }
            )

    start_bbox = tile_bbox(
        row=dataset.start_cell[0],
        col=dataset.start_cell[1],
        grid_left=grid_left,
        grid_top=grid_top,
        cell_size=cell_size,
        gap=int(render_params.cell_gap_px),
    )
    dest_bbox = tile_bbox(
        row=dataset.destination_cell[0],
        col=dataset.destination_cell[1],
        grid_left=grid_left,
        grid_top=grid_top,
        cell_size=cell_size,
        gap=int(render_params.cell_gap_px),
    )
    start_direction = _start_flow_direction(dataset)
    start_marker_bbox = draw_start_marker(
        draw,
        bbox=start_bbox,
        direction=start_direction,
        render_params=render_params,
    )
    finish_marker_bbox = draw_finish_flag(draw, bbox=dest_bbox, render_params=render_params)
    item_bbox_map["start_marker"] = tuple(float(value) for value in start_marker_bbox)
    item_bbox_map["finish_flag"] = tuple(float(value) for value in finish_marker_bbox)
    entities.extend(
        [
            {
                "id": "start_marker",
                "type": "pipe_flow_start_marker",
                "bbox_px": [int(value) for value in start_marker_bbox],
                "cell": [int(dataset.start_cell[0]), int(dataset.start_cell[1])],
                "direction": str(start_direction),
            },
            {
                "id": "finish_flag",
                "type": "pipe_flow_finish_flag",
                "bbox_px": [int(value) for value in finish_marker_bbox],
                "cell": [int(dataset.destination_cell[0]), int(dataset.destination_cell[1])],
            },
        ]
    )

    if missing_bboxes:
        missing_region_bbox = (
            min(box[0] for box in missing_bboxes),
            min(box[1] for box in missing_bboxes),
            max(box[2] for box in missing_bboxes),
            max(box[3] for box in missing_bboxes),
        )
        draw.rounded_rectangle(
            missing_region_bbox,
            radius=max(7, int(cell_size * 0.20)),
            fill=(18, 20, 24),
            outline=(245, 247, 250),
            width=max(2, int(render_params.cell_border_width_px + 1)),
        )
        item_bbox_map[str(dataset.missing_region_id)] = tuple(
            float(value) for value in missing_region_bbox
        )
        entities.append(
            {
                "id": str(dataset.missing_region_id),
                "type": "pipe_flow_missing_region",
                "bbox_px": [int(value) for value in missing_region_bbox],
                "gap_size_variant": str(dataset.gap_size_variant),
                "gap_size": int(dataset.gap_size),
                "origin_row": int(dataset.missing_origin[0]),
                "origin_col": int(dataset.missing_origin[1]),
                "cells": [[int(row), int(col)] for row, col in dataset.missing_cells],
            }
        )

    for option_index, option in enumerate(dataset.options):
        row = int(option_index // option_cols)
        col = int(option_index % option_cols)
        panel = (
            int(options_left + col * (option_panel_width + option_gap)),
            int(options_top + row * (option_panel_height + option_gap)),
            int(options_left + col * (option_panel_width + option_gap) + option_panel_width),
            int(options_top + row * (option_panel_height + option_gap) + option_panel_height),
        )
        entity, option_bboxes = draw_option_panel(
            draw,
            option=option,
            panel_bbox=panel,
            render_params=render_params,
            scene_variant=str(dataset.scene_variant),
            gap_size=int(dataset.gap_size),
        )
        entities.append(dict(entity))
        for key, value in option_bboxes.items():
            if key == str(option.option_id):
                item_bbox_map[str(key)] = tuple(float(v) for v in value)
            else:
                tile_bbox_map[str(key)] = tuple(float(v) for v in value)

    scene_bbox = (
        float(min(panel_bbox[0], options_panel_bbox[0])),
        float(min(panel_bbox[1], options_panel_bbox[1])),
        float(max(panel_bbox[2], options_panel_bbox[2])),
        float(max(panel_bbox[3], options_panel_bbox[3])),
    )
    return RenderedPipeFlowScene(
        image=image,
        scene_bbox_px=tuple(float(value) for value in scene_bbox),
        tile_bbox_map=dict(tile_bbox_map),
        item_bbox_map=dict(item_bbox_map),
        entities=tuple(entities),
    )


def render_pipe_flow_misrotated_scene(
    *,
    background: Image.Image,
    dataset: PipeFlowMisrotatedDataset,
    render_params: RenderParams,
) -> RenderedPipeFlowScene:
    """Render a compact pipe-flow board with four labeled candidate tiles."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    usable_width = int(render_params.canvas_width - (2 * render_params.scene_margin_px))
    usable_height = int(render_params.canvas_height - (2 * render_params.scene_margin_px))
    cell_size = int(
        min(
            (usable_width - ((dataset.cols - 1) * render_params.cell_gap_px)) / dataset.cols,
            (usable_height - ((dataset.rows - 1) * render_params.cell_gap_px)) / dataset.rows,
        )
    )
    cell_size = max(
        int(render_params.cell_size_min_px),
        min(int(render_params.cell_size_max_px), cell_size),
    )
    grid_width = int(dataset.cols * cell_size + (dataset.cols - 1) * render_params.cell_gap_px)
    grid_height = int(dataset.rows * cell_size + (dataset.rows - 1) * render_params.cell_gap_px)
    grid_left = int((render_params.canvas_width - grid_width) / 2)
    grid_top = int((render_params.canvas_height - grid_height) / 2)
    panel_bbox = (
        int(grid_left - render_params.panel_padding_px),
        int(grid_top - render_params.panel_padding_px),
        int(grid_left + grid_width + render_params.panel_padding_px),
        int(grid_top + grid_height + render_params.panel_padding_px),
    )
    draw_rounded_rect(
        draw,
        panel_bbox,
        radius=int(render_params.panel_corner_radius_px),
        fill=tuple(render_params.panel_fill_rgb),
        outline=tuple(render_params.grid_line_rgb),
        width=int(render_params.panel_border_width_px),
    )

    tile_bbox_map: Dict[str, BBox] = {}
    item_bbox_map: Dict[str, BBox] = {}
    entities: list[dict[str, Any]] = [
        {
            "id": "pipe_flow_panel",
            "type": "pipe_flow_panel",
            "bbox_px": [int(value) for value in panel_bbox],
            "rows": int(dataset.rows),
            "cols": int(dataset.cols),
            "scene_variant": str(dataset.scene_variant),
            "candidate_count": int(len(dataset.candidates)),
            "misrotated_tile_id": str(dataset.misrotated_tile_id),
        }
    ]
    by_cell = {(tile.row, tile.col): tile for tile in dataset.tiles}

    for row in range(dataset.rows):
        for col in range(dataset.cols):
            bbox = tile_bbox(
                row=row,
                col=col,
                grid_left=grid_left,
                grid_top=grid_top,
                cell_size=cell_size,
                gap=int(render_params.cell_gap_px),
            )
            draw.rounded_rectangle(
                bbox,
                radius=max(4, int(cell_size * 0.14)),
                fill=tuple(render_params.cell_fill_rgb),
                outline=tuple(render_params.grid_line_rgb),
                width=int(render_params.cell_border_width_px),
            )
            tile = by_cell.get((row, col))
            if tile is None:
                continue
            pipe_width = int(render_params.pipe_width_px)
            if dataset.scene_variant == "circuit_trace":
                pipe_width = max(7, pipe_width - 4)
            draw_pipe(
                draw,
                bbox=bbox,
                openings=tuple(tile.current_openings),
                pipe_rgb=tuple(render_params.pipe_rgb),
                shadow_rgb=tuple(render_params.pipe_shadow_rgb),
                pipe_width=max(5, min(pipe_width, int(cell_size * 0.38))),
                scene_variant=str(dataset.scene_variant),
            )
            label_bbox = draw_tile_label_badge(
                draw,
                label=str(tile.label),
                tile_box=bbox,
                render_params=render_params,
            )
            tile_bbox_map[str(tile.tile_id)] = tuple(float(value) for value in bbox)
            item_bbox_map[str(tile.tile_id)] = tuple(float(value) for value in bbox)
            if tile.label:
                candidate_id = f"candidate_tile_{str(tile.label).lower()}"
                item_bbox_map[str(candidate_id)] = tuple(float(value) for value in bbox)
                tile_bbox_map[f"{candidate_id}_label"] = tuple(
                    float(value) for value in label_bbox
                )
            entities.append(
                {
                    "id": str(tile.tile_id),
                    "type": "pipe_flow_tile",
                    "label": str(tile.label),
                    "row_index": int(tile.row),
                    "col_index": int(tile.col),
                    "bbox_px": [int(value) for value in bbox],
                    "current_openings": list(tile.current_openings),
                    "required_openings": list(tile.required_openings),
                    "is_path": bool(tile.is_path),
                    "is_branch": bool(tile.is_branch),
                    "is_candidate": bool(tile.label),
                }
            )

    for candidate in dataset.candidates:
        tile_id = str(candidate.tile_id)
        candidate_bbox = item_bbox_map[tile_id]
        entities.append(
            {
                "id": str(candidate.candidate_id),
                "type": "pipe_flow_candidate_tile",
                "label": str(candidate.label),
                "tile_id": tile_id,
                "bbox_px": [int(value) for value in candidate_bbox],
                "row_index": int(candidate.row),
                "col_index": int(candidate.col),
                "required_openings": list(candidate.required_openings),
                "current_openings": list(candidate.current_openings),
                "repair_rotation_turns": [int(value) for value in candidate.repair_rotation_turns],
                "is_correct": bool(candidate.is_correct),
                "connects_after_rotation": bool(candidate.connects_after_rotation),
            }
        )

    start_bbox = tile_bbox(
        row=dataset.start_cell[0],
        col=dataset.start_cell[1],
        grid_left=grid_left,
        grid_top=grid_top,
        cell_size=cell_size,
        gap=int(render_params.cell_gap_px),
    )
    dest_bbox = tile_bbox(
        row=dataset.destination_cell[0],
        col=dataset.destination_cell[1],
        grid_left=grid_left,
        grid_top=grid_top,
        cell_size=cell_size,
        gap=int(render_params.cell_gap_px),
    )
    start_direction = _start_flow_direction(dataset)
    start_marker_bbox = draw_start_marker(
        draw,
        bbox=start_bbox,
        direction=start_direction,
        render_params=render_params,
    )
    finish_marker_bbox = draw_finish_flag(draw, bbox=dest_bbox, render_params=render_params)
    item_bbox_map["start_marker"] = tuple(float(value) for value in start_marker_bbox)
    item_bbox_map["finish_flag"] = tuple(float(value) for value in finish_marker_bbox)
    entities.extend(
        [
            {
                "id": "start_marker",
                "type": "pipe_flow_start_marker",
                "bbox_px": [int(value) for value in start_marker_bbox],
                "cell": [int(dataset.start_cell[0]), int(dataset.start_cell[1])],
                "direction": str(start_direction),
            },
            {
                "id": "finish_flag",
                "type": "pipe_flow_finish_flag",
                "bbox_px": [int(value) for value in finish_marker_bbox],
                "cell": [int(dataset.destination_cell[0]), int(dataset.destination_cell[1])],
            },
        ]
    )
    return RenderedPipeFlowScene(
        image=image,
        scene_bbox_px=tuple(float(value) for value in panel_bbox),
        tile_bbox_map=dict(tile_bbox_map),
        item_bbox_map=dict(item_bbox_map),
        entities=tuple(entities),
    )


def _start_flow_direction(dataset: PipeFlowDataset | PipeFlowMisrotatedDataset) -> str:
    """Return the cardinal direction from the start cell to the next path cell."""

    if len(dataset.path_cells) < 2:
        return "E"
    start = tuple(int(value) for value in dataset.path_cells[0])
    nxt = tuple(int(value) for value in dataset.path_cells[1])
    delta = (int(nxt[0] - start[0]), int(nxt[1] - start[1]))
    if delta == (-1, 0):
        return "N"
    if delta == (1, 0):
        return "S"
    if delta == (0, -1):
        return "W"
    return "E"
