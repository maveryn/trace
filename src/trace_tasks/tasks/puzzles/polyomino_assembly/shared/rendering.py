"""Rendering for polyomino assembly option puzzles."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from PIL import Image, ImageDraw

from trace_tasks.tasks.shared.text_rendering import draw_text_centered, load_font

from .rules import canonicalize_cells, shape_bbox_dims
from .state import AssemblyRenderParams, Cell, Cells, RenderedPolyominoAssemblyScene


def render_polyomino_assembly_scene(
    background: Image.Image,
    *,
    scene_variant: str,
    dataset: Mapping[str, Any],
    render_params: AssemblyRenderParams,
) -> RenderedPolyominoAssemblyScene:
    """Render one target/source panel plus four option panels."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    entities: list[dict[str, Any]] = []
    item_bbox_map: dict[str, list[float]] = {}
    option_choice_bbox_map: dict[str, list[float]] = {}

    left = float(render_params.scene_margin_left_px)
    top = float(render_params.scene_margin_top_px)
    right = float(render_params.canvas_width - render_params.scene_margin_right_px)
    top_panel_bottom = top + float(render_params.top_panel_height_px)
    top_panel_bbox = [left, top, right, top_panel_bottom]
    _draw_panel(draw, top_panel_bbox, render_params, selected=False)

    top_kind = str(dataset.get("top_kind", "target"))
    if top_kind == "source_pair":
        _draw_source_pair(
            draw,
            panel_bbox=top_panel_bbox,
            dataset=dataset,
            render_params=render_params,
            entities=entities,
            item_bbox_map=item_bbox_map,
        )
    elif top_kind == "hole_board":
        _draw_hole_board(
            draw,
            panel_bbox=top_panel_bbox,
            dataset=dataset,
            render_params=render_params,
            entities=entities,
            item_bbox_map=item_bbox_map,
        )
    else:
        _draw_target_shape(
            draw,
            panel_bbox=top_panel_bbox,
            dataset=dataset,
            render_params=render_params,
            entities=entities,
            item_bbox_map=item_bbox_map,
        )

    option_panel_bboxes = _option_panel_bboxes(render_params)
    option_specs = list(dataset.get("option_specs", []))
    for option_index, option_spec in enumerate(option_specs):
        if option_index >= len(option_panel_bboxes):
            break
        panel_bbox = option_panel_bboxes[option_index]
        option_id = str(option_spec.get("option_choice_id", f"option_choice_{option_index + 1}"))
        option_choice_bbox_map[option_id] = [round(float(value), 3) for value in panel_bbox]
        item_bbox_map[option_id] = [round(float(value), 3) for value in panel_bbox]
        _draw_panel(draw, panel_bbox, render_params, selected=False)
        _draw_option_label(
            draw,
            panel_bbox=panel_bbox,
            label=str(option_spec.get("option_label", "")),
            render_params=render_params,
        )
        if "pieces" in option_spec:
            _draw_option_pair(
                draw,
                panel_bbox=panel_bbox,
                option_spec=option_spec,
                render_params=render_params,
                entities=entities,
                item_bbox_map=item_bbox_map,
            )
        else:
            _draw_option_shape(
                draw,
                panel_bbox=panel_bbox,
                option_spec=option_spec,
                render_params=render_params,
                entities=entities,
                item_bbox_map=item_bbox_map,
            )
        entities.append(
            {
                "entity_id": option_id,
                "entity_type": "polyomino_option_panel",
                "option_label": str(option_spec.get("option_label", "")),
                "bbox_px": [round(float(value), 3) for value in panel_bbox],
                "is_correct": bool(option_spec.get("is_correct", False)),
            }
        )

    scene_bbox = [
        min(left, *(bbox[0] for bbox in option_panel_bboxes)),
        top,
        max(right, *(bbox[2] for bbox in option_panel_bboxes)),
        max(top_panel_bottom, *(bbox[3] for bbox in option_panel_bboxes)),
    ]
    return RenderedPolyominoAssemblyScene(
        image=image,
        entities=entities,
        scene_bbox_px=[round(float(value), 3) for value in scene_bbox],
        item_bbox_map=item_bbox_map,
        option_choice_bbox_map=option_choice_bbox_map,
    )


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    bbox: list[float],
    render_params: AssemblyRenderParams,
    *,
    selected: bool,
) -> None:
    """Draw one rounded content panel."""

    fill = tuple(int(value) for value in render_params.option_panel_fill_rgb)
    outline = tuple(int(value) for value in render_params.border_color_rgb)
    width = max(1, int(render_params.border_width_px))
    draw.rounded_rectangle(
        [float(value) for value in bbox],
        radius=int(render_params.panel_corner_radius_px),
        fill=fill,
        outline=outline,
        width=width + (1 if bool(selected) else 0),
    )


def _draw_target_shape(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: list[float],
    dataset: Mapping[str, Any],
    render_params: AssemblyRenderParams,
    entities: list[dict[str, Any]],
    item_bbox_map: dict[str, list[float]],
) -> None:
    """Draw the target polyomino centered in the top panel."""

    center = _bbox_center(panel_bbox)
    cells = _cells_from_json(dataset.get("target_cells", []))
    _draw_polyomino_cells(
        draw,
        cells=cells,
        center=center,
        fill_rgb=render_params.shape_fill_rgb,
        shape_id="target_shape",
        render_params=render_params,
        entities=entities,
        item_bbox_map=item_bbox_map,
    )


def _draw_source_pair(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: list[float],
    dataset: Mapping[str, Any],
    render_params: AssemblyRenderParams,
    entities: list[dict[str, Any]],
    item_bbox_map: dict[str, list[float]],
) -> None:
    """Draw the two source pieces centered in the top panel."""

    source_pieces = list(dataset.get("source_pieces", []))
    if len(source_pieces) != 2:
        raise ValueError("source_pair scene requires exactly two source pieces")
    cells_a = _cells_from_json(source_pieces[0].get("cells", []))
    cells_b = _cells_from_json(source_pieces[1].get("cells", []))
    width_a, _height_a = _shape_pixel_size(cells_a, render_params)
    width_b, _height_b = _shape_pixel_size(cells_b, render_params)
    gap = float(render_params.source_gap_px)
    total_width = float(width_a + width_b) + gap
    center_x, center_y = _bbox_center(panel_bbox)
    center_a = (float(center_x - (total_width / 2.0) + (width_a / 2.0)), center_y)
    center_b = (float(center_x + (total_width / 2.0) - (width_b / 2.0)), center_y)
    _draw_polyomino_cells(
        draw,
        cells=cells_a,
        center=center_a,
        fill_rgb=render_params.shape_fill_rgb,
        shape_id="source_piece_a",
        render_params=render_params,
        entities=entities,
        item_bbox_map=item_bbox_map,
    )
    _draw_polyomino_cells(
        draw,
        cells=cells_b,
        center=center_b,
        fill_rgb=render_params.shape_fill_rgb,
        shape_id="source_piece_b",
        render_params=render_params,
        entities=entities,
        item_bbox_map=item_bbox_map,
    )


def _draw_hole_board(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: list[float],
    dataset: Mapping[str, Any],
    render_params: AssemblyRenderParams,
    entities: list[dict[str, Any]],
    item_bbox_map: dict[str, list[float]],
) -> None:
    """Draw a filled polyomino board with a blank interior hole."""

    board_width = int(dataset.get("board_width", 0))
    board_height = int(dataset.get("board_height", 0))
    if board_width <= 0 or board_height <= 0:
        raise ValueError("hole board requires positive board dimensions")
    board_cells = {
        (int(cell[0]), int(cell[1]))
        for cell in dataset.get("board_cells", [])
    }
    hole_cells = {
        (int(cell[0]), int(cell[1]))
        for cell in dataset.get("hole_cells", [])
    }
    step = int(render_params.cell_size_px + render_params.cell_gap_px)
    pixel_width = int(
        (board_width * render_params.cell_size_px)
        + ((board_width - 1) * render_params.cell_gap_px)
    )
    pixel_height = int(
        (board_height * render_params.cell_size_px)
        + ((board_height - 1) * render_params.cell_gap_px)
    )
    center_x, center_y = _bbox_center(panel_bbox)
    origin_x = float(center_x - (pixel_width / 2.0))
    origin_y = float(center_y - (pixel_height / 2.0))
    board_bboxes: list[list[float]] = []
    hole_bboxes: list[list[float]] = []
    for y in range(board_height):
        for x in range(board_width):
            cell = (int(x), int(y))
            x0 = origin_x + float(x * step)
            y0 = origin_y + float(y * step)
            bbox = [
                x0,
                y0,
                x0 + float(render_params.cell_size_px),
                y0 + float(render_params.cell_size_px),
            ]
            if cell in hole_cells:
                fill_rgb = render_params.option_panel_fill_rgb
                hole_bboxes.append([round(float(value), 3) for value in bbox])
            elif cell in board_cells:
                fill_rgb = render_params.shape_fill_rgb
                board_bboxes.append([round(float(value), 3) for value in bbox])
            else:
                continue
            draw.rounded_rectangle(
                bbox,
                radius=int(render_params.cell_corner_radius_px),
                fill=tuple(int(value) for value in fill_rgb),
                outline=tuple(int(value) for value in render_params.border_color_rgb),
                width=max(1, int(render_params.border_width_px)),
            )
    if not board_bboxes or not hole_bboxes:
        raise ValueError("hole board requires both filled cells and hole cells")
    board_bbox = [
        min(bbox[0] for bbox in [*board_bboxes, *hole_bboxes]),
        min(bbox[1] for bbox in [*board_bboxes, *hole_bboxes]),
        max(bbox[2] for bbox in [*board_bboxes, *hole_bboxes]),
        max(bbox[3] for bbox in [*board_bboxes, *hole_bboxes]),
    ]
    hole_bbox = [
        min(bbox[0] for bbox in hole_bboxes),
        min(bbox[1] for bbox in hole_bboxes),
        max(bbox[2] for bbox in hole_bboxes),
        max(bbox[3] for bbox in hole_bboxes),
    ]
    item_bbox_map["hole_board"] = [round(float(value), 3) for value in board_bbox]
    item_bbox_map["hole_region"] = [round(float(value), 3) for value in hole_bbox]
    entities.append(
        {
            "entity_id": "hole_board",
            "entity_type": "polyomino_hole_board",
            "board_width": int(board_width),
            "board_height": int(board_height),
            "board_cells": [[int(x), int(y)] for x, y in sorted(board_cells)],
            "hole_cells": [[int(x), int(y)] for x, y in sorted(hole_cells)],
            "bbox_px": [round(float(value), 3) for value in board_bbox],
            "hole_bbox_px": [round(float(value), 3) for value in hole_bbox],
        }
    )


def _draw_option_pair(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: list[float],
    option_spec: Mapping[str, Any],
    render_params: AssemblyRenderParams,
    entities: list[dict[str, Any]],
    item_bbox_map: dict[str, list[float]],
) -> None:
    """Draw two piece shapes inside one option card."""

    pieces = list(option_spec.get("pieces", []))
    cells_a = _cells_from_json(pieces[0].get("cells", []))
    cells_b = _cells_from_json(pieces[1].get("cells", []))
    width_a, _height_a = _shape_pixel_size(cells_a, render_params)
    width_b, _height_b = _shape_pixel_size(cells_b, render_params)
    gap = float(render_params.source_gap_px) * 0.62
    total_width = float(width_a + width_b) + gap
    center_x, center_y = _bbox_center(panel_bbox)
    center_a = (float(center_x - (total_width / 2.0) + (width_a / 2.0)), center_y)
    center_b = (float(center_x + (total_width / 2.0) - (width_b / 2.0)), center_y)
    label = str(option_spec.get("option_label", ""))
    _draw_polyomino_cells(
        draw,
        cells=cells_a,
        center=center_a,
        fill_rgb=render_params.shape_fill_rgb,
        shape_id=f"option_{label}_piece_a",
        render_params=render_params,
        entities=entities,
        item_bbox_map=item_bbox_map,
    )
    _draw_polyomino_cells(
        draw,
        cells=cells_b,
        center=center_b,
        fill_rgb=render_params.shape_fill_rgb,
        shape_id=f"option_{label}_piece_b",
        render_params=render_params,
        entities=entities,
        item_bbox_map=item_bbox_map,
    )


def _draw_option_shape(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: list[float],
    option_spec: Mapping[str, Any],
    render_params: AssemblyRenderParams,
    entities: list[dict[str, Any]],
    item_bbox_map: dict[str, list[float]],
) -> None:
    """Draw one composite option shape inside an option card."""

    label = str(option_spec.get("option_label", ""))
    cells = _cells_from_json(option_spec.get("cells", []))
    _draw_polyomino_cells(
        draw,
        cells=cells,
        center=_bbox_center(panel_bbox),
        fill_rgb=render_params.shape_fill_rgb,
        shape_id=f"option_{label}_shape",
        render_params=render_params,
        entities=entities,
        item_bbox_map=item_bbox_map,
    )


def _draw_option_label(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: list[float],
    label: str,
    render_params: AssemblyRenderParams,
) -> None:
    """Draw a compact option label in the upper-left of an option card."""

    radius = max(18, int(render_params.option_label_font_size_px * 0.62))
    center = (
        float(panel_bbox[0] + render_params.panel_padding_px + radius),
        float(panel_bbox[1] + render_params.panel_padding_px + radius),
    )
    circle = [
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    ]
    draw.ellipse(
        circle,
        fill=tuple(int(value) for value in render_params.panel_fill_rgb),
        outline=tuple(int(value) for value in render_params.border_color_rgb),
        width=max(1, int(render_params.border_width_px)),
    )
    font = load_font(int(render_params.option_label_font_size_px), bold=True)
    draw_text_centered(
        draw,
        text=str(label),
        center=center,
        font=font,
        fill=render_params.text_color_rgb,
        stroke_fill=render_params.text_stroke_rgb,
        stroke_width=1,
    )


def _draw_polyomino_cells(
    draw: ImageDraw.ImageDraw,
    *,
    cells: Cells,
    center: tuple[float, float],
    fill_rgb: tuple[int, int, int],
    shape_id: str,
    render_params: AssemblyRenderParams,
    entities: list[dict[str, Any]],
    item_bbox_map: dict[str, list[float]],
) -> list[float]:
    """Draw one polyomino and trace its enclosing bbox."""

    canonical = canonicalize_cells(cells)
    width, height = shape_bbox_dims(canonical)
    step = int(render_params.cell_size_px + render_params.cell_gap_px)
    pixel_width = int((width * render_params.cell_size_px) + ((width - 1) * render_params.cell_gap_px))
    pixel_height = int((height * render_params.cell_size_px) + ((height - 1) * render_params.cell_gap_px))
    origin_x = float(center[0] - (pixel_width / 2.0))
    origin_y = float(center[1] - (pixel_height / 2.0))
    cell_bboxes: list[list[float]] = []
    for cell_x, cell_y in canonical:
        x0 = origin_x + float(cell_x * step)
        y0 = origin_y + float(cell_y * step)
        x1 = x0 + float(render_params.cell_size_px)
        y1 = y0 + float(render_params.cell_size_px)
        bbox = [x0, y0, x1, y1]
        draw.rounded_rectangle(
            bbox,
            radius=int(render_params.cell_corner_radius_px),
            fill=tuple(int(value) for value in fill_rgb),
            outline=tuple(int(value) for value in render_params.border_color_rgb),
            width=max(1, int(render_params.border_width_px)),
        )
        cell_bboxes.append([round(float(value), 3) for value in bbox])
    shape_bbox = [
        min(bbox[0] for bbox in cell_bboxes),
        min(bbox[1] for bbox in cell_bboxes),
        max(bbox[2] for bbox in cell_bboxes),
        max(bbox[3] for bbox in cell_bboxes),
    ]
    item_bbox_map[str(shape_id)] = [round(float(value), 3) for value in shape_bbox]
    entities.append(
        {
            "entity_id": str(shape_id),
            "entity_type": "polyomino_shape",
            "cells": [[int(x), int(y)] for x, y in canonical],
            "bbox_px": [round(float(value), 3) for value in shape_bbox],
        }
    )
    return [round(float(value), 3) for value in shape_bbox]


def _option_panel_bboxes(render_params: AssemblyRenderParams) -> list[list[float]]:
    """Return the fixed 2x2 option-card bboxes."""

    panel_w = float(render_params.option_panel_width_px)
    panel_h = float(render_params.option_panel_height_px)
    gap_x = float(render_params.option_gap_px)
    gap_y = float(render_params.option_row_gap_px)
    total_w = (2.0 * panel_w) + gap_x
    x0 = (float(render_params.canvas_width) - total_w) / 2.0
    y0 = (
        float(render_params.scene_margin_top_px)
        + float(render_params.top_panel_height_px)
        + float(render_params.top_to_options_gap_px)
    )
    return [
        [x0, y0, x0 + panel_w, y0 + panel_h],
        [x0 + panel_w + gap_x, y0, x0 + (2.0 * panel_w) + gap_x, y0 + panel_h],
        [x0, y0 + panel_h + gap_y, x0 + panel_w, y0 + (2.0 * panel_h) + gap_y],
        [
            x0 + panel_w + gap_x,
            y0 + panel_h + gap_y,
            x0 + (2.0 * panel_w) + gap_x,
            y0 + (2.0 * panel_h) + gap_y,
        ],
    ]


def _shape_pixel_size(cells: Cells, render_params: AssemblyRenderParams) -> tuple[int, int]:
    """Return shape pixel width and height at the current render scale."""

    width, height = shape_bbox_dims(cells)
    pixel_width = int(
        (width * render_params.cell_size_px)
        + ((width - 1) * render_params.cell_gap_px)
    )
    pixel_height = int(
        (height * render_params.cell_size_px)
        + ((height - 1) * render_params.cell_gap_px)
    )
    return int(pixel_width), int(pixel_height)


def _cells_from_json(raw_cells: Any) -> Cells:
    """Convert JSON-style cell coordinates to canonical cells."""

    return canonicalize_cells((int(cell[0]), int(cell[1])) for cell in raw_cells)


def _bbox_center(bbox: Iterable[float]) -> tuple[float, float]:
    """Return the center of a bbox."""

    values = [float(value) for value in bbox]
    return ((values[0] + values[2]) / 2.0, (values[1] + values[3]) / 2.0)


__all__ = ["render_polyomino_assembly_scene"]
