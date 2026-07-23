"""Rendering helpers for maze-exit puzzle scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.shared.text_rendering import (
    draw_text_centered,
    fit_font_to_box,
    load_font,
)

from .state import BBox, Cell, Color, MazeExitRenderParams, RenderedMazeExitScene
from .topology import edge_key, maze_cell_id

def line_with_gap(
    draw: ImageDraw.ImageDraw,
    *,
    start: Tuple[float, float],
    end: Tuple[float, float],
    gap_center: Tuple[float, float] | None,
    gap_size: float,
    fill: Color,
    width: int,
) -> None:
    if gap_center is None:
        draw.line((float(start[0]), float(start[1]), float(end[0]), float(end[1])), fill=fill, width=int(width))
        return
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    cx, cy = float(gap_center[0]), float(gap_center[1])
    if abs(y0 - y1) <= 1e-6:
        half = 0.5 * float(gap_size)
        draw.line((x0, y0, max(x0, cx - half), y0), fill=fill, width=int(width))
        draw.line((min(x1, cx + half), y0, x1, y0), fill=fill, width=int(width))
    else:
        half = 0.5 * float(gap_size)
        draw.line((x0, y0, x0, max(y0, cy - half)), fill=fill, width=int(width))
        draw.line((x0, min(y1, cy + half), x0, y1), fill=fill, width=int(width))
def cell_center(*, left: float, top: float, cell_size: float, cell: Cell) -> Tuple[float, float]:
    return (
        float(left + ((float(cell[0]) + 0.5) * float(cell_size))),
        float(top + ((float(cell[1]) + 0.5) * float(cell_size))),
    )
def text_bbox_for_center(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font,
    pad: float,
    stroke_width: int,
) -> BBox:
    try:
        raw = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width)))
        width = float(raw[2] - raw[0])
        height = float(raw[3] - raw[1])
    except Exception:
        width, height = draw.textsize(str(text), font=font)
    cx, cy = float(center[0]), float(center[1])
    return (float(cx - (0.5 * width) - pad), float(cy - (0.5 * height) - pad), float(cx + (0.5 * width) + pad), float(cy + (0.5 * height) + pad))
def maze_exit_marker_bbox(
    *,
    center: Tuple[float, float],
    radius: float,
    side: str,
    shape: str,
) -> BBox:
    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    if str(shape) == "tab":
        if str(side) in {"top", "bottom"}:
            return (float(cx - (1.18 * r)), float(cy - (0.78 * r)), float(cx + (1.18 * r)), float(cy + (0.78 * r)))
        return (float(cx - (0.78 * r)), float(cy - (1.18 * r)), float(cx + (0.78 * r)), float(cy + (1.18 * r)))
    return (float(cx - r), float(cy - r), float(cx + r), float(cy + r))
def draw_maze_exit_marker(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: BBox,
    fill: Color,
    outline: Color,
    shape: str,
) -> None:
    if str(shape) == "square":
        radius = max(3, int(round((float(bbox[2]) - float(bbox[0])) * 0.10)))
        draw.rounded_rectangle(tuple(float(value) for value in bbox), radius=radius, fill=tuple(fill), outline=tuple(outline), width=3)
    elif str(shape) == "tab":
        radius = max(5, int(round(min(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])) * 0.22)))
        draw.rounded_rectangle(tuple(float(value) for value in bbox), radius=radius, fill=tuple(fill), outline=tuple(outline), width=3)
    else:
        draw.ellipse(tuple(float(value) for value in bbox), fill=tuple(fill), outline=tuple(outline), width=3)
def render_maze_exit_scene(
    background: Image.Image,
    *,
    dataset: Mapping[str, Any],
    scene_variant: str,
    render_params: MazeExitRenderParams,
) -> RenderedMazeExitScene:
    """Draw the complete maze panel and record every cell/exit bbox."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    rows = int(dataset["maze_rows"])
    cols = int(dataset["maze_cols"])
    usable_width = float(render_params.canvas_width - render_params.scene_margin_left_px - render_params.scene_margin_right_px)
    usable_height = float(render_params.canvas_height - render_params.scene_margin_top_px - render_params.scene_margin_bottom_px)
    cell_size = float(min(usable_width / float(cols), usable_height / float(rows))) * float(render_params.unit_size_scale)
    grid_width = float(cell_size * float(cols))
    grid_height = float(cell_size * float(rows))
    left = float((render_params.canvas_width - grid_width) / 2.0)
    top = float((render_params.canvas_height - grid_height) / 2.0)
    right = float(left + grid_width)
    bottom = float(top + grid_height)
    pad = float(max(18, int(round(cell_size * 0.26))))

    scene_panel = (float(left - pad), float(top - pad), float(right + pad), float(bottom + pad))
    draw.rectangle(scene_panel, fill=tuple(render_params.panel_fill_rgb), outline=tuple(render_params.border_color_rgb), width=2)
    draw.rectangle((float(left), float(top), float(right), float(bottom)), fill=tuple(render_params.floor_fill_rgb))

    if str(scene_variant) in {"paper_labyrinth_maze", "block_wall_maze"}:
        grid_width_px = 1 if str(scene_variant) == "paper_labyrinth_maze" else 2
        for col in range(1, int(cols)):
            x = float(left + (col * cell_size))
            draw.line((x, top, x, bottom), fill=tuple(render_params.subtle_grid_rgb), width=int(grid_width_px))
        for row in range(1, int(rows)):
            y = float(top + (row * cell_size))
            draw.line((left, y, right, y), fill=tuple(render_params.subtle_grid_rgb), width=int(grid_width_px))

    exits = [dict(exit_spec) for exit_spec in dataset["exits"]]
    exit_by_wall: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for exit_spec in exits:
        col, row = (int(value) for value in exit_spec["cell"])
        side = str(exit_spec["side"])
        if side == "top":
            exit_by_wall[("top", col)] = exit_spec
        elif side == "bottom":
            exit_by_wall[("bottom", col)] = exit_spec
        elif side == "left":
            exit_by_wall[("left", row)] = exit_spec
        else:
            exit_by_wall[("right", row)] = exit_spec

    open_edges = {
        edge_key(tuple(edge[0]), tuple(edge[1]))
        for edge in dataset["open_edges"]
    }
    cell_bbox_map: Dict[str, BBox] = {}
    wall_width = int(render_params.wall_stroke_width_px)
    outer_wall_width = int(render_params.outer_wall_stroke_width_px)
    gap_size = float(cell_size * 0.52)

    for row in range(int(rows)):
        y0 = float(top + (row * cell_size))
        y1 = float(top + ((row + 1) * cell_size))
        for col in range(int(cols)):
            x0 = float(left + (col * cell_size))
            x1 = float(left + ((col + 1) * cell_size))
            current = (int(col), int(row))
            cell_bbox_map[maze_cell_id(current)] = (float(x0), float(y0), float(x1), float(y1))
            if row == 0:
                exit_spec = exit_by_wall.get(("top", int(col)))
                line_with_gap(
                    draw,
                    start=(x0, y0),
                    end=(x1, y0),
                    gap_center=((x0 + x1) / 2.0, y0) if exit_spec is not None else None,
                    gap_size=gap_size,
                    fill=render_params.wall_color_rgb,
                    width=outer_wall_width,
                )
            if col == 0:
                exit_spec = exit_by_wall.get(("left", int(row)))
                line_with_gap(
                    draw,
                    start=(x0, y0),
                    end=(x0, y1),
                    gap_center=(x0, (y0 + y1) / 2.0) if exit_spec is not None else None,
                    gap_size=gap_size,
                    fill=render_params.wall_color_rgb,
                    width=outer_wall_width,
                )
            if col == int(cols) - 1:
                exit_spec = exit_by_wall.get(("right", int(row)))
                line_with_gap(
                    draw,
                    start=(x1, y0),
                    end=(x1, y1),
                    gap_center=(x1, (y0 + y1) / 2.0) if exit_spec is not None else None,
                    gap_size=gap_size,
                    fill=render_params.wall_color_rgb,
                    width=outer_wall_width,
                )
            else:
                neighbor = (int(col) + 1, int(row))
                if edge_key(current, neighbor) not in open_edges:
                    draw.line((x1, y0, x1, y1), fill=tuple(render_params.wall_color_rgb), width=wall_width)
            if row == int(rows) - 1:
                exit_spec = exit_by_wall.get(("bottom", int(col)))
                line_with_gap(
                    draw,
                    start=(x0, y1),
                    end=(x1, y1),
                    gap_center=((x0 + x1) / 2.0, y1) if exit_spec is not None else None,
                    gap_size=gap_size,
                    fill=render_params.wall_color_rgb,
                    width=outer_wall_width,
                )
            else:
                neighbor = (int(col), int(row) + 1)
                if edge_key(current, neighbor) not in open_edges:
                    draw.line((x0, y1, x1, y1), fill=tuple(render_params.wall_color_rgb), width=wall_width)

    entities: List[Dict[str, Any]] = []
    item_bbox_map: Dict[str, BBox] = {}
    item_point_map: Dict[str, Tuple[float, float]] = {}

    start_cell = tuple(int(value) for value in dataset["start_cell"])
    start_center = cell_center(left=left, top=top, cell_size=cell_size, cell=start_cell)
    start_box_w = float(cell_size * 0.72)
    start_box_h = float(cell_size * 0.42)
    start_bbox = (
        float(start_center[0] - (0.5 * start_box_w)),
        float(start_center[1] - (0.5 * start_box_h)),
        float(start_center[0] + (0.5 * start_box_w)),
        float(start_center[1] + (0.5 * start_box_h)),
    )
    draw.rounded_rectangle(start_bbox, radius=max(4, int(round(cell_size * 0.08))), fill=tuple(render_params.start_fill_rgb), outline=tuple(render_params.start_outline_rgb), width=3)
    start_font = fit_font_to_box(
        draw,
        text="START",
        max_width=float(start_box_w),
        max_height=float(start_box_h),
        bold=True,
        min_size_px=10,
        max_size_px=int(render_params.start_font_size_px),
        fill_ratio=0.86,
    )
    draw_text_centered(
        draw,
        text="START",
        center=start_center,
        font=start_font,
        fill=tuple(render_params.text_color_rgb),
        stroke_fill=tuple(render_params.text_stroke_rgb),
        stroke_width=1,
    )
    entities.append({"entity_id": "start", "role": "start", "label": "START", "cell": [int(start_cell[0]), int(start_cell[1])], "bbox_px": list(start_bbox)})

    exit_font = load_font(int(render_params.exit_label_font_size_px), bold=True)
    marker_radius = float(render_params.exit_marker_radius_px)
    label_offset = float(marker_radius + max(14.0, cell_size * 0.22))
    door_color = tuple(render_params.floor_fill_rgb)
    for index, exit_spec in enumerate(exits):
        col, row = (int(value) for value in exit_spec["cell"])
        side = str(exit_spec["side"])
        exit_cell_center = cell_center(left=left, top=top, cell_size=cell_size, cell=(col, row))
        if side == "top":
            marker_center = (float(exit_cell_center[0]), float(top - label_offset))
            door_bbox = (float(exit_cell_center[0] - (gap_size * 0.35)), float(top - outer_wall_width * 0.6), float(exit_cell_center[0] + (gap_size * 0.35)), float(top + outer_wall_width * 0.6))
        elif side == "bottom":
            marker_center = (float(exit_cell_center[0]), float(bottom + label_offset))
            door_bbox = (float(exit_cell_center[0] - (gap_size * 0.35)), float(bottom - outer_wall_width * 0.6), float(exit_cell_center[0] + (gap_size * 0.35)), float(bottom + outer_wall_width * 0.6))
        elif side == "left":
            marker_center = (float(left - label_offset), float(exit_cell_center[1]))
            door_bbox = (float(left - outer_wall_width * 0.6), float(exit_cell_center[1] - (gap_size * 0.35)), float(left + outer_wall_width * 0.6), float(exit_cell_center[1] + (gap_size * 0.35)))
        else:
            marker_center = (float(right + label_offset), float(exit_cell_center[1]))
            door_bbox = (float(right - outer_wall_width * 0.6), float(exit_cell_center[1] - (gap_size * 0.35)), float(right + outer_wall_width * 0.6), float(exit_cell_center[1] + (gap_size * 0.35)))

        draw.rectangle(door_bbox, fill=door_color)
        marker_fill = tuple(render_params.exit_palette[int(index) % len(render_params.exit_palette)])
        marker_bbox = maze_exit_marker_bbox(
            center=marker_center,
            radius=float(marker_radius),
            side=str(side),
            shape=str(render_params.exit_marker_shape),
        )
        draw_maze_exit_marker(
            draw,
            bbox=marker_bbox,
            fill=marker_fill,
            outline=tuple(render_params.exit_outline_rgb),
            shape=str(render_params.exit_marker_shape),
        )
        draw_text_centered(
            draw,
            text=str(exit_spec["label"]),
            center=marker_center,
            font=exit_font,
            fill=tuple(render_params.text_color_rgb),
            stroke_fill=tuple(render_params.text_stroke_rgb),
            stroke_width=1,
        )
        text_bbox = text_bbox_for_center(
            draw,
            text=str(exit_spec["label"]),
            center=marker_center,
            font=exit_font,
            pad=8.0,
            stroke_width=1,
        )
        item_bbox = (
            float(min(marker_bbox[0], text_bbox[0], door_bbox[0])),
            float(min(marker_bbox[1], text_bbox[1], door_bbox[1])),
            float(max(marker_bbox[2], text_bbox[2], door_bbox[2])),
            float(max(marker_bbox[3], text_bbox[3], door_bbox[3])),
        )
        item_bbox_map[str(exit_spec["item_id"])] = item_bbox
        item_point_map[str(exit_spec["item_id"])] = (
            round(float(marker_center[0]), 3),
            round(float(marker_center[1]), 3),
        )
        entities.append(
            {
                "entity_id": str(exit_spec["item_id"]),
                "role": "reachable_exit" if bool(exit_spec["reachable"]) else "unreachable_exit",
                "label": str(exit_spec["label"]),
                "cell": [int(col), int(row)],
                "side": str(side),
                "reachable": bool(exit_spec["reachable"]),
                "bbox_px": list(item_bbox),
                "point_px": list(item_point_map[str(exit_spec["item_id"])]),
            }
        )

    scene_bbox = (
        float(max(0.0, min(scene_panel[0], *(bbox[0] for bbox in item_bbox_map.values())))),
        float(max(0.0, min(scene_panel[1], *(bbox[1] for bbox in item_bbox_map.values())))),
        float(min(float(render_params.canvas_width), max(scene_panel[2], *(bbox[2] for bbox in item_bbox_map.values())))),
        float(min(float(render_params.canvas_height), max(scene_panel[3], *(bbox[3] for bbox in item_bbox_map.values())))),
    )
    return RenderedMazeExitScene(
        image=image,
        entities=tuple(entities),
        scene_bbox_px=scene_bbox,
        item_bbox_map=dict(item_bbox_map),
        item_point_map=dict(item_point_map),
        cell_bbox_map=dict(cell_bbox_map),
    )


__all__ = [
    "render_maze_exit_scene",
]
