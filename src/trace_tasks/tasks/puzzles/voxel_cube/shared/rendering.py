"""Rendering primitives for voxel-cube puzzle scenes."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from PIL import ImageDraw

from trace_tasks.tasks.puzzles.shared.scene_style import (
    PuzzleSceneStyle,
    draw_puzzle_option_card,
)
from trace_tasks.tasks.shared.text_rendering import draw_text_centered, load_font

from .state import (
    BBox,
    ChangeDataset,
    CountDataset,
    CubeStack,
    ProjectionCountDataset,
    ProjectionGrid,
    ProjectionMatchDataset,
    RenderedVoxelScene,
    VoxelPalette,
    VoxelRenderParams,
)


def render_single_stack_scene(
    background,
    *,
    dataset: CountDataset,
    style: PuzzleSceneStyle,
    render_params: VoxelRenderParams,
) -> RenderedVoxelScene:
    """Render one standalone voxel stack for a structure count task."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    stack_bbox = _draw_stack_centered(
        draw,
        dataset.stack,
        center=(render_params.canvas_width * 0.5, render_params.canvas_height * 0.52),
        cube_size=int(render_params.cube_size_px),
        palette=render_params.palette,
    )
    scene_bbox = _pad_bbox(stack_bbox, 30.0)
    return RenderedVoxelScene(
        image=image,
        scene_bbox_px=scene_bbox,
        stack_bbox_px=stack_bbox,
        reference_stack_bbox_px=None,
        changed_stack_bbox_px=None,
        projection_cell_bbox_map={},
        option_panel_bbox_map={},
    )


def render_change_scene(
    background,
    *,
    dataset: ChangeDataset,
    style: PuzzleSceneStyle,
    render_params: VoxelRenderParams,
) -> RenderedVoxelScene:
    """Render reference and changed voxel structures side by side."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    left_center = (
        render_params.canvas_width * 0.32,
        render_params.canvas_height * 0.56,
    )
    right_center = (
        render_params.canvas_width * 0.68,
        render_params.canvas_height * 0.56,
    )
    reference_bbox = _draw_stack_centered(
        draw,
        dataset.reference_stack,
        center=left_center,
        cube_size=int(render_params.cube_size_px),
        palette=render_params.palette,
    )
    changed_bbox = _draw_stack_centered(
        draw,
        dataset.changed_stack,
        center=right_center,
        cube_size=int(render_params.cube_size_px),
        palette=render_params.palette,
    )
    _draw_scene_label(
        draw,
        text="Reference",
        center=(left_center[0], reference_bbox[1] - 24.0),
        style=style,
        font_size=int(render_params.label_font_size_px),
    )
    _draw_scene_label(
        draw,
        text="Changed",
        center=(right_center[0], changed_bbox[1] - 24.0),
        style=style,
        font_size=int(render_params.label_font_size_px),
    )
    scene_bbox = _union_bboxes([reference_bbox, changed_bbox])
    return RenderedVoxelScene(
        image=image,
        scene_bbox_px=_pad_bbox(scene_bbox, 34.0),
        stack_bbox_px=None,
        reference_stack_bbox_px=reference_bbox,
        changed_stack_bbox_px=changed_bbox,
        projection_cell_bbox_map={},
        option_panel_bbox_map={},
    )


def render_projection_count_scene(
    background,
    *,
    dataset: ProjectionCountDataset,
    style: PuzzleSceneStyle,
    render_params: VoxelRenderParams,
) -> RenderedVoxelScene:
    """Render a stack beside an empty target projection grid."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    stack_bbox = _draw_stack_centered(
        draw,
        dataset.stack,
        center=(render_params.canvas_width * 0.35, render_params.canvas_height * 0.55),
        cube_size=int(render_params.cube_size_px),
        palette=render_params.palette,
    )
    orientation_bbox = _draw_projection_orientation_cue(
        draw,
        stack_bbox=stack_bbox,
        style=style,
        render_params=render_params,
    )
    grid_bbox, cell_map = _draw_projection_panel(
        draw,
        projection=dataset.projection,
        panel_bbox=_panel_bbox(
            center=(
                render_params.canvas_width * 0.71,
                render_params.canvas_height * 0.52,
            ),
            width=max(
                190.0,
                dataset.projection.cols * render_params.projection_cell_size_px + 56.0,
            ),
            height=max(
                190.0,
                dataset.projection.rows * render_params.projection_cell_size_px + 74.0,
            ),
        ),
        style=style,
        render_params=render_params,
        label=f"{dataset.projection.direction.title()} view",
        draw_filled=False,
    )
    scene_bbox = _union_bboxes([stack_bbox, orientation_bbox, grid_bbox])
    return RenderedVoxelScene(
        image=image,
        scene_bbox_px=_pad_bbox(scene_bbox, 24.0),
        stack_bbox_px=stack_bbox,
        reference_stack_bbox_px=None,
        changed_stack_bbox_px=None,
        projection_cell_bbox_map=cell_map,
        option_panel_bbox_map={},
    )


def render_projection_match_scene(
    background,
    *,
    dataset: ProjectionMatchDataset,
    style: PuzzleSceneStyle,
    render_params: VoxelRenderParams,
) -> RenderedVoxelScene:
    """Render a stack with several candidate projection option panels."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    stack_bbox = _draw_stack_centered(
        draw,
        dataset.stack,
        center=(render_params.canvas_width * 0.25, render_params.canvas_height * 0.43),
        cube_size=int(render_params.cube_size_px),
        palette=render_params.palette,
    )
    orientation_bbox = _draw_projection_orientation_cue(
        draw,
        stack_bbox=stack_bbox,
        style=style,
        render_params=render_params,
    )
    option_bboxes: dict[str, BBox] = {}
    cell_map: dict[str, BBox] = {}
    for option, panel_bbox in zip(
        dataset.options,
        _option_panel_bboxes(render_params, len(dataset.options)),
        strict=True,
    ):
        option_bbox, option_cells = _draw_projection_panel(
            draw,
            projection=option.projection,
            panel_bbox=panel_bbox,
            style=style,
            render_params=render_params,
            label=str(option.label),
            draw_filled=True,
        )
        option_bboxes[str(option.label)] = option_bbox
        for key, bbox in option_cells.items():
            cell_map[f"{option.label}:{key}"] = bbox
    scene_bbox = _union_bboxes(
        [stack_bbox, orientation_bbox, *option_bboxes.values()]
    )
    return RenderedVoxelScene(
        image=image,
        scene_bbox_px=_pad_bbox(scene_bbox, 24.0),
        stack_bbox_px=stack_bbox,
        reference_stack_bbox_px=None,
        changed_stack_bbox_px=None,
        projection_cell_bbox_map=cell_map,
        option_panel_bbox_map=option_bboxes,
    )


def _draw_stack_centered(
    draw: ImageDraw.ImageDraw,
    stack: CubeStack,
    *,
    center: tuple[float, float],
    cube_size: int,
    palette: VoxelPalette,
) -> BBox:
    """Draw one isometric stack centered on an approximate bounding region."""

    rel_bbox = _stack_relative_bbox(stack, int(cube_size), palette=palette)
    rel_cx = 0.5 * (rel_bbox[0] + rel_bbox[2])
    rel_cy = 0.5 * (rel_bbox[1] + rel_bbox[3])
    origin = (float(center[0]) - rel_cx, float(center[1]) - rel_cy)
    return _draw_stack(draw, stack, origin=origin, cube_size=int(cube_size), palette=palette)


def _draw_stack(
    draw: ImageDraw.ImageDraw,
    stack: CubeStack,
    *,
    origin: tuple[float, float],
    cube_size: int,
    palette: VoxelPalette,
) -> BBox:
    """Draw cubes in back-to-front order and return the pixel bbox."""

    polygons: list[tuple[float, list[tuple[float, float]], tuple[int, int, int]]] = []
    for row, values in enumerate(stack.heights):
        for col, height in enumerate(values):
            for level in range(int(height)):
                depth = float(row + col + level)
                polygons.extend(
                    _cube_faces(row, col, level, origin, int(cube_size), depth, palette=palette)
                )
    points: list[tuple[float, float]] = []
    for _depth, vertices, fill in sorted(polygons, key=lambda item: item[0]):
        draw.polygon(vertices, fill=fill, outline=palette.cube_edge_rgb)
        points.extend(vertices)
    return _points_bbox(points)


def _cube_faces(
    row: int,
    col: int,
    level: int,
    origin: tuple[float, float],
    cube_size: int,
    depth: float,
    palette: VoxelPalette,
) -> list[tuple[float, list[tuple[float, float]], tuple[int, int, int]]]:
    """Return left, right, and top face polygons for one cube."""

    size = float(cube_size)
    half_w = 0.58 * size
    half_d = 0.32 * size
    z_step = 0.58 * size
    cx = float(origin[0]) + (float(col) - float(row)) * half_w
    cy = float(origin[1]) + (float(col) + float(row)) * half_d - float(level) * z_step
    top = [(cx, cy - half_d), (cx + half_w, cy), (cx, cy + half_d), (cx - half_w, cy)]
    left = [
        (cx - half_w, cy),
        (cx, cy + half_d),
        (cx, cy + half_d + z_step),
        (cx - half_w, cy + z_step),
    ]
    right = [
        (cx + half_w, cy),
        (cx, cy + half_d),
        (cx, cy + half_d + z_step),
        (cx + half_w, cy + z_step),
    ]
    return [
        (depth + 0.1, left, palette.cube_left_rgb),
        (depth + 0.2, right, palette.cube_right_rgb),
        (depth + 0.3, top, palette.cube_top_rgb),
    ]


def _stack_relative_bbox(stack: CubeStack, cube_size: int, *, palette: VoxelPalette) -> BBox:
    """Return the bbox a stack would occupy with origin at zero."""

    points: list[tuple[float, float]] = []
    for row, values in enumerate(stack.heights):
        for col, height in enumerate(values):
            for level in range(int(height)):
                for _depth, vertices, _fill in _cube_faces(
                    row,
                    col,
                    level,
                    (0.0, 0.0),
                    int(cube_size),
                    0.0,
                    palette=palette,
                ):
                    points.extend(vertices)
    return _points_bbox(points)


def _draw_projection_panel(
    draw: ImageDraw.ImageDraw,
    *,
    projection: ProjectionGrid,
    panel_bbox: BBox,
    style: PuzzleSceneStyle,
    render_params: VoxelRenderParams,
    label: str,
    draw_filled: bool,
) -> tuple[BBox, dict[str, BBox]]:
    """Draw one labeled projection grid inside a panel."""

    x0, y0, x1, y1 = [float(value) for value in panel_bbox]
    draw_puzzle_option_card(
        draw,
        bbox=(x0, y0, x1, y1),
        style=style,
        fill=style.option_fill_rgb,
    )
    font = load_font(int(render_params.label_font_size_px), bold=True)
    draw_text_centered(
        draw,
        text=str(label),
        center=((x0 + x1) * 0.5, y0 + 24.0),
        font=font,
        fill=style.text_rgb,
        stroke_fill=style.text_stroke_rgb,
    )
    cell = float(render_params.projection_cell_size_px)
    grid_w = cell * float(projection.cols)
    grid_h = cell * float(projection.rows)
    gx0 = x0 + (x1 - x0 - grid_w) * 0.5
    gy0 = y0 + 50.0 + max(0.0, y1 - y0 - 58.0 - grid_h) * 0.5
    filled = {tuple(cell_id) for cell_id in projection.filled_cells}
    cell_map: dict[str, BBox] = {}
    for row in range(int(projection.rows)):
        for col in range(int(projection.cols)):
            cx0 = gx0 + float(col) * cell
            cy0 = gy0 + float(row) * cell
            bbox = (cx0, cy0, cx0 + cell, cy0 + cell)
            fill = (
                render_params.palette.projection_fill_rgb
                if draw_filled and (row, col) in filled
                else render_params.palette.projection_empty_rgb
            )
            draw.rectangle(bbox, fill=fill, outline=style.grid_rgb, width=2)
            key = f"{row}_{col}"
            cell_map[key] = bbox
    return (x0, y0, x1, y1), cell_map


def _draw_projection_orientation_cue(
    draw: ImageDraw.ImageDraw,
    *,
    stack_bbox: BBox,
    style: PuzzleSceneStyle,
    render_params: VoxelRenderParams,
) -> BBox:
    """Draw a small front/right cue for projection-oriented voxel tasks."""

    x0, _y0, x1, y1 = [float(value) for value in stack_bbox]
    anchor_x = min(
        max(x0 + 28.0, 60.0),
        float(render_params.canvas_width) - 120.0,
    )
    anchor_y = min(
        y1 + 44.0,
        float(render_params.canvas_height) - 58.0,
    )
    anchor = (anchor_x, anchor_y)
    front_end = (anchor_x - 52.0, anchor_y + 28.0)
    right_end = (anchor_x + 58.0, anchor_y + 28.0)
    line_rgb = tuple(int(value) for value in style.text_rgb)
    shadow_rgb = tuple(int(value) for value in style.text_stroke_rgb)
    _draw_arrow(
        draw,
        start=anchor,
        end=front_end,
        fill=line_rgb,
        shadow_fill=shadow_rgb,
        width=3,
    )
    _draw_arrow(
        draw,
        start=anchor,
        end=right_end,
        fill=line_rgb,
        shadow_fill=shadow_rgb,
        width=3,
    )
    font = load_font(max(12, min(15, int(render_params.label_font_size_px) - 4)), bold=True)
    draw_text_centered(
        draw,
        text="Front",
        center=(front_end[0] - 12.0, front_end[1] + 17.0),
        font=font,
        fill=line_rgb,
        stroke_fill=shadow_rgb,
    )
    draw_text_centered(
        draw,
        text="Right",
        center=(right_end[0] + 16.0, right_end[1] + 17.0),
        font=font,
        fill=line_rgb,
        stroke_fill=shadow_rgb,
    )
    return _points_bbox(
        (
            anchor,
            front_end,
            right_end,
            (front_end[0] - 42.0, front_end[1] + 28.0),
            (right_end[0] + 42.0, right_end[1] + 28.0),
        )
    )


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: tuple[int, int, int],
    shadow_fill: tuple[int, int, int],
    width: int,
) -> None:
    """Draw a short arrow with a contrast stroke."""

    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    draw.line((sx, sy, ex, ey), fill=shadow_fill, width=int(width) + 3)
    draw.line((sx, sy, ex, ey), fill=fill, width=int(width))
    dx = ex - sx
    dy = ey - sy
    length = max(1.0, math.hypot(dx, dy))
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    head_len = 11.0
    head_half_w = 6.5
    p1 = (
        ex - ux * head_len + px * head_half_w,
        ey - uy * head_len + py * head_half_w,
    )
    p2 = (
        ex - ux * head_len - px * head_half_w,
        ey - uy * head_len - py * head_half_w,
    )
    draw.polygon((end, p1, p2), fill=shadow_fill)
    inner_p1 = (
        ex - ux * (head_len - 2.5) + px * (head_half_w - 2.0),
        ey - uy * (head_len - 2.5) + py * (head_half_w - 2.0),
    )
    inner_p2 = (
        ex - ux * (head_len - 2.5) - px * (head_half_w - 2.0),
        ey - uy * (head_len - 2.5) - py * (head_half_w - 2.0),
    )
    draw.polygon((end, inner_p1, inner_p2), fill=fill)


def _option_panel_bboxes(
    render_params: VoxelRenderParams,
    option_count: int,
) -> tuple[BBox, ...]:
    """Return a balanced option-panel grid for projection options."""

    count = int(option_count)
    cols = 2 if count <= 4 else 3
    rows = 2
    panel_w = 190.0
    panel_h = 178.0
    gap = float(render_params.panel_gap_px)
    total_w = cols * panel_w + (cols - 1) * gap
    start_x = render_params.canvas_width - total_w - 44.0
    start_y = render_params.canvas_height - (rows * panel_h + gap) - 42.0
    bboxes: list[BBox] = []
    for index in range(count):
        row = index // cols
        col = index % cols
        x0 = start_x + float(col) * (panel_w + gap)
        y0 = start_y + float(row) * (panel_h + gap)
        bboxes.append((x0, y0, x0 + panel_w, y0 + panel_h))
    return tuple(bboxes)


def _panel_bbox(
    *,
    center: tuple[float, float],
    width: float,
    height: float,
) -> BBox:
    """Return an axis-aligned bbox centered at a point."""

    cx, cy = float(center[0]), float(center[1])
    return (cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width, cy + 0.5 * height)


def _draw_scene_label(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: tuple[float, float],
    style: PuzzleSceneStyle,
    font_size: int,
) -> None:
    """Draw short scene-internal labels that identify compared panels."""

    font = load_font(int(font_size), bold=True)
    draw_text_centered(
        draw,
        text=str(text),
        center=center,
        font=font,
        fill=style.text_rgb,
        stroke_fill=style.text_stroke_rgb,
    )


def _points_bbox(points: Sequence[tuple[float, float]]) -> BBox:
    """Return a padded bbox around a non-empty point set."""

    if not points:
        raise ValueError("cannot compute bbox for empty point set")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _union_bboxes(bboxes: Sequence[BBox]) -> BBox:
    """Return the union bbox of non-empty bboxes."""

    if not bboxes:
        raise ValueError("cannot union empty bbox list")
    return (
        min(float(bbox[0]) for bbox in bboxes),
        min(float(bbox[1]) for bbox in bboxes),
        max(float(bbox[2]) for bbox in bboxes),
        max(float(bbox[3]) for bbox in bboxes),
    )


def _pad_bbox(bbox: BBox, padding: float) -> BBox:
    """Return a bbox expanded by a fixed pixel padding."""

    pad = float(padding)
    return (
        float(bbox[0]) - pad,
        float(bbox[1]) - pad,
        float(bbox[2]) + pad,
        float(bbox[3]) + pad,
    )
