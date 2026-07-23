"""Rendering helpers for counterfactual-board game scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.games.shared.scene_style import (
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from trace_tasks.tasks.shared.visual_style.panel import (
    draw_panel_chrome_by_mode,
    resolve_panel_chrome_mode,
)
from trace_tasks.tasks.shared.bbox_projection import round_bbox
from trace_tasks.tasks.shared.color_distance import coerce_rgb
from trace_tasks.tasks.shared.config_defaults import group_default

from .layout import resolve_board_layout
from .state import (
    CELL_BOARD_KIND,
    CHESS_CHECKERS_STYLE,
    LINE_BOARD_KIND,
    RenderedCounterfactualBoard,
    SUDOKU_STYLE,
    XIANGQI_STYLE,
)


def _scale_bbox(bbox: Sequence[float], scale: int) -> list[float]:
    return [float(value) * int(scale) for value in bbox]


def _rgb(value: object, fallback: Sequence[int]) -> tuple[int, int, int]:
    return coerce_rgb(value, fallback)


def _draw_chess_checkers(
    draw: ImageDraw.ImageDraw,
    *,
    rows: int,
    cols: int,
    board_bbox: Sequence[float],
    scale: int,
    light_rgb: tuple[int, int, int],
    dark_rgb: tuple[int, int, int],
    outline_rgb: tuple[int, int, int],
) -> None:
    """Draw a checkered cell board without task-specific count overlays."""

    x0, y0, x1, y1 = [float(value) for value in board_bbox]
    cell_w = (x1 - x0) / float(max(1, int(cols)))
    cell_h = (y1 - y0) / float(max(1, int(rows)))
    for row in range(int(rows)):
        for col in range(int(cols)):
            fill = light_rgb if (row + col) % 2 == 0 else dark_rgb
            bbox = (
                x0 + float(col) * cell_w,
                y0 + float(row) * cell_h,
                x0 + float(col + 1) * cell_w,
                y0 + float(row + 1) * cell_h,
            )
            draw.rectangle(_scale_bbox(bbox, scale), fill=fill + (255,))
    draw.rectangle(
        _scale_bbox(board_bbox, scale),
        outline=outline_rgb + (245,),
        width=max(2, 3 * int(scale)),
    )


def _draw_sudoku(
    draw: ImageDraw.ImageDraw,
    *,
    rows: int,
    cols: int,
    board_bbox: Sequence[float],
    scale: int,
    fill_rgb: tuple[int, int, int],
    line_rgb: tuple[int, int, int],
) -> None:
    """Draw a Sudoku-like grid with thicker subgrid separators."""

    x0, y0, x1, y1 = [float(value) for value in board_bbox]
    draw.rectangle(_scale_bbox(board_bbox, scale), fill=fill_rgb + (255,))
    cell_w = (x1 - x0) / float(max(1, int(cols)))
    cell_h = (y1 - y0) / float(max(1, int(rows)))
    row_block = max(1, int(round(float(rows) / 3.0)))
    col_block = max(1, int(round(float(cols) / 3.0)))
    for row in range(int(rows) + 1):
        width = 3 if row in (0, int(rows)) or row % row_block == 0 else 1
        y = y0 + float(row) * cell_h
        draw.line(
            _scale_bbox((x0, y, x1, y), scale),
            fill=line_rgb + (245,),
            width=max(1, int(width) * int(scale)),
        )
    for col in range(int(cols) + 1):
        width = 3 if col in (0, int(cols)) or col % col_block == 0 else 1
        x = x0 + float(col) * cell_w
        draw.line(
            _scale_bbox((x, y0, x, y1), scale),
            fill=line_rgb + (245,),
            width=max(1, int(width) * int(scale)),
        )


def _draw_xiangqi(
    draw: ImageDraw.ImageDraw,
    *,
    rows: int,
    cols: int,
    board_bbox: Sequence[float],
    scale: int,
    board_rgb: tuple[int, int, int],
    line_rgb: tuple[int, int, int],
) -> None:
    """Draw a Xiangqi-like line board with river and palace diagonals."""

    x0, y0, x1, y1 = [float(value) for value in board_bbox]
    pad = max(20.0, (x1 - x0) / max(12.0, float(cols)))
    draw.rounded_rectangle(
        _scale_bbox((x0 - pad, y0 - pad, x1 + pad, y1 + pad), scale),
        radius=max(10, int(round(pad * 0.45))) * int(scale),
        fill=board_rgb + (255,),
        outline=line_rgb + (225,),
        width=max(2, 2 * int(scale)),
    )
    dx = (x1 - x0) / float(max(1, int(cols) - 1))
    dy = (y1 - y0) / float(max(1, int(rows) - 1))
    line_width = max(2, 2 * int(scale))
    river_after = max(2, int(rows) // 2 - 1)
    river_before = min(int(rows) - 2, river_after + 1)
    for row in range(int(rows)):
        y = y0 + float(row) * dy
        draw.line(_scale_bbox((x0, y, x1, y), scale), fill=line_rgb + (245,), width=line_width)
    for col in range(int(cols)):
        x = x0 + float(col) * dx
        if col in (0, int(cols) - 1):
            draw.line(_scale_bbox((x, y0, x, y1), scale), fill=line_rgb + (245,), width=line_width)
        else:
            draw.line(
                _scale_bbox((x, y0, x, y0 + float(river_after) * dy), scale),
                fill=line_rgb + (245,),
                width=line_width,
            )
            draw.line(
                _scale_bbox((x, y0 + float(river_before) * dy, x, y1), scale),
                fill=line_rgb + (245,),
                width=line_width,
            )
    if int(rows) >= 6 and int(cols) >= 5:
        mid = int(cols) // 2
        for c0, r0, c1, r1 in (
            (mid - 1, 0, mid + 1, 2),
            (mid + 1, 0, mid - 1, 2),
            (mid - 1, int(rows) - 3, mid + 1, int(rows) - 1),
            (mid + 1, int(rows) - 3, mid - 1, int(rows) - 1),
        ):
            draw.line(
                _scale_bbox((x0 + c0 * dx, y0 + r0 * dy, x0 + c1 * dx, y0 + r1 * dy), scale),
                fill=line_rgb + (245,),
                width=line_width,
            )


def render_counterfactual_board(
    *,
    style: str,
    rows: int,
    cols: int,
    board_kind: str,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    visual_defaults: Mapping[str, Any],
) -> RenderedCounterfactualBoard:
    """Render one board style and expose scene-level projection metadata."""

    layout_rng = spawn_rng(int(instance_seed), "games.counterfactual_board.layout")
    layout = resolve_board_layout(
        rng=layout_rng,
        rows=int(rows),
        cols=int(cols),
        board_kind=str(board_kind),
        params=params,
        rendering_defaults=rendering_defaults,
    )
    scale = int(params.get("scene_supersample_scale", group_default(rendering_defaults, "scene_supersample_scale", 2)))
    scene_style, scene_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace="games.counterfactual_board.scene_style",
    )
    chrome_mode, chrome_meta = resolve_panel_chrome_mode(
        instance_seed=int(instance_seed),
        namespace="games.counterfactual_board.chrome",
    )
    image, background_meta = make_panel_scene_background(
        canvas_width=int(layout.canvas_width_px) * int(scale),
        canvas_height=int(layout.canvas_height_px) * int(scale),
        style=scene_style,
    )
    draw = ImageDraw.Draw(image, "RGBA")
    x0, y0, x1, y1 = [float(value) for value in layout.board_bbox_px]
    panel_pad = max(8, int(round(float(layout.unit_size_px) * 0.22)))
    panel_bbox = (
        int(round((x0 - panel_pad) * scale)),
        int(round((y0 - panel_pad) * scale)),
        int(round((x1 + panel_pad) * scale)),
        int(round((y1 + panel_pad) * scale)),
    )
    draw_panel_chrome_by_mode(
        draw,
        bbox=panel_bbox,
        style=scene_style,
        radius=max(6, int(round(float(layout.unit_size_px) * 0.18))) * int(scale),
        border_width=max(1, int(round(float(layout.unit_size_px) * 0.05))) * int(scale),
        mode=str(chrome_mode),
    )
    if str(style) == CHESS_CHECKERS_STYLE:
        _draw_chess_checkers(
            draw,
            rows=int(rows),
            cols=int(cols),
            board_bbox=layout.board_bbox_px,
            scale=int(scale),
            light_rgb=_rgb(group_default(rendering_defaults, "chess_light_rgb", [238, 229, 206]), [238, 229, 206]),
            dark_rgb=_rgb(group_default(rendering_defaults, "chess_dark_rgb", [92, 124, 80]), [92, 124, 80]),
            outline_rgb=_rgb(group_default(rendering_defaults, "outline_rgb", [33, 37, 45]), [33, 37, 45]),
        )
    elif str(style) == SUDOKU_STYLE:
        _draw_sudoku(
            draw,
            rows=int(rows),
            cols=int(cols),
            board_bbox=layout.board_bbox_px,
            scale=int(scale),
            fill_rgb=_rgb(group_default(rendering_defaults, "sudoku_fill_rgb", [255, 255, 252]), [255, 255, 252]),
            line_rgb=_rgb(group_default(rendering_defaults, "sudoku_line_rgb", [38, 42, 50]), [38, 42, 50]),
        )
    elif str(style) == XIANGQI_STYLE:
        _draw_xiangqi(
            draw,
            rows=int(rows),
            cols=int(cols),
            board_bbox=layout.board_bbox_px,
            scale=int(scale),
            board_rgb=_rgb(group_default(rendering_defaults, "xiangqi_board_rgb", [241, 205, 142]), [241, 205, 142]),
            line_rgb=_rgb(group_default(rendering_defaults, "xiangqi_line_rgb", [72, 45, 24]), [72, 45, 24]),
        )
    else:
        raise ValueError(f"unsupported board style: {style!r}")
    image = image.resize((int(layout.canvas_width_px), int(layout.canvas_height_px)))
    image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=visual_defaults,
    )
    board_bbox = round_bbox(layout.board_bbox_px)
    entities = [
        {
            "entity_id": "board_0",
            "entity_type": "counterfactual_board",
            "board_style": str(style),
            "board_kind": str(board_kind),
            "visible_rows": int(rows),
            "visible_columns": int(cols),
            "bbox": list(board_bbox),
        }
    ]
    return RenderedCounterfactualBoard(
        image=image,
        layout=layout,
        entities=entities,
        render_meta={
            "scene_id": "counterfactual_board",
            "coord_space": "pixel",
            "canvas_width": int(layout.canvas_width_px),
            "canvas_height": int(layout.canvas_height_px),
            "scene_supersample_scale": int(scale),
            "scene_style": dict(scene_style_meta),
            "background_style": dict(background_meta),
            "panel_chrome": dict(chrome_meta),
            "board_bbox_px": list(board_bbox),
            "board_style": str(style),
            "board_kind": str(board_kind),
            "layout": dict(layout.placement_meta),
            "post_image_noise": dict(post_noise_meta),
        },
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


__all__ = ["render_counterfactual_board"]
