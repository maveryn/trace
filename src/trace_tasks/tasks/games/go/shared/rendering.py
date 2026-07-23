"""Shared Go-board renderer for games-domain liberty-count tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

from .rules import BLACK, WHITE, Board, Coord, color_name, coord_to_point_id, coord_to_stone_id
from ...shared.layout import apply_games_layout_jitter_to_bbox
from ...shared.marking import SemanticMarkerStyle, draw_semantic_ellipse_marker, resolve_semantic_marker_style
from ...shared.scene_style import GamePanelSceneStyle, draw_panel_scene_chrome, game_panel_scene_style_metadata
from ...shared.style import GoTheme, build_games_go_theme


GO_MARKED_GROUP_RED_RGB: Tuple[int, int, int] = (220, 38, 38)


@dataclass(frozen=True)
class GoRenderParams:
    """Resolved render controls for one visible Go board scene."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    max_board_size_px: int
    board_padding_px: int
    board_corner_radius_px: int
    board_frame_width_px: int
    line_width_px: int
    point_radius_px: int
    stone_radius_fraction: float
    highlight_outline_width_px: int
    liberty_bbox_fraction: float
    layout_jitter_meta: Dict[str, Any] | None = None
    instance_seed: int = 0


@dataclass(frozen=True)
class RenderedGoScene:
    """Rendered Go scene plus trace-friendly geometry maps."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


def _stone_bbox(center: Tuple[float, float], *, radius: float) -> Tuple[float, float, float, float]:
    """Return one inscribed stone bbox around an intersection center."""

    center_x, center_y = float(center[0]), float(center[1])
    return (
        round(float(center_x - radius), 3),
        round(float(center_y - radius), 3),
        round(float(center_x + radius), 3),
        round(float(center_y + radius), 3),
    )


def _point_bbox(center: Tuple[float, float], *, side: float) -> Tuple[float, float, float, float]:
    """Return one square bbox centered on an empty board intersection."""

    half_side = 0.5 * float(side)
    center_x, center_y = float(center[0]), float(center[1])
    return (
        round(float(center_x - half_side), 3),
        round(float(center_y - half_side), 3),
        round(float(center_x + half_side), 3),
        round(float(center_y + half_side), 3),
    )


def _draw_stone(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Tuple[float, float, float, float],
    theme: GoTheme,
    color: int,
    marked: bool,
    highlight_outline_width_px: int,
    marker_style: SemanticMarkerStyle | None = None,
    marker_metadata: Dict[str, Any] | None = None,
) -> None:
    """Draw one black or white Go stone with optional marked-group outline."""

    if int(color) == int(BLACK):
        fill_rgb = theme.black_stone_fill_rgb
        outline_rgb = theme.black_stone_outline_rgb
        shine_rgb = theme.black_stone_shine_rgb
    else:
        fill_rgb = theme.white_stone_fill_rgb
        outline_rgb = theme.white_stone_outline_rgb
        shine_rgb = theme.white_stone_shine_rgb
    draw.ellipse(
        bbox_px,
        fill=tuple(int(value) for value in fill_rgb),
        outline=tuple(int(value) for value in outline_rgb),
        width=int(theme.stone_outline_width_px),
    )
    left, top, right, bottom = bbox_px
    shine_w = 0.34 * (right - left)
    shine_h = 0.22 * (bottom - top)
    draw.ellipse(
        [
            left + 0.18 * (right - left),
            top + 0.16 * (bottom - top),
            left + 0.18 * (right - left) + shine_w,
            top + 0.16 * (bottom - top) + shine_h,
        ],
        fill=tuple(int(value) for value in shine_rgb),
    )
    if bool(marked):
        highlight_inset = -float(max(3, int(highlight_outline_width_px)))
        if marker_style is None:
            marker_style = resolve_semantic_marker_style(
                instance_seed=0,
                namespace="games.go.marked_group.fallback",
                role="go_marked_stone",
                surface_rgbs=(tuple(int(value) for value in fill_rgb), tuple(int(value) for value in outline_rgb)),
                preferred_rgbs=(GO_MARKED_GROUP_RED_RGB,),
                candidate_rgbs=(GO_MARKED_GROUP_RED_RGB,),
            )
        draw_semantic_ellipse_marker(
            draw,
            (
                bbox_px[0] + highlight_inset,
                bbox_px[1] + highlight_inset,
                bbox_px[2] - highlight_inset,
                bbox_px[3] - highlight_inset,
            ),
            style=marker_style,
            width=int(highlight_outline_width_px),
            marker_kind="go_marked_stone_ring",
            extra_metadata=marker_metadata,
        )


def render_go_board_scene(
    *,
    board: Sequence[Sequence[int]],
    background: Image.Image,
    scene_variant: str,
    style_variant: str,
    marked_group_coords: Sequence[Coord],
    liberty_coords: Sequence[Coord],
    params: GoRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedGoScene:
    """Render one visible Go board with a highlighted connected group."""

    del scene_variant
    board_size = int(len(board))
    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme = build_games_go_theme(style_variant=str(style_variant))

    board_size_px = min(
        int(params.max_board_size_px),
        int(params.canvas_width) - (2 * int(params.panel_margin_px)),
        int(params.canvas_height) - (2 * int(params.panel_margin_px)),
    )
    board_left = int(0.5 * (int(params.canvas_width) - int(board_size_px)))
    board_top = int(0.5 * (int(params.canvas_height) - int(board_size_px)))
    board_bbox = (
        round(float(board_left), 3),
        round(float(board_top), 3),
        round(float(board_left + board_size_px), 3),
        round(float(board_top + board_size_px), 3),
    )
    board_bbox, _dx, _dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=board_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    board_left, board_top = float(board_bbox[0]), float(board_bbox[1])
    panel_bbox: Tuple[int, int, int, int] | None = None
    if panel_style is not None:
        panel_pad = max(16, int(round(float(params.panel_margin_px) * 0.62)))
        panel_bbox = (
            max(4, int(round(board_bbox[0])) - panel_pad),
            max(4, int(round(board_bbox[1])) - panel_pad),
            min(int(params.canvas_width) - 4, int(round(board_bbox[2])) + panel_pad),
            min(int(params.canvas_height) - 4, int(round(board_bbox[3])) + panel_pad),
        )
        draw_panel_scene_chrome(
            draw,
            bbox=panel_bbox,
            style=panel_style,
            radius=max(18, int(params.board_corner_radius_px) + 10),
            border_width=max(2, int(round(float(params.board_frame_width_px) * 0.45))),
        )
    draw.rounded_rectangle(
        board_bbox,
        radius=int(params.board_corner_radius_px),
        fill=tuple(int(value) for value in theme.board_frame_rgb),
    )
    inner_inset = float(params.board_frame_width_px)
    inner_bbox = (
        round(float(board_bbox[0] + inner_inset), 3),
        round(float(board_bbox[1] + inner_inset), 3),
        round(float(board_bbox[2] - inner_inset), 3),
        round(float(board_bbox[3] - inner_inset), 3),
    )
    draw.rounded_rectangle(
        inner_bbox,
        radius=max(8, int(params.board_corner_radius_px) - int(params.board_frame_width_px)),
        fill=tuple(int(value) for value in theme.board_fill_rgb),
    )

    grid_left = float(inner_bbox[0] + int(params.board_padding_px))
    grid_top = float(inner_bbox[1] + int(params.board_padding_px))
    grid_right = float(inner_bbox[2] - int(params.board_padding_px))
    grid_bottom = float(inner_bbox[3] - int(params.board_padding_px))
    step = float((grid_right - grid_left) / max(1, int(board_size) - 1))
    centers_px: Dict[str, Tuple[float, float]] = {}
    point_bboxes_px: Dict[str, Tuple[float, float, float, float]] = {}
    stone_bboxes_px: Dict[str, Tuple[float, float, float, float]] = {}
    liberty_set = {(int(coord[0]), int(coord[1])) for coord in liberty_coords}
    marked_group = {(int(coord[0]), int(coord[1])) for coord in marked_group_coords}
    marked_group_colors = {
        int(board[row][col])
        for row, col in marked_group
        if 0 <= int(row) < int(board_size)
        and 0 <= int(col) < int(board_size)
        and int(board[row][col]) != 0
    }
    marked_group_color = next(iter(marked_group_colors), 0)
    marked_stone_surface_rgbs: Tuple[Tuple[int, int, int], ...]
    if int(marked_group_color) == int(BLACK):
        marked_stone_surface_rgbs = (
            tuple(int(value) for value in theme.black_stone_fill_rgb),
            tuple(int(value) for value in theme.black_stone_outline_rgb),
        )
    elif int(marked_group_color) == int(WHITE):
        marked_stone_surface_rgbs = (
            tuple(int(value) for value in theme.white_stone_fill_rgb),
            tuple(int(value) for value in theme.white_stone_outline_rgb),
        )
    else:
        marked_stone_surface_rgbs = ()
    marked_group_marker_style = (
        resolve_semantic_marker_style(
            instance_seed=int(params.instance_seed),
            namespace="games.go.marked_group",
            role="go_marked_stone_group",
            surface_rgbs=marked_stone_surface_rgbs or (tuple(int(value) for value in theme.board_fill_rgb),),
            preferred_rgbs=(GO_MARKED_GROUP_RED_RGB,),
            candidate_rgbs=(GO_MARKED_GROUP_RED_RGB,),
        )
        if marked_group
        else None
    )

    for index in range(board_size):
        offset = float(index) * float(step)
        x = float(grid_left + offset)
        y = float(grid_top + offset)
        draw.line(
            [(x, grid_top), (x, grid_bottom)],
            fill=tuple(int(value) for value in theme.grid_line_rgb),
            width=int(params.line_width_px),
        )
        draw.line(
            [(grid_left, y), (grid_right, y)],
            fill=tuple(int(value) for value in theme.grid_line_rgb),
            width=int(params.line_width_px),
        )

    scene_entities: List[Dict[str, Any]] = []
    point_side = float(step) * float(params.liberty_bbox_fraction)
    stone_radius = float(step) * float(params.stone_radius_fraction)
    for row in range(board_size):
        for col in range(board_size):
            center = (float(grid_left + (col * step)), float(grid_top + (row * step)))
            point_id = coord_to_point_id((int(row), int(col)))
            centers_px[point_id] = (round(float(center[0]), 3), round(float(center[1]), 3))
            point_bbox = _point_bbox(center, side=point_side)
            point_bboxes_px[point_id] = point_bbox
            draw.ellipse(
                [
                    center[0] - float(params.point_radius_px),
                    center[1] - float(params.point_radius_px),
                    center[0] + float(params.point_radius_px),
                    center[1] + float(params.point_radius_px),
                ],
                fill=tuple(int(value) for value in theme.point_rgb),
            )
            occupant = int(board[row][col])
            stone_id: str | None = None
            stone_bbox: Tuple[float, float, float, float] | None = None
            if int(occupant) != 0:
                stone_id = coord_to_stone_id((int(row), int(col)))
                stone_bbox = _stone_bbox(center, radius=stone_radius)
                stone_bboxes_px[stone_id] = stone_bbox
                _draw_stone(
                    draw,
                    bbox_px=stone_bbox,
                    theme=theme,
                    color=int(occupant),
                    marked=(int(row), int(col)) in marked_group,
                    highlight_outline_width_px=int(params.highlight_outline_width_px),
                    marker_style=marked_group_marker_style if (int(row), int(col)) in marked_group else None,
                    marker_metadata={"point_id": str(point_id), "stone_id": str(stone_id)},
                )
            scene_entities.append(
                {
                    "entity_id": str(point_id),
                    "entity_type": "go_intersection",
                    "row": int(row),
                    "col": int(col),
                    "occupant": "empty" if int(occupant) == 0 else str(color_name(int(occupant)).lower()),
                    "is_marked_group": bool((int(row), int(col)) in marked_group),
                    "is_liberty": bool((int(row), int(col)) in liberty_set),
                    "bbox_px": list(point_bbox),
                    "center_px": [round(float(center[0]), 3), round(float(center[1]), 3)],
                    "stone_id": None if stone_id is None else str(stone_id),
                    "stone_bbox_px": None if stone_bbox is None else list(stone_bbox),
                }
            )

    render_map = {
        "board_bbox_px": list(board_bbox),
        "inner_board_bbox_px": list(inner_bbox),
        "point_centers_px": {str(key): [float(value[0]), float(value[1])] for key, value in centers_px.items()},
        "point_bboxes_px": {str(key): list(value) for key, value in point_bboxes_px.items()},
        "stone_bboxes_px": {str(key): list(value) for key, value in stone_bboxes_px.items()},
        "marked_group_marker_style": None if marked_group_marker_style is None else marked_group_marker_style.metadata(),
        "layout_jitter": dict(layout_jitter),
        "scene_panel_bbox_px": None if panel_bbox is None else [int(value) for value in panel_bbox],
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
    }
    return RenderedGoScene(
        image=image,
        scene_entities=tuple(scene_entities),
        render_map=render_map,
    )


__all__ = [
    "GoRenderParams",
    "RenderedGoScene",
    "render_go_board_scene",
]
