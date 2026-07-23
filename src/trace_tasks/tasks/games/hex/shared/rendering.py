"""Shared Hex-board renderer for games-domain connection tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.text_rendering import fit_font_to_box
from ...shared.text import draw_game_text_traced as draw_text_traced
from .rules import BLUE, EMPTY, RED, Board, Coord, color_name, coord_to_cell_id
from ...shared.layout import apply_games_layout_jitter_to_bbox
from ...shared.scene_style import GamePanelSceneStyle, draw_panel_scene_chrome, game_panel_scene_style_metadata
from ...shared.style import HexTheme, build_games_hex_theme


@dataclass(frozen=True)
class HexRenderParams:
    """Resolved render controls for one Hex board scene."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    max_board_width_px: int
    max_board_height_px: int
    hex_border_width_px: int
    stone_radius_fraction: float
    candidate_label_font_size_px: int
    side_band_width_px: int
    layout_jitter_meta: Dict[str, Any] | None = None
    font_family: str = ""


@dataclass(frozen=True)
class RenderedHexScene:
    """Rendered Hex scene plus trace-friendly geometry maps."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


def _hex_points(cx: float, cy: float, radius: float) -> Tuple[Tuple[float, float], ...]:
    """Return pointy-top hexagon points centered at `(cx, cy)`."""

    points: list[tuple[float, float]] = []
    for index in range(6):
        angle = math.radians(30.0 + (60.0 * index))
        points.append((float(cx + (radius * math.cos(angle))), float(cy + (radius * math.sin(angle)))))
    return tuple(points)


def _bbox_for_points(points: Sequence[Tuple[float, float]], *, pad: float = 0.0) -> Tuple[float, float, float, float]:
    """Return a rounded bbox for polygon points."""

    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (
        round(float(min(xs) - float(pad)), 3),
        round(float(min(ys) - float(pad)), 3),
        round(float(max(xs) + float(pad)), 3),
        round(float(max(ys) + float(pad)), 3),
    )


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Tuple[float, float, float, float],
    text: str,
    fill: Tuple[int, int, int],
    max_size_px: int,
    bold: bool = False,
    font_family: str = "",
) -> None:
    """Draw centered text inside one bbox."""

    left, top, right, bottom = bbox_px
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=max(1.0, float(right - left)),
        max_height=max(1.0, float(bottom - top)),
        bold=bool(bold),
        min_size_px=9,
        max_size_px=int(max_size_px),
        font_family=str(font_family) or None,
        fill_ratio=0.80,
    )
    text_bbox = draw.textbbox((0, 0), str(text), font=font)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    x = float(left + (0.5 * (float(right - left) - text_w)) - float(text_bbox[0]))
    y = float(top + (0.5 * (float(bottom - top) - text_h)) - float(text_bbox[1]))
    draw_text_traced(draw,(x, y), str(text), fill=tuple(int(v) for v in fill), font=font, role="readout", required=False)


def _draw_stone(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    radius: float,
    value: int,
    theme: HexTheme,
) -> Tuple[float, float, float, float]:
    """Draw one red or blue Hex stone and return its bbox."""

    cx, cy = float(center[0]), float(center[1])
    bbox = (
        round(float(cx - radius), 3),
        round(float(cy - radius), 3),
        round(float(cx + radius), 3),
        round(float(cy + radius), 3),
    )
    if int(value) == int(RED):
        fill = theme.red_stone_fill_rgb
        outline = theme.red_stone_outline_rgb
        shine = theme.red_stone_shine_rgb
    else:
        fill = theme.blue_stone_fill_rgb
        outline = theme.blue_stone_outline_rgb
        shine = theme.blue_stone_shine_rgb
    draw.ellipse(
        bbox,
        fill=tuple(int(v) for v in fill),
        outline=tuple(int(v) for v in outline),
        width=max(2, int(round(0.07 * float(radius)))),
    )
    shine_radius = float(radius) * 0.24
    draw.ellipse(
        (
            float(cx - 0.36 * radius),
            float(cy - 0.42 * radius),
            float(cx - 0.36 * radius + shine_radius),
            float(cy - 0.42 * radius + shine_radius),
        ),
        fill=tuple(int(v) for v in shine),
    )
    return bbox


def render_hex_board_scene(
    *,
    board: Sequence[Sequence[int]],
    background: Image.Image,
    scene_variant: str,
    style_variant: str,
    player_color: str,
    candidate_labels_by_coord: Mapping[Coord, str],
    params: HexRenderParams,
    reference_coords: Sequence[Coord] | None = None,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedHexScene:
    """Render one Hex board with colored goal sides and optional candidate labels."""

    del scene_variant
    size = int(len(board))
    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme = build_games_hex_theme(style_variant=str(style_variant))

    # Pointy-top rhombus geometry. Width includes the row skew.
    max_w = min(int(params.max_board_width_px), int(params.canvas_width) - (2 * int(params.panel_margin_px)))
    max_h = min(int(params.max_board_height_px), int(params.canvas_height) - (2 * int(params.panel_margin_px)))
    radius_by_w = float(max_w) / max(1.0, math.sqrt(3.0) * (1.5 * float(size) + 0.5))
    radius_by_h = float(max_h) / max(1.0, 1.5 * float(size) + 0.5)
    radius = float(min(radius_by_w, radius_by_h))
    step_x = float(math.sqrt(3.0) * radius)
    step_y = float(1.5 * radius)
    board_w = float(step_x * (float(size) + (0.5 * float(size - 1))) + (math.sqrt(3.0) * radius))
    board_h = float(step_y * float(size - 1) + (2.0 * radius))
    left = float(0.5 * (int(params.canvas_width) - board_w))
    top = float(0.5 * (int(params.canvas_height) - board_h))
    board_bbox = (
        round(left, 3),
        round(top, 3),
        round(left + board_w, 3),
        round(top + board_h, 3),
    )
    board_bbox, dx, dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=board_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    left = float(board_bbox[0])
    top = float(board_bbox[1])
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
            radius=26,
            border_width=max(2, int(round(float(params.hex_border_width_px) * 0.8))),
        )

    centers_px: Dict[str, Tuple[float, float]] = {}
    cell_bboxes_px: Dict[str, List[float]] = {}
    stone_bboxes_px: Dict[str, List[float]] = {}
    scene_entities: List[Dict[str, Any]] = []
    polygons_by_coord: Dict[Coord, Tuple[Tuple[float, float], ...]] = {}
    center_by_coord: Dict[Coord, Tuple[float, float]] = {}

    for row in range(size):
        for col in range(size):
            cx = float(left + (math.sqrt(3.0) * radius) + (col * step_x) + (row * step_x * 0.5))
            cy = float(top + radius + (row * step_y))
            coord = (int(row), int(col))
            center_by_coord[coord] = (float(cx), float(cy))
            points = _hex_points(cx, cy, radius * 0.96)
            polygons_by_coord[coord] = points

    candidate_label_lookup = {
        (int(coord[0]), int(coord[1])): str(label)
        for coord, label in candidate_labels_by_coord.items()
    }
    reference_coord_lookup = {
        (int(coord[0]), int(coord[1]))
        for coord in tuple(reference_coords or tuple())
    }
    player_value = RED if str(player_color).lower() == "red" else BLUE
    for row in range(size):
        for col in range(size):
            coord = (int(row), int(col))
            cell_id = coord_to_cell_id(coord)
            points = polygons_by_coord[coord]
            value = int(board[row][col])
            fill = theme.cell_alt_fill_rgb if (row + col) % 2 else theme.cell_fill_rgb
            outline = theme.cell_outline_rgb
            if coord in reference_coord_lookup:
                fill = theme.reference_cell_fill_rgb
                outline = theme.reference_cell_outline_rgb
            draw.polygon(points, fill=tuple(int(v) for v in fill), outline=tuple(int(v) for v in outline))
            if int(value) == int(EMPTY) and coord in candidate_label_lookup:
                cx, cy = center_by_coord[coord]
                badge_r = float(radius) * 0.45
                badge_bbox = (
                    float(cx - badge_r),
                    float(cy - badge_r),
                    float(cx + badge_r),
                    float(cy + badge_r),
                )
                draw.ellipse(
                    badge_bbox,
                    fill=tuple(int(v) for v in theme.candidate_badge_fill_rgb),
                    outline=tuple(int(v) for v in theme.candidate_badge_outline_rgb),
                    width=max(2, int(round(0.035 * radius))),
                )
                _draw_centered_text(
                    draw,
                    bbox_px=badge_bbox,
                    text=str(candidate_label_lookup[coord]),
                    fill=tuple(int(v) for v in theme.candidate_badge_text_rgb),
                    max_size_px=int(params.candidate_label_font_size_px),
                    bold=True,
                    font_family=str(params.font_family),
                )
            stone_bbox = None
            if int(value) in {int(RED), int(BLUE)}:
                stone_bbox = _draw_stone(
                    draw,
                    center=center_by_coord[coord],
                    radius=float(radius) * float(params.stone_radius_fraction),
                    value=int(value),
                    theme=theme,
                )
                stone_bboxes_px[cell_id] = [float(v) for v in stone_bbox]
            cell_bbox = _bbox_for_points(points, pad=1.0)
            centers_px[cell_id] = (round(float(center_by_coord[coord][0]), 3), round(float(center_by_coord[coord][1]), 3))
            cell_bboxes_px[cell_id] = [float(v) for v in cell_bbox]
            scene_entities.append(
                {
                    "entity_id": str(cell_id),
                    "entity_type": "hex_cell",
                    "row": int(row),
                    "col": int(col),
                    "state": "empty" if int(value) == EMPTY else str(color_name(value)).lower(),
                    "is_candidate": bool(coord in candidate_label_lookup),
                    "candidate_label": candidate_label_lookup.get(coord),
                    "is_reference": bool(coord in reference_coord_lookup),
                    "is_query_player_stone": bool(int(value) == int(player_value)),
                    "bbox_px": [float(v) for v in cell_bbox],
                    "center_px": [float(v) for v in centers_px[cell_id]],
                    "stone_bbox_px": None if stone_bbox is None else [float(v) for v in stone_bbox],
                }
            )

    # Colored goal sides sit on the outside hex edges. They are drawn after cells
    # so they remain visible, but they never cut through interior cells.
    band_width = max(3, int(params.side_band_width_px))
    outline_width = max(2, int(params.hex_border_width_px))
    for row in range(size):
        left_points = polygons_by_coord[(row, 0)]
        right_points = polygons_by_coord[(row, size - 1)]
        draw.line(
            [left_points[2], left_points[3]],
            fill=tuple(int(v) for v in theme.red_goal_rgb),
            width=band_width,
        )
        draw.line(
            [right_points[5], right_points[0]],
            fill=tuple(int(v) for v in theme.red_goal_rgb),
            width=band_width,
        )
    for col in range(size):
        top_points = polygons_by_coord[(0, col)]
        bottom_points = polygons_by_coord[(size - 1, col)]
        draw.line(
            [top_points[1], top_points[2]],
            fill=tuple(int(v) for v in theme.blue_goal_rgb),
            width=band_width,
        )
        draw.line(
            [top_points[0], top_points[1]],
            fill=tuple(int(v) for v in theme.blue_goal_rgb),
            width=band_width,
        )
        draw.line(
            [bottom_points[3], bottom_points[4]],
            fill=tuple(int(v) for v in theme.blue_goal_rgb),
            width=band_width,
        )
        draw.line(
            [bottom_points[4], bottom_points[5]],
            fill=tuple(int(v) for v in theme.blue_goal_rgb),
            width=band_width,
        )
    for row in range(size):
        for col in range(size):
            points = polygons_by_coord[(row, col)]
            draw.line(
                [*points, points[0]],
                fill=tuple(int(v) for v in theme.board_outline_rgb),
                width=outline_width,
            )

    render_map = {
        "board_bbox_px": [float(v) for v in board_bbox],
        "cell_bboxes_px": dict(cell_bboxes_px),
        "cell_centers_px": {str(key): [float(v) for v in value] for key, value in centers_px.items()},
        "stone_bboxes_px": dict(stone_bboxes_px),
        "candidate_labels_by_cell_id": {
            coord_to_cell_id(coord): str(label)
            for coord, label in candidate_label_lookup.items()
        },
        "reference_cell_ids": [coord_to_cell_id(coord) for coord in sorted(reference_coord_lookup)],
        "player_color": str(player_color),
        "layout_jitter": {**dict(layout_jitter), "board_dx_px": float(dx), "board_dy_px": float(dy)},
        "scene_panel_bbox_px": None if panel_bbox is None else [int(value) for value in panel_bbox],
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
        "font_family": str(params.font_family),
    }
    return RenderedHexScene(
        image=image,
        scene_entities=tuple(scene_entities),
        render_map=render_map,
    )


__all__ = [
    "HexRenderParams",
    "RenderedHexScene",
    "render_hex_board_scene",
]
