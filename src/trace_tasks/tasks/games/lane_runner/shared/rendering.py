"""Shared lane-runner renderer for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.text_rendering import load_font
from .rules import cell_entity_id, path_option_entity_id, runner_entity_id
from .state import LaneRunnerCoin, LaneRunnerHazard, LaneRunnerPathOption
from ...shared.layout import apply_games_layout_jitter_to_bbox
from ...shared.marking import draw_semantic_ellipse_marker, resolve_semantic_marker_style
from ...shared.scene_style import (
    GamePanelSceneStyle,
    draw_panel_scene_chrome,
    game_panel_scene_style_metadata,
)
from ...shared.text import draw_centered_game_text


Color = Tuple[int, int, int]
BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


@dataclass(frozen=True)
class LaneRunnerRenderParams:
    """Resolved render controls for one lane-runner scene."""

    canvas_width: int
    canvas_height: int
    row_count: int
    lane_count: int
    cell_size_px: int
    cell_gap_px: int
    panel_margin_px: int
    start_band_height_px: int
    finish_band_height_px: int
    coin_radius_px: int
    runner_radius_px: int
    grid_line_width_px: int
    label_font_size_px: int
    hazard_radius_px: int = 18
    path_line_width_px: int = 7
    path_label_font_size_px: int = 18
    option_card_cell_size_px: int = 24
    option_card_gap_px: int = 4
    option_card_margin_px: int = 10
    option_card_area_gap_px: int = 28
    font_family: str = ""
    instance_seed: int = 0
    layout_jitter_meta: Dict[str, Any] | None = None
    unit_size_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class LaneRunnerTheme:
    """Scene-local lane-runner visual style."""

    track_fill_rgb: Color
    track_outline_rgb: Color
    cell_a_rgb: Color
    cell_b_rgb: Color
    grid_rgb: Color
    start_rgb: Color
    finish_rgb: Color
    runner_rgb: Color
    runner_outline_rgb: Color
    coin_rgb: Color
    coin_outline_rgb: Color
    label_rgb: Color
    label_stroke_rgb: Color
    pattern: str


@dataclass(frozen=True)
class RenderedLaneRunnerScene:
    """Rendered lane-runner image plus trace-friendly geometry."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


def build_games_lane_runner_theme(*, style_variant: str) -> LaneRunnerTheme:
    """Return one complete lane-runner board theme with readable contrast."""

    style = str(style_variant)
    if style == "city_road":
        return LaneRunnerTheme(
            track_fill_rgb=(73, 78, 86),
            track_outline_rgb=(37, 42, 50),
            cell_a_rgb=(96, 101, 110),
            cell_b_rgb=(86, 91, 100),
            grid_rgb=(229, 230, 221),
            start_rgb=(41, 127, 84),
            finish_rgb=(235, 237, 232),
            runner_rgb=(70, 150, 226),
            runner_outline_rgb=(16, 41, 78),
            coin_rgb=(242, 190, 42),
            coin_outline_rgb=(92, 63, 18),
            label_rgb=(251, 252, 246),
            label_stroke_rgb=(35, 38, 42),
            pattern="road",
        )
    if style == "forest_path":
        return LaneRunnerTheme(
            track_fill_rgb=(109, 142, 86),
            track_outline_rgb=(50, 84, 48),
            cell_a_rgb=(133, 166, 96),
            cell_b_rgb=(121, 153, 86),
            grid_rgb=(68, 99, 58),
            start_rgb=(81, 116, 72),
            finish_rgb=(235, 225, 184),
            runner_rgb=(226, 76, 58),
            runner_outline_rgb=(92, 35, 28),
            coin_rgb=(248, 203, 49),
            coin_outline_rgb=(98, 67, 21),
            label_rgb=(251, 247, 227),
            label_stroke_rgb=(49, 70, 42),
            pattern="grass",
        )
    if style == "neon_track":
        return LaneRunnerTheme(
            track_fill_rgb=(23, 28, 50),
            track_outline_rgb=(98, 230, 236),
            cell_a_rgb=(36, 44, 79),
            cell_b_rgb=(47, 39, 88),
            grid_rgb=(132, 116, 255),
            start_rgb=(18, 156, 140),
            finish_rgb=(245, 246, 252),
            runner_rgb=(255, 92, 168),
            runner_outline_rgb=(255, 235, 248),
            coin_rgb=(255, 218, 72),
            coin_outline_rgb=(25, 26, 42),
            label_rgb=(250, 252, 255),
            label_stroke_rgb=(19, 22, 39),
            pattern="scan",
        )
    if style == "paper_course":
        return LaneRunnerTheme(
            track_fill_rgb=(239, 230, 205),
            track_outline_rgb=(102, 91, 75),
            cell_a_rgb=(248, 239, 213),
            cell_b_rgb=(237, 223, 191),
            grid_rgb=(133, 116, 92),
            start_rgb=(105, 145, 108),
            finish_rgb=(252, 249, 238),
            runner_rgb=(64, 98, 170),
            runner_outline_rgb=(29, 42, 85),
            coin_rgb=(226, 163, 45),
            coin_outline_rgb=(89, 61, 25),
            label_rgb=(38, 44, 52),
            label_stroke_rgb=(250, 246, 232),
            pattern="paper",
        )
    return LaneRunnerTheme(
        track_fill_rgb=(46, 55, 88),
        track_outline_rgb=(21, 27, 52),
        cell_a_rgb=(66, 80, 126),
        cell_b_rgb=(76, 91, 139),
        grid_rgb=(210, 224, 255),
        start_rgb=(63, 194, 124),
        finish_rgb=(250, 250, 245),
        runner_rgb=(239, 77, 87),
        runner_outline_rgb=(71, 24, 31),
        coin_rgb=(247, 198, 45),
        coin_outline_rgb=(82, 58, 18),
        label_rgb=(248, 250, 255),
        label_stroke_rgb=(19, 24, 42),
        pattern="arcade",
    )


def _round_bbox(bbox: Sequence[float]) -> BBox:
    return tuple(round(float(value), 3) for value in bbox[:4])  # type: ignore[return-value]


def _center(bbox: Sequence[float]) -> Point:
    return (
        round((float(bbox[0]) + float(bbox[2])) * 0.5, 3),
        round((float(bbox[1]) + float(bbox[3])) * 0.5, 3),
    )


def _draw_track_pattern(draw: ImageDraw.ImageDraw, *, bbox: BBox, theme: LaneRunnerTheme) -> None:
    left, top, right, bottom = [float(value) for value in bbox]
    accent = tuple(max(0, min(255, int(channel))) for channel in theme.grid_rgb)
    if theme.pattern == "road":
        mid_x = (left + right) * 0.5
        dash_h = 22
        for y in range(int(top) + 18, int(bottom), dash_h * 2):
            draw.line([(mid_x, y), (mid_x, y + dash_h)], fill=accent, width=2)
    elif theme.pattern == "grass":
        for y in range(int(top) + 16, int(bottom), 32):
            for x in range(int(left) + 18, int(right), 44):
                draw.line([(x - 4, y + 3), (x, y - 5), (x + 5, y + 3)], fill=accent, width=1)
    elif theme.pattern == "scan":
        for y in range(int(top) + 14, int(bottom), 18):
            draw.line([(left + 8, y), (right - 8, y)], fill=accent, width=1)
    elif theme.pattern == "paper":
        for y in range(int(top) + 26, int(bottom), 42):
            draw.line([(left + 10, y), (right - 10, y)], fill=accent, width=1)


def _draw_coin(
    draw: ImageDraw.ImageDraw,
    *,
    center: Point,
    radius: float,
    theme: LaneRunnerTheme,
) -> BBox:
    cx, cy = float(center[0]), float(center[1])
    bbox = _round_bbox((cx - radius, cy - radius, cx + radius, cy + radius))
    draw.ellipse(
        bbox,
        fill=theme.coin_rgb,
        outline=theme.coin_outline_rgb,
        width=max(2, int(radius * 0.16)),
    )
    shine_r = max(2.0, float(radius) * 0.24)
    draw.ellipse(
        _round_bbox((cx - radius * 0.42, cy - radius * 0.48, cx - radius * 0.42 + shine_r, cy - radius * 0.48 + shine_r)),
        fill=(255, 243, 145),
    )
    return bbox


def _draw_hazard(
    draw: ImageDraw.ImageDraw,
    *,
    center: Point,
    radius: float,
    theme: LaneRunnerTheme,
) -> BBox:
    """Draw one visible hazard marker."""

    cx, cy = float(center[0]), float(center[1])
    bbox = _round_bbox((cx - radius, cy - radius, cx + radius, cy + radius))
    fill = (197, 48, 48)
    outline = (64, 18, 22)
    draw.rounded_rectangle(
        bbox,
        radius=max(4, int(radius * 0.25)),
        fill=fill,
        outline=outline,
        width=max(2, int(radius * 0.16)),
    )
    inset = float(radius) * 0.42
    draw.line((cx - inset, cy - inset, cx + inset, cy + inset), fill=(255, 240, 218), width=max(2, int(radius * 0.16)))
    draw.line((cx + inset, cy - inset, cx - inset, cy + inset), fill=(255, 240, 218), width=max(2, int(radius * 0.16)))
    return bbox


def _draw_runner(
    draw: ImageDraw.ImageDraw,
    *,
    center: Point,
    radius: float,
    theme: LaneRunnerTheme,
    marker_surface: Sequence[Sequence[int]],
    instance_seed: int,
) -> BBox:
    cx, cy = float(center[0]), float(center[1])
    bbox = _round_bbox((cx - radius, cy - radius, cx + radius, cy + radius))
    draw.ellipse(
        bbox,
        fill=theme.runner_rgb,
        outline=theme.runner_outline_rgb,
        width=max(2, int(radius * 0.16)),
    )
    arrow = (
        (cx, cy - radius * 0.68),
        (cx - radius * 0.44, cy + radius * 0.18),
        (cx - radius * 0.16, cy + radius * 0.12),
        (cx - radius * 0.16, cy + radius * 0.64),
        (cx + radius * 0.16, cy + radius * 0.64),
        (cx + radius * 0.16, cy + radius * 0.12),
        (cx + radius * 0.44, cy + radius * 0.18),
    )
    draw.polygon(arrow, fill=theme.runner_outline_rgb)
    marker = resolve_semantic_marker_style(
        surface_rgbs=marker_surface,
        role="runner_start",
        preferred_rgbs=((220, 40, 48),),
        instance_seed=int(instance_seed),
        namespace="games.lane_runner.runner_marker",
    )
    draw_semantic_ellipse_marker(draw, bbox=bbox, style=marker, width=max(3, int(radius * 0.20)))
    return bbox


def _path_color(label_index: int) -> Color:
    """Return a high-contrast candidate path color."""

    palette: Tuple[Color, ...] = (
        (34, 111, 220),
        (20, 145, 100),
        (132, 78, 190),
        (220, 122, 34),
        (20, 150, 164),
        (190, 64, 116),
    )
    return palette[int(label_index) % len(palette)]


def _draw_shown_board_path(
    draw: ImageDraw.ImageDraw,
    *,
    start_center: Point,
    finish_bbox: BBox,
    lanes_by_row: Sequence[int],
    cell_bboxes: Mapping[str, BBox],
    params: LaneRunnerRenderParams,
    theme: LaneRunnerTheme,
) -> Dict[str, Any]:
    """Draw one visible route through the main lane-runner board."""

    row_points: list[Point] = []
    for row, lane in enumerate(lanes_by_row):
        cell_id = cell_entity_id(int(row), int(lane))
        row_points.append(_center(cell_bboxes[cell_id]))
    if not row_points:
        return {"path_points_px": [], "polyline_points_px": [], "color_rgb": list(theme.runner_rgb)}

    last_lane = int(lanes_by_row[-1])
    finish_center = (
        round(float(row_points[-1][0]), 3),
        round((float(finish_bbox[1]) + float(finish_bbox[3])) * 0.5, 3),
    )
    polyline = [tuple(start_center), *row_points, finish_center]
    color = _path_color(0)
    shadow = theme.runner_outline_rgb if theme.pattern != "neon_track" else (245, 250, 255)
    width = max(3, int(params.path_line_width_px))
    draw.line(polyline, fill=shadow, width=width + 5, joint="curve")
    draw.line(polyline, fill=color, width=width, joint="curve")
    for row_index, point in enumerate(row_points):
        radius = max(4.0, float(width) * 0.58)
        draw.ellipse(
            _round_bbox((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius)),
            fill=color,
            outline=shadow,
            width=2,
        )
    return {
        "lanes_by_row": [int(value) for value in lanes_by_row],
        "path_points_px": [[round(float(x), 3), round(float(y), 3)] for x, y in row_points],
        "polyline_points_px": [[round(float(x), 3), round(float(y), 3)] for x, y in polyline],
        "color_rgb": list(color),
        "finish_lane": int(last_lane),
    }


def _draw_path_option_card(
    draw: ImageDraw.ImageDraw,
    *,
    option: LaneRunnerPathOption,
    option_index: int,
    card_bbox: BBox,
    hazards: Sequence[LaneRunnerHazard],
    params: LaneRunnerRenderParams,
    label_font,
    theme: LaneRunnerTheme,
) -> Dict[str, Any]:
    """Draw one compact labeled candidate route card."""

    color = _path_color(int(option_index))
    x0, y0, x1, y1 = [float(value) for value in card_bbox]
    card_fill = (246, 248, 252) if theme.pattern != "neon_track" else (30, 35, 62)
    card_outline = theme.track_outline_rgb if theme.pattern != "neon_track" else (190, 232, 245)
    draw.rounded_rectangle(card_bbox, radius=9, fill=card_fill, outline=card_outline, width=2)

    label_radius = max(10, int(params.path_label_font_size_px * 0.72))
    label_center = (x0 + label_radius + 6.0, y0 + label_radius + 6.0)
    label_bbox = _round_bbox(
        (
            label_center[0] - label_radius,
            label_center[1] - label_radius,
            label_center[0] + label_radius,
            label_center[1] + label_radius,
        )
    )
    draw.ellipse(label_bbox, fill=color, outline=card_outline, width=2)
    draw_centered_game_text(
        draw,
        text=str(option.label),
        center=label_center,
        font=label_font,
        fill=(255, 255, 255),
        stroke_fill=(25, 25, 30),
        stroke_width=1,
        role="lane_runner_path_label",
        required=True,
        surface_rgbs=(color,),
        preferred_rgbs=((255, 255, 255),),
        instance_seed=None,
        namespace=f"games.lane_runner.path_card_label.{str(option.label)}",
    )

    lanes = int(params.lane_count)
    rows = int(params.row_count)
    cell = float(params.option_card_cell_size_px)
    gap = float(params.option_card_gap_px)
    grid_w = (lanes * cell) + ((lanes - 1) * gap)
    grid_h = (rows * cell) + ((rows - 1) * gap)
    grid_left = x0 + ((x1 - x0) - grid_w) * 0.5
    grid_top = y0 + (2.0 * float(params.option_card_margin_px)) + (2.0 * float(label_radius))
    if grid_top + grid_h > y1 - float(params.option_card_margin_px):
        grid_top = y1 - float(params.option_card_margin_px) - grid_h

    hazard_cells = {(int(hazard.row), int(hazard.lane)) for hazard in hazards}
    cell_centers: Dict[Tuple[int, int], Point] = {}
    for row in range(rows):
        row_from_top = rows - 1 - int(row)
        cy0 = grid_top + (row_from_top * (cell + gap))
        for lane in range(lanes):
            cx0 = grid_left + (int(lane) * (cell + gap))
            bbox = _round_bbox((cx0, cy0, cx0 + cell, cy0 + cell))
            fill = theme.cell_a_rgb if (row + lane) % 2 == 0 else theme.cell_b_rgb
            draw.rounded_rectangle(
                bbox,
                radius=max(2, int(cell * 0.10)),
                fill=fill,
                outline=theme.grid_rgb,
                width=1,
            )
            center = _center(bbox)
            cell_centers[(int(row), int(lane))] = center
            if (int(row), int(lane)) in hazard_cells:
                hr = max(4.0, cell * 0.22)
                draw.rounded_rectangle(
                    _round_bbox((center[0] - hr, center[1] - hr, center[0] + hr, center[1] + hr)),
                    radius=max(2, int(hr * 0.28)),
                    fill=(197, 48, 48),
                    outline=(64, 18, 22),
                    width=1,
                )
                draw.line((center[0] - hr * 0.45, center[1] - hr * 0.45, center[0] + hr * 0.45, center[1] + hr * 0.45), fill=(255, 240, 218), width=1)
                draw.line((center[0] + hr * 0.45, center[1] - hr * 0.45, center[0] - hr * 0.45, center[1] + hr * 0.45), fill=(255, 240, 218), width=1)

    row_points = tuple(cell_centers[(row, int(lane))] for row, lane in enumerate(option.lanes_by_row))
    path_points = [(float(point[0]), float(point[1])) for point in row_points]
    shadow = (24, 27, 34) if theme.pattern != "neon_track" else (246, 250, 255)
    width = max(2, int(params.path_line_width_px * 0.72))
    if len(path_points) >= 2:
        draw.line(path_points, fill=shadow, width=width + 3, joint="curve")
        draw.line(path_points, fill=color, width=width, joint="curve")
    for point in path_points:
        radius = max(3.0, width * 0.72)
        draw.ellipse(
            _round_bbox((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius)),
            fill=color,
            outline=shadow,
            width=1,
        )

    return {
        "path_points_px": [[round(float(x), 3), round(float(y), 3)] for x, y in path_points],
        "label_bbox_px": list(label_bbox),
        "card_bbox_px": list(_round_bbox(card_bbox)),
        "color_rgb": list(color),
    }


def _option_card_column_count(option_count: int) -> int:
    """Return a compact, balanced option-card grid column count."""

    if int(option_count) == 4:
        return 2
    if int(option_count) == 6:
        return 3
    return 3


def render_lane_runner_scene(
    *,
    coins: Sequence[LaneRunnerCoin],
    hazards: Sequence[LaneRunnerHazard] = (),
    path_options: Sequence[LaneRunnerPathOption] = (),
    shown_path_lanes: Sequence[int] | None = None,
    start_lane: int,
    background: Image.Image,
    style_variant: str,
    params: LaneRunnerRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
    show_board: bool = True,
) -> RenderedLaneRunnerScene:
    """Render the lane-runner board or option-card panel with trace geometry."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    theme = build_games_lane_runner_theme(style_variant=str(style_variant))
    render_board = bool(show_board) or not path_options
    if not render_board and (coins or shown_path_lanes is not None):
        raise ValueError("lane-runner option-only rendering does not support coins or shown board paths")
    lanes = int(params.lane_count)
    rows = int(params.row_count)
    cell = float(params.cell_size_px)
    gap = float(params.cell_gap_px)
    board_w = (lanes * cell) + ((lanes - 1) * gap)
    board_h = (rows * cell) + ((rows - 1) * gap)
    option_area_w = 0.0
    option_area_h = 0.0
    option_cols = 0
    option_card_w = 0.0
    option_card_h = 0.0
    option_card_gap = 10.0
    if path_options:
        option_cols = _option_card_column_count(len(path_options))
        option_rows = int(math.ceil(len(path_options) / float(option_cols)))
        option_card_w = (
            (lanes * float(params.option_card_cell_size_px))
            + ((lanes - 1) * float(params.option_card_gap_px))
            + (2.0 * float(params.option_card_margin_px))
        )
        option_card_h = (
            (rows * float(params.option_card_cell_size_px))
            + ((rows - 1) * float(params.option_card_gap_px))
            + (4.0 * float(params.option_card_margin_px))
            + (2.0 * max(10.0, float(params.path_label_font_size_px) * 0.72))
        )
        option_area_w = (option_cols * option_card_w) + ((option_cols - 1) * option_card_gap)
        option_area_h = (option_rows * option_card_h) + ((option_rows - 1) * option_card_gap)
    if render_board:
        board_area_w = board_w + (float(params.option_card_area_gap_px) + option_area_w if path_options else 0.0)
    else:
        board_area_w = option_area_w
    content_w = board_area_w + (2.0 * float(params.panel_margin_px))
    if render_board:
        content_h = (
            float(params.finish_band_height_px)
            + gap
            + board_h
            + gap
            + float(params.start_band_height_px)
            + (2.0 * float(params.panel_margin_px))
        )
        content_h = max(content_h, option_area_h + (2.0 * float(params.panel_margin_px)))
    else:
        content_h = option_area_h + (2.0 * float(params.panel_margin_px))
    base_bbox = (
        0.5 * (float(params.canvas_width) - content_w),
        0.5 * (float(params.canvas_height) - content_h),
        0.5 * (float(params.canvas_width) + content_w),
        0.5 * (float(params.canvas_height) + content_h),
    )
    panel_bbox, _dx, _dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=base_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    left, top, right, bottom = [float(value) for value in panel_bbox]
    if panel_style is not None:
        draw_panel_scene_chrome(
            draw,
            bbox=(int(round(left - 12)), int(round(top - 12)), int(round(right + 12)), int(round(bottom + 12))),
            style=panel_style,
            radius=22,
            border_width=2,
        )
    draw.rounded_rectangle(
        panel_bbox,
        radius=18,
        fill=theme.track_fill_rgb,
        outline=theme.track_outline_rgb,
        width=max(2, int(params.grid_line_width_px)),
    )
    _draw_track_pattern(draw, bbox=panel_bbox, theme=theme)

    inner_left = left + float(params.panel_margin_px)
    inner_top = top + float(params.panel_margin_px)
    finish_bbox = _round_bbox((inner_left, inner_top, inner_left, inner_top))
    board_bbox = _round_bbox((inner_left, inner_top, inner_left, inner_top))
    start_bbox = _round_bbox((inner_left, inner_top, inner_left, inner_top))
    entities: list[Dict[str, Any]] = []
    entity_bboxes: Dict[str, BBox] = {}
    entity_points: Dict[str, Point] = {}
    cell_bboxes: Dict[str, BBox] = {}

    coin_points: Dict[str, Point] = {}
    coin_bboxes: Dict[str, BBox] = {}
    hazard_points: Dict[str, Point] = {}
    hazard_bboxes: Dict[str, BBox] = {}
    runner_center: Point = (inner_left, inner_top)
    if render_board:
        finish_bbox = _round_bbox((inner_left, inner_top, inner_left + board_w, inner_top + float(params.finish_band_height_px)))
        grid_top = float(finish_bbox[3]) + gap
        board_bbox = _round_bbox((inner_left, grid_top, inner_left + board_w, grid_top + board_h))
        start_top = float(board_bbox[3]) + gap
        start_bbox = _round_bbox((inner_left, start_top, inner_left + board_w, start_top + float(params.start_band_height_px)))

        draw.rounded_rectangle(finish_bbox, radius=8, fill=theme.finish_rgb, outline=theme.track_outline_rgb, width=2)
        draw.rounded_rectangle(start_bbox, radius=8, fill=theme.start_rgb, outline=theme.track_outline_rgb, width=2)
        font = load_font(int(params.label_font_size_px), bold=True, font_family=str(params.font_family or "") or None)
        draw_centered_game_text(
            draw,
            text="FINISH",
            center=_center(finish_bbox),
            font=font,
            fill=theme.label_rgb,
            stroke_fill=theme.label_stroke_rgb,
            stroke_width=1,
            role="lane_runner_finish_label",
            required=True,
            surface_rgbs=(theme.finish_rgb,),
            preferred_rgbs=(theme.label_rgb,),
            instance_seed=None,
            namespace="games.lane_runner.finish_label",
        )

        for row in range(rows):
            row_from_top = rows - 1 - int(row)
            y0 = grid_top + (row_from_top * (cell + gap))
            for lane in range(lanes):
                x0 = inner_left + (int(lane) * (cell + gap))
                bbox = _round_bbox((x0, y0, x0 + cell, y0 + cell))
                fill = theme.cell_a_rgb if (row + lane) % 2 == 0 else theme.cell_b_rgb
                draw.rounded_rectangle(
                    bbox,
                    radius=max(4, int(cell * 0.08)),
                    fill=fill,
                    outline=theme.grid_rgb,
                    width=max(1, int(params.grid_line_width_px)),
                )
                cell_id = cell_entity_id(row, lane)
                cell_bboxes[str(cell_id)] = bbox
                entity_bboxes[str(cell_id)] = bbox
                entities.append(
                    {
                        "entity_id": str(cell_id),
                        "entity_type": "lane_runner_cell",
                        "row": int(row),
                        "lane": int(lane),
                        "bbox_px": list(bbox),
                    }
                )

        runner_x = inner_left + (int(start_lane) * (cell + gap)) + (cell * 0.5)
        runner_center = (round(runner_x, 3), round(float(start_bbox[1]) + (float(params.start_band_height_px) * 0.5), 3))
        runner_bbox = _draw_runner(
            draw,
            center=runner_center,
            radius=float(params.runner_radius_px),
            theme=theme,
            marker_surface=(
                theme.runner_rgb,
                theme.start_rgb,
            ),
            instance_seed=int(params.instance_seed),
        )
        entity_bboxes[runner_entity_id()] = runner_bbox
        entity_points[runner_entity_id()] = runner_center
        entities.append(
            {
                "entity_id": runner_entity_id(),
                "entity_type": "lane_runner_start",
                "lane": int(start_lane),
                "point_px": list(runner_center),
                "bbox_px": list(runner_bbox),
            }
        )

        for hazard in hazards:
            cell_bbox = cell_bboxes[cell_entity_id(int(hazard.row), int(hazard.lane))]
            center = _center(cell_bbox)
            hazard_bbox = _draw_hazard(
                draw,
                center=center,
                radius=float(params.hazard_radius_px),
                theme=theme,
            )
            hazard_points[str(hazard.hazard_id)] = center
            hazard_bboxes[str(hazard.hazard_id)] = hazard_bbox
            entity_points[str(hazard.hazard_id)] = center
            entity_bboxes[str(hazard.hazard_id)] = hazard_bbox
            entities.append(
                {
                    "entity_id": str(hazard.hazard_id),
                    "entity_type": "lane_runner_hazard",
                    "row": int(hazard.row),
                    "lane": int(hazard.lane),
                    "point_px": list(center),
                    "bbox_px": list(hazard_bbox),
                    }
                )

    shown_path_render_map: Dict[str, Any] | None = None
    if render_board and shown_path_lanes is not None:
        shown_path_render_map = _draw_shown_board_path(
            draw,
            start_center=runner_center,
            finish_bbox=finish_bbox,
            lanes_by_row=tuple(int(value) for value in shown_path_lanes),
            cell_bboxes=cell_bboxes,
            params=params,
            theme=theme,
        )

    path_points: Dict[str, Tuple[Point, ...]] = {}
    path_render_map: Dict[str, Dict[str, Any]] = {}
    if path_options:
        path_font = load_font(int(params.path_label_font_size_px), bold=True, font_family=str(params.font_family or "") or None)
        option_left = inner_left + board_w + float(params.option_card_area_gap_px) if render_board else inner_left
        option_top = top + ((bottom - top) - option_area_h) * 0.5 if option_area_h > 0.0 else inner_top
        for option_index, option in enumerate(path_options):
            card_col = int(option_index) % max(1, int(option_cols))
            card_row = int(option_index) // max(1, int(option_cols))
            card_left = option_left + (card_col * (option_card_w + option_card_gap))
            card_top = option_top + (card_row * (option_card_h + option_card_gap))
            card_bbox = _round_bbox((card_left, card_top, card_left + option_card_w, card_top + option_card_h))
            path_entity_id = path_option_entity_id(str(option.label))
            metadata = _draw_path_option_card(
                draw,
                option=option,
                option_index=int(option_index),
                card_bbox=card_bbox,
                hazards=hazards,
                params=params,
                label_font=path_font,
                theme=theme,
            )
            row_points = tuple((float(point[0]), float(point[1])) for point in metadata["path_points_px"])
            path_points[str(path_entity_id)] = tuple(row_points)
            path_render_map[str(option.label)] = dict(metadata)
            option_bbox = _round_bbox(metadata["card_bbox_px"])
            entity_bboxes[str(path_entity_id)] = option_bbox
            entities.append(
                {
                    "entity_id": str(path_entity_id),
                    "entity_type": "lane_runner_path_option",
                    "label": str(option.label),
                    "lanes_by_row": [int(value) for value in option.lanes_by_row],
                    "path_points_px": [list(point) for point in row_points],
                    "label_bbox_px": list(metadata["label_bbox_px"]),
                    "bbox_px": list(option_bbox),
                }
            )

    if render_board:
        for coin in coins:
            cell_bbox = cell_bboxes[cell_entity_id(int(coin.row), int(coin.lane))]
            center = _center(cell_bbox)
            coin_bbox = _draw_coin(
                draw,
                center=center,
                radius=float(params.coin_radius_px),
                theme=theme,
            )
            coin_points[str(coin.coin_id)] = center
            coin_bboxes[str(coin.coin_id)] = coin_bbox
            entity_points[str(coin.coin_id)] = center
            entity_bboxes[str(coin.coin_id)] = coin_bbox
            entities.append(
                {
                    "entity_id": str(coin.coin_id),
                    "entity_type": "lane_runner_coin",
                    "row": int(coin.row),
                    "lane": int(coin.lane),
                    "point_px": list(center),
                    "bbox_px": list(coin_bbox),
                }
            )

    render_map = {
        "show_board": bool(render_board),
        "panel_bbox_px": list(_round_bbox(panel_bbox)),
        "board_bbox_px": list(board_bbox),
        "start_bbox_px": list(start_bbox),
        "finish_bbox_px": list(finish_bbox),
        "cell_bboxes_px": {str(key): list(value) for key, value in cell_bboxes.items()},
        "coin_points_px": {str(key): list(value) for key, value in coin_points.items()},
        "coin_bboxes_px": {str(key): list(value) for key, value in coin_bboxes.items()},
        "hazard_points_px": {str(key): list(value) for key, value in hazard_points.items()},
        "hazard_bboxes_px": {str(key): list(value) for key, value in hazard_bboxes.items()},
        "path_points_px": {str(key): [list(point) for point in value] for key, value in path_points.items()},
        "path_options_px": {str(key): dict(value) for key, value in path_render_map.items()},
        "shown_path_px": dict(shown_path_render_map or {}),
        "entity_points_px": {str(key): list(value) for key, value in entity_points.items()},
        "entity_bboxes_px": {str(key): list(value) for key, value in entity_bboxes.items()},
        "layout_jitter": {
            **dict(layout_jitter),
            "unit_size_jitter": dict(params.unit_size_meta or {}),
        },
        "lane_runner_style": {
            "style_variant": str(style_variant),
            "theme": {
                "track_fill_rgb": list(theme.track_fill_rgb),
                "track_outline_rgb": list(theme.track_outline_rgb),
                "cell_a_rgb": list(theme.cell_a_rgb),
                "cell_b_rgb": list(theme.cell_b_rgb),
                "coin_rgb": list(theme.coin_rgb),
                "runner_rgb": list(theme.runner_rgb),
            },
        },
        "panel_scene_style": game_panel_scene_style_metadata(panel_style) if panel_style is not None else {},
    }
    return RenderedLaneRunnerScene(
        image=image,
        scene_entities=tuple(entities),
        render_map=render_map,
    )


__all__ = [
    "LaneRunnerRenderParams",
    "RenderedLaneRunnerScene",
    "build_games_lane_runner_theme",
    "render_lane_runner_scene",
]
