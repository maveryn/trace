"""Shared Bubble-shooter renderer for games-domain tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import get_font_family_record

from ....shared.color_distance import (
    min_color_distance_to_anchors,
    resolve_contrasting_palette,
)
from ....shared.drawing import draw_dashed_line
from ...shared.text import draw_centered_game_text as draw_centered_text
from ....shared.text_rendering import load_font
from .state import (
    BUBBLE_COLOR_KEYS,
    Board,
    BubbleShooterLandingOption,
    BubbleShooterOption,
    Coord,
    bubble_entity_id,
    landing_option_entity_id,
    landing_slot_entity_id,
    option_entity_id,
    shooter_bubble_entity_id,
)
from .rules import board_value, occupied_coords
from ...shared.layout import apply_games_layout_jitter_to_bbox
from ...shared.scene_style import (
    GamePanelSceneStyle,
    draw_panel_scene_chrome,
    game_panel_contrast_anchor_colors,
    game_panel_scene_style_metadata,
)
from ...shared.visual_defaults import load_games_scene_noise_defaults


@dataclass(frozen=True)
class BubbleShooterRenderParams:
    """Resolved render controls for one Bubble-shooter scene."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    playfield_width_px: int
    playfield_height_px: int
    playfield_border_width_px: int
    board_top_px: int
    board_height_px: int
    bubble_gap_px: int
    path_width_px: int
    shooter_radius_px: int
    option_radius_px: int
    option_label_font_size_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class BubbleShooterTheme:
    """Resolved Bubble-shooter palette for one style variant."""

    playfield_fill_rgb: Tuple[int, int, int]
    playfield_outline_rgb: Tuple[int, int, int]
    slot_outline_rgb: Tuple[int, int, int]
    guide_line_rgb: Tuple[int, int, int]
    launcher_fill_rgb: Tuple[int, int, int]
    launcher_outline_rgb: Tuple[int, int, int]
    option_panel_fill_rgb: Tuple[int, int, int]
    option_panel_outline_rgb: Tuple[int, int, int]
    option_label_rgb: Tuple[int, int, int]
    color_palette_rgb: Dict[str, Tuple[int, int, int]]
    bubble_outline_rgb: Tuple[int, int, int]
    highlight_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedBubbleShooterScene:
    """Rendered Bubble-shooter image plus trace-friendly geometry."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class RenderedBubbleShooterTaskContext:
    """Rendered Bubble-shooter task image plus scene/review metadata."""

    image: Image.Image
    rendered_scene: RenderedBubbleShooterScene
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    panel_style_meta: Dict[str, Any]
    text_style_meta: Dict[str, Any]


POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(
    scene_id="bubble_shooter", apply_prob=0.5
)


def build_games_bubble_shooter_theme(*, style_variant: str) -> BubbleShooterTheme:
    """Return one Bubble-shooter visual theme with stable contrast-safe role colors."""

    style = str(style_variant)
    if style == "pastel":
        return BubbleShooterTheme(
            playfield_fill_rgb=(246, 242, 233),
            playfield_outline_rgb=(116, 111, 98),
            slot_outline_rgb=(97, 112, 131),
            guide_line_rgb=(68, 113, 169),
            launcher_fill_rgb=(223, 213, 192),
            launcher_outline_rgb=(97, 89, 76),
            option_panel_fill_rgb=(236, 230, 214),
            option_panel_outline_rgb=(127, 116, 96),
            option_label_rgb=(49, 43, 34),
            color_palette_rgb={
                "red": (222, 101, 109),
                "yellow": (232, 194, 95),
                "blue": (101, 154, 211),
                "green": (103, 176, 127),
                "purple": (163, 123, 196),
                "orange": (226, 139, 83),
            },
            bubble_outline_rgb=(91, 82, 73),
            highlight_rgb=(255, 255, 255),
        )
    if style == "neon":
        return BubbleShooterTheme(
            playfield_fill_rgb=(13, 14, 31),
            playfield_outline_rgb=(105, 104, 221),
            slot_outline_rgb=(255, 237, 107),
            guide_line_rgb=(80, 229, 238),
            launcher_fill_rgb=(29, 26, 72),
            launcher_outline_rgb=(168, 155, 255),
            option_panel_fill_rgb=(23, 22, 60),
            option_panel_outline_rgb=(125, 116, 244),
            option_label_rgb=(239, 241, 255),
            color_palette_rgb={
                "red": (255, 79, 124),
                "yellow": (255, 231, 89),
                "blue": (78, 204, 255),
                "green": (105, 239, 125),
                "purple": (203, 105, 255),
                "orange": (255, 151, 79),
            },
            bubble_outline_rgb=(243, 246, 255),
            highlight_rgb=(255, 255, 255),
        )
    if style == "paper":
        return BubbleShooterTheme(
            playfield_fill_rgb=(243, 236, 219),
            playfield_outline_rgb=(82, 74, 61),
            slot_outline_rgb=(44, 87, 136),
            guide_line_rgb=(45, 93, 150),
            launcher_fill_rgb=(220, 206, 174),
            launcher_outline_rgb=(75, 66, 52),
            option_panel_fill_rgb=(232, 220, 195),
            option_panel_outline_rgb=(92, 80, 60),
            option_label_rgb=(45, 39, 31),
            color_palette_rgb={
                "red": (201, 83, 76),
                "yellow": (221, 176, 71),
                "blue": (76, 126, 184),
                "green": (85, 151, 102),
                "purple": (137, 98, 169),
                "orange": (209, 116, 70),
            },
            bubble_outline_rgb=(65, 55, 45),
            highlight_rgb=(255, 248, 232),
        )
    if style == "arcade":
        return BubbleShooterTheme(
            playfield_fill_rgb=(30, 21, 37),
            playfield_outline_rgb=(226, 116, 94),
            slot_outline_rgb=(250, 211, 92),
            guide_line_rgb=(255, 219, 94),
            launcher_fill_rgb=(67, 35, 65),
            launcher_outline_rgb=(226, 126, 153),
            option_panel_fill_rgb=(57, 32, 62),
            option_panel_outline_rgb=(229, 126, 102),
            option_label_rgb=(255, 239, 218),
            color_palette_rgb={
                "red": (237, 69, 73),
                "yellow": (247, 199, 70),
                "blue": (72, 143, 224),
                "green": (76, 186, 104),
                "purple": (167, 86, 213),
                "orange": (238, 128, 64),
            },
            bubble_outline_rgb=(255, 231, 214),
            highlight_rgb=(255, 255, 238),
        )
    return BubbleShooterTheme(
        playfield_fill_rgb=(21, 29, 45),
        playfield_outline_rgb=(145, 165, 194),
        slot_outline_rgb=(255, 221, 104),
        guide_line_rgb=(255, 225, 108),
        launcher_fill_rgb=(42, 55, 78),
        launcher_outline_rgb=(172, 192, 218),
        option_panel_fill_rgb=(38, 50, 72),
        option_panel_outline_rgb=(150, 174, 207),
        option_label_rgb=(236, 242, 249),
        color_palette_rgb={
            "red": (225, 76, 83),
            "yellow": (233, 195, 75),
            "blue": (83, 146, 222),
            "green": (81, 183, 113),
            "purple": (158, 100, 209),
            "orange": (233, 132, 68),
        },
        bubble_outline_rgb=(238, 243, 247),
        highlight_rgb=(255, 255, 255),
    )


def _slot_centers(
    *,
    playfield_bbox: Tuple[float, float, float, float],
    row_count: int,
    col_count: int,
    params: BubbleShooterRenderParams,
) -> Tuple[Dict[Coord, Tuple[float, float]], float]:
    """Return close-packed bubble slot centers and radius."""

    left, top, right, _bottom = playfield_bbox
    board_left = float(left + 48.0)
    board_right = float(right - 48.0)
    board_top = float(top + float(params.board_top_px))
    board_height = float(params.board_height_px)
    gap = float(params.bubble_gap_px)
    width_radius = float(
        (board_right - board_left) / max(1.0, (2.0 * float(col_count)) + 1.1)
    )
    height_radius = float(
        board_height / max(1.0, 2.0 + (math.sqrt(3.0) * max(0, int(row_count) - 1)))
    )
    radius = max(13.0, min(float(width_radius), float(height_radius)) - (0.5 * gap))
    step_x = float((2.0 * radius) + gap)
    step_y = float((math.sqrt(3.0) * radius) + (0.72 * gap))
    used_width = float((2.0 * radius) + ((int(col_count) - 1) * step_x) + radius)
    x0 = float(
        board_left + (0.5 * max(0.0, (board_right - board_left) - used_width)) + radius
    )
    centers: Dict[Coord, Tuple[float, float]] = {}
    for row in range(int(row_count)):
        row_offset = radius if row % 2 else 0.0
        for col in range(int(col_count)):
            centers[(row, col)] = (
                round(float(x0 + row_offset + (float(col) * step_x)), 3),
                round(float(board_top + radius + (float(row) * step_y)), 3),
            )
    return centers, float(radius)


def _bubble_bbox(
    center: Tuple[float, float], radius: float
) -> Tuple[float, float, float, float]:
    """Return a circle bbox."""

    cx, cy = float(center[0]), float(center[1])
    return (
        round(cx - float(radius), 3),
        round(cy - float(radius), 3),
        round(cx + float(radius), 3),
        round(cy + float(radius), 3),
    )


def _segment_circle_entry_t(
    *,
    start: Tuple[float, float],
    end: Tuple[float, float],
    center: Tuple[float, float],
    radius: float,
) -> float | None:
    """Return the first segment parameter where a ray segment enters a circle."""

    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    cx, cy = float(center[0]), float(center[1])
    dx = float(ex - sx)
    dy = float(ey - sy)
    a = float((dx * dx) + (dy * dy))
    if a <= 1e-9:
        return None
    fx = float(sx - cx)
    fy = float(sy - cy)
    c = float((fx * fx) + (fy * fy) - (float(radius) * float(radius)))
    if c <= 0.0:
        return 0.0
    b = float(2.0 * ((fx * dx) + (fy * dy)))
    discriminant = float((b * b) - (4.0 * a * c))
    if discriminant < 0.0:
        return None
    root = math.sqrt(discriminant)
    candidates = sorted(
        (float((-b - root) / (2.0 * a)), float((-b + root) / (2.0 * a)))
    )
    for value in candidates:
        if 0.0 <= value <= 1.0:
            return float(value)
    return None


def _short_aim_cue_segment(
    *,
    path_start: Tuple[float, float],
    path_end: Tuple[float, float],
    occupied_centers: Tuple[Tuple[float, float], ...],
    bubble_radius: float,
    path_width_px: int,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return a short visible shot cue that stops before crossing board bubbles."""

    sx, sy = float(path_start[0]), float(path_start[1])
    ex, ey = float(path_end[0]), float(path_end[1])
    dx = float(ex - sx)
    dy = float(ey - sy)
    distance = math.hypot(dx, dy)
    if distance <= 1.0:
        return path_start, path_end

    start_t = 0.055
    max_len_px = min(92.0, max(48.0, distance * 0.16))
    max_t = min(0.22, float(max_len_px / distance))
    inflated_radius = float(bubble_radius + max(10.0, float(path_width_px) * 2.5))
    first_hit_t = 1.0
    for center in occupied_centers:
        hit_t = _segment_circle_entry_t(
            start=path_start, end=path_end, center=center, radius=inflated_radius
        )
        if hit_t is not None and hit_t > start_t:
            first_hit_t = min(first_hit_t, float(hit_t))

    clearance_t = float(
        max(20.0, (0.45 * float(bubble_radius)) + (2.0 * float(path_width_px)))
        / distance
    )
    safe_t = max(start_t, float(first_hit_t - clearance_t))
    desired_t = min(max_t, safe_t)
    if desired_t <= start_t:
        desired_t = min(max_t, start_t + min(0.045, 24.0 / distance))
        desired_t = min(desired_t, safe_t)

    cue_start = (round(sx + (start_t * dx), 3), round(sy + (start_t * dy), 3))
    cue_end = (round(sx + (desired_t * dx), 3), round(sy + (desired_t * dy), 3))
    return cue_start, cue_end


def _draw_bubble(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    radius: float,
    color_key: str,
    theme: BubbleShooterTheme,
    outline_width: int = 3,
) -> Tuple[float, float, float, float]:
    """Draw one glossy bubble and return its bbox."""

    bbox = _bubble_bbox(center, radius)
    fill = theme.color_palette_rgb.get(
        str(color_key), theme.color_palette_rgb[BUBBLE_COLOR_KEYS[0]]
    )
    draw.ellipse(
        bbox,
        fill=tuple(int(v) for v in fill),
        outline=tuple(int(v) for v in theme.bubble_outline_rgb),
        width=max(1, int(outline_width)),
    )
    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    draw.ellipse(
        (
            round(cx - (0.52 * r), 3),
            round(cy - (0.56 * r), 3),
            round(cx - (0.08 * r), 3),
            round(cy - (0.12 * r), 3),
        ),
        fill=tuple(int(v) for v in theme.highlight_rgb) + (145,),
    )
    return bbox


def _draw_labeled_option(
    draw: ImageDraw.ImageDraw,
    *,
    label: str,
    color_key: str,
    center: Tuple[float, float],
    radius: float,
    theme: BubbleShooterTheme,
    label_font_size_px: int,
    font_family: str,
) -> Tuple[float, float, float, float]:
    """Draw one labeled color option and return its bbox."""

    bbox = _draw_bubble(
        draw,
        center=center,
        radius=radius,
        color_key=color_key,
        theme=theme,
        outline_width=3,
    )
    font = load_font(
        max(10, int(label_font_size_px)),
        bold=True,
        font_family=str(font_family) or None,
    )
    draw_centered_text(
        draw,
        text=str(label),
        center=(float(center[0]), float(center[1] + (1.36 * radius))),
        font=font,
        fill=tuple(int(v) for v in theme.option_label_rgb),
        stroke_fill=tuple(int(v) for v in theme.option_panel_fill_rgb),
        stroke_width=2,
    )
    return (
        round(float(bbox[0] - 8.0), 3),
        round(float(bbox[1] - 8.0), 3),
        round(float(bbox[2] + 8.0), 3),
        round(float(bbox[3] + (1.95 * radius)), 3),
    )


def _landing_marker_colours(
    theme: BubbleShooterTheme,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Return contrast-first marker colors that are not bubble-palette colors."""

    fill = tuple(float(value) for value in theme.playfield_fill_rgb)
    luminance = (0.2126 * fill[0]) + (0.7152 * fill[1]) + (0.0722 * fill[2])
    if luminance < 120.0:
        return (255, 255, 255), (16, 18, 24)
    return (18, 22, 28), (255, 255, 255)


def _draw_landing_marker(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    radius: float,
    theme: BubbleShooterTheme,
) -> Tuple[float, float, float, float]:
    """Draw a non-bubble bullseye landing target marker and return its bbox."""

    cx, cy = float(center[0]), float(center[1])
    marker_radius = float(radius * 1.12)
    bbox = _bubble_bbox(center, marker_radius)
    primary_rgb, shadow_rgb = _landing_marker_colours(theme)
    outer_shadow_width = max(8, int(round(radius * 0.30)))
    outer_width = max(5, int(round(radius * 0.19)))
    inner_shadow_width = max(6, int(round(radius * 0.22)))
    inner_width = max(4, int(round(radius * 0.14)))
    inner_radius = float(radius * 0.46)
    inner_bbox = _bubble_bbox(center, inner_radius)
    for ring_bbox, shadow_width, width in (
        (bbox, outer_shadow_width, outer_width),
        (inner_bbox, inner_shadow_width, inner_width),
    ):
        draw.ellipse(
            ring_bbox,
            fill=None,
            outline=shadow_rgb + (245,),
            width=int(shadow_width),
        )
        draw.ellipse(
            ring_bbox,
            fill=None,
            outline=primary_rgb + (255,),
            width=int(width),
        )
    tick = float(marker_radius * 0.32)
    gap = float(marker_radius * 0.15)
    for start, end in (
        ((cx - tick, cy), (cx - gap, cy)),
        ((cx + gap, cy), (cx + tick, cy)),
        ((cx, cy - tick), (cx, cy - gap)),
        ((cx, cy + gap), (cx, cy + tick)),
    ):
        draw.line(
            (
                round(start[0], 3),
                round(start[1], 3),
                round(end[0], 3),
                round(end[1], 3),
            ),
            fill=shadow_rgb + (235,),
            width=max(5, int(round(radius * 0.18))),
        )
        draw.line(
            (
                round(start[0], 3),
                round(start[1], 3),
                round(end[0], 3),
                round(end[1], 3),
            ),
            fill=primary_rgb + (255,),
            width=max(3, int(round(radius * 0.10))),
        )
    return bbox


def _draw_labeled_landing_marker(
    draw: ImageDraw.ImageDraw,
    *,
    label: str,
    center: Tuple[float, float],
    radius: float,
    theme: BubbleShooterTheme,
    font_family: str,
) -> Tuple[float, float, float, float]:
    """Draw one simple labeled landing target marker and return its bbox."""

    primary_rgb, shadow_rgb = _landing_marker_colours(theme)
    marker_radius = float(radius * 1.08)
    bbox = _bubble_bbox(center, marker_radius)
    draw.ellipse(
        bbox,
        fill=tuple(int(v) for v in theme.playfield_fill_rgb) + (210,),
        outline=tuple(int(v) for v in shadow_rgb) + (235,),
        width=max(7, int(round(float(radius) * 0.24))),
    )
    draw.ellipse(
        bbox,
        fill=None,
        outline=tuple(int(v) for v in primary_rgb) + (255,),
        width=max(4, int(round(float(radius) * 0.13))),
    )
    font = load_font(
        max(15, int(round(float(radius) * 0.86))),
        bold=True,
        font_family=str(font_family) or None,
    )
    draw_centered_text(
        draw,
        text=str(label),
        center=(float(center[0]), float(center[1])),
        font=font,
        fill=tuple(int(v) for v in primary_rgb),
        stroke_fill=tuple(int(v) for v in shadow_rgb),
        stroke_width=max(2, int(round(float(radius) * 0.07))),
    )
    return bbox


def _guide_color_anchor_rgbs(
    *,
    theme: BubbleShooterTheme,
    panel_style: GamePanelSceneStyle | None,
) -> Tuple[Tuple[int, int, int], ...]:
    """Return known panel/playfield colors the aim cue must avoid."""

    anchors: list[Tuple[int, int, int]] = [
        tuple(int(v) for v in theme.playfield_fill_rgb),
        tuple(int(v) for v in theme.playfield_outline_rgb),
        tuple(int(v) for v in theme.slot_outline_rgb),
        tuple(int(v) for v in theme.launcher_fill_rgb),
        tuple(int(v) for v in theme.launcher_outline_rgb),
        tuple(int(v) for v in theme.option_panel_fill_rgb),
        tuple(int(v) for v in theme.option_panel_outline_rgb),
    ]
    return tuple(game_panel_contrast_anchor_colors(panel_style, extra_colors=anchors))


def render_bubble_shooter_scene(
    *,
    board: Board,
    landing_coord: Coord,
    shooter_color_key: str | None,
    option_specs: Tuple[BubbleShooterOption, ...],
    landing_option_specs: Tuple[BubbleShooterLandingOption, ...] = tuple(),
    background: Image.Image,
    scene_variant: str,
    style_variant: str,
    params: BubbleShooterRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedBubbleShooterScene:
    """Render one Bubble-shooter playfield with a marked shot landing slot."""

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme = build_games_bubble_shooter_theme(style_variant=str(style_variant))
    guide_color_anchors = _guide_color_anchor_rgbs(theme=theme, panel_style=panel_style)
    guide_line_rgb = resolve_contrasting_palette(
        (theme.guide_line_rgb,),
        anchor_colors=guide_color_anchors,
        min_anchor_distance=40.0,
        min_pairwise_distance=0.0,
        distance_space="lab",
    )[0]

    left = float((int(params.canvas_width) - int(params.playfield_width_px)) / 2.0)
    top = float((int(params.canvas_height) - int(params.playfield_height_px)) / 2.0)
    playfield_bbox = (
        left,
        top,
        left + float(params.playfield_width_px),
        top + float(params.playfield_height_px),
    )
    if isinstance(params.layout_jitter_meta, Mapping):
        playfield_bbox, _dx, _dy, layout_jitter = apply_games_layout_jitter_to_bbox(
            bbox_px=playfield_bbox,
            canvas_width=int(params.canvas_width),
            canvas_height=int(params.canvas_height),
            jitter=params.layout_jitter_meta,
        )
    else:
        layout_jitter = {}

    panel_bbox: Tuple[int, int, int, int] | None = None
    if panel_style is not None:
        panel_pad = max(18, int(round(float(params.panel_margin_px) * 0.56)))
        panel_bbox = (
            max(4, int(round(playfield_bbox[0])) - panel_pad),
            max(4, int(round(playfield_bbox[1])) - panel_pad),
            min(
                int(params.canvas_width) - 4, int(round(playfield_bbox[2])) + panel_pad
            ),
            min(
                int(params.canvas_height) - 4, int(round(playfield_bbox[3])) + panel_pad
            ),
        )
        draw_panel_scene_chrome(
            draw,
            bbox=panel_bbox,
            style=panel_style,
            radius=32,
            border_width=max(
                2, int(round(float(params.playfield_border_width_px) * 0.45))
            ),
        )

    draw.rounded_rectangle(
        playfield_bbox,
        radius=24,
        fill=tuple(int(v) for v in theme.playfield_fill_rgb) + (238,),
        outline=tuple(int(v) for v in theme.playfield_outline_rgb) + (255,),
        width=int(params.playfield_border_width_px),
    )

    rows = len(board)
    cols = len(board[0]) if rows else 0
    centers, radius = _slot_centers(
        playfield_bbox=playfield_bbox,
        row_count=rows,
        col_count=cols,
        params=params,
    )

    entity_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    entity_centers: Dict[str, Tuple[float, float]] = {}
    bubble_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    scene_entities: list[Dict[str, Any]] = []
    for coord in occupied_coords(board):
        color_key = str(board_value(board, coord))
        center = centers[coord]
        bbox = _draw_bubble(
            draw,
            center=center,
            radius=radius,
            color_key=color_key,
            theme=theme,
            outline_width=3 if str(scene_variant) == "dense_pack" else 2,
        )
        entity_id = bubble_entity_id(coord)
        entity_bboxes[str(entity_id)] = bbox
        entity_centers[str(entity_id)] = (
            round(float(center[0]), 3),
            round(float(center[1]), 3),
        )
        bubble_bboxes[str(entity_id)] = bbox
        scene_entities.append(
            {
                "entity_id": str(entity_id),
                "entity_type": "bubble_shooter_bubble",
                "row": int(coord[0]),
                "col": int(coord[1]),
                "color_key": str(color_key),
                "bbox_px": list(bbox),
                "center_px": list(entity_centers[str(entity_id)]),
            }
        )

    landing_center = centers[tuple(landing_coord)]
    slot_radius = float(radius * 0.92)
    landing_option_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    if landing_option_specs:
        landing_bbox = _bubble_bbox(landing_center, float(radius * 1.12))
        for option in landing_option_specs:
            option_coord = tuple(option.landing_coord)
            option_center = centers[option_coord]
            bbox = _draw_labeled_landing_marker(
                draw,
                label=str(option.label),
                center=option_center,
                radius=float(radius),
                theme=theme,
                font_family=str(params.font_family),
            )
            entity_id = landing_option_entity_id(str(option.label))
            entity_bboxes[str(entity_id)] = bbox
            entity_centers[str(entity_id)] = (
                round(float(option_center[0]), 3),
                round(float(option_center[1]), 3),
            )
            landing_option_bboxes[str(entity_id)] = bbox
            scene_entities.append(
                {
                    "entity_id": str(entity_id),
                    "entity_type": "bubble_shooter_landing_option",
                    "label": str(option.label),
                    "row": int(option_coord[0]),
                    "col": int(option_coord[1]),
                    "is_answer": bool(option.is_answer),
                    "bbox_px": list(bbox),
                    "center_px": list(entity_centers[str(entity_id)]),
                }
            )
    else:
        landing_bbox = _draw_landing_marker(
            draw, center=landing_center, radius=float(radius), theme=theme
        )
        entity_bboxes[landing_slot_entity_id()] = landing_bbox
        entity_centers[landing_slot_entity_id()] = (
            round(float(landing_center[0]), 3),
            round(float(landing_center[1]), 3),
        )
        scene_entities.append(
            {
                "entity_id": landing_slot_entity_id(),
                "entity_type": "bubble_shooter_landing_slot",
                "row": int(landing_coord[0]),
                "col": int(landing_coord[1]),
                "bbox_px": list(landing_bbox),
                "center_px": list(entity_centers[landing_slot_entity_id()]),
            }
        )

    play_left, _play_top, play_right, play_bottom = playfield_bbox
    shooter_center = (
        round(float((play_left + play_right) / 2.0), 3),
        round(float(play_bottom - 88.0), 3),
    )
    launcher_bbox = (
        round(float(shooter_center[0] - 62.0), 3),
        round(float(shooter_center[1] - 38.0), 3),
        round(float(shooter_center[0] + 62.0), 3),
        round(float(shooter_center[1] + 42.0), 3),
    )
    draw.rounded_rectangle(
        launcher_bbox,
        radius=26,
        fill=tuple(int(v) for v in theme.launcher_fill_rgb) + (230,),
        outline=tuple(int(v) for v in theme.launcher_outline_rgb) + (255,),
        width=3,
    )
    if shooter_color_key is not None:
        shooter_bbox = _draw_bubble(
            draw,
            center=shooter_center,
            radius=float(params.shooter_radius_px),
            color_key=str(shooter_color_key),
            theme=theme,
            outline_width=4,
        )
    else:
        shooter_bbox = (
            round(float(shooter_center[0] - params.shooter_radius_px), 3),
            round(float(shooter_center[1] - params.shooter_radius_px), 3),
            round(float(shooter_center[0] + params.shooter_radius_px), 3),
            round(float(shooter_center[1] + params.shooter_radius_px), 3),
        )
        draw.ellipse(
            shooter_bbox,
            fill=tuple(int(v) for v in theme.launcher_fill_rgb),
            outline=tuple(int(v) for v in theme.launcher_outline_rgb),
            width=3,
        )
    entity_bboxes[shooter_bubble_entity_id()] = shooter_bbox
    entity_centers[shooter_bubble_entity_id()] = (
        round(float(shooter_center[0]), 3),
        round(float(shooter_center[1]), 3),
    )
    scene_entities.append(
        {
            "entity_id": shooter_bubble_entity_id(),
            "entity_type": "bubble_shooter_launcher_bubble",
            "color_key": None if shooter_color_key is None else str(shooter_color_key),
            "bbox_px": list(shooter_bbox),
            "center_px": list(entity_centers[shooter_bubble_entity_id()]),
        }
    )

    path_start = (
        float(shooter_center[0]),
        float(shooter_center[1] - params.shooter_radius_px - 4.0),
    )
    path_end = (float(landing_center[0]), float(landing_center[1] + slot_radius + 3.0))
    cue_start, cue_end = _short_aim_cue_segment(
        path_start=path_start,
        path_end=path_end,
        occupied_centers=tuple(centers[coord] for coord in occupied_coords(board)),
        bubble_radius=float(radius),
        path_width_px=int(params.path_width_px),
    )
    if (
        not landing_option_specs
        and math.hypot(
            float(cue_end[0]) - float(cue_start[0]),
            float(cue_end[1]) - float(cue_start[1]),
        )
        >= 16.0
    ):
        draw_dashed_line(
            draw,
            start=cue_start,
            end=cue_end,
            fill=tuple(int(v) for v in guide_line_rgb),
            width=int(params.path_width_px),
            dash_px=16,
            gap_px=8,
        )

    option_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    if option_specs:
        panel_width = min(
            float(play_right - play_left - 80.0), float(128.0 * len(option_specs))
        )
        panel_cx = float((play_left + play_right) / 2.0)
        panel_bbox = (
            round(float(panel_cx - (0.5 * panel_width)), 3),
            round(float(play_bottom - 74.0), 3),
            round(float(panel_cx + (0.5 * panel_width)), 3),
            round(float(play_bottom - 12.0), 3),
        )
        draw.rounded_rectangle(
            panel_bbox,
            radius=18,
            fill=tuple(int(v) for v in theme.option_panel_fill_rgb) + (228,),
            outline=tuple(int(v) for v in theme.option_panel_outline_rgb) + (255,),
            width=2,
        )
        gap = float(panel_width / max(1, len(option_specs)))
        for index, option in enumerate(option_specs):
            center = (
                round(float(panel_bbox[0] + ((index + 0.5) * gap)), 3),
                round(float(panel_bbox[1] + 22.0), 3),
            )
            bbox = _draw_labeled_option(
                draw,
                label=str(option.label),
                color_key=str(option.color_key),
                center=center,
                radius=float(params.option_radius_px),
                theme=theme,
                label_font_size_px=int(params.option_label_font_size_px),
                font_family=str(params.font_family),
            )
            entity_id = option_entity_id(str(option.label))
            entity_bboxes[str(entity_id)] = bbox
            entity_centers[str(entity_id)] = (
                round(float(center[0]), 3),
                round(float(center[1]), 3),
            )
            option_bboxes[str(entity_id)] = bbox
            scene_entities.append(
                {
                    "entity_id": str(entity_id),
                    "entity_type": "bubble_shooter_color_option",
                    "label": str(option.label),
                    "color_key": str(option.color_key),
                    "is_answer": bool(option.is_answer),
                    "bbox_px": list(bbox),
                    "center_px": list(entity_centers[str(entity_id)]),
                }
            )

    render_map = {
        "playfield_bbox_px": [round(float(v), 3) for v in playfield_bbox],
        "panel_bbox_px": (
            None if panel_bbox is None else [int(value) for value in panel_bbox]
        ),
        "bubble_bboxes_px": {
            str(key): list(value) for key, value in bubble_bboxes.items()
        },
        "option_bboxes_px": {
            str(key): list(value) for key, value in option_bboxes.items()
        },
        "landing_option_bboxes_px": {
            str(key): list(value) for key, value in landing_option_bboxes.items()
        },
        "entity_bboxes_px": {
            str(key): list(value) for key, value in entity_bboxes.items()
        },
        "entity_centers_px": {
            str(key): list(value) for key, value in entity_centers.items()
        },
        "landing_slot_bbox_px": list(landing_bbox),
        "landing_slot_center_px": [
            round(float(landing_center[0]), 3),
            round(float(landing_center[1]), 3),
        ],
        "shooter_bubble_bbox_px": list(shooter_bbox),
        "shooter_bubble_center_px": list(entity_centers[shooter_bubble_entity_id()]),
        "slot_centers_px": {
            f"{row},{col}": [round(float(center[0]), 3), round(float(center[1]), 3)]
            for (row, col), center in centers.items()
        },
        "bubble_radius_px": round(float(radius), 3),
        "motion_path_px": {
            "start": [round(float(path_start[0]), 3), round(float(path_start[1]), 3)],
            "end": [round(float(path_end[0]), 3), round(float(path_end[1]), 3)],
        },
        "aim_cue_path_px": {
            "start": [round(float(cue_start[0]), 3), round(float(cue_start[1]), 3)],
            "end": [round(float(cue_end[0]), 3), round(float(cue_end[1]), 3)],
        },
        "layout_jitter": dict(layout_jitter),
        "font_family": str(params.font_family),
        "guide_line_rgb": list(guide_line_rgb),
        "guide_color_safety": {
            "distance_space": "lab",
            "min_anchor_distance_required": 40.0,
            "anchor_rgbs": [list(color) for color in guide_color_anchors],
            "guide_anchor_lab_distance": round(
                float(
                    min_color_distance_to_anchors(
                        guide_line_rgb, guide_color_anchors, distance_space="lab"
                    )
                ),
                3,
            ),
        },
        "panel_scene_style": (
            None
            if panel_style is None
            else game_panel_scene_style_metadata(panel_style)
        ),
    }
    return RenderedBubbleShooterScene(
        image=image.convert("RGB"),
        scene_entities=tuple(scene_entities),
        render_map=render_map,
    )


def render_bubble_shooter_task_scene(
    *,
    board: Board,
    landing_coord: Coord,
    shooter_color_key: str | None,
    option_specs: Tuple[BubbleShooterOption, ...],
    landing_option_specs: Tuple[BubbleShooterLandingOption, ...] = tuple(),
    scene_variant: str,
    style_variant: str,
    render_params: BubbleShooterRenderParams,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
) -> RenderedBubbleShooterTaskContext:
    """Render a Bubble-shooter task scene with shared panel/noise treatment."""

    from ...shared.scene_style import (
        make_panel_scene_background,
        resolve_game_panel_scene_style,
    )

    allowed_panel_treatments_raw = params.get(
        "panel_scene_treatments",
        group_default(render_defaults, "panel_scene_treatments", None),
    )
    if isinstance(allowed_panel_treatments_raw, str):
        allowed_panel_treatments = (str(allowed_panel_treatments_raw),)
    elif allowed_panel_treatments_raw is None:
        allowed_panel_treatments = None
    else:
        allowed_panel_treatments = tuple(
            str(item) for item in allowed_panel_treatments_raw
        )
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace="games.bubble_shooter.panel_scene_style",
        treatments=allowed_panel_treatments,
        treatment_weights=params.get(
            "panel_scene_treatment_weights",
            group_default(render_defaults, "panel_scene_treatment_weights", None),
        ),
        palette_weights=params.get(
            "panel_scene_palette_weights",
            group_default(render_defaults, "panel_scene_palette_weights", None),
        ),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered_scene = render_bubble_shooter_scene(
        board=board,
        landing_coord=landing_coord,
        shooter_color_key=shooter_color_key,
        option_specs=option_specs,
        landing_option_specs=landing_option_specs,
        background=background,
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        params=render_params,
        panel_style=panel_style,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    text_style_meta = {
        "font_family": str(render_params.font_family),
        "font_asset": get_font_family_record(str(render_params.font_family)).to_trace(),
    }
    return RenderedBubbleShooterTaskContext(
        image=image,
        rendered_scene=rendered_scene,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        panel_style_meta=dict(panel_style_meta),
        text_style_meta=dict(text_style_meta),
    )


__all__ = [
    "BubbleShooterRenderParams",
    "BubbleShooterTheme",
    "RenderedBubbleShooterScene",
    "RenderedBubbleShooterTaskContext",
    "build_games_bubble_shooter_theme",
    "render_bubble_shooter_scene",
    "render_bubble_shooter_task_scene",
]
