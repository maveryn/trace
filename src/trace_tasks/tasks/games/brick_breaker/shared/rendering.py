"""Shared Brick-breaker renderer for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.font_assets import get_font_family_record

from ....shared.color_distance import min_color_distance_to_anchors, resolve_contrasting_palette
from ....shared.drawing import draw_arrow, draw_dashed_line
from ....shared.text_rendering import fit_font_to_box
from ...shared.text import draw_game_text_traced as draw_text_traced
from .defaults import SCENE_ID
from .state import BrickBreakerBrick, brick_entity_id, lane_entity_id, lane_label
from ...shared.layout import apply_games_layout_jitter_to_bbox
from ...shared.scene_style import (
    GamePanelSceneStyle,
    draw_panel_scene_chrome,
    game_panel_contrast_anchor_colors,
    game_panel_scene_style_metadata,
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from ...shared.visual_defaults import load_games_scene_noise_defaults


_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


@dataclass(frozen=True)
class BrickBreakerRenderParams:
    """Resolved render controls for one Brick-breaker scene."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    playfield_width_px: int
    playfield_height_px: int
    playfield_border_width_px: int
    brick_wall_top_px: int
    brick_wall_height_px: int
    brick_gap_px: int
    lane_pad_height_px: int
    lane_pad_gap_px: int
    ball_radius_px: int
    path_width_px: int
    label_font_size_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class BrickBreakerTheme:
    """Resolved Brick-breaker palette for one style variant."""

    playfield_fill_rgb: Tuple[int, int, int]
    playfield_outline_rgb: Tuple[int, int, int]
    guide_line_rgb: Tuple[int, int, int]
    brick_palette_rgb: Tuple[Tuple[int, int, int], ...]
    brick_outline_rgb: Tuple[int, int, int]
    brick_text_rgb: Tuple[int, int, int]
    lane_fill_rgb: Tuple[int, int, int]
    lane_outline_rgb: Tuple[int, int, int]
    lane_text_rgb: Tuple[int, int, int]
    ball_fill_rgb: Tuple[int, int, int]
    ball_outline_rgb: Tuple[int, int, int]
    paddle_fill_rgb: Tuple[int, int, int]
    paddle_outline_rgb: Tuple[int, int, int]
    marker_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedBrickBreakerScene:
    """Rendered Brick-breaker image plus trace-friendly geometry."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class RenderedBrickBreakerTaskContext:
    """Rendered Brick-breaker scene plus reusable trace metadata."""

    rendered_scene: RenderedBrickBreakerScene
    image: Image.Image
    render_params: BrickBreakerRenderParams
    panel_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    text_style_meta: Dict[str, Any]


def build_games_brick_breaker_theme(*, style_variant: str) -> BrickBreakerTheme:
    """Return one Brick-breaker theme with safe contrast for labels, lanes, and paths."""

    style = str(style_variant)
    if style == "neon":
        return BrickBreakerTheme(
            playfield_fill_rgb=(12, 13, 30),
            playfield_outline_rgb=(122, 104, 255),
            guide_line_rgb=(255, 231, 83),
            brick_palette_rgb=((255, 72, 168), (73, 229, 242), (255, 145, 73), (153, 244, 93), (210, 105, 255)),
            brick_outline_rgb=(248, 232, 255),
            brick_text_rgb=(17, 12, 25),
            lane_fill_rgb=(29, 25, 74),
            lane_outline_rgb=(134, 126, 255),
            lane_text_rgb=(230, 234, 255),
            ball_fill_rgb=(255, 245, 112),
            ball_outline_rgb=(255, 255, 225),
            paddle_fill_rgb=(57, 224, 244),
            paddle_outline_rgb=(211, 251, 255),
            marker_rgb=(255, 145, 73),
        )
    if style == "paper":
        return BrickBreakerTheme(
            playfield_fill_rgb=(246, 240, 222),
            playfield_outline_rgb=(92, 84, 68),
            guide_line_rgb=(49, 94, 150),
            brick_palette_rgb=((218, 98, 84), (235, 165, 83), (96, 166, 122), (86, 141, 196), (155, 112, 190)),
            brick_outline_rgb=(79, 67, 55),
            brick_text_rgb=(42, 34, 28),
            lane_fill_rgb=(232, 221, 197),
            lane_outline_rgb=(111, 100, 78),
            lane_text_rgb=(45, 39, 32),
            ball_fill_rgb=(56, 104, 168),
            ball_outline_rgb=(31, 60, 103),
            paddle_fill_rgb=(80, 91, 104),
            paddle_outline_rgb=(36, 42, 48),
            marker_rgb=(192, 75, 66),
        )
    if style == "blueprint":
        return BrickBreakerTheme(
            playfield_fill_rgb=(18, 45, 82),
            playfield_outline_rgb=(151, 199, 230),
            guide_line_rgb=(226, 240, 255),
            brick_palette_rgb=((99, 171, 223), (121, 203, 231), (171, 222, 238), (83, 135, 197), (64, 106, 169)),
            brick_outline_rgb=(216, 239, 255),
            brick_text_rgb=(9, 32, 58),
            lane_fill_rgb=(27, 64, 108),
            lane_outline_rgb=(170, 212, 237),
            lane_text_rgb=(232, 245, 255),
            ball_fill_rgb=(248, 245, 216),
            ball_outline_rgb=(204, 219, 234),
            paddle_fill_rgb=(114, 189, 227),
            paddle_outline_rgb=(221, 244, 255),
            marker_rgb=(255, 209, 112),
        )
    if style == "arcade":
        return BrickBreakerTheme(
            playfield_fill_rgb=(32, 19, 28),
            playfield_outline_rgb=(228, 105, 87),
            guide_line_rgb=(255, 210, 89),
            brick_palette_rgb=((232, 75, 62), (239, 127, 64), (250, 191, 73), (80, 183, 102), (67, 144, 219)),
            brick_outline_rgb=(255, 232, 205),
            brick_text_rgb=(35, 19, 12),
            lane_fill_rgb=(75, 36, 50),
            lane_outline_rgb=(230, 130, 105),
            lane_text_rgb=(255, 235, 220),
            ball_fill_rgb=(255, 245, 141),
            ball_outline_rgb=(255, 255, 231),
            paddle_fill_rgb=(92, 208, 161),
            paddle_outline_rgb=(221, 255, 239),
            marker_rgb=(83, 197, 231),
        )
    return BrickBreakerTheme(
        playfield_fill_rgb=(21, 28, 46),
        playfield_outline_rgb=(143, 166, 196),
        guide_line_rgb=(255, 226, 104),
        brick_palette_rgb=((218, 82, 79), (229, 148, 69), (230, 203, 83), (83, 181, 127), (83, 139, 214)),
        brick_outline_rgb=(238, 243, 247),
        brick_text_rgb=(24, 29, 38),
        lane_fill_rgb=(45, 58, 82),
        lane_outline_rgb=(166, 188, 214),
        lane_text_rgb=(237, 242, 247),
        ball_fill_rgb=(248, 248, 238),
        ball_outline_rgb=(64, 72, 84),
        paddle_fill_rgb=(86, 171, 224),
        paddle_outline_rgb=(217, 238, 255),
        marker_rgb=(255, 128, 90),
    )


def _fit_text(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[float, float, float, float],
    text: str,
    fill: Tuple[int, int, int],
    max_size_px: int,
    font_family: str = "",
) -> None:
    """Draw centered text inside one bbox."""

    left, top, right, bottom = bbox
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=max(1.0, float(right - left)),
        max_height=max(1.0, float(bottom - top)),
        bold=False,
        min_size_px=7,
        max_size_px=int(max_size_px),
        fill_ratio=0.74,
        font_family=str(font_family) or None,
    )
    text_bbox = draw.textbbox((0, 0), str(text), font=font)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    draw_text_traced(
        draw,
        (
            float(left + (0.5 * (float(right - left) - text_w)) - float(text_bbox[0])),
            float(top + (0.5 * (float(bottom - top) - text_h)) - float(text_bbox[1])),
        ),
        str(text),
        fill=tuple(int(v) for v in fill),
        font=font,
        role="context_text",
        required=False,
        stroke_width=0,
    )


def _brick_bbox(
    *,
    playfield_bbox: Tuple[float, float, float, float],
    row: int,
    col: int,
    rows: int,
    cols: int,
    params: BrickBreakerRenderParams,
) -> Tuple[float, float, float, float]:
    """Return one brick bbox in playfield coordinates."""

    left, top, right, _bottom = playfield_bbox
    wall_left = float(left + 44.0)
    wall_right = float(right - 44.0)
    wall_top = float(top + float(params.brick_wall_top_px))
    gap = float(params.brick_gap_px)
    brick_w = float((wall_right - wall_left - (gap * max(0, int(cols) - 1))) / max(1, int(cols)))
    brick_h = float((float(params.brick_wall_height_px) - (gap * max(0, int(rows) - 1))) / max(1, int(rows)))
    x0 = float(wall_left + (int(col) * (brick_w + gap)))
    y0 = float(wall_top + (int(row) * (brick_h + gap)))
    return (
        round(x0, 3),
        round(y0, 3),
        round(x0 + brick_w, 3),
        round(y0 + brick_h, 3),
    )


def _draw_brick(
    draw: ImageDraw.ImageDraw,
    *,
    brick: BrickBreakerBrick,
    bbox: Tuple[float, float, float, float],
    theme: BrickBreakerTheme,
    label_font_size_px: int,
    font_family: str,
) -> None:
    """Draw one labeled brick."""

    fill = theme.brick_palette_rgb[int(brick.color_index) % len(theme.brick_palette_rgb)]
    radius = max(3, int(round(0.15 * min(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])))))
    draw.rounded_rectangle(
        bbox,
        radius=radius,
        fill=tuple(int(v) for v in fill),
        outline=tuple(int(v) for v in theme.brick_outline_rgb),
        width=2,
    )
    inner = (
        float(bbox[0] + 5),
        float(bbox[1] + 3),
        float(bbox[2] - 5),
        float(bbox[3] - 3),
    )
    _fit_text(
        draw,
        bbox=inner,
        text=str(brick.label),
        fill=tuple(int(v) for v in theme.brick_text_rgb),
        max_size_px=int(label_font_size_px),
        font_family=str(font_family),
    )


def _draw_lane(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[float, float, float, float],
    label: str,
    theme: BrickBreakerTheme,
    label_font_size_px: int,
    font_family: str,
) -> None:
    """Draw one bottom catch lane pad."""

    draw.rounded_rectangle(
        bbox,
        radius=8,
        fill=tuple(int(v) for v in theme.lane_fill_rgb) + (226,),
        outline=tuple(int(v) for v in theme.lane_outline_rgb) + (255,),
        width=2,
    )
    _fit_text(
        draw,
        bbox=(bbox[0] + 4, bbox[1] + 3, bbox[2] - 4, bbox[3] - 3),
        text=str(label),
        fill=tuple(int(v) for v in theme.lane_text_rgb),
        max_size_px=int(label_font_size_px),
        font_family=str(font_family),
    )


def _draw_ball(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    radius_px: float,
    theme: BrickBreakerTheme,
) -> Tuple[float, float, float, float]:
    """Draw the visible ball and return its bbox."""

    cx, cy = float(center[0]), float(center[1])
    r = float(radius_px)
    bbox = (round(cx - r, 3), round(cy - r, 3), round(cx + r, 3), round(cy + r, 3))
    draw.ellipse(
        bbox,
        fill=tuple(int(v) for v in theme.ball_fill_rgb),
        outline=tuple(int(v) for v in theme.ball_outline_rgb),
        width=3,
    )
    highlight = (cx - (0.45 * r), cy - (0.45 * r), cx - (0.05 * r), cy - (0.05 * r))
    draw.ellipse(highlight, fill=(255, 255, 255, 170))
    return bbox


def _guide_color_anchor_rgbs(
    *,
    theme: BrickBreakerTheme,
    panel_style: GamePanelSceneStyle | None,
) -> Tuple[Tuple[int, int, int], ...]:
    """Return known background/object colors the trajectory cue must avoid."""

    anchors: list[Tuple[int, int, int]] = [
        tuple(int(v) for v in theme.playfield_fill_rgb),
        tuple(int(v) for v in theme.playfield_outline_rgb),
        tuple(int(v) for v in theme.lane_fill_rgb),
        tuple(int(v) for v in theme.lane_outline_rgb),
        tuple(int(v) for v in theme.paddle_fill_rgb),
        tuple(int(v) for v in theme.paddle_outline_rgb),
    ]
    return tuple(game_panel_contrast_anchor_colors(panel_style, extra_colors=anchors))


def render_brick_breaker_scene(
    *,
    brick_rows: int,
    brick_cols: int,
    lane_count: int,
    bricks: Tuple[BrickBreakerBrick, ...],
    render_mode: str,
    target_brick_id: str | None,
    target_lane_index: int | None,
    ball_start_lane_index: int | None,
    background: Image.Image,
    style_variant: str,
    params: BrickBreakerRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedBrickBreakerScene:
    """Render the playfield and preserve brick/lane geometry for annotation projection."""

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme = build_games_brick_breaker_theme(style_variant=str(style_variant))
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

    clip_left, clip_top, clip_right, clip_bottom = playfield_bbox
    panel_bbox: Tuple[int, int, int, int] | None = None
    if panel_style is not None:
        panel_pad = max(16, int(round(float(params.panel_margin_px) * 0.56)))
        panel_bbox = (
            max(4, int(round(clip_left)) - panel_pad),
            max(4, int(round(clip_top)) - panel_pad),
            min(int(params.canvas_width) - 4, int(round(clip_right)) + panel_pad),
            min(int(params.canvas_height) - 4, int(round(clip_bottom)) + panel_pad),
        )
        draw_panel_scene_chrome(
            draw,
            bbox=panel_bbox,
            style=panel_style,
            radius=30,
            border_width=max(2, int(round(float(params.playfield_border_width_px) * 0.45))),
        )

    draw.rounded_rectangle(
        playfield_bbox,
        radius=22,
        fill=tuple(int(v) for v in theme.playfield_fill_rgb) + (238,),
        outline=tuple(int(v) for v in theme.playfield_outline_rgb) + (255,),
        width=int(params.playfield_border_width_px),
    )

    brick_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    entity_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    scene_entities: list[Dict[str, Any]] = []
    for brick in bricks:
        bbox = _brick_bbox(
            playfield_bbox=playfield_bbox,
            row=int(brick.row),
            col=int(brick.col),
            rows=int(brick_rows),
            cols=int(brick_cols),
            params=params,
        )
        _draw_brick(
            draw,
            brick=brick,
            bbox=bbox,
            theme=theme,
            label_font_size_px=int(params.label_font_size_px),
            font_family=str(params.font_family),
        )
        brick_bboxes[str(brick.brick_id)] = bbox
        entity_bboxes[str(brick.brick_id)] = bbox
        scene_entities.append(
            {
                "entity_id": str(brick.brick_id),
                "entity_type": "brick_breaker_brick",
                "label": str(brick.label),
                "row": int(brick.row),
                "col": int(brick.col),
                "bbox_px": list(bbox),
            }
        )

    lane_width = float((clip_right - clip_left) / max(1, int(lane_count)))
    lane_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    lane_top = float(clip_bottom - float(params.lane_pad_height_px) - 30.0)
    for lane in range(int(lane_count)):
        pad_left = float(clip_left + (lane * lane_width) + float(params.lane_pad_gap_px))
        pad_right = float(clip_left + ((lane + 1) * lane_width) - float(params.lane_pad_gap_px))
        bbox = (
            round(pad_left, 3),
            round(lane_top, 3),
            round(pad_right, 3),
            round(lane_top + float(params.lane_pad_height_px), 3),
        )
        lane_id = lane_entity_id(lane)
        lane_bboxes[str(lane_id)] = bbox
        entity_bboxes[str(lane_id)] = bbox
        _draw_lane(
            draw,
            bbox=bbox,
            label=lane_label(lane),
            theme=theme,
            label_font_size_px=int(params.label_font_size_px),
            font_family=str(params.font_family),
        )
        scene_entities.append(
            {
                "entity_id": str(lane_id),
                "entity_type": "brick_breaker_catch_lane",
                "label": lane_label(lane),
                "lane": int(lane),
                "bbox_px": list(bbox),
            }
        )

    paddle_height = max(6.0, min(9.0, float(params.lane_pad_height_px) * 0.22))
    paddle_y = float(lane_top - 24.0)
    first_lane_bbox = lane_bboxes[lane_entity_id(0)]
    paddle_w = round(float(first_lane_bbox[2] - first_lane_bbox[0]), 3)
    paddle_cx = float((clip_left + clip_right) / 2.0)
    paddle_left = round(paddle_cx - (0.5 * paddle_w), 3)
    paddle_bbox = (
        paddle_left,
        round(paddle_y, 3),
        round(paddle_left + paddle_w, 3),
        round(paddle_y + paddle_height, 3),
    )
    draw.rounded_rectangle(
        paddle_bbox,
        radius=7,
        fill=tuple(int(v) for v in theme.paddle_fill_rgb),
        outline=tuple(int(v) for v in theme.paddle_outline_rgb),
        width=2,
    )
    entity_bboxes["paddle"] = paddle_bbox
    scene_entities.append({"entity_id": "paddle", "entity_type": "brick_breaker_paddle", "bbox_px": list(paddle_bbox)})

    path_start: Tuple[float, float]
    path_end: Tuple[float, float]
    mode = str(render_mode)
    if mode == "brick_hit_path":
        if target_brick_id is None or str(target_brick_id) not in brick_bboxes:
            raise ValueError("brick-hit render requires target brick bbox")
        if ball_start_lane_index is None:
            raise ValueError("brick-hit render requires a ball start lane")
        target_bbox = brick_bboxes[str(target_brick_id)]
        path_end = (
            float((target_bbox[0] + target_bbox[2]) / 2.0),
            float((target_bbox[1] + target_bbox[3]) / 2.0),
        )
        start_lane = max(0, min(int(lane_count) - 1, int(ball_start_lane_index)))
        start_x = float(clip_left + ((start_lane + 0.5) * lane_width))
        path_start = (
            float(start_x),
            float(lane_top - 72.0),
        )
        is_brick_hit_path = True
    elif mode == "paddle_catch_path":
        if target_lane_index is None or ball_start_lane_index is None:
            raise ValueError("paddle_catch_label render requires target and start lanes")
        target_lane = lane_bboxes[lane_entity_id(int(target_lane_index))]
        start_lane = max(0, min(int(lane_count) - 1, int(ball_start_lane_index)))
        start_x = float(clip_left + ((start_lane + 0.5) * lane_width))
        path_start = (
            float(start_x),
            float(clip_top + float(params.brick_wall_top_px) + float(params.brick_wall_height_px) + 82.0),
        )
        path_end = (
            float((target_lane[0] + target_lane[2]) / 2.0),
            float(target_lane[1] + 3.0),
        )
        is_brick_hit_path = False
    else:
        raise ValueError(f"unsupported Brick-breaker render mode: {render_mode}")

    visible_fraction = 0.55 if bool(is_brick_hit_path) else 0.42
    visible_end = (
        float(path_start[0] + (float(visible_fraction) * (path_end[0] - path_start[0]))),
        float(path_start[1] + (float(visible_fraction) * (path_end[1] - path_start[1]))),
    )
    draw_dashed_line(
        draw,
        start=path_start,
        end=visible_end,
        fill=tuple(int(v) for v in guide_line_rgb),
        width=int(params.path_width_px),
        dash_px=18,
        gap_px=10,
    )
    draw_arrow(
        draw,
        start=(
            float(path_start[0] + (0.56 * (visible_end[0] - path_start[0]))),
            float(path_start[1] + (0.56 * (visible_end[1] - path_start[1]))),
        ),
        end=visible_end,
        fill=tuple(int(v) for v in guide_line_rgb),
        width=int(params.path_width_px),
        head_length_px=18 if is_brick_hit_path else 22,
        head_width_px=14 if is_brick_hit_path else 18,
    )
    ball_bbox = _draw_ball(
        draw,
        center=path_start,
        radius_px=float(params.ball_radius_px),
        theme=theme,
    )
    entity_bboxes["ball"] = ball_bbox
    scene_entities.append({"entity_id": "ball", "entity_type": "brick_breaker_ball", "bbox_px": list(ball_bbox)})

    render_map = {
        "playfield_bbox_px": [round(float(v), 3) for v in playfield_bbox],
        "panel_bbox_px": None if panel_bbox is None else [int(value) for value in panel_bbox],
        "brick_bboxes_px": {str(key): list(value) for key, value in brick_bboxes.items()},
        "lane_bboxes_px": {str(key): list(value) for key, value in lane_bboxes.items()},
        "entity_bboxes_px": {str(key): list(value) for key, value in entity_bboxes.items()},
        "paddle_bbox_px": list(paddle_bbox),
        "ball_bbox_px": list(ball_bbox),
        "motion_path_px": {
            "start": [round(float(path_start[0]), 3), round(float(path_start[1]), 3)],
            "end": [round(float(path_end[0]), 3), round(float(path_end[1]), 3)],
        },
        "visible_motion_path_px": {
            "start": [round(float(path_start[0]), 3), round(float(path_start[1]), 3)],
            "end": [round(float(visible_end[0]), 3), round(float(visible_end[1]), 3)],
            "fraction_of_full_path": float(visible_fraction),
        },
        "layout_jitter": dict(layout_jitter),
        "font_family": str(params.font_family),
        "guide_line_rgb": list(guide_line_rgb),
        "guide_color_safety": {
            "distance_space": "lab",
            "min_anchor_distance_required": 40.0,
            "anchor_rgbs": [list(color) for color in guide_color_anchors],
            "guide_anchor_lab_distance": round(
                float(min_color_distance_to_anchors(guide_line_rgb, guide_color_anchors, distance_space="lab")),
                3,
            ),
        },
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
    }
    return RenderedBrickBreakerScene(
        image=image.convert("RGB"),
        scene_entities=tuple(scene_entities),
        render_map=render_map,
    )


def render_brick_breaker_task_scene(
    *,
    brick_rows: int,
    brick_cols: int,
    lane_count: int,
    bricks: Tuple[BrickBreakerBrick, ...],
    render_mode: str,
    target_brick_id: str | None,
    target_lane_index: int | None,
    ball_start_lane_index: int | None,
    style_variant: str,
    render_params: BrickBreakerRenderParams,
    params: Mapping[str, Any],
    instance_seed: int,
) -> RenderedBrickBreakerTaskContext:
    """Render one Brick-breaker task scene with shared panel and noise treatment."""

    allowed_panel_treatments_raw = params.get(
        "panel_scene_treatments",
        group_default(_RENDER_DEFAULTS, "panel_scene_treatments", None),
    )
    if isinstance(allowed_panel_treatments_raw, str):
        allowed_panel_treatments = (str(allowed_panel_treatments_raw),)
    elif allowed_panel_treatments_raw is None:
        allowed_panel_treatments = None
    else:
        allowed_panel_treatments = tuple(str(item) for item in allowed_panel_treatments_raw)

    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace="games.brick_breaker.panel_scene_style",
        treatments=allowed_panel_treatments,
        treatment_weights=params.get(
            "panel_scene_treatment_weights",
            group_default(_RENDER_DEFAULTS, "panel_scene_treatment_weights", None),
        ),
        palette_weights=params.get(
            "panel_scene_palette_weights",
            group_default(_RENDER_DEFAULTS, "panel_scene_palette_weights", None),
        ),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered_scene = render_brick_breaker_scene(
        brick_rows=int(brick_rows),
        brick_cols=int(brick_cols),
        lane_count=int(lane_count),
        bricks=bricks,
        render_mode=str(render_mode),
        target_brick_id=target_brick_id,
        target_lane_index=target_lane_index,
        ball_start_lane_index=ball_start_lane_index,
        background=background,
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
    return RenderedBrickBreakerTaskContext(
        rendered_scene=rendered_scene,
        image=image,
        render_params=render_params,
        panel_style_meta=dict(panel_style_meta),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        text_style_meta=dict(text_style_meta),
    )


__all__ = [
    "BrickBreakerRenderParams",
    "BrickBreakerTheme",
    "RenderedBrickBreakerScene",
    "RenderedBrickBreakerTaskContext",
    "build_games_brick_breaker_theme",
    "render_brick_breaker_scene",
    "render_brick_breaker_task_scene",
]
