"""Shared Platformer level renderer for games-domain tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.games.shared.layout import (
    apply_games_layout_jitter_to_bbox,
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from trace_tasks.tasks.games.shared.scene_style import (
    GamePanelSceneStyle,
    draw_panel_scene_chrome,
    game_panel_scene_style_metadata,
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family

from ....shared.drawing import draw_dashed_line
from ....shared.text_rendering import fit_font_to_box
from ...shared.text import draw_centered_game_text, draw_game_text_traced as draw_text_traced
from .defaults import DEFAULTS, POST_IMAGE_NOISE_DEFAULTS
from .state import PlatformerCollectible, PlatformerHazard, PlatformerPlatform, PlatformerSample


@dataclass(frozen=True)
class PlatformerRenderParams:
    """Resolved render controls for one side-scroller platformer level."""

    canvas_width: int
    canvas_height: int
    level_width_px: int
    level_height_px: int
    level_border_width_px: int
    platform_height_px: int
    player_width_px: int
    player_height_px: int
    hazard_width_px: int
    hazard_height_px: int
    collectible_radius_px: int
    path_width_px: int
    label_font_size_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class PlatformerTheme:
    """Resolved visual theme for the platformer level."""

    sky_rgb: Tuple[int, int, int]
    far_rgb: Tuple[int, int, int]
    level_outline_rgb: Tuple[int, int, int]
    ground_rgb: Tuple[int, int, int]
    ground_outline_rgb: Tuple[int, int, int]
    platform_palette_rgb: Tuple[Tuple[int, int, int], ...]
    platform_outline_rgb: Tuple[int, int, int]
    platform_text_rgb: Tuple[int, int, int]
    hazard_palette_rgb: Tuple[Tuple[int, int, int], ...]
    hazard_outline_rgb: Tuple[int, int, int]
    hazard_text_rgb: Tuple[int, int, int]
    coin_palette_rgb: Tuple[Tuple[int, int, int], ...]
    coin_outline_rgb: Tuple[int, int, int]
    player_rgb: Tuple[int, int, int]
    player_outline_rgb: Tuple[int, int, int]
    path_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedPlatformerScene:
    """Rendered Platformer image plus trace-friendly geometry."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class RenderedPlatformerTaskContext:
    """Rendered Platformer image plus style and noise metadata."""

    image: Image.Image
    rendered_scene: RenderedPlatformerScene
    render_params: PlatformerRenderParams
    background_meta: Dict[str, Any]
    panel_style_meta: Dict[str, Any]
    text_style_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


def resolve_platformer_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> PlatformerRenderParams:
    """Resolve Platformer rendering parameters from config/defaults."""

    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            render_defaults,
            instance_seed=int(instance_seed),
            namespace=f"{str(namespace)}.layout",
        ),
        unit_scale_meta,
    )
    level_width_px = scale_games_px(
        params.get("level_width_px", group_default(render_defaults, "level_width_px", DEFAULTS.level_width_px)),
        unit_scale,
        min_px=430,
    )
    level_height_px = scale_games_px(
        params.get("level_height_px", group_default(render_defaults, "level_height_px", DEFAULTS.level_height_px)),
        unit_scale,
        min_px=305,
    )
    default_canvas_width = int(group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width))
    default_canvas_height = int(group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height))
    canvas_width = int(max(640, min(default_canvas_width, int(level_width_px) + 190)))
    canvas_height = int(max(500, min(default_canvas_height, int(level_height_px) + 160)))
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.font_family",
        params=params,
    )
    return PlatformerRenderParams(
        canvas_width=int(params.get("canvas_width", canvas_width)),
        canvas_height=int(params.get("canvas_height", canvas_height)),
        level_width_px=int(level_width_px),
        level_height_px=int(level_height_px),
        level_border_width_px=scale_games_px(params.get("level_border_width_px", group_default(render_defaults, "level_border_width_px", DEFAULTS.level_border_width_px)), unit_scale, min_px=2),
        platform_height_px=scale_games_px(params.get("platform_height_px", group_default(render_defaults, "platform_height_px", DEFAULTS.platform_height_px)), unit_scale, min_px=17),
        player_width_px=scale_games_px(params.get("player_width_px", group_default(render_defaults, "player_width_px", DEFAULTS.player_width_px)), unit_scale, min_px=19),
        player_height_px=scale_games_px(params.get("player_height_px", group_default(render_defaults, "player_height_px", DEFAULTS.player_height_px)), unit_scale, min_px=29),
        hazard_width_px=scale_games_px(params.get("hazard_width_px", group_default(render_defaults, "hazard_width_px", DEFAULTS.hazard_width_px)), unit_scale, min_px=27),
        hazard_height_px=scale_games_px(params.get("hazard_height_px", group_default(render_defaults, "hazard_height_px", DEFAULTS.hazard_height_px)), unit_scale, min_px=27),
        collectible_radius_px=scale_games_px(params.get("collectible_radius_px", group_default(render_defaults, "collectible_radius_px", DEFAULTS.collectible_radius_px)), unit_scale, min_px=9),
        path_width_px=scale_games_px(params.get("path_width_px", group_default(render_defaults, "path_width_px", DEFAULTS.path_width_px)), unit_scale, min_px=3),
        label_font_size_px=scale_games_px(params.get("label_font_size_px", group_default(render_defaults, "label_font_size_px", DEFAULTS.label_font_size_px)), unit_scale, min_px=12),
        font_family=str(font_family),
        layout_jitter_meta=layout_jitter,
    )


def build_games_platformer_theme(*, style_variant: str) -> PlatformerTheme:
    """Return one complete platformer palette for the requested style variant."""

    style = str(style_variant)
    if style == "cave":
        return PlatformerTheme(
            sky_rgb=(47, 43, 55),
            far_rgb=(70, 61, 82),
            level_outline_rgb=(28, 25, 32),
            ground_rgb=(94, 73, 55),
            ground_outline_rgb=(47, 35, 28),
            platform_palette_rgb=((126, 91, 68), (106, 108, 88), (143, 112, 74), (100, 86, 113)),
            platform_outline_rgb=(47, 35, 30),
            platform_text_rgb=(24, 21, 19),
            hazard_palette_rgb=((207, 72, 58), (192, 95, 54), (152, 71, 120)),
            hazard_outline_rgb=(42, 30, 29),
            hazard_text_rgb=(250, 238, 218),
            coin_palette_rgb=((238, 185, 68), (216, 136, 67), (154, 190, 198)),
            coin_outline_rgb=(60, 45, 33),
            player_rgb=(94, 183, 202),
            player_outline_rgb=(31, 39, 45),
            path_rgb=(245, 244, 231),
        )
    if style == "neon":
        return PlatformerTheme(
            sky_rgb=(21, 25, 39),
            far_rgb=(40, 46, 70),
            level_outline_rgb=(115, 234, 223),
            ground_rgb=(33, 56, 64),
            ground_outline_rgb=(105, 234, 223),
            platform_palette_rgb=((255, 89, 158), (91, 208, 255), (254, 207, 78), (139, 111, 255)),
            platform_outline_rgb=(223, 242, 255),
            platform_text_rgb=(18, 21, 28),
            hazard_palette_rgb=((255, 72, 92), (255, 142, 75), (185, 102, 255)),
            hazard_outline_rgb=(244, 249, 255),
            hazard_text_rgb=(12, 16, 24),
            coin_palette_rgb=((255, 222, 76), (95, 239, 184), (255, 117, 177)),
            coin_outline_rgb=(241, 249, 255),
            player_rgb=(92, 244, 181),
            player_outline_rgb=(241, 249, 255),
            path_rgb=(255, 246, 120),
        )
    if style == "snow":
        return PlatformerTheme(
            sky_rgb=(203, 224, 237),
            far_rgb=(173, 201, 218),
            level_outline_rgb=(72, 108, 128),
            ground_rgb=(236, 246, 249),
            ground_outline_rgb=(96, 126, 146),
            platform_palette_rgb=((180, 210, 222), (155, 184, 204), (221, 232, 236), (145, 169, 193)),
            platform_outline_rgb=(72, 105, 126),
            platform_text_rgb=(35, 58, 75),
            hazard_palette_rgb=((201, 72, 82), (229, 113, 91), (151, 88, 166)),
            hazard_outline_rgb=(64, 83, 96),
            hazard_text_rgb=(249, 244, 238),
            coin_palette_rgb=((234, 190, 72), (100, 169, 206), (180, 143, 220)),
            coin_outline_rgb=(64, 90, 112),
            player_rgb=(54, 127, 189),
            player_outline_rgb=(28, 55, 80),
            path_rgb=(62, 91, 112),
        )
    if style == "sunset":
        return PlatformerTheme(
            sky_rgb=(230, 157, 112),
            far_rgb=(180, 111, 123),
            level_outline_rgb=(89, 61, 69),
            ground_rgb=(118, 89, 72),
            ground_outline_rgb=(69, 48, 45),
            platform_palette_rgb=((188, 114, 76), (151, 104, 107), (211, 155, 84), (112, 115, 121)),
            platform_outline_rgb=(71, 49, 45),
            platform_text_rgb=(34, 27, 25),
            hazard_palette_rgb=((192, 54, 57), (222, 97, 64), (138, 77, 144)),
            hazard_outline_rgb=(58, 39, 42),
            hazard_text_rgb=(252, 234, 210),
            coin_palette_rgb=((244, 194, 71), (245, 136, 73), (112, 170, 181)),
            coin_outline_rgb=(71, 49, 43),
            player_rgb=(92, 170, 198),
            player_outline_rgb=(43, 43, 54),
            path_rgb=(252, 244, 214),
        )
    return PlatformerTheme(
        sky_rgb=(143, 197, 229),
        far_rgb=(105, 171, 210),
        level_outline_rgb=(44, 87, 117),
        ground_rgb=(83, 143, 72),
        ground_outline_rgb=(38, 82, 48),
        platform_palette_rgb=((114, 174, 84), (168, 128, 80), (84, 147, 183), (185, 145, 78)),
        platform_outline_rgb=(42, 80, 49),
        platform_text_rgb=(27, 42, 30),
        hazard_palette_rgb=((210, 62, 55), (224, 117, 54), (143, 83, 166)),
        hazard_outline_rgb=(57, 48, 42),
        hazard_text_rgb=(253, 241, 220),
        coin_palette_rgb=((241, 197, 67), (255, 222, 91), (98, 182, 204)),
        coin_outline_rgb=(74, 65, 40),
        player_rgb=(70, 123, 205),
        player_outline_rgb=(27, 48, 77),
        path_rgb=(255, 255, 246),
    )


def _level_bbox(params: PlatformerRenderParams) -> Tuple[Tuple[float, float, float, float], Dict[str, Any]]:
    """Return level bbox in canvas pixels plus resolved jitter metadata."""

    left = (float(params.canvas_width) - float(params.level_width_px)) / 2.0
    top = (float(params.canvas_height) - float(params.level_height_px)) / 2.0
    bbox = (left, top, left + float(params.level_width_px), top + float(params.level_height_px))
    if isinstance(params.layout_jitter_meta, Mapping):
        shifted, _dx, _dy, layout_jitter = apply_games_layout_jitter_to_bbox(
            bbox_px=bbox,
            canvas_width=int(params.canvas_width),
            canvas_height=int(params.canvas_height),
            jitter=params.layout_jitter_meta,
        )
        return shifted, dict(layout_jitter)
    return bbox, {}


def _to_px(level_bbox: Tuple[float, float, float, float], point: Tuple[float, float]) -> Tuple[float, float]:
    """Project normalized level coordinates to pixels."""

    left, top, right, bottom = level_bbox
    return (
        float(left + (float(point[0]) * (right - left))),
        float(top + (float(point[1]) * (bottom - top))),
    )


def _norm_box_to_px(
    level_bbox: Tuple[float, float, float, float],
    *,
    x_norm: float,
    y_norm: float,
    width_norm: float,
    height_norm: float,
) -> Tuple[float, float, float, float]:
    """Project a normalized center-size box to pixels."""

    cx, cy = _to_px(level_bbox, (float(x_norm), float(y_norm)))
    left, top, right, bottom = level_bbox
    width = float(width_norm) * float(right - left)
    height = float(height_norm) * float(bottom - top)
    return (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


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
        bold=True,
        font_family=str(font_family),
        min_size_px=7,
        max_size_px=int(max_size_px),
        fill_ratio=0.74,
    )
    text_bbox = draw.textbbox((0, 0), str(text), font=font)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    draw_text_traced(draw,
        (
            float(left + (0.5 * (float(right - left) - text_w)) - float(text_bbox[0])),
            float(top + (0.5 * (float(bottom - top) - text_h)) - float(text_bbox[1])),
        ),
        str(text),
        fill=tuple(int(v) for v in fill),
        font=font,
     role="readout", required=False,)


def _draw_background(draw: ImageDraw.ImageDraw, *, bbox: Tuple[float, float, float, float], theme: PlatformerTheme) -> None:
    """Draw the level background inside the framed play area."""

    left, top, right, bottom = bbox
    draw.rounded_rectangle(
        bbox,
        radius=30,
        fill=tuple(int(v) for v in theme.sky_rgb),
        outline=tuple(int(v) for v in theme.level_outline_rgb),
        width=5,
    )
    inset = 5.0
    for idx, y_frac in enumerate((0.30, 0.44, 0.57)):
        y = top + ((bottom - top) * y_frac)
        offset = (idx * 42) % 110
        points = [(left + inset, y + 34)]
        for step in range(9):
            x = left + inset + offset + (step * 112)
            x = max(left + inset, min(right - inset, x))
            points.append((x, y - (24 if step % 2 else 6)))
        points.extend([(right - inset, y + 42), (right - inset, bottom - inset), (left + inset, bottom - inset)])
        draw.polygon(points, fill=tuple(int(v) for v in theme.far_rgb))
    ground_top = top + ((bottom - top) * 0.88)
    draw.rectangle((left + 4, ground_top, right - 4, bottom - 4), fill=tuple(int(v) for v in theme.ground_rgb))
    draw.line((left + 8, ground_top, right - 8, ground_top), fill=tuple(int(v) for v in theme.ground_outline_rgb), width=4)


def _draw_path(
    draw: ImageDraw.ImageDraw,
    *,
    level_bbox: Tuple[float, float, float, float],
    path_points_norm: Tuple[Tuple[float, float], ...],
    visible_fraction: float,
    theme: PlatformerTheme,
    params: PlatformerRenderParams,
) -> Dict[str, Any]:
    """Draw the visible prefix of the jump arc."""

    if len(path_points_norm) < 2:
        return {"points": [], "visible_points": []}
    visible_segments = max(1, int(math.ceil((len(path_points_norm) - 1) * float(visible_fraction))))
    visible_points = path_points_norm[: visible_segments + 1]
    px_points = [_to_px(level_bbox, point) for point in path_points_norm]
    visible_px_points = [_to_px(level_bbox, point) for point in visible_points]
    for start, end in zip(visible_px_points, visible_px_points[1:]):
        draw_dashed_line(
            draw,
            start=start,
            end=end,
            fill=tuple(int(v) for v in theme.path_rgb),
            width=int(params.path_width_px),
            dash_px=12,
            gap_px=7,
        )
    if visible_px_points:
        end = visible_px_points[-1]
        r = max(4.0, float(params.path_width_px) * 0.9)
        draw.ellipse((end[0] - r, end[1] - r, end[0] + r, end[1] + r), fill=tuple(int(v) for v in theme.path_rgb))
    return {
        "points": [[round(float(x), 3), round(float(y), 3)] for x, y in px_points],
        "visible_points": [[round(float(x), 3), round(float(y), 3)] for x, y in visible_px_points],
        "visible_fraction": float(visible_fraction),
    }


def _draw_platform(
    draw: ImageDraw.ImageDraw,
    *,
    level_bbox: Tuple[float, float, float, float],
    platform: PlatformerPlatform,
    theme: PlatformerTheme,
    params: PlatformerRenderParams,
) -> Tuple[float, float, float, float]:
    """Draw one platform and return its bbox."""

    bbox = _norm_box_to_px(
        level_bbox,
        x_norm=float(platform.x_norm),
        y_norm=float(platform.y_norm),
        width_norm=float(platform.width_norm),
        height_norm=float(platform.height_norm),
    )
    fill = theme.platform_palette_rgb[int(platform.color_index) % len(theme.platform_palette_rgb)]
    draw.rounded_rectangle(
        bbox,
        radius=10,
        fill=tuple(int(v) for v in fill),
        outline=tuple(int(v) for v in theme.platform_outline_rgb),
        width=3,
    )
    draw.line((bbox[0] + 8, bbox[1] + 7, bbox[2] - 8, bbox[1] + 7), fill=(255, 255, 255), width=1)
    if str(platform.label):
        _fit_text(
            draw,
            bbox=(bbox[0] + 8, bbox[1] + 2, bbox[2] - 8, bbox[3] - 2),
            text=str(platform.label),
            fill=theme.platform_text_rgb,
            max_size_px=int(params.label_font_size_px),
            font_family=str(params.font_family),
        )
    return tuple(round(float(v), 3) for v in bbox)


def _draw_hazard(
    draw: ImageDraw.ImageDraw,
    *,
    level_bbox: Tuple[float, float, float, float],
    hazard: PlatformerHazard,
    theme: PlatformerTheme,
    params: PlatformerRenderParams,
) -> Tuple[float, float, float, float]:
    """Draw one hazard and return its bbox."""

    bbox = _norm_box_to_px(
        level_bbox,
        x_norm=float(hazard.x_norm),
        y_norm=float(hazard.y_norm),
        width_norm=float(hazard.width_norm),
        height_norm=float(hazard.height_norm),
    )
    fill = theme.hazard_palette_rgb[int(hazard.color_index) % len(theme.hazard_palette_rgb)]
    kind = str(hazard.kind)
    if kind == "patrol":
        draw.ellipse(bbox, fill=tuple(int(v) for v in fill), outline=tuple(int(v) for v in theme.hazard_outline_rgb), width=3)
        eye_y = bbox[1] + (bbox[3] - bbox[1]) * 0.38
        for x_frac in (0.38, 0.62):
            eye_x = bbox[0] + (bbox[2] - bbox[0]) * x_frac
            draw.ellipse((eye_x - 3, eye_y - 3, eye_x + 3, eye_y + 3), fill=tuple(int(v) for v in theme.hazard_outline_rgb))
    else:
        width = bbox[2] - bbox[0]
        tooth_count = 3
        for idx in range(tooth_count):
            x0 = bbox[0] + (width * idx / tooth_count)
            x1 = bbox[0] + (width * (idx + 1) / tooth_count)
            points = [(x0, bbox[3]), ((x0 + x1) / 2.0, bbox[1]), (x1, bbox[3])]
            draw.polygon(points, fill=tuple(int(v) for v in fill), outline=tuple(int(v) for v in theme.hazard_outline_rgb))
    if str(hazard.label):
        label_box = (bbox[0] + 5, bbox[1] + 4, bbox[2] - 5, bbox[3] - 4)
        _fit_text(
            draw,
            bbox=label_box,
            text=str(hazard.label),
            fill=theme.hazard_text_rgb,
            max_size_px=int(params.label_font_size_px),
            font_family=str(params.font_family),
        )
    return tuple(round(float(v), 3) for v in bbox)


def _draw_collectible(
    draw: ImageDraw.ImageDraw,
    *,
    level_bbox: Tuple[float, float, float, float],
    collectible: PlatformerCollectible,
    theme: PlatformerTheme,
    params: PlatformerRenderParams,
) -> Tuple[float, float, float, float]:
    """Draw one collectible and return its bbox."""

    cx, cy = _to_px(level_bbox, (float(collectible.x_norm), float(collectible.y_norm)))
    radius = float(params.collectible_radius_px) * (float(collectible.radius_norm) / 0.022)
    bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
    fill = theme.coin_palette_rgb[int(collectible.color_index) % len(theme.coin_palette_rgb)]
    outline = tuple(int(v) for v in theme.coin_outline_rgb)
    kind = str(collectible.kind or "coin")
    if collectible.score_value is None:
        draw.ellipse(bbox, fill=tuple(int(v) for v in fill), outline=outline, width=2)
        shine = (cx - radius * 0.35, cy - radius * 0.45, cx - radius * 0.06, cy - radius * 0.16)
        draw.ellipse(shine, fill=(255, 255, 246))
        return tuple(round(float(v), 3) for v in bbox)

    if kind == "star":
        points = []
        for index in range(10):
            angle = (-math.pi / 2.0) + (float(index) * math.pi / 5.0)
            point_radius = radius * (1.0 if index % 2 == 0 else 0.48)
            points.append((cx + (point_radius * math.cos(angle)), cy + (point_radius * math.sin(angle))))
        draw.polygon(points, fill=tuple(int(v) for v in fill), outline=outline)
    else:
        points = (
            (cx, cy - radius),
            (cx + radius * 0.92, cy),
            (cx, cy + radius),
            (cx - radius * 0.92, cy),
        )
        draw.polygon(points, fill=tuple(int(v) for v in fill), outline=outline)
        draw.line((cx - radius * 0.42, cy - radius * 0.38, cx + radius * 0.42, cy - radius * 0.38), fill=(255, 255, 246), width=2)
    text = str(int(collectible.score_value))
    font = fit_font_to_box(
        draw,
        text=text,
        max_width=max(8.0, radius * 1.22),
        max_height=max(8.0, radius * 1.02),
        bold=True,
        font_family=str(params.font_family),
        min_size_px=7,
        max_size_px=max(10, int(params.label_font_size_px)),
        fill_ratio=0.88,
    )
    draw_centered_game_text(
        draw,
        text=text,
        center=(cx, cy),
        font=font,
        fill=theme.coin_outline_rgb,
        stroke_fill=fill,
        stroke_width=2,
        role="readout",
        required=True,
        surface_rgbs=(fill,),
        preferred_rgbs=(theme.coin_outline_rgb,),
        namespace="games.platformer.collectible_score",
    )
    return tuple(round(float(v), 3) for v in bbox)


def _draw_player(
    draw: ImageDraw.ImageDraw,
    *,
    level_bbox: Tuple[float, float, float, float],
    player_xy_norm: Tuple[float, float],
    theme: PlatformerTheme,
    params: PlatformerRenderParams,
) -> Tuple[float, float, float, float]:
    """Draw the player avatar and return its bbox."""

    cx, cy = _to_px(level_bbox, player_xy_norm)
    w = float(params.player_width_px)
    h = float(params.player_height_px)
    body = (cx - w / 2.0, cy - h * 0.30, cx + w / 2.0, cy + h * 0.44)
    head_r = h * 0.18
    head = (cx - head_r, cy - h * 0.58, cx + head_r, cy - h * 0.22)
    draw.rounded_rectangle(body, radius=7, fill=tuple(int(v) for v in theme.player_rgb), outline=tuple(int(v) for v in theme.player_outline_rgb), width=3)
    draw.ellipse(head, fill=(248, 215, 176), outline=tuple(int(v) for v in theme.player_outline_rgb), width=2)
    draw.line((cx - w * 0.22, cy + h * 0.44, cx - w * 0.34, cy + h * 0.62), fill=tuple(int(v) for v in theme.player_outline_rgb), width=4)
    draw.line((cx + w * 0.22, cy + h * 0.44, cx + w * 0.34, cy + h * 0.62), fill=tuple(int(v) for v in theme.player_outline_rgb), width=4)
    return tuple(round(float(v), 3) for v in (cx - w / 2.0, cy - h * 0.60, cx + w / 2.0, cy + h * 0.66))


def render_platformer_scene(
    *,
    platforms: Tuple[PlatformerPlatform, ...],
    hazards: Tuple[PlatformerHazard, ...],
    collectibles: Tuple[PlatformerCollectible, ...],
    mode: str,
    player_xy_norm: Tuple[float, float],
    path_points_norm: Tuple[Tuple[float, float], ...],
    visible_path_fraction: float,
    background: Image.Image,
    style_variant: str,
    params: PlatformerRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedPlatformerScene:
    """Render one side-scroller level and record every entity projection."""

    image = background.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    theme = build_games_platformer_theme(style_variant=str(style_variant))
    level_bbox, layout_jitter = _level_bbox(params)
    if panel_style is not None:
        left, top, right, bottom = level_bbox
        draw_panel_scene_chrome(
            draw,
            bbox=(
                int(round(float(left - 24.0))),
                int(round(float(top - 24.0))),
                int(round(float(right + 24.0))),
                int(round(float(bottom + 24.0))),
            ),
            style=panel_style,
            radius=34,
            border_width=2,
        )
    _draw_background(draw, bbox=level_bbox, theme=theme)

    entity_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    entity_points: Dict[str, Tuple[float, float]] = {}
    scene_entities: list[Dict[str, Any]] = []
    motion_paths_px = {
        "jump_arc": _draw_path(
            draw,
            level_bbox=level_bbox,
            path_points_norm=path_points_norm,
            visible_fraction=float(visible_path_fraction),
            theme=theme,
            params=params,
        )
    }

    for platform in platforms:
        bbox = _draw_platform(draw, level_bbox=level_bbox, platform=platform, theme=theme, params=params)
        entity_bboxes[str(platform.platform_id)] = bbox
        entity_points[str(platform.platform_id)] = (
            round(float((bbox[0] + bbox[2]) / 2.0), 3),
            round(float((bbox[1] + bbox[3]) / 2.0), 3),
        )
        scene_entities.append(
            {
                "id": str(platform.platform_id),
                "type": "platformer_platform",
                "label": str(platform.label),
                "bbox": list(bbox),
                "point": list(entity_points[str(platform.platform_id)]),
            }
        )
    for hazard in hazards:
        bbox = _draw_hazard(draw, level_bbox=level_bbox, hazard=hazard, theme=theme, params=params)
        entity_bboxes[str(hazard.hazard_id)] = bbox
        entity_points[str(hazard.hazard_id)] = (
            round(float((bbox[0] + bbox[2]) / 2.0), 3),
            round(float((bbox[1] + bbox[3]) / 2.0), 3),
        )
        scene_entities.append(
            {
                "id": str(hazard.hazard_id),
                "type": "platformer_hazard",
                "label": str(hazard.label),
                "kind": str(hazard.kind),
                "bbox": list(bbox),
                "point": list(entity_points[str(hazard.hazard_id)]),
            }
        )
    for collectible in collectibles:
        bbox = _draw_collectible(draw, level_bbox=level_bbox, collectible=collectible, theme=theme, params=params)
        entity_bboxes[str(collectible.collectible_id)] = bbox
        entity_points[str(collectible.collectible_id)] = (
            round(float((bbox[0] + bbox[2]) / 2.0), 3),
            round(float((bbox[1] + bbox[3]) / 2.0), 3),
        )
        scene_entities.append(
            {
                "id": str(collectible.collectible_id),
                "type": "platformer_collectible",
                "on_path": bool(collectible.on_path),
                "kind": str(collectible.kind),
                "bbox": list(bbox),
                "point": list(entity_points[str(collectible.collectible_id)]),
            }
        )
        if collectible.score_value is not None:
            scene_entities[-1]["score_value"] = int(collectible.score_value)
            scene_entities[-1]["display_text"] = str(int(collectible.score_value))

    player_bbox = _draw_player(draw, level_bbox=level_bbox, player_xy_norm=player_xy_norm, theme=theme, params=params)
    entity_bboxes["player"] = player_bbox
    entity_points["player"] = (
        round(float((player_bbox[0] + player_bbox[2]) / 2.0), 3),
        round(float((player_bbox[1] + player_bbox[3]) / 2.0), 3),
    )
    scene_entities.append({"id": "player", "type": "platformer_player", "bbox": list(player_bbox), "point": list(entity_points["player"])})

    render_map: Dict[str, Any] = {
        "level_bbox_px": [round(float(v), 3) for v in level_bbox],
        "entity_bboxes_px": {str(key): list(value) for key, value in entity_bboxes.items()},
        "entity_points_px": {str(key): [float(x), float(y)] for key, (x, y) in entity_points.items()},
        "motion_paths_px": motion_paths_px,
        "layout_jitter": dict(layout_jitter),
        "font_family": str(params.font_family),
        "text_style": {"font_family": str(params.font_family)},
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
        "mode": str(mode),
    }
    return RenderedPlatformerScene(image=image, scene_entities=tuple(scene_entities), render_map=render_map)


def render_platformer_task_context(
    *,
    sample: PlatformerSample,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> RenderedPlatformerTaskContext:
    """Render one platformer sample and attach shared style/noise metadata."""

    render_params = resolve_platformer_render_params(
        params,
        render_defaults=render_defaults,
        namespace=str(namespace),
        instance_seed=int(instance_seed),
    )
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.panel_scene_style",
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
    rendered_scene = render_platformer_scene(
        platforms=sample.platforms,
        hazards=sample.hazards,
        collectibles=sample.collectibles,
        mode=str(sample.mode),
        player_xy_norm=(float(sample.player_x_norm), float(sample.player_y_norm)),
        path_points_norm=tuple(sample.path_points_norm),
        visible_path_fraction=float(sample.visible_path_fraction),
        background=background,
        style_variant=str(sample.style_variant),
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
    return RenderedPlatformerTaskContext(
        image=image,
        rendered_scene=rendered_scene,
        render_params=render_params,
        background_meta=dict(background_meta),
        panel_style_meta=dict(panel_style_meta),
        text_style_meta=dict(text_style_meta),
        post_noise_meta=dict(post_noise_meta),
    )


__all__ = [
    "PlatformerRenderParams",
    "RenderedPlatformerScene",
    "RenderedPlatformerTaskContext",
    "build_games_platformer_theme",
    "render_platformer_task_context",
    "render_platformer_scene",
    "resolve_platformer_render_params",
]
