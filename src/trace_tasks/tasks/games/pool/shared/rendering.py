"""Shared Pool-table renderer for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from ....shared.text_rendering import fit_font_to_box
from .defaults import DEFAULTS, POST_IMAGE_NOISE_DEFAULTS
from .rules import ball_group
from .state import POOL_POCKETS, PoolBall, PoolPocket, PoolSceneState
from ...shared.layout import apply_games_layout_jitter_to_bbox
from ...shared.layout import resolve_games_layout_jitter
from ...shared.scene_style import GamePanelSceneStyle, draw_panel_scene_chrome, game_panel_scene_style_metadata
from ...shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style
from ...shared.style import PoolTheme, build_games_pool_theme
from ...shared.text import draw_game_text_traced as draw_text_traced


@dataclass(frozen=True)
class PoolRenderParams:
    """Resolved render controls for one Pool-table scene."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    table_width_px: int
    table_height_px: int
    rail_width_px: int
    pocket_radius_px: int
    ball_radius_px: int
    ball_number_font_size_px: int
    badge_font_size_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class RenderedPoolScene:
    """Rendered Pool-table scene plus trace-friendly geometry."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class RenderedPoolTaskContext:
    """Rendered image and metadata needed for final pool task output."""

    image: Image.Image
    rendered_scene: RenderedPoolScene
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    panel_style_meta: Dict[str, Any]
    text_style_meta: Dict[str, Any]


_BALL_COLORS: Dict[int, Tuple[int, int, int]] = {
    1: (236, 197, 42),
    2: (42, 92, 190),
    3: (202, 55, 48),
    4: (94, 62, 160),
    5: (222, 116, 42),
    6: (46, 136, 82),
    7: (124, 62, 44),
    8: (22, 24, 28),
    9: (236, 197, 42),
    10: (42, 92, 190),
    11: (202, 55, 48),
    12: (94, 62, 160),
    13: (222, 116, 42),
    14: (46, 136, 82),
    15: (124, 62, 44),
}


def _norm_to_px(point: Tuple[float, float], *, table_bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Map normalized table coordinates into image pixel coordinates."""

    left, top, right, bottom = table_bbox
    return (
        float(left + (float(point[0]) * float(right - left))),
        float(top + (float(point[1]) * float(bottom - top))),
    )


def _bbox_from_center(center: Tuple[float, float], *, radius: float) -> Tuple[float, float, float, float]:
    """Return one circular-object bbox."""

    x, y = center
    r = float(radius)
    return (round(x - r, 3), round(y - r, 3), round(x + r, 3), round(y + r, 3))


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
    """Draw centered text inside a bbox."""

    left, top, right, bottom = bbox_px
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=max(1.0, float(right - left)),
        max_height=max(1.0, float(bottom - top)),
        bold=bool(bold),
        min_size_px=8,
        max_size_px=int(max_size_px),
        font_family=str(font_family),
        fill_ratio=0.84,
    )
    text_bbox = draw.textbbox((0, 0), str(text), font=font)
    tw = float(text_bbox[2] - text_bbox[0])
    th = float(text_bbox[3] - text_bbox[1])
    x = float(left + (0.5 * (float(right - left) - tw)) - text_bbox[0])
    y = float(top + (0.5 * (float(bottom - top) - th)) - text_bbox[1])
    draw_text_traced(draw, (x, y), str(text), fill=tuple(int(value) for value in fill), font=font, role="readout", required=False)


def _draw_ball(
    draw: ImageDraw.ImageDraw,
    *,
    ball: PoolBall,
    bbox_px: Tuple[float, float, float, float],
    theme: PoolTheme,
    font_size_px: int,
    font_family: str = "",
) -> None:
    """Draw one cue or object ball."""

    left, top, right, bottom = bbox_px
    width = float(right - left)
    shadow = max(2.0, 0.10 * width)
    draw.ellipse(
        (left + shadow, top + shadow, right + shadow, bottom + shadow),
        fill=tuple(int(value) for value in theme.ball_shadow_rgb) + (80,),
    )
    number = int(ball.number)
    if bool(ball.is_cue):
        draw.ellipse(bbox_px, fill=(246, 246, 238), outline=tuple(int(value) for value in theme.ball_outline_rgb), width=2)
        return
    group = ball_group(number)
    color = _BALL_COLORS.get(number, (190, 190, 190))
    if group == "stripe":
        draw.ellipse(bbox_px, fill=(248, 247, 238), outline=tuple(int(value) for value in theme.ball_outline_rgb), width=2)
        stripe_bbox = (left + (0.08 * width), top + (0.32 * width), right - (0.08 * width), bottom - (0.32 * width))
        draw.rounded_rectangle(stripe_bbox, radius=max(3, int(round(0.10 * width))), fill=color)
        draw.ellipse(bbox_px, outline=tuple(int(value) for value in theme.ball_outline_rgb), width=2)
    else:
        draw.ellipse(bbox_px, fill=color, outline=tuple(int(value) for value in theme.ball_outline_rgb), width=2)
    if str(theme.ball_rendering) == "glossy":
        shine = (left + (0.18 * width), top + (0.13 * width), left + (0.45 * width), top + (0.40 * width))
        draw.ellipse(shine, fill=(255, 255, 255, 82))
    number_box = (
        left + (0.28 * width),
        top + (0.28 * width),
        right - (0.28 * width),
        bottom - (0.28 * width),
    )
    draw.ellipse(number_box, fill=(250, 248, 240), outline=tuple(int(value) for value in theme.ball_outline_rgb), width=1)
    _draw_centered_text(
        draw,
        bbox_px=number_box,
        text=str(number),
        fill=(18, 20, 24),
        max_size_px=int(font_size_px),
        bold=True,
        font_family=str(font_family),
    )


def _draw_badge(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    table_bbox: Tuple[float, float, float, float],
    theme: PoolTheme,
    params: PoolRenderParams,
) -> Tuple[float, float, float, float] | None:
    """Draw an optional current-player badge."""

    if not str(text).strip():
        return None
    left, top, right, _bottom = table_bbox
    bbox = (
        round(float(right - 280), 3),
        round(float(top - 58), 3),
        round(float(right), 3),
        round(float(top - 14), 3),
    )
    draw.rounded_rectangle(
        bbox,
        radius=12,
        fill=tuple(int(value) for value in theme.badge_fill_rgb),
        outline=tuple(int(value) for value in theme.badge_outline_rgb),
        width=2,
    )
    _draw_centered_text(
        draw,
        bbox_px=bbox,
        text=str(text),
        fill=tuple(int(value) for value in theme.badge_text_rgb),
        max_size_px=int(params.badge_font_size_px),
        bold=True,
        font_family=str(params.font_family),
    )
    return bbox


def render_pool_table_scene(
    *,
    balls: Sequence[PoolBall],
    pockets: Sequence[PoolPocket] = POOL_POCKETS,
    background: Image.Image,
    style_variant: str,
    badge_text: str,
    marked_ball_id: str | None,
    marked_pocket_id: str | None,
    shot_path_ball_id: str | None,
    shot_path_pocket_id: str | None,
    params: PoolRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedPoolScene:
    """Render one Pool table with visible balls and optional shot indicators."""

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme = build_games_pool_theme(style_variant=str(style_variant))

    table_left = float((int(params.canvas_width) - int(params.table_width_px)) / 2)
    table_top = float((int(params.canvas_height) - int(params.table_height_px)) / 2 + 28)
    rail_bbox = (
        round(table_left, 3),
        round(table_top, 3),
        round(table_left + int(params.table_width_px), 3),
        round(table_top + int(params.table_height_px), 3),
    )
    rail_bbox, dx, dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=rail_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    if panel_style is not None:
        draw_panel_scene_chrome(
            draw,
            bbox=(
                int(round(float(rail_bbox[0] - 30.0))),
                int(round(float(rail_bbox[1] - 30.0))),
                int(round(float(rail_bbox[2] + 30.0))),
                int(round(float(rail_bbox[3] + 30.0))),
            ),
            style=panel_style,
            radius=36,
            border_width=2,
        )
    rail = float(params.rail_width_px)
    cloth_bbox = (
        round(float(rail_bbox[0] + rail), 3),
        round(float(rail_bbox[1] + rail), 3),
        round(float(rail_bbox[2] - rail), 3),
        round(float(rail_bbox[3] - rail), 3),
    )

    draw.rounded_rectangle(
        rail_bbox,
        radius=42,
        fill=tuple(int(value) for value in theme.rail_rgb),
        outline=tuple(int(value) for value in theme.rail_outline_rgb),
        width=5,
    )
    draw.rounded_rectangle(
        cloth_bbox,
        radius=24,
        fill=tuple(int(value) for value in theme.cloth_rgb),
        outline=tuple(int(value) for value in theme.cloth_line_rgb),
        width=2,
    )

    center_x = float((cloth_bbox[0] + cloth_bbox[2]) / 2)
    center_y = float((cloth_bbox[1] + cloth_bbox[3]) / 2)
    draw.line([(center_x, cloth_bbox[1] + 8), (center_x, cloth_bbox[3] - 8)], fill=tuple(int(v) for v in theme.cloth_line_rgb) + (92,), width=2)
    draw.ellipse(
        (center_x - 4, center_y - 4, center_x + 4, center_y + 4),
        fill=tuple(int(value) for value in theme.cloth_line_rgb) + (140,),
    )

    pocket_bboxes_px: Dict[str, List[float]] = {}
    pocket_centers_px: Dict[str, Tuple[float, float]] = {}
    for pocket in pockets:
        center = _norm_to_px(pocket.center, table_bbox=cloth_bbox)
        pocket_centers_px[str(pocket.pocket_id)] = center
        radius = float(params.pocket_radius_px)
        bbox = _bbox_from_center(center, radius=radius)
        pocket_bboxes_px[str(pocket.pocket_id)] = [float(value) for value in bbox]
        draw.ellipse(
            bbox,
            fill=tuple(int(value) for value in theme.pocket_rgb),
            outline=tuple(int(value) for value in theme.pocket_outline_rgb),
            width=3,
        )

    ball_centers_px: Dict[str, Tuple[float, float]] = {
        str(ball.ball_id): _norm_to_px(ball.center, table_bbox=cloth_bbox)
        for ball in balls
    }
    if shot_path_ball_id and shot_path_pocket_id and str(shot_path_ball_id) in ball_centers_px and str(shot_path_pocket_id) in pocket_centers_px:
        cue_center = ball_centers_px.get("cue_ball")
        target_center = ball_centers_px[str(shot_path_ball_id)]
        pocket_center = pocket_centers_px[str(shot_path_pocket_id)]
        if cue_center is not None:
            draw.line([cue_center, target_center, pocket_center], fill=tuple(int(value) for value in theme.shot_line_rgb) + (190,), width=5)
            draw.line([cue_center, target_center, pocket_center], fill=(255, 255, 255, 120), width=2)

    ball_bboxes_px: Dict[str, List[float]] = {}
    scene_entities: List[Dict[str, Any]] = []
    for ball in balls:
        center = ball_centers_px[str(ball.ball_id)]
        bbox = _bbox_from_center(center, radius=float(params.ball_radius_px))
        ball_bboxes_px[str(ball.ball_id)] = [float(value) for value in bbox]
        _draw_ball(
            draw,
            ball=ball,
            bbox_px=bbox,
            theme=theme,
            font_size_px=int(params.ball_number_font_size_px),
            font_family=str(params.font_family),
        )
        if str(ball.ball_id) == str(marked_ball_id):
            pad = float(params.ball_radius_px) * 0.32
            mark_bbox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
            draw.ellipse(mark_bbox, outline=tuple(int(value) for value in theme.marker_rgb), width=5)
            inner_pad = float(params.ball_radius_px) * 0.11
            draw.ellipse(
                (bbox[0] - inner_pad, bbox[1] - inner_pad, bbox[2] + inner_pad, bbox[3] + inner_pad),
                outline=(255, 255, 255, 150),
                width=2,
            )
        scene_entities.append(
            {
                "entity_id": str(ball.ball_id),
                "entity_type": "pool_ball",
                "number": int(ball.number),
                "group": str(ball.group),
                "is_cue": bool(ball.is_cue),
                "is_marked": bool(str(ball.ball_id) == str(marked_ball_id)),
                "center_norm": [float(ball.center[0]), float(ball.center[1])],
                "point_px": [round(float(center[0]), 3), round(float(center[1]), 3)],
                "bbox_px": list(bbox),
            }
        )

    if marked_pocket_id and str(marked_pocket_id) in pocket_bboxes_px:
        bbox = tuple(float(value) for value in pocket_bboxes_px[str(marked_pocket_id)])
        pad = float(params.pocket_radius_px) * 0.40
        draw.ellipse((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), outline=tuple(int(v) for v in theme.marker_rgb), width=5)

    for pocket in pockets:
        scene_entities.append(
            {
                "entity_id": str(pocket.pocket_id),
                "entity_type": "pool_pocket",
                "display_name": str(pocket.display_name),
                "center_norm": [float(pocket.center[0]), float(pocket.center[1])],
                "point_px": [
                    round(float(pocket_centers_px[str(pocket.pocket_id)][0]), 3),
                    round(float(pocket_centers_px[str(pocket.pocket_id)][1]), 3),
                ],
                "bbox_px": list(pocket_bboxes_px[str(pocket.pocket_id)]),
                "is_marked": bool(str(pocket.pocket_id) == str(marked_pocket_id)),
            }
        )

    badge_bbox = _draw_badge(
        draw,
        text=str(badge_text),
        table_bbox=rail_bbox,
        theme=theme,
        params=params,
    )
    if badge_bbox is not None:
        scene_entities.append(
            {
                "entity_id": "current_player_badge",
                "entity_type": "pool_badge",
                "text": str(badge_text),
                "bbox_px": list(badge_bbox),
            }
        )

    return RenderedPoolScene(
        image=image.convert("RGB"),
        scene_entities=tuple(scene_entities),
        render_map={
            "rail_bbox_px": list(rail_bbox),
            "cloth_bbox_px": list(cloth_bbox),
            "ball_bboxes_px": dict(ball_bboxes_px),
            "ball_points_px": {
                str(ball_id): [round(float(center[0]), 3), round(float(center[1]), 3)]
                for ball_id, center in ball_centers_px.items()
            },
            "pocket_bboxes_px": dict(pocket_bboxes_px),
            "pocket_points_px": {
                str(pocket_id): [round(float(center[0]), 3), round(float(center[1]), 3)]
                for pocket_id, center in pocket_centers_px.items()
            },
            "layout_jitter": {**dict(layout_jitter), "table_dx_px": float(dx), "table_dy_px": float(dy)},
            "font_family": str(params.font_family),
            "text_style": {"font_family": str(params.font_family)},
            "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
        },
    )


def resolve_pool_render_params(
    params: Dict[str, Any] | Any,
    *,
    render_defaults: Dict[str, Any] | Any,
    namespace: str,
    instance_seed: int,
) -> PoolRenderParams:
    """Resolve Pool rendering parameters from config/defaults."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.font_family",
        params=params,
    )
    return PoolRenderParams(
        canvas_width=int(params.get("canvas_width", group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width))),
        canvas_height=int(params.get("canvas_height", group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height))),
        panel_margin_px=int(params.get("panel_margin_px", group_default(render_defaults, "panel_margin_px", DEFAULTS.panel_margin_px))),
        table_width_px=int(params.get("table_width_px", group_default(render_defaults, "table_width_px", DEFAULTS.table_width_px))),
        table_height_px=int(params.get("table_height_px", group_default(render_defaults, "table_height_px", DEFAULTS.table_height_px))),
        rail_width_px=int(params.get("rail_width_px", group_default(render_defaults, "rail_width_px", DEFAULTS.rail_width_px))),
        pocket_radius_px=int(params.get("pocket_radius_px", group_default(render_defaults, "pocket_radius_px", DEFAULTS.pocket_radius_px))),
        ball_radius_px=int(params.get("ball_radius_px", group_default(render_defaults, "ball_radius_px", DEFAULTS.ball_radius_px))),
        ball_number_font_size_px=int(params.get("ball_number_font_size_px", group_default(render_defaults, "ball_number_font_size_px", DEFAULTS.ball_number_font_size_px))),
        badge_font_size_px=int(params.get("badge_font_size_px", group_default(render_defaults, "badge_font_size_px", DEFAULTS.badge_font_size_px))),
        font_family=str(font_family),
        layout_jitter_meta=resolve_games_layout_jitter(
            params,
            render_defaults,
            instance_seed=int(instance_seed),
            namespace=f"{str(namespace)}.layout",
        ),
    )


def render_pool_task_context(
    *,
    state: PoolSceneState,
    params: Dict[str, Any] | Any,
    render_defaults: Dict[str, Any] | Any,
    namespace: str,
    instance_seed: int,
    style_variant: str,
    badge_text: str = "",
    show_shot_path: bool = False,
) -> RenderedPoolTaskContext:
    """Render a pool scene and attach reusable style, font, and noise metadata."""

    render_params = resolve_pool_render_params(
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
    rendered_scene = render_pool_table_scene(
        balls=state.balls,
        pockets=state.pockets,
        background=background,
        style_variant=str(style_variant),
        badge_text=str(badge_text),
        marked_ball_id=state.marked_ball_id,
        marked_pocket_id=state.marked_pocket_id,
        shot_path_ball_id=state.marked_ball_id if bool(show_shot_path) else None,
        shot_path_pocket_id=state.marked_pocket_id if bool(show_shot_path) else None,
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
    return RenderedPoolTaskContext(
        image=image,
        rendered_scene=rendered_scene,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        panel_style_meta=dict(panel_style_meta),
        text_style_meta=dict(text_style_meta),
    )


__all__ = [
    "PoolRenderParams",
    "RenderedPoolScene",
    "RenderedPoolTaskContext",
    "render_pool_table_scene",
    "render_pool_task_context",
    "resolve_pool_render_params",
]
