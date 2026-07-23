"""Shared Space-shooter renderer for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

from .....core.seed import hash64
from ....shared.config_defaults import group_default
from ....shared.font_assets import sample_font_family
from ....shared.text_rendering import fit_font_to_box
from ...shared.text import draw_game_text_traced as draw_text_traced
from ...shared.layout import apply_games_layout_jitter_to_bbox, resolve_games_layout_jitter
from ...shared.scene_style import GamePanelSceneStyle, draw_panel_scene_chrome, game_panel_scene_style_metadata
from .defaults import DEFAULTS, RENDER_DEFAULTS
from .state import SpaceEnemy, SpaceProjectile, lane_entity_id


@dataclass(frozen=True)
class SpaceShooterRenderParams:
    """Resolved render controls for one Space-shooter scene."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    playfield_width_px: int
    playfield_height_px: int
    playfield_border_width_px: int
    lane_pad_height_px: int
    lane_pad_gap_px: int
    enemy_width_px: int
    enemy_height_px: int
    projectile_width_px: int
    projectile_height_px: int
    player_ship_width_px: int
    player_ship_height_px: int
    label_font_size_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class SpaceShooterTheme:
    """Resolved Space-shooter palette for one style variant."""

    screen_fill_rgb: Tuple[int, int, int]
    screen_outline_rgb: Tuple[int, int, int]
    star_rgb: Tuple[int, int, int]
    enemy_fill_rgb: Tuple[int, int, int]
    enemy_outline_rgb: Tuple[int, int, int]
    enemy_text_rgb: Tuple[int, int, int]
    enemy_projectile_fill_rgb: Tuple[int, int, int]
    player_projectile_fill_rgb: Tuple[int, int, int]
    projectile_outline_rgb: Tuple[int, int, int]
    lane_fill_rgb: Tuple[int, int, int]
    lane_outline_rgb: Tuple[int, int, int]
    player_fill_rgb: Tuple[int, int, int]
    player_outline_rgb: Tuple[int, int, int]
    accent_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedSpaceShooterScene:
    """Rendered Space-shooter image plus trace-friendly geometry."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


def resolve_space_shooter_render_params(params: Mapping[str, Any], *, instance_seed: int) -> SpaceShooterRenderParams:
    """Resolve render parameters from scene config and caller overrides."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="games.space_shooter.font_family",
        params=params,
    )
    return SpaceShooterRenderParams(
        canvas_width=int(params.get("canvas_width", group_default(RENDER_DEFAULTS, "canvas_width", DEFAULTS.canvas_width))),
        canvas_height=int(params.get("canvas_height", group_default(RENDER_DEFAULTS, "canvas_height", DEFAULTS.canvas_height))),
        panel_margin_px=int(params.get("panel_margin_px", group_default(RENDER_DEFAULTS, "panel_margin_px", DEFAULTS.panel_margin_px))),
        playfield_width_px=int(params.get("playfield_width_px", group_default(RENDER_DEFAULTS, "playfield_width_px", DEFAULTS.playfield_width_px))),
        playfield_height_px=int(params.get("playfield_height_px", group_default(RENDER_DEFAULTS, "playfield_height_px", DEFAULTS.playfield_height_px))),
        playfield_border_width_px=int(params.get("playfield_border_width_px", group_default(RENDER_DEFAULTS, "playfield_border_width_px", DEFAULTS.playfield_border_width_px))),
        lane_pad_height_px=int(params.get("lane_pad_height_px", group_default(RENDER_DEFAULTS, "lane_pad_height_px", DEFAULTS.lane_pad_height_px))),
        lane_pad_gap_px=int(params.get("lane_pad_gap_px", group_default(RENDER_DEFAULTS, "lane_pad_gap_px", DEFAULTS.lane_pad_gap_px))),
        enemy_width_px=int(params.get("enemy_width_px", group_default(RENDER_DEFAULTS, "enemy_width_px", DEFAULTS.enemy_width_px))),
        enemy_height_px=int(params.get("enemy_height_px", group_default(RENDER_DEFAULTS, "enemy_height_px", DEFAULTS.enemy_height_px))),
        projectile_width_px=int(params.get("projectile_width_px", group_default(RENDER_DEFAULTS, "projectile_width_px", DEFAULTS.projectile_width_px))),
        projectile_height_px=int(params.get("projectile_height_px", group_default(RENDER_DEFAULTS, "projectile_height_px", DEFAULTS.projectile_height_px))),
        player_ship_width_px=int(params.get("player_ship_width_px", group_default(RENDER_DEFAULTS, "player_ship_width_px", DEFAULTS.player_ship_width_px))),
        player_ship_height_px=int(params.get("player_ship_height_px", group_default(RENDER_DEFAULTS, "player_ship_height_px", DEFAULTS.player_ship_height_px))),
        label_font_size_px=int(params.get("label_font_size_px", group_default(RENDER_DEFAULTS, "label_font_size_px", DEFAULTS.label_font_size_px))),
        font_family=str(font_family),
        layout_jitter_meta=resolve_games_layout_jitter(
            params,
            RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace="games.space_shooter.layout",
        ),
    )


def build_games_space_shooter_theme(*, style_variant: str) -> SpaceShooterTheme:
    """Return one complete Space-shooter palette for a sampled style variant."""

    style = str(style_variant)
    if style == "deep_space":
        return SpaceShooterTheme(
            screen_fill_rgb=(10, 14, 33),
            screen_outline_rgb=(88, 130, 190),
            star_rgb=(190, 210, 236),
            enemy_fill_rgb=(82, 177, 232),
            enemy_outline_rgb=(206, 238, 255),
            enemy_text_rgb=(8, 18, 32),
            enemy_projectile_fill_rgb=(242, 66, 67),
            player_projectile_fill_rgb=(70, 170, 255),
            projectile_outline_rgb=(228, 243, 255),
            lane_fill_rgb=(18, 42, 72),
            lane_outline_rgb=(119, 183, 232),
            player_fill_rgb=(86, 230, 164),
            player_outline_rgb=(211, 255, 235),
            accent_rgb=(255, 214, 113),
        )
    if style == "vector":
        return SpaceShooterTheme(
            screen_fill_rgb=(17, 19, 24),
            screen_outline_rgb=(226, 226, 214),
            star_rgb=(166, 166, 150),
            enemy_fill_rgb=(230, 230, 214),
            enemy_outline_rgb=(24, 24, 24),
            enemy_text_rgb=(18, 18, 18),
            enemy_projectile_fill_rgb=(238, 63, 57),
            player_projectile_fill_rgb=(76, 166, 245),
            projectile_outline_rgb=(242, 242, 230),
            lane_fill_rgb=(36, 38, 42),
            lane_outline_rgb=(225, 225, 210),
            player_fill_rgb=(215, 242, 120),
            player_outline_rgb=(34, 36, 26),
            accent_rgb=(242, 202, 88),
        )
    if style == "amber":
        return SpaceShooterTheme(
            screen_fill_rgb=(28, 20, 9),
            screen_outline_rgb=(222, 163, 74),
            star_rgb=(232, 178, 94),
            enemy_fill_rgb=(246, 160, 58),
            enemy_outline_rgb=(255, 223, 143),
            enemy_text_rgb=(45, 24, 8),
            enemy_projectile_fill_rgb=(237, 61, 52),
            player_projectile_fill_rgb=(66, 166, 244),
            projectile_outline_rgb=(255, 232, 182),
            lane_fill_rgb=(68, 43, 20),
            lane_outline_rgb=(221, 141, 58),
            player_fill_rgb=(255, 214, 110),
            player_outline_rgb=(255, 238, 184),
            accent_rgb=(122, 212, 255),
        )
    if style == "terminal":
        return SpaceShooterTheme(
            screen_fill_rgb=(4, 22, 14),
            screen_outline_rgb=(53, 202, 122),
            star_rgb=(89, 220, 146),
            enemy_fill_rgb=(66, 221, 122),
            enemy_outline_rgb=(190, 255, 210),
            enemy_text_rgb=(3, 30, 16),
            enemy_projectile_fill_rgb=(244, 67, 70),
            player_projectile_fill_rgb=(78, 180, 255),
            projectile_outline_rgb=(196, 255, 220),
            lane_fill_rgb=(14, 54, 32),
            lane_outline_rgb=(79, 220, 135),
            player_fill_rgb=(166, 255, 184),
            player_outline_rgb=(220, 255, 228),
            accent_rgb=(255, 226, 109),
        )
    return SpaceShooterTheme(
        screen_fill_rgb=(13, 15, 35),
        screen_outline_rgb=(152, 91, 255),
        star_rgb=(214, 207, 255),
        enemy_fill_rgb=(255, 76, 174),
        enemy_outline_rgb=(255, 211, 239),
        enemy_text_rgb=(37, 5, 32),
        enemy_projectile_fill_rgb=(242, 65, 67),
        player_projectile_fill_rgb=(68, 178, 255),
        projectile_outline_rgb=(220, 230, 255),
        lane_fill_rgb=(30, 24, 76),
        lane_outline_rgb=(132, 116, 255),
        player_fill_rgb=(69, 230, 245),
        player_outline_rgb=(211, 250, 255),
        accent_rgb=(255, 143, 87),
    )


def _entity_jitter(seed_text: str, *, magnitude: float) -> float:
    """Return a deterministic small visual jitter."""

    raw = int(hash64(0, str(seed_text))) % 10001
    centered = (float(raw) / 10000.0) - 0.5
    return float(centered * 2.0 * float(magnitude))


def _fit_text(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[float, float, float, float],
    text: str,
    fill: Tuple[int, int, int],
    max_size_px: int,
    font_family: str = "",
    bold: bool = True,
) -> None:
    """Draw centered text inside one bbox."""

    left, top, right, bottom = bbox
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=max(1.0, float(right - left)),
        max_height=max(1.0, float(bottom - top)),
        bold=bool(bold),
        min_size_px=8,
        max_size_px=int(max_size_px),
        fill_ratio=0.78,
        font_family=str(font_family) or None,
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


def _draw_enemy(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[float, float, float, float],
    enemy: SpaceEnemy,
    theme: SpaceShooterTheme,
    label_font_size_px: int,
    font_family: str = "",
    show_label: bool = True,
) -> None:
    """Draw a downward enemy fighter, with optional task-visible text."""

    left, top, right, bottom = bbox
    width = float(right - left)
    height = float(bottom - top)
    fill = tuple(int(v) for v in theme.enemy_fill_rgb)
    outline = tuple(int(v) for v in theme.enemy_outline_rgb)
    stroke = max(2, int(round(0.045 * height)))
    left_wing = [
        (left + 0.06 * width, top + 0.43 * height),
        (left + 0.34 * width, top + 0.28 * height),
        (left + 0.41 * width, top + 0.62 * height),
        (left + 0.19 * width, top + 0.75 * height),
    ]
    right_wing = [
        (right - 0.06 * width, top + 0.43 * height),
        (right - 0.34 * width, top + 0.28 * height),
        (right - 0.41 * width, top + 0.62 * height),
        (right - 0.19 * width, top + 0.75 * height),
    ]
    hull = [
        (left + 0.50 * width, top + 0.10 * height),
        (left + 0.66 * width, top + 0.31 * height),
        (left + 0.61 * width, top + 0.72 * height),
        (left + 0.50 * width, bottom - 0.04 * height),
        (left + 0.39 * width, top + 0.72 * height),
        (left + 0.34 * width, top + 0.31 * height),
    ]
    tail_left = [
        (left + 0.27 * width, top + 0.16 * height),
        (left + 0.42 * width, top + 0.18 * height),
        (left + 0.36 * width, top + 0.38 * height),
    ]
    tail_right = [
        (right - 0.27 * width, top + 0.16 * height),
        (right - 0.42 * width, top + 0.18 * height),
        (right - 0.36 * width, top + 0.38 * height),
    ]
    for poly in (left_wing, right_wing, tail_left, tail_right, hull):
        draw.polygon(poly, fill=fill, outline=outline)
        draw.line(poly + [poly[0]], fill=outline, width=stroke)
    if not bool(show_label):
        return
    display_text = str(int(enemy.score_value)) if enemy.score_value is not None else str(enemy.label)
    label_box = (left + 0.22 * width, top + 0.26 * height, right - 0.22 * width, bottom - 0.34 * height)
    _fit_text(
        draw,
        bbox=label_box,
        text=str(display_text),
        fill=theme.enemy_text_rgb,
        max_size_px=int(label_font_size_px),
        font_family=str(font_family),
    )


def _draw_projectile(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[float, float, float, float],
    projectile: SpaceProjectile,
    theme: SpaceShooterTheme,
) -> None:
    """Draw one glowing tapered laser bolt with a clear travel direction."""

    left, top, right, bottom = bbox
    cx = float((left + right) / 2.0)
    width = float(right - left)
    height = float(bottom - top)
    fill = (
        tuple(int(v) for v in theme.player_projectile_fill_rgb)
        if str(projectile.owner) == "player"
        else tuple(int(v) for v in theme.enemy_projectile_fill_rgb)
    )
    outline = tuple(int(v) for v in theme.projectile_outline_rgb)
    owner = str(projectile.owner)
    hot_core = (245, 252, 255) if owner == "player" else (255, 242, 206)
    outline_width = max(1, int(round(0.04 * width)))

    glow_half_w = max(4.5, 0.22 * width)
    glow_bbox = (
        cx - glow_half_w,
        top + (0.04 * height),
        cx + glow_half_w,
        bottom - (0.04 * height),
    )
    draw.rounded_rectangle(
        glow_bbox,
        radius=max(4, int(round(0.38 * glow_half_w))),
        fill=fill + (70,),
    )
    draw.rounded_rectangle(
        (
            cx - (0.58 * glow_half_w),
            top + (0.13 * height),
            cx + (0.58 * glow_half_w),
            bottom - (0.13 * height),
        ),
        radius=max(3, int(round(0.24 * glow_half_w))),
        fill=fill + (104,),
    )

    shell_half_w = max(3.6, 0.16 * width)
    core_half_w = max(1.7, 0.065 * width)
    if str(projectile.owner) == "player":
        shell = [
            (cx, top + 0.02 * height),
            (cx + shell_half_w, top + 0.31 * height),
            (cx + 0.72 * shell_half_w, bottom - 0.16 * height),
            (cx + 0.30 * shell_half_w, bottom - 0.03 * height),
            (cx - 0.30 * shell_half_w, bottom - 0.03 * height),
            (cx - 0.72 * shell_half_w, bottom - 0.16 * height),
            (cx - shell_half_w, top + 0.31 * height),
        ]
        core = [
            (cx, top + 0.12 * height),
            (cx + core_half_w, top + 0.36 * height),
            (cx + 0.58 * core_half_w, bottom - 0.19 * height),
            (cx, bottom - 0.07 * height),
            (cx - 0.58 * core_half_w, bottom - 0.19 * height),
            (cx - core_half_w, top + 0.36 * height),
        ]
        tail_y = bottom - 0.08 * height
        draw.line(
            (cx - 0.34 * width, bottom - 0.02 * height, cx - 0.12 * width, tail_y),
            fill=fill + (150,),
            width=outline_width,
        )
        draw.line(
            (cx + 0.34 * width, bottom - 0.02 * height, cx + 0.12 * width, tail_y),
            fill=fill + (150,),
            width=outline_width,
        )
    else:
        shell = [
            (cx - 0.30 * shell_half_w, top + 0.03 * height),
            (cx + 0.30 * shell_half_w, top + 0.03 * height),
            (cx + 0.72 * shell_half_w, top + 0.16 * height),
            (cx + shell_half_w, bottom - 0.31 * height),
            (cx, bottom - 0.02 * height),
            (cx - shell_half_w, bottom - 0.31 * height),
            (cx - 0.72 * shell_half_w, top + 0.16 * height),
        ]
        core = [
            (cx, top + 0.07 * height),
            (cx + 0.58 * core_half_w, top + 0.19 * height),
            (cx + core_half_w, bottom - 0.36 * height),
            (cx, bottom - 0.12 * height),
            (cx - core_half_w, bottom - 0.36 * height),
            (cx - 0.58 * core_half_w, top + 0.19 * height),
        ]
        tail_y = top + 0.08 * height
        draw.line(
            (cx - 0.34 * width, top + 0.02 * height, cx - 0.12 * width, tail_y),
            fill=fill + (150,),
            width=outline_width,
        )
        draw.line(
            (cx + 0.34 * width, top + 0.02 * height, cx + 0.12 * width, tail_y),
            fill=fill + (150,),
            width=outline_width,
        )

    draw.polygon(shell, fill=fill + (238,), outline=outline + (238,))
    draw.line(shell + [shell[0]], fill=outline + (232,), width=outline_width)
    draw.polygon(core, fill=hot_core + (245,))


def _draw_player(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[float, float, float, float],
    theme: SpaceShooterTheme,
) -> None:
    """Draw the player ship."""

    left, top, right, bottom = bbox
    width = float(right - left)
    height = float(bottom - top)
    points = [
        (left + 0.50 * width, top + 0.04 * height),
        (right - 0.08 * width, bottom - 0.18 * height),
        (left + 0.66 * width, bottom - 0.12 * height),
        (left + 0.50 * width, bottom - 0.34 * height),
        (left + 0.34 * width, bottom - 0.12 * height),
        (left + 0.08 * width, bottom - 0.18 * height),
    ]
    draw.polygon(points, fill=tuple(int(v) for v in theme.player_fill_rgb), outline=tuple(int(v) for v in theme.player_outline_rgb))
    draw.line(points + [points[0]], fill=tuple(int(v) for v in theme.player_outline_rgb), width=max(2, int(round(0.05 * height))))


def render_space_shooter_scene(
    *,
    lane_count: int,
    player_lane: int,
    enemies: Tuple[SpaceEnemy, ...],
    projectiles: Tuple[SpaceProjectile, ...],
    background: Image.Image,
    style_variant: str,
    params: SpaceShooterRenderParams,
    highlight_player_lane: bool = False,
    show_enemy_labels: bool = True,
    visible_enemy_label_ids: Tuple[str, ...] | None = None,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedSpaceShooterScene:
    """Render one playfield and record every object box used by annotations."""

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme = build_games_space_shooter_theme(style_variant=str(style_variant))

    margin = float(params.panel_margin_px)
    playfield_width = float(params.playfield_width_px)
    playfield_height = float(params.playfield_height_px)
    left = float((int(params.canvas_width) - int(params.playfield_width_px)) / 2.0)
    top = float((int(params.canvas_height) - int(params.playfield_height_px)) / 2.0)
    playfield_bbox = (left, top, left + playfield_width, top + playfield_height)
    if isinstance(params.layout_jitter_meta, Mapping):
        playfield_bbox, _dx, _dy, layout_jitter = apply_games_layout_jitter_to_bbox(
            bbox_px=playfield_bbox,
            canvas_width=int(params.canvas_width),
            canvas_height=int(params.canvas_height),
            jitter=params.layout_jitter_meta,
        )
        left, top, right, bottom = playfield_bbox
        playfield_width = float(right - left)
        playfield_height = float(bottom - top)
    else:
        layout_jitter = {}

    clip_left, clip_top, clip_right, clip_bottom = playfield_bbox
    panel_bbox: Tuple[int, int, int, int] | None = None
    if panel_style is not None:
        panel_pad = max(16, int(round(float(params.panel_margin_px) * 0.55)))
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
        radius=20,
        fill=tuple(int(v) for v in theme.screen_fill_rgb) + (238,),
        outline=tuple(int(v) for v in theme.screen_outline_rgb) + (255,),
        width=int(params.playfield_border_width_px),
    )
    for star_index in range(70):
        sx = float(clip_left + 20 + (int(hash64(star_index, "space.star.x")) % max(1, int(playfield_width - 40))))
        sy = float(clip_top + 16 + (int(hash64(star_index, "space.star.y")) % max(1, int(playfield_height - 96))))
        alpha = 72 + int(hash64(star_index, "space.star.a")) % 112
        draw.ellipse((sx, sy, sx + 1.8, sy + 1.8), fill=tuple(int(v) for v in theme.star_rgb) + (alpha,))

    lane_width = float(playfield_width / max(1, int(lane_count)))
    lane_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    entity_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    scene_entities: list[Dict[str, Any]] = []
    pad_top = float(clip_bottom - float(params.lane_pad_height_px) - 24.0)
    for lane in range(int(lane_count)):
        pad_left = float(clip_left + (lane * lane_width) + float(params.lane_pad_gap_px))
        pad_right = float(clip_left + ((lane + 1) * lane_width) - float(params.lane_pad_gap_px))
        bbox = (
            round(pad_left, 3),
            round(pad_top, 3),
            round(pad_right, 3),
            round(pad_top + float(params.lane_pad_height_px), 3),
        )
        lane_id = lane_entity_id(lane)
        lane_bboxes[str(lane_id)] = bbox
        entity_bboxes[str(lane_id)] = bbox
        draw.rounded_rectangle(
            bbox,
            radius=7,
            fill=tuple(int(v) for v in theme.lane_fill_rgb) + (174,),
            outline=tuple(int(v) for v in theme.lane_outline_rgb) + (228,),
            width=2,
        )
        scene_entities.append(
            {
                "entity_id": str(lane_id),
                "entity_type": "bottom_lane_pad",
                "lane": int(lane),
                "bbox_px": list(bbox),
            }
        )

    player_cx = float(clip_left + ((int(player_lane) + 0.5) * lane_width))
    player_top = float(pad_top - float(params.player_ship_height_px) - 10)
    player_bbox = (
        round(player_cx - (0.5 * float(params.player_ship_width_px)), 3),
        round(player_top, 3),
        round(player_cx + (0.5 * float(params.player_ship_width_px)), 3),
        round(player_top + float(params.player_ship_height_px), 3),
    )
    _draw_player(draw, bbox=player_bbox, theme=theme)
    player_lane_bbox = lane_bboxes.get(lane_entity_id(int(player_lane)))
    if bool(highlight_player_lane) and player_lane_bbox is not None:
        draw.rounded_rectangle(
            player_lane_bbox,
            radius=7,
            outline=tuple(int(v) for v in theme.accent_rgb) + (255,),
            width=4,
        )
    entity_bboxes["player_ship"] = player_bbox
    scene_entities.append({"entity_id": "player_ship", "entity_type": "player_ship", "lane": int(player_lane), "bbox_px": list(player_bbox)})

    def object_bbox(*, lane: int, y_slot: int, width_px: float, height_px: float, dx_frac: float, dy_px: float) -> Tuple[float, float, float, float]:
        y_fracs = (0.13, 0.24, 0.35, 0.47, 0.59, 0.70)
        cx = float(clip_left + ((int(lane) + 0.5 + float(dx_frac)) * lane_width))
        if int(y_slot) >= len(y_fracs):
            cy = float(pad_top + (0.5 * float(params.lane_pad_height_px)) + float(dy_px) * 0.25)
        else:
            slot = max(0, min(len(y_fracs) - 1, int(y_slot)))
            cy = float(clip_top + (float(y_fracs[slot]) * playfield_height) + float(dy_px))
        return (
            round(cx - (0.5 * float(width_px)), 3),
            round(cy - (0.5 * float(height_px)), 3),
            round(cx + (0.5 * float(width_px)), 3),
            round(cy + (0.5 * float(height_px)), 3),
        )

    projectile_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    for projectile in projectiles:
        bbox = object_bbox(
            lane=int(projectile.lane),
            y_slot=int(projectile.y_slot),
            width_px=float(params.projectile_width_px),
            height_px=float(params.projectile_height_px),
            dx_frac=float(projectile.dx_frac),
            dy_px=float(projectile.dy_px),
        )
        _draw_projectile(draw, bbox=bbox, projectile=projectile, theme=theme)
        projectile_bboxes[str(projectile.projectile_id)] = bbox
        entity_bboxes[str(projectile.projectile_id)] = bbox
        scene_entities.append(
            {
                "entity_id": str(projectile.projectile_id),
                "entity_type": f"{str(projectile.owner)}_projectile",
                "owner": str(projectile.owner),
                "direction": "up" if str(projectile.owner) == "player" else "down",
                "display_text": None,
                "text_visible": False,
                "lane": int(projectile.lane),
                "y_slot": int(projectile.y_slot),
                "bbox_px": list(bbox),
            }
        )

    enemy_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    visible_label_id_set = (
        None
        if visible_enemy_label_ids is None
        else {str(enemy_id) for enemy_id in visible_enemy_label_ids}
    )
    for enemy in enemies:
        bbox = object_bbox(
            lane=int(enemy.lane),
            y_slot=int(enemy.y_slot),
            width_px=float(params.enemy_width_px),
            height_px=float(params.enemy_height_px),
            dx_frac=float(enemy.dx_frac),
            dy_px=float(enemy.dy_px),
        )
        enemy_label_visible = bool(show_enemy_labels) and (
            visible_label_id_set is None or str(enemy.enemy_id) in visible_label_id_set
        )
        _draw_enemy(
            draw,
            bbox=bbox,
            enemy=enemy,
            theme=theme,
            label_font_size_px=int(params.label_font_size_px),
            font_family=str(params.font_family),
            show_label=bool(enemy_label_visible),
        )
        enemy_bboxes[str(enemy.enemy_id)] = bbox
        entity_bboxes[str(enemy.enemy_id)] = bbox
        display_text = str(int(enemy.score_value)) if enemy.score_value is not None else str(enemy.label)
        scene_entities.append(
            {
                "entity_id": str(enemy.enemy_id),
                "entity_type": "enemy_ship",
                "label": str(enemy.label),
                "score_value": None if enemy.score_value is None else int(enemy.score_value),
                "display_text": display_text if bool(enemy_label_visible) else None,
                "text_visible": bool(enemy_label_visible),
                "lane": int(enemy.lane),
                "y_slot": int(enemy.y_slot),
                "bbox_px": list(bbox),
            }
        )

    render_map = {
        "playfield_bbox_px": [round(float(v), 3) for v in playfield_bbox],
        "panel_bbox_px": None if panel_bbox is None else [int(value) for value in panel_bbox],
        "lane_bboxes_px": {str(key): list(value) for key, value in lane_bboxes.items()},
        "enemy_bboxes_px": {str(key): list(value) for key, value in enemy_bboxes.items()},
        "projectile_bboxes_px": {str(key): list(value) for key, value in projectile_bboxes.items()},
        "entity_bboxes_px": {str(key): list(value) for key, value in entity_bboxes.items()},
        "player_ship_bbox_px": list(player_bbox),
        "layout_jitter": dict(layout_jitter),
        "font_family": str(params.font_family),
        "text_style": {"font_family": str(params.font_family)},
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
        "show_enemy_labels": bool(show_enemy_labels),
        "visible_enemy_label_ids": (
            None
            if visible_enemy_label_ids is None
            else [str(enemy_id) for enemy_id in visible_enemy_label_ids]
        ),
    }
    return RenderedSpaceShooterScene(
        image=image.convert("RGB"),
        scene_entities=tuple(scene_entities),
        render_map=render_map,
    )


__all__ = [
    "RenderedSpaceShooterScene",
    "SpaceShooterRenderParams",
    "SpaceShooterTheme",
    "build_games_space_shooter_theme",
    "resolve_space_shooter_render_params",
    "render_space_shooter_scene",
]
