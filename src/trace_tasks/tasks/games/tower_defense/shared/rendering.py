"""Shared tower-defense renderer for games-domain tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_rendering import fit_font_to_box
from trace_tasks.tasks.games.shared.text import draw_centered_game_text
from ....shared.color_distance import min_color_distance_to_anchors, resolve_contrasting_palette
from .rules import (
    candidate_tower_label_from_id,
    enemy_entity_id,
    path_segment_entity_id,
)
from .defaults import DEFAULTS, RENDER_DEFAULTS
from .state import (
    BBox,
    Color,
    Point,
    RenderedTowerDefenseScene,
    SCENE_NAMESPACE,
    TowerDefenseEnemy,
    TowerDefenseRenderParams,
    TowerDefenseTheme,
    TowerDefenseTower,
)
from ...shared.layout import apply_games_layout_jitter_to_bbox, resolve_games_layout_jitter
from ...shared.marking import draw_semantic_ellipse_marker, resolve_semantic_marker_style
from ...shared.scene_style import (
    GamePanelSceneStyle,
    draw_panel_scene_chrome,
    game_panel_contrast_anchor_colors,
    game_panel_scene_style_metadata,
)
from ....shared.config_defaults import group_default


def build_games_tower_defense_theme(*, style_variant: str) -> TowerDefenseTheme:
    """Return one tower-defense map visual theme."""

    style = str(style_variant)
    if style == "desert_path":
        return TowerDefenseTheme(
            map_fill_rgb=(219, 198, 150),
            map_outline_rgb=(111, 86, 52),
            terrain_accent_rgb=(190, 160, 111),
            path_fill_rgb=(160, 116, 76),
            path_outline_rgb=(89, 66, 45),
            path_node_rgb=(185, 137, 88),
            tower_fill_rgb=(82, 107, 127),
            tower_inner_rgb=(221, 232, 236),
            tower_outline_rgb=(40, 56, 70),
            enemy_fill_rgb=(201, 56, 54),
            enemy_outline_rgb=(255, 235, 206),
            range_palette_rgb=((42, 122, 171), (85, 101, 195), (28, 146, 128)),
            terrain_pattern="stones",
        )
    if style == "blueprint_grid":
        return TowerDefenseTheme(
            map_fill_rgb=(36, 69, 102),
            map_outline_rgb=(164, 209, 231),
            terrain_accent_rgb=(69, 109, 142),
            path_fill_rgb=(96, 151, 190),
            path_outline_rgb=(217, 239, 247),
            path_node_rgb=(126, 181, 214),
            tower_fill_rgb=(235, 241, 246),
            tower_inner_rgb=(43, 77, 112),
            tower_outline_rgb=(12, 35, 63),
            enemy_fill_rgb=(232, 74, 80),
            enemy_outline_rgb=(255, 244, 214),
            range_palette_rgb=((255, 214, 88), (75, 219, 202), (245, 126, 184)),
            terrain_pattern="grid",
        )
    if style == "night_ops":
        return TowerDefenseTheme(
            map_fill_rgb=(20, 37, 41),
            map_outline_rgb=(104, 180, 156),
            terrain_accent_rgb=(41, 68, 72),
            path_fill_rgb=(68, 95, 96),
            path_outline_rgb=(152, 211, 188),
            path_node_rgb=(85, 119, 119),
            tower_fill_rgb=(79, 212, 179),
            tower_inner_rgb=(12, 30, 36),
            tower_outline_rgb=(202, 252, 235),
            enemy_fill_rgb=(255, 91, 91),
            enemy_outline_rgb=(255, 237, 197),
            range_palette_rgb=((255, 214, 74), (42, 211, 238), (236, 72, 153)),
            terrain_pattern="scanlines",
        )
    if style == "paper_map":
        return TowerDefenseTheme(
            map_fill_rgb=(244, 234, 207),
            map_outline_rgb=(93, 82, 67),
            terrain_accent_rgb=(220, 202, 166),
            path_fill_rgb=(185, 155, 103),
            path_outline_rgb=(103, 78, 45),
            path_node_rgb=(205, 173, 118),
            tower_fill_rgb=(78, 118, 133),
            tower_inner_rgb=(238, 247, 248),
            tower_outline_rgb=(42, 61, 68),
            enemy_fill_rgb=(209, 67, 60),
            enemy_outline_rgb=(255, 244, 220),
            range_palette_rgb=((50, 130, 186), (131, 99, 193), (57, 146, 107)),
            terrain_pattern="contour",
        )
    return TowerDefenseTheme(
        map_fill_rgb=(84, 139, 92),
        map_outline_rgb=(45, 80, 54),
        terrain_accent_rgb=(104, 161, 111),
        path_fill_rgb=(151, 119, 78),
        path_outline_rgb=(80, 62, 43),
        path_node_rgb=(174, 138, 88),
        tower_fill_rgb=(75, 94, 148),
        tower_inner_rgb=(232, 238, 250),
        tower_outline_rgb=(35, 45, 82),
        enemy_fill_rgb=(210, 51, 61),
        enemy_outline_rgb=(255, 240, 218),
        range_palette_rgb=((61, 132, 210), (121, 92, 204), (230, 139, 42)),
        terrain_pattern="grass",
    )


def _round_bbox(bbox: Sequence[float]) -> BBox:
    return tuple(round(float(value), 3) for value in bbox[:4])  # type: ignore[return-value]


def _circle_bbox(center: Sequence[float], radius: float) -> BBox:
    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    return _round_bbox((cx - r, cy - r, cx + r, cy + r))


def _int_default(params: Mapping[str, Any], key: str, fallback: int) -> int:
    """Resolve one integer rendering control from params, config, or fallback."""

    if str(key) in params:
        return int(params[str(key)])
    return int(group_default(RENDER_DEFAULTS, str(key), int(fallback)))


def resolve_tower_defense_render_params(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
) -> TowerDefenseRenderParams:
    """Resolve deterministic render controls for one tower-defense map."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.text_font",
        params=params,
    )
    return TowerDefenseRenderParams(
        canvas_width=_int_default(params, "canvas_width", DEFAULTS.canvas_width),
        canvas_height=_int_default(params, "canvas_height", DEFAULTS.canvas_height),
        map_width_px=_int_default(params, "map_width_px", DEFAULTS.map_width_px),
        map_height_px=_int_default(params, "map_height_px", DEFAULTS.map_height_px),
        panel_margin_px=_int_default(params, "panel_margin_px", DEFAULTS.panel_margin_px),
        path_width_px=_int_default(params, "path_width_px", DEFAULTS.path_width_px),
        path_node_radius_px=_int_default(params, "path_node_radius_px", DEFAULTS.path_node_radius_px),
        tower_radius_px=_int_default(params, "tower_radius_px", DEFAULTS.tower_radius_px),
        enemy_radius_px=_int_default(params, "enemy_radius_px", DEFAULTS.enemy_radius_px),
        range_outline_width_px=_int_default(params, "range_outline_width_px", DEFAULTS.range_outline_width_px),
        label_font_size_px=_int_default(params, "label_font_size_px", DEFAULTS.label_font_size_px),
        layout_jitter_meta=resolve_games_layout_jitter(
            params,
            RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.layout_jitter",
        ),
        font_family=str(font_family),
    )


def _local_to_global(point: Sequence[float], *, map_bbox: Sequence[float]) -> Point:
    return (
        round(float(map_bbox[0]) + float(point[0]), 3),
        round(float(map_bbox[1]) + float(point[1]), 3),
    )


def _draw_terrain_pattern(draw: ImageDraw.ImageDraw, *, bbox: BBox, theme: TowerDefenseTheme) -> None:
    left, top, right, bottom = [float(value) for value in bbox]
    accent = tuple(int(value) for value in theme.terrain_accent_rgb)
    pattern = str(theme.terrain_pattern)
    if pattern == "grid":
        for x in range(int(left) + 30, int(right), 46):
            draw.line([(x, top + 8), (x, bottom - 8)], fill=accent, width=1)
        for y in range(int(top) + 30, int(bottom), 46):
            draw.line([(left + 8, y), (right - 8, y)], fill=accent, width=1)
    elif pattern == "scanlines":
        for y in range(int(top) + 16, int(bottom), 18):
            draw.line([(left + 12, y), (right - 12, y)], fill=accent, width=1)
    elif pattern == "contour":
        for offset in range(-120, int(right - left) + 140, 72):
            points = [
                (left + offset + 0, top + 36),
                (left + offset + 42, top + 118),
                (left + offset + 18, top + 212),
                (left + offset + 78, top + 312),
                (left + offset + 48, bottom - 40),
            ]
            draw.line(points, fill=accent, width=1)
    elif pattern == "stones":
        for y in range(int(top) + 36, int(bottom), 62):
            for x in range(int(left) + 38, int(right), 84):
                shift = 26 if ((y // 62) % 2) else 0
                draw.ellipse((x + shift - 3, y - 2, x + shift + 3, y + 2), fill=accent)
    else:
        for y in range(int(top) + 30, int(bottom), 54):
            for x in range(int(left) + 28, int(right), 70):
                draw.line([(x - 6, y + 4), (x, y - 6), (x + 7, y + 5)], fill=accent, width=1)


def _draw_tower(
    draw: ImageDraw.ImageDraw,
    *,
    center: Point,
    radius: float,
    theme: TowerDefenseTheme,
    draw_inner_detail: bool = True,
) -> BBox:
    bbox = _circle_bbox(center, radius)
    draw.ellipse(
        bbox,
        fill=theme.tower_fill_rgb,
        outline=theme.tower_outline_rgb,
        width=max(2, int(round(radius * 0.16))),
    )
    if not bool(draw_inner_detail):
        return bbox
    inner_radius = max(5.0, float(radius) * 0.42)
    draw.ellipse(
        _circle_bbox(center, inner_radius),
        fill=theme.tower_inner_rgb,
        outline=theme.tower_outline_rgb,
        width=max(1, int(round(radius * 0.08))),
    )
    cx, cy = float(center[0]), float(center[1])
    barrel = (
        round(cx - radius * 0.16, 3),
        round(cy - radius * 1.02, 3),
        round(cx + radius * 0.16, 3),
        round(cy - radius * 0.28, 3),
    )
    draw.rounded_rectangle(
        barrel,
        radius=max(2, int(radius * 0.08)),
        fill=theme.tower_inner_rgb,
        outline=theme.tower_outline_rgb,
        width=max(1, int(round(radius * 0.08))),
    )
    return bbox


def _draw_candidate_label(
    draw: ImageDraw.ImageDraw,
    *,
    center: Point,
    label: str,
    radius: float,
    theme: TowerDefenseTheme,
    params: TowerDefenseRenderParams,
) -> BBox:
    """Draw a centered A-D label on one candidate tower."""

    font = fit_font_to_box(
        draw,
        text=str(label),
        max_width=max(12.0, float(radius) * 1.15),
        max_height=max(12.0, float(radius) * 1.15),
        bold=True,
        font_family=str(params.font_family) or None,
        min_size_px=10,
        max_size_px=int(params.label_font_size_px),
        fill_ratio=0.88,
    )
    text_bbox = draw_centered_game_text(
        draw,
        text=str(label),
        center=(float(center[0]), float(center[1])),
        font=font,
        fill=(20, 24, 31),
        stroke_fill=(255, 255, 255),
        stroke_width=2,
        role="candidate_label",
        required=True,
        surface_rgbs=(theme.tower_fill_rgb, theme.tower_inner_rgb),
        preferred_rgbs=((20, 24, 31),),
        instance_seed=int(round(float(center[0]) * 19.0 + float(center[1]) * 23.0)),
        namespace="games.tower_defense.candidate_label",
    )
    return _round_bbox(text_bbox)


def _draw_enemy(
    draw: ImageDraw.ImageDraw,
    *,
    center: Point,
    radius: float,
    theme: TowerDefenseTheme,
    draw_eyes: bool = True,
) -> BBox:
    bbox = _circle_bbox(center, radius)
    draw.ellipse(
        bbox,
        fill=theme.enemy_fill_rgb,
        outline=theme.enemy_outline_rgb,
        width=max(2, int(round(radius * 0.18))),
    )
    if not bool(draw_eyes):
        return bbox
    cx, cy = float(center[0]), float(center[1])
    eye_r = max(1.5, float(radius) * 0.14)
    for dx in (-0.32 * float(radius), 0.32 * float(radius)):
        draw.ellipse(_circle_bbox((cx + dx, cy - 0.12 * float(radius)), eye_r), fill=(20, 24, 31))
    return bbox


def _draw_enemy_label(
    draw: ImageDraw.ImageDraw,
    *,
    center: Point,
    label: str,
    radius: float,
    theme: TowerDefenseTheme,
    params: TowerDefenseRenderParams,
) -> BBox:
    """Draw one A-F option label inside a path enemy marker."""

    font = fit_font_to_box(
        draw,
        text=str(label),
        max_width=max(12.0, float(radius) * 1.25),
        max_height=max(12.0, float(radius) * 1.25),
        bold=True,
        font_family=str(params.font_family) or None,
        min_size_px=10,
        max_size_px=int(params.label_font_size_px),
        fill_ratio=0.86,
    )
    text_bbox = draw_centered_game_text(
        draw,
        text=str(label),
        center=(float(center[0]), float(center[1])),
        font=font,
        fill=(18, 24, 33),
        stroke_fill=(255, 250, 235),
        stroke_width=2,
        role="enemy_option_label",
        required=True,
        surface_rgbs=(theme.enemy_fill_rgb, theme.enemy_outline_rgb),
        preferred_rgbs=((18, 24, 33),),
        instance_seed=int(round(float(center[0]) * 29.0 + float(center[1]) * 31.0)),
        namespace="games.tower_defense.enemy_option_label",
    )
    return _round_bbox(text_bbox)


def _draw_exit_marker(
    draw: ImageDraw.ImageDraw,
    *,
    final_point: Point,
    previous_point: Point,
    map_bbox: BBox,
    theme: TowerDefenseTheme,
    params: TowerDefenseRenderParams,
) -> BBox:
    """Draw a compact exit badge just beyond the final path point."""

    dx = float(final_point[0]) - float(previous_point[0])
    dy = float(final_point[1]) - float(previous_point[1])
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / length, dy / length
    raw_x = float(final_point[0]) + (ux * 34.0)
    raw_y = float(final_point[1]) + (uy * 34.0)
    left, top, right, bottom = [float(value) for value in map_bbox]
    cx = max(left + 44.0, min(right - 44.0, raw_x))
    cy = max(top + 20.0, min(bottom - 20.0, raw_y))
    badge_bbox = _round_bbox((cx - 36.0, cy - 15.0, cx + 36.0, cy + 15.0))
    draw.rounded_rectangle(
        badge_bbox,
        radius=10,
        fill=(222, 247, 229),
        outline=theme.map_outline_rgb,
        width=2,
    )
    font = fit_font_to_box(
        draw,
        text="EXIT",
        max_width=56,
        max_height=18,
        bold=True,
        font_family=str(params.font_family) or None,
        min_size_px=8,
        max_size_px=16,
        fill_ratio=0.90,
    )
    text_bbox = draw_centered_game_text(
        draw,
        text="EXIT",
        center=(cx, cy),
        font=font,
        fill=theme.map_outline_rgb,
        stroke_fill=(222, 247, 229),
        stroke_width=1,
        role="exit_marker",
        required=True,
        surface_rgbs=(theme.map_fill_rgb,),
        preferred_rgbs=(theme.map_outline_rgb,),
        instance_seed=int(round(cx * 17.0 + cy * 19.0)),
        namespace="games.tower_defense.exit_marker",
    )
    return _round_bbox(
        (
            min(badge_bbox[0], text_bbox[0]),
            min(badge_bbox[1], text_bbox[1]),
            max(badge_bbox[2], text_bbox[2]),
            max(badge_bbox[3], text_bbox[3]),
        )
    )


def render_tower_defense_scene(
    *,
    path_points_px: Sequence[Point],
    towers: Sequence[TowerDefenseTower],
    enemy: TowerDefenseEnemy | None,
    labeled_path_enemy_options: Sequence[tuple[int, str]] | Mapping[int, str] = (),
    show_exit_marker: bool = False,
    background: Image.Image,
    style_variant: str,
    params: TowerDefenseRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedTowerDefenseScene:
    """Render the complete tower-defense map and record annotation projections."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    theme = build_games_tower_defense_theme(style_variant=str(style_variant))
    base_bbox = (
        0.5 * (float(params.canvas_width) - float(params.map_width_px)),
        0.5 * (float(params.canvas_height) - float(params.map_height_px)),
        0.5 * (float(params.canvas_width) + float(params.map_width_px)),
        0.5 * (float(params.canvas_height) + float(params.map_height_px)),
    )
    map_bbox, _dx, _dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=base_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    left, top, right, bottom = [float(value) for value in map_bbox]
    chrome_bbox = (
        int(round(left - 24.0)),
        int(round(top - 24.0)),
        int(round(right + 24.0)),
        int(round(bottom + 24.0)),
    )
    if panel_style is not None:
        draw_panel_scene_chrome(
            draw,
            bbox=chrome_bbox,
            style=panel_style,
            radius=24,
            border_width=2,
        )
    draw.rounded_rectangle(
        map_bbox,
        radius=20,
        fill=theme.map_fill_rgb,
        outline=theme.map_outline_rgb,
        width=4,
    )
    _draw_terrain_pattern(draw, bbox=map_bbox, theme=theme)

    global_path_points = [_local_to_global(point, map_bbox=map_bbox) for point in path_points_px]
    if isinstance(labeled_path_enemy_options, Mapping):
        label_by_path_index = {int(index): str(label) for index, label in labeled_path_enemy_options.items()}
    else:
        label_by_path_index = {int(index): str(label) for index, label in labeled_path_enemy_options}
    entities: list[Dict[str, Any]] = []
    entity_bboxes: Dict[str, BBox] = {}
    entity_points: Dict[str, Point] = {}

    # Range rings sit under the path, optional enemy, and towers.
    anchor_colors = game_panel_contrast_anchor_colors(
        panel_style,
        extra_colors=(
            theme.map_fill_rgb,
            theme.map_outline_rgb,
            theme.terrain_accent_rgb,
            theme.path_fill_rgb,
            theme.path_outline_rgb,
            theme.path_node_rgb,
            theme.tower_fill_rgb,
            theme.tower_inner_rgb,
            theme.tower_outline_rgb,
            theme.enemy_fill_rgb,
        ),
    )
    range_palette = resolve_contrasting_palette(
        theme.range_palette_rgb,
        anchor_colors=anchor_colors,
        min_anchor_distance=55.0,
        min_pairwise_distance=24.0,
        distance_space="lab",
    )
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    range_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    range_draw = ImageDraw.Draw(range_layer)
    range_bboxes: Dict[str, BBox] = {}
    for index, tower in enumerate(towers):
        center = _local_to_global(tower.center_px, map_bbox=map_bbox)
        radius = float(tower.range_radius_px)
        color = tuple(int(value) for value in range_palette[int(index) % len(range_palette)])
        range_bbox = _circle_bbox(center, radius)
        range_draw.ellipse(
            range_bbox,
            fill=None,
            outline=(*color, 220),
            width=max(4, int(params.range_outline_width_px)),
        )
        range_bboxes[str(tower.tower_id)] = range_bbox
    map_mask = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(map_mask)
    mask_draw.rounded_rectangle(map_bbox, radius=20, fill=255)
    overlay = Image.composite(range_layer, overlay, map_mask)
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    if len(global_path_points) >= 2:
        draw.line(
            global_path_points,
            fill=theme.path_outline_rgb,
            width=max(4, int(params.path_width_px) + 8),
            joint="curve",
        )
        draw.line(
            global_path_points,
            fill=theme.path_fill_rgb,
            width=max(2, int(params.path_width_px)),
            joint="curve",
        )
    exit_marker_bbox = None
    if bool(show_exit_marker) and len(global_path_points) >= 2:
        exit_marker_bbox = _draw_exit_marker(
            draw,
            final_point=global_path_points[-1],
            previous_point=global_path_points[-2],
            map_bbox=_round_bbox(map_bbox),
            theme=theme,
            params=params,
        )
    for index, point in enumerate(global_path_points):
        option_label = label_by_path_index.get(int(index))
        halo_bbox = _circle_bbox(point, float(params.path_node_radius_px) + 4.0)
        draw.ellipse(halo_bbox, fill=theme.map_fill_rgb, outline=theme.path_outline_rgb, width=2)
        node_bbox = _draw_enemy(
            draw,
            center=point,
            radius=max(7.0, float(params.path_node_radius_px)),
            theme=theme,
            draw_eyes=option_label is None,
        )
        label_bbox = None
        if option_label is not None:
            label_bbox = _draw_enemy_label(
                draw,
                center=point,
                label=str(option_label),
                radius=max(7.0, float(params.path_node_radius_px)),
                theme=theme,
                params=params,
            )
        entity_id = path_segment_entity_id(index)
        entity_bboxes[entity_id] = node_bbox
        entity_points[entity_id] = (round(float(point[0]), 3), round(float(point[1]), 3))
        path_entity = {
            "entity_id": entity_id,
            "type": "path_enemy",
            "path_index": int(index),
            "bbox_px": list(node_bbox),
            "point_px": list(entity_points[entity_id]),
        }
        if option_label is not None:
            path_entity["option_label"] = str(option_label)
            path_entity["label_bbox_px"] = list(label_bbox or node_bbox)
        entities.append(path_entity)

    tower_radius = max(8.0, float(params.tower_radius_px))
    for tower in towers:
        center = _local_to_global(tower.center_px, map_bbox=map_bbox)
        candidate_label = candidate_tower_label_from_id(str(tower.tower_id))
        bbox = _draw_tower(
            draw,
            center=center,
            radius=tower_radius,
            theme=theme,
            draw_inner_detail=candidate_label is None,
        )
        label_bbox = None
        if candidate_label is not None:
            label_bbox = _draw_candidate_label(
                draw,
                center=center,
                label=str(candidate_label),
                radius=tower_radius,
                theme=theme,
                params=params,
            )
        entity_bboxes[str(tower.tower_id)] = bbox
        entity_points[str(tower.tower_id)] = (round(float(center[0]), 3), round(float(center[1]), 3))
        tower_entity = {
            "entity_id": str(tower.tower_id),
            "type": "tower",
            "bbox_px": list(bbox),
            "point_px": list(entity_points[str(tower.tower_id)]),
            "range_bbox_px": list(range_bboxes[str(tower.tower_id)]),
            "range_radius_px": round(float(tower.range_radius_px), 3),
            "covers_marked_enemy": bool(tower.covers_target),
        }
        if candidate_label is not None:
            tower_entity["candidate_label"] = str(candidate_label)
            tower_entity["label_bbox_px"] = list(label_bbox or bbox)
        entities.append(tower_entity)

    marked_enemy_id = None
    if enemy is not None:
        enemy_center = _local_to_global(enemy.center_px, map_bbox=map_bbox)
        enemy_bbox = _draw_enemy(
            draw,
            center=enemy_center,
            radius=max(7.0, float(params.enemy_radius_px)),
            theme=theme,
        )
        marker_style = resolve_semantic_marker_style(
            instance_seed=int(round(enemy_center[0] * 13.0 + enemy_center[1] * 17.0)),
            namespace="games.tower_defense.marked_enemy",
            role="marked_enemy",
            surface_rgbs=(theme.path_fill_rgb, theme.map_fill_rgb),
            preferred_rgbs=((239, 68, 68), (255, 214, 74), (255, 255, 255)),
            required=True,
        )
        marker_bbox = _circle_bbox(enemy_center, max(14.0, float(params.enemy_radius_px) + 8.0))
        draw_semantic_ellipse_marker(
            draw,
            marker_bbox,
            style=marker_style,
            width=4,
            marker_kind="marked_enemy_ring",
            extra_metadata={"entity_id": enemy_entity_id()},
        )
        marked_enemy_id = enemy_entity_id()
        entity_bboxes[enemy_entity_id()] = enemy_bbox
        entity_points[enemy_entity_id()] = (round(float(enemy_center[0]), 3), round(float(enemy_center[1]), 3))
        entities.append(
            {
                "entity_id": enemy_entity_id(),
                "type": "marked_enemy",
                "path_index": int(enemy.path_index),
                "bbox_px": list(enemy_bbox),
                "marker_bbox_px": list(marker_bbox),
                "point_px": list(entity_points[enemy_entity_id()]),
            }
        )

    render_map = {
        "map_bbox_px": list(_round_bbox(map_bbox)),
        "layout_jitter": dict(layout_jitter),
        "path_points_px": [list(point) for point in global_path_points],
        "entity_bboxes_px": {str(key): list(value) for key, value in sorted(entity_bboxes.items())},
        "entity_points_px": {str(key): list(value) for key, value in sorted(entity_points.items())},
        "tower_range_bboxes_px": {str(key): list(value) for key, value in sorted(range_bboxes.items())},
        "tower_range_radii_px": {str(tower.tower_id): round(float(tower.range_radius_px), 3) for tower in towers},
        "marked_enemy_id": marked_enemy_id,
        "labeled_path_enemy_options": [
            {
                "label": str(label),
                "path_index": int(index),
                "entity_id": path_segment_entity_id(int(index)),
            }
            for index, label in sorted(label_by_path_index.items())
        ],
        "exit_marker_bbox_px": [] if exit_marker_bbox is None else list(exit_marker_bbox),
        "panel_scene_style": game_panel_scene_style_metadata(panel_style) if panel_style is not None else {},
        "tower_defense_style": {
            "style_variant": str(style_variant),
            "terrain_pattern": str(theme.terrain_pattern),
            "font_family": str(params.font_family),
            "font_asset": get_font_family_record(str(params.font_family)).to_trace(),
            "range_palette_rgb": [list(color) for color in range_palette],
            "range_anchor_lab_distances": [
                round(float(min_color_distance_to_anchors(color, anchor_colors, distance_space="lab")), 3)
                for color in range_palette
            ],
            "range_clip_policy": "clipped_to_map_rounded_rect",
        },
    }
    return RenderedTowerDefenseScene(
        image=image,
        scene_entities=tuple(entities),
        render_map=render_map,
    )


__all__ = [
    "RenderedTowerDefenseScene",
    "TowerDefenseRenderParams",
    "TowerDefenseTheme",
    "build_games_tower_defense_theme",
    "render_tower_defense_scene",
    "resolve_tower_defense_render_params",
]
