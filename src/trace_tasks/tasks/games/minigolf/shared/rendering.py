"""Shared Mini-golf course renderer for games-domain tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

from ....shared.color_distance import min_color_distance_to_anchors, resolve_contrasting_palette
from ....shared.drawing import draw_dashed_line
from ....shared.text_rendering import fit_font_to_box
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import sample_font_family
from ...shared.text import draw_game_text_traced as draw_text_traced
from ...shared.layout import apply_games_layout_jitter_to_bbox, resolve_games_layout_jitter
from .defaults import DEFAULTS, FIRST_OBSTACLE_MODE
from .state import MinigolfObstacle, MinigolfShotOption
from ...shared.scene_style import (
    GamePanelSceneStyle,
    game_panel_contrast_anchor_colors,
    game_panel_scene_style_metadata,
)


@dataclass(frozen=True)
class MinigolfRenderParams:
    """Resolved render controls for one mini-golf course."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    course_width_px: int
    course_height_px: int
    course_border_width_px: int
    ball_radius_px: int
    hole_radius_px: int
    obstacle_radius_px: int
    path_width_px: int
    label_font_size_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class MinigolfTheme:
    """Resolved mini-golf visual theme."""

    course_fill_rgb: Tuple[int, int, int]
    course_outline_rgb: Tuple[int, int, int]
    fairway_line_rgb: Tuple[int, int, int]
    cup_fill_rgb: Tuple[int, int, int]
    cup_outline_rgb: Tuple[int, int, int]
    ball_fill_rgb: Tuple[int, int, int]
    ball_outline_rgb: Tuple[int, int, int]
    obstacle_palette_rgb: Tuple[Tuple[int, int, int], ...]
    obstacle_outline_rgb: Tuple[int, int, int]
    obstacle_text_rgb: Tuple[int, int, int]
    path_palette_rgb: Tuple[Tuple[int, int, int], ...]
    path_label_fill_rgb: Tuple[int, int, int]
    path_label_text_rgb: Tuple[int, int, int]
    flag_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedMinigolfScene:
    """Rendered Mini-golf image plus trace-friendly geometry."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


def build_games_minigolf_theme(*, style_variant: str) -> MinigolfTheme:
    """Return the full color contract for one Mini-golf course style."""

    style = str(style_variant)
    if style == "desert":
        return MinigolfTheme(
            course_fill_rgb=(191, 157, 91),
            course_outline_rgb=(93, 71, 43),
            fairway_line_rgb=(219, 190, 129),
            cup_fill_rgb=(35, 31, 29),
            cup_outline_rgb=(250, 238, 208),
            ball_fill_rgb=(252, 249, 232),
            ball_outline_rgb=(86, 67, 44),
            obstacle_palette_rgb=((142, 97, 58), (102, 139, 146), (214, 171, 89), (120, 86, 123), (86, 119, 73)),
            obstacle_outline_rgb=(73, 53, 38),
            obstacle_text_rgb=(28, 24, 20),
            path_palette_rgb=((38, 105, 154), (196, 76, 67), (62, 137, 89), (194, 126, 49), (126, 78, 150), (45, 139, 145)),
            path_label_fill_rgb=(250, 238, 207),
            path_label_text_rgb=(44, 34, 26),
            flag_rgb=(206, 61, 54),
        )
    if style == "neon":
        return MinigolfTheme(
            course_fill_rgb=(27, 43, 54),
            course_outline_rgb=(105, 242, 173),
            fairway_line_rgb=(52, 77, 89),
            cup_fill_rgb=(3, 7, 13),
            cup_outline_rgb=(239, 250, 255),
            ball_fill_rgb=(246, 255, 250),
            ball_outline_rgb=(100, 246, 215),
            obstacle_palette_rgb=((255, 86, 159), (93, 213, 255), (254, 211, 76), (154, 114, 255), (94, 244, 154)),
            obstacle_outline_rgb=(226, 242, 255),
            obstacle_text_rgb=(18, 22, 30),
            path_palette_rgb=((255, 214, 83), (96, 238, 181), (255, 111, 173), (121, 165, 255), (255, 147, 91), (198, 112, 255)),
            path_label_fill_rgb=(22, 28, 39),
            path_label_text_rgb=(245, 249, 255),
            flag_rgb=(255, 92, 176),
        )
    if style == "blueprint":
        return MinigolfTheme(
            course_fill_rgb=(218, 231, 237),
            course_outline_rgb=(46, 82, 112),
            fairway_line_rgb=(156, 187, 207),
            cup_fill_rgb=(34, 59, 80),
            cup_outline_rgb=(241, 248, 252),
            ball_fill_rgb=(255, 255, 255),
            ball_outline_rgb=(47, 78, 102),
            obstacle_palette_rgb=((128, 166, 191), (181, 143, 94), (109, 149, 130), (158, 125, 171), (185, 119, 110)),
            obstacle_outline_rgb=(43, 76, 102),
            obstacle_text_rgb=(25, 45, 62),
            path_palette_rgb=((33, 104, 171), (194, 65, 63), (46, 132, 88), (202, 143, 42), (129, 82, 164), (40, 137, 150)),
            path_label_fill_rgb=(242, 248, 252),
            path_label_text_rgb=(38, 65, 86),
            flag_rgb=(198, 64, 67),
        )
    if style == "garden":
        return MinigolfTheme(
            course_fill_rgb=(82, 147, 82),
            course_outline_rgb=(38, 85, 47),
            fairway_line_rgb=(118, 175, 104),
            cup_fill_rgb=(27, 33, 28),
            cup_outline_rgb=(236, 246, 222),
            ball_fill_rgb=(252, 252, 242),
            ball_outline_rgb=(45, 75, 55),
            obstacle_palette_rgb=((131, 100, 72), (89, 141, 175), (214, 191, 100), (122, 111, 80), (152, 108, 156)),
            obstacle_outline_rgb=(40, 72, 45),
            obstacle_text_rgb=(22, 34, 24),
            path_palette_rgb=((45, 100, 180), (207, 79, 67), (237, 171, 67), (139, 87, 174), (42, 139, 136), (186, 95, 138)),
            path_label_fill_rgb=(244, 248, 228),
            path_label_text_rgb=(32, 54, 35),
            flag_rgb=(215, 67, 62),
        )
    return MinigolfTheme(
        course_fill_rgb=(69, 139, 76),
        course_outline_rgb=(31, 80, 45),
        fairway_line_rgb=(103, 166, 92),
        cup_fill_rgb=(24, 28, 25),
        cup_outline_rgb=(240, 247, 229),
        ball_fill_rgb=(252, 252, 244),
        ball_outline_rgb=(47, 70, 55),
        obstacle_palette_rgb=((122, 96, 73), (80, 136, 170), (212, 174, 82), (142, 102, 160), (93, 130, 86)),
        obstacle_outline_rgb=(35, 66, 43),
        obstacle_text_rgb=(24, 31, 24),
        path_palette_rgb=((45, 101, 180), (203, 75, 65), (60, 145, 94), (218, 158, 60), (136, 91, 174), (48, 145, 153)),
        path_label_fill_rgb=(248, 247, 232),
        path_label_text_rgb=(30, 45, 32),
        flag_rgb=(210, 58, 54),
    )


def _fit_text(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[float, float, float, float],
    text: str,
    fill: Tuple[int, int, int],
    max_size_px: int,
    font_family: str | None = None,
) -> None:
    """Draw centered text inside one bbox."""

    left, top, right, bottom = bbox
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=max(1.0, float(right - left)),
        max_height=max(1.0, float(bottom - top)),
        bold=True,
        font_family=str(font_family or "") or None,
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


def _course_bbox(params: MinigolfRenderParams) -> Tuple[Tuple[float, float, float, float], Dict[str, Any]]:
    """Return the course bbox in canvas pixels plus resolved jitter metadata."""

    left = (float(params.canvas_width) - float(params.course_width_px)) / 2.0
    top = (float(params.canvas_height) - float(params.course_height_px)) / 2.0
    bbox = (left, top, left + float(params.course_width_px), top + float(params.course_height_px))
    if isinstance(params.layout_jitter_meta, Mapping):
        shifted, _dx, _dy, layout_jitter = apply_games_layout_jitter_to_bbox(
            bbox_px=bbox,
            canvas_width=int(params.canvas_width),
            canvas_height=int(params.canvas_height),
            jitter=params.layout_jitter_meta,
        )
        return shifted, dict(layout_jitter)
    return bbox, {}


def resolve_minigolf_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> MinigolfRenderParams:
    """Resolve Mini-golf rendering parameters from config/defaults."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font_family",
        params=params,
    )
    return MinigolfRenderParams(
        canvas_width=int(params.get("canvas_width", group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width))),
        canvas_height=int(params.get("canvas_height", group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height))),
        panel_margin_px=int(params.get("panel_margin_px", group_default(render_defaults, "panel_margin_px", DEFAULTS.panel_margin_px))),
        course_width_px=int(params.get("course_width_px", group_default(render_defaults, "course_width_px", DEFAULTS.course_width_px))),
        course_height_px=int(params.get("course_height_px", group_default(render_defaults, "course_height_px", DEFAULTS.course_height_px))),
        course_border_width_px=int(params.get("course_border_width_px", group_default(render_defaults, "course_border_width_px", DEFAULTS.course_border_width_px))),
        ball_radius_px=int(params.get("ball_radius_px", group_default(render_defaults, "ball_radius_px", DEFAULTS.ball_radius_px))),
        hole_radius_px=int(params.get("hole_radius_px", group_default(render_defaults, "hole_radius_px", DEFAULTS.hole_radius_px))),
        obstacle_radius_px=int(params.get("obstacle_radius_px", group_default(render_defaults, "obstacle_radius_px", DEFAULTS.obstacle_radius_px))),
        path_width_px=int(params.get("path_width_px", group_default(render_defaults, "path_width_px", DEFAULTS.path_width_px))),
        label_font_size_px=int(params.get("label_font_size_px", group_default(render_defaults, "label_font_size_px", DEFAULTS.label_font_size_px))),
        font_family=str(font_family),
        layout_jitter_meta=resolve_games_layout_jitter(
            params,
            render_defaults,
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.layout",
        ),
    )


def _to_px(course_bbox: Tuple[float, float, float, float], point: Tuple[float, float]) -> Tuple[float, float]:
    """Project normalized course coordinates to pixels."""

    left, top, right, bottom = course_bbox
    return (float(left + (float(point[0]) * (right - left))), float(top + (float(point[1]) * (bottom - top))))


def _draw_obstacle(
    draw: ImageDraw.ImageDraw,
    *,
    course_bbox: Tuple[float, float, float, float],
    obstacle: MinigolfObstacle,
    theme: MinigolfTheme,
    params: MinigolfRenderParams,
) -> Tuple[float, float, float, float]:
    """Draw one obstacle and return its bbox."""

    cx, cy = _to_px(course_bbox, (float(obstacle.x_norm), float(obstacle.y_norm)))
    radius = float(params.obstacle_radius_px) * (float(obstacle.radius_norm) / 0.045)
    bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
    fill = theme.obstacle_palette_rgb[int(obstacle.color_index) % len(theme.obstacle_palette_rgb)]
    kind = str(obstacle.kind)
    if kind == "sand":
        draw.ellipse(bbox, fill=tuple(int(v) for v in fill), outline=tuple(int(v) for v in theme.obstacle_outline_rgb), width=2)
        inner = (bbox[0] + radius * 0.24, bbox[1] + radius * 0.36, bbox[2] - radius * 0.18, bbox[3] - radius * 0.32)
        draw.arc(inner, start=190, end=350, fill=tuple(int(v) for v in theme.obstacle_outline_rgb), width=2)
    elif kind == "water":
        draw.ellipse(bbox, fill=tuple(int(v) for v in fill), outline=tuple(int(v) for v in theme.obstacle_outline_rgb), width=2)
        draw.arc((bbox[0] + radius * 0.18, cy - radius * 0.2, bbox[2] - radius * 0.18, cy + radius * 0.36), 190, 350, fill=(245, 250, 255), width=2)
    elif kind == "block":
        draw.rounded_rectangle(bbox, radius=max(5, int(radius * 0.24)), fill=tuple(int(v) for v in fill), outline=tuple(int(v) for v in theme.obstacle_outline_rgb), width=2)
    else:
        draw.ellipse(bbox, fill=tuple(int(v) for v in fill), outline=tuple(int(v) for v in theme.obstacle_outline_rgb), width=2)
    label_box = (cx - radius * 0.52, cy - radius * 0.52, cx + radius * 0.52, cy + radius * 0.52)
    _fit_text(
        draw,
        bbox=label_box,
        text=str(obstacle.label),
        fill=theme.obstacle_text_rgb,
        max_size_px=int(params.label_font_size_px),
        font_family=str(params.font_family),
    )
    return tuple(round(float(v), 3) for v in bbox)


def _draw_path_marker(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    label: str,
    color: Tuple[int, int, int],
    theme: MinigolfTheme,
    params: MinigolfRenderParams,
) -> Tuple[float, float, float, float]:
    """Draw one numbered shot marker and return its bbox."""

    cx, cy = float(center[0]), float(center[1])
    radius = max(16.0, float(params.label_font_size_px) * 0.78)
    bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.ellipse(bbox, fill=tuple(int(v) for v in theme.path_label_fill_rgb), outline=tuple(int(v) for v in color), width=4)
    _fit_text(
        draw,
        bbox=(bbox[0] + 3, bbox[1] + 2, bbox[2] - 3, bbox[3] - 2),
        text=str(label),
        fill=theme.path_label_text_rgb,
        max_size_px=int(params.label_font_size_px),
        font_family=str(params.font_family),
    )
    return tuple(round(float(v), 3) for v in bbox)


def _path_color_anchor_rgbs(
    *,
    theme: MinigolfTheme,
    panel_style: GamePanelSceneStyle | None,
) -> Tuple[Tuple[int, int, int], ...]:
    """Return known colors that Mini-golf cue/path colors must avoid."""

    anchors: list[Tuple[int, int, int]] = [
        tuple(int(v) for v in theme.course_fill_rgb),
        tuple(int(v) for v in theme.course_outline_rgb),
        tuple(int(v) for v in theme.fairway_line_rgb),
        tuple(int(v) for v in theme.path_label_fill_rgb),
        tuple(int(v) for v in theme.ball_fill_rgb),
        tuple(int(v) for v in theme.cup_outline_rgb),
    ]
    return tuple(game_panel_contrast_anchor_colors(panel_style, extra_colors=anchors))


def render_minigolf_scene(
    *,
    obstacles: Tuple[MinigolfObstacle, ...],
    shot_options: Tuple[MinigolfShotOption, ...],
    mode: str,
    ball_xy_norm: Tuple[float, float],
    hole_xy_norm: Tuple[float, float],
    cue_visible_fraction: float,
    hidden_paths_norm: Mapping[str, Tuple[Tuple[float, float], ...]],
    background: Image.Image,
    style_variant: str,
    params: MinigolfRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedMinigolfScene:
    """Render one course and preserve entity projections for annotation binding."""

    image = background.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    theme = build_games_minigolf_theme(style_variant=str(style_variant))
    path_color_anchors = _path_color_anchor_rgbs(theme=theme, panel_style=panel_style)
    path_palette_rgb = resolve_contrasting_palette(
        theme.path_palette_rgb,
        anchor_colors=path_color_anchors,
        min_anchor_distance=40.0,
        min_pairwise_distance=24.0,
        distance_space="lab",
    )
    cue_line_rgb = resolve_contrasting_palette(
        ((245, 246, 248),),
        anchor_colors=path_color_anchors,
        min_anchor_distance=40.0,
        min_pairwise_distance=0.0,
        distance_space="lab",
    )[0]
    course_bbox, layout_jitter = _course_bbox(params)
    left, top, right, bottom = course_bbox
    draw.rounded_rectangle(
        course_bbox,
        radius=44,
        fill=tuple(int(v) for v in theme.course_fill_rgb),
        outline=tuple(int(v) for v in theme.course_outline_rgb),
        width=int(params.course_border_width_px),
    )
    inset = 26.0
    draw.rounded_rectangle(
        (left + inset, top + inset, right - inset, bottom - inset),
        radius=34,
        outline=tuple(int(v) for v in theme.fairway_line_rgb),
        width=2,
    )
    for t in (0.25, 0.5, 0.75):
        y = top + (t * (bottom - top))
        draw.line((left + 34, y, right - 34, y), fill=tuple(int(v) for v in theme.fairway_line_rgb), width=1)

    entity_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    entity_points: Dict[str, Tuple[float, float]] = {}
    scene_entities: list[Dict[str, Any]] = []
    obstacle_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    for obstacle in obstacles:
        obstacle_center = _to_px(course_bbox, (float(obstacle.x_norm), float(obstacle.y_norm)))
        bbox = _draw_obstacle(draw, course_bbox=course_bbox, obstacle=obstacle, theme=theme, params=params)
        entity_bboxes[str(obstacle.obstacle_id)] = bbox
        entity_points[str(obstacle.obstacle_id)] = (
            round(float(obstacle_center[0]), 3),
            round(float(obstacle_center[1]), 3),
        )
        obstacle_bboxes[str(obstacle.obstacle_id)] = bbox
        scene_entities.append(
            {
                "id": str(obstacle.obstacle_id),
                "type": "minigolf_obstacle",
                "label": str(obstacle.label),
                "kind": str(obstacle.kind),
                "bbox": list(bbox),
                "point": list(entity_points[str(obstacle.obstacle_id)]),
            }
        )

    ball_px = _to_px(course_bbox, ball_xy_norm)
    hole_px = _to_px(course_bbox, hole_xy_norm)
    hole_r = float(params.hole_radius_px)
    hole_bbox = (hole_px[0] - hole_r, hole_px[1] - hole_r, hole_px[0] + hole_r, hole_px[1] + hole_r)
    draw.ellipse(hole_bbox, fill=tuple(int(v) for v in theme.cup_fill_rgb), outline=tuple(int(v) for v in theme.cup_outline_rgb), width=2)
    flag_x = hole_px[0] + hole_r * 0.85
    draw.line((flag_x, hole_px[1] - hole_r * 2.2, flag_x, hole_px[1] - hole_r * 0.1), fill=tuple(int(v) for v in theme.cup_outline_rgb), width=3)
    draw.polygon(
        [(flag_x, hole_px[1] - hole_r * 2.2), (flag_x + hole_r * 1.6, hole_px[1] - hole_r * 1.75), (flag_x, hole_px[1] - hole_r * 1.3)],
        fill=tuple(int(v) for v in theme.flag_rgb),
    )
    entity_bboxes["hole"] = tuple(round(float(v), 3) for v in hole_bbox)
    entity_points["hole"] = (round(float(hole_px[0]), 3), round(float(hole_px[1]), 3))
    scene_entities.append({"id": "hole", "type": "minigolf_hole", "bbox": list(entity_bboxes["hole"]), "point": list(entity_points["hole"])})

    ball_r = float(params.ball_radius_px)
    ball_bbox = (ball_px[0] - ball_r, ball_px[1] - ball_r, ball_px[0] + ball_r, ball_px[1] + ball_r)
    draw.ellipse(ball_bbox, fill=tuple(int(v) for v in theme.ball_fill_rgb), outline=tuple(int(v) for v in theme.ball_outline_rgb), width=3)
    entity_bboxes["ball"] = tuple(round(float(v), 3) for v in ball_bbox)
    entity_points["ball"] = (round(float(ball_px[0]), 3), round(float(ball_px[1]), 3))
    scene_entities.append({"id": "ball", "type": "minigolf_ball", "bbox": list(entity_bboxes["ball"]), "point": list(entity_points["ball"])})

    motion_paths_px: Dict[str, Dict[str, Any]] = {}
    path_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    path_point_pairs: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    if str(mode) == FIRST_OBSTACLE_MODE:
        path = hidden_paths_norm.get("shown_path", tuple())
        if len(path) >= 2:
            start = _to_px(course_bbox, path[0])
            full_end = _to_px(course_bbox, path[-1])
            visible_end = (
                start[0] + (float(cue_visible_fraction) * (full_end[0] - start[0])),
                start[1] + (float(cue_visible_fraction) * (full_end[1] - start[1])),
            )
            draw_dashed_line(
                draw,
                start=start,
                end=visible_end,
                fill=tuple(int(v) for v in cue_line_rgb),
                width=int(params.path_width_px),
                dash_px=14,
                gap_px=8,
            )
            motion_paths_px["shown_path"] = {
                "points": [list(_to_px(course_bbox, point)) for point in path],
                "visible_end": [float(visible_end[0]), float(visible_end[1])],
            }
    else:
        for path in shot_options:
            color = path_palette_rgb[int(path.color_index) % len(path_palette_rgb)]
            start = ball_px
            cue_len = min(float(params.course_width_px), float(params.course_height_px)) * 0.165
            end = (start[0] + (math.cos(float(path.angle_rad)) * cue_len), start[1] + (math.sin(float(path.angle_rad)) * cue_len))
            cue_midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
            draw_dashed_line(
                draw,
                start=start,
                end=end,
                fill=tuple(int(v) for v in color),
                width=int(params.path_width_px),
                dash_px=12,
                gap_px=7,
            )
            marker_center = (start[0] + (math.cos(float(path.angle_rad)) * (cue_len + 28.0)), start[1] + (math.sin(float(path.angle_rad)) * (cue_len + 28.0)))
            bbox = _draw_path_marker(draw, center=marker_center, label=str(path.label), color=color, theme=theme, params=params)
            entity_bboxes[str(path.path_id)] = bbox
            entity_points[str(path.path_id)] = (
                round(float(cue_midpoint[0]), 3),
                round(float(cue_midpoint[1]), 3),
            )
            path_point_pairs[str(path.path_id)] = (
                (round(float(start[0]), 3), round(float(start[1]), 3)),
                (round(float(end[0]), 3), round(float(end[1]), 3)),
            )
            path_bboxes[str(path.path_id)] = bbox
            scene_entities.append(
                {
                    "id": str(path.path_id),
                    "type": "minigolf_shot_option",
                    "label": str(path.label),
                    "bbox": list(bbox),
                    "point": list(entity_points[str(path.path_id)]),
                    "point_pair": [list(point) for point in path_point_pairs[str(path.path_id)]],
                }
            )
            hidden = hidden_paths_norm.get(str(path.path_id), tuple())
            motion_paths_px[str(path.path_id)] = {
                "points": [list(_to_px(course_bbox, point)) for point in hidden],
                "visible_end": [float(end[0]), float(end[1])],
            }

    render_map: Dict[str, Any] = {
        "course_bbox_px": [round(float(v), 3) for v in course_bbox],
        "entity_bboxes_px": {str(key): list(value) for key, value in entity_bboxes.items()},
        "entity_points_px": {str(key): list(value) for key, value in entity_points.items()},
        "obstacle_bboxes_px": {str(key): list(value) for key, value in obstacle_bboxes.items()},
        "path_bboxes_px": {str(key): list(value) for key, value in path_bboxes.items()},
        "path_point_pairs_px": {str(key): [list(point) for point in value] for key, value in path_point_pairs.items()},
        "motion_paths_px": motion_paths_px,
        "layout_jitter": dict(layout_jitter),
        "text_style": {
            "font_family": str(params.font_family),
        },
        "style_variant": str(style_variant),
        "font_family": str(params.font_family),
        "path_palette_rgb": [list(color) for color in path_palette_rgb],
        "cue_line_rgb": list(cue_line_rgb),
        "path_color_safety": {
            "distance_space": "lab",
            "min_anchor_distance_required": 40.0,
            "min_pairwise_distance_required": 24.0,
            "anchor_rgbs": [list(color) for color in path_color_anchors],
            "path_anchor_lab_distances": [
                round(float(min_color_distance_to_anchors(color, path_color_anchors, distance_space="lab")), 3)
                for color in path_palette_rgb
            ],
            "cue_anchor_lab_distance": round(
                float(min_color_distance_to_anchors(cue_line_rgb, path_color_anchors, distance_space="lab")),
                3,
            ),
            "min_path_anchor_lab_distance": round(
                min(float(min_color_distance_to_anchors(color, path_color_anchors, distance_space="lab")) for color in path_palette_rgb),
                3,
            ),
        },
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
    }
    return RenderedMinigolfScene(image=image, scene_entities=tuple(scene_entities), render_map=render_map)


__all__ = [
    "MinigolfRenderParams",
    "RenderedMinigolfScene",
    "build_games_minigolf_theme",
    "render_minigolf_scene",
    "resolve_minigolf_render_params",
]
