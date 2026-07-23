"""Shared racing-track renderer for games-domain tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from ....shared.text_rendering import load_font
from trace_tasks.tasks.games.shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import group_default

from ...shared.layout import apply_games_layout_jitter_to_bbox, resolve_games_layout_jitter
from .defaults import DEFAULTS, SCENE_ID
from .state import Point, RacingTrackCar, RacingTrackSceneState
from ...shared.scene_style import (
    GamePanelSceneStyle,
    draw_panel_scene_chrome,
    game_panel_scene_style_metadata,
)
from ...shared.text import draw_centered_game_text_traced


Color = Tuple[int, int, int]
BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class RacingTrackRenderParams:
    """Resolved render controls for one racing-track scene."""

    canvas_width: int
    canvas_height: int
    track_width_px: int
    track_height_px: int
    road_width_px: int
    road_border_width_px: int
    car_length_px: int
    car_width_px: int
    label_font_size_px: int
    marked_outline_width_px: int = 5
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class RacingTrackTheme:
    """Resolved racing-track visual theme."""

    infield_rgb: Color
    road_rgb: Color
    road_outline_rgb: Color
    lane_mark_rgb: Color
    finish_light_rgb: Color
    finish_dark_rgb: Color
    arrow_rgb: Color
    car_palette_rgb: Tuple[Color, ...]
    car_outline_rgb: Color
    label_fill_rgb: Color
    label_text_rgb: Color
    marked_car_outline_rgb: Color
    terrain_pattern: str


@dataclass(frozen=True)
class RenderedRacingTrackScene:
    """Rendered racing-track image plus trace-friendly geometry."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class RenderedRacingTrackTaskContext:
    """Rendered racing-track image plus background/noise metadata."""

    image: Image.Image
    rendered_scene: RenderedRacingTrackScene
    background_meta: Dict[str, Any]
    panel_style_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def resolve_racing_track_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    font_family: str,
    namespace: str = "games.racing_track.layout_jitter",
) -> RacingTrackRenderParams:
    """Resolve canvas, track, car, text, and layout-jitter render controls."""

    return RacingTrackRenderParams(
        canvas_width=int(params.get("canvas_width", group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width))),
        canvas_height=int(params.get("canvas_height", group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height))),
        track_width_px=int(params.get("track_width_px", group_default(render_defaults, "track_width_px", DEFAULTS.track_width_px))),
        track_height_px=int(params.get("track_height_px", group_default(render_defaults, "track_height_px", DEFAULTS.track_height_px))),
        road_width_px=int(params.get("road_width_px", group_default(render_defaults, "road_width_px", DEFAULTS.road_width_px))),
        road_border_width_px=int(params.get("road_border_width_px", group_default(render_defaults, "road_border_width_px", DEFAULTS.road_border_width_px))),
        car_length_px=int(params.get("car_length_px", group_default(render_defaults, "car_length_px", DEFAULTS.car_length_px))),
        car_width_px=int(params.get("car_width_px", group_default(render_defaults, "car_width_px", DEFAULTS.car_width_px))),
        marked_outline_width_px=int(params.get("marked_outline_width_px", group_default(render_defaults, "marked_outline_width_px", DEFAULTS.marked_outline_width_px))),
        label_font_size_px=int(params.get("label_font_size_px", group_default(render_defaults, "label_font_size_px", DEFAULTS.label_font_size_px))),
        font_family=str(font_family),
        layout_jitter_meta=resolve_games_layout_jitter(
            params,
            render_defaults,
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        ),
    )


def build_games_racing_track_theme(*, style_variant: str) -> RacingTrackTheme:
    """Return one racing-track theme with road, car, label, and terrain colors."""

    style = str(style_variant)
    if style == "rally_sand":
        return RacingTrackTheme(
            infield_rgb=(214, 185, 128),
            road_rgb=(118, 89, 61),
            road_outline_rgb=(76, 54, 38),
            lane_mark_rgb=(224, 198, 150),
            finish_light_rgb=(245, 237, 215),
            finish_dark_rgb=(48, 39, 34),
            arrow_rgb=(33, 103, 146),
            car_palette_rgb=((197, 54, 53), (43, 111, 178), (59, 143, 86), (221, 152, 48), (125, 82, 166), (36, 138, 139), (210, 93, 132)),
            car_outline_rgb=(43, 34, 28),
            label_fill_rgb=(246, 237, 212),
            label_text_rgb=(34, 28, 24),
            marked_car_outline_rgb=(230, 35, 35),
            terrain_pattern="dust",
        )
    if style == "neon_night":
        return RacingTrackTheme(
            infield_rgb=(21, 31, 45),
            road_rgb=(42, 49, 69),
            road_outline_rgb=(122, 244, 216),
            lane_mark_rgb=(224, 245, 255),
            finish_light_rgb=(237, 251, 255),
            finish_dark_rgb=(12, 18, 28),
            arrow_rgb=(255, 214, 72),
            car_palette_rgb=((255, 79, 141), (90, 207, 255), (94, 240, 158), (255, 190, 68), (170, 120, 255), (255, 111, 89), (80, 232, 218)),
            car_outline_rgb=(236, 249, 255),
            label_fill_rgb=(18, 25, 38),
            label_text_rgb=(245, 250, 255),
            marked_car_outline_rgb=(255, 45, 45),
            terrain_pattern="scanlines",
        )
    if style == "blueprint_track":
        return RacingTrackTheme(
            infield_rgb=(45, 78, 111),
            road_rgb=(80, 127, 160),
            road_outline_rgb=(205, 231, 245),
            lane_mark_rgb=(236, 247, 252),
            finish_light_rgb=(240, 249, 252),
            finish_dark_rgb=(26, 55, 82),
            arrow_rgb=(255, 207, 78),
            car_palette_rgb=((230, 82, 84), (245, 196, 73), (75, 203, 172), (112, 158, 236), (197, 113, 211), (247, 139, 86), (120, 216, 103)),
            car_outline_rgb=(18, 42, 66),
            label_fill_rgb=(240, 248, 252),
            label_text_rgb=(24, 50, 74),
            marked_car_outline_rgb=(230, 35, 35),
            terrain_pattern="grid",
        )
    if style == "paper_race":
        return RacingTrackTheme(
            infield_rgb=(235, 225, 198),
            road_rgb=(128, 132, 129),
            road_outline_rgb=(76, 73, 66),
            lane_mark_rgb=(246, 240, 220),
            finish_light_rgb=(248, 244, 230),
            finish_dark_rgb=(42, 39, 35),
            arrow_rgb=(184, 62, 58),
            car_palette_rgb=((195, 70, 64), (65, 116, 169), (67, 139, 89), (214, 152, 55), (133, 91, 158), (57, 142, 144), (196, 95, 126)),
            car_outline_rgb=(56, 51, 43),
            label_fill_rgb=(250, 246, 230),
            label_text_rgb=(36, 31, 25),
            marked_car_outline_rgb=(220, 30, 30),
            terrain_pattern="paper",
        )
    return RacingTrackTheme(
        infield_rgb=(74, 139, 85),
        road_rgb=(82, 89, 96),
        road_outline_rgb=(39, 48, 55),
        lane_mark_rgb=(232, 236, 224),
        finish_light_rgb=(248, 248, 242),
        finish_dark_rgb=(30, 34, 38),
        arrow_rgb=(222, 178, 51),
        car_palette_rgb=((207, 55, 58), (55, 120, 204), (53, 158, 92), (231, 159, 53), (139, 92, 197), (42, 151, 157), (216, 92, 130)),
        car_outline_rgb=(31, 38, 44),
        label_fill_rgb=(250, 250, 238),
        label_text_rgb=(30, 34, 38),
        marked_car_outline_rgb=(230, 35, 35),
        terrain_pattern="grass",
    )


def _round_bbox(bbox: Sequence[float]) -> BBox:
    return tuple(round(float(value), 3) for value in bbox[:4])  # type: ignore[return-value]


def _local_to_global(point: Sequence[float], *, track_bbox: Sequence[float]) -> Point:
    return (
        round(float(track_bbox[0]) + float(point[0]), 3),
        round(float(track_bbox[1]) + float(point[1]), 3),
    )


def _global_car(car: RacingTrackCar, *, track_bbox: Sequence[float]) -> RacingTrackCar:
    return RacingTrackCar(
        car_id=str(car.car_id),
        label=str(car.label),
        progress=float(car.progress),
        center_px=_local_to_global(car.center_px, track_bbox=track_bbox),
        tangent_px=car.tangent_px,
        remaining_distance=float(car.remaining_distance),
    )


def _polygon_bbox(points: Sequence[Point]) -> BBox:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return _round_bbox((min(xs), min(ys), max(xs), max(ys)))


def _draw_terrain_pattern(draw: ImageDraw.ImageDraw, *, bbox: BBox, theme: RacingTrackTheme) -> None:
    left, top, right, bottom = [float(value) for value in bbox]
    if theme.terrain_pattern == "grid":
        accent = (64, 99, 130)
        for x in range(int(left) + 30, int(right), 46):
            draw.line([(x, top + 10), (x, bottom - 10)], fill=accent, width=1)
        for y in range(int(top) + 30, int(bottom), 46):
            draw.line([(left + 10, y), (right - 10, y)], fill=accent, width=1)
    elif theme.terrain_pattern == "scanlines":
        for y in range(int(top) + 18, int(bottom), 22):
            draw.line([(left + 14, y), (right - 14, y)], fill=(31, 45, 63), width=1)
    elif theme.terrain_pattern == "dust":
        for y in range(int(top) + 38, int(bottom), 62):
            for x in range(int(left) + 42, int(right), 86):
                draw.ellipse((x - 3, y - 2, x + 3, y + 2), fill=(190, 158, 107))
    elif theme.terrain_pattern == "paper":
        for y in range(int(top) + 28, int(bottom), 52):
            draw.line([(left + 16, y), (right - 16, y)], fill=(219, 207, 176), width=1)
    else:
        for y in range(int(top) + 30, int(bottom), 56):
            for x in range(int(left) + 28, int(right), 76):
                draw.line([(x - 6, y + 4), (x, y - 6), (x + 7, y + 5)], fill=(95, 160, 102), width=1)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    start: Point,
    tangent: Point,
    color: Color,
    width: int,
) -> BBox:
    length = 76.0
    sx, sy = float(start[0]), float(start[1])
    tx, ty = float(tangent[0]), float(tangent[1])
    end = (sx + (tx * length), sy + (ty * length))
    draw.line([(sx, sy), end], fill=color, width=int(width))
    nx, ny = -ty, tx
    head_len = 18.0
    head_w = 12.0
    head = (
        end,
        (end[0] - (tx * head_len) + (nx * head_w), end[1] - (ty * head_len) + (ny * head_w)),
        (end[0] - (tx * head_len) - (nx * head_w), end[1] - (ty * head_len) - (ny * head_w)),
    )
    draw.polygon(head, fill=color)
    return _polygon_bbox(((sx, sy), end, head[1], head[2]))


def _draw_finish_line(
    draw: ImageDraw.ImageDraw,
    *,
    finish_point: Point,
    finish_tangent: Point,
    road_width: float,
    theme: RacingTrackTheme,
) -> BBox:
    tx, ty = float(finish_tangent[0]), float(finish_tangent[1])
    nx, ny = -ty, tx
    center_x, center_y = float(finish_point[0]), float(finish_point[1])
    segment_count = 8
    segment_len = float(road_width) / float(segment_count)
    points: list[Point] = []
    for index in range(segment_count):
        offset0 = (-0.5 * float(road_width)) + (float(index) * segment_len)
        offset1 = offset0 + segment_len
        start = (center_x + (nx * offset0), center_y + (ny * offset0))
        end = (center_x + (nx * offset1), center_y + (ny * offset1))
        color = theme.finish_light_rgb if index % 2 == 0 else theme.finish_dark_rgb
        draw.line([start, end], fill=color, width=12)
        points.extend((start, end))
    return _polygon_bbox(points)


def _draw_car(
    draw: ImageDraw.ImageDraw,
    *,
    car: RacingTrackCar,
    params: RacingTrackRenderParams,
    theme: RacingTrackTheme,
    color: Color,
) -> BBox:
    """Draw one oriented car body and centered label while preserving its bbox."""

    cx, cy = float(car.center_px[0]), float(car.center_px[1])
    tx, ty = float(car.tangent_px[0]), float(car.tangent_px[1])
    nx, ny = -ty, tx
    half_len = float(params.car_length_px) * 0.5
    half_w = float(params.car_width_px) * 0.5
    corners = (
        (cx + (tx * half_len) + (nx * half_w), cy + (ty * half_len) + (ny * half_w)),
        (cx + (tx * half_len) - (nx * half_w), cy + (ty * half_len) - (ny * half_w)),
        (cx - (tx * half_len) - (nx * half_w), cy - (ty * half_len) - (ny * half_w)),
        (cx - (tx * half_len) + (nx * half_w), cy - (ty * half_len) + (ny * half_w)),
    )
    draw.polygon(corners, fill=color, outline=theme.car_outline_rgb)
    draw.line([corners[0], corners[1]], fill=theme.car_outline_rgb, width=3)
    label_radius = max(10.0, float(params.car_width_px) * 0.42)
    label_bbox = (
        cx - label_radius,
        cy - label_radius,
        cx + label_radius,
        cy + label_radius,
    )
    draw.ellipse(label_bbox, fill=theme.label_fill_rgb, outline=theme.car_outline_rgb, width=2)
    font = load_font(int(params.label_font_size_px), bold=True, font_family=str(params.font_family or "") or None)
    draw_centered_game_text_traced(
        draw,
        center=(cx, cy),
        text=str(car.label),
        font=font,
        fill_rgb=theme.label_text_rgb,
        role="car_label",
        required=True,
        surface_rgbs=(theme.label_fill_rgb,),
        preferred_rgbs=(theme.label_text_rgb,),
        namespace="games.racing_track.car_label",
        extra_metadata={"entity_id": str(car.car_id), "label": str(car.label)},
    )
    bbox = _polygon_bbox(corners)
    return _round_bbox(
        (
            min(float(bbox[0]), float(label_bbox[0])),
            min(float(bbox[1]), float(label_bbox[1])),
            max(float(bbox[2]), float(label_bbox[2])),
            max(float(bbox[3]), float(label_bbox[3])),
        )
    )


def _draw_marked_car_outline(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    theme: RacingTrackTheme,
    width: int,
) -> None:
    pad = float(max(4, int(width)))
    expanded = (
        float(bbox[0]) - pad,
        float(bbox[1]) - pad,
        float(bbox[2]) + pad,
        float(bbox[3]) + pad,
    )
    draw.rounded_rectangle(
        expanded,
        radius=max(6, int(width) * 2),
        outline=theme.marked_car_outline_rgb,
        width=max(3, int(width)),
    )


def render_racing_track_scene(
    *,
    centerline_points_px: Sequence[Point],
    finish_point_px: Point,
    finish_tangent_px: Point,
    cars: Sequence[RacingTrackCar],
    background: Image.Image,
    style_variant: str,
    params: RacingTrackRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
    marked_car_id: str | None = None,
) -> RenderedRacingTrackScene:
    """Render the track, finish, direction arrow, cars, markers, and trace geometry."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    theme = build_games_racing_track_theme(style_variant=str(style_variant))
    base_bbox = (
        0.5 * (float(params.canvas_width) - float(params.track_width_px)),
        0.5 * (float(params.canvas_height) - float(params.track_height_px)),
        0.5 * (float(params.canvas_width) + float(params.track_width_px)),
        0.5 * (float(params.canvas_height) + float(params.track_height_px)),
    )
    track_bbox, _dx, _dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=base_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    left, top, right, bottom = [float(value) for value in track_bbox]
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
    draw.rounded_rectangle(track_bbox, radius=28, fill=theme.infield_rgb, outline=theme.road_outline_rgb, width=4)
    _draw_terrain_pattern(draw, bbox=_round_bbox(track_bbox), theme=theme)

    global_centerline = [_local_to_global(point, track_bbox=track_bbox) for point in centerline_points_px]
    closed_points = list(global_centerline) + [global_centerline[0]]
    road_width = int(params.road_width_px)
    draw.line(closed_points, fill=theme.road_outline_rgb, width=int(road_width + (2 * params.road_border_width_px)), joint="curve")
    draw.line(closed_points, fill=theme.road_rgb, width=int(road_width), joint="curve")
    draw.line(closed_points, fill=theme.lane_mark_rgb, width=max(2, int(road_width * 0.08)), joint="curve")

    finish_point = _local_to_global(finish_point_px, track_bbox=track_bbox)
    finish_bbox = _draw_finish_line(
        draw,
        finish_point=finish_point,
        finish_tangent=finish_tangent_px,
        road_width=float(params.road_width_px + 8),
        theme=theme,
    )
    arrow_index = max(4, min(len(global_centerline) - 3, int(len(global_centerline) * 0.52)))
    arrow_point = global_centerline[arrow_index]
    arrow_next = global_centerline[arrow_index + 2]
    arrow_tangent = (
        float(arrow_next[0]) - float(arrow_point[0]),
        float(arrow_next[1]) - float(arrow_point[1]),
    )
    arrow_len = math.hypot(float(arrow_tangent[0]), float(arrow_tangent[1])) or 1.0
    arrow_bbox = _draw_arrow(
        draw,
        start=(round(float(arrow_point[0]), 3), round(float(arrow_point[1]), 3)),
        tangent=(float(arrow_tangent[0]) / arrow_len, float(arrow_tangent[1]) / arrow_len),
        color=theme.arrow_rgb,
        width=max(4, int(params.road_width_px * 0.12)),
    )

    entities: list[Dict[str, Any]] = []
    entity_bboxes: Dict[str, BBox] = {}
    entity_points: Dict[str, Point] = {}

    global_cars = [_global_car(car, track_bbox=track_bbox) for car in cars]
    for index, car in enumerate(global_cars):
        color = theme.car_palette_rgb[int(index) % len(theme.car_palette_rgb)]
        bbox = _draw_car(draw, car=car, params=params, theme=theme, color=color)
        point = (round(float(car.center_px[0]), 3), round(float(car.center_px[1]), 3))
        entity_bboxes[str(car.car_id)] = bbox
        entity_points[str(car.car_id)] = point
        entities.append(
            {
                "entity_id": str(car.car_id),
                "type": "race_car",
                "label": str(car.label),
                "progress": round(float(car.progress), 6),
                "remaining_distance": round(float(car.remaining_distance), 6),
                "bbox_px": list(bbox),
                "point_px": list(point),
            }
        )
    if str(marked_car_id or "") in entity_bboxes:
        _draw_marked_car_outline(
            draw,
            bbox=entity_bboxes[str(marked_car_id)],
            theme=theme,
            width=int(params.marked_outline_width_px),
        )

    render_map = {
        "track_bbox_px": list(_round_bbox(track_bbox)),
        "layout_jitter": dict(layout_jitter),
        "centerline_points_px": [list(point) for point in global_centerline],
        "finish_point_px": list(finish_point),
        "finish_bbox_px": list(finish_bbox),
        "finish_tangent_px": [round(float(finish_tangent_px[0]), 6), round(float(finish_tangent_px[1]), 6)],
        "direction_arrow_bbox_px": list(arrow_bbox),
        "marked_car_id": str(marked_car_id or ""),
        "entity_bboxes_px": {str(key): list(value) for key, value in sorted(entity_bboxes.items())},
        "entity_points_px": {str(key): list(value) for key, value in sorted(entity_points.items())},
        "panel_scene_style": game_panel_scene_style_metadata(panel_style) if panel_style is not None else {},
        "racing_track_style": {
            "style_variant": str(style_variant),
            "terrain_pattern": str(theme.terrain_pattern),
            "car_palette_rgb": [list(color) for color in theme.car_palette_rgb],
        },
    }
    return RenderedRacingTrackScene(
        image=image,
        scene_entities=tuple(entities),
        render_map=render_map,
    )


def render_racing_track_task_context(
    *,
    state: RacingTrackSceneState,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    render_params: RacingTrackRenderParams,
    instance_seed: int,
    namespace: str,
    marked_car_id: str | None = None,
) -> RenderedRacingTrackTaskContext:
    """Render one racing-track state with shared game-panel styling and noise."""

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
    rendered_scene = render_racing_track_scene(
        centerline_points_px=state.centerline_points_px,
        finish_point_px=state.finish_point_px,
        finish_tangent_px=state.finish_tangent_px,
        cars=state.cars,
        background=background,
        style_variant=str(state.style_variant),
        params=render_params,
        panel_style=panel_style,
        marked_car_id=marked_car_id,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedRacingTrackTaskContext(
        image=image,
        rendered_scene=rendered_scene,
        background_meta=dict(background_meta),
        panel_style_meta=dict(panel_style_meta),
        post_noise_meta=dict(post_noise_meta),
    )


__all__ = [
    "RacingTrackRenderParams",
    "RacingTrackTheme",
    "RenderedRacingTrackTaskContext",
    "RenderedRacingTrackScene",
    "build_games_racing_track_theme",
    "render_racing_track_scene",
    "render_racing_track_task_context",
    "resolve_racing_track_render_params",
]
