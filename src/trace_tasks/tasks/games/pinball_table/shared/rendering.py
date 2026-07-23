"""Shared pinball-table renderer for games-domain tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from ....shared.color_distance import min_color_distance_to_anchors, resolve_contrasting_palette
from ....shared.drawing import draw_arrow
from ....shared.text_rendering import fit_font_to_box, resolve_text_stroke_fill
from ...shared.layout import apply_games_layout_jitter_to_bbox
from ...shared.layout import resolve_games_layout_jitter
from .defaults import DEFAULTS, POST_IMAGE_NOISE_DEFAULTS
from .state import PinballObject, PinballSceneState
from ...shared.scene_style import (
    GamePanelSceneStyle,
    game_panel_contrast_anchor_colors,
    game_panel_scene_style_metadata,
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from ...shared.text import draw_game_text_traced as draw_text_traced

_BUMPER_RADIUS_REF = 0.045
_RECT_WIDTH_REF = 0.116
_RECT_HEIGHT_REF = 0.052


@dataclass(frozen=True)
class PinballRenderParams:
    """Resolved render controls for one pinball table."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    table_width_px: int
    table_height_px: int
    table_border_width_px: int
    ball_radius_px: int
    bumper_radius_px: int
    target_width_px: int
    target_height_px: int
    cue_width_px: int
    label_font_size_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class PinballTheme:
    """Resolved pinball palette for one style variant."""

    table_fill_rgb: Tuple[int, int, int]
    table_outline_rgb: Tuple[int, int, int]
    rail_rgb: Tuple[int, int, int]
    lane_rgb: Tuple[int, int, int]
    flipper_rgb: Tuple[int, int, int]
    ball_fill_rgb: Tuple[int, int, int]
    ball_outline_rgb: Tuple[int, int, int]
    object_palette_rgb: Tuple[Tuple[int, int, int], ...]
    object_outline_rgb: Tuple[int, int, int]
    object_text_rgb: Tuple[int, int, int]
    cue_palette_rgb: Tuple[Tuple[int, int, int], ...]


@dataclass(frozen=True)
class RenderedPinballScene:
    """Rendered pinball image plus trace-friendly geometry."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class RenderedPinballTaskContext:
    """Rendered image and metadata needed for task output assembly."""

    image: Image.Image
    rendered_scene: RenderedPinballScene
    panel_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    text_style_meta: Dict[str, Any]


def resolve_pinball_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> PinballRenderParams:
    """Resolve pinball rendering parameters from config/defaults."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.font_family",
        params=params,
    )
    return PinballRenderParams(
        canvas_width=int(params.get("canvas_width", group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width))),
        canvas_height=int(params.get("canvas_height", group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height))),
        panel_margin_px=int(params.get("panel_margin_px", group_default(render_defaults, "panel_margin_px", DEFAULTS.panel_margin_px))),
        table_width_px=int(params.get("table_width_px", group_default(render_defaults, "table_width_px", DEFAULTS.table_width_px))),
        table_height_px=int(params.get("table_height_px", group_default(render_defaults, "table_height_px", DEFAULTS.table_height_px))),
        table_border_width_px=int(params.get("table_border_width_px", group_default(render_defaults, "table_border_width_px", DEFAULTS.table_border_width_px))),
        ball_radius_px=int(params.get("ball_radius_px", group_default(render_defaults, "ball_radius_px", DEFAULTS.ball_radius_px))),
        bumper_radius_px=int(params.get("bumper_radius_px", group_default(render_defaults, "bumper_radius_px", DEFAULTS.bumper_radius_px))),
        target_width_px=int(params.get("target_width_px", group_default(render_defaults, "target_width_px", DEFAULTS.target_width_px))),
        target_height_px=int(params.get("target_height_px", group_default(render_defaults, "target_height_px", DEFAULTS.target_height_px))),
        cue_width_px=int(params.get("cue_width_px", group_default(render_defaults, "cue_width_px", DEFAULTS.cue_width_px))),
        label_font_size_px=int(params.get("label_font_size_px", group_default(render_defaults, "label_font_size_px", DEFAULTS.label_font_size_px))),
        font_family=str(font_family),
        layout_jitter_meta=resolve_games_layout_jitter(
            params,
            render_defaults,
            instance_seed=int(instance_seed),
            namespace=f"{str(namespace)}.layout",
        ),
    )


def build_games_pinball_theme(*, style_variant: str) -> PinballTheme:
    """Return one complete readable pinball table palette."""

    style = str(style_variant)
    if style == "blueprint":
        return PinballTheme(
            table_fill_rgb=(220, 234, 242),
            table_outline_rgb=(48, 82, 113),
            rail_rgb=(94, 134, 166),
            lane_rgb=(165, 195, 216),
            flipper_rgb=(55, 89, 120),
            ball_fill_rgb=(248, 252, 255),
            ball_outline_rgb=(47, 74, 96),
            object_palette_rgb=((93, 149, 197), (204, 135, 83), (119, 164, 126), (156, 122, 181), (198, 91, 92), (218, 180, 92)),
            object_outline_rgb=(37, 66, 92),
            object_text_rgb=(18, 31, 46),
            cue_palette_rgb=((38, 94, 165), (188, 60, 71), (45, 130, 95), (202, 132, 47)),
        )
    if style == "neon":
        return PinballTheme(
            table_fill_rgb=(22, 24, 45),
            table_outline_rgb=(114, 242, 211),
            rail_rgb=(76, 95, 145),
            lane_rgb=(44, 51, 83),
            flipper_rgb=(247, 83, 168),
            ball_fill_rgb=(246, 255, 252),
            ball_outline_rgb=(102, 242, 214),
            object_palette_rgb=((255, 85, 166), (83, 217, 255), (254, 214, 77), (154, 115, 255), (98, 244, 156), (255, 146, 92)),
            object_outline_rgb=(232, 244, 255),
            object_text_rgb=(16, 20, 30),
            cue_palette_rgb=((255, 224, 83), (95, 238, 181), (255, 112, 174), (122, 166, 255)),
        )
    if style == "carnival":
        return PinballTheme(
            table_fill_rgb=(98, 43, 63),
            table_outline_rgb=(245, 207, 116),
            rail_rgb=(151, 62, 78),
            lane_rgb=(119, 54, 77),
            flipper_rgb=(243, 177, 78),
            ball_fill_rgb=(255, 250, 228),
            ball_outline_rgb=(102, 52, 45),
            object_palette_rgb=((231, 82, 78), (245, 169, 78), (83, 163, 193), (106, 181, 102), (159, 105, 198), (239, 216, 91)),
            object_outline_rgb=(76, 38, 45),
            object_text_rgb=(31, 22, 22),
            cue_palette_rgb=((255, 230, 115), (89, 196, 222), (116, 220, 142), (246, 115, 124)),
        )
    if style == "paper":
        return PinballTheme(
            table_fill_rgb=(241, 233, 211),
            table_outline_rgb=(91, 81, 64),
            rail_rgb=(160, 145, 116),
            lane_rgb=(225, 213, 185),
            flipper_rgb=(103, 92, 78),
            ball_fill_rgb=(255, 255, 248),
            ball_outline_rgb=(73, 67, 55),
            object_palette_rgb=((211, 95, 82), (231, 158, 75), (83, 151, 119), (81, 132, 183), (151, 104, 181), (195, 170, 82)),
            object_outline_rgb=(72, 61, 50),
            object_text_rgb=(35, 29, 24),
            cue_palette_rgb=((46, 92, 152), (190, 66, 61), (45, 127, 85), (195, 126, 47)),
        )
    return PinballTheme(
        table_fill_rgb=(42, 91, 73),
        table_outline_rgb=(226, 218, 178),
        rail_rgb=(35, 70, 58),
        lane_rgb=(57, 112, 91),
        flipper_rgb=(208, 62, 62),
        ball_fill_rgb=(250, 250, 238),
        ball_outline_rgb=(54, 65, 54),
        object_palette_rgb=((217, 70, 75), (234, 166, 68), (86, 151, 203), (116, 184, 101), (151, 107, 196), (221, 209, 83)),
        object_outline_rgb=(30, 57, 45),
        object_text_rgb=(22, 29, 24),
        cue_palette_rgb=((250, 239, 108), (72, 174, 224), (255, 126, 126), (112, 224, 151)),
    )


def _fit_text(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[float, float, float, float],
    text: str,
    fill: Tuple[int, int, int],
    max_size_px: int,
    font_family: str | None,
    surface_rgb: Tuple[int, int, int],
) -> None:
    """Draw centered text inside one bbox."""

    left, top, right, bottom = [float(value) for value in bbox]
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=max(1.0, right - left),
        max_height=max(1.0, bottom - top),
        bold=True,
        font_family=str(font_family or "") or None,
        min_size_px=8,
        max_size_px=int(max_size_px),
        fill_ratio=0.76,
    )
    text_bbox = draw.textbbox((0, 0), str(text), font=font)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    draw_text_traced(
        draw,
        (
            float(left + (0.5 * ((right - left) - text_w)) - float(text_bbox[0])),
            float(top + (0.5 * ((bottom - top) - text_h)) - float(text_bbox[1])),
        ),
        str(text),
        fill=tuple(int(v) for v in fill),
        font=font,
        role="readout",
        required=False,
        surface_rgbs=(tuple(int(v) for v in surface_rgb),),
    )


def _table_bbox(params: PinballRenderParams) -> Tuple[Tuple[float, float, float, float], Dict[str, Any]]:
    """Return table bbox plus resolved layout jitter metadata."""

    left = (float(params.canvas_width) - float(params.table_width_px)) / 2.0
    top = (float(params.canvas_height) - float(params.table_height_px)) / 2.0
    bbox = (left, top, left + float(params.table_width_px), top + float(params.table_height_px))
    if isinstance(params.layout_jitter_meta, Mapping):
        shifted, _dx, _dy, layout_jitter = apply_games_layout_jitter_to_bbox(
            bbox_px=bbox,
            canvas_width=int(params.canvas_width),
            canvas_height=int(params.canvas_height),
            jitter=params.layout_jitter_meta,
        )
        return shifted, dict(layout_jitter)
    return bbox, {}


def _to_px(table_bbox: Tuple[float, float, float, float], point: Tuple[float, float]) -> Tuple[float, float]:
    """Project normalized playfield coordinates to pixels."""

    top_left, top_right, _bottom_right, bottom_left = _logical_corners(table_bbox)
    x, y = float(point[0]), float(point[1])
    left_edge = _lerp_point(top_left, bottom_left, y)
    right_edge = _lerp_point(top_right, _bottom_right, y)
    return _lerp_point(left_edge, right_edge, x)


def _lerp(a: float, b: float, t: float) -> float:
    """Interpolate between two scalars."""

    return float(float(a) + ((float(b) - float(a)) * float(t)))


def _lerp_point(a: Tuple[float, float], b: Tuple[float, float], t: float) -> Tuple[float, float]:
    """Interpolate between two points."""

    return (_lerp(float(a[0]), float(b[0]), float(t)), _lerp(float(a[1]), float(b[1]), float(t)))


def _polygon_bbox(points: Tuple[Tuple[float, float], ...]) -> Tuple[float, float, float, float]:
    """Return bbox enclosing polygon points."""

    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _outer_corners(table_bbox: Tuple[float, float, float, float]) -> Tuple[Tuple[float, float], ...]:
    """Return outer playfield rim corners."""

    left, top, right, bottom = [float(value) for value in table_bbox]
    width = right - left
    height = bottom - top
    return (
        (left + (0.13 * width), top + (0.035 * height)),
        (right - (0.13 * width), top + (0.035 * height)),
        (right - (0.025 * width), bottom - (0.020 * height)),
        (left + (0.025 * width), bottom - (0.020 * height)),
    )


def _logical_corners(table_bbox: Tuple[float, float, float, float]) -> Tuple[Tuple[float, float], ...]:
    """Return normalized-coordinate playfield corners in pixel space."""

    left, top, right, bottom = [float(value) for value in table_bbox]
    width = right - left
    height = bottom - top
    return (
        (left + (0.20 * width), top + (0.085 * height)),
        (right - (0.20 * width), top + (0.085 * height)),
        (right - (0.075 * width), bottom - (0.075 * height)),
        (left + (0.075 * width), bottom - (0.075 * height)),
    )


def _scale_at_y(y_norm: float) -> float:
    """Return perspective scale for an object at a normalized y coordinate."""

    return float(0.78 + (0.28 * float(y_norm)))


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    *,
    table_bbox: Tuple[float, float, float, float],
    norm_points: Tuple[Tuple[float, float], ...],
    fill: Tuple[int, int, int],
    width: int,
) -> Tuple[float, float, float, float]:
    """Draw one projected normalized polyline."""

    points = tuple(_to_px(table_bbox, point) for point in norm_points)
    if len(points) >= 2:
        draw.line(points, fill=tuple(int(v) for v in fill), width=int(width), joint="curve")
    return tuple(round(float(v), 3) for v in _polygon_bbox(points))


def _draw_playfield_polygon(
    draw: ImageDraw.ImageDraw,
    *,
    points: Tuple[Tuple[float, float], ...],
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    width: int,
) -> Tuple[float, float, float, float]:
    """Draw a filled playfield polygon with a thick outline."""

    draw.polygon(points, fill=tuple(int(v) for v in fill))
    draw.line(points + (points[0],), fill=tuple(int(v) for v in outline), width=int(width), joint="curve")
    return tuple(round(float(v), 3) for v in _polygon_bbox(points))


def _draw_projected_polygon(
    draw: ImageDraw.ImageDraw,
    *,
    table_bbox: Tuple[float, float, float, float],
    norm_points: Tuple[Tuple[float, float], ...],
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    width: int,
) -> Tuple[float, float, float, float]:
    """Draw a projected normalized polygon."""

    points = tuple(_to_px(table_bbox, point) for point in norm_points)
    draw.polygon(points, fill=tuple(int(v) for v in fill))
    draw.line(points + (points[0],), fill=tuple(int(v) for v in outline), width=int(width), joint="curve")
    return tuple(round(float(v), 3) for v in _polygon_bbox(points))


def _top_return_arc_points() -> Tuple[Tuple[float, float], ...]:
    """Return normalized points for the top return lane arc."""

    points: list[Tuple[float, float]] = []
    for index in range(21):
        theta = math.radians(204.0 + (132.0 * (float(index) / 20.0)))
        points.append((0.50 + (0.39 * math.cos(theta)), 0.23 + (0.14 * math.sin(theta))))
    return tuple(points)


def _draw_cue_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    start: Tuple[float, float],
    end: Tuple[float, float],
    fill_rgb: Tuple[int, int, int],
    ball_radius_px: float,
    width_px: int,
) -> Dict[str, Any]:
    """Draw a single legible launch-cue arrow and return visible geometry."""

    start_x, start_y = float(start[0]), float(start[1])
    end_x, end_y = float(end[0]), float(end[1])
    dx = float(end_x - start_x)
    dy = float(end_y - start_y)
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return {
            "visible_start": [round(start_x, 3), round(start_y, 3)],
            "visible_end": [round(end_x, 3), round(end_y, 3)],
            "length_px": 0.0,
        }
    unit_x = float(dx / length)
    unit_y = float(dy / length)
    start_offset = min(max(float(ball_radius_px) + 5.0, 8.0), max(8.0, length * 0.28))
    visible_start = (
        float(start_x + (unit_x * start_offset)),
        float(start_y + (unit_y * start_offset)),
    )
    visible_length = max(0.0, float(length - start_offset))
    if visible_length <= 8.0:
        visible_start = (start_x, start_y)
        visible_length = length
    head_length = min(22.0, max(14.0, visible_length * 0.24))
    head_width = min(20.0, max(13.0, float(width_px) * 3.0))
    halo_rgb = tuple(int(v) for v in resolve_text_stroke_fill(fill_rgb))
    draw_arrow(
        draw,
        start=visible_start,
        end=(end_x, end_y),
        fill=halo_rgb,
        width=max(2, int(width_px) + 5),
        head_length_px=float(head_length + 4.0),
        head_width_px=float(head_width + 7.0),
    )
    draw_arrow(
        draw,
        start=visible_start,
        end=(end_x, end_y),
        fill=tuple(int(v) for v in fill_rgb),
        width=max(2, int(width_px)),
        head_length_px=float(head_length),
        head_width_px=float(head_width),
    )
    return {
        "visible_start": [round(float(visible_start[0]), 3), round(float(visible_start[1]), 3)],
        "visible_end": [round(float(end_x), 3), round(float(end_y), 3)],
        "length_px": round(float(visible_length), 3),
        "head_length_px": round(float(head_length), 3),
        "head_width_px": round(float(head_width), 3),
        "halo_rgb": [int(v) for v in halo_rgb],
    }


def _draw_full_path(
    draw: ImageDraw.ImageDraw,
    *,
    points_px: Tuple[Tuple[float, float], ...],
    fill_rgb: Tuple[int, int, int],
    width_px: int,
) -> Dict[str, Any]:
    """Draw a complete visible pinball trajectory with a final direction arrow."""

    rounded_points = [[round(float(x), 3), round(float(y), 3)] for x, y in points_px]
    if len(points_px) < 2:
        return {"points": rounded_points, "length_px": 0.0}
    halo_rgb = tuple(int(v) for v in resolve_text_stroke_fill(fill_rgb))
    total_length = 0.0
    for start, end in zip(points_px[:-1], points_px[1:]):
        total_length += math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
    for color, width in ((halo_rgb, max(2, int(width_px) + 5)), (tuple(int(v) for v in fill_rgb), max(2, int(width_px)))):
        draw.line(points_px, fill=tuple(int(v) for v in color), width=int(width), joint="curve")
    last_start = points_px[-2]
    last_end = points_px[-1]
    dx = float(last_end[0]) - float(last_start[0])
    dy = float(last_end[1]) - float(last_start[1])
    segment_length = math.hypot(dx, dy)
    if segment_length > 1e-6:
        unit_x = float(dx / segment_length)
        unit_y = float(dy / segment_length)
        arrow_start = (
            float(last_end[0] - (unit_x * min(44.0, max(24.0, segment_length * 0.38)))),
            float(last_end[1] - (unit_y * min(44.0, max(24.0, segment_length * 0.38)))),
        )
        for color, width, head_extra in (
            (halo_rgb, max(2, int(width_px) + 5), 5.0),
            (tuple(int(v) for v in fill_rgb), max(2, int(width_px)), 0.0),
        ):
            draw_arrow(
                draw,
                start=arrow_start,
                end=last_end,
                fill=tuple(int(v) for v in color),
                width=int(width),
                head_length_px=20.0 + float(head_extra),
                head_width_px=18.0 + float(head_extra),
            )
    for point in points_px[1:-1]:
        radius = max(3.0, float(width_px) * 0.72)
        bbox = (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius)
        draw.ellipse(bbox, fill=tuple(int(v) for v in fill_rgb), outline=halo_rgb, width=2)
    return {
        "points": rounded_points,
        "visible_start": rounded_points[0],
        "visible_end": rounded_points[-1],
        "length_px": round(float(total_length), 3),
        "halo_rgb": [int(v) for v in halo_rgb],
    }


def _object_bbox_px(
    *,
    table_bbox: Tuple[float, float, float, float],
    obj: PinballObject,
    params: PinballRenderParams,
) -> Tuple[float, float, float, float]:
    """Return projected bbox for one pinball object."""

    cx, cy = _to_px(table_bbox, (float(obj.x_norm), float(obj.y_norm)))
    scale = _scale_at_y(float(obj.y_norm))
    if str(obj.kind) in {"bumper", "standup_target"}:
        radius = float(params.bumper_radius_px) * (float(obj.radius_norm) / _BUMPER_RADIUS_REF) * scale
        return (cx - radius, cy - (0.84 * radius), cx + radius, cy + (0.84 * radius))
    width = float(params.target_width_px) * (float(obj.width_norm) / _RECT_WIDTH_REF) * scale
    height = float(params.target_height_px) * (float(obj.height_norm) / _RECT_HEIGHT_REF) * scale
    return (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def _draw_object(
    draw: ImageDraw.ImageDraw,
    *,
    table_bbox: Tuple[float, float, float, float],
    obj: PinballObject,
    theme: PinballTheme,
    params: PinballRenderParams,
) -> Tuple[float, float, float, float]:
    """Draw one pinball object and return its bbox."""

    bbox = _object_bbox_px(table_bbox=table_bbox, obj=obj, params=params)
    fill = theme.object_palette_rgb[int(obj.color_index) % len(theme.object_palette_rgb)]
    outline = tuple(int(v) for v in theme.object_outline_rgb)
    kind = str(obj.kind)
    if kind == "bumper":
        draw.ellipse(bbox, fill=tuple(int(v) for v in fill), outline=outline, width=3)
        inset = 0.20 * (bbox[2] - bbox[0])
        draw.ellipse(
            (bbox[0] + inset, bbox[1] + inset, bbox[2] - inset, bbox[3] - inset),
            outline=tuple(int(v) for v in theme.table_outline_rgb),
            width=2,
        )
    elif kind == "standup_target":
        draw.ellipse(bbox, fill=tuple(int(v) for v in fill), outline=outline, width=3)
        inset = 0.28 * (bbox[2] - bbox[0])
        draw.ellipse(
            (bbox[0] + inset, bbox[1] + inset, bbox[2] - inset, bbox[3] - inset),
            fill=tuple(int(v) for v in theme.object_text_rgb),
        )
    elif kind == "rollover_lane":
        draw.rounded_rectangle(
            bbox,
            radius=max(6, int((bbox[3] - bbox[1]) * 0.48)),
            fill=tuple(int(v) for v in fill),
            outline=outline,
            width=3,
        )
        mid_y = (bbox[1] + bbox[3]) / 2.0
        draw.line(
            (bbox[0] + 0.14 * (bbox[2] - bbox[0]), mid_y, bbox[2] - 0.14 * (bbox[2] - bbox[0]), mid_y),
            fill=tuple(int(v) for v in theme.table_outline_rgb),
            width=2,
        )
    else:
        draw.rounded_rectangle(
            bbox,
            radius=max(5, int((bbox[3] - bbox[1]) * 0.25)),
            fill=tuple(int(v) for v in fill),
            outline=outline,
            width=3,
        )
        inset_x = 0.18 * (bbox[2] - bbox[0])
        inset_y = 0.20 * (bbox[3] - bbox[1])
        draw.rectangle(
            (bbox[0] + inset_x, bbox[1] + inset_y, bbox[2] - inset_x, bbox[3] - inset_y),
            outline=tuple(int(v) for v in theme.table_outline_rgb),
            width=2,
        )
    label_box = (
        bbox[0] + 0.18 * (bbox[2] - bbox[0]),
        bbox[1] + 0.16 * (bbox[3] - bbox[1]),
        bbox[2] - 0.18 * (bbox[2] - bbox[0]),
        bbox[3] - 0.16 * (bbox[3] - bbox[1]),
    )
    display_text = str(int(obj.score_value)) if obj.score_value is not None else (str(obj.label) if bool(obj.show_label) else "")
    if display_text:
        _fit_text(
            draw,
            bbox=label_box,
            text=str(display_text),
            fill=theme.object_text_rgb,
            max_size_px=int(params.label_font_size_px),
            font_family=str(params.font_family),
            surface_rgb=tuple(int(v) for v in fill),
        )
    return tuple(round(float(v), 3) for v in bbox)


def _cue_color_anchors(
    *,
    theme: PinballTheme,
    panel_style: GamePanelSceneStyle | None,
) -> Tuple[Tuple[int, int, int], ...]:
    """Return colors that launch-cue colors must avoid."""

    anchors = (
        theme.table_fill_rgb,
        theme.table_outline_rgb,
        theme.rail_rgb,
        theme.lane_rgb,
        theme.ball_fill_rgb,
        theme.flipper_rgb,
    )
    return tuple(game_panel_contrast_anchor_colors(panel_style, extra_colors=anchors))


def render_pinball_scene(
    *,
    objects: Tuple[PinballObject, ...],
    ball_xy_norm: Tuple[float, float],
    hidden_path_norm: Tuple[Tuple[float, float], ...],
    cue_visible_fraction: float,
    background: Image.Image,
    style_variant: str,
    params: PinballRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedPinballScene:
    """Render the shared projected pinball playfield and all visible objects."""

    image = background.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    theme = build_games_pinball_theme(style_variant=str(style_variant))
    cue_anchors = _cue_color_anchors(theme=theme, panel_style=panel_style)
    cue_rgb = resolve_contrasting_palette(
        theme.cue_palette_rgb,
        anchor_colors=cue_anchors,
        min_anchor_distance=40.0,
        min_pairwise_distance=18.0,
        distance_space="lab",
    )[0]

    table_bbox, layout_jitter = _table_bbox(params)
    outer_corners = _outer_corners(table_bbox)
    logical_corners = _logical_corners(table_bbox)
    shadow_corners = tuple((float(x) + 10.0, float(y) + 12.0) for x, y in outer_corners)
    draw.polygon(shadow_corners, fill=(38, 42, 49))
    decorative_entities: list[Dict[str, Any]] = []
    rim_bbox = _draw_playfield_polygon(
        draw,
        points=outer_corners,
        fill=tuple(int(v) for v in theme.rail_rgb),
        outline=tuple(int(v) for v in theme.table_outline_rgb),
        width=max(5, int(params.table_border_width_px)),
    )
    playfield_bbox = _draw_playfield_polygon(
        draw,
        points=logical_corners,
        fill=tuple(int(v) for v in theme.table_fill_rgb),
        outline=tuple(int(v) for v in theme.table_outline_rgb),
        width=3,
    )
    decorative_entities.append({"id": "playfield_rim", "type": "pinball_rim", "bbox": list(rim_bbox)})
    decorative_entities.append({"id": "playfield_surface", "type": "pinball_surface", "bbox": list(playfield_bbox)})

    for rail_id, rail_points in (
        ("left_inner_lane", ((0.12, 0.12), (0.10, 0.42), (0.15, 0.77))),
        ("right_inner_lane", ((0.88, 0.12), (0.90, 0.42), (0.85, 0.77))),
        ("right_plunger_lane_outer", ((0.91, 0.08), (0.95, 0.47), (0.92, 0.88))),
        ("right_plunger_lane_inner", ((0.84, 0.12), (0.88, 0.49), (0.85, 0.82))),
    ):
        bbox = _draw_polyline(
            draw,
            table_bbox=table_bbox,
            norm_points=rail_points,
            fill=tuple(int(v) for v in theme.lane_rgb),
            width=4 if "plunger" in rail_id else 3,
        )
        decorative_entities.append({"id": rail_id, "type": "pinball_lane", "bbox": list(bbox)})

    arc_bbox = _draw_polyline(
        draw,
        table_bbox=table_bbox,
        norm_points=_top_return_arc_points(),
        fill=tuple(int(v) for v in theme.lane_rgb),
        width=4,
    )
    decorative_entities.append({"id": "top_return_arc", "type": "pinball_lane_arc", "bbox": list(arc_bbox)})

    for slingshot_id, slingshot_points in (
        ("left_slingshot", ((0.18, 0.67), (0.34, 0.74), (0.28, 0.56))),
        ("right_slingshot", ((0.82, 0.67), (0.66, 0.74), (0.72, 0.56))),
    ):
        bbox = _draw_projected_polygon(
            draw,
            table_bbox=table_bbox,
            norm_points=slingshot_points,
            fill=tuple(int(v) for v in theme.lane_rgb),
            outline=tuple(int(v) for v in theme.table_outline_rgb),
            width=3,
        )
        decorative_entities.append({"id": slingshot_id, "type": "pinball_slingshot", "bbox": list(bbox)})

    for flipper_id, flipper_points in (
        ("left_flipper", ((0.28, 0.88), (0.48, 0.81), (0.51, 0.85), (0.31, 0.92))),
        ("right_flipper", ((0.72, 0.88), (0.52, 0.81), (0.49, 0.85), (0.69, 0.92))),
    ):
        bbox = _draw_projected_polygon(
            draw,
            table_bbox=table_bbox,
            norm_points=flipper_points,
            fill=tuple(int(v) for v in theme.flipper_rgb),
            outline=tuple(int(v) for v in theme.table_outline_rgb),
            width=3,
        )
        decorative_entities.append({"id": flipper_id, "type": "pinball_flipper", "bbox": list(bbox)})

    for post_index, post_point in enumerate(((0.24, 0.24), (0.76, 0.24), (0.20, 0.78), (0.80, 0.78), (0.50, 0.91))):
        px, py = _to_px(table_bbox, post_point)
        radius = 5.5 * _scale_at_y(float(post_point[1]))
        bbox = (px - radius, py - radius, px + radius, py + radius)
        draw.ellipse(
            bbox,
            fill=tuple(int(v) for v in theme.table_outline_rgb),
            outline=tuple(int(v) for v in theme.ball_outline_rgb),
            width=1,
        )
        decorative_entities.append(
            {
                "id": f"post_{post_index}",
                "type": "pinball_post",
                "bbox": [round(float(v), 3) for v in bbox],
            }
        )

    entity_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    entity_points: Dict[str, Tuple[float, float]] = {}
    object_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    scene_entities: list[Dict[str, Any]] = []
    motion_paths_px: Dict[str, Dict[str, Any]] = {}
    full_path_mode = bool(float(cue_visible_fraction) >= 0.99)
    path_points_px = tuple(_to_px(table_bbox, point) for point in hidden_path_norm)
    if full_path_mode and len(path_points_px) >= 2:
        motion_paths_px["shown_path"] = _draw_full_path(
            draw,
            points_px=path_points_px,
            fill_rgb=tuple(int(v) for v in cue_rgb),
            width_px=max(2, int(params.cue_width_px)),
        )
    for obj in objects:
        center = _to_px(table_bbox, (float(obj.x_norm), float(obj.y_norm)))
        bbox = _draw_object(draw, table_bbox=table_bbox, obj=obj, theme=theme, params=params)
        point = (round(float(center[0]), 3), round(float(center[1]), 3))
        entity_bboxes[str(obj.object_id)] = bbox
        entity_points[str(obj.object_id)] = point
        object_bboxes[str(obj.object_id)] = bbox
        scene_entities.append(
            {
                "id": str(obj.object_id),
                "type": "pinball_object",
                "label": str(obj.label),
                "kind": str(obj.kind),
                "bbox": list(bbox),
                "point": list(point),
                "score_value": None if obj.score_value is None else int(obj.score_value),
                "show_label": bool(obj.show_label),
            }
        )

    ball_px = _to_px(table_bbox, ball_xy_norm)
    ball_r = float(params.ball_radius_px)
    ball_bbox = (ball_px[0] - ball_r, ball_px[1] - ball_r, ball_px[0] + ball_r, ball_px[1] + ball_r)

    if (not full_path_mode) and len(hidden_path_norm) >= 2:
        start = _to_px(table_bbox, hidden_path_norm[0])
        full_end = _to_px(table_bbox, hidden_path_norm[-1])
        visible_end = (
            float(start[0] + (float(cue_visible_fraction) * (full_end[0] - start[0]))),
            float(start[1] + (float(cue_visible_fraction) * (full_end[1] - start[1]))),
        )
        cue_meta = _draw_cue_arrow(
            draw,
            start=start,
            end=visible_end,
            fill_rgb=tuple(int(v) for v in cue_rgb),
            ball_radius_px=float(params.ball_radius_px),
            width_px=max(2, int(params.cue_width_px)),
        )
        motion_paths_px["shown_path"] = {
            "points": [list(_to_px(table_bbox, point)) for point in hidden_path_norm],
            "visible_end": [float(visible_end[0]), float(visible_end[1])],
            **cue_meta,
        }

    draw.ellipse(ball_bbox, fill=tuple(int(v) for v in theme.ball_fill_rgb), outline=tuple(int(v) for v in theme.ball_outline_rgb), width=3)
    entity_bboxes["ball"] = tuple(round(float(v), 3) for v in ball_bbox)
    entity_points["ball"] = (round(float(ball_px[0]), 3), round(float(ball_px[1]), 3))
    scene_entities.append({"id": "ball", "type": "pinball_ball", "bbox": list(entity_bboxes["ball"]), "point": list(entity_points["ball"])})

    render_map: Dict[str, Any] = {
        "table_bbox_px": [round(float(v), 3) for v in table_bbox],
        "playfield_projection": {
            "kind": "trapezoid_isometric",
            "logical_corners_px": [[round(float(x), 3), round(float(y), 3)] for x, y in logical_corners],
            "outer_corners_px": [[round(float(x), 3), round(float(y), 3)] for x, y in outer_corners],
        },
        "decorative_entities": [dict(entity) for entity in decorative_entities],
        "entity_bboxes_px": {str(key): list(value) for key, value in entity_bboxes.items()},
        "entity_points_px": {str(key): list(value) for key, value in entity_points.items()},
        "object_bboxes_px": {str(key): list(value) for key, value in object_bboxes.items()},
        "motion_paths_px": motion_paths_px,
        "layout_jitter": dict(layout_jitter),
        "style_variant": str(style_variant),
        "font_family": str(params.font_family),
        "text_style": {"font_family": str(params.font_family)},
        "cue_rgb": list(cue_rgb),
        "cue_color_safety": {
            "distance_space": "lab",
            "min_anchor_distance_required": 40.0,
            "anchor_rgbs": [list(color) for color in cue_anchors],
            "cue_anchor_lab_distance": round(float(min_color_distance_to_anchors(cue_rgb, cue_anchors, distance_space="lab")), 3),
        },
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
    }
    return RenderedPinballScene(image=image, scene_entities=tuple(scene_entities), render_map=render_map)


def render_pinball_task_context(
    *,
    scene: PinballSceneState,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> RenderedPinballTaskContext:
    """Render one pinball scene with panel style, layout jitter, and post noise."""

    render_params = resolve_pinball_render_params(
        params,
        render_defaults=render_defaults,
        namespace=str(namespace),
        instance_seed=int(instance_seed),
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
        allowed_panel_treatments = tuple(str(item) for item in allowed_panel_treatments_raw)
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.panel_scene_style",
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
    rendered_scene = render_pinball_scene(
        objects=scene.objects,
        ball_xy_norm=(float(scene.ball_x_norm), float(scene.ball_y_norm)),
        hidden_path_norm=scene.hidden_path_norm,
        cue_visible_fraction=float(scene.cue_visible_fraction),
        background=background,
        style_variant=str(scene.style_variant),
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
    return RenderedPinballTaskContext(
        image=image,
        rendered_scene=rendered_scene,
        panel_style_meta=dict(panel_style_meta),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        text_style_meta=dict(text_style_meta),
    )


__all__ = [
    "PinballRenderParams",
    "RenderedPinballTaskContext",
    "RenderedPinballScene",
    "build_games_pinball_theme",
    "render_pinball_scene",
    "render_pinball_task_context",
    "resolve_pinball_render_params",
]
