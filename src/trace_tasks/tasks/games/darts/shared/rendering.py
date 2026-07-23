"""Shared simplified dartboard renderer for games-domain tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.noise import apply_post_image_noise

from ....shared.color_distance import color_distance
from ....shared.config_defaults import group_default
from ....shared.font_assets import get_font_family_record
from ....shared.text_rendering import load_font
from ...shared.text import draw_game_text_traced as draw_text_traced
from ...shared.scene_style import (
    GamePanelSceneStyle,
    draw_panel_scene_chrome,
    game_panel_scene_style_metadata,
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from ...shared.visual_defaults import load_games_scene_noise_defaults

from .defaults import DARTS_NAMESPACE, SCENE_ID, STANDARD_DART_SECTORS
from .state import DartInstance


DARTBOARD_RADIUS_FRACTIONS: Mapping[str, float] = {
    "bullseye": 0.200,
    "sector_outer": 0.840,
    "frame": 0.840,
}

DARTBOARD_SAMPLE_RADIUS_FRACTIONS: Mapping[str, Tuple[float, float]] = {
    "bullseye": (0.000, 0.165),
    "sector": (0.295, 0.760),
}


@dataclass(frozen=True)
class DartboardRenderParams:
    """Resolved render controls for one simplified dartboard scene."""

    canvas_width: int
    canvas_height: int
    board_center_x_px: int
    board_center_y_px: int
    board_radius_px: int
    marker_radius_px: int
    number_font_size_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class RenderedDartSpec:
    """One rendered dart marker with trace-friendly metadata."""

    dart_id: str
    label: str | None
    area_kind: str
    sector_value: int | None
    score: int
    center_px: Tuple[float, float]
    bbox_px: Tuple[float, float, float, float]
    is_marked: bool


@dataclass(frozen=True)
class RenderedDartsScene:
    """Rendered dartboard scene plus metadata."""

    image: Image.Image
    dart_specs: Tuple[RenderedDartSpec, ...]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class RenderedDartsTaskContext:
    """Rendered darts image plus scene-wide visual metadata."""

    image: Image.Image
    rendered_scene: RenderedDartsScene
    panel_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    text_style_meta: Dict[str, Any]


POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


_STYLE_PALETTES: Mapping[str, Mapping[str, Tuple[int, int, int]]] = {
    "classic": {
        "board_frame": (28, 31, 35),
        "light_sector": (238, 224, 194),
        "dark_sector": (33, 36, 40),
        "bullseye": (196, 45, 48),
        "wire": (222, 218, 205),
        "number": (246, 246, 238),
        "dart": (244, 198, 65),
        "dart_outline": (41, 45, 52),
        "marked": (248, 72, 64),
    },
    "soft": {
        "board_frame": (51, 61, 68),
        "light_sector": (246, 231, 197),
        "dark_sector": (63, 69, 74),
        "bullseye": (205, 70, 70),
        "wire": (238, 232, 216),
        "number": (252, 249, 239),
        "dart": (102, 180, 232),
        "dart_outline": (29, 45, 58),
        "marked": (238, 74, 68),
    },
    "outlined": {
        "board_frame": (31, 41, 48),
        "light_sector": (242, 236, 219),
        "dark_sector": (59, 66, 73),
        "bullseye": (184, 54, 58),
        "wire": (32, 38, 44),
        "number": (248, 248, 240),
        "dart": (250, 218, 93),
        "dart_outline": (13, 22, 30),
        "marked": (231, 61, 54),
    },
    "league_blue": {
        "board_frame": (24, 48, 83),
        "light_sector": (239, 231, 210),
        "dark_sector": (31, 60, 96),
        "bullseye": (207, 58, 76),
        "wire": (226, 233, 240),
        "number": (250, 252, 255),
        "dart": (247, 197, 72),
        "dart_outline": (9, 24, 44),
        "marked": (248, 76, 68),
    },
    "parchment": {
        "board_frame": (76, 55, 38),
        "light_sector": (247, 227, 184),
        "dark_sector": (78, 63, 49),
        "bullseye": (174, 57, 55),
        "wire": (238, 217, 178),
        "number": (255, 244, 218),
        "dart": (73, 142, 191),
        "dart_outline": (40, 28, 21),
        "marked": (220, 64, 54),
    },
    "neon": {
        "board_frame": (17, 24, 39),
        "light_sector": (225, 236, 230),
        "dark_sector": (24, 34, 55),
        "bullseye": (224, 61, 104),
        "wire": (178, 220, 238),
        "number": (242, 250, 255),
        "dart": (250, 204, 21),
        "dart_outline": (4, 12, 24),
        "marked": (255, 84, 76),
    },
}

_DART_COLOR_CANDIDATES: Tuple[Tuple[int, int, int], ...] = (
    (37, 99, 235),
    (147, 51, 234),
    (217, 70, 239),
    (6, 182, 212),
    (14, 165, 233),
    (244, 114, 182),
    (236, 72, 153),
    (125, 58, 237),
    (79, 70, 229),
    (56, 189, 248),
)
_DART_COLOR_ANCHOR_KEYS: Tuple[str, ...] = (
    "board_frame",
    "light_sector",
    "dark_sector",
    "bullseye",
    "wire",
    "number",
)
_DART_COLOR_EXTRA_ANCHORS: Tuple[Tuple[int, int, int], ...] = (
    (20, 24, 28),
)


def _palette(style_variant: str) -> Mapping[str, Tuple[int, int, int]]:
    """Return one safe dartboard palette."""

    return _STYLE_PALETTES.get(str(style_variant), _STYLE_PALETTES["classic"])


def dartboard_anchor_colors(style_variant: str) -> Tuple[Tuple[int, int, int], ...]:
    """Return board and highlight colors that dart markers must avoid."""

    palette = _palette(str(style_variant))
    colors = [tuple(int(v) for v in palette[key]) for key in _DART_COLOR_ANCHOR_KEYS]
    colors.extend(tuple(int(v) for v in color) for color in _DART_COLOR_EXTRA_ANCHORS)
    return tuple(colors)


def sample_dart_marker_color(
    rng,
    *,
    style_variant: str,
    min_lab_distance: float = 40.0,
) -> Tuple[Tuple[int, int, int], float]:
    """Sample a dart fill color that is Lab-separated from board colors."""

    anchors = dartboard_anchor_colors(str(style_variant))
    candidates = list(_DART_COLOR_CANDIDATES)
    rng.shuffle(candidates)
    best_color = candidates[0]
    best_min_distance = -1.0
    for candidate in candidates:
        min_distance = min(float(color_distance(candidate, anchor, distance_space="lab")) for anchor in anchors)
        if float(min_distance) > float(best_min_distance):
            best_color = candidate
            best_min_distance = float(min_distance)
        if float(min_distance) >= float(min_lab_distance):
            return tuple(int(v) for v in candidate), float(min_distance)
    return tuple(int(v) for v in best_color), float(best_min_distance)


def polar_to_xy(*, cx: float, cy: float, radius: float, angle_deg: float) -> Tuple[float, float]:
    """Convert dartboard polar coordinates to image pixel coordinates."""

    theta = math.radians(float(angle_deg))
    return (
        float(cx + (float(radius) * math.sin(theta))),
        float(cy - (float(radius) * math.cos(theta))),
    )


def _ring_polygon(
    *,
    cx: float,
    cy: float,
    inner_radius: float,
    outer_radius: float,
    start_deg: float,
    end_deg: float,
    steps: int = 8,
) -> List[Tuple[float, float]]:
    """Return polygon points for one annular sector."""

    outer: List[Tuple[float, float]] = []
    inner: List[Tuple[float, float]] = []
    for index in range(int(steps) + 1):
        t = float(index) / float(max(1, steps))
        angle = float(start_deg) + (float(end_deg) - float(start_deg)) * t
        outer.append(polar_to_xy(cx=cx, cy=cy, radius=outer_radius, angle_deg=angle))
    for index in range(int(steps), -1, -1):
        t = float(index) / float(max(1, steps))
        angle = float(start_deg) + (float(end_deg) - float(start_deg)) * t
        inner.append(polar_to_xy(cx=cx, cy=cy, radius=inner_radius, angle_deg=angle))
    return outer + inner


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    *,
    font,
    fill: Tuple[int, int, int],
    stroke_fill: Tuple[int, int, int] | None = None,
    stroke_width: int = 0,
) -> None:
    """Draw text centered on one point."""

    bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=int(stroke_width))
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    draw_text_traced(
        draw,
        (float(xy[0]) - (0.5 * width), float(xy[1]) - (0.5 * height)),
        str(text),
        font=font,
        fill=fill,
        stroke_width=int(stroke_width),
        stroke_fill=stroke_fill,
        role="readout",
        required=False,
    )


def _draw_board(
    image: Image.Image,
    *,
    params: DartboardRenderParams,
    style_variant: str,
) -> Dict[str, Any]:
    """Draw the simplified dartboard and return board geometry metadata."""

    draw = ImageDraw.Draw(image)
    palette = _palette(str(style_variant))
    cx = float(params.board_center_x_px)
    cy = float(params.board_center_y_px)
    radius = float(params.board_radius_px)
    bull_radius = float(DARTBOARD_RADIUS_FRACTIONS["bullseye"]) * radius
    sector_outer = float(DARTBOARD_RADIUS_FRACTIONS["sector_outer"]) * radius
    sector_width_deg = 360.0 / float(max(1, len(STANDARD_DART_SECTORS)))
    sector_half_width_deg = 0.5 * float(sector_width_deg)

    # Draw only the scoring disk. The old filled outer frame looked like an extra
    # scoring ring, which made the simplified board harder to read.
    for sector_index, _sector_value in enumerate(STANDARD_DART_SECTORS):
        start_deg = float((sector_index * sector_width_deg) - sector_half_width_deg)
        end_deg = float((sector_index * sector_width_deg) + sector_half_width_deg)
        fill = palette["light_sector"] if sector_index % 2 == 0 else palette["dark_sector"]
        draw.polygon(
            _ring_polygon(
                cx=cx,
                cy=cy,
                inner_radius=float(bull_radius),
                outer_radius=float(sector_outer),
                start_deg=float(start_deg),
                end_deg=float(end_deg),
            ),
            fill=tuple(int(v) for v in fill),
        )

    wire = tuple(int(v) for v in palette["wire"])
    draw.ellipse(
        [cx - sector_outer, cy - sector_outer, cx + sector_outer, cy + sector_outer],
        outline=wire,
        width=3,
    )
    draw.ellipse(
        [cx - bull_radius, cy - bull_radius, cx + bull_radius, cy + bull_radius],
        fill=tuple(int(v) for v in palette["bullseye"]),
        outline=wire,
        width=3,
    )
    for sector_index in range(len(STANDARD_DART_SECTORS)):
        angle = float((sector_index * sector_width_deg) - sector_half_width_deg)
        x_outer, y_outer = polar_to_xy(cx=cx, cy=cy, radius=sector_outer, angle_deg=angle)
        x_inner, y_inner = polar_to_xy(cx=cx, cy=cy, radius=bull_radius, angle_deg=angle)
        draw.line([(x_inner, y_inner), (x_outer, y_outer)], fill=wire, width=2)

    number_font = load_font(
        int(params.number_font_size_px),
        bold=False,
        font_family=str(params.font_family) or None,
    )
    number_radius = float(sector_outer) + float(max(62, int(params.number_font_size_px) * 1.6))
    for sector_index, sector_value in enumerate(STANDARD_DART_SECTORS):
        angle = float(sector_index * sector_width_deg)
        x_text, y_text = polar_to_xy(cx=cx, cy=cy, radius=number_radius, angle_deg=angle)
        _draw_centered_text(
            draw,
            (x_text, y_text),
            str(sector_value),
            font=number_font,
            fill=tuple(int(v) for v in palette["number"]),
            stroke_fill=(20, 24, 28),
            stroke_width=1,
        )

    return {
        "center_px": [round(cx, 3), round(cy, 3)],
        "radius_px": round(radius, 3),
        "radii_px": {
            "bullseye": round(float(bull_radius), 3),
            "sector_outer": round(float(sector_outer), 3),
            "frame": round(float(sector_outer), 3),
            "number": round(float(number_radius), 3),
        },
        "sector_order_clockwise_from_top": [int(value) for value in STANDARD_DART_SECTORS],
        "bullseye_score": 50,
    }


def _draw_dart(
    draw: ImageDraw.ImageDraw,
    *,
    dart: DartInstance,
    params: DartboardRenderParams,
    style_variant: str,
    dart_fill_color: Tuple[int, int, int] | None = None,
) -> Tuple[float, float, float, float]:
    """Draw one dart marker and return its bbox."""

    palette = _palette(str(style_variant))
    radius = float(params.marker_radius_px)
    x = float(dart.x_px)
    y = float(dart.y_px)
    fill = palette["dart"] if dart_fill_color is None else tuple(int(v) for v in dart_fill_color)
    outline = palette["dart_outline"]
    bbox = (
        round(x - radius, 3),
        round(y - radius, 3),
        round(x + radius, 3),
        round(y + radius, 3),
    )
    arm_radius = round(radius * 1.55)
    x_i = round(x)
    y_i = round(y)
    draw.line([(x_i - arm_radius, y_i), (x_i + arm_radius, y_i)], fill=tuple(int(v) for v in outline), width=5)
    draw.line([(x_i, y_i - arm_radius), (x_i, y_i + arm_radius)], fill=tuple(int(v) for v in outline), width=5)
    draw.line([(x_i - arm_radius, y_i), (x_i + arm_radius, y_i)], fill=tuple(int(v) for v in fill), width=3)
    draw.line([(x_i, y_i - arm_radius), (x_i, y_i + arm_radius)], fill=tuple(int(v) for v in fill), width=3)
    draw.ellipse(bbox, fill=tuple(int(v) for v in fill), outline=tuple(int(v) for v in outline), width=2)
    if bool(dart.is_marked):
        ring_radius = radius * 1.72
        draw.ellipse(
            [x - ring_radius, y - ring_radius, x + ring_radius, y + ring_radius],
            outline=tuple(int(v) for v in palette["marked"]),
            width=5,
        )
    return bbox


def _draw_dart_label(
    draw: ImageDraw.ImageDraw,
    *,
    dart: DartInstance,
    params: DartboardRenderParams,
    style_variant: str,
) -> Tuple[float, float, float, float] | None:
    """Draw an optional letter badge next to one dart marker."""

    if dart.label is None:
        return None
    palette = _palette(str(style_variant))
    x = float(dart.x_px)
    y = float(dart.y_px)
    dx = x - float(params.board_center_x_px)
    dy = y - float(params.board_center_y_px)
    distance = math.hypot(dx, dy)
    if distance < 1.0:
        ux, uy = 0.72, -0.72
    else:
        ux, uy = dx / distance, dy / distance
    offset = max(30.0, float(params.marker_radius_px) * 2.3)
    badge_radius = max(13.0, float(params.marker_radius_px) * 0.95)
    cx = min(
        float(params.canvas_width) - badge_radius - 8.0,
        max(badge_radius + 8.0, x + (ux * offset)),
    )
    cy = min(
        float(params.canvas_height) - badge_radius - 8.0,
        max(badge_radius + 8.0, y + (uy * offset)),
    )
    outline = tuple(int(v) for v in palette["dart_outline"])
    draw.line([(x, y), (cx, cy)], fill=outline, width=2)
    bbox = (
        round(cx - badge_radius, 3),
        round(cy - badge_radius, 3),
        round(cx + badge_radius, 3),
        round(cy + badge_radius, 3),
    )
    draw.ellipse(bbox, fill=(255, 250, 232), outline=outline, width=2)
    font = load_font(
        max(12, int(round(badge_radius * 1.25))),
        bold=True,
        font_family=str(params.font_family) or None,
    )
    text_bbox = draw.textbbox((0, 0), str(dart.label), font=font)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    draw_text_traced(
        draw,
        (float(cx - (0.5 * text_w) - float(text_bbox[0])), float(cy - (0.5 * text_h) - float(text_bbox[1]))),
        str(dart.label),
        font=font,
        fill=(24, 28, 34),
        role="option_label",
        required=True,
    )
    return bbox


def _panel_bbox(*, params: DartboardRenderParams) -> Tuple[int, int, int, int]:
    """Return a backing panel bbox that follows the jittered dartboard."""

    cx = float(params.board_center_x_px)
    cy = float(params.board_center_y_px)
    radius = float(params.board_radius_px)
    x0 = max(10, int(round(cx - radius - 62.0)))
    y0 = max(10, int(round(cy - radius - 44.0)))
    x1 = min(int(params.canvas_width) - 10, int(round(cx + radius + 62.0)))
    y1 = min(int(params.canvas_height) - 10, int(round(cy + radius + 56.0)))
    return (int(x0), int(y0), int(x1), int(max(y0 + 80, y1)))


def render_darts_scene(
    *,
    darts: Sequence[DartInstance],
    background: Image.Image,
    style_variant: str,
    params: DartboardRenderParams,
    dart_fill_color: Tuple[int, int, int] | None = None,
    dart_fill_min_lab_distance: float | None = None,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedDartsScene:
    """Render the simplified board and darts."""

    image = background.convert("RGBA")
    panel_bbox = _panel_bbox(params=params)
    if panel_style is not None:
        draw_panel_scene_chrome(
            ImageDraw.Draw(image),
            bbox=panel_bbox,
            style=panel_style,
            radius=24,
            border_width=3,
        )
    board_meta = _draw_board(image, params=params, style_variant=str(style_variant))
    draw = ImageDraw.Draw(image)

    dart_specs: List[RenderedDartSpec] = []
    scene_entities: List[Dict[str, Any]] = []
    dart_bboxes: Dict[str, List[float]] = {}
    dart_centers: Dict[str, List[float]] = {}
    dart_label_bboxes: Dict[str, List[float]] = {}
    for dart in darts:
        bbox = _draw_dart(
            draw,
            dart=dart,
            params=params,
            style_variant=str(style_variant),
            dart_fill_color=dart_fill_color,
        )
        center = (round(float(dart.x_px), 3), round(float(dart.y_px), 3))
        dart_bboxes[str(dart.dart_id)] = [float(v) for v in bbox]
        dart_centers[str(dart.dart_id)] = [float(center[0]), float(center[1])]
        dart_specs.append(
            RenderedDartSpec(
                dart_id=str(dart.dart_id),
                label=None if dart.label is None else str(dart.label),
                area_kind=str(dart.area_kind),
                sector_value=None if dart.sector_value is None else int(dart.sector_value),
                score=int(dart.score),
                center_px=center,
                bbox_px=bbox,
                is_marked=bool(dart.is_marked),
            )
        )
        scene_entities.append(
            {
                "entity_id": str(dart.dart_id),
                "entity_type": "dart",
                "bbox_px": [float(v) for v in bbox],
                "attrs": {
                    "label": None if dart.label is None else str(dart.label),
                    "area_kind": str(dart.area_kind),
                    "sector_value": None if dart.sector_value is None else int(dart.sector_value),
                    "score": int(dart.score),
                    "is_marked": bool(dart.is_marked),
                },
            }
        )
    for dart in darts:
        label_bbox = _draw_dart_label(
            draw,
            dart=dart,
            params=params,
            style_variant=str(style_variant),
        )
        if label_bbox is not None:
            dart_label_bboxes[str(dart.dart_id)] = [float(value) for value in label_bbox]

    return RenderedDartsScene(
        image=image,
        dart_specs=tuple(dart_specs),
        scene_entities=tuple(scene_entities),
        render_map={
            "board": dict(board_meta),
            "dart_bboxes_px": dict(dart_bboxes),
            "dart_centers_px": dict(dart_centers),
            "dart_label_bboxes_px": dict(dart_label_bboxes),
            "dart_fill_color": None if dart_fill_color is None else [int(v) for v in dart_fill_color],
            "dart_fill_min_lab_distance": None
            if dart_fill_min_lab_distance is None
            else round(float(dart_fill_min_lab_distance), 3),
            "layout_jitter": dict(params.layout_jitter_meta or {}),
            "scene_panel_bbox_px": [int(value) for value in panel_bbox],
            "panel_scene_style": {}
            if panel_style is None
            else game_panel_scene_style_metadata(panel_style),
            "font_family": str(params.font_family),
        },
    )


def _allowed_panel_treatments(params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Resolve optional panel-scene treatment filters from rendering config."""

    raw = params.get(
        "panel_scene_treatments",
        group_default(render_defaults, "panel_scene_treatments", None),
    )
    if isinstance(raw, str):
        return (str(raw),)
    if raw is None:
        return None
    return tuple(str(item) for item in raw)


def render_darts_task_scene(
    *,
    darts: Sequence[DartInstance],
    style_variant: str,
    render_params: DartboardRenderParams,
    render_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
) -> RenderedDartsTaskContext:
    """Render a full darts task image with shared panel style and post-noise."""

    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{DARTS_NAMESPACE}.panel_scene_style",
        treatments=_allowed_panel_treatments(params, render_defaults),
        treatment_weights=params.get(
            "panel_scene_treatment_weights",
            group_default(render_defaults, "panel_scene_treatment_weights", None),
        ),
        palette_weights=params.get(
            "panel_scene_palette_weights",
            group_default(render_defaults, "panel_scene_palette_weights", None),
        ),
    )
    color_rng = spawn_rng(int(instance_seed), f"{DARTS_NAMESPACE}.dart_color")
    dart_fill_color, dart_fill_min_lab_distance = sample_dart_marker_color(
        color_rng,
        style_variant=str(style_variant),
        min_lab_distance=40.0,
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered_scene = render_darts_scene(
        darts=tuple(darts),
        background=background,
        style_variant=str(style_variant),
        params=render_params,
        dart_fill_color=dart_fill_color,
        dart_fill_min_lab_distance=float(dart_fill_min_lab_distance),
        panel_style=panel_style,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedDartsTaskContext(
        image=image,
        rendered_scene=rendered_scene,
        panel_style_meta=dict(panel_style_meta),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        text_style_meta={
            "font_family": str(render_params.font_family),
            "font_asset": get_font_family_record(str(render_params.font_family)).to_trace(),
        },
    )


__all__ = [
    "DARTBOARD_RADIUS_FRACTIONS",
    "DARTBOARD_SAMPLE_RADIUS_FRACTIONS",
    "DartboardRenderParams",
    "RenderedDartsTaskContext",
    "RenderedDartsScene",
    "STANDARD_DART_SECTORS",
    "dartboard_anchor_colors",
    "polar_to_xy",
    "render_darts_scene",
    "render_darts_task_scene",
    "sample_dart_marker_color",
]
