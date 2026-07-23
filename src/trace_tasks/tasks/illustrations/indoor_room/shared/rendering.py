"""Indoor-room illustration scene with semantic furniture and surfaces."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default
from ...shared.canvas_profiles import resolve_profile_render_params
from ...shared.render_geometry import scale_bbox as _scale_bbox, scale_points as _scale_points
from ...shared.object_library import (
    BBox,
    RGB,
    STYLE_IDS,
    aspect_ratio_for_object,
    choose_object_colors,
    display_name_for_object_type,
)
from ...shared.object_catalog import variant_ids_with_tag
from ...shared.object_rendering import (
    IllustrationObjectSpec,
    RenderContext,
    make_vector_scene_object_record,
    object_record_for_spec,
    render_illustration_object,
    render_vector_scene_object,
)
from ...shared.object_variants import RENDERER_STYLE_VECTOR
from ...shared.style_registry import resolve_art_style_weights, style_outline_params


INDOOR_THEME_IDS: Tuple[str, ...] = variant_ids_with_tag("indoor_theme")
INDOOR_OBJECT_TYPES: Tuple[str, ...] = variant_ids_with_tag("indoor_object")
INDOOR_SURFACE_TYPES: Tuple[str, ...] = variant_ids_with_tag("indoor_surface")
INDOOR_CONTAINER_TYPES: Tuple[str, ...] = variant_ids_with_tag("indoor_container")
INDOOR_FURNITURE_TYPES: Tuple[str, ...] = variant_ids_with_tag("indoor_furniture")
ANNOTATION_BBOX_MIN_SIDE_PX = 24.0
_INDOOR_OBJECT_MIN_PLACEMENT_SIZE: Dict[str, Tuple[float, float]] = {
    "bottle": (58.0, 42.0),
    "candle": (58.0, 42.0),
    "egg": (42.0, 48.0),
    "flower": (58.0, 42.0),
    "lightbulb": (52.0, 42.0),
    "mug": (52.0, 42.0),
    "mushroom": (54.0, 42.0),
    "potted_plant": (56.0, 42.0),
    "vase": (56.0, 42.0),
}


def _indoor_object_min_placement_size(object_type: str) -> Tuple[float, float]:
    """Return scene-local placement floors needed for countable object bboxes."""

    return _INDOOR_OBJECT_MIN_PLACEMENT_SIZE.get(str(object_type), (42.0, 42.0))


@dataclass(frozen=True)
class IndoorSurface:
    surface_id: str
    surface_type: str
    label: str
    bbox_xyxy: BBox
    support_bbox_xyxy: BBox
    furniture_id: str | None
    attributes: Mapping[str, Any]
    object_record: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class IndoorContainer:
    container_id: str
    container_type: str
    label: str
    bbox_xyxy: BBox
    interior_bbox_xyxy: BBox
    attributes: Mapping[str, Any]
    object_record: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class IndoorFurniture:
    furniture_id: str
    furniture_type: str
    label: str
    bbox_xyxy: BBox
    attributes: Mapping[str, Any]
    object_record: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class IndoorObjectSpec:
    object_type: str
    placement_kind: str
    target_type: str
    role: str = "distractor"


@dataclass(frozen=True)
class IndoorObjectPlacement:
    object_id: str
    object_type: str
    bbox_xyxy: BBox
    placement_kind: str
    surface_id: str | None
    surface_type: str | None
    surface_contact_px: Tuple[float, float] | None
    surface_depth: float | None
    container_id: str | None
    container_type: str | None
    region_relation: str | None
    region_furniture_id: str | None
    primary_color_rgb: RGB
    accent_color_rgb: RGB
    style_id: str
    relations: Mapping[str, Any]
    role: str
    object_record: Mapping[str, Any]


@dataclass(frozen=True)
class RenderedIndoorRoomScene:
    image: Image.Image
    theme_id: str
    surfaces: Tuple[IndoorSurface, ...]
    containers: Tuple[IndoorContainer, ...]
    furniture: Tuple[IndoorFurniture, ...]
    objects: Tuple[Any, ...]
    placements: Tuple[IndoorObjectPlacement, ...]
    canvas_width: int
    canvas_height: int
    render_scale: int
    style_id: str


def _choose_weighted(rng, weights: Mapping[str, float], support: Sequence[str]) -> str:
    choices = [(value, max(0.0, float(weights.get(value, 0.0)))) for value in support]
    total = sum(weight for _value, weight in choices)
    if total <= 0.0:
        return str(rng.choice(tuple(support)))
    threshold = float(rng.random()) * total
    running = 0.0
    for value, weight in choices:
        running += float(weight)
        if running >= threshold:
            return str(value)
    return str(choices[-1][0])


def _jitter_rgb(rng, color: RGB, amount: int = 16) -> RGB:
    return tuple(max(0, min(255, int(channel) + int(rng.randint(-int(amount), int(amount))))) for channel in color)  # type: ignore[return-value]


def _as_rgb(value: Any, fallback: RGB) -> RGB:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
        return tuple(int(max(0, min(255, int(channel)))) for channel in value)  # type: ignore[return-value]
    return tuple(int(channel) for channel in fallback)  # type: ignore[return-value]


def _blend_rgb(a: RGB, b: RGB, t: float) -> RGB:
    mix = max(0.0, min(1.0, float(t)))
    return tuple(int(round((1.0 - mix) * int(left) + mix * int(right))) for left, right in zip(a, b))  # type: ignore[return-value]


def _scale_design_box(box: Sequence[float], *, sx: float, sy: float) -> BBox:
    return (
        round(float(box[0]) * float(sx), 3),
        round(float(box[1]) * float(sy), 3),
        round(float(box[2]) * float(sx), 3),
        round(float(box[3]) * float(sy), 3),
    )


def _scale_design_point(point: Sequence[float], *, sx: float, sy: float) -> Tuple[float, float]:
    return (round(float(point[0]) * float(sx), 3), round(float(point[1]) * float(sy), 3))


def _plane_from_surface_box(x0: float, back_y: float, x1: float, front_y: float, *, inset_fraction: float = 0.07) -> Dict[str, Tuple[float, float]]:
    inset = max(12.0, float(x1 - x0) * float(inset_fraction))
    return {
        "back_left": (round(float(x0) + inset, 3), round(float(back_y), 3)),
        "back_right": (round(float(x1) - inset, 3), round(float(back_y), 3)),
        "front_left": (round(float(x0), 3), round(float(front_y), 3)),
        "front_right": (round(float(x1), 3), round(float(front_y), 3)),
    }


def _scale_plane(plane: Mapping[str, Tuple[float, float]], *, sx: float, sy: float) -> Dict[str, Tuple[float, float]]:
    return {str(key): _scale_design_point(value, sx=sx, sy=sy) for key, value in plane.items()}


def _draw_background(draw: ImageDraw.ImageDraw, *, rng, theme_id: str, style_id: str, width: int, height: int, scale: int) -> None:
    """Draw the room shell and decorative context without task-specific marks."""

    palettes = {
        "living_room": ((235, 226, 211), (196, 174, 151)),
        "kitchen": ((226, 238, 239), (199, 207, 194)),
        "study": ((225, 230, 214), (177, 154, 126)),
        "bedroom": ((235, 224, 235), (202, 184, 196)),
    }
    style = str(style_id)
    wall_base, floor_base = palettes.get(str(theme_id), palettes["living_room"])
    wall = _jitter_rgb(rng, wall_base, amount=10)
    floor = _jitter_rgb(rng, floor_base, amount=12)
    s = int(scale)
    draw.rectangle((0, 0, int(width) * s, int(height) * s), fill=tuple(wall))
    floor_y = int(round(float(rng.uniform(0.605, 0.635)) * int(height) * s))
    draw.rectangle((0, floor_y, int(width) * s, int(height) * s), fill=tuple(floor))
    horizon = _blend_rgb(floor, (96, 88, 82), 0.35)
    draw.line([(0, floor_y), (int(width) * s, floor_y)], fill=horizon, width=max(1, 2 * s))
    baseboard_h = max(5 * s, int(round(float(rng.uniform(8.0, 15.0)) * s)))
    draw.rectangle((0, floor_y - baseboard_h, int(width) * s, floor_y), fill=_blend_rgb(wall, floor, 0.34))
    draw.line([(0, floor_y - baseboard_h), (int(width) * s, floor_y - baseboard_h)], fill=_blend_rgb(horizon, (255, 255, 255), 0.25), width=max(1, s))

    floor_pattern = str(rng.choice(("planks", "diagonal_planks", "tiles", "plain_rug_band")))
    if style == "flat_vector" and floor_pattern != "tiles":
        floor_pattern = str(rng.choice(("plain_rug_band", "planks")))
    pattern_rgb = _blend_rgb(floor, (80, 72, 65), 0.16 if style != "outlined_cartoon" else 0.26)
    if floor_pattern == "tiles":
        spacing = int(round(float(rng.uniform(62.0, 92.0)) * s))
        offset = int(round(float(rng.uniform(0.0, spacing)) if spacing > 0 else 0.0))
        for x in range(offset, int(width) * s + spacing, max(1, spacing)):
            draw.line([(x, floor_y), (x, int(height) * s)], fill=pattern_rgb, width=max(1, s))
        for y in range(floor_y + spacing, int(height) * s, max(1, spacing)):
            draw.line([(0, y), (int(width) * s, y)], fill=pattern_rgb, width=max(1, s))
    elif floor_pattern in {"planks", "diagonal_planks"}:
        spacing = int(round(float(rng.uniform(44.0, 68.0)) * s))
        for y in range(floor_y + spacing, int(height) * s, max(1, spacing)):
            draw.line([(0, y), (int(width) * s, y)], fill=pattern_rgb, width=max(1, s))
        if floor_pattern == "diagonal_planks":
            step = int(round(float(rng.uniform(105.0, 145.0)) * s))
            for x in range(-int(height) * s, int(width) * s + step, max(1, step)):
                draw.line([(x, floor_y), (x + int(height) * s, int(height) * s)], fill=_blend_rgb(pattern_rgb, floor, 0.35), width=max(1, s))
    else:
        band_h = int(round(float(rng.uniform(18.0, 28.0)) * s))
        band_y = floor_y + int(round(float(rng.uniform(120.0, 190.0)) * s))
        if band_y + band_h < int(height) * s:
            draw.rectangle((0, band_y, int(width) * s, band_y + band_h), fill=_blend_rgb(floor, wall, 0.18))

    window_w = float(rng.uniform(170.0, 230.0)) * float(width) / 1280.0
    window_h = float(rng.uniform(130.0, 170.0)) * float(height) / 840.0
    window_x0 = float(rng.uniform(500.0, 600.0)) * float(width) / 1280.0
    window_y0 = float(rng.uniform(76.0, 118.0)) * float(height) / 840.0
    window = _scale_bbox((window_x0, window_y0, window_x0 + window_w, window_y0 + window_h), s)
    window_style = str(rng.choice(("plain", "curtains", "deep_frame", "arched_top")))
    frame_rgb = _jitter_rgb(rng, (111, 126, 136), amount=8)
    if style in {"paper_cutout", "soft_shadow"}:
        shadow_pad = 5 * s
        draw.rounded_rectangle((window[0] + shadow_pad, window[1] + shadow_pad, window[2] + shadow_pad, window[3] + shadow_pad), radius=max(1, 10 * s), fill=_blend_rgb(wall, (85, 78, 72), 0.12))
    if window_style == "curtains":
        curtain_rgb = _jitter_rgb(rng, rng.choice(((171, 107, 108), (107, 133, 164), (145, 137, 92), (143, 119, 151))), amount=12)
        rod_y = window[1] - 14 * s
        draw.line([(window[0] - 22 * s, rod_y), (window[2] + 22 * s, rod_y)], fill=_blend_rgb(curtain_rgb, (70, 60, 54), 0.30), width=max(2, 3 * s))
        draw.rounded_rectangle((window[0] - 24 * s, window[1] - 4 * s, window[0] + 18 * s, window[3] + 12 * s), radius=max(1, 8 * s), fill=curtain_rgb)
        draw.rounded_rectangle((window[2] - 18 * s, window[1] - 4 * s, window[2] + 24 * s, window[3] + 12 * s), radius=max(1, 8 * s), fill=_jitter_rgb(rng, curtain_rgb, amount=8))
    if window_style == "arched_top":
        arch = (window[0], window[1] - int(0.24 * (window[2] - window[0])), window[2], window[1] + int(0.52 * (window[3] - window[1])))
        draw.pieslice(arch, 180, 360, fill=(206, 226, 238), outline=frame_rgb, width=max(1, 3 * s))
    draw.rounded_rectangle(window, radius=max(1, 8 * s), fill=(206, 226, 238), outline=frame_rgb, width=max(1, (4 if style == "outlined_cartoon" else 3) * s))
    wx0, wy0, wx1, wy1 = window
    if window_style == "deep_frame":
        inner = (wx0 + 11 * s, wy0 + 11 * s, wx1 - 11 * s, wy1 - 11 * s)
        draw.rounded_rectangle(inner, radius=max(1, 5 * s), outline=_blend_rgb(frame_rgb, (255, 255, 255), 0.20), width=max(1, 2 * s))
    draw.line([((wx0 + wx1) // 2, wy0), ((wx0 + wx1) // 2, wy1)], fill=frame_rgb, width=max(1, 2 * s))
    draw.line([(wx0, (wy0 + wy1) // 2), (wx1, (wy0 + wy1) // 2)], fill=frame_rgb, width=max(1, 2 * s))

    art_count = int(rng.choice((0, 1, 1, 2)))
    art_slots = [
        (float(rng.uniform(120.0, 230.0)), float(rng.uniform(92.0, 148.0))),
        (float(rng.uniform(930.0, 1040.0)), float(rng.uniform(88.0, 152.0))),
    ]
    for index in range(art_count):
        ax, ay = art_slots[index]
        aw = float(rng.uniform(78.0, 116.0)) * float(width) / 1280.0
        ah = float(rng.uniform(52.0, 78.0)) * float(height) / 840.0
        frame = _scale_bbox((ax, ay, ax + aw, ay + ah), s)
        draw.rounded_rectangle(frame, radius=max(1, 5 * s), fill=_blend_rgb(wall, (255, 255, 255), 0.34), outline=_blend_rgb(horizon, (55, 48, 42), 0.26), width=max(1, 2 * s))
        ix0, iy0, ix1, iy1 = frame[0] + 7 * s, frame[1] + 7 * s, frame[2] - 7 * s, frame[3] - 7 * s
        draw.rectangle((ix0, iy0, ix1, iy1), fill=_jitter_rgb(rng, (204, 220, 211), amount=14))
        draw.polygon([(ix0, iy1), (ix0 + int(0.42 * (ix1 - ix0)), iy0 + int(0.50 * (iy1 - iy0))), (ix0 + int(0.72 * (ix1 - ix0)), iy1)], fill=_jitter_rgb(rng, (111, 151, 105), amount=12))
        draw.ellipse((ix1 - 22 * s, iy0 + 8 * s, ix1 - 8 * s, iy0 + 22 * s), fill=_jitter_rgb(rng, (232, 184, 79), amount=10))


def _layout(rng, *, width: int, height: int, theme_id: str, style_id: str) -> Tuple[Tuple[IndoorFurniture, ...], Tuple[IndoorSurface, ...], Tuple[IndoorContainer, ...]]:
    """Sample furniture, surface, and container geometry for one coherent room."""

    sx = float(width) / 1280.0
    sy = float(height) / 840.0
    style = str(style_id)
    wood_base = {
        "living_room": (171, 123, 80),
        "kitchen": (183, 142, 91),
        "study": (151, 103, 68),
        "bedroom": (178, 133, 88),
    }.get(str(theme_id), (171, 123, 80))
    wood = _jitter_rgb(rng, wood_base, amount=18)
    wood_dark = _jitter_rgb(rng, (105, 73, 48), amount=10)
    wood_light = _jitter_rgb(rng, (205, 166, 107), amount=18)
    sofa_fill = _jitter_rgb(rng, rng.choice(((105, 137, 164), (132, 126, 166), (137, 157, 130), (168, 124, 116))), amount=16)
    sofa_back = _jitter_rgb(rng, tuple(min(255, int(channel) + 14) for channel in sofa_fill), amount=8)
    rug_fill = _jitter_rgb(rng, rng.choice(((153, 130, 166), (143, 157, 128), (168, 135, 118), (128, 151, 166))), amount=12)
    rug_outline = _jitter_rgb(rng, (104, 91, 112), amount=10)

    layout_variant = str(rng.choice(("cabinet_right_sofa_left", "cabinet_left_sofa_right", "wide_left_storage", "wide_right_storage")))
    table_style = str(rng.choice(("straight_legs", "tapered_legs", "trestle")))
    sofa_style = str(rng.choice(("block", "rounded_arms", "split_cushions")))
    cabinet_style = str(rng.choice(("panel_doors", "open_shelves", "mixed_drawers")))
    shelf_style = str(rng.choice(("plank", "brackets", "cubby")))
    rug_pattern = str(rng.choice(("plain", "border", "stripes", "dots")))
    container_style = str(rng.choice(("plain", "slatted", "woven")))
    if style == "paper_cutout":
        rug_pattern = str(rng.choice(("border", "dots", "stripes")))
    elif style == "flat_vector":
        table_style = str(rng.choice(("straight_legs", "tapered_legs")))

    table_w = float(rng.uniform(360.0, 455.0))
    if layout_variant in {"cabinet_right_sofa_left", "wide_right_storage"}:
        table_x0 = float(rng.uniform(390.0, 480.0))
    else:
        table_x0 = float(rng.uniform(410.0, 520.0))
    table_x1 = min(850.0, table_x0 + table_w)
    table_back_y = float(rng.uniform(400.0, 430.0))
    table_front_y = table_back_y + float(rng.uniform(30.0, 42.0))
    table_lip_bottom = table_front_y + float(rng.uniform(34.0, 48.0))
    table_bottom = min(674.0, table_lip_bottom + float(rng.uniform(132.0, 164.0)))
    table_plane = _plane_from_surface_box(table_x0, table_back_y, table_x1, table_front_y, inset_fraction=float(rng.uniform(0.055, 0.09)))

    sofa_side = "left" if layout_variant in {"cabinet_right_sofa_left", "wide_right_storage"} else "right"
    cabinet_side = "right" if layout_variant in {"cabinet_right_sofa_left", "wide_right_storage"} else "left"
    shelf_side = "left" if cabinet_side == "right" else "right"
    basket_side = "right" if cabinet_side == "right" else "left"
    box_side = "left" if cabinet_side == "right" else "right"

    sofa_w = float(rng.uniform(270.0, 330.0))
    if sofa_side == "left":
        sofa_x0 = float(rng.uniform(58.0, 106.0))
        sofa_x1 = min(405.0, sofa_x0 + sofa_w)
    else:
        sofa_x0 = float(rng.uniform(850.0, 925.0))
        sofa_x1 = min(1218.0, sofa_x0 + sofa_w)
    sofa_y0 = float(rng.uniform(418.0, 448.0))
    sofa_h = float(rng.uniform(190.0, 232.0))
    sofa_y1 = min(670.0, sofa_y0 + sofa_h)

    cab_w = float(rng.uniform(270.0, 322.0))
    if cabinet_side == "right":
        cab_x0 = float(rng.uniform(878.0, 925.0))
        cab_x1 = min(1210.0, cab_x0 + cab_w)
    else:
        cab_x0 = float(rng.uniform(64.0, 122.0))
        cab_x1 = min(410.0, cab_x0 + cab_w)
    cab_y0 = float(rng.uniform(246.0, 292.0))
    cab_h = float(rng.uniform(300.0, 354.0))
    cab_y1 = min(600.0, cab_y0 + cab_h)
    counter_front_y = cab_y0 + float(rng.uniform(26.0, 36.0))
    counter_lip_bottom = counter_front_y + float(rng.uniform(34.0, 46.0))
    counter_plane = _plane_from_surface_box(cab_x0 + 10.0, cab_y0, cab_x1 - 10.0, counter_front_y, inset_fraction=float(rng.uniform(0.07, 0.10)))

    shelf_w = float(rng.uniform(320.0, 410.0))
    if shelf_side == "left":
        shelf_x0 = float(rng.uniform(72.0, 132.0))
    else:
        shelf_x0 = float(rng.uniform(780.0, 860.0))
    shelf_x1 = shelf_x0 + shelf_w
    shelf_back_y = float(rng.uniform(126.0, 162.0))
    shelf_front_y = shelf_back_y + float(rng.uniform(22.0, 32.0))
    shelf_lip_bottom = shelf_front_y + float(rng.uniform(24.0, 36.0))
    shelf_plane = _plane_from_surface_box(shelf_x0, shelf_back_y, shelf_x1, shelf_front_y, inset_fraction=float(rng.uniform(0.045, 0.065)))

    basket_w = float(rng.uniform(220.0, 260.0))
    basket_h = float(rng.uniform(148.0, 182.0))
    if basket_side == "right":
        basket_x0 = float(rng.uniform(820.0, 890.0))
    else:
        basket_x0 = float(rng.uniform(58.0, 115.0))
    basket_y0 = float(rng.uniform(635.0, 665.0))
    box_w = float(rng.uniform(220.0, 270.0))
    box_h = float(rng.uniform(142.0, 176.0))
    if box_side == "left":
        box_x0 = float(rng.uniform(132.0, 188.0))
    else:
        box_x0 = float(rng.uniform(820.0, 895.0))
    box_y0 = float(rng.uniform(638.0, 668.0))
    floor_container_bottom = 812.0
    basket_y0 = min(basket_y0, floor_container_bottom - basket_h)
    box_y0 = min(box_y0, floor_container_bottom - box_h)
    drawer_w = float(cab_x1 - cab_x0) * float(rng.uniform(0.62, 0.72))
    drawer_x0 = cab_x0 + float(cab_x1 - cab_x0) * float(rng.uniform(0.20, 0.26))
    drawer_y0 = cab_y0 + float(cab_y1 - cab_y0) * float(rng.uniform(0.48, 0.56))
    drawer_h = float(rng.uniform(108.0, 132.0))

    furniture = (
        IndoorFurniture(
            "furniture_table",
            "table",
            "table",
            _scale_design_box((table_x0, table_back_y, table_x1, table_bottom), sx=sx, sy=sy),
            {
                "role": "central_table",
                "layout_variant": layout_variant,
                "style_id": style,
                "table_style": table_style,
                "rug_pattern": rug_pattern,
                "fill_rgb": wood,
                "dark_rgb": wood_dark,
                "top_rgb": _jitter_rgb(rng, wood_light, amount=10),
                "lip_rgb": _jitter_rgb(rng, wood, amount=12),
                "leg_width": round(float(rng.uniform(28.0, 38.0)) * sx, 3),
                "rug_bbox": _scale_design_box((table_x0 - 50.0, table_bottom - 26.0, table_x1 + 50.0, table_bottom + float(rng.uniform(118.0, 152.0))), sx=sx, sy=sy),
                "rug_fill_rgb": rug_fill,
                "rug_outline_rgb": rug_outline,
            },
        ),
        IndoorFurniture(
            "furniture_sofa",
            "sofa",
            "sofa",
            _scale_design_box((sofa_x0, sofa_y0, sofa_x1, sofa_y1), sx=sx, sy=sy),
            {"role": "seating", "layout_variant": layout_variant, "side": sofa_side, "style_id": style, "sofa_style": sofa_style, "fill_rgb": sofa_fill, "back_fill_rgb": sofa_back, "outline_rgb": _jitter_rgb(rng, (69, 84, 101), amount=10)},
        ),
        IndoorFurniture(
            "furniture_cabinet",
            "cabinet",
            "cabinet",
            _scale_design_box((cab_x0, cab_y0, cab_x1, cab_y1), sx=sx, sy=sy),
            {"role": "storage", "layout_variant": layout_variant, "side": cabinet_side, "style_id": style, "cabinet_style": cabinet_style, "fill_rgb": _jitter_rgb(rng, wood, amount=10), "panel_rgb": _jitter_rgb(rng, wood_light, amount=12), "outline_rgb": wood_dark},
        ),
    )
    surfaces = (
        IndoorSurface(
            "surface_table",
            "table",
            "table",
            _scale_design_box((table_x0, table_back_y, table_x1, table_lip_bottom), sx=sx, sy=sy),
            _scale_design_box((table_x0 + 20.0, table_back_y - 88.0, table_x1 - 20.0, table_front_y), sx=sx, sy=sy),
            "furniture_table",
            {"plane": _scale_plane(table_plane, sx=sx, sy=sy), "lip_bottom_y": round(table_lip_bottom * sy, 3), "style_id": style, "surface_style": table_style, "top_fill_rgb": _jitter_rgb(rng, wood_light, amount=10), "lip_fill_rgb": _jitter_rgb(rng, wood, amount=12), "outline_rgb": wood_dark},
        ),
        IndoorSurface(
            "surface_shelf",
            "shelf",
            "shelf",
            _scale_design_box((shelf_x0, shelf_back_y, shelf_x1, shelf_lip_bottom), sx=sx, sy=sy),
            _scale_design_box((shelf_x0 + 20.0, shelf_back_y - 64.0, shelf_x1 - 20.0, shelf_front_y), sx=sx, sy=sy),
            None,
            {"plane": _scale_plane(shelf_plane, sx=sx, sy=sy), "lip_bottom_y": round(shelf_lip_bottom * sy, 3), "style_id": style, "shelf_style": shelf_style, "board_bbox": _scale_design_box((shelf_x0, shelf_back_y - 28.0, shelf_x1, shelf_back_y + 4.0), sx=sx, sy=sy), "top_fill_rgb": _jitter_rgb(rng, wood_light, amount=10), "lip_fill_rgb": _jitter_rgb(rng, wood_dark, amount=8), "outline_rgb": wood_dark},
        ),
        IndoorSurface(
            "surface_counter",
            "counter",
            "counter",
            _scale_design_box((cab_x0 + 10.0, cab_y0, cab_x1 - 10.0, counter_lip_bottom), sx=sx, sy=sy),
            _scale_design_box((cab_x0 + 30.0, cab_y0 - 62.0, cab_x1 - 30.0, counter_front_y), sx=sx, sy=sy),
            "furniture_cabinet",
            {"plane": _scale_plane(counter_plane, sx=sx, sy=sy), "lip_bottom_y": round(counter_lip_bottom * sy, 3), "style_id": style, "surface_style": cabinet_style, "top_fill_rgb": _jitter_rgb(rng, (222, 211, 188), amount=12), "lip_fill_rgb": _jitter_rgb(rng, wood_light, amount=12), "outline_rgb": _jitter_rgb(rng, (96, 82, 66), amount=8)},
        ),
    )
    containers = (
        IndoorContainer(
            "container_basket",
            "basket",
            "basket",
            _scale_design_box((basket_x0, basket_y0, basket_x0 + basket_w, basket_y0 + basket_h), sx=sx, sy=sy),
            _scale_design_box((basket_x0 + 22.0, basket_y0 - 54.0, basket_x0 + basket_w - 22.0, basket_y0 + 88.0), sx=sx, sy=sy),
            {"surface": "floor", "style_id": style, "container_style": container_style, "fill_rgb": _jitter_rgb(rng, (193, 145, 86), amount=14), "outline_rgb": _jitter_rgb(rng, (94, 69, 42), amount=10)},
        ),
        IndoorContainer(
            "container_box",
            "box",
            "box",
            _scale_design_box((box_x0, box_y0, box_x0 + box_w, box_y0 + box_h), sx=sx, sy=sy),
            _scale_design_box((box_x0 + 22.0, box_y0 - 52.0, box_x0 + box_w - 22.0, box_y0 + 88.0), sx=sx, sy=sy),
            {"surface": "floor", "style_id": style, "container_style": container_style, "fill_rgb": _jitter_rgb(rng, (177, 130, 72), amount=14), "outline_rgb": _jitter_rgb(rng, (99, 69, 38), amount=10)},
        ),
        IndoorContainer(
            "container_drawer",
            "drawer",
            "drawer",
            _scale_design_box((drawer_x0, drawer_y0, drawer_x0 + drawer_w, drawer_y0 + drawer_h), sx=sx, sy=sy),
            _scale_design_box((drawer_x0 + 20.0, drawer_y0 - 46.0, drawer_x0 + drawer_w - 20.0, drawer_y0 + 62.0), sx=sx, sy=sy),
            {"furniture_id": "furniture_cabinet", "style_id": style, "container_style": container_style, "fill_rgb": _jitter_rgb(rng, (151, 103, 69), amount=14), "outline_rgb": wood_dark},
        ),
    )
    return furniture, surfaces, containers


def _surface_plane(surface: str | IndoorSurface) -> Dict[str, Tuple[float, float]]:
    if isinstance(surface, IndoorSurface):
        raw_plane = surface.attributes.get("plane")
        if isinstance(raw_plane, Mapping):
            return {
                str(key): (float(value[0]), float(value[1]))
                for key, value in raw_plane.items()
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2
            }
        surface_type = str(surface.surface_type)
    else:
        surface_type = str(surface)
    planes = {
        "table": {
            "back_left": (440.0, 414.0),
            "back_right": (790.0, 414.0),
            "front_left": (410.0, 448.0),
            "front_right": (820.0, 448.0),
        },
        "shelf": {
            "back_left": (115.0, 148.0),
            "back_right": (442.0, 148.0),
            "front_left": (95.0, 176.0),
            "front_right": (462.0, 176.0),
        },
        "counter": {
            "back_left": (930.0, 248.0),
            "back_right": (1165.0, 248.0),
            "front_left": (910.0, 276.0),
            "front_right": (1185.0, 276.0),
        },
    }
    return dict(planes[str(surface_type)])


def _plane_point(plane: Mapping[str, Tuple[float, float]], *, x_fraction: float, depth: float) -> Tuple[float, float]:
    d = max(0.0, min(1.0, float(depth)))
    tx = max(0.0, min(1.0, float(x_fraction)))
    back_left = plane["back_left"]
    back_right = plane["back_right"]
    front_left = plane["front_left"]
    front_right = plane["front_right"]
    left = (
        float(back_left[0]) + d * (float(front_left[0]) - float(back_left[0])),
        float(back_left[1]) + d * (float(front_left[1]) - float(back_left[1])),
    )
    right = (
        float(back_right[0]) + d * (float(front_right[0]) - float(back_right[0])),
        float(back_right[1]) + d * (float(front_right[1]) - float(back_right[1])),
    )
    return (
        float(left[0]) + tx * (float(right[0]) - float(left[0])),
        float(left[1]) + tx * (float(right[1]) - float(left[1])),
    )


def _render_room_fixtures(
    draw: ImageDraw.ImageDraw,
    *,
    furniture: Sequence[IndoorFurniture],
    surfaces: Sequence[IndoorSurface],
    containers: Sequence[IndoorContainer],
    scale: int,
) -> Tuple[Tuple[IndoorFurniture, ...], Tuple[IndoorSurface, ...], Tuple[IndoorContainer, ...]]:
    """Render fixed room fixtures before task objects while preserving records."""

    furniture_by_type = {item.furniture_type: item for item in furniture}
    surface_by_type = {item.surface_type: item for item in surfaces}
    furniture_records: Dict[str, Mapping[str, Any]] = {}
    surface_records: Dict[str, Mapping[str, Any]] = {}
    container_records: Dict[str, Mapping[str, Any]] = {}

    table = furniture_by_type["table"]
    table_surface = surface_by_type["table"]
    style_id = str(table.attributes.get("style_id", "flat_vector"))
    table_semantic = {
        "furniture_type": str(table.furniture_type),
        "label": str(table.label),
        **dict(table.attributes),
    }
    table_visual = {**dict(table.attributes), "surface_bbox": tuple(float(v) for v in table_surface.bbox_xyxy), "draw_phase": "rug"}
    render_vector_scene_object(
        draw,
        object_id=str(table.furniture_id),
        object_type="furniture",
        bbox_xyxy=table.bbox_xyxy,
        renderer_id="indoor_furniture",
        renderer_variant_id=str(table.furniture_type),
        semantic_attributes=table_semantic,
        visual_attributes=table_visual,
        source_entity_type="indoor_furniture",
        render_scale=int(scale),
        style_id=style_id,
    )

    shelf = surface_by_type["shelf"]
    rendered_shelf = render_vector_scene_object(
        draw,
        object_id=str(shelf.surface_id),
        object_type="surface",
        bbox_xyxy=shelf.bbox_xyxy,
        renderer_id="indoor_surface",
        renderer_variant_id=str(shelf.surface_type),
        semantic_attributes={
            "surface_type": str(shelf.surface_type),
            "label": str(shelf.label),
            "furniture_id": shelf.furniture_id,
            **dict(shelf.attributes),
        },
        visual_attributes=dict(shelf.attributes),
        source_entity_type="indoor_surface",
        render_scale=int(scale),
        style_id=style_id,
    )
    surface_records[str(shelf.surface_id)] = rendered_shelf.object_record

    sofa = furniture_by_type["sofa"]
    rendered_sofa = render_vector_scene_object(
        draw,
        object_id=str(sofa.furniture_id),
        object_type="furniture",
        bbox_xyxy=sofa.bbox_xyxy,
        renderer_id="indoor_furniture",
        renderer_variant_id=str(sofa.furniture_type),
        semantic_attributes={
            "furniture_type": str(sofa.furniture_type),
            "label": str(sofa.label),
            **dict(sofa.attributes),
        },
        visual_attributes=dict(sofa.attributes),
        source_entity_type="indoor_furniture",
        render_scale=int(scale),
        style_id=style_id,
    )
    furniture_records[str(sofa.furniture_id)] = rendered_sofa.object_record

    cabinet = furniture_by_type["cabinet"]
    rendered_cabinet = render_vector_scene_object(
        draw,
        object_id=str(cabinet.furniture_id),
        object_type="furniture",
        bbox_xyxy=cabinet.bbox_xyxy,
        renderer_id="indoor_furniture",
        renderer_variant_id=str(cabinet.furniture_type),
        semantic_attributes={
            "furniture_type": str(cabinet.furniture_type),
            "label": str(cabinet.label),
            **dict(cabinet.attributes),
        },
        visual_attributes=dict(cabinet.attributes),
        source_entity_type="indoor_furniture",
        render_scale=int(scale),
        style_id=style_id,
    )
    furniture_records[str(cabinet.furniture_id)] = rendered_cabinet.object_record

    counter = surface_by_type["counter"]
    rendered_counter = render_vector_scene_object(
        draw,
        object_id=str(counter.surface_id),
        object_type="surface",
        bbox_xyxy=counter.bbox_xyxy,
        renderer_id="indoor_surface",
        renderer_variant_id=str(counter.surface_type),
        semantic_attributes={
            "surface_type": str(counter.surface_type),
            "label": str(counter.label),
            "furniture_id": counter.furniture_id,
            **dict(counter.attributes),
        },
        visual_attributes=dict(counter.attributes),
        source_entity_type="indoor_surface",
        render_scale=int(scale),
        style_id=style_id,
    )
    surface_records[str(counter.surface_id)] = rendered_counter.object_record

    rendered_table_surface = render_vector_scene_object(
        draw,
        object_id=str(table_surface.surface_id),
        object_type="surface",
        bbox_xyxy=table_surface.bbox_xyxy,
        renderer_id="indoor_surface",
        renderer_variant_id=str(table_surface.surface_type),
        semantic_attributes={
            "surface_type": str(table_surface.surface_type),
            "label": str(table_surface.label),
            "furniture_id": table_surface.furniture_id,
            **dict(table_surface.attributes),
        },
        visual_attributes=dict(table_surface.attributes),
        source_entity_type="indoor_surface",
        render_scale=int(scale),
        style_id=style_id,
    )
    surface_records[str(table_surface.surface_id)] = rendered_table_surface.object_record

    rendered_table_legs = render_vector_scene_object(
        draw,
        object_id=str(table.furniture_id),
        object_type="furniture",
        bbox_xyxy=table.bbox_xyxy,
        renderer_id="indoor_furniture",
        renderer_variant_id=str(table.furniture_type),
        semantic_attributes=table_semantic,
        visual_attributes={**dict(table.attributes), **dict(table_surface.attributes), "surface_bbox": tuple(float(v) for v in table_surface.bbox_xyxy), "draw_phase": "legs"},
        source_entity_type="indoor_furniture",
        render_scale=int(scale),
        style_id=style_id,
    )
    furniture_records[str(table.furniture_id)] = rendered_table_legs.object_record

    for container in containers:
        rendered_container = render_vector_scene_object(
            draw,
            object_id=str(container.container_id),
            object_type="container",
            bbox_xyxy=container.bbox_xyxy,
            renderer_id="indoor_container",
            renderer_variant_id=str(container.container_type),
            semantic_attributes={
                "container_type": str(container.container_type),
                "label": str(container.label),
                **dict(container.attributes),
            },
            visual_attributes=dict(container.attributes),
            source_entity_type="indoor_container",
            render_scale=int(scale),
            style_id=style_id,
        )
        container_records[str(container.container_id)] = rendered_container.object_record
    return (
        tuple(replace(item, object_record=furniture_records.get(str(item.furniture_id))) for item in furniture),
        tuple(replace(item, object_record=surface_records.get(str(item.surface_id))) for item in surfaces),
        tuple(replace(item, object_record=container_records.get(str(item.container_id))) for item in containers),
    )


def _surface_map(surfaces: Sequence[IndoorSurface]) -> Dict[str, IndoorSurface]:
    return {surface.surface_type: surface for surface in surfaces}


def _container_map(containers: Sequence[IndoorContainer]) -> Dict[str, IndoorContainer]:
    return {container.container_type: container for container in containers}


def _furniture_map(furniture: Sequence[IndoorFurniture]) -> Dict[str, IndoorFurniture]:
    return {item.furniture_type: item for item in furniture}


def _region_bbox(relation: str, furniture: IndoorFurniture, *, width: int, height: int) -> BBox:
    x0, y0, x1, y1 = [float(v) for v in furniture.bbox_xyxy]
    if relation == "left":
        box = (54.0, max(250.0, y0 - 110.0), max(88.0, x0 - 48.0), min(float(height) - 54.0, y1 + 110.0))
    elif relation == "right":
        box = (min(float(width) - 88.0, x1 + 48.0), max(250.0, y0 - 110.0), float(width) - 54.0, min(float(height) - 54.0, y1 + 110.0))
    elif relation == "above":
        box = (max(54.0, x0 - 170.0), 74.0, min(float(width) - 54.0, x1 + 170.0), max(120.0, y0 - 46.0))
    elif relation == "below":
        box = (max(54.0, x0 - 170.0), min(float(height) - 120.0, y1 + 36.0), min(float(width) - 54.0, x1 + 170.0), float(height) - 54.0)
    else:
        raise ValueError(f"unsupported relation {relation!r}")
    if box[2] - box[0] < 80.0 or box[3] - box[1] < 70.0:
        raise ValueError(f"relation region is too small for {furniture.furniture_type}/{relation}")
    return box


def _surface_contact_ratio(object_type: str) -> float:
    """Approximate where the visible drawing touches a supporting surface."""

    ratios = {
        "apple": 0.84,
        "backpack": 0.86,
        "bottle": 0.86,
        "book": 0.88,
        "bowl": 0.84,
        "camera": 0.78,
        "candle": 0.92,
        "clock": 0.90,
        "egg": 0.72,
        "flower": 0.88,
        "gift": 0.84,
        "key": 0.66,
        "lamp": 0.88,
        "lightbulb": 0.86,
        "mug": 0.82,
        "mushroom": 0.86,
        "pencil": 0.62,
        "plate": 0.78,
        "potted_plant": 0.92,
        "remote": 0.94,
        "ruler": 0.66,
        "rugby_ball": 0.81,
        "scissors": 0.78,
        "soccer_ball": 0.88,
        "spoon": 0.75,
        "teapot": 0.78,
        "vase": 0.86,
    }
    return float(ratios.get(str(object_type), 0.88))


def _slot_boxes(
    rng,
    specs: Sequence[Tuple[int, IndoorObjectSpec]],
    area: BBox,
    *,
    size_min_px: int,
    size_max_px: int,
    align_bottom: bool,
) -> Dict[int, BBox]:
    """Assign non-overlapping boxes inside a free region or container area."""

    if not specs:
        return {}
    x0, y0, x1, y1 = [float(v) for v in area]
    n = len(specs)
    area_w = max(1.0, x1 - x0)
    area_h = max(1.0, y1 - y0)
    if align_bottom:
        cols = max(1, int(n))
        rows = 1
    else:
        cols = max(1, int(math.ceil(math.sqrt(float(n) * area_w / max(1.0, area_h)))))
        rows = max(1, int(math.ceil(float(n) / float(cols))))
    cell_w = area_w / float(cols)
    cell_h = area_h / float(rows)
    ordered = list(specs)
    rng.shuffle(ordered)
    boxes: Dict[int, BBox] = {}
    for slot_index, (global_index, spec) in enumerate(ordered):
        row = slot_index // cols
        col = slot_index % cols
        cx = x0 + (float(col) + 0.5) * cell_w + float(rng.uniform(-0.12, 0.12)) * cell_w
        aspect = max(0.35, float(aspect_ratio_for_object(str(spec.object_type))))
        h = min(float(size_max_px), 0.72 * cell_h, 0.82 * cell_w / aspect)
        h = max(30.0, min(float(size_min_px), h) if h < float(size_min_px) else h)
        min_w, min_h = _indoor_object_min_placement_size(str(spec.object_type))
        h = max(float(min_h), h)
        w = max(26.0, float(min_w), min(0.82 * cell_w, h * aspect))
        if align_bottom:
            contact_y = y1 - 2.0 + float(rng.uniform(-0.01, 0.006)) * cell_h
            contact_ratio = _surface_contact_ratio(str(spec.object_type))
            yy0 = max(y0 + 2.0, contact_y - contact_ratio * h)
            yy1 = yy0 + h
        else:
            cy = y0 + (float(row) + 0.5) * cell_h + float(rng.uniform(-0.12, 0.12)) * cell_h
            yy0 = max(y0 + 2.0, min(y1 - h - 2.0, cy - 0.5 * h))
            yy1 = yy0 + h
        xx0 = max(x0 + 2.0, min(x1 - w - 2.0, cx - 0.5 * w))
        boxes[int(global_index)] = (float(xx0), float(yy0), float(xx0 + w), float(yy1))
    return boxes


def _surface_slot_boxes(
    rng,
    specs: Sequence[Tuple[int, IndoorObjectSpec]],
    surface: IndoorSurface,
    *,
    size_min_px: int,
    size_max_px: int,
) -> Tuple[Dict[int, BBox], Dict[int, Tuple[float, float]], Dict[int, float]]:
    """Place objects on perspective surface planes and record contact points."""

    if not specs:
        return {}, {}, {}
    n = len(specs)
    plane = _surface_plane(surface)
    back_rows_allowed = str(surface.surface_type) in {"table", "counter"} and n >= 5
    rows = 2 if back_rows_allowed else 1
    cols = max(1, int(math.ceil(float(n) / float(rows))))
    ordered = list(specs)
    rng.shuffle(ordered)
    support_x0, support_y0, support_x1, _support_y1 = [float(v) for v in surface.support_bbox_xyxy]
    boxes: Dict[int, BBox] = {}
    contact_points: Dict[int, Tuple[float, float]] = {}
    depths: Dict[int, float] = {}
    for slot_index, (global_index, spec) in enumerate(ordered):
        row = min(rows - 1, slot_index // cols)
        col = slot_index % cols
        x_fraction = (float(col) + 0.5) / float(cols)
        x_fraction += float(rng.uniform(-0.10, 0.10)) / float(max(1, cols))
        if rows == 1:
            base_depth = 0.78 if str(surface.surface_type) != "shelf" else 0.72
        else:
            base_depth = 0.43 if row == 0 else 0.82
        depth = max(0.18, min(0.92, float(base_depth) + float(rng.uniform(-0.04, 0.04))))
        contact_x, contact_y = _plane_point(plane, x_fraction=x_fraction, depth=depth)
        visible_scale = 0.82 + 0.22 * depth
        plane_left = _plane_point(plane, x_fraction=0.0, depth=depth)[0]
        plane_right = _plane_point(plane, x_fraction=1.0, depth=depth)[0]
        cell_w = max(24.0, (float(plane_right) - float(plane_left)) / float(cols))
        aspect = max(0.35, float(aspect_ratio_for_object(str(spec.object_type))))
        h = min(float(size_max_px) * visible_scale, 0.82 * cell_w / aspect)
        h = max(30.0, min(float(size_min_px) * visible_scale, h) if h < float(size_min_px) * visible_scale else h)
        min_w, min_h = _indoor_object_min_placement_size(str(spec.object_type))
        h = max(float(min_h), h)
        w = max(24.0, float(min_w), min(0.84 * cell_w, h * aspect))
        contact_ratio = _surface_contact_ratio(str(spec.object_type))
        xx0 = max(support_x0 + 2.0, min(support_x1 - w - 2.0, float(contact_x) - 0.5 * w))
        contact_x = xx0 + 0.5 * w
        yy0 = max(support_y0 + 2.0, float(contact_y) - contact_ratio * h)
        boxes[int(global_index)] = (float(xx0), float(yy0), float(xx0 + w), float(yy0 + h))
        contact_points[int(global_index)] = (round(float(contact_x), 3), round(float(contact_y), 3))
        depths[int(global_index)] = round(float(depth), 4)
    return boxes, contact_points, depths


def _draw_surface_shadow(draw: ImageDraw.ImageDraw, *, box: BBox, contact_px: Tuple[float, float], depth: float, scale: int) -> None:
    x0, _y0, x1, _y1 = [float(v) for v in box]
    cx, cy = [float(v) for v in contact_px]
    width = max(14.0, min(58.0, 0.58 * (x1 - x0)))
    height = max(4.0, 5.0 + 5.0 * float(depth))
    shadow = (cx - 0.5 * width, cy - 0.45 * height, cx + 0.5 * width, cy + 0.55 * height)
    draw.ellipse(_scale_bbox(shadow, scale), fill=(164, 151, 136))


def _relation_to_furniture(box: BBox, furniture: IndoorFurniture) -> Dict[str, Any]:
    cx = 0.5 * (float(box[0]) + float(box[2]))
    cy = 0.5 * (float(box[1]) + float(box[3]))
    fx = 0.5 * (float(furniture.bbox_xyxy[0]) + float(furniture.bbox_xyxy[2]))
    fy = 0.5 * (float(furniture.bbox_xyxy[1]) + float(furniture.bbox_xyxy[3]))
    return {
        "left": bool(cx < fx),
        "right": bool(cx > fx),
        "above": bool(cy < fy),
        "below": bool(cy > fy),
        "center_px": [round(float(cx), 3), round(float(cy), 3)],
        "furniture_center_px": [round(float(fx), 3), round(float(fy), 3)],
    }


def render_indoor_room_scene(
    *,
    rng,
    object_specs: Sequence[IndoorObjectSpec],
    canvas_width: int = 1280,
    canvas_height: int = 840,
    render_scale: int = 2,
    theme_weights: Mapping[str, float] | None = None,
    style_weights: Mapping[str, float] | None = None,
    object_size_min_px: int = 52,
    object_size_max_px: int = 86,
    highlight_container_type: str | None = None,
) -> RenderedIndoorRoomScene:
    """Render the full indoor-room scene from neutral object placement specs."""

    width = int(canvas_width)
    height = int(canvas_height)
    scale = max(1, int(render_scale))
    theme_id = _choose_weighted(rng, theme_weights or {theme: 1.0 for theme in INDOOR_THEME_IDS}, INDOOR_THEME_IDS)
    style_id = _choose_weighted(rng, style_weights or {style: 1.0 for style in STYLE_IDS}, STYLE_IDS)
    _, _, draw_style_shadow = style_outline_params(str(style_id))
    furniture, surfaces, containers = _layout(rng, width=width, height=height, theme_id=str(theme_id), style_id=str(style_id))
    surfaces_by_type = _surface_map(surfaces)
    containers_by_type = _container_map(containers)
    furniture_by_type = _furniture_map(furniture)

    grouped: Dict[Tuple[str, str, str | None], List[Tuple[int, IndoorObjectSpec]]] = {}
    for index, spec in enumerate(object_specs):
        relation = None
        target_type = str(spec.target_type)
        if str(spec.placement_kind) == "region":
            target_type, relation = target_type.split(":", 1)
        grouped.setdefault((str(spec.placement_kind), target_type, relation), []).append((index, spec))

    boxes: Dict[int, BBox] = {}
    surface_contact_points: Dict[int, Tuple[float, float]] = {}
    surface_depths: Dict[int, float] = {}
    for (placement_kind, target_type, relation), specs in grouped.items():
        if placement_kind == "surface":
            surface = surfaces_by_type[str(target_type)]
            surface_boxes, contact_points, depths = _surface_slot_boxes(
                rng,
                specs,
                surface,
                size_min_px=int(object_size_min_px),
                size_max_px=int(object_size_max_px),
            )
            boxes.update(surface_boxes)
            surface_contact_points.update(contact_points)
            surface_depths.update(depths)
        elif placement_kind == "container":
            container = containers_by_type[str(target_type)]
            boxes.update(
                _slot_boxes(
                    rng,
                    specs,
                    container.interior_bbox_xyxy,
                    size_min_px=max(34, int(object_size_min_px) - 12),
                    size_max_px=max(40, int(object_size_max_px) - 16),
                    align_bottom=False,
                )
            )
        elif placement_kind == "region":
            assert relation is not None
            area = _region_bbox(str(relation), furniture_by_type[str(target_type)], width=width, height=height)
            boxes.update(
                _slot_boxes(
                    rng,
                    specs,
                    area,
                    size_min_px=int(object_size_min_px),
                    size_max_px=int(object_size_max_px),
                    align_bottom=False,
                )
            )
        else:
            raise ValueError(f"unsupported indoor placement kind {placement_kind!r}")

    image = Image.new("RGB", (width * scale, height * scale), (246, 246, 241))
    draw = ImageDraw.Draw(image)
    _draw_background(draw, rng=rng, theme_id=theme_id, style_id=str(style_id), width=width, height=height, scale=scale)
    furniture, surfaces, containers = _render_room_fixtures(draw, furniture=furniture, surfaces=surfaces, containers=containers, scale=scale)

    placements: List[IndoorObjectPlacement] = []
    rendered_objects = []
    for index, spec in sorted(enumerate(object_specs), key=lambda item: float(boxes[int(item[0])][3])):
        box = tuple(float(v) for v in boxes[int(index)])
        surface_id = surface_type = container_id = container_type = region_relation = region_furniture_id = None
        surface_contact_px = None
        surface_depth = None
        if spec.placement_kind == "surface":
            surface = surfaces_by_type[str(spec.target_type)]
            surface_id = str(surface.surface_id)
            surface_type = str(surface.surface_type)
            surface_contact_px = surface_contact_points.get(int(index))
            surface_depth = surface_depths.get(int(index))
        elif spec.placement_kind == "container":
            container = containers_by_type[str(spec.target_type)]
            container_id = str(container.container_id)
            container_type = str(container.container_type)
        elif spec.placement_kind == "region":
            target_type, relation = str(spec.target_type).split(":", 1)
            region_relation = str(relation)
            region_furniture_id = str(furniture_by_type[target_type].furniture_id)
        primary, accent = choose_object_colors(rng, str(spec.object_type))
        object_id = f"indoor_obj_{index:02d}"
        visual_attributes = {
            "primary_color_rgb": primary,
            "accent_color_rgb": accent,
            "style_id": str(style_id),
        }
        if surface_contact_px is not None and draw_style_shadow:
            _draw_surface_shadow(
                draw,
                box=box,
                contact_px=surface_contact_px,
                depth=float(surface_depth or 0.75),
                scale=scale,
            )
        rendered_object = render_illustration_object(
            IllustrationObjectSpec(
                object_id=object_id,
                object_type=str(spec.object_type),
                bbox_xyxy=box,
                visual_attributes=visual_attributes,
                role=str(spec.role),
                source_entity_type="illustration_object",
            ),
            RenderContext(
                renderer_style=RENDERER_STYLE_VECTOR,
                draw=draw,
                render_scale=scale,
            ),
        )
        actual_box = tuple(float(v) for v in rendered_object.bbox_xyxy)
        relations = {
            item.furniture_id: _relation_to_furniture(actual_box, item)
            for item in furniture
        }
        object_record = object_record_for_spec(
            IllustrationObjectSpec(
                object_id=object_id,
                object_type=str(spec.object_type),
                bbox_xyxy=actual_box,
                semantic_attributes={
                    "placement_kind": str(spec.placement_kind),
                    "surface_id": surface_id,
                    "surface_type": surface_type,
                    "container_id": container_id,
                    "container_type": container_type,
                    "region_relation": region_relation,
                    "region_furniture_id": region_furniture_id,
                    "relations": relations,
                },
                visual_attributes=visual_attributes,
                role=str(spec.role),
                source_entity_type="illustration_object",
            ),
            RenderContext(renderer_style=RENDERER_STYLE_VECTOR),
        )
        placements.append(
            IndoorObjectPlacement(
                object_id=object_id,
                object_type=str(spec.object_type),
                bbox_xyxy=actual_box,
                placement_kind=str(spec.placement_kind),
                surface_id=surface_id,
                surface_type=surface_type,
                surface_contact_px=surface_contact_px,
                surface_depth=surface_depth,
                container_id=container_id,
                container_type=container_type,
                region_relation=region_relation,
                region_furniture_id=region_furniture_id,
                primary_color_rgb=primary,
                accent_color_rgb=accent,
                style_id=str(style_id),
                relations=relations,
                role=str(spec.role),
                object_record=object_record,
            )
        )
        rendered_objects.append(rendered_object)

    highlight_type = str(highlight_container_type or "").strip()
    if highlight_type:
        for container in containers:
            if str(container.container_type) != highlight_type:
                continue
            x0, y0, x1, y1 = [float(value) for value in container.bbox_xyxy]
            pad = 8.0
            outline_box = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
            draw.rounded_rectangle(
                _scale_bbox(outline_box, scale),
                radius=max(1, 18 * scale),
                outline=(39, 116, 224),
                width=max(2, 6 * scale),
            )
            break

    if scale != 1:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    placement_by_id = {placement.object_id: placement for placement in placements}
    ordered_placements = tuple(placement_by_id[f"indoor_obj_{index:02d}"] for index in range(len(object_specs)))
    return RenderedIndoorRoomScene(
        image=image,
        theme_id=str(theme_id),
        surfaces=tuple(surfaces),
        containers=tuple(containers),
        furniture=tuple(furniture),
        objects=tuple(rendered_objects),
        placements=ordered_placements,
        canvas_width=width,
        canvas_height=height,
        render_scale=scale,
        style_id=str(style_id),
    )


def _indoor_theme_weights(theme_id: str) -> Dict[str, float]:
    return {theme: (1.0 if str(theme) == str(theme_id) else 0.0) for theme in INDOOR_THEME_IDS}


def _indoor_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    fallback: Mapping[str, Any],
    instance_seed: int | None = None,
    namespace: str = "indoor_room:canvas_profile",
) -> Dict[str, Any]:
    profile_params = resolve_profile_render_params(
        params,
        render_defaults,
        prefix="indoor",
        fallback_width=int(fallback["canvas_width"]),
        fallback_height=int(fallback["canvas_height"]),
        fallback_scale=int(fallback["render_scale"]),
        instance_seed=instance_seed,
        namespace=namespace,
    )
    return {
        "canvas_width": int(profile_params["canvas_width"]),
        "canvas_height": int(profile_params["canvas_height"]),
        "canvas_profile": str(profile_params["canvas_profile"]),
        "canvas_profile_size": list(profile_params["canvas_profile_size"]),
        "canvas_profile_probabilities": dict(profile_params["canvas_profile_probabilities"]),
        "object_size_min_px": int(
            params.get(
                "object_size_min_px",
                group_default(render_defaults, "indoor_object_size_min_px", int(fallback["object_size_min_px"])),
            )
        ),
        "object_size_max_px": int(
            params.get(
                "object_size_max_px",
                group_default(render_defaults, "indoor_object_size_max_px", int(fallback["object_size_max_px"])),
            )
        ),
        "render_scale": int(profile_params["render_scale"]),
    }


def render_indoor_scene_from_specs(
    *,
    render_namespace: str,
    instance_seed: int,
    attempt_index: int,
    specs: Sequence[IndoorObjectSpec],
    theme_id: str,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    fallback: Mapping[str, Any],
) -> RenderedIndoorRoomScene:
    render_params = _indoor_render_params(
        params,
        render_defaults,
        fallback=fallback,
        instance_seed=instance_seed,
        namespace=f"{render_namespace}:canvas_profile",
    )
    rng = spawn_rng(int(instance_seed), f"{render_namespace}:indoor-scene", int(attempt_index))
    return render_indoor_room_scene(
        rng=rng,
        object_specs=tuple(specs),
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        render_scale=int(render_params["render_scale"]),
        theme_weights=_indoor_theme_weights(str(theme_id)),
        style_weights=resolve_art_style_weights(params, render_defaults, style_ids=STYLE_IDS),
        object_size_min_px=int(render_params["object_size_min_px"]),
        object_size_max_px=int(render_params["object_size_max_px"]),
        highlight_container_type=str(params.get("highlight_container_type", "") or ""),
    )


def indoor_scene_entities(scene: RenderedIndoorRoomScene) -> List[Dict[str, Any]]:
    """Serialize room fixtures and placed objects into trace-safe entities."""

    def json_safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, float):
            return round(float(value), 3)
        return value

    entities: List[Dict[str, Any]] = []
    for furniture in scene.furniture:
        object_record = (
            dict(furniture.object_record)
            if furniture.object_record is not None
            else make_vector_scene_object_record(
                object_id=str(furniture.furniture_id),
                object_type="furniture",
                bbox_xyxy=furniture.bbox_xyxy,
                semantic_attributes={
                    "furniture_type": str(furniture.furniture_type),
                    "label": str(furniture.label),
                    **dict(furniture.attributes),
                },
                source_entity_type="indoor_furniture",
                render_scale=int(scene.render_scale),
                style_id=str(scene.style_id),
            )
        )
        entities.append(
            {
                "entity_id": str(furniture.furniture_id),
                "entity_type": "indoor_furniture",
                "furniture_type": str(furniture.furniture_type),
                "label": str(furniture.label),
                "bbox": [round(float(v), 3) for v in furniture.bbox_xyxy],
                "attributes": json_safe(furniture.attributes),
                "object_record": object_record,
            }
        )
    for surface in scene.surfaces:
        object_record = (
            dict(surface.object_record)
            if surface.object_record is not None
            else make_vector_scene_object_record(
                object_id=str(surface.surface_id),
                object_type="surface",
                bbox_xyxy=surface.bbox_xyxy,
                semantic_attributes={
                    "surface_type": str(surface.surface_type),
                    "label": str(surface.label),
                    "furniture_id": surface.furniture_id,
                    **dict(surface.attributes),
                },
                source_entity_type="indoor_surface",
                render_scale=int(scene.render_scale),
                style_id=str(scene.style_id),
            )
        )
        entities.append(
            {
                "entity_id": str(surface.surface_id),
                "entity_type": "indoor_surface",
                "surface_type": str(surface.surface_type),
                "label": str(surface.label),
                "bbox": [round(float(v), 3) for v in surface.bbox_xyxy],
                "support_bbox": [round(float(v), 3) for v in surface.support_bbox_xyxy],
                "furniture_id": surface.furniture_id,
                "attributes": json_safe(surface.attributes),
                "object_record": object_record,
            }
        )
    for container in scene.containers:
        object_record = (
            dict(container.object_record)
            if container.object_record is not None
            else make_vector_scene_object_record(
                object_id=str(container.container_id),
                object_type="container",
                bbox_xyxy=container.bbox_xyxy,
                semantic_attributes={
                    "container_type": str(container.container_type),
                    "label": str(container.label),
                    **dict(container.attributes),
                },
                source_entity_type="indoor_container",
                render_scale=int(scene.render_scale),
                style_id=str(scene.style_id),
            )
        )
        entities.append(
            {
                "entity_id": str(container.container_id),
                "entity_type": "indoor_container",
                "container_type": str(container.container_type),
                "label": str(container.label),
                "bbox": [round(float(v), 3) for v in container.bbox_xyxy],
                "interior_bbox": [round(float(v), 3) for v in container.interior_bbox_xyxy],
                "attributes": json_safe(container.attributes),
                "object_record": object_record,
            }
        )
    for placement in scene.placements:
        entities.append(
            {
                "entity_id": str(placement.object_id),
                "entity_type": "illustration_object",
                "object_type": str(placement.object_type),
                "object_name": display_name_for_object_type(str(placement.object_type)),
                "bbox": [round(float(v), 3) for v in placement.bbox_xyxy],
                "placement_kind": str(placement.placement_kind),
                "surface_id": placement.surface_id,
                "surface_type": placement.surface_type,
                "surface_contact_px": [round(float(v), 3) for v in placement.surface_contact_px] if placement.surface_contact_px else None,
                "surface_depth": round(float(placement.surface_depth), 4) if placement.surface_depth is not None else None,
                "container_id": placement.container_id,
                "container_type": placement.container_type,
                "region_relation": placement.region_relation,
                "region_furniture_id": placement.region_furniture_id,
                "relations": json_safe(placement.relations),
                "role": str(placement.role),
                "object_record": placement.object_record,
            }
        )
    return entities


__all__ = [
    "INDOOR_CONTAINER_TYPES",
    "INDOOR_FURNITURE_TYPES",
    "INDOOR_OBJECT_TYPES",
    "INDOOR_SURFACE_TYPES",
    "INDOOR_THEME_IDS",
    "IndoorObjectSpec",
    "RenderedIndoorRoomScene",
    "indoor_scene_entities",
    "render_indoor_scene_from_specs",
    "render_indoor_room_scene",
]
