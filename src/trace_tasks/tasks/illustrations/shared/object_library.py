"""Reusable synthetic object drawing primitives for illustration tasks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from .render_geometry import scale_bbox as _scale_bbox, scale_points as _scale_points
from .object_schema import ObjectRecord
from .object_variants import (
    RENDERER_STYLE_VECTOR,
    normalize_object_variant_id,
    variant_visual_metadata,
)
from .person_rendering import normalize_person_gender
from .style_registry import STYLE_IDS, style_object_colors, style_outline_params
from .vector_person_rendering import draw_vector_person


BBox = Tuple[float, float, float, float]
RGB = Tuple[int, int, int]

PART_PLURALS: Dict[str, str] = {
    "arm": "arms",
    "bag": "bags",
    "door": "doors",
    "eye": "eyes",
    "flower": "flowers",
    "fruit": "fruits",
    "handle": "handles",
    "hat": "hats",
    "head": "heads",
    "leaf": "leaves",
    "leg": "legs",
    "light": "lights",
    "tail": "tails",
    "wheel": "wheels",
    "window": "windows",
    "wing": "wings",
}


@dataclass(frozen=True)
class ObjectTemplate:
    """Static drawing metadata for one recognizable object type."""

    object_type: str
    family: str
    aspect_ratio: float
    part_counts: Mapping[str, int]


@dataclass(frozen=True)
class IllustrationPart:
    """One visible semantic part rendered for an object."""

    part_id: str
    part_kind: str
    bbox_xyxy: BBox
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class IllustrationObject:
    """One rendered object and its trace-ready semantic parts."""

    object_id: str
    object_type: str
    family: str
    bbox_xyxy: BBox
    primary_color_rgb: RGB
    accent_color_rgb: RGB
    style_id: str
    parts: Tuple[IllustrationPart, ...]
    attributes: Mapping[str, Any]


OBJECT_TEMPLATES: Dict[str, ObjectTemplate] = {
    "bird": ObjectTemplate("bird", "animal", 1.35, {"leg": 2, "wing": 1, "eye": 1, "tail": 1}),
    "duck": ObjectTemplate("duck", "animal", 1.45, {"leg": 2, "wing": 1, "eye": 1, "tail": 1}),
    "quadruped": ObjectTemplate("quadruped", "animal", 1.75, {"leg": 4, "eye": 1, "tail": 1}),
    "rabbit": ObjectTemplate("rabbit", "animal", 1.35, {"leg": 4, "eye": 1, "tail": 1}),
    "fish": ObjectTemplate("fish", "animal", 1.60, {"eye": 1, "tail": 1}),
    "butterfly": ObjectTemplate("butterfly", "animal", 1.25, {"wing": 4, "eye": 2}),
    "car": ObjectTemplate("car", "vehicle", 1.95, {"wheel": 2, "window": 2, "door": 1, "light": 2}),
    "bus": ObjectTemplate("bus", "vehicle", 2.20, {"wheel": 3, "window": 5, "door": 2, "light": 2}),
    "truck": ObjectTemplate("truck", "vehicle", 2.25, {"wheel": 3, "window": 2, "door": 1, "light": 2}),
    "taxi": ObjectTemplate("taxi", "vehicle", 1.95, {"wheel": 2, "window": 2, "door": 1, "light": 2}),
    "van": ObjectTemplate("van", "vehicle", 2.05, {"wheel": 2, "window": 3, "door": 2, "light": 2}),
    "bicycle": ObjectTemplate("bicycle", "vehicle", 1.80, {"wheel": 2, "handle": 1, "light": 1}),
    "scooter": ObjectTemplate("scooter", "vehicle", 1.45, {"wheel": 2, "handle": 1, "light": 1}),
    "train": ObjectTemplate("train", "vehicle", 1.75, {"wheel": 4, "window": 4, "light": 2}),
    "airplane": ObjectTemplate("airplane", "vehicle", 2.10, {"wing": 2, "window": 4}),
    "boat": ObjectTemplate("boat", "vehicle", 1.85, {"window": 2}),
    "sailboat": ObjectTemplate("sailboat", "vehicle", 1.35, {}),
    "buoy": ObjectTemplate("buoy", "object", 0.55, {}),
    "tree": ObjectTemplate("tree", "plant", 1.10, {"leaf": 6, "fruit": 4}),
    "flower": ObjectTemplate("flower", "plant", 0.80, {"flower": 1, "leaf": 3}),
    "cactus": ObjectTemplate("cactus", "plant", 0.80, {"flower": 1}),
    "mushroom": ObjectTemplate("mushroom", "plant", 0.90, {}),
    "potted_plant": ObjectTemplate("potted_plant", "plant", 0.90, {"leaf": 5, "flower": 2, "fruit": 3}),
    "lily_pad": ObjectTemplate("lily_pad", "plant", 1.25, {"leaf": 1, "flower": 1}),
    "person": ObjectTemplate("person", "person", 0.55, {"head": 1, "arm": 2, "leg": 2}),
    "pedestrian_with_bag": ObjectTemplate("pedestrian_with_bag", "person", 0.64, {"head": 1, "arm": 2, "leg": 2, "bag": 1}),
    "apple": ObjectTemplate("apple", "object", 0.85, {"fruit": 1}),
    "egg": ObjectTemplate("egg", "object", 1.15, {}),
    "soccer_ball": ObjectTemplate("soccer_ball", "object", 1.00, {}),
    "rugby_ball": ObjectTemplate("rugby_ball", "object", 1.75, {}),
    "spoon": ObjectTemplate("spoon", "object", 2.20, {}),
    "plate": ObjectTemplate("plate", "object", 1.20, {}),
    "book": ObjectTemplate("book", "object", 1.45, {}),
    "camera": ObjectTemplate("camera", "object", 1.25, {}),
    "remote": ObjectTemplate("remote", "object", 0.48, {}),
    "pencil": ObjectTemplate("pencil", "object", 2.70, {}),
    "ruler": ObjectTemplate("ruler", "object", 2.80, {}),
    "clock": ObjectTemplate("clock", "object", 1.00, {}),
    "gift": ObjectTemplate("gift", "object", 1.00, {}),
    "vase": ObjectTemplate("vase", "object", 0.65, {}),
    "bowl": ObjectTemplate("bowl", "object", 1.40, {}),
    "candle": ObjectTemplate("candle", "object", 0.55, {}),
    "lightbulb": ObjectTemplate("lightbulb", "object", 0.75, {"light": 1}),
    "mug": ObjectTemplate("mug", "object", 0.90, {"handle": 1}),
    "teapot": ObjectTemplate("teapot", "object", 1.25, {"handle": 1}),
    "chair": ObjectTemplate("chair", "object", 0.85, {"leg": 4}),
    "table": ObjectTemplate("table", "object", 1.35, {"leg": 4}),
    "lamp": ObjectTemplate("lamp", "object", 0.80, {"light": 1}),
    "streetlamp": ObjectTemplate("streetlamp", "object", 0.45, {"light": 1}),
    "traffic_light": ObjectTemplate("traffic_light", "object", 0.48, {"light": 3}),
    "road_sign": ObjectTemplate("road_sign", "object", 0.80, {}),
    "mailbox": ObjectTemplate("mailbox", "object", 1.15, {}),
    "bench": ObjectTemplate("bench", "object", 1.75, {"leg": 4}),
    "trash_bin": ObjectTemplate("trash_bin", "object", 0.70, {}),
    "backpack": ObjectTemplate("backpack", "object", 0.78, {"handle": 1}),
    "umbrella": ObjectTemplate("umbrella", "object", 1.10, {"handle": 1}),
    "kite": ObjectTemplate("kite", "object", 0.78, {}),
    "bucket": ObjectTemplate("bucket", "object", 0.85, {"handle": 1}),
    "shovel": ObjectTemplate("shovel", "object", 1.50, {"handle": 1}),
    "guitar": ObjectTemplate("guitar", "object", 1.65, {}),
    "scissors": ObjectTemplate("scissors", "object", 1.20, {"handle": 2}),
    "key": ObjectTemplate("key", "object", 1.60, {"handle": 1}),
    "bottle": ObjectTemplate("bottle", "object", 0.55, {}),
    "cloud": ObjectTemplate("cloud", "sky", 1.90, {}),
    "sun": ObjectTemplate("sun", "sky", 1.00, {}),
}

OBJECT_DISPLAY_NAMES: Dict[str, str] = {
    "pedestrian_with_bag": "pedestrian with bag",
    "potted_plant": "potted plant",
    "lily_pad": "lily pad",
    "traffic_light": "traffic light",
    "road_sign": "road sign",
    "trash_bin": "trash bin",
    "soccer_ball": "soccer ball",
    "rugby_ball": "rugby ball",
    "quadruped": "animal",
}


PALETTES: Dict[str, Tuple[RGB, ...]] = {
    "animal": ((93, 129, 176), (126, 102, 157), (176, 112, 74), (92, 147, 116), (206, 145, 83)),
    "vehicle": ((63, 114, 176), (204, 90, 82), (229, 171, 72), (80, 145, 145), (120, 129, 152)),
    "plant": ((78, 143, 96), (114, 161, 82), (55, 132, 121), (164, 142, 78), (92, 152, 122)),
    "person": ((72, 115, 166), (184, 93, 91), (219, 168, 76), (91, 146, 132), (122, 104, 153)),
    "object": ((198, 105, 83), (86, 126, 177), (212, 169, 78), (116, 111, 152), (87, 148, 139)),
    "sky": ((248, 248, 245), (247, 214, 101), (211, 227, 239), (238, 241, 244)),
}
ACCENT_PALETTE: Tuple[RGB, ...] = (
    (245, 190, 88),
    (238, 129, 86),
    (236, 103, 132),
    (92, 167, 208),
    (248, 235, 164),
    (245, 245, 245),
)
OUTLINE_RGB: RGB = (35, 39, 48)
SOFT_OUTLINE_RGB: RGB = (79, 84, 94)


def object_types() -> Tuple[str, ...]:
    """Return all supported object type ids."""

    return tuple(sorted(OBJECT_TEMPLATES))


def display_name_for_object_type(object_type: str) -> str:
    """Return a prompt-facing name for one object type."""

    key = str(object_type)
    if key in OBJECT_DISPLAY_NAMES:
        return str(OBJECT_DISPLAY_NAMES[key])
    return key.replace("_", " ")


def supported_part_kinds() -> Tuple[str, ...]:
    """Return all part kinds that can be queried by the v0 counting task."""

    kinds = sorted({kind for template in OBJECT_TEMPLATES.values() for kind, count in template.part_counts.items() if int(count) > 0})
    return tuple(str(kind) for kind in kinds)


def object_types_with_part(part_kind: str) -> Tuple[str, ...]:
    """Return object types with at least one visible part of the requested kind."""

    return tuple(
        object_type
        for object_type, template in sorted(OBJECT_TEMPLATES.items())
        if int(template.part_counts.get(str(part_kind), 0)) > 0
    )


def object_types_without_part(part_kind: str) -> Tuple[str, ...]:
    """Return object types with no visible part of the requested kind."""

    return tuple(
        object_type
        for object_type, template in sorted(OBJECT_TEMPLATES.items())
        if int(template.part_counts.get(str(part_kind), 0)) == 0
    )


def part_count_for_object(object_type: str, part_kind: str) -> int:
    """Return the static count for one part kind on one object type."""

    return int(OBJECT_TEMPLATES[str(object_type)].part_counts.get(str(part_kind), 0))


def family_for_object(object_type: str) -> str:
    """Return the family id for one object type."""

    return str(OBJECT_TEMPLATES[str(object_type)].family)


def aspect_ratio_for_object(object_type: str) -> float:
    """Return the preferred footprint aspect ratio for one object type."""

    return float(OBJECT_TEMPLATES[str(object_type)].aspect_ratio)


def choose_object_colors(rng, object_type: str) -> Tuple[RGB, RGB]:
    """Choose deterministic primary/accent colors for one object."""

    family = family_for_object(str(object_type))
    palette = PALETTES.get(str(family), PALETTES["object"])
    primary = tuple(int(v) for v in rng.choice(tuple(palette)))
    accent = tuple(int(v) for v in rng.choice(ACCENT_PALETTE))
    return primary, accent




def _expand(bbox: BBox, pad: float) -> BBox:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    p = max(0.0, float(pad))
    return (x0 - p, y0 - p, x1 + p, y1 + p)


def _union(bboxes: Iterable[BBox]) -> BBox:
    boxes = [tuple(float(v) for v in box) for box in bboxes]
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _blend_rgb(a: RGB, b: RGB, t: float) -> RGB:
    mix = max(0.0, min(1.0, float(t)))
    return tuple(
        int(round((1.0 - mix) * int(left) + mix * int(right)))
        for left, right in zip(a, b)
    )  # type: ignore[return-value]


def _shade_rgb(rgb: RGB, amount: int) -> RGB:
    return tuple(max(0, min(255, int(value) + int(amount))) for value in rgb)  # type: ignore[return-value]


def _rel_bbox(box: BBox, rx0: float, ry0: float, rx1: float, ry1: float) -> BBox:
    x0, y0, x1, y1 = box
    w = float(x1) - float(x0)
    h = float(y1) - float(y0)
    return (x0 + rx0 * w, y0 + ry0 * h, x0 + rx1 * w, y0 + ry1 * h)


def _ellipse(draw: ImageDraw.ImageDraw, bbox: BBox, *, fill: RGB, outline: RGB | None, width: int, scale: int) -> None:
    draw.ellipse(_scale_bbox(bbox, scale), fill=tuple(fill), outline=tuple(outline) if outline else None, width=max(1, int(width) * int(scale)))


def _rect(draw: ImageDraw.ImageDraw, bbox: BBox, *, fill: RGB, outline: RGB | None, width: int, scale: int, radius: float = 0.0) -> None:
    sb = _scale_bbox(bbox, scale)
    if radius > 0:
        draw.rounded_rectangle(sb, radius=int(round(float(radius) * int(scale))), fill=tuple(fill), outline=tuple(outline) if outline else None, width=max(1, int(width) * int(scale)))
    else:
        draw.rectangle(sb, fill=tuple(fill), outline=tuple(outline) if outline else None, width=max(1, int(width) * int(scale)))


def _poly(draw: ImageDraw.ImageDraw, points: Sequence[Tuple[float, float]], *, fill: RGB, outline: RGB | None, width: int, scale: int) -> None:
    draw.polygon(_scale_points(points, scale), fill=tuple(fill))
    if outline:
        draw.line(_scale_points([*points, points[0]], scale), fill=tuple(outline), width=max(1, int(width) * int(scale)), joint="curve")


def _line(draw: ImageDraw.ImageDraw, points: Sequence[Tuple[float, float]], *, fill: RGB, width: int, scale: int) -> None:
    draw.line(_scale_points(points, scale), fill=tuple(fill), width=max(1, int(width) * int(scale)), joint="curve")


def _arc(draw: ImageDraw.ImageDraw, bbox: BBox, *, start: float, end: float, fill: RGB, width: int, scale: int) -> None:
    draw.arc(_scale_bbox(bbox, scale), start=float(start), end=float(end), fill=tuple(fill), width=max(1, int(width) * int(scale)))


def _regular_polygon_points(cx: float, cy: float, radius: float, sides: int, *, start_degrees: float = -90.0) -> List[Tuple[float, float]]:
    count = max(3, int(sides))
    return [
        (
            float(cx) + float(radius) * math.cos(math.radians(float(start_degrees) + 360.0 * float(index) / float(count))),
            float(cy) + float(radius) * math.sin(math.radians(float(start_degrees) + 360.0 * float(index) / float(count))),
        )
        for index in range(count)
    ]


def _style_params(style_id: str) -> Tuple[RGB | None, int, bool]:
    return style_outline_params(str(style_id))


def _draw_shadow(draw: ImageDraw.ImageDraw, bbox: BBox, *, scale: int) -> None:
    x0, y0, x1, y1 = bbox
    shadow = (x0 + 0.08 * (x1 - x0), y1 - 0.10 * (y1 - y0), x1 - 0.04 * (x1 - x0), y1 + 0.02 * (y1 - y0))
    _ellipse(draw, shadow, fill=(205, 210, 216), outline=None, width=1, scale=scale)


class _PartBuilder:
    def __init__(self, object_id: str) -> None:
        self.object_id = str(object_id)
        self._counts: Dict[str, int] = {}
        self.parts: List[IllustrationPart] = []

    def add(self, kind: str, bbox: BBox, **attrs: Any) -> None:
        index = int(self._counts.get(str(kind), 0))
        self._counts[str(kind)] = index + 1
        self.parts.append(
            IllustrationPart(
                part_id=f"{self.object_id}_{kind}_{index}",
                part_kind=str(kind),
                bbox_xyxy=tuple(float(v) for v in bbox),
                attributes=dict(attrs),
            )
        )


def _draw_bird_like(draw: ImageDraw.ImageDraw, *, box: BBox, object_type: str, primary: RGB, accent: RGB, outline: RGB | None, width: int, scale: int, parts: _PartBuilder) -> List[BBox]:
    body = _rel_bbox(box, 0.18, 0.34, 0.76, 0.72)
    head = _rel_bbox(box, 0.67, 0.22, 0.88, 0.43)
    wing = _rel_bbox(box, 0.36, 0.42, 0.62, 0.64)
    beak = ((box[0] + 0.86 * (box[2] - box[0]), box[1] + 0.31 * (box[3] - box[1])), (box[0] + 0.98 * (box[2] - box[0]), box[1] + 0.35 * (box[3] - box[1])), (box[0] + 0.86 * (box[2] - box[0]), box[1] + 0.39 * (box[3] - box[1])))
    tail = ((box[0] + 0.18 * (box[2] - box[0]), box[1] + 0.45 * (box[3] - box[1])), (box[0] + 0.02 * (box[2] - box[0]), box[1] + 0.35 * (box[3] - box[1])), (box[0] + 0.07 * (box[2] - box[0]), box[1] + 0.59 * (box[3] - box[1])))
    _poly(draw, tail, fill=primary, outline=outline, width=width, scale=scale)
    _ellipse(draw, body, fill=primary, outline=outline, width=width, scale=scale)
    _ellipse(draw, wing, fill=accent, outline=outline, width=max(1, width - 1), scale=scale)
    _ellipse(draw, head, fill=primary, outline=outline, width=width, scale=scale)
    _poly(draw, beak, fill=(235, 161, 65), outline=outline, width=max(1, width - 1), scale=scale)
    eye = _rel_bbox(box, 0.78, 0.30, 0.82, 0.34)
    _ellipse(draw, eye, fill=(20, 24, 30), outline=None, width=1, scale=scale)
    parts.add("wing", wing)
    parts.add("eye", _expand(eye, 2.0))
    parts.add("tail", _rel_bbox(box, 0.02, 0.35, 0.20, 0.60))
    leg_xs = (0.43, 0.57) if object_type == "duck" else (0.46, 0.56)
    for lx in leg_xs:
        leg = _rel_bbox(box, lx, 0.68, lx + 0.035, 0.94)
        _line(draw, [(leg[0], leg[1]), (leg[0] + 0.02 * (box[2] - box[0]), leg[3])], fill=(93, 72, 48), width=max(2, width), scale=scale)
        parts.add("leg", _expand(leg, 3.0))
    return [body, head, wing, _rel_bbox(box, 0.02, 0.35, 0.98, 0.94)]


def _draw_quadruped_like(draw: ImageDraw.ImageDraw, *, box: BBox, object_type: str, primary: RGB, accent: RGB, outline: RGB | None, width: int, scale: int, parts: _PartBuilder) -> List[BBox]:
    body = _rel_bbox(box, 0.18, 0.35, 0.78, 0.68)
    head = _rel_bbox(box, 0.70, 0.23, 0.93, 0.48)
    _ellipse(draw, body, fill=primary, outline=outline, width=width, scale=scale)
    _ellipse(draw, head, fill=primary, outline=outline, width=width, scale=scale)
    if object_type == "rabbit":
        for ear in (_rel_bbox(box, 0.73, 0.02, 0.80, 0.28), _rel_bbox(box, 0.83, 0.03, 0.90, 0.29)):
            _ellipse(draw, ear, fill=primary, outline=outline, width=max(1, width - 1), scale=scale)
    else:
        ear = _rel_bbox(box, 0.77, 0.17, 0.86, 0.28)
        _poly(draw, [(ear[0], ear[3]), (0.5 * (ear[0] + ear[2]), ear[1]), (ear[2], ear[3])], fill=accent, outline=outline, width=max(1, width - 1), scale=scale)
    eye = _rel_bbox(box, 0.82, 0.32, 0.86, 0.36)
    _ellipse(draw, eye, fill=(20, 24, 30), outline=None, width=1, scale=scale)
    tail = _rel_bbox(box, 0.07, 0.39, 0.21, 0.53)
    _ellipse(draw, tail, fill=accent, outline=outline, width=max(1, width - 1), scale=scale)
    parts.add("eye", _expand(eye, 2.0))
    parts.add("tail", tail)
    for lx in (0.28, 0.42, 0.58, 0.72):
        leg = _rel_bbox(box, lx, 0.62, lx + 0.055, 0.93)
        _rect(draw, leg, fill=primary, outline=outline, width=max(1, width - 1), scale=scale, radius=3)
        parts.add("leg", leg)
    return [body, head, tail, _rel_bbox(box, 0.07, 0.02, 0.93, 0.93)]


def _draw_fish(draw: ImageDraw.ImageDraw, *, box: BBox, primary: RGB, accent: RGB, outline: RGB | None, width: int, scale: int, parts: _PartBuilder) -> List[BBox]:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    fin_rgb = _shade_rgb(primary, -16)
    tail = [
        (x0 + 0.18 * w, y0 + 0.50 * h),
        (x0 + 0.02 * w, y0 + 0.28 * h),
        (x0 + 0.05 * w, y0 + 0.50 * h),
        (x0 + 0.02 * w, y0 + 0.72 * h),
    ]
    body = _rel_bbox(box, 0.14, 0.27, 0.93, 0.74)
    dorsal = [
        (x0 + 0.46 * w, y0 + 0.31 * h),
        (x0 + 0.56 * w, y0 + 0.17 * h),
        (x0 + 0.66 * w, y0 + 0.32 * h),
    ]
    belly_fin = [
        (x0 + 0.48 * w, y0 + 0.66 * h),
        (x0 + 0.56 * w, y0 + 0.82 * h),
        (x0 + 0.63 * w, y0 + 0.65 * h),
    ]
    side_fin = [
        (x0 + 0.56 * w, y0 + 0.50 * h),
        (x0 + 0.43 * w, y0 + 0.60 * h),
        (x0 + 0.57 * w, y0 + 0.65 * h),
    ]
    _poly(draw, tail, fill=fin_rgb, outline=outline, width=width, scale=scale)
    _poly(draw, dorsal, fill=fin_rgb, outline=outline, width=max(1, width - 1), scale=scale)
    _poly(draw, belly_fin, fill=fin_rgb, outline=outline, width=max(1, width - 1), scale=scale)
    _ellipse(draw, body, fill=primary, outline=outline, width=width, scale=scale)
    _poly(draw, side_fin, fill=fin_rgb, outline=outline, width=max(1, width - 1), scale=scale)
    gill_x = x0 + 0.72 * w
    _line(draw, [(gill_x, y0 + 0.37 * h), (gill_x - 0.035 * w, y0 + 0.50 * h), (gill_x, y0 + 0.63 * h)], fill=_shade_rgb(primary, -45), width=max(1, width - 1), scale=scale)
    mouth_rgb = _shade_rgb(primary, -58)
    _line(
        draw,
        [(x0 + 0.885 * w, y0 + 0.505 * h), (x0 + 0.93 * w, y0 + 0.49 * h)],
        fill=mouth_rgb,
        width=max(1, width - 1),
        scale=scale,
    )
    _line(
        draw,
        [(x0 + 0.885 * w, y0 + 0.535 * h), (x0 + 0.925 * w, y0 + 0.545 * h)],
        fill=mouth_rgb,
        width=max(1, width - 1),
        scale=scale,
    )
    eye = _rel_bbox(box, 0.78, 0.38, 0.84, 0.45)
    _ellipse(draw, eye, fill=(20, 24, 30), outline=None, width=1, scale=scale)
    highlight = _rel_bbox(box, 0.80, 0.39, 0.815, 0.405)
    _ellipse(draw, highlight, fill=(245, 248, 252), outline=None, width=1, scale=scale)
    parts.add("eye", _expand(eye, 2.0))
    parts.add("tail", _rel_bbox(box, 0.02, 0.28, 0.19, 0.72))
    return [body, _rel_bbox(box, 0.02, 0.17, 0.98, 0.82)]


def _draw_butterfly(draw: ImageDraw.ImageDraw, *, box: BBox, primary: RGB, accent: RGB, outline: RGB | None, width: int, scale: int, parts: _PartBuilder) -> List[BBox]:
    wings = (
        _rel_bbox(box, 0.08, 0.08, 0.45, 0.46),
        _rel_bbox(box, 0.55, 0.08, 0.92, 0.46),
        _rel_bbox(box, 0.16, 0.48, 0.46, 0.86),
        _rel_bbox(box, 0.54, 0.48, 0.84, 0.86),
    )
    for wing in wings:
        _ellipse(draw, wing, fill=accent, outline=outline, width=width, scale=scale)
        parts.add("wing", wing)
    body = _rel_bbox(box, 0.45, 0.18, 0.55, 0.84)
    _ellipse(draw, body, fill=primary, outline=outline, width=width, scale=scale)
    for eye in (_rel_bbox(box, 0.45, 0.13, 0.49, 0.18), _rel_bbox(box, 0.51, 0.13, 0.55, 0.18)):
        _ellipse(draw, eye, fill=(20, 24, 30), outline=None, width=1, scale=scale)
        parts.add("eye", _expand(eye, 2.0))
    return [*wings, body]


def _draw_vehicle(draw: ImageDraw.ImageDraw, *, box: BBox, object_type: str, primary: RGB, accent: RGB, outline: RGB | None, width: int, scale: int, parts: _PartBuilder) -> List[BBox]:
    body = _rel_bbox(box, 0.08, 0.42, 0.92, 0.74)
    cab = _rel_bbox(box, 0.22, 0.24, 0.70, 0.48)
    if object_type == "truck":
        cab = _rel_bbox(box, 0.62, 0.25, 0.90, 0.48)
        trailer = _rel_bbox(box, 0.08, 0.34, 0.64, 0.72)
        _rect(draw, trailer, fill=primary, outline=outline, width=width, scale=scale, radius=6)
    elif object_type == "van":
        cab = _rel_bbox(box, 0.18, 0.25, 0.84, 0.50)
    _rect(draw, body, fill=primary, outline=outline, width=width, scale=scale, radius=8)
    _rect(draw, cab, fill=primary, outline=outline, width=width, scale=scale, radius=8)
    if object_type in {"car", "taxi", "van"}:
        wheel_xs = (0.25, 0.75)
    else:
        wheel_xs = (0.20, 0.50, 0.80)
    for wx in wheel_xs:
        wheel = _rel_bbox(box, wx - 0.06, 0.66, wx + 0.06, 0.88)
        _ellipse(draw, wheel, fill=(37, 42, 50), outline=outline, width=max(1, width - 1), scale=scale)
        parts.add("wheel", wheel)
    window_count = {"car": 2, "taxi": 2, "bus": 5, "truck": 2, "van": 3}.get(object_type, 2)
    for i in range(window_count):
        t0 = 0.24 + i * (0.44 / max(1, window_count))
        win = _rel_bbox(box, t0, 0.30, t0 + 0.32 / max(1, window_count), 0.44)
        _rect(draw, win, fill=(213, 235, 244), outline=outline, width=1, scale=scale, radius=3)
        parts.add("window", win)
    door_count = 2 if object_type in {"bus", "van"} else 1
    for i in range(door_count):
        dx = 0.62 + i * 0.14 if object_type == "bus" else 0.55
        door = _rel_bbox(box, dx, 0.45, dx + 0.09, 0.72)
        _rect(draw, door, fill=accent, outline=outline, width=1, scale=scale, radius=2)
        parts.add("door", door)
    for light in (_rel_bbox(box, 0.08, 0.54, 0.14, 0.61), _rel_bbox(box, 0.86, 0.54, 0.92, 0.61)):
        _ellipse(draw, light, fill=(248, 224, 109), outline=outline, width=1, scale=scale)
        parts.add("light", light)
    if object_type == "taxi":
        sign = _rel_bbox(box, 0.43, 0.18, 0.58, 0.28)
        _rect(draw, sign, fill=(248, 224, 109), outline=outline, width=1, scale=scale, radius=2)
    return [body, cab, _rel_bbox(box, 0.08, 0.24, 0.92, 0.88)]


def _draw_train(draw: ImageDraw.ImageDraw, *, box: BBox, primary: RGB, accent: RGB, outline: RGB | None, width: int, scale: int, parts: _PartBuilder) -> List[BBox]:
    body = _rel_bbox(box, 0.10, 0.26, 0.88, 0.70)
    nose = [(body[2], body[1] + 0.08 * (body[3] - body[1])), (box[2] - 0.02 * (box[2] - box[0]), (body[1] + body[3]) * 0.5), (body[2], body[3] - 0.08 * (body[3] - body[1]))]
    roof = _rel_bbox(box, 0.22, 0.16, 0.62, 0.30)
    rail = _rel_bbox(box, 0.06, 0.82, 0.94, 0.88)
    _rect(draw, body, fill=primary, outline=outline, width=width, scale=scale, radius=8)
    _poly(draw, nose, fill=primary, outline=outline, width=width, scale=scale)
    _rect(draw, roof, fill=accent, outline=outline, width=max(1, width - 1), scale=scale, radius=5)
    for index in range(4):
        x0 = 0.20 + 0.14 * index
        win = _rel_bbox(box, x0, 0.36, x0 + 0.09, 0.49)
        _rect(draw, win, fill=(213, 235, 244), outline=outline, width=1, scale=scale, radius=2)
        parts.add("window", win)
    for wx in (0.18, 0.38, 0.62, 0.82):
        wheel = _rel_bbox(box, wx - 0.045, 0.64, wx + 0.045, 0.80)
        _ellipse(draw, wheel, fill=(37, 42, 50), outline=outline, width=max(1, width - 1), scale=scale)
        hub = _rel_bbox(box, wx - 0.018, 0.69, wx + 0.018, 0.75)
        _ellipse(draw, hub, fill=(216, 222, 225), outline=None, width=1, scale=scale)
        parts.add("wheel", wheel)
    for light in (_rel_bbox(box, 0.11, 0.50, 0.16, 0.57), _rel_bbox(box, 0.88, 0.48, 0.94, 0.56)):
        _ellipse(draw, light, fill=(248, 224, 109), outline=outline, width=1, scale=scale)
        parts.add("light", light)
    _line(draw, [(rail[0], rail[1]), (rail[2], rail[1])], fill=SOFT_OUTLINE_RGB, width=max(2, width), scale=scale)
    _line(draw, [(rail[0], rail[3]), (rail[2], rail[3])], fill=SOFT_OUTLINE_RGB, width=max(2, width), scale=scale)
    return [body, roof, rail, _rel_bbox(box, 0.06, 0.16, 0.98, 0.88)]


def _draw_bicycle(draw: ImageDraw.ImageDraw, *, box: BBox, primary: RGB, accent: RGB, outline: RGB | None, width: int, scale: int, parts: _PartBuilder) -> List[BBox]:
    wheels = (_rel_bbox(box, 0.10, 0.50, 0.35, 0.82), _rel_bbox(box, 0.66, 0.50, 0.91, 0.82))
    for wheel in wheels:
        _ellipse(draw, wheel, fill=(245, 247, 250), outline=OUTLINE_RGB, width=max(2, width), scale=scale)
        hub = _rel_bbox(wheel, 0.40, 0.40, 0.60, 0.60)
        _ellipse(draw, hub, fill=SOFT_OUTLINE_RGB, outline=None, width=1, scale=scale)
        center = ((hub[0] + hub[2]) * 0.5, (hub[1] + hub[3]) * 0.5)
        for angle in (0, 60, 120):
            rx = 0.5 * (wheel[2] - wheel[0])
            ry = 0.5 * (wheel[3] - wheel[1])
            dx = rx * math.cos(math.radians(angle))
            dy = ry * math.sin(math.radians(angle))
            _line(
                draw,
                [(center[0] - dx, center[1] - dy), (center[0] + dx, center[1] + dy)],
                fill=(160, 168, 176),
                width=max(1, width - 1),
                scale=scale,
            )
        parts.add("wheel", wheel)
    x0, y0, x1, y1 = box
    bw = x1 - x0
    bh = y1 - y0
    crank = (x0 + 0.50 * bw, y0 + 0.64 * bh)
    seat = _rel_bbox(box, 0.40, 0.29, 0.58, 0.36)
    handle = _rel_bbox(box, 0.74, 0.27, 0.94, 0.45)
    frame_points = (
        (x0 + 0.225 * bw, y0 + 0.66 * bh),
        (x0 + 0.48 * bw, y0 + 0.40 * bh),
        (x0 + 0.785 * bw, y0 + 0.66 * bh),
        (x0 + 0.50 * bw, y0 + 0.64 * bh),
        (x0 + 0.225 * bw, y0 + 0.66 * bh),
        (x0 + 0.62 * bw, y0 + 0.66 * bh),
        (x0 + 0.48 * bw, y0 + 0.40 * bh),
    )
    _line(draw, frame_points, fill=primary, width=max(3, width + 1), scale=scale)
    _line(draw, [(x0 + 0.48 * bw, y0 + 0.40 * bh), (seat[0] + 0.5 * (seat[2] - seat[0]), seat[1])], fill=primary, width=max(2, width), scale=scale)
    _rect(draw, seat, fill=SOFT_OUTLINE_RGB, outline=outline, width=1, scale=scale, radius=4)
    _ellipse(draw, (crank[0] - 3.5, crank[1] - 3.5, crank[0] + 3.5, crank[1] + 3.5), fill=accent, outline=outline, width=1, scale=scale)
    _line(draw, [(handle[0], handle[3]), (x0 + 0.80 * bw, y0 + 0.35 * bh), (handle[2], handle[1])], fill=primary, width=max(3, width + 1), scale=scale)
    _line(draw, [(handle[2] - 0.04 * bw, handle[1]), (handle[2] + 0.04 * bw, handle[1])], fill=SOFT_OUTLINE_RGB, width=max(2, width), scale=scale)
    parts.add("handle", _expand(handle, 4.0))
    light = _rel_bbox(box, 0.88, 0.38, 0.94, 0.45)
    _ellipse(draw, light, fill=(248, 224, 109), outline=outline, width=1, scale=scale)
    parts.add("light", light)
    return [*wheels, seat, _rel_bbox(box, 0.10, 0.27, 0.96, 0.82)]


def _draw_scooter(draw: ImageDraw.ImageDraw, *, box: BBox, primary: RGB, accent: RGB, outline: RGB | None, width: int, scale: int, parts: _PartBuilder) -> List[BBox]:
    wheels = (_rel_bbox(box, 0.16, 0.64, 0.34, 0.84), _rel_bbox(box, 0.66, 0.64, 0.84, 0.84))
    deck = _rel_bbox(box, 0.24, 0.56, 0.76, 0.66)
    stem = _rel_bbox(box, 0.66, 0.22, 0.72, 0.62)
    handle = _rel_bbox(box, 0.56, 0.18, 0.86, 0.28)
    for wheel in wheels:
        _ellipse(draw, wheel, fill=(37, 42, 50), outline=outline, width=max(1, width - 1), scale=scale)
        parts.add("wheel", wheel)
    _rect(draw, deck, fill=primary, outline=outline, width=width, scale=scale, radius=4)
    _line(draw, [(stem[0], stem[3]), (stem[2], stem[1])], fill=primary, width=max(3, width + 1), scale=scale)
    _line(draw, [(handle[0], handle[1]), (handle[2], handle[1])], fill=primary, width=max(3, width + 1), scale=scale)
    light = _rel_bbox(box, 0.78, 0.22, 0.86, 0.30)
    _ellipse(draw, light, fill=(248, 224, 109), outline=outline, width=1, scale=scale)
    parts.add("handle", _expand(handle, 3.0))
    parts.add("light", light)
    return [*wheels, deck, stem, handle]


def _draw_airplane(draw: ImageDraw.ImageDraw, *, box: BBox, primary: RGB, accent: RGB, outline: RGB | None, width: int, scale: int, parts: _PartBuilder) -> List[BBox]:
    body = _rel_bbox(box, 0.10, 0.38, 0.90, 0.58)
    wing_top = [(box[0] + 0.46 * (box[2] - box[0]), box[1] + 0.42 * (box[3] - box[1])), (box[0] + 0.27 * (box[2] - box[0]), box[1] + 0.12 * (box[3] - box[1])), (box[0] + 0.58 * (box[2] - box[0]), box[1] + 0.37 * (box[3] - box[1]))]
    wing_bottom = [(box[0] + 0.52 * (box[2] - box[0]), box[1] + 0.55 * (box[3] - box[1])), (box[0] + 0.32 * (box[2] - box[0]), box[1] + 0.84 * (box[3] - box[1])), (box[0] + 0.64 * (box[2] - box[0]), box[1] + 0.59 * (box[3] - box[1]))]
    _poly(draw, wing_top, fill=accent, outline=outline, width=width, scale=scale)
    _poly(draw, wing_bottom, fill=accent, outline=outline, width=width, scale=scale)
    _ellipse(draw, body, fill=primary, outline=outline, width=width, scale=scale)
    parts.add("wing", _rel_bbox(box, 0.27, 0.12, 0.58, 0.42))
    parts.add("wing", _rel_bbox(box, 0.32, 0.55, 0.64, 0.84))
    for i in range(4):
        win = _rel_bbox(box, 0.46 + i * 0.08, 0.43, 0.50 + i * 0.08, 0.49)
        _ellipse(draw, win, fill=(213, 235, 244), outline=outline, width=1, scale=scale)
        parts.add("window", win)
    return [body, _rel_bbox(box, 0.10, 0.12, 0.90, 0.84)]


def _draw_boat(draw: ImageDraw.ImageDraw, *, box: BBox, primary: RGB, accent: RGB, outline: RGB | None, width: int, scale: int, parts: _PartBuilder) -> List[BBox]:
    hull = [(box[0] + 0.12 * (box[2] - box[0]), box[1] + 0.60 * (box[3] - box[1])), (box[0] + 0.88 * (box[2] - box[0]), box[1] + 0.60 * (box[3] - box[1])), (box[0] + 0.74 * (box[2] - box[0]), box[1] + 0.82 * (box[3] - box[1])), (box[0] + 0.24 * (box[2] - box[0]), box[1] + 0.82 * (box[3] - box[1]))]
    cabin = _rel_bbox(box, 0.34, 0.34, 0.66, 0.60)
    _poly(draw, hull, fill=primary, outline=outline, width=width, scale=scale)
    _rect(draw, cabin, fill=accent, outline=outline, width=width, scale=scale, radius=4)
    for win in (_rel_bbox(box, 0.40, 0.42, 0.48, 0.50), _rel_bbox(box, 0.53, 0.42, 0.61, 0.50)):
        _rect(draw, win, fill=(213, 235, 244), outline=outline, width=1, scale=scale, radius=2)
        parts.add("window", win)
    return [_rel_bbox(box, 0.12, 0.34, 0.88, 0.82)]


def _draw_sailboat(draw: ImageDraw.ImageDraw, *, box: BBox, primary: RGB, accent: RGB, outline: RGB | None, width: int, scale: int, parts: _PartBuilder) -> List[BBox]:
    mast = _rel_bbox(box, 0.47, 0.18, 0.53, 0.66)
    sail = [(mast[0], mast[1]), (mast[0], mast[3]), (box[0] + 0.18 * (box[2] - box[0]), mast[3])]
    hull = [(box[0] + 0.10 * (box[2] - box[0]), box[1] + 0.66 * (box[3] - box[1])), (box[0] + 0.90 * (box[2] - box[0]), box[1] + 0.66 * (box[3] - box[1])), (box[0] + 0.74 * (box[2] - box[0]), box[1] + 0.84 * (box[3] - box[1])), (box[0] + 0.28 * (box[2] - box[0]), box[1] + 0.84 * (box[3] - box[1]))]
    _rect(draw, mast, fill=(95, 87, 75), outline=None, width=1, scale=scale)
    _poly(draw, sail, fill=(248, 246, 231), outline=outline, width=width, scale=scale)
    _poly(draw, hull, fill=primary, outline=outline, width=width, scale=scale)
    return [_rel_bbox(box, 0.10, 0.18, 0.90, 0.84)]


def _draw_person_like(
    draw: ImageDraw.ImageDraw,
    *,
    object_id: str,
    box: BBox,
    object_type: str,
    primary: RGB,
    accent: RGB,
    outline: RGB | None,
    width: int,
    scale: int,
    style_id: str,
    parts: _PartBuilder,
    gender_id: str,
    person_variant_id: str = "adult",
) -> List[BBox]:
    _ = outline, width
    rendered = draw_vector_person(
        draw,
        object_id=str(object_id),
        object_type=str(object_type),
        bbox_xyxy=box,
        renderer_id="object_library_person",
        style_id=str(style_id),
        render_scale=int(scale),
        primary_color_rgb=primary,
        accent_color_rgb=accent,
        skin_color_rgb=(199, 139, 96),
        gender_id=str(gender_id),
        person_variant_id=str(person_variant_id),
        apply_style_colors=False,
        draw_shadow=False,
    )
    for part in rendered.parts:
        attrs = part.get("attributes", {}) if isinstance(part, Mapping) else {}
        parts.add(
            str(part["part_kind"]),
            tuple(float(value) for value in part["bbox"]),  # type: ignore[arg-type]
            **(dict(attrs) if isinstance(attrs, Mapping) else {}),
        )
    return [tuple(float(value) for value in rendered.bbox_xyxy)]


def _draw_plant(
    draw: ImageDraw.ImageDraw,
    *,
    box: BBox,
    object_type: str,
    primary: RGB,
    accent: RGB,
    outline: RGB | None,
    width: int,
    scale: int,
    parts: _PartBuilder,
    object_variant_id: str | None = None,
) -> List[BBox]:
    boxes: List[BBox] = []
    if object_type == "tree":
        if object_variant_id is not None:
            tree_variant = normalize_object_variant_id("tree", str(object_variant_id))
            trunk = _rel_bbox(box, 0.44, 0.50, 0.57, 0.91)
            _rect(draw, trunk, fill=(123, 87, 55), outline=outline, width=max(1, width - 1), scale=scale, radius=4)
            boxes.append(trunk)
            if tree_variant == "pine":
                leaf_specs = (
                    _rel_bbox(box, 0.37, 0.08, 0.63, 0.32),
                    _rel_bbox(box, 0.28, 0.24, 0.72, 0.52),
                    _rel_bbox(box, 0.20, 0.42, 0.80, 0.72),
                )
                for index, leaf_box in enumerate(leaf_specs):
                    x0, y0, x1, y1 = leaf_box
                    points = ((0.5 * (x0 + x1), y0), (x1, y1), (x0, y1))
                    _poly(draw, points, fill=primary, outline=outline, width=max(1, width - 1), scale=scale)
                    parts.add("leaf", leaf_box)
                    boxes.append(leaf_box)
                    if index < 3:
                        parts.add("leaf", _expand(leaf_box, -2.0))
                return boxes
            leaf_centers = {
                "oak": ((0.35, 0.28), (0.50, 0.17), (0.65, 0.28), (0.29, 0.44), (0.50, 0.42), (0.71, 0.44)),
                "maple": ((0.34, 0.26), (0.50, 0.17), (0.66, 0.26), (0.27, 0.43), (0.50, 0.45), (0.73, 0.43)),
                "fruit_tree": ((0.35, 0.30), (0.50, 0.18), (0.64, 0.30), (0.28, 0.45), (0.50, 0.42), (0.72, 0.45)),
            }.get(tree_variant, ((0.35, 0.30), (0.50, 0.18), (0.64, 0.30), (0.28, 0.45), (0.50, 0.42), (0.72, 0.45)))
            if tree_variant == "maple":
                crown = (
                    (box[0] + 0.50 * (box[2] - box[0]), box[1] + 0.06 * (box[3] - box[1])),
                    (box[0] + 0.75 * (box[2] - box[0]), box[1] + 0.23 * (box[3] - box[1])),
                    (box[0] + 0.86 * (box[2] - box[0]), box[1] + 0.47 * (box[3] - box[1])),
                    (box[0] + 0.67 * (box[2] - box[0]), box[1] + 0.66 * (box[3] - box[1])),
                    (box[0] + 0.50 * (box[2] - box[0]), box[1] + 0.58 * (box[3] - box[1])),
                    (box[0] + 0.33 * (box[2] - box[0]), box[1] + 0.66 * (box[3] - box[1])),
                    (box[0] + 0.14 * (box[2] - box[0]), box[1] + 0.47 * (box[3] - box[1])),
                    (box[0] + 0.25 * (box[2] - box[0]), box[1] + 0.23 * (box[3] - box[1])),
                )
                _poly(draw, crown, fill=_blend_rgb(primary, (255, 255, 255), 0.08), outline=outline, width=max(1, width - 1), scale=scale)
                boxes.append(_rel_bbox(box, 0.14, 0.06, 0.86, 0.66))
            for cx, cy in leaf_centers:
                leaf_w = 0.17 if tree_variant == "maple" else 0.13
                leaf_h = 0.15 if tree_variant == "maple" else 0.12
                leaf = _rel_bbox(box, cx - leaf_w, cy - leaf_h, cx + leaf_w, cy + leaf_h)
                if tree_variant == "maple":
                    points = (
                        (leaf[0] + 0.50 * (leaf[2] - leaf[0]), leaf[1]),
                        (leaf[0] + 0.78 * (leaf[2] - leaf[0]), leaf[1] + 0.33 * (leaf[3] - leaf[1])),
                        (leaf[2], leaf[1] + 0.54 * (leaf[3] - leaf[1])),
                        (leaf[0] + 0.62 * (leaf[2] - leaf[0]), leaf[1] + 0.64 * (leaf[3] - leaf[1])),
                        (leaf[0] + 0.54 * (leaf[2] - leaf[0]), leaf[3]),
                        (leaf[0] + 0.40 * (leaf[2] - leaf[0]), leaf[1] + 0.66 * (leaf[3] - leaf[1])),
                        (leaf[0], leaf[1] + 0.56 * (leaf[3] - leaf[1])),
                        (leaf[0] + 0.22 * (leaf[2] - leaf[0]), leaf[1] + 0.34 * (leaf[3] - leaf[1])),
                    )
                    _poly(draw, points, fill=primary, outline=outline, width=max(1, width - 1), scale=scale)
                else:
                    _ellipse(draw, leaf, fill=primary, outline=outline, width=max(1, width - 1), scale=scale)
                parts.add("leaf", leaf)
                boxes.append(leaf)
            if tree_variant == "fruit_tree":
                for fx, fy in ((0.42, 0.31), (0.56, 0.27), (0.34, 0.45), (0.62, 0.48)):
                    fruit = _rel_bbox(box, fx - 0.035, fy - 0.035, fx + 0.035, fy + 0.035)
                    _ellipse(draw, fruit, fill=accent, outline=None, width=1, scale=scale)
                    parts.add("fruit", fruit)
                    boxes.append(fruit)
            return boxes
        trunk = _rel_bbox(box, 0.44, 0.48, 0.57, 0.90)
        _rect(draw, trunk, fill=(123, 87, 55), outline=outline, width=max(1, width - 1), scale=scale, radius=4)
        leaf_centers = ((0.35, 0.30), (0.50, 0.18), (0.64, 0.30), (0.28, 0.45), (0.50, 0.42), (0.72, 0.45))
        for cx, cy in leaf_centers:
            leaf = _rel_bbox(box, cx - 0.13, cy - 0.12, cx + 0.13, cy + 0.12)
            _ellipse(draw, leaf, fill=primary, outline=outline, width=max(1, width - 1), scale=scale)
            parts.add("leaf", leaf)
            boxes.append(leaf)
        for fx, fy in ((0.42, 0.31), (0.56, 0.27), (0.34, 0.45), (0.62, 0.48)):
            fruit = _rel_bbox(box, fx - 0.035, fy - 0.035, fx + 0.035, fy + 0.035)
            _ellipse(draw, fruit, fill=accent, outline=outline, width=1, scale=scale)
            parts.add("fruit", fruit)
            boxes.append(fruit)
        boxes.append(trunk)
    elif object_type == "flower":
        stem = _rel_bbox(box, 0.48, 0.40, 0.53, 0.88)
        _rect(draw, stem, fill=primary, outline=None, width=1, scale=scale)
        for leaf in (_rel_bbox(box, 0.30, 0.54, 0.50, 0.66), _rel_bbox(box, 0.52, 0.62, 0.74, 0.75), _rel_bbox(box, 0.34, 0.72, 0.50, 0.84)):
            _ellipse(draw, leaf, fill=primary, outline=outline, width=1, scale=scale)
            parts.add("leaf", leaf)
        flower = _rel_bbox(box, 0.28, 0.08, 0.74, 0.46)
        for petal in (_rel_bbox(box, 0.40, 0.05, 0.62, 0.26), _rel_bbox(box, 0.25, 0.20, 0.48, 0.42), _rel_bbox(box, 0.55, 0.20, 0.78, 0.42), _rel_bbox(box, 0.40, 0.34, 0.62, 0.54)):
            _ellipse(draw, petal, fill=accent, outline=outline, width=1, scale=scale)
        _ellipse(draw, _rel_bbox(box, 0.45, 0.25, 0.58, 0.38), fill=(238, 206, 86), outline=outline, width=1, scale=scale)
        parts.add("flower", flower)
        boxes.extend([stem, flower])
    elif object_type == "cactus":
        stem = _rel_bbox(box, 0.38, 0.22, 0.62, 0.88)
        arm_l = _rel_bbox(box, 0.18, 0.42, 0.42, 0.60)
        arm_r = _rel_bbox(box, 0.58, 0.36, 0.82, 0.54)
        _rect(draw, stem, fill=primary, outline=outline, width=width, scale=scale, radius=12)
        _rect(draw, arm_l, fill=primary, outline=outline, width=width, scale=scale, radius=10)
        _rect(draw, arm_r, fill=primary, outline=outline, width=width, scale=scale, radius=10)
        flower = _rel_bbox(box, 0.45, 0.08, 0.58, 0.22)
        _ellipse(draw, flower, fill=accent, outline=outline, width=1, scale=scale)
        parts.add("flower", flower)
        boxes.extend([stem, arm_l, arm_r, flower])
    elif object_type == "potted_plant":
        pot = _rel_bbox(box, 0.28, 0.66, 0.72, 0.92)
        _rect(draw, pot, fill=(173, 103, 72), outline=outline, width=width, scale=scale, radius=5)
        leaf_boxes = (_rel_bbox(box, 0.20, 0.34, 0.44, 0.56), _rel_bbox(box, 0.38, 0.18, 0.62, 0.50), _rel_bbox(box, 0.58, 0.34, 0.82, 0.56), _rel_bbox(box, 0.28, 0.48, 0.52, 0.70), _rel_bbox(box, 0.50, 0.48, 0.74, 0.70))
        for leaf in leaf_boxes:
            _ellipse(draw, leaf, fill=primary, outline=outline, width=1, scale=scale)
            parts.add("leaf", leaf)
        for flower in (_rel_bbox(box, 0.33, 0.28, 0.44, 0.40), _rel_bbox(box, 0.62, 0.30, 0.73, 0.42)):
            _ellipse(draw, flower, fill=accent, outline=outline, width=1, scale=scale)
            parts.add("flower", flower)
        for fruit in (_rel_bbox(box, 0.44, 0.44, 0.50, 0.50), _rel_bbox(box, 0.54, 0.42, 0.60, 0.48), _rel_bbox(box, 0.48, 0.54, 0.54, 0.60)):
            _ellipse(draw, fruit, fill=(228, 93, 83), outline=outline, width=1, scale=scale)
            parts.add("fruit", fruit)
        boxes.extend([pot, *leaf_boxes])
    elif object_type == "lily_pad":
        x0, y0, x1, y1 = box
        bw = x1 - x0
        bh = y1 - y0
        leaf_points = (
            (x0 + 0.10 * bw, y0 + 0.56 * bh),
            (x0 + 0.18 * bw, y0 + 0.34 * bh),
            (x0 + 0.38 * bw, y0 + 0.23 * bh),
            (x0 + 0.62 * bw, y0 + 0.24 * bh),
            (x0 + 0.84 * bw, y0 + 0.39 * bh),
            (x0 + 0.90 * bw, y0 + 0.58 * bh),
            (x0 + 0.76 * bw, y0 + 0.76 * bh),
            (x0 + 0.45 * bw, y0 + 0.82 * bh),
            (x0 + 0.20 * bw, y0 + 0.72 * bh),
        )
        leaf = _rel_bbox(box, 0.10, 0.23, 0.90, 0.82)
        _poly(draw, leaf_points, fill=primary, outline=outline, width=width, scale=scale)
        notch = (
            (x0 + 0.58 * bw, y0 + 0.54 * bh),
            (x0 + 0.90 * bw, y0 + 0.42 * bh),
            (x0 + 0.88 * bw, y0 + 0.66 * bh),
        )
        _poly(draw, notch, fill=(88, 169, 205), outline=None, width=1, scale=scale)
        center = (x0 + 0.55 * bw, y0 + 0.54 * bh)
        for end in ((0.30, 0.40), (0.27, 0.64), (0.66, 0.32), (0.74, 0.70)):
            _line(
                draw,
                [center, (x0 + end[0] * bw, y0 + end[1] * bh)],
                fill=_blend_rgb(primary, (255, 255, 255), 0.26),
                width=max(1, width - 1),
                scale=scale,
            )
        flower = _rel_bbox(box, 0.34, 0.15, 0.54, 0.34)
        for petal in (
            _rel_bbox(box, 0.39, 0.13, 0.49, 0.24),
            _rel_bbox(box, 0.30, 0.20, 0.43, 0.31),
            _rel_bbox(box, 0.46, 0.20, 0.59, 0.31),
            _rel_bbox(box, 0.39, 0.27, 0.49, 0.38),
        ):
            _ellipse(draw, petal, fill=accent, outline=outline, width=1, scale=scale)
        _ellipse(draw, _rel_bbox(box, 0.42, 0.23, 0.47, 0.28), fill=(248, 229, 132), outline=None, width=1, scale=scale)
        parts.add("leaf", leaf)
        parts.add("flower", flower)
        boxes.extend([leaf, flower])
    else:
        stem = _rel_bbox(box, 0.40, 0.42, 0.60, 0.86)
        cap = _rel_bbox(box, 0.18, 0.16, 0.82, 0.50)
        _ellipse(draw, cap, fill=accent, outline=outline, width=width, scale=scale)
        _rect(draw, stem, fill=(238, 222, 188), outline=outline, width=width, scale=scale, radius=8)
        boxes.extend([stem, cap])
    return boxes


def _draw_object_item(draw: ImageDraw.ImageDraw, *, box: BBox, object_type: str, primary: RGB, accent: RGB, outline: RGB | None, width: int, scale: int, parts: _PartBuilder) -> List[BBox]:
    boxes: List[BBox] = []
    if object_type == "cloud":
        blobs = (
            _rel_bbox(box, 0.10, 0.42, 0.42, 0.72),
            _rel_bbox(box, 0.28, 0.28, 0.62, 0.70),
            _rel_bbox(box, 0.52, 0.38, 0.90, 0.72),
        )
        for blob in blobs:
            _ellipse(draw, blob, fill=(248, 250, 252), outline=outline, width=max(1, width - 1), scale=scale)
        base = _rel_bbox(box, 0.18, 0.56, 0.84, 0.78)
        _rect(draw, base, fill=(248, 250, 252), outline=None, width=1, scale=scale, radius=8)
        boxes.extend([*blobs, base])
    elif object_type == "sun":
        cx = 0.5 * (box[0] + box[2])
        cy = 0.5 * (box[1] + box[3])
        ray_r0 = 0.34 * min(box[2] - box[0], box[3] - box[1])
        ray_r1 = 0.48 * min(box[2] - box[0], box[3] - box[1])
        for idx in range(12):
            angle = 2.0 * 3.141592653589793 * idx / 12.0
            _line(
                draw,
                [(cx + ray_r0 * math.cos(angle), cy + ray_r0 * math.sin(angle)), (cx + ray_r1 * math.cos(angle), cy + ray_r1 * math.sin(angle))],
                fill=(238, 190, 67),
                width=max(2, width),
                scale=scale,
            )
        disc = _rel_bbox(box, 0.24, 0.24, 0.76, 0.76)
        _ellipse(draw, disc, fill=(247, 213, 91), outline=outline, width=width, scale=scale)
        boxes.append(disc)
    elif object_type == "buoy":
        body = _rel_bbox(box, 0.26, 0.18, 0.74, 0.80)
        cap_top = _rel_bbox(box, 0.28, 0.18, 0.72, 0.36)
        cap_bottom = _rel_bbox(box, 0.28, 0.62, 0.72, 0.80)
        _ellipse(draw, body, fill=(248, 248, 241), outline=outline, width=width, scale=scale)
        _rect(draw, cap_top, fill=primary, outline=None, width=1, scale=scale, radius=4)
        _rect(draw, cap_bottom, fill=primary, outline=None, width=1, scale=scale, radius=4)
        boxes.append(body)
    elif object_type == "streetlamp":
        pole = _rel_bbox(box, 0.47, 0.28, 0.54, 0.86)
        lamp = _rel_bbox(box, 0.28, 0.08, 0.74, 0.34)
        glow = _rel_bbox(box, 0.34, 0.14, 0.68, 0.32)
        base = _rel_bbox(box, 0.30, 0.84, 0.72, 0.92)
        _rect(draw, pole, fill=SOFT_OUTLINE_RGB, outline=None, width=1, scale=scale, radius=2)
        _ellipse(draw, lamp, fill=accent, outline=outline, width=width, scale=scale)
        _ellipse(draw, glow, fill=(248, 235, 164), outline=None, width=1, scale=scale)
        _ellipse(draw, base, fill=primary, outline=outline, width=width, scale=scale)
        parts.add("light", glow)
        boxes.extend([pole, lamp, base])
    elif object_type == "traffic_light":
        pole = _rel_bbox(box, 0.47, 0.48, 0.54, 0.92)
        housing = _rel_bbox(box, 0.28, 0.08, 0.74, 0.56)
        _rect(draw, pole, fill=SOFT_OUTLINE_RGB, outline=None, width=1, scale=scale, radius=2)
        _rect(draw, housing, fill=(49, 57, 66), outline=outline, width=width, scale=scale, radius=8)
        for idx, fill in enumerate(((219, 75, 71), (232, 191, 69), (82, 169, 103))):
            light = _rel_bbox(box, 0.38, 0.13 + idx * 0.13, 0.64, 0.25 + idx * 0.13)
            _ellipse(draw, light, fill=fill, outline=(26, 31, 37), width=1, scale=scale)
            highlight = _rel_bbox(light, 0.18, 0.18, 0.42, 0.42)
            _ellipse(draw, highlight, fill=(249, 241, 210), outline=None, width=1, scale=scale)
            parts.add("light", light)
        boxes.extend([pole, housing])
    elif object_type == "road_sign":
        post = _rel_bbox(box, 0.47, 0.42, 0.54, 0.90)
        panel = _rel_bbox(box, 0.18, 0.12, 0.82, 0.45)
        _rect(draw, post, fill=SOFT_OUTLINE_RGB, outline=None, width=1, scale=scale, radius=2)
        _rect(draw, panel, fill=accent, outline=outline, width=width, scale=scale, radius=5)
        boxes.extend([post, panel])
    elif object_type == "mailbox":
        post = _rel_bbox(box, 0.40, 0.68, 0.48, 0.92)
        body = _rel_bbox(box, 0.16, 0.30, 0.74, 0.72)
        lower = _rel_bbox(box, 0.16, 0.50, 0.74, 0.72)
        door_line = _rel_bbox(box, 0.66, 0.40, 0.70, 0.70)
        flag_post = _rel_bbox(box, 0.72, 0.33, 0.77, 0.54)
        flag = _rel_bbox(box, 0.76, 0.32, 0.92, 0.42)
        _rect(draw, post, fill=SOFT_OUTLINE_RGB, outline=None, width=1, scale=scale, radius=2)
        _ellipse(draw, body, fill=primary, outline=outline, width=width, scale=scale)
        _rect(draw, lower, fill=primary, outline=outline, width=width, scale=scale, radius=3)
        _line(draw, [(door_line[0], door_line[1]), (door_line[0], door_line[3])], fill=outline or SOFT_OUTLINE_RGB, width=max(1, width - 1), scale=scale)
        _rect(draw, flag_post, fill=SOFT_OUTLINE_RGB, outline=None, width=1, scale=scale, radius=1)
        _rect(draw, flag, fill=accent, outline=outline, width=max(1, width - 1), scale=scale, radius=2)
        boxes.extend([post, body, lower, flag])
    elif object_type == "bench":
        seat = _rel_bbox(box, 0.10, 0.52, 0.90, 0.64)
        back = _rel_bbox(box, 0.12, 0.30, 0.88, 0.46)
        _rect(draw, back, fill=primary, outline=outline, width=width, scale=scale, radius=4)
        _rect(draw, seat, fill=primary, outline=outline, width=width, scale=scale, radius=4)
        for lx in (0.18, 0.34, 0.66, 0.82):
            leg = _rel_bbox(box, lx, 0.62, lx + 0.035, 0.88)
            _rect(draw, leg, fill=SOFT_OUTLINE_RGB, outline=None, width=1, scale=scale, radius=2)
            parts.add("leg", leg)
        boxes.extend([seat, back])
    elif object_type == "trash_bin":
        x0, y0, x1, y1 = box
        bw = x1 - x0
        bh = y1 - y0
        body_points = (
            (x0 + 0.28 * bw, y0 + 0.28 * bh),
            (x0 + 0.72 * bw, y0 + 0.28 * bh),
            (x0 + 0.66 * bw, y0 + 0.88 * bh),
            (x0 + 0.34 * bw, y0 + 0.88 * bh),
        )
        body = _rel_bbox(box, 0.28, 0.28, 0.72, 0.88)
        lid = _rel_bbox(box, 0.18, 0.17, 0.82, 0.30)
        base = _rel_bbox(box, 0.32, 0.84, 0.68, 0.91)
        _poly(draw, body_points, fill=primary, outline=outline, width=width, scale=scale)
        _rect(draw, lid, fill=accent, outline=outline, width=width, scale=scale, radius=5)
        _rect(draw, base, fill=_blend_rgb(primary, (0, 0, 0), 0.22), outline=outline, width=max(1, width - 1), scale=scale, radius=3)
        for rx in (0.40, 0.50, 0.60):
            _line(
                draw,
                [(x0 + rx * bw, y0 + 0.34 * bh), (x0 + (rx - 0.02) * bw, y0 + 0.80 * bh)],
                fill=_blend_rgb(primary, (255, 255, 255), 0.22),
                width=max(1, width - 1),
                scale=scale,
            )
        boxes.extend([body, lid, base])
    elif object_type == "backpack":
        body = _rel_bbox(box, 0.22, 0.22, 0.78, 0.86)
        pocket = _rel_bbox(box, 0.32, 0.54, 0.68, 0.78)
        handle = _rel_bbox(box, 0.39, 0.12, 0.61, 0.30)
        strap_l = _rel_bbox(box, 0.12, 0.30, 0.32, 0.78)
        strap_r = _rel_bbox(box, 0.68, 0.30, 0.88, 0.78)
        _arc(draw, handle, start=180, end=360, fill=outline or SOFT_OUTLINE_RGB, width=max(2, width), scale=scale)
        _rect(draw, body, fill=primary, outline=outline, width=width, scale=scale, radius=10)
        _rect(draw, pocket, fill=accent, outline=outline, width=max(1, width - 1), scale=scale, radius=6)
        _line(draw, [(strap_l[0], strap_l[1]), (strap_l[2], strap_l[3])], fill=accent, width=max(2, width), scale=scale)
        _line(draw, [(strap_r[2], strap_r[1]), (strap_r[0], strap_r[3])], fill=accent, width=max(2, width), scale=scale)
        parts.add("handle", handle)
        boxes.extend([body, pocket, handle, strap_l, strap_r])
    elif object_type == "apple":
        body = _rel_bbox(box, 0.20, 0.25, 0.80, 0.84)
        left_lobe = _rel_bbox(box, 0.18, 0.30, 0.55, 0.76)
        right_lobe = _rel_bbox(box, 0.45, 0.30, 0.82, 0.76)
        fruit_fill = (205, 72, 68)
        _ellipse(draw, body, fill=fruit_fill, outline=outline, width=width, scale=scale)
        _ellipse(draw, left_lobe, fill=fruit_fill, outline=outline, width=max(1, width - 1), scale=scale)
        _ellipse(draw, right_lobe, fill=fruit_fill, outline=outline, width=max(1, width - 1), scale=scale)
        stem = _rel_bbox(box, 0.48, 0.12, 0.56, 0.32)
        _line(draw, [(stem[0], stem[3]), (stem[2], stem[1])], fill=(103, 75, 45), width=max(2, width), scale=scale)
        leaf = _rel_bbox(box, 0.56, 0.13, 0.78, 0.31)
        _ellipse(draw, leaf, fill=(91, 151, 88), outline=outline, width=1, scale=scale)
        fruit = _union([body, left_lobe, right_lobe])
        parts.add("fruit", fruit)
        boxes.extend([fruit, stem, leaf])
    elif object_type == "egg":
        egg = _rel_bbox(box, 0.16, 0.28, 0.84, 0.72)
        _ellipse(draw, egg, fill=(246, 238, 207), outline=outline, width=width, scale=scale)
        highlight = _rel_bbox(box, 0.30, 0.38, 0.42, 0.50)
        _ellipse(draw, highlight, fill=(255, 251, 231), outline=None, width=1, scale=scale)
        boxes.append(egg)
    elif object_type == "soccer_ball":
        ball = _rel_bbox(box, 0.12, 0.12, 0.88, 0.88)
        _ellipse(draw, ball, fill=(246, 247, 245), outline=outline, width=width, scale=scale)
        cx = (ball[0] + ball[2]) * 0.5
        cy = (ball[1] + ball[3]) * 0.5
        radius = (ball[2] - ball[0]) * 0.5
        panel = (36, 40, 48)
        seam = (58, 64, 74)
        center_poly = _regular_polygon_points(cx, cy, radius * 0.24, 5, start_degrees=-90.0)
        outer_centers: list[tuple[float, float]] = []
        for angle in (-90, -18, 54, 126, 198):
            a = math.radians(float(angle))
            outer_centers.append((cx + radius * 0.57 * math.cos(a), cy + radius * 0.57 * math.sin(a)))
        for index, point in enumerate(center_poly):
            _line(draw, [point, outer_centers[index]], fill=seam, width=max(2, width), scale=scale)
        for angle, center in zip((-90, -18, 54, 126, 198), outer_centers):
            ex = cx + radius * 0.86 * math.cos(math.radians(float(angle)))
            ey = cy + radius * 0.86 * math.sin(math.radians(float(angle)))
            _line(draw, [center, (ex, ey)], fill=seam, width=max(1, width), scale=scale)
        _poly(draw, center_poly, fill=panel, outline=seam, width=max(1, width - 1), scale=scale)
        for angle, center in zip((-90, -18, 54, 126, 198), outer_centers):
            outer_poly = _regular_polygon_points(center[0], center[1], radius * 0.16, 5, start_degrees=float(angle))
            _poly(draw, outer_poly, fill=panel, outline=seam, width=max(1, width - 1), scale=scale)
        highlight = _rel_bbox(box, 0.26, 0.22, 0.42, 0.34)
        _arc(draw, highlight, start=205, end=335, fill=(255, 255, 255), width=max(1, width - 1), scale=scale)
        boxes.append(ball)
    elif object_type == "rugby_ball":
        ball_points: list[tuple[float, float]] = []
        cx = box[0] + 0.50 * (box[2] - box[0])
        cy = box[1] + 0.50 * (box[3] - box[1])
        rx = 0.43 * (box[2] - box[0])
        ry = 0.31 * (box[3] - box[1])
        for index in range(13):
            theta = math.pi * float(index) / 12.0
            ball_points.append((cx + rx * math.cos(theta), cy - ry * math.sin(theta)))
        for index in range(13):
            theta = math.pi * float(12 - index) / 12.0
            ball_points.append((cx + rx * math.cos(theta), cy + ry * math.sin(theta)))
        _poly(draw, ball_points, fill=(151, 87, 50), outline=outline, width=width, scale=scale)
        seam = _rel_bbox(box, 0.32, 0.43, 0.68, 0.57)
        _line(draw, [(seam[0], cy), (seam[2], cy)], fill=(246, 238, 218), width=max(2, width), scale=scale)
        for tick_x in (0.39, 0.45, 0.51, 0.57):
            _line(
                draw,
                [
                    (box[0] + tick_x * (box[2] - box[0]), cy - 0.055 * (box[3] - box[1])),
                    (box[0] + tick_x * (box[2] - box[0]), cy + 0.055 * (box[3] - box[1])),
                ],
                fill=(246, 238, 218),
                width=max(1, width - 1),
                scale=scale,
            )
        _arc(draw, _rel_bbox(box, 0.18, 0.24, 0.82, 0.76), start=202, end=338, fill=(105, 61, 39), width=max(1, width - 1), scale=scale)
        boxes.append(_rel_bbox(box, 0.07, 0.19, 0.93, 0.81))
    elif object_type == "spoon":
        bowl = _rel_bbox(box, 0.06, 0.16, 0.38, 0.84)
        handle = _rel_bbox(box, 0.33, 0.36, 0.94, 0.64)
        neck = _rel_bbox(box, 0.30, 0.39, 0.44, 0.61)
        _ellipse(draw, bowl, fill=(230, 224, 210), outline=outline, width=width, scale=scale)
        _rect(draw, neck, fill=(201, 194, 181), outline=outline, width=max(1, width - 1), scale=scale, radius=4)
        _rect(draw, handle, fill=(201, 194, 181), outline=outline, width=max(1, width - 1), scale=scale, radius=5)
        _arc(draw, _rel_bbox(bowl, 0.24, 0.20, 0.80, 0.82), start=205, end=335, fill=(246, 242, 232), width=max(1, width - 1), scale=scale)
        boxes.extend([bowl, neck, handle])
    elif object_type == "plate":
        rim = _rel_bbox(box, 0.08, 0.14, 0.92, 0.86)
        well = _rel_bbox(box, 0.22, 0.28, 0.78, 0.72)
        _ellipse(draw, rim, fill=(238, 238, 230), outline=outline, width=width, scale=scale)
        _ellipse(draw, well, fill=(220, 228, 228), outline=(178, 187, 188) if outline else None, width=max(1, width - 1), scale=scale)
        boxes.append(rim)
    elif object_type == "book":
        pages = _rel_bbox(box, 0.18, 0.30, 0.90, 0.88)
        cover = _rel_bbox(box, 0.10, 0.20, 0.86, 0.80)
        spine = _rel_bbox(box, 0.10, 0.20, 0.24, 0.82)
        fore_edge = _rel_bbox(box, 0.78, 0.30, 0.90, 0.88)
        page_fill = (242, 232, 205)
        _poly(
            draw,
            (
                (pages[0], pages[1]),
                (pages[2], pages[1] + 0.08 * (pages[3] - pages[1])),
                (pages[2], pages[3]),
                (pages[0], pages[3] - 0.06 * (pages[3] - pages[1])),
            ),
            fill=page_fill,
            outline=outline,
            width=max(1, width - 1),
            scale=scale,
        )
        _rect(draw, cover, fill=primary, outline=outline, width=width, scale=scale, radius=5)
        _rect(draw, spine, fill=_blend_rgb(primary, (0, 0, 0), 0.20), outline=outline, width=max(1, width - 1), scale=scale, radius=4)
        _line(draw, [(fore_edge[0], fore_edge[1]), (fore_edge[0], fore_edge[3])], fill=_shade_rgb(page_fill, -42), width=max(1, width), scale=scale)
        for line_y in (0.38, 0.50, 0.62, 0.74):
            y = pages[1] + line_y * (pages[3] - pages[1])
            _line(
                draw,
                [(fore_edge[0] + 2.0, y), (fore_edge[2] - 2.0, y + 0.02 * (pages[3] - pages[1]))],
                fill=_shade_rgb(page_fill, -34),
                width=max(1, width - 1),
                scale=scale,
            )
        _line(
            draw,
            [(spine[2] + 0.06 * (cover[2] - cover[0]), cover[1] + 4.0), (spine[2] + 0.06 * (cover[2] - cover[0]), cover[3] - 4.0)],
            fill=accent,
            width=max(2, width),
            scale=scale,
        )
        boxes.extend([cover, pages, spine])
    elif object_type == "camera":
        body = _rel_bbox(box, 0.12, 0.28, 0.88, 0.78)
        top = _rel_bbox(box, 0.27, 0.18, 0.54, 0.34)
        lens = _rel_bbox(box, 0.38, 0.38, 0.68, 0.68)
        inner = _rel_bbox(box, 0.46, 0.46, 0.60, 0.60)
        flash = _rel_bbox(box, 0.72, 0.36, 0.80, 0.45)
        _rect(draw, body, fill=primary, outline=outline, width=width, scale=scale, radius=8)
        _rect(draw, top, fill=accent, outline=outline, width=max(1, width - 1), scale=scale, radius=5)
        _ellipse(draw, lens, fill=(226, 232, 236), outline=outline, width=width, scale=scale)
        _ellipse(draw, inner, fill=(52, 59, 70), outline=None, width=1, scale=scale)
        _rect(draw, flash, fill=(248, 235, 164), outline=outline, width=1, scale=scale, radius=2)
        boxes.extend([body, top, lens, flash])
    elif object_type == "remote":
        body = _rel_bbox(box, 0.10, 0.06, 0.90, 0.94)
        _rect(draw, body, fill=(65, 72, 82), outline=outline, width=width, scale=scale, radius=7)
        for row in range(4):
            for col in range(2):
                cx0 = 0.34 + 0.20 * col
                cy0 = 0.30 + 0.13 * row
                button = _rel_bbox(box, cx0, cy0, cx0 + 0.12, cy0 + 0.055)
                _ellipse(draw, button, fill=(216, 222, 225), outline=None, width=1, scale=scale)
        red = _rel_bbox(box, 0.42, 0.14, 0.58, 0.21)
        _ellipse(draw, red, fill=(220, 80, 75), outline=None, width=1, scale=scale)
        boxes.append(body)
    elif object_type == "pencil":
        shaft = _rel_bbox(box, 0.16, 0.14, 0.76, 0.88)
        tip = _rel_bbox(box, 0.74, 0.08, 0.94, 0.94)
        eraser = _rel_bbox(box, 0.06, 0.18, 0.18, 0.84)
        band = _rel_bbox(box, 0.16, 0.18, 0.22, 0.84)
        _rect(draw, shaft, fill=(235, 186, 72), outline=outline, width=width, scale=scale, radius=4)
        _rect(draw, eraser, fill=(214, 118, 132), outline=outline, width=max(1, width - 1), scale=scale, radius=3)
        _rect(draw, band, fill=(186, 188, 182), outline=outline, width=max(1, width - 1), scale=scale, radius=1)
        for ridge_y in (0.34, 0.51, 0.68):
            y = box[1] + ridge_y * (box[3] - box[1])
            _line(draw, [(shaft[0] + 2.0, y), (shaft[2] - 2.0, y)], fill=(204, 147, 46), width=max(1, width - 1), scale=scale)
        _poly(
            draw,
            [(tip[0], tip[1]), (tip[2], (tip[1] + tip[3]) / 2.0), (tip[0], tip[3])],
            fill=(222, 177, 118),
            outline=outline,
            width=width,
            scale=scale,
        )
        graphite = _rel_bbox(box, 0.88, 0.36, 0.96, 0.66)
        _poly(
            draw,
            [(graphite[0], graphite[1]), (graphite[2], (graphite[1] + graphite[3]) / 2.0), (graphite[0], graphite[3])],
            fill=SOFT_OUTLINE_RGB,
            outline=None,
            width=1,
            scale=scale,
        )
        boxes.extend([shaft, eraser, band, tip, graphite])
    elif object_type == "ruler":
        body = _rel_bbox(box, 0.06, 0.08, 0.94, 0.92)
        _rect(draw, body, fill=(235, 208, 105), outline=outline, width=width, scale=scale, radius=4)
        for idx in range(1, 10):
            x = body[0] + idx * (body[2] - body[0]) / 10.0
            tick_h = 0.36 if idx % 2 else 0.52
            _line(
                draw,
                [(x, body[1] + 0.08 * (body[3] - body[1])), (x, body[1] + tick_h * (body[3] - body[1]))],
                fill=(128, 105, 55),
                width=max(1, width - 1),
                scale=scale,
            )
        _line(
            draw,
            [(body[0] + 0.10 * (body[2] - body[0]), body[1] + 0.74 * (body[3] - body[1])), (body[0] + 0.90 * (body[2] - body[0]), body[1] + 0.74 * (body[3] - body[1]))],
            fill=(190, 160, 82),
            width=max(1, width),
            scale=scale,
        )
        boxes.append(body)
    elif object_type == "clock":
        face = _rel_bbox(box, 0.14, 0.12, 0.86, 0.84)
        _ellipse(draw, face, fill=(244, 242, 226), outline=outline, width=width, scale=scale)
        center_x = (face[0] + face[2]) / 2.0
        center_y = (face[1] + face[3]) / 2.0
        _line(
            draw,
            [(center_x, center_y), (center_x, face[1] + 0.25 * (face[3] - face[1]))],
            fill=SOFT_OUTLINE_RGB,
            width=max(2, width),
            scale=scale,
        )
        _line(
            draw,
            [(center_x, center_y), (face[0] + 0.70 * (face[2] - face[0]), face[1] + 0.60 * (face[3] - face[1]))],
            fill=SOFT_OUTLINE_RGB,
            width=max(2, width),
            scale=scale,
        )
        _ellipse(draw, _rel_bbox(box, 0.47, 0.48, 0.53, 0.54), fill=SOFT_OUTLINE_RGB, outline=None, width=1, scale=scale)
        base = _rel_bbox(box, 0.30, 0.82, 0.70, 0.90)
        _rect(draw, base, fill=primary, outline=outline, width=max(1, width - 1), scale=scale, radius=4)
        boxes.extend([face, base])
    elif object_type == "gift":
        box_body = _rel_bbox(box, 0.18, 0.36, 0.82, 0.84)
        lid = _rel_bbox(box, 0.12, 0.28, 0.88, 0.44)
        ribbon_v = _rel_bbox(box, 0.44, 0.28, 0.56, 0.84)
        ribbon_h = _rel_bbox(box, 0.18, 0.50, 0.82, 0.60)
        left_bow = _rel_bbox(box, 0.28, 0.12, 0.50, 0.32)
        right_bow = _rel_bbox(box, 0.50, 0.12, 0.72, 0.32)
        _rect(draw, box_body, fill=primary, outline=outline, width=width, scale=scale, radius=5)
        _rect(draw, lid, fill=accent, outline=outline, width=width, scale=scale, radius=5)
        _rect(draw, ribbon_v, fill=(248, 235, 164), outline=outline, width=1, scale=scale, radius=2)
        _rect(draw, ribbon_h, fill=(248, 235, 164), outline=None, width=1, scale=scale, radius=1)
        _ellipse(draw, left_bow, fill=accent, outline=outline, width=max(1, width - 1), scale=scale)
        _ellipse(draw, right_bow, fill=accent, outline=outline, width=max(1, width - 1), scale=scale)
        boxes.extend([box_body, lid, left_bow, right_bow])
    elif object_type == "vase":
        neck = _rel_bbox(box, 0.38, 0.10, 0.62, 0.34)
        body = _rel_bbox(box, 0.20, 0.28, 0.80, 0.86)
        lip = _rel_bbox(box, 0.32, 0.08, 0.68, 0.20)
        _ellipse(draw, body, fill=primary, outline=outline, width=width, scale=scale)
        _rect(draw, neck, fill=accent, outline=outline, width=max(1, width - 1), scale=scale, radius=5)
        _ellipse(draw, lip, fill=(238, 238, 230), outline=outline, width=max(1, width - 1), scale=scale)
        boxes.extend([body, neck, lip])
    elif object_type == "bowl":
        rim = _rel_bbox(box, 0.12, 0.26, 0.88, 0.56)
        body = _rel_bbox(box, 0.18, 0.34, 0.82, 0.84)
        _ellipse(draw, body, fill=primary, outline=outline, width=width, scale=scale)
        _ellipse(draw, rim, fill=(242, 238, 225), outline=outline, width=width, scale=scale)
        inner = _rel_bbox(box, 0.24, 0.34, 0.76, 0.50)
        _ellipse(draw, inner, fill=(219, 214, 202), outline=None, width=1, scale=scale)
        boxes.extend([body, rim])
    elif object_type == "candle":
        wax = _rel_bbox(box, 0.30, 0.34, 0.70, 0.86)
        flame = _rel_bbox(box, 0.38, 0.10, 0.62, 0.36)
        wick = _rel_bbox(box, 0.48, 0.28, 0.52, 0.38)
        _rect(draw, wax, fill=(248, 239, 212), outline=outline, width=width, scale=scale, radius=5)
        _line(draw, [(wick[0], wick[3]), (wick[2], wick[1])], fill=SOFT_OUTLINE_RGB, width=max(2, width), scale=scale)
        _ellipse(draw, flame, fill=(244, 151, 61), outline=(170, 91, 45) if outline else None, width=max(1, width - 1), scale=scale)
        base = _rel_bbox(box, 0.24, 0.82, 0.76, 0.92)
        _ellipse(draw, base, fill=accent, outline=outline, width=max(1, width - 1), scale=scale)
        boxes.extend([wax, flame, base])
    elif object_type == "lightbulb":
        bulb = _rel_bbox(box, 0.22, 0.12, 0.78, 0.60)
        neck = _rel_bbox(box, 0.36, 0.56, 0.64, 0.72)
        base = _rel_bbox(box, 0.34, 0.68, 0.66, 0.86)
        cx = (bulb[0] + bulb[2]) * 0.5
        cy = bulb[1] + 0.28 * (bulb[3] - bulb[1])
        ray_r0 = 0.34 * (bulb[2] - bulb[0])
        ray_r1 = 0.48 * (bulb[2] - bulb[0])
        for angle in (-150, -118, -62, -30, 0):
            a = math.radians(float(angle))
            _line(draw, [(cx + ray_r0 * math.cos(a), cy + ray_r0 * math.sin(a)), (cx + ray_r1 * math.cos(a), cy + ray_r1 * math.sin(a))], fill=(238, 190, 67), width=max(1, width - 1), scale=scale)
        _ellipse(draw, bulb, fill=(248, 235, 164), outline=outline, width=width, scale=scale)
        _poly(draw, [(neck[0], neck[1]), (neck[2], neck[1]), (base[2], base[1]), (base[0], base[1])], fill=(248, 235, 164), outline=outline, width=max(1, width - 1), scale=scale)
        _rect(draw, base, fill=(159, 164, 166), outline=outline, width=max(1, width - 1), scale=scale, radius=3)
        for y in (0.72, 0.79):
            _line(draw, [(box[0] + 0.36 * (box[2] - box[0]), box[1] + y * (box[3] - box[1])), (box[0] + 0.64 * (box[2] - box[0]), box[1] + y * (box[3] - box[1]))], fill=SOFT_OUTLINE_RGB, width=1, scale=scale)
        parts.add("light", bulb)
        boxes.extend([bulb, neck, base])
    elif object_type == "mug":
        cup = _rel_bbox(box, 0.24, 0.30, 0.70, 0.82)
        handle = _rel_bbox(box, 0.62, 0.42, 0.90, 0.70)
        _ellipse(draw, handle, fill=(246, 246, 242), outline=outline, width=max(2, width), scale=scale)
        _rect(draw, cup, fill=primary, outline=outline, width=width, scale=scale, radius=8)
        parts.add("handle", handle)
        boxes.extend([cup, handle])
    elif object_type == "teapot":
        body = _rel_bbox(box, 0.22, 0.34, 0.70, 0.78)
        spout = [
            (box[0] + 0.24 * (box[2] - box[0]), box[1] + 0.48 * (box[3] - box[1])),
            (box[0] + 0.04 * (box[2] - box[0]), box[1] + 0.40 * (box[3] - box[1])),
            (box[0] + 0.18 * (box[2] - box[0]), box[1] + 0.63 * (box[3] - box[1])),
        ]
        handle = _rel_bbox(box, 0.60, 0.40, 0.92, 0.72)
        inner = _rel_bbox(box, 0.69, 0.49, 0.83, 0.63)
        lid = _rel_bbox(box, 0.36, 0.24, 0.58, 0.38)
        knob = _rel_bbox(box, 0.43, 0.16, 0.52, 0.25)
        _poly(draw, spout, fill=primary, outline=outline, width=width, scale=scale)
        _ellipse(draw, handle, fill=accent, outline=outline, width=width, scale=scale)
        _ellipse(draw, inner, fill=(246, 246, 242), outline=None, width=1, scale=scale)
        _ellipse(draw, body, fill=primary, outline=outline, width=width, scale=scale)
        _rect(draw, lid, fill=accent, outline=outline, width=max(1, width - 1), scale=scale, radius=5)
        _ellipse(draw, knob, fill=accent, outline=outline, width=1, scale=scale)
        parts.add("handle", handle)
        boxes.extend([body, _rel_bbox(box, 0.04, 0.40, 0.24, 0.63), handle, lid, knob])
    elif object_type == "chair":
        back = _rel_bbox(box, 0.26, 0.16, 0.74, 0.52)
        seat = _rel_bbox(box, 0.20, 0.50, 0.80, 0.62)
        _rect(draw, back, fill=primary, outline=outline, width=width, scale=scale, radius=4)
        _rect(draw, seat, fill=accent, outline=outline, width=width, scale=scale, radius=4)
        for leg in (_rel_bbox(box, 0.23, 0.60, 0.30, 0.90), _rel_bbox(box, 0.38, 0.60, 0.45, 0.90), _rel_bbox(box, 0.58, 0.60, 0.65, 0.90), _rel_bbox(box, 0.73, 0.60, 0.80, 0.90)):
            _rect(draw, leg, fill=primary, outline=outline, width=1, scale=scale, radius=2)
            parts.add("leg", leg)
        boxes.extend([back, seat])
    elif object_type == "table":
        top = _rel_bbox(box, 0.10, 0.26, 0.90, 0.46)
        lip = _rel_bbox(box, 0.14, 0.45, 0.86, 0.56)
        top_poly = (
            (top[0] + 0.08 * (top[2] - top[0]), top[1]),
            (top[2] - 0.08 * (top[2] - top[0]), top[1]),
            (top[2], top[3]),
            (top[0], top[3]),
        )
        _poly(draw, top_poly, fill=_blend_rgb(primary, (255, 255, 255), 0.12), outline=outline, width=width, scale=scale)
        _rect(draw, lip, fill=primary, outline=outline, width=max(1, width - 1), scale=scale, radius=4)
        leg_specs = (
            _rel_bbox(box, 0.18, 0.52, 0.27, 0.88),
            _rel_bbox(box, 0.33, 0.52, 0.42, 0.86),
            _rel_bbox(box, 0.58, 0.52, 0.67, 0.86),
            _rel_bbox(box, 0.73, 0.52, 0.82, 0.88),
        )
        for leg in leg_specs:
            leg_points = (
                (leg[0] + 0.12 * (leg[2] - leg[0]), leg[1]),
                (leg[2] - 0.12 * (leg[2] - leg[0]), leg[1]),
                (leg[2], leg[3]),
                (leg[0], leg[3]),
            )
            _poly(draw, leg_points, fill=_blend_rgb(primary, (0, 0, 0), 0.12), outline=outline, width=1, scale=scale)
            parts.add("leg", leg)
        boxes.extend([top, lip])
    elif object_type == "lamp":
        shade = _rel_bbox(box, 0.24, 0.14, 0.76, 0.42)
        stand = _rel_bbox(box, 0.47, 0.40, 0.54, 0.78)
        base = _rel_bbox(box, 0.30, 0.76, 0.70, 0.88)
        light = _rel_bbox(box, 0.34, 0.23, 0.66, 0.39)
        _poly(draw, [(shade[0], shade[3]), (shade[1], shade[1]) if False else (shade[0] + 0.08 * (shade[2] - shade[0]), shade[1]), (shade[2] - 0.08 * (shade[2] - shade[0]), shade[1]), (shade[2], shade[3])], fill=accent, outline=outline, width=width, scale=scale)
        _rect(draw, stand, fill=primary, outline=outline, width=1, scale=scale, radius=2)
        _ellipse(draw, base, fill=primary, outline=outline, width=width, scale=scale)
        parts.add("light", light)
        boxes.extend([shade, stand, base])
    elif object_type == "umbrella":
        canopy_points = [
            (box[0] + 0.10 * (box[2] - box[0]), box[1] + 0.53 * (box[3] - box[1])),
            (box[0] + 0.16 * (box[2] - box[0]), box[1] + 0.36 * (box[3] - box[1])),
            (box[0] + 0.32 * (box[2] - box[0]), box[1] + 0.20 * (box[3] - box[1])),
            (box[0] + 0.50 * (box[2] - box[0]), box[1] + 0.14 * (box[3] - box[1])),
            (box[0] + 0.68 * (box[2] - box[0]), box[1] + 0.20 * (box[3] - box[1])),
            (box[0] + 0.84 * (box[2] - box[0]), box[1] + 0.36 * (box[3] - box[1])),
            (box[0] + 0.90 * (box[2] - box[0]), box[1] + 0.53 * (box[3] - box[1])),
            (box[0] + 0.70 * (box[2] - box[0]), box[1] + 0.57 * (box[3] - box[1])),
            (box[0] + 0.50 * (box[2] - box[0]), box[1] + 0.58 * (box[3] - box[1])),
            (box[0] + 0.30 * (box[2] - box[0]), box[1] + 0.57 * (box[3] - box[1])),
        ]
        _poly(draw, canopy_points, fill=primary, outline=outline, width=width, scale=scale)
        canopy = _rel_bbox(box, 0.10, 0.14, 0.90, 0.58)
        for rib_x in (0.30, 0.50, 0.70):
            _line(
                draw,
                [
                    (box[0] + 0.50 * (box[2] - box[0]), box[1] + 0.16 * (box[3] - box[1])),
                    (box[0] + rib_x * (box[2] - box[0]), box[1] + 0.56 * (box[3] - box[1])),
                ],
                fill=SOFT_OUTLINE_RGB,
                width=max(1, width - 1),
                scale=scale,
            )
        for scallop in (_rel_bbox(box, 0.12, 0.48, 0.34, 0.62), _rel_bbox(box, 0.32, 0.50, 0.52, 0.64), _rel_bbox(box, 0.50, 0.50, 0.70, 0.64), _rel_bbox(box, 0.68, 0.48, 0.90, 0.62)):
            _arc(draw, scallop, start=0, end=180, fill=outline or SOFT_OUTLINE_RGB, width=max(1, width), scale=scale)
        shaft = _rel_bbox(box, 0.48, 0.48, 0.53, 0.82)
        _rect(draw, shaft, fill=SOFT_OUTLINE_RGB, outline=None, width=1, scale=scale)
        handle = _rel_bbox(box, 0.39, 0.74, 0.65, 0.94)
        _line(
            draw,
            [
                (handle[0] + 0.44 * (handle[2] - handle[0]), handle[1]),
                (handle[0] + 0.44 * (handle[2] - handle[0]), handle[3] - 0.20 * (handle[3] - handle[1])),
                (handle[2], handle[3] - 0.20 * (handle[3] - handle[1])),
                (handle[2] - 0.08 * (handle[2] - handle[0]), handle[3]),
                (handle[0] + 0.52 * (handle[2] - handle[0]), handle[3]),
            ],
            fill=SOFT_OUTLINE_RGB,
            width=max(3, width),
            scale=scale,
        )
        parts.add("handle", _expand(handle, 4.0))
        boxes.extend([canopy, shaft, handle])
    elif object_type == "kite":
        diamond = [
            (box[0] + 0.50 * (box[2] - box[0]), box[1] + 0.08 * (box[3] - box[1])),
            (box[0] + 0.84 * (box[2] - box[0]), box[1] + 0.42 * (box[3] - box[1])),
            (box[0] + 0.50 * (box[2] - box[0]), box[1] + 0.78 * (box[3] - box[1])),
            (box[0] + 0.16 * (box[2] - box[0]), box[1] + 0.42 * (box[3] - box[1])),
        ]
        center = (box[0] + 0.50 * (box[2] - box[0]), box[1] + 0.42 * (box[3] - box[1]))
        _poly(draw, [diamond[0], diamond[1], center], fill=primary, outline=outline, width=width, scale=scale)
        _poly(draw, [diamond[1], diamond[2], center], fill=accent, outline=outline, width=width, scale=scale)
        _poly(draw, [diamond[2], diamond[3], center], fill=primary, outline=outline, width=width, scale=scale)
        _poly(draw, [diamond[3], diamond[0], center], fill=accent, outline=outline, width=width, scale=scale)
        _line(draw, [(diamond[2][0], diamond[2][1]), (box[0] + 0.42 * (box[2] - box[0]), box[1] + 0.88 * (box[3] - box[1])), (box[0] + 0.58 * (box[2] - box[0]), box[1] + 0.94 * (box[3] - box[1]))], fill=SOFT_OUTLINE_RGB, width=max(1, width - 1), scale=scale)
        boxes.append(_rel_bbox(box, 0.16, 0.08, 0.84, 0.94))
    elif object_type == "bucket":
        body_pts = [
            (box[0] + 0.22 * (box[2] - box[0]), box[1] + 0.36 * (box[3] - box[1])),
            (box[0] + 0.78 * (box[2] - box[0]), box[1] + 0.36 * (box[3] - box[1])),
            (box[0] + 0.68 * (box[2] - box[0]), box[1] + 0.84 * (box[3] - box[1])),
            (box[0] + 0.32 * (box[2] - box[0]), box[1] + 0.84 * (box[3] - box[1])),
        ]
        rim = _rel_bbox(box, 0.18, 0.28, 0.82, 0.42)
        handle = _rel_bbox(box, 0.26, 0.10, 0.74, 0.54)
        _arc(draw, handle, start=195, end=345, fill=SOFT_OUTLINE_RGB, width=max(2, width), scale=scale)
        _poly(draw, body_pts, fill=primary, outline=outline, width=width, scale=scale)
        _rect(draw, rim, fill=accent, outline=outline, width=width, scale=scale, radius=5)
        parts.add("handle", _expand(handle, 2.0))
        boxes.append(_rel_bbox(box, 0.18, 0.10, 0.82, 0.84))
    elif object_type == "shovel":
        handle = _rel_bbox(box, 0.16, 0.12, 0.58, 0.48)
        shaft_start = (box[0] + 0.30 * (box[2] - box[0]), box[1] + 0.22 * (box[3] - box[1]))
        shaft_end = (box[0] + 0.66 * (box[2] - box[0]), box[1] + 0.58 * (box[3] - box[1]))
        blade = [
            (box[0] + 0.58 * (box[2] - box[0]), box[1] + 0.52 * (box[3] - box[1])),
            (box[0] + 0.88 * (box[2] - box[0]), box[1] + 0.66 * (box[3] - box[1])),
            (box[0] + 0.74 * (box[2] - box[0]), box[1] + 0.90 * (box[3] - box[1])),
            (box[0] + 0.50 * (box[2] - box[0]), box[1] + 0.66 * (box[3] - box[1])),
        ]
        grip = _rel_bbox(box, 0.12, 0.06, 0.34, 0.28)
        _ellipse(draw, grip, fill=accent, outline=outline, width=width, scale=scale)
        _ellipse(draw, _rel_bbox(grip, 0.28, 0.28, 0.72, 0.72), fill=(246, 246, 242), outline=None, width=1, scale=scale)
        _line(draw, [shaft_start, shaft_end], fill=SOFT_OUTLINE_RGB, width=max(3, width + 1), scale=scale)
        _poly(draw, blade, fill=primary, outline=outline, width=width, scale=scale)
        parts.add("handle", _expand(handle, 3.0))
        boxes.extend([handle, grip, _rel_bbox(box, 0.50, 0.52, 0.88, 0.90)])
    elif object_type == "guitar":
        lower = _rel_bbox(box, 0.10, 0.38, 0.54, 0.84)
        upper = _rel_bbox(box, 0.34, 0.26, 0.66, 0.58)
        neck = _rel_bbox(box, 0.58, 0.37, 0.88, 0.47)
        head = _rel_bbox(box, 0.84, 0.30, 0.98, 0.54)
        _ellipse(draw, lower, fill=primary, outline=outline, width=width, scale=scale)
        _ellipse(draw, upper, fill=primary, outline=outline, width=width, scale=scale)
        _rect(draw, neck, fill=accent, outline=outline, width=max(1, width - 1), scale=scale, radius=3)
        _rect(draw, head, fill=accent, outline=outline, width=max(1, width - 1), scale=scale, radius=4)
        _ellipse(draw, _rel_bbox(box, 0.34, 0.52, 0.48, 0.66), fill=(45, 38, 32), outline=None, width=1, scale=scale)
        for offset in (-0.04, 0.0, 0.04):
            _line(draw, [(box[0] + 0.44 * (box[2] - box[0]), box[1] + (0.59 + offset) * (box[3] - box[1])), (box[0] + 0.92 * (box[2] - box[0]), box[1] + (0.42 + offset) * (box[3] - box[1]))], fill=(72, 61, 50), width=1, scale=scale)
        boxes.extend([lower, upper, neck, head])
    elif object_type == "scissors":
        for handle in (_rel_bbox(box, 0.12, 0.48, 0.36, 0.74), _rel_bbox(box, 0.12, 0.18, 0.36, 0.44)):
            _ellipse(draw, handle, fill=(246, 246, 242), outline=primary, width=max(3, width), scale=scale)
            parts.add("handle", handle)
        _line(draw, [(box[0] + 0.33 * (box[2] - box[0]), box[1] + 0.45 * (box[3] - box[1])), (box[0] + 0.88 * (box[2] - box[0]), box[1] + 0.16 * (box[3] - box[1]))], fill=SOFT_OUTLINE_RGB, width=max(4, width + 1), scale=scale)
        _line(draw, [(box[0] + 0.33 * (box[2] - box[0]), box[1] + 0.47 * (box[3] - box[1])), (box[0] + 0.88 * (box[2] - box[0]), box[1] + 0.78 * (box[3] - box[1]))], fill=SOFT_OUTLINE_RGB, width=max(4, width + 1), scale=scale)
        boxes.append(_rel_bbox(box, 0.12, 0.16, 0.88, 0.78))
    elif object_type == "key":
        handle = _rel_bbox(box, 0.06, 0.08, 0.38, 0.92)
        shaft = _rel_bbox(box, 0.34, 0.32, 0.84, 0.68)
        bridge = _rel_bbox(box, 0.30, 0.36, 0.45, 0.64)
        _ellipse(draw, handle, fill=(246, 246, 242), outline=primary, width=max(4, width + 1), scale=scale)
        _rect(draw, bridge, fill=primary, outline=outline, width=1, scale=scale, radius=3)
        _rect(draw, shaft, fill=primary, outline=outline, width=1, scale=scale, radius=3)
        _rect(draw, _rel_bbox(box, 0.74, 0.18, 0.88, 0.40), fill=primary, outline=None, width=1, scale=scale)
        _rect(draw, _rel_bbox(box, 0.76, 0.60, 0.92, 0.84), fill=primary, outline=None, width=1, scale=scale)
        _ellipse(draw, _rel_bbox(handle, 0.28, 0.28, 0.72, 0.72), fill=(246, 246, 242), outline=primary, width=max(1, width - 1), scale=scale)
        parts.add("handle", handle)
        boxes.extend([handle, bridge, shaft, _rel_bbox(box, 0.74, 0.18, 0.92, 0.84)])
    else:
        body = _rel_bbox(box, 0.30, 0.16, 0.70, 0.86)
        neck = _rel_bbox(box, 0.38, 0.06, 0.62, 0.20)
        _rect(draw, body, fill=primary, outline=outline, width=width, scale=scale, radius=10)
        _rect(draw, neck, fill=accent, outline=outline, width=width, scale=scale, radius=4)
        boxes.extend([body, neck])
    return boxes


def draw_illustration_object(
    draw: ImageDraw.ImageDraw,
    *,
    object_id: str,
    object_type: str,
    bbox_xyxy: BBox,
    primary_color_rgb: RGB,
    accent_color_rgb: RGB,
    style_id: str,
    render_scale: int,
    gender_id: str | None = None,
    object_variant_id: str | None = None,
) -> IllustrationObject:
    """Draw one object and return semantic bbox metadata for all visible parts."""

    template = OBJECT_TEMPLATES[str(object_type)]
    outline, width, shadow = _style_params(str(style_id))
    primary_color_rgb, accent_color_rgb = style_object_colors(
        str(style_id),
        tuple(int(v) for v in primary_color_rgb),
        tuple(int(v) for v in accent_color_rgb),
    )
    box = tuple(float(v) for v in bbox_xyxy)
    if shadow:
        _draw_shadow(draw, box, scale=int(render_scale))
    parts = _PartBuilder(str(object_id))
    family = str(template.family)
    if object_type in {"bird", "duck"}:
        boxes = _draw_bird_like(draw, box=box, object_type=str(object_type), primary=primary_color_rgb, accent=accent_color_rgb, outline=outline, width=width, scale=int(render_scale), parts=parts)
    elif object_type in {"quadruped", "rabbit"}:
        boxes = _draw_quadruped_like(draw, box=box, object_type=str(object_type), primary=primary_color_rgb, accent=accent_color_rgb, outline=outline, width=width, scale=int(render_scale), parts=parts)
    elif object_type == "fish":
        boxes = _draw_fish(draw, box=box, primary=primary_color_rgb, accent=accent_color_rgb, outline=outline, width=width, scale=int(render_scale), parts=parts)
    elif object_type == "butterfly":
        boxes = _draw_butterfly(draw, box=box, primary=primary_color_rgb, accent=accent_color_rgb, outline=outline, width=width, scale=int(render_scale), parts=parts)
    elif object_type in {"car", "bus", "truck", "taxi", "van"}:
        boxes = _draw_vehicle(draw, box=box, object_type=str(object_type), primary=primary_color_rgb, accent=accent_color_rgb, outline=outline, width=width, scale=int(render_scale), parts=parts)
    elif object_type == "train":
        boxes = _draw_train(draw, box=box, primary=primary_color_rgb, accent=accent_color_rgb, outline=outline, width=width, scale=int(render_scale), parts=parts)
    elif object_type == "bicycle":
        boxes = _draw_bicycle(draw, box=box, primary=primary_color_rgb, accent=accent_color_rgb, outline=outline, width=width, scale=int(render_scale), parts=parts)
    elif object_type == "scooter":
        boxes = _draw_scooter(draw, box=box, primary=primary_color_rgb, accent=accent_color_rgb, outline=outline, width=width, scale=int(render_scale), parts=parts)
    elif object_type == "airplane":
        boxes = _draw_airplane(draw, box=box, primary=primary_color_rgb, accent=accent_color_rgb, outline=outline, width=width, scale=int(render_scale), parts=parts)
    elif object_type == "boat":
        boxes = _draw_boat(draw, box=box, primary=primary_color_rgb, accent=accent_color_rgb, outline=outline, width=width, scale=int(render_scale), parts=parts)
    elif object_type == "sailboat":
        boxes = _draw_sailboat(draw, box=box, primary=primary_color_rgb, accent=accent_color_rgb, outline=outline, width=width, scale=int(render_scale), parts=parts)
    elif family == "person":
        gender = normalize_person_gender(gender_id)
        person_variant_id = normalize_object_variant_id("person", object_variant_id) if object_variant_id is not None else "adult"
        boxes = _draw_person_like(draw, object_id=str(object_id), box=box, object_type=str(object_type), primary=primary_color_rgb, accent=accent_color_rgb, outline=outline, width=width, scale=int(render_scale), style_id=str(style_id), parts=parts, gender_id=gender, person_variant_id=person_variant_id)
    elif family == "plant":
        boxes = _draw_plant(draw, box=box, object_type=str(object_type), primary=primary_color_rgb, accent=accent_color_rgb, outline=outline, width=width, scale=int(render_scale), parts=parts, object_variant_id=object_variant_id)
    else:
        boxes = _draw_object_item(draw, box=box, object_type=str(object_type), primary=primary_color_rgb, accent=accent_color_rgb, outline=outline, width=width, scale=int(render_scale), parts=parts)
    object_bbox = _expand(_union([*boxes, *(part.bbox_xyxy for part in parts.parts)]), 2.0)
    attributes = {
        "object_type": str(object_type),
        "family": family,
        "style_id": str(style_id),
    }
    if family == "person":
        attributes["gender_id"] = normalize_person_gender(gender_id)
    if object_variant_id is not None and object_type in {"person", "pedestrian_with_bag", "tree"}:
        variant_object_type = "person" if family == "person" else str(object_type)
        attributes.update(variant_visual_metadata(variant_object_type, object_variant_id, RENDERER_STYLE_VECTOR))
    return IllustrationObject(
        object_id=str(object_id),
        object_type=str(object_type),
        family=family,
        bbox_xyxy=tuple(float(v) for v in object_bbox),
        primary_color_rgb=tuple(int(v) for v in primary_color_rgb),
        accent_color_rgb=tuple(int(v) for v in accent_color_rgb),
        style_id=str(style_id),
        parts=tuple(parts.parts),
        attributes=attributes,
    )


def serialize_object(rendered: IllustrationObject) -> Dict[str, Any]:
    """Serialize one rendered object for trace payloads."""

    parts = [
        {
            "part_id": part.part_id,
            "part_kind": part.part_kind,
            "bbox": [round(float(v), 3) for v in part.bbox_xyxy],
            "attributes": dict(part.attributes),
        }
        for part in rendered.parts
    ]
    visual_attributes = {
        "primary_color_rgb": [int(v) for v in rendered.primary_color_rgb],
        "accent_color_rgb": [int(v) for v in rendered.accent_color_rgb],
        "style_id": str(rendered.style_id),
    }
    if isinstance(rendered.attributes, Mapping) and "gender_id" in rendered.attributes:
        visual_attributes["gender_id"] = str(rendered.attributes["gender_id"])
    variant_visual_keys = (
        "accessory",
        "crown_shape",
        "fruit_visible",
        "object_variant_id",
        "object_variant_public_name",
        "object_variant_queryable",
        "person_variant_id",
        "renderer_style",
        "renderer_variant_id",
        "scale",
        "tree_style",
    )
    if isinstance(rendered.attributes, Mapping):
        for key in variant_visual_keys:
            if key in rendered.attributes:
                visual_attributes[key] = rendered.attributes[key]
    role = str(rendered.attributes.get("role", "distractor")) if isinstance(rendered.attributes, Mapping) else "distractor"
    semantic_attributes = dict(rendered.attributes)
    semantic_attributes.pop("gender_id", None)
    for key in variant_visual_keys:
        semantic_attributes.pop(key, None)
    object_record = ObjectRecord(
        object_id=str(rendered.object_id),
        object_type=str(rendered.object_type),
        public_name=display_name_for_object_type(str(rendered.object_type)),
        family=str(rendered.family),
        bbox_xyxy=tuple(float(v) for v in rendered.bbox_xyxy),
        semantic_attributes=semantic_attributes,
        visual_attributes=visual_attributes,
        role=role,
        source_entity_type="illustration_object",
        parts=tuple(parts),
    ).as_dict()
    return {
        "object_id": rendered.object_id,
        "object_type": rendered.object_type,
        "family": rendered.family,
        "bbox": [round(float(v), 3) for v in rendered.bbox_xyxy],
        "primary_color_rgb": [int(v) for v in rendered.primary_color_rgb],
        "accent_color_rgb": [int(v) for v in rendered.accent_color_rgb],
        "style_id": rendered.style_id,
        "parts": parts,
        "attributes": dict(rendered.attributes),
        "object_record": object_record,
    }


__all__ = [
    "ACCENT_PALETTE",
    "BBox",
    "IllustrationObject",
    "IllustrationPart",
    "OBJECT_TEMPLATES",
    "PART_PLURALS",
    "RGB",
    "STYLE_IDS",
    "aspect_ratio_for_object",
    "choose_object_colors",
    "display_name_for_object_type",
    "draw_illustration_object",
    "family_for_object",
    "object_types",
    "object_types_with_part",
    "object_types_without_part",
    "part_count_for_object",
    "serialize_object",
    "supported_part_kinds",
]
