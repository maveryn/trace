"""Shared vector person drawing for illustration renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from PIL import ImageDraw

from .object_schema import BBox
from .object_variants import normalize_object_variant_id
from .person_rendering import draw_person_hair_back, draw_person_hair_front, draw_person_skirt, normalize_person_gender
from .render_geometry import scale_bbox as _scale_bbox, scale_points as _scale_points
from .style_registry import style_object_colors, style_outline_params


RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class VectorPersonRenderResult:
    """Rendered person geometry emitted by the shared vector renderer."""

    bbox_xyxy: BBox
    parts: Tuple[Mapping[str, Any], ...]
    visual_attributes: Mapping[str, Any]
    support_items: Tuple[Mapping[str, Any], ...] = ()


def _rgb(value: Any, fallback: RGB) -> RGB:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
        return tuple(int(max(0, min(255, int(channel)))) for channel in value)  # type: ignore[return-value]
    return tuple(int(channel) for channel in fallback)  # type: ignore[return-value]


def _rel(box: BBox, x0: float, y0: float, x1: float, y1: float) -> BBox:
    bx0, by0, bx1, by1 = [float(value) for value in box]
    width = bx1 - bx0
    height = by1 - by0
    return (bx0 + x0 * width, by0 + y0 * height, bx0 + x1 * width, by0 + y1 * height)


def _expand(box: BBox, pad: float) -> BBox:
    return (float(box[0]) - pad, float(box[1]) - pad, float(box[2]) + pad, float(box[3]) + pad)


def _union(boxes: Sequence[BBox]) -> BBox:
    return (
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    )


def _rect(
    draw: ImageDraw.ImageDraw,
    box: BBox,
    *,
    fill: RGB,
    outline: RGB | None,
    width: int,
    scale: int,
    radius: float = 0.0,
) -> None:
    scaled = _scale_bbox(box, scale)
    line_width = max(1, int(width) * int(scale))
    if float(radius) > 0.0:
        draw.rounded_rectangle(
            scaled,
            radius=max(1, int(round(float(radius) * int(scale)))),
            fill=tuple(fill),
            outline=tuple(outline) if outline else None,
            width=line_width,
        )
        return
    draw.rectangle(scaled, fill=tuple(fill), outline=tuple(outline) if outline else None, width=line_width)


def _ellipse(
    draw: ImageDraw.ImageDraw,
    box: BBox,
    *,
    fill: RGB,
    outline: RGB | None,
    width: int,
    scale: int,
) -> None:
    draw.ellipse(_scale_bbox(box, scale), fill=tuple(fill), outline=tuple(outline) if outline else None, width=max(1, int(width) * int(scale)))


def _poly(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Tuple[float, float]],
    *,
    fill: RGB,
    outline: RGB | None,
    width: int,
    scale: int,
) -> None:
    draw.polygon(_scale_points(points, scale), fill=tuple(fill))
    if outline:
        draw.line(_scale_points([*points, points[0]], scale), fill=tuple(outline), width=max(1, int(width) * int(scale)), joint="curve")


def _line(draw: ImageDraw.ImageDraw, points: Sequence[Tuple[float, float]], *, fill: RGB, width: int, scale: int) -> None:
    draw.line(_scale_points(points, scale), fill=tuple(fill), width=max(1, int(width) * int(scale)), joint="curve")


def _shadow(draw: ImageDraw.ImageDraw, box: BBox, *, scale: int) -> None:
    x0, y0, x1, y1 = [float(value) for value in box]
    height = y1 - y0
    _ellipse(
        draw,
        (x0 + 0.08 * (x1 - x0), y1 - 0.08 * height, x1 - 0.05 * (x1 - x0), y1 + 0.03 * height),
        fill=(132, 128, 119),
        outline=None,
        width=1,
        scale=scale,
    )


def _part(parts: list[dict[str, Any]], counts: dict[str, int], object_id: str, kind: str, box: BBox, **attrs: Any) -> None:
    index = int(counts.get(str(kind), 0))
    counts[str(kind)] = index + 1
    parts.append(
        {
            "part_id": f"{object_id}_{kind}_{index}",
            "part_kind": str(kind),
            "bbox": [round(float(value), 3) for value in box],
            "attributes": dict(attrs),
        }
    )


def _draw_accessory(
    draw: ImageDraw.ImageDraw,
    *,
    box: BBox,
    variant: str,
    accent: RGB,
    outline: RGB | None,
    line_width: int,
    scale: int,
) -> list[BBox]:
    if variant == "farmer":
        brim = _rel(box, 0.20, 0.04, 0.80, 0.11)
        crown = _rel(box, 0.34, 0.00, 0.66, 0.09)
        _rect(draw, crown, fill=(218, 174, 84), outline=outline, width=max(1, line_width - 1), scale=scale, radius=4)
        _rect(draw, brim, fill=(226, 190, 105), outline=outline, width=max(1, line_width - 1), scale=scale, radius=4)
        return [brim, crown]
    if variant == "worker":
        helmet = _rel(box, 0.26, 0.00, 0.74, 0.13)
        _rect(draw, helmet, fill=(237, 195, 63), outline=outline, width=max(1, line_width - 1), scale=scale, radius=5)
        _line(draw, [(0.5 * (helmet[0] + helmet[2]), helmet[1]), (0.5 * (helmet[0] + helmet[2]), helmet[3])], fill=(126, 91, 34), width=max(1, line_width - 1), scale=scale)
        return [helmet]
    if variant == "vendor":
        cap = _rel(box, 0.28, 0.01, 0.76, 0.12)
        _rect(draw, cap, fill=accent, outline=outline, width=max(1, line_width - 1), scale=scale, radius=5)
        return [cap]
    if variant == "soldier":
        helmet = _rel(box, 0.25, 0.00, 0.75, 0.13)
        brim = _rel(box, 0.18, 0.10, 0.82, 0.16)
        _rect(draw, helmet, fill=(77, 96, 58), outline=outline, width=max(1, line_width - 1), scale=scale, radius=6)
        _rect(draw, brim, fill=(57, 73, 46), outline=outline, width=max(1, line_width - 1), scale=scale, radius=3)
        _line(
            draw,
            [
                (helmet[0] + 0.30 * (helmet[2] - helmet[0]), helmet[1] + 2.0),
                (helmet[2] - 2.0, helmet[3] - 1.0),
            ],
            fill=(111, 128, 76),
            width=max(1, line_width - 1),
            scale=scale,
        )
        return [helmet, brim]
    return []


def _draw_tool(
    draw: ImageDraw.ImageDraw,
    *,
    box: BBox,
    tool_type: str,
    outline: RGB | None,
    scale: int,
) -> list[BBox]:
    x0, y0, x1, y1 = [float(value) for value in box]
    width = x1 - x0
    height = y1 - y0
    if str(tool_type) == "shovel":
        handle = (x0 + width * 0.84, y0 + height * 0.55, x0 + width * 1.03, y1 - height * 0.04)
        blade = (x0 + width * 0.96, y1 - height * 0.09, x0 + width * 1.10, y1 + height * 0.02)
        _line(draw, [(handle[0], handle[1]), (handle[2], handle[3])], fill=(82, 68, 49), width=3, scale=scale)
        _ellipse(draw, blade, fill=(101, 105, 108), outline=outline, width=1, scale=scale)
        return [_expand(handle, 3.0), blade]
    if str(tool_type) == "hammer":
        handle = (x0 + width * 0.82, y0 + height * 0.58, x0 + width * 0.98, y0 + height * 0.76)
        head = (x0 + width * 0.91, y0 + height * 0.51, x0 + width * 1.08, y0 + height * 0.59)
        _line(draw, [(handle[0], handle[1]), (handle[2], handle[3])], fill=(86, 65, 44), width=4, scale=scale)
        _rect(draw, head, fill=(107, 112, 118), outline=outline, width=1, scale=scale, radius=2)
        return [_expand(handle, 3.0), head]
    wrench = (x0 + width * 0.80, y0 + height * 0.58, x0 + width * 1.10, y0 + height * 0.78)
    _line(draw, [(wrench[0], wrench[1]), (x0 + width * 1.04, y0 + height * 0.70)], fill=(97, 101, 106), width=4, scale=scale)
    _ellipse(draw, (x0 + width * 0.98, y0 + height * 0.66, wrench[2], wrench[3]), fill=(97, 101, 106), outline=outline, width=1, scale=scale)
    return [wrench]


def draw_vector_person(
    draw: ImageDraw.ImageDraw,
    *,
    object_id: str,
    object_type: str,
    bbox_xyxy: BBox,
    renderer_id: str,
    renderer_variant_id: str = "",
    style_id: str,
    render_scale: int,
    primary_color_rgb: RGB,
    accent_color_rgb: RGB,
    skin_color_rgb: RGB | None = None,
    gender_id: str | None = None,
    person_variant_id: str = "adult",
    hard_hat_color_rgb: RGB | None = None,
    vest_color_rgb: RGB | None = None,
    tool_type: str | None = None,
    apply_style_colors: bool = True,
    draw_shadow: bool = True,
) -> VectorPersonRenderResult:
    """Draw a vector person-like object and return trace-ready geometry."""

    scale = max(1, int(render_scale))
    outline_raw, line_width_raw, shadow = style_outline_params(str(style_id))
    outline = tuple(outline_raw or (66, 73, 84))  # type: ignore[arg-type]
    line_width = int(line_width_raw)
    if bool(apply_style_colors):
        primary, accent = style_object_colors(str(style_id), primary_color_rgb, accent_color_rgb)
    else:
        primary = _rgb(primary_color_rgb, (72, 115, 166))
        accent = _rgb(accent_color_rgb, (245, 190, 88))
    skin = _rgb(skin_color_rgb, (199, 139, 96))
    gender = normalize_person_gender(gender_id)
    renderer = str(renderer_id)
    variant = normalize_object_variant_id("person", str(person_variant_id or "adult"))
    if variant == "soldier":
        primary = (78, 103, 61)
        accent = (106, 124, 76)
    box = tuple(float(value) for value in bbox_xyxy)  # type: ignore[assignment]
    if shadow and bool(draw_shadow):
        _shadow(draw, box, scale=scale)

    parts: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    support_items: list[dict[str, Any]] = []
    boxes: list[BBox] = []
    arm_trace_boxes: list[BBox] = []
    leg_trace_boxes: list[BBox] = []
    head: BBox = box

    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0

    pose = str(renderer_variant_id or "standing")
    if renderer == "park_person" and pose == "sitting":
        head = _rel(box, 0.31, 0.04, 0.69, 0.32)
        body = _rel(box, 0.28, 0.32, 0.76, 0.56 if gender == "female" else 0.62)
        leg_boxes = (
            _expand((x0 + 0.25 * width, y0 + 0.62 * height, x0 + 0.55 * width, y0 + 0.90 * height), 3.0),
            _expand((x0 + 0.60 * width, y0 + 0.62 * height, x0 + 0.96 * width, y0 + 0.86 * height), 3.0),
        )
        arm_boxes = (_rel(box, 0.14, 0.40, 0.31, 0.64), _rel(box, 0.72, 0.40, 0.92, 0.62))
        draw_person_hair_back(draw, head_bbox=head, gender_id=gender, scale=scale, outline=outline)
        _ellipse(draw, head, fill=skin, outline=outline, width=line_width, scale=scale)
        draw_person_hair_front(draw, head_bbox=head, gender_id=gender, scale=scale, outline=outline)
        _rect(draw, body, fill=primary, outline=outline, width=line_width, scale=scale, radius=8)
        for arm in arm_boxes:
            _line(draw, [(arm[0], arm[1]), (arm[2], arm[3])], fill=primary, width=max(3, line_width + 1), scale=scale)
        leg_color = (65, 74, 91)
        _line(draw, [(x0 + 0.40 * width, y0 + 0.62 * height), (x0 + 0.25 * width, y0 + 0.87 * height), (x0 + 0.55 * width, y0 + 0.90 * height)], fill=leg_color, width=max(5, line_width + 3), scale=scale)
        _line(draw, [(x0 + 0.60 * width, y0 + 0.62 * height), (x0 + 0.75 * width, y0 + 0.86 * height), (x0 + 0.96 * width, y0 + 0.84 * height)], fill=leg_color, width=max(5, line_width + 3), scale=scale)
        if gender == "female":
            body = _union([body, draw_person_skirt(draw, torso_bbox=body, bottom_y=y0 + 0.70 * height, fill=primary, outline=outline, scale=scale, width=line_width)])
        boxes.extend([head, body, *arm_boxes, *leg_boxes])
        arm_trace_boxes = [tuple(float(value) for value in arm) for arm in arm_boxes]  # type: ignore[list-item]
        leg_trace_boxes = [tuple(float(value) for value in leg) for leg in leg_boxes]  # type: ignore[list-item]
    elif renderer == "construction_worker":
        hat = _rgb(hard_hat_color_rgb, (238, 194, 64))
        vest = _rgb(vest_color_rgb, (232, 126, 54))
        shirt = (84, 118, 154)
        pants = (68, 83, 104)
        head = _rel(box, 0.33, 0.13, 0.67, 0.43)
        hair_head = _rel(box, 0.37, 0.19, 0.63, 0.43)
        helmet = _rel(box, 0.25, 0.09, 0.75, 0.25)
        brim = _rel(box, 0.20, 0.22, 0.80, 0.28)
        torso = _rel(box, 0.24, 0.42, 0.76, 0.72)
        draw_person_hair_back(draw, head_bbox=hair_head, gender_id=gender, scale=scale, outline=outline)
        _ellipse(draw, head, fill=skin, outline=outline, width=line_width, scale=scale)
        draw_person_hair_front(draw, head_bbox=hair_head, gender_id=gender, scale=scale, outline=outline)
        _rect(draw, helmet, fill=hat, outline=outline, width=line_width, scale=scale, radius=8)
        _rect(draw, brim, fill=hat, outline=outline, width=line_width, scale=scale, radius=4)
        _rect(draw, torso, fill=shirt, outline=outline, width=line_width, scale=scale, radius=7)
        _poly(draw, [(x0 + width * 0.26, y0 + height * 0.43), (x0 + width * 0.43, y0 + height * 0.43), (x0 + width * 0.56, y0 + height * 0.72), (x0 + width * 0.38, y0 + height * 0.72)], fill=vest, outline=None, width=1, scale=scale)
        _poly(draw, [(x0 + width * 0.57, y0 + height * 0.43), (x0 + width * 0.74, y0 + height * 0.43), (x0 + width * 0.62, y0 + height * 0.72), (x0 + width * 0.46, y0 + height * 0.72)], fill=vest, outline=None, width=1, scale=scale)
        _line(draw, [(x0 + width * 0.23, y0 + height * 0.48), (x0 + width * 0.10, y0 + height * 0.66)], fill=skin, width=5, scale=scale)
        _line(draw, [(x0 + width * 0.77, y0 + height * 0.48), (x0 + width * 0.90, y0 + height * 0.66)], fill=skin, width=5, scale=scale)
        _line(draw, [(x0 + width * 0.41, y0 + height * 0.72), (x0 + width * 0.34, y1 - height * 0.06)], fill=pants, width=7, scale=scale)
        _line(draw, [(x0 + width * 0.59, y0 + height * 0.72), (x0 + width * 0.68, y1 - height * 0.06)], fill=pants, width=7, scale=scale)
        tool_boxes = _draw_tool(draw, box=box, tool_type=str(tool_type), outline=outline, scale=scale) if tool_type else []
        boxes.extend([head, helmet, brim, torso, *tool_boxes])
        arm_trace_boxes = [_expand((x0 + width * 0.10, y0 + height * 0.48, x0 + width * 0.23, y0 + height * 0.66), 3.0), _expand((x0 + width * 0.77, y0 + height * 0.48, x0 + width * 0.90, y0 + height * 0.66), 3.0)]
        leg_trace_boxes = [_expand((x0 + width * 0.34, y0 + height * 0.72, x0 + width * 0.41, y1 - height * 0.06), 4.0), _expand((x0 + width * 0.59, y0 + height * 0.72, x0 + width * 0.68, y1 - height * 0.06), 4.0)]
    else:
        head = _rel(box, 0.31, 0.04, 0.69, 0.28)
        torso = _rel(box, 0.28, 0.29, 0.74, 0.53 if gender == "female" else 0.60)
        draw_person_hair_back(draw, head_bbox=head, gender_id=gender, scale=scale, outline=outline)
        _ellipse(draw, head, fill=skin, outline=outline, width=line_width, scale=scale)
        draw_person_hair_front(draw, head_bbox=head, gender_id=gender, scale=scale, outline=outline)
        accessories = _draw_accessory(draw, box=box, variant=variant, accent=accent, outline=outline, line_width=line_width, scale=scale)
        _rect(draw, torso, fill=primary, outline=outline, width=line_width, scale=scale, radius=8)
        body_boxes = [torso]
        if variant == "soldier":
            x0_t, y0_t, x1_t, y1_t = torso
            _line(
                draw,
                [
                    (x0_t + 0.18 * (x1_t - x0_t), y0_t + 0.34 * (y1_t - y0_t)),
                    (x0_t + 0.44 * (x1_t - x0_t), y0_t + 0.48 * (y1_t - y0_t)),
                ],
                fill=(42, 59, 39),
                width=max(1, line_width - 1),
                scale=scale,
            )
            _line(
                draw,
                [
                    (x0_t + 0.50 * (x1_t - x0_t), y0_t + 0.25 * (y1_t - y0_t)),
                    (x0_t + 0.78 * (x1_t - x0_t), y0_t + 0.42 * (y1_t - y0_t)),
                ],
                fill=(118, 136, 82),
                width=max(1, line_width - 1),
                scale=scale,
            )
        if gender == "female":
            skirt_bottom = y0 + (0.70 if pose == "seated" else 0.73) * height
            body_boxes.append(draw_person_skirt(draw, torso_bbox=torso, bottom_y=skirt_bottom, fill=primary, outline=outline, scale=scale, width=line_width))
        if variant in {"farmer", "worker", "vendor"}:
            apron = _rel(box, 0.36, 0.36, 0.64, 0.61)
            _rect(draw, apron, fill=(232, 211, 130) if variant == "farmer" else accent, outline=None, width=1, scale=scale, radius=3)
            body_boxes.append(apron)
        leg_color = (63, 71, 88)
        arm_boxes = (_rel(box, 0.14, 0.34, 0.31, 0.58), _rel(box, 0.69, 0.34, 0.88, 0.58))
        if renderer == "park_person" and pose == "playing_ball":
            arm_boxes = (_rel(box, 0.02, 0.26, 0.31, 0.42), _rel(box, 0.71, 0.28, 1.03, 0.42))
        for arm in arm_boxes:
            _line(draw, [(arm[0], arm[1]), (arm[2], arm[3])], fill=primary, width=max(3, line_width + 2), scale=scale)
        arm_trace_boxes = [tuple(float(value) for value in _expand(arm, 3.0)) for arm in arm_boxes]  # type: ignore[list-item]
        if pose == "walking":
            leg_lines = (((0.42, 0.58), (0.18, 0.96)), ((0.58, 0.58), (0.82, 0.94)))
        elif pose == "seated":
            leg_lines = (((0.38, 0.60), (0.20, 0.82), (0.48, 0.84)), ((0.60, 0.60), (0.78, 0.82), (0.98, 0.80)))
        elif pose == "playing_ball":
            leg_lines = (((0.38, 0.58), (0.28, 0.92)), ((0.60, 0.58), (0.82, 0.80)))
        else:
            leg_lines = (((0.42, 0.58), (0.36, 0.96)), ((0.58, 0.58), (0.66, 0.96)))
        leg_boxes: list[BBox] = []
        for line in leg_lines:
            points = [(x0 + px * width, y0 + py * height) for px, py in line]
            _line(draw, points, fill=leg_color, width=max(4, line_width + 3), scale=scale)
            leg_boxes.append(_expand(_union([(min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points))]), 4.0))
        leg_trace_boxes = list(leg_boxes)
        if renderer == "park_person" and pose == "playing_ball":
            ball = _rel(box, 0.36, 0.00, 0.64, 0.09)
            _ellipse(draw, ball, fill=accent, outline=outline, width=1, scale=scale)
            boxes.append(ball)
        if str(object_type) == "pedestrian_with_bag":
            bag = _rel(box, 0.70, 0.42, 0.92, 0.66)
            _rect(draw, bag, fill=accent, outline=outline, width=line_width, scale=scale, radius=5)
            boxes.append(bag)
            _part(parts, counts, object_id, "bag", bag)
        boxes.extend([head, *accessories, *body_boxes, *arm_boxes, *leg_boxes])

    _part(parts, counts, object_id, "head", head)
    for arm_box in arm_trace_boxes:
        _part(parts, counts, object_id, "arm", arm_box)
    for leg_box in leg_trace_boxes:
        _part(parts, counts, object_id, "leg", leg_box)
    object_bbox = _expand(_union(boxes or [box]), 2.0)
    visual_attrs = {
        "primary_color_rgb": [int(value) for value in primary],
        "accent_color_rgb": [int(value) for value in accent],
        "skin_color_rgb": [int(value) for value in skin],
        "gender_id": gender,
        "person_variant_id": variant,
        "renderer_id": renderer,
        "renderer_variant_id": str(renderer_variant_id),
    }
    return VectorPersonRenderResult(
        bbox_xyxy=tuple(round(float(value), 3) for value in object_bbox),  # type: ignore[return-value]
        parts=tuple(parts),
        visual_attributes=visual_attrs,
        support_items=tuple(support_items),
    )


__all__ = ["VectorPersonRenderResult", "draw_vector_person"]
