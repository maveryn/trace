"""Shared rendering helpers for geometry measurement scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from ...shared.color_distance import color_distance
from ...shared.text_legibility import (
    READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
    READ_REQUIRED_TEXT_MIN_LAB_DISTANCE,
    contrast_ratio,
    draw_text_traced,
)
from ...shared.text_rendering import symbol_safe_font_for_text

from .vector2d import add, mid, mul, perp, sub, unit

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


def round1(value: float) -> float:
    """Round one measurement value to one decimal with stable near-integer handling."""

    return round(float(value) + 1e-9, 1)


def fmt_measure(value: float) -> str:
    """Format one measurement value without a trailing decimal when integral."""

    rounded = round(float(value))
    if abs(float(value) - float(rounded)) <= 1e-9:
        return str(int(rounded))
    return f"{float(value):.1f}"


def bbox_to_list(bbox: Sequence[float]) -> list[float]:
    """Return a rounded bbox list for trace payloads."""

    return [round(float(value), 3) for value in bbox]


def clamp_bbox(bbox: Sequence[float], *, width: int, height: int) -> BBox:
    """Clamp an xyxy bbox to a canvas while normalizing inverted coordinates."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    return (
        round(max(0.0, min(float(width), min(x0, x1))), 3),
        round(max(0.0, min(float(height), min(y0, y1))), 3),
        round(max(0.0, min(float(width), max(x0, x1))), 3),
        round(max(0.0, min(float(height), max(y0, y1))), 3),
    )


def pad_bbox(bbox: Sequence[float], pad: float, *, width: int, height: int) -> BBox:
    """Pad and clamp an xyxy bbox."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    return clamp_bbox((x0 - pad, y0 - pad, x1 + pad, y1 + pad), width=width, height=height)


def bbox_from_points(points: Sequence[Point], *, width: int, height: int, pad: float = 0.0) -> BBox:
    """Return a padded/clamped bbox covering point coordinates."""

    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return pad_bbox((min(xs), min(ys), max(xs), max(ys)), pad, width=width, height=height)


def bbox_union_from_bboxes(
    bboxes: Sequence[Sequence[float]],
    *,
    width: int,
    height: int,
    pad: float = 0.0,
) -> BBox:
    """Return a padded/clamped bbox covering already-projected bboxes."""

    x0 = min(float(bbox[0]) for bbox in bboxes)
    y0 = min(float(bbox[1]) for bbox in bboxes)
    x1 = max(float(bbox[2]) for bbox in bboxes)
    y1 = max(float(bbox[3]) for bbox in bboxes)
    return pad_bbox((x0, y0, x1, y1), float(pad), width=int(width), height=int(height))


def _coerce_color(value: Any, fallback: Color) -> Color:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        try:
            return tuple(max(0, min(255, int(channel))) for channel in value[:3])  # type: ignore[return-value]
        except Exception:
            return fallback
    return fallback


def label_backing_fill(ctx: Any) -> Color:
    """Return a neutral fill used behind required geometry readout text."""

    for attr in ("label_backing_color", "panel_fill", "panel_alt_fill", "face_top", "face_front"):
        if hasattr(ctx, attr):
            return _coerce_color(getattr(ctx, attr), (255, 255, 255))
    return (255, 255, 255)


def readout_text_fill(ctx: Any, requested: Sequence[int] | None = None) -> Color:
    """Choose readout ink that remains high-contrast on the label backing."""

    backing_rgb = label_backing_fill(ctx)
    fallback = max(
        ((10, 14, 22), (245, 247, 250)),
        key=lambda candidate: float(contrast_ratio(candidate, backing_rgb)),
    )
    if requested is None:
        return fallback
    requested_rgb = _coerce_color(requested, fallback)
    if float(contrast_ratio(requested_rgb, backing_rgb)) >= float(READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO):
        return requested_rgb
    return fallback


def readout_text_metadata(ctx: Any, fill: Sequence[int]) -> dict[str, Any]:
    """Return contrast metadata for backed required geometry readout text."""

    fill_rgb = _coerce_color(fill, (10, 14, 22))
    surface_rgb = label_backing_fill(ctx)
    min_contrast = float(contrast_ratio(fill_rgb, surface_rgb))
    min_lab = float(color_distance(fill_rgb, surface_rgb, distance_space="lab"))
    return {
        "surface_rgbs": [list(surface_rgb)],
        "surface_sample_method": "rendered_label_backplate",
        "min_contrast_ratio": round(min_contrast, 3),
        "min_lab_distance": round(min_lab, 3),
        "min_contrast_required": round(float(READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO), 3),
        "min_lab_distance_required": round(float(READ_REQUIRED_TEXT_MIN_LAB_DISTANCE), 3),
        "passes": bool(
            min_contrast >= float(READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO)
            and min_lab >= float(READ_REQUIRED_TEXT_MIN_LAB_DISTANCE)
        ),
    }


def draw_label_backplate(
    ctx: Any,
    bbox: Sequence[float],
    *,
    pad: float = 10.0,
    fill: Sequence[int] | None = None,
    radius: int = 4,
) -> BBox:
    """Draw a small non-semantic backing behind one measured text bbox."""

    backing_bbox = pad_bbox(bbox, float(pad), width=int(ctx.width), height=int(ctx.height))
    backing_fill = _coerce_color(fill, label_backing_fill(ctx)) if fill is not None else label_backing_fill(ctx)
    try:
        ctx.draw.rounded_rectangle(backing_bbox, radius=max(0, int(radius)), fill=backing_fill)
    except Exception:
        ctx.draw.rectangle(backing_bbox, fill=backing_fill)
    return backing_bbox


def draw_label(ctx: Any, text: str, center: Point, *, small: bool = False) -> BBox:
    """Draw centered measurement text using the task render context contract."""

    font = symbol_safe_font_for_text(str(text), ctx.small_font if bool(small) else ctx.font)
    stroke_width = max(0, int(getattr(ctx, "label_stroke_width", 1)))
    bbox = ctx.draw.textbbox((0, 0), str(text), font=font, stroke_width=stroke_width)
    text_w = float(bbox[2] - bbox[0])
    text_h = float(bbox[3] - bbox[1])
    left = float(center[0]) - (text_w / 2.0)
    top = float(center[1]) - (text_h / 2.0)
    draw_text_traced(ctx.draw,
        (left, top),
        str(text),
        font=font,
        fill=ctx.label_color,
        stroke_width=stroke_width,
        stroke_fill=ctx.label_stroke_color,
     role="readout", required=False,)
    return pad_bbox((left, top, left + text_w, top + text_h), 4.0, width=ctx.width, height=ctx.height)


def draw_readout_centered(
    ctx: Any,
    text: str,
    center: Point,
    *,
    small: bool = True,
    required: bool = True,
    backed: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
) -> BBox:
    """Draw centered readout text, with label backing only when explicitly requested."""

    font = symbol_safe_font_for_text(str(text), ctx.small_font if bool(small) else ctx.font)
    stroke_width = max(0, int(getattr(ctx, "label_stroke_width", 0)))
    bbox = ctx.draw.textbbox(
        (float(center[0]), float(center[1])),
        str(text),
        anchor="mm",
        font=font,
        stroke_width=stroke_width,
    )
    if bool(backed):
        draw_label_backplate(ctx, bbox)
        fill = readout_text_fill(ctx, ctx.label_color)
        stroke_fill = readout_text_fill(ctx, ctx.label_stroke_color)
        extra_metadata = readout_text_metadata(ctx, fill)
    else:
        fill = _coerce_color(getattr(ctx, "label_color", (10, 14, 22)), (10, 14, 22))
        stroke_fill = _coerce_color(getattr(ctx, "label_stroke_color", (255, 255, 255)), (255, 255, 255))
        extra_metadata = dict(extra_metadata) if extra_metadata is not None else None
    draw_text_traced(
        ctx.draw,
        (float(center[0]), float(center[1])),
        str(text),
        anchor="mm",
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
        role="readout",
        required=bool(required),
        extra_metadata=extra_metadata,
    )
    return pad_bbox(bbox, 4.0, width=int(ctx.width), height=int(ctx.height))


def draw_dimension_line(
    ctx: Any,
    start: Point,
    end: Point,
    label: str,
    *,
    label_offset: Point,
    color: Color | None = None,
    tick_px: float | None = None,
    backed: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
) -> BBox:
    """Draw a measured segment with endpoint ticks and a plain label by default."""

    draw_color = color or ctx.secondary_color
    line_width = int(getattr(ctx, "line_width", 2))
    ctx.draw.line([start, end], fill=draw_color, width=max(2, line_width - 1))
    direction = unit(sub(end, start))
    normal = perp(direction)
    tick = float(tick_px) if tick_px is not None else max(6.0, float(line_width) * 2.5)
    for point in (start, end):
        ctx.draw.line(
            [add(point, mul(normal, -tick)), add(point, mul(normal, tick))],
            fill=draw_color,
            width=max(1, line_width - 2),
        )
    return draw_readout_centered(
        ctx,
        label,
        add(mid(start, end), label_offset),
        small=True,
        backed=bool(backed),
        extra_metadata=extra_metadata,
    )


def draw_right_angle_marker(
    ctx: Any,
    corner: Point,
    *,
    arm_a: Point,
    arm_b: Point,
    side_px: float | None = None,
    color: Color | None = None,
    width: int | None = None,
) -> BBox:
    """Draw a right-angle marker from two arm directions and return its bbox."""

    line_width = int(getattr(ctx, "line_width", 2))
    side = float(side_px) if side_px is not None else float(max(13, line_width * 4))
    u = unit(arm_a)
    v = unit(arm_b)
    p0 = add(corner, mul(u, side))
    p1 = add(p0, mul(v, side))
    p2 = add(corner, mul(v, side))
    draw_color = color or ctx.line_color
    marker_width = max(1, int(width) if width is not None else line_width - 2)
    ctx.draw.line([p0, p1, p2], fill=draw_color, width=marker_width)
    return bbox_from_points((p0, p1, p2, corner), width=int(ctx.width), height=int(ctx.height), pad=4.0)


def assert_bboxes_inside(
    bboxes: Sequence[Sequence[float]],
    *,
    width: int,
    height: int,
    margin: float = 3.0,
    error_message: str = "geometry measurement bbox too close to canvas edge",
) -> None:
    """Raise when any bbox is too close to or outside a canvas edge."""

    edge_margin = float(margin)
    for bbox in bboxes:
        x0, y0, x1, y1 = [float(value) for value in bbox]
        if (
            x0 <= edge_margin
            or y0 <= edge_margin
            or x1 >= float(width) - edge_margin
            or y1 >= float(height) - edge_margin
        ):
            raise ValueError(str(error_message))


__all__ = [
    "BBox",
    "Color",
    "Point",
    "assert_bboxes_inside",
    "bbox_from_points",
    "bbox_to_list",
    "bbox_union_from_bboxes",
    "clamp_bbox",
    "draw_dimension_line",
    "draw_label_backplate",
    "draw_label",
    "draw_readout_centered",
    "draw_right_angle_marker",
    "fmt_measure",
    "label_backing_fill",
    "pad_bbox",
    "readout_text_metadata",
    "readout_text_fill",
    "round1",
]
