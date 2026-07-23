"""Render review-only overlays from public projected annotations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import io
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

_COLORS = (
    (229, 57, 53, 255),
    (30, 136, 229, 255),
    (0, 137, 123, 255),
    (251, 140, 0, 255),
    (142, 36, 170, 255),
)


def render_annotation_overlay(
    image_path: Path | str,
    annotation_gt: Mapping[str, Any],
) -> bytes:
    """Return a PNG with public bbox/point/segment annotations overlaid."""

    with Image.open(image_path) as opened:
        image = opened.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    annotation_type = str(annotation_gt.get("type", ""))
    value = annotation_gt.get("value")
    labeled = list(_annotation_items(annotation_type, value))
    for index, (label, geometry_kind, geometry) in enumerate(labeled):
        color = _COLORS[index % len(_COLORS)]
        if geometry_kind == "bbox":
            _draw_bbox(draw, geometry, color=color, label=label)
        elif geometry_kind == "point":
            _draw_point(draw, geometry, color=color, label=label)
        elif geometry_kind == "segment":
            _draw_segment(draw, geometry, color=color, label=label)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _annotation_items(annotation_type: str, value: Any):
    if annotation_type == "bbox":
        yield "answer", "bbox", value
    elif annotation_type in {"bbox_set", "bbox_sequence"} and _sequence(value):
        for index, bbox in enumerate(value):
            yield str(index + 1), "bbox", bbox
    elif annotation_type == "bbox_map" and isinstance(value, Mapping):
        for key, bbox in sorted(value.items(), key=lambda item: str(item[0])):
            yield str(key), "bbox", bbox
    elif annotation_type == "bbox_set_map" and isinstance(value, Mapping):
        for key, boxes in sorted(value.items(), key=lambda item: str(item[0])):
            if not _sequence(boxes):
                continue
            for index, bbox in enumerate(boxes):
                yield f"{key}:{index + 1}", "bbox", bbox
    elif annotation_type == "point":
        yield "answer", "point", value
    elif annotation_type in {"point_set", "point_sequence"} and _sequence(value):
        for index, point in enumerate(value):
            yield str(index + 1), "point", point
    elif annotation_type == "point_map" and isinstance(value, Mapping):
        for key, point in sorted(value.items(), key=lambda item: str(item[0])):
            yield str(key), "point", point
    elif annotation_type == "point_set_map" and isinstance(value, Mapping):
        for key, points in sorted(value.items(), key=lambda item: str(item[0])):
            if not _sequence(points):
                continue
            for index, point in enumerate(points):
                yield f"{key}:{index + 1}", "point", point
    elif annotation_type == "segment":
        yield "answer", "segment", value
    elif annotation_type == "segment_set" and _sequence(value):
        for index, segment in enumerate(value):
            yield str(index + 1), "segment", segment


def _draw_bbox(
    draw: ImageDraw.ImageDraw, value: Any, *, color: tuple[int, ...], label: str
) -> None:
    if not _numeric_sequence(value, 4):
        return
    x0, y0, x1, y1 = (float(item) for item in value)
    draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
    _draw_label(draw, (x0, y0), label, color)


def _draw_point(
    draw: ImageDraw.ImageDraw, value: Any, *, color: tuple[int, ...], label: str
) -> None:
    if not _numeric_sequence(value, 2):
        return
    x, y = (float(item) for item in value)
    radius = 7.0
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    _draw_label(draw, (x + radius, y - radius), label, color)


def _draw_segment(
    draw: ImageDraw.ImageDraw, value: Any, *, color: tuple[int, ...], label: str
) -> None:
    if not _sequence(value) or len(value) != 2:
        return
    if not _numeric_sequence(value[0], 2) or not _numeric_sequence(value[1], 2):
        return
    points = [(float(point[0]), float(point[1])) for point in value]
    draw.line(points, fill=color, width=5)
    _draw_label(draw, points[0], label, color)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    point: tuple[float, float],
    label: str,
    color: tuple[int, ...],
) -> None:
    text = str(label)[:48]
    if not text:
        return
    font = ImageFont.load_default()
    left, top = float(point[0]), max(0.0, float(point[1]) - 15.0)
    right = left + max(18.0, float(draw.textlength(text, font=font)) + 8.0)
    draw.rectangle((left, top, right, top + 15.0), fill=color)
    draw.text((left + 4.0, top + 2.0), text, fill=(255, 255, 255, 255), font=font)


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _numeric_sequence(value: Any, length: int) -> bool:
    if not _sequence(value) or len(value) != length:
        return False
    return all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    )


__all__ = ["render_annotation_overlay"]
