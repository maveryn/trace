"""Shared text legibility and non-semantic text color utilities."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from ...core.seed import spawn_rng
from .color_distance import color_distance

Color = Tuple[int, int, int]
BBox = Tuple[float, float, float, float]

READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO = 7.0
LARGE_TEXT_MIN_CONTRAST_RATIO = 4.5
CONTEXT_TEXT_MIN_CONTRAST_RATIO = 3.0
READ_REQUIRED_TEXT_MIN_LAB_DISTANCE = 38.0
TEXT_LEGIBILITY_POLICY_VERSION = "text_legibility_v1"
_ACTIVE_DRAWN_TEXT_RECORDS: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "trace_active_drawn_text_records",
    default=None,
)

READABLE_TEXT_INK_POOL: Tuple[Color, ...] = (
    (10, 14, 22),
    (24, 31, 44),
    (36, 45, 64),
    (22, 52, 70),
    (18, 70, 69),
    (28, 82, 56),
    (73, 53, 112),
    (103, 42, 82),
    (122, 45, 54),
    (118, 66, 28),
    (54, 54, 52),
    (236, 242, 252),
    (250, 252, 255),
    (245, 248, 241),
    (255, 247, 229),
    (245, 235, 255),
    (232, 250, 246),
    (255, 237, 236),
)

_STROKE_CANDIDATES: Tuple[Color, ...] = (
    (255, 255, 255),
    (250, 252, 255),
    (10, 14, 22),
    (28, 34, 44),
)


@dataclass(frozen=True)
class ReadableTextStyle:
    """Resolved non-semantic text styling for one visible text role."""

    role: str
    fill_rgb: Color
    stroke_rgb: Color
    surface_rgbs: Tuple[Color, ...]
    min_contrast_ratio: float
    min_lab_distance: float
    required: bool = True
    min_contrast_required: float = READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO
    min_lab_distance_required: float = READ_REQUIRED_TEXT_MIN_LAB_DISTANCE

    def metadata(self) -> dict[str, Any]:
        return {
            "role": str(self.role),
            "required": bool(self.required),
            "fill_rgb": list(self.fill_rgb),
            "stroke_rgb": list(self.stroke_rgb),
            "surface_rgbs": [list(color) for color in self.surface_rgbs],
            "min_contrast_ratio": round(float(self.min_contrast_ratio), 3),
            "min_lab_distance": round(float(self.min_lab_distance), 3),
            "min_contrast_required": round(float(self.min_contrast_required), 3),
            "min_lab_distance_required": round(float(self.min_lab_distance_required), 3),
            "passes": bool(
                float(self.min_contrast_ratio) >= float(self.min_contrast_required)
                and float(self.min_lab_distance) >= float(self.min_lab_distance_required)
            ),
        }


def normalize_rgb(value: Sequence[int]) -> Color:
    """Normalize one RGB-like sequence into a clamped integer triplet."""

    if len(value) < 3:
        raise ValueError("RGB values require at least three channels")
    return tuple(max(0, min(255, int(channel))) for channel in value[:3])  # type: ignore[return-value]


def relative_luminance(color: Sequence[int]) -> float:
    """Return WCAG relative luminance for one sRGB color."""

    rgb = normalize_rgb(color)

    def channel(value: int) -> float:
        normalized = float(value) / 255.0
        if normalized <= 0.03928:
            return normalized / 12.92
        return ((normalized + 0.055) / 1.055) ** 2.4

    return (
        (0.2126 * channel(int(rgb[0])))
        + (0.7152 * channel(int(rgb[1])))
        + (0.0722 * channel(int(rgb[2])))
    )


def contrast_ratio(color_a: Sequence[int], color_b: Sequence[int]) -> float:
    """Return WCAG contrast ratio between two sRGB colors."""

    lum_a = relative_luminance(color_a)
    lum_b = relative_luminance(color_b)
    lighter = max(float(lum_a), float(lum_b))
    darker = min(float(lum_a), float(lum_b))
    return (lighter + 0.05) / (darker + 0.05)


def text_legibility_metadata_for_surfaces(
    *,
    fill_rgb: Sequence[int],
    surface_rgbs: Sequence[Sequence[int]],
    required: bool = True,
    min_contrast_ratio: float = READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
    min_lab_distance: float = READ_REQUIRED_TEXT_MIN_LAB_DISTANCE,
) -> dict[str, Any]:
    """Return explicit text-legibility metadata for known solid text surfaces."""

    fill = normalize_rgb(fill_rgb)
    surfaces = _unique_colors(surface_rgbs)
    min_contrast = _min_contrast(fill, surfaces)
    min_lab = _min_lab_distance(fill, surfaces)
    return {
        "required": bool(required),
        "surface_rgbs": [list(color) for color in surfaces],
        "surface_sample_method": "declared_solid_surface",
        "min_contrast_ratio": round(float(min_contrast), 3),
        "min_lab_distance": round(float(min_lab), 3),
        "min_contrast_required": round(float(min_contrast_ratio), 3),
        "min_lab_distance_required": round(float(min_lab_distance), 3),
        "passes": bool(float(min_contrast) >= float(min_contrast_ratio) and float(min_lab) >= float(min_lab_distance)),
    }


def _unique_colors(colors: Iterable[Sequence[int]]) -> Tuple[Color, ...]:
    seen: set[Color] = set()
    resolved: list[Color] = []
    for color in colors:
        rgb = normalize_rgb(color)
        if rgb in seen:
            continue
        seen.add(rgb)
        resolved.append(rgb)
    return tuple(resolved)


def _min_contrast(color: Color, surfaces: Sequence[Color]) -> float:
    if not surfaces:
        return float("inf")
    return min(float(contrast_ratio(color, surface)) for surface in surfaces)


def _min_lab_distance(color: Color, surfaces: Sequence[Color]) -> float:
    if not surfaces:
        return float("inf")
    return min(float(color_distance(color, surface, distance_space="lab")) for surface in surfaces)


def _choose_text_stroke(fill_rgb: Color, surface_rgbs: Sequence[Color]) -> Color:
    """Choose a deterministic outline color that separates glyph fill and surface."""

    surfaces = tuple(normalize_rgb(surface) for surface in surface_rgbs)

    def score(candidate: Color) -> tuple[float, float, float]:
        fill_ratio = contrast_ratio(candidate, fill_rgb)
        surface_ratio = _min_contrast(candidate, surfaces)
        current_bonus = 1.0 if candidate == (255, 255, 255) else 0.0
        return (float(fill_ratio), float(surface_ratio), float(current_bonus))

    return max(_STROKE_CANDIDATES, key=score)


def resolve_readable_text_style(
    *,
    instance_seed: int,
    namespace: str,
    role: str,
    surface_rgbs: Sequence[Sequence[int]],
    preferred_rgbs: Sequence[Sequence[int]] | None = None,
    candidate_rgbs: Sequence[Sequence[int]] | None = None,
    min_contrast_ratio: float = READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
    min_lab_distance: float = READ_REQUIRED_TEXT_MIN_LAB_DISTANCE,
    required: bool = True,
) -> ReadableTextStyle:
    """Resolve one random non-semantic text color that remains readable.

    The candidate ink is sampled deterministically from a shared readable pool.
    Preferred colors remain eligible, but they are not allowed to bypass the
    contrast requirements for required/read-off text.
    """

    surfaces = _unique_colors(surface_rgbs)
    preferred = tuple(preferred_rgbs or ())
    pool = _unique_colors((*preferred, *(candidate_rgbs or READABLE_TEXT_INK_POOL)))
    rng = spawn_rng(instance_seed=int(instance_seed), namespace=str(namespace))
    shuffled = list(pool)
    rng.shuffle(shuffled)
    preferred_set = set(_unique_colors(preferred))
    eligible = [
        color
        for color in shuffled
        if _min_contrast(color, surfaces) >= float(min_contrast_ratio)
        and _min_lab_distance(color, surfaces) >= float(min_lab_distance)
    ]
    if eligible:
        preferred_eligible = [color for color in eligible if color in preferred_set]
        if preferred_eligible and rng.random() < 0.35:
            fill_rgb = preferred_eligible[int(rng.randint(0, len(preferred_eligible) - 1))]
        else:
            fill_rgb = eligible[int(rng.randint(0, len(eligible) - 1))]
    else:
        fill_rgb = max(
            shuffled or list(READABLE_TEXT_INK_POOL),
            key=lambda color: (_min_contrast(color, surfaces), _min_lab_distance(color, surfaces)),
        )
    stroke_rgb = _choose_text_stroke(fill_rgb, surfaces)
    return ReadableTextStyle(
        role=str(role),
        fill_rgb=tuple(int(channel) for channel in fill_rgb),
        stroke_rgb=tuple(int(channel) for channel in stroke_rgb),
        surface_rgbs=tuple(surfaces),
        min_contrast_ratio=float(_min_contrast(fill_rgb, surfaces)),
        min_lab_distance=float(_min_lab_distance(fill_rgb, surfaces)),
        required=bool(required),
        min_contrast_required=float(min_contrast_ratio),
        min_lab_distance_required=float(min_lab_distance),
    )


def text_legibility_summary(
    styles: Sequence[ReadableTextStyle],
    *,
    drawn_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return trace metadata for a collection of resolved text roles."""

    records = [style.metadata() for style in styles]
    records.extend(dict(record) for record in (drawn_records or ()))
    return text_legibility_summary_from_records(
        records,
        drawn_text_record_count=int(len(drawn_records or ())),
    )


def text_legibility_summary_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    drawn_text_record_count: int = 0,
) -> dict[str, Any]:
    """Return trace metadata from already-serialized text legibility records."""

    records = [dict(record) for record in records]
    required_records = [record for record in records if bool(record.get("required", True))]
    failures = [record for record in required_records if not bool(record.get("passes", False))]
    min_required_contrast = min(
        (float(record.get("min_contrast_ratio", 0.0)) for record in required_records),
        default=0.0,
    )
    return {
        "enabled": True,
        "policy_version": TEXT_LEGIBILITY_POLICY_VERSION,
        "policy": "required_text_uses_random_nonsemantic_readable_ink",
        "required_role_count": int(len(required_records)),
        "drawn_text_record_count": int(max(0, int(drawn_text_record_count))),
        "min_required_contrast_ratio": round(float(min_required_contrast), 3),
        "failure_count": int(len(failures)),
        "records": records,
    }


@contextmanager
def collect_traced_text_records() -> Iterable[list[dict[str, Any]]]:
    """Collect visible text draw records emitted through shared text helpers."""

    records: list[dict[str, Any]] = []
    token = _ACTIVE_DRAWN_TEXT_RECORDS.set(records)
    try:
        yield records
    finally:
        _ACTIVE_DRAWN_TEXT_RECORDS.reset(token)


def _collected_record_copy(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validation-safe copy of one drawn text record.

    Some migrated compatibility call sites pass ``required=True`` before they
    have per-surface contrast metadata. Keep those records useful for coverage
    auditing, but do not let them masquerade as validated required text.
    Required/read-off validation remains attached to resolved
    ``ReadableTextStyle`` records.
    """

    out = dict(record)
    if bool(out.get("declared_required_without_contrast_metadata", False)):
        out["required"] = False
    elif bool(out.get("required", False)) and "passes" not in out:
        out["declared_required_without_contrast_metadata"] = True
        out["required"] = False
    return out


def record_traced_text(record: Mapping[str, Any]) -> None:
    """Append one drawn text record to the active generation collector."""

    records = _ACTIVE_DRAWN_TEXT_RECORDS.get()
    if records is not None:
        records.append(dict(record))


def _coerce_bbox_px(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(item))) for item in value]
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _surface_samples_around_bbox(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    margin_px: int = 3,
    max_colors: int = 6,
) -> Tuple[Color, ...]:
    """Sample likely background colors around a final drawn text bbox."""

    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    x0, y0, x1, y1 = bbox
    left = max(0, x0 - int(margin_px))
    right = min(width - 1, x1 + int(margin_px))
    top = max(0, y0 - int(margin_px))
    bottom = min(height - 1, y1 + int(margin_px))
    if right <= left or bottom <= top:
        return ()
    x_step = max(1, int(round((right - left + 1) / 16.0)))
    y_step = max(1, int(round((bottom - top + 1) / 8.0)))
    samples: list[Color] = []
    for x in range(left, right + 1, x_step):
        if top < y0:
            samples.append(normalize_rgb(rgb_image.getpixel((x, top))))
        if bottom > y1:
            samples.append(normalize_rgb(rgb_image.getpixel((x, bottom))))
    for y in range(top, bottom + 1, y_step):
        if left < x0:
            samples.append(normalize_rgb(rgb_image.getpixel((left, y))))
        if right > x1:
            samples.append(normalize_rgb(rgb_image.getpixel((right, y))))
    if not samples:
        return ()
    counter = Counter(samples)
    return tuple(color for color, _count in counter.most_common(max(1, int(max_colors))))


def annotate_traced_text_records_with_image_contrast(
    records: Sequence[Mapping[str, Any]],
    *,
    image: Image.Image | None,
) -> list[dict[str, Any]]:
    """Attach approximate final-image contrast metadata to drawn text records.

    This is a compatibility bridge for renderers that route text through the
    shared draw wrappers but have not yet resolved a role-specific
    ``ReadableTextStyle``. It samples the immediate final-image neighborhood
    around each text bbox, so it should be used for auditing and metadata, not
    as a replacement for semantic render-time style resolution.
    """

    out: list[dict[str, Any]] = []
    for record in records:
        copied = dict(record)
        if image is not None and "passes" not in copied and "fill_rgb" in copied and "bbox_px" in copied:
            bbox = _coerce_bbox_px(copied.get("bbox_px"))
            if bbox is not None:
                surfaces = _surface_samples_around_bbox(image, bbox)
                if surfaces:
                    fill_rgb = normalize_rgb(copied["fill_rgb"])
                    copied["surface_rgbs"] = [list(color) for color in surfaces]
                    copied["surface_sample_method"] = "final_image_bbox_perimeter"
                    copied["min_contrast_ratio"] = round(float(_min_contrast(fill_rgb, surfaces)), 3)
                    copied["min_lab_distance"] = round(float(_min_lab_distance(fill_rgb, surfaces)), 3)
                    if bool(copied.get("required", False)):
                        contrast_required = READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO
                        lab_required = READ_REQUIRED_TEXT_MIN_LAB_DISTANCE
                    else:
                        contrast_required = CONTEXT_TEXT_MIN_CONTRAST_RATIO
                        lab_required = 0.0
                    copied["min_contrast_required"] = round(float(contrast_required), 3)
                    copied["min_lab_distance_required"] = round(float(lab_required), 3)
                    copied["passes"] = bool(
                        float(copied["min_contrast_ratio"]) >= float(contrast_required)
                        and float(copied["min_lab_distance"]) >= float(lab_required)
                    )
        if bool(copied.get("required", False)) and "passes" not in copied:
            copied["declared_required_without_contrast_metadata"] = True
            copied["required"] = False
        out.append(_collected_record_copy(copied))
    return out


def traced_text_records_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    image: Image.Image | None = None,
) -> dict[str, Any]:
    """Return a trace metadata block for automatically collected text draws."""

    copied = annotate_traced_text_records_with_image_contrast(records, image=image)
    summary = text_legibility_summary_from_records(
        copied,
        drawn_text_record_count=int(len(copied)),
    )
    summary["source"] = "automatic_drawn_text_collector"
    return summary


@dataclass
class TextLegibilityRecorder:
    """Collect text-legibility metadata while rendering one image or panel.

    Use this when a renderer draws required/read-off text and wants the trace
    to record both the resolved style and the final projected glyph box. The
    recorder is intentionally lightweight so renderers can adopt it without
    changing their geometry/annotation source of truth.
    """

    canvas_size_px: tuple[int, int] | None = None
    styles: list[ReadableTextStyle] = field(default_factory=list)
    drawn_records: list[dict[str, Any]] = field(default_factory=list)

    def add_style(self, style: ReadableTextStyle) -> ReadableTextStyle:
        self.styles.append(style)
        return style

    def add_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(record)
        if self.canvas_size_px is not None:
            out.setdefault("canvas_size_px", [int(self.canvas_size_px[0]), int(self.canvas_size_px[1])])
        self.drawn_records.append(out)
        return out

    def draw_text(
        self,
        draw: ImageDraw.ImageDraw,
        *,
        xy: tuple[float, float],
        text: str,
        font: ImageFont.ImageFont,
        style: ReadableTextStyle,
        stroke_width: int = 1,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = draw_readable_text(
            draw,
            xy=xy,
            text=str(text),
            font=font,
            style=style,
            stroke_width=max(0, int(stroke_width)),
            extra_metadata=extra_metadata,
        )
        return self.add_record(record)

    def draw_centered_text(
        self,
        draw: ImageDraw.ImageDraw,
        *,
        center: tuple[float, float],
        text: str,
        font: ImageFont.ImageFont,
        style: ReadableTextStyle,
        stroke_width: int = 1,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = draw_centered_readable_text(
            draw,
            center=center,
            text=str(text),
            font=font,
            style=style,
            stroke_width=max(0, int(stroke_width)),
            extra_metadata=extra_metadata,
        )
        return self.add_record(record)

    def metadata(self) -> dict[str, Any]:
        summary = text_legibility_summary(self.styles, drawn_records=self.drawn_records)
        if self.canvas_size_px is not None:
            summary["canvas_size_px"] = [int(self.canvas_size_px[0]), int(self.canvas_size_px[1])]
        return summary


def text_bbox(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    *,
    stroke_width: int = 0,
) -> BBox:
    """Return one rendered text bbox."""

    try:
        bbox = draw.textbbox(
            (float(xy[0]), float(xy[1])),
            str(text),
            font=font,
            stroke_width=max(0, int(stroke_width)),
        )
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        x, y = float(xy[0]), float(xy[1])
        return (x, y, x + float(width), y + float(height))


def draw_readable_text(
    draw: ImageDraw.ImageDraw,
    *,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    style: ReadableTextStyle,
    stroke_width: int = 1,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Draw text using a resolved readable style and return trace metadata."""

    draw.text(
        (float(xy[0]), float(xy[1])),
        str(text),
        font=font,
        fill=tuple(int(channel) for channel in style.fill_rgb),
        stroke_width=max(0, int(stroke_width)),
        stroke_fill=tuple(int(channel) for channel in style.stroke_rgb),
    )
    bbox = text_bbox(draw, (float(xy[0]), float(xy[1])), str(text), font, stroke_width=max(0, int(stroke_width)))
    record = style.metadata()
    record.update(
        {
            "text": str(text),
            "bbox_px": [round(float(value), 3) for value in bbox],
            "font_size_px": int(getattr(font, "size", 0) or 0),
            "stroke_width_px": max(0, int(stroke_width)),
        }
    )
    if extra_metadata:
        record.update(dict(extra_metadata))
    record_traced_text(record)
    return record


def draw_traced_text(
    draw: ImageDraw.ImageDraw,
    *,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill_rgb: Sequence[int],
    stroke_rgb: Sequence[int] | None = None,
    stroke_width: int = 0,
    role: str = "visible_text",
    required: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Draw visible text through the shared text path and return trace metadata.

    This wrapper is for migrated renderers that already resolved their colors
    elsewhere or are drawing non-answer context text. Required/read-off text
    should still record contrast metadata through `ReadableTextStyle` records in
    render metadata.
    """

    fill = normalize_rgb(fill_rgb)
    stroke = normalize_rgb(stroke_rgb) if stroke_rgb is not None else fill
    draw.text(
        (float(xy[0]), float(xy[1])),
        str(text),
        font=font,
        fill=fill,
        stroke_width=max(0, int(stroke_width)),
        stroke_fill=stroke,
    )
    bbox = text_bbox(draw, (float(xy[0]), float(xy[1])), str(text), font, stroke_width=max(0, int(stroke_width)))
    record = {
        "role": str(role),
        "required": bool(required),
        "text": str(text),
        "fill_rgb": list(fill),
        "stroke_rgb": list(stroke),
        "bbox_px": [round(float(value), 3) for value in bbox],
        "font_size_px": int(getattr(font, "size", 0) or 0),
        "stroke_width_px": max(0, int(stroke_width)),
    }
    if extra_metadata:
        record.update(dict(extra_metadata))
    record_traced_text(record)
    return record


def draw_text_traced(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float] | Sequence[float],
    text: str,
    *args: Any,
    role: str = "visible_text",
    required: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compatibility wrapper for migrated `ImageDraw.text` call sites.

    The signature intentionally accepts PIL's text kwargs so bulk migrations can
    replace `draw.text(...)` with `draw_text_traced(draw, ...)` without changing
    layout behavior.
    """

    draw.text(xy, str(text), *args, **kwargs)
    font = kwargs.get("font")
    fill = kwargs.get("fill")
    stroke_fill = kwargs.get("stroke_fill", fill)
    stroke_width = max(0, int(kwargs.get("stroke_width", 0) or 0))
    anchor = kwargs.get("anchor")
    try:
        bbox = draw.textbbox(xy, str(text), font=font, stroke_width=stroke_width, anchor=anchor)
    except Exception:
        bbox = text_bbox(draw, (float(xy[0]), float(xy[1])), str(text), font, stroke_width=stroke_width)  # type: ignore[index]
    record = {
        "role": str(role),
        "required": bool(required),
        "text": str(text),
        "bbox_px": [round(float(value), 3) for value in bbox],
        "font_size_px": int(getattr(font, "size", 0) or 0),
        "stroke_width_px": int(stroke_width),
    }
    if fill is not None:
        try:
            record["fill_rgb"] = list(normalize_rgb(fill))
        except Exception:
            pass
    if stroke_fill is not None:
        try:
            record["stroke_rgb"] = list(normalize_rgb(stroke_fill))
        except Exception:
            pass
    if extra_metadata:
        record.update(dict(extra_metadata))
    record_traced_text(record)
    return record


def draw_centered_traced_text(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill_rgb: Sequence[int],
    stroke_rgb: Sequence[int] | None = None,
    stroke_width: int = 0,
    role: str = "visible_text",
    required: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Draw centered visible text through the shared text path."""

    bbox0 = text_bbox(draw, (0.0, 0.0), str(text), font, stroke_width=max(0, int(stroke_width)))
    width = float(bbox0[2] - bbox0[0])
    height = float(bbox0[3] - bbox0[1])
    x = float(center[0]) - (width / 2.0) - float(bbox0[0])
    y = float(center[1]) - (height / 2.0) - float(bbox0[1])
    return draw_traced_text(
        draw,
        xy=(float(x), float(y)),
        text=str(text),
        font=font,
        fill_rgb=fill_rgb,
        stroke_rgb=stroke_rgb,
        stroke_width=max(0, int(stroke_width)),
        role=str(role),
        required=bool(required),
        extra_metadata=extra_metadata,
    )


def draw_centered_readable_text(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    style: ReadableTextStyle,
    stroke_width: int = 1,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Draw centered text using a resolved readable style and return metadata."""

    bbox0 = text_bbox(draw, (0.0, 0.0), str(text), font, stroke_width=max(0, int(stroke_width)))
    width = float(bbox0[2] - bbox0[0])
    height = float(bbox0[3] - bbox0[1])
    x = float(center[0]) - (width / 2.0) - float(bbox0[0])
    y = float(center[1]) - (height / 2.0) - float(bbox0[1])
    return draw_readable_text(
        draw,
        xy=(float(x), float(y)),
        text=str(text),
        font=font,
        style=style,
        stroke_width=max(0, int(stroke_width)),
        extra_metadata=extra_metadata,
    )
