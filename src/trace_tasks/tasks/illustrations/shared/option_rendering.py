"""Scene-neutral option rendering helpers for illustration visual tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat

from ...shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from ...shared.text_legibility import draw_text_traced
from ...shared.text_rendering import fit_font_to_box, load_font


def bbox_list(box: Sequence[float], *, dx: float = 0.0, dy: float = 0.0) -> list[float]:
    """Return a rounded pixel bbox list with an optional offset."""

    return [
        round(float(box[0]) + float(dx), 3),
        round(float(box[1]) + float(dy), 3),
        round(float(box[2]) + float(dx), 3),
        round(float(box[3]) + float(dy), 3),
    ]


def sort_bboxes_by_position(boxes: Sequence[Sequence[float]]) -> list[list[float]]:
    """Sort bboxes top-to-bottom, left-to-right."""

    return [
        bbox_list(box)
        for box in sorted(
            boxes,
            key=lambda box: (float(box[1]), float(box[0]), float(box[3]), float(box[2])),
        )
    ]


def default_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    """Load the active shared font for compact illustration labels."""

    return load_font(int(size), bold=bool(bold))


def draw_panel_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    xy: Tuple[int, int],
    *,
    size: int = 24,
    font_family: str | None = None,
) -> None:
    """Draw a compact panel label."""

    x, y = int(xy[0]), int(xy[1])
    font = (
        load_font(int(size), bold=True, font_family=str(font_family or ""))
        if font_family
        else default_font(int(size), bold=True)
    )
    text = str(label)
    bbox = draw.textbbox((x, y), text, font=font)
    pad_x = 8
    pad_y = 5
    draw.rounded_rectangle(
        (bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y),
        radius=6,
        fill=(255, 255, 255),
        outline=(42, 49, 58),
        width=2,
    )
    draw_text_traced(
        draw,
        (x, y),
        text,
        fill=(22, 28, 36),
        font=font,
        role="readout",
        required=False,
    )


def sample_visual_label_font_trace(
    *,
    namespace_prefix: str,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace_suffix: str,
    explicit_key: str,
    weights_key: str,
) -> Dict[str, Any]:
    """Sample one approved font family for a visual option-label role."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace_prefix}:{namespace_suffix}",
        params=params,
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    record = get_font_family_record(str(font_family))
    return {
        "font_asset_version": font_asset_version(),
        "pool": "global_approved_font_pool",
        **record.to_trace(),
    }


def draw_label_badge(
    draw: ImageDraw.ImageDraw,
    label: str,
    bbox_xyxy: Sequence[float],
    *,
    font_family: str | None = None,
    fill: Tuple[int, int, int] = (255, 255, 255),
    outline: Tuple[int, int, int] = (44, 52, 65),
    text_fill: Tuple[int, int, int] = (18, 25, 35),
    radius: int = 5,
    width: int = 2,
) -> None:
    """Draw a fitted compact label badge."""

    x0, y0, x1, y1 = [float(value) for value in bbox_xyxy]
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=int(radius),
        fill=fill,
        outline=outline,
        width=int(width),
    )
    text = str(label)
    font = fit_font_to_box(
        draw,
        text=text,
        max_width=max(1.0, (x1 - x0) - 8.0),
        max_height=max(1.0, (y1 - y0) - 6.0),
        bold=True,
        font_family=font_family,
        min_size_px=8,
        max_size_px=int(max(8.0, (y1 - y0) - 4.0)),
        fill_ratio=1.0,
    )
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    text_x = float(x0 + ((x1 - x0) - text_w) * 0.5 - float(text_bbox[0]))
    text_y = float(y0 + ((y1 - y0) - text_h) * 0.5 - float(text_bbox[1]))
    draw_text_traced(
        draw,
        (text_x, text_y),
        text,
        fill=text_fill,
        font=font,
        role="readout",
        required=False,
    )


def fit_source_image(image: Image.Image, *, width: int, height: int) -> Image.Image:
    """Crop-fit a source illustration to a fixed panel without blank padding."""

    return ImageOps.fit(
        image.convert("RGB"),
        (int(width), int(height)),
        method=Image.Resampling.LANCZOS,
    )


def image_detail_score(image: Image.Image) -> float:
    """Cheap crop-detail score used to avoid blank visual reconstruction crops."""

    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    variance = float(stat.var[0]) if stat.var else 0.0
    extrema = gray.getextrema()
    contrast = float(extrema[1] - extrema[0]) if extrema else 0.0
    return variance + 2.0 * contrast


__all__ = [
    "bbox_list",
    "default_font",
    "draw_label_badge",
    "draw_panel_label",
    "fit_source_image",
    "image_detail_score",
    "sample_visual_label_font_trace",
    "sort_bboxes_by_position",
]
