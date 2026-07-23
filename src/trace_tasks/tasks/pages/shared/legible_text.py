"""Pages-specific helpers for required text contrast in structured page renderers."""

from __future__ import annotations

from typing import Any, Sequence, Tuple

from PIL import ImageDraw

from ...shared.text_legibility import (
    READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
    draw_readable_text,
    contrast_ratio,
    normalize_rgb,
    resolve_readable_text_style,
)

Color = Tuple[int, int, int]


def darken_surface_for_light_text(
    surface_rgb: Sequence[int],
    *,
    text_rgb: Sequence[int] = (255, 255, 255),
    min_contrast_ratio: float = READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
) -> Color:
    """Darken a header-like fill until light required text can pass contrast."""

    surface = normalize_rgb(surface_rgb)
    text = normalize_rgb(text_rgb)
    if contrast_ratio(text, surface) >= float(min_contrast_ratio):
        return surface
    for step in range(1, 33):
        weight_black = float(step) / 32.0
        candidate = tuple(int(round(float(channel) * (1.0 - weight_black))) for channel in surface)
        if contrast_ratio(text, candidate) >= float(min_contrast_ratio):
            return candidate  # type: ignore[return-value]
    return (0, 0, 0)


def draw_required_page_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: Any,
    *,
    role: str,
    surface_rgbs: Sequence[Sequence[int]],
    instance_seed: int,
    namespace: str,
    preferred_rgbs: Sequence[Sequence[int]] | None = None,
    stroke_width: int = 1,
    required: bool = True,
) -> list[float]:
    """Draw required page text and record contrast against known semantic surfaces."""

    style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        role=str(role),
        surface_rgbs=surface_rgbs,
        preferred_rgbs=preferred_rgbs,
        min_contrast_ratio=READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
        required=bool(required),
    )
    record = draw_readable_text(
        draw,
        xy=(float(xy[0]), float(xy[1])),
        text=str(text),
        font=font,
        style=style,
        stroke_width=max(0, int(stroke_width)),
    )
    return [float(value) for value in record["bbox_px"]]
