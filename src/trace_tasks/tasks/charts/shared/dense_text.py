"""Shared styling helpers for dense semantic chart text."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Tuple

from PIL import ImageFont

from trace_tasks.tasks.shared.text_legibility import contrast_ratio
from trace_tasks.tasks.shared.text_rendering import load_font

RGB = Tuple[int, int, int]

DENSE_TEXT_POLICY_VERSION = "charts_dense_semantic_text_v1"
DENSE_TEXT_DARK_RGB: RGB = (22, 28, 38)
DENSE_TEXT_LIGHT_RGB: RGB = (246, 250, 255)
DENSE_TEXT_MUTED_RGB: RGB = (58, 68, 82)
DENSE_TEXT_STROKE_WIDTH_PX = 0
DENSE_TEXT_FONT_FAMILY_WEIGHTS: dict[str, float] = {
    "roboto": 1.0,
    "source_sans_3": 1.0,
    "nunito_sans": 1.0,
    "fira_sans": 1.0,
    "barlow": 1.0,
    "karla": 1.0,
    "cabin": 1.0,
}


def dense_text_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return params with a readable dense-label font subset if none is set."""

    resolved = dict(params)
    resolved.setdefault("chart_font_family_weights", dict(DENSE_TEXT_FONT_FAMILY_WEIGHTS))
    return resolved


def dense_font(size_px: int, *, font_family: str | None = None, medium: bool = False) -> ImageFont.ImageFont:
    """Load a dense-label-safe font without heavy bold outlines."""

    return load_font(int(size_px), bold=bool(medium), font_family=font_family)


def dense_fit_bold() -> bool:
    """Return the standard bold flag for dense fitted labels."""

    return False


def dense_stroke_width() -> int:
    """Return the default outline width for dense semantic text."""

    return int(DENSE_TEXT_STROKE_WIDTH_PX)


def normalize_rgb(color: Sequence[int]) -> RGB:
    return tuple(max(0, min(255, int(channel))) for channel in color[:3])  # type: ignore[return-value]


def dense_text_fill_for_surface(surface_rgb: Sequence[int]) -> RGB:
    """Choose a single no-outline dense-text ink for one surface."""

    surface = normalize_rgb(surface_rgb)
    dark_ratio = contrast_ratio(DENSE_TEXT_DARK_RGB, surface)
    light_ratio = contrast_ratio(DENSE_TEXT_LIGHT_RGB, surface)
    return DENSE_TEXT_DARK_RGB if dark_ratio >= light_ratio else DENSE_TEXT_LIGHT_RGB


def lighten_for_dense_text(color: Sequence[int], amount: float = 0.34) -> RGB:
    """Lighten a semantic mark fill so dark no-outline labels stay readable."""

    factor = max(0.0, min(1.0, float(amount)))
    rgb = normalize_rgb(color)
    return tuple(int(round(channel + ((255 - channel) * factor))) for channel in rgb)  # type: ignore[return-value]


def dense_text_style_meta(*, role: str = "dense_semantic_text") -> dict[str, Any]:
    """Serialize the chart dense-text policy for trace metadata."""

    return {
        "policy_version": DENSE_TEXT_POLICY_VERSION,
        "role": str(role),
        "default_fill_rgb": list(DENSE_TEXT_DARK_RGB),
        "alternate_fill_rgb": list(DENSE_TEXT_LIGHT_RGB),
        "stroke_width_px": int(DENSE_TEXT_STROKE_WIDTH_PX),
        "font_weight": "regular",
        "font_family_weights": dict(DENSE_TEXT_FONT_FAMILY_WEIGHTS),
    }


__all__ = [
    "DENSE_TEXT_DARK_RGB",
    "DENSE_TEXT_FONT_FAMILY_WEIGHTS",
    "DENSE_TEXT_LIGHT_RGB",
    "DENSE_TEXT_MUTED_RGB",
    "DENSE_TEXT_POLICY_VERSION",
    "dense_fit_bold",
    "dense_font",
    "dense_stroke_width",
    "dense_text_fill_for_surface",
    "dense_text_params",
    "dense_text_style_meta",
    "lighten_for_dense_text",
]
