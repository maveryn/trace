"""Readable fill/label pairing for plain geometry readout diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.text_legibility import (
    READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
    READ_REQUIRED_TEXT_MIN_LAB_DISTANCE,
    resolve_readable_text_style,
)

from .diagram_style import GeometryDiagramStyle

Color = tuple[int, int, int]

_DARK_INK: Color = (10, 14, 22)
_LIGHT_INK: Color = (250, 252, 255)

_SAFE_LIGHT_FILLS: tuple[Color, ...] = (
    (236, 246, 255),
    (238, 248, 239),
    (255, 246, 226),
    (246, 242, 255),
    (255, 240, 236),
    (232, 247, 249),
)
_SAFE_DARK_FILLS: tuple[Color, ...] = (
    (34, 48, 68),
    (38, 64, 58),
    (68, 52, 80),
    (78, 49, 56),
    (72, 58, 35),
    (42, 57, 86),
)


@dataclass(frozen=True)
class ReadoutFillStyle:
    """Resolved fill palette and readout ink for a diagram scene."""

    fill_colors: tuple[Color, ...]
    label_color: Color
    label_stroke_color: Color
    metadata: dict[str, Any]


def resolve_readout_fill_style(
    *,
    instance_seed: int,
    namespace: str,
    diagram_style: GeometryDiagramStyle,
    background_meta: Mapping[str, Any],
    candidate_palettes: Sequence[Sequence[Sequence[int]]],
    params: Mapping[str, Any] | None = None,
) -> ReadoutFillStyle:
    """Choose a fill palette and label ink that pass required-readout contrast.

    Thin readout text is often drawn directly on both the diagram background
    and object fills. The fill palette and text ink therefore need to be
    resolved together; otherwise a theme can pick individually attractive
    colors that fail when stroke width is zero.
    """

    normalized_palettes = tuple(
        tuple(_rgb(color) for color in palette)
        for palette in candidate_palettes
        if len(tuple(palette)) > 0
    )
    if not normalized_palettes:
        raise ValueError("at least one fill palette is required")
    fill_count = len(normalized_palettes[0])
    if any(len(palette) != fill_count for palette in normalized_palettes):
        raise ValueError("all fill palettes must have the same length")

    anchors = _surface_anchor_colors(diagram_style=diagram_style, background_meta=background_meta)
    palette_candidates = _rotated_palettes(
        (
            *normalized_palettes,
            *_safe_palette_candidates(fill_count),
        ),
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.palette_order",
        params=params or {},
    )
    preferred = (
        _rgb(diagram_style.label_rgb),
        _rgb(diagram_style.stroke_rgb),
        _rgb(diagram_style.secondary_stroke_rgb),
        _DARK_INK,
        _LIGHT_INK,
    )

    best: tuple[tuple[float, float], tuple[Color, ...], Any] | None = None
    for index, palette in enumerate(palette_candidates):
        surfaces = (*anchors, *palette)
        text_style = resolve_readable_text_style(
            instance_seed=int(instance_seed) + int(index),
            namespace=f"{namespace}.readout_text.{index}",
            role="geometry_plain_readout",
            surface_rgbs=surfaces,
            preferred_rgbs=preferred,
            candidate_rgbs=preferred,
            min_contrast_ratio=READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
            min_lab_distance=READ_REQUIRED_TEXT_MIN_LAB_DISTANCE,
            required=True,
        )
        score = (
            float(text_style.min_contrast_ratio),
            float(text_style.min_lab_distance),
        )
        if bool(text_style.metadata().get("passes", False)):
            return _style_result(
                palette=palette,
                text_style=text_style,
                anchors=anchors,
                palette_index=index,
                candidate_count=len(palette_candidates),
            )
        if best is None or score > best[0]:
            best = (score, palette, text_style)

    assert best is not None
    return _style_result(
        palette=best[1],
        text_style=best[2],
        anchors=anchors,
        palette_index=-1,
        candidate_count=len(palette_candidates),
    )


def _style_result(
    *,
    palette: Sequence[Color],
    text_style: Any,
    anchors: Sequence[Color],
    palette_index: int,
    candidate_count: int,
) -> ReadoutFillStyle:
    metadata = {
        "policy": "geometry_plain_readout_fill_contrast_v1",
        "palette_index": int(palette_index),
        "candidate_palette_count": int(candidate_count),
        "surface_anchor_rgb": [list(color) for color in anchors],
        "fill_colors": [list(color) for color in palette],
        "readout_text_style": text_style.metadata(),
    }
    return ReadoutFillStyle(
        fill_colors=tuple(_rgb(color) for color in palette),
        label_color=_rgb(text_style.fill_rgb),
        label_stroke_color=_rgb(text_style.stroke_rgb),
        metadata=metadata,
    )


def _safe_palette_candidates(fill_count: int) -> tuple[tuple[Color, ...], ...]:
    light = _palette_windows(_SAFE_LIGHT_FILLS, fill_count)
    dark = _palette_windows(_SAFE_DARK_FILLS, fill_count)
    return (*light, *dark)


def _palette_windows(colors: Sequence[Color], fill_count: int) -> tuple[tuple[Color, ...], ...]:
    if fill_count <= 0:
        return tuple()
    normalized = tuple(_rgb(color) for color in colors)
    return tuple(
        tuple(normalized[(start + offset) % len(normalized)] for offset in range(fill_count))
        for start in range(len(normalized))
    )


def _rotated_palettes(
    palettes: Sequence[Sequence[Color]],
    *,
    instance_seed: int,
    namespace: str,
    params: Mapping[str, Any],
) -> tuple[tuple[Color, ...], ...]:
    normalized = _unique_palettes(palettes)
    if not normalized:
        return tuple()
    forced = params.get("readout_fill_palette_index")
    if forced is None:
        rng = spawn_rng(int(instance_seed), str(namespace))
        offset = int(rng.randrange(len(normalized)))
    else:
        offset = int(forced) % len(normalized)
    return (*normalized[offset:], *normalized[:offset])


def _surface_anchor_colors(
    *,
    diagram_style: GeometryDiagramStyle,
    background_meta: Mapping[str, Any],
) -> tuple[Color, ...]:
    anchors: list[Color] = [
        _rgb(diagram_style.canvas_rgb),
        _rgb(diagram_style.canvas_accent_rgb),
        _rgb(diagram_style.paper_rgb),
        _rgb(diagram_style.panel_fill_rgb),
        _rgb(diagram_style.panel_alt_fill_rgb),
        _rgb(diagram_style.option_fill_rgb),
    ]
    style_spec = background_meta.get("style_spec") if isinstance(background_meta, Mapping) else None
    if isinstance(style_spec, Mapping):
        background_style = style_spec.get("background_style")
        if isinstance(background_style, Mapping):
            for key in ("canvas_rgb", "canvas_accent_rgb", "paper_rgb"):
                value = background_style.get(key)
                if _is_rgb_like(value):
                    anchors.append(_rgb(value))
    return _unique_colors(anchors)


def _unique_palettes(palettes: Sequence[Sequence[Color]]) -> tuple[tuple[Color, ...], ...]:
    seen: set[tuple[Color, ...]] = set()
    resolved: list[tuple[Color, ...]] = []
    for palette in palettes:
        item = tuple(_rgb(color) for color in palette)
        if item in seen:
            continue
        seen.add(item)
        resolved.append(item)
    return tuple(resolved)


def _unique_colors(colors: Sequence[Color]) -> tuple[Color, ...]:
    seen: set[Color] = set()
    resolved: list[Color] = []
    for color in colors:
        rgb = _rgb(color)
        if rgb in seen:
            continue
        seen.add(rgb)
        resolved.append(rgb)
    return tuple(resolved)


def _is_rgb_like(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3


def _rgb(value: Sequence[int]) -> Color:
    if len(value) < 3:
        raise ValueError("RGB values require at least three channels")
    return tuple(max(0, min(255, int(channel))) for channel in value[:3])  # type: ignore[return-value]


__all__ = ["ReadoutFillStyle", "resolve_readout_fill_style"]
