"""Scene-local style resolution for composite-shape geometry diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trace_tasks.tasks.geometry.shared.diagram_style import GeometryDiagramStyle
from trace_tasks.tasks.shared.color_distance import color_distance
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.text_legibility import (
    READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
    READ_REQUIRED_TEXT_MIN_LAB_DISTANCE,
    contrast_ratio,
    resolve_readable_text_style,
)

from .state import Color

MIN_FILL_BACKGROUND_LAB_DISTANCE = 28.0
MIN_FILL_PAIRWISE_LAB_DISTANCE = 16.0


@dataclass(frozen=True)
class CompositeShapeStyle:
    """Resolved semantic fill and readout colors for one composite-shape render."""

    fill_color: Color
    secondary_fill_color: Color
    accent_color: Color
    label_color: Color
    label_stroke_color: Color
    metadata: dict[str, Any]


def resolve_composite_shape_style(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_namespace: str,
    diagram_style: GeometryDiagramStyle,
    background_meta: Mapping[str, Any],
) -> CompositeShapeStyle:
    """Resolve visible composite-shape fills against the selected diagram surface.

    The shared technical-diagram theme exposes panel and option colors that may
    intentionally sit close to the paper color. Composite-shape shaded regions
    are semantic objects, so they must remain visibly separated from all sampled
    background anchors.
    """

    anchors = _surface_anchor_colors(diagram_style=diagram_style, background_meta=background_meta)
    selection_index = int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{render_namespace}.fill",
        )
    )
    candidates = _rotated_sequence(_fill_candidates(diagram_style), selection_index)
    primary = _choose_visible_color(
        candidates,
        background_anchors=anchors,
        pairwise_anchors=(),
        label_anchor=diagram_style.label_rgb,
    )
    secondary = _choose_visible_color(
        candidates,
        background_anchors=anchors,
        pairwise_anchors=(primary,),
        label_anchor=diagram_style.label_rgb,
    )
    accent = _choose_visible_color(
        _rotated_sequence(_accent_candidates(diagram_style), selection_index),
        background_anchors=anchors,
        pairwise_anchors=(primary, secondary),
        min_background_distance=20.0,
        min_pairwise_distance=12.0,
    )
    text_style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace=f"{render_namespace}.composite_shape_text",
        role="composite_shape_readout",
        surface_rgbs=anchors,
        preferred_rgbs=(
            tuple(int(value) for value in diagram_style.label_rgb),
            tuple(int(value) for value in diagram_style.stroke_rgb),
            tuple(int(value) for value in diagram_style.secondary_stroke_rgb),
        ),
        min_contrast_ratio=READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
        min_lab_distance=READ_REQUIRED_TEXT_MIN_LAB_DISTANCE,
        required=True,
    )
    metadata = {
        "policy": "composite_shape_semantic_fill_contrast_v1",
        "surface_anchor_rgb": [list(color) for color in anchors],
        "min_fill_background_lab_distance_required": MIN_FILL_BACKGROUND_LAB_DISTANCE,
        "min_fill_pairwise_lab_distance_required": MIN_FILL_PAIRWISE_LAB_DISTANCE,
        "fill_color": list(primary),
        "secondary_fill_color": list(secondary),
        "accent_color": list(accent),
        "fill_background_lab_distance_min": round(_min_lab_distance(primary, anchors), 3),
        "secondary_fill_background_lab_distance_min": round(_min_lab_distance(secondary, anchors), 3),
        "fill_pairwise_lab_distance": round(float(color_distance(primary, secondary, distance_space="lab")), 3),
        "readout_text_style": text_style.metadata(),
    }
    return CompositeShapeStyle(
        fill_color=primary,
        secondary_fill_color=secondary,
        accent_color=accent,
        label_color=tuple(int(value) for value in text_style.fill_rgb),
        label_stroke_color=tuple(int(value) for value in text_style.stroke_rgb),
        metadata=metadata,
    )


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


def _fill_candidates(style: GeometryDiagramStyle) -> tuple[Color, ...]:
    surface = _rgb(style.paper_rgb)
    canvas = _rgb(style.canvas_rgb)
    accents = (
        _rgb(style.fill_rgb),
        _rgb(style.muted_fill_rgb),
        _rgb(style.highlight_rgb),
        _rgb(style.accent_rgb),
        _rgb(style.secondary_accent_rgb),
        _rgb(style.guide_rgb),
    )
    derived: list[Color] = []
    for color in accents:
        derived.append(color)
        for weight in (0.36, 0.46, 0.56, 0.66):
            derived.append(_blend(surface, color, weight))
            derived.append(_blend(canvas, color, weight))
    return _unique_colors(derived)


def _accent_candidates(style: GeometryDiagramStyle) -> tuple[Color, ...]:
    return _unique_colors(
        (
            _rgb(style.accent_rgb),
            _rgb(style.secondary_accent_rgb),
            _rgb(style.highlight_rgb),
            _rgb(style.stroke_rgb),
            _rgb(style.secondary_stroke_rgb),
            _rgb(style.guide_rgb),
        )
    )


def _choose_visible_color(
    candidates: Sequence[Color],
    *,
    background_anchors: Sequence[Color],
    pairwise_anchors: Sequence[Color],
    label_anchor: Sequence[int] | None = None,
    min_background_distance: float = MIN_FILL_BACKGROUND_LAB_DISTANCE,
    min_pairwise_distance: float = MIN_FILL_PAIRWISE_LAB_DISTANCE,
) -> Color:
    threshold = max(0.0, float(min_background_distance))
    pair_threshold = max(0.0, float(min_pairwise_distance))
    normalized = _unique_colors(candidates)
    if not normalized:
        return (0, 114, 178)

    def score(candidate: Color) -> tuple[float, float, float, float, float]:
        background_distance = _min_lab_distance(candidate, background_anchors)
        pair_distance = _min_lab_distance(candidate, pairwise_anchors)
        label_contrast = (
            float("inf")
            if label_anchor is None
            else float(contrast_ratio(_rgb(label_anchor), candidate))
        )
        return (
            1.0 if background_distance >= threshold else 0.0,
            1.0 if pair_distance >= pair_threshold else 0.0,
            1.0 if label_contrast >= READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO else 0.0,
            min(background_distance, pair_distance),
            label_contrast,
        )

    for color in normalized:
        color_score = score(color)
        if color_score[0] >= 1.0 and color_score[1] >= 1.0 and color_score[2] >= 1.0:
            return color
    return max(normalized, key=score)


def _min_lab_distance(color: Color, anchors: Sequence[Color]) -> float:
    if not anchors:
        return float("inf")
    return min(float(color_distance(color, anchor, distance_space="lab")) for anchor in anchors)


def _rotated_sequence(colors: Sequence[Color], selection_index: int) -> tuple[Color, ...]:
    normalized = _unique_colors(colors)
    if not normalized:
        return tuple()
    offset = int(selection_index) % len(normalized)
    return (*normalized[offset:], *normalized[:offset])


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


def _blend(base: Color, overlay: Color, overlay_weight: float) -> Color:
    weight = max(0.0, min(1.0, float(overlay_weight)))
    return tuple(
        max(0, min(255, int(round((float(base[index]) * (1.0 - weight)) + (float(overlay[index]) * weight)))))
        for index in range(3)
    )  # type: ignore[return-value]


def _is_rgb_like(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3


def _rgb(value: Sequence[int]) -> Color:
    if len(value) < 3:
        raise ValueError("RGB values require at least three channels")
    return tuple(max(0, min(255, int(channel))) for channel in value[:3])  # type: ignore[return-value]
