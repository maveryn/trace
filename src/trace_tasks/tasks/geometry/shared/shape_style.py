"""Shared deterministic geometry-ink style sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from ...shared.config_defaults import group_default, resolve_required_int_bounds
from ...shared.color_distance import (
    DEFAULT_COLOR_DISTANCE_SPACE,
    DEFAULT_COLOR_SAMPLING_ATTEMPTS,
    DEFAULT_MIN_COLOR_DISTANCE,
    sample_color_palette_with_distance_constraints,
    sample_color_with_distance_constraints,
)

Color = Tuple[int, int, int]


@dataclass(frozen=True)
class GeometryShapeStyle:
    """Per-instance shape/label ink colors for geometry scenes."""

    line_color: Color
    label_color: Color
    label_stroke_color: Color

    def to_trace_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable shape-style metadata for trace payloads."""
        return {
            "line_color": [int(self.line_color[0]), int(self.line_color[1]), int(self.line_color[2])],
            "label_color": [int(self.label_color[0]), int(self.label_color[1]), int(self.label_color[2])],
            "label_stroke_color": [
                int(self.label_stroke_color[0]),
                int(self.label_stroke_color[1]),
                int(self.label_stroke_color[2]),
            ],
        }


def _resolve_gray_range(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
) -> Tuple[int, int]:
    """Resolve one inclusive grayscale range from params/defaults with validation."""
    min_value, max_value = resolve_required_int_bounds(
        params,
        render_defaults,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context="geometry shape-style grayscale defaults",
    )
    low = max(0, min(255, int(min_value)))
    high = max(0, min(255, int(max_value)))
    if int(low) > int(high):
        raise ValueError(f"{min_key} must be <= {max_key}")
    return (int(low), int(high))


def _sample_gray_triplet(rng, low: int, high: int) -> Color:
    """Sample one grayscale RGB triplet with inclusive integer bounds."""
    value = int(rng.randint(int(low), int(high)))
    return (int(value), int(value), int(value))


def _coerce_color_triplet(value: Any) -> Color | None:
    """Coerce one RGB-like value into a clamped integer color tuple."""
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    return (
        max(0, min(255, int(value[0]))),
        max(0, min(255, int(value[1]))),
        max(0, min(255, int(value[2]))),
    )


def extract_background_anchor_colors(background_meta: Mapping[str, Any] | None) -> Tuple[Color, ...]:
    """Extract background ink colors that geometry marks should avoid.

    For graph-paper scenes this includes grid/axis/center/origin colors. For solid
    backgrounds this includes the fill color.
    """
    if not isinstance(background_meta, Mapping):
        return tuple()
    style_spec = background_meta.get("style_spec", {})
    if not isinstance(style_spec, Mapping):
        return tuple()

    anchors: list[Color] = []
    for key in (
        "color",
        "base_color",
        "background_rgb",
        "background_accent_rgb",
        "line_color",
        "major_line_color",
        "axis_color",
        "center_point_color",
        "origin_label_color",
    ):
        sampled = _coerce_color_triplet(style_spec.get(str(key)))
        if sampled is not None:
            anchors.append(sampled)
    unique = sorted(set(anchors))
    return tuple(unique)


def sample_geometry_shape_style(
    rng,
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    anchor_colors: Sequence[Color] | None = None,
) -> GeometryShapeStyle:
    """Sample deterministic geometry shape colors with perceptual distance constraints."""
    color_mode = str(group_default(render_defaults, "shape_color_mode", "rgb_constrained")).strip().lower()
    if "shape_color_mode" in params and params.get("shape_color_mode") is not None:
        color_mode = str(params.get("shape_color_mode")).strip().lower()
    if str(color_mode) == "grayscale":
        line_low, line_high = _resolve_gray_range(
            params,
            render_defaults,
            min_key="shape_line_gray_min",
            max_key="shape_line_gray_max",
            fallback_min=10,
            fallback_max=28,
        )
        label_low, label_high = _resolve_gray_range(
            params,
            render_defaults,
            min_key="shape_label_gray_min",
            max_key="shape_label_gray_max",
            fallback_min=14,
            fallback_max=36,
        )
        stroke_low, stroke_high = _resolve_gray_range(
            params,
            render_defaults,
            min_key="shape_label_stroke_gray_min",
            max_key="shape_label_stroke_gray_max",
            fallback_min=248,
            fallback_max=255,
        )
        return GeometryShapeStyle(
            line_color=_sample_gray_triplet(rng, line_low, line_high),
            label_color=_sample_gray_triplet(rng, label_low, label_high),
            label_stroke_color=_sample_gray_triplet(rng, stroke_low, stroke_high),
        )

    line_low, line_high = resolve_required_int_bounds(
        params,
        render_defaults,
        min_key="shape_line_channel_min",
        max_key="shape_line_channel_max",
        fallback_min=12,
        fallback_max=172,
        context="geometry shape-style line channel defaults",
    )
    stroke_low, stroke_high = _resolve_gray_range(
        params,
        render_defaults,
        min_key="shape_label_stroke_gray_min",
        max_key="shape_label_stroke_gray_max",
        fallback_min=238,
        fallback_max=255,
    )
    distance_space = str(
        group_default(render_defaults, "color_distance_space", DEFAULT_COLOR_DISTANCE_SPACE)
    ).strip().lower()
    min_color_distance = float(group_default(render_defaults, "color_min_distance", DEFAULT_MIN_COLOR_DISTANCE))
    sampling_attempts = int(
        group_default(render_defaults, "color_sampling_attempts", DEFAULT_COLOR_SAMPLING_ATTEMPTS)
    )
    if "color_min_distance" in params and params.get("color_min_distance") is not None:
        min_color_distance = float(params.get("color_min_distance"))
    if "color_sampling_attempts" in params and params.get("color_sampling_attempts") is not None:
        sampling_attempts = int(params.get("color_sampling_attempts"))
    if "color_distance_space" in params and params.get("color_distance_space") is not None:
        distance_space = str(params.get("color_distance_space")).strip().lower()

    anchors: list[Color] = []
    if isinstance(anchor_colors, Iterable):
        for color in anchor_colors:
            anchors.append((int(color[0]), int(color[1]), int(color[2])))
    include_white = bool(group_default(render_defaults, "shape_include_white_anchor", True))
    include_black = bool(group_default(render_defaults, "shape_include_black_anchor", False))
    if "shape_include_white_anchor" in params and params.get("shape_include_white_anchor") is not None:
        include_white = bool(params.get("shape_include_white_anchor"))
    if "shape_include_black_anchor" in params and params.get("shape_include_black_anchor") is not None:
        include_black = bool(params.get("shape_include_black_anchor"))
    if bool(include_white):
        anchors.append((255, 255, 255))
    if bool(include_black):
        anchors.append((0, 0, 0))

    line_palette = sample_color_palette_with_distance_constraints(
        rng,
        palette_size=1,
        channel_min=int(line_low),
        channel_max=int(line_high),
        anchor_colors=tuple(anchors),
        min_distance=float(min_color_distance),
        max_attempts=max(1, int(sampling_attempts)),
        distance_space=str(distance_space),
    )
    line_color = line_palette[0]
    label_color = line_color
    stroke_min_distance = float(group_default(render_defaults, "shape_label_stroke_min_distance", 40.0))
    if "shape_label_stroke_min_distance" in params and params.get("shape_label_stroke_min_distance") is not None:
        stroke_min_distance = float(params.get("shape_label_stroke_min_distance"))
    label_stroke_color = sample_color_with_distance_constraints(
        rng,
        channel_min=int(stroke_low),
        channel_max=int(stroke_high),
        anchor_colors=(line_color,),
        min_distance=float(stroke_min_distance),
        max_attempts=max(1, int(sampling_attempts)),
        distance_space=str(distance_space),
    )
    return GeometryShapeStyle(
        line_color=line_color,
        label_color=label_color,
        label_stroke_color=label_stroke_color,
    )
