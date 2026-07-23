"""Shared RGB/Lab color-distance and constrained color-sampling utilities."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence, Tuple

Color = Tuple[int, int, int]
DEFAULT_COLOR_DISTANCE_SPACE = "lab"
DEFAULT_MIN_COLOR_DISTANCE = 60.0
DEFAULT_COLOR_SAMPLING_ATTEMPTS = 256
VISIBILITY_SAFE_FALLBACK_COLORS: Tuple[Color, ...] = (
    (255, 255, 255),
    (0, 0, 0),
    (0, 114, 178),
    (213, 94, 0),
    (0, 158, 115),
    (204, 121, 167),
    (230, 159, 0),
    (86, 180, 233),
    (240, 228, 66),
    (180, 40, 80),
    (60, 70, 180),
    (30, 170, 210),
)


def _clamp_channel(value: int) -> int:
    """Clamp one integer color channel into the `[0, 255]` range."""
    return max(0, min(255, int(value)))


def _rgb_channel_to_linear(channel_value: int) -> float:
    """Convert one 8-bit sRGB channel into linear-light space."""
    normalized = float(_clamp_channel(channel_value)) / 255.0
    if normalized <= 0.04045:
        return normalized / 12.92
    return ((normalized + 0.055) / 1.055) ** 2.4


def rgb_to_xyz(color: Color) -> Tuple[float, float, float]:
    """Convert one sRGB color (D65) into XYZ space."""
    red_linear = _rgb_channel_to_linear(int(color[0]))
    green_linear = _rgb_channel_to_linear(int(color[1]))
    blue_linear = _rgb_channel_to_linear(int(color[2]))
    x_value = (0.4124564 * red_linear) + (0.3575761 * green_linear) + (0.1804375 * blue_linear)
    y_value = (0.2126729 * red_linear) + (0.7151522 * green_linear) + (0.0721750 * blue_linear)
    z_value = (0.0193339 * red_linear) + (0.1191920 * green_linear) + (0.9503041 * blue_linear)
    return (float(x_value), float(y_value), float(z_value))


def _xyz_component_to_lab(component: float) -> float:
    """Convert one normalized XYZ component for CIE Lab conversion."""
    if float(component) > 0.008856:
        return float(component) ** (1.0 / 3.0)
    return (7.787 * float(component)) + (16.0 / 116.0)


def xyz_to_lab(xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert one XYZ color (D65) into CIE Lab."""
    x_value, y_value, z_value = xyz
    ref_x, ref_y, ref_z = 0.95047, 1.00000, 1.08883
    fx = _xyz_component_to_lab(float(x_value) / float(ref_x))
    fy = _xyz_component_to_lab(float(y_value) / float(ref_y))
    fz = _xyz_component_to_lab(float(z_value) / float(ref_z))
    l_value = (116.0 * float(fy)) - 16.0
    a_value = 500.0 * (float(fx) - float(fy))
    b_value = 200.0 * (float(fy) - float(fz))
    return (float(l_value), float(a_value), float(b_value))


def delta_e_cie76(color_a: Color, color_b: Color) -> float:
    """Return CIE76 (ΔE*ab) distance between two sRGB colors."""
    lab_a = xyz_to_lab(rgb_to_xyz(color_a))
    lab_b = xyz_to_lab(rgb_to_xyz(color_b))
    delta_l = float(lab_a[0]) - float(lab_b[0])
    delta_a = float(lab_a[1]) - float(lab_b[1])
    delta_b = float(lab_a[2]) - float(lab_b[2])
    return math.sqrt((delta_l * delta_l) + (delta_a * delta_a) + (delta_b * delta_b))


def rgb_euclidean_distance(color_a: Color, color_b: Color) -> float:
    """Return Euclidean distance between two colors in raw RGB space."""
    delta_red = float(int(color_a[0]) - int(color_b[0]))
    delta_green = float(int(color_a[1]) - int(color_b[1]))
    delta_blue = float(int(color_a[2]) - int(color_b[2]))
    return math.sqrt((delta_red * delta_red) + (delta_green * delta_green) + (delta_blue * delta_blue))


def color_distance(color_a: Color, color_b: Color, *, distance_space: str = "lab") -> float:
    """Return color distance in `lab` (CIE76) or `rgb` Euclidean space."""
    space = str(distance_space).strip().lower()
    if space == "lab":
        return float(delta_e_cie76(color_a, color_b))
    if space == "rgb":
        return float(rgb_euclidean_distance(color_a, color_b))
    raise ValueError(f"unsupported color distance space: {distance_space!r}")


def min_color_distance_to_anchors(
    color: Color,
    anchor_colors: Iterable[Color],
    *,
    distance_space: str = DEFAULT_COLOR_DISTANCE_SPACE,
) -> float:
    """Return the nearest distance from one color to any normalized anchor color."""

    anchors = tuple(_normalize_colors(anchor_colors))
    if not anchors:
        return float("inf")
    normalized = _normalize_color(color)
    return min(float(color_distance(normalized, anchor, distance_space=str(distance_space))) for anchor in anchors)


def _unique_colors_in_order(colors: Iterable[Color]) -> Tuple[Color, ...]:
    """Return normalized unique colors while preserving first-seen order."""

    seen: set[Color] = set()
    ordered: list[Color] = []
    for color in colors:
        normalized = _normalize_color(color)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def resolve_contrasting_palette(
    candidate_colors: Sequence[Color],
    *,
    anchor_colors: Iterable[Color],
    min_anchor_distance: float = 40.0,
    min_pairwise_distance: float = 20.0,
    distance_space: str = DEFAULT_COLOR_DISTANCE_SPACE,
    fallback_colors: Sequence[Color] = VISIBILITY_SAFE_FALLBACK_COLORS,
) -> Tuple[Color, ...]:
    """Return a palette whose colors avoid known background/anchor colors.

    Candidate colors are preserved when they satisfy the anchor and pairwise
    thresholds. Unsafe candidates are deterministically replaced from the
    remaining candidate pool plus a small shared high-contrast fallback palette.
    If every option misses the threshold, the color with the best minimum anchor
    distance is selected rather than failing generation.
    """

    normalized_candidates = tuple(_normalize_color(color) for color in candidate_colors)
    if not normalized_candidates:
        return tuple()
    anchors = tuple(_normalize_colors(anchor_colors))
    pool = _unique_colors_in_order((*normalized_candidates, *tuple(fallback_colors)))
    threshold = max(0.0, float(min_anchor_distance))
    pair_threshold = max(0.0, float(min_pairwise_distance))
    resolved: list[Color] = []

    def _pairwise_distance(candidate: Color) -> float:
        if not resolved:
            return float("inf")
        return min(float(color_distance(candidate, color, distance_space=str(distance_space))) for color in resolved)

    def _score(candidate: Color, *, preferred: Color) -> tuple[float, float, float, float, float]:
        anchor_distance = min_color_distance_to_anchors(candidate, anchors, distance_space=str(distance_space))
        pair_distance = _pairwise_distance(candidate)
        duplicate_penalty = -1.0 if candidate in resolved else 0.0
        preferred_bonus = 1.0 if candidate == preferred else 0.0
        return (
            1.0 if float(anchor_distance) >= float(threshold) else 0.0,
            1.0 if float(pair_distance) >= float(pair_threshold) else 0.0,
            float(anchor_distance),
            float(pair_distance),
            float(preferred_bonus + duplicate_penalty),
        )

    for preferred in normalized_candidates:
        preferred_score = _score(preferred, preferred=preferred)
        if preferred_score[0] >= 1.0 and preferred_score[1] >= 1.0:
            resolved.append(preferred)
            continue
        best = max(pool, key=lambda candidate: _score(candidate, preferred=preferred))
        resolved.append(_normalize_color(best))
    return tuple(resolved)


def _sample_random_color(rng, *, channel_min: int, channel_max: int) -> Color:
    """Sample one RGB color with per-channel bounds."""
    return (
        int(rng.randint(int(channel_min), int(channel_max))),
        int(rng.randint(int(channel_min), int(channel_max))),
        int(rng.randint(int(channel_min), int(channel_max))),
    )


def _normalize_color(color: Color) -> Color:
    """Clamp one RGB triplet into a normalized integer color tuple."""
    return (_clamp_channel(int(color[0])), _clamp_channel(int(color[1])), _clamp_channel(int(color[2])))


def normalize_rgb(color: Sequence[int]) -> Color:
    """Return one clamped RGB tuple from a sequence with at least three channels."""

    if len(color) < 3:
        raise ValueError("normalize_rgb requires at least three channels")
    return (_clamp_channel(int(color[0])), _clamp_channel(int(color[1])), _clamp_channel(int(color[2])))


def coerce_rgb(value: Any, fallback: Sequence[int]) -> Color:
    """Return a clamped RGB tuple from ``value`` or a trusted fallback RGB tuple."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        return normalize_rgb(value[:3])
    return (int(fallback[0]), int(fallback[1]), int(fallback[2]))


def _normalize_colors(colors: Iterable[Color]) -> tuple[Color, ...]:
    """Return unique normalized RGB triplets in deterministic order."""
    unique = sorted({_normalize_color(color) for color in colors})
    return tuple(unique)


def sample_color_with_distance_constraints(
    rng,
    *,
    channel_min: int,
    channel_max: int,
    anchor_colors: Iterable[Color],
    min_distance: float = DEFAULT_MIN_COLOR_DISTANCE,
    max_attempts: int = DEFAULT_COLOR_SAMPLING_ATTEMPTS,
    distance_space: str = DEFAULT_COLOR_DISTANCE_SPACE,
) -> Color:
    """Sample one color at least `min_distance` from all anchors, with deterministic fallback.

    Fallback behavior keeps determinism and avoids hard failure: if no candidate meets
    the threshold within `max_attempts`, this returns the sampled color that maximizes
    minimum anchor distance among attempted candidates.
    """
    low = max(0, min(255, int(channel_min)))
    high = max(0, min(255, int(channel_max)))
    if int(low) > int(high):
        low, high = high, low

    anchors = list(_normalize_colors(anchor_colors))
    attempts = max(1, int(max_attempts))
    threshold = max(0.0, float(min_distance))
    if not anchors:
        return _sample_random_color(rng, channel_min=int(low), channel_max=int(high))

    best_candidate: Color | None = None
    best_candidate_min_distance = -1.0
    for _ in range(int(attempts)):
        candidate = _sample_random_color(rng, channel_min=int(low), channel_max=int(high))
        nearest_anchor_distance = min(
            float(color_distance(candidate, anchor, distance_space=str(distance_space)))
            for anchor in anchors
        )
        if float(nearest_anchor_distance) >= float(threshold):
            return candidate
        if float(nearest_anchor_distance) > float(best_candidate_min_distance):
            best_candidate = candidate
            best_candidate_min_distance = float(nearest_anchor_distance)

    if best_candidate is not None:
        return best_candidate
    return _sample_random_color(rng, channel_min=int(low), channel_max=int(high))


def sample_color_palette_with_distance_constraints(
    rng,
    *,
    palette_size: int,
    channel_min: int,
    channel_max: int,
    anchor_colors: Iterable[Color] = (),
    min_distance: float = DEFAULT_MIN_COLOR_DISTANCE,
    max_attempts: int = DEFAULT_COLOR_SAMPLING_ATTEMPTS,
    distance_space: str = DEFAULT_COLOR_DISTANCE_SPACE,
) -> tuple[Color, ...]:
    """Sample `palette_size` colors with anchor + pairwise distance constraints.

    The sampler is deterministic and best-effort: each sampled color is selected by
    constrained sampling against all anchors plus already sampled colors.
    """
    size = max(0, int(palette_size))
    if int(size) == 0:
        return tuple()

    sampled: list[Color] = []
    anchors: list[Color] = list(_normalize_colors(anchor_colors))
    for _ in range(int(size)):
        constraints = tuple([*anchors, *sampled])
        color = sample_color_with_distance_constraints(
            rng,
            channel_min=int(channel_min),
            channel_max=int(channel_max),
            anchor_colors=constraints,
            min_distance=float(min_distance),
            max_attempts=max(1, int(max_attempts)),
            distance_space=str(distance_space),
        )
        sampled.append((_clamp_channel(int(color[0])), _clamp_channel(int(color[1])), _clamp_channel(int(color[2]))))
    return tuple(sampled)
