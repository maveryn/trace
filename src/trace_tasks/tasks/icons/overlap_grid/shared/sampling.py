"""Neutral sampling helpers for overlap-grid icon scenes."""

from __future__ import annotations

from typing import Sequence, Tuple

from ....shared.color_distance import color_distance


def order_id_for_front_role(front_role: str) -> str:
    """Return a stable order id from one front-role token."""

    if str(front_role) == "a":
        return "a_over_b"
    if str(front_role) == "b":
        return "b_over_a"
    raise ValueError(f"unsupported front_role: {front_role}")


def sample_tint_pair(
    rng,
    *,
    palette: Sequence[Tuple[int, int, int]],
    pair_min_color_distance: float,
    distance_space: str,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Sample two distinct icon tints from one already-separated palette."""

    if len(palette) < 2:
        raise ValueError("overlap-grid scenes require at least two palette colors")
    qualifying_pairs = [
        (left, right)
        for left_index, left in enumerate(palette)
        for right in palette[left_index + 1 :]
        if float(color_distance(left, right, distance_space=str(distance_space))) >= float(pair_min_color_distance)
    ]
    if not qualifying_pairs:
        raise ValueError("overlap-grid palette resolved no pair with sufficient color separation")
    first, second = rng.choice(qualifying_pairs)
    if bool(rng.randint(0, 1)):
        first, second = second, first
    return tuple(int(channel) for channel in first), tuple(int(channel) for channel in second)


def sample_overlap_offsets(
    rng,
    *,
    overlap_ratio_range: Sequence[float],
    max_offset_frac: float = 0.45,
) -> Tuple[float, float, float]:
    """Sample normalized icon offsets whose nominal overlap stays in the requested range."""

    overlap_min = max(0.0, min(0.95, float(overlap_ratio_range[0])))
    overlap_max = max(overlap_min, min(0.95, float(overlap_ratio_range[1])))
    for _ in range(120):
        dx_abs = float(rng.uniform(0.08, float(max_offset_frac)))
        dy_abs = float(rng.uniform(0.08, float(max_offset_frac)))
        overlap_ratio = float((1.0 - dx_abs) * (1.0 - dy_abs))
        if overlap_min <= overlap_ratio <= overlap_max:
            dx = dx_abs * float(rng.choice((-1.0, 1.0)))
            dy = dy_abs * float(rng.choice((-1.0, 1.0)))
            return float(dx), float(dy), float(overlap_ratio)
    target = 0.5 * float(overlap_min + overlap_max)
    dx_abs = min(float(max_offset_frac), max(0.08, 1.0 - target))
    dy_abs = min(float(max_offset_frac), max(0.08, 1.0 - (target / max(1e-6, 1.0 - dx_abs))))
    overlap_ratio = float((1.0 - dx_abs) * (1.0 - dy_abs))
    return (
        float(dx_abs * float(rng.choice((-1.0, 1.0)))),
        float(dy_abs * float(rng.choice((-1.0, 1.0)))),
        float(overlap_ratio),
    )


__all__ = ["order_id_for_front_role", "sample_overlap_offsets", "sample_tint_pair"]
