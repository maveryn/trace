"""Tests for shared color-distance constrained sampling utilities."""

from __future__ import annotations

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.color_distance import (
    color_distance,
    coerce_rgb,
    normalize_rgb,
    sample_color_palette_with_distance_constraints,
    sample_color_with_distance_constraints,
)


def test_palette_sampling_respects_anchor_and_pairwise_distance() -> None:
    rng = spawn_rng(9001, "color_palette_anchor")
    anchors = ((248, 248, 248), (220, 228, 238), (118, 128, 146))
    palette = sample_color_palette_with_distance_constraints(
        rng,
        palette_size=3,
        channel_min=8,
        channel_max=172,
        anchor_colors=anchors,
        min_distance=60.0,
        max_attempts=1024,
        distance_space="lab",
    )
    assert len(palette) == 3
    for sampled in palette:
        assert all(color_distance(sampled, anchor, distance_space="lab") >= 60.0 for anchor in anchors)
    assert color_distance(palette[0], palette[1], distance_space="lab") >= 60.0
    assert color_distance(palette[0], palette[2], distance_space="lab") >= 60.0
    assert color_distance(palette[1], palette[2], distance_space="lab") >= 60.0


def test_palette_sampling_without_anchors_respects_pairwise_distance() -> None:
    rng = spawn_rng(9002, "color_palette_no_anchor")
    palette = sample_color_palette_with_distance_constraints(
        rng,
        palette_size=4,
        channel_min=0,
        channel_max=255,
        min_distance=45.0,
        max_attempts=1024,
        distance_space="lab",
    )
    assert len(palette) == 4
    for left in range(len(palette)):
        for right in range(left + 1, len(palette)):
            assert color_distance(palette[left], palette[right], distance_space="lab") >= 45.0


def test_palette_sampling_impossible_constraints_is_deterministic() -> None:
    rng_a = spawn_rng(9003, "color_palette_impossible")
    palette_a = sample_color_palette_with_distance_constraints(
        rng_a,
        palette_size=3,
        channel_min=120,
        channel_max=122,
        anchor_colors=((121, 121, 121),),
        min_distance=120.0,
        max_attempts=12,
        distance_space="lab",
    )
    rng_b = spawn_rng(9003, "color_palette_impossible")
    palette_b = sample_color_palette_with_distance_constraints(
        rng_b,
        palette_size=3,
        channel_min=120,
        channel_max=122,
        anchor_colors=((121, 121, 121),),
        min_distance=120.0,
        max_attempts=12,
        distance_space="lab",
    )
    assert palette_a == palette_b
    assert len(palette_a) == 3


def test_single_color_sampling_uses_global_default_min_distance() -> None:
    rng = spawn_rng(9004, "single_color_default_distance")
    anchors = ((250, 250, 250), (220, 230, 240), (120, 130, 145))
    sampled = sample_color_with_distance_constraints(
        rng,
        channel_min=8,
        channel_max=172,
        anchor_colors=anchors,
        max_attempts=1024,
        distance_space="lab",
    )
    assert all(color_distance(sampled, anchor, distance_space="lab") >= 60.0 for anchor in anchors)


def test_rgb_normalization_and_coercion() -> None:
    assert normalize_rgb([300, -4, 12.8]) == (255, 0, 12)
    assert coerce_rgb([301, -1, 42, 99], (1, 2, 3)) == (255, 0, 42)
    assert coerce_rgb("not-rgb", (11, 12, 13)) == (11, 12, 13)
