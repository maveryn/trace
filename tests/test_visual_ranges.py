"""Tests for shared visual-range normalization helpers."""

from __future__ import annotations

from trace_tasks.core.visual.ranges import normalize_int_range, normalize_non_negative_int_range


def test_normalize_int_range_orders_bounds() -> None:
    assert normalize_int_range([9, 2], fallback_min=0, fallback_max=0) == (2, 9)
    assert normalize_int_range(None, fallback_min=3, fallback_max=7) == (3, 7)
    assert normalize_int_range(5, fallback_min=0, fallback_max=0) == (5, 5)


def test_normalize_non_negative_int_range_clamps_at_zero() -> None:
    assert normalize_non_negative_int_range([-4, 2], fallback_min=0, fallback_max=0) == (0, 2)
    assert normalize_non_negative_int_range([-5, -1], fallback_min=0, fallback_max=0) == (0, 0)
