"""Tests for shared physics vector-arrow helpers."""

from __future__ import annotations

from trace_tasks.tasks.physics.shared.vector_arrows import (
    SEMANTIC_DIRECTION_VECTORS,
    arrow_bbox,
    centered_arrow_endpoints,
    direction_endpoint,
    direction_unit_vector,
)


def test_direction_helpers_use_physics_semantics_with_screen_y_down() -> None:
    assert SEMANTIC_DIRECTION_VECTORS["north"] == (0, 1)
    assert direction_unit_vector("east") == (1.0, -0.0)
    assert direction_unit_vector("north") == (0.0, -1.0)
    assert direction_endpoint((100.0, 100.0), direction="north", length_px=25.0) == (100.0, 75.0)
    assert direction_endpoint((100.0, 100.0), direction="south", length_px=25.0) == (100.0, 125.0)


def test_centered_arrow_endpoints_and_bbox_are_symmetric() -> None:
    start, end = centered_arrow_endpoints((50.0, 50.0), direction="east", length_px=40.0, half_fraction=0.5)

    assert start == (30.0, 50.0)
    assert end == (70.0, 50.0)
    assert arrow_bbox(start, end, padding_px=5.0) == [25.0, 45.0, 75.0, 55.0]
