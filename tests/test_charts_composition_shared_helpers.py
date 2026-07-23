"""Tests for shared composition-chart helper primitives."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.charts.shared.composition.palette import (
    composition_hsv_color,
    darken_rgb,
    lighten_rgb,
)
from trace_tasks.tasks.charts.shared.composition.values import (
    count_from_percent_share,
    counts_from_percent_shares,
    int_sum,
    select_unique_extremum,
    select_unique_nearest,
)


def test_composition_value_helpers_convert_percent_counts() -> None:
    assert count_from_percent_share(1200, 35) == 420
    assert counts_from_percent_shares(900, {"A": 10, "B": 25}) == {"A": 90, "B": 225}
    assert int_sum([1, "2", 3]) == 6


def test_select_unique_extremum_enforces_ties_and_margin() -> None:
    rows = (("A", 12), ("B", 20), ("C", 15))
    largest = select_unique_extremum(rows, select_largest=True, error_label="demo")
    smallest = select_unique_extremum(rows, select_largest=False, error_label="demo")
    assert largest.item == "B"
    assert largest.value == 20
    assert largest.margin == 5
    assert smallest.item == "A"
    assert smallest.value == 12

    with pytest.raises(ValueError, match="tied"):
        select_unique_extremum((("A", 20), ("B", 20)), select_largest=True, error_label="demo")
    with pytest.raises(ValueError, match="margin"):
        select_unique_extremum(rows, select_largest=True, min_margin=8, error_label="demo")


def test_select_unique_nearest_enforces_ties_and_margin() -> None:
    rows = (("A", 12), ("B", 20), ("C", 31))
    selected = select_unique_nearest(
        rows,
        value_fn=lambda item: int(item[1]),
        target_value=22,
        min_margin=1,
        error_label="demo",
    )
    assert selected.item == ("B", 20)
    assert selected.value == 20
    assert selected.distance == 2
    assert selected.margin == 7

    with pytest.raises(ValueError, match="tied"):
        select_unique_nearest(
            (("A", 18), ("B", 26)),
            value_fn=lambda item: int(item[1]),
            target_value=22,
            min_margin=1,
            error_label="demo",
        )


def test_composition_palette_helpers_are_deterministic() -> None:
    assert lighten_rgb((100, 150, 250), 0.25) == (139, 176, 251)
    assert darken_rgb((100, 150, 250), 0.25) == (75, 112, 188)
    first = composition_hsv_color(
        1,
        5,
        instance_seed=123,
        namespace="tests.composition.palette",
        saturation_base=0.45,
        saturation_jitter=0.18,
        value_base=0.74,
        value_jitter=0.12,
    )
    second = composition_hsv_color(
        1,
        5,
        instance_seed=123,
        namespace="tests.composition.palette",
        saturation_base=0.45,
        saturation_jitter=0.18,
        value_base=0.74,
        value_jitter=0.12,
    )
    assert first == second
    assert len(first) == 3
    assert all(0 <= channel <= 255 for channel in first)
