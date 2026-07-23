"""Tests for shared games visual option layout helpers."""

from __future__ import annotations

from trace_tasks.tasks.games.shared.option_layout import balanced_option_grid_spec, option_grid_position


def _row_counts(option_count: int) -> tuple[int, ...]:
    spec = balanced_option_grid_spec(option_count)
    counts: list[int] = []
    for index in range(option_count):
        row, _col, _left, _top = option_grid_position(
            index,
            option_count,
            left=0.0,
            top=0.0,
            item_width=100.0,
            item_height=50.0,
            gap_x=10.0,
            gap_y=12.0,
        )
        if row >= len(counts):
            counts.append(0)
        counts[row] += 1
    return tuple(counts)


def test_balanced_games_option_grid_uses_two_by_two_for_four_options() -> None:
    assert balanced_option_grid_spec(4).columns == 2
    assert _row_counts(4) == (2, 2)


def test_balanced_games_option_grid_keeps_three_by_two_for_six_options() -> None:
    assert balanced_option_grid_spec(6).columns == 3
    assert _row_counts(6) == (3, 3)


def test_balanced_games_option_grid_centers_short_final_rows() -> None:
    spec = balanced_option_grid_spec(5)
    first_row = [
        option_grid_position(
            index,
            5,
            left=0.0,
            top=0.0,
            item_width=100.0,
            item_height=50.0,
            gap_x=10.0,
            gap_y=12.0,
        )[2]
        for index in range(3)
    ]
    final_row = [
        option_grid_position(
            index,
            5,
            left=0.0,
            top=0.0,
            item_width=100.0,
            item_height=50.0,
            gap_x=10.0,
            gap_y=12.0,
        )[2]
        for index in range(3, 5)
    ]

    assert spec.columns == 3
    assert first_row == [0.0, 110.0, 220.0]
    assert final_row == [55.0, 165.0]

