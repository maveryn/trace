"""Tests for shared sequence helpers."""

from __future__ import annotations

from trace_tasks.tasks.shared.sequence import rotate_sequence


def test_rotate_sequence_basic_behavior() -> None:
    assert rotate_sequence([1, 2, 3, 4], shift=1) == [2, 3, 4, 1]
    assert rotate_sequence([1, 2, 3, 4], shift=4) == [1, 2, 3, 4]
    assert rotate_sequence([1, 2, 3, 4], shift=-1) == [4, 1, 2, 3]


def test_rotate_sequence_empty() -> None:
    assert rotate_sequence([], shift=5) == []
