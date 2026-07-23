"""Tests for shared bbox projection helpers."""

from trace_tasks.tasks.shared.bbox_projection import bbox_union, bbox_union_many, bbox_union_raw, round_bbox


def test_round_bbox_uses_three_decimal_default() -> None:
    assert round_bbox([1, 2.34567, 3.4444, 4.5555]) == [1.0, 2.346, 3.444, 4.556]


def test_bbox_union_handles_padding_empty_and_inverted_boxes() -> None:
    assert bbox_union([], padding=3) == [0.0, 0.0, 0.0, 0.0]
    assert bbox_union([[10, 20, 4, 8], [2.1119, 30, 7, 40]], padding=1.5) == [0.612, 6.5, 11.5, 41.5]


def test_bbox_union_many_matches_iterable_adapter() -> None:
    boxes = ([5.1234, 2.0, 9.0, 8.0], [1.0, 4.0, 6.0, 12.0])
    assert bbox_union_many(*boxes) == bbox_union(boxes) == [1.0, 2.0, 9.0, 12.0]


def test_bbox_union_raw_preserves_input_orientation() -> None:
    boxes = ([10, 20, 4, 8], [2.1119, 30, 7, 40])
    assert bbox_union_raw(boxes, padding=1.5) == [0.612, 18.5, 8.5, 41.5]
