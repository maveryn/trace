"""Identity rule (DESIGN 2.3, D4), frames (2.2) and source-pixel annotation views."""

from __future__ import annotations

import pytest

from trace_tasks.experimental.certificates.geometry import (
    Frame,
    annotation_view,
    round_box_half_up,
    round_half_up,
    scale_annotation,
    shown_frame,
    shown_size,
    smart_resize,
    source_frame,
)
from trace_tasks.experimental.certificates.identity import (
    AMBIGUOUS,
    DUPLICATE,
    INVALID,
    RESOLVED,
    UNRESOLVED,
    IdentityParams,
    box_matches,
    dedupe,
    match_stats,
    resolve_box,
    resolve_boxes,
    resolve_point,
    set_f1,
)

LEVER_BLOCK = (499.0, 278.5, 557.0, 330.5)  # 58 x 52, DESIGN 3.3 record 3
LEVER_RIGHT = (565.0, 278.5, 623.0, 330.5)  # neighbour 8 px to the right
LEVER_LEFT = (433.0, 278.5, 491.0, 330.5)


def shifted(box, dx=0.0, dy=0.0):
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def table_column(x0=414.0, x1=532.0, top=125.0, height=56.0, rows=6):
    return [(x0, top + r * height, x1, top + (r + 1) * height) for r in range(rows)]


# ------------------------------------------------------------------------------------------
# box rule


@pytest.mark.parametrize("sign", [1, -1])
def test_horizontal_shift_sweep_isolated_block(sign: int) -> None:
    """IoU of a same-size box shifted by dx is (w-dx)/(w+dx) >= 0.5 iff dx <= w/3 (19.3 px)."""
    for dx in range(0, 41):
        resolution = resolve_box(shifted(LEVER_BLOCK, dx=sign * dx), [LEVER_BLOCK])
        assert resolution.resolved == (dx <= 19), dx
        assert resolution.iou == pytest.approx((58 - dx) / (58 + dx) if dx < 58 else 0.0)


def test_vertical_shift_sweep() -> None:
    for dy in range(0, 30):
        assert resolve_box(shifted(LEVER_BLOCK, dy=dy), [LEVER_BLOCK]).resolved == (dy <= 17), dy


def test_shift_toward_neighbour_never_ambiguous() -> None:
    candidates = [LEVER_LEFT, LEVER_BLOCK, LEVER_RIGHT]
    for dx in range(0, 90):
        resolution = resolve_box(shifted(LEVER_BLOCK, dx=dx), candidates)
        assert resolution.status != AMBIGUOUS
        if dx <= 19:
            assert resolution.candidate == 1
        elif dx < 47:
            assert resolution.status == UNRESOLVED, dx
        elif dx <= 85:
            assert resolution.candidate == 2, dx


def test_straddling_table_rows_do_not_resolve() -> None:
    column = table_column()
    half = (414.0, 153.0, 532.0, 209.0)  # half in row 0, half in row 1: center on the boundary
    assert resolve_box(half, column).status == UNRESOLVED
    mostly_row0 = (414.0, 145.0, 532.0, 201.0)  # 36/56 in row 0: IoU 0.47, containment 0.64
    resolution = resolve_box(mostly_row0, column)
    assert resolution.status == UNRESOLVED and resolution.best_candidate == 0
    assert resolution.iou == pytest.approx(36 / 76) and resolution.containment == pytest.approx(36 / 56)
    nudged = (414.0, 135.0, 532.0, 191.0)  # 10 px into row 1: IoU 46/66
    assert resolve_box(nudged, column).candidate == 0
    two_rows = (414.0, 125.0, 532.0, 237.0)  # covers rows 0 and 1 exactly
    assert resolve_box(two_rows, column).status == UNRESOLVED


def test_digit_tight_box_resolves_by_containment() -> None:
    cell = (414.0, 125.0, 532.0, 181.0)
    digits = (460.0, 140.0, 486.0, 166.0)
    stats = match_stats(digits, cell)
    assert stats.iou < 0.5 and stats.containment == 1.0 and stats.center_inside
    assert resolve_box(digits, table_column()).candidate == 0
    poking_out = (460.0, 160.0, 486.0, 197.0)  # center 178.5 inside row 0, containment 21/37
    assert resolve_box(poking_out, table_column()).status == UNRESOLVED


def test_chess_cell_sized_box_and_oversized_box() -> None:
    piece = (342.0, 423.0, 376.0, 457.0)  # 34 x 34 piece box
    cell_box = (336.6, 417.6, 381.4, 462.4)  # 44.8 px square around it: IoU 0.576
    assert match_stats(cell_box, piece).iou == pytest.approx(1156 / 44.8**2)
    assert resolve_box(cell_box, [piece]).resolved
    oversized = (331.0, 412.0, 387.0, 468.0)  # 56 px square: IoU 0.37, containment 0.37
    assert resolve_box(oversized, [piece]).status == UNRESOLVED


def test_center_must_be_strictly_inside() -> None:
    candidate = (0.0, 0.0, 100.0, 100.0)
    # covers the left half of the candidate plus as much outside: center exactly on the left edge
    assert not box_matches((-50.0, 0.0, 50.0, 100.0), candidate)
    # tiny box inside but centred on the edge is rejected even with containment 0.5
    assert not box_matches((-1.0, 10.0, 1.0, 12.0), candidate)


def test_ambiguous_when_two_candidates_match() -> None:
    a = (0.0, 0.0, 100.0, 100.0)
    b = (10.0, 10.0, 110.0, 110.0)  # overlapping candidates (never happens in our scenes)
    resolution = resolve_box((5.0, 5.0, 105.0, 105.0), [a, b])
    assert resolution.status == AMBIGUOUS and resolution.hits == (0, 1) and resolution.candidate is None


def test_invalid_box() -> None:
    assert resolve_box(None, [LEVER_BLOCK]).status == INVALID
    frame = Frame("px_source", 1.0, 1.0)
    assert frame.box([1, 1, 1, 5]) is None
    assert frame.box([True, 1, 2, 5]) is None
    assert frame.box([1, 2, 3]) is None
    assert frame.box([3, 4, 1, 2]) == (1.0, 2.0, 3.0, 4.0)


def test_threshold_sensitivity_parameters() -> None:
    candidate = (0.0, 0.0, 100.0, 100.0)
    box_iou_06 = (0.0, 0.0, 100.0, 60.0)  # IoU 0.6, containment 1.0 -> containment branch
    assert resolve_box(box_iou_06, [candidate]).resolved
    strict = IdentityParams(iou_threshold=0.7, containment_threshold=1.01)
    assert not resolve_box(box_iou_06, [candidate], strict).resolved
    loose = IdentityParams(iou_threshold=0.3, containment_threshold=1.01)
    offset = (40.0, 0.0, 140.0, 100.0)  # IoU 60/140 = 0.43, containment 0.6
    assert not resolve_box(offset, [candidate]).resolved
    assert resolve_box(offset, [candidate], loose).resolved


# ------------------------------------------------------------------------------------------
# one-to-one dedupe


def test_duplicates_one_to_one_in_increasing_index() -> None:
    candidates = [LEVER_LEFT, LEVER_BLOCK, LEVER_RIGHT]
    boxes = [LEVER_BLOCK, shifted(LEVER_BLOCK, dx=5), LEVER_RIGHT, shifted(LEVER_BLOCK, dy=-4), None]
    resolutions = resolve_boxes(boxes, candidates)
    assert [r.status for r in resolutions] == [RESOLVED, DUPLICATE, RESOLVED, DUPLICATE, INVALID]
    assert resolutions[1].duplicate_of == 0 and resolutions[3].duplicate_of == 0
    assert resolutions[1].candidate == 1
    # order of the input list does not matter, only the annotation index
    reordered = dedupe(list(reversed([resolve_box(b, candidates, index=i) for i, b in enumerate(boxes)])))
    assert [(r.index, r.status) for r in reordered] == [(r.index, r.status) for r in resolutions]


def test_set_f1() -> None:
    candidates = [LEVER_LEFT, LEVER_BLOCK, LEVER_RIGHT]
    assert set_f1(resolve_boxes([LEVER_LEFT, LEVER_BLOCK], candidates), [0, 1]) == 1.0
    assert set_f1(resolve_boxes([LEVER_LEFT, LEVER_LEFT], candidates), [0, 1]) == pytest.approx(0.5)
    assert set_f1(resolve_boxes([LEVER_RIGHT], candidates), [0, 1]) == 0.0
    assert set_f1([], []) == 1.0 and set_f1([], [0]) == 0.0


# ------------------------------------------------------------------------------------------
# points


def test_boundary_points_are_unresolved() -> None:
    tiles = [(0.0, 0.0, 80.0, 80.0), (80.0, 0.0, 160.0, 80.0), (0.0, 80.0, 80.0, 160.0), (80.0, 80.0, 160.0, 160.0)]
    assert resolve_point((40.0, 40.0), tiles).candidate == 0
    assert resolve_point((159.9, 159.9), tiles).candidate == 3
    assert resolve_point((80.0, 40.0), tiles).status == UNRESOLVED  # shared edge
    assert resolve_point((80.0, 80.0), tiles).status == UNRESOLVED  # corner
    assert resolve_point((0.0, 40.0), tiles).status == UNRESOLVED  # outer edge
    assert resolve_point((-5.0, 40.0), tiles).status == UNRESOLVED
    assert resolve_point(None, tiles).status == INVALID
    assert Frame("px_source", 1.0, 1.0).point([True, 3]) is None
    assert Frame("px_source", 1.0, 1.0).point([float("nan"), 3]) is None


# ------------------------------------------------------------------------------------------
# frames


def test_rel1000_conversion() -> None:
    frame = source_frame("rel1000", (1280, 640))
    assert frame.box([500, 500, 600, 600]) == pytest.approx((640.0, 320.0, 768.0, 384.0))
    assert frame.box([600, 600, 500, 500]) == pytest.approx((640.0, 320.0, 768.0, 384.0))
    assert frame.point([1000, 1000]) == pytest.approx((1280.0, 640.0))
    assert frame.point([0, 0]) == (0.0, 0.0)
    rel = [coord * 1000 / size for coord, size in zip(LEVER_BLOCK, (1280, 640, 1280, 640))]
    assert resolve_box(frame.box(rel), [LEVER_LEFT, LEVER_BLOCK, LEVER_RIGHT]).candidate == 1
    assert source_frame("px_source", (1280, 640)).is_identity
    with pytest.raises(ValueError):
        source_frame("normalized", (1280, 640))


def test_scale_annotation_preserves_structure() -> None:
    frame = source_frame("rel1000", (2000, 1000))
    value = {"filter_cells": [[100, 100, 200, 200]], "target_cells": [[1, 2, 3], "x"]}
    scaled = scale_annotation(value, "bbox_set_map", frame)
    assert scaled == {"filter_cells": [[200.0, 100.0, 400.0, 200.0]], "target_cells": [[1, 2, 3], "x"]}
    assert scale_annotation([[500, 500]], "point_set", frame) == [[1000.0, 500.0]]
    identity = source_frame("px_source", (2000, 1000))
    assert scale_annotation(value, "bbox_set_map", identity) is value
    deep = [[[[[[[[[[1, 2, 3, 4]]]]]]]]]]
    assert scale_annotation(deep, "bbox_set", frame) is not None


def test_smart_resize_matches_qwen25_reference() -> None:
    # reference values from trace_tool.tools.base.smart_resize(factor=28, 262144, 4194304)
    cases = {
        (726, 718): (728, 728),
        (640, 1280): (644, 1288),
        (900, 940): (896, 952),
        (640, 960): (644, 952),
        (960, 640): (952, 644),
        (800, 800): (812, 812),
        (3000, 4000): (1764, 2352),
        (200, 300): (420, 644),
    }
    for (height, width), expected in cases.items():
        assert smart_resize(height, width) == expected
    assert shown_size((718, 726)) == (728, 728)
    frame = shown_frame((718, 726))
    assert frame.scale_x == pytest.approx(718 / 728) and frame.scale_y == pytest.approx(726 / 728)


def test_shown_frame_maps_shown_coordinates_back() -> None:
    width, height = 718, 726
    frame = shown_frame((width, height))
    shown_w, shown_h = shown_size((width, height))
    piece = (342.0, 423.0, 376.0, 457.0)
    scales = (shown_w / width, shown_h / height) * 2
    as_shown = [coord * scale for coord, scale in zip(piece, scales)]
    assert frame.box(as_shown) == pytest.approx(piece)


# ------------------------------------------------------------------------------------------
# annotation views


def test_annotation_view_keeps_indices_of_malformed_elements() -> None:
    frame = source_frame("px_source", (1000, 1000))
    view = annotation_view([[1, 2, 3, 4], [5, 5, 5, 9], "x", [9, 9, 1, 1]], "bbox_set", frame)
    assert view.structure_ok and view.size() == 4
    assert view.items() == ((1.0, 2.0, 3.0, 4.0), None, None, (1.0, 1.0, 9.0, 9.0))
    assert annotation_view([1, 2, 3, 4], "bbox_set", frame).items() == ((1.0, 2.0, 3.0, 4.0),)
    assert annotation_view({"bboxes": [[1, 2, 3, 4]]}, "bbox_set", frame).size() == 1
    for bad in (None, "[[1,2,3,4]]", 7, {"x": 1}):
        bad_view = annotation_view(bad, "bbox_set", frame)
        assert not bad_view.structure_ok and bad_view.size() == 0


def test_annotation_view_maps() -> None:
    frame = source_frame("rel1000", (2000, 1000))
    value = {"filter_cells": [[0, 0, 10, 10]], " target_cells ": [[0, 0, 10, 10], [1]]}
    view = annotation_view(value, "bbox_set_map", frame)
    assert view.items("filter_cells") == ((0.0, 0.0, 20.0, 10.0),)
    assert view.items("target_cells") == ((0.0, 0.0, 20.0, 10.0), None)
    endpoints = annotation_view({"player_cell": [560, 160, 640, 240], "target_cell": "x"}, "bbox_map", frame)
    assert endpoints.items("player_cell") == ((1120.0, 160.0, 1280.0, 240.0),)
    assert endpoints.items("target_cell") == (None,) and not endpoints.structure_ok
    assert annotation_view([[1, 2, 3, 4]], "bbox_map", frame).roles == {}


def test_round_half_up() -> None:
    assert round_half_up(278.5) == 279
    assert round_half_up(2.5) == 3 and round_half_up(3.5) == 4
    assert round_half_up(0.49999999999999994) == 0
    assert round_half_up(330.4999) == 330
    assert round_box_half_up([499.0, 278.5, 557.0, 330.5]) == [499, 279, 557, 331]
