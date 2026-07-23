"""Contract tests for toggle-grid puzzle source-layout tasks."""

from __future__ import annotations

from typing import Mapping, Sequence

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks import TASK_REGISTRY, create_task
from trace_tasks.tasks.puzzles.toggle_grid.shared.rules import apply_toggles, toggle_once
from trace_tasks.tasks.puzzles.toggle_grid.shared.state import SCENE_ID
from trace_tasks.tasks.puzzles.toggle_grid.toggle_repair_switch_label import (
    PuzzlesToggleGridToggleRepairSwitchLabelTask,
)
from trace_tasks.tasks.puzzles.toggle_grid.toggle_result_label import (
    PuzzlesToggleGridToggleResultLabelTask,
)

RESULT_TASK_ID = "task_puzzles__toggle_grid__toggle_result_label"
REPAIR_TASK_ID = "task_puzzles__toggle_grid__toggle_repair_switch_label"
_NO_NOISE_PARAMS = {"visual": {"noise": {"apply_prob": 0.0}}}


def _assert_bbox_in_image(bbox: list[float], image_size: tuple[int, int]) -> None:
    """Assert one scalar bbox lies inside the rendered image."""

    assert len(bbox) == 4
    assert 0 <= float(bbox[0]) < float(bbox[2]) <= image_size[0]
    assert 0 <= float(bbox[1]) < float(bbox[3]) <= image_size[1]


def _rgb_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Return simple channel distance between two sampled pixels."""

    return sum(abs(int(a) - int(b)) for a, b in zip(left[:3], right[:3]))


def _center_pixel(image, bbox: Sequence[float]) -> tuple[int, int, int]:
    """Sample the center of one rendered bbox."""

    x = int(round((float(bbox[0]) + float(bbox[2])) / 2.0))
    y = int(round((float(bbox[1]) + float(bbox[3])) / 2.0))
    return tuple(int(value) for value in image.getpixel((x, y))[:3])


def _sample_on_off_pixels(
    *,
    out,
    state: Sequence[Sequence[int]],
    bbox_map: Mapping[str, Sequence[float]],
    excluded_cells: set[tuple[int, int]] | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return representative OFF and ON cell pixels away from labels/markers."""

    excluded = set(excluded_cells or set())
    off_pixel: tuple[int, int, int] | None = None
    on_pixel: tuple[int, int, int] | None = None
    for row_index, row in enumerate(state):
        for col_index, value in enumerate(row):
            if (int(row_index), int(col_index)) in excluded:
                continue
            bbox = bbox_map[f"cell_{row_index}_{col_index}"]
            pixel = _center_pixel(out.image, bbox)
            if int(value):
                on_pixel = on_pixel or pixel
            else:
                off_pixel = off_pixel or pixel
            if on_pixel is not None and off_pixel is not None:
                return off_pixel, on_pixel
    raise AssertionError("could not find both OFF and ON cell pixels")


def test_toggle_grid_tasks_are_registered() -> None:
    """The public ids route to one task file each."""

    assert TASK_REGISTRY[RESULT_TASK_ID] is PuzzlesToggleGridToggleResultLabelTask
    assert TASK_REGISTRY[REPAIR_TASK_ID] is PuzzlesToggleGridToggleRepairSwitchLabelTask

    for task_id in (RESULT_TASK_ID, REPAIR_TASK_ID):
        task = create_task(task_id)
        assert task.domain == "puzzles"
        assert task.supported_query_ids == (SINGLE_QUERY_ID,)
        assert not hasattr(task, "scene_id")


def test_toggle_result_contract() -> None:
    """Result task answer is the option matching sequential toggle simulation."""

    out = create_task(RESULT_TASK_ID).generate(2026053001, params={}, max_attempts=64)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox"
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert len(execution["pressed_cells"]) == 1
    assert [option["option_label"] for option in execution["result_options"]] == [
        "A",
        "B",
        "C",
        "D",
    ]

    recomputed = apply_toggles(
        tuple(tuple(int(value) for value in row) for row in execution["start_state"]),
        tuple(tuple(int(value) for value in cell) for cell in execution["pressed_cells"]),
    )
    assert [list(row) for row in recomputed] == execution["target_state"]
    correct = next(
        option
        for option in execution["result_options"]
        if option["option_label"] == execution["answer_value"]
    )
    assert correct["state"] == execution["target_state"]
    assert bool(correct["is_correct"])
    assert str(out.answer_gt.value) == str(execution["answer_value"])
    selected_bbox = trace["render_map"]["option_panel_bboxes_px"][
        f"option_{execution['answer_value']}"
    ]
    assert out.annotation_gt.value == selected_bbox
    _assert_bbox_in_image(out.annotation_gt.value, out.image.size)


def test_toggle_grid_rendered_on_off_cells_are_visually_distinct() -> None:
    """Both toggle tasks render ON cells with clear contrast from OFF cells."""

    result_out = create_task(RESULT_TASK_ID).generate(
        2026053001,
        params=_NO_NOISE_PARAMS,
        max_attempts=64,
    )
    result_trace = result_out.trace_payload
    result_execution = result_trace["execution_trace"]
    result_pressed = {
        (int(row), int(col)) for row, col in result_execution["pressed_cells"]
    }
    result_off, result_on = _sample_on_off_pixels(
        out=result_out,
        state=result_execution["start_state"],
        bbox_map=result_trace["render_map"]["start_cell_bboxes_px"],
        excluded_cells=result_pressed,
    )
    assert _rgb_distance(result_off, result_on) >= 96

    repair_out = create_task(REPAIR_TASK_ID).generate(
        2026053002,
        params=_NO_NOISE_PARAMS,
        max_attempts=64,
    )
    repair_trace = repair_out.trace_payload
    repair_execution = repair_trace["execution_trace"]
    repair_off, repair_on = _sample_on_off_pixels(
        out=repair_out,
        state=repair_execution["target_state"],
        bbox_map=repair_trace["render_map"]["target_cell_bboxes_px"],
    )
    assert _rgb_distance(repair_off, repair_on) >= 96


def test_toggle_repair_contract() -> None:
    """Repair task answer is the switch whose one press reaches the target grid."""

    out = create_task(REPAIR_TASK_ID).generate(2026053002, params={}, max_attempts=64)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox"
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert [option["option_label"] for option in execution["switch_options"]] == [
        "A",
        "B",
        "C",
        "D",
    ]

    correct = next(
        option
        for option in execution["switch_options"]
        if option["option_label"] == execution["answer_value"]
    )
    recomputed = toggle_once(
        tuple(tuple(int(value) for value in row) for row in execution["start_state"]),
        (int(correct["row"]), int(correct["col"])),
    )
    assert [list(row) for row in recomputed] == execution["target_state"]
    assert bool(correct["is_correct"])
    assert str(out.answer_gt.value) == str(execution["answer_value"])
    selected_bbox = trace["render_map"]["start_cell_bboxes_px"][
        f"cell_{correct['row']}_{correct['col']}"
    ]
    assert out.annotation_gt.value == selected_bbox
    _assert_bbox_in_image(out.annotation_gt.value, out.image.size)
