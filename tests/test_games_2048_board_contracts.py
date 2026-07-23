"""Contract tests for games 2048-board tasks."""

from __future__ import annotations

import inspect
from importlib import import_module
from pathlib import Path

import pytest

from trace_tasks.core.source_layout_policy import parse_public_task_id

_mechanics = import_module("trace_tasks.tasks.games.2048.shared.rules")
_state = import_module("trace_tasks.tasks.games.2048.shared.state")
_max_tile_task = import_module("trace_tasks.tasks.games.2048.max_tile_value")
_merge_task = import_module("trace_tasks.tasks.games.2048.merge_count")
_move_result_task = import_module("trace_tasks.tasks.games.2048.move_result_board_label")

SUPPORTED_2048_STYLE_VARIANTS = _state.SUPPORTED_2048_STYLE_VARIANTS
board_max_tile = _mechanics.board_max_tile
coord_to_cell_id = _state.coord_to_cell_id
simulate_2048_move = _mechanics.simulate_2048_move

Games2048MaxTileValueTask = _max_tile_task.Games2048MaxTileValueTask
Games2048MergeCountTask = _merge_task.Games2048MergeCountTask
Games2048MoveResultBoardLabelTask = _move_result_task.Games2048MoveResultBoardLabelTask


def test_games_2048_scene_package_source_layout() -> None:
    expected_sources = {
        Games2048MaxTileValueTask: Path("src/trace_tasks/tasks/games/2048/max_tile_value.py"),
        Games2048MergeCountTask: Path("src/trace_tasks/tasks/games/2048/merge_count.py"),
        Games2048MoveResultBoardLabelTask: Path("src/trace_tasks/tasks/games/2048/move_result_board_label.py"),
    }

    for task_cls, relative_path in expected_sources.items():
        source_path = Path(inspect.getsourcefile(task_cls) or "").resolve()
        assert source_path == (Path.cwd() / relative_path).resolve()
        assert parse_public_task_id(str(task_cls.task_id)).scene_id == "2048"


def _board(rows: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """Return a trace board as immutable rows."""

    return tuple(tuple(int(value) for value in row) for row in rows)


def _source_ids_for_max(result) -> tuple[str, ...]:
    """Return source cell ids for the unique max tile after a move."""

    max_value = board_max_tile(result.after)
    max_cells = [
        coord
        for coord, sources in result.result_sources.items()
        if int(result.after[coord[0]][coord[1]]) == int(max_value) and sources
    ]
    return tuple(coord_to_cell_id(coord) for cell in max_cells for coord in result.result_sources[cell])


def _bbox_row_counts(bboxes: dict[str, list[float]], *, tolerance_px: float = 4.0) -> tuple[int, ...]:
    centers = sorted((float(bbox[1]) + float(bbox[3])) / 2.0 for bbox in bboxes.values())
    rows: list[list[float]] = []
    for center_y in centers:
        if rows and abs(float(rows[-1][0]) - center_y) <= float(tolerance_px):
            rows[-1].append(center_y)
        else:
            rows.append([center_y])
    return tuple(len(row) for row in rows)


def _bbox_center(bbox: list[float]) -> list[float]:
    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
    ]


@pytest.mark.parametrize(
    ("task_cls", "params", "expected_prompt_query", "expected_answer"),
    (
        (Games2048MergeCountTask, {"target_answer": 3}, "merge_count", 3),
        (Games2048MaxTileValueTask, {"target_answer": 128}, "max_tile_value", 128),
    ),
)
def test_games_2048_move_result_value_emits_expected_contract(
    task_cls,
    params: dict[str, int | str],
    expected_prompt_query: str,
    expected_answer: int,
) -> None:
    out = task_cls().generate(204801, params=params, max_attempts=128)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == int(expected_answer)
    expected_annotation_type = "segment_set" if expected_prompt_query == "merge_count" else "bbox_set"
    assert out.annotation_gt.type == expected_annotation_type
    assert out.query_id == "single"
    assert out.scene_id == "2048"
    assert trace["query_spec"]["params"]["query_id"] == "single"
    assert execution["query_id"] == "single"
    assert trace["query_spec"]["prompt_variant"]["selected_keys"]["query"] == expected_prompt_query
    if expected_annotation_type == "segment_set":
        assert trace["projected_annotation"]["type"] == "segment_set"
        assert trace["projected_annotation"]["segment_set"] == out.annotation_gt.value
        assert len(out.annotation_gt.value) == len(execution["move_result"]["merge_pairs"])
        assert execution["annotation_entity_id_pairs"]
        assert len(execution["annotation_entity_id_pairs"]) == len(out.annotation_gt.value)
    else:
        assert trace["projected_annotation"]["type"] == "bbox_set"
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert len(execution["annotation_entity_ids"]) == len(out.annotation_gt.value)
    assert trace["render_spec"]["canvas_width"] <= 900
    assert trace["render_spec"]["canvas_height"] <= 900
    assert float(trace["render_spec"]["effective_cell_size_px"]) >= 28.0
    assert trace["render_spec"]["text_style"]["font_family"]
    assert trace["render_map"]["font_family"] == trace["render_spec"]["text_style"]["font_family"]


def test_games_2048_max_tile_annotation_uses_source_cells_for_unique_max() -> None:
    out = Games2048MaxTileValueTask().generate(
        204812,
        params={"target_answer": 256, "move_direction": "up"},
        max_attempts=128,
    )
    execution = out.trace_payload["execution_trace"]
    result = simulate_2048_move(_board(execution["board_before"]), str(execution["move_direction"]))

    assert int(out.answer_gt.value) == board_max_tile(result.after) == 256
    assert out.annotation_gt.type == "bbox_set"
    assert tuple(execution["annotation_entity_ids"]) == _source_ids_for_max(result)
    assert len(out.annotation_gt.value) == 2


def test_games_2048_move_result_board_label_has_unique_option_board() -> None:
    out = Games2048MoveResultBoardLabelTask().generate(
        204841,
        params={"target_label": "F", "move_direction": "left"},
        max_attempts=128,
    )
    execution = out.trace_payload["execution_trace"]
    board = _board(execution["board_before"])
    result = simulate_2048_move(board, str(execution["move_direction"]))
    options = {
        str(label): _board(option_board)
        for label, option_board in execution["result_option_boards"].items()
    }
    matching_labels = [label for label, option_board in options.items() if option_board == result.after]

    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "F"
    assert out.annotation_gt.type == "bbox"
    assert matching_labels == ["F"]
    assert tuple(execution["annotation_entity_ids"]) == ("result_option_F",)
    assert out.trace_payload["projected_annotation"]["type"] == "bbox"
    assert out.trace_payload["projected_annotation"]["bbox"] == out.annotation_gt.value


def test_games_2048_move_result_board_label_lays_out_four_options_as_two_by_two() -> None:
    out = Games2048MoveResultBoardLabelTask().generate(
        204842,
        params={"target_label": "D", "move_direction": "left", "result_board_option_count": 4},
        max_attempts=128,
    )
    option_bboxes = out.trace_payload["render_map"]["result_option_bboxes_px"]

    assert set(option_bboxes) == {"A", "B", "C", "D"}
    assert _bbox_row_counts(option_bboxes) == (2, 2)


def test_games_2048_query_cycles_cover_supports() -> None:
    value_tasks = (Games2048MergeCountTask(), Games2048MaxTileValueTask())
    result_board_task = Games2048MoveResultBoardLabelTask()
    runtime_query_ids: set[str] = set()
    answers_by_task: dict[str, set[int]] = {
        Games2048MergeCountTask.task_id: set(),
        Games2048MaxTileValueTask.task_id: set(),
    }
    styles: set[str] = set()
    result_labels: set[str] = set()

    for sampling_index in range(240):
        value_task = value_tasks[int(sampling_index) % len(value_tasks)]
        out = value_task.generate(
            204900 + int(sampling_index),
            params={},
            max_attempts=128,
        )
        execution = out.trace_payload["execution_trace"]
        runtime_query_ids.add(str(out.query_id))
        answers_by_task[str(value_task.task_id)].add(int(out.answer_gt.value))
        styles.add(str(execution["style_variant"]))

    for sampling_index in range(72):
        out = result_board_task.generate(
            205600 + int(sampling_index),
            params={"_sample_cursor": int(sampling_index)},
            max_attempts=128,
        )
        result_labels.add(str(out.answer_gt.value))

    assert runtime_query_ids == {"single"}
    assert answers_by_task[Games2048MergeCountTask.task_id] == {0, 1, 2, 3, 4}
    assert answers_by_task[Games2048MaxTileValueTask.task_id] == {16, 32, 64, 128, 256}
    assert styles == set(SUPPORTED_2048_STYLE_VARIANTS)
    assert result_labels == set("ABCDEF")


def test_games_2048_generation_is_deterministic() -> None:
    params = {"target_answer": 64, "move_direction": "down"}
    task = Games2048MaxTileValueTask()
    out_a = task.generate(204831, params=params, max_attempts=128)
    out_b = task.generate(204831, params=params, max_attempts=128)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()
