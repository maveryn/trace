"""Contract tests for games Bubble-shooter board tasks."""

from __future__ import annotations

from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.games.bubble_shooter.drop_count import GamesBubbleShooterDropCountTask
from trace_tasks.tasks.games.bubble_shooter.pop_color_label import (
    GamesBubbleShooterPopColorLabelTask,
)
from trace_tasks.tasks.games.bubble_shooter.pop_count import GamesBubbleShooterPopCountTask
from trace_tasks.tasks.games.bubble_shooter.pop_target_label import (
    GamesBubbleShooterPopTargetLabelTask,
)
from trace_tasks.tasks.games.bubble_shooter.shared.rules import (
    compute_shot_outcome,
    is_playable_landing_coord,
    sorted_coords,
)
from trace_tasks.tasks.games.bubble_shooter.shared.state import (
    BUBBLE_OPTION_LABELS,
    bubble_entity_id,
    landing_option_entity_id,
    landing_slot_entity_id,
)
from tests.helpers import read_jsonl


def _board_from_trace(execution: dict) -> tuple[tuple[str | None, ...], ...]:
    rows = int(execution["row_count"])
    cols = int(execution["col_count"])
    values: list[list[str | None]] = [
        [None for _col in range(cols)] for _row in range(rows)
    ]
    for bubble in execution["board_bubbles"]:
        row, col = [int(value) for value in bubble["coord"]]
        values[row][col] = str(bubble["color_key"])
    return tuple(tuple(row) for row in values)


def _assert_trace_landing_is_playable(execution: dict) -> None:
    board = _board_from_trace(execution)
    landing = tuple(int(value) for value in execution["landing_coord"])
    assert is_playable_landing_coord(board, landing)


def test_games_bubble_shooter_rejects_enclosed_landing_slot() -> None:
    board = (
        ("red", "yellow", "blue", "green", "red"),
        ("yellow", None, "green", "blue", "yellow"),
        ("blue", "green", "red", "yellow", "blue"),
        (None, None, None, None, None),
        (None, None, None, None, None),
    )

    assert not is_playable_landing_coord(board, (1, 1))


@pytest.mark.parametrize(
    ("task_cls", "params", "expected_prompt_query", "answer_type", "annotation_type"),
    (
        (
            GamesBubbleShooterPopCountTask,
            {"target_answer": 5, "row_count": 7, "col_count": 9},
            "pop_count",
            "integer",
            "bbox_set",
        ),
        (
            GamesBubbleShooterDropCountTask,
            {"target_answer": 4, "row_count": 7, "col_count": 9},
            "drop_count",
            "integer",
            "bbox_set",
        ),
        (
            GamesBubbleShooterPopColorLabelTask,
            {"target_label": "E", "option_count": 6},
            "pop_color_label",
            "string",
            "bbox_set",
        ),
        (
            GamesBubbleShooterPopTargetLabelTask,
            {"target_label": "D", "positive_pop_count": 4},
            "pop_target_label",
            "string",
            "bbox",
        ),
    ),
)
def test_games_bubble_shooter_public_tasks_emit_expected_contract(
    task_cls,
    params: dict[str, int | str],
    expected_prompt_query: str,
    answer_type: str,
    annotation_type: str,
) -> None:
    out = task_cls().generate(102000, params=params, max_attempts=256)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    _assert_trace_landing_is_playable(execution)
    assert out.answer_gt.type == answer_type
    assert out.annotation_gt.type == annotation_type
    assert out.query_id == "single"
    assert out.scene_id == "bubble_shooter"
    assert trace["query_spec"]["query_id"] == "single"
    assert trace["query_spec"]["params"]["query_id"] == "single"
    assert execution["query_id"] == "single"
    assert (
        trace["query_spec"]["prompt_variant"]["selected_keys"]["query"]
        == expected_prompt_query
    )
    assert trace["projected_annotation"]["type"] == annotation_type
    assert trace["render_spec"]["canvas_width"] <= 980
    assert trace["render_spec"]["canvas_height"] <= 820
    assert trace["render_spec"]["panel_scene_style"]["treatment"]
    assert trace["render_spec"]["text_style"]["font_family"]
    assert (
        trace["render_map"]["font_family"]
        == trace["render_spec"]["text_style"]["font_family"]
    )
    assert (
        float(trace["render_map"]["guide_color_safety"]["guide_anchor_lab_distance"])
        >= 40.0
    )
    if annotation_type == "bbox_set":
        assert len(execution["annotation_entity_ids"]) == len(out.annotation_gt.value)
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert (
            trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value
        )
        expected_bboxes = [
            trace["render_map"]["entity_bboxes_px"][str(entity_id)]
            for entity_id in execution["annotation_entity_ids"]
        ]
        assert out.annotation_gt.value == expected_bboxes
        boxes = out.annotation_gt.value
    else:
        assert len(execution["annotation_entity_ids"]) == 1
        assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_bbox"] == out.annotation_gt.value
        assert (
            out.annotation_gt.value
            == trace["render_map"]["entity_bboxes_px"][
                execution["annotation_entity_ids"][0]
            ]
        )
        boxes = [out.annotation_gt.value]
    for x0, y0, x1, y1 in boxes:
        assert (
            0 <= float(x0) <= float(x1) <= float(trace["render_spec"]["canvas_width"])
        )
        assert (
            0 <= float(y0) <= float(y1) <= float(trace["render_spec"]["canvas_height"])
        )


def test_games_bubble_shooter_pop_count_matches_computed_outcome() -> None:
    out = GamesBubbleShooterPopCountTask().generate(
        102010,
        params={"target_answer": 5, "row_count": 7, "col_count": 10},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    _assert_trace_landing_is_playable(execution)
    board = _board_from_trace(execution)
    landing = tuple(int(value) for value in execution["landing_coord"])
    outcome = compute_shot_outcome(
        board, landing_coord=landing, color_key=str(execution["shooter_color_key"])
    )
    annotation_coords = sorted_coords(
        tuple(int(part) for part in entity_id.replace("bubble_r", "").split("_c"))
        for entity_id in execution["annotation_entity_ids"]
    )

    assert int(out.answer_gt.value) == len(outcome.popped_coords) == 5
    assert annotation_coords == outcome.popped_coords
    assert set(execution["annotation_entity_ids"]) == {
        bubble_entity_id(coord) for coord in outcome.popped_coords
    }


def test_games_bubble_shooter_pop_count_prompt_excludes_non_board_bubbles() -> None:
    out = GamesBubbleShooterPopCountTask().generate(
        102012,
        params={"target_answer": 3, "row_count": 7, "col_count": 9},
        max_attempts=256,
    )
    prompt = str(out.prompt).lower()

    assert "do not count the shot bubble" in prompt
    assert "placed shot bubble" not in prompt


def test_games_bubble_shooter_pop_count_allows_zero_pop_case() -> None:
    out = GamesBubbleShooterPopCountTask().generate(
        102015,
        params={"target_answer": 0, "row_count": 7, "col_count": 9},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    _assert_trace_landing_is_playable(execution)
    board = _board_from_trace(execution)
    landing = tuple(int(value) for value in execution["landing_coord"])
    outcome = compute_shot_outcome(
        board, landing_coord=landing, color_key=str(execution["shooter_color_key"])
    )

    assert int(out.answer_gt.value) == len(outcome.popped_coords) == 0
    assert execution["annotation_entity_ids"] == []
    assert out.annotation_gt.value == []


def test_games_bubble_shooter_drop_count_matches_computed_outcome() -> None:
    out = GamesBubbleShooterDropCountTask().generate(
        102020,
        params={"target_answer": 4, "row_count": 7, "col_count": 10},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    _assert_trace_landing_is_playable(execution)
    board = _board_from_trace(execution)
    landing = tuple(int(value) for value in execution["landing_coord"])
    outcome = compute_shot_outcome(
        board, landing_coord=landing, color_key=str(execution["shooter_color_key"])
    )

    assert int(out.answer_gt.value) == len(outcome.dropped_coords) == 4
    assert set(execution["annotation_entity_ids"]) == {
        bubble_entity_id(coord) for coord in outcome.dropped_coords
    }


def test_games_bubble_shooter_drop_count_allows_zero_drop_case() -> None:
    out = GamesBubbleShooterDropCountTask().generate(
        102025,
        params={"target_answer": 0, "row_count": 7, "col_count": 9},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    _assert_trace_landing_is_playable(execution)
    board = _board_from_trace(execution)
    landing = tuple(int(value) for value in execution["landing_coord"])
    outcome = compute_shot_outcome(
        board, landing_coord=landing, color_key=str(execution["shooter_color_key"])
    )

    assert int(out.answer_gt.value) == len(outcome.dropped_coords) == 0
    assert execution["annotation_entity_ids"] == []
    assert out.annotation_gt.value == []


def test_games_bubble_shooter_pop_color_label_has_one_displayed_popping_option() -> (
    None
):
    out = GamesBubbleShooterPopColorLabelTask().generate(
        102030,
        params={"target_label": "F", "option_count": 6, "row_count": 8, "col_count": 9},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    _assert_trace_landing_is_playable(execution)
    board = _board_from_trace(execution)
    landing = tuple(int(value) for value in execution["landing_coord"])
    positives: list[str] = []
    for option in execution["option_specs"]:
        outcome = compute_shot_outcome(
            board, landing_coord=landing, color_key=str(option["color_key"])
        )
        if len(outcome.popped_coords) > 0:
            positives.append(str(option["label"]))

    assert positives == ["F"]
    assert out.answer_gt.value == "F"
    assert landing_slot_entity_id() not in set(execution["annotation_entity_ids"])
    assert not any(
        str(entity_id).startswith("option_")
        for entity_id in execution["annotation_entity_ids"]
    )
    assert set(execution["annotation_entity_ids"]) == {
        bubble_entity_id(coord)
        for coord in compute_shot_outcome(
            board,
            landing_coord=landing,
            color_key=str(execution["outcome"]["color_key"]),
        ).popped_coords
    }


def test_games_bubble_shooter_pop_target_label_has_one_displayed_popping_target() -> (
    None
):
    out = GamesBubbleShooterPopTargetLabelTask().generate(
        102035,
        params={
            "target_label": "D",
            "positive_pop_count": 4,
            "row_count": 8,
            "col_count": 9,
        },
        max_attempts=256,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    _assert_trace_landing_is_playable(execution)
    board = _board_from_trace(execution)
    positives: list[str] = []
    target_coord: tuple[int, int] | None = None
    for option in execution["landing_option_specs"]:
        landing = tuple(int(value) for value in option["coord"])
        outcome = compute_shot_outcome(
            board, landing_coord=landing, color_key=str(execution["shooter_color_key"])
        )
        if outcome.popped_coords:
            positives.append(str(option["label"]))
        if str(option["label"]) == "D":
            target_coord = landing

    assert positives == ["D"]
    assert target_coord is not None
    assert out.answer_gt.value == "D"
    assert tuple(execution["landing_coord"]) == target_coord
    assert len(execution["landing_option_specs"]) == 4
    assert execution["annotation_entity_ids"] == [landing_option_entity_id("D")]
    assert out.annotation_gt.type == "bbox"
    assert (
        out.annotation_gt.value
        == trace["render_map"]["entity_bboxes_px"][landing_option_entity_id("D")]
    )
    assert not execution["option_specs"]


def test_games_bubble_shooter_pop_color_prompt_excludes_non_board_bubbles() -> None:
    out = GamesBubbleShooterPopColorLabelTask().generate(
        102032,
        params={"target_label": "F", "option_count": 6, "row_count": 8, "col_count": 9},
        max_attempts=256,
    )
    prompt = str(out.prompt).lower()

    assert "placed shot bubble" not in prompt
    assert "selected color option" not in prompt


@pytest.mark.parametrize(
    "task_cls",
    (
        GamesBubbleShooterPopCountTask,
        GamesBubbleShooterDropCountTask,
        GamesBubbleShooterPopColorLabelTask,
        GamesBubbleShooterPopTargetLabelTask,
    ),
)
def test_games_bubble_shooter_fixed_tasks_reject_unsupported_query_id(task_cls) -> None:
    task = task_cls()
    expected_query = str(task.supported_query_ids[0])
    assert (
        task.generate(
            102090, params={"query_id": expected_query}, max_attempts=256
        ).query_id
        == expected_query
    )
    assert (
        task.generate(102090, params={"query_id": "default"}, max_attempts=256).query_id
        == expected_query
    )
    with pytest.raises(ValueError, match="unsupported query_id"):
        task.generate(
            102090, params={"query_id": "__unsupported_query_id__"}, max_attempts=256
        )


def test_games_bubble_shooter_task_axes_cover_support() -> None:
    queries: set[str] = set()
    rows: set[int] = set()
    cols: set[int] = set()
    labels: set[str] = set()
    counts: set[int] = set()

    for sampling_index in range(180):
        for task_cls in (
            GamesBubbleShooterPopCountTask,
            GamesBubbleShooterDropCountTask,
            GamesBubbleShooterPopColorLabelTask,
            GamesBubbleShooterPopTargetLabelTask,
        ):
            out = task_cls().generate(
                102100 + sampling_index,
                params={},
                max_attempts=256,
            )
            execution = out.trace_payload["execution_trace"]
            _assert_trace_landing_is_playable(execution)
            queries.add(str(out.query_id))
            rows.add(int(execution["row_count"]))
            cols.add(int(execution["col_count"]))
            if out.answer_gt.type == "string":
                labels.add(str(out.answer_gt.value))
            else:
                counts.add(int(out.answer_gt.value))

    assert queries == {"single"}
    assert rows <= {7, 8, 9}
    assert 7 in rows
    assert cols == {8, 9, 10}
    assert labels == set(BUBBLE_OPTION_LABELS)
    assert counts >= {0, 1, 2, 3, 4}


def test_games_bubble_shooter_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__bubble_shooter"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__bubble_shooter",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_games__bubble_shooter__pop_count", count=2, params={}
            ),
            BuildTaskConfig(
                task_id="task_games__bubble_shooter__pop_color_label",
                count=1,
                params={},
            ),
            BuildTaskConfig(
                task_id="task_games__bubble_shooter__pop_target_label",
                count=1,
                params={},
            ),
        ],
        max_attempts_per_instance=256,
        workers=1,
    )
    final_path = build_dataset(config, code_hash="games-bubble-shooter-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")

    assert len(rows) == 4
    assert all(row["domain"] == "games" for row in rows)
    assert all(
        str(row["task"]).startswith("task_games__bubble_shooter__") for row in rows
    )
