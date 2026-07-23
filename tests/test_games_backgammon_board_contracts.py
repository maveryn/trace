"""Contract tests for games Backgammon board tasks."""

from __future__ import annotations

from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.games.backgammon.destination_count import (
    DESTINATION_QUERY_SPECS,
    SUPPORTED_QUERY_IDS as DESTINATION_QUERY_IDS,
    GamesBackgammonDestinationCountTask,
)
from trace_tasks.tasks.games.backgammon.pip_count_value import (
    PIP_COUNT_SUPPORT,
    GamesBackgammonPipCountValueTask,
)
from trace_tasks.tasks.games.backgammon.point_state_count import (
    GamesBackgammonPointStateCountTask,
)
from trace_tasks.tasks.games.backgammon.shared.rules import (
    compute_single_die_destinations,
    pip_count_contributions_for_player,
    pip_count_for_player,
    target_destinations_for_status,
    target_points_for_stack_state,
)
from trace_tasks.tasks.games.backgammon.shared.state import (
    PLAYER_BLACK,
    PLAYER_WHITE,
    STACK_STATE_SINGLE,
    STACK_STATE_TWO_OR_MORE,
    SUPPORTED_BACKGAMMON_STYLE_VARIANTS,
    BackgammonPoint,
    point_entity_id,
)
from tests.helpers import read_jsonl


def _points_from_trace(execution: dict) -> dict[int, BackgammonPoint]:
    return {
        int(row["point_id"]): BackgammonPoint(
            owner=None if row["owner"] is None else str(row["owner"]),
            count=int(row["count"]),
        )
        for row in execution["points"]
    }


@pytest.mark.parametrize(
    ("task_cls", "params", "expected_query"),
    (
        (
            GamesBackgammonDestinationCountTask,
            {"target_answer": 4, "query_id": "legal_move_count"},
            "legal_move_count",
        ),
        (
            GamesBackgammonDestinationCountTask,
            {"target_answer": 3, "query_id": "hit_move_count"},
            "hit_move_count",
        ),
        (
            GamesBackgammonDestinationCountTask,
            {"target_answer": 4, "query_id": "blocked_destination_count"},
            "blocked_destination_count",
        ),
        (
            GamesBackgammonPointStateCountTask,
            {
                "target_answer": 5,
                "target_checker_color": PLAYER_BLACK,
                "target_stack_state": STACK_STATE_SINGLE,
            },
            "single",
        ),
        (
            GamesBackgammonPointStateCountTask,
            {
                "target_answer": 6,
                "target_checker_color": PLAYER_WHITE,
                "target_stack_state": STACK_STATE_TWO_OR_MORE,
            },
            "single",
        ),
    ),
)
def test_games_backgammon_public_tasks_emit_expected_contract(
    task_cls,
    params: dict[str, int],
    expected_query: str,
) -> None:
    out = task_cls().generate(820000, params=params, max_attempts=256)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_set"
    assert out.query_id == expected_query
    assert out.scene_id == "backgammon"
    assert trace["query_spec"]["query_id"] == expected_query
    assert trace["query_spec"]["params"]["query_id"] == expected_query
    assert execution["query_id"] == expected_query
    assert trace["query_spec"]["prompt_variant"]["prompt_schema_version"] == "v1"
    expected_prompt_query = (
        "point_state_count"
        if task_cls is GamesBackgammonPointStateCountTask
        else expected_query
    )
    assert (
        trace["query_spec"]["prompt_variant"]["selected_keys"]["query"]
        == expected_prompt_query
    )
    assert execution["active_player"] in {PLAYER_BLACK, PLAYER_WHITE}
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    assert len(execution["target_points"]) == int(out.answer_gt.value)
    assert trace["render_spec"]["canvas_width"] <= 1000
    assert trace["render_spec"]["canvas_height"] <= 720
    assert float(trace["render_spec"]["effective_point_width_px"]) >= 28.0
    assert trace["render_spec"]["text_style"]["font_family"]
    assert (
        trace["render_map"]["font_family"]
        == trace["render_spec"]["text_style"]["font_family"]
    )
    for x0, y0, x1, y1 in out.annotation_gt.value:
        assert (
            0 <= float(x0) <= float(x1) <= float(trace["render_spec"]["canvas_width"])
        )
        assert (
            0 <= float(y0) <= float(y1) <= float(trace["render_spec"]["canvas_height"])
        )


@pytest.mark.parametrize(
    ("task_cls", "target_answer", "expected_query", "active_player"),
    (
        (GamesBackgammonDestinationCountTask, 4, "legal_move_count", PLAYER_BLACK),
        (GamesBackgammonDestinationCountTask, 5, "hit_move_count", PLAYER_WHITE),
        (
            GamesBackgammonDestinationCountTask,
            4,
            "blocked_destination_count",
            PLAYER_WHITE,
        ),
    ),
)
def test_games_backgammon_answers_match_recomputed_destination_sets(
    task_cls,
    target_answer: int,
    expected_query: str,
    active_player: str,
) -> None:
    out = task_cls().generate(
        820100 + target_answer,
        params={
            "target_answer": int(target_answer),
            "query_id": str(expected_query),
            "active_player": str(active_player),
        },
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    points = _points_from_trace(execution)
    dice = tuple(int(value) for value in execution["dice"])
    outcome = compute_single_die_destinations(
        points,
        dice=(dice[0], dice[1]),
        active_player=str(execution["active_player"]),
    )
    expected_destinations = target_destinations_for_status(
        outcome,
        destination_status=DESTINATION_QUERY_SPECS[str(expected_query)][0],
    )
    expected_entity_ids = {point_entity_id(point) for point in expected_destinations}

    assert int(out.answer_gt.value) == len(expected_destinations) == int(target_answer)
    assert tuple(int(value) for value in execution["target_destinations"]) == tuple(
        expected_destinations
    )
    assert set(execution["annotation_entity_ids"]) == expected_entity_ids


@pytest.mark.parametrize(
    ("checker_color", "stack_state", "target_answer"),
    (
        (PLAYER_BLACK, STACK_STATE_SINGLE, 0),
        (PLAYER_WHITE, STACK_STATE_SINGLE, 2),
        (PLAYER_BLACK, STACK_STATE_TWO_OR_MORE, 4),
        (PLAYER_WHITE, STACK_STATE_TWO_OR_MORE, 6),
    ),
)
def test_games_backgammon_point_state_answers_match_recomputed_points(
    checker_color: str,
    stack_state: str,
    target_answer: int,
) -> None:
    out = GamesBackgammonPointStateCountTask().generate(
        820500 + target_answer,
        params={
            "target_answer": int(target_answer),
            "target_checker_color": str(checker_color),
            "target_stack_state": str(stack_state),
        },
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    points = _points_from_trace(execution)
    expected_points = target_points_for_stack_state(
        points,
        checker_color=str(checker_color),
        stack_state=str(stack_state),
    )
    expected_entity_ids = {point_entity_id(point) for point in expected_points}

    assert int(out.answer_gt.value) == len(expected_points) == int(target_answer)
    assert tuple(int(value) for value in execution["target_points"]) == tuple(
        expected_points
    )
    assert tuple(execution["target_destinations"]) == ()
    assert set(execution["annotation_entity_ids"]) == expected_entity_ids
    assert str(execution["construction_mode"]) == "exact_point_state_count"
    assert out.query_id == "single"
    assert execution["checker_color"] == str(checker_color)
    assert execution["stack_state"] == str(stack_state)


@pytest.mark.parametrize(
    ("target_answer", "active_player"),
    (
        (6, PLAYER_BLACK),
        (14, PLAYER_WHITE),
        (24, PLAYER_BLACK),
    ),
)
def test_games_backgammon_pip_count_matches_recomputed_sum(
    target_answer: int,
    active_player: str,
) -> None:
    out = GamesBackgammonPipCountValueTask().generate(
        822000 + target_answer,
        params={
            "target_answer": int(target_answer),
            "active_player": str(active_player),
        },
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    points = _points_from_trace(execution)
    expected_contributions = pip_count_contributions_for_player(
        points,
        active_player=str(active_player),
    )
    expected_points = tuple(sorted(int(point) for point in expected_contributions))
    expected_entity_ids = {point_entity_id(point) for point in expected_points}

    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_set"
    assert out.query_id == "single"
    assert out.scene_id == "backgammon"
    assert int(out.answer_gt.value) == pip_count_for_player(
        points,
        active_player=str(active_player),
    ) == int(target_answer)
    assert tuple(int(value) for value in execution["target_points"]) == expected_points
    assert tuple(execution["target_destinations"]) == ()
    assert {
        str(point): int(value)
        for point, value in sorted(expected_contributions.items())
    } == dict(execution["pip_count_contributions"])
    assert set(execution["annotation_entity_ids"]) == expected_entity_ids
    assert len(out.annotation_gt.value) == len(expected_points)
    assert str(execution["construction_mode"]) == "exact_pip_count"
    assert bool(execution["use_dice_for_moves"]) is False
    assert bool(out.trace_payload["render_map"]["use_dice_for_moves"]) is False
    assert (
        out.trace_payload["query_spec"]["prompt_variant"]["selected_keys"]["query"]
        == "pip_count_value"
    )


def test_games_backgammon_query_cycle_covers_support_and_styles() -> None:
    destination_task = GamesBackgammonDestinationCountTask()
    point_state_task = GamesBackgammonPointStateCountTask()
    pip_count_task = GamesBackgammonPipCountValueTask()
    destination_queries: set[str] = set()
    point_state_queries: set[str] = set()
    point_state_operands: set[tuple[str, str]] = set()
    pip_count_queries: set[str] = set()
    pip_counts: set[int] = set()
    counts: set[int] = set()
    styles: set[str] = set()
    active_players: set[str] = set()

    for sampling_index in range(240):
        out = destination_task.generate(
            820300 + sampling_index,
            params={"_sample_cursor": int(sampling_index)},
            max_attempts=512,
        )
        execution = out.trace_payload["execution_trace"]
        destination_queries.add(str(out.query_id))
        counts.add(int(out.answer_gt.value))
        styles.add(str(execution["style_variant"]))
        active_players.add(str(execution["active_player"]))

    for sampling_index in range(280):
        out = point_state_task.generate(
            821300 + sampling_index,
            params={"_sample_cursor": int(sampling_index)},
            max_attempts=512,
        )
        execution = out.trace_payload["execution_trace"]
        point_state_queries.add(str(out.query_id))
        point_state_operands.add(
            (str(execution["checker_color"]), str(execution["stack_state"]))
        )
        counts.add(int(out.answer_gt.value))
        styles.add(str(execution["style_variant"]))
        active_players.add(str(execution["active_player"]))

    for sampling_index in range(360):
        out = pip_count_task.generate(
            822300 + sampling_index,
            params={"_sample_cursor": int(sampling_index)},
            max_attempts=512,
        )
        execution = out.trace_payload["execution_trace"]
        pip_count_queries.add(str(out.query_id))
        pip_counts.add(int(out.answer_gt.value))
        styles.add(str(execution["style_variant"]))
        active_players.add(str(execution["active_player"]))

    assert destination_queries == set(DESTINATION_QUERY_IDS)
    assert point_state_queries == {"single"}
    assert pip_count_queries == {"single"}
    assert point_state_operands == {
        (PLAYER_BLACK, STACK_STATE_SINGLE),
        (PLAYER_WHITE, STACK_STATE_SINGLE),
        (PLAYER_BLACK, STACK_STATE_TWO_OR_MORE),
        (PLAYER_WHITE, STACK_STATE_TWO_OR_MORE),
    }
    assert counts >= {0, 1, 2, 3, 4, 5, 6}
    assert pip_counts == set(PIP_COUNT_SUPPORT)
    assert styles == set(SUPPORTED_BACKGAMMON_STYLE_VARIANTS)
    assert active_players == {PLAYER_BLACK, PLAYER_WHITE}


def test_games_backgammon_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__backgammon"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__backgammon",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_games__backgammon__destination_count", count=3, params={}
            ),
            BuildTaskConfig(
                task_id="task_games__backgammon__point_state_count", count=3, params={}
            ),
            BuildTaskConfig(
                task_id="task_games__backgammon__pip_count_value", count=3, params={}
            ),
        ],
        max_attempts_per_instance=512,
        workers=1,
    )
    final_path = build_dataset(config, code_hash="games-backgammon-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")

    assert len(rows) == 9
    assert all(row["domain"] == "games" for row in rows)
    assert all("scene_id" not in row for row in rows)
    assert {
        "task_games__backgammon__destination_count",
        "task_games__backgammon__point_state_count",
        "task_games__backgammon__pip_count_value",
    } == {row["task"] for row in rows}
