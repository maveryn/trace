"""Contract tests for the games Connect Four move-count task."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.games.connect_four.blocking_move_column_label import GamesConnectFourBlockingMoveColumnLabelTask
from trace_tasks.tasks.games.connect_four.column_disc_profile_label import GamesConnectFourColumnDiscProfileLabelTask
from trace_tasks.tasks.games.connect_four.shared.rules import RED, YELLOW, drop_disc, opponent, player_name, winning_drop_map
from trace_tasks.tasks.games.connect_four.winning_move_column_label import GamesConnectFourWinningMoveColumnLabelTask
from trace_tasks.tasks.games.connect_four.winning_move_count import GamesConnectFourWinningMoveCountTask
from tests.helpers import read_jsonl


def test_games_connect_four_move_count_emits_expected_contract() -> None:
    out = GamesConnectFourWinningMoveCountTask().generate(
        31001,
        params={
            "scene_variant": "midgame_board",
            "target_answer": 3,
            "board_size_variant": "standard_7x6",
        },
        max_attempts=48,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 3
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == 3
    assert trace["query_spec"]["params"]["query_id"] == out.query_id
    assert int(execution["target_answer"]) == 3
    assert int(execution["board_row_count"]) == 6
    assert int(execution["board_column_count"]) == 7
    assert int(trace["render_map"]["rows"]) == 6
    assert int(trace["render_map"]["columns"]) == 7
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert len(execution["annotation_entity_ids"]) == 3
    assert all(str(entity_id).startswith("cell_r") for entity_id in execution["annotation_entity_ids"])


def test_games_connect_four_move_count_winning_annotation_stays_on_immediate_wins() -> None:
    out = GamesConnectFourWinningMoveCountTask().generate(
        31011,
        params={
            "scene_variant": "midgame_board",
            "target_answer": 4,
            "board_size_variant": "standard_7x6",
        },
        max_attempts=48,
    )
    execution = out.trace_payload["execution_trace"]
    annotation_coords = {tuple(coord) for coord in execution["annotation_coords"]}
    winning_coords = {tuple(coord) for coord in execution["winning_move_coords"]}

    assert annotation_coords == winning_coords
    assert len(annotation_coords) == 4


@pytest.mark.parametrize(
    ("params", "expected_label", "expected_column", "expected_columns"),
    (
        (
            {
                "scene_variant": "midgame_board",
                "board_size_variant": "small_6x5",
                "target_column_label": "C",
                "winning_move_label_threat_kind": "horizontal_threat",
            },
            "C",
            2,
            6,
        ),
        (
            {
                "scene_variant": "crowded_board",
                "board_size_variant": "standard_7x6",
                "target_column_label": "G",
                "winning_move_label_threat_kind": "vertical_threat",
            },
            "G",
            6,
            7,
        ),
    ),
)
def test_games_connect_four_winning_move_column_label_contract(
    params: dict[str, str],
    expected_label: str,
    expected_column: int,
    expected_columns: int,
) -> None:
    out = GamesConnectFourWinningMoveColumnLabelTask().generate(31023, params=params, max_attempts=96)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "string"
    assert str(out.answer_gt.value) == str(expected_label)
    assert out.annotation_gt.type == "point"
    assert len(out.annotation_gt.value) == 2
    assert str(out.query_id) == "single"
    assert trace["query_spec"]["prompt_variant"]["selected_keys"]["query"] == "winning_move_column_label"
    assert int(execution["answer_column"]) == int(expected_column)
    assert execution["column_labels"] == list("ABCDEFG"[: int(expected_columns)])
    assert trace["render_map"]["column_label_to_col"][str(expected_label)] == int(expected_column)
    assert len(execution["winning_move_coords"]) == 1
    assert int(execution["winning_move_coords"][0][1]) == int(expected_column)
    assert execution["annotation_coords"] == execution["winning_move_coords"]
    assert trace["projected_annotation"]["point"] == out.annotation_gt.value
    assert trace["render_map"]["marked_square_bbox_px"] is None


@pytest.mark.parametrize(
    ("params", "expected_label", "expected_column", "expected_columns"),
    (
        (
            {
                "scene_variant": "midgame_board",
                "board_size_variant": "small_6x5",
                "target_column_label": "C",
                "blocking_move_label_threat_kind": "horizontal_threat",
            },
            "C",
            2,
            6,
        ),
        (
            {
                "scene_variant": "crowded_board",
                "board_size_variant": "standard_7x6",
                "target_column_label": "G",
                "blocking_move_label_threat_kind": "vertical_threat",
            },
            "G",
            6,
            7,
        ),
    ),
)
def test_games_connect_four_blocking_move_column_label_contract(
    params: dict[str, str],
    expected_label: str,
    expected_column: int,
    expected_columns: int,
) -> None:
    out = GamesConnectFourBlockingMoveColumnLabelTask().generate(31027, params=params, max_attempts=128)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    board = tuple(tuple(int(cell) for cell in row) for row in execution["board"])
    current_player = RED if str(execution["current_player"]) == "red" else YELLOW
    opposing_player = opponent(int(current_player))

    assert out.answer_gt.type == "string"
    assert str(out.answer_gt.value) == str(expected_label)
    assert out.annotation_gt.type == "point"
    assert len(out.annotation_gt.value) == 2
    assert str(out.query_id) == "single"
    assert trace["query_spec"]["prompt_variant"]["selected_keys"]["query"] == "blocking_move_column_label"
    assert int(execution["answer_column"]) == int(expected_column)
    assert execution["column_labels"] == list("ABCDEFG"[: int(expected_columns)])
    assert trace["render_map"]["column_label_to_col"][str(expected_label)] == int(expected_column)
    assert execution["annotation_coords"] == execution["blocking_move_coords"]
    assert int(execution["blocking_move_coords"][0][1]) == int(expected_column)
    assert trace["projected_annotation"]["point"] == out.annotation_gt.value
    assert trace["render_map"]["marked_square_bbox_px"] is None
    assert execution["opponent_player"] == player_name(int(opposing_player)).lower()

    opponent_wins = winning_drop_map(board, int(opposing_player))
    assert set(opponent_wins.keys()) == {int(expected_column)}
    assert not winning_drop_map(board, int(current_player))
    blocked_board, landing_coord = drop_disc(board, int(current_player), int(expected_column))
    assert list(landing_coord) == execution["blocking_move_coords"][0]
    assert not winning_drop_map(blocked_board, int(opposing_player))


def test_games_connect_four_column_disc_profile_label_contract() -> None:
    out = GamesConnectFourColumnDiscProfileLabelTask().generate(
        31025,
        params={
            "scene_variant": "midgame_board",
            "board_size_variant": "small_6x5",
            "target_column_label": "D",
            "target_red_count": 2,
            "target_yellow_count": 3,
        },
        max_attempts=48,
    )
    trace = out.trace_payload
    execution = out.trace_payload["execution_trace"]

    assert out.answer_gt.type == "string"
    assert str(out.answer_gt.value) == "D"
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == 5
    assert str(out.query_id) == "single"
    assert trace["query_spec"]["prompt_variant"]["selected_keys"]["query"] == "column_disc_profile_label"
    assert int(execution["answer_column"]) == 3
    assert int(execution["target_red_count"]) == 2
    assert int(execution["target_yellow_count"]) == 3
    assert execution["column_labels"] == list("ABCDEF")
    assert trace["render_map"]["column_label_to_col"]["D"] == 3
    assert all(int(coord[1]) == 3 for coord in execution["annotation_coords"])
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value


def test_games_connect_four_move_count_tasks_cover_style_axis() -> None:
    styles_by_variant: dict[str, set[str]] = {
        "column_disc_profile_label": set(),
        "winning_move_count": set(),
    }
    cases = (
        (
            "winning_move_count",
            GamesConnectFourWinningMoveCountTask,
            {
                "scene_variant": "midgame_board",
                "board_size_variant": "standard_7x6",
                "target_answer": 3,
            },
        ),
        (
            "column_disc_profile_label",
            GamesConnectFourColumnDiscProfileLabelTask,
            {
                "scene_variant": "crowded_board",
                "board_size_variant": "small_6x5",
                "target_column_label": "B",
                "target_red_count": 1,
                "target_yellow_count": 2,
            },
        ),
    )
    expected_styles = {"classic", "soft", "outlined", "arcade_blue", "teal_frame", "charcoal"}
    for prompt_query_key, task_cls, base_params in cases:
        for sampling_index, style_variant in enumerate(sorted(expected_styles)):
            params = dict(base_params)
            params["style_variant"] = str(style_variant)
            params["_sample_cursor"] = int(sampling_index)
            out = task_cls().generate(
                20260506 + int(sampling_index),
                params=params,
                max_attempts=96,
            )
            assert str(out.query_id) == "single"
            assert out.trace_payload["query_spec"]["prompt_variant"]["selected_keys"]["query"] == str(prompt_query_key)
            execution = out.trace_payload["execution_trace"]
            styles_by_variant[str(prompt_query_key)].add(str(execution["style_variant"]))

    assert styles_by_variant == {
        "column_disc_profile_label": expected_styles,
        "winning_move_count": expected_styles,
    }


def test_games_connect_four_move_count_is_deterministic() -> None:
    params = {
        "scene_variant": "crowded_board",
        "target_column_label": "A",
        "target_red_count": 2,
        "target_yellow_count": 1,
    }
    task = GamesConnectFourColumnDiscProfileLabelTask()
    out_a = task.generate(31041, params=params, max_attempts=64)
    out_b = task.generate(31041, params=params, max_attempts=64)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_games_connect_four_move_count_prompt_bundle_requires_rule_text_for_query_specific_prompts() -> None:
    bundle = json.loads(Path("src/trace_tasks/resources/prompts/games/connect_four/games_connect_four_v1.json").read_text(encoding="utf-8"))
    required = bundle["required_slots_by_key"]
    assert required["query:winning_move_count"] == [
        "current_player_name",
        "legal_drop_rule_text",
        "winning_rule_text",
    ]
    assert required["query:winning_move_column_label"] == [
        "current_player_name",
        "legal_drop_rule_text",
        "winning_rule_text",
    ]
    assert required["query:blocking_move_column_label"] == [
        "current_player_name",
        "opponent_player_name",
        "legal_drop_rule_text",
        "winning_rule_text",
    ]
    assert required["query:column_disc_profile_label"] == [
        "target_red_count",
        "target_yellow_count",
    ]


def test_games_connect_four_move_count_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__connect_four__winning_move_count"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__connect_four__winning_move_count",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_games__connect_four__winning_move_count",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=64,
        sampling_seed=79,
    )
    final_path = build_dataset(config, code_hash="games-connect-four-move-count-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "games" for record in train_records)
    assert all(record["task"] == "task_games__connect_four__winning_move_count" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_games__connect_four__winning_move_count"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
