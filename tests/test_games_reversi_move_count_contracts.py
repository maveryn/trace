"""Contract tests for games Reversi source-layout tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.games.reversi.legal_destination_count import (
    GamesReversiLegalDestinationCountTask,
)
from trace_tasks.tasks.games.reversi.shared.rules import frontier_disc_coords
from trace_tasks.tasks.games.reversi.shared.state import BLACK, WHITE
from trace_tasks.tasks.games.shared.style import (
    SUPPORTED_REVERSI_STYLE_VARIANTS,
    build_games_reversi_theme,
)
from trace_tasks.tasks.registry import create_task
from tests.helpers import read_jsonl


@pytest.mark.parametrize(
    ("task_id", "params", "expected_answer", "expected_annotation_type"),
    (
        (
            "task_games__reversi__legal_destination_count",
            {"scene_variant": "compact_board", "target_answer": 4},
            4,
            "bbox_set",
        ),
        (
            "task_games__reversi__marked_move_flip_count",
            {"scene_variant": "classic_board", "target_answer": 5},
            5,
            "point_set",
        ),
    ),
)
def test_games_reversi_tasks_emit_expected_contract(
    task_id: str,
    params: dict[str, int | str],
    expected_answer: int,
    expected_annotation_type: str,
) -> None:
    out = create_task(task_id).generate(28001, params=params, max_attempts=64)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == int(expected_answer)
    assert out.annotation_gt.type == str(expected_annotation_type)
    assert len(out.annotation_gt.value) == int(expected_answer)
    assert trace["query_spec"]["query_id"] == out.query_id
    assert int(execution["target_answer"]) == int(expected_answer)
    assert (
        trace["projected_annotation"][str(expected_annotation_type)]
        == out.annotation_gt.value
    )
    assert len(execution["annotation_entity_ids"]) == int(expected_answer)
    assert all(
        str(entity_id).startswith("cell_r")
        for entity_id in execution["annotation_entity_ids"]
    )


def test_games_reversi_marked_move_keeps_annotation_on_flipped_discs() -> None:
    out = create_task("task_games__reversi__marked_move_flip_count").generate(
        28021,
        params={"scene_variant": "classic_board", "target_answer": 4},
        max_attempts=64,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.query_id == "single"
    assert execution["prompt_query_key"] == "flip_count_for_marked_move"
    assert execution["marked_move"] is not None
    assert execution["marked_move_cell_id"] is not None
    assert trace["render_map"]["marked_square_bbox_px"] is not None
    assert execution["marked_move_cell_id"] not in set(
        execution["annotation_entity_ids"]
    )
    assert len(execution["marked_move_flip_coords"]) == 4
    assert out.annotation_gt.type == "point_set"
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    marker_records = trace["render_spec"]["drawn_markers"]["marker_legibility"][
        "records"
    ]
    expected_red = build_games_reversi_theme(
        style_variant=str(execution["style_variant"])
    ).marked_square_outline_rgb
    assert len(marker_records) == 1
    assert marker_records[0]["inner_rgb"] == list(expected_red)


@pytest.mark.parametrize(
    ("target_player", "player_value"),
    (
        ("black", BLACK),
        ("white", WHITE),
    ),
)
def test_games_reversi_frontier_disc_count_matches_visible_board(
    target_player: str, player_value: int
) -> None:
    out = create_task("task_games__reversi__frontier_disc_count").generate(
        28041,
        params={
            "scene_variant": "classic_board",
            "target_player": target_player,
            "target_answer": 5,
        },
        max_attempts=96,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    board = tuple(tuple(int(cell) for cell in row) for row in execution["board_rows"])
    expected_coords = frontier_disc_coords(board, int(player_value))
    expected_entity_ids = [
        f"cell_r{int(row)}_c{int(col)}" for row, col in expected_coords
    ]

    assert out.scene_id == "reversi"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 5 == len(expected_coords)
    assert out.annotation_gt.type == "point_set"
    assert execution["annotation_entity_ids"] == expected_entity_ids
    assert execution["target_player"] == target_player
    assert execution["annotation_coords"] == [
        [int(row), int(col)] for row, col in expected_coords
    ]
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    for entity_id, point in zip(expected_entity_ids, out.annotation_gt.value):
        assert trace["render_map"]["disc_points_px"][entity_id] == point


def test_games_reversi_legal_destination_cycle_covers_answer_scene_and_style_support() -> (
    None
):
    task = GamesReversiLegalDestinationCountTask()
    answers: set[int] = set()
    scenes: set[str] = set()
    styles: set[str] = set()

    for sampling_index in range(84):
        out = task.generate(
            28101 + int(sampling_index),
            params={"_sample_cursor": sampling_index},
            max_attempts=192,
        )
        execution = out.trace_payload["execution_trace"]
        assert str(out.query_id) == "single"
        answers.add(int(out.answer_gt.value))
        scenes.add(str(execution["scene_variant"]))
        styles.add(str(execution["style_variant"]))

    assert answers == {0, 1, 2, 3, 4, 5}
    assert scenes == {"compact_board", "classic_board"}
    assert styles == set(SUPPORTED_REVERSI_STYLE_VARIANTS)


def test_games_reversi_task_is_deterministic() -> None:
    params = {"scene_variant": "compact_board", "target_answer": 5}
    task = GamesReversiLegalDestinationCountTask()
    out_a = task.generate(28031, params=params, max_attempts=64)
    out_b = task.generate(28031, params=params, max_attempts=64)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert (
        out_a.trace_payload["query_spec"]["prompt_variant"]
        == out_b.trace_payload["query_spec"]["prompt_variant"]
    )
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_games_reversi_prompt_bundle_requires_rule_text_for_query_specific_prompts() -> (
    None
):
    bundle = json.loads(
        Path("src/trace_tasks/resources/prompts/games/reversi/games_reversi_v1.json").read_text(encoding="utf-8")
    )
    required = bundle["required_slots_by_key"]
    assert required["query:legal_move_count"] == [
        "current_player_name",
        "legal_move_rule_text",
    ]
    assert required["query:flip_count_for_marked_move"] == [
        "current_player_name",
        "legal_move_rule_text",
        "marked_move_rule_text",
        "flip_rule_text",
    ]
    assert required["query:frontier_disc_count"] == [
        "frontier_rule_text",
        "query_player",
    ]


def test_games_reversi_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__reversi__legal_destination_count"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__reversi__legal_destination_count",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_games__reversi__legal_destination_count",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=64,
        sampling_seed=73,
    )
    final_path = build_dataset(config, code_hash="games-reversi-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "games" for record in train_records)
    assert all(record.get("scene_id") == "reversi" for record in train_records)

    build_report = json.loads(
        (final_path / "build_report.json").read_text(encoding="utf-8")
    )
    assert (
        int(
            build_report["accepted_counts_by_task"][
                "task_games__reversi__legal_destination_count"
            ]
        )
        == 4
    )

    validation = json.loads(
        (final_path / "validation_report.json").read_text(encoding="utf-8")
    )
    assert validation["total_errors"] == 0
