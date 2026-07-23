"""Contract tests for Sixteen Soldiers games tasks."""

from __future__ import annotations

import json
from pathlib import Path

import trace_tasks.tasks  # noqa: F401
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.games.sixteen_soldiers.shared.rules import (
    BLUE,
    EDGES,
    JUMP_SPECS,
    POINT_COORDS,
    RED,
    board_piece_count,
    capturable_opponent_points,
    freeze_board,
    legal_destinations,
    piece_to_entity_id,
)
from trace_tasks.tasks.registry import create_task
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


MARKED_DESTINATION_TASK_ID = "task_games__sixteen_soldiers__marked_piece_destination_count"
MARKED_CAPTURE_TASK_ID = "task_games__sixteen_soldiers__marked_piece_capture_count"


def _board_from_trace(out) -> tuple[tuple[str, int], ...]:
    state_to_value = {"empty": 0, "red": RED, "blue": BLUE}
    return freeze_board(
        {
            str(item["point_id"]): int(state_to_value[str(item["state"])])
            for item in out.trace_payload["execution_trace"]["point_values"]
        }
    )


def test_games_sixteen_soldiers_topology_is_canonical_and_usable() -> None:
    assert len(POINT_COORDS) == 37
    assert len(EDGES) == 76
    assert len(JUMP_SPECS) == 112
    assert all(a != b for a, b in EDGES)
    assert len(set(EDGES)) == len(EDGES)


def test_games_sixteen_soldiers_defaults_expose_axes_and_prompt_bundle() -> None:
    cfg = get_scene_defaults("games", "sixteen_soldiers")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id=MARKED_DESTINATION_TASK_ID,
    )

    assert set(generation["scene_variant_weights"].keys()) == {
        "balanced_midgame",
        "center_crossroads_midgame",
        "triangle_wing_midgame",
    }
    assert "query_id_weights" not in generation
    assert set(generation["style_variant_weights"].keys()) == {
        "ground_court",
        "ink_court",
        "cloth_board",
        "slate_court",
        "sand_court",
    }
    assert list(generation["marked_piece_destination_count_support"]) == [0, 1, 2, 3, 4, 5]
    assert list(generation["marked_piece_capture_count_support"]) == [0, 1, 2, 3, 4]
    assert list(generation["piece_count_per_side_support"]) == [6, 7, 8, 9, 10]
    assert int(rendering["max_board_height_px"]) > int(rendering["max_board_width_px"])
    assert str(prompt["bundle_id"]) == "games_sixteen_soldiers_v1"


def test_games_sixteen_soldiers_prompt_bundle_has_queries() -> None:
    bundle = json.loads(
        Path("src/trace_tasks/resources/prompts/games/sixteen_soldiers/games_sixteen_soldiers_v1.json").read_text(encoding="utf-8")
    )
    assert set(bundle["templates"]["query"].keys()) == {
        "marked_piece_destination_count",
        "marked_piece_capture_count",
    }
    assert bundle["required_slots_by_key"]["query:marked_piece_capture_count"] == ["capture_rule_text"]


def test_games_sixteen_soldiers_marked_destination_answer_matches_trace() -> None:
    out = create_task(MARKED_DESTINATION_TASK_ID).generate(
        81701,
        params={"target_answer": 5, "piece_count_per_side": 10},
        max_attempts=80,
    )
    board = _board_from_trace(out)
    execution = out.trace_payload["execution_trace"]
    marked_point_id = str(execution["marked_point_id"])
    expected_ids = list(legal_destinations(board, marked_point_id))
    expected_points = [out.trace_payload["render_map"]["point_centers_px"][point_id] for point_id in expected_ids]

    assert out.scene_id == "sixteen_soldiers"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 5
    assert out.annotation_gt.type == "point_set"
    assert execution["annotation_point_ids"] == expected_ids
    assert execution["prompt_query_key"] == "marked_piece_destination_count"
    assert out.trace_payload["query_spec"]["params"]["prompt_query_key"] == "marked_piece_destination_count"
    assert out.annotation_gt.value == expected_points
    assert out.trace_payload["projected_annotation"]["point_set"] == expected_points


def test_games_sixteen_soldiers_marked_capture_answer_matches_trace() -> None:
    out = create_task(MARKED_CAPTURE_TASK_ID).generate(
        81702,
        params={"target_answer": 4, "piece_count_per_side": 10},
        max_attempts=100,
    )
    board = _board_from_trace(out)
    execution = out.trace_payload["execution_trace"]
    marked_point_id = str(execution["marked_point_id"])
    expected_points_ids = list(capturable_opponent_points(board, marked_point_id))
    expected_entity_ids = [piece_to_entity_id(point_id) for point_id in expected_points_ids]
    expected_points = [out.trace_payload["render_map"]["piece_centers_px"][piece_id] for piece_id in expected_entity_ids]

    assert out.scene_id == "sixteen_soldiers"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 4
    assert out.annotation_gt.type == "point_set"
    assert execution["annotation_point_ids"] == expected_points_ids
    assert execution["annotation_entity_ids"] == expected_entity_ids
    assert execution["prompt_query_key"] == "marked_piece_capture_count"
    assert out.trace_payload["query_spec"]["params"]["prompt_query_key"] == "marked_piece_capture_count"
    assert out.annotation_gt.value == expected_points
    assert out.trace_payload["projected_annotation"]["pixel_point_set"] == expected_points
    assert len(execution["capture_lines_for_marked_piece"]) >= 4


def test_games_sixteen_soldiers_zero_capture_answer_uses_empty_point_set() -> None:
    out = create_task(MARKED_CAPTURE_TASK_ID).generate(
        81703,
        params={"target_answer": 0, "piece_count_per_side": 9},
        max_attempts=100,
    )
    board = _board_from_trace(out)
    marked_point_id = str(out.trace_payload["execution_trace"]["marked_point_id"])

    assert capturable_opponent_points(board, marked_point_id) == ()
    assert int(out.answer_gt.value) == 0
    assert out.annotation_gt.type == "point_set"
    assert out.annotation_gt.value == []
    assert out.trace_payload["projected_annotation"]["point_set"] == []
    assert board_piece_count(board, RED) == 9
    assert board_piece_count(board, BLUE) == 9


def test_games_sixteen_soldiers_taxonomy_mapping() -> None:
    for task_id in (MARKED_DESTINATION_TASK_ID, MARKED_CAPTURE_TASK_ID):
        taxonomy = resolve_task_taxonomy(task_id)
        assert taxonomy.domain == "games"
        assert taxonomy.scene_id == "sixteen_soldiers"
        assert taxonomy.source_domain == "games"
        assert taxonomy.source_scene_id == ""
