"""Contract tests for Ludo-board games tasks."""

from __future__ import annotations

import json
from pathlib import Path

import trace_tasks.tasks  # noqa: F401
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.games.ludo_board.capture_roll_option_label import TASK_ID as CAPTURE_ROLL_TASK_ID
from trace_tasks.tasks.games.ludo_board.move_result_option_label import TASK_ID as MOVE_RESULT_TASK_ID
from trace_tasks.tasks.games.ludo_board.shared.rules import roll_option_text, roll_sequence_for_total, route_for_color
from trace_tasks.tasks.games.ludo_board.shared.state import (
    FLOW_ARROW_CELLS,
    FLOW_ARROW_SPECS,
    HOME_LANES,
    MAIN_PATH,
    OPTION_LABELS,
    PLAYER_COLORS,
    START_COORDS,
)
from trace_tasks.tasks.games.ludo_board.winning_roll_value import TASK_ID as WINNING_ROLL_TASK_ID
from trace_tasks.tasks.registry import create_task, ensure_scene_tasks_registered, is_default_dataset_task
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_games_ludo_board_defaults_and_prompt_bundle() -> None:
    cfg = get_scene_defaults("games", "ludo_board")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(cfg)

    assert set(generation["style_variant_weights"].keys()) == {
        "classic_bright",
        "ivory_board",
        "slate_table",
        "soft_plastic",
        "arcade_gloss",
    }
    assert set(generation["query_color_weights"].keys()) == set(PLAYER_COLORS)
    task_overrides = cfg["generation"]["task_overrides"]
    assert list(task_overrides[WINNING_ROLL_TASK_ID]["winning_roll_support"]) == [1, 2, 3, 4, 5]
    assert list(task_overrides[CAPTURE_ROLL_TASK_ID]["capture_distance_support"]) == list(range(1, 12))
    assert list(task_overrides[CAPTURE_ROLL_TASK_ID]["answer_option_label_weights"].keys()) == list(OPTION_LABELS)
    assert list(task_overrides[MOVE_RESULT_TASK_ID]["move_roll_total_support"]) == list(range(1, 12))
    assert list(task_overrides[MOVE_RESULT_TASK_ID]["answer_option_label_weights"].keys()) == list(OPTION_LABELS)
    assert int(rendering["cell_size_min_px"]) == 36
    assert int(rendering["cell_size_max_px"]) == 48
    assert bool(rendering["flow_arrow_enabled"]) is True
    assert int(rendering["flow_arrow_width_px"]) == 2
    assert str(prompt["bundle_id"]) == "games_ludo_board_v1"


def test_games_ludo_board_prompt_bundle_has_queries() -> None:
    bundle = json.loads(Path("src/trace_tasks/resources/prompts/games/ludo_board/games_ludo_board_v1.json").read_text(encoding="utf-8"))

    assert bundle["schema_version"] == "v1"
    assert set(bundle["templates"]["query"].keys()) == {
        "winning_roll_value",
        "capture_roll_option_label",
        "move_result_option_label",
    }
    assert bundle["required_slots_by_key"]["query:winning_roll_value"] == ["exact_finish_rule_text"]
    assert bundle["required_slots_by_key"]["query:capture_roll_option_label"] == ["capture_option_rule_text"]
    assert bundle["required_slots_by_key"]["query:move_result_option_label"] == ["move_sequence_rule_text"]
    assert bool(bundle["allow_empty_task_templates"])


def test_games_ludo_board_path_and_roll_encoding() -> None:
    assert len(MAIN_PATH) == 52
    assert len(set(MAIN_PATH)) == 52
    for color, start in START_COORDS.items():
        assert start in MAIN_PATH
        assert len(HOME_LANES[color]) == 5
    assert FLOW_ARROW_SPECS == (
        ((6, 2), (6, 3), "start_forward_red"),
        ((2, 8), (3, 8), "start_forward_green"),
        ((8, 12), (8, 11), "start_forward_yellow"),
        ((12, 6), (11, 6), "start_forward_blue"),
        ((6, 5), (5, 6), "corner_turn_top_left"),
        ((5, 8), (6, 9), "corner_turn_top_right"),
        ((8, 9), (9, 8), "corner_turn_bottom_right"),
        ((9, 6), (8, 5), "corner_turn_bottom_left"),
        ((7, 0), (7, 1), "home_entry_red"),
        ((0, 7), (1, 7), "home_entry_green"),
        ((7, 14), (7, 13), "home_entry_yellow"),
        ((14, 7), (13, 7), "home_entry_blue"),
    )
    playable_cells = set(MAIN_PATH)
    for lane in HOME_LANES.values():
        playable_cells.update(lane)
    assert len(FLOW_ARROW_SPECS) == 12
    assert len(FLOW_ARROW_CELLS) == 24
    assert all(coord in playable_cells for coord in FLOW_ARROW_CELLS)
    for start, end, _role in FLOW_ARROW_SPECS:
        assert max(abs(int(start[0]) - int(end[0])), abs(int(start[1]) - int(end[1]))) == 1

    assert roll_option_text(1) == "1"
    assert roll_option_text(6) == "6"
    assert roll_option_text(7) == "6 then 1"
    assert roll_option_text(11) == "6 then 5"
    assert roll_sequence_for_total(1) == (1,)
    assert roll_sequence_for_total(6) == (6,)
    assert roll_sequence_for_total(7) == (6, 1)
    assert roll_sequence_for_total(11) == (6, 5)


def test_games_ludo_board_registry_and_taxonomy() -> None:
    ensure_scene_tasks_registered("games", "ludo_board")
    assert is_default_dataset_task(WINNING_ROLL_TASK_ID)
    assert is_default_dataset_task(CAPTURE_ROLL_TASK_ID)
    assert is_default_dataset_task(MOVE_RESULT_TASK_ID)
    for task_id in (WINNING_ROLL_TASK_ID, CAPTURE_ROLL_TASK_ID, MOVE_RESULT_TASK_ID):
        taxonomy = resolve_task_taxonomy(task_id)
        assert taxonomy.domain == "games"
        assert taxonomy.scene_id == "ludo_board"
        assert taxonomy.source_scene_id == ""


def test_games_ludo_board_winning_roll_answer_matches_trace() -> None:
    for roll in (1, 5):
        out = create_task(WINNING_ROLL_TASK_ID).generate(
            980100 + roll,
            params={"winning_roll": roll, "query_color": "blue"},
            max_attempts=100,
        )
        execution = out.trace_payload["execution_trace"]

        assert out.scene_id == "ludo_board"
        assert out.query_id == "single"
        assert out.trace_payload["query_spec"]["params"]["prompt_query_key"] == "winning_roll_value"
        assert out.answer_gt.type == "integer"
        assert int(out.answer_gt.value) == roll
        assert execution["winning_roll"] == roll
        assert execution["query_color"] == "blue"
        assert execution["token_coords_by_color"]["blue"] == list(HOME_LANES["blue"][5 - roll])
        assert out.annotation_gt.type == "point"
        assert len(out.annotation_gt.value) == 2
        assert out.trace_payload["projected_annotation"]["type"] == "point"
        assert out.trace_payload["projected_annotation"]["pixel_point"] == out.annotation_gt.value
        assert out.trace_payload["scene_ir"]["relations"]["annotation_entity_ids"] == {"point_0": "token_blue"}
        assert len(out.trace_payload["render_map"]["flow_arrow_markers_px"]) == len(FLOW_ARROW_SPECS)
        assert set(out.trace_payload["render_map"]["flow_arrow_markers_px"][0]) >= {"start_coord", "end_coord", "role"}


def test_games_ludo_board_capture_option_answer_matches_trace() -> None:
    for distance in (1, 6, 7, 11):
        out = create_task(CAPTURE_ROLL_TASK_ID).generate(
            981100 + distance,
            params={
                "capture_distance": distance,
                "query_color": "red",
                "target_color": "green",
                "answer_option_label": "F",
                "option_count": 6,
            },
            max_attempts=100,
        )
        execution = out.trace_payload["execution_trace"]
        correct_options = [option for option in execution["options"] if int(option["distance"]) == distance]

        assert out.scene_id == "ludo_board"
        assert out.query_id == "single"
        assert out.trace_payload["query_spec"]["params"]["prompt_query_key"] == "capture_roll_option_label"
        assert out.answer_gt.type == "option_letter"
        assert out.answer_gt.value == "F"
        assert execution["capture_distance"] == distance
        assert execution["answer"] == "F"
        assert correct_options == [{"label": "F", "distance": distance, "text": roll_option_text(distance)}]
        assert len(execution["options"]) == 6
        assert len(out.trace_payload["render_map"]["flow_arrow_markers_px"]) == len(FLOW_ARROW_SPECS)
        assert out.annotation_gt.type == "point_map"
        assert set(out.annotation_gt.value.keys()) == {"mover_token", "target_token"}
        assert set(out.trace_payload["projected_annotation"]["pixel_point_map"].keys()) == {
            "mover_token",
            "target_token",
        }


def test_games_ludo_board_move_result_answer_matches_trace() -> None:
    for total in (1, 7, 10, 11):
        out = create_task(MOVE_RESULT_TASK_ID).generate(
            982100 + total,
            params={
                "move_roll_total": total,
                "query_color": "blue",
                "answer_option_label": "D",
                "option_count": 6,
            },
            max_attempts=100,
        )
        execution = out.trace_payload["execution_trace"]
        route = route_for_color("blue")
        start_coord = tuple(execution["token_coords_by_color"]["blue"])
        start_index = route.index(start_coord)
        expected_destination = tuple(route[start_index + total])
        destination_options = {
            str(option["label"]): tuple(option["coord"])
            for option in execution["destination_options"]
        }
        token_coords = {tuple(coord) for coord in execution["token_coords_by_color"].values()}

        assert out.scene_id == "ludo_board"
        assert out.query_id == "single"
        assert out.trace_payload["query_spec"]["params"]["prompt_query_key"] == "move_result_option_label"
        assert out.answer_gt.type == "option_letter"
        assert out.answer_gt.value == "D"
        assert execution["move_roll_total"] == total
        assert execution["roll_sequence"] == list(roll_sequence_for_total(total))
        assert len(execution["roll_sequence"]) <= 2
        assert execution["answer"] == "D"
        assert destination_options["D"] == expected_destination
        assert len(destination_options) == len(execution["destination_options"])
        assert len(set(destination_options.values())) == len(execution["destination_options"])
        assert not set(destination_options.values()) & token_coords
        assert set(out.annotation_gt.value.keys()) == {"moving_token", "destination_cell"}
        assert out.annotation_gt.type == "point_map"
        assert set(out.trace_payload["projected_annotation"]["pixel_point_map"].keys()) == {
            "moving_token",
            "destination_cell",
        }
        assert set(out.trace_payload["render_map"]["destination_option_cell_bboxes_px"].keys()) == set(destination_options)
        assert out.trace_payload["render_map"]["roll_sequence_px"]["values"] == list(roll_sequence_for_total(total))
