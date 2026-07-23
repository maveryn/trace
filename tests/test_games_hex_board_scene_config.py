"""Config regression tests for games Hex-board defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.games.shared.style import SUPPORTED_HEX_STYLE_VARIANTS


def test_games_hex_board_defaults_expose_scene_target_board_and_style_axes() -> None:
    cfg = get_scene_defaults("games", "hex")
    generation_shared = cfg["generation"]["shared"]
    task_overrides = cfg["generation"]["task_overrides"]
    rendering = cfg["rendering"]["shared"]
    prompt = cfg["prompt"]["shared"]

    assert "query_id_weights" not in generation_shared
    assert "balanced_query_id_sampling" not in generation_shared
    assert bool(generation_shared["balanced_scene_variant_sampling"]) is True
    assert bool(generation_shared["balanced_player_color_sampling"]) is True
    assert bool(generation_shared["balanced_style_variant_sampling"]) is True
    assert bool(generation_shared["balanced_board_size_sampling"]) is True
    assert set(generation_shared["scene_variant_weights"].keys()) == {"open_board", "crowded_board"}
    assert set(generation_shared["player_color_weights"].keys()) == {"red", "blue"}
    assert set(generation_shared["style_variant_weights"].keys()) == set(SUPPORTED_HEX_STYLE_VARIANTS)
    assert list(generation_shared["board_size_support"]) == [5, 6, 7, 8]
    neighbor = task_overrides["task_games__hex__candidate_neighbor_count"]
    assert bool(neighbor["balanced_target_answer_sampling"]) is True
    assert list(neighbor["neighbor_count_support"]) == [0, 1, 2, 3, 4, 5, 6]
    gap = task_overrides["task_games__hex__connection_gap_count"]
    assert bool(gap["balanced_target_answer_sampling"]) is True
    assert list(gap["connection_gap_count_support"]) == [1, 2, 3, 4, 5]
    winning = task_overrides["task_games__hex__winning_move_cell_label"]
    assert bool(winning["balanced_target_label_sampling"]) is True
    assert bool(winning["balanced_candidate_count_sampling"]) is True
    assert list(winning["candidate_count_support"]) == [4, 5, 6]
    assert list(winning["winning_move_label_support"]) == list("ABCDEF")

    assert int(rendering["canvas_width"]) == 980
    assert int(rendering["max_board_width_px"]) > 0
    assert bool(rendering["dynamic_canvas_size_enabled"]) is True
    assert int(rendering["canvas_min_width_px"]) >= 560
    assert str(prompt["bundle_id"]) == "games_hex_v1"
    assert str(prompt["scene_key"]) == "visible_hex_board"
    assert str(prompt["task_key"]) == "hex_connection_query"
