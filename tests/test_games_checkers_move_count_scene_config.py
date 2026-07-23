"""Config regression tests for games Checkers defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_games_checkers_move_count_defaults_expose_scene_query_and_answer_axes() -> None:
    cfg = get_scene_defaults("games", "checkers")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__checkers__move_count",
    )

    assert bool(generation["balanced_scene_variant_sampling"]) is True
    assert bool(generation["balanced_style_variant_sampling"]) is True
    assert bool(generation["balanced_target_answer_sampling"]) is True
    assert set(generation["scene_variant_weights"].keys()) == {"midgame_board", "crowded_board"}
    assert set(generation["style_variant_weights"].keys()) == {
        "classic",
        "soft",
        "outlined",
        "wood_token",
        "blue_table",
        "charcoal",
    }
    assert list(generation["legal_move_count_support"]) == [0, 1, 2, 3, 4, 5]
    assert list(generation["capture_move_count_support"]) == [0, 1, 2, 3, 4]
    chain_generation, _chain_rendering, _chain_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__checkers__max_capture_chain_length",
    )
    mobility_generation, _mobility_rendering, _mobility_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__checkers__piece_mobility_count",
    )
    state_generation, _state_rendering, _state_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__checkers__piece_state_count",
    )
    assert list(chain_generation["max_capture_chain_length_support"]) == [1, 2, 3, 4, 5]
    assert list(mobility_generation["piece_with_legal_move_count_support"]) == [0, 1, 2, 3, 4, 5]
    assert list(mobility_generation["piece_with_capture_move_count_support"]) == [0, 1, 2, 3, 4]
    assert list(state_generation["piece_state_count_support"]) == [0, 1, 2, 3, 4, 5, 6]
    assert int(rendering["max_board_size_px"]) > 0
    assert int(rendering["player_badge_height_px"]) > 0
    assert bool(rendering["dynamic_canvas_size_enabled"]) is True
    assert int(rendering["canvas_min_width_px"]) > 0
    assert int(rendering["canvas_min_height_px"]) > 0
    assert str(prompt["bundle_id"]) == "games_checkers_v1"
    assert str(prompt["scene_key"]) == "visible_checkers_board"
    assert str(prompt["task_key"]) == "checkers_board_query"
