"""Config regression tests for games Snakes and Ladders defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.games.snakes_ladders.shared.state import SUPPORTED_SNAKES_LADDERS_STYLE_VARIANTS
from trace_tasks.tasks.shared.config_defaults import required_group_default, split_generation_rendering_prompt_defaults


def test_games_snakes_ladders_defaults_expose_query_style_and_answer_axes() -> None:
    cfg = get_scene_defaults("games", "snakes_ladders")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__snakes_ladders__remaining_to_finish_value",
    )

    assert bool(generation["balanced_scene_variant_sampling"]) is True
    assert bool(generation["balanced_style_variant_sampling"]) is True
    assert bool(generation["balanced_board_side_sampling"]) is True
    assert bool(generation["balanced_target_answer_sampling"]) is True
    assert bool(generation["balanced_die_value_sampling"]) is True
    assert set(generation["scene_variant_weights"].keys()) == {"standard_board"}
    assert set(generation["style_variant_weights"].keys()) == set(SUPPORTED_SNAKES_LADDERS_STYLE_VARIANTS)
    assert list(generation["board_side_support"]) == [5, 6, 7]
    assert float(generation["move_outcome_jump_probability"]) == 0.30
    assert list(generation["special_square_count_support"]) == [1, 2, 3, 4]
    assert list(generation["remaining_to_finish_support"]) == list(range(1, 26))
    assert int(rendering["canvas_width"]) == 1000
    assert int(rendering["canvas_height"]) == 760
    assert int(rendering["board_size_px"]) > 0
    assert str(prompt["bundle_id"]) == "games_snakes_ladders_v1"
    assert "Snakes and Ladders board" in str(required_group_default(prompt, "object_description_standard_board", context="snakes ladders prompt defaults"))
    assert "token_square" in str(required_group_default(prompt, "annotation_hint_remaining_to_finish_value", context="snakes ladders prompt defaults"))
    assert "visible ladders" in str(required_group_default(prompt, "answer_hint_ladder_count", context="snakes ladders prompt defaults"))
    assert "snake-head" in str(required_group_default(prompt, "annotation_hint_snake_count", context="snakes ladders prompt defaults"))
