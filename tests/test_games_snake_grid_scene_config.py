"""Config regression tests for games Snake-grid defaults."""

from __future__ import annotations

import json
from pathlib import Path

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.games.snake.shared.state import SCENE_VARIANTS, STYLE_VARIANTS
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_games_snake_grid_defaults_present() -> None:
    cfg = get_scene_defaults("games", "snake")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__snake__safe_direction_count",
    )

    assert bool(generation["balanced_style_variant_sampling"]) is True
    assert bool(generation["balanced_snake_length_count_sampling"]) is True
    assert set(generation["scene_variant_weights"].keys()) == set(SCENE_VARIANTS)
    assert set(generation["style_variant_weights"].keys()) == set(STYLE_VARIANTS)
    assert list(generation["board_size_support"]) == [7, 8, 9, 10]
    assert list(generation["safe_direction_count_support"]) == [0, 1, 2, 3]
    assert list(generation["snake_length_count_support"]) == [6, 7, 8, 9, 10, 11, 12]
    assert list(generation["planned_move_outcome_support"]) == ["point", "game_over"]
    assert list(generation["obstacle_count_support"]) == [2, 3, 4, 5, 6]
    assert int(rendering["canvas_width"]) == 900
    assert int(rendering["max_board_size_px"]) == 720
    assert str(prompt["bundle_id"]) == "games_snake_v1"
    prompt_bundle = json.loads(Path("src/trace_tasks/resources/prompts/games/snake/games_snake_v1.json").read_text(encoding="utf-8"))
    code_defaults = prompt_bundle["code_prompt_defaults"]
    assert "connected body cells" in str(code_defaults["object_description_square_grid"]).lower()
    assert "snake-occupied cells" in str(code_defaults["answer_hint_snake_length_count"]).lower()
