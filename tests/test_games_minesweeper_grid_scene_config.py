"""Config regression tests for games Minesweeper-grid defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.games.shared.style import SUPPORTED_MINESWEEPER_STYLE_VARIANTS
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_games_minesweeper_grid_defaults_expose_scene_target_and_board_axes() -> None:
    cfg = get_scene_defaults("games", "minesweeper")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__minesweeper__forced_cell_count",
    )

    assert bool(generation["balanced_scene_variant_sampling"]) is True
    assert bool(generation["balanced_style_variant_sampling"]) is True
    assert bool(generation["balanced_board_size_sampling"]) is True
    assert bool(generation["balanced_target_answer_sampling"]) is True
    assert set(generation["scene_variant_weights"].keys()) == {"open_grid", "mixed_grid"}
    assert set(generation["style_variant_weights"].keys()) == set(SUPPORTED_MINESWEEPER_STYLE_VARIANTS)
    assert list(generation["forced_cell_board_size_support"]) == [4, 5]
    assert list(generation["forced_mine_count_support"]) == [1, 2, 3, 4, 5]
    assert list(generation["forced_safe_count_support"]) == [1, 2, 3, 4, 5]
    assert "query_id_weights" not in generation
    assert "balanced_query_id_sampling" not in generation
    assert int(rendering["canvas_width"]) == 900
    assert int(rendering["canvas_height"]) == 900
    assert int(rendering["max_board_size_px"]) > 0
    assert str(prompt["bundle_id"]) == "games_minesweeper_v1"


def test_games_minesweeper_task_override_defaults_expose_specific_supports() -> None:
    cfg = get_scene_defaults("games", "minesweeper")

    remaining_generation, _, _ = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__minesweeper__remaining_mine_count_value",
    )
    assert list(remaining_generation["board_size_support"]) == [4, 5, 6, 7, 8]
    assert list(remaining_generation["remaining_mine_count_support"]) == [0, 1, 2, 3, 4, 5]

    label_generation, _, _ = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__minesweeper__forced_mine_cell_label",
    )
    assert list(label_generation["forced_mine_cell_label_board_size_support"]) == [4, 5]
    assert list(label_generation["forced_mine_cell_label_support"]) == [0, 1, 2, 3]
