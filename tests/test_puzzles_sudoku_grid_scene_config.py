"""Config regression tests for puzzle Sudoku-grid defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import (
    split_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.puzzles.sudoku.shared.styles import SUPPORTED_SUDOKU_STYLE_VARIANTS


def test_puzzles_sudoku_grid_defaults_present() -> None:
    cfg = get_scene_defaults("puzzles", "sudoku")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_puzzles__sudoku__marked_cell_value",
    )

    assert bool(generation["balanced_scene_variant_sampling"]) is True
    assert bool(generation["balanced_unit_type_sampling"]) is True
    assert bool(generation["balanced_style_variant_sampling"]) is True
    assert bool(generation["balanced_target_answer_sampling"]) is True
    assert set(generation["scene_variant_weights"].keys()) == {
        "sparse_grid",
        "filled_grid",
    }
    assert set(generation["unit_type_weights"].keys()) == {"row", "column", "box"}
    assert set(generation["style_variant_weights"].keys()) == set(
        SUPPORTED_SUDOKU_STYLE_VARIANTS
    )
    assert list(generation["marked_cell_value_support"]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert list(generation["option_label_support"]) == ["A", "B", "C", "D"]
    assert int(rendering["max_board_size_px"]) > 0
    assert int(rendering["marked_cell_outline_width_px"]) > 0
    assert str(prompt["bundle_id"]) == "puzzles_sudoku_v1"
    assert str(prompt["scene_key"]) == "visible_sudoku_grid"
    assert str(prompt["task_key"]) == "sudoku_grid_query"
