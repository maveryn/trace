"""Config regression tests for games bingo defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_games_bingo_completed_column_label_defaults_expose_scene_query_and_target_axes() -> None:
    cfg = get_scene_defaults("games", "bingo")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__bingo__completed_column_label",
    )
    line_sum_generation, _, _ = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__bingo__completed_line_sum_value",
    )
    near_complete_generation, _, _ = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__bingo__near_complete_line_count",
    )
    called_generation, _, _ = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__bingo__called_number_match_count",
    )

    assert bool(generation["balanced_scene_variant_sampling"]) is True
    assert bool(generation["balanced_style_variant_sampling"]) is True
    assert bool(generation["balanced_target_answer_sampling"]) is True
    assert set(generation["scene_variant_weights"].keys()) == {"single_card"}
    assert "query_id_weights" not in generation
    assert "balanced_query_id_sampling" not in generation
    assert set(generation["target_column_label_weights"].keys()) == {"B", "I", "N", "G", "O"}
    assert bool(generation["balanced_target_column_label_sampling"]) is True
    assert "extremum_weights" not in generation
    assert set(generation["style_variant_weights"].keys()) == {
        "classic",
        "soft",
        "outlined",
        "mint",
        "lavender",
        "amber",
        "slate",
    }
    assert set(generation["mark_shape_weights"].keys()) == {"ellipse", "cell", "ring"}
    assert set(generation["cell_fill_pattern_weights"].keys()) == {"solid", "column_tint", "checker_tint"}
    assert "line_sum_completed_line_count_support" not in generation
    assert "extremum_weights" not in line_sum_generation
    assert "balanced_extremum_sampling" not in line_sum_generation
    assert list(line_sum_generation["target_line_index_support"]) == [0, 1, 2, 3, 4]
    assert bool(line_sum_generation["balanced_target_line_index_sampling"]) is True
    assert list(near_complete_generation["near_complete_line_count_support"]) == [0, 1, 2, 3, 4]
    assert list(called_generation["called_number_match_count_support"]) == [0, 1, 2, 3, 4, 5]
    assert list(called_generation["called_number_count_support"]) == [5, 6, 7, 8]
    assert "line_sum_distractor_mark_prob" not in generation
    assert float(line_sum_generation["line_sum_distractor_mark_prob"]) == 0.2
    assert int(rendering["card_width_px"]) > 0
    assert int(rendering["card_height_px"]) > 0
    assert float(rendering["unit_size_scale_min"]) == 0.5
    assert float(rendering["unit_size_scale_max"]) == 1.0
    assert bool(rendering["dynamic_canvas_size_enabled"]) is True
    assert int(rendering["number_font_size_px"]) > 0
    assert int(rendering["called_panel_width_px"]) > 0
    assert str(prompt["bundle_id"]) == "games_bingo_v1"
    assert str(prompt["scene_key"]) == "visible_bingo_card"
    assert str(prompt["task_key"]) == "bingo_card_query"
    assert "object_description_single_card" not in prompt
    assert "answer_hint_completed_column_label" not in prompt
