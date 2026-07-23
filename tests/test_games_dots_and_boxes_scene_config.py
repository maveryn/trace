"""Scene config tests for the games dots-and-boxes scene."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_games_dots_and_boxes_scene_id_defaults_present() -> None:
    defaults = get_scene_defaults("games", "dots_and_boxes")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        defaults,
        task_id="task_games__dots_and_boxes__completable_box_label",
    )
    assert generation["scene_variant_weights"] == {"single_board": 1.0}
    assert "query_id_weights" not in generation
    assert "capture_move_query_id_weights" not in generation
    assert set(generation["style_variant_weights"].keys()) == {
        "classic",
        "soft",
        "outlined",
        "notebook",
        "slate",
        "wood_panel",
    }
    assert generation["option_label_support"] == ["A", "B", "C", "D", "E", "F"]
    assert generation["box_rows_support"] == [3, 4]
    assert generation["box_cols_support"] == [3, 4]
    assert generation["balanced_target_answer_sampling"] is True
    assert generation["balanced_board_shape_sampling"] is True
    assert "balanced_capture_move_query_id_sampling" not in generation

    owned_generation, _owned_rendering, _owned_prompt = split_generation_rendering_prompt_defaults(
        defaults,
        task_id="task_games__dots_and_boxes__owned_box_count",
    )
    assert owned_generation["owned_box_count_support"] == [0, 1, 2, 3, 4, 5, 6, 7, 8]

    three_generation, _three_rendering, _three_prompt = split_generation_rendering_prompt_defaults(
        defaults,
        task_id="task_games__dots_and_boxes__three_sided_box_count",
    )
    assert three_generation["three_sided_box_count_support"] == [0, 1, 2, 3, 4, 5]

    assert int(rendering["board_width_px"]) > 0
    assert int(rendering["board_height_px"]) > 0
    assert rendering["dynamic_canvas_size_enabled"] is True
    assert int(rendering["canvas_min_width_px"]) >= 620
    assert int(rendering["canvas_min_height_px"]) >= 520
    assert str(prompt["bundle_id"]) == "games_dots_and_boxes_v1"
    assert str(prompt["scene_key"]) == "visible_dots_and_boxes_board"
    assert str(prompt["task_key"]) == "dots_and_boxes_query"
