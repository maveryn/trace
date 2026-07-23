"""Config regression tests for games lane-crossing defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults


def test_games_crossing_lane_defaults_present() -> None:
    cfg = get_scene_defaults("games", "crossing")
    shared_generation, rendering, prompt = split_scene_generation_rendering_prompt_defaults(cfg)
    route_generation, _route_rendering, _route_prompt = split_scene_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__crossing__hit_object_label",
    )
    first_exit_generation, _first_exit_rendering, _first_exit_prompt = split_scene_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__crossing__first_exit_object_label",
    )
    direction_generation, _direction_rendering, _direction_prompt = split_scene_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__crossing__moving_object_direction_count",
    )

    assert bool(shared_generation["balanced_scene_variant_sampling"]) is True
    assert bool(shared_generation["balanced_style_variant_sampling"]) is True
    assert bool(shared_generation["balanced_lane_count_sampling"]) is True
    assert bool(shared_generation["balanced_row_count_sampling"]) is True
    assert bool(shared_generation["balanced_target_answer_sampling"]) is True
    assert "query_id_weights" not in shared_generation
    assert set(shared_generation["scene_variant_weights"].keys()) == {"traffic_crossing"}
    assert set(shared_generation["style_variant_weights"].keys()) == {
        "day",
        "night",
        "retro",
        "paper",
        "construction",
    }
    assert list(shared_generation["lane_count_support"]) == [5, 6, 7, 8]
    assert list(shared_generation["row_count_support"]) == [5, 6, 7]
    assert "moving_object_count_support" not in shared_generation
    assert "first_exit_label_index_support" not in shared_generation
    assert "hit_object_label_index_support" not in shared_generation
    assert "left_moving_object_count_support" not in shared_generation
    assert "right_moving_object_count_support" not in shared_generation
    assert list(first_exit_generation["first_exit_label_index_support"]) == [0, 1, 2, 3]
    assert list(route_generation["hit_object_label_index_support"]) == [0, 1, 2, 3]
    assert int(route_generation["hit_object_max_extra_per_row"]) == 1
    assert list(direction_generation["left_moving_object_count_support"]) == [1, 2, 3, 4, 5, 6]
    assert list(direction_generation["right_moving_object_count_support"]) == [1, 2, 3, 4, 5, 6]
    assert int(rendering["canvas_width"]) == 1000
    assert int(rendering["canvas_height"]) == 780
    assert int(rendering["vehicle_width_px"]) > 0
    assert str(prompt["bundle_id"]) == "games_crossing_v1"
