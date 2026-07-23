"""Config regression tests for physics collision scene defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_physics_collision_defaults_expose_scene_axes_and_answer_support() -> None:
    cfg = get_scene_defaults("physics", "collision")
    direction_generation, direction_rendering, direction_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__collision__sticky_collision_direction_choice",
    )
    speed_generation, speed_rendering, speed_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__collision__sticky_collision_speed_value",
    )


    assert bool(direction_generation["balanced_scene_variant_sampling"]) is True

    assert "query_id_weights" not in direction_generation

    assert "balanced_query_id_sampling" not in direction_generation

    assert bool(direction_generation["balanced_target_answer_sampling"]) is True

    assert bool(direction_generation["balanced_correct_option_letter_sampling"]) is True

    assert set(direction_generation["scene_variant_weights"].keys()) == {"wide_table", "compact_table", "gridded_table"}

    assert set(direction_generation["correct_option_letter_weights"].keys()) == {"A", "B", "C", "D"}

    assert list(direction_generation["component_answer_support"]) == [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]

    assert list(direction_generation["mass_support"]) == [1, 2, 3, 4, 5, 6]

    assert list(direction_generation["speed_support"]) == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    assert list(speed_generation["speed_answer_tenths_support"]) == [
        14,
        22,
        28,
        32,
        36,
        41,
        42,
        45,
        50,
        51,
        54,
        57,
        58,
        61,
        63,
        64,
        67,
        71,
        72,
        78,
        85,
    ]


    assert int(direction_rendering["canvas_width"]) == 1180

    assert int(direction_rendering["canvas_height"]) == 760

    assert int(direction_rendering["puck_radius_px"]) == 42

    assert int(direction_rendering["option_arrow_length_px"]) == 74
    assert bool(speed_rendering["show_candidate_options"]) is False
    assert int(speed_rendering["canvas_height"]) == 640


    assert str(direction_prompt["bundle_id"]) == "physics_collision_v1"

    assert str(direction_prompt["task_key"]) == "sticky_collision_direction_choice_query"

    assert str(speed_prompt["task_key"]) == "sticky_collision_speed_value_query"

    assert "scene_key" not in direction_prompt
