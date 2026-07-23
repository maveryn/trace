"""Config regression tests for physics waves interference-tank defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_physics_waves_defaults_expose_scene_query_axes_and_supports() -> None:
    cfg = get_scene_defaults("physics", "wave_interference")
    choice_generation, rendering, choice_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__wave_interference__interference_point_choice",
    )
    path_generation, _, path_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__wave_interference__path_difference_value",
    )


    assert bool(choice_generation["balanced_scene_variant_sampling"]) is True

    assert "query_id_weights" not in choice_generation
    assert "balanced_query_id_sampling" not in choice_generation

    assert bool(choice_generation["balanced_phase_relation_sampling"]) is True

    assert bool(choice_generation["balanced_option_letter_sampling"]) is True

    assert bool(path_generation["balanced_target_answer_sampling"]) is True

    assert set(choice_generation["scene_variant_weights"].keys()) == {"clean_tank", "grid_tank", "lab_sheet"}

    assert "query_id_weights" not in path_generation
    assert "balanced_query_id_sampling" not in path_generation

    assert set(choice_generation["phase_relation_weights"].keys()) == {"in_phase", "opposite_phase"}

    assert set(choice_generation["option_letter_weights"].keys()) == {"A", "B", "C", "D", "E"}

    assert path_generation["path_difference_step_support"] == [1, 2, 3, 4, 5]


    assert int(rendering["canvas_width"]) == 1180

    assert int(rendering["canvas_height"]) == 760

    assert int(rendering["board_width_px"]) == 980

    assert int(rendering["board_height_px"]) == 620

    assert int(rendering["half_wavelength_px"]) == 50

    assert int(rendering["wavefront_width_px"]) == 2

    assert bool(rendering["layout_jitter_enabled"]) is True

    assert int(rendering["layout_jitter_min_margin_px"]) == 18


    assert str(choice_prompt["bundle_id"]) == "physics_wave_interference_v1"

    assert str(choice_prompt["task_key"]) == "interference_point_choice_query"

    assert str(path_prompt["bundle_id"]) == "physics_wave_interference_v1"

    assert str(path_prompt["task_key"]) == "path_difference_value_query"
