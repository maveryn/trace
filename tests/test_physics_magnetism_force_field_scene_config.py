"""Config regression tests for magnetic-force scene defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_physics_magnetic_force_defaults_expose_scene_axes_and_supports() -> None:
    cfg = get_scene_defaults("physics", "magnetic_force")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__magnetic_force__force_direction_choice",
    )


    assert bool(generation["balanced_scene_variant_sampling"]) is True

    assert bool(generation["balanced_field_orientation_sampling"]) is True

    assert bool(generation["balanced_velocity_direction_sampling"]) is True

    assert bool(generation["balanced_charge_sign_sampling"]) is True

    assert bool(generation["balanced_direction_option_letter_sampling"]) is True

    assert set(generation["scene_variant_weights"].keys()) == {"field_grid"}

    assert set(generation["field_orientation_weights"].keys()) == {"out_of_page", "into_page"}

    assert set(generation["velocity_direction_weights"].keys()) == {
        "east",
        "northeast",
        "north",
        "northwest",
        "west",
        "southwest",
        "south",
        "southeast",
    }

    assert set(generation["direction_option_letter_weights"].keys()) == {"B", "C", "D", "E", "G", "H"}


    assert int(rendering["canvas_width"]) == 1180

    assert int(rendering["canvas_height"]) == 760

    assert int(rendering["panel_width_px"]) == 760

    assert int(rendering["arrow_length_px"]) == 148

    assert int(rendering["arrow_width_px"]) == 9

    assert int(rendering["option_cell_width_px"]) == 126

    assert int(rendering["option_arrow_length_px"]) == 64

    assert int(rendering["option_arrow_width_px"]) == 8

    assert int(rendering["option_arrow_head_length_px"]) == 20

    assert int(rendering["option_arrow_head_width_px"]) == 18

    assert int(rendering["particle_font_size_px"]) == 31

    assert bool(rendering["layout_jitter_enabled"]) is True


    assert str(prompt["bundle_id"]) == "physics_magnetic_force_v1"

    assert str(prompt["task_key"]) == "force_direction_choice_query"

    assert "query_id_weights" not in generation
