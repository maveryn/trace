"""Config regression tests for physics electrostatic-field defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_physics_electrostatic_field_defaults_expose_scene_and_answer_support() -> None:
    cfg = get_scene_defaults("physics", "electrostatic_field")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__electrostatic_field__field_direction_choice",
    )

    assert bool(generation["balanced_scene_variant_sampling"]) is True
    assert "query_id_weights" not in generation
    assert bool(generation["balanced_direction_mode_sampling"]) is True
    assert bool(generation["balanced_target_direction_sampling"]) is True
    assert bool(generation["balanced_direction_option_letter_sampling"]) is True
    assert bool(generation["balanced_point_option_letter_sampling"]) is True
    assert bool(generation["balanced_target_answer_sampling"]) is True
    assert set(generation["scene_variant_weights"].keys()) == {"clean_grid", "paper_grid", "dense_grid"}
    assert set(generation["direction_mode_weights"].keys()) == {
        "electric_field_direction",
        "force_on_positive_charge",
        "force_on_negative_charge",
    }
    assert set(generation["target_direction_weights"].keys()) == {
        "east",
        "northeast",
        "north",
        "northwest",
        "west",
        "southwest",
        "south",
        "southeast",
    }
    assert set(generation["direction_option_letter_weights"].keys()) == {"A", "B", "C", "D", "E", "F", "G", "H"}
    assert set(generation["point_option_letter_weights"].keys()) == {"A", "B", "C", "D", "E", "F"}
    assert -9 in generation["potential_answer_support"]
    assert 9 in generation["potential_answer_support"]

    assert int(rendering["canvas_width"]) == 1180
    assert int(rendering["canvas_height"]) == 760
    assert int(rendering["board_width_px"]) == 760
    assert int(rendering["option_cell_width_px"]) == 140
    assert bool(rendering["layout_jitter_enabled"]) is True

    assert str(prompt["bundle_id"]) == "physics_electrostatic_field_v1"
    assert str(prompt["task_key"]) == "field_direction_choice_query"
