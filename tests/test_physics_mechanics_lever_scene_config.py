"""Config regression tests for physics lever-balance defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_physics_lever_defaults_expose_scene_axes_and_answer_support() -> None:
    cfg = get_scene_defaults("physics", "lever")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__lever__side_torque_value",
    )


    assert bool(generation["balanced_scene_variant_sampling"]) is True

    assert bool(generation["balanced_target_answer_sampling"]) is True

    assert bool(generation["balanced_accent_color_name_sampling"]) is True

    assert set(generation["scene_variant_weights"].keys()) == {
        "center_fulcrum",
        "offset_fulcrum",
        "textured_beam",
    }

    assert set(generation["accent_color_name_weights"].keys()) == {
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "brown",
        "cyan",
        "magenta",
        "maroon",
    }

    assert set(generation["torque_side_weights"].keys()) == {"left", "right"}

    assert list(generation["distance_support"]) == list(range(1, 9))

    assert list(generation["missing_weight_support"]) == list(range(1, 7))

    assert int(generation["max_side_weights"]) == 4

    assert int(generation["missing_weight_max_side_weights"]) == 2

    assert set(generation["missing_weight_scene_variant_weights"].keys()) == {"textured_beam"}

    assert int(rendering["canvas_width"]) == 1280

    assert int(rendering["beam_width_px"]) > 0

    assert list(rendering["distance_support"]) == list(range(1, 9))

    assert int(rendering["weight_box_width_px"]) > 0
    assert bool(rendering["layout_jitter_enabled"]) is True

    assert str(prompt["bundle_id"]) == "physics_lever_v1"

    assert str(prompt["task_key"]) == "side_torque_value_query"


def test_physics_lever_missing_weight_prompt_override() -> None:
    cfg = get_scene_defaults("physics", "lever")
    generation, _rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__lever__missing_weight_balance_value",
    )

    assert set(generation["missing_weight_scene_variant_weights"].keys()) == {"textured_beam"}

    assert str(prompt["bundle_id"]) == "physics_lever_v1"

    assert str(prompt["task_key"]) == "missing_weight_balance_value_query"
