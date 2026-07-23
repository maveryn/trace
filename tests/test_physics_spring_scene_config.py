"""Config regression tests for migrated spring scene defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_physics_spring_defaults_expose_scene_and_answer_support() -> None:
    cfg = get_scene_defaults("physics", "spring")
    missing_generation, rendering, missing_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__spring__spring_missing_value",
    )
    difference_generation, _, difference_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__spring__spring_extension_difference",
    )

    assert bool(missing_generation["balanced_scene_variant_sampling"]) is True
    assert bool(missing_generation["balanced_target_answer_sampling"]) is True
    assert bool(missing_generation["balanced_accent_color_name_sampling"]) is True
    assert "query_id_weights" not in missing_generation
    assert "solve_for_weights" not in missing_generation
    assert set(missing_generation["scene_variant_weights"].keys()) == {
        "paired_springs",
        "staggered_springs",
        "textured_spring",
    }

    assert list(missing_generation["scale_factor_support"]) == [1, 2, 3]
    assert list(missing_generation["extension_difference_scale_factor_support"]) == [2]
    assert int(missing_generation["weight_value_max"]) == 12
    assert list(missing_generation["missing_weight_support"]) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert list(missing_generation["missing_extension_support"]) == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ]
    assert int(missing_generation["extension_value_max"]) == 14
    assert list(difference_generation["extension_difference_support"]) == [2, 4, 8, 10, 12]
    assert int(rendering["canvas_width"]) == 980
    assert int(rendering["canvas_height"]) == 660
    assert int(rendering["card_width_px"]) == 304
    assert int(rendering["ruler_value_max"]) == 14
    assert int(rendering["weight_box_width_px"]) == 78
    assert bool(rendering["layout_jitter_enabled"]) is True
    assert int(rendering["layout_jitter_min_margin_px"]) == 14

    assert str(missing_prompt["bundle_id"]) == "physics_spring_v1"
    assert str(missing_prompt["task_key"]) == "spring_missing_value_query"
    assert str(difference_prompt["bundle_id"]) == "physics_spring_v1"
    assert str(difference_prompt["task_key"]) == "spring_extension_difference_query"
