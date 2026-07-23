"""Config regression tests for physics optics defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_physics_optics_ray_defaults_expose_scene_query_and_answer_support() -> None:
    cfg = get_scene_defaults("physics", "ray_optics")
    bounce_generation, rendering, bounce_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__ray_optics__ray_bounce_count",
    )
    target_generation, _target_rendering, target_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__ray_optics__ray_target_hit_count",
    )


    assert bool(bounce_generation["balanced_scene_variant_sampling"]) is True

    assert bool(bounce_generation["balanced_target_answer_sampling"]) is True

    assert bool(bounce_generation["balanced_accent_color_name_sampling"]) is True

    assert set(bounce_generation["scene_variant_weights"].keys()) == {
        "quad_mirror",
        "five_mirror",
    }

    assert set(target_generation["scene_variant_weights"].keys()) == {
        "single_mirror",
        "double_mirror",
        "triple_mirror",
    }

    assert list(target_generation["bounce_count_support_single_mirror"]) == [0, 1]

    assert list(target_generation["bounce_count_support_double_mirror"]) == [0, 1, 2]

    assert list(target_generation["bounce_count_support_triple_mirror"]) == [0, 1, 2, 3]

    assert list(bounce_generation["bounce_count_support_quad_mirror"]) == [0, 1, 2, 3, 4]

    assert list(bounce_generation["bounce_count_support_five_mirror"]) == [1, 2, 3, 4, 5]

    assert list(target_generation["target_hit_count_support"]) == [1, 2, 3, 4, 5]

    assert int(target_generation["target_count_max"]) == 5

    assert int(rendering["board_cols"]) == 8

    assert int(rendering["cell_size_px"]) > 0

    assert int(rendering["mirror_width_px"]) == 7

    assert int(rendering["mirror_padding_px"]) == 6

    assert int(rendering["target_radius_px"]) == 18

    assert bool(rendering["layout_jitter_enabled"]) is True

    assert int(rendering["layout_jitter_min_margin_px"]) == 8

    assert str(bounce_prompt["bundle_id"]) == "physics_ray_optics_v1"

    assert str(bounce_prompt["task_key"]) == "ray_bounce_count"

    assert str(target_prompt["task_key"]) == "ray_target_hit_count"
