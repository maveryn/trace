from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_vernier_caliper_scene_defaults_are_scene_keyed() -> None:
    scene = get_scene_defaults("physics", "vernier_caliper")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        scene,
        task_id="task_physics__vernier_caliper__length_readout_value",
    )

    assert "query_id_weights" not in generation
    assert "balanced_query_id_sampling" not in generation
    assert set(generation["main_mm_support"]) == set(range(8, 56))
    assert set(generation["aligned_vernier_tick_support"]) == set(range(1, 10))
    assert bool(generation["balanced_target_answer_sampling"]) is True
    assert bool(generation["balanced_correct_option_letter_sampling"]) is True
    assert set(generation["correct_option_letter_weights"]) == {"A", "B", "C", "D", "E", "F"}

    assert int(rendering["canvas_width"]) == 1180
    assert int(rendering["canvas_height"]) == 720
    assert int(rendering["main_scale_max_mm"]) == 62
    assert int(rendering["mm_px"]) == 13
    assert int(rendering["option_cell_height_px"]) == 42

    assert str(prompt["bundle_id"]) == "physics_vernier_caliper_v1"
    assert str(prompt["task_key"]) == "vernier_caliper_readout_query"
