"""Config regression tests for the migrated physics manometer scene."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_physics_manometer_defaults_expose_scene_axes_and_supports() -> None:
    cfg = get_scene_defaults("physics", "manometer")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__manometer__pressure_difference_value",
    )

    assert "query_id_weights" not in generation
    assert bool(generation["balanced_target_answer_sampling"]) is True
    assert bool(generation["balanced_height_sampling"]) is True
    assert bool(generation["balanced_kpa_per_cm_sampling"]) is True
    assert set(generation["height_cm_support"]) == set(range(2, 13))
    assert set(generation["kpa_per_cm_support"]) == {1, 2, 3, 4, 5}

    assert int(rendering["canvas_width"]) == 1120
    assert int(rendering["canvas_height"]) == 720
    assert int(rendering["px_per_cm"]) == 17
    assert int(rendering["tube_top_px"]) == 188
    assert int(rendering["tube_bottom_px"]) == 560
    assert int(rendering["tube_width_px"]) == 86
    assert int(rendering["tube_gap_px"]) == 290

    assert str(prompt["bundle_id"]) == "physics_manometer_v1"
    assert str(prompt["task_key"]) == "pressure_difference_value_query"
