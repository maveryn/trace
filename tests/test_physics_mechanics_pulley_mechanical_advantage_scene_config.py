"""Config regression tests for physics pulley defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_physics_mechanics_pulley_defaults_expose_scene_query_and_answer_support() -> None:
    cfg = get_scene_defaults("physics", "pulley")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__pulley__pulley_mechanical_advantage",
    )


    assert bool(generation["balanced_scene_variant_sampling"]) is True

    assert bool(generation["balanced_target_answer_sampling"]) is True

    assert bool(generation["balanced_accent_color_name_sampling"]) is True

    assert set(generation["scene_variant_weights"].keys()) == {
        "open_block",
        "compact_block",
        "tall_block",
    }

    assert "solve_for_weights" not in generation

    assert "balanced_solve_for_sampling" not in generation

    assert bool(generation["balanced_connected_support_count_sampling"]) is False

    assert bool(generation["balanced_disconnected_segment_count_sampling"]) is False

    assert list(generation["connected_support_count_support"]) == [2, 3, 4, 5, 6]

    assert list(generation["disconnected_segment_count_support"]) == [0, 1, 2, 3, 4]

    assert list(generation["effort_force_support"]) == list(range(4, 19))

    assert int(min(generation["load_force_support"])) == 8

    assert int(max(generation["load_force_support"])) == 108

    assert int(generation["effort_force_min"]) == 4

    assert int(generation["effort_force_max"]) == 18

    assert int(rendering["canvas_width"]) == 1280

    assert int(rendering["canvas_height"]) == 760

    assert int(rendering["support_segment_gap_px"]) > int(rendering["pulley_radius_px"])

    assert int(rendering["load_width_px"]) > 0

    assert bool(rendering["layout_jitter_enabled"]) is True

    assert "query_id_weights" not in generation

    assert "balanced_query_id_sampling" not in generation

    assert str(prompt["bundle_id"]) == "physics_pulley_v1"

    assert str(prompt["task_key"]) == "pulley_mechanical_advantage_query"

    assert {
        "bundle_id",
        "task_key",
        "json_output_contract",
        "json_output_contract_answer_only",
    }.issubset(set(prompt.keys()))

    assert not any("object_description" in key for key in prompt)

    assert not any("annotation_hint" in key for key in prompt)
