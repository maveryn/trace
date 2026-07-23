"""Config regression tests for physics circuit-equivalent defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_physics_circuit_equivalent_defaults_expose_scene_task_and_answer_support() -> None:
    cfg = get_scene_defaults("physics", "circuit_equivalent")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__circuit_equivalent__total_resistance_value",
    )

    assert bool(generation["balanced_scene_variant_sampling"]) is True
    assert bool(generation["balanced_target_answer_sampling"]) is True
    assert bool(generation["balanced_accent_color_name_sampling"]) is True
    assert set(generation["scene_variant_weights"].keys()) == {
        "series_parallel",
    }
    assert "query_id_weights" not in generation
    assert "balanced_query_id_sampling" not in generation
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
    assert list(generation["total_resistance_target_answer_support"]) == list(range(1, 21))
    assert list(generation["total_capacitance_target_answer_support"]) == list(range(1, 21))
    assert int(generation["component_value_max"]) == 60
    assert list(generation["parallel_component_count_options"]) == [2, 3, 4]
    assert list(generation["parallel_block_count_options"]) == [1, 2]
    assert list(generation["series_parallel_branch_count_options"]) == [2, 3]

    assert int(rendering["component_symbol_width_px"]) > 0
    assert int(rendering["component_symbol_height_px"]) > 0
    assert int(rendering["wire_width_px"]) > 0
    assert bool(rendering["layout_jitter_enabled"]) is True

    assert str(prompt["bundle_id"]) == "physics_circuit_equivalent_v1"
    assert str(prompt["task_key"]) == "total_resistance_value_query"

    _, _, capacitance_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__circuit_equivalent__total_capacitance_value",
    )
    assert str(capacitance_prompt["bundle_id"]) == "physics_circuit_equivalent_v1"
    assert str(capacitance_prompt["task_key"]) == "total_capacitance_value_query"
