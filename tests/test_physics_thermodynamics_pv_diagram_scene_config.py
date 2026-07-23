"""Config regression tests for physics PV-diagram defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_physics_pv_defaults_expose_scene_and_answer_support() -> None:
    cfg = get_scene_defaults("physics", "pv_diagram")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__pv_diagram__pv_work_value",
    )

    assert bool(generation["balanced_scene_variant_sampling"]) is True
    assert bool(generation["balanced_work_mode_sampling"]) is True
    assert bool(generation["balanced_target_answer_sampling"]) is True
    assert set(generation["scene_variant_weights"].keys()) == {"clean_grid", "paper_grid", "bold_grid"}
    assert set(generation["work_mode_weights"].keys()) == {"single_process", "rectangular_cycle"}
    assert list(generation["pressure_support"]) == [2, 3, 4, 5, 6]
    assert list(generation["volume_support"]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert list(generation["work_answer_support"]) == [
        -24,
        -20,
        -18,
        -16,
        -15,
        -12,
        -10,
        -9,
        -8,
        -6,
        -4,
        4,
        6,
        8,
        9,
        10,
        12,
        15,
        16,
        18,
        20,
        24,
    ]
    assert int(rendering["canvas_width"]) == 1180
    assert int(rendering["canvas_height"]) == 760
    assert int(rendering["plot_width_px"]) == 760
    assert int(rendering["mini_cell_width_px"]) == 262
    assert bool(rendering["layout_jitter_enabled"]) is True
    assert int(rendering["layout_jitter_min_margin_px"]) == 8
    assert str(prompt["bundle_id"]) == "physics_pv_diagram_v1"
    assert str(prompt["task_key"]) == "pv_work_value"


def test_physics_pv_sign_choice_defaults_expose_option_support() -> None:
    cfg = get_scene_defaults("physics", "pv_diagram")
    generation, _rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__pv_diagram__pv_process_sign_choice",
    )

    assert bool(generation["balanced_target_sign_sampling"]) is True
    assert bool(generation["balanced_correct_option_letter_sampling"]) is True
    assert set(generation["target_sign_weights"].keys()) == {"positive", "negative", "zero"}
    assert set(generation["correct_option_letter_weights"].keys()) == {
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
    }
    assert str(prompt["bundle_id"]) == "physics_pv_diagram_v1"
    assert str(prompt["task_key"]) == "pv_process_sign_choice"
