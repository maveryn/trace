"""Config regression tests for games Mini-golf course defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.games.minigolf.shared.defaults import SCENE_VARIANTS, STYLE_VARIANTS
from trace_tasks.tasks.shared.config_defaults import required_group_default, split_generation_rendering_prompt_defaults


def test_games_minigolf_course_defaults_present() -> None:
    cfg = get_scene_defaults("games", "minigolf")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__minigolf__first_obstacle_label",
    )

    assert bool(generation["balanced_scene_variant_sampling"]) is True
    assert bool(generation["balanced_style_variant_sampling"]) is True
    assert bool(generation["balanced_obstacle_count_sampling"]) is True
    assert bool(generation["balanced_target_obstacle_label_sampling"]) is True
    assert set(generation["scene_variant_weights"].keys()) == set(SCENE_VARIANTS)
    assert set(generation["style_variant_weights"].keys()) == set(STYLE_VARIANTS)
    assert list(generation["obstacle_count_support"]) == [4, 6]
    assert "target_obstacle_label_support" not in generation
    assert int(rendering["canvas_width"]) == 1000
    assert int(rendering["canvas_height"]) == 740
    assert int(rendering["obstacle_radius_px"]) > 0
    assert str(prompt["bundle_id"]) == "games_minigolf_v1"
    assert "straight line" in str(required_group_default(prompt, "minigolf_cue_rule_text", context="minigolf prompt")).lower()
    assert "[x, y] pixel point" in str(
        required_group_default(prompt, "annotation_hint_first_obstacle_label", context="minigolf prompt")
    )
    assert "path segment" in str(required_group_default(prompt, "annotation_hint_shot_path_label", context="minigolf prompt"))


def test_games_minigolf_shot_path_task_defaults_present() -> None:
    cfg = get_scene_defaults("games", "minigolf")
    generation, _rendering, _prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__minigolf__shot_path_label",
    )

    assert bool(generation["balanced_path_option_count_sampling"]) is True
    assert bool(generation["balanced_target_path_sampling"]) is True
    assert list(generation["path_option_count_support"]) == [4, 5, 6]
    assert list(generation["target_path_index_support"]) == list(range(6))
