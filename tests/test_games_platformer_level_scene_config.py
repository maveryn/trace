"""Config regression tests for games Platformer level defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.games.platformer.shared.state import (
    SUPPORTED_PLATFORMER_SCENE_VARIANTS,
    SUPPORTED_PLATFORMER_STYLE_VARIANTS,
)
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_games_platformer_level_defaults_present() -> None:
    cfg = get_scene_defaults("games", "platformer")
    landing_generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__platformer__jump_landing_label",
    )
    count_generation, _count_rendering, _count_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__platformer__collectible_count",
    )
    score_generation, _score_rendering, _score_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__platformer__jump_collectible_score_value",
    )

    assert bool(landing_generation["balanced_scene_variant_sampling"]) is True
    assert bool(landing_generation["balanced_style_variant_sampling"]) is True
    assert bool(landing_generation["balanced_platform_count_sampling"]) is True
    assert bool(landing_generation["balanced_hazard_count_sampling"]) is True
    assert set(landing_generation["scene_variant_weights"].keys()) == set(SUPPORTED_PLATFORMER_SCENE_VARIANTS)
    assert "query_id_weights" not in landing_generation
    assert "balanced_query_id_sampling" not in landing_generation
    assert set(landing_generation["style_variant_weights"].keys()) == set(SUPPORTED_PLATFORMER_STYLE_VARIANTS)
    assert list(landing_generation["platform_count_support"]) == [4, 5, 6, 7]
    assert list(landing_generation["hazard_count_support"]) == [4, 5, 6, 7, 8]
    assert list(landing_generation["target_platform_label_support"]) == list("ABCDEFGH")
    assert list(count_generation["target_collectible_count_support"]) == [2, 3, 4, 5, 6, 7]
    assert bool(count_generation["balanced_target_collectible_count_sampling"]) is True
    assert list(score_generation["score_on_arc_coin_count_support"]) == [1, 2, 3, 4]
    assert list(score_generation["score_on_arc_bonus_count_support"]) == [1, 2]
    assert list(score_generation["score_off_arc_bonus_count_support"]) == [1, 2, 3]
    assert list(score_generation["score_bonus_value_support"]) == [2, 3, 4]
    assert float(landing_generation["jump_visible_after_peak_min"]) == 0.08
    assert float(landing_generation["jump_visible_after_peak_max"]) == 0.14
    assert int(rendering["canvas_width"]) == 1000
    assert int(rendering["canvas_height"]) == 740
    assert int(rendering["collectible_radius_px"]) > 0
    assert str(prompt["bundle_id"]) == "games_platformer_v1"
