"""Config regression tests for games Brick-breaker defaults."""

from __future__ import annotations

from trace_tasks.tasks.games.brick_breaker.shared.state import (
    SUPPORTED_BRICK_BREAKER_SCENE_VARIANTS,
    SUPPORTED_BRICK_BREAKER_STYLE_VARIANTS,
)
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults


def test_games_brick_breaker_defaults_present() -> None:
    generation, rendering, prompt = load_scene_generation_rendering_prompt_defaults(
        "games",
        "brick_breaker",
        task_id="task_games__brick_breaker__next_hit_label",
    )
    row_generation, _row_rendering, row_prompt = load_scene_generation_rendering_prompt_defaults(
        "games",
        "brick_breaker",
        task_id="task_games__brick_breaker__hit_row_remaining_count",
    )

    assert bool(generation["balanced_scene_variant_sampling"]) is True
    assert bool(generation["balanced_style_variant_sampling"]) is True
    assert bool(generation["balanced_brick_row_sampling"]) is True
    assert bool(generation["balanced_brick_col_sampling"]) is True
    assert bool(generation["balanced_lane_count_sampling"]) is True
    assert "query_id_weights" not in generation
    assert "balanced_query_id_sampling" not in generation
    assert set(generation["scene_variant_weights"].keys()) == set(SUPPORTED_BRICK_BREAKER_SCENE_VARIANTS)
    assert set(generation["style_variant_weights"].keys()) == set(SUPPORTED_BRICK_BREAKER_STYLE_VARIANTS)
    assert len(SUPPORTED_BRICK_BREAKER_STYLE_VARIANTS) >= 5
    assert list(generation["brick_row_count_support"]) == [4, 5]
    assert list(generation["brick_col_count_support"]) == [5, 6]
    assert list(generation["catch_lane_count_support"]) == [5, 6]
    assert int(rendering["canvas_width"]) == 980
    assert int(rendering["canvas_height"]) == 740
    assert bool(rendering["dynamic_canvas_size_enabled"]) is True
    assert int(rendering["ball_radius_px"]) > 0
    assert str(prompt["bundle_id"]) == "games_brick_breaker_v1"
    assert str(row_prompt["bundle_id"]) == "games_brick_breaker_v1"
    assert {"bundle_id", "scene_key", "task_key"}.issubset(prompt.keys())
    assert "brick_breaker_motion_rule_text" not in prompt
    assert "annotation_hint_next_hit_label" not in prompt
    assert "answer_hint_hit_row_remaining_count" not in prompt
    assert bool(row_generation["balanced_row_remaining_count_sampling"]) is True
    assert list(row_generation["row_remaining_count_support"]) == [1, 2, 3, 4, 5]
