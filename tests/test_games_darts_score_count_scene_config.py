"""Config regression tests for simplified games darts defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_games_darts_defaults_expose_simplified_scene_style_and_prompt_axes() -> None:
    cfg = get_scene_defaults("games", "darts")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__darts__dart_score_value",
    )

    shared_generation = cfg["generation"]["shared"]
    assert "query_id_weights" not in shared_generation
    assert "balanced_query_id_sampling" not in shared_generation
    assert bool(generation["balanced_scene_variant_sampling"]) is True
    assert bool(generation["balanced_style_variant_sampling"]) is True
    assert bool(generation["balanced_score_value_sampling"]) is True
    assert set(generation["scene_variant_weights"].keys()) == {"single_board"}
    assert set(generation["style_variant_weights"].keys()) == {
        "classic",
        "soft",
        "outlined",
        "league_blue",
        "parchment",
        "neon",
    }
    assert list(generation["score_value_support"]) == list(range(1, 11)) + [50]
    assert int(rendering["board_radius_px"]) > 0
    assert int(rendering["marker_radius_px"]) > 0
    assert str(prompt["bundle_id"]) == "games_darts_v1"
    assert str(prompt["task_key"]) == "darts_query"


def test_games_darts_count_task_overrides_are_task_owned() -> None:
    cfg = get_scene_defaults("games", "darts")
    bull_generation, _rendering, _prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__darts__bullseye_membership_count",
    )

    assert bool(bull_generation["balanced_target_answer_sampling"]) is True
    assert list(bull_generation["count_target_answer_support"]) == [0, 1, 2, 3, 4, 5]
    assert list(bull_generation["count_query_dart_count_support"]) == [4, 5, 6, 7]

    label_generation, _rendering, _prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__darts__highest_scoring_dart_label",
    )
    assert bool(label_generation["balanced_target_answer_sampling"]) is True
    assert list(label_generation["highest_scoring_dart_label_support"]) == [0, 1, 2, 3]
