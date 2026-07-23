"""Config regression tests for games nine-men's-morris defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.games.shared.style import SUPPORTED_NINE_MENS_MORRIS_STYLE_VARIANTS
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_games_nine_mens_morris_scene_defaults_present() -> None:
    cfg = get_scene_defaults("games", "nine_mens_morris")
    pieces_generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__nine_mens_morris__pieces_in_mill_count",
    )
    completion_generation, _completion_rendering, completion_prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__nine_mens_morris__mill_completion_point_count",
    )

    assert bool(pieces_generation["balanced_scene_variant_sampling"]) is True
    assert bool(pieces_generation["balanced_style_variant_sampling"]) is True
    assert bool(pieces_generation["balanced_target_answer_sampling"]) is True
    assert set(pieces_generation["scene_variant_weights"].keys()) == {"single_board"}
    assert "balanced_query_id_sampling" not in pieces_generation
    assert "query_id_weights" not in pieces_generation
    assert set(pieces_generation["style_variant_weights"].keys()) == set(SUPPORTED_NINE_MENS_MORRIS_STYLE_VARIANTS)
    assert list(pieces_generation["all_pieces_in_mill_count_support"]) == [0, 3, 5, 6, 7, 8, 9]
    assert "white_mill_completion_point_count_support" not in pieces_generation
    assert "black_mill_completion_point_count_support" not in pieces_generation
    assert list(completion_generation["white_mill_completion_point_count_support"]) == [0, 1, 2, 3, 4, 5]
    assert list(completion_generation["black_mill_completion_point_count_support"]) == [0, 1, 2, 3, 4, 5]
    assert int(rendering["board_width_px"]) > 0
    assert int(rendering["piece_radius_px"]) > 0
    assert str(prompt["bundle_id"]) == "games_nine_mens_morris_v1"
    assert str(completion_prompt["bundle_id"]) == "games_nine_mens_morris_v1"
