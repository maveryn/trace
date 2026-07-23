"""Config regression tests for games dominoes defaults."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def test_games_dominoes_defaults_expose_scene_and_candidate_axes() -> None:
    cfg = get_scene_defaults("games", "dominoes")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__dominoes__double_count",
    )

    assert bool(generation["balanced_scene_variant_sampling"]) is True
    assert bool(generation["balanced_style_variant_sampling"]) is True
    assert bool(generation["balanced_target_answer_sampling"]) is True
    assert bool(generation["balanced_candidate_count_sampling"]) is True
    assert "query_id_weights" not in generation
    assert set(generation["scene_variant_weights"].keys()) == {"single_row", "two_row"}
    assert set(generation["style_variant_weights"].keys()) == {
        "classic",
        "soft",
        "outlined",
        "ivory",
        "charcoal_tile",
        "wood_tile",
    }
    assert list(generation["double_target_answer_support"]) == [0, 1, 2, 3, 4, 5]
    assert list(generation["single_row_candidate_count_support"]) == [7, 8, 9]
    assert list(generation["two_row_candidate_count_support"]) == [10, 11, 12]
    assert int(rendering["chain_gap_px"]) < int(rendering["candidate_gap_px"])
    assert int(rendering["tile_width_px"]) > 0
    assert int(rendering["tile_height_px"]) > 0
    assert int(rendering["reference_tag_font_size_px"]) > 0
    assert int(rendering["section_label_font_size_px"]) > 0
    assert int(rendering["section_separator_width_px"]) > 0
    assert str(prompt["bundle_id"]) == "games_dominoes_v1"
    assert str(prompt["scene_key"]) == "visible_domino_chain"
    assert str(prompt["task_key"]) == "domino_chain_query"


def test_games_dominoes_task_overrides_remain_task_owned() -> None:
    cfg = get_scene_defaults("games", "dominoes")
    expected = {
        "task_games__dominoes__matching_end_count": ("matching_end_target_answer_support", [0, 1, 2, 3, 4, 5]),
        "task_games__dominoes__longest_chain_length_value": ("longest_chain_length_answer_support", [1, 2, 3, 4, 5]),
        "task_games__dominoes__invalid_join_label": ("invalid_join_label_support", ["A", "B", "C", "D", "E", "F"]),
        "task_games__dominoes__higher_sum_than_reference_count": ("higher_sum_target_answer_support", [0, 1, 2, 3, 4, 5]),
        "task_games__dominoes__sum_to_target_count": ("sum_to_target_answer_support", [0, 1, 2, 3, 4]),
        "task_games__dominoes__double_count": ("double_target_answer_support", [0, 1, 2, 3, 4, 5]),
    }
    for task_id, (support_key, support) in expected.items():
        generation, _rendering, _prompt = split_generation_rendering_prompt_defaults(cfg, task_id=task_id)
        assert list(generation[support_key]) == support
    generation, _rendering, _prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__dominoes__longest_chain_length_value",
    )
    assert list(generation["single_row_candidate_count_support"]) == [7]
    assert list(generation["two_row_candidate_count_support"]) == [7]
    generation, _rendering, _prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_games__dominoes__sum_to_target_count",
    )
    assert list(generation["sum_target_total_support"]) == [2, 3, 4, 5, 6, 7, 8, 9, 10]
