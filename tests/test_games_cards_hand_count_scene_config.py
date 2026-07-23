"""Config regression tests for games cards scene defaults."""
from __future__ import annotations
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults

def test_games_cards_scene_defaults_expose_shared_rendering_and_task_axes() -> None:
    cfg = get_scene_defaults('games', 'cards')
    shared_generation = cfg['generation']['shared']
    assert 'query_id_weights' not in shared_generation
    assert 'balanced_query_id_sampling' not in shared_generation
    assert 'balanced_target_answer_sampling' not in shared_generation
    assert 'balanced_card_count_sampling' not in shared_generation
    assert 'balanced_option_count_sampling' not in shared_generation
    assert bool(shared_generation['balanced_scene_variant_sampling']) is True
    assert bool(shared_generation['balanced_style_variant_sampling']) is True
    assert set(shared_generation['scene_variant_weights'].keys()) == {'multi_row'}
    assert set(shared_generation['style_variant_weights'].keys()) == {'classic', 'soft', 'outlined', 'ivory', 'slate'}
    same_generation, rendering, prompt = split_generation_rendering_prompt_defaults(cfg, task_id='task_games__cards__same_suit_as_reference_count')
    assert bool(same_generation['balanced_target_answer_sampling']) is True
    assert bool(same_generation['balanced_card_count_sampling']) is True
    assert list(same_generation['same_suit_target_answer_support']) == [0, 1, 2, 3, 4, 5]
    assert bool(same_generation['same_suit_order_by_suit']) is False
    assert list(same_generation['same_suit_as_reference_count_card_count_support']) == list(range(16, 27))
    higher_generation, _rendering, _prompt = split_generation_rendering_prompt_defaults(cfg, task_id='task_games__cards__higher_than_reference_count')
    assert bool(higher_generation['balanced_target_answer_sampling']) is True
    assert bool(higher_generation['balanced_card_count_sampling']) is True
    assert list(higher_generation['higher_rank_target_answer_support']) == [0, 1, 2, 3, 4, 5]
    assert bool(higher_generation['higher_rank_order_by_rank']) is False
    assert list(higher_generation['higher_than_reference_count_card_count_support']) == list(range(10, 16))
    triple_generation, _rendering, _prompt = split_generation_rendering_prompt_defaults(cfg, task_id='task_games__cards__exact_triple_count')
    assert bool(triple_generation['balanced_target_answer_sampling']) is True
    assert bool(triple_generation['balanced_card_count_sampling']) is True
    assert list(triple_generation['exact_triple_count_support']) == [0, 1, 2, 3, 4]
    assert list(triple_generation['exact_triple_count_card_count_support']) == list(range(12, 23))
    run_generation, _rendering, _prompt = split_generation_rendering_prompt_defaults(cfg, task_id='task_games__cards__longest_run_length')
    assert bool(run_generation['balanced_target_answer_sampling']) is True
    assert bool(run_generation['balanced_card_count_sampling']) is True
    assert list(run_generation['longest_run_length_support']) == [2, 3, 4, 5, 6]
    assert list(run_generation['card_count_support']) == list(range(16, 41))
    blackjack_generation, _rendering, _prompt = split_generation_rendering_prompt_defaults(cfg, task_id='task_games__cards__blackjack_best_hand_label')
    assert bool(blackjack_generation['balanced_option_count_sampling']) is True
    assert list(blackjack_generation['blackjack_hand_count_support']) == [4, 6]
    assert list(blackjack_generation['blackjack_cards_per_hand_support']) == [3, 4]
    poker_draw_generation, _rendering, _prompt = split_generation_rendering_prompt_defaults(cfg, task_id='task_games__cards__poker_draw_card_label')
    assert list(poker_draw_generation['poker_draw_candidate_count_support']) == [4, 6]
    assert bool(poker_draw_generation['balanced_poker_draw_target_category_sampling']) is True
    assert set(poker_draw_generation['poker_draw_target_category_weights']) >= {'straight', 'flush', 'full_house', 'four_of_a_kind', 'straight_flush'}
    missing_generation, _rendering, _prompt = split_generation_rendering_prompt_defaults(cfg, task_id='task_games__cards__missing_card_to_complete_hand_label')
    assert 'missing_card_query_id_weights' not in missing_generation
    assert 'balanced_missing_card_query_id_sampling' not in missing_generation
    assert list(missing_generation['missing_card_candidate_count_support']) == [4, 6]
    trick_generation, _rendering, _prompt = split_generation_rendering_prompt_defaults(cfg, task_id='task_games__cards__trick_winning_play_label')
    assert list(trick_generation['trick_play_candidate_count_support']) == [4, 6]
    assert list(trick_generation['trick_play_played_count_support']) == [3, 4]
    assert set(trick_generation['trick_play_trump_mode_weights']) == {'no_trump', 'with_trump'}
    assert int(rendering['card_width_px']) > 0
    assert int(rendering['card_height_px']) > 0
    assert int(rendering['max_cards_per_row']) == 8
    assert int(rendering['group_label_font_size_px']) > 0
    assert int(rendering['continuation_font_size_px']) > 0
    assert str(prompt['bundle_id']) == 'games_cards_v1'
    assert str(prompt['scene_key']) == 'visible_card_hand'
    assert str(prompt['task_key']) == 'cards_hand_query'
