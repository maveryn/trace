"""Config regression tests for games Chess-board defaults."""
from __future__ import annotations
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.games.shared.style import SUPPORTED_CHESS_STYLE_VARIANTS, build_games_chess_theme
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.text_legibility import contrast_ratio

def test_games_chess_board_defaults_present() -> None:
    cfg = get_scene_defaults('games', 'chess')
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(cfg, task_id='task_games__chess__marked_piece_destination_count')
    assert bool(generation['balanced_scene_variant_sampling']) is True
    assert bool(generation['balanced_style_variant_sampling']) is True
    assert bool(generation['balanced_target_answer_sampling']) is True
    assert set(generation['scene_variant_weights'].keys()) == {'sparse_board', 'crowded_board'}
    assert 'balanced_query_id_sampling' not in generation
    assert 'query_id_weights' not in generation
    assert bool(generation['balanced_marked_piece_kind_sampling']) is True
    assert list(generation['marked_piece_destination_count_support']) == [1, 2, 3, 4, 5, 6]
    assert set(generation['marked_piece_kind_weights'].keys()) == {'knight', 'bishop', 'rook', 'queen'}
    capture_generation, _capture_rendering, _capture_prompt = split_generation_rendering_prompt_defaults(cfg, task_id='task_games__chess__marked_piece_capture_count')
    player_generation, _player_rendering, _player_prompt = split_generation_rendering_prompt_defaults(cfg, task_id='task_games__chess__player_capture_piece_count')
    attacker_generation, _attacker_rendering, _attacker_prompt = split_generation_rendering_prompt_defaults(cfg, task_id='task_games__chess__target_square_attacker_count')
    escape_generation, _escape_rendering, _escape_prompt = split_generation_rendering_prompt_defaults(cfg, task_id='task_games__chess__king_escape_square_count')
    piece_generation, _piece_rendering, _piece_prompt = split_generation_rendering_prompt_defaults(cfg, task_id='task_games__chess__piece_kind_count')
    colored_piece_generation, _colored_piece_rendering, _colored_piece_prompt = split_generation_rendering_prompt_defaults(cfg, task_id='task_games__chess__colored_piece_kind_count')
    checkmate_generation, _checkmate_rendering, _checkmate_prompt = split_generation_rendering_prompt_defaults(cfg, task_id='task_games__chess__checkmate_move_label')
    assert list(capture_generation['marked_piece_capture_count_support']) == [0, 1, 2, 3, 4]
    assert set(capture_generation['marked_piece_kind_weights'].keys()) == {'knight', 'bishop', 'rook', 'queen'}
    assert list(player_generation['player_capture_piece_count_support']) == [1, 2, 3, 4, 5, 6]
    assert list(attacker_generation['target_square_attacker_count_support']) == [0, 1, 2, 3, 4]
    assert list(escape_generation['king_escape_square_count_support']) == [0, 1, 2, 3, 4, 5]
    assert list(piece_generation['piece_type_count_support']) == [0, 1, 2, 3, 4, 5, 6]
    assert list(piece_generation['piece_count_distractor_count_support']) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert set(piece_generation['target_piece_kind_weights'].keys()) == {'king', 'queen', 'rook', 'bishop', 'knight', 'pawn'}
    assert list(colored_piece_generation['piece_type_count_support']) == [0, 1, 2, 3, 4, 5, 6]
    assert set(colored_piece_generation['target_piece_kind_weights'].keys()) == {'king', 'queen', 'rook', 'bishop', 'knight', 'pawn'}
    assert set(colored_piece_generation['target_piece_color_weights'].keys()) == {'white', 'black'}
    assert list(checkmate_generation['checkmate_option_count_support']) == [4, 6]
    assert int(rendering['max_board_size_px']) > 0
    assert int(rendering['marked_square_outline_width_px']) > 0
    assert bool(rendering['dynamic_canvas_size_enabled']) is True
    assert int(rendering['canvas_min_width_px']) > 0
    assert int(rendering['canvas_min_height_px']) > 0
    assert str(prompt['bundle_id']) == 'games_chess_v1'
    assert str(prompt['scene_key']) == 'visible_chess_board'
    assert str(prompt['task_key']) == 'chess_board_query'

def test_games_chess_board_themes_keep_fixed_pieces_readable() -> None:
    for style_variant in SUPPORTED_CHESS_STYLE_VARIANTS:
        theme = build_games_chess_theme(style_variant=style_variant)
        assert theme.piece_rendering == 'glyph'
        assert theme.white_piece_fill_rgb == (255, 255, 255)
        assert theme.white_piece_outline_rgb == (0, 0, 0)
        assert theme.black_piece_fill_rgb == (0, 0, 0)
        assert theme.black_piece_outline_rgb == (255, 255, 255)
        assert theme.marked_square_outline_rgb == (220, 38, 38)
        assert theme.marked_square_fill_rgba == (220, 38, 38, 0)
        for square_rgb in (theme.light_square_rgb, theme.dark_square_rgb):
            assert contrast_ratio(theme.white_piece_fill_rgb, square_rgb) >= 1.9
            assert contrast_ratio(theme.black_piece_fill_rgb, square_rgb) >= 2.8
