"""Regression tests for the games shared style facade."""

from __future__ import annotations

import math

from trace_tasks.tasks.games.shared import style


def test_games_shared_style_facade_exports_all_public_symbols() -> None:
    missing = [name for name in style.__all__ if not hasattr(style, name)]

    assert missing == []


def test_games_shared_style_builders_cover_supported_variants() -> None:
    builder_support_pairs = (
        ("build_games_card_theme", "SUPPORTED_CARD_STYLE_VARIANTS"),
        ("build_games_domino_theme", "SUPPORTED_DOMINO_STYLE_VARIANTS"),
        ("build_games_reversi_theme", "SUPPORTED_REVERSI_STYLE_VARIANTS"),
        ("build_games_connect_four_theme", "SUPPORTED_CONNECT_FOUR_STYLE_VARIANTS"),
        ("build_games_checkers_theme", "SUPPORTED_CHECKERS_STYLE_VARIANTS"),
        ("build_games_chess_theme", "SUPPORTED_CHESS_STYLE_VARIANTS"),
        ("build_games_bingo_theme", "SUPPORTED_BINGO_STYLE_VARIANTS"),
        ("build_games_dots_and_boxes_theme", "SUPPORTED_DOTS_AND_BOXES_STYLE_VARIANTS"),
        ("build_games_nine_mens_morris_theme", "SUPPORTED_NINE_MENS_MORRIS_STYLE_VARIANTS"),
        ("build_games_go_theme", "SUPPORTED_GO_STYLE_VARIANTS"),
        ("build_games_minesweeper_theme", "SUPPORTED_MINESWEEPER_STYLE_VARIANTS"),
        ("build_games_battleship_theme", "SUPPORTED_BATTLESHIP_STYLE_VARIANTS"),
        ("build_games_pool_theme", "SUPPORTED_POOL_STYLE_VARIANTS"),
        ("build_games_hex_theme", "SUPPORTED_HEX_STYLE_VARIANTS"),
    )

    for builder_name, support_name in builder_support_pairs:
        builder = getattr(style, builder_name)
        support = getattr(style, support_name)
        assert support
        for variant in support:
            theme = builder(style_variant=str(variant))
            assert theme.__class__.__name__.endswith("Theme")


def test_games_shared_style_probability_map_stays_uniform() -> None:
    probabilities = style.style_probability_map()

    assert set(probabilities) == set(style.SUPPORTED_GAMES_STYLE_VARIANTS)
    assert math.isclose(sum(probabilities.values()), 1.0)
    assert set(probabilities.values()) == {1.0 / len(style.SUPPORTED_GAMES_STYLE_VARIANTS)}
