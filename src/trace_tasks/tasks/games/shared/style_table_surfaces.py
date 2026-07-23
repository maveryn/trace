"""Table-surface style helpers for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


SUPPORTED_DARTS_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "soft",
    "outlined",
    "league_blue",
    "parchment",
    "neon",
)


SUPPORTED_POOL_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "tournament_blue",
    "burgundy",
    "charcoal",
    "light_rail",
)


@dataclass(frozen=True)
class PoolTheme:
    """Resolved Pool-table palette for one style variant."""

    rail_rgb: Tuple[int, int, int]
    rail_outline_rgb: Tuple[int, int, int]
    cloth_rgb: Tuple[int, int, int]
    cloth_line_rgb: Tuple[int, int, int]
    pocket_rgb: Tuple[int, int, int]
    pocket_outline_rgb: Tuple[int, int, int]
    ball_outline_rgb: Tuple[int, int, int]
    ball_shadow_rgb: Tuple[int, int, int]
    marker_rgb: Tuple[int, int, int]
    marker_fill_rgba: Tuple[int, int, int, int]
    badge_fill_rgb: Tuple[int, int, int]
    badge_outline_rgb: Tuple[int, int, int]
    badge_text_rgb: Tuple[int, int, int]
    shot_line_rgb: Tuple[int, int, int]
    ball_rendering: str = "glossy"


def build_games_pool_theme(*, style_variant: str) -> PoolTheme:
    """Return one resolved Pool-table theme for the active style variant."""

    variant = str(style_variant)
    if variant == "tournament_blue":
        return PoolTheme(
            rail_rgb=(34, 42, 54),
            rail_outline_rgb=(10, 16, 24),
            cloth_rgb=(38, 94, 132),
            cloth_line_rgb=(96, 156, 190),
            pocket_rgb=(5, 8, 12),
            pocket_outline_rgb=(194, 214, 226),
            ball_outline_rgb=(26, 30, 38),
            ball_shadow_rgb=(6, 10, 14),
            marker_rgb=(255, 214, 92),
            marker_fill_rgba=(255, 214, 92, 50),
            badge_fill_rgb=(238, 246, 250),
            badge_outline_rgb=(50, 78, 102),
            badge_text_rgb=(20, 32, 44),
            shot_line_rgb=(255, 224, 92),
        )
    if variant == "burgundy":
        return PoolTheme(
            rail_rgb=(78, 42, 36),
            rail_outline_rgb=(32, 18, 16),
            cloth_rgb=(92, 36, 52),
            cloth_line_rgb=(154, 94, 108),
            pocket_rgb=(12, 8, 10),
            pocket_outline_rgb=(218, 186, 160),
            ball_outline_rgb=(34, 24, 24),
            ball_shadow_rgb=(16, 6, 10),
            marker_rgb=(250, 206, 94),
            marker_fill_rgba=(250, 206, 94, 52),
            badge_fill_rgb=(252, 242, 232),
            badge_outline_rgb=(116, 72, 62),
            badge_text_rgb=(48, 28, 24),
            shot_line_rgb=(255, 220, 104),
        )
    if variant == "charcoal":
        return PoolTheme(
            rail_rgb=(32, 36, 42),
            rail_outline_rgb=(8, 10, 12),
            cloth_rgb=(48, 74, 68),
            cloth_line_rgb=(116, 144, 136),
            pocket_rgb=(4, 5, 6),
            pocket_outline_rgb=(176, 188, 184),
            ball_outline_rgb=(18, 20, 22),
            ball_shadow_rgb=(4, 5, 6),
            marker_rgb=(248, 214, 76),
            marker_fill_rgba=(248, 214, 76, 52),
            badge_fill_rgb=(236, 240, 238),
            badge_outline_rgb=(72, 84, 82),
            badge_text_rgb=(28, 34, 34),
            shot_line_rgb=(250, 216, 78),
        )
    if variant == "light_rail":
        return PoolTheme(
            rail_rgb=(156, 128, 88),
            rail_outline_rgb=(78, 58, 36),
            cloth_rgb=(54, 126, 94),
            cloth_line_rgb=(128, 190, 154),
            pocket_rgb=(10, 8, 6),
            pocket_outline_rgb=(80, 58, 34),
            ball_outline_rgb=(40, 34, 26),
            ball_shadow_rgb=(28, 18, 10),
            marker_rgb=(34, 54, 76),
            marker_fill_rgba=(34, 54, 76, 40),
            badge_fill_rgb=(255, 250, 238),
            badge_outline_rgb=(108, 82, 48),
            badge_text_rgb=(42, 32, 22),
            shot_line_rgb=(34, 54, 76),
        )
    return PoolTheme(
        rail_rgb=(66, 54, 38),
        rail_outline_rgb=(26, 18, 12),
        cloth_rgb=(40, 112, 76),
        cloth_line_rgb=(104, 174, 132),
        pocket_rgb=(6, 8, 8),
        pocket_outline_rgb=(210, 186, 124),
        ball_outline_rgb=(22, 24, 24),
        ball_shadow_rgb=(6, 10, 8),
        marker_rgb=(246, 206, 60),
        marker_fill_rgba=(246, 206, 60, 55),
        badge_fill_rgb=(248, 246, 236),
        badge_outline_rgb=(82, 70, 46),
        badge_text_rgb=(28, 34, 28),
        shot_line_rgb=(248, 214, 76),
    )
