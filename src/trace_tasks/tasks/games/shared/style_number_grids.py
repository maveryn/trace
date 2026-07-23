"""Number-grid style helpers for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


SUPPORTED_BINGO_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "soft",
    "outlined",
    "mint",
    "lavender",
    "amber",
    "slate",
)


SUPPORTED_MINESWEEPER_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "soft",
    "outlined",
    "notebook",
    "dark",
    "retro",
)


SUPPORTED_BATTLESHIP_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "soft",
    "outlined",
    "navy",
    "radar",
    "paper",
)


@dataclass(frozen=True)
class BingoTheme:
    """Resolved bingo-card palette for one style variant."""

    card_fill_rgb: Tuple[int, int, int]
    card_border_rgb: Tuple[int, int, int]
    card_border_width_px: int
    shadow_rgb: Tuple[int, int, int]
    shadow_alpha: int
    shadow_offset_px: Tuple[int, int]
    title_rgb: Tuple[int, int, int]
    header_rgb: Tuple[int, int, int]
    grid_line_rgb: Tuple[int, int, int]
    cell_fill_rgb: Tuple[int, int, int]
    cell_alt_fill_rgb: Tuple[int, int, int]
    number_rgb: Tuple[int, int, int]
    mark_fill_rgba: Tuple[int, int, int, int]
    mark_outline_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class MinesweeperTheme:
    """Resolved Minesweeper-grid palette for one style variant."""

    board_fill_rgb: Tuple[int, int, int]
    board_border_rgb: Tuple[int, int, int]
    grid_line_rgb: Tuple[int, int, int]
    hidden_cell_fill_rgb: Tuple[int, int, int]
    hidden_cell_border_rgb: Tuple[int, int, int]
    revealed_cell_fill_rgb: Tuple[int, int, int]
    revealed_cell_alt_fill_rgb: Tuple[int, int, int]
    number_rgb_by_value: Tuple[Tuple[int, int, int], ...]
    flag_rgb: Tuple[int, int, int]
    flag_pole_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class BattleshipTheme:
    """Resolved Battleship tracking-grid palette for one style variant."""

    board_fill_rgb: Tuple[int, int, int]
    board_border_rgb: Tuple[int, int, int]
    grid_line_rgb: Tuple[int, int, int]
    cell_fill_rgb: Tuple[int, int, int]
    cell_alt_fill_rgb: Tuple[int, int, int]
    hit_fill_rgb: Tuple[int, int, int]
    hit_outline_rgb: Tuple[int, int, int]
    miss_rgb: Tuple[int, int, int]
    panel_fill_rgb: Tuple[int, int, int]
    panel_border_rgb: Tuple[int, int, int]
    panel_text_rgb: Tuple[int, int, int]
    ship_icon_fill_rgb: Tuple[int, int, int]
    ship_icon_outline_rgb: Tuple[int, int, int]
    hit_marker_style: str = "disc"
    miss_marker_style: str = "ring"


def build_games_bingo_theme(*, style_variant: str) -> BingoTheme:
    """Return one resolved bingo-card theme for the active style variant."""

    variant = str(style_variant)
    if variant == "soft":
        return BingoTheme(
            card_fill_rgb=(251, 246, 236),
            card_border_rgb=(102, 112, 124),
            card_border_width_px=3,
            shadow_rgb=(18, 24, 20),
            shadow_alpha=56,
            shadow_offset_px=(5, 6),
            title_rgb=(40, 63, 121),
            header_rgb=(56, 84, 146),
            grid_line_rgb=(132, 142, 152),
            cell_fill_rgb=(255, 252, 247),
            cell_alt_fill_rgb=(248, 241, 231),
            number_rgb=(41, 46, 54),
            mark_fill_rgba=(214, 84, 76, 136),
            mark_outline_rgb=(184, 58, 54),
        )
    if variant == "outlined":
        return BingoTheme(
            card_fill_rgb=(255, 255, 255),
            card_border_rgb=(62, 70, 80),
            card_border_width_px=4,
            shadow_rgb=(14, 18, 20),
            shadow_alpha=40,
            shadow_offset_px=(4, 5),
            title_rgb=(44, 75, 168),
            header_rgb=(51, 88, 186),
            grid_line_rgb=(92, 100, 110),
            cell_fill_rgb=(255, 255, 255),
            cell_alt_fill_rgb=(242, 246, 255),
            number_rgb=(33, 38, 44),
            mark_fill_rgba=(218, 66, 60, 126),
            mark_outline_rgb=(196, 48, 44),
        )
    if variant == "mint":
        return BingoTheme(
            card_fill_rgb=(239, 250, 242),
            card_border_rgb=(63, 118, 95),
            card_border_width_px=3,
            shadow_rgb=(12, 30, 23),
            shadow_alpha=46,
            shadow_offset_px=(5, 6),
            title_rgb=(29, 94, 76),
            header_rgb=(35, 104, 86),
            grid_line_rgb=(103, 151, 132),
            cell_fill_rgb=(252, 255, 250),
            cell_alt_fill_rgb=(229, 246, 238),
            number_rgb=(28, 47, 43),
            mark_fill_rgba=(56, 122, 210, 130),
            mark_outline_rgb=(42, 91, 170),
        )
    if variant == "lavender":
        return BingoTheme(
            card_fill_rgb=(246, 243, 255),
            card_border_rgb=(102, 91, 151),
            card_border_width_px=3,
            shadow_rgb=(24, 21, 42),
            shadow_alpha=44,
            shadow_offset_px=(5, 5),
            title_rgb=(91, 69, 153),
            header_rgb=(105, 75, 166),
            grid_line_rgb=(145, 134, 182),
            cell_fill_rgb=(255, 253, 255),
            cell_alt_fill_rgb=(238, 235, 252),
            number_rgb=(38, 35, 50),
            mark_fill_rgba=(218, 141, 45, 138),
            mark_outline_rgb=(180, 112, 28),
        )
    if variant == "amber":
        return BingoTheme(
            card_fill_rgb=(255, 248, 231),
            card_border_rgb=(129, 91, 41),
            card_border_width_px=3,
            shadow_rgb=(36, 25, 13),
            shadow_alpha=48,
            shadow_offset_px=(5, 6),
            title_rgb=(127, 78, 25),
            header_rgb=(145, 85, 28),
            grid_line_rgb=(173, 135, 82),
            cell_fill_rgb=(255, 254, 248),
            cell_alt_fill_rgb=(250, 238, 212),
            number_rgb=(44, 36, 26),
            mark_fill_rgba=(48, 128, 116, 132),
            mark_outline_rgb=(28, 101, 93),
        )
    if variant == "slate":
        return BingoTheme(
            card_fill_rgb=(241, 245, 249),
            card_border_rgb=(67, 78, 93),
            card_border_width_px=4,
            shadow_rgb=(8, 12, 18),
            shadow_alpha=50,
            shadow_offset_px=(4, 6),
            title_rgb=(31, 71, 118),
            header_rgb=(39, 82, 129),
            grid_line_rgb=(111, 123, 138),
            cell_fill_rgb=(255, 255, 255),
            cell_alt_fill_rgb=(231, 238, 246),
            number_rgb=(24, 31, 40),
            mark_fill_rgba=(196, 64, 102, 132),
            mark_outline_rgb=(166, 43, 78),
        )
    return BingoTheme(
        card_fill_rgb=(255, 253, 247),
        card_border_rgb=(74, 82, 92),
        card_border_width_px=3,
        shadow_rgb=(18, 22, 24),
        shadow_alpha=48,
        shadow_offset_px=(4, 5),
        title_rgb=(42, 72, 160),
        header_rgb=(48, 82, 180),
        grid_line_rgb=(108, 116, 126),
        cell_fill_rgb=(255, 255, 252),
        cell_alt_fill_rgb=(242, 247, 255),
        number_rgb=(29, 34, 40),
        mark_fill_rgba=(212, 62, 56, 132),
        mark_outline_rgb=(190, 46, 42),
    )


def build_games_minesweeper_theme(*, style_variant: str) -> MinesweeperTheme:
    """Return one resolved Minesweeper-grid theme for the active style variant."""

    number_colors: Tuple[Tuple[int, int, int], ...] = (
        (92, 96, 104),
        (42, 96, 196),
        (38, 132, 72),
        (195, 54, 62),
        (100, 74, 170),
        (178, 92, 36),
        (38, 136, 152),
        (72, 78, 88),
        (36, 40, 46),
    )
    dark_number_colors: Tuple[Tuple[int, int, int], ...] = (
        (176, 184, 194),
        (92, 156, 255),
        (118, 210, 142),
        (255, 120, 126),
        (190, 150, 255),
        (244, 174, 96),
        (92, 210, 224),
        (210, 218, 228),
        (246, 248, 250),
    )
    variant = str(style_variant)
    if variant == "notebook":
        return MinesweeperTheme(
            board_fill_rgb=(246, 250, 255),
            board_border_rgb=(54, 82, 118),
            grid_line_rgb=(174, 198, 222),
            hidden_cell_fill_rgb=(210, 226, 242),
            hidden_cell_border_rgb=(92, 126, 162),
            revealed_cell_fill_rgb=(252, 254, 255),
            revealed_cell_alt_fill_rgb=(244, 249, 255),
            number_rgb_by_value=number_colors,
            flag_rgb=(204, 50, 64),
            flag_pole_rgb=(42, 64, 92),
        )
    if variant == "dark":
        return MinesweeperTheme(
            board_fill_rgb=(36, 44, 54),
            board_border_rgb=(14, 18, 24),
            grid_line_rgb=(82, 94, 108),
            hidden_cell_fill_rgb=(72, 86, 102),
            hidden_cell_border_rgb=(142, 154, 168),
            revealed_cell_fill_rgb=(48, 58, 70),
            revealed_cell_alt_fill_rgb=(54, 66, 78),
            number_rgb_by_value=dark_number_colors,
            flag_rgb=(244, 82, 92),
            flag_pole_rgb=(226, 232, 238),
        )
    if variant == "retro":
        return MinesweeperTheme(
            board_fill_rgb=(184, 188, 192),
            board_border_rgb=(70, 74, 78),
            grid_line_rgb=(122, 126, 130),
            hidden_cell_fill_rgb=(198, 202, 206),
            hidden_cell_border_rgb=(86, 90, 94),
            revealed_cell_fill_rgb=(230, 230, 226),
            revealed_cell_alt_fill_rgb=(222, 222, 218),
            number_rgb_by_value=number_colors,
            flag_rgb=(206, 30, 42),
            flag_pole_rgb=(34, 36, 38),
        )
    if variant == "soft":
        return MinesweeperTheme(
            board_fill_rgb=(240, 237, 228),
            board_border_rgb=(104, 110, 120),
            grid_line_rgb=(178, 182, 190),
            hidden_cell_fill_rgb=(202, 210, 216),
            hidden_cell_border_rgb=(128, 136, 146),
            revealed_cell_fill_rgb=(248, 246, 239),
            revealed_cell_alt_fill_rgb=(241, 239, 232),
            number_rgb_by_value=number_colors,
            flag_rgb=(210, 54, 64),
            flag_pole_rgb=(66, 72, 82),
        )
    if variant == "outlined":
        return MinesweeperTheme(
            board_fill_rgb=(252, 253, 255),
            board_border_rgb=(54, 62, 74),
            grid_line_rgb=(142, 150, 160),
            hidden_cell_fill_rgb=(212, 218, 225),
            hidden_cell_border_rgb=(82, 92, 106),
            revealed_cell_fill_rgb=(255, 255, 255),
            revealed_cell_alt_fill_rgb=(248, 250, 252),
            number_rgb_by_value=number_colors,
            flag_rgb=(202, 42, 58),
            flag_pole_rgb=(42, 50, 60),
        )
    return MinesweeperTheme(
        board_fill_rgb=(235, 232, 224),
        board_border_rgb=(72, 78, 88),
        grid_line_rgb=(152, 158, 166),
        hidden_cell_fill_rgb=(190, 198, 206),
        hidden_cell_border_rgb=(96, 104, 114),
        revealed_cell_fill_rgb=(246, 244, 238),
        revealed_cell_alt_fill_rgb=(238, 236, 230),
        number_rgb_by_value=number_colors,
        flag_rgb=(204, 48, 58),
        flag_pole_rgb=(54, 60, 70),
    )


def build_games_battleship_theme(*, style_variant: str) -> BattleshipTheme:
    """Return one resolved Battleship tracking-grid theme for the active style variant."""

    variant = str(style_variant)
    if variant == "soft":
        return BattleshipTheme(
            board_fill_rgb=(226, 238, 244),
            board_border_rgb=(68, 92, 108),
            grid_line_rgb=(150, 174, 190),
            cell_fill_rgb=(238, 248, 252),
            cell_alt_fill_rgb=(230, 242, 248),
            hit_fill_rgb=(218, 62, 66),
            hit_outline_rgb=(128, 32, 38),
            miss_rgb=(78, 122, 148),
            panel_fill_rgb=(248, 250, 246),
            panel_border_rgb=(108, 122, 132),
            panel_text_rgb=(34, 44, 54),
            ship_icon_fill_rgb=(104, 122, 136),
            ship_icon_outline_rgb=(42, 52, 64),
            hit_marker_style="disc",
            miss_marker_style="ring",
        )
    if variant == "outlined":
        return BattleshipTheme(
            board_fill_rgb=(255, 255, 255),
            board_border_rgb=(42, 52, 64),
            grid_line_rgb=(120, 132, 146),
            cell_fill_rgb=(253, 254, 255),
            cell_alt_fill_rgb=(246, 248, 251),
            hit_fill_rgb=(210, 44, 58),
            hit_outline_rgb=(96, 22, 32),
            miss_rgb=(54, 94, 126),
            panel_fill_rgb=(255, 255, 255),
            panel_border_rgb=(52, 62, 76),
            panel_text_rgb=(24, 30, 38),
            ship_icon_fill_rgb=(96, 104, 116),
            ship_icon_outline_rgb=(28, 34, 44),
            hit_marker_style="cross",
            miss_marker_style="cross",
        )
    if variant == "navy":
        return BattleshipTheme(
            board_fill_rgb=(20, 42, 66),
            board_border_rgb=(8, 16, 28),
            grid_line_rgb=(70, 108, 140),
            cell_fill_rgb=(31, 62, 92),
            cell_alt_fill_rgb=(26, 54, 82),
            hit_fill_rgb=(246, 78, 78),
            hit_outline_rgb=(255, 198, 198),
            miss_rgb=(160, 210, 232),
            panel_fill_rgb=(24, 38, 54),
            panel_border_rgb=(118, 154, 178),
            panel_text_rgb=(230, 238, 244),
            ship_icon_fill_rgb=(150, 166, 178),
            ship_icon_outline_rgb=(230, 238, 244),
            hit_marker_style="disc",
            miss_marker_style="ring",
        )
    if variant == "radar":
        return BattleshipTheme(
            board_fill_rgb=(24, 58, 46),
            board_border_rgb=(10, 30, 24),
            grid_line_rgb=(80, 150, 112),
            cell_fill_rgb=(34, 76, 58),
            cell_alt_fill_rgb=(30, 68, 52),
            hit_fill_rgb=(238, 74, 58),
            hit_outline_rgb=(122, 26, 24),
            miss_rgb=(176, 224, 190),
            panel_fill_rgb=(232, 244, 232),
            panel_border_rgb=(62, 104, 78),
            panel_text_rgb=(24, 52, 38),
            ship_icon_fill_rgb=(86, 120, 92),
            ship_icon_outline_rgb=(24, 52, 38),
            hit_marker_style="square",
            miss_marker_style="ring",
        )
    if variant == "paper":
        return BattleshipTheme(
            board_fill_rgb=(244, 239, 224),
            board_border_rgb=(90, 78, 62),
            grid_line_rgb=(176, 158, 132),
            cell_fill_rgb=(255, 252, 240),
            cell_alt_fill_rgb=(248, 243, 228),
            hit_fill_rgb=(198, 52, 50),
            hit_outline_rgb=(116, 34, 30),
            miss_rgb=(78, 106, 138),
            panel_fill_rgb=(252, 248, 236),
            panel_border_rgb=(112, 96, 74),
            panel_text_rgb=(48, 42, 34),
            ship_icon_fill_rgb=(122, 112, 98),
            ship_icon_outline_rgb=(58, 50, 42),
            hit_marker_style="cross",
            miss_marker_style="dot",
        )
    return BattleshipTheme(
        board_fill_rgb=(212, 230, 238),
        board_border_rgb=(42, 72, 94),
        grid_line_rgb=(126, 160, 182),
        cell_fill_rgb=(232, 246, 252),
        cell_alt_fill_rgb=(222, 238, 246),
        hit_fill_rgb=(222, 58, 64),
        hit_outline_rgb=(110, 26, 34),
        miss_rgb=(58, 108, 142),
        panel_fill_rgb=(246, 248, 244),
        panel_border_rgb=(86, 104, 116),
        panel_text_rgb=(30, 38, 48),
        ship_icon_fill_rgb=(98, 116, 130),
        ship_icon_outline_rgb=(36, 46, 58),
        hit_marker_style="disc",
        miss_marker_style="ring",
    )
