"""Node and hex-board style helpers for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


SUPPORTED_GO_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "soft",
    "outlined",
    "wood_board",
    "slate_board",
    "paper_board",
)


SUPPORTED_HEX_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "soft",
    "outlined",
    "slate",
    "paper",
)


SUPPORTED_NINE_MENS_MORRIS_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "soft",
    "outlined",
    "wood_panel",
    "slate",
    "parchment",
)


@dataclass(frozen=True)
class NineMensMorrisTheme:
    """Resolved nine-men's-morris palette for one style variant."""

    board_fill_rgb: Tuple[int, int, int]
    board_border_rgb: Tuple[int, int, int]
    board_border_width_px: int
    shadow_rgb: Tuple[int, int, int]
    shadow_alpha: int
    shadow_offset_px: Tuple[int, int]
    title_rgb: Tuple[int, int, int]
    line_rgb: Tuple[int, int, int]
    line_width_px: int
    node_rgb: Tuple[int, int, int]
    white_piece_fill_rgb: Tuple[int, int, int]
    white_piece_outline_rgb: Tuple[int, int, int]
    black_piece_fill_rgb: Tuple[int, int, int]
    black_piece_outline_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class GoTheme:
    """Resolved Go-board palette for one style variant."""

    board_frame_rgb: Tuple[int, int, int]
    board_fill_rgb: Tuple[int, int, int]
    grid_line_rgb: Tuple[int, int, int]
    point_rgb: Tuple[int, int, int]
    black_stone_fill_rgb: Tuple[int, int, int]
    black_stone_outline_rgb: Tuple[int, int, int]
    black_stone_shine_rgb: Tuple[int, int, int]
    white_stone_fill_rgb: Tuple[int, int, int]
    white_stone_outline_rgb: Tuple[int, int, int]
    white_stone_shine_rgb: Tuple[int, int, int]
    stone_outline_width_px: int
    highlight_outline_rgb: Tuple[int, int, int]
    highlight_fill_rgba: Tuple[int, int, int, int]


@dataclass(frozen=True)
class HexTheme:
    """Resolved Hex-board palette for one style variant."""

    cell_fill_rgb: Tuple[int, int, int]
    cell_alt_fill_rgb: Tuple[int, int, int]
    cell_outline_rgb: Tuple[int, int, int]
    board_outline_rgb: Tuple[int, int, int]
    red_goal_rgb: Tuple[int, int, int]
    blue_goal_rgb: Tuple[int, int, int]
    red_stone_fill_rgb: Tuple[int, int, int]
    red_stone_outline_rgb: Tuple[int, int, int]
    red_stone_shine_rgb: Tuple[int, int, int]
    blue_stone_fill_rgb: Tuple[int, int, int]
    blue_stone_outline_rgb: Tuple[int, int, int]
    blue_stone_shine_rgb: Tuple[int, int, int]
    candidate_badge_fill_rgb: Tuple[int, int, int]
    candidate_badge_outline_rgb: Tuple[int, int, int]
    candidate_badge_text_rgb: Tuple[int, int, int]
    reference_cell_fill_rgb: Tuple[int, int, int]
    reference_cell_outline_rgb: Tuple[int, int, int]


def build_games_nine_mens_morris_theme(*, style_variant: str) -> NineMensMorrisTheme:
    """Return one resolved nine-men's-morris theme for the active style variant."""

    variant = str(style_variant)
    if variant == "wood_panel":
        return NineMensMorrisTheme(
            board_fill_rgb=(222, 184, 128),
            board_border_rgb=(100, 66, 38),
            board_border_width_px=4,
            shadow_rgb=(18, 16, 12),
            shadow_alpha=58,
            shadow_offset_px=(6, 7),
            title_rgb=(78, 48, 28),
            line_rgb=(86, 56, 34),
            line_width_px=7,
            node_rgb=(70, 44, 28),
            white_piece_fill_rgb=(252, 246, 232),
            white_piece_outline_rgb=(118, 94, 70),
            black_piece_fill_rgb=(48, 36, 30),
            black_piece_outline_rgb=(18, 14, 12),
        )
    if variant == "slate":
        return NineMensMorrisTheme(
            board_fill_rgb=(54, 66, 78),
            board_border_rgb=(24, 30, 38),
            board_border_width_px=4,
            shadow_rgb=(8, 10, 14),
            shadow_alpha=72,
            shadow_offset_px=(5, 6),
            title_rgb=(226, 232, 238),
            line_rgb=(198, 206, 214),
            line_width_px=6,
            node_rgb=(222, 228, 234),
            white_piece_fill_rgb=(246, 248, 250),
            white_piece_outline_rgb=(132, 144, 156),
            black_piece_fill_rgb=(22, 28, 36),
            black_piece_outline_rgb=(6, 8, 12),
        )
    if variant == "parchment":
        return NineMensMorrisTheme(
            board_fill_rgb=(248, 236, 204),
            board_border_rgb=(124, 88, 52),
            board_border_width_px=4,
            shadow_rgb=(38, 28, 18),
            shadow_alpha=44,
            shadow_offset_px=(4, 5),
            title_rgb=(98, 66, 38),
            line_rgb=(110, 78, 48),
            line_width_px=6,
            node_rgb=(92, 62, 40),
            white_piece_fill_rgb=(255, 252, 242),
            white_piece_outline_rgb=(128, 112, 90),
            black_piece_fill_rgb=(54, 48, 42),
            black_piece_outline_rgb=(22, 18, 16),
        )
    if variant == "soft":
        return NineMensMorrisTheme(
            board_fill_rgb=(248, 242, 230),
            board_border_rgb=(102, 112, 124),
            board_border_width_px=3,
            shadow_rgb=(18, 24, 20),
            shadow_alpha=56,
            shadow_offset_px=(5, 6),
            title_rgb=(48, 76, 134),
            line_rgb=(86, 94, 106),
            line_width_px=6,
            node_rgb=(72, 80, 92),
            white_piece_fill_rgb=(252, 250, 244),
            white_piece_outline_rgb=(112, 118, 126),
            black_piece_fill_rgb=(58, 64, 72),
            black_piece_outline_rgb=(24, 28, 34),
        )
    if variant == "outlined":
        return NineMensMorrisTheme(
            board_fill_rgb=(255, 255, 255),
            board_border_rgb=(62, 70, 80),
            board_border_width_px=4,
            shadow_rgb=(14, 18, 20),
            shadow_alpha=40,
            shadow_offset_px=(4, 5),
            title_rgb=(44, 75, 168),
            line_rgb=(58, 64, 72),
            line_width_px=6,
            node_rgb=(46, 52, 60),
            white_piece_fill_rgb=(255, 255, 255),
            white_piece_outline_rgb=(112, 118, 126),
            black_piece_fill_rgb=(42, 48, 56),
            black_piece_outline_rgb=(18, 22, 28),
        )
    return NineMensMorrisTheme(
        board_fill_rgb=(255, 252, 244),
        board_border_rgb=(74, 82, 92),
        board_border_width_px=3,
        shadow_rgb=(18, 22, 24),
        shadow_alpha=48,
        shadow_offset_px=(4, 5),
        title_rgb=(45, 76, 160),
        line_rgb=(66, 72, 82),
        line_width_px=6,
        node_rgb=(44, 50, 58),
        white_piece_fill_rgb=(254, 252, 248),
        white_piece_outline_rgb=(106, 112, 120),
        black_piece_fill_rgb=(36, 42, 50),
        black_piece_outline_rgb=(18, 22, 28),
    )


def build_games_go_theme(*, style_variant: str) -> GoTheme:
    """Return one resolved Go-board theme for the active style variant."""

    variant = str(style_variant)
    if variant == "wood_board":
        return GoTheme(
            board_frame_rgb=(126, 78, 38),
            board_fill_rgb=(222, 172, 94),
            grid_line_rgb=(92, 54, 24),
            point_rgb=(86, 50, 22),
            black_stone_fill_rgb=(34, 32, 30),
            black_stone_outline_rgb=(12, 12, 12),
            black_stone_shine_rgb=(88, 82, 74),
            white_stone_fill_rgb=(252, 246, 232),
            white_stone_outline_rgb=(124, 112, 96),
            white_stone_shine_rgb=(255, 255, 255),
            stone_outline_width_px=2,
            highlight_outline_rgb=(42, 116, 218),
            highlight_fill_rgba=(82, 142, 236, 52),
        )
    if variant == "slate_board":
        return GoTheme(
            board_frame_rgb=(42, 50, 60),
            board_fill_rgb=(96, 114, 126),
            grid_line_rgb=(34, 42, 52),
            point_rgb=(30, 38, 48),
            black_stone_fill_rgb=(18, 22, 28),
            black_stone_outline_rgb=(6, 8, 12),
            black_stone_shine_rgb=(78, 88, 98),
            white_stone_fill_rgb=(240, 244, 246),
            white_stone_outline_rgb=(118, 128, 138),
            white_stone_shine_rgb=(255, 255, 255),
            stone_outline_width_px=2,
            highlight_outline_rgb=(78, 158, 244),
            highlight_fill_rgba=(86, 158, 244, 58),
        )
    if variant == "paper_board":
        return GoTheme(
            board_frame_rgb=(102, 110, 118),
            board_fill_rgb=(251, 247, 236),
            grid_line_rgb=(76, 82, 88),
            point_rgb=(68, 74, 82),
            black_stone_fill_rgb=(42, 46, 54),
            black_stone_outline_rgb=(16, 20, 26),
            black_stone_shine_rgb=(108, 116, 126),
            white_stone_fill_rgb=(255, 255, 255),
            white_stone_outline_rgb=(126, 134, 142),
            white_stone_shine_rgb=(255, 255, 255),
            stone_outline_width_px=3,
            highlight_outline_rgb=(52, 116, 222),
            highlight_fill_rgba=(86, 145, 235, 46),
        )
    if variant == "soft":
        return GoTheme(
            board_frame_rgb=(120, 89, 56),
            board_fill_rgb=(214, 181, 126),
            grid_line_rgb=(88, 64, 38),
            point_rgb=(80, 58, 34),
            black_stone_fill_rgb=(50, 56, 64),
            black_stone_outline_rgb=(20, 24, 30),
            black_stone_shine_rgb=(107, 114, 122),
            white_stone_fill_rgb=(247, 245, 238),
            white_stone_outline_rgb=(128, 132, 140),
            white_stone_shine_rgb=(255, 255, 255),
            stone_outline_width_px=2,
            highlight_outline_rgb=(67, 132, 223),
            highlight_fill_rgba=(93, 156, 240, 52),
        )
    if variant == "outlined":
        return GoTheme(
            board_frame_rgb=(82, 88, 98),
            board_fill_rgb=(255, 250, 241),
            grid_line_rgb=(78, 82, 90),
            point_rgb=(72, 78, 88),
            black_stone_fill_rgb=(48, 54, 62),
            black_stone_outline_rgb=(18, 22, 28),
            black_stone_shine_rgb=(110, 116, 126),
            white_stone_fill_rgb=(255, 255, 255),
            white_stone_outline_rgb=(130, 136, 144),
            white_stone_shine_rgb=(255, 255, 255),
            stone_outline_width_px=3,
            highlight_outline_rgb=(52, 116, 222),
            highlight_fill_rgba=(86, 145, 235, 44),
        )
    return GoTheme(
        board_frame_rgb=(118, 86, 48),
        board_fill_rgb=(225, 190, 128),
        grid_line_rgb=(86, 58, 30),
        point_rgb=(80, 54, 28),
        black_stone_fill_rgb=(42, 48, 56),
        black_stone_outline_rgb=(18, 22, 28),
        black_stone_shine_rgb=(102, 108, 118),
        white_stone_fill_rgb=(252, 250, 244),
        white_stone_outline_rgb=(124, 128, 136),
        white_stone_shine_rgb=(255, 255, 255),
        stone_outline_width_px=2,
        highlight_outline_rgb=(46, 113, 228),
        highlight_fill_rgba=(82, 142, 236, 46),
    )


def build_games_hex_theme(*, style_variant: str) -> HexTheme:
    """Return one resolved Hex-board theme for the active style variant."""

    variant = str(style_variant)
    if variant == "soft":
        return HexTheme(
            cell_fill_rgb=(238, 232, 212),
            cell_alt_fill_rgb=(229, 222, 202),
            cell_outline_rgb=(119, 104, 82),
            board_outline_rgb=(81, 66, 46),
            red_goal_rgb=(204, 72, 65),
            blue_goal_rgb=(58, 112, 204),
            red_stone_fill_rgb=(208, 66, 62),
            red_stone_outline_rgb=(108, 35, 35),
            red_stone_shine_rgb=(244, 150, 140),
            blue_stone_fill_rgb=(53, 110, 200),
            blue_stone_outline_rgb=(28, 58, 112),
            blue_stone_shine_rgb=(142, 190, 244),
            candidate_badge_fill_rgb=(255, 250, 236),
            candidate_badge_outline_rgb=(89, 78, 62),
            candidate_badge_text_rgb=(38, 34, 28),
            reference_cell_fill_rgb=(105, 190, 116),
            reference_cell_outline_rgb=(24, 104, 48),
        )
    if variant == "outlined":
        return HexTheme(
            cell_fill_rgb=(250, 250, 248),
            cell_alt_fill_rgb=(241, 242, 238),
            cell_outline_rgb=(64, 70, 78),
            board_outline_rgb=(34, 38, 46),
            red_goal_rgb=(190, 52, 56),
            blue_goal_rgb=(43, 94, 178),
            red_stone_fill_rgb=(224, 62, 64),
            red_stone_outline_rgb=(74, 26, 28),
            red_stone_shine_rgb=(255, 154, 150),
            blue_stone_fill_rgb=(48, 104, 200),
            blue_stone_outline_rgb=(20, 42, 86),
            blue_stone_shine_rgb=(144, 190, 250),
            candidate_badge_fill_rgb=(255, 255, 255),
            candidate_badge_outline_rgb=(40, 46, 54),
            candidate_badge_text_rgb=(18, 24, 30),
            reference_cell_fill_rgb=(92, 198, 118),
            reference_cell_outline_rgb=(18, 112, 52),
        )
    if variant == "slate":
        return HexTheme(
            cell_fill_rgb=(74, 88, 92),
            cell_alt_fill_rgb=(66, 78, 84),
            cell_outline_rgb=(174, 188, 190),
            board_outline_rgb=(226, 232, 230),
            red_goal_rgb=(236, 92, 82),
            blue_goal_rgb=(94, 164, 236),
            red_stone_fill_rgb=(214, 68, 62),
            red_stone_outline_rgb=(80, 24, 22),
            red_stone_shine_rgb=(252, 148, 140),
            blue_stone_fill_rgb=(58, 126, 218),
            blue_stone_outline_rgb=(18, 48, 94),
            blue_stone_shine_rgb=(142, 204, 255),
            candidate_badge_fill_rgb=(244, 248, 246),
            candidate_badge_outline_rgb=(28, 36, 40),
            candidate_badge_text_rgb=(16, 20, 22),
            reference_cell_fill_rgb=(92, 208, 132),
            reference_cell_outline_rgb=(210, 255, 226),
        )
    if variant == "paper":
        return HexTheme(
            cell_fill_rgb=(250, 241, 216),
            cell_alt_fill_rgb=(242, 230, 200),
            cell_outline_rgb=(136, 104, 68),
            board_outline_rgb=(82, 58, 34),
            red_goal_rgb=(184, 66, 54),
            blue_goal_rgb=(52, 104, 166),
            red_stone_fill_rgb=(202, 76, 62),
            red_stone_outline_rgb=(92, 40, 32),
            red_stone_shine_rgb=(242, 154, 132),
            blue_stone_fill_rgb=(56, 112, 184),
            blue_stone_outline_rgb=(30, 60, 96),
            blue_stone_shine_rgb=(142, 194, 238),
            candidate_badge_fill_rgb=(255, 249, 226),
            candidate_badge_outline_rgb=(112, 82, 46),
            candidate_badge_text_rgb=(44, 30, 18),
            reference_cell_fill_rgb=(112, 188, 104),
            reference_cell_outline_rgb=(30, 104, 42),
        )
    return HexTheme(
        cell_fill_rgb=(232, 210, 164),
        cell_alt_fill_rgb=(220, 198, 150),
        cell_outline_rgb=(104, 78, 46),
        board_outline_rgb=(66, 48, 28),
        red_goal_rgb=(194, 50, 54),
        blue_goal_rgb=(44, 92, 178),
        red_stone_fill_rgb=(214, 54, 58),
        red_stone_outline_rgb=(82, 24, 28),
        red_stone_shine_rgb=(248, 138, 136),
        blue_stone_fill_rgb=(48, 100, 196),
        blue_stone_outline_rgb=(22, 44, 92),
        blue_stone_shine_rgb=(136, 184, 248),
        candidate_badge_fill_rgb=(255, 248, 224),
        candidate_badge_outline_rgb=(78, 58, 34),
        candidate_badge_text_rgb=(30, 24, 18),
        reference_cell_fill_rgb=(105, 186, 94),
        reference_cell_outline_rgb=(22, 92, 38),
    )
