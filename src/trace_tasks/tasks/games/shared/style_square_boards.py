"""Square-board style helpers for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


SUPPORTED_CHESS_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "soft",
    "outlined",
    "wood_token",
    "blue_glyph",
    "monochrome_glyph",
)


SUPPORTED_CHECKERS_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "soft",
    "outlined",
    "wood_token",
    "blue_table",
    "charcoal",
)


SUPPORTED_CONNECT_FOUR_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "soft",
    "outlined",
    "arcade_blue",
    "teal_frame",
    "charcoal",
)


SUPPORTED_DOTS_AND_BOXES_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "soft",
    "outlined",
    "notebook",
    "slate",
    "wood_panel",
)


SUPPORTED_REVERSI_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "soft",
    "outlined",
    "wood_board",
    "slate_board",
    "blue_board",
)


@dataclass(frozen=True)
class ReversiTheme:
    """Resolved Reversi-board palette for one style variant."""

    board_frame_rgb: Tuple[int, int, int]
    board_fill_rgb: Tuple[int, int, int]
    grid_line_rgb: Tuple[int, int, int]
    badge_fill_rgb: Tuple[int, int, int]
    badge_outline_rgb: Tuple[int, int, int]
    badge_text_rgb: Tuple[int, int, int]
    black_disc_fill_rgb: Tuple[int, int, int]
    black_disc_outline_rgb: Tuple[int, int, int]
    black_disc_shine_rgb: Tuple[int, int, int]
    white_disc_fill_rgb: Tuple[int, int, int]
    white_disc_outline_rgb: Tuple[int, int, int]
    white_disc_shine_rgb: Tuple[int, int, int]
    disc_outline_width_px: int
    marked_square_outline_rgb: Tuple[int, int, int]
    marked_square_fill_rgba: Tuple[int, int, int, int]


@dataclass(frozen=True)
class ConnectFourTheme:
    """Resolved Connect Four board palette for one style variant."""

    board_frame_rgb: Tuple[int, int, int]
    board_fill_rgb: Tuple[int, int, int]
    cell_well_rgb: Tuple[int, int, int]
    cell_well_outline_rgb: Tuple[int, int, int]
    cell_well_outline_width_px: int
    badge_fill_rgb: Tuple[int, int, int]
    badge_outline_rgb: Tuple[int, int, int]
    badge_text_rgb: Tuple[int, int, int]
    red_disc_fill_rgb: Tuple[int, int, int]
    red_disc_outline_rgb: Tuple[int, int, int]
    red_disc_shine_rgb: Tuple[int, int, int]
    yellow_disc_fill_rgb: Tuple[int, int, int]
    yellow_disc_outline_rgb: Tuple[int, int, int]
    yellow_disc_shine_rgb: Tuple[int, int, int]
    disc_outline_width_px: int
    marked_square_outline_rgb: Tuple[int, int, int]
    marked_square_fill_rgba: Tuple[int, int, int, int]
    board_shadow_rgb: Tuple[int, int, int] = (12, 16, 24)
    board_shadow_alpha: int = 0
    board_shadow_offset_px: Tuple[int, int] = (0, 0)
    board_rendering: str = "flat"
    cell_well_rendering: str = "flat"
    disc_rendering: str = "glossy"


@dataclass(frozen=True)
class CheckersTheme:
    """Resolved Checkers board palette for one style variant."""

    board_frame_rgb: Tuple[int, int, int]
    light_square_rgb: Tuple[int, int, int]
    dark_square_rgb: Tuple[int, int, int]
    grid_line_rgb: Tuple[int, int, int]
    grid_line_width_px: int
    badge_fill_rgb: Tuple[int, int, int]
    badge_outline_rgb: Tuple[int, int, int]
    badge_text_rgb: Tuple[int, int, int]
    red_piece_fill_rgb: Tuple[int, int, int]
    red_piece_outline_rgb: Tuple[int, int, int]
    red_piece_shine_rgb: Tuple[int, int, int]
    black_piece_fill_rgb: Tuple[int, int, int]
    black_piece_outline_rgb: Tuple[int, int, int]
    black_piece_shine_rgb: Tuple[int, int, int]
    piece_outline_width_px: int
    piece_shadow_rgb: Tuple[int, int, int] = (12, 14, 16)
    piece_shadow_alpha: int = 0
    piece_rendering: str = "ring"
    square_rendering: str = "flat"


@dataclass(frozen=True)
class ChessTheme:
    """Resolved Chess board palette for one style variant."""

    board_frame_rgb: Tuple[int, int, int]
    light_square_rgb: Tuple[int, int, int]
    dark_square_rgb: Tuple[int, int, int]
    grid_line_rgb: Tuple[int, int, int]
    grid_line_width_px: int
    badge_fill_rgb: Tuple[int, int, int]
    badge_outline_rgb: Tuple[int, int, int]
    badge_text_rgb: Tuple[int, int, int]
    white_piece_fill_rgb: Tuple[int, int, int]
    white_piece_outline_rgb: Tuple[int, int, int]
    black_piece_fill_rgb: Tuple[int, int, int]
    black_piece_outline_rgb: Tuple[int, int, int]
    marked_square_outline_rgb: Tuple[int, int, int]
    marked_square_fill_rgba: Tuple[int, int, int, int]
    piece_shadow_rgb: Tuple[int, int, int]
    piece_shadow_alpha: int
    piece_rendering: str = "token"
    square_rendering: str = "flat"


@dataclass(frozen=True)
class DotsAndBoxesTheme:
    """Resolved dots-and-boxes palette for one style variant."""

    board_fill_rgb: Tuple[int, int, int]
    board_border_rgb: Tuple[int, int, int]
    board_border_width_px: int
    shadow_rgb: Tuple[int, int, int]
    shadow_alpha: int
    shadow_offset_px: Tuple[int, int]
    title_rgb: Tuple[int, int, int]
    dot_rgb: Tuple[int, int, int]
    edge_rgb: Tuple[int, int, int]
    edge_width_px: int
    highlight_rgb: Tuple[int, int, int]
    highlight_width_px: int
    guide_rgb: Tuple[int, int, int]
    board_inner_fill_rgb: Tuple[int, int, int] | None = None
    board_pattern_rgb: Tuple[int, int, int] | None = None
    board_pattern_alpha: int = 0
    board_rendering: str = "flat"
    dot_outline_rgb: Tuple[int, int, int] | None = None
    dot_rendering: str = "flat"


def build_games_reversi_theme(*, style_variant: str) -> ReversiTheme:
    """Return one resolved Reversi-board theme for the active style variant."""

    variant = str(style_variant)
    if variant == "wood_board":
        return ReversiTheme(
            board_frame_rgb=(92, 58, 32),
            board_fill_rgb=(82, 132, 78),
            grid_line_rgb=(52, 92, 54),
            badge_fill_rgb=(252, 247, 238),
            badge_outline_rgb=(118, 86, 54),
            badge_text_rgb=(42, 34, 28),
            black_disc_fill_rgb=(34, 29, 25),
            black_disc_outline_rgb=(14, 12, 10),
            black_disc_shine_rgb=(94, 82, 70),
            white_disc_fill_rgb=(251, 246, 234),
            white_disc_outline_rgb=(128, 116, 98),
            white_disc_shine_rgb=(255, 255, 255),
            disc_outline_width_px=3,
            marked_square_outline_rgb=(203, 66, 54),
            marked_square_fill_rgba=(203, 66, 54, 36),
        )
    if variant == "slate_board":
        return ReversiTheme(
            board_frame_rgb=(35, 43, 52),
            board_fill_rgb=(62, 92, 84),
            grid_line_rgb=(28, 48, 46),
            badge_fill_rgb=(238, 242, 245),
            badge_outline_rgb=(84, 96, 108),
            badge_text_rgb=(24, 30, 36),
            black_disc_fill_rgb=(18, 23, 30),
            black_disc_outline_rgb=(8, 10, 13),
            black_disc_shine_rgb=(78, 88, 98),
            white_disc_fill_rgb=(240, 244, 246),
            white_disc_outline_rgb=(118, 128, 138),
            white_disc_shine_rgb=(255, 255, 255),
            disc_outline_width_px=3,
            marked_square_outline_rgb=(226, 74, 66),
            marked_square_fill_rgba=(226, 74, 66, 38),
        )
    if variant == "blue_board":
        return ReversiTheme(
            board_frame_rgb=(32, 55, 104),
            board_fill_rgb=(74, 128, 156),
            grid_line_rgb=(30, 74, 98),
            badge_fill_rgb=(242, 247, 252),
            badge_outline_rgb=(72, 92, 130),
            badge_text_rgb=(22, 38, 62),
            black_disc_fill_rgb=(26, 30, 38),
            black_disc_outline_rgb=(10, 12, 18),
            black_disc_shine_rgb=(82, 92, 112),
            white_disc_fill_rgb=(250, 253, 255),
            white_disc_outline_rgb=(116, 130, 146),
            white_disc_shine_rgb=(255, 255, 255),
            disc_outline_width_px=3,
            marked_square_outline_rgb=(215, 64, 76),
            marked_square_fill_rgba=(215, 64, 76, 36),
        )
    if variant == "soft":
        return ReversiTheme(
            board_frame_rgb=(76, 56, 38),
            board_fill_rgb=(66, 126, 86),
            grid_line_rgb=(29, 74, 46),
            badge_fill_rgb=(241, 244, 247),
            badge_outline_rgb=(102, 112, 122),
            badge_text_rgb=(32, 38, 44),
            black_disc_fill_rgb=(35, 39, 46),
            black_disc_outline_rgb=(18, 22, 28),
            black_disc_shine_rgb=(92, 100, 110),
            white_disc_fill_rgb=(245, 247, 250),
            white_disc_outline_rgb=(124, 132, 142),
            white_disc_shine_rgb=(255, 255, 255),
            disc_outline_width_px=3,
            marked_square_outline_rgb=(199, 63, 59),
            marked_square_fill_rgba=(199, 63, 59, 34),
        )
    if variant == "outlined":
        return ReversiTheme(
            board_frame_rgb=(58, 43, 31),
            board_fill_rgb=(72, 135, 88),
            grid_line_rgb=(24, 68, 41),
            badge_fill_rgb=(255, 255, 255),
            badge_outline_rgb=(72, 82, 92),
            badge_text_rgb=(26, 30, 36),
            black_disc_fill_rgb=(28, 31, 36),
            black_disc_outline_rgb=(10, 12, 15),
            black_disc_shine_rgb=(86, 94, 104),
            white_disc_fill_rgb=(255, 255, 255),
            white_disc_outline_rgb=(122, 130, 140),
            white_disc_shine_rgb=(255, 255, 255),
            disc_outline_width_px=4,
            marked_square_outline_rgb=(208, 60, 58),
            marked_square_fill_rgba=(208, 60, 58, 28),
        )
    return ReversiTheme(
        board_frame_rgb=(70, 50, 34),
        board_fill_rgb=(60, 121, 79),
        grid_line_rgb=(26, 72, 43),
        badge_fill_rgb=(248, 249, 251),
        badge_outline_rgb=(94, 102, 110),
        badge_text_rgb=(27, 32, 38),
        black_disc_fill_rgb=(31, 34, 39),
        black_disc_outline_rgb=(13, 15, 19),
        black_disc_shine_rgb=(84, 92, 102),
        white_disc_fill_rgb=(250, 251, 253),
        white_disc_outline_rgb=(128, 136, 145),
        white_disc_shine_rgb=(255, 255, 255),
        disc_outline_width_px=3,
        marked_square_outline_rgb=(203, 58, 57),
        marked_square_fill_rgba=(203, 58, 57, 32),
    )


def build_games_connect_four_theme(*, style_variant: str) -> ConnectFourTheme:
    """Return one resolved Connect Four theme for the active style variant."""

    variant = str(style_variant)
    if variant == "arcade_blue":
        return ConnectFourTheme(
            board_frame_rgb=(18, 33, 92),
            board_fill_rgb=(34, 78, 190),
            cell_well_rgb=(224, 234, 252),
            cell_well_outline_rgb=(13, 42, 132),
            cell_well_outline_width_px=4,
            badge_fill_rgb=(245, 249, 255),
            badge_outline_rgb=(35, 54, 112),
            badge_text_rgb=(20, 31, 62),
            red_disc_fill_rgb=(221, 48, 51),
            red_disc_outline_rgb=(132, 24, 28),
            red_disc_shine_rgb=(252, 150, 145),
            yellow_disc_fill_rgb=(252, 208, 45),
            yellow_disc_outline_rgb=(174, 126, 12),
            yellow_disc_shine_rgb=(255, 238, 142),
            disc_outline_width_px=4,
            marked_square_outline_rgb=(255, 255, 255),
            marked_square_fill_rgba=(255, 255, 255, 34),
            board_shadow_rgb=(8, 12, 28),
            board_shadow_alpha=58,
            board_shadow_offset_px=(8, 10),
            board_rendering="inset",
            cell_well_rendering="inset",
            disc_rendering="glossy",
        )
    if variant == "teal_frame":
        return ConnectFourTheme(
            board_frame_rgb=(18, 93, 104),
            board_fill_rgb=(34, 139, 151),
            cell_well_rgb=(235, 249, 250),
            cell_well_outline_rgb=(102, 188, 194),
            cell_well_outline_width_px=2,
            badge_fill_rgb=(245, 251, 250),
            badge_outline_rgb=(55, 127, 134),
            badge_text_rgb=(17, 52, 56),
            red_disc_fill_rgb=(205, 67, 68),
            red_disc_outline_rgb=(132, 42, 42),
            red_disc_shine_rgb=(234, 132, 130),
            yellow_disc_fill_rgb=(238, 197, 70),
            yellow_disc_outline_rgb=(164, 127, 37),
            yellow_disc_shine_rgb=(255, 226, 139),
            disc_outline_width_px=3,
            marked_square_outline_rgb=(255, 248, 180),
            marked_square_fill_rgba=(255, 248, 180, 32),
            board_shadow_rgb=(10, 34, 38),
            board_shadow_alpha=46,
            board_shadow_offset_px=(7, 8),
            board_rendering="inset",
            cell_well_rendering="ring",
            disc_rendering="flat",
        )
    if variant == "charcoal":
        return ConnectFourTheme(
            board_frame_rgb=(28, 31, 38),
            board_fill_rgb=(58, 64, 78),
            cell_well_rgb=(236, 238, 242),
            cell_well_outline_rgb=(33, 37, 46),
            cell_well_outline_width_px=3,
            badge_fill_rgb=(247, 248, 250),
            badge_outline_rgb=(71, 78, 92),
            badge_text_rgb=(24, 28, 35),
            red_disc_fill_rgb=(219, 54, 61),
            red_disc_outline_rgb=(118, 25, 33),
            red_disc_shine_rgb=(246, 139, 145),
            yellow_disc_fill_rgb=(244, 204, 64),
            yellow_disc_outline_rgb=(157, 116, 26),
            yellow_disc_shine_rgb=(255, 232, 135),
            disc_outline_width_px=4,
            marked_square_outline_rgb=(89, 170, 255),
            marked_square_fill_rgba=(89, 170, 255, 34),
            board_shadow_rgb=(4, 5, 8),
            board_shadow_alpha=56,
            board_shadow_offset_px=(7, 9),
            board_rendering="flat",
            cell_well_rendering="inset",
            disc_rendering="token",
        )
    if variant == "soft":
        return ConnectFourTheme(
            board_frame_rgb=(39, 61, 138),
            board_fill_rgb=(66, 99, 208),
            cell_well_rgb=(235, 241, 255),
            cell_well_outline_rgb=(184, 198, 236),
            cell_well_outline_width_px=2,
            badge_fill_rgb=(241, 244, 247),
            badge_outline_rgb=(102, 112, 122),
            badge_text_rgb=(32, 38, 44),
            red_disc_fill_rgb=(212, 72, 68),
            red_disc_outline_rgb=(150, 45, 42),
            red_disc_shine_rgb=(241, 160, 154),
            yellow_disc_fill_rgb=(245, 201, 72),
            yellow_disc_outline_rgb=(186, 142, 34),
            yellow_disc_shine_rgb=(255, 232, 150),
            disc_outline_width_px=3,
            marked_square_outline_rgb=(199, 63, 59),
            marked_square_fill_rgba=(199, 63, 59, 30),
        )
    if variant == "outlined":
        return ConnectFourTheme(
            board_frame_rgb=(32, 48, 118),
            board_fill_rgb=(54, 86, 196),
            cell_well_rgb=(255, 255, 255),
            cell_well_outline_rgb=(193, 206, 242),
            cell_well_outline_width_px=3,
            badge_fill_rgb=(255, 255, 255),
            badge_outline_rgb=(72, 82, 92),
            badge_text_rgb=(26, 30, 36),
            red_disc_fill_rgb=(220, 60, 58),
            red_disc_outline_rgb=(150, 34, 32),
            red_disc_shine_rgb=(247, 160, 156),
            yellow_disc_fill_rgb=(250, 206, 54),
            yellow_disc_outline_rgb=(182, 136, 18),
            yellow_disc_shine_rgb=(255, 237, 150),
            disc_outline_width_px=4,
            marked_square_outline_rgb=(208, 60, 58),
            marked_square_fill_rgba=(208, 60, 58, 24),
        )
    return ConnectFourTheme(
        board_frame_rgb=(35, 55, 126),
        board_fill_rgb=(58, 91, 204),
        cell_well_rgb=(238, 244, 255),
        cell_well_outline_rgb=(187, 201, 238),
        cell_well_outline_width_px=2,
        badge_fill_rgb=(248, 249, 251),
        badge_outline_rgb=(94, 102, 110),
        badge_text_rgb=(27, 32, 38),
        red_disc_fill_rgb=(216, 66, 62),
        red_disc_outline_rgb=(146, 40, 36),
        red_disc_shine_rgb=(244, 158, 152),
        yellow_disc_fill_rgb=(246, 203, 60),
        yellow_disc_outline_rgb=(185, 139, 24),
        yellow_disc_shine_rgb=(255, 233, 148),
        disc_outline_width_px=3,
        marked_square_outline_rgb=(203, 58, 57),
        marked_square_fill_rgba=(203, 58, 57, 28),
    )


def build_games_checkers_theme(*, style_variant: str) -> CheckersTheme:
    """Return one resolved Checkers theme for the active style variant."""

    variant = str(style_variant)
    if variant == "wood_token":
        return CheckersTheme(
            board_frame_rgb=(91, 54, 28),
            light_square_rgb=(238, 202, 154),
            dark_square_rgb=(139, 83, 42),
            grid_line_rgb=(101, 59, 30),
            grid_line_width_px=2,
            badge_fill_rgb=(250, 245, 238),
            badge_outline_rgb=(111, 75, 46),
            badge_text_rgb=(40, 27, 18),
            red_piece_fill_rgb=(206, 57, 51),
            red_piece_outline_rgb=(126, 32, 28),
            red_piece_shine_rgb=(244, 158, 150),
            black_piece_fill_rgb=(37, 29, 26),
            black_piece_outline_rgb=(14, 11, 10),
            black_piece_shine_rgb=(108, 92, 82),
            piece_outline_width_px=4,
            piece_shadow_rgb=(16, 10, 6),
            piece_shadow_alpha=46,
            piece_rendering="double_ring",
            square_rendering="inset",
        )
    if variant == "blue_table":
        return CheckersTheme(
            board_frame_rgb=(42, 62, 92),
            light_square_rgb=(240, 247, 253),
            dark_square_rgb=(94, 128, 172),
            grid_line_rgb=(60, 80, 112),
            grid_line_width_px=2,
            badge_fill_rgb=(246, 249, 253),
            badge_outline_rgb=(74, 91, 116),
            badge_text_rgb=(20, 29, 42),
            red_piece_fill_rgb=(218, 70, 64),
            red_piece_outline_rgb=(137, 38, 34),
            red_piece_shine_rgb=(246, 170, 160),
            black_piece_fill_rgb=(23, 29, 40),
            black_piece_outline_rgb=(8, 12, 18),
            black_piece_shine_rgb=(86, 102, 124),
            piece_outline_width_px=3,
            piece_shadow_rgb=(8, 13, 20),
            piece_shadow_alpha=34,
            piece_rendering="flat",
            square_rendering="flat",
        )
    if variant == "charcoal":
        return CheckersTheme(
            board_frame_rgb=(52, 54, 58),
            light_square_rgb=(236, 236, 230),
            dark_square_rgb=(126, 128, 132),
            grid_line_rgb=(66, 68, 72),
            grid_line_width_px=3,
            badge_fill_rgb=(248, 248, 246),
            badge_outline_rgb=(84, 86, 90),
            badge_text_rgb=(24, 25, 27),
            red_piece_fill_rgb=(196, 50, 54),
            red_piece_outline_rgb=(122, 27, 31),
            red_piece_shine_rgb=(235, 144, 146),
            black_piece_fill_rgb=(22, 23, 25),
            black_piece_outline_rgb=(6, 7, 8),
            black_piece_shine_rgb=(92, 96, 102),
            piece_outline_width_px=4,
            piece_shadow_rgb=(6, 7, 8),
            piece_shadow_alpha=28,
            piece_rendering="ring",
            square_rendering="inset",
        )
    if variant == "soft":
        return CheckersTheme(
            board_frame_rgb=(103, 70, 44),
            light_square_rgb=(235, 223, 198),
            dark_square_rgb=(110, 76, 48),
            grid_line_rgb=(88, 63, 42),
            grid_line_width_px=2,
            badge_fill_rgb=(241, 244, 247),
            badge_outline_rgb=(102, 112, 122),
            badge_text_rgb=(32, 38, 44),
            red_piece_fill_rgb=(205, 74, 69),
            red_piece_outline_rgb=(144, 45, 40),
            red_piece_shine_rgb=(238, 166, 158),
            black_piece_fill_rgb=(47, 50, 56),
            black_piece_outline_rgb=(21, 24, 29),
            black_piece_shine_rgb=(112, 118, 128),
            piece_outline_width_px=3,
            piece_shadow_rgb=(14, 16, 18),
            piece_shadow_alpha=24,
            piece_rendering="ring",
            square_rendering="flat",
        )
    if variant == "outlined":
        return CheckersTheme(
            board_frame_rgb=(94, 62, 37),
            light_square_rgb=(246, 239, 220),
            dark_square_rgb=(117, 80, 46),
            grid_line_rgb=(82, 59, 40),
            grid_line_width_px=3,
            badge_fill_rgb=(255, 255, 255),
            badge_outline_rgb=(72, 82, 92),
            badge_text_rgb=(26, 30, 36),
            red_piece_fill_rgb=(215, 61, 58),
            red_piece_outline_rgb=(147, 34, 31),
            red_piece_shine_rgb=(244, 165, 160),
            black_piece_fill_rgb=(36, 39, 44),
            black_piece_outline_rgb=(9, 12, 15),
            black_piece_shine_rgb=(104, 112, 122),
            piece_outline_width_px=4,
            piece_shadow_rgb=(10, 12, 14),
            piece_shadow_alpha=20,
            piece_rendering="double_ring",
            square_rendering="inset",
        )
    return CheckersTheme(
        board_frame_rgb=(98, 66, 41),
        light_square_rgb=(241, 232, 210),
        dark_square_rgb=(103, 70, 42),
        grid_line_rgb=(84, 60, 39),
        grid_line_width_px=2,
        badge_fill_rgb=(248, 249, 251),
        badge_outline_rgb=(94, 102, 110),
        badge_text_rgb=(27, 32, 38),
        red_piece_fill_rgb=(210, 67, 63),
        red_piece_outline_rgb=(143, 41, 37),
        red_piece_shine_rgb=(241, 162, 156),
        black_piece_fill_rgb=(41, 44, 50),
        black_piece_outline_rgb=(15, 18, 22),
        black_piece_shine_rgb=(106, 114, 124),
        piece_outline_width_px=3,
        piece_shadow_rgb=(12, 14, 16),
        piece_shadow_alpha=26,
        piece_rendering="ring",
        square_rendering="flat",
    )


def build_games_chess_theme(*, style_variant: str) -> ChessTheme:
    """Return one resolved Chess-board theme for the active style variant."""

    variant = str(style_variant)
    if variant == "wood_token":
        return ChessTheme(
            board_frame_rgb=(82, 55, 34),
            light_square_rgb=(194, 151, 100),
            dark_square_rgb=(137, 95, 60),
            grid_line_rgb=(84, 58, 39),
            grid_line_width_px=2,
            badge_fill_rgb=(250, 245, 238),
            badge_outline_rgb=(111, 75, 46),
            badge_text_rgb=(40, 27, 18),
            white_piece_fill_rgb=(255, 255, 255),
            white_piece_outline_rgb=(0, 0, 0),
            black_piece_fill_rgb=(0, 0, 0),
            black_piece_outline_rgb=(255, 255, 255),
            marked_square_outline_rgb=(220, 38, 38),
            marked_square_fill_rgba=(220, 38, 38, 0),
            piece_shadow_rgb=(18, 12, 8),
            piece_shadow_alpha=52,
            piece_rendering="glyph",
            square_rendering="inset",
        )
    if variant == "blue_glyph":
        return ChessTheme(
            board_frame_rgb=(38, 58, 88),
            light_square_rgb=(158, 188, 218),
            dark_square_rgb=(88, 122, 164),
            grid_line_rgb=(50, 74, 108),
            grid_line_width_px=3,
            badge_fill_rgb=(246, 249, 253),
            badge_outline_rgb=(74, 91, 116),
            badge_text_rgb=(20, 29, 42),
            white_piece_fill_rgb=(255, 255, 255),
            white_piece_outline_rgb=(0, 0, 0),
            black_piece_fill_rgb=(0, 0, 0),
            black_piece_outline_rgb=(255, 255, 255),
            marked_square_outline_rgb=(220, 38, 38),
            marked_square_fill_rgba=(220, 38, 38, 0),
            piece_shadow_rgb=(8, 13, 20),
            piece_shadow_alpha=26,
            piece_rendering="glyph",
            square_rendering="flat",
        )
    if variant == "monochrome_glyph":
        return ChessTheme(
            board_frame_rgb=(54, 56, 60),
            light_square_rgb=(186, 188, 182),
            dark_square_rgb=(105, 110, 116),
            grid_line_rgb=(66, 68, 72),
            grid_line_width_px=3,
            badge_fill_rgb=(248, 248, 246),
            badge_outline_rgb=(84, 86, 90),
            badge_text_rgb=(24, 25, 27),
            white_piece_fill_rgb=(255, 255, 255),
            white_piece_outline_rgb=(0, 0, 0),
            black_piece_fill_rgb=(0, 0, 0),
            black_piece_outline_rgb=(255, 255, 255),
            marked_square_outline_rgb=(220, 38, 38),
            marked_square_fill_rgba=(220, 38, 38, 0),
            piece_shadow_rgb=(6, 7, 8),
            piece_shadow_alpha=20,
            piece_rendering="glyph",
            square_rendering="inset",
        )
    if variant == "soft":
        return ChessTheme(
            board_frame_rgb=(82, 72, 62),
            light_square_rgb=(184, 180, 158),
            dark_square_rgb=(112, 142, 112),
            grid_line_rgb=(76, 91, 77),
            grid_line_width_px=2,
            badge_fill_rgb=(241, 244, 247),
            badge_outline_rgb=(102, 112, 122),
            badge_text_rgb=(32, 38, 44),
            white_piece_fill_rgb=(255, 255, 255),
            white_piece_outline_rgb=(0, 0, 0),
            black_piece_fill_rgb=(0, 0, 0),
            black_piece_outline_rgb=(255, 255, 255),
            marked_square_outline_rgb=(220, 38, 38),
            marked_square_fill_rgba=(220, 38, 38, 0),
            piece_shadow_rgb=(15, 20, 18),
            piece_shadow_alpha=46,
            piece_rendering="glyph",
            square_rendering="flat",
        )
    if variant == "outlined":
        return ChessTheme(
            board_frame_rgb=(62, 70, 80),
            light_square_rgb=(176, 188, 178),
            dark_square_rgb=(102, 137, 112),
            grid_line_rgb=(78, 90, 98),
            grid_line_width_px=3,
            badge_fill_rgb=(255, 255, 255),
            badge_outline_rgb=(72, 82, 92),
            badge_text_rgb=(26, 30, 36),
            white_piece_fill_rgb=(255, 255, 255),
            white_piece_outline_rgb=(0, 0, 0),
            black_piece_fill_rgb=(0, 0, 0),
            black_piece_outline_rgb=(255, 255, 255),
            marked_square_outline_rgb=(220, 38, 38),
            marked_square_fill_rgba=(220, 38, 38, 0),
            piece_shadow_rgb=(12, 16, 18),
            piece_shadow_alpha=34,
            piece_rendering="glyph",
            square_rendering="inset",
        )
    return ChessTheme(
        board_frame_rgb=(76, 59, 42),
        light_square_rgb=(196, 180, 150),
        dark_square_rgb=(111, 140, 100),
        grid_line_rgb=(76, 86, 72),
        grid_line_width_px=2,
        badge_fill_rgb=(248, 249, 251),
        badge_outline_rgb=(94, 102, 110),
        badge_text_rgb=(27, 32, 38),
        white_piece_fill_rgb=(255, 255, 255),
        white_piece_outline_rgb=(0, 0, 0),
        black_piece_fill_rgb=(0, 0, 0),
        black_piece_outline_rgb=(255, 255, 255),
        marked_square_outline_rgb=(220, 38, 38),
        marked_square_fill_rgba=(220, 38, 38, 0),
        piece_shadow_rgb=(14, 18, 20),
        piece_shadow_alpha=42,
        piece_rendering="glyph",
        square_rendering="flat",
    )


def build_games_dots_and_boxes_theme(*, style_variant: str) -> DotsAndBoxesTheme:
    """Return one resolved dots-and-boxes theme for the active style variant."""

    variant = str(style_variant)
    if variant == "notebook":
        return DotsAndBoxesTheme(
            board_fill_rgb=(255, 253, 246),
            board_border_rgb=(94, 112, 145),
            board_border_width_px=3,
            shadow_rgb=(15, 23, 42),
            shadow_alpha=44,
            shadow_offset_px=(5, 6),
            title_rgb=(42, 85, 176),
            dot_rgb=(25, 31, 42),
            edge_rgb=(38, 48, 66),
            edge_width_px=7,
            highlight_rgb=(214, 62, 58),
            highlight_width_px=10,
            guide_rgb=(184, 199, 224),
            board_inner_fill_rgb=(255, 255, 252),
            board_pattern_rgb=(159, 191, 226),
            board_pattern_alpha=46,
            board_rendering="notebook",
            dot_outline_rgb=(255, 255, 255),
            dot_rendering="outlined",
        )
    if variant == "slate":
        return DotsAndBoxesTheme(
            board_fill_rgb=(53, 61, 73),
            board_border_rgb=(194, 204, 216),
            board_border_width_px=4,
            shadow_rgb=(3, 6, 12),
            shadow_alpha=62,
            shadow_offset_px=(5, 7),
            title_rgb=(240, 244, 248),
            dot_rgb=(236, 240, 245),
            edge_rgb=(224, 231, 238),
            edge_width_px=8,
            highlight_rgb=(90, 203, 255),
            highlight_width_px=10,
            guide_rgb=(120, 132, 150),
            board_inner_fill_rgb=(62, 71, 84),
            board_pattern_rgb=(148, 163, 184),
            board_pattern_alpha=28,
            board_rendering="inset",
            dot_outline_rgb=(23, 29, 38),
            dot_rendering="outlined",
        )
    if variant == "wood_panel":
        return DotsAndBoxesTheme(
            board_fill_rgb=(172, 118, 72),
            board_border_rgb=(91, 55, 30),
            board_border_width_px=4,
            shadow_rgb=(18, 10, 6),
            shadow_alpha=58,
            shadow_offset_px=(6, 7),
            title_rgb=(56, 34, 20),
            dot_rgb=(42, 25, 15),
            edge_rgb=(55, 33, 20),
            edge_width_px=8,
            highlight_rgb=(33, 104, 196),
            highlight_width_px=10,
            guide_rgb=(136, 89, 52),
            board_inner_fill_rgb=(205, 154, 101),
            board_pattern_rgb=(121, 75, 39),
            board_pattern_alpha=34,
            board_rendering="wood",
            dot_outline_rgb=(232, 190, 137),
            dot_rendering="outlined",
        )
    if variant == "soft":
        return DotsAndBoxesTheme(
            board_fill_rgb=(249, 244, 234),
            board_border_rgb=(102, 112, 124),
            board_border_width_px=3,
            shadow_rgb=(18, 24, 20),
            shadow_alpha=56,
            shadow_offset_px=(5, 6),
            title_rgb=(52, 76, 132),
            dot_rgb=(54, 60, 68),
            edge_rgb=(66, 74, 86),
            edge_width_px=8,
            highlight_rgb=(202, 74, 60),
            highlight_width_px=10,
            guide_rgb=(176, 184, 194),
        )
    if variant == "outlined":
        return DotsAndBoxesTheme(
            board_fill_rgb=(255, 255, 255),
            board_border_rgb=(62, 70, 80),
            board_border_width_px=4,
            shadow_rgb=(14, 18, 20),
            shadow_alpha=40,
            shadow_offset_px=(4, 5),
            title_rgb=(44, 75, 168),
            dot_rgb=(36, 40, 46),
            edge_rgb=(52, 58, 68),
            edge_width_px=8,
            highlight_rgb=(214, 60, 54),
            highlight_width_px=10,
            guide_rgb=(196, 202, 210),
        )
    return DotsAndBoxesTheme(
        board_fill_rgb=(255, 252, 244),
        board_border_rgb=(74, 82, 92),
        board_border_width_px=3,
        shadow_rgb=(18, 22, 24),
        shadow_alpha=48,
        shadow_offset_px=(4, 5),
        title_rgb=(45, 76, 160),
        dot_rgb=(30, 34, 40),
        edge_rgb=(46, 52, 60),
        edge_width_px=8,
        highlight_rgb=(210, 58, 52),
        highlight_width_px=10,
        guide_rgb=(184, 190, 198),
    )
