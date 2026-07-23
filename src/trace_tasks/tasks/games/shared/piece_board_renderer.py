"""Shared Chess board renderer for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ...shared.text_legibility import draw_text_traced as draw_fixed_text_traced
from ...shared.text_rendering import fit_font_to_box, load_font, resolve_text_stroke_fill, temporary_default_font_family
from .text import draw_game_text_traced
from .piece_board_rules import (
    BLACK,
    BOARD_SIZE,
    WHITE,
    ChessPiece,
    Coord,
    color_name,
    coord_to_cell_id,
    piece_to_entity_id,
)
from .layout import apply_games_layout_jitter_to_bbox, offset_bbox
from .marking import draw_semantic_bbox_marker, resolve_semantic_marker_style
from .scene_style import GamePanelSceneStyle, draw_panel_scene_chrome, game_panel_scene_style_metadata
from .style import ChessTheme, build_games_chess_theme


@dataclass(frozen=True)
class ChessRenderParams:
    """Resolved render controls for one Chess scene."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    player_badge_height_px: int
    player_badge_width_px: int
    header_gap_px: int
    max_board_size_px: int
    board_corner_radius_px: int
    board_frame_width_px: int
    piece_inset_fraction: float
    piece_font_size_px: int
    marked_square_outline_width_px: int
    player_badge_font_size_px: int
    coordinate_label_font_size_px: int = 18
    option_panel_gap_px: int = 18
    option_panel_height_px: int = 0
    option_panel_font_size_px: int = 20
    layout_jitter_meta: Dict[str, Any] | None = None
    font_family: str = ""
    instance_seed: int = 0


@dataclass(frozen=True)
class ChessCellSpec:
    """One board cell after layout/render assignment."""

    cell_id: str
    row: int
    col: int
    occupant: str
    bbox_px: Tuple[float, float, float, float]
    piece_bbox_px: Tuple[float, float, float, float] | None


@dataclass(frozen=True)
class RenderedChessScene:
    """Rendered Chess scene plus trace-friendly metadata."""

    image: Image.Image
    cell_specs: Tuple[ChessCellSpec, ...]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


_FILLED_PIECE_CODEPOINTS: Dict[str, int] = {
    "king": 0x265A,
    "queen": 0x265B,
    "rook": 0x265C,
    "bishop": 0x265D,
    "knight": 0x265E,
    "pawn": 0x265F,
}


def _adjust_rgb(rgb: Sequence[int], delta: int) -> Tuple[int, int, int]:
    """Return an RGB color lightened/darkened by a small channel delta."""

    return tuple(max(0, min(255, int(value) + int(delta))) for value in rgb[:3])


def _inset_square_rgb(rgb: Sequence[int]) -> Tuple[int, int, int]:
    """Return a subtle inner-square shade for inset board styles."""

    brightness = sum(int(value) for value in rgb[:3]) / 3.0
    return _adjust_rgb(rgb, 10 if brightness < 158.0 else -7)


def _piece_bbox(cell_bbox: Tuple[float, float, float, float], *, inset_fraction: float) -> Tuple[float, float, float, float]:
    """Return an inscribed piece bbox inside one board cell."""

    left, top, right, bottom = cell_bbox
    inset = float(max(4.0, inset_fraction * min(right - left, bottom - top)))
    return (
        round(float(left + inset), 3),
        round(float(top + inset), 3),
        round(float(right - inset), 3),
        round(float(bottom - inset), 3),
    )


def _fit_chess_symbol_font(
    draw: ImageDraw.ImageDraw,
    *,
    glyph: str,
    max_width: float,
    max_height: float,
    min_size_px: int,
    max_size_px: int,
    fill_ratio: float,
):
    """Fit a chess-symbol font independent of the sampled scene text font."""

    with temporary_default_font_family(""):
        return fit_font_to_box(
            draw,
            text=str(glyph),
            max_width=float(max_width),
            max_height=float(max_height),
            bold=False,
            min_size_px=int(min_size_px),
            max_size_px=int(max_size_px),
            fill_ratio=float(fill_ratio),
        )


def draw_chess_piece_symbol(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Tuple[float, float, float, float],
    piece: ChessPiece,
    theme: ChessTheme,
    font_size_px: int,
) -> None:
    """Draw one chess piece glyph centered in its cell."""

    left, top, right, bottom = bbox_px
    width = float(right - left)
    height = float(bottom - top)
    if str(piece.color) == WHITE:
        glyph_fill = tuple(int(value) for value in theme.white_piece_fill_rgb)
        glyph_stroke = tuple(int(value) for value in theme.white_piece_outline_rgb)
    else:
        glyph_fill = tuple(int(value) for value in theme.black_piece_fill_rgb)
        glyph_stroke = tuple(int(value) for value in theme.black_piece_outline_rgb)
    glyph = chr(_FILLED_PIECE_CODEPOINTS[str(piece.kind)])
    font = _fit_chess_symbol_font(
        draw,
        glyph=glyph,
        max_width=width,
        max_height=height,
        min_size_px=18,
        max_size_px=int(font_size_px),
        fill_ratio=0.98,
    )
    stroke_width = max(1, int(round(0.035 * min(width, height))))
    text_bbox = draw.textbbox((0, 0), glyph, font=font, stroke_width=stroke_width)
    text_width = float(text_bbox[2] - text_bbox[0])
    text_height = float(text_bbox[3] - text_bbox[1])
    x = float(left + (0.5 * (width - text_width)) - text_bbox[0])
    y = float(top + (0.5 * (height - text_height)) - text_bbox[1] - (0.02 * height))
    draw_fixed_text_traced(draw,
        (x, y),
        glyph,
        font=font,
        fill=glyph_fill,
        stroke_width=stroke_width,
        stroke_fill=glyph_stroke,
    role="game_piece", required=False,)


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    bbox: Tuple[float, float, float, float],
    font,
    fill: Tuple[int, int, int],
    role: str,
    stroke_width: int = 0,
) -> Tuple[float, float, float, float]:
    """Draw text centered in a box and return its drawn bbox."""

    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=int(stroke_width))
    text_width = float(text_bbox[2] - text_bbox[0])
    text_height = float(text_bbox[3] - text_bbox[1])
    x = float(bbox[0] + (0.5 * ((bbox[2] - bbox[0]) - text_width)) - text_bbox[0])
    y = float(bbox[1] + (0.5 * ((bbox[3] - bbox[1]) - text_height)) - text_bbox[1])
    stroke_fill = tuple(int(value) for value in resolve_text_stroke_fill(fill)) if int(stroke_width) > 0 else None
    draw_game_text_traced(
        draw,
        (x, y),
        str(text),
        font=font,
        fill=fill,
        stroke_width=int(stroke_width),
        stroke_fill=stroke_fill,
        role=str(role),
        required=False,
    )
    return (
        round(float(x + text_bbox[0]), 3),
        round(float(y + text_bbox[1]), 3),
        round(float(x + text_bbox[2]), 3),
        round(float(y + text_bbox[3]), 3),
    )


def _draw_coordinate_labels(
    draw: ImageDraw.ImageDraw,
    *,
    board_bbox: Tuple[float, float, float, float],
    cell_size: int,
    params: ChessRenderParams,
    theme: ChessTheme,
) -> Dict[str, Any]:
    """Draw standard file/rank labels in the chess board frame."""

    gutter = max(26.0, float(params.coordinate_label_font_size_px) + 16.0)
    if gutter <= 0.0:
        return {}
    label_rgb = tuple(int(value) for value in theme.badge_text_rgb)
    labels: Dict[str, Any] = {"files": {}, "ranks": {}}
    inner_left = float(board_bbox[0])
    inner_top = float(board_bbox[1])
    inner_bottom = float(board_bbox[3])
    bottom_label_box_height = float(gutter)
    left_label_box_width = float(gutter)
    for col in range(BOARD_SIZE):
        file_label = chr(ord("A") + int(col))
        x0 = float(inner_left + (col * cell_size))
        x1 = float(inner_left + ((col + 1) * cell_size))
        bbox = (x0, inner_bottom + 2.0, x1, inner_bottom + bottom_label_box_height)
        font = fit_font_to_box(
            draw,
            text=file_label,
            max_width=max(8.0, float(bbox[2] - bbox[0]) - 6.0),
            max_height=max(8.0, float(bbox[3] - bbox[1]) - 6.0),
            bold=True,
            font_family=str(params.font_family) or None,
            min_size_px=9,
            max_size_px=int(params.coordinate_label_font_size_px),
            fill_ratio=0.95,
        )
        labels["files"][file_label] = list(
            _draw_centered_text(
                draw,
                text=file_label,
                bbox=bbox,
                font=font,
                fill=label_rgb,
                role="axis_label",
                stroke_width=1,
            )
        )
    for row in range(BOARD_SIZE):
        rank_label = str(BOARD_SIZE - int(row))
        y0 = float(inner_top + (row * cell_size))
        y1 = float(inner_top + ((row + 1) * cell_size))
        bbox = (float(inner_left - left_label_box_width - 2.0), y0, float(inner_left - 2.0), y1)
        font = fit_font_to_box(
            draw,
            text=rank_label,
            max_width=max(8.0, float(bbox[2] - bbox[0]) - 6.0),
            max_height=max(8.0, float(bbox[3] - bbox[1]) - 6.0),
            bold=True,
            font_family=str(params.font_family) or None,
            min_size_px=9,
            max_size_px=int(params.coordinate_label_font_size_px),
            fill_ratio=0.95,
        )
        labels["ranks"][rank_label] = list(
            _draw_centered_text(
                draw,
                text=rank_label,
                bbox=bbox,
                font=font,
                fill=label_rgb,
                role="axis_label",
                stroke_width=1,
            )
        )
    return labels


def _draw_move_option_panel(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[float, float, float, float],
    move_options: Sequence[Mapping[str, Any]],
    params: ChessRenderParams,
    theme: ChessTheme,
) -> Dict[str, Any]:
    """Draw the encoded move-option panel below the chess board."""

    if not move_options:
        return {}
    panel_bbox = tuple(round(float(value), 3) for value in bbox)
    draw.rounded_rectangle(
        panel_bbox,
        radius=max(10, int(round(0.22 * min(panel_bbox[2] - panel_bbox[0], panel_bbox[3] - panel_bbox[1])))),
        fill=tuple(int(value) for value in theme.badge_fill_rgb),
        outline=tuple(int(value) for value in theme.badge_outline_rgb),
        width=2,
    )
    count = len(move_options)
    columns = 3 if int(count) == 6 else (2 if int(count) > 3 else 1)
    rows = int((int(count) + int(columns) - 1) // int(columns))
    pad_x = max(14.0, 0.035 * float(panel_bbox[2] - panel_bbox[0]))
    pad_y = max(10.0, 0.09 * float(panel_bbox[3] - panel_bbox[1]))
    inner_left = float(panel_bbox[0] + pad_x)
    inner_top = float(panel_bbox[1] + pad_y)
    inner_width = float(panel_bbox[2] - panel_bbox[0] - (2.0 * pad_x))
    inner_height = float(panel_bbox[3] - panel_bbox[1] - (2.0 * pad_y))
    cell_width = float(inner_width / float(columns))
    cell_height = float(inner_height / float(rows))
    text_rgb = tuple(int(value) for value in theme.badge_text_rgb)
    option_bboxes: Dict[str, Any] = {}
    option_text_bboxes: Dict[str, Any] = {}
    for index, option in enumerate(move_options):
        label = str(option.get("label", "")).strip()
        text = str(option.get("text", "")).strip()
        row = int(index) // int(columns)
        col = int(index) % int(columns)
        option_bbox = (
            round(float(inner_left + (col * cell_width)), 3),
            round(float(inner_top + (row * cell_height)), 3),
            round(float(inner_left + ((col + 1) * cell_width) - 8.0), 3),
            round(float(inner_top + ((row + 1) * cell_height) - 4.0), 3),
        )
        if not label or not text:
            continue
        option_bboxes[label] = list(option_bbox)
        font = fit_font_to_box(
            draw,
            text=text,
            max_width=max(1.0, float(option_bbox[2] - option_bbox[0])),
            max_height=max(1.0, float(option_bbox[3] - option_bbox[1])),
            bold=True,
            font_family=str(params.font_family) or None,
            min_size_px=11,
            max_size_px=int(params.option_panel_font_size_px),
            fill_ratio=0.94,
        )
        text_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
        text_height = float(text_bbox[3] - text_bbox[1])
        x = float(option_bbox[0] - text_bbox[0] + 1.0)
        y = float(option_bbox[1] + (0.5 * ((option_bbox[3] - option_bbox[1]) - text_height)) - text_bbox[1])
        draw_game_text_traced(
            draw,
            (x, y),
            text,
            font=font,
            fill=text_rgb,
            stroke_width=1,
            stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(text_rgb)),
            role="option_label",
            required=False,
        )
        option_text_bboxes[label] = [
            round(float(x + text_bbox[0]), 3),
            round(float(y + text_bbox[1]), 3),
            round(float(x + text_bbox[2]), 3),
            round(float(y + text_bbox[3]), 3),
        ]
    return {
        "panel_bbox_px": list(panel_bbox),
        "option_bboxes_px": option_bboxes,
        "option_text_bboxes_px": option_text_bboxes,
    }


def render_chess_board_scene(
    *,
    board: Sequence[Sequence[ChessPiece | None]],
    background: Image.Image,
    scene_variant: str,
    style_variant: str,
    badge_text: str,
    marked_coord: Coord | None,
    params: ChessRenderParams,
    target_coord: Coord | None = None,
    panel_style: GamePanelSceneStyle | None = None,
    show_coordinates: bool = False,
    move_options: Sequence[Mapping[str, Any]] = (),
) -> RenderedChessScene:
    """Render one visible Chess board state."""

    del scene_variant
    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image)
    theme = build_games_chess_theme(style_variant=str(style_variant))

    option_panel_height = int(params.option_panel_height_px) if move_options else 0
    option_panel_gap = int(params.option_panel_gap_px) if move_options else 0
    cell_size = min(
        int(params.max_board_size_px) // BOARD_SIZE,
        (int(params.canvas_width) - (2 * int(params.panel_margin_px))) // BOARD_SIZE,
        (
            int(params.canvas_height)
            - (2 * int(params.panel_margin_px))
            - int(params.player_badge_height_px)
            - int(params.header_gap_px)
            - int(option_panel_gap)
            - int(option_panel_height)
        )
        // BOARD_SIZE,
    )
    board_size_px = int(cell_size) * BOARD_SIZE
    board_left = int(0.5 * (int(params.canvas_width) - int(board_size_px)))
    available_height = (
        int(params.canvas_height)
        - (2 * int(params.panel_margin_px))
        - int(params.player_badge_height_px)
        - int(params.header_gap_px)
        - int(option_panel_gap)
        - int(option_panel_height)
    )
    board_top = int(
        params.panel_margin_px
        + params.player_badge_height_px
        + params.header_gap_px
        + max(0, 0.5 * (available_height - int(board_size_px)))
    )
    board_bbox = (
        round(float(board_left), 3),
        round(float(board_top), 3),
        round(float(board_left + board_size_px), 3),
        round(float(board_top + board_size_px), 3),
    )

    badge_font = load_font(
        int(params.player_badge_font_size_px),
        bold=True,
        font_family=str(params.font_family) or None,
    )
    badge_text_bbox = draw.textbbox((0, 0), str(badge_text), font=badge_font, stroke_width=1)
    badge_width = max(
        int(params.player_badge_width_px),
        int((badge_text_bbox[2] - badge_text_bbox[0]) + 44),
    )
    badge_left = int(0.5 * (int(params.canvas_width) - int(badge_width)))
    badge_top = int(params.panel_margin_px)
    badge_bbox = (
        round(float(badge_left), 3),
        round(float(badge_top), 3),
        round(float(badge_left + badge_width), 3),
        round(float(badge_top + params.player_badge_height_px), 3),
    )
    coordinate_label_gutter = max(26.0, float(params.coordinate_label_font_size_px) + 16.0) if bool(show_coordinates) else 0.0
    move_option_count = len(move_options)
    option_panel_columns = 3 if int(move_option_count) == 6 else (2 if int(move_option_count) > 3 else 1)
    option_panel_min_width = 720.0 if int(option_panel_columns) >= 3 else (600.0 if int(option_panel_columns) == 2 else float(board_size_px))
    option_panel_max_width = max(
        float(board_size_px),
        float(int(params.canvas_width) - (2 * int(params.panel_margin_px))),
    )
    option_panel_width = float(board_size_px)
    if move_options:
        option_panel_width = min(float(option_panel_max_width), max(float(board_size_px), float(option_panel_min_width)))
    option_panel_left = float(board_left)
    if move_options:
        board_center_x = float(board_left + (0.5 * float(board_size_px)))
        min_left = float(params.panel_margin_px)
        max_left = float(int(params.canvas_width) - int(params.panel_margin_px) - float(option_panel_width))
        if max_left >= min_left:
            option_panel_left = min(max_left, max(min_left, board_center_x - (0.5 * float(option_panel_width))))
        else:
            option_panel_left = float(board_left)
    option_panel_bbox = (
        round(float(option_panel_left), 3),
        round(float(board_top + board_size_px + option_panel_gap), 3),
        round(float(option_panel_left + option_panel_width), 3),
        round(float(board_top + board_size_px + option_panel_gap + option_panel_height), 3),
    )
    group_bbox = (
        min(
            float(board_bbox[0]) - float(coordinate_label_gutter),
            float(badge_bbox[0]),
            float(option_panel_bbox[0]) if move_options else float(board_bbox[0]),
        ),
        min(float(board_bbox[1]), float(badge_bbox[1])),
        max(float(board_bbox[2]), float(badge_bbox[2]), float(option_panel_bbox[2]) if move_options else float(board_bbox[2])),
        max(
            float(board_bbox[3]) + float(coordinate_label_gutter),
            float(badge_bbox[3]),
            float(option_panel_bbox[3]) if move_options else float(board_bbox[3]),
        ),
    )
    _group_bbox, dx, dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=group_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    board_left = float(board_left + dx)
    board_top = float(board_top + dy)
    badge_left = float(badge_left + dx)
    badge_top = float(badge_top + dy)
    board_bbox = offset_bbox(board_bbox, dx=dx, dy=dy)
    badge_bbox = offset_bbox(badge_bbox, dx=dx, dy=dy)
    option_panel_bbox = offset_bbox(option_panel_bbox, dx=dx, dy=dy)
    group_bbox = offset_bbox(group_bbox, dx=dx, dy=dy)

    scene_panel_bbox: Tuple[int, int, int, int] | None = None
    if panel_style is not None:
        panel_pad = max(18, int(round(float(params.panel_margin_px) * 0.42)))
        scene_panel_bbox = (
            max(4, int(round(float(group_bbox[0]))) - panel_pad),
            max(4, int(round(float(group_bbox[1]))) - panel_pad),
            min(int(params.canvas_width) - 4, int(round(float(group_bbox[2]))) + panel_pad),
            min(int(params.canvas_height) - 4, int(round(float(group_bbox[3]))) + panel_pad),
        )
        draw_panel_scene_chrome(
            draw,
            bbox=scene_panel_bbox,
            style=panel_style,
            radius=26,
            border_width=max(2, int(round(float(params.board_frame_width_px) * 0.55))),
        )

    draw.rounded_rectangle(
        board_bbox,
        radius=int(params.board_corner_radius_px),
        fill=tuple(int(value) for value in theme.board_frame_rgb),
    )
    draw.rounded_rectangle(
        badge_bbox,
        radius=int(0.5 * int(params.player_badge_height_px)),
        fill=tuple(int(value) for value in theme.badge_fill_rgb),
        outline=tuple(int(value) for value in theme.badge_outline_rgb),
        width=2,
    )
    badge_text_rgb = tuple(int(value) for value in theme.badge_text_rgb)
    draw_game_text_traced(draw,
        (
            float(badge_left + 22),
            float(badge_top + 0.5 * (int(params.player_badge_height_px) - (badge_text_bbox[3] - badge_text_bbox[1]))),
        ),
        str(badge_text),
        font=badge_font,
        fill=badge_text_rgb,
        stroke_width=1,
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(badge_text_rgb)),
     role="readout", required=False,)

    cell_specs: List[ChessCellSpec] = []
    scene_entities: List[Dict[str, Any]] = []
    cell_bboxes_px: Dict[str, Tuple[float, float, float, float]] = {}
    cell_surface_rgbs: Dict[str, Tuple[int, int, int]] = {}
    piece_bboxes_px: Dict[str, Tuple[float, float, float, float]] = {}
    inner_inset = float(params.board_frame_width_px)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            left = float(board_left + (col * cell_size) + inner_inset)
            top = float(board_top + (row * cell_size) + inner_inset)
            right = float(board_left + ((col + 1) * cell_size) - inner_inset)
            bottom = float(board_top + ((row + 1) * cell_size) - inner_inset)
            cell_bbox = (round(left, 3), round(top, 3), round(right, 3), round(bottom, 3))
            is_light = (row + col) % 2 == 0
            square_rgb = tuple(int(value) for value in (theme.light_square_rgb if is_light else theme.dark_square_rgb))
            draw.rectangle(
                cell_bbox,
                fill=square_rgb,
                outline=tuple(int(value) for value in theme.grid_line_rgb),
                width=int(theme.grid_line_width_px),
            )
            if str(theme.square_rendering) == "inset":
                inset = max(2.0, 0.045 * min(cell_bbox[2] - cell_bbox[0], cell_bbox[3] - cell_bbox[1]))
                draw.rectangle(
                    [
                        cell_bbox[0] + inset,
                        cell_bbox[1] + inset,
                        cell_bbox[2] - inset,
                        cell_bbox[3] - inset,
                    ],
                    fill=_inset_square_rgb(square_rgb),
                )
            cell_id = coord_to_cell_id((int(row), int(col)))
            cell_surface_rgbs[str(cell_id)] = square_rgb
            occupant_piece = board[row][col]
            occupant = "empty" if occupant_piece is None else f"{occupant_piece.color}_{occupant_piece.kind}"
            piece_bbox_px: Tuple[float, float, float, float] | None = None
            if occupant_piece is not None:
                piece_bbox_px = _piece_bbox(cell_bbox, inset_fraction=float(params.piece_inset_fraction))
                draw_chess_piece_symbol(
                    draw,
                    bbox_px=piece_bbox_px,
                    piece=occupant_piece,
                    theme=theme,
                    font_size_px=int(params.piece_font_size_px),
                )
                piece_id = piece_to_entity_id((int(row), int(col)), occupant_piece)
                piece_bboxes_px[piece_id] = piece_bbox_px
                scene_entities.append(
                    {
                        "id": str(piece_id),
                        "type": "chess_piece",
                        "color": str(occupant_piece.color),
                        "kind": str(occupant_piece.kind),
                        "row": int(row),
                        "col": int(col),
                        "bbox_px": list(piece_bbox_px),
                    }
                )
            cell_bboxes_px[cell_id] = cell_bbox
            cell_specs.append(
                ChessCellSpec(
                    cell_id=str(cell_id),
                    row=int(row),
                    col=int(col),
                    occupant=str(occupant),
                    bbox_px=cell_bbox,
                    piece_bbox_px=piece_bbox_px,
                )
            )
            scene_entities.append(
                {
                    "id": str(cell_id),
                    "type": "chess_board_cell",
                    "row": int(row),
                    "col": int(col),
                    "occupant": str(occupant),
                    "bbox_px": list(cell_bbox),
                }
            )

    coordinate_label_bboxes = {}
    if bool(show_coordinates):
        coordinate_label_bboxes = _draw_coordinate_labels(
            draw,
            board_bbox=board_bbox,
            cell_size=int(cell_size),
            params=params,
            theme=theme,
        )

    move_option_render = {}
    if move_options:
        move_option_render = _draw_move_option_panel(
            draw,
            bbox=option_panel_bbox,
            move_options=move_options,
            params=params,
            theme=theme,
        )

    if marked_coord is not None:
        marked_id = coord_to_cell_id(marked_coord)
        marked_bbox = cell_bboxes_px[str(marked_id)]
        inset = max(3.0, 0.06 * min(marked_bbox[2] - marked_bbox[0], marked_bbox[3] - marked_bbox[1]))
        marker_bbox = [
            marked_bbox[0] + inset,
            marked_bbox[1] + inset,
            marked_bbox[2] - inset,
            marked_bbox[3] - inset,
        ]
        draw.rectangle(
            marker_bbox,
            outline=tuple(int(value) for value in theme.marked_square_outline_rgb),
            width=int(params.marked_square_outline_width_px),
        )

    target_marker_record: Dict[str, Any] | None = None
    if target_coord is not None:
        target_id = coord_to_cell_id(target_coord)
        target_bbox = cell_bboxes_px[str(target_id)]
        inset = max(3.0, 0.06 * min(target_bbox[2] - target_bbox[0], target_bbox[3] - target_bbox[1]))
        marker_bbox = [
            target_bbox[0] + inset,
            target_bbox[1] + inset,
            target_bbox[2] - inset,
            target_bbox[3] - inset,
        ]
        target_marker_style = resolve_semantic_marker_style(
            instance_seed=int(params.instance_seed),
            namespace="games.chess.target_cell",
            role="target_cell_outline",
            surface_rgbs=(cell_surface_rgbs[str(target_id)],),
            preferred_rgbs=((37, 99, 235),),
            candidate_rgbs=((37, 99, 235),),
        )
        target_marker_record = draw_semantic_bbox_marker(
            draw,
            marker_bbox,
            radius=max(5, int(0.10 * min(target_bbox[2] - target_bbox[0], target_bbox[3] - target_bbox[1]))),
            style=target_marker_style,
            width=int(params.marked_square_outline_width_px),
            marker_kind="target_cell_outline",
            extra_metadata={"source": "games_chess_target_cell"},
        )

    render_map: Dict[str, Any] = {
        "board_bbox_px": list(board_bbox),
        "scene_panel_bbox_px": None if scene_panel_bbox is None else [int(value) for value in scene_panel_bbox],
        "badge_bbox_px": list(badge_bbox),
        "cell_bboxes_px": {str(key): list(value) for key, value in cell_bboxes_px.items()},
        "piece_bboxes_px": {str(key): list(value) for key, value in piece_bboxes_px.items()},
        "marked_cell_id": None if marked_coord is None else coord_to_cell_id(marked_coord),
        "target_cell_id": None if target_coord is None else coord_to_cell_id(target_coord),
        "target_marker": target_marker_record,
        "coordinate_label_bboxes_px": coordinate_label_bboxes,
        "move_option_panel": move_option_render,
        "effective_cell_size_px": float(cell_size),
        "layout_jitter": dict(layout_jitter),
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
        "font_family": str(params.font_family),
        "board_size": int(BOARD_SIZE),
    }
    return RenderedChessScene(
        image=image,
        cell_specs=tuple(cell_specs),
        scene_entities=tuple(scene_entities),
        render_map=render_map,
    )


__all__ = [
    "ChessCellSpec",
    "ChessRenderParams",
    "RenderedChessScene",
    "draw_chess_piece_symbol",
    "render_chess_board_scene",
]
