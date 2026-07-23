"""Shared Reversi board renderer for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.marker_legibility import draw_semantic_bbox_marker, resolve_semantic_marker_style
from ....shared.text_rendering import load_font, resolve_text_stroke_fill
from ...shared.layout import apply_games_layout_jitter_to_bbox, offset_bbox
from ...shared.text import draw_game_text_traced as draw_text_traced
from .rules import coord_to_cell_id, player_name
from .state import BLACK, WHITE, Coord
from ...shared.scene_style import GamePanelSceneStyle, draw_panel_scene_chrome, game_panel_scene_style_metadata
from ...shared.style import ReversiTheme, build_games_reversi_theme


@dataclass(frozen=True)
class ReversiRenderParams:
    """Resolved render controls for one Reversi board scene."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    player_badge_height_px: int
    player_badge_width_px: int
    header_gap_px: int
    max_board_size_px: int
    board_corner_radius_px: int
    board_frame_width_px: int
    cell_line_width_px: int
    marked_square_outline_width_px: int
    disc_inset_fraction: float
    player_badge_font_size_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None
    instance_seed: int = 0


@dataclass(frozen=True)
class ReversiCellSpec:
    """One board cell after layout/render assignment."""

    cell_id: str
    row: int
    col: int
    occupant: str
    bbox_px: Tuple[float, float, float, float]
    disc_bbox_px: Tuple[float, float, float, float] | None


@dataclass(frozen=True)
class RenderedReversiScene:
    """Rendered Reversi scene plus trace-friendly metadata."""

    image: Image.Image
    cell_specs: Tuple[ReversiCellSpec, ...]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


def _disc_bbox(cell_bbox: Tuple[float, float, float, float], *, inset_fraction: float) -> Tuple[float, float, float, float]:
    """Return one inscribed disc bbox inside a board cell."""

    left, top, right, bottom = cell_bbox
    inset = float(max(5.0, inset_fraction * min(right - left, bottom - top)))
    return (
        round(float(left + inset), 3),
        round(float(top + inset), 3),
        round(float(right - inset), 3),
        round(float(bottom - inset), 3),
    )


def _draw_disc(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Tuple[float, float, float, float],
    theme: ReversiTheme,
    player: int,
) -> None:
    """Draw one Reversi disc with simple highlight chrome."""

    if int(player) == int(BLACK):
        fill_rgb = theme.black_disc_fill_rgb
        outline_rgb = theme.black_disc_outline_rgb
        shine_rgb = theme.black_disc_shine_rgb
    else:
        fill_rgb = theme.white_disc_fill_rgb
        outline_rgb = theme.white_disc_outline_rgb
        shine_rgb = theme.white_disc_shine_rgb
    draw.ellipse(
        bbox_px,
        fill=tuple(int(value) for value in fill_rgb),
        outline=tuple(int(value) for value in outline_rgb),
        width=int(theme.disc_outline_width_px),
    )
    left, top, right, bottom = bbox_px
    shine_w = 0.34 * (right - left)
    shine_h = 0.24 * (bottom - top)
    draw.ellipse(
        [
            left + 0.18 * (right - left),
            top + 0.16 * (bottom - top),
            left + 0.18 * (right - left) + shine_w,
            top + 0.16 * (bottom - top) + shine_h,
        ],
        fill=tuple(int(value) for value in shine_rgb),
    )


def render_reversi_board_scene(
    *,
    board: Sequence[Sequence[int]],
    background: Image.Image,
    scene_variant: str,
    style_variant: str,
    current_player: int,
    params: ReversiRenderParams,
    marked_move: Coord | None,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedReversiScene:
    """Render one visible Reversi board with an optional marked legal move."""

    board_size = int(len(board))
    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme = build_games_reversi_theme(style_variant=str(style_variant))

    cell_size = min(
        int(params.max_board_size_px) // int(board_size),
        (int(params.canvas_width) - (2 * int(params.panel_margin_px))) // int(board_size),
        (
            int(params.canvas_height)
            - (2 * int(params.panel_margin_px))
            - int(params.player_badge_height_px)
            - int(params.header_gap_px)
        )
        // int(board_size),
    )
    board_width = int(cell_size) * int(board_size)
    board_height = int(cell_size) * int(board_size)
    board_left = int(0.5 * (int(params.canvas_width) - int(board_width)))
    available_height = (
        int(params.canvas_height)
        - (2 * int(params.panel_margin_px))
        - int(params.player_badge_height_px)
        - int(params.header_gap_px)
    )
    board_top = int(
        params.panel_margin_px
        + params.player_badge_height_px
        + params.header_gap_px
        + max(0, 0.5 * (available_height - int(board_height)))
    )
    board_bbox = (
        round(float(board_left), 3),
        round(float(board_top), 3),
        round(float(board_left + board_width), 3),
        round(float(board_top + board_height), 3),
    )

    badge_font = load_font(int(params.player_badge_font_size_px), bold=True, font_family=str(params.font_family))
    badge_text = f"{player_name(int(current_player))} to move"
    badge_text_bbox = draw.textbbox((0, 0), badge_text, font=badge_font, stroke_width=1)
    badge_width = max(
        int(params.player_badge_width_px),
        int((badge_text_bbox[2] - badge_text_bbox[0]) + params.player_badge_height_px + 34),
    )
    badge_left = int(0.5 * (int(params.canvas_width) - int(badge_width)))
    badge_top = int(params.panel_margin_px)
    badge_bbox = (
        round(float(badge_left), 3),
        round(float(badge_top), 3),
        round(float(badge_left + badge_width), 3),
        round(float(badge_top + params.player_badge_height_px), 3),
    )
    group_bbox = (
        min(float(board_bbox[0]), float(badge_bbox[0])),
        min(float(board_bbox[1]), float(badge_bbox[1])),
        max(float(board_bbox[2]), float(badge_bbox[2])),
        max(float(board_bbox[3]), float(badge_bbox[3])),
    )
    group_bbox, dx, dy, layout_jitter = apply_games_layout_jitter_to_bbox(
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

    if panel_style is not None:
        panel_pad = 24.0
        panel_bbox = (
            int(round(max(6.0, float(group_bbox[0]) - panel_pad))),
            int(round(max(6.0, float(group_bbox[1]) - panel_pad))),
            int(round(min(float(params.canvas_width) - 6.0, float(group_bbox[2]) + panel_pad))),
            int(round(min(float(params.canvas_height) - 6.0, float(group_bbox[3]) + panel_pad))),
        )
        draw_panel_scene_chrome(
            draw,
            bbox=panel_bbox,
            style=panel_style,
            radius=max(18, int(params.board_corner_radius_px)),
            border_width=2,
        )

    draw.rounded_rectangle(
        board_bbox,
        radius=int(params.board_corner_radius_px),
        fill=tuple(int(value) for value in theme.board_frame_rgb),
    )
    inner_inset = float(params.board_frame_width_px)
    inner_bbox = (
        round(float(board_bbox[0] + inner_inset), 3),
        round(float(board_bbox[1] + inner_inset), 3),
        round(float(board_bbox[2] - inner_inset), 3),
        round(float(board_bbox[3] - inner_inset), 3),
    )
    draw.rounded_rectangle(
        inner_bbox,
        radius=max(8, int(params.board_corner_radius_px) - int(params.board_frame_width_px)),
        fill=tuple(int(value) for value in theme.board_fill_rgb),
    )
    draw.rounded_rectangle(
        badge_bbox,
        radius=int(0.5 * int(params.player_badge_height_px)),
        fill=tuple(int(value) for value in theme.badge_fill_rgb),
        outline=tuple(int(value) for value in theme.badge_outline_rgb),
        width=2,
    )
    disc_d = int(params.player_badge_height_px) - 16
    disc_left = int(badge_left + 12)
    disc_top = int(badge_top + 8)
    _draw_disc(
        draw,
        bbox_px=(float(disc_left), float(disc_top), float(disc_left + disc_d), float(disc_top + disc_d)),
        theme=theme,
        player=int(current_player),
    )
    badge_text_rgb = tuple(int(value) for value in theme.badge_text_rgb)
    draw_text_traced(
        draw,
        (
            float(disc_left + disc_d + 12),
            float(badge_top + 0.5 * (int(params.player_badge_height_px) - (badge_text_bbox[3] - badge_text_bbox[1]))),
        ),
        badge_text,
        font=badge_font,
        fill=badge_text_rgb,
        stroke_width=1,
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(badge_text_rgb)),
        role="readout",
        required=False,
    )
    cell_specs: List[ReversiCellSpec] = []
    scene_entities: List[Dict[str, Any]] = []
    cell_bboxes_px: Dict[str, Tuple[float, float, float, float]] = {}
    disc_bboxes_px: Dict[str, Tuple[float, float, float, float]] = {}
    disc_points_px: Dict[str, Tuple[float, float]] = {}

    for index in range(1, board_size):
        x = float(board_left + (index * cell_size))
        y = float(board_top + (index * cell_size))
        draw.line(
            [(x, float(board_top)), (x, float(board_top + board_height))],
            fill=tuple(int(value) for value in theme.grid_line_rgb),
            width=int(params.cell_line_width_px),
        )
        draw.line(
            [(float(board_left), y), (float(board_left + board_width), y)],
            fill=tuple(int(value) for value in theme.grid_line_rgb),
            width=int(params.cell_line_width_px),
        )

    marked_square_bbox_px: Tuple[float, float, float, float] | None = None
    for row in range(board_size):
        for col in range(board_size):
            cell_id = coord_to_cell_id((int(row), int(col)))
            cell_bbox = (
                round(float(board_left + (col * cell_size)), 3),
                round(float(board_top + (row * cell_size)), 3),
                round(float(board_left + ((col + 1) * cell_size)), 3),
                round(float(board_top + ((row + 1) * cell_size)), 3),
            )
            if marked_move is not None and (int(row), int(col)) == (int(marked_move[0]), int(marked_move[1])):
                marked_square_bbox_px = cell_bbox
                inset = 6.0
                marker_bbox = [
                    cell_bbox[0] + inset,
                    cell_bbox[1] + inset,
                    cell_bbox[2] - inset,
                    cell_bbox[3] - inset,
                ]
                marker_style = resolve_semantic_marker_style(
                    instance_seed=int(params.instance_seed),
                    namespace=f"games.reversi.marked_square.{row}.{col}",
                    role="marked_square_outline",
                    surface_rgbs=(theme.board_fill_rgb,),
                    preferred_rgbs=(theme.marked_square_outline_rgb,),
                    candidate_rgbs=(theme.marked_square_outline_rgb,),
                )
                draw_semantic_bbox_marker(
                    draw,
                    marker_bbox,
                    radius=max(8, int(0.18 * cell_size)),
                    style=marker_style,
                    width=int(params.marked_square_outline_width_px),
                    marker_kind="cell_outline",
                    extra_metadata={"source": "games_reversi_marked_square", "cell_id": str(cell_id)},
                )
            occupant_value = int(board[row][col])
            occupant_name = "empty" if int(occupant_value) == 0 else "black" if int(occupant_value) == int(BLACK) else "white"
            disc_bbox_px = None
            if int(occupant_value) in {int(BLACK), int(WHITE)}:
                disc_bbox_px = _disc_bbox(cell_bbox, inset_fraction=float(params.disc_inset_fraction))
                disc_bboxes_px[str(cell_id)] = disc_bbox_px
                disc_points_px[str(cell_id)] = (
                    round(float((disc_bbox_px[0] + disc_bbox_px[2]) / 2.0), 3),
                    round(float((disc_bbox_px[1] + disc_bbox_px[3]) / 2.0), 3),
                )
                _draw_disc(draw, bbox_px=disc_bbox_px, theme=theme, player=int(occupant_value))
            cell_specs.append(
                ReversiCellSpec(
                    cell_id=str(cell_id),
                    row=int(row),
                    col=int(col),
                    occupant=str(occupant_name),
                    bbox_px=cell_bbox,
                    disc_bbox_px=disc_bbox_px,
                )
            )
            cell_bboxes_px[str(cell_id)] = cell_bbox
            entity: Dict[str, Any] = {
                "entity_id": str(cell_id),
                "entity_type": "board_cell",
                "row": int(row),
                "col": int(col),
                "occupant": str(occupant_name),
                "bbox": list(cell_bbox),
            }
            if disc_bbox_px is not None:
                entity["disc_bbox"] = list(disc_bbox_px)
            scene_entities.append(entity)

    return RenderedReversiScene(
        image=image,
        cell_specs=tuple(cell_specs),
        scene_entities=tuple(scene_entities),
        render_map={
            "board_bbox_px": list(board_bbox),
            "player_badge_bbox_px": list(badge_bbox),
            "cell_bboxes_px": {str(key): list(value) for key, value in cell_bboxes_px.items()},
            "disc_bboxes_px": {str(key): list(value) for key, value in disc_bboxes_px.items()},
            "disc_points_px": {str(key): list(value) for key, value in disc_points_px.items()},
            "marked_square_bbox_px": None if marked_square_bbox_px is None else list(marked_square_bbox_px),
            "board_size": int(board_size),
            "scene_variant": str(scene_variant),
            "style_variant": str(style_variant),
            "layout_jitter": dict(layout_jitter),
            "font_family": str(params.font_family),
            "text_style": {"font_family": str(params.font_family)},
            "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
        },
    )


__all__ = [
    "RenderedReversiScene",
    "ReversiCellSpec",
    "ReversiRenderParams",
    "render_reversi_board_scene",
]
