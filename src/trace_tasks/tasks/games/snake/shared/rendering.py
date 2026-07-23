"""Shared Snake-grid renderer for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ...shared.layout import apply_games_layout_jitter_to_bbox
from ...shared.scene_style import GamePanelSceneStyle, draw_panel_scene_chrome, game_panel_scene_style_metadata
from .rules import all_coords, coord_to_cell_id
from .state import Coord, SnakeState


@dataclass(frozen=True)
class SnakeRenderParams:
    """Resolved render controls for one Snake scene."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    max_board_size_px: int
    board_border_width_px: int
    grid_line_width_px: int
    cell_padding_px: int
    food_radius_px: int
    eye_radius_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class SnakeTheme:
    """Resolved visual theme for one Snake board."""

    board_fill_rgb: Tuple[int, int, int]
    board_border_rgb: Tuple[int, int, int]
    grid_rgb: Tuple[int, int, int]
    cell_alt_rgba: Tuple[int, int, int, int]
    head_rgb: Tuple[int, int, int]
    head_outline_rgb: Tuple[int, int, int]
    body_rgb: Tuple[int, int, int]
    body_outline_rgb: Tuple[int, int, int]
    food_rgb: Tuple[int, int, int]
    food_outline_rgb: Tuple[int, int, int]
    food_leaf_rgb: Tuple[int, int, int]
    eye_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedSnakeScene:
    """Rendered Snake image plus trace-friendly geometry."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


def build_games_snake_theme(*, style_variant: str) -> SnakeTheme:
    """Return one visual theme for Snake."""

    style = str(style_variant)
    if style == "neon":
        return SnakeTheme(
            board_fill_rgb=(229, 235, 250),
            board_border_rgb=(72, 80, 126),
            grid_rgb=(176, 187, 214),
            cell_alt_rgba=(255, 255, 255, 66),
            head_rgb=(250, 218, 73),
            head_outline_rgb=(74, 68, 38),
            body_rgb=(67, 117, 202),
            body_outline_rgb=(31, 63, 126),
            food_rgb=(255, 78, 118),
            food_outline_rgb=(112, 42, 62),
            food_leaf_rgb=(71, 160, 98),
            eye_rgb=(22, 26, 40),
        )
    if style == "forest":
        return SnakeTheme(
            board_fill_rgb=(180, 211, 137),
            board_border_rgb=(54, 91, 51),
            grid_rgb=(135, 174, 102),
            cell_alt_rgba=(255, 255, 220, 24),
            head_rgb=(238, 190, 59),
            head_outline_rgb=(61, 84, 47),
            body_rgb=(64, 139, 82),
            body_outline_rgb=(43, 88, 57),
            food_rgb=(208, 57, 54),
            food_outline_rgb=(82, 46, 38),
            food_leaf_rgb=(55, 135, 71),
            eye_rgb=(26, 44, 30),
        )
    if style == "paper":
        return SnakeTheme(
            board_fill_rgb=(238, 226, 184),
            board_border_rgb=(105, 72, 45),
            grid_rgb=(184, 142, 83),
            cell_alt_rgba=(255, 244, 186, 32),
            head_rgb=(229, 184, 66),
            head_outline_rgb=(83, 75, 58),
            body_rgb=(83, 135, 166),
            body_outline_rgb=(52, 78, 96),
            food_rgb=(203, 66, 57),
            food_outline_rgb=(90, 62, 52),
            food_leaf_rgb=(79, 139, 83),
            eye_rgb=(40, 39, 36),
        )
    if style == "candy":
        return SnakeTheme(
            board_fill_rgb=(247, 224, 233),
            board_border_rgb=(133, 79, 117),
            grid_rgb=(222, 177, 201),
            cell_alt_rgba=(255, 255, 255, 46),
            head_rgb=(255, 205, 89),
            head_outline_rgb=(134, 78, 94),
            body_rgb=(84, 191, 203),
            body_outline_rgb=(57, 114, 130),
            food_rgb=(239, 86, 105),
            food_outline_rgb=(131, 67, 82),
            food_leaf_rgb=(86, 183, 106),
            eye_rgb=(69, 44, 64),
        )
    return SnakeTheme(
        board_fill_rgb=(205, 224, 155),
        board_border_rgb=(58, 83, 57),
        grid_rgb=(158, 186, 121),
        cell_alt_rgba=(255, 255, 255, 25),
        head_rgb=(246, 196, 60),
        head_outline_rgb=(55, 72, 47),
        body_rgb=(57, 135, 207),
        body_outline_rgb=(37, 77, 118),
        food_rgb=(215, 58, 52),
        food_outline_rgb=(80, 53, 43),
        food_leaf_rgb=(59, 147, 74),
        eye_rgb=(24, 32, 25),
    )


def _cell_bbox(
    *,
    board_left: float,
    board_top: float,
    cell_size: float,
    row: int,
    col: int,
    padding_px: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Return one cell bbox."""

    left = float(board_left + (int(col) * float(cell_size)) + float(padding_px))
    top = float(board_top + (int(row) * float(cell_size)) + float(padding_px))
    right = float(board_left + ((int(col) + 1) * float(cell_size)) - float(padding_px))
    bottom = float(board_top + ((int(row) + 1) * float(cell_size)) - float(padding_px))
    return (round(left, 3), round(top, 3), round(right, 3), round(bottom, 3))


def _cell_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Return the center of one bbox."""

    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _direction_between_coords(start: Coord, end: Coord) -> Tuple[int, int]:
    """Return a row/col unit direction from adjacent `start` to `end`."""

    dr = int(end[0]) - int(start[0])
    dc = int(end[1]) - int(start[1])
    if abs(dr) + abs(dc) != 1:
        return (0, 1)
    return (dr, dc)


def _piece_bbox_from_cell(
    cell_bbox: Tuple[float, float, float, float],
    *,
    cell_size: float,
    extra_scale: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Return a smaller centered bbox for visible snake pieces."""

    inset = max(10.0, float(cell_size) * 0.24) * float(extra_scale)
    max_inset = min(float(cell_bbox[2] - cell_bbox[0]), float(cell_bbox[3] - cell_bbox[1])) * 0.34
    inset = min(float(inset), float(max_inset))
    return (
        round(float(cell_bbox[0]) + inset, 3),
        round(float(cell_bbox[1]) + inset, 3),
        round(float(cell_bbox[2]) - inset, 3),
        round(float(cell_bbox[3]) - inset, 3),
    )


def _draw_food(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[float, float, float, float],
    theme: SnakeTheme,
    radius_px: int,
) -> None:
    """Draw the visible food marker."""

    cx, cy = _cell_center(bbox)
    cell_radius = min(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])) * 0.40
    radius = min(float(radius_px), float(cell_radius))
    apple = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.ellipse(apple, fill=tuple(int(v) for v in theme.food_rgb), outline=tuple(int(v) for v in theme.food_outline_rgb), width=3)
    draw.ellipse((cx - radius * 0.35, cy - radius * 0.46, cx - radius * 0.05, cy - radius * 0.16), fill=(255, 224, 220))
    stem_top = cy - radius * 1.08
    draw.line((cx, cy - radius * 0.76, cx + radius * 0.08, stem_top), fill=tuple(int(v) for v in theme.food_outline_rgb), width=3)
    leaf = (
        cx + radius * 0.08,
        stem_top - radius * 0.10,
        cx + radius * 0.55,
        stem_top + radius * 0.22,
    )
    draw.ellipse(leaf, fill=tuple(int(v) for v in theme.food_leaf_rgb), outline=tuple(int(v) for v in theme.food_outline_rgb), width=1)


def _draw_wall_cell(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[float, float, float, float],
) -> None:
    """Draw one blocked wall cell."""

    fill = (76, 82, 94)
    outline = (30, 34, 42)
    highlight = (142, 149, 162)
    draw.rounded_rectangle(
        bbox,
        radius=max(5, int(round((float(bbox[2]) - float(bbox[0])) * 0.10))),
        fill=fill,
        outline=outline,
        width=3,
    )
    left, top, right, bottom = (float(value) for value in bbox)
    width = max(1.0, right - left)
    pad = max(4.0, width * 0.12)
    for frac in (0.22, 0.42, 0.62, 0.82):
        x0 = max(left + pad, min(right - pad, left + (width * (frac - 0.16))))
        x1 = max(left + pad, min(right - pad, left + (width * (frac + 0.16))))
        draw.line((x0, bottom - pad, x1, top + pad), fill=highlight, width=2)


def _draw_head(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[float, float, float, float],
    theme: SnakeTheme,
    eye_radius_px: int,
    facing: Tuple[int, int],
) -> None:
    """Draw the directional snake head from adjacent path geometry."""

    width = max(2, int(round((float(bbox[2]) - float(bbox[0])) * 0.055)))
    draw.rounded_rectangle(
        bbox,
        radius=max(7, int(round((float(bbox[2]) - float(bbox[0])) * 0.30))),
        fill=tuple(int(v) for v in theme.head_rgb),
        outline=tuple(int(v) for v in theme.head_outline_rgb),
        width=width,
    )
    cx, cy = _cell_center(bbox)
    piece_w = float(bbox[2]) - float(bbox[0])
    piece_h = float(bbox[3]) - float(bbox[1])
    dr, dc = int(facing[0]), int(facing[1])
    if abs(dr) + abs(dc) != 1:
        dr, dc = 0, 1
    forward_x = float(dc)
    forward_y = float(dr)
    perp_x = -float(dr)
    perp_y = float(dc)
    eye_forward = min(piece_w, piece_h) * 0.16
    eye_separation = min(piece_w, piece_h) * 0.15
    r = max(2.0, min(float(eye_radius_px), min(piece_w, piece_h) * 0.055))
    for side in (-1.0, 1.0):
        eye_x = cx + (forward_x * eye_forward) + (perp_x * eye_separation * side)
        eye_y = cy + (forward_y * eye_forward) + (perp_y * eye_separation * side)
        draw.ellipse((eye_x - r * 1.45, eye_y - r * 1.45, eye_x + r * 1.45, eye_y + r * 1.45), fill=(255, 255, 235))
        draw.ellipse((eye_x - r * 0.65, eye_y - r * 0.65, eye_x + r * 0.65, eye_y + r * 0.65), fill=tuple(int(v) for v in theme.eye_rgb))

    mouth_center_x = cx + (forward_x * min(piece_w, piece_h) * 0.34)
    mouth_center_y = cy + (forward_y * min(piece_w, piece_h) * 0.34)
    mouth_span = min(piece_w, piece_h) * 0.12
    draw.line(
        (
            mouth_center_x - (perp_x * mouth_span),
            mouth_center_y - (perp_y * mouth_span),
            mouth_center_x + (perp_x * mouth_span),
            mouth_center_y + (perp_y * mouth_span),
        ),
        fill=tuple(int(v) for v in theme.eye_rgb),
        width=max(1, width - 1),
    )


def _draw_body_segment(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[float, float, float, float],
    theme: SnakeTheme,
) -> None:
    """Draw one ordinary snake body segment."""

    piece_w = float(bbox[2]) - float(bbox[0])
    draw.rounded_rectangle(
        bbox,
        radius=max(7, int(round(piece_w * 0.28))),
        fill=tuple(int(v) for v in theme.body_rgb),
        outline=tuple(int(v) for v in theme.body_outline_rgb),
        width=max(2, int(round(piece_w * 0.045))),
    )


def _draw_tail_segment(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[float, float, float, float],
    theme: SnakeTheme,
    tail_direction: Tuple[int, int],
) -> None:
    """Draw the terminal snake segment with a visible tapered tail."""

    cx, cy = _cell_center(bbox)
    piece_w = float(bbox[2]) - float(bbox[0])
    piece_h = float(bbox[3]) - float(bbox[1])
    dr, dc = int(tail_direction[0]), int(tail_direction[1])
    if abs(dr) + abs(dc) != 1:
        dr, dc = 0, -1
    forward_x = float(dc)
    forward_y = float(dr)
    perp_x = -float(dr)
    perp_y = float(dc)
    tip = (
        cx + (forward_x * piece_w * 0.47),
        cy + (forward_y * piece_h * 0.47),
    )
    base_center = (
        cx - (forward_x * piece_w * 0.14),
        cy - (forward_y * piece_h * 0.14),
    )
    half_span = min(piece_w, piece_h) * 0.25
    polygon = (
        tip,
        (
            base_center[0] + (perp_x * half_span),
            base_center[1] + (perp_y * half_span),
        ),
        (
            base_center[0] - (perp_x * half_span),
            base_center[1] - (perp_y * half_span),
        ),
    )
    draw.polygon(polygon, fill=tuple(int(v) for v in theme.body_rgb), outline=tuple(int(v) for v in theme.body_outline_rgb))
    cap_radius = min(piece_w, piece_h) * 0.30
    cap = (
        cx - cap_radius - (forward_x * piece_w * 0.08),
        cy - cap_radius - (forward_y * piece_h * 0.08),
        cx + cap_radius - (forward_x * piece_w * 0.08),
        cy + cap_radius - (forward_y * piece_h * 0.08),
    )
    draw.ellipse(cap, fill=tuple(int(v) for v in theme.body_rgb), outline=tuple(int(v) for v in theme.body_outline_rgb), width=max(2, int(round(piece_w * 0.04))))


def _draw_segment_connectors(
    draw: ImageDraw.ImageDraw,
    *,
    path: Sequence[Coord],
    piece_bboxes_px: Mapping[str, Sequence[float]],
    theme: SnakeTheme,
    cell_size: float,
) -> None:
    """Draw subtle connectors so smaller pieces still read as one snake."""

    width = max(5, int(round(float(cell_size) * 0.14)))
    outline_width = max(width + 4, int(round(float(cell_size) * 0.19)))
    for start, end in zip(tuple(path), tuple(path)[1:]):
        start_bbox = tuple(float(v) for v in piece_bboxes_px[coord_to_cell_id(start)])
        end_bbox = tuple(float(v) for v in piece_bboxes_px[coord_to_cell_id(end)])
        start_center = _cell_center(start_bbox)
        end_center = _cell_center(end_bbox)
        draw.line((*start_center, *end_center), fill=tuple(int(v) for v in theme.body_outline_rgb), width=outline_width)
        draw.line((*start_center, *end_center), fill=tuple(int(v) for v in theme.body_rgb), width=width)


def render_snake_grid_scene(
    *,
    state: SnakeState,
    background: Image.Image,
    style_variant: str,
    params: SnakeRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedSnakeScene:
    """Render one metadata-bound Snake board shared by every objective."""

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme = build_games_snake_theme(style_variant=str(style_variant))

    board_size = int(state.board_size)
    board_size_px = min(
        int(params.max_board_size_px),
        int(params.canvas_width) - (2 * int(params.panel_margin_px)),
        int(params.canvas_height) - (2 * int(params.panel_margin_px)),
    )
    board_left = int(0.5 * (int(params.canvas_width) - int(board_size_px)))
    board_top = int(0.5 * (int(params.canvas_height) - int(board_size_px)))
    board_bbox = (
        round(float(board_left), 3),
        round(float(board_top), 3),
        round(float(board_left + board_size_px), 3),
        round(float(board_top + board_size_px), 3),
    )
    board_bbox, _dx, _dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=board_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    board_left = float(board_bbox[0])
    board_top = float(board_bbox[1])
    cell_size = float((float(board_bbox[2]) - float(board_bbox[0])) / float(board_size))

    if panel_style is not None:
        panel_pad = 22.0
        panel_bbox = (
            int(round(max(6.0, float(board_bbox[0]) - panel_pad))),
            int(round(max(6.0, float(board_bbox[1]) - panel_pad))),
            int(round(min(float(params.canvas_width) - 6.0, float(board_bbox[2]) + panel_pad))),
            int(round(min(float(params.canvas_height) - 6.0, float(board_bbox[3]) + panel_pad))),
        )
        draw_panel_scene_chrome(
            draw,
            bbox=panel_bbox,
            style=panel_style,
            radius=24,
            border_width=2,
        )

    draw.rounded_rectangle(
        board_bbox,
        radius=16,
        fill=tuple(int(v) for v in theme.board_fill_rgb),
        outline=tuple(int(v) for v in theme.board_border_rgb),
        width=int(params.board_border_width_px),
    )

    cell_bboxes_px: Dict[str, List[float]] = {}
    piece_bboxes_px: Dict[str, List[float]] = {}
    scene_entities: List[Dict[str, Any]] = []
    snake_body = {(int(row), int(col)) for row, col in state.body}
    head = (int(state.head[0]), int(state.head[1]))
    food = (int(state.food[0]), int(state.food[1]))
    obstacles = {(int(row), int(col)) for row, col in state.obstacles}

    for row, col in all_coords(board_size):
        full_bbox = _cell_bbox(board_left=board_left, board_top=board_top, cell_size=cell_size, row=row, col=col)
        annotation_bbox = _cell_bbox(
            board_left=board_left,
            board_top=board_top,
            cell_size=cell_size,
            row=row,
            col=col,
            padding_px=max(float(params.cell_padding_px), float(cell_size) * 0.10),
        )
        piece_bbox = _piece_bbox_from_cell(full_bbox, cell_size=cell_size)
        if (int(row) + int(col)) % 2:
            draw.rectangle(full_bbox, fill=tuple(int(v) for v in theme.cell_alt_rgba))
        cell_id = coord_to_cell_id((row, col))
        cell_bboxes_px[cell_id] = list(annotation_bbox)
        piece_bboxes_px[cell_id] = list(piece_bbox)
        occupancy = "empty"
        if (row, col) == head:
            occupancy = "head"
        elif (row, col) in snake_body:
            occupancy = "body"
        elif (row, col) == food:
            occupancy = "food"
        elif (row, col) in obstacles:
            occupancy = "wall"
        scene_entities.append(
            {
                "entity_id": str(cell_id),
                "entity_type": "snake_cell",
                "row": int(row),
                "col": int(col),
                "occupancy": str(occupancy),
                "bbox_px": list(annotation_bbox),
            }
        )

    for row, col in sorted(obstacles):
        _draw_wall_cell(draw, bbox=tuple(cell_bboxes_px[coord_to_cell_id((row, col))]))

    # Body first, then food, then head so the important game pieces remain legible.
    snake_path = (head,) + tuple((int(row), int(col)) for row, col in state.body)
    _draw_segment_connectors(
        draw,
        path=snake_path,
        piece_bboxes_px=piece_bboxes_px,
        theme=theme,
        cell_size=cell_size,
    )
    tail = tuple(state.body[-1]) if state.body else None
    for row, col in reversed(tuple(state.body)):
        coord = (int(row), int(col))
        bbox = tuple(piece_bboxes_px[coord_to_cell_id(coord)])
        if tail is not None and coord == tail:
            neighbor = tuple(state.body[-2]) if len(tuple(state.body)) >= 2 else head
            _draw_tail_segment(
                draw,
                bbox=bbox,
                theme=theme,
                tail_direction=_direction_between_coords(neighbor, coord),
            )
        else:
            _draw_body_segment(
                draw,
                bbox=bbox,
                theme=theme,
            )

    _draw_food(
        draw,
        bbox=tuple(
            _piece_bbox_from_cell(
                tuple(cell_bboxes_px[coord_to_cell_id(food)]),
                cell_size=cell_size,
                extra_scale=0.35,
            )
        ),
        theme=theme,
        radius_px=int(params.food_radius_px),
    )
    facing = _direction_between_coords(tuple(state.body[0]), head) if state.body else (0, 1)
    _draw_head(
        draw,
        bbox=tuple(piece_bboxes_px[coord_to_cell_id(head)]),
        theme=theme,
        eye_radius_px=int(params.eye_radius_px),
        facing=facing,
    )

    # Grid lines last, lightly, to keep empty-cell annotation boundaries clear.
    for index in range(board_size + 1):
        x = float(board_left + (index * cell_size))
        y = float(board_top + (index * cell_size))
        draw.line((x, board_top, x, float(board_bbox[3])), fill=tuple(int(v) for v in theme.grid_rgb), width=int(params.grid_line_width_px))
        draw.line((board_left, y, float(board_bbox[2]), y), fill=tuple(int(v) for v in theme.grid_rgb), width=int(params.grid_line_width_px))
    draw.rounded_rectangle(
        board_bbox,
        radius=16,
        outline=tuple(int(v) for v in theme.board_border_rgb),
        width=int(params.board_border_width_px),
    )

    render_map = {
        "board_bbox_px": list(board_bbox),
        "cell_bboxes_px": dict(cell_bboxes_px),
        "piece_bboxes_px": dict(piece_bboxes_px),
        "layout_jitter": dict(layout_jitter),
        "head_cell_id": coord_to_cell_id(head),
        "body_cell_ids": [coord_to_cell_id(coord) for coord in state.body],
        "food_cell_id": coord_to_cell_id(food),
        "wall_cell_ids": [coord_to_cell_id(coord) for coord in sorted(obstacles)],
        "font_family": str(params.font_family),
        "text_style": {"font_family": str(params.font_family)},
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
    }
    return RenderedSnakeScene(
        image=image.convert("RGB"),
        scene_entities=tuple(scene_entities),
        render_map=render_map,
    )


__all__ = [
    "RenderedSnakeScene",
    "SnakeRenderParams",
    "SnakeTheme",
    "build_games_snake_theme",
    "render_snake_grid_scene",
]
