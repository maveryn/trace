"""Shared renderer for Snakes and Ladders board scenes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.drawing import draw_rounded_rect
from ...shared.text import draw_centered_game_text as draw_centered_text
from ....shared.text_rendering import fit_font_to_box
from ...shared.layout import apply_games_layout_jitter_to_bbox
from ...shared.scene_style import GamePanelSceneStyle, draw_panel_scene_chrome, game_panel_scene_style_metadata
from .rules import board_last_square, square_to_cell_id, square_to_coord
from .state import BOARD_ROWS, SnakesLaddersJump


@dataclass(frozen=True)
class SnakesLaddersRenderParams:
    """Resolved render controls for one Snakes and Ladders board."""

    canvas_width: int
    canvas_height: int
    board_side: int
    board_left_px: int
    board_top_px: int
    board_size_px: int
    side_panel_width_px: int
    cell_gap_px: int
    cell_radius_px: int
    number_font_size_px: int
    token_radius_px: int
    die_size_px: int
    jump_width_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class SnakesLaddersTheme:
    """Resolved palette for one visual style."""

    panel_fill_rgb: Tuple[int, int, int]
    panel_outline_rgb: Tuple[int, int, int]
    cell_fill_a_rgb: Tuple[int, int, int]
    cell_fill_b_rgb: Tuple[int, int, int]
    cell_outline_rgb: Tuple[int, int, int]
    number_rgb: Tuple[int, int, int]
    token_fill_rgb: Tuple[int, int, int]
    token_outline_rgb: Tuple[int, int, int]
    die_fill_rgb: Tuple[int, int, int]
    die_outline_rgb: Tuple[int, int, int]
    ladder_rgb: Tuple[int, int, int]
    ladder_outline_rgb: Tuple[int, int, int]
    snake_rgb: Tuple[int, int, int]
    snake_outline_rgb: Tuple[int, int, int]
    note_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedSnakesLaddersScene:
    """Rendered scene plus trace-friendly geometry."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


def build_snakes_ladders_theme(*, style_variant: str) -> SnakesLaddersTheme:
    """Return a palette for one Snakes and Ladders style."""

    palettes: Dict[str, SnakesLaddersTheme] = {
        "classic": SnakesLaddersTheme(
            panel_fill_rgb=(247, 238, 212),
            panel_outline_rgb=(112, 86, 55),
            cell_fill_a_rgb=(255, 247, 226),
            cell_fill_b_rgb=(239, 223, 184),
            cell_outline_rgb=(128, 101, 69),
            number_rgb=(55, 43, 31),
            token_fill_rgb=(79, 95, 204),
            token_outline_rgb=(26, 36, 112),
            die_fill_rgb=(255, 255, 248),
            die_outline_rgb=(77, 65, 54),
            ladder_rgb=(45, 135, 77),
            ladder_outline_rgb=(24, 73, 44),
            snake_rgb=(202, 76, 59),
            snake_outline_rgb=(104, 37, 30),
            note_rgb=(59, 47, 36),
        ),
        "paper": SnakesLaddersTheme(
            panel_fill_rgb=(249, 244, 232),
            panel_outline_rgb=(100, 94, 82),
            cell_fill_a_rgb=(255, 253, 244),
            cell_fill_b_rgb=(232, 237, 226),
            cell_outline_rgb=(136, 129, 116),
            number_rgb=(55, 57, 54),
            token_fill_rgb=(196, 82, 72),
            token_outline_rgb=(91, 41, 38),
            die_fill_rgb=(252, 250, 241),
            die_outline_rgb=(88, 84, 75),
            ladder_rgb=(76, 132, 94),
            ladder_outline_rgb=(42, 75, 54),
            snake_rgb=(167, 91, 77),
            snake_outline_rgb=(88, 46, 40),
            note_rgb=(72, 66, 56),
        ),
        "neon": SnakesLaddersTheme(
            panel_fill_rgb=(24, 27, 58),
            panel_outline_rgb=(98, 111, 202),
            cell_fill_a_rgb=(37, 42, 86),
            cell_fill_b_rgb=(30, 35, 73),
            cell_outline_rgb=(103, 118, 210),
            number_rgb=(239, 246, 255),
            token_fill_rgb=(255, 97, 166),
            token_outline_rgb=(255, 220, 241),
            die_fill_rgb=(246, 251, 255),
            die_outline_rgb=(51, 70, 130),
            ladder_rgb=(74, 236, 185),
            ladder_outline_rgb=(17, 99, 80),
            snake_rgb=(255, 128, 76),
            snake_outline_rgb=(110, 43, 31),
            note_rgb=(237, 242, 255),
        ),
        "pastel": SnakesLaddersTheme(
            panel_fill_rgb=(236, 242, 248),
            panel_outline_rgb=(95, 112, 132),
            cell_fill_a_rgb=(246, 250, 255),
            cell_fill_b_rgb=(224, 235, 243),
            cell_outline_rgb=(126, 142, 159),
            number_rgb=(48, 61, 76),
            token_fill_rgb=(122, 92, 195),
            token_outline_rgb=(64, 45, 112),
            die_fill_rgb=(254, 254, 255),
            die_outline_rgb=(91, 105, 121),
            ladder_rgb=(91, 159, 129),
            ladder_outline_rgb=(48, 91, 76),
            snake_rgb=(218, 112, 104),
            snake_outline_rgb=(114, 55, 52),
            note_rgb=(53, 67, 83),
        ),
        "wood": SnakesLaddersTheme(
            panel_fill_rgb=(218, 174, 112),
            panel_outline_rgb=(93, 57, 32),
            cell_fill_a_rgb=(245, 206, 143),
            cell_fill_b_rgb=(225, 181, 112),
            cell_outline_rgb=(111, 67, 37),
            number_rgb=(50, 31, 19),
            token_fill_rgb=(35, 109, 147),
            token_outline_rgb=(15, 56, 80),
            die_fill_rgb=(255, 247, 228),
            die_outline_rgb=(86, 52, 31),
            ladder_rgb=(57, 119, 68),
            ladder_outline_rgb=(31, 68, 37),
            snake_rgb=(177, 67, 52),
            snake_outline_rgb=(88, 34, 29),
            note_rgb=(57, 36, 24),
        ),
    }
    return palettes.get(str(style_variant), palettes["classic"])


def _board_bbox(params: SnakesLaddersRenderParams) -> Tuple[float, float, float, float]:
    """Return the outer board bbox."""

    meta = params.layout_jitter_meta if isinstance(params.layout_jitter_meta, Mapping) else {}
    base = (
        float(params.board_left_px),
        float(params.board_top_px),
        float(params.board_left_px + params.board_size_px),
        float(params.board_top_px + params.board_size_px),
    )
    shifted, _dx, _dy, _resolved = apply_games_layout_jitter_to_bbox(
        bbox_px=base,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=meta,
    )
    return shifted


def _square_bbox(
    *,
    board_bbox: Tuple[float, float, float, float],
    square: int,
    board_side: int,
    gap_px: float,
) -> Tuple[float, float, float, float]:
    """Return the pixel bbox for one numbered square."""

    board_left, board_top, board_right, board_bottom = [float(value) for value in board_bbox]
    board_size = min(float(board_right - board_left), float(board_bottom - board_top))
    side = int(board_side)
    cell_size = (float(board_size) - (float(gap_px) * (side + 1))) / side
    row, col = square_to_coord(int(square), board_side=side)
    left = float(board_left + gap_px + (int(col) * (cell_size + gap_px)))
    top = float(board_top + gap_px + (int(row) * (cell_size + gap_px)))
    return (
        round(left, 3),
        round(top, 3),
        round(float(left + cell_size), 3),
        round(float(top + cell_size), 3),
    )


def _center(bbox: Sequence[float]) -> Tuple[float, float]:
    """Return bbox center."""

    return (float((float(bbox[0]) + float(bbox[2])) / 2.0), float((float(bbox[1]) + float(bbox[3])) / 2.0))


def _draw_direction_arrowhead(
    draw: ImageDraw.ImageDraw,
    *,
    tip: Tuple[float, float],
    direction: Tuple[float, float],
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    size: float,
    width: int,
) -> None:
    """Draw one triangular arrowhead pointing along direction."""

    dx, dy = float(direction[0]), float(direction[1])
    length = max(1.0, math.hypot(dx, dy))
    ux, uy = float(dx / length), float(dy / length)
    px, py = float(-uy), float(ux)
    tx, ty = float(tip[0]), float(tip[1])
    back_x = float(tx - ux * size)
    back_y = float(ty - uy * size)
    half_width = float(size * 0.52)
    points = [
        (tx, ty),
        (back_x + px * half_width, back_y + py * half_width),
        (back_x - px * half_width, back_y - py * half_width),
    ]
    outline_size = max(2.0, float(width) * 0.65)
    outline_points = [
        (float(tx + ux * outline_size), float(ty + uy * outline_size)),
        (float(back_x + px * (half_width + outline_size)), float(back_y + py * (half_width + outline_size))),
        (float(back_x - px * (half_width + outline_size)), float(back_y - py * (half_width + outline_size))),
    ]
    draw.polygon(outline_points, fill=outline)
    draw.polygon(points, fill=fill)


def _draw_ladder(
    draw: ImageDraw.ImageDraw,
    *,
    start: Tuple[float, float],
    end: Tuple[float, float],
    theme: SnakesLaddersTheme,
    width: int,
) -> None:
    """Draw a simple ladder between two square centers."""

    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    dx, dy = float(ex - sx), float(ey - sy)
    length = max(1.0, math.hypot(dx, dy))
    px, py = float(-dy / length), float(dx / length)
    rail_gap = 9.0
    rail_a = [(sx + px * rail_gap, sy + py * rail_gap), (ex + px * rail_gap, ey + py * rail_gap)]
    rail_b = [(sx - px * rail_gap, sy - py * rail_gap), (ex - px * rail_gap, ey - py * rail_gap)]
    for rail in (rail_a, rail_b):
        draw.line(rail, fill=theme.ladder_outline_rgb, width=max(1, int(width) + 4))
        draw.line(rail, fill=theme.ladder_rgb, width=max(1, int(width)))
    rung_count = max(3, min(7, int(length // 70)))
    for index in range(1, rung_count):
        t = float(index / rung_count)
        cx = float(sx + dx * t)
        cy = float(sy + dy * t)
        a = (cx + px * (rail_gap + 3), cy + py * (rail_gap + 3))
        b = (cx - px * (rail_gap + 3), cy - py * (rail_gap + 3))
        draw.line([a, b], fill=theme.ladder_outline_rgb, width=max(1, int(width)))
        draw.line([a, b], fill=theme.ladder_rgb, width=max(1, int(width) - 1))
    ux, uy = float(dx / length), float(dy / length)
    _draw_direction_arrowhead(
        draw,
        tip=(float(ex - ux * max(5.0, float(width) * 0.8)), float(ey - uy * max(5.0, float(width) * 0.8))),
        direction=(ux, uy),
        fill=theme.ladder_rgb,
        outline=theme.ladder_outline_rgb,
        size=max(17.0, float(width) * 3.2),
        width=max(2, int(width)),
    )


def _draw_snake(
    draw: ImageDraw.ImageDraw,
    *,
    start: Tuple[float, float],
    end: Tuple[float, float],
    theme: SnakesLaddersTheme,
    width: int,
) -> None:
    """Draw a curved snake-like connector between two square centers."""

    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    dx, dy = float(ex - sx), float(ey - sy)
    length = max(1.0, math.hypot(dx, dy))
    px, py = float(-dy / length), float(dx / length)
    wave = min(28.0, max(12.0, length * 0.08))
    points = []
    for idx in range(18):
        t = float(idx / 17.0)
        wobble = math.sin(t * math.pi * 4.0) * wave
        points.append((sx + dx * t + px * wobble, sy + dy * t + py * wobble))
    draw.line(points, fill=theme.snake_outline_rgb, width=max(1, int(width) + 6), joint="curve")
    draw.line(points, fill=theme.snake_rgb, width=max(1, int(width) + 1), joint="curve")

    body_dx = float(points[1][0] - points[0][0])
    body_dy = float(points[1][1] - points[0][1])
    body_len = max(1.0, math.hypot(body_dx, body_dy))
    face_x = float(-body_dx / body_len)
    face_y = float(-body_dy / body_len)
    eye_px = float(-face_y)
    eye_py = float(face_x)

    head_r = max(18, int(width * 2.35) + 8)
    draw.ellipse(
        (sx - head_r, sy - head_r, sx + head_r, sy + head_r),
        fill=theme.snake_rgb,
        outline=theme.snake_outline_rgb,
        width=max(2, int(width // 2)),
    )
    eye_r = max(3.0, float(head_r) * 0.18)
    pupil_r = max(1.5, float(eye_r) * 0.48)
    eye_forward = float(head_r) * 0.18
    eye_spread = float(head_r) * 0.38
    for side in (-1.0, 1.0):
        eye_x = float(sx + face_x * eye_forward + eye_px * eye_spread * side)
        eye_y = float(sy + face_y * eye_forward + eye_py * eye_spread * side)
        draw.ellipse(
            (eye_x - eye_r, eye_y - eye_r, eye_x + eye_r, eye_y + eye_r),
            fill=(255, 255, 245),
            outline=theme.snake_outline_rgb,
            width=1,
        )
        draw.ellipse(
            (eye_x - pupil_r, eye_y - pupil_r, eye_x + pupil_r, eye_y + pupil_r),
            fill=theme.snake_outline_rgb,
        )
    tongue_base = (float(sx + face_x * head_r * 0.78), float(sy + face_y * head_r * 0.78))
    tongue_tip = (float(sx + face_x * head_r * 1.28), float(sy + face_y * head_r * 1.28))
    tongue_color = (225, 40, 83)
    draw.line([tongue_base, tongue_tip], fill=tongue_color, width=max(2, int(width // 3)))
    fork_len = float(head_r) * 0.22
    fork_a = (
        float(tongue_tip[0] + eye_px * fork_len + face_x * fork_len * 0.25),
        float(tongue_tip[1] + eye_py * fork_len + face_y * fork_len * 0.25),
    )
    fork_b = (
        float(tongue_tip[0] - eye_px * fork_len + face_x * fork_len * 0.25),
        float(tongue_tip[1] - eye_py * fork_len + face_y * fork_len * 0.25),
    )
    draw.line([tongue_tip, fork_a], fill=tongue_color, width=max(2, int(width // 3)))
    draw.line([tongue_tip, fork_b], fill=tongue_color, width=max(2, int(width // 3)))
    tail_dx = float(points[-1][0] - points[-2][0])
    tail_dy = float(points[-1][1] - points[-2][1])
    _draw_direction_arrowhead(
        draw,
        tip=(ex, ey),
        direction=(tail_dx, tail_dy),
        fill=theme.snake_rgb,
        outline=theme.snake_outline_rgb,
        size=max(16.0, float(width) * 3.0),
        width=max(2, int(width)),
    )


def _draw_die(draw: ImageDraw.ImageDraw, *, bbox: Tuple[float, float, float, float], value: int, theme: SnakesLaddersTheme) -> None:
    """Draw one die face with pips."""

    draw_rounded_rect(draw, bbox, radius=10, fill=theme.die_fill_rgb, outline=theme.die_outline_rgb, width=3)
    left, top, right, bottom = [float(v) for v in bbox]
    cx, cy = float((left + right) / 2.0), float((top + bottom) / 2.0)
    offsets = {
        1: [(0, 0)],
        2: [(-1, -1), (1, 1)],
        3: [(-1, -1), (0, 0), (1, 1)],
        4: [(-1, -1), (1, -1), (-1, 1), (1, 1)],
        5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)],
        6: [(-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)],
    }[int(value)]
    gap = float((right - left) * 0.24)
    radius = max(3.0, float((right - left) * 0.055))
    for ox, oy in offsets:
        x = float(cx + ox * gap)
        y = float(cy + oy * gap)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=theme.die_outline_rgb)


def render_snakes_ladders_board_scene(
    *,
    jumps: Sequence[SnakesLaddersJump],
    background: Image.Image,
    style_variant: str,
    params: SnakesLaddersRenderParams,
    start_square: int,
    die_value: int | None = None,
    horizon_roll_count: int | None = None,
    show_roll_panel: bool = True,
    highlight_token_square: bool = True,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedSnakesLaddersScene:
    """Render one Snakes and Ladders board scene."""

    image = background.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    theme = build_snakes_ladders_theme(style_variant=str(style_variant))
    board_side = int(params.board_side)
    last_square = board_last_square(board_side)
    board_bbox = _board_bbox(params)
    side_left = float(board_bbox[2] + 32)
    side_top = float(board_bbox[1])
    side_right = min(float(params.canvas_width - 48), float(side_left + params.side_panel_width_px))
    side_bottom = float(board_bbox[3])
    show_side_panel = bool(die_value is not None or show_roll_panel or horizon_roll_count is not None)
    if panel_style is not None:
        panel_pad = 18.0
        content_right = float(side_right) if bool(show_side_panel) else float(board_bbox[2])
        panel_bbox = (
            int(round(max(6.0, float(board_bbox[0]) - panel_pad))),
            int(round(max(6.0, float(board_bbox[1]) - panel_pad))),
            int(round(min(float(params.canvas_width) - 6.0, float(content_right) + panel_pad))),
            int(round(min(float(params.canvas_height) - 6.0, float(board_bbox[3]) + panel_pad))),
        )
        draw_panel_scene_chrome(
            draw,
            bbox=panel_bbox,
            style=panel_style,
            radius=24,
            border_width=2,
        )
    if bool(show_side_panel):
        draw_rounded_rect(
            draw,
            (side_left, side_top, side_right, side_bottom),
            radius=18,
            fill=theme.panel_fill_rgb,
            outline=theme.panel_outline_rgb,
            width=4,
        )
    draw_rounded_rect(
        draw,
        board_bbox,
        radius=20,
        fill=theme.panel_fill_rgb,
        outline=theme.panel_outline_rgb,
        width=5,
    )

    entity_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    entities: list[Dict[str, Any]] = []
    number_font_cache: Dict[int, Any] = {}
    for square in range(1, last_square + 1):
        bbox = _square_bbox(
            board_bbox=board_bbox,
            square=int(square),
            board_side=int(board_side),
            gap_px=float(params.cell_gap_px),
        )
        entity_id = square_to_cell_id(int(square))
        row, col = square_to_coord(int(square), board_side=int(board_side))
        fill = theme.cell_fill_a_rgb if (int(row) + int(col)) % 2 == 0 else theme.cell_fill_b_rgb
        draw_rounded_rect(
            draw,
            bbox,
            radius=int(params.cell_radius_px),
            fill=fill,
            outline=theme.cell_outline_rgb,
            width=1,
        )
        text = str(int(square))
        max_font = int(params.number_font_size_px)
        if max_font not in number_font_cache:
            number_font_cache[max_font] = fit_font_to_box(
                draw,
                text=str(int(last_square)),
                max_width=float(bbox[2] - bbox[0]) * 0.70,
                max_height=float(bbox[3] - bbox[1]) * 0.52,
                bold=False,
                font_family=str(params.font_family),
                min_size_px=18,
                max_size_px=max_font,
            )
        draw_centered_text(
            draw,
            text=text,
            center=(
                float(bbox[0] + (float(bbox[2] - bbox[0]) * 0.32)),
                float(bbox[1] + (float(bbox[3] - bbox[1]) * 0.26)),
            ),
            font=number_font_cache[max_font],
            fill=theme.number_rgb,
            stroke_fill=theme.number_rgb,
            stroke_width=0,
        )
        entity_bboxes[entity_id] = tuple(float(v) for v in bbox)
        entities.append(
            {
                "id": entity_id,
                "type": "board_square",
                "square": int(square),
                "row": int(row),
                "col": int(col),
                "bbox_px": [float(v) for v in bbox],
            }
        )

    centers = {square: _center(entity_bboxes[square_to_cell_id(square)]) for square in range(1, last_square + 1)}
    for jump in jumps:
        start = centers[int(jump.start_square)]
        end = centers[int(jump.end_square)]
        if str(jump.kind) == "ladder":
            _draw_ladder(draw, start=start, end=end, theme=theme, width=int(params.jump_width_px))
        else:
            _draw_snake(draw, start=start, end=end, theme=theme, width=int(params.jump_width_px))
        entities.append(
            {
                "id": str(jump.jump_id),
                "type": str(jump.kind),
                "start_square": int(jump.start_square),
                "end_square": int(jump.end_square),
                "start_xy_px": [float(start[0]), float(start[1])],
                "end_xy_px": [float(end[0]), float(end[1])],
            }
        )

    start_square_bbox = entity_bboxes[square_to_cell_id(int(start_square))]
    if bool(highlight_token_square):
        draw.rounded_rectangle(
            start_square_bbox,
            radius=int(params.cell_radius_px) + 2,
            fill=None,
            outline=theme.token_outline_rgb,
            width=max(5, int(params.jump_width_px)),
        )

    token_center = centers[int(start_square)]
    token_radius = float(params.token_radius_px)
    token_bbox = (
        float(token_center[0] - token_radius),
        float(token_center[1] - token_radius),
        float(token_center[0] + token_radius),
        float(token_center[1] + token_radius),
    )
    draw.ellipse(token_bbox, fill=theme.token_outline_rgb)
    inset = 5.0
    draw.ellipse(
        (
            float(token_bbox[0] + inset),
            float(token_bbox[1] + inset),
            float(token_bbox[2] - inset),
            float(token_bbox[3] - inset),
        ),
        fill=theme.token_fill_rgb,
    )
    entity_bboxes["token"] = tuple(float(v) for v in token_bbox)
    entities.append(
        {
            "id": "token",
            "type": "token",
            "square": int(start_square),
            "bbox_px": [float(v) for v in token_bbox],
        }
    )

    if bool(show_side_panel):
        title_font = fit_font_to_box(
            draw,
            text="ROLL",
            max_width=float(side_right - side_left - 24),
            max_height=32,
            bold=True,
            font_family=str(params.font_family),
            min_size_px=16,
            max_size_px=26,
        )
        if die_value is not None:
            draw_centered_text(
                draw,
                text="DIE",
                center=(float((side_left + side_right) / 2.0), float(side_top + 40)),
                font=title_font,
                fill=theme.note_rgb,
                stroke_fill=theme.note_rgb,
                stroke_width=0,
            )
            die_bbox = (
                float((side_left + side_right - params.die_size_px) / 2.0),
                float(side_top + 66),
                float((side_left + side_right + params.die_size_px) / 2.0),
                float(side_top + 66 + params.die_size_px),
            )
            _draw_die(draw, bbox=die_bbox, value=int(die_value), theme=theme)
            entity_bboxes["die"] = tuple(float(v) for v in die_bbox)
            entities.append({"id": "die", "type": "die", "value": int(die_value), "bbox_px": [float(v) for v in die_bbox]})
        elif bool(show_roll_panel):
            draw_centered_text(
                draw,
                text="ROLLS 1-6",
                center=(float((side_left + side_right) / 2.0), float(side_top + 44)),
                font=title_font,
                fill=theme.note_rgb,
                stroke_fill=theme.note_rgb,
                stroke_width=0,
            )
            small = max(30, int(params.die_size_px * 0.48))
            top = float(side_top + 76)
            for idx, value in enumerate((1, 2, 3, 4, 5, 6)):
                row = int(idx // 3)
                col = int(idx % 3)
                left = float(side_left + 24 + col * (small + 12))
                die_bbox = (left, top + row * (small + 12), left + small, top + row * (small + 12) + small)
                _draw_die(draw, bbox=die_bbox, value=int(value), theme=theme)

        if horizon_roll_count is not None:
            note_font = fit_font_to_box(
                draw,
                text="3 ROLLS",
                max_width=float(side_right - side_left - 24),
                max_height=34,
                bold=True,
                font_family=str(params.font_family),
                min_size_px=14,
                max_size_px=24,
            )
            draw_centered_text(
                draw,
                text=f"{int(horizon_roll_count)} ROLL{'S' if int(horizon_roll_count) != 1 else ''}",
                center=(float((side_left + side_right) / 2.0), float(side_bottom - 60)),
                font=note_font,
                fill=theme.note_rgb,
                stroke_fill=theme.note_rgb,
                stroke_width=0,
            )

    render_map = {
        "entity_bboxes_px": {str(key): [float(v) for v in bbox] for key, bbox in entity_bboxes.items()},
        "board_bbox_px": [float(v) for v in board_bbox],
        "board_side": int(board_side),
        "last_square": int(last_square),
        "square_centers_px": {str(square): [float(v) for v in centers[int(square)]] for square in centers},
        "layout_jitter": dict(params.layout_jitter_meta or {}),
        "font_family": str(params.font_family),
        "text_style": {"font_family": str(params.font_family)},
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
    }
    return RenderedSnakesLaddersScene(image=image, scene_entities=tuple(entities), render_map=render_map)


__all__ = [
    "RenderedSnakesLaddersScene",
    "SnakesLaddersRenderParams",
    "SnakesLaddersTheme",
    "build_snakes_ladders_theme",
    "render_snakes_ladders_board_scene",
]
