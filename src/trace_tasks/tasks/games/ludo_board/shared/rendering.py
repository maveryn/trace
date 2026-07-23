"""Rendering helpers for Ludo board scene tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.games.shared.layout import apply_games_layout_jitter_to_bbox, resolve_games_layout_jitter
from trace_tasks.tasks.games.shared.scene_style import (
    draw_panel_option_card,
    draw_panel_scene_chrome,
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from trace_tasks.tasks.games.shared.text import draw_centered_game_text_traced
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import font_role_trace, sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font

from .rules import color_rgb, soft_color
from .state import (
    BBox,
    DEFAULTS,
    FLOW_ARROW_SPECS,
    HOME_LANES,
    MAIN_PATH,
    PLAYER_COLORS,
    START_COORDS,
    YARD_BBOX_CELLS,
    Coord,
    LudoDestinationOption,
    LudoRenderState,
    LudoRollOption,
    LudoSceneAxes,
    STYLE_VARIANTS,
)


@dataclass(frozen=True)
class LudoTheme:
    """Scene-local board theme while preserving semantic player colors."""

    board_fill_rgb: Tuple[int, int, int]
    board_border_rgb: Tuple[int, int, int]
    track_fill_rgb: Tuple[int, int, int]
    track_outline_rgb: Tuple[int, int, int]
    yard_inner_rgb: Tuple[int, int, int]
    token_outline_rgb: Tuple[int, int, int]
    token_highlight_rgb: Tuple[int, int, int]
    flow_arrow_rgb: Tuple[int, int, int]
    option_text_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedLudoScene:
    """Rendered Ludo board plus trace-friendly maps."""

    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]


def make_ludo_render_state(
    *,
    style_variant: str,
    token_coords: Mapping[str, Coord],
    query_color: str,
    target_color: str | None = None,
    roll_sequence: Sequence[int] = (),
    roll_options: Sequence[LudoRollOption] = (),
    destination_options: Sequence[LudoDestinationOption] = (),
) -> LudoRenderState:
    """Normalize task-owned Ludo tokens, rolls, and options into renderer input."""

    return LudoRenderState(
        style_variant=str(style_variant),
        token_coords={str(color): tuple(coord) for color, coord in dict(token_coords).items()},
        query_color=str(query_color),
        target_color=None if target_color is None else str(target_color),
        roll_sequence=tuple(int(value) for value in roll_sequence),
        roll_options=tuple(roll_options),
        destination_options=tuple(destination_options),
    )


def theme_for_style(style_variant: str) -> Tuple[LudoTheme, Dict[str, Any]]:
    """Resolve one of the scene-local board skins used by the renderer."""

    themes: dict[str, LudoTheme] = {
        "classic_bright": LudoTheme(
            board_fill_rgb=(242, 238, 220),
            board_border_rgb=(54, 59, 70),
            track_fill_rgb=(252, 250, 238),
            track_outline_rgb=(55, 61, 74),
            yard_inner_rgb=(255, 255, 246),
            token_outline_rgb=(35, 39, 50),
            token_highlight_rgb=(255, 255, 255),
            flow_arrow_rgb=(33, 39, 54),
            option_text_rgb=(31, 36, 48),
        ),
        "ivory_board": LudoTheme(
            board_fill_rgb=(235, 225, 199),
            board_border_rgb=(92, 77, 55),
            track_fill_rgb=(255, 249, 226),
            track_outline_rgb=(102, 86, 62),
            yard_inner_rgb=(255, 251, 234),
            token_outline_rgb=(70, 57, 42),
            token_highlight_rgb=(255, 255, 244),
            flow_arrow_rgb=(78, 59, 37),
            option_text_rgb=(61, 50, 38),
        ),
        "slate_table": LudoTheme(
            board_fill_rgb=(55, 65, 78),
            board_border_rgb=(220, 228, 236),
            track_fill_rgb=(229, 234, 238),
            track_outline_rgb=(39, 48, 60),
            yard_inner_rgb=(245, 248, 250),
            token_outline_rgb=(17, 24, 33),
            token_highlight_rgb=(255, 255, 255),
            flow_arrow_rgb=(18, 26, 38),
            option_text_rgb=(24, 31, 42),
        ),
        "soft_plastic": LudoTheme(
            board_fill_rgb=(224, 236, 232),
            board_border_rgb=(66, 96, 99),
            track_fill_rgb=(252, 253, 246),
            track_outline_rgb=(78, 111, 112),
            yard_inner_rgb=(255, 255, 249),
            token_outline_rgb=(42, 61, 65),
            token_highlight_rgb=(255, 255, 255),
            flow_arrow_rgb=(35, 72, 78),
            option_text_rgb=(35, 55, 58),
        ),
        "arcade_gloss": LudoTheme(
            board_fill_rgb=(38, 39, 73),
            board_border_rgb=(255, 221, 78),
            track_fill_rgb=(244, 247, 255),
            track_outline_rgb=(34, 38, 72),
            yard_inner_rgb=(255, 255, 250),
            token_outline_rgb=(12, 16, 35),
            token_highlight_rgb=(255, 255, 255),
            flow_arrow_rgb=(28, 34, 82),
            option_text_rgb=(23, 27, 54),
        ),
    }
    resolved = str(style_variant) if str(style_variant) in themes else "classic_bright"
    return themes[resolved], {
        "style_variant": str(resolved),
        "available_styles": list(STYLE_VARIANTS),
        "board_style_policy": "semantic_ludo_colors_with_scene_local_board_theme",
    }


def _cell_bbox(board_bbox: Sequence[float], cell_size: float, coord: Coord) -> BBox:
    row, col = int(coord[0]), int(coord[1])
    x0 = float(board_bbox[0]) + (float(col) * float(cell_size))
    y0 = float(board_bbox[1]) + (float(row) * float(cell_size))
    return (round(x0, 3), round(y0, 3), round(x0 + float(cell_size), 3), round(y0 + float(cell_size), 3))


def _bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    return (0.5 * (float(bbox[0]) + float(bbox[2])), 0.5 * (float(bbox[1]) + float(bbox[3])))


def _draw_cell(draw: ImageDraw.ImageDraw, bbox: Sequence[float], *, fill: Sequence[int], outline: Sequence[int], width: int) -> None:
    draw.rectangle(tuple(float(v) for v in bbox), fill=tuple(int(v) for v in fill[:3]) + (255,), outline=tuple(int(v) for v in outline[:3]) + (255,), width=max(1, int(width)))


def _draw_token(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    color: str,
    theme: LudoTheme,
    width: int,
) -> BBox:
    cx, cy = _bbox_center(bbox)
    radius = 0.5 * min(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])) * 0.74
    token_bbox = (
        round(float(cx) - radius, 3),
        round(float(cy) - radius, 3),
        round(float(cx) + radius, 3),
        round(float(cy) + radius, 3),
    )
    draw.ellipse(
        token_bbox,
        fill=color_rgb(str(color)) + (255,),
        outline=tuple(theme.token_outline_rgb) + (255,),
        width=max(2, int(width)),
    )
    shine_radius = max(3.0, radius * 0.26)
    draw.ellipse(
        (
            float(cx) - radius * 0.42,
            float(cy) - radius * 0.48,
            float(cx) - radius * 0.42 + shine_radius,
            float(cy) - radius * 0.48 + shine_radius,
        ),
        fill=tuple(theme.token_highlight_rgb) + (95,),
    )
    return token_bbox


def _draw_flow_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    start_cell_bbox: Sequence[float],
    end_cell_bbox: Sequence[float],
    role: str,
    fill_rgb: Sequence[int],
    outline_rgb: Sequence[int],
    width: int,
) -> Dict[str, Any]:
    """Draw one two-cell flow arrow and return its visible marker geometry."""

    start_center = _bbox_center(start_cell_bbox)
    end_center = _bbox_center(end_cell_bbox)
    dx = float(end_center[0]) - float(start_center[0])
    dy = float(end_center[1]) - float(start_center[1])
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    unit = min(
        float(start_cell_bbox[2]) - float(start_cell_bbox[0]),
        float(start_cell_bbox[3]) - float(start_cell_bbox[1]),
        float(end_cell_bbox[2]) - float(end_cell_bbox[0]),
        float(end_cell_bbox[3]) - float(end_cell_bbox[1]),
    )
    extension = 0.18 * unit
    head_len = 0.21 * unit
    head_half_width = 0.16 * unit
    start = (float(start_center[0]) - ux * extension, float(start_center[1]) - uy * extension)
    end = (float(end_center[0]) + ux * extension, float(end_center[1]) + uy * extension)
    head_base = (end[0] - ux * head_len, end[1] - uy * head_len)
    head_points = (
        end,
        (head_base[0] + px * head_half_width, head_base[1] + py * head_half_width),
        (head_base[0] - px * head_half_width, head_base[1] - py * head_half_width),
    )
    outline_rgba = tuple(int(v) for v in outline_rgb[:3]) + (235,)
    fill_rgba = tuple(int(v) for v in fill_rgb[:3]) + (225,)
    line_width = max(3, int(width) + 2)
    draw.line((start, end), fill=outline_rgba, width=line_width + 2)
    draw.line((start, end), fill=fill_rgba, width=line_width)
    draw.polygon(head_points, fill=fill_rgba, outline=outline_rgba)
    points = (start, end) + head_points
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return {
        "start_center_px": [round(float(start_center[0]), 3), round(float(start_center[1]), 3)],
        "end_center_px": [round(float(end_center[0]), 3), round(float(end_center[1]), 3)],
        "bbox_px": [round(min(xs), 3), round(min(ys), 3), round(max(xs), 3), round(max(ys), 3)],
        "points_px": [[round(float(x), 3), round(float(y), 3)] for x, y in points],
        "role": str(role),
    }


def _draw_ludo_board(
    draw: ImageDraw.ImageDraw,
    *,
    board_bbox: Sequence[float],
    cell_size: float,
    theme: LudoTheme,
    grid_width: int,
    flow_arrow_enabled: bool,
    flow_arrow_width: int,
) -> Dict[str, Any]:
    """Draw the canonical Ludo cross board and return board-level markers."""

    main_set = set(MAIN_PATH)
    lane_cells = {coord: color for color, lane in HOME_LANES.items() for coord in lane}
    board_bg_bbox = tuple(float(value) for value in board_bbox)
    draw.rounded_rectangle(
        board_bg_bbox,
        radius=max(12, int(round(float(cell_size) * 0.35))),
        fill=tuple(theme.board_fill_rgb) + (255,),
        outline=tuple(theme.board_border_rgb) + (255,),
        width=max(2, int(grid_width) + 1),
    )
    for color, cells in YARD_BBOX_CELLS.items():
        r0, c0, r1, c1 = cells
        yard_bbox = (
            float(board_bbox[0]) + c0 * float(cell_size),
            float(board_bbox[1]) + r0 * float(cell_size),
            float(board_bbox[0]) + c1 * float(cell_size),
            float(board_bbox[1]) + r1 * float(cell_size),
        )
        draw.rectangle(yard_bbox, fill=soft_color(color, amount=0.72) + (255,), outline=tuple(theme.board_border_rgb) + (255,), width=max(1, int(grid_width)))
        inner_pad = float(cell_size) * 0.82
        inner_bbox = (
            yard_bbox[0] + inner_pad,
            yard_bbox[1] + inner_pad,
            yard_bbox[2] - inner_pad,
            yard_bbox[3] - inner_pad,
        )
        draw.rounded_rectangle(
            inner_bbox,
            radius=max(10, int(round(float(cell_size) * 0.35))),
            fill=tuple(theme.yard_inner_rgb) + (245,),
            outline=tuple(theme.track_outline_rgb) + (255,),
            width=max(1, int(grid_width)),
        )
        for dy in (0.32, 0.68):
            for dx in (0.32, 0.68):
                cx = inner_bbox[0] + (inner_bbox[2] - inner_bbox[0]) * dx
                cy = inner_bbox[1] + (inner_bbox[3] - inner_bbox[1]) * dy
                half_side = float(cell_size) * 0.28
                slot_bbox = (cx - half_side, cy - half_side, cx + half_side, cy + half_side)
                draw.rounded_rectangle(
                    slot_bbox,
                    radius=max(3, int(round(float(cell_size) * 0.08))),
                    fill=soft_color(color, amount=0.48) + (245,),
                    outline=tuple(theme.track_outline_rgb) + (190,),
                    width=max(1, int(grid_width)),
                )

    for coord in sorted(main_set):
        fill = theme.track_fill_rgb
        for color, start in START_COORDS.items():
            if tuple(coord) == tuple(start):
                fill = soft_color(color, amount=0.85)
        _draw_cell(
            draw,
            _cell_bbox(board_bbox, cell_size, coord),
            fill=fill,
            outline=theme.track_outline_rgb,
            width=max(1, int(grid_width)),
        )
    for coord, color in lane_cells.items():
        _draw_cell(
            draw,
            _cell_bbox(board_bbox, cell_size, coord),
            fill=soft_color(color, amount=0.72),
            outline=theme.track_outline_rgb,
            width=max(1, int(grid_width)),
        )

    flow_arrow_markers: list[dict[str, Any]] = []
    if bool(flow_arrow_enabled):
        for start_coord, end_coord, role in FLOW_ARROW_SPECS:
            marker = _draw_flow_arrow(
                draw,
                start_cell_bbox=_cell_bbox(board_bbox, cell_size, start_coord),
                end_cell_bbox=_cell_bbox(board_bbox, cell_size, end_coord),
                role=str(role),
                fill_rgb=theme.flow_arrow_rgb,
                outline_rgb=theme.track_fill_rgb,
                width=max(1, int(flow_arrow_width)),
            )
            marker["start_coord"] = [int(start_coord[0]), int(start_coord[1])]
            marker["end_coord"] = [int(end_coord[0]), int(end_coord[1])]
            flow_arrow_markers.append(marker)

    finish_bbox = (
        float(board_bbox[0]) + 6.0 * float(cell_size),
        float(board_bbox[1]) + 6.0 * float(cell_size),
        float(board_bbox[0]) + 9.0 * float(cell_size),
        float(board_bbox[1]) + 9.0 * float(cell_size),
    )
    fx = 0.5 * (finish_bbox[0] + finish_bbox[2])
    fy = 0.5 * (finish_bbox[1] + finish_bbox[3])
    triangles = {
        "red": ((finish_bbox[0], finish_bbox[0] * 0 + finish_bbox[1]), (finish_bbox[0], finish_bbox[3]), (fx, fy)),
        "green": ((finish_bbox[0], finish_bbox[1]), (finish_bbox[2], finish_bbox[1]), (fx, fy)),
        "yellow": ((finish_bbox[2], finish_bbox[1]), (finish_bbox[2], finish_bbox[3]), (fx, fy)),
        "blue": ((finish_bbox[0], finish_bbox[3]), (finish_bbox[2], finish_bbox[3]), (fx, fy)),
    }
    for color, points in triangles.items():
        draw.polygon(points, fill=soft_color(color, amount=0.84) + (255,), outline=tuple(theme.track_outline_rgb) + (255,))
    draw.rectangle(finish_bbox, outline=tuple(theme.track_outline_rgb) + (255,), width=max(1, int(grid_width) + 1))
    return {
        "finish_bbox_px": [round(float(v), 3) for v in finish_bbox],
        "finish_center_px": [round(float(fx), 3), round(float(fy), 3)],
        "flow_arrow_markers_px": [dict(marker) for marker in flow_arrow_markers],
    }


def _draw_options(
    draw: ImageDraw.ImageDraw,
    *,
    options: Sequence[LudoRollOption],
    canvas_width: int,
    option_top: float,
    card_width: int,
    card_height: int,
    card_gap: int,
    panel_style: Any,
    theme: LudoTheme,
    font_family: str,
    instance_seed: int,
    namespace: str,
) -> Dict[str, List[float]]:
    """Draw the bottom roll-option cards and record each option bbox."""

    option_bboxes: dict[str, list[float]] = {}
    if not options:
        return option_bboxes
    label_font = load_font(max(15, int(round(float(card_height) * 0.34))), bold=True, font_family=font_family)
    text_font = load_font(max(13, int(round(float(card_height) * 0.28))), bold=True, font_family=font_family)
    total_width = (len(options) * int(card_width)) + ((len(options) - 1) * int(card_gap))
    left = 0.5 * (float(canvas_width) - float(total_width))
    for index, option in enumerate(options):
        x0 = left + (float(index) * float(card_width + card_gap))
        bbox = (x0, float(option_top), x0 + float(card_width), float(option_top) + float(card_height))
        draw_panel_option_card(draw, bbox=tuple(int(round(v)) for v in bbox), style=panel_style, radius=10, border_width=2)
        label_badge = (bbox[0] + 6.0, bbox[1] + 6.0, bbox[0] + 30.0, bbox[1] + 30.0)
        draw.ellipse(label_badge, fill=(255, 255, 255, 245), outline=tuple(theme.track_outline_rgb) + (255,), width=1)
        draw_centered_game_text_traced(
            draw,
            center=_bbox_center(label_badge),
            text=str(option.label),
            font=label_font,
            fill_rgb=theme.option_text_rgb,
            surface_rgbs=((255, 255, 255),),
            role="option_label",
            required=True,
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.option_label.{option.label}",
        )
        draw_centered_game_text_traced(
            draw,
            center=(bbox[0] + (0.60 * float(card_width)), bbox[1] + (0.53 * float(card_height))),
            text=str(option.text),
            font=text_font,
            fill_rgb=theme.option_text_rgb,
            surface_rgbs=((255, 255, 255), panel_style.panel_fill_rgb),
            role="option_text",
            required=True,
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.option_text.{option.label}",
        )
        option_bboxes[str(option.label)] = [round(float(value), 3) for value in bbox]
    return option_bboxes


def _draw_destination_options(
    draw: ImageDraw.ImageDraw,
    *,
    destination_options: Sequence[LudoDestinationOption],
    board_bbox: Sequence[float],
    cell_size: float,
    theme: LudoTheme,
    font_family: str,
    instance_seed: int,
    namespace: str,
) -> Dict[str, Dict[str, List[float]]]:
    """Draw destination letters inside board cells and record cell/text geometry."""

    cell_bboxes: dict[str, list[float]] = {}
    text_bboxes: dict[str, list[float]] = {}
    centers: dict[str, list[float]] = {}
    if not destination_options:
        return {"cell_bboxes": cell_bboxes, "text_bboxes": text_bboxes, "centers": centers}
    font = load_font(max(22, int(round(float(cell_size) * 0.64))), bold=True, font_family=font_family)
    surface_rgbs = (
        theme.track_fill_rgb,
        soft_color("red", amount=0.72),
        soft_color("green", amount=0.72),
        soft_color("yellow", amount=0.72),
        soft_color("blue", amount=0.72),
    )
    for option in destination_options:
        cell_bbox = _cell_bbox(board_bbox, float(cell_size), tuple(option.coord))
        center = _bbox_center(cell_bbox)
        disk_radius = max(9.0, float(cell_size) * 0.34)
        disk_bbox = (
            float(center[0]) - disk_radius,
            float(center[1]) - disk_radius,
            float(center[0]) + disk_radius,
            float(center[1]) + disk_radius,
        )
        draw.ellipse(
            disk_bbox,
            fill=(255, 255, 255, 238),
            outline=tuple(theme.track_outline_rgb) + (255,),
            width=max(2, int(round(float(cell_size) * 0.06))),
        )
        record = draw_centered_game_text_traced(
            draw,
            center=center,
            text=str(option.label),
            font=font,
            fill_rgb=theme.option_text_rgb,
            stroke_width=2,
            role="board_mark",
            required=True,
            surface_rgbs=((255, 255, 255),) + surface_rgbs,
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.destination_option.{option.label}",
        )
        text_bbox = record.get("bbox_px")
        cell_bboxes[str(option.label)] = [round(float(value), 3) for value in cell_bbox]
        centers[str(option.label)] = [round(float(center[0]), 3), round(float(center[1]), 3)]
        if isinstance(text_bbox, list) and len(text_bbox) == 4:
            text_bboxes[str(option.label)] = [round(float(value), 3) for value in text_bbox]
        else:
            text_bboxes[str(option.label)] = list(cell_bboxes[str(option.label)])
    return {"cell_bboxes": cell_bboxes, "text_bboxes": text_bboxes, "centers": centers}


def _draw_roll_sequence(
    draw: ImageDraw.ImageDraw,
    *,
    roll_sequence: Sequence[int],
    canvas_width: int,
    option_top: float,
    card_height: int,
    panel_style: Any,
    theme: LudoTheme,
    font_family: str,
    instance_seed: int,
    namespace: str,
) -> Dict[str, Any]:
    """Draw the visible dice-sequence strip used by move-result tasks."""

    if not roll_sequence:
        return {}
    box_size = max(34, int(round(float(card_height) * 0.70)))
    gap = max(8, int(round(float(box_size) * 0.22)))
    total_width = (len(roll_sequence) * int(box_size)) + ((len(roll_sequence) - 1) * int(gap))
    left = 0.5 * (float(canvas_width) - float(total_width))
    top = float(option_top) + max(0.0, 0.5 * (float(card_height) - float(box_size)))
    font = load_font(max(16, int(round(float(box_size) * 0.48))), bold=True, font_family=font_family)
    sequence_bbox = (left, top, left + total_width, top + box_size)
    box_bboxes: list[list[float]] = []
    for index, value in enumerate(roll_sequence):
        x0 = left + (float(index) * float(box_size + gap))
        bbox = (x0, top, x0 + float(box_size), top + float(box_size))
        draw.rounded_rectangle(
            bbox,
            radius=max(6, int(round(float(box_size) * 0.16))),
            fill=tuple(panel_style.panel_fill_rgb) + (245,),
            outline=tuple(theme.track_outline_rgb) + (255,),
            width=2,
        )
        draw_centered_game_text_traced(
            draw,
            center=_bbox_center(bbox),
            text=str(int(value)),
            font=font,
            fill_rgb=theme.option_text_rgb,
            stroke_width=1,
            role="readout",
            required=True,
            surface_rgbs=(panel_style.panel_fill_rgb,),
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.roll_sequence.{int(index)}",
        )
        box_bboxes.append([round(float(coord), 3) for coord in bbox])
    return {
        "bbox_px": [round(float(value), 3) for value in sequence_bbox],
        "center_px": [round(float(_bbox_center(sequence_bbox)[0]), 3), round(float(_bbox_center(sequence_bbox)[1]), 3)],
        "box_bboxes_px": box_bboxes,
        "values": [int(value) for value in roll_sequence],
    }


def render_ludo_scene(
    *,
    render_state: LudoRenderState,
    axes: LudoSceneAxes,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    namespace: str,
) -> RenderedLudoScene:
    """Render a complete Ludo board from task-owned symbolic state."""

    from trace_tasks.core.seed import spawn_rng

    rng = spawn_rng(int(instance_seed), f"{namespace}.render")
    cell_min = int(params.get("cell_size_min_px", group_default(render_defaults, "cell_size_min_px", DEFAULTS.cell_size_min_px)))
    cell_max = int(params.get("cell_size_max_px", group_default(render_defaults, "cell_size_max_px", DEFAULTS.cell_size_max_px)))
    cell_size = int(rng.randint(min(cell_min, cell_max), max(cell_min, cell_max)))
    board_size = int(cell_size) * 15
    card_width = int(params.get("option_card_width_px", group_default(render_defaults, "option_card_width_px", DEFAULTS.option_card_width_px)))
    card_height = int(params.get("option_card_height_px", group_default(render_defaults, "option_card_height_px", DEFAULTS.option_card_height_px)))
    card_gap = int(params.get("option_card_gap_px", group_default(render_defaults, "option_card_gap_px", DEFAULTS.option_card_gap_px)))
    has_options = bool(render_state.roll_options)
    has_roll_sequence = bool(render_state.roll_sequence)
    has_bottom_area = bool(has_options or has_roll_sequence)
    side_padding = int(params.get("canvas_side_padding_px", group_default(render_defaults, "canvas_side_padding_px", DEFAULTS.canvas_side_padding_px)))
    vertical_padding = int(params.get("canvas_vertical_padding_px", group_default(render_defaults, "canvas_vertical_padding_px", DEFAULTS.canvas_vertical_padding_px)))
    options_height = (card_height + 34) if has_bottom_area else 0
    option_panel_count = len(render_state.roll_options) if has_options else 0
    option_panel_width = (
        (int(option_panel_count) * card_width) + ((int(option_panel_count) - 1) * card_gap) + 80
        if int(option_panel_count) > 0
        else 0
    )
    canvas_width = max(int(board_size + side_padding), int(option_panel_width))
    canvas_width = min(int(params.get("canvas_width", group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width))), int(canvas_width))
    canvas_height = min(
        int(params.get("canvas_height", group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height))),
        int(board_size + vertical_padding + options_height),
    )
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.panel_scene_style",
        treatment_weights=params.get("panel_scene_treatment_weights", group_default(render_defaults, "panel_scene_treatment_weights", None)),
        palette_weights=params.get("panel_scene_palette_weights", group_default(render_defaults, "panel_scene_palette_weights", None)),
    )
    image, background_meta = make_panel_scene_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=panel_style,
    )
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme, theme_meta = theme_for_style(str(axes.style_variant))
    option_area_height = float(card_height + 34) if has_bottom_area else 0.0
    board_bbox = (
        round(0.5 * (float(canvas_width) - float(board_size)), 3),
        round(0.5 * (float(canvas_height) - option_area_height - float(board_size)), 3),
        round(0.5 * (float(canvas_width) + float(board_size)), 3),
        round(0.5 * (float(canvas_height) - option_area_height + float(board_size)), 3),
    )
    layout_jitter = resolve_games_layout_jitter(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.layout",
    )
    board_bbox, _dx, _dy, resolved_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=board_bbox,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height - option_area_height),
        jitter=layout_jitter,
    )
    board_padding = int(params.get("board_padding_px", group_default(render_defaults, "board_padding_px", DEFAULTS.board_padding_px)))
    panel_bbox = (
        int(round(float(board_bbox[0]) - float(board_padding))),
        int(round(float(board_bbox[1]) - float(board_padding))),
        int(round(float(board_bbox[2]) + float(board_padding))),
        int(round(float(board_bbox[3]) + float(board_padding))),
    )
    draw_panel_scene_chrome(draw, bbox=panel_bbox, style=panel_style, radius=24, border_width=2)
    grid_width = int(params.get("grid_width_px", group_default(render_defaults, "grid_width_px", DEFAULTS.grid_width_px)))
    flow_arrow_enabled = bool(params.get("flow_arrow_enabled", group_default(render_defaults, "flow_arrow_enabled", DEFAULTS.flow_arrow_enabled)))
    flow_arrow_width = int(params.get("flow_arrow_width_px", group_default(render_defaults, "flow_arrow_width_px", DEFAULTS.flow_arrow_width_px)))
    board_meta = _draw_ludo_board(
        draw,
        board_bbox=board_bbox,
        cell_size=float(cell_size),
        theme=theme,
        grid_width=max(1, int(grid_width)),
        flow_arrow_enabled=bool(flow_arrow_enabled),
        flow_arrow_width=max(1, int(flow_arrow_width)),
    )
    token_bboxes: dict[str, list[float]] = {}
    token_centers: dict[str, list[float]] = {}
    entities: list[dict[str, Any]] = []
    for color in PLAYER_COLORS:
        coord = tuple(render_state.token_coords[str(color)])
        cell_bbox = _cell_bbox(board_bbox, float(cell_size), coord)
        token_bbox = _draw_token(draw, bbox=cell_bbox, color=str(color), theme=theme, width=max(2, int(grid_width) + 1))
        center = _bbox_center(token_bbox)
        token_id = f"token_{color}"
        token_bboxes[token_id] = [round(float(value), 3) for value in token_bbox]
        token_centers[token_id] = [round(float(center[0]), 3), round(float(center[1]), 3)]
        entities.append(
            {
                "entity_id": str(token_id),
                "entity_type": "ludo_token",
                "color": str(color),
                "coord": [int(coord[0]), int(coord[1])],
                "center_px": list(token_centers[token_id]),
                "bbox_px": list(token_bboxes[token_id]),
            }
        )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.labels",
        params=params,
        explicit_key="label_font_family",
        weights_key="label_font_family_weights",
    )
    destination_maps = _draw_destination_options(
        draw,
        destination_options=render_state.destination_options,
        board_bbox=board_bbox,
        cell_size=float(cell_size),
        theme=theme,
        font_family=str(font_family),
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    for option in render_state.destination_options:
        label = str(option.label)
        entities.append(
            {
                "entity_id": f"destination_option_{label}",
                "entity_type": "ludo_destination_option",
                "label": label,
                "coord": [int(option.coord[0]), int(option.coord[1])],
                "center_px": list(destination_maps["centers"].get(label, [])),
                "bbox_px": list(destination_maps["cell_bboxes"].get(label, [])),
            }
        )
    option_top = float(board_bbox[3]) + 24.0
    option_bboxes = {}
    roll_sequence_map: dict[str, Any] = {}
    if has_options:
        option_bboxes = _draw_options(
            draw,
            options=render_state.roll_options,
            canvas_width=int(canvas_width),
            option_top=float(option_top),
            card_width=int(card_width),
            card_height=int(card_height),
            card_gap=int(card_gap),
            panel_style=panel_style,
            theme=theme,
            font_family=str(font_family),
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
    elif has_roll_sequence:
        roll_sequence_map = _draw_roll_sequence(
            draw,
            roll_sequence=render_state.roll_sequence,
            canvas_width=int(canvas_width),
            option_top=float(option_top),
            card_height=int(card_height),
            panel_style=panel_style,
            theme=theme,
            font_family=str(font_family),
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
    render_map = {
        "board_bbox_px": [round(float(value), 3) for value in board_bbox],
        "finish_bbox_px": list(board_meta["finish_bbox_px"]),
        "finish_center_px": list(board_meta["finish_center_px"]),
        "flow_arrow_markers_px": [dict(marker) for marker in board_meta.get("flow_arrow_markers_px", [])],
        "token_bboxes_px": dict(token_bboxes),
        "token_centers_px": dict(token_centers),
        "option_bboxes_px": dict(option_bboxes),
        "destination_option_cell_bboxes_px": dict(destination_maps["cell_bboxes"]),
        "destination_option_text_bboxes_px": dict(destination_maps["text_bboxes"]),
        "destination_option_centers_px": dict(destination_maps["centers"]),
        "roll_sequence_px": dict(roll_sequence_map),
        "layout_jitter": dict(resolved_jitter),
        "effective_cell_size_px": int(cell_size),
        "effective_board_size_px": int(board_size),
        "label_font": font_role_trace(str(font_family), role="readout"),
    }
    return RenderedLudoScene(
        image=image.convert("RGB"),
        entities=tuple(entities),
        render_map=render_map,
        style_meta={
            "panel_scene_style": dict(panel_style_meta),
            "ludo_board_style": dict(theme_meta),
            "label_font": font_role_trace(str(font_family), role="readout"),
        },
        background_meta=dict(background_meta),
    )



__all__ = [
    "RenderedLudoScene",
    "render_ludo_scene",
]
