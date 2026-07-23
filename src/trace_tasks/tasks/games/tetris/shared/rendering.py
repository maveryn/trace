"""Rendering helpers for Tetris board scenes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.text_rendering import fit_font_to_box, resolve_text_stroke_fill
from trace_tasks.tasks.games.shared.layout import apply_games_layout_jitter_to_bbox, resolve_games_layout_jitter
from trace_tasks.tasks.games.shared.scene_style import draw_panel_grid_cell, draw_panel_option_card, make_panel_scene_background, resolve_game_panel_scene_style
from trace_tasks.tasks.games.shared.text import draw_game_text_traced as draw_text_traced

from .defaults import DEFAULTS, RENDER_DEFAULTS
from .rules import TETROMINOES, board_size, piece_cells, shape_size
from .state import Board, Coord, EMPTY, OPTION_LABELS, PIECE_ORDER, RenderParams, RenderedTetrisScene, SUPPORTED_STYLE_VARIANTS, TetrisSample

RENDER_MODE_STATIC_BOARD = "static_board"
RENDER_MODE_LINE_CLEAR = "line_clear"
RENDER_MODE_COLLISION = "collision"
RENDER_MODE_RESULT_OPTIONS = "result_options"
RENDER_MODE_ACTIVE_SHAPE = "active_shape"


def render_params(params: Mapping[str, Any], *, instance_seed: int) -> RenderParams:
    """Resolve canvas, font, and layout jitter controls for one render."""

    layout_jitter = resolve_games_layout_jitter(
        params,
        RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace="games.tetris.layout",
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="games.tetris.font_family",
        params=params,
    )
    return RenderParams(
        canvas_width=int(params.get("canvas_width", group_default(RENDER_DEFAULTS, "canvas_width", DEFAULTS.canvas_width))),
        canvas_height=int(params.get("canvas_height", group_default(RENDER_DEFAULTS, "canvas_height", DEFAULTS.canvas_height))),
        panel_margin_px=int(params.get("panel_margin_px", group_default(RENDER_DEFAULTS, "panel_margin_px", DEFAULTS.panel_margin_px))),
        board_gap_px=int(params.get("board_gap_px", group_default(RENDER_DEFAULTS, "board_gap_px", DEFAULTS.board_gap_px))),
        cell_size_px=int(params.get("cell_size_px", group_default(RENDER_DEFAULTS, "cell_size_px", DEFAULTS.cell_size_px))),
        line_cell_size_px=int(params.get("line_cell_size_px", group_default(RENDER_DEFAULTS, "line_cell_size_px", DEFAULTS.line_cell_size_px))),
        cell_gap_px=int(params.get("cell_gap_px", group_default(RENDER_DEFAULTS, "cell_gap_px", DEFAULTS.cell_gap_px))),
        panel_pad_px=int(params.get("panel_pad_px", group_default(RENDER_DEFAULTS, "panel_pad_px", DEFAULTS.panel_pad_px))),
        label_band_height_px=int(params.get("label_band_height_px", group_default(RENDER_DEFAULTS, "label_band_height_px", DEFAULTS.label_band_height_px))),
        cell_outline_width_px=int(params.get("cell_outline_width_px", group_default(RENDER_DEFAULTS, "cell_outline_width_px", DEFAULTS.cell_outline_width_px))),
        ghost_outline_width_px=int(params.get("ghost_outline_width_px", group_default(RENDER_DEFAULTS, "ghost_outline_width_px", DEFAULTS.ghost_outline_width_px))),
        label_font_size_px=int(params.get("label_font_size_px", group_default(RENDER_DEFAULTS, "label_font_size_px", DEFAULTS.label_font_size_px))),
        small_label_font_size_px=int(params.get("small_label_font_size_px", group_default(RENDER_DEFAULTS, "small_label_font_size_px", DEFAULTS.small_label_font_size_px))),
        font_family=str(font_family),
        layout_jitter_meta=dict(layout_jitter),
    )


def piece_color(piece: str, *, style_variant: str, state_colors: Sequence[Sequence[int]]) -> Tuple[int, int, int]:
    palettes: Dict[str, Dict[str, Tuple[int, int, int]]] = {
        "classic_blocks": {"I": (38, 180, 210), "O": (239, 193, 50), "T": (154, 90, 204), "S": (78, 174, 86), "Z": (220, 72, 82), "J": (65, 112, 210), "L": (230, 145, 52)},
        "beveled_blocks": {"I": (33, 166, 196), "O": (226, 177, 42), "T": (142, 82, 190), "S": (64, 156, 78), "Z": (207, 63, 78), "J": (54, 103, 194), "L": (218, 126, 42)},
        "paper_tiles": {"I": (88, 166, 184), "O": (214, 178, 86), "T": (151, 113, 174), "S": (105, 166, 112), "Z": (195, 102, 107), "J": (95, 128, 188), "L": (206, 142, 82)},
        "neon_blocks": {"I": (42, 230, 244), "O": (255, 228, 78), "T": (210, 96, 255), "S": (86, 242, 126), "Z": (255, 86, 116), "J": (92, 164, 255), "L": (255, 172, 70)},
    }
    fallback = palettes["classic_blocks"]
    palette = palettes.get(str(style_variant), fallback)
    if str(style_variant) == "panel_state_colors":
        index = PIECE_ORDER.index(str(piece)) if str(piece) in PIECE_ORDER else 0
        if len(state_colors) >= len(PIECE_ORDER):
            return tuple(int(v) for v in state_colors[index % len(state_colors)])
    return palette.get(str(piece), (110, 130, 155))


def draw_tetris_block(draw: ImageDraw.ImageDraw, *, bbox: Tuple[int, int, int, int], fill: Sequence[int], style, params: RenderParams, style_variant: str, outline: Sequence[int] | None = None) -> None:
    """Draw one locked Tetris block with scene-local nonsemantic styling."""

    fill_rgb = tuple(int(v) for v in fill)
    draw_panel_grid_cell(draw, bbox=bbox, fill=fill_rgb, style=style, outline=outline or style.grid_rgb, width=int(params.cell_outline_width_px))
    x0, y0, x1, y1 = [int(v) for v in bbox]
    inset = max(2, int(round(min(x1 - x0, y1 - y0) * 0.14)))
    variant = str(style_variant)
    if variant == "beveled_blocks":
        highlight = tuple(min(255, int(v) + 54) for v in fill_rgb)
        shadow = tuple(max(0, int(v) - 64) for v in fill_rgb)
        draw.line((x0 + inset, y0 + inset, x1 - inset, y0 + inset), fill=highlight, width=2)
        draw.line((x0 + inset, y0 + inset, x0 + inset, y1 - inset), fill=highlight, width=2)
        draw.line((x0 + inset, y1 - inset, x1 - inset, y1 - inset), fill=shadow, width=2)
        draw.line((x1 - inset, y0 + inset, x1 - inset, y1 - inset), fill=shadow, width=2)
    elif variant == "paper_tiles":
        accent = tuple(max(0, int(v) - 42) for v in fill_rgb)
        draw.line((x0 + inset, y1 - inset, x1 - inset, y0 + inset), fill=accent, width=1)
    elif variant == "neon_blocks":
        glow = tuple(min(255, int(v) + 40) for v in fill_rgb)
        draw.rounded_rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), radius=max(3, inset), outline=glow + (230,), width=max(2, int(params.cell_outline_width_px) + 1))


def board_panel_size(params: RenderParams, *, board_rows: int, board_cols: int) -> Tuple[int, int]:
    cell = int(params.cell_size_px)
    gap = int(params.cell_gap_px)
    board_w = (int(board_cols) * cell) + ((int(board_cols) - 1) * gap)
    board_h = (int(board_rows) * cell) + ((int(board_rows) - 1) * gap)
    panel_w = board_w + (2 * int(params.panel_pad_px))
    panel_h = board_h + (2 * int(params.panel_pad_px)) + int(params.label_band_height_px)
    return int(panel_w), int(panel_h)


def piece_preview_panel_size(params: RenderParams) -> Tuple[int, int]:
    cell = int(params.cell_size_px)
    gap = int(params.cell_gap_px)
    preview_w = (4 * cell) + (3 * gap)
    preview_h = (4 * cell) + (3 * gap)
    return int(preview_w + (2 * int(params.panel_pad_px))), int(preview_h + (2 * int(params.panel_pad_px)) + int(params.label_band_height_px))


def shape_legend_panel_size(params: RenderParams, *, option_count: int = 4) -> Tuple[int, int]:
    """Return the footprint for four text shape-name option cards."""

    count = max(1, int(option_count))
    option_w = max(92, int(round(int(params.cell_size_px) * 2.65)))
    option_h = int(params.label_band_height_px) + 30
    legend_w = (count * option_w) + ((count - 1) * 10) + (2 * int(params.panel_pad_px))
    legend_h = option_h + (2 * int(params.panel_pad_px))
    return int(legend_w), int(legend_h)


def compact_canvas_params_for_mode(params: RenderParams, *, render_mode: str, board_rows: int, board_cols: int) -> RenderParams:
    """Shrink non-option Tetris canvases around the board while keeping readable cells."""

    if str(render_mode) == RENDER_MODE_RESULT_OPTIONS:
        grid_rows = 3
        grid_cols = 2
        margin = max(24, int(params.panel_margin_px))
        gap = int(params.board_gap_px)
        panel_pad = int(params.panel_pad_px)
        label_band = int(params.label_band_height_px)
        cell_gap = int(params.cell_gap_px)
        usable_h = (
            int(params.canvas_height)
            - (2 * margin)
            - ((grid_rows - 1) * gap)
            - (grid_rows * ((2 * panel_pad) + label_band))
            - (grid_rows * (max(0, int(board_rows) - 1) * cell_gap))
        )
        fitted_cell = max(10, min(int(params.cell_size_px), usable_h // max(1, grid_rows * int(board_rows))))
        panel_w = (int(board_cols) * fitted_cell) + (max(0, int(board_cols) - 1) * cell_gap) + (2 * panel_pad)
        compact_w = (grid_cols * panel_w) + ((grid_cols - 1) * gap) + (2 * margin)
        return replace(params, canvas_width=max(560, min(int(params.canvas_width), int(compact_w))))
    row_params = replace(params, cell_size_px=int(params.line_cell_size_px))
    board_w, board_h = board_panel_size(row_params, board_rows=int(board_rows), board_cols=int(board_cols))
    margin = max(24, int(params.panel_margin_px))
    if str(render_mode) == RENDER_MODE_ACTIVE_SHAPE:
        legend_w, legend_h = shape_legend_panel_size(row_params)
        total_w = max(int(board_w), int(legend_w))
        total_h = int(board_h) + int(params.board_gap_px) + int(legend_h)
        canvas_w = int(total_w + (2 * margin) + 40)
        canvas_h = int(total_h + (2 * margin))
        return replace(params, canvas_width=max(canvas_w, 620), canvas_height=max(canvas_h, 700))
    if str(render_mode) == RENDER_MODE_LINE_CLEAR:
        preview_w, preview_h = piece_preview_panel_size(row_params)
        total_w = max(int(board_w), int(preview_w))
        total_h = int(preview_h) + int(params.board_gap_px) + int(board_h)
        canvas_w = int(total_w + (2 * margin) + 60)
        canvas_h = int(total_h + (2 * margin))
        return replace(params, canvas_width=max(canvas_w, 520), canvas_height=max(canvas_h, 680))
    canvas_w = int(board_w + (2 * margin) + 60)
    canvas_h = int(board_h + (2 * margin) + 56)
    return replace(params, canvas_width=max(canvas_w, 500), canvas_height=max(canvas_h, 560))


def label_text(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], text: str, *, font_size: int, fill: Sequence[int], font_family: str = "") -> None:
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=max(12, float(bbox[2] - bbox[0] - 8)),
        max_height=max(10, float(bbox[3] - bbox[1] - 4)),
        bold=True,
        min_size_px=8,
        max_size_px=int(font_size),
        fill_ratio=0.95,
        font_family=str(font_family) or None,
    )
    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=1)
    fill_rgb = tuple(int(v) for v in fill)
    draw_text_traced(
        draw,
        (bbox[0] + ((bbox[2] - bbox[0]) - (text_bbox[2] - text_bbox[0])) / 2.0, bbox[1] + ((bbox[3] - bbox[1]) - (text_bbox[3] - text_bbox[1])) / 2.0 - text_bbox[1]),
        str(text),
        font=font,
        fill=fill_rgb,
        stroke_width=1,
        stroke_fill=tuple(resolve_text_stroke_fill(fill_rgb)),
        role="readout",
        required=False,
    )


def draw_board_panel(
    image: Image.Image,
    *,
    panel_bbox: Tuple[int, int, int, int],
    label: str,
    board: Board,
    style,
    params: RenderParams,
    ghost_cells: Sequence[Coord] = (),
    ghost_piece: str | None = None,
    falling_cells: Sequence[Coord] = (),
    falling_piece: str | None = None,
    selected_rows: Sequence[int] = (),
    entity_prefix: str,
    style_variant: str,
) -> Tuple[Dict[str, Any], Tuple[Dict[str, Any], ...]]:
    """Draw a labeled Tetris board and return projection maps for cells and rows."""

    draw = ImageDraw.Draw(image, "RGBA")
    draw_panel_option_card(draw, bbox=panel_bbox, style=style, radius=12, border_width=2)
    label_bbox = (int(panel_bbox[0] + params.panel_pad_px), int(panel_bbox[1] + params.panel_pad_px), int(panel_bbox[2] - params.panel_pad_px), int(panel_bbox[1] + params.panel_pad_px + params.label_band_height_px))
    label_text(draw, label_bbox, str(label), font_size=int(params.label_font_size_px), fill=style.text_rgb, font_family=str(params.font_family))
    board_left = int(panel_bbox[0] + params.panel_pad_px)
    board_top = int(panel_bbox[1] + params.panel_pad_px + params.label_band_height_px)
    cell = int(params.cell_size_px)
    gap = int(params.cell_gap_px)
    cell_bboxes: Dict[str, List[float]] = {}
    row_bboxes: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = [{"id": str(entity_prefix), "type": "tetris_board_panel", "label": str(label), "bbox_px": [float(v) for v in panel_bbox]}]
    ghost_set = {tuple(coord) for coord in ghost_cells}
    falling_set = {tuple(coord) for coord in falling_cells}
    selected_row_set = {int(row) for row in selected_rows}
    board_rows, board_cols = board_size(board)
    for row in range(board_rows):
        row_left = board_left
        row_top = board_top + row * (cell + gap)
        row_right = board_left + board_cols * cell + (board_cols - 1) * gap
        row_bottom = row_top + cell
        row_id = f"{entity_prefix}_row_{row}"
        row_bboxes[row_id] = [float(row_left), float(row_top), float(row_right), float(row_bottom)]
        if row in selected_row_set:
            draw.rectangle((row_left - 2, row_top - 2, row_right + 2, row_bottom + 2), outline=style.mark_rgb, width=3)
        for col in range(board_cols):
            left = board_left + col * (cell + gap)
            top = board_top + row * (cell + gap)
            bbox = (int(left), int(top), int(left + cell), int(top + cell))
            cell_value = board[row][col]
            if cell_value == EMPTY:
                draw_panel_grid_cell(draw, bbox=bbox, fill=tuple(style.panel_fill_rgb), style=style, outline=style.grid_rgb, width=int(params.cell_outline_width_px))
            else:
                fill = piece_color(str(cell_value), style_variant=str(style_variant), state_colors=style.state_colors)
                draw_tetris_block(draw, bbox=bbox, fill=fill, style=style, params=params, style_variant=str(style_variant), outline=style.grid_rgb)
            if (row, col) in ghost_set:
                ghost_fill = piece_color(str(ghost_piece or "I"), style_variant=str(style_variant), state_colors=style.state_colors)
                draw.rounded_rectangle((bbox[0] + 3, bbox[1] + 3, bbox[2] - 3, bbox[3] - 3), radius=5, fill=tuple(ghost_fill) + (78,), outline=tuple(style.mark_rgb) + (255,), width=int(params.ghost_outline_width_px))
            if (row, col) in falling_set:
                falling_fill = piece_color(str(falling_piece or "I"), style_variant=str(style_variant), state_colors=style.state_colors)
                draw_tetris_block(draw, bbox=bbox, fill=falling_fill, style=style, params=params, style_variant=str(style_variant), outline=style.grid_rgb)
            cell_id = f"{entity_prefix}_cell_{row}_{col}"
            cell_bboxes[cell_id] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            entities.append({"id": cell_id, "type": "tetris_cell", "row": int(row), "col": int(col), "value": str(cell_value), "ghost": bool((row, col) in ghost_set), "falling": bool((row, col) in falling_set), "bbox_px": list(cell_bboxes[cell_id])})
    return {"panel_bbox_px": [float(v) for v in panel_bbox], "cell_bboxes_px": dict(cell_bboxes), "row_bboxes_px": dict(row_bboxes)}, tuple(entities)


def draw_piece_preview_panel(image: Image.Image, *, panel_bbox: Tuple[int, int, int, int], label: str, piece: str, orientation_index: int, style, params: RenderParams, entity_id: str, style_variant: str) -> Tuple[Dict[str, Any], Tuple[Dict[str, Any], ...]]:
    """Draw the 4x4 NEXT-piece preview panel and projection boxes."""

    draw = ImageDraw.Draw(image, "RGBA")
    draw_panel_option_card(draw, bbox=panel_bbox, style=style, radius=12, border_width=2)
    label_bbox = (int(panel_bbox[0] + params.panel_pad_px), int(panel_bbox[1] + params.panel_pad_px), int(panel_bbox[2] - params.panel_pad_px), int(panel_bbox[1] + params.panel_pad_px + params.label_band_height_px))
    label_text(draw, label_bbox, str(label), font_size=int(params.label_font_size_px), fill=style.text_rgb, font_family=str(params.font_family))
    cell = int(params.cell_size_px)
    gap = int(params.cell_gap_px)
    grid_left = int(panel_bbox[0] + params.panel_pad_px)
    grid_top = int(panel_bbox[1] + params.panel_pad_px + params.label_band_height_px)
    grid_w = (4 * cell) + (3 * gap)
    grid_h = (4 * cell) + (3 * gap)
    piece_shape = TETROMINOES[str(piece)][int(orientation_index)]
    shape_h, shape_w = shape_size(piece_shape)
    offset_col = (4 - int(shape_w)) // 2
    offset_row = (4 - int(shape_h)) // 2
    entities: List[Dict[str, Any]] = [{"id": str(entity_id), "type": "tetris_next_piece_panel", "piece": str(piece), "orientation_index": int(orientation_index), "bbox_px": [float(v) for v in panel_bbox]}]
    cell_bboxes: Dict[str, List[float]] = {}
    for row in range(4):
        for col in range(4):
            left = grid_left + col * (cell + gap)
            top = grid_top + row * (cell + gap)
            bbox = (int(left), int(top), int(left + cell), int(top + cell))
            draw_panel_grid_cell(draw, bbox=bbox, fill=tuple(style.panel_fill_rgb), style=style, outline=style.grid_rgb, width=int(params.cell_outline_width_px))
    for index, (row, col) in enumerate(piece_shape):
        r = int(row) + offset_row
        c = int(col) + offset_col
        left = grid_left + c * (cell + gap)
        top = grid_top + r * (cell + gap)
        bbox = (int(left), int(top), int(left + cell), int(top + cell))
        fill = piece_color(str(piece), style_variant=str(style_variant), state_colors=style.state_colors)
        draw_tetris_block(draw, bbox=bbox, fill=fill, style=style, params=params, style_variant=str(style_variant), outline=style.grid_rgb)
        cell_id = f"{entity_id}_cell_{index}"
        cell_bboxes[cell_id] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        entities.append({"id": cell_id, "type": "tetris_next_piece_cell", "piece": str(piece), "bbox_px": list(cell_bboxes[cell_id])})
    return {"panel_bbox_px": [float(v) for v in panel_bbox], "cell_bboxes_px": dict(cell_bboxes), "grid_bbox_px": [float(grid_left), float(grid_top), float(grid_left + grid_w), float(grid_top + grid_h)]}, tuple(entities)


def draw_shape_legend_panel(
    image: Image.Image,
    *,
    panel_bbox: Tuple[int, int, int, int],
    option_entries: Sequence[Mapping[str, str]],
    style,
    params: RenderParams,
    style_variant: str,
) -> Tuple[Dict[str, Any], Tuple[Dict[str, Any], ...]]:
    """Draw text shape-name options without visual mini-piece distractors."""

    draw = ImageDraw.Draw(image, "RGBA")
    draw_panel_option_card(draw, bbox=panel_bbox, style=style, radius=12, border_width=2)
    inner_left = int(panel_bbox[0] + params.panel_pad_px)
    inner_top = int(panel_bbox[1] + params.panel_pad_px)
    option_gap = 10
    raw_entries = tuple(option_entries)
    option_count = max(1, len(raw_entries))
    option_w = max(92, int((int(panel_bbox[2]) - int(panel_bbox[0]) - (2 * int(params.panel_pad_px)) - ((option_count - 1) * option_gap)) / option_count))
    option_h = int(panel_bbox[3]) - int(panel_bbox[1]) - (2 * int(params.panel_pad_px))
    option_bboxes: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = [
        {"id": "shape_options", "type": "tetris_shape_option_legend", "bbox_px": [float(v) for v in panel_bbox]}
    ]
    normalized_entries: List[Dict[str, str]] = []
    for index, raw_entry in enumerate(raw_entries):
        label = str(raw_entry.get("label", OPTION_LABELS[index]))
        piece = str(raw_entry.get("piece", "")).strip().upper()
        if piece not in PIECE_ORDER:
            raise ValueError(f"unsupported Tetris shape option piece: {piece}")
        x0 = int(inner_left + index * (option_w + option_gap))
        y0 = int(inner_top)
        option_bbox = (x0, y0, int(x0 + option_w), int(y0 + option_h))
        draw.rounded_rectangle(
            option_bbox,
            radius=8,
            fill=tuple(style.option_fill_rgb),
            outline=tuple(style.grid_rgb),
            width=1,
        )
        label_bbox = (x0 + 5, y0 + 4, x0 + option_w - 5, y0 + option_h - 4)
        label_text(
            draw,
            label_bbox,
            f"{label}: {piece}",
            font_size=max(int(params.label_font_size_px), int(params.small_label_font_size_px) + 2),
            fill=style.text_rgb,
            font_family=str(params.font_family),
        )
        entity_id = f"shape_option_{str(label).lower()}"
        option_bboxes[entity_id] = [float(value) for value in option_bbox]
        normalized_entries.append({"label": str(label), "piece": str(piece), "entity_id": str(entity_id)})
        entities.append(
            {
                "id": entity_id,
                "type": "tetris_shape_option",
                "piece": str(piece),
                "label": str(label),
                "bbox_px": list(option_bboxes[entity_id]),
            }
        )
    return {"panel_bbox_px": [float(v) for v in panel_bbox], "option_bboxes_px": dict(option_bboxes), "option_entries": tuple(normalized_entries)}, tuple(entities)


def result_option_grid_params(params: RenderParams, *, panel_count: int, board_rows: int, board_cols: int) -> RenderParams:
    """Return a fitted START-plus-2x2 result-board option grid."""

    if int(panel_count) != 5:
        raise ValueError("Tetris result-board layout requires START plus exactly four options")
    cols = 2
    rows = 3
    cell_gap = int(params.cell_gap_px)
    panel_pad = int(params.panel_pad_px)
    label_band = int(params.label_band_height_px)
    board_gap = int(params.board_gap_px)
    max_cell_w = (int(params.canvas_width) - ((int(cols) - 1) * board_gap) - (int(cols) * 2 * panel_pad) - (int(cols) * (int(board_cols) - 1) * cell_gap)) // max(1, int(cols) * int(board_cols))
    max_cell_h = (int(params.canvas_height) - ((int(rows) - 1) * board_gap) - (int(rows) * ((2 * panel_pad) + label_band)) - (int(rows) * (int(board_rows) - 1) * cell_gap)) // max(1, int(rows) * int(board_rows))
    fitted_cell = max(10, min(int(params.cell_size_px), int(max_cell_w), int(max_cell_h)))
    return replace(params, cell_size_px=int(fitted_cell))


def result_option_panel_bboxes(params: RenderParams, *, board_rows: int, board_cols: int) -> Tuple[Tuple[int, int, int, int], Tuple[Tuple[int, int, int, int], ...]]:
    """Place START centered above a two-by-two option-board grid."""

    panel_w, panel_h = board_panel_size(params, board_rows=int(board_rows), board_cols=int(board_cols))
    gap = int(params.board_gap_px)
    total_w = (2 * int(panel_w)) + gap
    total_h = (3 * int(panel_h)) + (2 * gap)
    left = int((int(params.canvas_width) - total_w) / 2.0)
    top = int((int(params.canvas_height) - total_h) / 2.0)
    group_bbox = (float(left), float(top), float(left + total_w), float(top + total_h))
    _shifted, dx, dy, _resolved = apply_games_layout_jitter_to_bbox(
        bbox_px=group_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    shift_x = int(round(dx))
    shift_y = int(round(dy))
    start_left = int(left + ((total_w - int(panel_w)) / 2.0) + shift_x)
    start_top = int(top + shift_y)
    start_bbox = (start_left, start_top, int(start_left + panel_w), int(start_top + panel_h))
    option_bboxes: List[Tuple[int, int, int, int]] = []
    option_top = int(top + int(panel_h) + gap + shift_y)
    for row in range(2):
        for col in range(2):
            x0 = int(left + col * (int(panel_w) + gap) + shift_x)
            y0 = int(option_top + row * (int(panel_h) + gap))
            option_bboxes.append((x0, y0, int(x0 + panel_w), int(y0 + panel_h)))
    return start_bbox, tuple(option_bboxes)


def render_tetris_scene(*, sample: TetrisSample, render_mode: str, style_variant: str, params: RenderParams, instance_seed: int) -> RenderedTetrisScene:
    """Render one Tetris sample using a semantic layout mode chosen by the task."""

    board_rows, board_cols = board_size(sample.board)
    params = compact_canvas_params_for_mode(
        params,
        render_mode=str(render_mode),
        board_rows=int(board_rows),
        board_cols=int(board_cols),
    )
    style, panel_style_meta = resolve_game_panel_scene_style(instance_seed=int(instance_seed), namespace="games.tetris.panel_style")
    background, background_meta = make_panel_scene_background(canvas_width=int(params.canvas_width), canvas_height=int(params.canvas_height), style=style)
    image = background.convert("RGBA")
    entities: List[Dict[str, Any]] = []
    render_map: Dict[str, Any] = {
        "board_rows": int(board_rows),
        "board_cols": int(board_cols),
        "panels": {},
        "cell_bboxes_px": {},
        "row_bboxes_px": {},
        "option_bboxes_px": {},
        "layout_jitter": dict(params.layout_jitter_meta),
        "font_family": str(params.font_family),
        "text_style": {"font_family": str(params.font_family)},
        "panel_scene_style": dict(panel_style_meta),
        "tetris_board_style": {"style_variant": str(style_variant), "available_styles": list(SUPPORTED_STYLE_VARIANTS), "piece_palette_policy": "scene_local_tetromino_piece_palette"},
    }

    if str(render_mode) in {RENDER_MODE_STATIC_BOARD, RENDER_MODE_COLLISION, RENDER_MODE_ACTIVE_SHAPE}:
        row_params = replace(params, cell_size_px=int(params.line_cell_size_px))
        board_w, board_h = board_panel_size(row_params, board_rows=board_rows, board_cols=board_cols)
        shape_option_entries = tuple(sample.metadata.get("shape_option_entries") or ())
        legend_w, legend_h = shape_legend_panel_size(row_params, option_count=len(shape_option_entries) or 4) if str(render_mode) == RENDER_MODE_ACTIVE_SHAPE else (0, 0)
        total_w = max(int(board_w), int(legend_w))
        total_h = int(board_h) + (int(row_params.board_gap_px) + int(legend_h) if str(render_mode) == RENDER_MODE_ACTIVE_SHAPE else 0)
        left = int((int(row_params.canvas_width) - total_w) / 2.0)
        top = int((int(row_params.canvas_height) - total_h) / 2.0)
        group_bbox = (float(left), float(top), float(left + total_w), float(top + total_h))
        _shifted, dx, dy, _resolved = apply_games_layout_jitter_to_bbox(bbox_px=group_bbox, canvas_width=int(row_params.canvas_width), canvas_height=int(row_params.canvas_height), jitter=row_params.layout_jitter_meta)
        panel_left = int(left + ((total_w - int(board_w)) / 2.0) + round(dx))
        panel_top = int(top + round(dy))
        panel_bbox = (panel_left, panel_top, int(panel_left + board_w), int(panel_top + board_h))
        is_collision = str(render_mode) == RENDER_MODE_COLLISION
        panel_map, panel_entities = draw_board_panel(
            image,
            panel_bbox=panel_bbox,
            label="START" if is_collision else "BOARD",
            board=sample.board,
            style=style,
            params=row_params,
            falling_cells=piece_cells(sample.falling_placement) if sample.falling_placement is not None else (),
            falling_piece=sample.piece,
            entity_prefix="start" if is_collision else "main",
            style_variant=str(style_variant),
        )
        entities.extend(panel_entities)
        panel_key = "start" if is_collision else "main"
        render_map["panels"][panel_key] = panel_map["panel_bbox_px"]
        render_map["option_bboxes_px"][panel_key] = panel_map["panel_bbox_px"]
        render_map["cell_bboxes_px"].update(panel_map["cell_bboxes_px"])
        render_map["row_bboxes_px"].update(panel_map["row_bboxes_px"])
        if str(render_mode) == RENDER_MODE_ACTIVE_SHAPE:
            legend_left = int(left + ((total_w - int(legend_w)) / 2.0) + round(dx))
            legend_top = int(panel_bbox[3] + int(row_params.board_gap_px))
            legend_bbox = (legend_left, legend_top, int(legend_left + legend_w), int(legend_top + legend_h))
            legend_map, legend_entities = draw_shape_legend_panel(
                image,
                panel_bbox=legend_bbox,
                option_entries=shape_option_entries,
                style=style,
                params=row_params,
                style_variant=str(style_variant),
            )
            entities.extend(legend_entities)
            render_map["panels"]["shape_options"] = legend_map["panel_bbox_px"]
            render_map["shape_option_bboxes_px"] = dict(legend_map["option_bboxes_px"])
            render_map["shape_option_entries"] = list(legend_map["option_entries"])
            render_map["shape_option_labels"] = [str(entry["label"]) for entry in legend_map["option_entries"]]
            render_map["shape_option_pieces"] = [str(entry["piece"]) for entry in legend_map["option_entries"]]
    elif str(render_mode) == RENDER_MODE_LINE_CLEAR:
        line_params = replace(params, cell_size_px=int(params.line_cell_size_px))
        board_w, board_h = board_panel_size(line_params, board_rows=board_rows, board_cols=board_cols)
        preview_w, preview_h = piece_preview_panel_size(line_params)
        total_h = preview_h + int(line_params.board_gap_px) + board_h
        left = int((int(line_params.canvas_width) - board_w) / 2.0)
        top = int((int(line_params.canvas_height) - total_h) / 2.0)
        group_bbox = (float(left), float(top), float(left + board_w), float(top + total_h))
        _shifted, dx, dy, _resolved = apply_games_layout_jitter_to_bbox(bbox_px=group_bbox, canvas_width=int(line_params.canvas_width), canvas_height=int(line_params.canvas_height), jitter=line_params.layout_jitter_meta)
        left = int(left + round(dx))
        top = int(top + round(dy))
        preview_bbox = (int(left + (board_w - preview_w) / 2.0), int(top), int(left + (board_w - preview_w) / 2.0 + preview_w), int(top + preview_h))
        panel_bbox = (int(left), int(top + preview_h + int(line_params.board_gap_px)), int(left + board_w), int(top + preview_h + int(line_params.board_gap_px) + board_h))
        preview_map, preview_entities = draw_piece_preview_panel(image, panel_bbox=preview_bbox, label="NEXT", piece=sample.piece, orientation_index=int(sample.preview_orientation_index), style=style, params=line_params, entity_id="next_piece", style_variant=str(style_variant))
        entities.extend(preview_entities)
        render_map["panels"]["next_piece"] = preview_map["panel_bbox_px"]
        render_map["option_bboxes_px"]["next_piece"] = preview_map["panel_bbox_px"]
        render_map["cell_bboxes_px"].update(preview_map["cell_bboxes_px"])
        panel_map, panel_entities = draw_board_panel(image, panel_bbox=panel_bbox, label="BOARD", board=sample.board, style=style, params=line_params, entity_prefix="main", style_variant=str(style_variant))
        entities.extend(panel_entities)
        render_map["panels"]["main"] = panel_map["panel_bbox_px"]
        render_map["option_bboxes_px"]["main"] = panel_map["panel_bbox_px"]
        render_map["cell_bboxes_px"].update(panel_map["cell_bboxes_px"])
        render_map["row_bboxes_px"].update(panel_map["row_bboxes_px"])
    else:
        grid_params = result_option_grid_params(params, panel_count=1 + len(sample.options), board_rows=board_rows, board_cols=board_cols)
        start_bbox, option_panel_bboxes = result_option_panel_bboxes(grid_params, board_rows=board_rows, board_cols=board_cols)
        start_map, start_entities = draw_board_panel(image, panel_bbox=start_bbox, label="START", board=sample.board, style=style, params=grid_params, falling_cells=piece_cells(sample.falling_placement) if sample.falling_placement is not None else (), falling_piece=sample.piece, entity_prefix="start", style_variant=str(style_variant))
        entities.extend(start_entities)
        render_map["panels"]["start"] = list(start_map["panel_bbox_px"])
        render_map["cell_bboxes_px"].update(start_map["cell_bboxes_px"])
        render_map["row_bboxes_px"].update(start_map["row_bboxes_px"])
        render_map["result_option_layout"] = {"start_row": 0, "option_grid_rows": 2, "option_grid_cols": 2}
        for option, panel_bbox in zip(sample.options, option_panel_bboxes):
            panel_map, panel_entities = draw_board_panel(image, panel_bbox=panel_bbox, label=str(option.label), board=option.board, style=style, params=grid_params, entity_prefix=f"option_{option.label.lower()}", style_variant=str(style_variant))
            entities.extend(panel_entities)
            render_map["option_bboxes_px"][option.entity_id] = list(panel_map["panel_bbox_px"])
            render_map["panels"][option.entity_id] = list(panel_map["panel_bbox_px"])
            render_map["cell_bboxes_px"].update(panel_map["cell_bboxes_px"])
            render_map["row_bboxes_px"].update(panel_map["row_bboxes_px"])
    return RenderedTetrisScene(
        image=image,
        entities=tuple(entities),
        render_map=render_map,
        style_meta={"panel_scene_style": dict(panel_style_meta), "tetris_board_style": dict(render_map["tetris_board_style"])},
        background_meta=dict(background_meta),
    )
