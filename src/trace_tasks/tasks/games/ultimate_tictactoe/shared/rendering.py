"""Rendering primitives for Ultimate Tic-Tac-Toe boards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Tuple

from PIL import ImageDraw

from trace_tasks.tasks.games.shared.layout import (
    apply_games_layout_jitter_to_bbox,
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from trace_tasks.tasks.games.shared.scene_style import (
    draw_panel_grid_cell,
    draw_panel_scene_chrome,
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from trace_tasks.tasks.games.shared.text import draw_game_text_traced as draw_text_traced
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .defaults import DEFAULTS, GEN_DEFAULTS, RENDER_DEFAULTS, int_render_default
from .rules import board_entity_id, cell_entity_id
from .state import (
    BBox,
    MACRO_LABELS,
    OPTION_LABELS,
    PLAYER_X,
    SUPPORTED_STYLE_VARIANTS,
    UltimateBoardVisualStyle,
    UltimateSample,
    RenderedUltimateScene,
)


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[float],
    text: str,
    *,
    font,
    fill: Tuple[int, int, int],
    stroke_width: int = 1,
) -> None:
    """Draw centered text while preserving text tracing and contrast stroke rules."""

    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=int(stroke_width))
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    x0, y0, x1, y1 = [float(value) for value in bbox]
    draw_text_traced(
        draw,
        (
            float(x0 + ((x1 - x0) - text_w) / 2.0 - float(text_bbox[0])),
            float(y0 + ((y1 - y0) - text_h) / 2.0 - float(text_bbox[1])),
        ),
        str(text),
        font=font,
        fill=tuple(fill),
        stroke_width=int(stroke_width),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(fill)),
        role="readout",
        required=False,
    )


def _rgb(values: Sequence[int]) -> Tuple[int, int, int]:
    return tuple(max(0, min(255, int(value))) for value in values[:3])  # type: ignore[return-value]


def _blend_rgb(a: Sequence[int], b: Sequence[int], alpha: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, float(alpha)))
    return tuple(int(round((float(a[index]) * (1.0 - t)) + (float(b[index]) * t))) for index in range(3))


def resolve_ultimate_board_visual_style(style_variant: str, panel_style) -> tuple[UltimateBoardVisualStyle, dict[str, Any]]:
    """Resolve scene-local board colors independent of the shared panel background."""

    panel_fill = _rgb(panel_style.panel_fill_rgb)
    panel_background = _rgb(panel_style.background_rgb)
    default_board = _blend_rgb(panel_fill, panel_background, 0.12)
    styles: dict[str, UltimateBoardVisualStyle] = {
        "classic_grid": UltimateBoardVisualStyle(
            cell_fill_rgb=_blend_rgb(panel_fill, (255, 255, 255), 0.22),
            board_fill_rgb=default_board,
            grid_rgb=(116, 132, 158),
            border_rgb=(46, 61, 92),
            highlight_rgb=(232, 162, 42),
            x_rgb=(42, 92, 205),
            o_rgb=(42, 92, 205),
            option_fill_rgb=(255, 232, 102),
            option_outline_rgb=(48, 63, 95),
            option_text_rgb=(32, 36, 45),
        ),
        "soft_marker": UltimateBoardVisualStyle(
            cell_fill_rgb=(238, 244, 247),
            board_fill_rgb=(226, 235, 239),
            grid_rgb=(133, 152, 158),
            border_rgb=(86, 103, 112),
            highlight_rgb=(78, 158, 151),
            x_rgb=(44, 118, 172),
            o_rgb=(44, 118, 172),
            option_fill_rgb=(239, 220, 143),
            option_outline_rgb=(65, 101, 109),
            option_text_rgb=(36, 45, 48),
        ),
        "paper_grid": UltimateBoardVisualStyle(
            cell_fill_rgb=(248, 241, 222),
            board_fill_rgb=(237, 226, 202),
            grid_rgb=(166, 133, 94),
            border_rgb=(95, 68, 48),
            highlight_rgb=(185, 108, 54),
            x_rgb=(45, 82, 130),
            o_rgb=(45, 82, 130),
            option_fill_rgb=(252, 218, 128),
            option_outline_rgb=(113, 79, 45),
            option_text_rgb=(52, 40, 31),
        ),
        "neon_board": UltimateBoardVisualStyle(
            cell_fill_rgb=(28, 34, 56),
            board_fill_rgb=(18, 23, 42),
            grid_rgb=(87, 119, 169),
            border_rgb=(73, 232, 237),
            highlight_rgb=(252, 211, 64),
            x_rgb=(87, 229, 244),
            o_rgb=(87, 229, 244),
            option_fill_rgb=(255, 224, 76),
            option_outline_rgb=(255, 255, 255),
            option_text_rgb=(28, 31, 39),
        ),
        "tournament_board": UltimateBoardVisualStyle(
            cell_fill_rgb=(224, 235, 227),
            board_fill_rgb=(207, 224, 211),
            grid_rgb=(94, 128, 104),
            border_rgb=(40, 91, 64),
            highlight_rgb=(216, 158, 59),
            x_rgb=(32, 93, 164),
            o_rgb=(32, 93, 164),
            option_fill_rgb=(247, 230, 134),
            option_outline_rgb=(36, 92, 67),
            option_text_rgb=(26, 44, 35),
        ),
    }
    resolved_variant = str(style_variant) if str(style_variant) in styles else "classic_grid"
    resolved = styles[resolved_variant]
    return resolved, {
        "style_variant": str(resolved_variant),
        "available_styles": list(SUPPORTED_STYLE_VARIANTS),
        "board_style_policy": "scene_local_ultimate_tictactoe_board_palette",
    }


def render_ultimate_tictactoe_scene(
    *,
    sample: UltimateSample,
    namespace: str,
    style_variant: str,
    instance_seed: int,
    params: Mapping[str, Any],
) -> RenderedUltimateScene:
    """Render the whole Ultimate board and expose entity bboxes for annotation."""

    base_canvas_width = int_render_default(params, "canvas_width", DEFAULTS.canvas_width)
    base_canvas_height = int_render_default(params, "canvas_height", DEFAULTS.canvas_height)
    style, style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.panel_style",
        treatment_weights=group_default(GEN_DEFAULTS, "panel_scene_treatment_weights", None),
        palette_weights=group_default(GEN_DEFAULTS, "panel_scene_palette_weights", None),
    )
    layout_jitter = resolve_games_layout_jitter(
        params,
        RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.layout",
    )
    unit_scale, unit_meta = resolve_games_unit_size_scale(
        params,
        RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.unit_size",
    )
    local_cell = scale_games_px(int_render_default(params, "local_cell_size_px", DEFAULTS.local_cell_size_px), unit_scale, min_px=32)
    local_gap = scale_games_px(int_render_default(params, "local_gap_px", DEFAULTS.local_gap_px), unit_scale, min_px=2)
    macro_gap = scale_games_px(int_render_default(params, "macro_gap_px", DEFAULTS.macro_gap_px), unit_scale, min_px=6)
    inner_margin = scale_games_px(int_render_default(params, "board_inner_margin_px", DEFAULTS.board_inner_margin_px), unit_scale, min_px=28)
    local_size = int(3 * local_cell + 2 * local_gap)
    grid_size = int(3 * local_size + 2 * macro_gap)
    raw_panel_size = int(grid_size + 2 * inner_margin)
    dynamic_canvas_enabled = bool(
        params.get(
            "dynamic_canvas_size_enabled",
            group_default(RENDER_DEFAULTS, "dynamic_canvas_size_enabled", True),
        )
    )
    canvas_width = int(base_canvas_width)
    canvas_height = int(base_canvas_height)
    if dynamic_canvas_enabled and params.get("canvas_width") is None:
        side_padding = int_render_default(params, "canvas_side_padding_px", 72)
        canvas_width = min(
            int(base_canvas_width),
            max(int_render_default(params, "canvas_min_width_px", 560), int(raw_panel_size + (2 * side_padding))),
        )
    if dynamic_canvas_enabled and params.get("canvas_height") is None:
        side_padding = int_render_default(params, "canvas_side_padding_px", 72)
        canvas_height = min(
            int(base_canvas_height),
            max(int_render_default(params, "canvas_min_height_px", 560), int(raw_panel_size + (2 * side_padding))),
        )
    panel_size = min(
        int(raw_panel_size),
        int(min(canvas_width, canvas_height) - 2 * int_render_default(params, "panel_margin_px", DEFAULTS.panel_margin_px)),
    )
    image, background_meta = make_panel_scene_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=style,
    )
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    base_panel = (
        float((canvas_width - panel_size) / 2.0),
        float((canvas_height - panel_size) / 2.0),
        float((canvas_width + panel_size) / 2.0),
        float((canvas_height + panel_size) / 2.0),
    )
    panel_bbox, _dx, _dy, resolved_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=base_panel,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        jitter=layout_jitter,
    )
    panel_bbox_i = tuple(int(round(value)) for value in panel_bbox)
    draw_panel_scene_chrome(draw, bbox=panel_bbox_i, style=style, radius=18, border_width=3)
    grid_left = int(round(panel_bbox[0])) + inner_margin
    grid_top = int(round(panel_bbox[1])) + inner_margin

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.font_family",
        params=params,
    )
    symbol_font = load_font(
        scale_games_px(int_render_default(params, "symbol_font_size_px", DEFAULTS.symbol_font_size_px), unit_scale, min_px=18),
        bold=True,
        font_family=str(font_family),
    )
    option_font = load_font(
        scale_games_px(int_render_default(params, "option_font_size_px", DEFAULTS.option_font_size_px), unit_scale, min_px=16),
        bold=True,
        font_family=str(font_family),
    )
    board_style, board_style_meta = resolve_ultimate_board_visual_style(str(style_variant), style)
    grid_rgb = tuple(int(value) for value in board_style.grid_rgb)
    border_rgb = tuple(int(value) for value in board_style.border_rgb)
    mark_rgb = tuple(int(value) for value in board_style.option_outline_rgb)
    open_rgb = tuple(int(value) for value in board_style.cell_fill_rgb)
    board_fill_rgb = tuple(int(value) for value in board_style.board_fill_rgb)
    x_rgb = tuple(int(value) for value in board_style.x_rgb)
    o_rgb = tuple(int(value) for value in board_style.o_rgb)
    highlight_rgb = tuple(int(value) for value in board_style.highlight_rgb)
    small_board_border_width = scale_games_px(
        int_render_default(params, "small_board_border_width_px", DEFAULTS.small_board_border_width_px),
        unit_scale,
        min_px=5,
    )

    entities: list[dict[str, Any]] = []
    entity_bboxes: dict[str, list[float]] = {}
    small_board_bboxes: dict[str, list[float]] = {}
    cell_bboxes_all: dict[str, list[float]] = {}
    board_cell_bboxes: dict[int, dict[int, BBox]] = {}

    for board_index, local in enumerate(sample.board):
        macro_row = int(board_index // 3)
        macro_col = int(board_index % 3)
        bx0 = int(grid_left + macro_col * (local_size + macro_gap))
        by0 = int(grid_top + macro_row * (local_size + macro_gap))
        board_bbox = (float(bx0), float(by0), float(bx0 + local_size), float(by0 + local_size))
        board_id = board_entity_id(int(board_index))
        draw.rounded_rectangle(board_bbox, radius=6, fill=board_fill_rgb, outline=None)
        board_bbox_list = [round(float(value), 3) for value in board_bbox]
        entity_bboxes[board_id] = list(board_bbox_list)
        small_board_bboxes[board_id] = list(board_bbox_list)
        entities.append(
            {
                "entity_id": str(board_id),
                "entity_type": "ultimate_tictactoe_small_board",
                "label": MACRO_LABELS[int(board_index)],
                "status": str(local.status),
                "bbox_px": list(board_bbox_list),
            }
        )
        board_cell_bboxes[int(board_index)] = {}
        for cell_index, value in enumerate(local.cells):
            row = int(cell_index // 3)
            col = int(cell_index % 3)
            x0 = int(bx0 + col * (local_cell + local_gap))
            y0 = int(by0 + row * (local_cell + local_gap))
            cell_bbox = (float(x0), float(y0), float(x0 + local_cell), float(y0 + local_cell))
            draw_panel_grid_cell(
                draw,
                bbox=tuple(int(round(v)) for v in cell_bbox),
                fill=open_rgb,
                style=style,
                outline=grid_rgb,
                width=1,
            )
            cell_id = cell_entity_id(int(board_index), int(cell_index))
            board_cell_bboxes[int(board_index)][int(cell_index)] = tuple(cell_bbox)
            cell_bbox_list = [round(float(v), 3) for v in cell_bbox]
            entity_bboxes[cell_id] = list(cell_bbox_list)
            cell_bboxes_all[cell_id] = list(cell_bbox_list)
            entities.append(
                {
                    "entity_id": str(cell_id),
                    "entity_type": "ultimate_tictactoe_cell",
                    "small_board": MACRO_LABELS[int(board_index)],
                    "cell_index": int(cell_index + 1),
                    "mark": str(value),
                    "bbox_px": list(cell_bbox_list),
                }
            )
            if value:
                _draw_centered_text(
                    draw,
                    cell_bbox,
                    str(value),
                    font=symbol_font,
                    fill=x_rgb if str(value) == PLAYER_X else o_rgb,
                    stroke_width=1,
                )
        draw.rounded_rectangle(board_bbox, radius=6, fill=None, outline=border_rgb, width=int(small_board_border_width))
        if sample.highlighted_board_index is not None and int(board_index) == int(sample.highlighted_board_index):
            hw = scale_games_px(int_render_default(params, "highlight_width_px", DEFAULTS.highlight_width_px), unit_scale, min_px=4)
            draw.rounded_rectangle(
                (board_bbox[0] - 5, board_bbox[1] - 5, board_bbox[2] + 5, board_bbox[3] + 5),
                radius=9,
                outline=highlight_rgb,
                width=int(hw),
            )
    if sample.highlighted_board_index is not None:
        for option_index, cell_index in enumerate(sample.option_cells):
            label = OPTION_LABELS[int(option_index)]
            bbox = board_cell_bboxes[int(sample.highlighted_board_index)][int(cell_index)]
            radius = max(14.0, local_cell * 0.3)
            label_bbox = (bbox[2] - 2 * radius - 3, bbox[1] + 3, bbox[2] - 3, bbox[1] + 2 * radius + 3)
            draw.ellipse(label_bbox, fill=tuple(board_style.option_fill_rgb), outline=mark_rgb, width=2)
            _draw_centered_text(draw, label_bbox, label, font=option_font, fill=tuple(board_style.option_text_rgb), stroke_width=0)

    render_map = {
        "entity_bboxes_px": dict(entity_bboxes),
        "small_board_bboxes_px": dict(small_board_bboxes),
        "cell_bboxes_px": dict(cell_bboxes_all),
        "grid_bbox_px": [float(grid_left), float(grid_top), float(grid_left + grid_size), float(grid_top + grid_size)],
        "style": dict(style_meta),
        "panel_scene_style": dict(style_meta),
        "ultimate_tictactoe_board_style": dict(board_style_meta),
        "font_family": str(font_family),
        "text_style": {"font_family": str(font_family)},
        "layout_jitter": attach_games_unit_size_jitter(resolved_jitter, unit_meta),
        "effective_local_cell_size_px": int(local_cell),
        "effective_local_gap_px": int(local_gap),
        "dynamic_canvas": {
            "enabled": bool(dynamic_canvas_enabled),
            "base_canvas_width": int(base_canvas_width),
            "base_canvas_height": int(base_canvas_height),
            "raw_panel_size_px": int(raw_panel_size),
            "resolved_canvas_width": int(canvas_width),
            "resolved_canvas_height": int(canvas_height),
        },
    }
    return RenderedUltimateScene(
        image=image.convert("RGB"),
        entities=tuple(entities),
        render_map=dict(render_map),
        style_meta={
            "panel_scene_style": dict(style_meta),
            "ultimate_tictactoe_board_style": dict(board_style_meta),
            "text_style": {
                "font_family": str(font_family),
                "font_asset": get_font_family_record(str(font_family)).to_trace(),
            },
        },
        background_meta=dict(background_meta),
    )
