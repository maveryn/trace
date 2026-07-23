"""Rendering helpers for rule-override board scene-package tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.games.shared.layout import (
    apply_games_layout_jitter_to_bbox,
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from trace_tasks.tasks.games.shared.scene_style import (
    draw_panel_scene_chrome,
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from trace_tasks.tasks.games.shared.text import draw_centered_game_text_traced as draw_centered_traced_text
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.text_rendering import fit_font_to_box, load_font

from .defaults import DEFAULTS
from .state import (
    LINE_BOARD_FAMILY,
    BoardPanel,
    RenderedRuleOverrideScene,
    RuleOverrideAxes,
    RuleOverrideRenderParams,
    RuleOverrideSceneSample,
    SCENE_NAMESPACE,
)


def board_grid_dimensions(board_count: int) -> tuple[int, int]:
    """Return display columns and rows for one mini-board collection."""

    count = max(1, int(board_count))
    cols = 2 if count <= 4 else 3
    rows = int(math.ceil(float(count) / float(cols)))
    return int(cols), int(rows)


def resolve_render_params(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    axes: RuleOverrideAxes,
) -> RuleOverrideRenderParams:
    """Resolve dimensions, font, and jitter for one rendered scene sample."""

    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            render_defaults,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.layout",
        ),
        unit_scale_meta,
    )
    cell_size = scale_games_px(
        params.get("cell_size_px", group_default(render_defaults, "cell_size_px", DEFAULTS.cell_size_px)),
        unit_scale,
        min_px=28,
    )
    board_gap = scale_games_px(
        params.get("board_gap_px", group_default(render_defaults, "board_gap_px", DEFAULTS.board_gap_px)),
        unit_scale,
        min_px=14,
    )
    board_padding = scale_games_px(
        params.get("board_padding_px", group_default(render_defaults, "board_padding_px", DEFAULTS.board_padding_px)),
        unit_scale,
        min_px=9,
    )
    board_label_height = scale_games_px(
        params.get(
            "board_label_height_px",
            group_default(render_defaults, "board_label_height_px", DEFAULTS.board_label_height_px),
        ),
        unit_scale,
        min_px=27,
    )
    cols, rows = board_grid_dimensions(int(axes.board_count))
    board_panel_w = (int(axes.board_size) * int(cell_size)) + (2 * int(board_padding))
    board_panel_h = int(board_label_height) + (int(axes.board_size) * int(cell_size)) + int(board_padding)
    content_w = (int(cols) * int(board_panel_w)) + ((int(cols) - 1) * int(board_gap))
    content_h = (int(rows) * int(board_panel_h)) + ((int(rows) - 1) * int(board_gap))
    content_margin = int(group_default(render_defaults, "content_margin_px", DEFAULTS.content_margin_px))
    min_canvas_width = int(group_default(render_defaults, "min_canvas_width_px", DEFAULTS.min_canvas_width_px))
    min_canvas_height = int(group_default(render_defaults, "min_canvas_height_px", DEFAULTS.min_canvas_height_px))
    canvas_width = int(params.get("canvas_width", max(min_canvas_width, content_w + (2 * content_margin))))
    canvas_height = int(params.get("canvas_height", max(min_canvas_height, content_h + (2 * content_margin))))
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.font_family",
        params=params,
    )
    return RuleOverrideRenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        cell_size_px=int(cell_size),
        board_gap_px=int(board_gap),
        board_padding_px=int(board_padding),
        board_label_height_px=int(board_label_height),
        content_margin_px=int(content_margin),
        panel_radius_px=scale_games_px(
            params.get("panel_radius_px", group_default(render_defaults, "panel_radius_px", DEFAULTS.panel_radius_px)),
            unit_scale,
            min_px=8,
        ),
        panel_border_width_px=scale_games_px(
            params.get(
                "panel_border_width_px",
                group_default(render_defaults, "panel_border_width_px", DEFAULTS.panel_border_width_px),
            ),
            unit_scale,
            min_px=1,
        ),
        board_border_width_px=scale_games_px(
            params.get(
                "board_border_width_px",
                group_default(render_defaults, "board_border_width_px", DEFAULTS.board_border_width_px),
            ),
            unit_scale,
            min_px=2,
        ),
        grid_width_px=scale_games_px(
            params.get("grid_width_px", group_default(render_defaults, "grid_width_px", DEFAULTS.grid_width_px)),
            unit_scale,
            min_px=1,
        ),
        board_label_font_size_px=scale_games_px(
            params.get(
                "board_label_font_size_px",
                group_default(render_defaults, "board_label_font_size_px", DEFAULTS.board_label_font_size_px),
            ),
            unit_scale,
            min_px=15,
        ),
        mark_font_size_px=scale_games_px(
            params.get("mark_font_size_px", group_default(render_defaults, "mark_font_size_px", DEFAULTS.mark_font_size_px)),
            unit_scale,
            min_px=23,
        ),
        font_family=str(font_family),
        layout_jitter_meta=dict(layout_jitter),
        unit_size_meta=dict(unit_scale_meta),
    )


def make_rule_override_background(
    *,
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    render_params: RuleOverrideRenderParams,
) -> tuple[Image.Image, Mapping[str, Any], Any, Mapping[str, Any]]:
    """Resolve panel scene style and background image for one sample."""

    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.panel_scene",
        treatment_weights=group_default(gen_defaults, "panel_scene_treatment_weights", None),
        palette_weights=group_default(gen_defaults, "panel_scene_palette_weights", None),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    return background, background_meta, panel_style, panel_style_meta


def theme(style_name: str) -> Dict[str, Tuple[int, int, int]]:
    """Return local board colors for one nonsemantic board style."""

    themes = {
        "classic": {
            "panel": (244, 239, 223),
            "grid": (45, 49, 58),
            "cell_a": (250, 247, 238),
            "cell_b": (231, 224, 205),
            "x": (45, 91, 188),
            "o": (45, 91, 188),
            "black": (36, 39, 48),
            "white": (244, 245, 238),
            "accent": (80, 118, 88),
        },
        "paper": {
            "panel": (248, 250, 244),
            "grid": (86, 94, 105),
            "cell_a": (255, 255, 250),
            "cell_b": (235, 241, 246),
            "x": (35, 82, 121),
            "o": (35, 82, 121),
            "black": (50, 54, 60),
            "white": (249, 250, 244),
            "accent": (106, 133, 167),
        },
        "chalkboard": {
            "panel": (35, 67, 55),
            "grid": (219, 236, 219),
            "cell_a": (44, 83, 68),
            "cell_b": (40, 75, 62),
            "x": (243, 230, 118),
            "o": (243, 230, 118),
            "black": (24, 31, 30),
            "white": (232, 235, 218),
            "accent": (231, 237, 198),
        },
        "arcade": {
            "panel": (24, 30, 56),
            "grid": (87, 219, 237),
            "cell_a": (31, 39, 72),
            "cell_b": (39, 48, 87),
            "x": (255, 95, 158),
            "o": (255, 95, 158),
            "black": (20, 22, 36),
            "white": (232, 246, 255),
            "accent": (252, 220, 88),
        },
        "wood": {
            "panel": (168, 116, 74),
            "grid": (79, 50, 36),
            "cell_a": (218, 170, 107),
            "cell_b": (188, 132, 82),
            "x": (44, 74, 127),
            "o": (44, 74, 127),
            "black": (34, 29, 25),
            "white": (245, 232, 198),
            "accent": (91, 62, 42),
        },
    }
    return dict(themes.get(str(style_name), themes["classic"]))


def draw_line_board(
    draw: ImageDraw.ImageDraw,
    *,
    board: BoardPanel,
    grid_bbox: Tuple[int, int, int, int],
    cell_size: int,
    grid_width: int,
    colors: Mapping[str, Tuple[int, int, int]],
    font_family: str,
    mark_font_size: int,
) -> None:
    """Draw one X/O anti-line mini-board grid."""

    size = len(board.cells)
    mark_font = fit_font_to_box(
        draw,
        text="X",
        max_width=float(cell_size),
        max_height=float(cell_size),
        bold=True,
        font_family=font_family,
        min_size_px=max(12, int(mark_font_size) // 2),
        max_size_px=int(mark_font_size),
        fill_ratio=0.74,
    )
    for row in range(size):
        for col in range(size):
            x0 = int(grid_bbox[0] + (col * int(cell_size)))
            y0 = int(grid_bbox[1] + (row * int(cell_size)))
            x1 = int(x0 + int(cell_size))
            y1 = int(y0 + int(cell_size))
            fill = colors["cell_a"] if (row + col) % 2 == 0 else colors["cell_b"]
            draw.rectangle((x0, y0, x1, y1), fill=fill)
            value = str(board.cells[row][col])
            if value:
                color = colors["x"] if value == "X" else colors["o"]
                draw_centered_traced_text(
                    draw,
                    center=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                    text=value,
                    font=mark_font,
                    fill_rgb=color,
                    stroke_rgb=colors["panel"],
                    stroke_width=1,
                    role="board_mark",
                    required=False,
                    extra_metadata={"board_id": str(board.board_id), "cell": [int(row), int(col)]},
                )
    for index in range(size + 1):
        x = int(grid_bbox[0] + (index * int(cell_size)))
        y = int(grid_bbox[1] + (index * int(cell_size)))
        draw.line((x, grid_bbox[1], x, grid_bbox[3]), fill=colors["grid"], width=int(grid_width))
        draw.line((grid_bbox[0], y, grid_bbox[2], y), fill=colors["grid"], width=int(grid_width))


def draw_piece_board(
    draw: ImageDraw.ImageDraw,
    *,
    board: BoardPanel,
    grid_bbox: Tuple[int, int, int, int],
    cell_size: int,
    grid_width: int,
    colors: Mapping[str, Tuple[int, int, int]],
) -> None:
    """Draw one black/white piece-count mini-board."""

    size = len(board.cells)
    for row in range(size):
        for col in range(size):
            x0 = int(grid_bbox[0] + (col * int(cell_size)))
            y0 = int(grid_bbox[1] + (row * int(cell_size)))
            x1 = int(x0 + int(cell_size))
            y1 = int(y0 + int(cell_size))
            fill = colors["cell_a"] if (row + col) % 2 == 0 else colors["cell_b"]
            draw.rectangle((x0, y0, x1, y1), fill=fill)
            value = str(board.cells[row][col])
            if value:
                inset = max(5, int(round(float(cell_size) * 0.18)))
                piece_bbox = (x0 + inset, y0 + inset, x1 - inset, y1 - inset)
                piece_fill = colors["black"] if value == "Black" else colors["white"]
                piece_outline = colors["white"] if value == "Black" else colors["black"]
                draw.ellipse(piece_bbox, fill=piece_fill, outline=piece_outline, width=max(2, int(grid_width)))
    for index in range(size + 1):
        x = int(grid_bbox[0] + (index * int(cell_size)))
        y = int(grid_bbox[1] + (index * int(cell_size)))
        draw.line((x, grid_bbox[1], x, grid_bbox[3]), fill=colors["grid"], width=int(grid_width))
        draw.line((grid_bbox[0], y, grid_bbox[2], y), fill=colors["grid"], width=int(grid_width))


def render_rule_override_scene(
    *,
    sample: RuleOverrideSceneSample,
    params: RuleOverrideRenderParams,
    panel_style: Any,
    background: Image.Image,
) -> RenderedRuleOverrideScene:
    """Render all mini-boards and emit panel bbox projections for annotations."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    colors = theme(str(sample.board_style))
    cols, rows = board_grid_dimensions(len(sample.boards))
    board_panel_w = (int(sample.board_size) * int(params.cell_size_px)) + (2 * int(params.board_padding_px))
    board_panel_h = int(params.board_label_height_px) + (int(sample.board_size) * int(params.cell_size_px)) + int(params.board_padding_px)
    content_w = (cols * board_panel_w) + ((cols - 1) * int(params.board_gap_px))
    content_h = (rows * board_panel_h) + ((rows - 1) * int(params.board_gap_px))
    natural_bbox = (
        int(params.content_margin_px),
        int(params.content_margin_px),
        int(params.content_margin_px + content_w),
        int(params.content_margin_px + content_h),
    )
    content_bbox, dx, dy, jitter_meta = apply_games_layout_jitter_to_bbox(
        bbox_px=natural_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    content_bbox_i = tuple(int(round(value)) for value in content_bbox)
    outer_bbox = (
        content_bbox_i[0] - 18,
        content_bbox_i[1] - 18,
        content_bbox_i[2] + 18,
        content_bbox_i[3] + 18,
    )
    draw_panel_scene_chrome(
        draw,
        bbox=outer_bbox,
        style=panel_style,
        radius=int(params.panel_radius_px),
        border_width=int(params.panel_border_width_px),
    )
    entity_bboxes: dict[str, list[float]] = {}
    entities: list[dict[str, Any]] = []
    label_font = load_font(int(params.board_label_font_size_px), bold=True, font_family=str(params.font_family))
    grid_top0 = int(content_bbox_i[1])
    for index, board in enumerate(sample.boards):
        row = index // cols
        col = index % cols
        row_item_count = min(int(cols), max(0, len(sample.boards) - (int(row) * int(cols))))
        row_width = (int(row_item_count) * int(board_panel_w)) + ((int(row_item_count) - 1) * int(params.board_gap_px))
        row_offset = int(round((float(content_w) - float(row_width)) / 2.0))
        left = int(content_bbox_i[0] + row_offset + (col * (board_panel_w + int(params.board_gap_px))))
        top = int(grid_top0 + (row * (board_panel_h + int(params.board_gap_px))))
        panel_bbox = (left, top, left + board_panel_w, top + board_panel_h)
        draw.rounded_rectangle(
            panel_bbox,
            radius=12,
            fill=colors["panel"],
            outline=colors["accent"],
            width=int(params.board_border_width_px),
        )
        draw_centered_traced_text(
            draw,
            center=((panel_bbox[0] + panel_bbox[2]) / 2.0, panel_bbox[1] + (int(params.board_label_height_px) / 2.0)),
            text=str(board.label),
            font=label_font,
            fill_rgb=colors["grid"],
            stroke_rgb=colors["panel"],
            stroke_width=1,
            role="mini_board_label",
            required=False,
            extra_metadata={"board_id": str(board.board_id)},
        )
        grid_bbox = (
            int(left + int(params.board_padding_px)),
            int(top + int(params.board_label_height_px)),
            int(left + int(params.board_padding_px) + (int(sample.board_size) * int(params.cell_size_px))),
            int(top + int(params.board_label_height_px) + (int(sample.board_size) * int(params.cell_size_px))),
        )
        if str(sample.board_family) == LINE_BOARD_FAMILY:
            draw_line_board(
                draw,
                board=board,
                grid_bbox=grid_bbox,
                cell_size=int(params.cell_size_px),
                grid_width=int(params.grid_width_px),
                colors=colors,
                font_family=str(params.font_family),
                mark_font_size=int(params.mark_font_size_px),
            )
        else:
            draw_piece_board(
                draw,
                board=board,
                grid_bbox=grid_bbox,
                cell_size=int(params.cell_size_px),
                grid_width=int(params.grid_width_px),
                colors=colors,
            )
        entity_bboxes[str(board.board_id)] = [float(value) for value in panel_bbox]
        entities.append(
            {
                "entity_id": str(board.board_id),
                "entity_type": "mini_board",
                "label": str(board.label),
                "bbox_px": [float(value) for value in panel_bbox],
                "grid_bbox_px": [float(value) for value in grid_bbox],
                "counted": bool(board.counted),
                "result": str(board.result),
            }
        )
    return RenderedRuleOverrideScene(
        image=image,
        render_map={
            "entity_bboxes_px": entity_bboxes,
            "layout_jitter": {
                **dict(jitter_meta),
                "dx_px": round(float(dx), 3),
                "dy_px": round(float(dy), 3),
                "unit_size_jitter": dict(params.unit_size_meta),
            },
            "board_style": str(sample.board_style),
            "text_style": {"font_family": str(params.font_family)},
        },
        scene_entities=tuple(entities),
    )


__all__ = [
    "board_grid_dimensions",
    "draw_line_board",
    "draw_piece_board",
    "make_rule_override_background",
    "render_rule_override_scene",
    "resolve_render_params",
    "theme",
]
