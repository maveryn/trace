"""Rendering and geometry projection for sliding-block boards."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.tasks.games.shared.layout import attach_games_unit_size_jitter, resolve_games_unit_size_scale, scale_games_px
from trace_tasks.tasks.games.shared.option_layout import balanced_option_grid_spec, option_grid_position, option_grid_size
from trace_tasks.tasks.games.shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style
from trace_tasks.tasks.shared.bbox_projection import bbox_union_raw as _bbox_union
from trace_tasks.tasks.shared.color_distance import coerce_rgb as _rgb
from trace_tasks.tasks.shared.drawing import draw_arrow, draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import RENDER_DEFAULTS
from .state import STYLE_COLORS, RenderedSlidingBlockScene, SlidingBlockRenderParams


def _int_value(mapping: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(mapping.get(str(key), int(fallback)))


def resolve_render_params(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
) -> SlidingBlockRenderParams:
    """Resolve render defaults, unit-size jitter, and explicit visual overrides."""

    merged = dict(RENDER_DEFAULTS)
    merged.update(dict(params))
    unit_scale, unit_meta = resolve_games_unit_size_scale(
        params,
        RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace="games.sliding_block.unit_size",
    )
    return SlidingBlockRenderParams(
        canvas_width=_int_value(merged, "canvas_width", 980),
        canvas_height=_int_value(merged, "canvas_height", 900),
        board_size_px=scale_games_px(_int_value(merged, "board_size_px", 660), unit_scale, min_px=320),
        panel_padding_px=scale_games_px(_int_value(merged, "panel_padding_px", 44), unit_scale, min_px=18),
        panel_corner_radius_px=scale_games_px(_int_value(merged, "panel_corner_radius_px", 28), unit_scale, min_px=10),
        block_corner_radius_px=scale_games_px(_int_value(merged, "block_corner_radius_px", 16), unit_scale, min_px=6),
        board_border_width_px=scale_games_px(_int_value(merged, "board_border_width_px", 5), unit_scale, min_px=2),
        grid_width_px=scale_games_px(_int_value(merged, "grid_width_px", 2), unit_scale, min_px=1),
        block_gap_px=scale_games_px(_int_value(merged, "block_gap_px", 7), unit_scale, min_px=3),
        target_outline_width_px=scale_games_px(_int_value(merged, "target_outline_width_px", 6), unit_scale, min_px=2),
        label_font_size_px=scale_games_px(_int_value(merged, "label_font_size_px", 30), unit_scale, min_px=14),
        target_label_font_size_px=scale_games_px(_int_value(merged, "target_label_font_size_px", 34), unit_scale, min_px=16),
        arrow_width_px=scale_games_px(_int_value(merged, "arrow_width_px", 8), unit_scale, min_px=3),
        arrow_head_length_px=scale_games_px(_int_value(merged, "arrow_head_length_px", 28), unit_scale, min_px=12),
        arrow_head_width_px=scale_games_px(_int_value(merged, "arrow_head_width_px", 28), unit_scale, min_px=12),
        panel_fill_rgb=None,
        board_fill_rgb=None,
        grid_rgb=None,
        border_rgb=None,
        path_rgb=None,
        exit_rgb=None,
        text_color_rgb=_rgb(merged.get("text_color_rgb"), (28, 32, 38)),
        text_stroke_rgb=_rgb(merged.get("text_stroke_rgb"), (255, 255, 255)),
        unit_size_jitter=dict(unit_meta),
    )


def apply_panel_style(
    render_params: SlidingBlockRenderParams,
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[SlidingBlockRenderParams, Image.Image, dict[str, Any], dict[str, Any]]:
    """Apply the shared games panel style and return a drawable background."""

    scene_style, scene_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    styled_params = replace(
        render_params,
        panel_fill_rgb=tuple(int(value) for value in scene_style.panel_fill_rgb),
        board_fill_rgb=tuple(int(value) for value in scene_style.option_fill_rgb),
        grid_rgb=tuple(int(value) for value in scene_style.grid_rgb),
        border_rgb=tuple(int(value) for value in scene_style.panel_border_rgb),
        path_rgb=tuple(int(value) for value in scene_style.step_fill_rgb),
        exit_rgb=tuple(int(value) for value in scene_style.mark_rgb),
        text_color_rgb=tuple(int(value) for value in scene_style.text_rgb),
        text_stroke_rgb=tuple(int(value) for value in scene_style.text_stroke_rgb),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(styled_params.canvas_width),
        canvas_height=int(styled_params.canvas_height),
        style=scene_style,
    )
    return styled_params, background, dict(background_meta), dict(scene_style_meta)


def style_colors(scene_variant: str, render_params: SlidingBlockRenderParams) -> dict[str, tuple[int, int, int]]:
    """Return style colors after per-instance panel overrides are applied."""

    colors = dict(STYLE_COLORS[str(scene_variant)])
    overrides = {
        "panel_fill": render_params.panel_fill_rgb,
        "board_fill": render_params.board_fill_rgb,
        "grid": render_params.grid_rgb,
        "border": render_params.border_rgb,
        "path": render_params.path_rgb,
        "exit": render_params.exit_rgb,
    }
    for key, value in overrides.items():
        if value is not None:
            colors[str(key)] = tuple(int(component) for component in value)
    return colors


def cell_bbox(
    *,
    board_left: float,
    board_top: float,
    cell_size: float,
    row: int,
    col: int,
    row_span: int = 1,
    col_span: int = 1,
    inset: float = 0.0,
) -> list[float]:
    """Project one board cell or rectangular cell span into pixel coordinates."""

    return [
        round(float(board_left + (int(col) * float(cell_size)) + float(inset)), 3),
        round(float(board_top + (int(row) * float(cell_size)) + float(inset)), 3),
        round(float(board_left + ((int(col) + int(col_span)) * float(cell_size)) - float(inset)), 3),
        round(float(board_top + ((int(row) + int(row_span)) * float(cell_size)) - float(inset)), 3),
    ]


def exit_arrow_points(
    *,
    exit_side: str,
    board_bbox: Sequence[float],
    target_block: Mapping[str, Any],
    cell_size: float,
) -> tuple[tuple[float, float], tuple[float, float], list[float]]:
    """Project the target block's exit arrow endpoints and enclosing bbox."""

    x0, y0, x1, y1 = [float(value) for value in board_bbox]
    row = int(target_block["row"])
    col = int(target_block["col"])
    height = int(target_block["height"])
    width = int(target_block["width"])
    if str(exit_side) == "right":
        cy = y0 + (float(row) + (0.5 * float(height))) * float(cell_size)
        start, end = (x1 - 8.0, cy), (x1 + 72.0, cy)
    elif str(exit_side) == "left":
        cy = y0 + (float(row) + (0.5 * float(height))) * float(cell_size)
        start, end = (x0 + 8.0, cy), (x0 - 72.0, cy)
    elif str(exit_side) == "bottom":
        cx = x0 + (float(col) + (0.5 * float(width))) * float(cell_size)
        start, end = (cx, y1 - 8.0), (cx, y1 + 72.0)
    else:
        cx = x0 + (float(col) + (0.5 * float(width))) * float(cell_size)
        start, end = (cx, y0 + 8.0), (cx, y0 - 72.0)
    arrow_bbox = _bbox_union([[start[0], start[1], end[0], end[1]]])
    return start, end, arrow_bbox


def _draw_board_at(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: Mapping[str, Any],
    blocks: Sequence[Mapping[str, Any]],
    board_bbox: Sequence[float],
    scene_variant: str,
    render_params: SlidingBlockRenderParams,
    show_path: bool,
    show_arrow: bool,
    label_font_size: int,
    target_label_font_size: int,
    block_gap_px: float,
    entity_prefix: str,
) -> tuple[dict[str, list[float]], list[dict[str, Any]], list[float], list[float], list[float]]:
    """Draw one board at an arbitrary bbox and return its projected block geometry."""

    rows = int(dataset["rows"])
    cols = int(dataset["cols"])
    x0, y0, x1, y1 = [float(value) for value in board_bbox]
    cell_size = min((x1 - x0) / float(cols), (y1 - y0) / float(rows))
    board_width = float(cell_size * int(cols))
    board_height = float(cell_size * int(rows))
    board_left = x0 + ((x1 - x0) - board_width) / 2.0
    board_top = y0 + ((y1 - y0) - board_height) / 2.0
    board_rect = [
        round(board_left, 3),
        round(board_top, 3),
        round(board_left + board_width, 3),
        round(board_top + board_height, 3),
    ]
    colors = style_colors(str(scene_variant), render_params)
    draw_rounded_rect(
        draw,
        tuple(board_rect),
        radius=max(6, int(18 * (float(cell_size) / 80.0))),
        fill=colors["board_fill"],
        outline=colors["border"],
        width=max(1, int(render_params.board_border_width_px)),
    )
    path_cell_boxes = [
        cell_bbox(
            board_left=board_left,
            board_top=board_top,
            cell_size=cell_size,
            row=int(cell[0]),
            col=int(cell[1]),
            inset=max(1.0, float(cell_size) * 0.04),
        )
        for cell in dataset["target_path_cells"]
    ]
    if bool(show_path):
        for bbox in path_cell_boxes:
            draw.rounded_rectangle(tuple(bbox), radius=max(2, int(cell_size * 0.10)), fill=tuple(colors["path"]))

    for row in range(1, int(rows)):
        y = board_top + (float(row) * cell_size)
        draw.line([(board_left, y), (board_left + board_width, y)], fill=tuple(colors["grid"]), width=max(1, int(render_params.grid_width_px)))
    for col in range(1, int(cols)):
        x = board_left + (float(col) * cell_size)
        draw.line([(x, board_top), (x, board_top + board_height)], fill=tuple(colors["grid"]), width=max(1, int(render_params.grid_width_px)))

    block_bbox_map: dict[str, list[float]] = {}
    entities: list[dict[str, Any]] = [
        {
            "entity_id": f"{entity_prefix}board",
            "entity_type": "sliding_block",
            "bbox_px": list(board_rect),
            "rows": int(rows),
            "cols": int(cols),
        }
    ]
    label_font = load_font(max(10, int(label_font_size)), bold=True)
    target_font = load_font(max(10, int(target_label_font_size)), bold=True)
    for block in blocks:
        bbox = cell_bbox(
            board_left=board_left,
            board_top=board_top,
            cell_size=cell_size,
            row=int(block["row"]),
            col=int(block["col"]),
            row_span=int(block["height"]),
            col_span=int(block["width"]),
            inset=float(block_gap_px),
        )
        fill = tuple(int(value) for value in block["fill_rgb"])
        outline = (30, 35, 42) if str(block["role"]) == "target" else colors["border"]
        line_width = max(1, int(render_params.target_outline_width_px if str(block["role"]) == "target" else 2))
        draw_rounded_rect(
            draw,
            tuple(bbox),
            radius=max(4, min(int(render_params.block_corner_radius_px), int(cell_size * 0.18))),
            fill=fill,
            outline=outline,
            width=int(line_width),
        )
        font = target_font if str(block["role"]) == "target" else label_font
        draw_centered_text(
            draw,
            text=str(block["label"]),
            center=((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0),
            font=font,
            fill=tuple(render_params.text_color_rgb),
            stroke_fill=tuple(render_params.text_stroke_rgb),
            stroke_width=2 if int(label_font_size) >= 18 else 1,
        )
        block_bbox_map[str(block["block_id"])] = list(bbox)
        entities.append(
            {
                "entity_id": f"{entity_prefix}{block['block_id']}",
                "entity_type": "sliding_block",
                "label": str(block["label"]),
                "role": str(block["role"]),
                "bbox_px": list(bbox),
                "row": int(block["row"]),
                "col": int(block["col"]),
                "height": int(block["height"]),
                "width": int(block["width"]),
            }
        )

    exit_arrow_bbox: list[float] = [0.0, 0.0, 0.0, 0.0]
    if bool(show_arrow):
        target_block = next(block for block in blocks if str(block["block_id"]) == "target")
        arrow_start, arrow_end, _arrow_bbox = exit_arrow_points(
            exit_side=str(dataset["exit_side"]),
            board_bbox=board_rect,
            target_block=target_block,
            cell_size=cell_size,
        )
        draw_arrow(
            draw,
            start=arrow_start,
            end=arrow_end,
            fill=colors["exit"],
            width=max(2, int(render_params.arrow_width_px)),
            head_length_px=float(render_params.arrow_head_length_px),
            head_width_px=float(render_params.arrow_head_width_px),
        )
        exit_arrow_bbox = [
            round(min(float(arrow_start[0]), float(arrow_end[0])) - 18.0, 3),
            round(min(float(arrow_start[1]), float(arrow_end[1])) - 18.0, 3),
            round(max(float(arrow_start[0]), float(arrow_end[0])) + 18.0, 3),
            round(max(float(arrow_start[1]), float(arrow_end[1])) + 18.0, 3),
        ]
        entities.append(
            {
                "entity_id": f"{entity_prefix}exit_arrow",
                "entity_type": "sliding_block_exit_arrow",
                "exit_side": str(dataset["exit_side"]),
                "bbox_px": list(exit_arrow_bbox),
            }
        )
    path_bbox = _bbox_union(path_cell_boxes) if bool(show_path) else [0.0, 0.0, 0.0, 0.0]
    return block_bbox_map, entities, path_bbox, exit_arrow_bbox, board_rect


def render_final_board_options(
    image: Image.Image,
    *,
    dataset: Mapping[str, Any],
    scene_variant: str,
    render_params: SlidingBlockRenderParams,
) -> RenderedSlidingBlockScene:
    """Render source board on top and visual final-board options below."""

    draw = ImageDraw.Draw(image)
    colors = style_colors(str(scene_variant), render_params)
    width = int(render_params.canvas_width)
    height = int(render_params.canvas_height)
    option_boards = [dict(option) for option in dataset.get("option_boards", [])]
    option_grid = balanced_option_grid_spec(len(option_boards))
    grid_cols = int(option_grid.columns)
    grid_rows = int(option_grid.rows) + 1
    margin_x = 48.0
    margin_y = 54.0
    gap_x = 24.0
    gap_y = 26.0
    panel_w = (float(width) - (2.0 * margin_x) - (float(grid_cols - 1) * gap_x)) / float(grid_cols)
    panel_h = (float(height) - (2.0 * margin_y) - (float(grid_rows - 1) * gap_y)) / float(grid_rows)
    board_size = min(panel_w - 44.0, panel_h - 76.0)
    option_area_width, _option_area_height = option_grid_size(
        len(option_boards),
        item_width=panel_w,
        item_height=panel_h,
        gap_x=gap_x,
        gap_y=gap_y,
        columns=grid_cols,
    )
    block_bbox_map: dict[str, list[float]] = {}
    option_panel_bbox_map: dict[str, list[float]] = {}
    entities: list[dict[str, Any]] = []
    panel_bboxes: list[list[float]] = []
    source_board_bbox = [0.0, 0.0, 0.0, 0.0]
    title_font = load_font(18, bold=True)
    panel_items = [
        {
            "option_id": "source",
            "label": "Original",
            "is_source": True,
            "blocks": list(dataset["blocks"]),
        },
        *option_boards,
    ]
    for index, option in enumerate(panel_items):
        if bool(option.get("is_source", False)):
            panel_left = margin_x + ((option_area_width - panel_w) / 2.0)
            panel_top = margin_y
        else:
            option_index = int(index) - 1
            _row, _col, panel_left, panel_top = option_grid_position(
                option_index,
                len(option_boards),
                left=margin_x,
                top=margin_y + panel_h + gap_y,
                item_width=panel_w,
                item_height=panel_h,
                gap_x=gap_x,
                gap_y=gap_y,
                columns=grid_cols,
            )
        panel_bbox = [
            round(panel_left, 3),
            round(panel_top, 3),
            round(panel_left + panel_w, 3),
            round(panel_top + panel_h, 3),
        ]
        option_id = str(option["option_id"])
        draw_rounded_rect(
            draw,
            tuple(panel_bbox),
            radius=18,
            fill=colors["panel_fill"],
            outline=colors["border"],
            width=2,
        )
        draw_centered_text(
            draw,
            text=str(option["label"]),
            center=(panel_left + (70.0 if bool(option.get("is_source", False)) else 42.0), panel_top + 25.0),
            font=title_font,
            fill=tuple(render_params.text_color_rgb),
            stroke_fill=tuple(render_params.text_stroke_rgb),
            stroke_width=1,
        )
        board_left = panel_left + ((panel_w - board_size) / 2.0)
        board_top = panel_top + 58.0
        rendered_blocks, rendered_entities, _path_bbox, _exit_bbox, _board_rect = _draw_board_at(
            draw,
            dataset=dataset,
            blocks=option["blocks"],
            board_bbox=[board_left, board_top, board_left + board_size, board_top + board_size],
            scene_variant=str(scene_variant),
            render_params=render_params,
            show_path=False,
            show_arrow=False,
            label_font_size=14,
            target_label_font_size=15,
            block_gap_px=max(2.0, float(render_params.block_gap_px) * 0.35),
            entity_prefix=f"{option_id}_",
        )
        entities.extend(rendered_entities)
        panel_bboxes.append(list(panel_bbox))
        if bool(option.get("is_source", False)):
            block_bbox_map = dict(rendered_blocks)
            source_board_bbox = [
                round(float(board_left), 3),
                round(float(board_top), 3),
                round(float(board_left + board_size), 3),
                round(float(board_top + board_size), 3),
            ]
            entities.append({"entity_id": "source_panel", "entity_type": "sliding_block_source_panel", "bbox_px": list(panel_bbox)})
        else:
            option_panel_bbox_map[option_id] = list(panel_bbox)
            entities.append(
                {
                    "entity_id": option_id,
                    "entity_type": "sliding_block_result_option",
                    "label": str(option["label"]),
                    "is_correct": bool(option.get("is_correct", False)),
                    "bbox_px": list(panel_bbox),
                }
            )

    return RenderedSlidingBlockScene(
        image=image,
        entities=entities,
        scene_bbox_px=list(_bbox_union(panel_bboxes)),
        board_bbox_px=list(source_board_bbox),
        path_bbox_px=[0.0, 0.0, 0.0, 0.0],
        exit_arrow_bbox_px=[0.0, 0.0, 0.0, 0.0],
        block_bbox_map=block_bbox_map,
        option_panel_bbox_map=option_panel_bbox_map,
    )


def render_single_board(
    image: Image.Image,
    *,
    dataset: Mapping[str, Any],
    scene_variant: str,
    render_params: SlidingBlockRenderParams,
    show_exit_path: bool,
) -> RenderedSlidingBlockScene:
    """Render the one-board count tasks with optional exit path and arrow cues."""

    draw = ImageDraw.Draw(image)
    rows = int(dataset["rows"])
    cols = int(dataset["cols"])
    max_dim = max(int(rows), int(cols))
    cell_size = float(render_params.board_size_px) / float(max_dim)
    board_width = float(cell_size * int(cols))
    board_height = float(cell_size * int(rows))
    board_left = float((int(render_params.canvas_width) - board_width) / 2.0)
    board_top = float((int(render_params.canvas_height) - board_height) / 2.0 + 16.0)
    panel_bbox = [
        round(board_left - int(render_params.panel_padding_px), 3),
        round(board_top - int(render_params.panel_padding_px), 3),
        round(board_left + board_width + int(render_params.panel_padding_px), 3),
        round(board_top + board_height + int(render_params.panel_padding_px), 3),
    ]
    colors = style_colors(str(scene_variant), render_params)
    draw_rounded_rect(
        draw,
        tuple(panel_bbox),
        radius=int(render_params.panel_corner_radius_px),
        fill=colors["panel_fill"],
        outline=colors["border"],
        width=max(1, int(render_params.board_border_width_px) - 1),
    )
    block_bboxes, entities, path_bbox, exit_arrow_bbox, board_rect = _draw_board_at(
        draw,
        dataset=dataset,
        blocks=dataset["blocks"],
        board_bbox=[board_left, board_top, board_left + board_width, board_top + board_height],
        scene_variant=str(scene_variant),
        render_params=render_params,
        show_path=bool(show_exit_path),
        show_arrow=bool(show_exit_path),
        label_font_size=int(render_params.label_font_size_px),
        target_label_font_size=int(render_params.target_label_font_size_px),
        block_gap_px=float(render_params.block_gap_px),
        entity_prefix="",
    )
    scene_bbox = _bbox_union([panel_bbox, exit_arrow_bbox]) if bool(show_exit_path) else list(panel_bbox)
    return RenderedSlidingBlockScene(
        image=image,
        entities=entities,
        scene_bbox_px=list(scene_bbox),
        board_bbox_px=list(board_rect),
        path_bbox_px=list(path_bbox),
        exit_arrow_bbox_px=list(exit_arrow_bbox),
        block_bbox_map=block_bboxes,
        option_panel_bbox_map={},
    )


def render_sliding_block_scene(
    image: Image.Image,
    *,
    dataset: Mapping[str, Any],
    scene_variant: str,
    render_params: SlidingBlockRenderParams,
    render_mode: str,
) -> RenderedSlidingBlockScene:
    """Dispatch between the scene's single-board and option-board render grammars."""

    if str(render_mode) == "final_board_options":
        return render_final_board_options(
            image,
            dataset=dataset,
            scene_variant=str(scene_variant),
            render_params=render_params,
        )
    return render_single_board(
        image,
        dataset=dataset,
        scene_variant=str(scene_variant),
        render_params=render_params,
        show_exit_path=str(render_mode) == "exit_path",
    )


def render_map_payload(
    *,
    rendered_scene: RenderedSlidingBlockScene,
    render_params: SlidingBlockRenderParams,
    annotation_source: str,
) -> dict[str, Any]:
    """Build a common render map and attach unit-size jitter metadata."""

    return attach_games_unit_size_jitter(
        {
            "image_id": "img0",
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "board_bbox_px": list(rendered_scene.board_bbox_px),
            "path_bbox_px": list(rendered_scene.path_bbox_px),
            "exit_arrow_bbox_px": list(rendered_scene.exit_arrow_bbox_px),
            "block_bboxes_px": {str(key): list(value) for key, value in rendered_scene.block_bbox_map.items()},
            "option_panel_bboxes_px": {str(key): list(value) for key, value in rendered_scene.option_panel_bbox_map.items()},
            "annotation_source": str(annotation_source),
        },
        render_params.unit_size_jitter,
    )


__all__ = [
    "apply_panel_style",
    "render_map_payload",
    "render_sliding_block_scene",
    "resolve_render_params",
    "style_colors",
]
