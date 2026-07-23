"""Renderer for word-search puzzle tasks."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw

from trace_tasks.tasks.puzzles.shared.drawing import draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.puzzles.shared.word_grid import cell_key, option_key
from trace_tasks.tasks.shared.bbox_projection import round_bbox
from trace_tasks.tasks.shared.text_rendering import fit_font_to_box, load_font

from .sampling import option_text
from .state import RenderedWordSearch, WordSearchDataset, WordSearchRenderParams

_WORD_SEARCH_FONT_FAMILY = "source_sans_3"


def render_word_search_scene(
    image: Image.Image,
    *,
    dataset: WordSearchDataset,
    render_params: WordSearchRenderParams,
    rng,
) -> RenderedWordSearch:
    """Render the grid plus optional option/word-bank side panel."""

    draw = ImageDraw.Draw(image)
    rows = int(dataset.rows)
    cols = int(dataset.cols)
    cell = int(render_params.cell_size_px)
    header = int(render_params.header_size_px)
    padding = int(render_params.panel_padding_px)
    grid_w = int(header + cols * cell)
    grid_h = int(header + rows * cell)
    option_grid_w, option_grid_h = _option_grid_size(dataset, render_params)
    option_gap_y = int(render_params.option_gap_px) * 2 if dataset.option_specs else 0
    content_w = max(int(grid_w), int(option_grid_w))
    content_h = int(grid_h + option_gap_y + option_grid_h)
    panel_w = int(content_w + 2 * padding)
    panel_h = int(content_h + 2 * padding)
    canvas_margin = 34
    max_panel_x0 = max(
        canvas_margin,
        int(render_params.canvas_width) - canvas_margin - panel_w,
    )
    max_panel_y0 = max(
        canvas_margin,
        int(render_params.canvas_height) - canvas_margin - panel_h,
    )
    panel_x0 = int(
        canvas_margin + rng.randrange(max(1, max_panel_x0 - canvas_margin + 1))
    )
    panel_y0 = int(
        canvas_margin + rng.randrange(max(1, max_panel_y0 - canvas_margin + 1))
    )
    panel_x1 = int(panel_x0 + panel_w)
    panel_y1 = int(panel_y0 + panel_h)
    grid_x0 = int(panel_x0 + padding + max(0, (content_w - grid_w) // 2))
    grid_y0 = int(panel_y0 + padding)
    option_x0 = int(panel_x0 + padding + max(0, (content_w - option_grid_w) // 2))
    option_y0 = int(grid_y0 + grid_h + option_gap_y)
    option_columns, option_rows = _option_grid_shape(len(dataset.option_specs))
    layout_jitter = {
        "enabled": True,
        "panel_x0_px": int(panel_x0),
        "panel_y0_px": int(panel_y0),
        "grid_x0_px": int(grid_x0),
        "grid_y0_px": int(grid_y0),
        "option_x0_px": int(option_x0) if dataset.option_specs else None,
        "option_y0_px": int(option_y0) if dataset.option_specs else None,
        "option_columns": int(option_columns),
        "option_rows": int(option_rows),
        "available_x0_min_px": int(canvas_margin),
        "available_x0_max_px": int(max_panel_x0),
        "available_y0_min_px": int(canvas_margin),
        "available_y0_max_px": int(max_panel_y0),
    }

    draw_rounded_rect(
        draw,
        (panel_x0, panel_y0, panel_x1, panel_y1),
        radius=int(render_params.panel_corner_radius_px),
        fill=render_params.panel_fill_rgb,
        outline=render_params.grid_line_rgb,
        width=max(1, int(render_params.grid_line_width_px)),
    )
    if str(dataset.scene_variant) == "word_search_notebook":
        for y_pos in range(panel_y0 + 14, panel_y1 - 8, max(16, cell // 2)):
            draw.line(
                (panel_x0 + 8, y_pos, panel_x1 - 8, y_pos),
                fill=render_params.grid_line_rgb,
                width=1,
            )

    letter_font = load_font(
        int(render_params.letter_font_size_px),
        bold=True,
        font_family=_WORD_SEARCH_FONT_FAMILY,
    )
    index_font = load_font(
        int(render_params.index_font_size_px),
        bold=True,
        font_family=_WORD_SEARCH_FONT_FAMILY,
    )
    item_bbox_map: dict[str, list[float]] = {}
    cell_bbox_map: dict[str, list[float]] = {}
    cell_centers_px: dict[str, tuple[float, float]] = {}
    entities: list[dict[str, Any]] = [
        {
            "entity_id": "word_search_panel",
            "entity_type": "puzzle_word_search_panel",
            "bbox_px": round_bbox((panel_x0, panel_y0, panel_x1, panel_y1)),
            "scene_variant": str(dataset.scene_variant),
        }
    ]
    _draw_grid(
        draw,
        dataset=dataset,
        render_params=render_params,
        grid_x0=grid_x0,
        grid_y0=grid_y0,
        letter_font=letter_font,
        index_font=index_font,
        item_bbox_map=item_bbox_map,
        cell_bbox_map=cell_bbox_map,
        cell_centers_px=cell_centers_px,
        entities=entities,
    )
    if dataset.option_specs:
        _draw_option_cards(
            draw,
            dataset=dataset,
            render_params=render_params,
            option_x0=option_x0,
            option_y0=option_y0,
            item_bbox_map=item_bbox_map,
            entities=entities,
        )

    return RenderedWordSearch(
        image=image,
        entities=tuple(entities),
        scene_bbox_px=round_bbox((panel_x0, panel_y0, panel_x1, panel_y1)),
        item_bbox_map=dict(item_bbox_map),
        cell_bbox_map=dict(cell_bbox_map),
        cell_centers_px=dict(cell_centers_px),
        layout_jitter=dict(layout_jitter),
    )


def _draw_grid(
    draw,
    *,
    dataset: WordSearchDataset,
    render_params: WordSearchRenderParams,
    grid_x0: int,
    grid_y0: int,
    letter_font,
    index_font,
    item_bbox_map: dict[str, list[float]],
    cell_bbox_map: dict[str, list[float]],
    cell_centers_px: dict[str, tuple[float, float]],
    entities: list[dict[str, Any]],
) -> None:
    """Draw the labeled letter grid and trace every data cell."""

    rows = int(dataset.rows)
    cols = int(dataset.cols)
    cell = int(render_params.cell_size_px)
    header = int(render_params.header_size_px)
    for row in range(rows + 1):
        for col in range(cols + 1):
            bbox, fill = _cell_bbox_and_fill(
                row=row,
                col=col,
                grid_x0=grid_x0,
                grid_y0=grid_y0,
                header=header,
                cell=cell,
                render_params=render_params,
            )
            draw.rectangle(
                bbox,
                fill=fill,
                outline=render_params.grid_line_rgb,
                width=max(1, int(render_params.grid_line_width_px)),
            )
            if row == 0 and col > 0:
                draw_centered_text(
                    draw,
                    text=str(col),
                    center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                    font=index_font,
                    fill=render_params.text_rgb,
                    stroke_fill=render_params.text_rgb,
                    stroke_width=1,
                )
            elif col == 0 and row > 0:
                draw_centered_text(
                    draw,
                    text=str(row),
                    center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                    font=index_font,
                    fill=render_params.text_rgb,
                    stroke_fill=render_params.text_rgb,
                    stroke_width=1,
                )
            elif row > 0 and col > 0:
                letter = str(dataset.grid[row - 1][col - 1])
                entity_id = cell_key((row - 1, col - 1))
                rounded = round_bbox(bbox)
                center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                item_bbox_map[entity_id] = rounded
                cell_bbox_map[entity_id] = rounded
                cell_centers_px[entity_id] = (round(center[0], 3), round(center[1], 3))
                draw_centered_text(
                    draw,
                    text=letter,
                    center=center,
                    font=letter_font,
                    fill=render_params.text_rgb,
                    stroke_fill=render_params.text_rgb,
                    stroke_width=1,
                )
                entities.append(
                    {
                        "entity_id": entity_id,
                        "entity_type": "puzzle_word_search_cell",
                        "bbox_px": rounded,
                        "row": int(row - 1),
                        "col": int(col - 1),
                        "letter": letter,
                    }
                )


def _draw_option_cards(
    draw,
    *,
    dataset: WordSearchDataset,
    render_params: WordSearchRenderParams,
    option_x0: int,
    option_y0: int,
    item_bbox_map: dict[str, list[float]],
    entities: list[dict[str, Any]],
) -> None:
    """Draw visible option cards below the letter grid."""

    columns, _rows = _option_grid_shape(len(dataset.option_specs))
    for option_index, spec in enumerate(dataset.option_specs):
        row_index, col_index = divmod(int(option_index), int(columns))
        x0 = int(
            option_x0
            + col_index
            * (int(render_params.option_panel_width_px) + int(render_params.option_gap_px))
        )
        y0 = int(
            option_y0
            + row_index
            * (int(render_params.option_panel_height_px) + int(render_params.option_gap_px))
        )
        bbox = (
            int(x0),
            int(y0),
            int(x0 + render_params.option_panel_width_px),
            int(y0 + render_params.option_panel_height_px),
        )
        draw.rounded_rectangle(
            bbox,
            radius=10,
            fill=render_params.option_fill_rgb,
            outline=render_params.option_border_rgb,
            width=2,
        )
        text = option_text(spec)
        option_card_font = fit_font_to_box(
            draw,
            text=text,
            max_width=max(1, (bbox[2] - bbox[0]) - 24),
            max_height=max(1, (bbox[3] - bbox[1]) - 12),
            bold=True,
            font_family=_WORD_SEARCH_FONT_FAMILY,
            min_size_px=13,
            max_size_px=max(int(render_params.option_font_size_px) + 8, 20),
            fill_ratio=0.9,
        )
        draw_centered_text(
            draw,
            text=text,
            center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
            font=option_card_font,
            fill=render_params.option_text_rgb,
            stroke_fill=render_params.option_text_rgb,
            stroke_width=0,
        )
        key = option_key(spec.label)
        item_bbox_map[key] = round_bbox(bbox)
        entities.append(
            {
                "entity_id": key,
                "entity_type": "puzzle_word_search_option",
                "bbox_px": round_bbox(bbox),
                "label": str(spec.label),
                "text": text,
                "word": str(spec.word),
                "is_correct": bool(spec.is_correct),
            }
        )


def _option_grid_size(
    dataset: WordSearchDataset,
    render_params: WordSearchRenderParams,
) -> tuple[int, int]:
    """Return the width and height needed by the bottom option grid."""

    option_count = len(dataset.option_specs)
    if option_count <= 0:
        return 0, 0
    columns, rows = _option_grid_shape(option_count)
    width = int(
        columns * int(render_params.option_panel_width_px)
        + max(0, columns - 1) * int(render_params.option_gap_px)
    )
    height = int(
        rows * int(render_params.option_panel_height_px)
        + max(0, rows - 1) * int(render_params.option_gap_px)
    )
    return int(width), int(height)


def _option_grid_shape(option_count: int) -> tuple[int, int]:
    """Return deterministic bottom-grid columns and rows for visible options."""

    count = int(option_count)
    if count <= 0:
        return 0, 0
    if count == 4:
        return 2, 2
    if count == 6:
        return 3, 2
    columns = min(3, count)
    rows = (count + columns - 1) // columns
    return int(columns), int(rows)


def _cell_bbox_and_fill(
    *,
    row: int,
    col: int,
    grid_x0: int,
    grid_y0: int,
    header: int,
    cell: int,
    render_params: WordSearchRenderParams,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int]]:
    """Return one grid/header bbox and fill color."""

    if row == 0 and col == 0:
        bbox = (grid_x0, grid_y0, grid_x0 + header, grid_y0 + header)
        fill = render_params.header_fill_rgb
    elif row == 0:
        bbox = (
            grid_x0 + header + ((col - 1) * cell),
            grid_y0,
            grid_x0 + header + (col * cell),
            grid_y0 + header,
        )
        fill = render_params.header_fill_rgb
    elif col == 0:
        bbox = (
            grid_x0,
            grid_y0 + header + ((row - 1) * cell),
            grid_x0 + header,
            grid_y0 + header + (row * cell),
        )
        fill = render_params.header_fill_rgb
    else:
        bbox = (
            grid_x0 + header + ((col - 1) * cell),
            grid_y0 + header + ((row - 1) * cell),
            grid_x0 + header + (col * cell),
            grid_y0 + header + (row * cell),
        )
        fill = render_params.grid_fill_rgb
    return bbox, fill


__all__ = ["render_word_search_scene"]
