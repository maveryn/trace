"""Rendering helpers for color-gradient puzzle scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.shared.drawing import draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.shared.text_rendering import fit_font_to_box, load_font

from .state import (
    CompletionDataset,
    RenderParams,
    RenderedScene,
    ViolationDataset,
)


def draw_notebook_lines(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
) -> None:
    """Draw subtle guide lines behind the swatches in notebook mode."""

    left, top, right, bottom = bbox
    for y in range(int(top) + 28, int(bottom), 28):
        draw.line(
            [(int(left) + 10, int(y)), (int(right) - 10, int(y))],
            fill=tuple(color),
            width=1,
        )
    for x in range(int(left) + 34, int(right), 34):
        draw.line(
            [(int(x), int(top) + 10), (int(x), int(bottom) - 10)],
            fill=tuple(color),
            width=1,
        )


def draw_label_chip(
    draw: ImageDraw.ImageDraw,
    *,
    label: str,
    swatch_bbox: Tuple[int, int, int, int],
    render_params: RenderParams,
) -> None:
    """Draw the fixed white/black label chip used on color swatches."""

    chip_size = int(render_params.label_chip_size_px)
    chip_margin = int(render_params.label_margin_px)
    chip_bbox = (
        int(swatch_bbox[0] + chip_margin),
        int(swatch_bbox[1] + chip_margin),
        int(swatch_bbox[0] + chip_margin + chip_size),
        int(swatch_bbox[1] + chip_margin + chip_size),
    )
    chip_fill = (255, 255, 255)
    chip_outline = (36, 42, 52)
    label_fill = (28, 32, 38)
    label_font = fit_font_to_box(
        draw,
        text=str(label),
        max_width=max(1.0, float(chip_size) - 7.0),
        max_height=max(1.0, float(chip_size) - 7.0),
        bold=True,
        min_size_px=min(16, int(render_params.label_font_size_px)),
        max_size_px=int(render_params.label_font_size_px),
        fill_ratio=0.92,
    )
    draw.rounded_rectangle(
        chip_bbox, radius=8, fill=chip_fill, outline=chip_outline, width=1
    )
    draw_centered_text(
        draw,
        text=str(label),
        center=(
            (chip_bbox[0] + chip_bbox[2]) / 2.0,
            (chip_bbox[1] + chip_bbox[3]) / 2.0,
        ),
        font=label_font,
        fill=label_fill,
        stroke_fill=chip_outline,
        stroke_width=0,
    )


def render_violation_scene(
    *,
    background: Image.Image,
    dataset: ViolationDataset,
    scene_variant: str,
    render_params: RenderParams,
) -> RenderedScene:
    """Render the color-gradient violation swatch grid."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    grid_width = (int(dataset.cols) * int(render_params.swatch_size_px)) + (
        (int(dataset.cols) - 1) * int(render_params.swatch_gap_px)
    )
    grid_height = (int(dataset.rows) * int(render_params.swatch_size_px)) + (
        (int(dataset.rows) - 1) * int(render_params.swatch_gap_px)
    )
    grid_left = int(round((int(render_params.canvas_width) - int(grid_width)) / 2.0))
    grid_top = int(round((int(render_params.canvas_height) - int(grid_height)) / 2.0))
    panel_bbox = (
        int(grid_left - int(render_params.panel_padding_px)),
        int(grid_top - int(render_params.panel_padding_px)),
        int(grid_left + int(grid_width) + int(render_params.panel_padding_px)),
        int(grid_top + int(grid_height) + int(render_params.panel_padding_px)),
    )

    _draw_optional_panel(
        draw,
        scene_variant=str(scene_variant),
        panel_bbox=panel_bbox,
        render_params=render_params,
    )

    cell_bbox_map: Dict[str, Tuple[int, int, int, int]] = {}
    item_bbox_map: Dict[str, Tuple[int, int, int, int]] = {}
    entities: List[Dict[str, Any]] = [
        {
            "id": "gradient_grid_panel",
            "type": "color_gradient_panel",
            "bbox_px": [int(value) for value in panel_bbox],
            "rows": int(dataset.rows),
            "cols": int(dataset.cols),
            "scene_variant": str(scene_variant),
        }
    ]

    for cell in dataset.cells:
        x0 = int(
            grid_left
            + (
                int(cell.col)
                * (int(render_params.swatch_size_px) + int(render_params.swatch_gap_px))
            )
        )
        y0 = int(
            grid_top
            + (
                int(cell.row)
                * (int(render_params.swatch_size_px) + int(render_params.swatch_gap_px))
            )
        )
        bbox = (
            int(x0),
            int(y0),
            int(x0 + int(render_params.swatch_size_px)),
            int(y0 + int(render_params.swatch_size_px)),
        )
        draw_rounded_rect(
            draw,
            bbox,
            radius=int(render_params.swatch_corner_radius_px),
            fill=tuple(cell.observed_rgb),
            outline=tuple(render_params.swatch_border_rgb),
            width=int(render_params.swatch_border_width_px),
        )
        draw_label_chip(
            draw,
            label=str(cell.label),
            swatch_bbox=bbox,
            render_params=render_params,
        )

        cell_bbox_map[str(cell.cell_id)] = tuple(int(value) for value in bbox)
        item_bbox_map[str(cell.cell_id)] = tuple(int(value) for value in bbox)
        entities.append(
            {
                "id": str(cell.cell_id),
                "type": "color_gradient_swatch_cell",
                "label": str(cell.label),
                "row_index": int(cell.row),
                "col_index": int(cell.col),
                "bbox_px": [int(value) for value in bbox],
                "expected_hsl": [round(float(value), 6) for value in cell.expected_hsl],
                "observed_hsl": [round(float(value), 6) for value in cell.observed_hsl],
                "expected_rgb": [int(value) for value in cell.expected_rgb],
                "observed_rgb": [int(value) for value in cell.observed_rgb],
                "is_violation": bool(cell.is_violation),
            }
        )

    return RenderedScene(
        image=image,
        scene_bbox_px=tuple(int(value) for value in panel_bbox),
        cell_bbox_map=dict(cell_bbox_map),
        item_bbox_map=dict(item_bbox_map),
        entities=tuple(entities),
    )


def draw_missing_swatch(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[int, int, int, int],
    render_params: RenderParams,
) -> None:
    """Draw the blank swatch used by the completion task."""

    fill = (239, 242, 247)
    outline = tuple(render_params.swatch_border_rgb)
    draw_rounded_rect(
        draw,
        bbox,
        radius=int(render_params.swatch_corner_radius_px),
        fill=fill,
        outline=outline,
        width=int(render_params.swatch_border_width_px),
    )
    for offset in range(-int(bbox[3] - bbox[1]), int(bbox[2] - bbox[0]), 18):
        start_x = max(int(bbox[0]), int(bbox[0] + offset))
        start_y = int(bbox[1]) if offset >= 0 else int(bbox[1] - offset)
        end_x = min(int(bbox[2]), int(bbox[0] + offset + int(bbox[3] - bbox[1])))
        end_y = int(bbox[1] + (end_x - (int(bbox[0]) + offset)))
        draw.line(
            [(start_x, start_y), (end_x, min(int(bbox[3]), end_y))],
            fill=(207, 214, 225),
            width=2,
        )
    question_font = load_font(
        max(28, int(render_params.label_font_size_px) + 12),
        bold=True,
    )
    draw_centered_text(
        draw,
        text="?",
        center=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
        font=question_font,
        fill=(55, 62, 75),
        stroke_fill=(255, 255, 255),
        stroke_width=2,
    )


def render_completion_scene(
    *,
    background: Image.Image,
    dataset: CompletionDataset,
    scene_variant: str,
    render_params: RenderParams,
) -> RenderedScene:
    """Render a linear gradient with one missing swatch and visual options."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    gap = int(render_params.swatch_gap_px)
    swatch_size = min(
        int(render_params.swatch_size_px),
        max(
            76,
            int(
                (
                    int(render_params.canvas_width)
                    - (2 * int(render_params.panel_padding_px))
                    - ((int(dataset.sequence_length) - 1) * gap)
                    - 80
                )
                / max(1, int(dataset.sequence_length))
            ),
        ),
    )
    option_size = min(int(swatch_size), 104)
    sequence_width = int(dataset.sequence_length) * int(swatch_size) + (
        (int(dataset.sequence_length) - 1) * gap
    )
    options_width = int(dataset.option_count) * int(option_size) + (
        (int(dataset.option_count) - 1) * gap
    )
    sequence_left = int(
        round((int(render_params.canvas_width) - int(sequence_width)) / 2.0)
    )
    option_left = int(
        round((int(render_params.canvas_width) - int(options_width)) / 2.0)
    )
    sequence_top = 188
    option_top = int(sequence_top + int(swatch_size) + 118)
    panel_bbox = (
        int(min(sequence_left, option_left) - int(render_params.panel_padding_px)),
        int(sequence_top - int(render_params.panel_padding_px)),
        int(
            max(sequence_left + sequence_width, option_left + options_width)
            + int(render_params.panel_padding_px)
        ),
        int(option_top + int(option_size) + int(render_params.panel_padding_px)),
    )

    _draw_optional_panel(
        draw,
        scene_variant=str(scene_variant),
        panel_bbox=panel_bbox,
        render_params=render_params,
    )

    cell_bbox_map: Dict[str, Tuple[int, int, int, int]] = {}
    item_bbox_map: Dict[str, Tuple[int, int, int, int]] = {}
    entities: List[Dict[str, Any]] = [
        {
            "id": "linear_gradient_panel",
            "type": "linear_color_gradient_panel",
            "bbox_px": [int(value) for value in panel_bbox],
            "sequence_length": int(dataset.sequence_length),
            "option_count": int(dataset.option_count),
            "scene_variant": str(scene_variant),
        }
    ]

    for cell in dataset.cells:
        x0 = int(sequence_left + (int(cell.index) * (int(swatch_size) + gap)))
        y0 = int(sequence_top)
        bbox = (int(x0), int(y0), int(x0 + swatch_size), int(y0 + swatch_size))
        if bool(cell.is_missing):
            draw_missing_swatch(draw, bbox=bbox, render_params=render_params)
        else:
            draw_rounded_rect(
                draw,
                bbox,
                radius=int(render_params.swatch_corner_radius_px),
                fill=tuple(cell.expected_rgb),
                outline=tuple(render_params.swatch_border_rgb),
                width=int(render_params.swatch_border_width_px),
            )
        cell_bbox_map[str(cell.cell_id)] = tuple(int(value) for value in bbox)
        item_bbox_map[str(cell.cell_id)] = tuple(int(value) for value in bbox)
        entities.append(
            {
                "id": str(cell.cell_id),
                "type": "linear_gradient_sequence_cell",
                "index": int(cell.index),
                "bbox_px": [int(value) for value in bbox],
                "expected_hsl": [round(float(value), 6) for value in cell.expected_hsl],
                "expected_rgb": [int(value) for value in cell.expected_rgb],
                "is_missing": bool(cell.is_missing),
            }
        )

    for option_index, option in enumerate(dataset.options):
        x0 = int(option_left + (int(option_index) * (int(option_size) + gap)))
        y0 = int(option_top)
        bbox = (int(x0), int(y0), int(x0 + option_size), int(y0 + option_size))
        draw_rounded_rect(
            draw,
            bbox,
            radius=int(render_params.swatch_corner_radius_px),
            fill=tuple(option.rgb),
            outline=tuple(render_params.swatch_border_rgb),
            width=int(render_params.swatch_border_width_px),
        )
        draw_label_chip(
            draw,
            label=str(option.label),
            swatch_bbox=bbox,
            render_params=render_params,
        )
        cell_bbox_map[str(option.option_id)] = tuple(int(value) for value in bbox)
        item_bbox_map[str(option.option_id)] = tuple(int(value) for value in bbox)
        entities.append(
            {
                "id": str(option.option_id),
                "type": "linear_gradient_option_swatch",
                "label": str(option.label),
                "option_index": int(option_index),
                "bbox_px": [int(value) for value in bbox],
                "rgb": [int(value) for value in option.rgb],
                "is_correct": bool(option.is_correct),
            }
        )

    return RenderedScene(
        image=image,
        scene_bbox_px=tuple(int(value) for value in panel_bbox),
        cell_bbox_map=dict(cell_bbox_map),
        item_bbox_map=dict(item_bbox_map),
        entities=tuple(entities),
    )


def _draw_optional_panel(
    draw: ImageDraw.ImageDraw,
    *,
    scene_variant: str,
    panel_bbox: Tuple[int, int, int, int],
    render_params: RenderParams,
) -> None:
    """Draw scene panel treatments that sit behind the semantic swatches."""

    if str(scene_variant) not in {"swatch_card", "swatch_notebook"}:
        return
    draw_rounded_rect(
        draw,
        panel_bbox,
        radius=int(render_params.panel_corner_radius_px),
        fill=tuple(render_params.panel_fill_rgb),
        outline=tuple(render_params.panel_border_rgb),
        width=int(render_params.panel_border_width_px),
    )
    if str(scene_variant) == "swatch_notebook":
        draw_notebook_lines(
            draw,
            bbox=panel_bbox,
            color=tuple(render_params.notebook_line_rgb),
        )


__all__ = [
    "render_completion_scene",
    "render_violation_scene",
]
