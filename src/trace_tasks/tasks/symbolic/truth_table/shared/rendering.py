"""Renderer for symbolic truth-table scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from ....shared.text_rendering import load_font
from ...shared.drawing import draw_centered_text, draw_rounded_rect
from ...shared.scene_style import SymbolicSceneStyle

from .rules import truth_rows
from .state import (
    TRUTH_VARIABLES,
    RenderedTruthTableScene,
    TruthExpressionSpec,
    TruthTableRenderParams,
)
from .styles import truth_table_variant_palette


def _rounded_bbox(values: Sequence[float]) -> list[float]:
    return [round(float(value), 3) for value in values[:4]]


def _table_bbox(
    *,
    left: float,
    top: float,
    columns: Sequence[tuple[str, float]],
    params: TruthTableRenderParams,
) -> list[float]:
    width = sum(float(width) for _key, width in columns)
    height = float(params.header_height_px + (len(truth_rows()) * params.row_height_px))
    return _rounded_bbox([left, top, left + width, top + height])


def _draw_text(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    bbox: Sequence[float],
    font,
    fill: Sequence[int],
    stroke: Sequence[int],
    stroke_width: int = 1,
) -> list[float]:
    left, top, right, bottom = [float(value) for value in bbox[:4]]
    return draw_centered_text(
        draw,
        text=str(text),
        center=(0.5 * (left + right), 0.5 * (top + bottom)),
        font=font,
        fill=fill,
        stroke_fill=stroke,
        stroke_width=int(stroke_width),
    )


def _draw_table_grid(
    draw: ImageDraw.ImageDraw,
    *,
    left: float,
    top: float,
    columns: Sequence[tuple[str, float]],
    headers: Mapping[str, str],
    values_by_column: Mapping[str, Sequence[int | str]],
    params: TruthTableRenderParams,
    palette: Mapping[str, Sequence[int]],
    scene_variant: str,
) -> tuple[
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    list[dict[str, Any]],
]:
    """Draw a truth table and return cell/column boxes plus entities."""

    header_font = load_font(int(params.header_font_size_px), bold=True)
    cell_font = load_font(int(params.cell_font_size_px), bold=False)
    table_box = _table_bbox(left=left, top=top, columns=columns, params=params)
    draw_rounded_rect(
        draw,
        tuple(table_box),
        radius=int(params.card_corner_radius_px),
        fill=palette["panel_fill"],
        outline=palette["grid"],
        width=int(params.card_border_width_px),
    )
    cell_bboxes: dict[str, list[float]] = {}
    column_bboxes: dict[str, list[float]] = {}
    row_bboxes: dict[str, list[float]] = {}
    entities: list[dict[str, Any]] = []
    cursor_x = float(left)
    for column_key, column_width in columns:
        col_box = _rounded_bbox(
            [
                cursor_x,
                top,
                cursor_x + float(column_width),
                top
                + float(params.header_height_px)
                + (len(truth_rows()) * float(params.row_height_px)),
            ]
        )
        column_bboxes[str(column_key)] = list(col_box)
        header_box = _rounded_bbox(
            [
                cursor_x,
                top,
                cursor_x + float(column_width),
                top + float(params.header_height_px),
            ]
        )
        draw.rectangle(
            tuple(header_box),
            fill=tuple(int(value) for value in palette["header_fill"]),
        )
        _draw_text(
            draw,
            text=str(headers[column_key]),
            bbox=header_box,
            font=header_font,
            fill=palette["text"],
            stroke=palette["header_fill"],
            stroke_width=1,
        )
        cell_bboxes[f"header_{column_key}"] = list(header_box)
        entities.append(
            {
                "item_id": f"header_{column_key}",
                "entity_type": "truth_table_header",
                "role": "header",
                "label": str(headers[column_key]),
                "bbox_px": list(header_box),
            }
        )
        values = tuple(values_by_column.get(str(column_key), ()))
        for row_index, row in enumerate(truth_rows()):
            cell_top = (
                top
                + float(params.header_height_px)
                + (row_index * float(params.row_height_px))
            )
            cell_box = _rounded_bbox(
                [
                    cursor_x,
                    cell_top,
                    cursor_x + float(column_width),
                    cell_top + float(params.row_height_px),
                ]
            )
            fill = palette["cell_fill"]
            if str(scene_variant) == "notebook_table" and row_index % 2 == 1:
                fill = (248, 251, 255)
            draw.rectangle(tuple(cell_box), fill=tuple(int(value) for value in fill))
            value = values[row_index] if row_index < len(values) else ""
            _draw_text(
                draw,
                text=str(value),
                bbox=cell_box,
                font=cell_font,
                fill=palette["text"],
                stroke=fill,
                stroke_width=0,
            )
            cell_id = f"{column_key}_row_{row.row_label}"
            cell_bboxes[cell_id] = list(cell_box)
            entities.append(
                {
                    "item_id": str(cell_id),
                    "entity_type": "truth_table_cell",
                    "role": str(column_key),
                    "label": str(row.row_label),
                    "value": str(value),
                    "bbox_px": list(cell_box),
                }
            )
        cursor_x += float(column_width)

    x = float(left)
    for _column_key, column_width in columns:
        draw.line(
            (x, top, x, table_box[3]),
            fill=tuple(int(value) for value in palette["grid"]),
            width=int(params.grid_line_width_px),
        )
        x += float(column_width)
    draw.line(
        (x, top, x, table_box[3]),
        fill=tuple(int(value) for value in palette["grid"]),
        width=int(params.grid_line_width_px),
    )
    y = float(top)
    for row_index in range(len(truth_rows()) + 2):
        draw.line(
            (left, y, table_box[2], y),
            fill=tuple(int(value) for value in palette["grid"]),
            width=int(params.grid_line_width_px),
        )
        y += float(params.header_height_px if row_index == 0 else params.row_height_px)
    entities.append(
        {
            "item_id": "truth_table",
            "entity_type": "truth_table",
            "role": "table",
            "bbox_px": list(table_box),
        }
    )
    for row in truth_rows():
        row_cells = [
            cell_bboxes[f"{column_key}_row_{row.row_label}"]
            for column_key, _width in columns
        ]
        row_bboxes[f"row_{row.row_label}"] = _rounded_bbox(
            [
                min(float(bbox[0]) for bbox in row_cells),
                min(float(bbox[1]) for bbox in row_cells),
                max(float(bbox[2]) for bbox in row_cells),
                max(float(bbox[3]) for bbox in row_cells),
            ]
        )
    return cell_bboxes, column_bboxes, row_bboxes, entities


def _input_columns() -> tuple[tuple[str, float], ...]:
    return (
        ("row_label", 58.0),
        ("A", 66.0),
        ("B", 66.0),
        ("C", 66.0),
    )


def _input_headers() -> dict[str, str]:
    return {"row_label": "#", "A": "A", "B": "B", "C": "C"}


def _input_values() -> dict[str, tuple[int | str, ...]]:
    rows = truth_rows()
    return {
        "row_label": tuple(row.row_label for row in rows),
        "A": tuple(row.values["A"] for row in rows),
        "B": tuple(row.values["B"] for row in rows),
        "C": tuple(row.values["C"] for row in rows),
    }


def _scene_bbox_from_maps(*maps: Mapping[str, Sequence[float]]) -> list[float]:
    boxes = [bbox for mapping in maps for bbox in mapping.values()]
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    return _rounded_bbox(
        [
            min(float(bbox[0]) for bbox in boxes),
            min(float(bbox[1]) for bbox in boxes),
            max(float(bbox[2]) for bbox in boxes),
            max(float(bbox[3]) for bbox in boxes),
        ]
    )


def render_count_scene(
    image: Image.Image,
    *,
    expression: TruthExpressionSpec,
    params: TruthTableRenderParams,
    style: SymbolicSceneStyle,
    scene_variant: str,
) -> RenderedTruthTableScene:
    """Render a single-output truth table for true-row counting."""

    draw = ImageDraw.Draw(image)
    palette = truth_table_variant_palette(scene_variant)
    columns = (
        *_input_columns(),
        ("output_P", float(params.output_cell_width_px)),
    )
    table_width = sum(float(width) for _key, width in columns)
    table_left = 0.5 * (float(params.canvas_width) - float(table_width))
    headers = {
        **_input_headers(),
        "output_P": f"P={expression.display}",
    }
    values = {
        **_input_values(),
        "output_P": tuple("" for _value in expression.pattern),
    }
    cell_bboxes, column_bboxes, row_bboxes, entities = _draw_table_grid(
        draw,
        left=float(table_left),
        top=float(params.table_top_px),
        columns=columns,
        headers=headers,
        values_by_column=values,
        params=params,
        palette=palette,
        scene_variant=str(scene_variant),
    )
    item_bboxes = {"truth_table": _scene_bbox_from_maps(cell_bboxes)}
    item_bboxes.update(column_bboxes)
    return RenderedTruthTableScene(
        image=image,
        entities=tuple(entities),
        item_bboxes=dict(item_bboxes),
        cell_bboxes=dict(cell_bboxes),
        column_bboxes=dict(column_bboxes),
        row_bboxes=dict(row_bboxes),
        scene_bbox_px=_scene_bbox_from_maps(cell_bboxes, item_bboxes),
        style_metadata={
            "scene_variant": str(scene_variant),
            "variables": list(TRUTH_VARIABLES),
        },
    )


def _draw_expression_card(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    label: str,
    expression: str,
    params: TruthTableRenderParams,
    palette: Mapping[str, Sequence[int]],
    item_id: str | None = None,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Draw one expression card and keep its bbox traceable."""

    box = _rounded_bbox(bbox)
    draw_rounded_rect(
        draw,
        tuple(box),
        radius=int(params.card_corner_radius_px),
        fill=palette["panel_fill"],
        outline=palette["accent"],
        width=int(params.card_border_width_px),
    )
    label_font = load_font(int(params.option_label_font_size_px), bold=True)
    expression_font = load_font(int(params.expression_font_size_px), bold=False)
    _draw_text(
        draw,
        text=str(label),
        bbox=[box[0] + 14, box[1] + 8, box[0] + 72, box[3] - 8],
        font=label_font,
        fill=palette["accent"],
        stroke=palette["panel_fill"],
        stroke_width=1,
    )
    _draw_text(
        draw,
        text=str(expression),
        bbox=[box[0] + 78, box[1] + 8, box[2] - 14, box[3] - 8],
        font=expression_font,
        fill=palette["text"],
        stroke=palette["panel_fill"],
        stroke_width=1,
    )
    return box, [
        {
            "item_id": str(item_id or label),
            "entity_type": "truth_expression_card",
            "role": "expression_card",
            "label": str(label),
            "expression": str(expression),
            "bbox_px": list(box),
        }
    ]


def render_pattern_scene(
    image: Image.Image,
    *,
    expression: TruthExpressionSpec,
    options: Sequence[tuple[str, str]],
    params: TruthTableRenderParams,
    style: SymbolicSceneStyle,
    scene_variant: str,
) -> RenderedTruthTableScene:
    """Render a target expression with output-pattern option cards."""

    draw = ImageDraw.Draw(image)
    palette = truth_table_variant_palette(scene_variant)
    columns = _input_columns()
    cell_bboxes, column_bboxes, row_bboxes, entities = _draw_table_grid(
        draw,
        left=float(params.table_left_px),
        top=170.0,
        columns=columns,
        headers=_input_headers(),
        values_by_column=_input_values(),
        params=params,
        palette=palette,
        scene_variant=str(scene_variant),
    )
    item_bboxes: dict[str, list[float]] = {
        "truth_table": _scene_bbox_from_maps(cell_bboxes)
    }
    item_bboxes.update(column_bboxes)
    card_box, card_entities = _draw_expression_card(
        draw,
        bbox=[430, 96, 1032, 164],
        label="P",
        expression=str(expression.display),
        params=params,
        palette=palette,
    )
    item_bboxes["target_expression"] = list(card_box)
    entities.extend(card_entities)

    option_font = load_font(int(params.pattern_font_size_px), bold=False)
    label_font = load_font(int(params.option_label_font_size_px), bold=True)
    option_left = 430.0
    option_top = 220.0
    option_width = 282.0
    option_height = 104.0
    gap_x = 38.0
    gap_y = 48.0
    for index, (label, pattern) in enumerate(options):
        row = index // 2
        col = index % 2
        left = option_left + (col * (option_width + gap_x))
        top = option_top + (row * (option_height + gap_y))
        option_id = f"pattern_option_{label}"
        box = _rounded_bbox([left, top, left + option_width, top + option_height])
        draw_rounded_rect(
            draw,
            tuple(box),
            radius=int(params.card_corner_radius_px),
            fill=palette["panel_fill"],
            outline=palette["grid"],
            width=int(params.card_border_width_px),
        )
        _draw_text(
            draw,
            text=str(label),
            bbox=[box[0] + 12, box[1] + 12, box[0] + 60, box[3] - 12],
            font=label_font,
            fill=palette["accent"],
            stroke=palette["panel_fill"],
            stroke_width=1,
        )
        _draw_text(
            draw,
            text=" ".join(str(pattern)),
            bbox=[box[0] + 68, box[1] + 12, box[2] - 12, box[3] - 12],
            font=option_font,
            fill=palette["text"],
            stroke=palette["panel_fill"],
            stroke_width=0,
        )
        item_bboxes[option_id] = list(box)
        entities.append(
            {
                "item_id": str(option_id),
                "entity_type": "truth_pattern_option",
                "role": "option_card",
                "label": str(label),
                "pattern": str(pattern),
                "bbox_px": list(box),
            }
        )
    return RenderedTruthTableScene(
        image=image,
        entities=tuple(entities),
        item_bboxes=dict(item_bboxes),
        cell_bboxes=dict(cell_bboxes),
        column_bboxes=dict(column_bboxes),
        row_bboxes=dict(row_bboxes),
        scene_bbox_px=_scene_bbox_from_maps(cell_bboxes, item_bboxes),
        style_metadata={
            "scene_variant": str(scene_variant),
            "variables": list(TRUTH_VARIABLES),
        },
    )


def render_expression_from_rows_scene(
    image: Image.Image,
    *,
    expression: TruthExpressionSpec,
    candidates: Sequence[tuple[str, TruthExpressionSpec]],
    params: TruthTableRenderParams,
    style: SymbolicSceneStyle,
    scene_variant: str,
) -> RenderedTruthTableScene:
    """Render a completed output column with candidate expression cards."""

    draw = ImageDraw.Draw(image)
    palette = truth_table_variant_palette(scene_variant)
    columns = (
        *_input_columns(),
        ("output_P", float(params.compact_output_cell_width_px)),
    )
    headers = {**_input_headers(), "output_P": "P"}
    values = {
        **_input_values(),
        "output_P": tuple(int(value) for value in expression.pattern),
    }
    cell_bboxes, column_bboxes, row_bboxes, entities = _draw_table_grid(
        draw,
        left=float(params.table_left_px),
        top=float(params.table_top_px),
        columns=tuple(columns),
        headers=headers,
        values_by_column=values,
        params=params,
        palette=palette,
        scene_variant=str(scene_variant),
    )
    item_bboxes = {"truth_table": _scene_bbox_from_maps(cell_bboxes)}
    item_bboxes.update(column_bboxes)

    card_left = 560.0
    card_top = 128.0
    card_width = 482.0
    card_height = 76.0
    card_gap = 34.0
    for index, (label, spec) in enumerate(candidates):
        option_id = f"expression_option_{label}"
        top = card_top + (index * (card_height + card_gap))
        card_box, card_entities = _draw_expression_card(
            draw,
            bbox=[card_left, top, card_left + card_width, top + card_height],
            label=str(label),
            expression=str(spec.display),
            params=params,
            palette=palette,
            item_id=str(option_id),
        )
        item_bboxes[str(option_id)] = list(card_box)
        entities.extend(card_entities)
    return RenderedTruthTableScene(
        image=image,
        entities=tuple(entities),
        item_bboxes=dict(item_bboxes),
        cell_bboxes=dict(cell_bboxes),
        column_bboxes=dict(column_bboxes),
        row_bboxes=dict(row_bboxes),
        scene_bbox_px=_scene_bbox_from_maps(cell_bboxes, column_bboxes),
        style_metadata={
            "scene_variant": str(scene_variant),
            "variables": list(TRUTH_VARIABLES),
        },
    )


__all__ = [
    "render_count_scene",
    "render_expression_from_rows_scene",
    "render_pattern_scene",
]
