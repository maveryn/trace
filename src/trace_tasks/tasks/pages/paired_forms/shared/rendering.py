"""Renderer for paired-form reconciliation document scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.bbox_projection import round_bbox as _round_bbox
from ....shared.drawing import draw_rounded_rect
from ....shared.render_variation import apply_resolved_layout_jitter_to_margins
from ....shared.text_legibility import draw_text_traced
from ....shared.text_rendering import fit_font_to_box
from .sampling import ReconciliationRenderParams


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class RenderedReconciliationScene:
    """Rendered paired-form reconciliation scene plus traced text bboxes."""

    image: Image.Image
    entities: List[Dict[str, object]]
    panel_bboxes_px: Dict[str, List[float]]
    title_bboxes_px: Dict[str, List[float]]
    header_value_bbox_map: Dict[str, List[float]]
    cell_value_bbox_map: Dict[str, List[float]]
    row_bbox_map: Dict[str, List[float]]
    layout_jitter_meta: Dict[str, object]


def _draw_text_in_box(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: BBox,
    text: str,
    font_size_px: int,
    bold: bool,
    fill: Sequence[int],
    stroke_fill: Sequence[int],
    align: str,
    padding_px: int,
) -> List[float]:
    """Draw fitted text in a box and return the text bbox."""

    left, top, right, bottom = [float(value) for value in bbox]
    width = max(1.0, float(right - left) - (2.0 * float(padding_px)))
    height = max(1.0, float(bottom - top) - (2.0 * float(padding_px)))
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=float(width),
        max_height=float(height),
        bold=bool(bold),
        min_size_px=max(9, int(font_size_px * 0.62)),
        max_size_px=int(font_size_px),
        fill_ratio=0.98,
    )
    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=1)
    text_left, text_top, text_right, text_bottom = [float(value) for value in text_bbox]
    if str(align) == "right":
        origin_x = float(right - padding_px - text_right)
    elif str(align) == "center":
        origin_x = float(((left + right) * 0.5) - (0.5 * (text_left + text_right)))
    else:
        origin_x = float(left + padding_px - text_left)
    origin_y = float(((top + bottom) * 0.5) - (0.5 * (text_top + text_bottom)))
    draw_text_traced(draw,
        (float(origin_x), float(origin_y)),
        str(text),
        font=font,
        fill=tuple(int(value) for value in fill),
        stroke_width=1,
        stroke_fill=tuple(int(value) for value in stroke_fill),
     role="readout", required=False,)
    return _round_bbox(
        [
            float(origin_x + text_left),
            float(origin_y + text_top),
            float(origin_x + text_right),
            float(origin_y + text_bottom),
        ]
    )


def _panel_bboxes(render_params: ReconciliationRenderParams) -> Tuple[BBox, BBox, Dict[str, object]]:
    width = float(render_params.canvas_width)
    height = float(render_params.canvas_height)
    margin = float(render_params.outer_margin_px)
    gap = float(render_params.panel_gap_px)
    jitter_left, jitter_right, jitter_top, jitter_bottom, layout_jitter_meta = apply_resolved_layout_jitter_to_margins(
        left_px=float(margin),
        right_px=float(margin),
        top_px=float(margin),
        bottom_px=float(margin),
        jitter=render_params.layout_jitter_meta,
    )
    left_margin = float(jitter_left)
    right_margin = float(jitter_right)
    top_margin = float(jitter_top)
    bottom_margin = float(jitter_bottom)
    panel_width = float((width - left_margin - right_margin - gap) / 2.0)
    panel_height = float(height - top_margin - bottom_margin)
    left_panel = (left_margin, top_margin, left_margin + panel_width, top_margin + panel_height)
    right_left = float(left_margin + panel_width + gap)
    right_panel = (right_left, top_margin, right_left + panel_width, top_margin + panel_height)
    return left_panel, right_panel, dict(layout_jitter_meta)


def _header_boxes(panel_bbox: BBox, render_params: ReconciliationRenderParams) -> List[BBox]:
    left, top, right, _ = [float(value) for value in panel_bbox]
    inner_left = float(left + 22.0)
    inner_right = float(right - 22.0)
    content_top = float(top + render_params.title_band_height_px + 18.0)
    gap = 12.0
    field_height = float(render_params.header_field_height_px)
    field_width = float((inner_right - inner_left - gap) / 2.0)
    return [
        (inner_left, content_top, inner_left + field_width, content_top + field_height),
        (inner_left + field_width + gap, content_top, inner_right, content_top + field_height),
        (inner_left, content_top + field_height + 10.0, inner_right, content_top + 2.0 * field_height + 10.0),
    ]


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: BBox,
    panel_id: str,
    title: str,
    subtitle: str,
    header_specs: Sequence[Mapping[str, str]],
    render_params: ReconciliationRenderParams,
    entities: List[Dict[str, object]],
    panel_bboxes_px: Dict[str, List[float]],
    title_bboxes_px: Dict[str, List[float]],
    header_value_bbox_map: Dict[str, List[float]],
) -> None:
    """Draw one form panel with title band and header fields."""

    draw_rounded_rect(
        draw,
        panel_bbox,
        radius=int(render_params.panel_corner_radius_px),
        fill=render_params.panel_fill_rgb,
        outline=render_params.panel_border_rgb,
        width=int(render_params.panel_border_width_px),
    )
    title_band = (
        float(panel_bbox[0]),
        float(panel_bbox[1]),
        float(panel_bbox[2]),
        float(panel_bbox[1] + render_params.title_band_height_px),
    )
    draw_rounded_rect(
        draw,
        title_band,
        radius=int(render_params.panel_corner_radius_px),
        fill=render_params.title_fill_rgb,
        outline=render_params.title_fill_rgb,
        width=0,
    )
    title_bbox = _draw_text_in_box(
        draw,
        bbox=(title_band[0] + 22.0, title_band[1] + 8.0, title_band[2] - 22.0, title_band[1] + 42.0),
        text=str(title),
        font_size_px=int(render_params.title_font_size_px),
        bold=True,
        fill=render_params.title_text_rgb,
        stroke_fill=render_params.title_fill_rgb,
        align="left",
        padding_px=0,
    )
    _draw_text_in_box(
        draw,
        bbox=(title_band[0] + 22.0, title_band[1] + 42.0, title_band[2] - 22.0, title_band[3] - 8.0),
        text=str(subtitle),
        font_size_px=int(render_params.subtitle_font_size_px),
        bold=False,
        fill=render_params.title_text_rgb,
        stroke_fill=render_params.title_fill_rgb,
        align="left",
        padding_px=0,
    )
    panel_bboxes_px[str(panel_id)] = _round_bbox(panel_bbox)
    title_bboxes_px[str(panel_id)] = list(title_bbox)
    entities.extend(
        [
            {
                "entity_id": f"{panel_id}:panel",
                "entity_type": "reconciliation_document_panel",
                "bbox_id": str(panel_id),
                "bbox_px": _round_bbox(panel_bbox),
            },
            {
                "entity_id": f"{panel_id}:title",
                "entity_type": "reconciliation_document_title",
                "bbox_id": f"{panel_id}:title",
                "bbox_px": list(title_bbox),
                "text": str(title),
            },
        ]
    )

    for header_spec, box in zip(header_specs, _header_boxes(panel_bbox, render_params), strict=True):
        draw_rounded_rect(
            draw,
            box,
            radius=10,
            fill=render_params.field_fill_rgb,
            outline=render_params.field_border_rgb,
            width=1,
        )
        label_box = (box[0] + 10.0, box[1] + 5.0, box[2] - 10.0, box[1] + 22.0)
        value_box = (box[0] + 10.0, box[1] + 23.0, box[2] - 10.0, box[3] - 5.0)
        _draw_text_in_box(
            draw,
            bbox=label_box,
            text=str(header_spec["field_label"]),
            font_size_px=int(render_params.label_font_size_px),
            bold=True,
            fill=render_params.label_rgb,
            stroke_fill=render_params.stroke_rgb,
            align="left",
            padding_px=0,
        )
        value_bbox = _draw_text_in_box(
            draw,
            bbox=value_box,
            text=str(header_spec["field_value"]),
            font_size_px=int(render_params.cell_font_size_px),
            bold=False,
            fill=render_params.text_rgb,
            stroke_fill=render_params.stroke_rgb,
            align="left",
            padding_px=0,
        )
        bbox_id = f"{panel_id}:header:{header_spec['field_id']}"
        header_value_bbox_map[bbox_id] = list(value_bbox)
        entities.append(
            {
                "entity_id": bbox_id,
                "entity_type": "reconciliation_header_value",
                "bbox_id": bbox_id,
                "bbox_px": list(value_bbox),
                "text": str(header_spec["field_value"]),
            }
        )


def _table_geometry(
    panel_bbox: BBox,
    *,
    item_count: int,
    render_params: ReconciliationRenderParams,
) -> Tuple[BBox, float]:
    left, top, right, bottom = [float(value) for value in panel_bbox]
    inner_left = float(left + 22.0)
    inner_right = float(right - 22.0)
    table_top = float(top + render_params.title_band_height_px + 18.0 + 2.0 * render_params.header_field_height_px + 34.0)
    table_bottom = float(bottom - 24.0)
    available = max(1.0, float(table_bottom - table_top - render_params.table_header_height_px))
    row_height = float(available / max(1, int(item_count)))
    row_height = max(float(render_params.row_min_height_px), min(float(render_params.row_max_height_px), row_height))
    return (inner_left, table_top, inner_right, table_bottom), row_height


def _draw_table(
    draw: ImageDraw.ImageDraw,
    *,
    panel_id: str,
    panel_bbox: BBox,
    item_specs: Sequence[Mapping[str, object]],
    columns: Sequence[Tuple[str, str, float, str]],
    render_params: ReconciliationRenderParams,
    entities: List[Dict[str, object]],
    cell_value_bbox_map: Dict[str, List[float]],
    row_bbox_map: Dict[str, List[float]],
) -> None:
    """Draw one line-item table and trace cell values."""

    table_bbox, row_height = _table_geometry(panel_bbox, item_count=len(item_specs), render_params=render_params)
    left, top, right, _ = [float(value) for value in table_bbox]
    table_width = float(right - left)
    header_height = float(render_params.table_header_height_px)
    x_edges = [left]
    cursor = left
    for _, _, fraction, _ in columns:
        cursor += float(table_width * float(fraction))
        x_edges.append(cursor)
    x_edges[-1] = right

    header_bbox = (left, top, right, top + header_height)
    draw_rounded_rect(
        draw,
        header_bbox,
        radius=8,
        fill=render_params.table_header_fill_rgb,
        outline=render_params.divider_rgb,
        width=1,
    )
    for col_index, (_, column_label, _, align) in enumerate(columns):
        cell = (x_edges[col_index] + 5.0, top + 4.0, x_edges[col_index + 1] - 5.0, top + header_height - 4.0)
        _draw_text_in_box(
            draw,
            bbox=cell,
            text=str(column_label),
            font_size_px=int(render_params.label_font_size_px),
            bold=True,
            fill=render_params.label_rgb,
            stroke_fill=render_params.stroke_rgb,
            align=str(align),
            padding_px=0,
        )

    y = float(top + header_height)
    for row_index, spec in enumerate(item_specs):
        row_bbox = (left, y, right, y + row_height)
        fill = render_params.row_alt_fill_rgb if int(row_index) % 2 else render_params.field_fill_rgb
        draw.rectangle(row_bbox, fill=tuple(int(value) for value in fill), outline=tuple(int(value) for value in render_params.divider_rgb), width=1)
        row_bbox_map[f"{panel_id}:{spec['item_id']}"] = _round_bbox(row_bbox)
        for col_index, (value_key, _, _, align) in enumerate(columns):
            raw_value = spec[str(value_key)]
            bbox_id = f"{panel_id}:{spec['item_id']}:{value_key}"
            cell = (x_edges[col_index] + 7.0, y + 4.0, x_edges[col_index + 1] - 7.0, y + row_height - 4.0)
            value_bbox = _draw_text_in_box(
                draw,
                bbox=cell,
                text=str(raw_value),
                font_size_px=int(render_params.cell_font_size_px),
                bold=False,
                fill=render_params.text_rgb,
                stroke_fill=render_params.stroke_rgb,
                align=str(align),
                padding_px=0,
            )
            cell_value_bbox_map[bbox_id] = list(value_bbox)
            entities.append(
                {
                    "entity_id": bbox_id,
                    "entity_type": "reconciliation_table_cell",
                    "bbox_id": bbox_id,
                    "bbox_px": list(value_bbox),
                    "document_id": str(panel_id),
                    "item_id": str(spec["item_id"]),
                    "column_key": str(value_key),
                    "text": str(raw_value),
                }
            )
        y += float(row_height)


def render_reconciliation_scene(
    background: Image.Image,
    *,
    scene_title: str,
    purchase_title: str,
    receiving_title: str,
    purchase_header_specs: Sequence[Mapping[str, str]],
    receiving_header_specs: Sequence[Mapping[str, str]],
    item_specs: Sequence[Mapping[str, object]],
    receiving_item_specs: Sequence[Mapping[str, object]],
    render_params: ReconciliationRenderParams,
) -> RenderedReconciliationScene:
    """Render one side-by-side purchase-order / receiving-slip scene."""

    del scene_title
    image = background.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    left_panel, right_panel, layout_jitter_meta = _panel_bboxes(render_params)
    entities: List[Dict[str, object]] = []
    panel_bboxes_px: Dict[str, List[float]] = {}
    title_bboxes_px: Dict[str, List[float]] = {}
    header_value_bbox_map: Dict[str, List[float]] = {}
    cell_value_bbox_map: Dict[str, List[float]] = {}
    row_bbox_map: Dict[str, List[float]] = {}

    _draw_panel(
        draw,
        panel_bbox=left_panel,
        panel_id="po",
        title=str(purchase_title),
        subtitle="Ordered quantities and unit values",
        header_specs=purchase_header_specs,
        render_params=render_params,
        entities=entities,
        panel_bboxes_px=panel_bboxes_px,
        title_bboxes_px=title_bboxes_px,
        header_value_bbox_map=header_value_bbox_map,
    )
    _draw_panel(
        draw,
        panel_bbox=right_panel,
        panel_id="recv",
        title=str(receiving_title),
        subtitle="Received quantities by matching item code",
        header_specs=receiving_header_specs,
        render_params=render_params,
        entities=entities,
        panel_bboxes_px=panel_bboxes_px,
        title_bboxes_px=title_bboxes_px,
        header_value_bbox_map=header_value_bbox_map,
    )

    po_columns = (
        ("item_code", "Code", 0.21, "left"),
        ("item_name", "Item", 0.35, "left"),
        ("order_qty", "Ordered", 0.21, "right"),
        ("unit_value", "Unit Value", 0.23, "right"),
    )
    recv_columns = (
        ("item_code", "Code", 0.21, "left"),
        ("item_name", "Item", 0.35, "left"),
        ("received_qty", "Received", 0.24, "right"),
        ("dock_code", "Dock", 0.20, "center"),
    )
    _draw_table(
        draw,
        panel_id="po",
        panel_bbox=left_panel,
        item_specs=item_specs,
        columns=po_columns,
        render_params=render_params,
        entities=entities,
        cell_value_bbox_map=cell_value_bbox_map,
        row_bbox_map=row_bbox_map,
    )
    _draw_table(
        draw,
        panel_id="recv",
        panel_bbox=right_panel,
        item_specs=receiving_item_specs,
        columns=recv_columns,
        render_params=render_params,
        entities=entities,
        cell_value_bbox_map=cell_value_bbox_map,
        row_bbox_map=row_bbox_map,
    )

    return RenderedReconciliationScene(
        image=image,
        entities=entities,
        panel_bboxes_px=panel_bboxes_px,
        title_bboxes_px=title_bboxes_px,
        header_value_bbox_map=header_value_bbox_map,
        cell_value_bbox_map=cell_value_bbox_map,
        row_bbox_map=row_bbox_map,
        layout_jitter_meta=dict(layout_jitter_meta),
    )


__all__ = ["RenderedReconciliationScene", "render_reconciliation_scene"]
