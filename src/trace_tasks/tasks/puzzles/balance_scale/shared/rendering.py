"""Shared renderer for worksheet-style balance-scale puzzle scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.puzzles.shared.drawing import draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.puzzles.shared.scene_style import (
    PuzzleSceneStyle,
    draw_puzzle_panel_chrome,
)
from trace_tasks.tasks.puzzles.shared.scene_style import (
    make_puzzle_scene_background,
    resolve_puzzle_scene_style,
)
from trace_tasks.tasks.puzzles.shared.symbol_rendering import draw_puzzle_shape_icon
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import with_puzzle_unit_size_jitter
from trace_tasks.tasks.shared.bbox_projection import round_bbox
from trace_tasks.tasks.shared.font_assets import (
    font_asset_version,
    get_font_family_record,
    sample_font_family,
)
from trace_tasks.tasks.shared.text_rendering import load_font, temporary_default_font_family

from .sampling import resolve_render_params
from .rules import expand_terms
from .state import (
    BalanceScaleRenderParams,
    RenderedBalanceContext,
    RenderedBalanceScaleScene,
    EQUIVALENT_COUNT_ROW_KIND,
    MISSING_WEIGHT_ROW_KIND,
    SCENE_ID,
    SIDE_RELATION_ROW_KIND,
    WEIGHT_ORDER_ROW_KIND,
)


def _panel_bbox(params: BalanceScaleRenderParams) -> Tuple[float, float, float, float]:
    return (
        float(params.scene_margin_left_px),
        float(params.scene_margin_top_px),
        float(params.canvas_width - params.scene_margin_right_px),
        float(params.canvas_height - params.scene_margin_bottom_px),
    )


def _draw_object_token(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    object_label: str,
    object_type: str,
    fill_rgb: Sequence[int],
    outline_rgb: Sequence[int],
    text_rgb: Sequence[int],
    text_stroke_rgb: Sequence[int],
    stroke_width: int,
    label_font_size_px: int,
    highlight_rgb: Sequence[int] | None = None,
) -> None:
    """Draw one labeled object token while preserving its item bbox contract."""

    box = tuple(float(value) for value in bbox)
    if highlight_rgb is not None:
        pad = max(3.0, float(stroke_width) + 1.0)
        draw.rounded_rectangle(
            (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
            radius=max(6, int(0.18 * (box[2] - box[0]))),
            outline=tuple(int(value) for value in highlight_rgb),
            width=max(2, int(stroke_width) + 1),
        )
    draw_puzzle_shape_icon(
        draw,
        bbox=(box[0], box[1], box[2], box[3]),
        object_type=str(object_type),
        fill_rgb=fill_rgb,
        outline_rgb=outline_rgb,
        width=max(1, int(stroke_width)),
        inset_px=max(4.0, 0.10 * float(box[2] - box[0])),
    )
    draw_centered_text(
        draw,
        text=str(object_label),
        center=((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5),
        font=load_font(max(12, int(label_font_size_px)), bold=True),
        fill=tuple(int(value) for value in text_rgb),
        stroke_fill=tuple(int(value) for value in text_stroke_rgb),
        stroke_width=1,
    )


def _draw_numeric_chip(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    value: int,
    fill_rgb: Sequence[int],
    outline_rgb: Sequence[int],
    text_rgb: Sequence[int],
    text_stroke_rgb: Sequence[int],
    stroke_width: int,
    value_font_size_px: int,
) -> None:
    box = tuple(float(item) for item in bbox)
    draw_rounded_rect(
        draw,
        box,
        radius=max(6, int(0.14 * (box[3] - box[1]))),
        fill=tuple(int(value) for value in fill_rgb),
        outline=tuple(int(value) for value in outline_rgb),
        width=max(1, int(stroke_width)),
    )
    draw_centered_text(
        draw,
        text=str(int(value)),
        center=((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5),
        font=load_font(max(12, int(value_font_size_px)), bold=True),
        fill=tuple(int(item) for item in text_rgb),
        stroke_fill=tuple(int(item) for item in text_stroke_rgb),
        stroke_width=1,
    )


def _item_bbox_grid(
    *,
    pan_bbox: Sequence[float],
    item_count: int,
    token_size_px: int,
    token_gap_px: int,
) -> List[Tuple[float, float, float, float]]:
    if int(item_count) <= 0:
        return []
    left, top, right, bottom = [float(value) for value in pan_bbox]
    max_cols = 3 if int(item_count) > 4 else 4
    cols = min(int(max_cols), int(item_count))
    rows = (int(item_count) + int(cols) - 1) // int(cols)
    size = float(token_size_px)
    if rows > 1:
        available_h = max(1.0, bottom - top - float(token_gap_px))
        size = min(size, available_h / float(rows))
    row_gap = float(token_gap_px)
    col_gap = float(token_gap_px)
    total_h = rows * size + (rows - 1) * row_gap
    y0 = top + 0.5 * ((bottom - top) - total_h)
    boxes: List[Tuple[float, float, float, float]] = []
    remaining = int(item_count)
    index = 0
    for row in range(rows):
        row_cols = min(cols, remaining)
        total_w = row_cols * size + (row_cols - 1) * col_gap
        x0 = left + 0.5 * ((right - left) - total_w)
        for col in range(row_cols):
            x = x0 + col * (size + col_gap)
            y = y0 + row * (size + row_gap)
            boxes.append((x, y, x + size, y + size))
            index += 1
        remaining -= row_cols
    return boxes


def _draw_pan_items(
    draw: ImageDraw.ImageDraw,
    *,
    items: Sequence[Mapping[str, Any]],
    pan_bbox: Sequence[float],
    colors: Mapping[str, Tuple[int, int, int]],
    object_specs: Mapping[str, Mapping[str, Any]],
    item_bbox_map: Dict[str, List[float]],
    entities: List[Dict[str, Any]],
    params: BalanceScaleRenderParams,
    target_label: str,
    highlight_target: bool,
) -> None:
    """Draw all pan terms and register one bbox/entity per expanded item."""

    boxes = _item_bbox_grid(
        pan_bbox=pan_bbox,
        item_count=len(items),
        token_size_px=int(params.token_size_px),
        token_gap_px=int(params.token_gap_px),
    )
    for item, bbox in zip(items, boxes):
        item_id = str(item["item_id"])
        item_bbox_map[item_id] = round_bbox(bbox)
        if str(item["kind"]) == "object":
            object_label = str(item["object_label"])
            object_spec = object_specs[object_label]
            object_type = str(object_spec["object_type"])
            fill_rgb = tuple(int(value) for value in object_spec["fill_rgb"])
            _draw_object_token(
                draw,
                bbox=bbox,
                object_label=object_label,
                object_type=object_type,
                fill_rgb=fill_rgb,
                outline_rgb=colors["line"],
                text_rgb=colors["text"],
                text_stroke_rgb=colors["stroke"],
                stroke_width=int(params.line_width_px),
                label_font_size_px=int(params.label_font_size_px),
                highlight_rgb=(
                    colors["mark"]
                    if highlight_target and object_label == target_label
                    else None
                ),
            )
            entity_type = "balance_object_token"
        else:
            _draw_numeric_chip(
                draw,
                bbox=bbox,
                value=int(item["value"]),
                fill_rgb=colors["weight_fill"],
                outline_rgb=colors["line"],
                text_rgb=colors["text"],
                text_stroke_rgb=colors["stroke"],
                stroke_width=int(params.line_width_px),
                value_font_size_px=int(params.value_font_size_px),
            )
            entity_type = "balance_numeric_weight"
        entity = dict(item)
        entity.update(
            {
                "entity_id": item_id,
                "entity_type": entity_type,
                "bbox_px": list(item_bbox_map[item_id]),
            }
        )
        entities.append(entity)


def _draw_hanging_tray(
    draw: ImageDraw.ImageDraw,
    *,
    anchor_xy: Tuple[float, float],
    pan_bbox: Sequence[float],
    colors: Mapping[str, Tuple[int, int, int]],
    line_width: int,
) -> Tuple[float, float, float, float]:
    """Draw one hanging scale tray and return its usable content area."""

    anchor_x, anchor_y = [float(value) for value in anchor_xy]
    left, top, right, bottom = [float(value) for value in pan_bbox]
    width = right - left
    hanger_y = top + max(6.0, 0.09 * (bottom - top))
    tray_top = top + max(15.0, 0.18 * (bottom - top))
    tray_bottom = bottom
    left_hook = (left + 0.16 * width, hanger_y)
    right_hook = (right - 0.16 * width, hanger_y)
    instrument_width = max(4, int(line_width) + 2)
    hanger_width = max(3, int(line_width) + 1)

    draw.line(
        [(anchor_x, anchor_y), left_hook],
        fill=colors["line"],
        width=hanger_width,
    )
    draw.line(
        [(anchor_x, anchor_y), right_hook],
        fill=colors["line"],
        width=hanger_width,
    )
    draw.ellipse(
        (
            anchor_x - 5.0,
            anchor_y - 5.0,
            anchor_x + 5.0,
            anchor_y + 5.0,
        ),
        fill=colors["line"],
    )

    tray_points = [
        (left + 0.08 * width, tray_top),
        (right - 0.08 * width, tray_top),
        (right - 0.03 * width, tray_bottom - 4.0),
        (left + 0.03 * width, tray_bottom - 4.0),
    ]
    draw.polygon(
        tray_points,
        fill=colors["pan_fill"],
        outline=colors["line"],
    )
    draw.line(
        [tray_points[0], tray_points[1]],
        fill=colors["line"],
        width=instrument_width,
    )
    draw.line(
        [
            (left + 0.10 * width, tray_bottom - 8.0),
            (right - 0.10 * width, tray_bottom - 8.0),
        ],
        fill=colors["line"],
        width=max(3, int(line_width) + 1),
    )
    return (
        left + 0.07 * width,
        top + 8.0,
        right - 0.07 * width,
        tray_bottom - 12.0,
    )


def _draw_scale_panel(
    draw: ImageDraw.ImageDraw,
    *,
    panel: Mapping[str, Any],
    panel_bbox: Sequence[float],
    colors: Mapping[str, Tuple[int, int, int]],
    object_specs: Mapping[str, Mapping[str, Any]],
    item_bbox_map: Dict[str, List[float]],
    entities: List[Dict[str, Any]],
    params: BalanceScaleRenderParams,
    target_label: str,
    highlight_target: bool,
) -> None:
    """Draw one scale panel with beam tilt determined by left/right totals."""

    left, top, right, bottom = [float(value) for value in panel_bbox]
    pad = float(params.panel_padding_px)
    width = right - left
    scale_cx = 0.5 * (left + right)
    beam_y = top + 0.42 * (bottom - top)
    raw_pan_w = min(float(params.pan_width_px), 0.36 * width)
    pan_h = min(float(params.pan_height_px), max(88.0, 0.54 * (bottom - top)))
    max_beam_half = max(1.0, 0.5 * (width - raw_pan_w - 2.0 * pad))
    beam_half = min(0.5 * float(params.beam_width_px), max_beam_half)
    left_x = scale_cx - beam_half
    right_x = scale_cx + beam_half
    pan_w = min(raw_pan_w, max(1.0, 2.0 * (left_x - left - pad)))
    line_width = max(2, int(params.line_width_px))
    instrument_width = max(4, int(line_width) + 2)
    left_total = int(panel.get("left_total", 0))
    right_total = int(panel.get("right_total", 0))
    tilt_px = 0.0
    if left_total != right_total:
        tilt_px = min(36.0, max(26.0, 0.18 * (bottom - top)))
    left_offset = (
        tilt_px
        if left_total > right_total
        else -tilt_px if left_total < right_total else 0.0
    )
    right_offset = -left_offset

    title = str(panel.get("panel_label", ""))
    if title:
        draw_centered_text(
            draw,
            text=title,
            center=(left + pad + 36, top + pad * 0.75),
            font=load_font(max(14, int(0.90 * params.label_font_size_px)), bold=True),
            fill=colors["line"],
            stroke_fill=colors["stroke"],
            stroke_width=0,
        )

    beam_left = (left_x, beam_y + left_offset)
    beam_right = (right_x, beam_y + right_offset)
    draw.line([beam_left, beam_right], fill=colors["line"], width=instrument_width)
    draw.ellipse(
        (
            scale_cx - 8.5,
            beam_y - 8.5,
            scale_cx + 8.5,
            beam_y + 8.5,
        ),
        fill=colors["stand_fill"],
        outline=colors["line"],
        width=max(3, line_width + 1),
    )
    draw.polygon(
        [
            (scale_cx, beam_y + 10),
            (scale_cx - 30, beam_y + 78),
            (scale_cx + 30, beam_y + 78),
        ],
        fill=colors["stand_fill"],
        outline=colors["line"],
    )
    draw.line(
        [(scale_cx, beam_y + 70), (scale_cx, min(bottom - pad, beam_y + 92))],
        fill=colors["line"],
        width=instrument_width,
    )

    hanger_length = max(20.0, min(34.0, 0.14 * (bottom - top)))
    left_pan_top = beam_left[1] + hanger_length
    right_pan_top = beam_right[1] + hanger_length
    max_pan_bottom = bottom - max(2.0, 0.18 * pad)
    if max(left_pan_top, right_pan_top) + pan_h > max_pan_bottom:
        pan_h = max(72.0, max_pan_bottom - max(left_pan_top, right_pan_top))
    left_pan = (
        left_x - pan_w / 2.0,
        left_pan_top,
        left_x + pan_w / 2.0,
        left_pan_top + pan_h,
    )
    right_pan = (
        right_x - pan_w / 2.0,
        right_pan_top,
        right_x + pan_w / 2.0,
        right_pan_top + pan_h,
    )

    left_content_bbox = _draw_hanging_tray(
        draw,
        anchor_xy=beam_left,
        pan_bbox=left_pan,
        colors=colors,
        line_width=line_width,
    )
    right_content_bbox = _draw_hanging_tray(
        draw,
        anchor_xy=beam_right,
        pan_bbox=right_pan,
        colors=colors,
        line_width=line_width,
    )

    _draw_pan_items(
        draw,
        items=panel["left_items"],
        pan_bbox=left_content_bbox,
        colors=colors,
        object_specs=object_specs,
        item_bbox_map=item_bbox_map,
        entities=entities,
        params=params,
        target_label=target_label,
        highlight_target=highlight_target,
    )
    _draw_pan_items(
        draw,
        items=panel["right_items"],
        pan_bbox=right_content_bbox,
        colors=colors,
        object_specs=object_specs,
        item_bbox_map=item_bbox_map,
        entities=entities,
        params=params,
        target_label=target_label,
        highlight_target=highlight_target,
    )

    panel_id = str(panel["panel_id"])
    item_bbox_map[panel_id] = round_bbox(panel_bbox)
    entities.append(
        {
            "entity_id": panel_id,
            "entity_type": "balance_scale_panel",
            "bbox_px": list(item_bbox_map[panel_id]),
            "left_total": int(panel["left_total"]),
            "right_total": int(panel["right_total"]),
            "is_balanced": bool(panel["left_total"] == panel["right_total"]),
            "balance_state": str(panel.get("balance_state", "balanced")),
            "heavier_side": str(panel.get("heavier_side", "none")),
        }
    )


def _draw_missing_weight_query_row(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: Mapping[str, Any],
    row_bbox: Sequence[float],
    colors: Mapping[str, Tuple[int, int, int]],
    item_bbox_map: Dict[str, List[float]],
    entities: List[Dict[str, Any]],
    params: BalanceScaleRenderParams,
) -> None:
    """Draw the query row for object-to-missing-value questions."""

    left, top, right, bottom = [float(value) for value in row_bbox]
    cx = 0.5 * (left + right)
    cy = 0.5 * (top + bottom)
    token_size = float(params.token_size_px) * 1.18
    gap = 34.0
    eq_w = 34.0
    q_w = token_size * 1.22
    total_w = token_size + gap + eq_w + gap + q_w
    token_left = cx - total_w / 2.0
    token_bbox = (
        token_left,
        cy - token_size / 2.0,
        token_left + token_size,
        cy + token_size / 2.0,
    )
    eq_center_x = token_bbox[2] + gap + eq_w / 2.0
    q_left = eq_center_x + eq_w / 2.0 + gap
    question_bbox = (q_left, cy - token_size / 2.0, q_left + q_w, cy + token_size / 2.0)
    target_label = str(dataset["target_label"])
    object_spec = dataset["object_specs"][target_label]

    draw_centered_text(
        draw,
        text="Query",
        center=(left + 54, top + 0.5 * (bottom - top)),
        font=load_font(max(14, int(0.85 * params.label_font_size_px)), bold=True),
        fill=colors["line"],
        stroke_fill=colors["stroke"],
        stroke_width=0,
    )
    _draw_object_token(
        draw,
        bbox=token_bbox,
        object_label=target_label,
        object_type=str(object_spec["object_type"]),
        fill_rgb=object_spec["fill_rgb"],
        outline_rgb=colors["line"],
        text_rgb=colors["text"],
        text_stroke_rgb=colors["stroke"],
        stroke_width=int(params.line_width_px),
        label_font_size_px=int(params.query_font_size_px),
        highlight_rgb=colors["mark"],
    )
    draw_centered_text(
        draw,
        text="=",
        center=(eq_center_x, cy),
        font=load_font(max(18, int(params.query_font_size_px * 1.05)), bold=True),
        fill=colors["line"],
        stroke_fill=colors["stroke"],
        stroke_width=1,
    )
    draw_rounded_rect(
        draw,
        question_bbox,
        radius=12,
        fill=colors["query_fill"],
        outline=colors["mark"],
        width=max(2, int(params.line_width_px) + 1),
    )
    draw_centered_text(
        draw,
        text="?",
        center=((question_bbox[0] + question_bbox[2]) * 0.5, cy),
        font=load_font(max(20, int(params.query_font_size_px * 1.15)), bold=True),
        fill=colors["line"],
        stroke_fill=colors["stroke"],
        stroke_width=1,
    )

    item_bbox_map["query_object"] = round_bbox(token_bbox)
    item_bbox_map["missing_value_box"] = round_bbox(question_bbox)
    entities.append(
        {
            "entity_id": "query_object",
            "entity_type": "balance_query_object",
            "object_label": target_label,
            "bbox_px": list(item_bbox_map["query_object"]),
        }
    )
    entities.append(
        {
            "entity_id": "missing_value_box",
            "entity_type": "balance_missing_value_box",
            "bbox_px": list(item_bbox_map["missing_value_box"]),
        }
    )


def _draw_equivalent_count_query_row(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: Mapping[str, Any],
    row_bbox: Sequence[float],
    colors: Mapping[str, Tuple[int, int, int]],
    item_bbox_map: Dict[str, List[float]],
    entities: List[Dict[str, Any]],
    params: BalanceScaleRenderParams,
) -> None:
    """Draw the query row for source-object equals N repeated objects."""

    left, top, right, bottom = [float(value) for value in row_bbox]
    cx = 0.5 * (left + right)
    cy = 0.5 * (top + bottom)
    token_size = float(params.token_size_px) * 1.06
    gap = 24.0
    eq_w = 28.0
    mult_w = 28.0
    q_w = token_size * 1.10
    total_w = token_size + gap + eq_w + gap + q_w + gap + mult_w + gap + token_size
    source_left = cx - total_w / 2.0
    source_bbox = (
        source_left,
        cy - token_size / 2.0,
        source_left + token_size,
        cy + token_size / 2.0,
    )
    eq_center_x = source_bbox[2] + gap + eq_w / 2.0
    question_left = eq_center_x + eq_w / 2.0 + gap
    question_bbox = (
        question_left,
        cy - token_size / 2.0,
        question_left + q_w,
        cy + token_size / 2.0,
    )
    mult_center_x = question_bbox[2] + gap + mult_w / 2.0
    repeated_left = mult_center_x + mult_w / 2.0 + gap
    repeated_bbox = (
        repeated_left,
        cy - token_size / 2.0,
        repeated_left + token_size,
        cy + token_size / 2.0,
    )
    source_label = str(dataset["source_label"])
    repeated_label = str(dataset["repeated_label"])
    source_spec = dataset["object_specs"][source_label]
    repeated_spec = dataset["object_specs"][repeated_label]

    draw_centered_text(
        draw,
        text="Query",
        center=(left + 54, top + 0.5 * (bottom - top)),
        font=load_font(max(14, int(0.85 * params.label_font_size_px)), bold=True),
        fill=colors["line"],
        stroke_fill=colors["stroke"],
        stroke_width=0,
    )
    _draw_object_token(
        draw,
        bbox=source_bbox,
        object_label=source_label,
        object_type=str(source_spec["object_type"]),
        fill_rgb=source_spec["fill_rgb"],
        outline_rgb=colors["line"],
        text_rgb=colors["text"],
        text_stroke_rgb=colors["stroke"],
        stroke_width=int(params.line_width_px),
        label_font_size_px=int(params.query_font_size_px),
        highlight_rgb=colors["mark"],
    )
    draw_centered_text(
        draw,
        text="=",
        center=(eq_center_x, cy),
        font=load_font(max(18, int(params.query_font_size_px)), bold=True),
        fill=colors["line"],
        stroke_fill=colors["stroke"],
        stroke_width=0,
    )
    draw_rounded_rect(
        draw,
        question_bbox,
        radius=12,
        fill=colors["query_fill"],
        outline=colors["mark"],
        width=max(2, int(params.line_width_px) + 1),
    )
    draw_centered_text(
        draw,
        text="?",
        center=((question_bbox[0] + question_bbox[2]) * 0.5, cy),
        font=load_font(max(20, int(params.query_font_size_px * 1.05)), bold=True),
        fill=colors["text"],
        stroke_fill=colors["stroke"],
        stroke_width=1,
    )
    draw_centered_text(
        draw,
        text="x",
        center=(mult_center_x, cy),
        font=load_font(max(18, int(params.query_font_size_px * 0.92)), bold=True),
        fill=colors["text"],
        stroke_fill=colors["stroke"],
        stroke_width=1,
    )
    _draw_object_token(
        draw,
        bbox=repeated_bbox,
        object_label=repeated_label,
        object_type=str(repeated_spec["object_type"]),
        fill_rgb=repeated_spec["fill_rgb"],
        outline_rgb=colors["line"],
        text_rgb=colors["text"],
        text_stroke_rgb=colors["stroke"],
        stroke_width=int(params.line_width_px),
        label_font_size_px=int(params.query_font_size_px),
        highlight_rgb=colors["mark"],
    )

    item_bbox_map["source_object"] = round_bbox(source_bbox)
    item_bbox_map["repeated_object"] = round_bbox(repeated_bbox)
    item_bbox_map["missing_count_box"] = round_bbox(question_bbox)
    entities.append(
        {
            "entity_id": "source_object",
            "entity_type": "balance_query_source_object",
            "object_label": source_label,
            "bbox_px": list(item_bbox_map["source_object"]),
        }
    )
    entities.append(
        {
            "entity_id": "repeated_object",
            "entity_type": "balance_query_repeated_object",
            "object_label": repeated_label,
            "bbox_px": list(item_bbox_map["repeated_object"]),
        }
    )
    entities.append(
        {
            "entity_id": "missing_count_box",
            "entity_type": "balance_missing_count_box",
            "bbox_px": list(item_bbox_map["missing_count_box"]),
        }
    )


def _draw_weight_order_query_row(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: Mapping[str, Any],
    row_bbox: Sequence[float],
    colors: Mapping[str, Tuple[int, int, int]],
    item_bbox_map: Dict[str, List[float]],
    entities: List[Dict[str, Any]],
    params: BalanceScaleRenderParams,
) -> None:
    """Draw four lightest-to-heaviest order options in the query row."""

    left, top, right, bottom = [float(value) for value in row_bbox]
    cy = 0.5 * (top + bottom)
    options = [dict(option) for option in dataset.get("order_options", [])]
    if len(options) != 4:
        raise RuntimeError("weight-order query row requires exactly four options")

    title_w = 112.0
    gap = max(12.0, float(params.token_gap_px) * 1.5)
    option_w = max(160.0, (right - left - title_w - gap * 5) / 4.0)
    option_h = min(72.0, max(54.0, bottom - top - 20.0))
    start_x = left + 0.5 * ((right - left) - (title_w + gap + 4 * option_w + 3 * gap))

    draw_centered_text(
        draw,
        text="Order",
        center=(start_x + title_w / 2.0, cy),
        font=load_font(max(16, int(params.query_font_size_px * 0.76)), bold=True),
        fill=colors["text"],
        stroke_fill=colors["stroke"],
        stroke_width=1,
    )

    option_x = start_x + title_w + gap
    for index, option in enumerate(options):
        label = str(option["option_label"])
        order_text = str(option["order_text"])
        x0 = option_x + index * (option_w + gap)
        bbox = (x0, cy - option_h / 2.0, x0 + option_w, cy + option_h / 2.0)
        draw_rounded_rect(
            draw,
            bbox,
            radius=10,
            fill=colors["query_fill"],
            outline=colors["line"],
            width=max(1, int(params.line_width_px)),
        )
        badge_size = min(34.0, option_h - 16.0)
        badge_bbox = (
            bbox[0] + 10.0,
            cy - badge_size / 2.0,
            bbox[0] + 10.0 + badge_size,
            cy + badge_size / 2.0,
        )
        draw_rounded_rect(
            draw,
            badge_bbox,
            radius=int(badge_size / 2.0),
            fill=colors["panel_fill"],
            outline=colors["line"],
            width=max(1, int(params.line_width_px) - 1),
        )
        draw_centered_text(
            draw,
            text=label,
            center=((badge_bbox[0] + badge_bbox[2]) * 0.5, cy),
            font=load_font(max(14, int(params.label_font_size_px * 0.84)), bold=True),
            fill=colors["text"],
            stroke_fill=colors["stroke"],
            stroke_width=1,
        )
        draw_centered_text(
            draw,
            text=order_text,
            center=(
                bbox[0]
                + badge_size
                + 18.0
                + (bbox[2] - bbox[0] - badge_size - 24.0) / 2.0,
                cy,
            ),
            font=load_font(max(14, int(params.label_font_size_px * 0.84)), bold=True),
            fill=colors["text"],
            stroke_fill=colors["stroke"],
            stroke_width=1,
        )
        item_id = f"option_{label}"
        item_bbox_map[item_id] = round_bbox(bbox)
        entities.append(
            {
                "entity_id": item_id,
                "entity_type": "balance_order_option",
                "option_label": label,
                "order_text": order_text,
                "option_index": int(index),
                "bbox_px": list(item_bbox_map[item_id]),
            }
        )


def _draw_query_side_relation_row(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: Mapping[str, Any],
    row_bbox: Sequence[float],
    colors: Mapping[str, Tuple[int, int, int]],
    item_bbox_map: Dict[str, List[float]],
    entities: List[Dict[str, Any]],
    params: BalanceScaleRenderParams,
) -> None:
    """Draw one query comparison plus four relation-answer options."""

    left, top, right, bottom = [float(value) for value in row_bbox]
    row_h = max(1.0, bottom - top)
    title_w = 92.0
    gap = max(8.0, float(params.token_gap_px) * 1.35)
    content_left = left + title_w + gap
    content_right = right - gap
    expr_h = min(58.0, max(36.0, 0.50 * row_h))
    expr_top = top + max(5.0, 0.08 * row_h)
    expr_bottom = expr_top + expr_h
    option_top = expr_bottom + max(5.0, 0.05 * row_h)
    option_bottom = bottom - max(4.0, 0.04 * row_h)
    option_h = max(26.0, option_bottom - option_top)
    option_cy = option_top + 0.5 * option_h
    vs_w = 46.0
    expr_w = max(180.0, (content_right - content_left - vs_w - 2.0 * gap) / 2.0)
    expr_y = (expr_top, expr_bottom)
    left_expr = (
        content_left,
        expr_y[0],
        content_left + expr_w,
        expr_y[1],
    )
    vs_center_x = left_expr[2] + gap + 0.5 * vs_w
    right_expr = (
        vs_center_x + 0.5 * vs_w + gap,
        expr_y[0],
        min(content_right, vs_center_x + 0.5 * vs_w + gap + expr_w),
        expr_y[1],
    )

    draw_centered_text(
        draw,
        text="Query",
        center=(left + title_w * 0.5, top + 0.5 * row_h),
        font=load_font(max(14, int(params.query_font_size_px * 0.58)), bold=True),
        fill=colors["text"],
        stroke_fill=colors["stroke"],
        stroke_width=1,
    )
    for bbox in (left_expr, right_expr):
        draw_rounded_rect(
            draw,
            bbox,
            radius=10,
            fill=colors["query_fill"],
            outline=colors["line"],
            width=max(1, int(params.line_width_px)),
        )
    draw_centered_text(
        draw,
        text="vs",
        center=(vs_center_x, 0.5 * (expr_top + expr_bottom)),
        font=load_font(max(14, int(params.query_font_size_px * 0.55)), bold=True),
        fill=colors["text"],
        stroke_fill=colors["stroke"],
        stroke_width=1,
    )

    left_items = expand_terms(
        dataset["query_left_terms"],
        panel_name="query_relation",
        side_name="left",
        weights=dataset["object_weights"],
    )
    right_items = expand_terms(
        dataset["query_right_terms"],
        panel_name="query_relation",
        side_name="right",
        weights=dataset["object_weights"],
    )
    _draw_pan_items(
        draw,
        items=left_items,
        pan_bbox=(
            left_expr[0] + 8.0,
            left_expr[1] + 3.0,
            left_expr[2] - 8.0,
            left_expr[3] - 3.0,
        ),
        colors=colors,
        object_specs=dataset["object_specs"],
        item_bbox_map=item_bbox_map,
        entities=entities,
        params=params,
        target_label="",
        highlight_target=False,
    )
    _draw_pan_items(
        draw,
        items=right_items,
        pan_bbox=(
            right_expr[0] + 8.0,
            right_expr[1] + 3.0,
            right_expr[2] - 8.0,
            right_expr[3] - 3.0,
        ),
        colors=colors,
        object_specs=dataset["object_specs"],
        item_bbox_map=item_bbox_map,
        entities=entities,
        params=params,
        target_label="",
        highlight_target=False,
    )

    item_bbox_map["query_relation_box"] = round_bbox(
        (
            left_expr[0],
            expr_top,
            right_expr[2],
            expr_bottom,
        )
    )
    entities.append(
        {
            "entity_id": "query_relation_box",
            "entity_type": "balance_query_relation_box",
            "bbox_px": list(item_bbox_map["query_relation_box"]),
            "left_terms": [dict(term) for term in dataset["query_left_terms"]],
            "right_terms": [dict(term) for term in dataset["query_right_terms"]],
        }
    )

    options = [dict(option) for option in dataset.get("relation_options", [])]
    if len(options) != 4:
        raise RuntimeError("query-side relation row requires exactly four options")
    option_gap = max(7.0, float(params.token_gap_px))
    option_w = (content_right - content_left - 3.0 * option_gap) / 4.0
    badge_size = min(28.0, option_h - 8.0)
    for index, option in enumerate(options):
        option_label = str(option["option_label"])
        display_text = str(option["display_text"])
        x0 = content_left + int(index) * (option_w + option_gap)
        option_bbox = (x0, option_top, x0 + option_w, option_top + option_h)
        draw_rounded_rect(
            draw,
            option_bbox,
            radius=9,
            fill=colors["panel_fill"],
            outline=colors["line"],
            width=max(1, int(params.line_width_px) - 1),
        )
        badge_bbox = (
            option_bbox[0] + 7.0,
            option_cy - badge_size / 2.0,
            option_bbox[0] + 7.0 + badge_size,
            option_cy + badge_size / 2.0,
        )
        draw_rounded_rect(
            draw,
            badge_bbox,
            radius=int(badge_size / 2.0),
            fill=colors["query_fill"],
            outline=colors["line"],
            width=max(1, int(params.line_width_px) - 1),
        )
        draw_centered_text(
            draw,
            text=option_label,
            center=((badge_bbox[0] + badge_bbox[2]) * 0.5, option_cy),
            font=load_font(max(12, int(params.label_font_size_px * 0.62)), bold=True),
            fill=colors["text"],
            stroke_fill=colors["stroke"],
            stroke_width=1,
        )
        text_left = badge_bbox[2] + 7.0
        text_center_x = text_left + 0.5 * max(1.0, option_bbox[2] - text_left - 4.0)
        text_font_size = (
            int(params.label_font_size_px * 0.50)
            if len(display_text) > 11
            else int(params.label_font_size_px * 0.58)
        )
        draw_centered_text(
            draw,
            text=display_text,
            center=(text_center_x, option_cy),
            font=load_font(max(12, text_font_size), bold=True),
            fill=colors["text"],
            stroke_fill=colors["stroke"],
            stroke_width=1,
        )
        item_id = f"option_{option_label}"
        item_bbox_map[item_id] = round_bbox(option_bbox)
        entities.append(
            {
                "entity_id": item_id,
                "entity_type": "balance_relation_option",
                "option_label": option_label,
                "relation": str(option["relation"]),
                "display_text": display_text,
                "option_index": int(index),
                "bbox_px": list(item_bbox_map[item_id]),
            }
        )


def _draw_query_row(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: Mapping[str, Any],
    row_bbox: Sequence[float],
    colors: Mapping[str, Tuple[int, int, int]],
    item_bbox_map: Dict[str, List[float]],
    entities: List[Dict[str, Any]],
    params: BalanceScaleRenderParams,
) -> None:
    """Dispatch to the neutral query-row drawing grammar named by the dataset."""

    query_row_kind = str(dataset.get("query_row_kind", MISSING_WEIGHT_ROW_KIND))
    if query_row_kind == SIDE_RELATION_ROW_KIND:
        _draw_query_side_relation_row(
            draw,
            dataset=dataset,
            row_bbox=row_bbox,
            colors=colors,
            item_bbox_map=item_bbox_map,
            entities=entities,
            params=params,
        )
        return
    if query_row_kind == WEIGHT_ORDER_ROW_KIND:
        _draw_weight_order_query_row(
            draw,
            dataset=dataset,
            row_bbox=row_bbox,
            colors=colors,
            item_bbox_map=item_bbox_map,
            entities=entities,
            params=params,
        )
        return
    if query_row_kind == EQUIVALENT_COUNT_ROW_KIND:
        _draw_equivalent_count_query_row(
            draw,
            dataset=dataset,
            row_bbox=row_bbox,
            colors=colors,
            item_bbox_map=item_bbox_map,
            entities=entities,
            params=params,
        )
        return
    _draw_missing_weight_query_row(
        draw,
        dataset=dataset,
        row_bbox=row_bbox,
        colors=colors,
        item_bbox_map=item_bbox_map,
        entities=entities,
        params=params,
    )


def render_balance_scale_scene(
    image: Image.Image,
    *,
    dataset: Mapping[str, Any],
    scene_variant: str,
    render_params: BalanceScaleRenderParams,
    scene_style: PuzzleSceneStyle,
) -> RenderedBalanceScaleScene:
    """Render a static balance-scale puzzle scene."""

    draw = ImageDraw.Draw(image)
    panel_bbox = _panel_bbox(render_params)
    item_bbox_map: Dict[str, List[float]] = {"diagram_panel": round_bbox(panel_bbox)}
    entities: List[Dict[str, Any]] = [
        {
            "entity_id": "diagram_panel",
            "entity_type": "balance_scale_scene",
            "bbox_px": list(item_bbox_map["diagram_panel"]),
            "scene_variant": str(scene_variant),
            "query_row_kind": str(
                dataset.get("query_row_kind", MISSING_WEIGHT_ROW_KIND)
            ),
        }
    ]
    colors = {
        "panel_fill": tuple(int(value) for value in scene_style.panel_fill_rgb),
        "line": tuple(int(value) for value in scene_style.panel_border_rgb),
        "text": tuple(int(value) for value in scene_style.text_rgb),
        "muted_text": tuple(int(value) for value in scene_style.grid_rgb),
        "stroke": tuple(int(value) for value in scene_style.text_stroke_rgb),
        "mark": tuple(int(value) for value in scene_style.mark_rgb),
        "pan_fill": tuple(int(value) for value in scene_style.option_fill_rgb),
        "weight_fill": tuple(int(value) for value in scene_style.step_fill_rgb),
        "stand_fill": tuple(int(value) for value in scene_style.panel_accent_rgb),
        "query_fill": tuple(int(value) for value in scene_style.option_fill_rgb),
    }
    if str(scene_variant) == "balance_outline":
        draw.rounded_rectangle(
            panel_bbox,
            radius=int(render_params.panel_corner_radius_px),
            outline=colors["line"],
            width=int(render_params.panel_border_width_px),
        )
    elif str(scene_variant) == "balance_card":
        draw_puzzle_panel_chrome(
            draw,
            bbox=tuple(int(round(value)) for value in panel_bbox),
            style=scene_style,
            radius=int(render_params.panel_corner_radius_px),
            border_width=int(render_params.panel_border_width_px),
        )
    else:
        draw_rounded_rect(
            draw,
            panel_bbox,
            radius=int(render_params.panel_corner_radius_px),
            fill=colors["panel_fill"],
            outline=colors["line"],
            width=int(render_params.panel_border_width_px),
        )

    panels = [dict(panel) for panel in dataset["panels"]]
    gap = float(render_params.scale_panel_gap_px)
    inner_left = panel_bbox[0] + float(render_params.panel_padding_px)
    inner_right = panel_bbox[2] - float(render_params.panel_padding_px)
    inner_top = panel_bbox[1] + float(render_params.panel_padding_px)
    inner_bottom = panel_bbox[3] - float(render_params.panel_padding_px)
    query_h = float(render_params.query_row_height_px)
    available_h = inner_bottom - inner_top - query_h - gap * len(panels)
    scale_h = available_h / max(1, len(panels))
    y = inner_top
    highlight_target = False
    for panel in panels:
        scale_bbox = (inner_left, y, inner_right, y + scale_h)
        _draw_scale_panel(
            draw,
            panel=panel,
            panel_bbox=scale_bbox,
            colors=colors,
            object_specs=dataset["object_specs"],
            item_bbox_map=item_bbox_map,
            entities=entities,
            params=render_params,
            target_label=str(dataset["target_label"]),
            highlight_target=highlight_target,
        )
        y += scale_h + gap
    query_bbox = (inner_left, y, inner_right, min(inner_bottom, y + query_h))
    _draw_query_row(
        draw,
        dataset=dataset,
        row_bbox=query_bbox,
        colors=colors,
        item_bbox_map=item_bbox_map,
        entities=entities,
        params=render_params,
    )
    item_bbox_map["query_row"] = round_bbox(query_bbox)
    entities.append(
        {
            "entity_id": "query_row",
            "entity_type": "balance_query_row",
            "bbox_px": list(item_bbox_map["query_row"]),
        }
    )
    return RenderedBalanceScaleScene(
        image=image,
        entities=entities,
        scene_bbox_px=round_bbox(panel_bbox),
        item_bbox_map={str(key): list(value) for key, value in item_bbox_map.items()},
    )


def render_balance_scene_context(
    *,
    dataset: Mapping[str, Any],
    axes: Any,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    visual_defaults: Mapping[str, Any],
    namespace: str,
) -> RenderedBalanceContext:
    """Render one balance-scale dataset with resolved scene styling metadata."""

    render_params = resolve_render_params(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    scene_style, scene_style_meta = resolve_puzzle_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.scene_style",
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font_family",
        params={**dict(render_defaults), **dict(params)},
    )
    font_meta = {
        **get_font_family_record(str(font_family)).to_trace(),
        "font_asset_version": font_asset_version(),
        "selection_scope": "balance_scale_panel",
        "include_tags": [],
        "exclude_tags": [],
    }
    background, background_meta = make_puzzle_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=scene_style,
    )
    with temporary_default_font_family(str(font_family)):
        rendered_scene = render_balance_scale_scene(
            background,
            dataset=dataset,
            scene_variant=str(axes.scene_variant),
            render_params=render_params,
            scene_style=scene_style,
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=visual_defaults,
    )
    render_meta = {
        "scene_id": SCENE_ID,
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "coord_space": "pixel",
        "scene_variant": str(axes.scene_variant),
        "scene_style": dict(scene_style_meta),
        "background_style": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
        "scene_bbox_px": list(rendered_scene.scene_bbox_px),
        "text_style": {
            "font": dict(font_meta),
            "value_font_size_px": int(render_params.value_font_size_px),
            "label_font_size_px": int(render_params.label_font_size_px),
            "query_font_size_px": int(render_params.query_font_size_px),
        },
        "unit_size_jitter": dict(render_params.unit_size_jitter),
    }
    return RenderedBalanceContext(
        image=image,
        rendered_scene=rendered_scene,
        render_meta=render_meta,
        scene_style_meta=dict(scene_style_meta),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        font_meta=dict(font_meta),
    )


def balance_render_map(rendered_context: RenderedBalanceContext) -> Dict[str, Any]:
    """Return the common render-map payload for projected balance annotations."""

    rendered_scene = rendered_context.rendered_scene
    unit_jitter = dict(rendered_context.render_meta.get("unit_size_jitter", {}))
    return with_puzzle_unit_size_jitter(
        {
            "image_id": "img0",
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "item_bboxes_px": {
                str(item_name): list(value)
                for item_name, value in rendered_scene.item_bbox_map.items()
            },
            "annotation_source": "item_bboxes_px",
        },
        unit_jitter,
    )


__all__ = [
    "balance_render_map",
    "render_balance_scene_context",
    "render_balance_scale_scene",
]
