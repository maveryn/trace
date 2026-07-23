"""Rendering for the arithmetic-constraint puzzle scene."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.drawing import draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.shared.font_assets import (
    font_asset_version,
    font_role_trace,
    sample_font_family,
)
from trace_tasks.tasks.shared.text_rendering import load_font, temporary_default_font_family
from trace_tasks.tasks.shared.visual_defaults import default_noise_fallback

from .sampling import resolve_render_params
from .state import (
    ArithmeticCase,
    ArithmeticRenderParams,
    RenderedArithmeticScene,
    SCENE_ID,
)
from .styles import make_puzzle_scene_background, resolve_arithmetic_panel_style

ARITHMETIC_NOISE_FALLBACK = default_noise_fallback(apply_prob=0.12)
TARGET_SLOT_FILL_RGB = (255, 228, 225)
TARGET_SLOT_OUTLINE_RGB = (190, 38, 38)
TARGET_SLOT_TEXT_RGB = (125, 18, 18)
TARGET_SLOT_TEXT_STROKE_RGB = (255, 246, 244)


@dataclass(frozen=True)
class RenderedArithmeticContext:
    """Rendered arithmetic image with trace metadata and final geometry."""

    rendered_scene: RenderedArithmeticScene
    image: Image.Image
    render_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    panel_style_meta: Dict[str, Any]
    font_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


def _as_bbox(values: Sequence[float]) -> Tuple[float, float, float, float]:
    """Normalize one four-value bbox for trace storage."""

    rounded = tuple(round(float(value), 3) for value in values[:4])
    return rounded  # type: ignore[return-value]


def _expanded_bbox(
    cx: float, cy: float, radius_x: float, radius_y: float
) -> Tuple[float, float, float, float]:
    """Return a centered bbox around a rendered arithmetic token."""

    return _as_bbox(
        (
            float(cx - radius_x),
            float(cy - radius_y),
            float(cx + radius_x),
            float(cy + radius_y),
        )
    )


def _union_bbox(boxes: Iterable[Sequence[float]]) -> Tuple[float, float, float, float]:
    """Return the enclosing bbox for visible arithmetic content."""

    resolved = [tuple(float(value) for value in box[:4]) for box in boxes]
    if not resolved:
        return (0.0, 0.0, 1.0, 1.0)
    return _as_bbox(
        (
            min(box[0] for box in resolved),
            min(box[1] for box in resolved),
            max(box[2] for box in resolved),
            max(box[3] for box in resolved),
        )
    )


def _draw_value_box(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    text: str,
    render_params: ArithmeticRenderParams,
    style: Any,
    font,
    is_target: bool = False,
    entity_id: str,
    entity_type: str,
    item_bboxes: Dict[str, Tuple[float, float, float, float]],
    entities: list[Dict[str, Any]],
) -> None:
    """Draw one rectangular arithmetic slot and record its bbox/entity trace."""

    box = _as_bbox(bbox)
    fill = TARGET_SLOT_FILL_RGB if is_target else style.panel_fill_rgb
    outline = TARGET_SLOT_OUTLINE_RGB if is_target else style.grid_rgb
    width = max(2, int(render_params.panel_border_width_px + (2 if is_target else 0)))
    draw_rounded_rect(
        draw,
        tuple(box),
        radius=max(5, int(min(box[2] - box[0], box[3] - box[1]) * 0.14)),
        fill=fill,
        outline=outline,
        width=width,
    )
    draw_centered_text(
        draw,
        text=str(text),
        center=(0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])),
        font=font,
        fill=TARGET_SLOT_TEXT_RGB if is_target else style.text_rgb,
        stroke_fill=TARGET_SLOT_TEXT_STROKE_RGB if is_target else style.text_stroke_rgb,
        stroke_width=1,
    )
    item_bboxes[str(entity_id)] = box
    entities.append(
        {
            "entity_id": str(entity_id),
            "entity_type": str(entity_type),
            "bbox_px": list(box),
            "attrs": {
                "text": str(text),
                "is_target": bool(is_target),
            },
        }
    )


def _draw_value_node(
    draw: ImageDraw.ImageDraw,
    *,
    center: Sequence[float],
    radius: float,
    text: str,
    render_params: ArithmeticRenderParams,
    style: Any,
    font,
    is_target: bool,
    entity_id: str,
    item_bboxes: Dict[str, Tuple[float, float, float, float]],
    entities: list[Dict[str, Any]],
) -> None:
    """Draw one circular arithmetic node and record its bbox/entity trace."""

    cx, cy = float(center[0]), float(center[1])
    bbox = _expanded_bbox(cx, cy, float(radius), float(radius))
    fill = TARGET_SLOT_FILL_RGB if bool(is_target) else style.panel_fill_rgb
    outline = TARGET_SLOT_OUTLINE_RGB if bool(is_target) else style.grid_rgb
    draw.ellipse(
        tuple(bbox),
        fill=tuple(int(value) for value in fill),
        outline=tuple(int(value) for value in outline),
        width=max(
            2, int(render_params.panel_border_width_px + (2 if is_target else 0))
        ),
    )
    draw_centered_text(
        draw,
        text=str(text),
        center=(cx, cy),
        font=font,
        fill=TARGET_SLOT_TEXT_RGB if is_target else style.text_rgb,
        stroke_fill=TARGET_SLOT_TEXT_STROKE_RGB if is_target else style.text_stroke_rgb,
        stroke_width=1,
    )
    item_bboxes[str(entity_id)] = bbox
    entities.append(
        {
            "entity_id": str(entity_id),
            "entity_type": "arithmetic_node",
            "bbox_px": list(bbox),
            "attrs": {"text": str(text), "is_target": bool(is_target)},
        }
    )


def _draw_operator_text(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Sequence[float],
    font,
    style: Any,
    entities: list[Dict[str, Any]],
    entity_id: str,
) -> Tuple[float, float, float, float]:
    """Draw a visible operator or label token and return its bbox."""

    bbox = _as_bbox(
        draw_centered_text(
            draw,
            text=str(text),
            center=(float(center[0]), float(center[1])),
            font=font,
            fill=style.text_rgb,
            stroke_fill=style.text_stroke_rgb,
            stroke_width=1,
        )
    )
    entities.append(
        {
            "entity_id": str(entity_id),
            "entity_type": "arithmetic_text",
            "bbox_px": list(bbox),
            "attrs": {"text": str(text)},
        }
    )
    return bbox


def _draw_equal_sum_case(
    draw: ImageDraw.ImageDraw,
    *,
    case: ArithmeticCase,
    origin: Sequence[float],
    render_params: ArithmeticRenderParams,
    style: Any,
    fonts: Mapping[str, Any],
    item_bboxes: Dict[str, Tuple[float, float, float, float]],
    entities: list[Dict[str, Any]],
) -> list[Tuple[float, float, float, float]]:
    """Render a polygon-side arithmetic constraint with one hidden midpoint."""

    cx, cy = float(origin[0]), float(origin[1])
    radius = float(
        max(render_params.cell_width_px * 1.8, render_params.node_radius_px * 3.0)
    )
    side_count = int(case.data["side_count"])
    corners = [int(value) for value in case.data["corner_values"]]
    mids = list(case.data["middle_values"])
    angle_offset = -math.pi / 2.0
    corner_points = [
        (
            float(
                cx
                + radius * math.cos(angle_offset + (2.0 * math.pi * index / side_count))
            ),
            float(
                cy
                + radius * math.sin(angle_offset + (2.0 * math.pi * index / side_count))
            ),
        )
        for index in range(side_count)
    ]
    boxes: list[Tuple[float, float, float, float]] = []
    for index in range(side_count):
        start = corner_points[index]
        end = corner_points[(index + 1) % side_count]
        draw.line(
            [start, end],
            fill=style.grid_rgb,
            width=max(2, int(render_params.line_width_px)),
        )
    for index, point in enumerate(corner_points):
        _draw_value_node(
            draw,
            center=point,
            radius=render_params.node_radius_px,
            text=str(corners[index]),
            render_params=render_params,
            style=style,
            font=fonts["value"],
            is_target=False,
            entity_id=f"corner_{index}",
            item_bboxes=item_bboxes,
            entities=entities,
        )
        boxes.append(item_bboxes[f"corner_{index}"])
    target_side = int(case.data["target_side"])
    for index in range(side_count):
        start = corner_points[index]
        end = corner_points[(index + 1) % side_count]
        point = (0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1]))
        is_target = index == target_side
        entity_id = "target" if is_target else f"side_mid_{index}"
        _draw_value_node(
            draw,
            center=point,
            radius=render_params.node_radius_px,
            text="?" if is_target else str(mids[index]),
            render_params=render_params,
            style=style,
            font=fonts["value"],
            is_target=is_target,
            entity_id=entity_id,
            item_bboxes=item_bboxes,
            entities=entities,
        )
        boxes.append(item_bboxes[entity_id])
    return boxes


def _draw_vertical_case(
    draw: ImageDraw.ImageDraw,
    *,
    case: ArithmeticCase,
    origin: Sequence[float],
    render_params: ArithmeticRenderParams,
    style: Any,
    fonts: Mapping[str, Any],
    item_bboxes: Dict[str, Tuple[float, float, float, float]],
    entities: list[Dict[str, Any]],
) -> list[Tuple[float, float, float, float]]:
    """Render a stacked addition/subtraction problem with one hidden digit."""

    rows = list(case.data["rows"])
    cell = float(render_params.cell_height_px)
    gap = float(max(4, cell * 0.08))
    max_width = max(len(row[1]) for row in rows)
    total_w = max_width * cell + (max_width - 1) * gap
    left = float(origin[0] - total_w / 2.0)
    top = float(origin[1] - len(rows) * cell * 0.55)
    boxes: list[Tuple[float, float, float, float]] = []
    for row_index, (_role, digits, hidden_place) in enumerate(rows):
        row_y = float(top + row_index * (cell + gap))
        for col_index, digit in enumerate(digits):
            x = float(left + col_index * (cell + gap))
            is_target = hidden_place is not None and int(col_index) == int(hidden_place)
            entity_id = "target" if is_target else f"digit_{row_index}_{col_index}"
            _draw_value_box(
                draw,
                bbox=(x, row_y, x + cell, row_y + cell),
                text="?" if is_target else str(digit),
                render_params=render_params,
                style=style,
                font=fonts["value"],
                is_target=is_target,
                entity_id=entity_id,
                entity_type="vertical_digit",
                item_bboxes=item_bboxes,
                entities=entities,
            )
            boxes.append(item_bboxes[entity_id])
    operator = "+" if str(case.data["operation"]) == "addition" else "-"
    boxes.append(
        _draw_operator_text(
            draw,
            text=operator,
            center=(left - cell * 0.45, top + cell * 1.5),
            font=fonts["symbol"],
            style=style,
            entities=entities,
            entity_id="vertical_operator",
        )
    )
    line_y = float(top + 2.0 * (cell + gap) - gap * 0.35)
    draw.line(
        [(left - cell * 0.25, line_y), (left + total_w, line_y)],
        fill=style.grid_rgb,
        width=max(2, int(render_params.line_width_px)),
    )
    return boxes


def _draw_operation_table_case(
    draw: ImageDraw.ImageDraw,
    *,
    case: ArithmeticCase,
    origin: Sequence[float],
    render_params: ArithmeticRenderParams,
    style: Any,
    fonts: Mapping[str, Any],
    item_bboxes: Dict[str, Tuple[float, float, float, float]],
    entities: list[Dict[str, Any]],
) -> list[Tuple[float, float, float, float]]:
    """Render a small arithmetic operation table with one target body cell."""

    row_headers = [int(value) for value in case.data["row_headers"]]
    col_headers = [int(value) for value in case.data["col_headers"]]
    operator = str(case.data["operator"])
    cell_w = float(render_params.cell_width_px)
    cell_h = float(render_params.cell_height_px)
    rows = len(row_headers) + 1
    cols = len(col_headers) + 1
    total_w = cols * cell_w
    total_h = rows * cell_h
    left = float(origin[0] - total_w / 2.0)
    top = float(origin[1] - total_h / 2.0)
    boxes: list[Tuple[float, float, float, float]] = []
    for r in range(rows):
        for c in range(cols):
            x0 = left + c * cell_w
            y0 = top + r * cell_h
            is_corner = r == 0 and c == 0
            is_header = r == 0 or c == 0
            is_target = r == 1 and c == 1
            if is_corner:
                text = operator
            elif r == 0:
                text = str(col_headers[c - 1])
            elif c == 0:
                text = str(row_headers[r - 1])
            elif is_target:
                text = "?"
            else:
                rv, cv = row_headers[r - 1], col_headers[c - 1]
                text = str(rv * cv if operator == "x" else rv + cv)
            entity_id = "target" if is_target else f"table_{r}_{c}"
            _draw_value_box(
                draw,
                bbox=(x0, y0, x0 + cell_w, y0 + cell_h),
                text=text,
                render_params=render_params,
                style=style,
                font=fonts["value"] if not is_header else fonts["symbol"],
                is_target=is_target,
                entity_id=entity_id,
                entity_type="operation_table_cell",
                item_bboxes=item_bboxes,
                entities=entities,
            )
            boxes.append(item_bboxes[entity_id])
    return boxes


def _draw_row_column_case(
    draw: ImageDraw.ImageDraw,
    *,
    case: ArithmeticCase,
    origin: Sequence[float],
    render_params: ArithmeticRenderParams,
    style: Any,
    fonts: Mapping[str, Any],
    item_bboxes: Dict[str, Tuple[float, float, float, float]],
    entities: list[Dict[str, Any]],
) -> list[Tuple[float, float, float, float]]:
    """Render a sum grid with row and column totals on the outside."""

    values = [[int(value) for value in row] for row in case.data["values"]]
    row_totals = [int(value) for value in case.data["row_totals"]]
    col_totals = [int(value) for value in case.data["col_totals"]]
    target_row = int(case.data["target_row"])
    target_col = int(case.data["target_col"])
    cell_w = float(render_params.cell_width_px)
    cell_h = float(render_params.cell_height_px)
    rows = len(values) + 1
    cols = len(values[0]) + 1
    total_w = cols * cell_w
    total_h = rows * cell_h
    left = float(origin[0] - total_w / 2.0)
    top = float(origin[1] - total_h / 2.0)
    boxes: list[Tuple[float, float, float, float]] = []
    for r in range(rows):
        for c in range(cols):
            if r == rows - 1 and c == cols - 1:
                continue
            x0 = left + c * cell_w
            y0 = top + r * cell_h
            is_total = r == rows - 1 or c == cols - 1
            is_target = r == target_row and c == target_col
            if r == rows - 1:
                text = str(col_totals[c])
            elif c == cols - 1:
                text = str(row_totals[r])
            elif is_target:
                text = "?"
            else:
                text = str(values[r][c])
            entity_id = "target" if is_target else f"sum_{r}_{c}"
            _draw_value_box(
                draw,
                bbox=(x0, y0, x0 + cell_w, y0 + cell_h),
                text=text,
                render_params=render_params,
                style=style,
                font=fonts["value"] if not is_total else fonts["symbol"],
                is_target=is_target,
                entity_id=entity_id,
                entity_type="sum_grid_cell",
                item_bboxes=item_bboxes,
                entities=entities,
            )
            boxes.append(item_bboxes[entity_id])
    return boxes


def _draw_number_wall_case(
    draw: ImageDraw.ImageDraw,
    *,
    case: ArithmeticCase,
    origin: Sequence[float],
    render_params: ArithmeticRenderParams,
    style: Any,
    fonts: Mapping[str, Any],
    item_bboxes: Dict[str, Tuple[float, float, float, float]],
    entities: list[Dict[str, Any]],
) -> list[Tuple[float, float, float, float]]:
    """Render a stacked number wall with one hidden brick at any level."""

    levels = [[int(value) for value in level] for level in case.data["levels"]]
    brick_w = float(render_params.cell_width_px * 1.2)
    brick_h = float(render_params.cell_height_px)
    gap_y = float(max(4, brick_h * 0.08))
    max_count = max(len(level) for level in levels)
    total_w = max_count * brick_w
    total_h = len(levels) * brick_h + (len(levels) - 1) * gap_y
    left = float(origin[0] - total_w / 2.0)
    bottom = float(origin[1] + total_h / 2.0)
    boxes: list[Tuple[float, float, float, float]] = []
    for level_index, level in enumerate(levels):
        y1 = bottom - (level_index + 1) * brick_h - level_index * gap_y
        row_left = left + (max_count - len(level)) * brick_w / 2.0
        for brick_index, value in enumerate(level):
            x0 = row_left + brick_index * brick_w
            is_target = level_index == int(
                case.data["target_level"]
            ) and brick_index == int(case.data["target_index"])
            entity_id = "target" if is_target else f"brick_{level_index}_{brick_index}"
            _draw_value_box(
                draw,
                bbox=(x0, y1, x0 + brick_w, y1 + brick_h),
                text="?" if is_target else str(value),
                render_params=render_params,
                style=style,
                font=fonts["value"],
                is_target=is_target,
                entity_id=entity_id,
                entity_type="number_wall_brick",
                item_bboxes=item_bboxes,
                entities=entities,
            )
            boxes.append(item_bboxes[entity_id])
    return boxes


def _draw_case_body(
    draw: ImageDraw.ImageDraw,
    *,
    case: ArithmeticCase,
    body_center: Sequence[float],
    render_params: ArithmeticRenderParams,
    style: Any,
    fonts: Mapping[str, Any],
    item_bboxes: Dict[str, Tuple[float, float, float, float]],
    entities: list[Dict[str, Any]],
) -> list[Tuple[float, float, float, float]]:
    """Dispatch one scene-internal arithmetic case kind to its renderer."""

    if case.layout_style == "polygon_side_sum":
        return _draw_equal_sum_case(
            draw,
            case=case,
            origin=body_center,
            render_params=render_params,
            style=style,
            fonts=fonts,
            item_bboxes=item_bboxes,
            entities=entities,
        )
    if case.layout_style == "stacked_arithmetic":
        return _draw_vertical_case(
            draw,
            case=case,
            origin=body_center,
            render_params=render_params,
            style=style,
            fonts=fonts,
            item_bboxes=item_bboxes,
            entities=entities,
        )
    if case.layout_style == "operator_grid":
        return _draw_operation_table_case(
            draw,
            case=case,
            origin=body_center,
            render_params=render_params,
            style=style,
            fonts=fonts,
            item_bboxes=item_bboxes,
            entities=entities,
        )
    if case.layout_style == "sum_grid":
        return _draw_row_column_case(
            draw,
            case=case,
            origin=body_center,
            render_params=render_params,
            style=style,
            fonts=fonts,
            item_bboxes=item_bboxes,
            entities=entities,
        )
    if case.layout_style == "stacked_wall":
        return _draw_number_wall_case(
            draw,
            case=case,
            origin=body_center,
            render_params=render_params,
            style=style,
            fonts=fonts,
            item_bboxes=item_bboxes,
            entities=entities,
        )
    raise ValueError(f"unsupported arithmetic layout style: {case.layout_style}")


def render_arithmetic_case(
    *,
    case: ArithmeticCase,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    visual_defaults: Mapping[str, Any] | None = None,
) -> RenderedArithmeticContext:
    """Render one arithmetic puzzle from symbolic case data and final layout."""

    unit_jitter = params.get("unit_size_jitter")
    render_params = resolve_render_params(
        params=params,
        render_defaults=render_defaults,
        unit_size_jitter=unit_jitter if isinstance(unit_jitter, Mapping) else {},
    )
    style, style_meta = resolve_arithmetic_panel_style(
        instance_seed=int(instance_seed), params=params
    )
    style_meta["target_slot_fill_rgb"] = list(TARGET_SLOT_FILL_RGB)
    style_meta["target_slot_outline_rgb"] = list(TARGET_SLOT_OUTLINE_RGB)
    style_meta["target_slot_text_rgb"] = list(TARGET_SLOT_TEXT_RGB)
    background, background_meta = make_puzzle_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=style,
    )
    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)

    rng = spawn_rng(int(instance_seed), "puzzles.arithmetic_panel.layout")
    panel_w = int(render_params.canvas_width * 0.80)
    panel_h = int(render_params.canvas_height * 0.74)
    free_x = max(0, int(render_params.canvas_width - panel_w))
    free_y = max(0, int(render_params.canvas_height - panel_h))
    offset_x = int(rng.randint(-max(1, free_x // 5), max(1, free_x // 5)))
    offset_y = int(rng.randint(-max(1, free_y // 5), max(1, free_y // 5)))
    panel_left = int((render_params.canvas_width - panel_w) / 2 + offset_x)
    panel_top = int((render_params.canvas_height - panel_h) / 2 + offset_y)
    panel_bbox = (
        max(16, panel_left),
        max(16, panel_top),
        min(render_params.canvas_width - 16, panel_left + panel_w),
        min(render_params.canvas_height - 16, panel_top + panel_h),
    )
    scene_variant = str(params.get("scene_variant", "constraint_sheet"))
    panel_fill = (
        style.background_rgb
        if scene_variant == "constraint_outline"
        else style.panel_fill_rgb
    )
    panel_outline = (
        style.panel_accent_rgb
        if scene_variant == "constraint_outline"
        else style.panel_border_rgb
    )
    panel_width = max(
        1,
        int(
            render_params.panel_border_width_px
            + (1 if scene_variant == "constraint_outline" else 0)
        ),
    )
    draw_rounded_rect(
        draw,
        panel_bbox,
        radius=int(render_params.panel_corner_radius_px),
        fill=panel_fill,
        outline=panel_outline,
        width=panel_width,
    )

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="puzzles.arithmetic_panel.font_family",
        params=params,
    )
    font_meta = font_role_trace(str(font_family), role="readout")
    font_meta["font_size_px"] = int(render_params.value_font_size_px)
    font_meta["font_asset_version"] = font_asset_version()
    fonts = {
        "value": load_font(
            int(render_params.value_font_size_px),
            bold=True,
            font_family=str(font_family),
        ),
        "note": load_font(
            int(render_params.note_font_size_px),
            bold=False,
            font_family=str(font_family),
        ),
        "symbol": load_font(
            int(render_params.symbol_font_size_px),
            bold=True,
            font_family=str(font_family),
        ),
    }

    item_bboxes: Dict[str, Tuple[float, float, float, float]] = {
        "diagram_panel": _as_bbox(panel_bbox),
    }
    entities: list[Dict[str, Any]] = [
        {
            "entity_id": "diagram_panel",
            "entity_type": "arithmetic_panel",
            "bbox_px": list(item_bboxes["diagram_panel"]),
            "attrs": {"scene_variant": str(case.kind)},
        }
    ]
    body_center = (
        0.5 * (panel_bbox[0] + panel_bbox[2]),
        panel_bbox[1] + 0.50 * (panel_bbox[3] - panel_bbox[1]),
    )
    with temporary_default_font_family(str(font_family)):
        content_boxes = _draw_case_body(
            draw,
            case=case,
            body_center=body_center,
            render_params=render_params,
            style=style,
            fonts=fonts,
            item_bboxes=item_bboxes,
            entities=entities,
        )
    scene_bbox = _union_bbox(content_boxes + [item_bboxes["diagram_panel"]])
    rendered_scene = RenderedArithmeticScene(
        image=image,
        entities=tuple(entities),
        scene_bbox_px=scene_bbox,
        item_bbox_map=dict(item_bboxes),
    )
    final_image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=visual_defaults or ARITHMETIC_NOISE_FALLBACK,
    )
    render_meta = {
        "scene_id": SCENE_ID,
        "case_kind": str(case.kind),
        "layout_style": str(case.layout_style),
        "canvas_width": int(final_image.size[0]),
        "canvas_height": int(final_image.size[1]),
        "layout_jitter": {
            "panel_offset_px": [int(offset_x), int(offset_y)],
            "based_on_available_space": True,
            "applied_before_annotation_projection": True,
        },
        "unit_size_jitter": dict(render_params.unit_size_jitter),
    }
    return RenderedArithmeticContext(
        rendered_scene=rendered_scene,
        image=final_image,
        render_meta=render_meta,
        background_meta=dict(background_meta),
        panel_style_meta=dict(style_meta),
        font_meta=dict(font_meta),
        post_noise_meta=dict(post_noise_meta),
    )


__all__ = [
    "ARITHMETIC_NOISE_FALLBACK",
    "RenderedArithmeticContext",
    "render_arithmetic_case",
]
