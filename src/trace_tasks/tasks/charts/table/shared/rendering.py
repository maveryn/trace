"""Scene-local rendering helpers for styled table chart tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.charts.shared.dense_text import dense_fit_bold, dense_stroke_width
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.text_rendering import draw_text_centered, fit_font_to_box, load_font

from .state import TableDefaults


SUPPORTED_TABLE_SCENE_VARIANTS: Tuple[str, ...] = (
    "spreadsheet",
    "zebra",
    "ledger",
    "card_table",
)


@dataclass(frozen=True)
class TableRenderParams:
    """Resolved render parameters for one table scene."""

    canvas_width: int
    canvas_height: int
    table_margin_left_px: int
    table_margin_right_px: int
    table_margin_top_px: int
    table_margin_bottom_px: int
    row_label_width_fraction: float
    row_label_min_width_px: int
    header_fill_rgb: Tuple[int, int, int]
    zebra_row_fill_rgb: Tuple[int, int, int]
    card_fill_rgb: Tuple[int, int, int]
    border_color_rgb: Tuple[int, int, int]
    grid_color_rgb: Tuple[int, int, int]
    text_color_rgb: Tuple[int, int, int]
    text_stroke_rgb: Tuple[int, int, int]
    header_style: str
    header_dark_fill_rgb: Tuple[int, int, int]
    header_dark_text_rgb: Tuple[int, int, int]
    frame_style: str
    shadow_color_rgb: Tuple[int, int, int]
    shadow_offset_px: int
    inner_rule_style: str
    numeric_alignment: str
    label_font_size_px: int
    value_font_size_px: int
    border_width_px: int
    grid_width_px: int
    rounded_corner_radius_px: int
    cell_padding_px: int


@dataclass(frozen=True)
class RenderedTableScene:
    """Rendered image plus table geometry traces."""

    image: Image.Image
    entities: List[Dict[str, Any]]
    cell_traces: List[Dict[str, Any]]
    table_bbox_px: List[float]
    numeric_table_region_bbox: List[float]
    row_region_bboxes: Dict[str, List[float]]
    column_region_bboxes: Dict[str, List[float]]
    row_label_bboxes: Dict[str, List[float]]
    header_bboxes: Dict[str, List[float]]


def _text_size(draw: ImageDraw.ImageDraw, text: str, *, font) -> Tuple[float, float]:
    """Return text width and height in pixels."""

    bbox = draw.textbbox((0, 0), str(text), font=font)
    return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])


def _text_bbox_from_center(*, center: Tuple[float, float], text: str, draw: ImageDraw.ImageDraw, font) -> List[float]:
    """Return one centered text bbox in pixel coordinates."""

    width, height = _text_size(draw, str(text), font=font)
    cx, cy = float(center[0]), float(center[1])
    return [
        round(float(cx - (0.5 * width)), 3),
        round(float(cy - (0.5 * height)), 3),
        round(float(cx + (0.5 * width)), 3),
        round(float(cy + (0.5 * height)), 3),
    ]


def _draw_text_in_cell(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    bbox: Tuple[float, float, float, float],
    font,
    bold: bool,
    fill_rgb: Sequence[int],
    stroke_rgb: Sequence[int],
    align: str,
    padding_px: int,
) -> List[float]:
    """Draw one short text string inside a cell and return its bbox."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    max_width = max(1.0, float(x1 - x0) - (2.0 * float(padding_px)))
    max_height = max(1.0, float(y1 - y0) - (2.0 * float(padding_px)))
    fitted_font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=max_width,
        max_height=max_height,
        bold=bool(bold),
        min_size_px=8,
        max_size_px=int(getattr(font, "size", 14)),
        fill_ratio=0.96,
    )
    text_width, _ = _text_size(draw, str(text), font=fitted_font)
    if str(align) == "left":
        center = (
            float(x0 + float(padding_px) + (0.5 * text_width)),
            float(0.5 * (y0 + y1)),
        )
    elif str(align) == "right":
        center = (
            float(x1 - float(padding_px) - (0.5 * text_width)),
            float(0.5 * (y0 + y1)),
        )
    else:
        center = (
            float(0.5 * (x0 + x1)),
            float(0.5 * (y0 + y1)),
        )
    draw_text_centered(
        draw,
        text=str(text),
        center=center,
        font=fitted_font,
        fill=tuple(int(value) for value in fill_rgb),
        stroke_fill=tuple(int(value) for value in stroke_rgb),
        stroke_width=dense_stroke_width(),
    )
    return _text_bbox_from_center(center=center, text=str(text), draw=draw, font=fitted_font)


def _union_bboxes(boxes: Sequence[Sequence[float]]) -> List[float]:
    """Return one union bbox over one non-empty bbox sequence."""

    if not boxes:
        raise ValueError("boxes must be non-empty")
    return [
        round(float(min(float(box[0]) for box in boxes)), 3),
        round(float(min(float(box[1]) for box in boxes)), 3),
        round(float(max(float(box[2]) for box in boxes)), 3),
        round(float(max(float(box[3]) for box in boxes)), 3),
    ]


def _trace_cell_value(value: Any | None) -> Any | None:
    """Return a JSON-safe trace value for one rendered table cell."""

    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return int(value)
    return str(value)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Tuple[float, float]],
    *,
    fill: Sequence[int],
    width: int,
    dash_px: int = 8,
    gap_px: int = 5,
) -> None:
    """Draw one horizontal or vertical dashed line."""

    if len(points) != 2:
        raise ValueError("dashed table rules require exactly two points")
    (x0, y0), (x1, y1) = points
    x0 = float(x0)
    x1 = float(x1)
    y0 = float(y0)
    y1 = float(y1)
    if abs(x0 - x1) >= abs(y0 - y1):
        start = min(x0, x1)
        end = max(x0, x1)
        cursor = start
        y = y0
        while cursor < end:
            segment_end = min(end, cursor + int(dash_px))
            draw.line([(cursor, y), (segment_end, y)], fill=tuple(int(v) for v in fill), width=int(width))
            cursor = segment_end + int(gap_px)
    else:
        start = min(y0, y1)
        end = max(y0, y1)
        cursor = start
        x = x0
        while cursor < end:
            segment_end = min(end, cursor + int(dash_px))
            draw.line([(x, cursor), (x, segment_end)], fill=tuple(int(v) for v in fill), width=int(width))
            cursor = segment_end + int(gap_px)


def _draw_inner_rule(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Tuple[float, float]],
    *,
    params: TableRenderParams,
) -> None:
    """Draw one deterministic inner table rule."""

    style = str(params.inner_rule_style)
    width = max(1, int(params.grid_width_px))
    fill_rgb = tuple(int(value) for value in params.grid_color_rgb)
    if style == "soft":
        width = 1
        fill_rgb = tuple(int(0.55 * int(value) + 0.45 * 255) for value in params.grid_color_rgb)
    if style == "dashed":
        _draw_dashed_line(
            draw,
            points,
            fill=fill_rgb,
            width=width,
        )
        return
    draw.line(
        points,
        fill=fill_rgb,
        width=width,
    )


def _style_fill_for_cell(
    *,
    scene_variant: str,
    row_index: int,
    col_index: int,
    params: TableRenderParams,
) -> Tuple[int, int, int]:
    """Resolve the fill color for one cell under the selected style."""

    if int(row_index) == 0:
        if str(params.header_style) == "dark":
            return tuple(int(value) for value in params.header_dark_fill_rgb)
        return tuple(int(value) for value in params.header_fill_rgb)
    if str(scene_variant) == "zebra" and int(row_index) % 2 == 0:
        return tuple(int(value) for value in params.zebra_row_fill_rgb)
    return tuple(int(value) for value in params.card_fill_rgb)


def render_table_scene(
    background: Image.Image,
    *,
    scene_variant: str,
    row_labels: Sequence[str],
    column_headers: Sequence[str],
    values_by_row: Mapping[str, Mapping[str, Any]],
    render_params: TableRenderParams,
) -> RenderedTableScene:
    """Render one styled table with one leftmost row-label column."""

    selected_variant = str(scene_variant)
    if selected_variant not in set(SUPPORTED_TABLE_SCENE_VARIANTS):
        raise ValueError(f"unsupported table scene_variant: {selected_variant}")
    resolved_row_labels = [str(label) for label in row_labels]
    resolved_headers = [str(header) for header in column_headers]
    if not resolved_row_labels or not resolved_headers:
        raise ValueError("tables require non-empty row labels and column headers")

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    label_font = load_font(int(render_params.label_font_size_px), bold=dense_fit_bold())
    value_font = load_font(int(render_params.value_font_size_px), bold=False)

    table_left = float(render_params.table_margin_left_px)
    table_top = float(render_params.table_margin_top_px)
    table_right = float(render_params.canvas_width - render_params.table_margin_right_px)
    table_bottom = float(render_params.canvas_height - render_params.table_margin_bottom_px)
    table_bbox = (table_left, table_top, table_right, table_bottom)
    table_width = float(table_right - table_left)
    table_height = float(table_bottom - table_top)
    if table_width <= 0 or table_height <= 0:
        raise ValueError("table margins must leave positive renderable table area")

    row_label_width = max(
        float(render_params.row_label_min_width_px),
        float(render_params.row_label_width_fraction) * float(table_width),
    )
    if table_width <= row_label_width:
        raise ValueError("table layout must leave positive numeric-column area")
    numeric_col_width = float(table_width - row_label_width) / float(len(resolved_headers))
    total_rows = int(len(resolved_row_labels) + 1)
    row_height = float(table_height) / float(total_rows)

    if str(render_params.frame_style) == "shadow":
        shadow_offset = max(1, int(render_params.shadow_offset_px))
        shadow_bbox = (
            float(table_bbox[0] + shadow_offset),
            float(table_bbox[1] + shadow_offset),
            float(table_bbox[2] + shadow_offset),
            float(table_bbox[3] + shadow_offset),
        )
        if selected_variant == "card_table":
            draw.rounded_rectangle(
                shadow_bbox,
                radius=int(render_params.rounded_corner_radius_px),
                fill=tuple(int(value) for value in render_params.shadow_color_rgb),
            )
        else:
            draw.rectangle(
                shadow_bbox,
                fill=tuple(int(value) for value in render_params.shadow_color_rgb),
            )

    if selected_variant == "card_table":
        draw.rounded_rectangle(
            table_bbox,
            radius=int(render_params.rounded_corner_radius_px),
            fill=tuple(int(value) for value in render_params.card_fill_rgb),
            outline=tuple(int(value) for value in render_params.border_color_rgb),
            width=int(render_params.border_width_px),
        )
    else:
        draw.rectangle(
            table_bbox,
            fill=tuple(int(value) for value in render_params.card_fill_rgb),
            outline=tuple(int(value) for value in render_params.border_color_rgb),
            width=int(render_params.border_width_px),
        )

    cell_traces: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    row_region_inputs: Dict[str, List[List[float]]] = {str(label): [] for label in resolved_row_labels}
    column_region_inputs: Dict[str, List[List[float]]] = {str(header): [] for header in resolved_headers}
    row_label_bboxes: Dict[str, List[float]] = {}
    header_bboxes: Dict[str, List[float]] = {}

    def cell_bbox(*, row_index: int, col_index: int) -> Tuple[float, float, float, float]:
        left = float(table_left if int(col_index) == 0 else table_left + row_label_width + ((int(col_index) - 1) * numeric_col_width))
        width = float(row_label_width if int(col_index) == 0 else numeric_col_width)
        top = float(table_top + (int(row_index) * row_height))
        return (
            float(left),
            float(top),
            float(left + width),
            float(top + row_height),
        )

    def draw_cell(
        *,
        row_index: int,
        col_index: int,
        text: str,
        role: str,
        row_label: str | None,
        column_header: str | None,
        value: Any | None,
    ) -> List[float]:
        """Draw one table cell and record the exact annotation/projection bbox."""

        bbox = cell_bbox(row_index=int(row_index), col_index=int(col_index))
        fill_rgb = _style_fill_for_cell(
            scene_variant=selected_variant,
            row_index=int(row_index),
            col_index=int(col_index),
            params=render_params,
        )
        if selected_variant in {"spreadsheet", "zebra"}:
            draw.rectangle(
                bbox,
                fill=fill_rgb,
                outline=None,
            )
            _draw_inner_rule(
                draw,
                [(float(bbox[0]), float(bbox[1])), (float(bbox[2]), float(bbox[1]))],
                params=render_params,
            )
            _draw_inner_rule(
                draw,
                [(float(bbox[0]), float(bbox[3])), (float(bbox[2]), float(bbox[3]))],
                params=render_params,
            )
            _draw_inner_rule(
                draw,
                [(float(bbox[0]), float(bbox[1])), (float(bbox[0]), float(bbox[3]))],
                params=render_params,
            )
            _draw_inner_rule(
                draw,
                [(float(bbox[2]), float(bbox[1])), (float(bbox[2]), float(bbox[3]))],
                params=render_params,
            )
        elif selected_variant == "ledger":
            if int(row_index) == 0:
                draw.rectangle(bbox, fill=fill_rgb)
            line_y = float(bbox[3])
            _draw_inner_rule(
                draw,
                [(float(table_left), float(line_y)), (float(table_right), float(line_y))],
                params=render_params,
            )
            if int(col_index) == 0:
                _draw_inner_rule(
                    draw,
                    [(float(bbox[2]), float(table_top)), (float(bbox[2]), float(table_bottom))],
                    params=render_params,
                )
        else:
            if int(row_index) == 0:
                draw.rectangle(bbox, fill=fill_rgb)
            if int(col_index) > 0:
                _draw_inner_rule(
                    draw,
                    [(float(bbox[0]), float(table_top)), (float(bbox[0]), float(table_bottom))],
                    params=render_params,
                )
            if int(row_index) > 0:
                _draw_inner_rule(
                    draw,
                    [(float(table_left), float(bbox[1])), (float(table_right), float(bbox[1]))],
                    params=render_params,
                )

        if int(row_index) == 0 and str(render_params.header_style) == "accent":
            draw.line(
                [(float(bbox[0]), float(bbox[3])), (float(bbox[2]), float(bbox[3]))],
                fill=tuple(int(value) for value in render_params.border_color_rgb),
                width=max(1, int(render_params.border_width_px)),
            )

        text_fill_rgb = (
            tuple(int(value) for value in render_params.header_dark_text_rgb)
            if str(role) == "header" and str(render_params.header_style) == "dark"
            else tuple(int(value) for value in render_params.text_color_rgb)
        )
        text_align = (
            "left"
            if str(role) == "row_label"
            else str(render_params.numeric_alignment)
            if str(role) == "value" and isinstance(value, int) and not isinstance(value, bool)
            else "center"
        )
        text_bbox = _draw_text_in_cell(
            draw,
            text=str(text),
            bbox=bbox,
            font=label_font if str(role) in {"header", "row_label"} else value_font,
            bold=dense_fit_bold() if str(role) in {"header", "row_label"} else False,
            fill_rgb=text_fill_rgb,
            stroke_rgb=render_params.text_stroke_rgb,
            align=str(text_align),
            padding_px=int(render_params.cell_padding_px),
        )
        bbox_list = [round(float(value), 3) for value in bbox]
        cell_id = f"cell_r{int(row_index)}_c{int(col_index)}"
        trace = {
            "cell_id": str(cell_id),
            "row_index": int(row_index),
            "column_index": int(col_index),
            "cell_role": str(role),
            "row_label": None if row_label is None else str(row_label),
            "column_header": None if column_header is None else str(column_header),
            "text": str(text),
            "value": _trace_cell_value(value),
            "bbox_px": list(bbox_list),
            "text_bbox_px": list(text_bbox),
            "center_px": [
                round(float(0.5 * (bbox[0] + bbox[2])), 3),
                round(float(0.5 * (bbox[1] + bbox[3])), 3),
            ],
        }
        cell_traces.append(dict(trace))
        entities.append(
            {
                "entity_id": str(cell_id),
                "entity_type": "table_cell",
                "bbox_px": list(bbox_list),
                "attrs": {
                    "cell_role": str(role),
                    "row_index": int(row_index),
                    "column_index": int(col_index),
                    "row_label": None if row_label is None else str(row_label),
                    "column_header": None if column_header is None else str(column_header),
                    "text": str(text),
                    "value": _trace_cell_value(value),
                    "text_bbox_px": list(text_bbox),
                },
            }
        )
        return bbox_list

    draw_cell(
        row_index=0,
        col_index=0,
        text="Name",
        role="header",
        row_label=None,
        column_header="Name",
        value=None,
    )
    for column_index, header in enumerate(resolved_headers, start=1):
        header_bbox = draw_cell(
            row_index=0,
            col_index=int(column_index),
            text=str(header),
            role="header",
            row_label=None,
            column_header=str(header),
            value=None,
        )
        header_bboxes[str(header)] = list(header_bbox)

    for row_offset, row_label in enumerate(resolved_row_labels, start=1):
        label_bbox = draw_cell(
            row_index=int(row_offset),
            col_index=0,
            text=str(row_label),
            role="row_label",
            row_label=str(row_label),
            column_header="Name",
            value=None,
        )
        row_label_bboxes[str(row_label)] = list(label_bbox)
        values_for_row = values_by_row[str(row_label)]
        for column_offset, header in enumerate(resolved_headers, start=1):
            value = values_for_row[str(header)]
            value_bbox = draw_cell(
                row_index=int(row_offset),
                col_index=int(column_offset),
                text=str(value),
                role="value",
                row_label=str(row_label),
                column_header=str(header),
                value=value,
            )
            row_region_inputs[str(row_label)].append(list(value_bbox))
            column_region_inputs[str(header)].append(list(value_bbox))

    if selected_variant == "card_table":
        draw.rounded_rectangle(
            table_bbox,
            radius=int(render_params.rounded_corner_radius_px),
            fill=None,
            outline=tuple(int(value) for value in render_params.border_color_rgb),
            width=int(render_params.border_width_px),
        )
    else:
        draw.rectangle(
            table_bbox,
            fill=None,
            outline=tuple(int(value) for value in render_params.border_color_rgb),
            width=int(render_params.border_width_px),
        )

    row_region_bboxes = {
        str(row_label): _union_bboxes(boxes)
        for row_label, boxes in row_region_inputs.items()
    }
    column_region_bboxes = {
        str(header): _union_bboxes(boxes)
        for header, boxes in column_region_inputs.items()
    }
    numeric_table_region_bbox = _union_bboxes(list(column_region_bboxes.values()))
    return RenderedTableScene(
        image=image,
        entities=list(entities),
        cell_traces=list(cell_traces),
        table_bbox_px=[round(float(value), 3) for value in table_bbox],
        numeric_table_region_bbox=list(numeric_table_region_bbox),
        row_region_bboxes=dict(row_region_bboxes),
        column_region_bboxes=dict(column_region_bboxes),
        row_label_bboxes=dict(row_label_bboxes),
        header_bboxes=dict(header_bboxes),
    )


def resolve_table_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    defaults: TableDefaults,
    instance_seed: int | None = None,
) -> TableRenderParams:
    """Resolve one normalized table-render parameter block."""

    def _selection_index(key: str) -> int:
        seed = 0 if instance_seed is None else int(instance_seed)
        return abs(int(hash64(int(seed), f"table_render:{str(key)}", 45131)))

    def _int_value(key: str, fallback: int, *, minimum: int = 1) -> int:
        if params.get(str(key)) is not None:
            return max(int(minimum), int(params[str(key)]))
        low_raw = params.get(f"{str(key)}_min", group_default(render_defaults, f"{str(key)}_min", None))
        high_raw = params.get(f"{str(key)}_max", group_default(render_defaults, f"{str(key)}_max", None))
        if low_raw is not None or high_raw is not None:
            default_value = int(group_default(render_defaults, str(key), int(fallback)))
            low = int(default_value if low_raw is None else low_raw)
            high = int(default_value if high_raw is None else high_raw)
            if int(low) > int(high):
                raise ValueError(f"{str(key)}_min must be <= {str(key)}_max")
            return max(int(minimum), int(low) + (_selection_index(str(key)) % (int(high) - int(low) + 1)))
        return max(int(minimum), int(group_default(render_defaults, str(key), int(fallback))))

    def _rgb(key: str, fallback: Sequence[int]) -> Tuple[int, int, int]:
        if params.get(str(key)) is not None:
            raw = params[str(key)]
            return (int(raw[0]), int(raw[1]), int(raw[2]))
        options = params.get(f"{str(key)}_options", group_default(render_defaults, f"{str(key)}_options", None))
        if isinstance(options, Sequence) and options and not isinstance(options, (str, bytes)):
            raw = options[_selection_index(str(key)) % len(options)]
            return (int(raw[0]), int(raw[1]), int(raw[2]))
        raw = params.get(key, group_default(render_defaults, key, list(fallback)))
        return (int(raw[0]), int(raw[1]), int(raw[2]))

    def _choice_value(key: str, fallback: str, *, allowed: Sequence[str]) -> str:
        allowed_set = set(str(item) for item in allowed)
        if params.get(str(key)) is not None:
            value = str(params[str(key)])
        else:
            options = params.get(f"{str(key)}_options", group_default(render_defaults, f"{str(key)}_options", None))
            if isinstance(options, Sequence) and options and not isinstance(options, (str, bytes)):
                value = str(options[_selection_index(str(key)) % len(options)])
            else:
                value = str(group_default(render_defaults, str(key), str(fallback)))
        if value not in allowed_set:
            raise ValueError(f"unsupported {str(key)}: {value}")
        return str(value)

    return TableRenderParams(
        canvas_width=int(params.get("canvas_width", group_default(render_defaults, "canvas_width", defaults.canvas_width))),
        canvas_height=int(params.get("canvas_height", group_default(render_defaults, "canvas_height", defaults.canvas_height))),
        table_margin_left_px=_int_value("table_margin_left_px", defaults.table_margin_left_px, minimum=0),
        table_margin_right_px=_int_value("table_margin_right_px", defaults.table_margin_right_px, minimum=0),
        table_margin_top_px=_int_value("table_margin_top_px", defaults.table_margin_top_px, minimum=0),
        table_margin_bottom_px=_int_value("table_margin_bottom_px", defaults.table_margin_bottom_px, minimum=0),
        row_label_width_fraction=float(
            params.get(
                "row_label_width_fraction",
                group_default(render_defaults, "row_label_width_fraction", defaults.row_label_width_fraction),
            )
        ),
        row_label_min_width_px=int(
            params.get(
                "row_label_min_width_px",
                group_default(render_defaults, "row_label_min_width_px", defaults.row_label_min_width_px),
            )
        ),
        header_fill_rgb=_rgb("header_fill_rgb", (232, 236, 243)),
        zebra_row_fill_rgb=_rgb("zebra_row_fill_rgb", (247, 249, 252)),
        card_fill_rgb=_rgb("card_fill_rgb", (250, 250, 252)),
        border_color_rgb=_rgb("border_color_rgb", (92, 98, 109)),
        grid_color_rgb=_rgb("grid_color_rgb", (204, 209, 218)),
        text_color_rgb=_rgb("text_color_rgb", (36, 39, 45)),
        text_stroke_rgb=_rgb("text_stroke_rgb", (255, 255, 255)),
        header_style=_choice_value("header_style", defaults.header_style, allowed=("light", "accent", "dark")),
        header_dark_fill_rgb=_rgb("header_dark_fill_rgb", (63, 72, 86)),
        header_dark_text_rgb=_rgb("header_dark_text_rgb", (255, 255, 255)),
        frame_style=_choice_value("frame_style", defaults.frame_style, allowed=("flat", "shadow")),
        shadow_color_rgb=_rgb("shadow_color_rgb", (214, 218, 224)),
        shadow_offset_px=_int_value("shadow_offset_px", defaults.shadow_offset_px, minimum=1),
        inner_rule_style=_choice_value("inner_rule_style", defaults.inner_rule_style, allowed=("solid", "soft", "dashed")),
        numeric_alignment=_choice_value("numeric_alignment", defaults.numeric_alignment, allowed=("center", "right")),
        label_font_size_px=_int_value("label_font_size_px", defaults.label_font_size_px, minimum=10),
        value_font_size_px=_int_value("value_font_size_px", defaults.value_font_size_px, minimum=10),
        border_width_px=_int_value("border_width_px", defaults.border_width_px),
        grid_width_px=_int_value("grid_width_px", defaults.grid_width_px),
        rounded_corner_radius_px=_int_value("rounded_corner_radius_px", defaults.rounded_corner_radius_px),
        cell_padding_px=_int_value("cell_padding_px", defaults.cell_padding_px),
    )


def table_render_style_spec(render_params: TableRenderParams) -> Dict[str, Any]:
    """Serialize resolved non-semantic table style axes for trace metadata."""

    return {
        "header_style": str(render_params.header_style),
        "frame_style": str(render_params.frame_style),
        "inner_rule_style": str(render_params.inner_rule_style),
        "numeric_alignment": str(render_params.numeric_alignment),
        "header_fill_rgb": list(render_params.header_fill_rgb),
        "header_dark_fill_rgb": list(render_params.header_dark_fill_rgb),
        "header_dark_text_rgb": list(render_params.header_dark_text_rgb),
        "zebra_row_fill_rgb": list(render_params.zebra_row_fill_rgb),
        "card_fill_rgb": list(render_params.card_fill_rgb),
        "border_color_rgb": list(render_params.border_color_rgb),
        "grid_color_rgb": list(render_params.grid_color_rgb),
        "text_color_rgb": list(render_params.text_color_rgb),
        "text_stroke_rgb": list(render_params.text_stroke_rgb),
        "shadow_color_rgb": list(render_params.shadow_color_rgb),
        "shadow_offset_px": int(render_params.shadow_offset_px),
        "rounded_corner_radius_px": int(render_params.rounded_corner_radius_px),
        "cell_padding_px": int(render_params.cell_padding_px),
    }


__all__ = [
    "RenderedTableScene",
    "SUPPORTED_TABLE_SCENE_VARIANTS",
    "TableRenderParams",
    "resolve_table_render_params",
    "render_table_scene",
    "table_render_style_spec",
]
