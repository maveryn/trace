"""Rendering primitives for part-whole chart scenes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.information_style import (
    make_chart_information_background,
    resolve_chart_information_style,
)
from trace_tasks.tasks.charts.shared.dense_text import dense_fit_bold, dense_stroke_width
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.shared.visual_defaults import chart_font_asset_metadata, sample_chart_font_family
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.render_variation import apply_layout_jitter_to_margins, resolve_render_int, resolve_render_rgb
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font, temporary_default_font_family

from .defaults import (
    DEFAULTS,
    POST_IMAGE_NOISE_DEFAULTS,
    RENDER_DEFAULTS,
    SAMPLING_NAMESPACE,
    SCENE_ID,
)
from .state import PartWholeDataset, RenderedShareChart


@dataclass(frozen=True)
class PartWholeRenderResult:
    """Rendered chart plus non-semantic visual metadata."""

    image: Image.Image
    rendered_scene: RenderedShareChart
    background_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    font_assets: dict[str, str]
    canvas_width: int
    canvas_height: int


def _bbox_center(bbox: Sequence[float]) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return [(float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0]


def _text_bbox_at_origin(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: Any,
    *,
    stroke_width: int = 0,
) -> tuple[float, float, float, float]:
    try:
        bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width)))
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        pad = float(max(0, int(stroke_width)))
        return float(-pad), float(-pad), float(width + pad), float(height + pad)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: Any, *, stroke_width: int = 0) -> tuple[float, float]:
    bbox = _text_bbox_at_origin(draw, str(text), font, stroke_width=max(0, int(stroke_width)))
    return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: Any,
    fill: tuple[int, int, int],
    *,
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int] = (255, 255, 255),
) -> list[float]:
    draw_text_traced(
        draw,
        (float(xy[0]), float(xy[1])),
        str(text),
        font=font,
        fill=fill,
        stroke_width=max(0, int(stroke_width)),
        stroke_fill=stroke_fill,
        role="readout",
        required=False,
    )
    raw = _text_bbox_at_origin(draw, str(text), font, stroke_width=max(0, int(stroke_width)))
    return [
        float(xy[0]) + float(raw[0]),
        float(xy[1]) + float(raw[1]),
        float(xy[0]) + float(raw[2]),
        float(xy[1]) + float(raw[3]),
    ]


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: Any,
    fill: tuple[int, int, int],
    *,
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int] = (255, 255, 255),
) -> list[float]:
    raw = _text_bbox_at_origin(draw, str(text), font, stroke_width=max(0, int(stroke_width)))
    width = float(raw[2] - raw[0])
    height = float(raw[3] - raw[1])
    left = float(xy[0]) - (width / 2.0) - float(raw[0])
    top = float(xy[1]) - (height / 2.0) - float(raw[1])
    return _draw_text(
        draw,
        (float(left), float(top)),
        str(text),
        font,
        fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )


def _centered_text_bbox(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: Any,
    *,
    stroke_width: int = 0,
) -> list[float]:
    raw = _text_bbox_at_origin(draw, str(text), font, stroke_width=max(0, int(stroke_width)))
    width = float(raw[2] - raw[0])
    height = float(raw[3] - raw[1])
    left = float(xy[0]) - (width / 2.0) - float(raw[0])
    top = float(xy[1]) - (height / 2.0) - float(raw[1])
    return [
        float(left) + float(raw[0]),
        float(top) + float(raw[1]),
        float(left) + float(raw[2]),
        float(top) + float(raw[3]),
    ]


def _bboxes_overlap(left: Sequence[float], right: Sequence[float], *, padding: float = 0.0) -> bool:
    return not (
        float(left[2]) + float(padding) <= float(right[0])
        or float(right[2]) + float(padding) <= float(left[0])
        or float(left[3]) + float(padding) <= float(right[1])
        or float(right[3]) + float(padding) <= float(left[1])
    )


def _contrast_text(color: Sequence[int]) -> tuple[int, int, int]:
    red, green, blue = (int(color[0]), int(color[1]), int(color[2]))
    luminance = (0.299 * float(red)) + (0.587 * float(green)) + (0.114 * float(blue))
    return (22, 25, 32) if float(luminance) > 150.0 else (255, 255, 255)


def _render_pie_like(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: PartWholeDataset,
    scene_variant: str,
    chart_bbox: tuple[float, float, float, float],
    text_color: tuple[int, int, int],
    panel_fill: tuple[int, int, int],
    outline_width: int,
    label_font: Any,
) -> tuple[list[dict[str, Any]], tuple[int, int, int, int]]:
    """Draw circular share marks and record stable segment-center projections.

    This helper owns only pie/donut geometry.  It never chooses answer
    categories; every returned projection is keyed by visible category label so
    task-owned annotation labels can be projected later without reinterpreting
    pixels.
    """

    x0, y0, x1, y1 = [float(value) for value in chart_bbox]
    center = ((float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0 + 8.0)
    radius = min((float(x1) - float(x0)) * 0.42, (float(y1) - float(y0)) * 0.43)
    pie_box = (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)
    label_radius = float(radius) * (0.72 if str(scene_variant) == "pie" else 0.78)
    min_label_share = 5
    label_specs: list[dict[str, Any]] = []
    candidate_bboxes: list[list[float]] = []
    start_angle = -90.0
    for category in dataset.categories:
        share = int(category.value)
        end_angle = float(start_angle + (float(share) * 3.6))
        mid_angle = math.radians((float(start_angle) + float(end_angle)) / 2.0)
        label_x = float(center[0] + (math.cos(mid_angle) * label_radius))
        label_y = float(center[1] + (math.sin(mid_angle) * label_radius))
        candidate_bbox = _centered_text_bbox(
            draw,
            (label_x, label_y),
            str(category.label),
            label_font,
            stroke_width=dense_stroke_width(),
        )
        label_specs.append(
            {
                "category": category,
                "label_x": float(label_x),
                "label_y": float(label_y),
                "candidate_bbox": list(candidate_bbox),
            }
        )
        candidate_bboxes.append(list(candidate_bbox))
        start_angle = float(end_angle)

    labels_fit = all(int(category.value) >= int(min_label_share) for category in dataset.categories)
    if labels_fit:
        for left_index, left_bbox in enumerate(candidate_bboxes):
            for right_bbox in candidate_bboxes[int(left_index) + 1 :]:
                if _bboxes_overlap(left_bbox, right_bbox, padding=4.0):
                    labels_fit = False
                    break
            if not labels_fit:
                break

    traces: list[dict[str, Any]] = []
    start_angle = -90.0
    for spec in label_specs:
        category = spec["category"]
        share = int(category.value)
        end_angle = float(start_angle + (float(share) * 3.6))
        draw.pieslice(
            pie_box,
            start=float(start_angle),
            end=float(end_angle),
            fill=tuple(category.color_rgb),
            outline=(255, 255, 255),
            width=max(1, int(outline_width)),
        )
        if str(scene_variant) == "donut":
            inner = float(radius) * 0.52
            draw.ellipse(
                (center[0] - inner, center[1] - inner, center[0] + inner, center[1] + inner),
                fill=panel_fill,
                outline=panel_fill,
            )
        label_bbox: list[float] | None = None
        if labels_fit:
            fill = _contrast_text(category.color_rgb)
            label_bbox = _draw_centered(
                draw,
                (float(spec["label_x"]), float(spec["label_y"])),
                str(category.label),
                label_font,
                fill,
                stroke_width=dense_stroke_width(),
                stroke_fill=(32, 36, 44) if fill == (255, 255, 255) else (255, 255, 255),
            )
        traces.append(
            {
                "label": str(category.label),
                "value": int(category.value),
                "fill_rgb": [int(channel) for channel in category.color_rgb],
                "slice_angle_start": float(start_angle),
                "slice_angle_end": float(end_angle),
                "slice_center_px": [float(spec["label_x"]), float(spec["label_y"])],
                "label_bbox_px": list(label_bbox) if label_bbox is not None else None,
                "label_display_mode": "all" if labels_fit else "none",
            }
        )
        start_angle = float(end_angle)
    if str(scene_variant) == "donut":
        _draw_centered(draw, center, "100%", label_font, text_color)
    return traces, tuple(int(round(value)) for value in pie_box)


def _coerce_position_options(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ("right", "left", "bottom", "top")
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, Sequence):
        values = [str(value).strip() for value in raw]
    else:
        values = []
    allowed = {"right", "left", "bottom", "top"}
    options = tuple(value for value in values if value in allowed)
    return options or ("right", "left", "bottom", "top")


def _resolve_table_position(params: Mapping[str, Any], *, instance_seed: int) -> str:
    explicit = params.get("table_position", group_default(RENDER_DEFAULTS, "table_position", None))
    if explicit is not None:
        value = str(explicit).strip()
        if value not in {"right", "left", "bottom", "top"}:
            raise ValueError("table_position must be one of right, left, bottom, top")
        return value
    options = _coerce_position_options(
        params.get(
            "table_position_options",
            group_default(RENDER_DEFAULTS, "table_position_options", ("right", "left", "bottom", "top")),
        )
    )
    rng = spawn_rng(int(instance_seed), f"{SAMPLING_NAMESPACE}.table_position")
    return str(options[int(rng.randrange(0, len(options)))])


def _layout_share_chart_regions(
    *,
    width: int,
    height: int,
    margin_left: int,
    margin_right: int,
    margin_top: int,
    margin_bottom: int,
    table_width: int,
    table_height: int,
    chart_gap: int,
    table_position: str,
) -> tuple[tuple[int, int, int, int], tuple[float, float, float, float], tuple[int, int, int, int]]:
    content_x0 = int(margin_left)
    content_x1 = int(width - margin_right)
    content_y0 = int(margin_top + 72)
    content_y1 = int(height - margin_bottom)
    plot_bbox = (int(margin_left), int(margin_top), int(width - margin_right), int(height - margin_bottom))
    if str(table_position) == "left":
        table_bbox = (content_x0, content_y0, min(content_x1, content_x0 + int(table_width)), content_y1)
        chart_bbox = (float(table_bbox[2] + int(chart_gap)), float(content_y0), float(content_x1), float(content_y1))
    elif str(table_position) == "top":
        table_bottom = min(content_y1, content_y0 + int(table_height))
        table_bbox = (content_x0, content_y0, content_x1, table_bottom)
        chart_bbox = (float(content_x0), float(table_bbox[3] + int(chart_gap)), float(content_x1), float(content_y1))
    elif str(table_position) == "bottom":
        table_top = max(content_y0, content_y1 - int(table_height))
        table_bbox = (content_x0, table_top, content_x1, content_y1)
        chart_bbox = (float(content_x0), float(content_y0), float(content_x1), float(table_bbox[1] - int(chart_gap)))
    else:
        table_bbox = (max(content_x0, content_x1 - int(table_width)), content_y0, content_x1, content_y1)
        chart_bbox = (float(content_x0), float(content_y0), float(table_bbox[0] - int(chart_gap)), float(content_y1))
    if float(chart_bbox[2] - chart_bbox[0]) < 320.0 or float(chart_bbox[3] - chart_bbox[1]) < 320.0:
        table_bbox = (max(content_x0, content_x1 - int(table_width)), content_y0, content_x1, content_y1)
        chart_bbox = (float(content_x0), float(content_y0), float(table_bbox[0] - int(chart_gap)), float(content_y1))
    return tuple(int(value) for value in table_bbox), chart_bbox, plot_bbox


def _table_column_count(*, table_position: str, category_count: int) -> int:
    if str(table_position) in {"left", "right"}:
        return 1
    return 2 if int(category_count) <= 10 else 3


def _render_share_chart(
    *,
    base_image: Image.Image,
    dataset: PartWholeDataset,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> RenderedShareChart:
    """Render the circular chart and exact-value table for one dataset.

    The renderer keeps chart segment centers and table rows aligned to the same
    category records.  Objective code may select any subset of labels, but this
    function never branches on public task or query identity.
    """

    image = base_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    text_color = resolve_render_rgb(
        params,
        RENDER_DEFAULTS,
        "text_color_rgb",
        [36, 40, 48],
        instance_seed=int(instance_seed),
        namespace=SAMPLING_NAMESPACE,
    )
    grid_color = resolve_render_rgb(
        params,
        RENDER_DEFAULTS,
        "grid_color_rgb",
        [214, 219, 228],
        instance_seed=int(instance_seed),
        namespace=SAMPLING_NAMESPACE,
    )
    panel_fill = resolve_render_rgb(
        params,
        RENDER_DEFAULTS,
        "plot_fill_rgb",
        [255, 255, 255],
        instance_seed=int(instance_seed),
        namespace=SAMPLING_NAMESPACE,
    )
    zebra_row_fill = resolve_render_rgb(
        params,
        RENDER_DEFAULTS,
        "zebra_row_fill_rgb",
        [248, 249, 251],
        instance_seed=int(instance_seed),
        namespace=SAMPLING_NAMESPACE,
    )
    outline_width = resolve_render_int(
        params,
        RENDER_DEFAULTS,
        "mark_outline_width_px",
        2,
        instance_seed=int(instance_seed),
        namespace=SAMPLING_NAMESPACE,
    )
    label_font = load_font(
        int(params.get("label_font_size_px", group_default(RENDER_DEFAULTS, "label_font_size_px", 18))),
        bold=dense_fit_bold(),
    )
    table_font = load_font(
        int(params.get("table_font_size_px", group_default(RENDER_DEFAULTS, "table_font_size_px", 18))),
        bold=False,
    )
    table_header_font = load_font(
        int(params.get("table_header_font_size_px", group_default(RENDER_DEFAULTS, "table_header_font_size_px", 19))),
        bold=False,
    )

    margin_left = int(params.get("plot_margin_left_px", group_default(RENDER_DEFAULTS, "plot_margin_left_px", 42)))
    margin_right = int(params.get("plot_margin_right_px", group_default(RENDER_DEFAULTS, "plot_margin_right_px", 42)))
    margin_top = int(params.get("plot_margin_top_px", group_default(RENDER_DEFAULTS, "plot_margin_top_px", 46)))
    margin_bottom = int(params.get("plot_margin_bottom_px", group_default(RENDER_DEFAULTS, "plot_margin_bottom_px", 42)))
    margin_left, margin_right, margin_top, margin_bottom, layout_jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(margin_left),
        right_px=int(margin_right),
        top_px=int(margin_top),
        bottom_px=int(margin_bottom),
        params=params,
        defaults=RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.layout",
    )
    table_width = int(params.get("table_width_px", group_default(RENDER_DEFAULTS, "table_width_px", 450)))
    table_height = int(params.get("table_height_px", group_default(RENDER_DEFAULTS, "table_height_px", 280)))
    chart_gap = int(params.get("chart_table_gap_px", group_default(RENDER_DEFAULTS, "chart_table_gap_px", 34)))
    table_position = _resolve_table_position(params, instance_seed=int(instance_seed))
    table_bbox, chart_bbox, plot_bbox = _layout_share_chart_regions(
        width=int(width),
        height=int(height),
        margin_left=int(margin_left),
        margin_right=int(margin_right),
        margin_top=int(margin_top),
        margin_bottom=int(margin_bottom),
        table_width=int(table_width),
        table_height=int(table_height),
        chart_gap=int(chart_gap),
        table_position=str(table_position),
    )
    table_columns = _table_column_count(table_position=str(table_position), category_count=len(dataset.categories))
    layout_jitter_meta = {
        **dict(layout_jitter_meta),
        "table_position": str(table_position),
        "table_columns": int(table_columns),
    }
    draw.rounded_rectangle(chart_bbox, radius=8, fill=panel_fill, outline=grid_color, width=max(1, int(outline_width)))
    draw.rounded_rectangle(table_bbox, radius=8, fill=panel_fill, outline=grid_color, width=max(1, int(outline_width)))

    chart_traces, _chart_mark_bbox = _render_pie_like(
        draw,
        dataset=dataset,
        scene_variant=str(scene_variant),
        chart_bbox=chart_bbox,
        text_color=text_color,
        panel_fill=panel_fill,
        outline_width=int(outline_width),
        label_font=label_font,
    )

    table_inner_left = float(table_bbox[0]) + 14.0
    table_inner_right = float(table_bbox[2]) - 14.0
    table_top = float(table_bbox[1]) + 12.0
    column_gap = 14.0 if int(table_columns) > 1 else 0.0
    column_width = (
        float(table_inner_right - table_inner_left) - (float(table_columns - 1) * float(column_gap))
    ) / float(table_columns)
    rows_per_column = int(math.ceil(len(dataset.categories) / float(table_columns)))
    row_area_top = float(table_bbox[1]) + 54.0
    row_area_bottom = float(table_bbox[3]) - 14.0
    row_height = float(row_area_bottom - row_area_top) / float(max(1, rows_per_column))
    swatch_size = min(22.0, max(14.0, float(row_height) * 0.58))
    annotation_bbox_by_label: dict[str, list[float]] = {}
    annotation_point_by_label: dict[str, list[float]] = {}
    for trace in chart_traces:
        label = str(trace["label"])
        annotation_point_by_label[label] = [float(value) for value in trace["slice_center_px"]]
    category_traces: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    table_categories = tuple(sorted(dataset.categories, key=lambda item: str(item.label)))
    for column_index in range(int(table_columns)):
        col_x0 = float(table_inner_left + (float(column_index) * (float(column_width) + float(column_gap))))
        col_x1 = float(col_x0 + float(column_width))
        _draw_text(draw, (float(col_x0) + 6.0, float(table_top)), "Category", table_header_font, text_color)
        _draw_text(draw, (float(col_x1) - 88.0, float(table_top)), "Share", table_header_font, text_color)
        header_y = float(table_bbox[1]) + 48.0
        draw.line((float(col_x0) + 4.0, header_y, float(col_x1) - 4.0, header_y), fill=grid_color, width=2)
        if int(column_index) > 0:
            separator_x = float(col_x0) - (float(column_gap) / 2.0)
            draw.line(
                (separator_x, float(table_bbox[1]) + 16.0, separator_x, float(table_bbox[3]) - 16.0),
                fill=grid_color,
                width=1,
            )
    for index, category in enumerate(table_categories):
        column_index = int(index) // int(rows_per_column)
        row_index = int(index) % int(rows_per_column)
        col_x0 = float(table_inner_left + (float(column_index) * (float(column_width) + float(column_gap))))
        col_x1 = float(col_x0 + float(column_width))
        row_y0 = float(row_area_top + (float(row_index) * float(row_height)))
        row_y1 = float(row_area_top + (float(row_index + 1) * float(row_height)))
        if int(index) % 2 == 1:
            draw.rectangle(
                (float(col_x0) + 4.0, row_y0, float(col_x1) - 4.0, row_y1),
                fill=zebra_row_fill,
            )
        if int(row_index) > 0:
            draw.line((float(col_x0) + 4.0, row_y0, float(col_x1) - 4.0, row_y0), fill=grid_color, width=1)
        swatch_y0 = row_y0 + ((float(row_height) - float(swatch_size)) / 2.0)
        swatch_x0 = float(col_x0) + 8.0
        draw.rounded_rectangle(
            (swatch_x0, swatch_y0, swatch_x0 + swatch_size, swatch_y0 + swatch_size),
            radius=3,
            fill=tuple(category.color_rgb),
            outline=(80, 84, 92),
            width=1,
        )
        text_y = row_y0 + ((float(row_height) - _text_size(draw, str(category.label), table_font)[1]) / 2.0) - 1.0
        label_bbox = _draw_text(draw, (swatch_x0 + swatch_size + 12.0, text_y), str(category.label), table_font, text_color)
        value_text = f"{int(category.value)}%"
        value_width = _text_size(draw, value_text, table_font)[0]
        value_bbox = _draw_text(draw, (float(col_x1) - 12.0 - value_width, text_y), value_text, table_font, text_color)
        full_row_bbox = [float(col_x0) + 5.0, float(row_y0), float(col_x1) - 5.0, float(row_y1)]
        annotation_bbox = [
            max(float(col_x0) + 5.0, float(swatch_x0) - 5.0),
            max(float(row_y0) + 3.0, min(float(swatch_y0), float(label_bbox[1]), float(value_bbox[1])) - 4.0),
            min(float(col_x1) - 5.0, max(float(swatch_x0 + swatch_size), float(label_bbox[2]), float(value_bbox[2])) + 5.0),
            min(float(row_y1) - 3.0, max(float(swatch_y0 + swatch_size), float(label_bbox[3]), float(value_bbox[3])) + 4.0),
        ]
        annotation_bbox_by_label[str(category.label)] = list(annotation_bbox)
        category_trace = {
            "label": str(category.label),
            "value": int(category.value),
            "fill_rgb": [int(channel) for channel in category.color_rgb],
            "table_order_index": int(index),
            "table_column_index": int(column_index),
            "table_row_index": int(row_index),
            "row_bbox_px": list(full_row_bbox),
            "annotation_bbox_px": list(annotation_bbox),
            "chart_annotation_point_px": list(annotation_point_by_label.get(str(category.label), [])),
            "label_bbox_px": list(label_bbox),
            "value_bbox_px": list(value_bbox),
        }
        category_traces.append(dict(category_trace))
        entities.append(
            {
                "entity_id": str(category.label),
                "kind": "composition_category",
                "attrs": dict(category_trace),
            }
        )

    for trace in chart_traces:
        entities.append(
            {
                "entity_id": f"chart:{trace['label']}",
                "kind": "composition_mark",
                "attrs": dict(trace),
            }
        )

    return RenderedShareChart(
        image=image,
        entities=tuple(entities),
        plot_bbox_px=tuple(int(value) for value in plot_bbox),
        table_bbox_px=tuple(int(value) for value in table_bbox),
        chart_traces=tuple(dict(trace) for trace in chart_traces),
        category_traces=tuple(dict(trace) for trace in category_traces),
        annotation_bbox_by_label=dict(annotation_bbox_by_label),
        annotation_point_by_label=dict(annotation_point_by_label),
        layout_jitter_meta=dict(layout_jitter_meta),
    )


def render_part_whole_dataset(
    *,
    dataset: PartWholeDataset,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> PartWholeRenderResult:
    """Render a task-bound part-whole dataset into chart geometry."""

    canvas_width = int(params.get("canvas_width", group_default(RENDER_DEFAULTS, "canvas_width", DEFAULTS.canvas_width)))
    canvas_height = int(params.get("canvas_height", group_default(RENDER_DEFAULTS, "canvas_height", DEFAULTS.canvas_height)))
    information_style, information_style_meta = resolve_chart_information_style(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="part_whole",
        protected_colors=[tuple(int(channel) for channel in category.color_rgb) for category in dataset.categories],
    )
    styled_params = {
        **dict(params),
        "text_color_rgb": tuple(int(value) for value in information_style.text_rgb),
        "grid_color_rgb": tuple(int(value) for value in information_style.guide_rgb),
        "plot_fill_rgb": tuple(int(value) for value in information_style.panel_fill_rgb),
        "zebra_row_fill_rgb": tuple(int(value) for value in information_style.surface_alt_rgb),
    }
    background, background_meta = make_chart_information_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=information_style,
        instance_seed=int(instance_seed),
        namespace="charts.part_whole.information_scene_background",
    )
    chart_font_family = sample_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.chart_font",
        params=params,
    )
    with temporary_default_font_family(str(chart_font_family)):
        rendered_scene = _render_share_chart(
            base_image=background,
            dataset=dataset,
            scene_variant=str(scene_variant),
            params=styled_params,
            instance_seed=int(instance_seed),
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    rendered_scene = RenderedShareChart(
        image=image,
        entities=tuple(dict(entity) for entity in rendered_scene.entities),
        plot_bbox_px=tuple(int(value) for value in rendered_scene.plot_bbox_px),
        table_bbox_px=tuple(int(value) for value in rendered_scene.table_bbox_px),
        chart_traces=tuple(dict(trace) for trace in rendered_scene.chart_traces),
        category_traces=tuple(dict(trace) for trace in rendered_scene.category_traces),
        annotation_bbox_by_label=dict(rendered_scene.annotation_bbox_by_label),
        annotation_point_by_label=dict(rendered_scene.annotation_point_by_label),
        layout_jitter_meta=dict(rendered_scene.layout_jitter_meta),
    )
    return PartWholeRenderResult(
        image=image,
        rendered_scene=rendered_scene,
        background_meta={**dict(background_meta), "information_scene_style": dict(information_style_meta)},
        post_noise_meta=dict(post_noise_meta),
        font_assets=chart_font_asset_metadata(str(chart_font_family)),
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
    )


__all__ = ["PartWholeRenderResult", "render_part_whole_dataset"]
