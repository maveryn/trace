"""Rendering helpers for size-encoded chart comparison tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from .....core.seed import spawn_rng
from .....core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.shared.information_style import make_chart_information_background, resolve_chart_information_style
from trace_tasks.tasks.charts.shared.dense_text import (
    DENSE_TEXT_DARK_RGB,
    dense_fit_bold,
    dense_stroke_width,
    dense_text_params,
    dense_text_style_meta,
)
from ....shared.render_variation import apply_layout_jitter_to_margins
from ....shared.text_legibility import (
    LARGE_TEXT_MIN_CONTRAST_RATIO,
    READ_REQUIRED_TEXT_MIN_LAB_DISTANCE,
    READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
    draw_centered_readable_text,
    draw_readable_text,
    draw_text_traced,
    resolve_readable_text_style,
    text_legibility_summary,
)
from ....shared.text_rendering import fit_font_to_box, load_font, resolve_text_stroke_fill
from .defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    RENDER_DEFAULTS,
    category_palette,
    resolve_int,
)
from .state import (
    BBox,
    RGB,
    SCENE_NAMESPACE,
    SizeEncodingDataset,
    SizeEncodingItem,
    RenderedSizeEncodingScene,
)

def _lighten(color: RGB, factor: float) -> RGB:
    factor = max(0.0, min(1.0, float(factor)))
    return tuple(int(round(int(channel) + ((255 - int(channel)) * factor))) for channel in color)  # type: ignore[return-value]

def _darken(color: RGB, factor: float) -> RGB:
    factor = max(0.0, min(1.0, float(factor)))
    return tuple(int(round(int(channel) * factor)) for channel in color)  # type: ignore[return-value]

def _category_text_candidates(color: RGB) -> Tuple[RGB, ...]:
    """Return category-hued text candidates from raw swatch to dark readable tints."""

    return (
        tuple(int(channel) for channel in color),
        _darken(color, 0.82),
        _darken(color, 0.68),
        _darken(color, 0.54),
        _darken(color, 0.42),
        _lighten(color, 0.14),
        _lighten(color, 0.34),
        _lighten(color, 0.54),
        _lighten(color, 0.70),
    )

def _bubble_fill(color: RGB) -> RGB:
    return _lighten(color, 0.55)

def _text_bbox(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, font: ImageFont.ImageFont, *, stroke_width: int = 0) -> BBox:
    try:
        bbox = draw.textbbox(xy, str(text), font=font, stroke_width=max(0, int(stroke_width)))
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        x, y = float(xy[0]), float(xy[1])
        return (x, y, x + float(width), y + float(height))

def _center_text(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: RGB,
    stroke_width: int = 0,
    stroke_fill: RGB | None = None,
) -> List[float]:
    bbox0 = _text_bbox(draw, (0.0, 0.0), str(text), font, stroke_width=stroke_width)
    width = float(bbox0[2] - bbox0[0])
    height = float(bbox0[3] - bbox0[1])
    x = float(center[0]) - (width / 2.0) - float(bbox0[0])
    y = float(center[1]) - (height / 2.0) - float(bbox0[1])
    draw_text_traced(draw,
        (x, y),
        str(text),
        font=font,
        fill=tuple(int(channel) for channel in fill),
        stroke_width=max(0, int(stroke_width)),
        stroke_fill=tuple(int(channel) for channel in (stroke_fill or resolve_text_stroke_fill(fill))),
     role="readout", required=False,)
    return [float(value) for value in _text_bbox(draw, (x, y), str(text), font, stroke_width=stroke_width)]

def _union_bboxes(boxes: Sequence[Sequence[float]]) -> List[float]:
    valid = [tuple(float(value) for value in box[:4]) for box in boxes if len(box) >= 4]
    if not valid:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    ]

def _format_bbox(bbox: Sequence[float]) -> List[float]:
    return [round(float(value), 3) for value in bbox[:4]]

def _expand_bbox_to_min_side(
    bbox: Sequence[float],
    *,
    min_side_px: float,
    clip_bbox: Sequence[float] | None = None,
) -> List[float]:
    """Return a centered bbox expanded to the annotation minimum side."""

    x0, y0, x1, y1 = (float(value) for value in bbox[:4])
    width = max(0.0, float(x1 - x0))
    height = max(0.0, float(y1 - y0))
    target_w = max(width, float(min_side_px))
    target_h = max(height, float(min_side_px))
    cx = float((x0 + x1) / 2.0)
    cy = float((y0 + y1) / 2.0)

    if clip_bbox is not None:
        clip_x0, clip_y0, clip_x1, clip_y1 = (float(value) for value in clip_bbox[:4])
        target_w = min(target_w, max(0.0, float(clip_x1 - clip_x0)))
        target_h = min(target_h, max(0.0, float(clip_y1 - clip_y0)))
    else:
        clip_x0 = clip_y0 = float("-inf")
        clip_x1 = clip_y1 = float("inf")

    expanded = [
        cx - (target_w / 2.0),
        cy - (target_h / 2.0),
        cx + (target_w / 2.0),
        cy + (target_h / 2.0),
    ]

    if expanded[0] < clip_x0:
        shift = clip_x0 - expanded[0]
        expanded[0] += shift
        expanded[2] += shift
    if expanded[2] > clip_x1:
        shift = expanded[2] - clip_x1
        expanded[0] -= shift
        expanded[2] -= shift
    if expanded[1] < clip_y0:
        shift = clip_y0 - expanded[1]
        expanded[1] += shift
        expanded[3] += shift
    if expanded[3] > clip_y1:
        shift = expanded[3] - clip_y1
        expanded[1] -= shift
        expanded[3] -= shift

    return [
        max(clip_x0, min(clip_x1, expanded[0])),
        max(clip_y0, min(clip_y1, expanded[1])),
        max(clip_x0, min(clip_x1, expanded[2])),
        max(clip_y0, min(clip_y1, expanded[3])),
    ]

def _explicit_rgb(params: Mapping[str, Any], key: str, fallback: RGB) -> RGB:
    raw = params.get(str(key))
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 3:
        return (
            max(0, min(255, int(raw[0]))),
            max(0, min(255, int(raw[1]))),
            max(0, min(255, int(raw[2]))),
        )
    return tuple(int(channel) for channel in fallback)

def _cell_centers(
    bbox: BBox,
    *,
    count: int,
    instance_seed: int,
    namespace: str,
    circular: bool = False,
) -> List[Tuple[float, float, float, float]]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    cols = max(1, int(math.ceil(math.sqrt(float(count) * 1.35))))
    rows = max(1, int(math.ceil(float(count) / float(cols))))
    cells: List[Tuple[float, float, float, float]] = []
    while True:
        cells.clear()
        cell_w = width / float(cols)
        cell_h = height / float(rows)
        cx0 = (x1 + x2) / 2.0
        cy0 = (y1 + y2) / 2.0
        rx = width * 0.48
        ry = height * 0.48
        for row in range(rows):
            for col in range(cols):
                cx = x1 + (float(col) + 0.5) * cell_w
                cy = y1 + (float(row) + 0.5) * cell_h
                if circular:
                    norm = ((cx - cx0) / max(1.0, rx)) ** 2 + ((cy - cy0) / max(1.0, ry)) ** 2
                    if norm > 0.90:
                        continue
                cells.append((cx, cy, cell_w, cell_h))
        if len(cells) >= int(count) or not circular:
            break
        cols += 1
        rows += 1
    rng = spawn_rng(int(instance_seed), str(namespace))
    rng.shuffle(cells)
    return cells[: int(count)]

def _value_scale(items: Sequence[SizeEncodingItem]) -> Tuple[int, int]:
    values = [int(item.value) for item in items]
    return min(values), max(values)

def _value_fraction(value: int, *, value_min: int, value_max: int) -> float:
    if int(value_max) <= int(value_min):
        return 0.5
    return (float(value) - float(value_min)) / (float(value_max) - float(value_min))

def _draw_legend(
    draw: ImageDraw.ImageDraw,
    *,
    categories: Sequence[str],
    colors: Mapping[str, RGB],
    bbox: BBox,
    params: Mapping[str, Any],
    text_style,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], List[Dict[str, Any]]]:
    """Draw category swatches and labels without making them task witnesses."""

    x1, y1, x2, y2 = (float(value) for value in bbox)
    font = load_font(resolve_int(params, "legend_font_size_px", 18), bold=False)
    entities: List[Dict[str, Any]] = []
    text_records: List[Dict[str, Any]] = []
    bboxes: Dict[str, List[float]] = {}
    column_w = max(80.0, (x2 - x1) / max(1, len(categories)))
    for index, category in enumerate(categories):
        left = x1 + (index * column_w) + 4
        cy = (y1 + y2) / 2.0
        color = colors[str(category)]
        swatch = [left, cy - 8, left + 18, cy + 10]
        draw.rounded_rectangle(swatch, radius=4, fill=color, outline=_darken(color, 0.62), width=1)
        text_xy = (left + 26, cy - 12)
        text_records.append(
            draw_readable_text(
                draw,
                xy=text_xy,
                text=str(category),
                font=font,
                style=text_style,
                stroke_width=dense_stroke_width(),
                extra_metadata={"category": str(category), "source": "size_encoding_legend"},
            )
        )
        text_box = _text_bbox(draw, text_xy, str(category), font)
        full_box = [
            float(min(swatch[0], text_box[0])),
            float(min(swatch[1], text_box[1])),
            float(max(swatch[2], text_box[2])),
            float(max(swatch[3], text_box[3])),
        ]
        bboxes[str(category)] = _format_bbox(full_box)
        entities.append(
            {
                "entity_id": f"legend_{str(category).lower()}",
                "entity_type": "chart_size_category_legend",
                "bbox_px": _format_bbox(full_box),
                "attrs": {"category": str(category), "fill_rgb": list(color)},
            }
        )
    return entities, bboxes, text_records

def _draw_word_cloud(
    draw: ImageDraw.ImageDraw,
    *,
    items: Sequence[SizeEncodingItem],
    bbox: BBox,
    category_colors: Mapping[str, RGB],
    params: Mapping[str, Any],
    instance_seed: int,
    circular: bool,
    text_styles: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], List[Dict[str, Any]]]:
    """Render text-size encoded items and preserve each item bbox."""

    item_bboxes: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = []
    text_records: List[Dict[str, Any]] = []
    value_min, value_max = _value_scale(items)
    min_font = resolve_int(params, "min_word_font_size_px", 18)
    max_font = resolve_int(params, "max_word_font_size_px", 52)
    stroke_width = resolve_int(params, "word_text_stroke_width_px", 1)
    if circular:
        draw.ellipse(bbox, outline=(210, 216, 226), width=2)
    cells = _cell_centers(
        bbox,
        count=len(items),
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.word_cells.{int(circular)}",
        circular=bool(circular),
    )
    for item, (cx, cy, cell_w, cell_h) in zip(items, cells):
        frac = _value_fraction(int(item.value), value_min=int(value_min), value_max=int(value_max))
        target_font = int(round(float(min_font) + (float(max_font - min_font) * math.sqrt(float(frac)))))
        font = fit_font_to_box(
            draw,
            text=str(item.label),
            max_width=float(cell_w) * 0.78,
            max_height=float(cell_h) * 0.90,
                bold=dense_fit_bold(),
            min_size_px=max(8, int(min_font) - 6),
            max_size_px=max(int(min_font), int(target_font)),
            fill_ratio=0.95,
        )
        text_box0 = _text_bbox(draw, (0.0, 0.0), str(item.label), font, stroke_width=stroke_width)
        text_w = float(text_box0[2] - text_box0[0])
        marker_d = max(8.0, min(18.0, float(getattr(font, "size", target_font)) * 0.42))
        marker_gap = max(5.0, float(getattr(font, "size", target_font)) * 0.18)
        group_w = float(marker_d + marker_gap + text_w)
        marker_left = float(cx - (group_w / 2.0))
        marker_bbox = [
            marker_left,
            float(cy - (marker_d / 2.0)),
            float(marker_left + marker_d),
            float(cy + (marker_d / 2.0)),
        ]
        marker_color = category_colors[str(item.category)]
        draw.rounded_rectangle(
            marker_bbox,
            radius=max(2, int(round(marker_d / 2.0))),
            fill=marker_color,
            outline=_darken(marker_color, 0.55),
            width=1,
        )
        text_center_x = float(marker_bbox[2] + marker_gap + (text_w / 2.0))
        text_record = draw_centered_readable_text(
            draw,
            center=(float(text_center_x), float(cy)),
            text=str(item.label),
            font=font,
            style=text_styles[str(item.category)],
            stroke_width=dense_stroke_width(),
            extra_metadata={"item_id": str(item.item_id), "source": "size_encoding_word_item"},
        )
        text_records.append(text_record)
        bbox_px = _union_bboxes((marker_bbox, text_record["bbox_px"]))
        visual_bbox = [bbox_px[0] - 3, bbox_px[1] - 3, bbox_px[2] + 3, bbox_px[3] + 3]
        annotation_min_side_px = 24.5
        annotation_bbox = _expand_bbox_to_min_side(
            visual_bbox,
            min_side_px=annotation_min_side_px,
            clip_bbox=bbox,
        )
        item_bboxes[str(item.item_id)] = _format_bbox(annotation_bbox)
        entities.append(
            {
                "entity_id": str(item.item_id),
                "entity_type": "chart_size_word_item",
                "bbox_px": _format_bbox(annotation_bbox),
                "attrs": {
                    "label": str(item.label),
                    "category": str(item.category),
                    "panel": str(item.panel),
                    "value": int(item.value),
                    "font_size_px": int(getattr(font, "size", target_font)),
                    "visual_bbox_px": _format_bbox(visual_bbox),
                    "annotation_bbox_min_side_px": annotation_min_side_px,
                    "category_marker_rgb": list(marker_color),
                    "text_color_policy": "category_tinted_readable_ink",
                },
            }
        )
    return entities, item_bboxes, text_records

def _draw_bubbles(
    draw: ImageDraw.ImageDraw,
    *,
    items: Sequence[SizeEncodingItem],
    bbox: BBox,
    category_colors: Mapping[str, RGB],
    params: Mapping[str, Any],
    instance_seed: int,
    text_styles: Mapping[str, Any],
    value_scale: Tuple[int, int] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], List[Dict[str, Any]]]:
    """Render bubble-size encoded items and preserve each circle bbox."""

    item_bboxes: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = []
    text_records: List[Dict[str, Any]] = []
    value_min, value_max = value_scale if value_scale is not None else _value_scale(items)
    stroke_width = dense_stroke_width()
    cells = _cell_centers(
        bbox,
        count=len(items),
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.bubble_cells",
        circular=False,
    )
    for item, (cx, cy, cell_w, cell_h) in zip(items, cells):
        frac = _value_fraction(int(item.value), value_min=int(value_min), value_max=int(value_max))
        max_r = max(10.0, min(float(cell_w), float(cell_h)) * 0.43)
        min_r = max(10.0, max_r * 0.42)
        radius = min_r + ((max_r - min_r) * math.sqrt(float(frac)))
        color = category_colors[str(item.category)]
        fill = _bubble_fill(color)
        outline = _darken(color, 0.55)
        circle = [float(cx - radius), float(cy - radius), float(cx + radius), float(cy + radius)]
        draw.ellipse(circle, fill=fill, outline=outline, width=2)
        font = fit_font_to_box(
            draw,
            text=str(item.label),
            max_width=float(radius) * 1.78,
            max_height=float(radius) * 0.90,
            bold=dense_fit_bold(),
            min_size_px=resolve_int(params, "bubble_label_min_font_size_px", 11),
            max_size_px=resolve_int(params, "bubble_label_font_size_px", 19),
            fill_ratio=0.98,
        )
        text_records.append(
            draw_centered_readable_text(
                draw,
                center=(float(cx), float(cy)),
                text=str(item.label),
                font=font,
                style=text_styles[str(item.category)],
                stroke_width=dense_stroke_width(),
                extra_metadata={"item_id": str(item.item_id), "source": "size_encoding_bubble_item"},
            )
        )
        item_bboxes[str(item.item_id)] = _format_bbox(circle)
        entities.append(
            {
                "entity_id": str(item.item_id),
                "entity_type": "chart_size_bubble_item",
                "bbox_px": _format_bbox(circle),
                "attrs": {
                    "label": str(item.label),
                    "category": str(item.category),
                    "panel": str(item.panel),
                    "value": int(item.value),
                    "radius_px": round(float(radius), 3),
                },
            }
        )
    return entities, item_bboxes, text_records

def _panel_layout(plot_bbox: BBox, panel_count: int, gap: float) -> List[BBox]:
    x1, y1, x2, y2 = (float(value) for value in plot_bbox)
    if int(panel_count) <= 1:
        return [(x1, y1, x2, y2)]
    cols = 2 if int(panel_count) <= 4 else 3
    rows = int(math.ceil(float(panel_count) / float(cols)))
    cell_w = (x2 - x1 - (float(cols - 1) * gap)) / float(cols)
    cell_h = (y2 - y1 - (float(rows - 1) * gap)) / float(rows)
    bboxes: List[BBox] = []
    for index in range(int(panel_count)):
        row = index // cols
        col = index % cols
        px1 = x1 + (col * (cell_w + gap))
        py1 = y1 + (row * (cell_h + gap))
        bboxes.append((px1, py1, px1 + cell_w, py1 + cell_h))
    return bboxes

def render_size_encoding_scene(
    dataset: SizeEncodingDataset,
    *,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> RenderedSizeEncodingScene:
    """Render one size-encoding dataset without reading public task identity."""

    params = dense_text_params({**dict(params), "_render_style_seed": int(instance_seed)})
    canvas_width = resolve_int(params, "canvas_width", 1320)
    canvas_height = resolve_int(params, "canvas_height", 900)
    palette = category_palette(params, len(dataset.categories))
    category_colors = {str(category): palette[index] for index, category in enumerate(dataset.categories)}
    information_style, information_style_meta = resolve_chart_information_style(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="size_encoding",
        protected_colors=tuple(palette),
    )
    background, background_meta = make_chart_information_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=information_style,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.information_scene_background",
    )
    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    outer = resolve_int(params, "outer_margin_px", 44)
    margin_left, margin_right, margin_top, margin_bottom, layout_jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(outer),
        right_px=int(outer),
        top_px=int(outer),
        bottom_px=int(outer),
        params=params,
        defaults=RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.layout",
    )
    title_band = resolve_int(params, "title_band_height_px", 74)
    legend_height = resolve_int(params, "legend_height_px", 58)
    panel_gap = resolve_int(params, "panel_gap_px", 28)
    panel_padding = resolve_int(params, "panel_padding_px", 18)
    title_rgb = _explicit_rgb(params, "title_rgb", information_style.text_rgb)
    subtitle_rgb = _explicit_rgb(params, "subtitle_rgb", information_style.muted_text_rgb)
    panel_fill = _explicit_rgb(params, "panel_fill_rgb", information_style.surface_rgb)
    panel_border = _explicit_rgb(params, "panel_border_rgb", information_style.panel_border_rgb)
    background_rgb = tuple(int(channel) for channel in image.getpixel((0, 0))[:3])
    word_text_surfaces: Tuple[RGB, ...] = (tuple(panel_fill),)
    title_text_style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.title_text",
        role="read_required_chart_title",
        surface_rgbs=(background_rgb, tuple(panel_fill)),
        preferred_rgbs=(title_rgb,),
        min_contrast_ratio=LARGE_TEXT_MIN_CONTRAST_RATIO,
        required=True,
    )
    subtitle_text_style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.subtitle_text",
        role="read_required_chart_subtitle",
        surface_rgbs=(background_rgb, tuple(panel_fill)),
        preferred_rgbs=(subtitle_rgb, title_rgb),
        min_contrast_ratio=LARGE_TEXT_MIN_CONTRAST_RATIO,
        min_lab_distance=28.0,
        required=True,
    )
    word_text_style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.word_text.uniform",
        role="read_required_size_encoding_word_label",
        surface_rgbs=word_text_surfaces,
        preferred_rgbs=(DENSE_TEXT_DARK_RGB, title_rgb),
        candidate_rgbs=(DENSE_TEXT_DARK_RGB, title_rgb, (246, 250, 255)),
        min_contrast_ratio=READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
        min_lab_distance=READ_REQUIRED_TEXT_MIN_LAB_DISTANCE,
        required=True,
    )
    word_text_styles = {
        str(category): resolve_readable_text_style(
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.word_text.{category}",
            role="read_required_size_encoding_word_label",
            surface_rgbs=word_text_surfaces,
            preferred_rgbs=(word_text_style.fill_rgb,),
            candidate_rgbs=(word_text_style.fill_rgb,),
            min_contrast_ratio=READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
            min_lab_distance=READ_REQUIRED_TEXT_MIN_LAB_DISTANCE,
            required=True,
        )
        for category, color in category_colors.items()
    }
    bubble_text_styles = {
        str(category): resolve_readable_text_style(
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.bubble_text.{category}",
            role="read_required_size_encoding_bubble_label",
            surface_rgbs=(_bubble_fill(color),),
            preferred_rgbs=((38, 44, 58), title_rgb),
            min_contrast_ratio=READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
            min_lab_distance=READ_REQUIRED_TEXT_MIN_LAB_DISTANCE,
            required=True,
        )
        for category, color in category_colors.items()
    }
    secondary_text_style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.secondary_text",
        role="read_required_size_encoding_secondary_label",
        surface_rgbs=(background_rgb, tuple(panel_fill)),
        preferred_rgbs=((50, 58, 74), subtitle_rgb),
        min_contrast_ratio=LARGE_TEXT_MIN_CONTRAST_RATIO,
        min_lab_distance=28.0,
        required=True,
    )
    text_records: List[Dict[str, Any]] = []

    title_font = load_font(resolve_int(params, "title_font_size_px", 28), bold=False)
    subtitle_font = load_font(resolve_int(params, "subtitle_font_size_px", 18), bold=False)
    text_records.append(
        draw_readable_text(
            draw,
            xy=(margin_left, 24 + int(layout_jitter_meta.get("dy_px", 0))),
            text="Size-Encoded Category Chart",
            font=title_font,
            style=title_text_style,
            stroke_width=dense_stroke_width(),
            extra_metadata={"source": "size_encoding_title"},
        )
    )
    text_records.append(
        draw_readable_text(
            draw,
            xy=(margin_left, 58 + int(layout_jitter_meta.get("dy_px", 0))),
            text="Relative size shows value; marker colors show category.",
            font=subtitle_font,
            style=subtitle_text_style,
            stroke_width=dense_stroke_width(),
            extra_metadata={"source": "size_encoding_subtitle"},
        )
    )

    legend_bbox = (float(margin_left), float(title_band + int(layout_jitter_meta.get("dy_px", 0))), float(canvas_width - margin_right), float(title_band + legend_height + int(layout_jitter_meta.get("dy_px", 0))))
    legend_entities, legend_bboxes, legend_text_records = _draw_legend(
        draw,
        categories=dataset.categories,
        colors=category_colors,
        bbox=legend_bbox,
        params=params,
        text_style=secondary_text_style,
    )
    text_records.extend(legend_text_records)
    plot_bbox = [
        float(margin_left),
        float(title_band + legend_height + 16 + int(layout_jitter_meta.get("dy_px", 0))),
        float(canvas_width - margin_right),
        float(canvas_height - margin_bottom),
    ]

    entities: List[Dict[str, Any]] = list(legend_entities)
    item_bboxes: Dict[str, List[float]] = {}
    panel_title_bboxes: Dict[str, List[float]] = {}
    panel_bboxes = _panel_layout(tuple(plot_bbox), panel_count=len(dataset.panels), gap=float(panel_gap))
    panel_title_font = load_font(resolve_int(params, "panel_title_font_size_px", 20), bold=False)
    bubble_value_scale = _value_scale(dataset.items)

    for panel_index, (panel, panel_bbox) in enumerate(zip(dataset.panels, panel_bboxes)):
        px1, py1, px2, py2 = (float(value) for value in panel_bbox)
        draw.rounded_rectangle(
            [px1, py1, px2, py2],
            radius=resolve_int(params, "panel_corner_radius_px", 10),
            fill=panel_fill,
            outline=panel_border,
            width=resolve_int(params, "panel_border_width_px", 2),
        )
        if len(dataset.panels) > 1:
            title_xy = (px1 + 14, py1 + 10)
            text_records.append(
                draw_readable_text(
                    draw,
                    xy=title_xy,
                    text=str(panel),
                    font=panel_title_font,
                    style=secondary_text_style,
                    stroke_width=dense_stroke_width(),
                    extra_metadata={"panel": str(panel), "source": "size_encoding_panel_title"},
                )
            )
            title_bbox = _format_bbox(_text_bbox(draw, title_xy, str(panel), panel_title_font))
            panel_title_bboxes[str(panel)] = title_bbox
            content_top = py1 + 40
        else:
            panel_title_bboxes[str(panel)] = _format_bbox([px1, py1, px2, min(py2, py1 + 1)])
            content_top = py1 + panel_padding
        content_bbox: BBox = (
            px1 + panel_padding,
            content_top + panel_padding,
            px2 - panel_padding,
            py2 - panel_padding,
        )
        panel_items = [item for item in dataset.items if str(item.panel) == str(panel)]
        if str(scene_variant) == "packed_bubble_cloud" or str(scene_variant) == "small_multiple_bubble_cloud":
            rendered_entities, rendered_bboxes, rendered_text_records = _draw_bubbles(
                draw,
                items=panel_items,
                bbox=content_bbox,
                category_colors=category_colors,
                params=params,
                instance_seed=int(instance_seed) + int(panel_index),
                text_styles=bubble_text_styles,
                value_scale=bubble_value_scale,
            )
        else:
            rendered_entities, rendered_bboxes, rendered_text_records = _draw_word_cloud(
                draw,
                items=panel_items,
                bbox=content_bbox,
                category_colors=category_colors,
                params=params,
                instance_seed=int(instance_seed) + int(panel_index),
                circular=str(scene_variant) == "circle_word_cloud",
                text_styles=word_text_styles,
            )
        entities.extend(rendered_entities)
        item_bboxes.update(rendered_bboxes)
        text_records.extend(rendered_text_records)
        entities.append(
            {
                "entity_id": f"panel_{str(panel)}",
                "entity_type": "chart_size_panel",
                "bbox_px": _format_bbox([px1, py1, px2, py2]),
                "attrs": {"panel": str(panel), "scene_variant": str(scene_variant), "item_count": int(len(panel_items))},
            }
        )

    image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedSizeEncodingScene(
        image=image,
        entities=tuple(entities),
        item_bboxes=dict(item_bboxes),
        panel_title_bboxes=dict(panel_title_bboxes),
        category_legend_bboxes=dict(legend_bboxes),
        plot_bbox_px=_format_bbox(plot_bbox),
        render_meta={
            "background_style": dict(background_meta),
            "information_scene_style": dict(information_style_meta),
            "post_image_noise": dict(post_noise_meta),
            "layout_jitter": dict(layout_jitter_meta),
            "category_colors_rgb": {str(key): list(value) for key, value in category_colors.items()},
            "category_color_policy": "category is encoded by swatches, bubble fills, word markers, and contrast-adjusted category-tinted word text",
            "size_encoding_text_color_policy": "word-cloud item text uses readable category-tinted color; bubble item text uses high-contrast readable ink over category-colored bubbles",
            "size_value_scale_scope": "global",
            "size_value_scale": [int(bubble_value_scale[0]), int(bubble_value_scale[1])],
            "text_legibility": {
                **text_legibility_summary(
                    (
                        title_text_style,
                        subtitle_text_style,
                        secondary_text_style,
                        *tuple(word_text_styles.values()),
                        *tuple(bubble_text_styles.values()),
                    )
                ),
                "drawn_text_records": list(text_records),
            },
            "panel_bboxes_px": {str(panel): _format_bbox(bbox) for panel, bbox in zip(dataset.panels, panel_bboxes)},
        },
    )
