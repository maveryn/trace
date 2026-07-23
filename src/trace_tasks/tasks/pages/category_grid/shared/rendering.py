"""Rendering helpers for category-grid scene packages."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.pages.shared.information_style import resolve_pages_information_style
from trace_tasks.tasks.pages.shared.legible_text import darken_surface_for_light_text, draw_required_page_text
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import fit_font_to_box, load_font
from trace_tasks.tasks.shared.visual_style.information_scene import make_information_scene_background

from .defaults import NAMESPACE_ROOT, POST_IMAGE_NOISE_DEFAULTS, RENDER_FALLBACKS, RENDERING_DEFAULTS, SCENE
from .state import CategoryGridCase, CategoryGridSpec, RenderParams, RenderedCategoryGrid, RenderedCategoryGridBundle


def resolve_render_params(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> RenderParams:
    """Resolve render dimensions and typography parameters."""

    def _int_value(key: str, fallback: int, *, minimum: int = 1) -> int:
        return max(int(minimum), int(params.get(key, group_default(defaults, key, fallback))))

    return RenderParams(
        canvas_width=_int_value("canvas_width", int(RENDER_FALLBACKS["canvas_width"]), minimum=540),
        canvas_height=_int_value("canvas_height", int(RENDER_FALLBACKS["canvas_height"]), minimum=560),
        outer_margin_px=_int_value("outer_margin_px", int(RENDER_FALLBACKS["outer_margin_px"]), minimum=0),
        header_height_px=_int_value("header_height_px", int(RENDER_FALLBACKS["header_height_px"]), minimum=54),
        gap_px=_int_value("gap_px", int(RENDER_FALLBACKS["gap_px"]), minimum=4),
        corner_radius_px=_int_value("corner_radius_px", int(RENDER_FALLBACKS["corner_radius_px"]), minimum=0),
        outline_width_px=_int_value("outline_width_px", int(RENDER_FALLBACKS["outline_width_px"]), minimum=1),
        title_font_size_px=_int_value("title_font_size_px", int(RENDER_FALLBACKS["title_font_size_px"]), minimum=14),
        subtitle_font_size_px=_int_value("subtitle_font_size_px", int(RENDER_FALLBACKS["subtitle_font_size_px"]), minimum=10),
        category_title_font_size_px=_int_value("category_title_font_size_px", int(RENDER_FALLBACKS["category_title_font_size_px"]), minimum=11),
        subcategory_title_font_size_px=_int_value("subcategory_title_font_size_px", int(RENDER_FALLBACKS["subcategory_title_font_size_px"]), minimum=9),
        item_font_size_px=_int_value("item_font_size_px", int(RENDER_FALLBACKS["item_font_size_px"]), minimum=8),
    )


def _text_bbox(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font: Any,
    *,
    stroke_width: int = 0,
) -> List[float]:
    try:
        bbox = draw.textbbox((float(xy[0]), float(xy[1])), str(text), font=font, stroke_width=int(stroke_width))
        return [float(value) for value in bbox]
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        return [float(xy[0]), float(xy[1]), float(xy[0]) + float(width), float(xy[1]) + float(height)]


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font: Any,
    fill: Tuple[int, int, int],
    *,
    stroke_width: int = 1,
    role: str = "readout",
    surface_rgbs: Sequence[Sequence[int]] | None = None,
    instance_seed: int = 0,
    namespace: str = "",
) -> List[float]:
    if surface_rgbs:
        return draw_required_page_text(
            draw,
            (float(xy[0]), float(xy[1])),
            str(text),
            font,
            role=str(role),
            surface_rgbs=surface_rgbs,
            instance_seed=int(instance_seed),
            namespace=str(namespace or f"{NAMESPACE_ROOT}.{role}"),
            preferred_rgbs=(fill,),
            stroke_width=max(0, int(stroke_width)),
        )
    draw_text_traced(
        draw,
        (float(xy[0]), float(xy[1])),
        str(text),
        fill=fill,
        font=font,
        stroke_width=max(0, int(stroke_width)),
        stroke_fill=fill,
        role=str(role),
        required=True,
    )
    return _text_bbox(draw, xy, str(text), font, stroke_width=max(0, int(stroke_width)))


def _blend_rgb(color_a: Sequence[int], color_b: Sequence[int], weight_b: float) -> Tuple[int, int, int]:
    weight = max(0.0, min(1.0, float(weight_b)))
    return tuple(
        int(round((float(color_a[index]) * (1.0 - weight)) + (float(color_b[index]) * weight)))
        for index in range(3)
    )


def _layout_category_boxes(
    *,
    scene_variant: str,
    category_count: int,
    content_bbox: Sequence[float],
    gap_px: int,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    x0, y0, x1, y1 = [float(value) for value in content_bbox]
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    gap = float(gap_px)
    if str(scene_variant) == "column_groups":
        columns = min(max(1, int(category_count)), 4)
        rows = int((int(category_count) + int(columns) - 1) // int(columns))
    elif str(scene_variant) == "compact_index":
        columns = 1
        rows = max(1, int(category_count))
    else:
        columns = 2 if int(category_count) > 1 else 1
        rows = int((int(category_count) + int(columns) - 1) // int(columns))
    cell_w = (width - (float(columns - 1) * gap)) / float(columns)
    cell_h = (height - (float(rows - 1) * gap)) / float(rows)
    boxes: List[List[float]] = []
    for index in range(int(category_count)):
        row = int(index) // int(columns)
        col = int(index) % int(columns)
        left = x0 + float(col) * (cell_w + gap)
        top = y0 + float(row) * (cell_h + gap)
        boxes.append([left, top, left + cell_w, top + cell_h])
    return boxes, {
        "layout_columns": int(columns),
        "layout_rows": int(rows),
        "category_cell_width_px": float(cell_w),
        "category_cell_height_px": float(cell_h),
    }


def render_category_grid(
    background: Image.Image,
    *,
    spec: CategoryGridSpec,
    scene_variant: str,
    style: Any,
    render_params: RenderParams,
    instance_seed: int,
) -> RenderedCategoryGrid:
    """Draw the full category-grid page and return projected geometry."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    margin = int(render_params.outer_margin_px)
    panel_bbox = [
        float(margin),
        float(margin),
        float(render_params.canvas_width - margin),
        float(render_params.canvas_height - margin),
    ]
    radius = max(int(render_params.corner_radius_px), int(style.corner_radius_px))
    shadow = max(0, int(style.shadow_offset_px))
    if shadow:
        draw.rounded_rectangle(
            (panel_bbox[0] + shadow, panel_bbox[1] + shadow, panel_bbox[2] + shadow, panel_bbox[3] + shadow),
            radius=radius,
            fill=tuple(int(value) for value in style.shadow_rgb),
        )
    draw.rounded_rectangle(
        tuple(panel_bbox),
        radius=radius,
        fill=tuple(int(value) for value in style.surface_rgb),
        outline=tuple(int(value) for value in style.panel_border_rgb),
        width=max(1, int(render_params.outline_width_px), int(style.frame_width_px)),
    )
    header_h = max(int(render_params.header_height_px), int(style.title_band_height_px))
    header_fill = tuple(int(value) for value in style.header_rgb)
    title_surface_rgb = tuple(int(value) for value in style.surface_rgb)
    if str(style.chrome_kind) in {"accent_header", "rule_header", "accent_frame"} or str(style.chrome_mode) == "accent_frame":
        header_fill = darken_surface_for_light_text(header_fill, text_rgb=style.header_text_rgb)
        draw.rounded_rectangle(
            (panel_bbox[0], panel_bbox[1], panel_bbox[2], panel_bbox[1] + float(header_h)),
            radius=radius,
            fill=header_fill,
        )
        draw.rectangle(
            (panel_bbox[0], panel_bbox[1] + float(header_h) - float(radius), panel_bbox[2], panel_bbox[1] + float(header_h)),
            fill=header_fill,
        )
        title_rgb = tuple(int(value) for value in style.header_text_rgb)
        subtitle_rgb = tuple(int(value) for value in style.header_text_rgb)
        title_surface_rgb = tuple(int(value) for value in header_fill)
    else:
        title_rgb = tuple(int(value) for value in style.text_rgb)
        subtitle_rgb = tuple(int(value) for value in style.muted_text_rgb)
        y_rule = panel_bbox[1] + float(header_h)
        draw.line((panel_bbox[0] + 20.0, y_rule, panel_bbox[2] - 20.0, y_rule), fill=style.guide_rgb, width=1)

    scale = float(style.typography_scale)
    title_font = load_font(max(10, int(round(render_params.title_font_size_px * scale))), bold=True)
    subtitle_font = load_font(max(8, int(round(render_params.subtitle_font_size_px * scale))), bold=False)
    category_font_size = max(10, int(round(render_params.category_title_font_size_px * scale)))
    subcategory_font_size = max(9, int(round(render_params.subcategory_title_font_size_px * scale)))
    item_font_size = max(8, int(round(render_params.item_font_size_px * scale)))
    title_bbox = _draw_text(
        draw,
        (panel_bbox[0] + 24.0, panel_bbox[1] + 18.0),
        str(spec.title),
        title_font,
        title_rgb,
        role="page_title",
        surface_rgbs=(title_surface_rgb,),
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.page_title.{scene_variant}",
    )
    _draw_text(
        draw,
        (panel_bbox[0] + 24.0, panel_bbox[1] + 54.0),
        str(spec.subtitle),
        subtitle_font,
        subtitle_rgb,
        stroke_width=0,
        role="page_subtitle",
        surface_rgbs=(title_surface_rgb,),
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.page_subtitle.{scene_variant}",
    )

    content_bbox = [
        panel_bbox[0] + float(style.panel_padding_px),
        panel_bbox[1] + float(header_h) + float(render_params.gap_px),
        panel_bbox[2] - float(style.panel_padding_px),
        panel_bbox[3] - float(style.panel_padding_px),
    ]
    category_boxes, layout_meta = _layout_category_boxes(
        scene_variant=str(scene_variant),
        category_count=len(spec.categories),
        content_bbox=content_bbox,
        gap_px=int(render_params.gap_px),
    )
    text_rgb = tuple(int(value) for value in style.text_rgb)
    border_rgb = tuple(int(value) for value in style.panel_border_rgb)
    guide_rgb = tuple(int(value) for value in style.guide_rgb)
    category_header_bboxes: Dict[str, List[float]] = {}
    subcategory_header_bboxes: Dict[str, Dict[str, List[float]]] = {}
    item_row_bboxes: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    item_label_bboxes: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    entities: List[Dict[str, Any]] = []

    for category, category_bbox in zip(spec.categories, category_boxes):
        x0, y0, x1, y1 = [float(value) for value in category_bbox]
        accent = tuple(int(value) for value in category.accent_rgb)
        card_radius = max(4, int(radius * 0.65))
        card_fill = _blend_rgb(style.panel_fill_rgb, accent, 0.06 if str(scene_variant) != "compact_index" else 0.035)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=card_radius, fill=card_fill, outline=border_rgb, width=1)
        cat_band_h = max(34.0, min(46.0, (y1 - y0) * (0.20 if str(scene_variant) == "compact_index" else 0.16)))
        band_fill = darken_surface_for_light_text(
            _blend_rgb(style.header_rgb, accent, 0.24),
            text_rgb=style.header_text_rgb,
        )
        draw.rounded_rectangle((x0, y0, x1, y0 + cat_band_h), radius=card_radius, fill=band_fill)
        draw.rectangle((x0, y0 + cat_band_h - float(card_radius), x1, y0 + cat_band_h), fill=band_fill)
        category_font = fit_font_to_box(
            draw,
            text=str(category.label),
            max_width=max(40.0, x1 - x0 - 30.0),
            max_height=max(16.0, cat_band_h - 12.0),
            bold=True,
            min_size_px=10,
            max_size_px=int(category_font_size),
            fill_ratio=0.97,
        )
        cat_probe = _text_bbox(draw, (x0 + 16.0, y0 + 8.0), str(category.label), category_font)
        category_header_bboxes[str(category.category_id)] = _draw_text(
            draw,
            (x0 + 16.0, y0 + max(7.0, (cat_band_h - (cat_probe[3] - cat_probe[1])) / 2.0)),
            str(category.label),
            category_font,
            tuple(int(value) for value in style.header_text_rgb),
            role="category_header",
            surface_rgbs=(band_fill,),
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}.category_header.{category.category_id}",
        )
        subcategory_header_bboxes[str(category.category_id)] = {}
        item_row_bboxes[str(category.category_id)] = {}
        item_label_bboxes[str(category.category_id)] = {}
        sub_area = [
            x0 + 10.0,
            y0 + cat_band_h + 10.0,
            x1 - 10.0,
            y1 - 10.0,
        ]
        sub_count = len(category.subcategories)
        sub_gap = max(5.0, float(render_params.gap_px) * 0.55)
        if str(scene_variant) == "compact_index":
            sub_w = (sub_area[2] - sub_area[0] - float(sub_count - 1) * sub_gap) / float(max(1, sub_count))
            sub_h = sub_area[3] - sub_area[1]
            sub_boxes = [
                [
                    sub_area[0] + float(index) * (sub_w + sub_gap),
                    sub_area[1],
                    sub_area[0] + float(index) * (sub_w + sub_gap) + sub_w,
                    sub_area[1] + sub_h,
                ]
                for index in range(sub_count)
            ]
        else:
            sub_h = (sub_area[3] - sub_area[1] - float(sub_count - 1) * sub_gap) / float(max(1, sub_count))
            sub_boxes = [
                [
                    sub_area[0],
                    sub_area[1] + float(index) * (sub_h + sub_gap),
                    sub_area[2],
                    sub_area[1] + float(index) * (sub_h + sub_gap) + sub_h,
                ]
                for index in range(sub_count)
            ]

        category_entity = {
            "entity_id": str(category.category_id),
            "kind": "category_grid_category",
            "label": str(category.label),
            "accent_rgb": [int(value) for value in accent],
            "bbox_px": [float(value) for value in category_bbox],
            "header_bbox_px": [float(value) for value in category_header_bboxes[str(category.category_id)]],
            "subcategories": [],
        }
        for subcategory, sub_bbox in zip(category.subcategories, sub_boxes):
            sx0, sy0, sx1, sy1 = [float(value) for value in sub_bbox]
            sub_radius = max(3, int(card_radius * 0.55))
            sub_panel_fill = _blend_rgb(style.surface_alt_rgb, accent, 0.055)
            draw.rounded_rectangle(
                (sx0, sy0, sx1, sy1),
                radius=sub_radius,
                fill=sub_panel_fill,
                outline=guide_rgb,
                width=1,
            )
            sub_band_h = max(24.0, min(32.0, (sy1 - sy0) * 0.28))
            sub_band_fill = _blend_rgb(style.panel_fill_rgb, accent, 0.12)
            draw.rounded_rectangle((sx0, sy0, sx1, sy0 + sub_band_h), radius=sub_radius, fill=sub_band_fill)
            draw.rectangle((sx0, sy0 + sub_band_h - float(sub_radius), sx1, sy0 + sub_band_h), fill=sub_band_fill)
            sub_font = fit_font_to_box(
                draw,
                text=str(subcategory.label),
                max_width=max(30.0, sx1 - sx0 - 20.0),
                max_height=max(12.0, sub_band_h - 8.0),
                bold=True,
                min_size_px=8,
                max_size_px=int(subcategory_font_size),
                fill_ratio=0.97,
            )
            subcategory_header_bboxes[str(category.category_id)][str(subcategory.subcategory_id)] = _draw_text(
                draw,
                (sx0 + 10.0, sy0 + max(5.0, (sub_band_h - float(getattr(sub_font, "size", subcategory_font_size))) / 2.0)),
                str(subcategory.label),
                sub_font,
                text_rgb,
                role="subcategory_header",
                surface_rgbs=(sub_band_fill,),
                instance_seed=int(instance_seed),
                namespace=f"{NAMESPACE_ROOT}.subcategory_header.{category.category_id}.{subcategory.subcategory_id}",
            )
            item_row_bboxes[str(category.category_id)][str(subcategory.subcategory_id)] = {}
            item_label_bboxes[str(category.category_id)][str(subcategory.subcategory_id)] = {}
            item_top = sy0 + sub_band_h + 5.0
            item_bottom = sy1 - 5.0
            row_h = max(9.0, (item_bottom - item_top) / float(max(1, len(subcategory.items))))
            item_font_base = load_font(max(6, min(item_font_size, int(row_h * 0.64))), bold=True)
            subcategory_entity = {
                "subcategory_id": str(subcategory.subcategory_id),
                "label": str(subcategory.label),
                "header_bbox_px": [
                    float(value)
                    for value in subcategory_header_bboxes[str(category.category_id)][str(subcategory.subcategory_id)]
                ],
                "items": [],
            }
            for item_index, item in enumerate(subcategory.items):
                row_y0 = item_top + float(item_index) * row_h
                row_y1 = item_top + float(item_index + 1) * row_h
                row_bbox = [sx0 + 7.0, row_y0 + 1.5, sx1 - 7.0, row_y1 - 1.5]
                row_fill = sub_panel_fill
                if int(item_index) % 2 == 0 or str(scene_variant) == "compact_index":
                    row_fill = _blend_rgb(style.panel_fill_rgb, accent, 0.035)
                    draw.rounded_rectangle(tuple(row_bbox), radius=4, fill=row_fill)
                marker_r = min(4.5, max(2.5, row_h * 0.15))
                marker_cx = row_bbox[0] + 10.0
                marker_cy = (row_bbox[1] + row_bbox[3]) / 2.0
                draw.ellipse((marker_cx - marker_r, marker_cy - marker_r, marker_cx + marker_r, marker_cy + marker_r), fill=accent)
                item_font = fit_font_to_box(
                    draw,
                    text=str(item.label),
                    max_width=max(28.0, row_bbox[2] - row_bbox[0] - 28.0),
                    max_height=max(10.0, row_h - 5.0),
                    bold=True,
                    min_size_px=6,
                    max_size_px=max(6, int(getattr(item_font_base, "size", item_font_size) or item_font_size)),
                    fill_ratio=0.98,
                )
                label_bbox = _draw_text(
                    draw,
                    (row_bbox[0] + 22.0, row_bbox[1] + max(1.0, (row_h - float(getattr(item_font, "size", item_font_size))) / 2.0)),
                    str(item.label),
                    item_font,
                    text_rgb,
                    role="category_grid_item_label",
                    surface_rgbs=(row_fill,),
                    instance_seed=int(instance_seed),
                    namespace=f"{NAMESPACE_ROOT}.item_label.{item.item_id}",
                )
                full_row_bbox = [
                    float(row_bbox[0]),
                    min(float(row_bbox[1]), float(label_bbox[1]), marker_cy - marker_r),
                    float(row_bbox[2]),
                    max(float(row_bbox[3]), float(label_bbox[3]), marker_cy + marker_r),
                ]
                item_row_bboxes[str(category.category_id)][str(subcategory.subcategory_id)][str(item.item_id)] = full_row_bbox
                item_label_bboxes[str(category.category_id)][str(subcategory.subcategory_id)][str(item.item_id)] = [
                    float(value) for value in label_bbox
                ]
                subcategory_entity["items"].append(
                    {
                        "item_id": str(item.item_id),
                        "label": str(item.label),
                        "slot_index": int(item_index) + 1,
                        "item_row_bbox_px": [float(value) for value in full_row_bbox],
                        "item_label_bbox_px": [float(value) for value in label_bbox],
                    }
                )
                if int(item_index) < len(subcategory.items) - 1:
                    draw.line((sx0 + 8.0, row_y1, sx1 - 8.0, row_y1), fill=guide_rgb, width=1)
            category_entity["subcategories"].append(subcategory_entity)
        entities.append(category_entity)

    return RenderedCategoryGrid(
        image=image,
        entities=entities,
        panel_bbox_px=[float(value) for value in panel_bbox],
        title_bbox_px=[float(value) for value in title_bbox],
        category_header_bboxes_px=category_header_bboxes,
        subcategory_header_bboxes_px=subcategory_header_bboxes,
        item_row_bboxes_px=item_row_bboxes,
        item_label_bboxes_px=item_label_bboxes,
        layout_meta={
            "scene_variant": str(scene_variant),
            "content_bbox_px": [float(value) for value in content_bbox],
            "category_count": int(len(spec.categories)),
            "subcategory_count_per_category": int(len(spec.categories[0].subcategories)) if spec.categories else 0,
            **dict(layout_meta),
        },
    )


def render_category_grid_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    case: CategoryGridCase,
) -> RenderedCategoryGridBundle:
    """Render a resolved category-grid case and post-processing metadata."""

    render_params = resolve_render_params(params, RENDERING_DEFAULTS)
    style, style_meta = resolve_pages_information_style(
        instance_seed=int(instance_seed),
        params={**dict(RENDERING_DEFAULTS), **dict(params or {})},
        scene_id=SCENE,
    )
    background, background_meta = make_information_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=style,
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.information_scene_background",
    )
    rendered = render_category_grid(
        background,
        spec=case.spec,
        scene_variant=str(case.scene_variant),
        style=style,
        render_params=render_params,
        instance_seed=int(instance_seed),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedCategoryGridBundle(
        image=image,
        rendered_grid=rendered,
        render_params=render_params,
        background_meta=dict(background_meta),
        style_meta=dict(style_meta),
        post_noise_meta=dict(post_noise_meta),
    )
