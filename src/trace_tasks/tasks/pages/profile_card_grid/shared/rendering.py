"""Rendering helpers for profile-card-grid page scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.background import make_background_canvas
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.text_rendering import fit_font_to_box, load_font
from trace_tasks.tasks.pages.shared.legible_text import draw_required_page_text

from .defaults import POST_IMAGE_BACKGROUND_DEFAULTS, POST_IMAGE_NOISE_DEFAULTS, RENDER_FALLBACKS, RENDERING_DEFAULTS
from .state import (
    ProfileCard,
    ProfileCardGridCase,
    ProfileCardRenderParams,
    RenderedProfileCardGrid,
    RenderedProfileCardGridBundle,
)


def resolve_render_params(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> ProfileCardRenderParams:
    """Resolve render dimensions and typography parameters."""

    def _int_value(key: str, fallback: int, *, minimum: int = 1) -> int:
        return max(int(minimum), int(params.get(key, group_default(defaults, key, fallback))))

    return ProfileCardRenderParams(
        canvas_width=_int_value("canvas_width", int(RENDER_FALLBACKS["canvas_width"]), minimum=540),
        canvas_height=_int_value("canvas_height", int(RENDER_FALLBACKS["canvas_height"]), minimum=560),
        outer_margin_px=_int_value("outer_margin_px", int(RENDER_FALLBACKS["outer_margin_px"]), minimum=0),
        header_height_px=_int_value("header_height_px", int(RENDER_FALLBACKS["header_height_px"]), minimum=54),
        gap_px=_int_value("gap_px", int(RENDER_FALLBACKS["gap_px"]), minimum=4),
        corner_radius_px=_int_value("corner_radius_px", int(RENDER_FALLBACKS["corner_radius_px"]), minimum=0),
        outline_width_px=_int_value("outline_width_px", int(RENDER_FALLBACKS["outline_width_px"]), minimum=1),
        title_font_size_px=_int_value("title_font_size_px", int(RENDER_FALLBACKS["title_font_size_px"]), minimum=14),
        subtitle_font_size_px=_int_value("subtitle_font_size_px", int(RENDER_FALLBACKS["subtitle_font_size_px"]), minimum=10),
        card_title_font_size_px=_int_value("card_title_font_size_px", int(RENDER_FALLBACKS["card_title_font_size_px"]), minimum=12),
        label_font_size_px=_int_value("label_font_size_px", int(RENDER_FALLBACKS["label_font_size_px"]), minimum=9),
        value_font_size_px=_int_value("value_font_size_px", int(RENDER_FALLBACKS["value_font_size_px"]), minimum=10),
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
    role: str = "readout",
    surface_rgbs: Sequence[Sequence[int]],
    instance_seed: int,
    namespace: str,
) -> List[float]:
    return draw_required_page_text(
        draw,
        (float(xy[0]), float(xy[1])),
        str(text),
        font,
        role=str(role),
        surface_rgbs=surface_rgbs,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        preferred_rgbs=(fill,),
        stroke_width=1,
        required=True,
    )


def _blend_rgb(color_a: Sequence[int], color_b: Sequence[int], weight_b: float) -> Tuple[int, int, int]:
    weight = max(0.0, min(1.0, float(weight_b)))
    return tuple(
        int(round((float(color_a[index]) * (1.0 - weight)) + (float(color_b[index]) * weight)))
        for index in range(3)
    )


def _layout_grid(
    *,
    count: int,
    columns: int,
    render_params: ProfileCardRenderParams,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    rows = int((int(count) + int(columns) - 1) // int(columns))
    margin = int(render_params.outer_margin_px)
    gap = int(render_params.gap_px)
    top = float(margin + render_params.header_height_px)
    left = float(margin)
    right = float(render_params.canvas_width - margin)
    bottom = float(render_params.canvas_height - margin)
    inner_w = max(1.0, right - left)
    inner_h = max(1.0, bottom - top)
    card_w = (inner_w - (float(columns - 1) * float(gap))) / float(columns)
    card_h = (inner_h - (float(rows - 1) * float(gap))) / float(rows)
    bboxes: List[List[float]] = []
    for index in range(int(count)):
        row = int(index) // int(columns)
        col = int(index) % int(columns)
        x0 = left + (float(col) * (card_w + float(gap)))
        y0 = top + (float(row) * (card_h + float(gap)))
        bboxes.append([x0, y0, x0 + card_w, y0 + card_h])
    return bboxes, {"layout_columns": int(columns), "layout_rows": int(rows)}


def _vertical_text_rows(
    *,
    y0: float,
    y1: float,
    top_offset: float,
    row_count: int,
    bottom_padding: float,
    preferred_min_gap: float,
    max_gap: float,
    max_text_height: float,
) -> Tuple[float, float, float]:
    top = float(y0) + float(top_offset)
    count = max(1, int(row_count))
    available = max(1.0, float(y1) - float(top) - float(bottom_padding))
    text_height = min(float(max_text_height), max(8.0, (float(available) / float(count)) * 0.82))
    if count == 1:
        return float(top), 0.0, float(text_height)
    fit_gap = max(1.0, (float(available) - float(text_height)) / float(count - 1))
    if float(fit_gap) >= float(preferred_min_gap):
        row_gap = max(float(preferred_min_gap), min(float(max_gap), float(fit_gap)))
    else:
        row_gap = float(fit_gap)
    return float(top), float(row_gap), float(text_height)


def _draw_panel_header(
    image: Image.Image,
    *,
    render_params: ProfileCardRenderParams,
    page_title: str,
    page_subtitle: str,
    instance_seed: int,
) -> Tuple[ImageDraw.ImageDraw, List[float], List[float]]:
    """Draw the page frame/header; required text must stay contrast-safe against the panel fill."""

    draw = ImageDraw.Draw(image)
    margin = int(render_params.outer_margin_px)
    panel_bbox = [
        float(margin),
        float(margin),
        float(render_params.canvas_width - margin),
        float(render_params.canvas_height - margin),
    ]
    panel_fill = (249, 250, 250)
    draw.rounded_rectangle(
        tuple(panel_bbox),
        radius=18,
        fill=panel_fill,
        outline=(205, 212, 220),
        width=2,
    )
    title_font = load_font(int(render_params.title_font_size_px), bold=True)
    subtitle_font = load_font(int(render_params.subtitle_font_size_px), bold=False)
    title_xy = (float(margin + 24), float(margin + 18))
    title_bbox = _draw_text(
        draw,
        title_xy,
        str(page_title),
        title_font,
        (35, 42, 50),
        role="page_title",
        surface_rgbs=(panel_fill,),
        instance_seed=int(instance_seed),
        namespace="pages.profile_card_grid.header.title",
    )
    _draw_text(
        draw,
        (title_xy[0], title_xy[1] + 38.0),
        str(page_subtitle),
        subtitle_font,
        (91, 101, 113),
        role="page_subtitle",
        surface_rgbs=(panel_fill,),
        instance_seed=int(instance_seed),
        namespace="pages.profile_card_grid.header.subtitle",
    )
    return draw, panel_bbox, title_bbox


def render_profile_cards(
    background: Image.Image,
    *,
    cards: Sequence[ProfileCard],
    page_title: str,
    page_subtitle: str,
    scene_variant: str,
    render_params: ProfileCardRenderParams,
    instance_seed: int,
) -> RenderedProfileCardGrid:
    """Draw the full profile-card grid and retain all card/field boxes."""

    image = background.convert("RGB")
    draw, panel_bbox, title_bbox = _draw_panel_header(
        image,
        render_params=render_params,
        page_title=str(page_title),
        page_subtitle=str(page_subtitle),
        instance_seed=int(instance_seed),
    )
    columns = 3 if str(scene_variant) == "directory_grid" else 2
    card_bboxes, layout_meta = _layout_grid(count=len(cards), columns=int(columns), render_params=render_params)
    text_rgb = (35, 42, 50)
    muted_rgb = (91, 101, 113)
    entities: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    card_map: Dict[str, List[float]] = {}
    name_map: Dict[str, List[float]] = {}
    label_map: Dict[str, Dict[str, List[float]]] = {}
    value_map: Dict[str, Dict[str, List[float]]] = {}
    for card, bbox in zip(cards, card_bboxes):
        x0, y0, x1, y1 = [float(value) for value in bbox]
        accent = tuple(int(channel) for channel in card.accent_rgb)
        card_fill = _blend_rgb((255, 255, 255), accent, 0.035)
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=int(render_params.corner_radius_px),
            fill=card_fill,
            outline=(205, 212, 220),
            width=int(render_params.outline_width_px),
        )
        draw.rounded_rectangle((x0, y0, x1, y0 + 8.0), radius=int(render_params.corner_radius_px), fill=accent)
        name_font = fit_font_to_box(
            draw,
            text=str(card.name),
            max_width=max(40.0, x1 - x0 - 32.0),
            max_height=30.0,
            bold=True,
            min_size_px=12,
            max_size_px=int(render_params.card_title_font_size_px),
            fill_ratio=0.97,
        )
        profile_id = str(card.profile_id)
        name_bbox = _draw_text(
            draw,
            (x0 + 16.0, y0 + 22.0),
            str(card.name),
            name_font,
            text_rgb,
            role="profile_name",
            surface_rgbs=(card_fill,),
            instance_seed=int(instance_seed),
            namespace=f"pages.profile_card_grid.{profile_id}.name",
        )
        row_top, row_gap, row_text_height = _vertical_text_rows(
            y0=y0,
            y1=y1,
            top_offset=62.0,
            row_count=len(card.fields),
            bottom_padding=16.0,
            preferred_min_gap=28.0,
            max_gap=42.0,
            max_text_height=24.0,
        )
        label_bboxes: Dict[str, List[float]] = {}
        value_bboxes: Dict[str, List[float]] = {}
        for row_index, (field_label, field_value) in enumerate(card.fields.items()):
            fy = row_top + (float(row_index) * float(row_gap))
            label_font = fit_font_to_box(
                draw,
                text=f"{field_label}:",
                max_width=82.0,
                max_height=min(22.0, row_text_height),
                bold=True,
                min_size_px=7,
                max_size_px=min(int(render_params.label_font_size_px), max(7, int(row_text_height))),
                fill_ratio=0.96,
            )
            value_font = fit_font_to_box(
                draw,
                text=str(field_value),
                max_width=max(40.0, x1 - x0 - 118.0),
                max_height=float(row_text_height),
                bold=False,
                min_size_px=7,
                max_size_px=min(int(render_params.value_font_size_px), max(7, int(row_text_height))),
                fill_ratio=0.97,
            )
            label_bbox = _draw_text(
                draw,
                (x0 + 18.0, fy),
                f"{field_label}:",
                label_font,
                muted_rgb,
                role="field_label",
                surface_rgbs=(card_fill,),
                instance_seed=int(instance_seed),
                namespace=f"pages.profile_card_grid.{profile_id}.field_label.{row_index}",
            )
            value_bbox = _draw_text(
                draw,
                (x0 + 102.0, fy),
                str(field_value),
                value_font,
                text_rgb,
                role="field_value",
                surface_rgbs=(card_fill,),
                instance_seed=int(instance_seed),
                namespace=f"pages.profile_card_grid.{profile_id}.field_value.{row_index}",
            )
            label_bboxes[str(field_label)] = [float(value) for value in label_bbox]
            value_bboxes[str(field_label)] = [float(value) for value in value_bbox]
        trace = {
            "profile_id": profile_id,
            "name": str(card.name),
            "fields": dict(card.fields),
            "card_bbox_px": [float(value) for value in bbox],
            "name_bbox_px": [float(value) for value in name_bbox],
            "field_label_bboxes_px": dict(label_bboxes),
            "field_value_bboxes_px": dict(value_bboxes),
            "accent_rgb": [int(channel) for channel in accent],
        }
        entities.append(
            {
                "id": profile_id,
                "type": "profile_card",
                "bbox_px": [float(value) for value in bbox],
                "attrs": {"name": str(card.name), "fields": dict(card.fields)},
            }
        )
        traces.append(trace)
        card_map[profile_id] = [float(value) for value in bbox]
        name_map[profile_id] = [float(value) for value in name_bbox]
        label_map[profile_id] = dict(label_bboxes)
        value_map[profile_id] = dict(value_bboxes)
    return RenderedProfileCardGrid(
        image=image,
        entities=tuple(dict(entity) for entity in entities),
        card_traces=tuple(dict(trace) for trace in traces),
        panel_bbox_px=list(panel_bbox),
        title_bbox_px=list(title_bbox),
        layout_meta={"scene_variant": str(scene_variant), "card_count": int(len(cards)), **dict(layout_meta)},
        card_bboxes_px=dict(card_map),
        name_bboxes_px=dict(name_map),
        field_label_bboxes_px=dict(label_map),
        field_value_bboxes_px=dict(value_map),
    )


def render_profile_card_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    case: ProfileCardGridCase,
) -> RenderedProfileCardGridBundle:
    """Render one profile-card scene case and attach image-level metadata."""

    render_params = resolve_render_params(params, RENDERING_DEFAULTS)
    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_BACKGROUND_DEFAULTS,
    )
    rendered_grid = render_profile_cards(
        background,
        cards=case.spec.cards,
        page_title=str(case.spec.title),
        page_subtitle=str(case.spec.subtitle),
        scene_variant=str(case.scene_variant),
        render_params=render_params,
        instance_seed=int(instance_seed),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_grid.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedProfileCardGridBundle(
        image=image,
        rendered_grid=rendered_grid,
        render_params=render_params,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )
