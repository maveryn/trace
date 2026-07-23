"""Rendering helpers for mixed infographic page tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from .....core.seed import spawn_rng
from ....shared.font_assets import font_asset_version, font_role_trace, sample_font_family
from ....shared.text_rendering import fit_font_to_box, load_font
from ...shared.legible_text import darken_surface_for_light_text, draw_required_page_text
from ...shared.page_visual_assets import PageVisualAssetSelection, render_page_visual_asset_rgba
from .layout import _RenderParams, _layout_slots, _resolve_native_layout
from .state import _InfographicTextBlock, _MixedInfographicSpec, _MixedModule


_MIXED_INFOGRAPHIC_RENDER_NAMESPACE = "pages.mixed_infographic_page"
_ModuleDrawResult = Tuple[
    Dict[str, List[float]],
    Dict[str, List[float]],
    Dict[str, List[float]],
    Dict[str, Dict[str, List[float]]],
    Dict[str, List[float]],
]


class _MixedInfographicLayoutError(ValueError):
    """Raised when a sampled scene cannot fit its required visible content."""


@dataclass(frozen=True)
class _MixedFontProfile:
    readout_family: str
    section_header_family: str
    accent_context_family: str
    module_title_families_by_id: Dict[str, str]


@dataclass(frozen=True)
class _RenderedMixedInfographic:
    image: Image.Image
    entities: List[Dict[str, Any]]
    page_bbox_px: List[float]
    title_bbox_px: List[float]
    module_bboxes_px: Dict[str, List[float]]
    module_title_bboxes_px: Dict[str, List[float]]
    item_label_bboxes_px: Dict[str, Dict[str, List[float]]]
    item_container_bboxes_px: Dict[str, Dict[str, List[float]]]
    field_label_bboxes_px: Dict[str, Dict[str, List[float]]]
    value_cell_bboxes_px: Dict[str, Dict[str, Dict[str, List[float]]]]
    icon_bboxes_px: Dict[str, Dict[str, List[float]]]
    section_asset_bboxes_px: Dict[str, List[float]]
    decorative_asset_bboxes_px: Dict[str, List[float]]
    text_block_bboxes_px: Dict[str, List[float]]
    text_blocks: List[Dict[str, Any]]
    font_profile_meta: Dict[str, Any]
    layout_meta: Dict[str, Any]


def _sample_distinct_font_family(
    *,
    role: str,
    instance_seed: int,
    namespace: str,
    params: Mapping[str, Any],
    explicit_key: str,
    weights_key: str,
    avoid: Sequence[str],
) -> str:
    """Sample a font family for one role while keeping key page roles visually distinct."""

    avoided = {str(value) for value in avoid if str(value)}
    first = ""
    for offset in range(8):
        if str(role) == "readout":
            family = sample_font_family(
                role="readout",
                instance_seed=int(instance_seed),
                namespace=f"{namespace}.{offset}",
                params=params,
                explicit_key=str(explicit_key),
                weights_key=str(weights_key),
            )
        elif str(role) == "context":
            family = sample_font_family(
                role="context",
                instance_seed=int(instance_seed),
                namespace=f"{namespace}.{offset}",
                params=params,
                explicit_key=str(explicit_key),
                weights_key=str(weights_key),
            )
        elif str(role) == "decorative":
            family = sample_font_family(
                role="decorative",
                instance_seed=int(instance_seed),
                namespace=f"{namespace}.{offset}",
                params=params,
                explicit_key=str(explicit_key),
                weights_key=str(weights_key),
            )
        else:
            raise ValueError(f"unsupported mixed infographic font role: {role}")
        if not first:
            first = str(family)
        if str(family) not in avoided:
            return str(family)
    return str(first)


def _resolve_mixed_font_profile(
    *,
    modules: Sequence[_MixedModule],
    params: Mapping[str, Any],
    instance_seed: int,
) -> _MixedFontProfile:
    """Resolve page-wide font roles; item labels stay consistent within each module."""

    readout_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.font_profile.readout",
        params=params,
        explicit_key="mixed_infographic_readout_font_family",
        weights_key="mixed_infographic_readout_font_family_weights",
    )
    section_header_family = _sample_distinct_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.font_profile.section_header",
        params=params,
        explicit_key="mixed_infographic_section_header_font_family",
        weights_key="mixed_infographic_section_header_font_family_weights",
        avoid=(str(readout_family),),
    )
    accent_context_family = _sample_distinct_font_family(
        role="context",
        instance_seed=int(instance_seed),
        namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.font_profile.accent_context",
        params=params,
        explicit_key="mixed_infographic_context_font_family",
        weights_key="mixed_infographic_context_font_family_weights",
        avoid=(str(readout_family), str(section_header_family)),
    )
    title_families = (str(section_header_family), str(readout_family))
    module_title_families_by_id = {
        str(module.module_id): str(title_families[int(index) % len(title_families)])
        for index, module in enumerate(modules)
    }
    return _MixedFontProfile(
        readout_family=str(readout_family),
        section_header_family=str(section_header_family),
        accent_context_family=str(accent_context_family),
        module_title_families_by_id=dict(module_title_families_by_id),
    )


def _font_profile_metadata(font_profile: _MixedFontProfile) -> Dict[str, Any]:
    return {
        "asset_version": font_asset_version(),
        "policy": "mixed_infographic_three_family_profile",
        "answer_bearing_policy": "field_labels_item_labels_and_values_use_readout_family",
        "readout_family": str(font_profile.readout_family),
        "section_header_family": str(font_profile.section_header_family),
        "accent_context_family": str(font_profile.accent_context_family),
        "role_traces": {
            "readout": font_role_trace(str(font_profile.readout_family), role="readout"),
            "section_header": font_role_trace(str(font_profile.section_header_family), role="readout"),
            "accent_context": font_role_trace(str(font_profile.accent_context_family), role="context"),
        },
        "module_title_families_by_id": dict(font_profile.module_title_families_by_id),
    }


def _blend_rgb(color_a: Sequence[int], color_b: Sequence[int], weight_b: float) -> Tuple[int, int, int]:
    weight = max(0.0, min(1.0, float(weight_b)))
    return tuple(
        int(round((float(color_a[index]) * (1.0 - weight)) + (float(color_b[index]) * weight)))
        for index in range(3)
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
        return [
            float(value)
            for value in draw.textbbox((float(xy[0]), float(xy[1])), str(text), font=font, stroke_width=int(stroke_width))
        ]
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        return [float(xy[0]), float(xy[1]), float(xy[0]) + float(width), float(xy[1]) + float(height)]


def _draw_fitted_text(
    draw: ImageDraw.ImageDraw,
    *,
    box: Sequence[float],
    text: str,
    max_size_px: int,
    bold: bool,
    fill_rgb: Sequence[int],
    surface_rgbs: Sequence[Sequence[int]],
    instance_seed: int,
    namespace: str,
    role: str,
    align: str = "left",
    stroke_width: int = 1,
    font_family: str | None = None,
    required: bool = True,
) -> List[float]:
    """Draw one required text witness fitted to its bbox and return the text bbox."""

    x0, y0, x1, y1 = [float(value) for value in box]
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=max(8.0, x1 - x0),
        max_height=max(8.0, y1 - y0),
        bold=bool(bold),
        font_family=font_family,
        min_size_px=7,
        max_size_px=max(7, int(max_size_px)),
        fill_ratio=0.92,
    )
    probe = _text_bbox(draw, (x0, y0), str(text), font, stroke_width=max(0, int(stroke_width)))
    text_w = float(probe[2] - probe[0])
    text_h = float(probe[3] - probe[1])
    if str(align) == "center":
        tx = x0 + max(0.0, (x1 - x0 - text_w) / 2.0)
    elif str(align) == "right":
        tx = x1 - text_w
    else:
        tx = x0
    ty = y0 + max(0.0, (y1 - y0 - text_h) / 2.0)
    return draw_required_page_text(
        draw,
        (float(tx), float(ty)),
        str(text),
        font,
        role=str(role),
        surface_rgbs=surface_rgbs,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        preferred_rgbs=(tuple(int(value) for value in fill_rgb),),
        stroke_width=max(0, int(stroke_width)),
        required=bool(required),
    )


def _set_alpha_opacity(image: Image.Image, opacity: float) -> Image.Image:
    alpha_scale = max(0.0, min(1.0, float(opacity)))
    if alpha_scale >= 0.999:
        return image
    adjusted = image.copy()
    alpha = adjusted.getchannel("A").point(lambda value: int(round(float(value) * alpha_scale)))
    adjusted.putalpha(alpha)
    return adjusted


def _draw_visual_asset(
    image: Image.Image,
    *,
    selection: PageVisualAssetSelection,
    bbox: Sequence[float],
    tint_rgb: Sequence[int],
    opacity: float = 1.0,
) -> List[float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    asset = render_page_visual_asset_rgba(
        selection.asset,
        size_px=(max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0)))),
        tint_rgb=tuple(int(value) for value in tint_rgb),
    )
    asset = _set_alpha_opacity(asset, float(opacity))
    px = int(round(x0 + max(0.0, (x1 - x0 - asset.width) / 2.0)))
    py = int(round(y0 + max(0.0, (y1 - y0 - asset.height) / 2.0)))
    image.alpha_composite(asset, (px, py))
    return [float(px), float(py), float(px + asset.width), float(py + asset.height)]


def _draw_module_section_asset(
    image: Image.Image,
    *,
    module: _MixedModule,
    bbox: Sequence[float],
    header_height: float,
    opacity: float = 0.18,
) -> List[float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    size = min(max(42.0, width * 0.24), max(42.0, height * 0.26), 98.0)
    asset_box = [
        x1 - size - max(10.0, width * 0.035),
        y0 + float(header_height) + max(8.0, height * 0.035),
        x1 - max(10.0, width * 0.035),
        y0 + float(header_height) + max(8.0, height * 0.035) + size,
    ]
    return _draw_visual_asset(
        image,
        selection=module.section_asset_selection,
        bbox=asset_box,
        tint_rgb=module.accent_rgb,
        opacity=float(opacity),
    )


def _module_surface_rgb(style: Any, accent_rgb: Sequence[int], module_id: str) -> Tuple[int, int, int]:
    index_text = "".join(ch for ch in str(module_id) if ch.isdigit())
    index = int(index_text or 0)
    accent_weight = 0.12 + 0.04 * float(index % 4)
    alt_weight = 0.24 + 0.09 * float((index + 1) % 3)
    base = _blend_rgb(style.panel_fill_rgb, style.surface_alt_rgb, alt_weight)
    return _blend_rgb(base, accent_rgb, accent_weight)


def _draw_page_backdrops(
    draw: ImageDraw.ImageDraw,
    *,
    page_bbox: Sequence[float],
    style: Any,
    instance_seed: int,
    blend_scale: float,
) -> List[Dict[str, Any]]:
    """Draw decorative page backdrops behind modules without creating answer witnesses."""

    x0, y0, x1, y1 = [float(value) for value in page_bbox]
    width = x1 - x0
    height = y1 - y0
    rng = spawn_rng(int(instance_seed), f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.page_backdrops")
    backdrop_meta: List[Dict[str, Any]] = []
    scale = max(0.0, min(1.0, float(blend_scale)))
    colors = (
        _blend_rgb(style.surface_rgb, style.accent_rgb, 0.13 * scale),
        _blend_rgb(style.surface_rgb, style.surface_alt_rgb, 0.58 * scale),
        _blend_rgb(style.surface_rgb, style.header_rgb, 0.10 * scale),
    )
    polygons = (
        (
            (x0 + width * 0.03, y0 + height * 0.10),
            (x0 + width * 0.55, y0 + height * 0.06),
            (x0 + width * 0.50, y0 + height * 0.27),
            (x0 + width * 0.02, y0 + height * 0.31),
        ),
        (
            (x0 + width * 0.58, y0 + height * 0.16),
            (x0 + width * 0.98, y0 + height * 0.12),
            (x0 + width * 0.95, y0 + height * 0.44),
            (x0 + width * 0.63, y0 + height * 0.38),
        ),
        (
            (x0 + width * 0.08, y0 + height * 0.66),
            (x0 + width * 0.93, y0 + height * 0.58),
            (x0 + width * 0.97, y0 + height * 0.92),
            (x0 + width * 0.04, y0 + height * 0.96),
        ),
    )
    for index, polygon in enumerate(polygons):
        dx = (float(rng.random()) - 0.5) * width * 0.025
        dy = (float(rng.random()) - 0.5) * height * 0.018
        shifted = tuple((float(px) + dx, float(py) + dy) for px, py in polygon)
        fill = tuple(int(value) for value in colors[int(index) % len(colors)])
        draw.polygon(shifted, fill=fill)
        backdrop_meta.append(
            {
                "kind": "irregular_page_band",
                "polygon_px": [[float(px), float(py)] for px, py in shifted],
                "fill_rgb": [int(value) for value in fill],
                "blend_scale": float(scale),
            }
        )
    return backdrop_meta


def _bbox_overlap_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(value) for value in a]
    bx0, by0, bx1, by1 = [float(value) for value in b]
    overlap_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    overlap_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    return float(overlap_w * overlap_h)


def _wrap_text_lines(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    font: Any,
    max_width: float,
) -> List[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: List[str] = []
    current = str(words[0])
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = _text_bbox(draw, (0.0, 0.0), candidate, font, stroke_width=1)
        if float(bbox[2] - bbox[0]) <= float(max_width):
            current = candidate
        else:
            lines.append(str(current))
            current = str(word)
    lines.append(str(current))
    return lines


def _draw_wrapped_fitted_text(
    draw: ImageDraw.ImageDraw,
    *,
    box: Sequence[float],
    text: str,
    max_size_px: int,
    bold: bool,
    fill_rgb: Sequence[int],
    surface_rgbs: Sequence[Sequence[int]],
    instance_seed: int,
    namespace: str,
    role: str,
    stroke_width: int = 1,
    font_family: str | None = None,
    required: bool = True,
) -> List[float]:
    """Draw wrapped context text inside one box while preserving visible text bounds."""

    x0, y0, x1, y1 = [float(value) for value in box]
    max_width = max(8.0, x1 - x0)
    max_height = max(8.0, y1 - y0)
    selected_font = None
    selected_lines: List[str] = []
    selected_line_gap = 2.0
    for size_px in range(max(8, int(max_size_px)), 7, -1):
        font = load_font(int(size_px), bold=bool(bold), font_family=font_family)
        lines = _wrap_text_lines(draw, text=str(text), font=font, max_width=max_width * 0.95)
        line_heights = [
            max(
                1.0,
                _text_bbox(draw, (0.0, 0.0), line, font, stroke_width=max(0, int(stroke_width)))[3]
                - _text_bbox(draw, (0.0, 0.0), line, font, stroke_width=max(0, int(stroke_width)))[1],
            )
            for line in lines
        ]
        line_gap = max(2.0, float(size_px) * 0.18)
        total_height = sum(line_heights) + max(0, len(lines) - 1) * line_gap
        if total_height <= max_height * 0.95:
            selected_font = font
            selected_lines = list(lines)
            selected_line_gap = float(line_gap)
            break
    if selected_font is None:
        selected_font = load_font(8, bold=bool(bold), font_family=font_family)
        selected_lines = _wrap_text_lines(draw, text=str(text), font=selected_font, max_width=max_width * 0.95)
        selected_line_gap = 2.0

    current_y = y0
    drawn_bboxes: List[List[float]] = []
    for line_index, line in enumerate(selected_lines):
        bbox = draw_required_page_text(
            draw,
            (float(x0), float(current_y)),
            str(line),
            selected_font,
            role=str(role),
            surface_rgbs=surface_rgbs,
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.line_{line_index}",
            preferred_rgbs=(tuple(int(value) for value in fill_rgb),),
            stroke_width=max(0, int(stroke_width)),
            required=bool(required),
        )
        drawn_bboxes.append([float(value) for value in bbox])
        current_y = float(bbox[3]) + selected_line_gap
        if current_y > y1:
            break
    if not drawn_bboxes:
        return [float(x0), float(y0), float(x0), float(y0)]
    return [
        min(bbox[0] for bbox in drawn_bboxes),
        min(bbox[1] for bbox in drawn_bboxes),
        max(bbox[2] for bbox in drawn_bboxes),
        max(bbox[3] for bbox in drawn_bboxes),
    ]


def _draw_native_text_blocks(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    text_blocks: Sequence[_InfographicTextBlock],
    native_layout: _NativeLayoutPlan,
    hero_asset_selection: PageVisualAssetSelection,
    style: Any,
    font_profile: _MixedFontProfile,
    instance_seed: int,
) -> Tuple[Dict[str, List[float]], List[Dict[str, Any]], Dict[str, List[float]]]:
    """Render non-answer text/decorative blocks; bboxes remain separate from module witnesses."""

    text_block_bboxes: Dict[str, List[float]] = {}
    text_block_meta: List[Dict[str, Any]] = []
    decorative_asset_bboxes: Dict[str, List[float]] = {}
    occupied: List[List[float]] = []
    accent_cycle = (
        style.accent_rgb,
        style.header_rgb,
        style.callout_border_rgb,
        style.connector_rgb,
    )

    for index, block in enumerate(text_blocks):
        slot = [float(value) for value in native_layout.block_slots_px.get(str(block.block_id), ())]
        if not slot:
            continue
        while any(_bbox_overlap_area(slot, existing) > 0.0 for existing in occupied):
            slot = [slot[0], slot[1] + 2.0, slot[2], slot[3] + 2.0]
        accent = tuple(int(value) for value in accent_cycle[int(index) % len(accent_cycle)])
        if str(block.kind) == "paragraph_note":
            fill = _blend_rgb((246, 249, 252), accent, 0.07)
            outline = _blend_rgb(style.panel_border_rgb, accent, 0.28)
            radius = 10
        elif str(block.kind) == "source_line":
            fill = _blend_rgb(style.surface_rgb, style.surface_alt_rgb, 0.45)
            outline = _blend_rgb(style.guide_rgb, accent, 0.12)
            radius = 5
        elif str(block.kind) == "badge_note":
            fill = _blend_rgb(style.callout_fill_rgb, accent, 0.20)
            outline = _blend_rgb(style.callout_border_rgb, accent, 0.24)
            radius = 14
        else:
            fill = _blend_rgb(style.panel_fill_rgb, accent, 0.10)
            outline = _blend_rgb(style.panel_border_rgb, accent, 0.22)
            radius = 8
        draw.rounded_rectangle(
            tuple(slot),
            radius=int(radius),
            fill=fill,
            outline=outline,
            width=1,
        )
        if str(block.kind) == "paragraph_note":
            draw.rectangle((slot[0] + 8.0, slot[1] + 9.0, slot[0] + 13.0, slot[3] - 9.0), fill=accent)
            draw.line((slot[0] + 24.0, slot[1] + 15.0, slot[2] - 14.0, slot[1] + 15.0), fill=outline, width=1)
            text_box = (slot[0] + 24.0, slot[1] + 23.0, slot[2] - 14.0, slot[3] - 12.0)
            block_font_family = str(font_profile.readout_family)
            text_bbox = _draw_wrapped_fitted_text(
                draw,
                box=text_box,
                text=str(block.text),
                max_size_px=16,
                bold=True,
                fill_rgb=(10, 14, 22),
                surface_rgbs=(fill, style.surface_rgb),
                instance_seed=int(instance_seed),
                namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.{block.block_id}.native_text",
                role="mixed_infographic_native_context_text",
                stroke_width=1,
                font_family=block_font_family,
                required=False,
            )
        elif native_layout.hero_block_id == str(block.block_id) and native_layout.hero_slot_px is not None:
            decorative_asset_bboxes["hero_anchor"] = _draw_visual_asset(
                image,
                selection=hero_asset_selection,
                bbox=native_layout.hero_slot_px,
                tint_rgb=accent,
                opacity=0.94,
            )
            text_box = tuple(native_layout.hero_text_box_px or [slot[0] + 12.0, slot[1] + 5.0, slot[2] - 10.0, slot[3] - 5.0])
            block_font_family = str(font_profile.accent_context_family)
            text_bbox = _draw_fitted_text(
                draw,
                box=text_box,
                text=str(block.text),
                max_size_px=13,
                bold=str(block.kind) in {"summary_note", "callout_quote"},
                fill_rgb=style.text_rgb,
                surface_rgbs=(fill, style.surface_rgb),
                instance_seed=int(instance_seed),
                namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.{block.block_id}.native_text",
                role="mixed_infographic_native_context_text",
                align="left",
                stroke_width=1,
                font_family=block_font_family,
                required=False,
            )
        elif str(block.kind) == "badge_note":
            draw.ellipse(
                (slot[0] + 7.0, slot[1] + 8.0, slot[0] + 19.0, slot[1] + 20.0),
                fill=accent,
            )
            text_box = (slot[0] + 25.0, slot[1] + 4.0, slot[2] - 8.0, slot[3] - 4.0)
            block_font_family = str(font_profile.accent_context_family)
            text_bbox = _draw_fitted_text(
                draw,
                box=text_box,
                text=str(block.text),
                max_size_px=12,
                bold=True,
                fill_rgb=style.text_rgb,
                surface_rgbs=(fill, style.surface_rgb),
                instance_seed=int(instance_seed),
                namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.{block.block_id}.native_text",
                role="mixed_infographic_native_context_text",
                align="center",
                stroke_width=1,
                font_family=block_font_family,
                required=False,
            )
        else:
            draw.rectangle((slot[0] + 7.0, slot[1] + 7.0, slot[0] + 10.0, slot[3] - 7.0), fill=accent)
            text_box = (slot[0] + 16.0, slot[1] + 4.0, slot[2] - 8.0, slot[3] - 4.0)
            block_font_family = str(font_profile.accent_context_family)
            text_bbox = _draw_fitted_text(
                draw,
                box=text_box,
                text=str(block.text),
                max_size_px=12,
                bold=str(block.kind) in {"summary_note", "callout_quote"},
                fill_rgb=style.muted_text_rgb if str(block.kind) == "source_line" else style.text_rgb,
                surface_rgbs=(fill, style.surface_rgb),
                instance_seed=int(instance_seed),
                namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.{block.block_id}.native_text",
                role="mixed_infographic_native_context_text",
                align="left",
                stroke_width=1,
                font_family=block_font_family,
                required=False,
            )
        bbox = [float(value) for value in slot]
        text_block_bboxes[str(block.block_id)] = list(bbox)
        occupied.append(list(bbox))
        text_block_meta.append(
            {
                "block_id": str(block.block_id),
                "kind": str(block.kind),
                "text": str(block.text),
                "placement_region": str(native_layout.block_regions.get(str(block.block_id), block.placement_region)),
                "font_role": str(block.font_role),
                "font_family": str(block_font_family),
                "bbox_px": list(bbox),
                "text_bbox_px": [float(value) for value in text_bbox],
                "fill_rgb": [int(value) for value in fill],
                "outline_rgb": [int(value) for value in outline],
            }
        )
    return text_block_bboxes, text_block_meta, decorative_asset_bboxes


def _draw_module_shell(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    title: str,
    accent_rgb: Sequence[int],
    module_kind: str,
    style: Any,
    render_params: _RenderParams,
    instance_seed: int,
    module_id: str,
    font_profile: _MixedFontProfile,
) -> Tuple[List[float], List[float]]:
    """Draw a module container and title; returned title bbox anchors task annotations."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    fill = _module_surface_rgb(style, accent_rgb, str(module_id))
    border = tuple(int(value) for value in style.panel_border_rgb)
    header_h = min(46.0, max(34.0, (y1 - y0) * 0.16))
    header_fill = darken_surface_for_light_text(
        _blend_rgb(style.header_rgb, accent_rgb, 0.22),
        min_contrast_ratio=7.6,
    )

    if str(module_kind) in {"radial_bubbles", "ring_summary"}:
        draw.ellipse(
            (x0, y0, x1, y1),
            fill=fill,
            outline=border,
            width=max(1, int(render_params.outline_width_px)),
        )
        inset = min(18.0, max(8.0, (x1 - x0) * 0.04))
        draw.ellipse(
            (x0 + inset, y0 + inset, x1 - inset, y1 - inset),
            outline=_blend_rgb(fill, accent_rgb, 0.32),
            width=2,
        )
        title_box = (x0 + 25.0, y0 + 10.0, x1 - 25.0, y0 + header_h + 5.0)
        draw.rounded_rectangle(
            title_box,
            radius=max(10, int(render_params.corner_radius_px) + 4),
            fill=header_fill,
        )
        title_text_box = (title_box[0] + 8.0, title_box[1] + 4.0, title_box[2] - 8.0, title_box[3] - 4.0)
    else:
        radius = max(0, int(render_params.corner_radius_px))
        if str(module_kind) == "callout_stats":
            radius += 10
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=radius,
            fill=fill,
            outline=border,
            width=max(1, int(render_params.outline_width_px)),
        )
        if str(module_kind) == "timeline_snippet":
            draw.polygon(
                (
                    (x1 - 54.0, y0),
                    (x1, y0),
                    (x1, y0 + 54.0),
                    (x1 - 28.0, y0 + 32.0),
                ),
                fill=_blend_rgb(fill, accent_rgb, 0.28),
            )
        elif str(module_kind) == "profile_cards":
            draw.ellipse(
                (x1 - 76.0, y1 - 76.0, x1 - 18.0, y1 - 18.0),
                outline=_blend_rgb(fill, accent_rgb, 0.30),
                width=3,
            )
        draw.rounded_rectangle(
            (x0, y0, x1, y0 + header_h),
            radius=radius,
            fill=header_fill,
        )
        draw.rectangle((x0, y0 + header_h - 5.0, x1, y0 + header_h), fill=header_fill)
        draw.rectangle((x0, y0, x0 + 7.0, y1), fill=tuple(int(value) for value in accent_rgb))
        title_text_box = (x0 + 16.0, y0 + 5.0, x1 - 12.0, y0 + header_h - 5.0)

    title_bbox = _draw_fitted_text(
        draw,
        box=title_text_box,
        text=str(title),
        max_size_px=int(render_params.module_title_font_size_px),
        bold=True,
        fill_rgb=style.header_text_rgb,
        surface_rgbs=(header_fill,),
        instance_seed=int(instance_seed),
        namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.{module_id}.title",
        role="mixed_infographic_module_title",
        stroke_width=1,
        font_family=str(font_profile.module_title_families_by_id.get(str(module_id), font_profile.section_header_family)),
    )
    return [float(x0), float(y0), float(x1), float(y1)], [float(value) for value in title_bbox]


def _draw_table_like_module(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    module: _MixedModule,
    bbox: Sequence[float],
    header_height: float,
    style: Any,
    render_params: _RenderParams,
    instance_seed: int,
    font_profile: _MixedFontProfile,
) -> _ModuleDrawResult:
    """Render grid/table modules; item, field, and value bboxes must align by ids."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    pad = 10.0
    top = y0 + float(header_height) + pad
    left = x0 + pad + 4.0
    right = x1 - pad
    bottom = y1 - pad
    field_h = min(28.0, max(18.0, (bottom - top) * 0.16))
    row_count = max(1, len(module.items))
    row_h = max(14.0, (bottom - top - field_h) / float(row_count))
    label_col_w = min((right - left) * 0.42, 118.0)
    field_w = max(34.0, (right - left - label_col_w) / float(max(1, len(module.fields))))
    guide_rgb = tuple(int(value) for value in style.guide_rgb)
    item_bboxes: Dict[str, List[float]] = {}
    item_container_bboxes: Dict[str, List[float]] = {}
    field_bboxes: Dict[str, List[float]] = {}
    value_bboxes: Dict[str, Dict[str, List[float]]] = {}
    icon_bboxes: Dict[str, List[float]] = {}

    draw.rectangle((left, top, right, top + field_h), fill=_blend_rgb(style.surface_alt_rgb, style.accent_rgb, 0.08))
    for field_index, field in enumerate(module.fields):
        fx0 = left + label_col_w + float(field_index) * field_w
        fx1 = fx0 + field_w
        field_bboxes[str(field.field_id)] = _draw_fitted_text(
            draw,
            box=(fx0 + 3.0, top + 3.0, fx1 - 3.0, top + field_h - 3.0),
            text=str(field.label),
            max_size_px=int(render_params.field_font_size_px),
            bold=True,
            fill_rgb=style.muted_text_rgb,
            surface_rgbs=(style.surface_alt_rgb, style.panel_fill_rgb),
            instance_seed=int(instance_seed),
            namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.{module.module_id}.{field.field_id}.field",
            role="mixed_infographic_field_label",
            align="center",
            stroke_width=1,
            font_family=str(font_profile.readout_family),
        )
        draw.line((fx0, top, fx0, bottom), fill=guide_rgb, width=1)
    draw.line((left, top + field_h, right, top + field_h), fill=guide_rgb, width=1)

    for item_index, item in enumerate(module.items):
        row_top = top + field_h + float(item_index) * row_h
        row_bottom = min(bottom, row_top + row_h)
        row_box = [float(left), float(row_top), float(right), float(row_bottom)]
        item_container_bboxes[str(item.item_id)] = list(row_box)
        if item_index % 2 == 0:
            draw.rectangle((left, row_top, right, row_bottom), fill=_blend_rgb(style.panel_fill_rgb, style.surface_alt_rgb, 0.28))
        icon_box = (left + 3.0, row_top + 5.0, left + 23.0, row_bottom - 5.0)
        icon_bboxes[str(item.item_id)] = _draw_visual_asset(
            image,
            selection=item.visual_asset_selection,
            bbox=icon_box,
            tint_rgb=module.accent_rgb,
        )
        item_bboxes[str(item.item_id)] = _draw_fitted_text(
            draw,
            box=(left + 28.0, row_top + 4.0, left + label_col_w - 5.0, row_bottom - 4.0),
            text=str(item.label),
            max_size_px=int(render_params.label_font_size_px),
            bold=True,
            fill_rgb=style.text_rgb,
            surface_rgbs=(style.panel_fill_rgb, style.surface_alt_rgb),
            instance_seed=int(instance_seed),
            namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.{module.module_id}.{item.item_id}.item",
            role="mixed_infographic_item_label",
            stroke_width=1,
            font_family=str(font_profile.readout_family),
        )
        value_bboxes[str(item.item_id)] = {}
        for field_index, field in enumerate(module.fields):
            fx0 = left + label_col_w + float(field_index) * field_w
            fx1 = fx0 + field_w
            value = str(item.values_by_field_id[str(field.field_id)])
            value_box = [float(fx0), float(row_top), float(fx1), float(row_bottom)]
            draw.rectangle(
                tuple(value_box),
                fill=_blend_rgb(style.panel_fill_rgb, style.surface_alt_rgb, 0.16 if item_index % 2 else 0.24),
                outline=_blend_rgb(style.guide_rgb, module.accent_rgb, 0.10),
                width=1,
            )
            _draw_fitted_text(
                draw,
                box=(fx0 + 4.0, row_top + 4.0, fx1 - 4.0, row_bottom - 4.0),
                text=value,
                max_size_px=int(render_params.value_font_size_px),
                bold=True,
                fill_rgb=style.text_rgb,
                surface_rgbs=(style.panel_fill_rgb, style.surface_alt_rgb),
                instance_seed=int(instance_seed),
                namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.{module.module_id}.{item.item_id}.{field.field_id}.value",
                role="mixed_infographic_value_cell",
                align="center",
                stroke_width=1,
                font_family=str(font_profile.readout_family),
            )
            value_bboxes[str(item.item_id)][str(field.field_id)] = list(value_box)
        draw.line((left, row_bottom, right, row_bottom), fill=guide_rgb, width=1)
    return item_bboxes, item_container_bboxes, field_bboxes, value_bboxes, icon_bboxes


def _draw_card_like_module(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    module: _MixedModule,
    bbox: Sequence[float],
    header_height: float,
    style: Any,
    render_params: _RenderParams,
    instance_seed: int,
    font_profile: _MixedFontProfile,
) -> _ModuleDrawResult:
    """Render compact card modules while preserving per-item and per-field bbox maps."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    pad = 10.0
    top = y0 + float(header_height) + pad
    left = x0 + pad + 4.0
    right = x1 - pad
    bottom = y1 - pad
    chip_h = min(24.0, max(18.0, (bottom - top) * 0.16))
    chip_w = (right - left - max(0, len(module.fields) - 1) * 6.0) / float(max(1, len(module.fields)))
    item_bboxes: Dict[str, List[float]] = {}
    item_container_bboxes: Dict[str, List[float]] = {}
    field_bboxes: Dict[str, List[float]] = {}
    value_bboxes: Dict[str, Dict[str, List[float]]] = {}
    icon_bboxes: Dict[str, List[float]] = {}

    for field_index, field in enumerate(module.fields):
        cx0 = left + float(field_index) * (chip_w + 6.0)
        cx1 = cx0 + chip_w
        draw.rounded_rectangle(
            (cx0, top, cx1, top + chip_h),
            radius=5,
            fill=_blend_rgb(style.callout_fill_rgb, module.accent_rgb, 0.10),
            outline=tuple(int(value) for value in style.callout_border_rgb),
            width=1,
        )
        field_bboxes[str(field.field_id)] = _draw_fitted_text(
            draw,
            box=(cx0 + 5.0, top + 3.0, cx1 - 5.0, top + chip_h - 3.0),
            text=str(field.label),
            max_size_px=int(render_params.field_font_size_px),
            bold=True,
            fill_rgb=style.muted_text_rgb,
            surface_rgbs=(style.callout_fill_rgb, style.panel_fill_rgb),
            instance_seed=int(instance_seed),
            namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.{module.module_id}.{field.field_id}.field",
            role="mixed_infographic_field_label",
            align="center",
            stroke_width=1,
            font_family=str(font_profile.readout_family),
        )

    card_top = top + chip_h + 8.0
    cols = 2 if len(module.items) > 2 and (right - left) > 250.0 else 1
    rows = max(1, int(math.ceil(float(len(module.items)) / float(cols))))
    min_card_content_h = chip_h + 8.0 + float(rows) * 32.0 + float(max(0, rows - 1)) * 8.0
    if (bottom - top) < min_card_content_h:
        return _draw_table_like_module(
            image,
            draw,
            module=module,
            bbox=bbox,
            header_height=header_height,
            style=style,
            render_params=render_params,
            instance_seed=instance_seed,
            font_profile=font_profile,
        )
    card_w = (right - left - (cols - 1) * 8.0) / float(cols)
    card_h = max(22.0, (bottom - card_top - (rows - 1) * 8.0) / float(rows))
    for item_index, item in enumerate(module.items):
        row = int(item_index // cols)
        col = int(item_index % cols)
        cx0 = left + float(col) * (card_w + 8.0)
        cy0 = card_top + float(row) * (card_h + 8.0)
        cx1 = cx0 + card_w
        if cy0 >= bottom:
            cy0 = max(card_top, bottom - card_h)
        cy1 = min(bottom, cy0 + card_h)
        card_box = [float(cx0), float(cy0), float(cx1), float(cy1)]
        item_container_bboxes[str(item.item_id)] = list(card_box)
        draw.rounded_rectangle(
            tuple(card_box),
            radius=7,
            fill=_blend_rgb(style.panel_fill_rgb, style.surface_alt_rgb, 0.45),
            outline=tuple(int(value) for value in style.guide_rgb),
            width=1,
        )
        icon_bboxes[str(item.item_id)] = _draw_visual_asset(
            image,
            selection=item.visual_asset_selection,
            bbox=(cx0 + 6.0, cy0 + 6.0, cx0 + 26.0, cy0 + 26.0),
            tint_rgb=module.accent_rgb,
        )
        label_y0 = cy0 + 5.0
        value_bottom = max(cy0 + 18.0, cy1 - 4.0)
        value_top = min(
            value_bottom - 10.0,
            max(label_y0 + 18.0, min(cy1 - 16.0, cy0 + card_h * 0.58)),
        )
        label_y1 = min(value_top - 8.0, cy0 + 24.0)
        if label_y1 < label_y0 + 8.0:
            label_y1 = label_y0 + 8.0
            value_top = max(value_top, label_y1 + 8.0)

        item_bboxes[str(item.item_id)] = _draw_fitted_text(
            draw,
            box=(cx0 + 31.0, label_y0, cx1 - 6.0, label_y1),
            text=str(item.label),
            max_size_px=int(render_params.label_font_size_px),
            bold=True,
            fill_rgb=style.text_rgb,
            surface_rgbs=(style.panel_fill_rgb, style.surface_alt_rgb),
            instance_seed=int(instance_seed),
            namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.{module.module_id}.{item.item_id}.item",
            role="mixed_infographic_item_label",
            stroke_width=1,
            font_family=str(font_profile.readout_family),
        )
        value_bboxes[str(item.item_id)] = {}
        value_gap = 4.0
        field_count = max(1, len(module.fields))
        value_w = max(18.0, (cx1 - cx0 - 14.0 - value_gap * float(field_count - 1)) / float(field_count))
        for field_index, field in enumerate(module.fields):
            vx0 = cx0 + 7.0 + float(field_index) * (value_w + value_gap)
            vx1 = min(cx1 - 7.0, vx0 + value_w)
            value = str(item.values_by_field_id[str(field.field_id)])
            value_box = [float(vx0), float(value_top), float(vx1), float(value_bottom)]
            value_fill = _blend_rgb(style.callout_fill_rgb, module.accent_rgb, 0.08 + 0.03 * float(field_index % 2))
            draw.rounded_rectangle(
                tuple(value_box),
                radius=6,
                fill=value_fill,
                outline=_blend_rgb(style.guide_rgb, module.accent_rgb, 0.14),
                width=1,
            )
            _draw_fitted_text(
                draw,
                box=(vx0 + 3.0, value_top + 2.0, vx1 - 3.0, value_bottom - 2.0),
                text=value,
                max_size_px=int(render_params.value_font_size_px),
                bold=True,
                fill_rgb=style.text_rgb,
                surface_rgbs=(value_fill, style.panel_fill_rgb, style.surface_alt_rgb),
                instance_seed=int(instance_seed),
                namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.{module.module_id}.{item.item_id}.{field.field_id}.value",
                role="mixed_infographic_value_cell",
                align="center",
                stroke_width=1,
                font_family=str(font_profile.readout_family),
            )
            value_bboxes[str(item.item_id)][str(field.field_id)] = list(value_box)
    return item_bboxes, item_container_bboxes, field_bboxes, value_bboxes, icon_bboxes


def _draw_radial_like_module(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    module: _MixedModule,
    bbox: Sequence[float],
    header_height: float,
    style: Any,
    render_params: _RenderParams,
    instance_seed: int,
    font_profile: _MixedFontProfile,
) -> _ModuleDrawResult:
    """Render radial modules; polar layout still records rectangular value witnesses."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    width = x1 - x0
    height = y1 - y0
    left = x0 + max(22.0, width * 0.10)
    right = x1 - max(22.0, width * 0.10)
    top = y0 + float(header_height) + max(16.0, height * 0.06)
    bottom = y1 - max(18.0, height * 0.09)
    center_x = (left + right) / 2.0
    field_count = max(1, len(module.fields))
    item_count = max(1, len(module.items))
    item_bboxes: Dict[str, List[float]] = {}
    item_container_bboxes: Dict[str, List[float]] = {}
    field_bboxes: Dict[str, List[float]] = {}
    value_bboxes: Dict[str, Dict[str, List[float]]] = {}
    icon_bboxes: Dict[str, List[float]] = {}
    surface_rgb = _module_surface_rgb(style, module.accent_rgb, str(module.module_id))

    chip_gap = 7.0
    chip_h = 24.0
    chip_w = min(92.0, max(54.0, (right - left - chip_gap * float(field_count - 1)) / float(field_count)))
    chip_total_w = chip_w * float(field_count) + chip_gap * float(field_count - 1)
    chip_left = center_x - chip_total_w / 2.0
    for field_index, field in enumerate(module.fields):
        cx0 = chip_left + float(field_index) * (chip_w + chip_gap)
        cy0 = top
        cx1 = cx0 + chip_w
        cy1 = cy0 + chip_h
        chip_fill = _blend_rgb(style.callout_fill_rgb, module.accent_rgb, 0.15 + 0.04 * float(field_index % 2))
        draw.rounded_rectangle(
            (cx0, cy0, cx1, cy1),
            radius=12,
            fill=chip_fill,
            outline=_blend_rgb(style.callout_border_rgb, module.accent_rgb, 0.18),
            width=1,
        )
        field_bboxes[str(field.field_id)] = _draw_fitted_text(
            draw,
            box=(cx0 + 5.0, cy0 + 3.0, cx1 - 5.0, cy1 - 3.0),
            text=str(field.label),
            max_size_px=int(render_params.field_font_size_px),
            bold=True,
            fill_rgb=style.muted_text_rgb,
            surface_rgbs=(chip_fill, surface_rgb),
            instance_seed=int(instance_seed),
            namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.{module.module_id}.{field.field_id}.field",
            role="mixed_infographic_field_label",
            align="center",
            stroke_width=1,
            font_family=str(font_profile.readout_family),
        )

    bubble_area_top = top + chip_h + 12.0
    bubble_area_h = max(18.0, bottom - bubble_area_top)
    cols = 1 if item_count == 1 else 2
    rows = int(math.ceil(float(item_count) / float(cols)))
    bubble_gap = min(10.0, max(3.0, bubble_area_h * 0.08))
    cell_w = max(34.0, (right - left - bubble_gap * float(cols - 1)) / float(cols))
    cell_h = max(10.0, (bubble_area_h - bubble_gap * float(rows - 1)) / float(rows))
    bubble_w = min(cell_w * 0.92, 148.0)
    bubble_h = min(cell_h * 0.88, 112.0)

    for item_index, item in enumerate(module.items):
        row = int(item_index // cols)
        col = int(item_index % cols)
        cell_x0 = left + float(col) * (cell_w + bubble_gap)
        cell_y0 = bubble_area_top + float(row) * (cell_h + bubble_gap)
        cell_x1 = cell_x0 + cell_w
        cell_y1 = min(bottom, cell_y0 + cell_h)
        if item_count == 3 and item_index == 2:
            cell_x0 = left + (right - left - cell_w) / 2.0
            cell_x1 = cell_x0 + cell_w
        cx = (cell_x0 + cell_x1) / 2.0
        cy = (cell_y0 + cell_y1) / 2.0
        bx0 = max(cell_x0, cx - bubble_w / 2.0)
        by0 = max(cell_y0, cy - bubble_h / 2.0)
        bx1 = min(right, bx0 + bubble_w)
        by1 = min(bottom, by0 + bubble_h)
        bx0 = max(cell_x0, bx1 - bubble_w)
        by0 = max(cell_y0, by1 - bubble_h)
        bubble_fill = _blend_rgb(surface_rgb, module.accent_rgb, 0.10 + 0.05 * float(item_index % 3))
        outline = _blend_rgb(style.panel_border_rgb, module.accent_rgb, 0.28)
        bubble_box = [float(bx0), float(by0), float(bx1), float(by1)]
        item_container_bboxes[str(item.item_id)] = list(bubble_box)
        if str(module.kind) == "ring_summary":
            draw.rounded_rectangle((bx0, by0, bx1, by1), radius=18, fill=bubble_fill, outline=outline, width=1)
            draw.arc((bx0 + 5.0, by0 + 5.0, bx1 - 5.0, by1 - 5.0), 205, 338, fill=tuple(int(value) for value in module.accent_rgb), width=3)
        else:
            draw.ellipse((bx0, by0, bx1, by1), fill=bubble_fill, outline=outline, width=1)

        value_gap = 5.0
        value_band_h = min(22.0, max(10.0, bubble_h * 0.24))
        value_bottom = by1 - max(5.0, bubble_h * 0.08)
        value_top = max(by0 + 28.0, value_bottom - value_band_h)
        label_y0 = by0 + 7.0
        label_bottom = min(value_top - 5.0, by0 + max(20.0, bubble_h * 0.48))
        label_bottom = max(label_y0 + 8.0, label_bottom)
        icon_size = min(15.0, max(8.0, bubble_h * 0.15))
        icon_y0 = by0 + max(6.0, bubble_h * 0.09)
        icon_bboxes[str(item.item_id)] = _draw_visual_asset(
            image,
            selection=item.visual_asset_selection,
            bbox=(bx0 + 12.0, icon_y0, bx0 + 12.0 + icon_size, icon_y0 + icon_size),
            tint_rgb=module.accent_rgb,
        )
        item_bboxes[str(item.item_id)] = _draw_fitted_text(
            draw,
            box=(bx0 + 30.0, label_y0, bx1 - 12.0, label_bottom),
            text=str(item.label),
            max_size_px=int(render_params.label_font_size_px),
            bold=True,
            fill_rgb=style.text_rgb,
            surface_rgbs=(bubble_fill, surface_rgb),
            instance_seed=int(instance_seed),
            namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.{module.module_id}.{item.item_id}.item",
            role="mixed_infographic_item_label",
            align="center",
            stroke_width=1,
            font_family=str(font_profile.readout_family),
        )
        value_bboxes[str(item.item_id)] = {}
        value_w = max(9.0, (bx1 - bx0 - 20.0 - value_gap * float(field_count - 1)) / float(field_count))
        value_left = bx0 + 10.0
        for field_index, field in enumerate(module.fields):
            vx0 = value_left + float(field_index) * (value_w + value_gap)
            vx1 = max(vx0 + 8.0, min(bx1 - 8.0, vx0 + value_w))
            vy0 = value_top
            vy1 = max(vy0 + 8.0, value_bottom)
            value_fill = _blend_rgb(style.panel_fill_rgb, module.accent_rgb, 0.08)
            draw.rounded_rectangle(
                (vx0, vy0, vx1, vy1),
                radius=8,
                fill=value_fill,
                outline=_blend_rgb(style.guide_rgb, module.accent_rgb, 0.16),
                width=1,
            )
            value_box = [float(vx0), float(vy0), float(vx1), float(vy1)]
            _draw_fitted_text(
                draw,
                box=(vx0 + 4.0, vy0 + 2.0, vx1 - 4.0, vy1 - 2.0),
                text=str(item.values_by_field_id[str(field.field_id)]),
                max_size_px=int(render_params.value_font_size_px),
                bold=True,
                fill_rgb=style.text_rgb,
                surface_rgbs=(value_fill, bubble_fill),
                instance_seed=int(instance_seed),
                namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.{module.module_id}.{item.item_id}.{field.field_id}.value",
                role="mixed_infographic_value_cell",
                align="center",
                stroke_width=1,
                font_family=str(font_profile.readout_family),
            )
            value_bboxes[str(item.item_id)][str(field.field_id)] = list(value_box)
    return item_bboxes, item_container_bboxes, field_bboxes, value_bboxes, icon_bboxes


def _draw_row_like_module(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    module: _MixedModule,
    bbox: Sequence[float],
    header_height: float,
    style: Any,
    render_params: _RenderParams,
    instance_seed: int,
    font_profile: _MixedFontProfile,
) -> _ModuleDrawResult:
    """Render row-summary modules with id-aligned item, field, value, and icon bboxes."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    pad = 10.0
    top = y0 + float(header_height) + pad
    left = x0 + pad + 4.0
    right = x1 - pad
    bottom = y1 - pad
    item_bboxes: Dict[str, List[float]] = {}
    item_container_bboxes: Dict[str, List[float]] = {}
    field_bboxes: Dict[str, List[float]] = {}
    value_bboxes: Dict[str, Dict[str, List[float]]] = {}
    icon_bboxes: Dict[str, List[float]] = {}
    field_h = min(22.0, max(16.0, (bottom - top) * 0.14))
    label_w = min(128.0, (right - left) * 0.44)
    field_w = max(32.0, (right - left - label_w) / float(max(1, len(module.fields))))
    for field_index, field in enumerate(module.fields):
        fx0 = left + label_w + float(field_index) * field_w
        fx1 = fx0 + field_w
        field_bboxes[str(field.field_id)] = _draw_fitted_text(
            draw,
            box=(fx0 + 3.0, top, fx1 - 3.0, top + field_h),
            text=str(field.label),
            max_size_px=int(render_params.field_font_size_px),
            bold=True,
            fill_rgb=style.muted_text_rgb,
            surface_rgbs=(style.panel_fill_rgb,),
            instance_seed=int(instance_seed),
            namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.{module.module_id}.{field.field_id}.field",
            role="mixed_infographic_field_label",
            align="center",
            stroke_width=1,
            font_family=str(font_profile.readout_family),
        )
    row_top = top + field_h + 3.0
    row_count = max(1, len(module.items))
    available_row_height = float(bottom - row_top)
    minimum_row_height = 15.0
    if available_row_height < minimum_row_height * float(row_count):
        raise _MixedInfographicLayoutError(
            f"row-like module {module.module_id} has insufficient vertical space "
            f"for {row_count} items"
        )
    row_h = available_row_height / float(row_count)
    for item_index, item in enumerate(module.items):
        yy0 = row_top + float(item_index) * row_h
        yy1 = min(bottom, yy0 + row_h - 2.0)
        row_box = [float(left), float(yy0), float(right), float(yy1)]
        item_container_bboxes[str(item.item_id)] = list(row_box)
        row_fill = _blend_rgb(style.panel_fill_rgb, style.surface_alt_rgb, 0.30 if item_index % 2 else 0.42)
        draw.rounded_rectangle(
            tuple(row_box),
            radius=6,
            fill=row_fill,
            outline=_blend_rgb(style.guide_rgb, module.accent_rgb, 0.10),
            width=1,
        )
        if str(module.kind) == "ranked_list":
            draw.ellipse((left + 2.0, yy0 + 5.0, left + 25.0, yy0 + 28.0), fill=tuple(int(value) for value in module.accent_rgb))
            icon_bboxes[str(item.item_id)] = _draw_visual_asset(
                image,
                selection=item.visual_asset_selection,
                bbox=(left + 6.0, yy0 + 8.0, left + 21.0, yy0 + 25.0),
                tint_rgb=(255, 255, 255),
            )
        elif str(module.kind) == "timeline_snippet":
            cx = left + 14.0
            cy = yy0 + max(10.0, (yy1 - yy0) / 2.0)
            draw.line((cx, row_top, cx, bottom), fill=tuple(int(value) for value in style.connector_rgb), width=2)
            draw.ellipse((cx - 7.0, cy - 7.0, cx + 7.0, cy + 7.0), fill=tuple(int(value) for value in module.accent_rgb))
            icon_bboxes[str(item.item_id)] = _draw_visual_asset(
                image,
                selection=item.visual_asset_selection,
                bbox=(cx - 5.0, cy - 5.0, cx + 5.0, cy + 5.0),
                tint_rgb=(255, 255, 255),
            )
        else:
            icon_bboxes[str(item.item_id)] = _draw_visual_asset(
                image,
                selection=item.visual_asset_selection,
                bbox=(left + 2.0, yy0 + 4.0, left + 26.0, yy1 - 4.0),
                tint_rgb=module.accent_rgb,
            )
        item_bboxes[str(item.item_id)] = _draw_fitted_text(
            draw,
            box=(left + 31.0, yy0 + 3.0, left + label_w - 4.0, yy1 - 3.0),
            text=str(item.label),
            max_size_px=int(render_params.label_font_size_px),
            bold=True,
            fill_rgb=style.text_rgb,
            surface_rgbs=(style.panel_fill_rgb, style.surface_alt_rgb),
            instance_seed=int(instance_seed),
            namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.{module.module_id}.{item.item_id}.item",
            role="mixed_infographic_item_label",
            stroke_width=1,
            font_family=str(font_profile.readout_family),
        )
        value_bboxes[str(item.item_id)] = {}
        for field_index, field in enumerate(module.fields):
            fx0 = left + label_w + float(field_index) * field_w
            fx1 = fx0 + field_w
            value_box = [float(fx0), float(yy0), float(fx1), float(yy1)]
            value_fill = _blend_rgb(row_fill, module.accent_rgb, 0.05 + 0.03 * float(field_index % 2))
            draw.rounded_rectangle(
                tuple(value_box),
                radius=5,
                fill=value_fill,
                outline=_blend_rgb(style.guide_rgb, module.accent_rgb, 0.12),
                width=1,
            )
            _draw_fitted_text(
                draw,
                box=(fx0 + 3.0, yy0 + 3.0, fx1 - 3.0, yy1 - 3.0),
                text=str(item.values_by_field_id[str(field.field_id)]),
                max_size_px=int(render_params.value_font_size_px),
                bold=True,
                fill_rgb=style.text_rgb,
                surface_rgbs=(value_fill, style.panel_fill_rgb, style.surface_alt_rgb),
                instance_seed=int(instance_seed),
                namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.{module.module_id}.{item.item_id}.{field.field_id}.value",
                role="mixed_infographic_value_cell",
                align="center",
                stroke_width=1,
                font_family=str(font_profile.readout_family),
            )
            value_bboxes[str(item.item_id)][str(field.field_id)] = list(value_box)
        draw.line((left, yy1, right, yy1), fill=tuple(int(value) for value in style.guide_rgb), width=1)
    return item_bboxes, item_container_bboxes, field_bboxes, value_bboxes, icon_bboxes


def _render_mixed_infographic(
    background: Image.Image,
    *,
    spec: _MixedInfographicSpec,
    scene_variant: str,
    native_layout_mode: str,
    style: Any,
    render_params: _RenderParams,
    instance_seed: int,
    font_profile: _MixedFontProfile,
) -> _RenderedMixedInfographic:
    """Render the full page and collect every bbox map consumed by task annotations."""

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image)
    margin = float(render_params.outer_margin_px)
    page_bbox = [
        margin,
        margin,
        float(render_params.canvas_width) - margin,
        float(render_params.canvas_height) - margin,
    ]
    draw.rounded_rectangle(
        tuple(page_bbox),
        radius=max(0, int(render_params.corner_radius_px) + 4),
        fill=tuple(int(value) for value in style.surface_rgb),
        outline=tuple(int(value) for value in style.panel_border_rgb),
        width=max(1, int(render_params.outline_width_px)),
    )
    page_backdrops = _draw_page_backdrops(
        draw,
        page_bbox=page_bbox,
        style=style,
        instance_seed=int(instance_seed),
        blend_scale=float(render_params.page_backdrop_blend_scale),
    )
    native_layout = _resolve_native_layout(
        page_bbox=page_bbox,
        render_params=render_params,
        native_layout_mode=str(native_layout_mode),
        text_blocks=spec.text_blocks,
    )
    title_bbox = _draw_fitted_text(
        draw,
        box=native_layout.title_bbox_px,
        text=str(spec.title),
        max_size_px=int(render_params.title_font_size_px),
        bold=True,
        fill_rgb=style.text_rgb,
        surface_rgbs=(style.surface_rgb,),
        instance_seed=int(instance_seed),
        namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.page_title",
        role="mixed_infographic_page_title",
        stroke_width=1,
        font_family=str(font_profile.section_header_family),
    )
    _draw_fitted_text(
        draw,
        box=native_layout.subtitle_bbox_px,
        text=str(spec.subtitle),
        max_size_px=int(render_params.subtitle_font_size_px),
        bold=False,
        fill_rgb=style.muted_text_rgb,
        surface_rgbs=(style.surface_rgb,),
        instance_seed=int(instance_seed),
        namespace=f"{_MIXED_INFOGRAPHIC_RENDER_NAMESPACE}.page_subtitle",
        role="mixed_infographic_page_subtitle",
        stroke_width=1,
        font_family=str(font_profile.readout_family),
    )
    slots, layout_meta = _layout_slots(
        render_params=render_params,
        scene_variant=str(scene_variant),
        module_count=len(spec.modules),
        instance_seed=int(instance_seed),
        content_bbox=native_layout.content_bbox_px,
        footer_bbox=native_layout.footer_bbox_px,
        layout_namespace=_MIXED_INFOGRAPHIC_RENDER_NAMESPACE,
    )
    layout_meta["page_backdrops"] = list(page_backdrops)
    layout_meta.update(dict(native_layout.meta))
    module_bboxes: Dict[str, List[float]] = {}
    module_title_bboxes: Dict[str, List[float]] = {}
    item_label_bboxes: Dict[str, Dict[str, List[float]]] = {}
    item_container_bboxes: Dict[str, Dict[str, List[float]]] = {}
    field_label_bboxes: Dict[str, Dict[str, List[float]]] = {}
    value_cell_bboxes: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    icon_bboxes: Dict[str, Dict[str, List[float]]] = {}
    section_asset_bboxes: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = []

    for module, slot in zip(spec.modules, slots):
        header_h = min(46.0, max(34.0, (float(slot[3]) - float(slot[1])) * 0.16))
        module_bbox, title_box = _draw_module_shell(
            draw,
            bbox=slot,
            title=str(module.title),
            accent_rgb=module.accent_rgb,
            module_kind=str(module.kind),
            style=style,
            render_params=render_params,
            instance_seed=int(instance_seed),
            module_id=str(module.module_id),
            font_profile=font_profile,
        )
        section_asset_bboxes[str(module.module_id)] = _draw_module_section_asset(
            image,
            module=module,
            bbox=slot,
            header_height=header_h,
            opacity=0.17 if str(module.section_asset_selection.asset.render_mode) == "color" else 0.22,
        )
        if str(module.kind) in {"radial_bubbles", "ring_summary"}:
            item_boxes, item_container_boxes, field_boxes, value_boxes, icon_boxes = _draw_radial_like_module(
                image,
                draw,
                module=module,
                bbox=slot,
                header_height=header_h,
                style=style,
                render_params=render_params,
                instance_seed=int(instance_seed),
                font_profile=font_profile,
            )
        elif str(module.kind) in {"profile_cards", "callout_stats"}:
            item_boxes, item_container_boxes, field_boxes, value_boxes, icon_boxes = _draw_card_like_module(
                image,
                draw,
                module=module,
                bbox=slot,
                header_height=header_h,
                style=style,
                render_params=render_params,
                instance_seed=int(instance_seed),
                font_profile=font_profile,
            )
        elif str(module.kind) in {"icon_metric_list", "ranked_list", "timeline_snippet"}:
            item_boxes, item_container_boxes, field_boxes, value_boxes, icon_boxes = _draw_row_like_module(
                image,
                draw,
                module=module,
                bbox=slot,
                header_height=header_h,
                style=style,
                render_params=render_params,
                instance_seed=int(instance_seed),
                font_profile=font_profile,
            )
        else:
            item_boxes, item_container_boxes, field_boxes, value_boxes, icon_boxes = _draw_table_like_module(
                image,
                draw,
                module=module,
                bbox=slot,
                header_height=header_h,
                style=style,
                render_params=render_params,
                instance_seed=int(instance_seed),
                font_profile=font_profile,
            )
        module_bboxes[str(module.module_id)] = [float(value) for value in module_bbox]
        module_title_bboxes[str(module.module_id)] = [float(value) for value in title_box]
        item_label_bboxes[str(module.module_id)] = dict(item_boxes)
        item_container_bboxes[str(module.module_id)] = dict(item_container_boxes)
        field_label_bboxes[str(module.module_id)] = dict(field_boxes)
        value_cell_bboxes[str(module.module_id)] = {str(item_id): dict(fields) for item_id, fields in value_boxes.items()}
        icon_bboxes[str(module.module_id)] = dict(icon_boxes)
        entities.append(
            {
                "entity_id": str(module.module_id),
                "kind": "mixed_infographic_module",
                "module_kind": str(module.kind),
                "title": str(module.title),
                "bbox_px": [float(value) for value in module_bbox],
                "title_bbox_px": [float(value) for value in title_box],
                "accent_rgb": [int(value) for value in module.accent_rgb],
                "section_visual_asset_id": str(module.section_asset_selection.asset.asset_id),
                "section_visual_asset_bbox_px": [float(value) for value in section_asset_bboxes[str(module.module_id)]],
                "fields": [
                    {
                        "field_id": str(field.field_id),
                        "label": str(field.label),
                        "bbox_px": [float(value) for value in field_boxes[str(field.field_id)]],
                    }
                    for field in module.fields
                ],
                "items": [
                    {
                        "item_id": str(item.item_id),
                        "label": str(item.label),
                        "visual_asset_id": str(item.visual_asset_selection.asset.asset_id),
                        "visual_asset_source_id": str(item.visual_asset_selection.asset.source_id),
                        "label_bbox_px": [float(value) for value in item_boxes[str(item.item_id)]],
                        "item_container_bbox_px": [
                            float(value)
                            for value in item_container_boxes[str(item.item_id)]
                        ],
                        "visual_asset_bbox_px": [float(value) for value in icon_boxes[str(item.item_id)]],
                        "values": [
                            {
                                "field_id": str(field.field_id),
                                "field_label": str(field.label),
                                "value": str(item.values_by_field_id[str(field.field_id)]),
                                "bbox_px": [
                                    float(value)
                                    for value in value_boxes[str(item.item_id)][str(field.field_id)]
                                ],
                            }
                            for field in module.fields
                        ],
                    }
                    for item in module.items
                ],
            }
        )

    text_block_bboxes, text_block_meta, decorative_asset_bboxes = _draw_native_text_blocks(
        image,
        draw,
        text_blocks=spec.text_blocks,
        native_layout=native_layout,
        hero_asset_selection=spec.hero_asset_selection,
        style=style,
        font_profile=font_profile,
        instance_seed=int(instance_seed),
    )
    for block_meta in text_block_meta:
        entities.append(
            {
                "entity_id": str(block_meta["block_id"]),
                "kind": "mixed_infographic_text_block",
                "block_kind": str(block_meta["kind"]),
                "text": str(block_meta["text"]),
                "bbox_px": [float(value) for value in block_meta["bbox_px"]],
                "text_bbox_px": [float(value) for value in block_meta["text_bbox_px"]],
                "font_family": str(block_meta["font_family"]),
                "placement_region": str(block_meta["placement_region"]),
            }
        )

    rendered = image.convert("RGB")
    return _RenderedMixedInfographic(
        image=rendered,
        entities=entities,
        page_bbox_px=[float(value) for value in page_bbox],
        title_bbox_px=[float(value) for value in title_bbox],
        module_bboxes_px=module_bboxes,
        module_title_bboxes_px=module_title_bboxes,
        item_label_bboxes_px=item_label_bboxes,
        item_container_bboxes_px=item_container_bboxes,
        field_label_bboxes_px=field_label_bboxes,
        value_cell_bboxes_px=value_cell_bboxes,
        icon_bboxes_px=icon_bboxes,
        section_asset_bboxes_px=section_asset_bboxes,
        decorative_asset_bboxes_px=decorative_asset_bboxes,
        text_block_bboxes_px=text_block_bboxes,
        text_blocks=list(text_block_meta),
        font_profile_meta=_font_profile_metadata(font_profile),
        layout_meta=dict(layout_meta),
    )

__all__ = [
    "_MixedFontProfile",
    "_MixedInfographicLayoutError",
    "_RenderedMixedInfographic",
    "_render_mixed_infographic",
    "_resolve_mixed_font_profile",
]
