"""Layout planning and render-parameter helpers for mixed infographic pages."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default
from .state import _InfographicTextBlock
from .sampling import (
    resolve_named_variant as _resolve_named_variant,
    resolve_supported_int as _resolve_supported_int,
)


NATIVE_LAYOUT_MODES: Tuple[str, ...] = (
    "footer_only",
    "top_right_callout",
    "top_left_callout",
    "left_side_rail",
    "right_side_rail",
    "poster_anchor_strip",
    "corner_stamp",
)
NATIVE_LAYOUT_MODE_WEIGHTS: Dict[str, float] = {
    "footer_only": 0.18,
    "top_right_callout": 0.16,
    "top_left_callout": 0.14,
    "left_side_rail": 0.16,
    "right_side_rail": 0.16,
    "poster_anchor_strip": 0.12,
    "corner_stamp": 0.08,
}


@dataclass(frozen=True)
class _RenderParams:
    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    header_height_px: int
    gap_px: int
    corner_radius_px: int
    outline_width_px: int
    title_font_size_px: int
    subtitle_font_size_px: int
    module_title_font_size_px: int
    label_font_size_px: int
    value_font_size_px: int
    field_font_size_px: int
    native_text_footer_height_px: int
    page_backdrop_blend_scale: float


@dataclass(frozen=True)
class _NativeLayoutPlan:
    mode: str
    title_bbox_px: List[float]
    subtitle_bbox_px: List[float]
    content_bbox_px: List[float]
    footer_bbox_px: List[float]
    block_slots_px: Dict[str, List[float]]
    block_regions: Dict[str, str]
    hero_slot_px: List[float] | None
    hero_text_box_px: List[float] | None
    hero_block_id: str | None
    meta: Dict[str, Any]


def _resolve_native_text_block_count(
    *,
    sampling_namespace: str,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[int, Tuple[int, ...], Dict[str, float]]:
    return _resolve_supported_int(
        sampling_namespace=str(sampling_namespace),
        params=params,
        gen_defaults=render_defaults,
        explicit_key="native_text_block_count",
        support_key="native_text_block_count_support",
        fallback=(4, 5),
        instance_seed=int(instance_seed),
        namespace="native_text_block_count",
    )


def _resolve_native_layout_mode(
    *,
    sampling_namespace: str,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[str, Dict[str, float]]:
    defaults = dict(render_defaults)
    defaults.setdefault("native_layout_mode_weights", dict(NATIVE_LAYOUT_MODE_WEIGHTS))
    defaults.setdefault("balanced_native_layout_mode_sampling", False)
    return _resolve_named_variant(
        sampling_namespace=str(sampling_namespace),
        gen_defaults=defaults,
        params=params,
        instance_seed=int(instance_seed),
        supported=NATIVE_LAYOUT_MODES,
        explicit_key="native_layout_mode",
        weights_key="native_layout_mode_weights",
        balance_flag_key="balanced_native_layout_mode_sampling",
        namespace="native_layout_mode",
    )


def _resolve_render_params(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> _RenderParams:
    def _int_value(key: str, fallback: int, *, minimum: int = 1) -> int:
        return max(int(minimum), int(params.get(key, group_default(defaults, key, fallback))))

    def _float_value(key: str, fallback: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
        value = float(params.get(key, group_default(defaults, key, fallback)))
        return max(float(minimum), min(float(maximum), value))

    return _RenderParams(
        canvas_width=_int_value("canvas_width", 1040, minimum=520),
        canvas_height=_int_value("canvas_height", 1320, minimum=700),
        outer_margin_px=_int_value("outer_margin_px", 36, minimum=12),
        header_height_px=_int_value("header_height_px", 106, minimum=64),
        gap_px=_int_value("gap_px", 14, minimum=6),
        corner_radius_px=_int_value("corner_radius_px", 10, minimum=0),
        outline_width_px=_int_value("outline_width_px", 2, minimum=1),
        title_font_size_px=_int_value("title_font_size_px", 32, minimum=16),
        subtitle_font_size_px=_int_value("subtitle_font_size_px", 17, minimum=10),
        module_title_font_size_px=_int_value("module_title_font_size_px", 17, minimum=10),
        label_font_size_px=_int_value("label_font_size_px", 13, minimum=8),
        value_font_size_px=_int_value("value_font_size_px", 14, minimum=8),
        field_font_size_px=_int_value("field_font_size_px", 11, minimum=7),
        native_text_footer_height_px=_int_value("native_text_footer_height_px", 138, minimum=92),
        page_backdrop_blend_scale=_float_value("page_backdrop_blend_scale", 0.35),
    )


def _layout_slots(
    *,
    render_params: _RenderParams,
    scene_variant: str,
    module_count: int,
    instance_seed: int,
    layout_namespace: str,
    content_bbox: Sequence[float],
    footer_bbox: Sequence[float],
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Place module slots inside content bounds while preserving readable minimum sizes."""

    gap = float(render_params.gap_px)
    left, top, right, bottom = [float(value) for value in content_bbox]
    width = right - left
    height = bottom - top
    slots: List[List[float]] = []
    variant = str(scene_variant)
    rng = spawn_rng(int(instance_seed), f"{layout_namespace}.layout.{variant}.{module_count}")

    def _clamp_slot(slot: Sequence[float]) -> List[float]:
        x0, y0, x1, y1 = [float(value) for value in slot]
        min_w = min(220.0, max(160.0, width * 0.18))
        min_h = 142.0
        x0 = max(left, min(right - min_w, x0))
        y0 = max(top, min(bottom - min_h, y0))
        x1 = max(x0 + min_w, min(right, x1))
        y1 = max(y0 + min_h, min(bottom, y1))
        return [float(x0), float(y0), float(x1), float(y1)]

    def _scaled_slot(frac: Sequence[float]) -> List[float]:
        x0, y0, x1, y1 = [float(value) for value in frac]
        return _clamp_slot(
            [
                left + x0 * width,
                top + y0 * height,
                left + x1 * width,
                top + y1 * height,
            ]
        )

    def _jitter_slots(base_slots: Sequence[Sequence[float]], *, amount: float) -> List[List[float]]:
        jittered: List[List[float]] = []
        for index, slot in enumerate(base_slots):
            x0, y0, x1, y1 = [float(value) for value in slot]
            slot_w = x1 - x0
            slot_h = y1 - y0
            dx = (float(rng.random()) - 0.5) * float(amount) * min(gap * 2.6, slot_w * 0.08)
            dy = (float(rng.random()) - 0.5) * float(amount) * min(gap * 2.8, slot_h * 0.08)
            grow_x = (float(rng.random()) - 0.5) * float(amount) * min(gap * 1.6, slot_w * 0.04)
            grow_y = (float(rng.random()) - 0.5) * float(amount) * min(gap * 1.8, slot_h * 0.04)
            if index % 3 == 1:
                dy += min(gap * 0.55, slot_h * 0.025)
            elif index % 3 == 2:
                dy -= min(gap * 0.35, slot_h * 0.02)
            jittered.append(_clamp_slot([x0 + dx - grow_x, y0 + dy - grow_y, x1 + dx + grow_x, y1 + dy + grow_y]))
        return jittered

    def _append_grid(*, count: int, cols: int, grid_left: float, grid_top: float, grid_width: float, grid_height: float) -> None:
        rows = max(1, int(math.ceil(float(count) / float(cols))))
        cell_h = (float(grid_height) - float(rows - 1) * gap) / float(rows)
        for index in range(int(count)):
            row = int(index // cols)
            default_col = int(index % cols)
            remaining = int(count) - int(row) * int(cols)
            row_cols = int(cols) if int(remaining) >= int(cols) else int(remaining)
            row_cols = max(1, int(row_cols))
            col = int(default_col) if int(row_cols) == int(cols) else int(index - int(row) * int(cols))
            cell_w = (float(grid_width) - float(row_cols - 1) * gap) / float(row_cols)
            x0 = float(grid_left) + float(col) * (cell_w + gap)
            y0 = float(grid_top) + float(row) * (cell_h + gap)
            slots.append([x0, y0, x0 + cell_w, y0 + cell_h])

    irregular_patterns: Dict[str, Tuple[Tuple[float, float, float, float], ...]] = {
        "collage_board": (
            (0.00, 0.00, 0.36, 0.22),
            (0.39, 0.02, 0.66, 0.20),
            (0.69, 0.00, 1.00, 0.29),
            (0.02, 0.26, 0.41, 0.50),
            (0.44, 0.24, 0.72, 0.48),
            (0.75, 0.33, 1.00, 0.56),
            (0.00, 0.56, 0.30, 0.86),
            (0.33, 0.52, 0.67, 0.83),
            (0.70, 0.61, 1.00, 0.90),
        ),
        "radial_mosaic": (
            (0.34, 0.22, 0.69, 0.52),
            (0.02, 0.01, 0.35, 0.21),
            (0.67, 0.02, 0.98, 0.22),
            (0.00, 0.27, 0.32, 0.50),
            (0.72, 0.28, 1.00, 0.51),
            (0.05, 0.57, 0.36, 0.84),
            (0.39, 0.55, 0.67, 0.86),
            (0.70, 0.58, 0.98, 0.86),
            (0.17, 0.86, 0.83, 1.00),
        ),
    }

    if variant in irregular_patterns:
        slots = [_scaled_slot(pattern) for pattern in irregular_patterns[str(variant)][: int(module_count)]]
        slots = _jitter_slots(slots, amount=1.0)
    elif variant == "poster_sections":
        hero_h = min(210.0, max(170.0, height * 0.18))
        slots.append([left, top, right, top + hero_h])
        remaining = int(module_count) - 1
        _append_grid(
            count=remaining,
            cols=2,
            grid_left=left,
            grid_top=top + hero_h + gap,
            grid_width=width,
            grid_height=height - hero_h - gap,
        )
    elif variant == "compact_newsletter":
        _append_grid(count=int(module_count), cols=2, grid_left=left, grid_top=top, grid_width=width, grid_height=height)
    elif variant == "dashboard_blocks" and int(module_count) >= 7:
        top_h = min(270.0, max(230.0, height * 0.23))
        top_w = (width - gap) / 2.0
        slots.append([left, top, left + top_w, top + top_h])
        slots.append([left + top_w + gap, top, right, top + top_h])
        remaining = int(module_count) - 2
        _append_grid(
            count=remaining,
            cols=3,
            grid_left=left,
            grid_top=top + top_h + gap,
            grid_width=width,
            grid_height=height - top_h - gap,
        )
    else:
        _append_grid(count=int(module_count), cols=3, grid_left=left, grid_top=top, grid_width=width, grid_height=height)

    if variant not in irregular_patterns:
        slots = _jitter_slots(slots, amount=0.45 if variant in {"masonry_report", "dashboard_blocks"} else 0.28)

    return slots[: int(module_count)], {
        "scene_variant": str(scene_variant),
        "module_count": int(module_count),
        "placement_mode": "irregular_fractional_slots" if variant in irregular_patterns else "jittered_structured_slots",
        "content_bbox_px": [left, top, right, bottom],
        "native_text_footer_bbox_px": [float(value) for value in footer_bbox],
        "native_text_footer_height_px": int(render_params.native_text_footer_height_px),
        "slot_bboxes_px": [list(slot) for slot in slots[: int(module_count)]],
        "layout_jitter_seed": int(instance_seed),
    }


def _footer_text_slot(footer_bbox: Sequence[float], region: str) -> List[float]:
    x0, y0, x1, y1 = [float(value) for value in footer_bbox]
    gap = 12.0
    width = x1 - x0
    paragraph_bottom = y1 - 38.0
    half_w = (width - gap) / 2.0
    small_w = (width - 2.0 * gap) / 3.0
    regions = {
        "paragraph_left": [x0, y0, x0 + half_w, paragraph_bottom],
        "paragraph_right": [x0 + half_w + gap, y0, x1, paragraph_bottom],
        "footer_badge": [x0, paragraph_bottom + 8.0, x0 + small_w, y1],
        "footer_source": [x0 + small_w + gap, paragraph_bottom + 8.0, x0 + 2.0 * small_w + gap, y1],
        "footer_note": [x0 + 2.0 * (small_w + gap), paragraph_bottom + 8.0, x1, y1],
    }
    return [float(value) for value in regions.get(str(region), regions["footer_note"])]


def _resolve_native_layout(
    *,
    page_bbox: Sequence[float],
    render_params: _RenderParams,
    native_layout_mode: str,
    text_blocks: Sequence[_InfographicTextBlock],
) -> _NativeLayoutPlan:
    """Reserve header/footer/native text regions before answer-bearing module layout."""

    x0, y0, x1, y1 = [float(value) for value in page_bbox]
    gap = float(render_params.gap_px)
    header_bottom = y0 + float(render_params.header_height_px)
    footer_h = float(render_params.native_text_footer_height_px)
    default_content = [x0, header_bottom + gap, x1, y1 - footer_h]
    default_footer = [x0, default_content[3] + gap * 0.45, x1, y1]
    title_box = [x0 + 22.0, y0 + 12.0, x1 - 22.0, y0 + 55.0]
    subtitle_box = [x0 + 22.0, y0 + 58.0, x1 - 22.0, y0 + 88.0]
    block_slots: Dict[str, List[float]] = {}
    block_regions: Dict[str, str] = {}
    hero_slot: List[float] | None = None
    hero_text_box: List[float] | None = None
    hero_block_id: str | None = None

    paragraphs = [block for block in text_blocks if str(block.kind) == "paragraph_note"]
    small_blocks = [block for block in text_blocks if str(block.kind) != "paragraph_note"]

    def _assign_block(block: _InfographicTextBlock | None, slot: Sequence[float], region: str) -> None:
        if block is None:
            return
        block_slots[str(block.block_id)] = [float(value) for value in slot]
        block_regions[str(block.block_id)] = str(region)

    def _assign_footer_blocks(footer_bbox: Sequence[float], *, paragraph_blocks: Sequence[_InfographicTextBlock], note_blocks: Sequence[_InfographicTextBlock]) -> None:
        footer_regions = ("footer_badge", "footer_source", "footer_note")
        for index, block in enumerate(paragraph_blocks[:2]):
            region = "paragraph_left" if int(index) == 0 else "paragraph_right"
            _assign_block(block, _footer_text_slot(footer_bbox, region), region)
        for index, block in enumerate(note_blocks):
            region = footer_regions[int(index) % len(footer_regions)]
            _assign_block(block, _footer_text_slot(footer_bbox, region), region)

    content_bbox = list(default_content)
    footer_bbox = list(default_footer)
    mode = str(native_layout_mode)
    note_blocks = list(small_blocks)

    if mode == "top_right_callout":
        callout = [x1 - 318.0, y0 + 16.0, x1 - 24.0, y0 + 86.0]
        title_box[2] = max(title_box[0] + 360.0, callout[0] - 18.0)
        subtitle_box[2] = title_box[2]
        hero_block = note_blocks.pop(0) if note_blocks else None
        _assign_block(hero_block, callout, "top_right_callout")
        if hero_block is not None:
            hero_block_id = str(hero_block.block_id)
            hero_slot = [callout[0] + 9.0, callout[1] + 8.0, callout[0] + 64.0, callout[3] - 8.0]
            hero_text_box = [callout[0] + 72.0, callout[1] + 5.0, callout[2] - 10.0, callout[3] - 5.0]
        _assign_footer_blocks(footer_bbox, paragraph_blocks=paragraphs, note_blocks=note_blocks)
    elif mode == "top_left_callout":
        callout = [x0 + 24.0, y0 + 16.0, x0 + 318.0, y0 + 86.0]
        title_box[0] = min(title_box[2] - 360.0, callout[2] + 18.0)
        subtitle_box[0] = title_box[0]
        hero_block = note_blocks.pop(0) if note_blocks else None
        _assign_block(hero_block, callout, "top_left_callout")
        if hero_block is not None:
            hero_block_id = str(hero_block.block_id)
            hero_slot = [callout[0] + 9.0, callout[1] + 8.0, callout[0] + 64.0, callout[3] - 8.0]
            hero_text_box = [callout[0] + 72.0, callout[1] + 5.0, callout[2] - 10.0, callout[3] - 5.0]
        _assign_footer_blocks(footer_bbox, paragraph_blocks=paragraphs, note_blocks=note_blocks)
    elif mode in {"left_side_rail", "right_side_rail"}:
        rail_w = min(212.0, max(176.0, (x1 - x0) * 0.20))
        rail_left = x0 + 16.0 if mode == "left_side_rail" else x1 - rail_w - 16.0
        rail_right = rail_left + rail_w
        rail_top = header_bottom + gap
        rail_bottom = y1 - 16.0
        if mode == "left_side_rail":
            content_bbox = [rail_right + gap, rail_top, x1, y1 - footer_h]
        else:
            content_bbox = [x0, rail_top, rail_left - gap, y1 - footer_h]
        footer_bbox = [content_bbox[0], content_bbox[3] + gap * 0.45, content_bbox[2], y1]
        rail_inner = [rail_left, rail_top, rail_right, rail_bottom]
        hero_block = note_blocks.pop(0) if note_blocks else None
        hero_slot_box = [rail_inner[0], rail_inner[1], rail_inner[2], rail_inner[1] + 92.0]
        _assign_block(hero_block, hero_slot_box, f"{mode}_hero_note")
        if hero_block is not None:
            hero_block_id = str(hero_block.block_id)
            hero_slot = [hero_slot_box[0] + 10.0, hero_slot_box[1] + 10.0, hero_slot_box[0] + 58.0, hero_slot_box[1] + 58.0]
            hero_text_box = [hero_slot_box[0] + 12.0, hero_slot_box[1] + 60.0, hero_slot_box[2] - 12.0, hero_slot_box[3] - 8.0]
        paragraph_top = hero_slot_box[3] + gap
        available_h = max(250.0, rail_inner[3] - paragraph_top - gap)
        paragraph_h = min(270.0, (available_h - gap) / 2.0)
        for index, block in enumerate(paragraphs[:2]):
            py0 = paragraph_top + float(index) * (paragraph_h + gap)
            _assign_block(block, [rail_inner[0], py0, rail_inner[2], py0 + paragraph_h], f"{mode}_paragraph_{index + 1}")
        _assign_footer_blocks(footer_bbox, paragraph_blocks=(), note_blocks=note_blocks)
    elif mode == "poster_anchor_strip":
        strip = [x0 + 20.0, header_bottom + gap * 0.35, x1 - 20.0, header_bottom + gap * 0.35 + 108.0]
        content_bbox = [x0, strip[3] + gap, x1, y1 - footer_h]
        footer_bbox = [x0, content_bbox[3] + gap * 0.45, x1, y1]
        hero_block = note_blocks.pop(0) if note_blocks else None
        _assign_block(hero_block, strip, "poster_anchor_strip")
        if hero_block is not None:
            hero_block_id = str(hero_block.block_id)
            hero_slot = [strip[0] + 16.0, strip[1] + 12.0, strip[0] + 102.0, strip[3] - 12.0]
            hero_text_box = [strip[0] + 122.0, strip[1] + 14.0, strip[2] - 18.0, strip[3] - 14.0]
        _assign_footer_blocks(footer_bbox, paragraph_blocks=paragraphs, note_blocks=note_blocks)
    elif mode == "corner_stamp":
        stamp = [x1 - 238.0, y0 + 15.0, x1 - 24.0, y0 + 68.0]
        title_box[2] = max(title_box[0] + 360.0, stamp[0] - 14.0)
        subtitle_box[2] = title_box[2]
        hero_block = note_blocks.pop(0) if note_blocks else None
        _assign_block(hero_block, stamp, "corner_stamp")
        if hero_block is not None:
            hero_block_id = str(hero_block.block_id)
            hero_slot = [stamp[0] + 8.0, stamp[1] + 7.0, stamp[0] + 48.0, stamp[3] - 7.0]
            hero_text_box = [stamp[0] + 56.0, stamp[1] + 5.0, stamp[2] - 8.0, stamp[3] - 5.0]
        _assign_footer_blocks(footer_bbox, paragraph_blocks=paragraphs, note_blocks=note_blocks)
    else:
        mode = "footer_only"
        _assign_footer_blocks(footer_bbox, paragraph_blocks=paragraphs, note_blocks=note_blocks)

    meta = {
        "native_layout_mode": str(mode),
        "title_bbox_px": list(title_box),
        "subtitle_bbox_px": list(subtitle_box),
        "content_bbox_px": list(content_bbox),
        "native_text_footer_bbox_px": list(footer_bbox),
        "native_text_block_slots_px": {str(key): list(value) for key, value in block_slots.items()},
        "native_text_block_regions": dict(block_regions),
        "hero_block_id": hero_block_id,
        "hero_slot_px": list(hero_slot) if hero_slot is not None else None,
        "hero_text_box_px": list(hero_text_box) if hero_text_box is not None else None,
    }
    return _NativeLayoutPlan(
        mode=str(mode),
        title_bbox_px=list(title_box),
        subtitle_bbox_px=list(subtitle_box),
        content_bbox_px=list(content_bbox),
        footer_bbox_px=list(footer_bbox),
        block_slots_px={str(key): list(value) for key, value in block_slots.items()},
        block_regions=dict(block_regions),
        hero_slot_px=list(hero_slot) if hero_slot is not None else None,
        hero_text_box_px=list(hero_text_box) if hero_text_box is not None else None,
        hero_block_id=hero_block_id,
        meta=meta,
    )

__all__ = [
    "NATIVE_LAYOUT_MODES",
    "NATIVE_LAYOUT_MODE_WEIGHTS",
    "_NativeLayoutPlan",
    "_RenderParams",
    "_layout_slots",
    "_resolve_native_layout",
    "_resolve_native_layout_mode",
    "_resolve_native_text_block_count",
    "_resolve_render_params",
]
