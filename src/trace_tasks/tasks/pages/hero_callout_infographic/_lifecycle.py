"""Scene-private rendering and trace assembly for hero-callout infographic tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....core.types import TypedValue
from ....core.seed import spawn_rng
from ....core.scene_config import get_scene_defaults
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from ...shared.deterministic_sampling import resolve_selection_index
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)
from ...shared.fixed_query import select_task_query_id
from ...shared.text_rendering import fit_font_to_box, load_font
from ...shared.visual_style.information_scene import make_information_scene_background
from ..shared.information_style import resolve_pages_information_style
from ..shared.legible_text import darken_surface_for_light_text, draw_required_page_text
from ..shared.page_text_resources import page_text_resource_metadata, sample_page_context_batch, sample_page_label_batch
from ..shared.page_visual_assets import (
    PageVisualAssetSelection,
    page_visual_asset_version,
    render_page_visual_asset_rgba,
    sample_page_visual_asset,
)
from ..shared.sampling import (
    resolve_int_support as resolve_pages_int_support,
    resolve_named_axis as resolve_pages_named_axis,
    resolve_supported_int as resolve_pages_supported_int,
)
from ..shared.visual_defaults import load_pages_noise_defaults


SCENE_ID = "hero_callout_infographic"
NAMESPACE_ROOT = "pages.hero_callout_infographic"
PROMPT_BUNDLE = "pages_hero_callout_infographic_v1"
PROMPT_SCENE_KEY = "hero_callout_infographic"
PROMPT_TASK_KEY = "hero_callout_infographic_query"
SCENE_VARIANTS: Tuple[str, ...] = (
    "radial_halo",
    "side_rail_poster",
    "diagonal_chain",
    "split_poster",
    "stacked_feature",
)
FIELD_LABELS: Tuple[str, ...] = ("Score", "Count", "Reach", "Rate", "Cost", "Rank")
CONDITION_OPERATORS: Tuple[str, ...] = ("above", "below")

_SCENE_DEFAULTS = get_scene_defaults("pages", SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
)
POST_IMAGE_NOISE_DEFAULTS = load_pages_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)
_ACCENTS: Tuple[Tuple[int, int, int], ...] = (
    (47, 112, 184),
    (34, 144, 116),
    (187, 91, 73),
    (129, 94, 177),
    (198, 139, 48),
    (76, 132, 150),
    (183, 78, 132),
)
_VALUE_BANKS: Dict[str, Tuple[str, ...]] = {
    "Score": tuple(str(value) for value in range(24, 96)),
    "Count": tuple(str(value) for value in range(8, 86)),
    "Reach": tuple(f"{value}k" for value in range(12, 92)),
    "Rate": tuple(f"{value}%" for value in range(14, 89)),
    "Cost": tuple(f"${value}" for value in range(16, 96)),
    "Rank": tuple(f"#{value}" for value in range(1, 80)),
}


@dataclass(frozen=True)
class _RenderParams:
    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    header_height_px: int
    callout_gap_px: int
    corner_radius_px: int
    outline_width_px: int
    title_font_size_px: int
    subtitle_font_size_px: int
    callout_title_font_size_px: int
    field_font_size_px: int
    value_font_size_px: int
    hero_min_size_px: int
    hero_max_size_px: int


@dataclass(frozen=True)
class _FieldValue:
    field_id: str
    label: str
    visible_value: str
    numeric_value: int


@dataclass(frozen=True)
class _Callout:
    callout_id: str
    title: str
    accent_rgb: Tuple[int, int, int]
    fields: Tuple[_FieldValue, ...]
    badge_asset_selection: PageVisualAssetSelection


@dataclass(frozen=True)
class _HeroCalloutSpec:
    title: str
    subtitle: str
    callouts: Tuple[_Callout, ...]
    hero_asset_selection: PageVisualAssetSelection
    section_asset_selection: PageVisualAssetSelection
    text_resource_metadata: Dict[str, Any]


@dataclass(frozen=True)
class _RenderedHeroCallout:
    image: Image.Image
    entities: List[Dict[str, Any]]
    page_bbox_px: List[float]
    hero_bbox_px: List[float]
    callout_bboxes_px: Dict[str, List[float]]
    callout_title_bboxes_px: Dict[str, List[float]]
    field_row_bboxes_px: Dict[str, Dict[str, List[float]]]
    field_label_bboxes_px: Dict[str, Dict[str, List[float]]]
    value_cell_bboxes_px: Dict[str, Dict[str, List[float]]]
    badge_bboxes_px: Dict[str, List[float]]
    decorative_asset_bboxes_px: Dict[str, List[float]]
    layout_meta: Dict[str, Any]


@dataclass(frozen=True)
class _HeroSceneContext:
    selected_branch: str
    gen_defaults: Dict[str, Any]
    render_defaults: Dict[str, Any]
    prompt_defaults: Dict[str, Any]
    branch_probabilities: Dict[str, float]
    scene_variant: str
    scene_variant_probabilities: Dict[str, float]
    callout_count: int
    callout_count_support: Tuple[int, ...]
    callout_count_probabilities: Dict[str, float]
    field_count_support: Tuple[int, ...]
    spec: _HeroCalloutSpec
    rendered: _RenderedHeroCallout
    image: Image.Image
    render_params: _RenderParams
    background_meta: Dict[str, Any]
    style_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


def _resolve_named_variant(
    *,
    namespace_root: str,
    gen_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    supported: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    return resolve_pages_named_axis(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace_root=namespace_root,
        supported=supported,
        explicit_key=explicit_key,
        weights_key=weights_key,
        balance_flag_key=balance_flag_key,
        namespace=namespace,
    )


def _resolve_int_support(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
) -> Tuple[int, ...]:
    return resolve_pages_int_support(params=params, gen_defaults=gen_defaults, key=key, fallback=fallback)


def _resolve_supported_int(
    *,
    namespace_root: str,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    explicit_key: str,
    support_key: str,
    fallback: Sequence[int],
    instance_seed: int,
    namespace: str,
) -> Tuple[int, Tuple[int, ...], Dict[str, float]]:
    return resolve_pages_supported_int(
        params=params,
        gen_defaults=gen_defaults,
        namespace_root=namespace_root,
        explicit_key=explicit_key,
        support_key=support_key,
        fallback=fallback,
        instance_seed=int(instance_seed),
        namespace=namespace,
    )


def _resolve_render_params(params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> _RenderParams:
    def _int(name: str, fallback: int) -> int:
        return int(params.get(name, render_defaults.get(name, fallback)))

    return _RenderParams(
        canvas_width=_int("canvas_width", 1040),
        canvas_height=_int("canvas_height", 1320),
        outer_margin_px=_int("outer_margin_px", 36),
        header_height_px=_int("header_height_px", 104),
        callout_gap_px=_int("callout_gap_px", 14),
        corner_radius_px=_int("corner_radius_px", 14),
        outline_width_px=_int("outline_width_px", 2),
        title_font_size_px=_int("title_font_size_px", 32),
        subtitle_font_size_px=_int("subtitle_font_size_px", 17),
        callout_title_font_size_px=_int("callout_title_font_size_px", 17),
        field_font_size_px=_int("field_font_size_px", 12),
        value_font_size_px=_int("value_font_size_px", 15),
        hero_min_size_px=_int("hero_min_size_px", 260),
        hero_max_size_px=_int("hero_max_size_px", 380),
    )


def _parse_numeric_value(value: str) -> int:
    digits = "".join(char for char in str(value) if char.isdigit())
    if not digits:
        raise ValueError(f"hero callout value is not numeric: {value!r}")
    return int(digits)


def _unique_value(
    *,
    field_label: str,
    values: Sequence[str],
    used_values: set[str],
    fallback_counter: int,
) -> str:
    for raw_value in values:
        value = str(raw_value)
        key = f"{field_label}:{value}"
        if key not in used_values:
            used_values.add(key)
            return value
    while True:
        fallback = str(100 + int(fallback_counter))
        key = f"{field_label}:{fallback}"
        if key not in used_values:
            used_values.add(key)
            return fallback


def _build_spec(
    *,
    callout_count: int,
    field_count_support: Sequence[int],
    required_field_labels: Sequence[str] = (),
    instance_seed: int,
) -> _HeroCalloutSpec:
    """Build deterministic callout data while keeping field values unique per label."""

    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.spec")
    title_batch = sample_page_context_batch(
        rng,
        role="hero_callout_title",
        count=1,
        manifest_names=("phrases/headlines.txt",),
    )
    subtitle_batch = sample_page_context_batch(
        rng,
        role="hero_callout_subtitle",
        count=1,
        manifest_names=("phrases/captions.txt", "phrases/legend_notes.txt"),
    )
    callout_title_batch = sample_page_label_batch(
        rng,
        role="hero_callout_card_title",
        count=int(callout_count),
        manifest_name="categories/product_labels.txt",
        min_chars=3,
        max_chars=16,
        allow_spaces=True,
        allow_punctuation=False,
    )
    text_resource_meta = page_text_resource_metadata(title_batch, subtitle_batch, callout_title_batch)
    asset_rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.visual_assets")
    hero_asset_selection = sample_page_visual_asset(asset_rng, role="hero_anchor")
    section_asset_selection = sample_page_visual_asset(asset_rng, role="section_illustration")
    used_values: set[str] = set()
    fallback_counter = 1
    callouts: List[_Callout] = []
    required_labels = [
        str(label)
        for label in required_field_labels
        if str(label) in set(FIELD_LABELS)
    ]
    for callout_index in range(int(callout_count)):
        field_count = int(field_count_support[int(rng.randrange(len(field_count_support)))])
        if len(required_labels) > field_count:
            field_count = len(required_labels)
        field_labels = list(FIELD_LABELS)
        rng.shuffle(field_labels)
        if required_labels:
            remaining_labels = [label for label in field_labels if str(label) not in set(required_labels)]
            selected_labels = list(required_labels) + remaining_labels[: max(0, field_count - len(required_labels))]
            rng.shuffle(selected_labels)
        else:
            if "Score" not in field_labels[:field_count]:
                field_labels[0] = "Score"
            selected_labels = field_labels[:field_count]
        fields: List[_FieldValue] = []
        for field_index, field_label in enumerate(selected_labels):
            values = list(_VALUE_BANKS[str(field_label)])
            offset = (int(callout_index) * 7 + int(field_index) * 13 + int(rng.randrange(len(values)))) % len(values)
            ordered_values = values[offset:] + values[:offset]
            visible_value = _unique_value(
                field_label=str(field_label),
                values=ordered_values,
                used_values=used_values,
                fallback_counter=int(fallback_counter),
            )
            fallback_counter += 1
            fields.append(
                _FieldValue(
                    field_id=f"field_{field_index + 1}",
                    label=str(field_label),
                    visible_value=str(visible_value),
                    numeric_value=int(_parse_numeric_value(str(visible_value))),
                )
            )
        callouts.append(
            _Callout(
                callout_id=f"callout_{callout_index + 1}",
                title=str(callout_title_batch.values[int(callout_index)]),
                accent_rgb=_ACCENTS[int(callout_index) % len(_ACCENTS)],
                fields=tuple(fields),
                badge_asset_selection=sample_page_visual_asset(asset_rng, role="badge_spot", render_modes=("monochrome",)),
            )
        )
    return _HeroCalloutSpec(
        title=str(title_batch.values[0]),
        subtitle=str(subtitle_batch.values[0]),
        callouts=tuple(callouts),
        hero_asset_selection=hero_asset_selection,
        section_asset_selection=section_asset_selection,
        text_resource_metadata=text_resource_meta,
    )


def _lerp_color(a: Sequence[int], b: Sequence[int], weight: float) -> Tuple[int, int, int]:
    return tuple(int(round(float(a[index]) * (1.0 - float(weight)) + float(b[index]) * float(weight))) for index in range(3))


def _paste_rgba(base: Image.Image, overlay: Image.Image, xy: Tuple[int, int]) -> None:
    base.paste(overlay, (int(xy[0]), int(xy[1])), overlay)


def _asset_bbox_for_center(*, center: Tuple[float, float], size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    width, height = int(size[0]), int(size[1])
    x0 = int(round(float(center[0]) - (float(width) / 2.0)))
    y0 = int(round(float(center[1]) - (float(height) / 2.0)))
    return x0, y0, x0 + width, y0 + height


def _callout_slots(
    *,
    variant: str,
    count: int,
    content_box: Tuple[float, float, float, float],
    gap: int,
) -> Tuple[List[Tuple[float, float, float, float]], Tuple[float, float], Tuple[int, int]]:
    """Resolve variant-specific card geometry around one central hero asset."""

    left, top, right, bottom = [float(value) for value in content_box]
    width = float(right - left)
    height = float(bottom - top)
    if str(variant) == "side_rail_poster":
        hero_center = (left + width * 0.34, top + height * 0.49)
        hero_size = (int(width * 0.32), int(height * 0.34))
        x0 = left + width * 0.56
        slot_h = (height - float(gap) * float(count - 1)) / float(count)
        slots = [(x0, top + i * (slot_h + gap), right, top + i * (slot_h + gap) + slot_h) for i in range(count)]
    elif str(variant) == "diagonal_chain":
        hero_center = (left + width * 0.80, top + height * 0.22)
        hero_size = (int(width * 0.30), int(height * 0.24))
        slot_w = width * 0.35
        slot_h = min(150.0, max(118.0, (height - gap * (count - 1)) / float(max(6, count + 1))))
        slots = []
        for i in range(count):
            x0 = min(right - slot_w, left + width * 0.05 + float(i) * width * 0.105)
            y0 = min(bottom - slot_h, top + 8.0 + float(i) * (slot_h * 0.88 + gap))
            slots.append((x0, y0, x0 + slot_w, y0 + slot_h))
    elif str(variant) == "split_poster":
        hero_center = (left + width * 0.50, top + height * 0.50)
        hero_size = (int(width * 0.28), int(height * 0.34))
        side_w = width * 0.32
        left_count = int(math.ceil(float(count) / 2.0))
        right_count = int(count - left_count)
        slots = []
        for side, side_count in (("left", left_count), ("right", right_count)):
            slot_h = (height - gap * max(0, side_count - 1)) / float(max(1, side_count))
            x0 = left if side == "left" else right - side_w
            for i in range(side_count):
                y0 = top + i * (slot_h + gap)
                slots.append((x0, y0, x0 + side_w, y0 + slot_h))
    elif str(variant) == "stacked_feature":
        hero_center = (left + width * 0.50, top + height * 0.27)
        hero_size = (int(width * 0.33), int(height * 0.24))
        grid_top = top + height * 0.46
        cols = 2
        rows = int(math.ceil(float(count) / float(cols)))
        slot_w = (width - gap) / 2.0
        slot_h = (bottom - grid_top - gap * (rows - 1)) / float(rows)
        slots = []
        for i in range(count):
            row = i // cols
            col = i % cols
            x0 = left + col * (slot_w + gap)
            y0 = grid_top + row * (slot_h + gap)
            slots.append((x0, y0, x0 + slot_w, y0 + slot_h))
    elif str(variant) == "radial_halo":
        hero_center = (left + width * 0.50, top + height * 0.48)
        hero_size = (int(width * 0.30), int(width * 0.30))
        slot_w = min(286.0, max(252.0, width * 0.30))
        slot_h = min(142.0, max(124.0, height * 0.12))
        edge_pad = max(18.0, float(gap))
        left_x = left + edge_pad
        right_x = right - slot_w - edge_pad
        center_x = hero_center[0] - slot_w / 2.0
        top_y = top + edge_pad
        upper_y = top + height * 0.23
        middle_y = hero_center[1] - slot_h / 2.0
        lower_y = bottom - height * 0.23 - slot_h
        bottom_y = bottom - slot_h - edge_pad
        slot_by_name = {
            "top": (center_x, top_y),
            "right_upper": (right_x, upper_y),
            "right_middle": (right_x, middle_y),
            "right_lower": (right_x, lower_y),
            "bottom": (center_x, bottom_y),
            "left_lower": (left_x, lower_y),
            "left_middle": (left_x, middle_y),
            "left_upper": (left_x, upper_y),
        }
        if int(count) <= 5:
            order = ("top", "right_middle", "bottom", "left_lower", "left_upper")
        elif int(count) == 6:
            order = ("top", "right_upper", "right_lower", "bottom", "left_lower", "left_upper")
        else:
            order = ("top", "right_upper", "right_middle", "right_lower", "bottom", "left_lower", "left_upper")
        slots = [
            (slot_by_name[name][0], slot_by_name[name][1], slot_by_name[name][0] + slot_w, slot_by_name[name][1] + slot_h)
            for name in order[:count]
        ]
    else:
        raise ValueError(f"unsupported hero callout scene variant: {variant}")
    return slots[:count], hero_center, hero_size


def _draw_callout_card(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    callout: _Callout,
    bbox: Tuple[float, float, float, float],
    style: Any,
    render_params: _RenderParams,
    instance_seed: int,
) -> Tuple[List[float], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], List[float]]:
    """Draw one callout and return text boxes plus larger field-row boxes."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    radius = int(render_params.corner_radius_px)
    fill = _lerp_color(style.panel_fill_rgb, callout.accent_rgb, 0.08)
    border = tuple(int(value) for value in style.panel_border_rgb)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill, outline=border, width=int(render_params.outline_width_px))
    accent = darken_surface_for_light_text(callout.accent_rgb)
    strip_h = min(34.0, max(24.0, (y1 - y0) * 0.22))
    draw.rounded_rectangle([x0, y0, x1, y0 + strip_h], radius=radius, fill=accent)
    draw.rectangle([x0, y0 + strip_h - radius, x1, y0 + strip_h], fill=accent)
    title_font = fit_font_to_box(
        draw,
        text=str(callout.title),
        max_width=x1 - x0 - 58,
        max_height=strip_h - 8,
        bold=True,
        max_size_px=int(render_params.callout_title_font_size_px),
    )
    title_bbox = draw_required_page_text(
        draw,
        (x0 + 12, y0 + 6),
        str(callout.title),
        title_font,
        role="hero_callout_title",
        surface_rgbs=[accent],
        preferred_rgbs=[(255, 255, 255)],
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.{callout.callout_id}.title",
        stroke_width=0,
    )
    badge_size = int(min(30, max(22, strip_h - 8)))
    badge = render_page_visual_asset_rgba(
        callout.badge_asset_selection.asset,
        size_px=badge_size,
        tint_rgb=tuple(int(value) for value in style.accent_rgb),
    )
    badge_x = int(round(x1 - badge.width - 10))
    badge_y = int(round(y0 + (strip_h - badge.height) / 2.0))
    _paste_rgba(image, badge, (badge_x, badge_y))
    badge_bbox = [float(badge_x), float(badge_y), float(badge_x + badge.width), float(badge_y + badge.height)]
    row_boxes: Dict[str, List[float]] = {}
    field_boxes: Dict[str, List[float]] = {}
    value_boxes: Dict[str, List[float]] = {}
    content_top = y0 + strip_h + 8
    row_h = max(25.0, (y1 - content_top - 10) / float(max(1, len(callout.fields))))
    label_font = load_font(int(render_params.field_font_size_px), bold=True)
    value_font = load_font(int(render_params.value_font_size_px), bold=True)
    for row_index, field in enumerate(callout.fields):
        row_y0 = content_top + row_index * row_h
        row_y1 = min(y1 - 8, row_y0 + row_h - 4)
        row_box = [x0 + 10, row_y0 - 1, x1 - 10, row_y1 + 1]
        label_box = [x0 + 12, row_y0, x0 + (x1 - x0) * 0.55, row_y1]
        value_box = [x0 + (x1 - x0) * 0.59, row_y0, x1 - 12, row_y1]
        row_boxes[str(field.field_id)] = [float(value) for value in row_box]
        draw.rounded_rectangle(value_box, radius=6, fill=tuple(int(value) for value in style.surface_rgb), outline=border, width=1)
        fitted_label = fit_font_to_box(
            draw,
            text=str(field.label),
            max_width=label_box[2] - label_box[0],
            max_height=label_box[3] - label_box[1],
            bold=True,
            max_size_px=int(getattr(label_font, "size", render_params.field_font_size_px)),
        )
        fitted_value = fit_font_to_box(
            draw,
            text=str(field.visible_value),
            max_width=value_box[2] - value_box[0] - 8,
            max_height=value_box[3] - value_box[1] - 4,
            bold=True,
            max_size_px=int(getattr(value_font, "size", render_params.value_font_size_px)),
        )
        field_boxes[str(field.field_id)] = draw_required_page_text(
            draw,
            (label_box[0], label_box[1] + 3),
            str(field.label),
            fitted_label,
            role="hero_callout_field_label",
            surface_rgbs=[fill],
            preferred_rgbs=[style.text_rgb],
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}.{callout.callout_id}.{field.field_id}.label",
            stroke_width=0,
        )
        value_boxes[str(field.field_id)] = draw_required_page_text(
            draw,
            (value_box[0] + 5, value_box[1] + 2),
            str(field.visible_value),
            fitted_value,
            role="hero_callout_value",
            surface_rgbs=[style.surface_rgb],
            preferred_rgbs=[style.text_rgb],
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}.{callout.callout_id}.{field.field_id}.value",
            stroke_width=0,
        )
    return title_bbox, row_boxes, field_boxes, value_boxes, badge_bbox


def _render_scene(
    background: Image.Image,
    *,
    spec: _HeroCalloutSpec,
    scene_variant: str,
    style: Any,
    render_params: _RenderParams,
    instance_seed: int,
) -> _RenderedHeroCallout:
    """Render the full hero-callout page and collect deterministic geometry maps."""

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image)
    w, h = int(render_params.canvas_width), int(render_params.canvas_height)
    margin = int(render_params.outer_margin_px)
    page_bbox = [float(margin), float(margin), float(w - margin), float(h - margin)]
    draw.rounded_rectangle(page_bbox, radius=20, fill=tuple(style.surface_rgb), outline=tuple(style.panel_border_rgb), width=2)
    title_font = load_font(int(render_params.title_font_size_px), bold=True)
    subtitle_font = load_font(int(render_params.subtitle_font_size_px), bold=False)
    title_bbox = draw_required_page_text(
        draw,
        (margin + 24, margin + 20),
        str(spec.title),
        title_font,
        role="hero_callout_page_title",
        surface_rgbs=[style.surface_rgb],
        preferred_rgbs=[style.text_rgb],
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.page_title",
        stroke_width=0,
    )
    draw_required_page_text(
        draw,
        (margin + 24, margin + 58),
        str(spec.subtitle),
        subtitle_font,
        role="hero_callout_page_subtitle",
        surface_rgbs=[style.surface_rgb],
        preferred_rgbs=[style.muted_text_rgb],
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.page_subtitle",
        stroke_width=0,
    )
    content_box = (
        float(margin + 28),
        float(margin + int(render_params.header_height_px)),
        float(w - margin - 28),
        float(h - margin - 24),
    )
    slots, hero_center, hero_size = _callout_slots(
        variant=str(scene_variant),
        count=len(spec.callouts),
        content_box=content_box,
        gap=int(render_params.callout_gap_px),
    )
    hero_size = (
        int(max(render_params.hero_min_size_px, min(render_params.hero_max_size_px, hero_size[0]))),
        int(max(render_params.hero_min_size_px, min(render_params.hero_max_size_px, hero_size[1]))),
    )
    hero = render_page_visual_asset_rgba(spec.hero_asset_selection.asset, size_px=hero_size)
    hero_bbox_tuple = _asset_bbox_for_center(center=hero_center, size=hero.size)
    aura = [
        hero_bbox_tuple[0] - 18,
        hero_bbox_tuple[1] - 18,
        hero_bbox_tuple[2] + 18,
        hero_bbox_tuple[3] + 18,
    ]
    draw.rounded_rectangle(aura, radius=30, fill=_lerp_color(style.panel_fill_rgb, style.accent_rgb, 0.12), outline=tuple(style.guide_rgb), width=2)
    for callout, slot in zip(spec.callouts, slots):
        cx = (slot[0] + slot[2]) / 2.0
        cy = (slot[1] + slot[3]) / 2.0
        draw.line([hero_center, (cx, cy)], fill=tuple(style.guide_rgb), width=2)
    _paste_rgba(image, hero, (hero_bbox_tuple[0], hero_bbox_tuple[1]))
    section_asset = render_page_visual_asset_rgba(
        spec.section_asset_selection.asset,
        size_px=(92, 92),
        tint_rgb=tuple(int(value) for value in style.accent_rgb),
    )
    section_x = int(page_bbox[2] - section_asset.width - 30)
    section_y = int(page_bbox[1] + 18)
    _paste_rgba(image, section_asset, (section_x, section_y))
    decorative_asset_bboxes = {
        "hero_anchor": [float(value) for value in hero_bbox_tuple],
        "section_illustration": [
            float(section_x),
            float(section_y),
            float(section_x + section_asset.width),
            float(section_y + section_asset.height),
        ],
    }
    callout_bboxes: Dict[str, List[float]] = {}
    callout_title_bboxes: Dict[str, List[float]] = {}
    field_row_bboxes: Dict[str, Dict[str, List[float]]] = {}
    field_bboxes: Dict[str, Dict[str, List[float]]] = {}
    value_bboxes: Dict[str, Dict[str, List[float]]] = {}
    badge_bboxes: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = []
    for callout, slot in zip(spec.callouts, slots):
        title_box, row_boxes, field_boxes, value_boxes, badge_box = _draw_callout_card(
            image,
            draw,
            callout=callout,
            bbox=slot,
            style=style,
            render_params=render_params,
            instance_seed=int(instance_seed),
        )
        callout_bboxes[str(callout.callout_id)] = [float(value) for value in slot]
        callout_title_bboxes[str(callout.callout_id)] = list(title_box)
        field_row_bboxes[str(callout.callout_id)] = dict(row_boxes)
        field_bboxes[str(callout.callout_id)] = dict(field_boxes)
        value_bboxes[str(callout.callout_id)] = dict(value_boxes)
        badge_bboxes[str(callout.callout_id)] = list(badge_box)
        entities.append(
            {
                "entity_id": str(callout.callout_id),
                "kind": "hero_callout_card",
                "title": str(callout.title),
                "bbox_px": [float(value) for value in slot],
                "title_bbox_px": list(title_box),
                "badge_asset": callout.badge_asset_selection.to_metadata(),
                "badge_bbox_px": list(badge_box),
                "fields": [
                    {
                        "field_id": str(field.field_id),
                        "label": str(field.label),
                        "visible_value": str(field.visible_value),
                        "numeric_value": int(field.numeric_value),
                        "field_row_bbox_px": list(row_boxes[str(field.field_id)]),
                        "field_bbox_px": list(field_boxes[str(field.field_id)]),
                        "value_bbox_px": list(value_boxes[str(field.field_id)]),
                    }
                    for field in callout.fields
                ],
            }
        )
    entities.append(
        {
            "entity_id": "hero_anchor",
            "kind": "hero_callout_anchor_asset",
            "asset": spec.hero_asset_selection.to_metadata(),
            "bbox_px": [float(value) for value in hero_bbox_tuple],
        }
    )
    layout_meta = {
        "scene_variant": str(scene_variant),
        "content_box_px": [float(value) for value in content_box],
        "hero_center_px": [float(hero_center[0]), float(hero_center[1])],
        "hero_size_px": [int(hero.size[0]), int(hero.size[1])],
        "callout_count": len(spec.callouts),
    }
    return _RenderedHeroCallout(
        image=image.convert("RGB"),
        entities=entities,
        page_bbox_px=list(page_bbox),
        hero_bbox_px=[float(value) for value in hero_bbox_tuple],
        callout_bboxes_px=callout_bboxes,
        callout_title_bboxes_px=callout_title_bboxes,
        field_row_bboxes_px=field_row_bboxes,
        field_label_bboxes_px=field_bboxes,
        value_cell_bboxes_px=value_bboxes,
        badge_bboxes_px=badge_bboxes,
        decorative_asset_bboxes_px=decorative_asset_bboxes,
        layout_meta=layout_meta,
    )


def select_public_branch(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    supported: Tuple[str, ...],
    default: str,
    public_task: str,
) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
    """Resolve the public branch with the repo-wide fixed-query policy."""

    branch, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported),
        default_query_id=str(default),
        task_id=str(public_task),
    )
    return str(branch), dict(probabilities), dict(task_params)


def resolve_scene_context(
    *,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    instance_seed: int,
) -> _HeroSceneContext:
    """Sample and render the shared hero-callout scene before task binding."""

    scene_variant, scene_variant_probabilities = _resolve_named_variant(
        namespace_root=NAMESPACE_ROOT,
        gen_defaults=GEN_DEFAULTS,
        params=params,
        instance_seed=int(instance_seed),
        supported=SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace="scene_variant",
    )
    callout_count, callout_count_support, callout_count_probabilities = _resolve_supported_int(
        namespace_root=NAMESPACE_ROOT,
        params=params,
        gen_defaults=GEN_DEFAULTS,
        explicit_key="callout_count",
        support_key="callout_count_support",
        fallback=(5, 6, 7),
        instance_seed=int(instance_seed),
        namespace="callout_count",
    )
    field_count_support = _resolve_int_support(
        params=params,
        gen_defaults=GEN_DEFAULTS,
        key="field_count_support",
        fallback=(2, 3),
    )
    required_field_labels = tuple(str(value) for value in params.get("required_field_labels", ()))
    spec = _build_spec(
        callout_count=int(callout_count),
        field_count_support=field_count_support,
        required_field_labels=required_field_labels,
        instance_seed=int(instance_seed),
    )
    render_params = _resolve_render_params(params, RENDER_DEFAULTS)
    style, style_meta = resolve_pages_information_style(
        instance_seed=int(instance_seed),
        params={**dict(RENDER_DEFAULTS), **dict(params or {})},
        scene_id=SCENE_ID,
    )
    background, background_meta = make_information_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=style,
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.information_scene_background",
    )
    rendered = _render_scene(
        background,
        spec=spec,
        scene_variant=str(scene_variant),
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
    return _HeroSceneContext(
        selected_branch=str(selected_branch),
        gen_defaults=dict(GEN_DEFAULTS),
        render_defaults=dict(RENDER_DEFAULTS),
        prompt_defaults=dict(PROMPT_DEFAULTS),
        branch_probabilities={str(key): float(value) for key, value in branch_probabilities.items()},
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        callout_count=int(callout_count),
        callout_count_support=tuple(int(value) for value in callout_count_support),
        callout_count_probabilities=dict(callout_count_probabilities),
        field_count_support=tuple(int(value) for value in field_count_support),
        spec=spec,
        rendered=rendered,
        image=image,
        render_params=render_params,
        background_meta=dict(background_meta),
        style_meta=dict(style_meta),
        post_noise_meta=dict(post_noise_meta),
    )


def _field_by_id(callout: _Callout, field_id: str) -> _FieldValue:
    for field in callout.fields:
        if str(field.field_id) == str(field_id):
            return field
    raise ValueError("field is not part of callout")


def _field_values_for_label(spec: _HeroCalloutSpec, field_label: str) -> List[Dict[str, Any]]:
    values: List[Dict[str, Any]] = []
    for callout_index, callout in enumerate(spec.callouts):
        matching_fields = [field for field in callout.fields if str(field.label) == str(field_label)]
        if not matching_fields:
            continue
        field = matching_fields[0]
        values.append(
            {
                "callout_index": int(callout_index),
                "callout_id": str(callout.callout_id),
                "callout_title": str(callout.title),
                "field_id": str(field.field_id),
                "field_label": str(field.label),
                "visible_value": str(field.visible_value),
                "numeric_value": int(field.numeric_value),
            }
        )
    return values


def _uniform_probs(values: Sequence[Any], selected: Any, *, locked: bool = False) -> Dict[str, float]:
    keys = sorted({str(value) for value in values})
    if not keys:
        return {}
    if bool(locked):
        return {str(selected): 1.0}
    p = 1.0 / float(len(keys))
    return {str(key): float(p) for key in keys}


def select_lookup_target(
    *,
    spec: _HeroCalloutSpec,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[_Callout, _FieldValue, Dict[str, float], Dict[str, float]]:
    """Select a visible callout-field pair for direct value lookup."""

    requested_callout_index = params.get("target_callout_index")
    requested_field_label = params.get("target_field_label")
    candidates: List[Tuple[int, _Callout, int, _FieldValue]] = []
    for callout_index, callout in enumerate(spec.callouts):
        if requested_callout_index is not None and int(callout_index) != int(requested_callout_index):
            continue
        for field_index, field in enumerate(callout.fields):
            if requested_field_label is not None and str(field.label) != str(requested_field_label):
                continue
            candidates.append((int(callout_index), callout, int(field_index), field))
    if not candidates:
        raise ValueError("no eligible hero callout lookup target")
    selected = candidates[
        int(resolve_selection_index(params=params, instance_seed=int(instance_seed), namespace=f"{NAMESPACE_ROOT}.lookup_target"))
        % int(len(candidates))
    ]
    callout_indices = [candidate[0] for candidate in candidates]
    field_labels = [str(candidate[3].label) for candidate in candidates if candidate[0] == selected[0]]
    return (
        selected[1],
        selected[3],
        _uniform_probs(callout_indices, selected[0], locked=requested_callout_index is not None),
        _uniform_probs(field_labels, str(selected[3].label), locked=requested_field_label is not None),
    )


def select_extremum_target(
    *,
    gen_defaults: Mapping[str, Any],
    spec: _HeroCalloutSpec,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[_Callout, _FieldValue, Dict[str, Any], Dict[str, float]]:
    """Select a field whose visible values have a unique requested extremum."""

    del gen_defaults
    direction = str(params.get("rank_direction", ""))
    if direction not in {"highest", "lowest"}:
        raise ValueError(f"unsupported hero callout rank_direction: {direction}")
    requested_field_label = params.get("target_field_label")
    candidates_by_label: Dict[str, Dict[str, Any]] = {}
    for field_label in sorted(set(FIELD_LABELS)):
        if requested_field_label is not None and str(field_label) != str(requested_field_label):
            continue
        values = _field_values_for_label(spec, str(field_label))
        if len(values) < 3:
            continue
        numeric_values = [int(value["numeric_value"]) for value in values]
        target_value = max(numeric_values) if str(direction) == "highest" else min(numeric_values)
        winners = [dict(value) for value in values if int(value["numeric_value"]) == int(target_value)]
        if len(winners) != 1:
            continue
        candidates_by_label[str(field_label)] = {
            "field_label": str(field_label),
            "winner": dict(winners[0]),
            "candidate_values": [dict(value) for value in values],
        }
    if not candidates_by_label:
        raise ValueError("no unique hero callout extremum target")
    labels = sorted(candidates_by_label)
    selected_label = str(requested_field_label) if requested_field_label is not None else labels[
        int(resolve_selection_index(params=params, instance_seed=int(instance_seed), namespace=f"{NAMESPACE_ROOT}.extremum_field.{direction}"))
        % int(len(labels))
    ]
    selected = dict(candidates_by_label[str(selected_label)])
    winner = dict(selected["winner"])
    callout = spec.callouts[int(winner["callout_index"])]
    field = _field_by_id(callout, str(winner["field_id"]))
    target = {
        "callout_id": str(callout.callout_id),
        "callout_title": str(callout.title),
        "field_id": str(field.field_id),
        "field_label": str(field.label),
        "rank_direction": str(direction),
        "rank_order_phrase": "highest to lowest" if str(direction) == "highest" else "lowest to highest",
        "visible_value": str(field.visible_value),
        "numeric_value": int(field.numeric_value),
        "candidate_values": [dict(value) for value in selected["candidate_values"]],
        "answer_value": str(callout.title),
    }
    return callout, field, target, _uniform_probs(labels, str(selected_label), locked=requested_field_label is not None)


def select_composite_extremum_target(
    *,
    spec: _HeroCalloutSpec,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[_Callout, Dict[str, Any], Dict[str, float]]:
    """Select a unique callout extremum by the sum of two visible field values."""

    direction = str(params.get("rank_direction", ""))
    if direction not in {"highest", "lowest"}:
        raise ValueError(f"unsupported hero callout composite rank_direction: {direction}")
    requested_labels = tuple(str(value) for value in params.get("target_field_labels", ("Score", "Count")))
    if len(requested_labels) != 2:
        raise ValueError("target_field_labels must contain exactly two field labels")
    field_a, field_b = requested_labels
    candidates: List[Dict[str, Any]] = []
    for callout_index, callout in enumerate(spec.callouts):
        by_label = {str(field.label): field for field in callout.fields}
        if field_a not in by_label or field_b not in by_label:
            continue
        first = by_label[field_a]
        second = by_label[field_b]
        composite_value = int(first.numeric_value) + int(second.numeric_value)
        candidates.append(
            {
                "callout_index": int(callout_index),
                "callout_id": str(callout.callout_id),
                "callout_title": str(callout.title),
                "first_field_id": str(first.field_id),
                "first_field_label": str(first.label),
                "first_visible_value": str(first.visible_value),
                "first_numeric_value": int(first.numeric_value),
                "second_field_id": str(second.field_id),
                "second_field_label": str(second.label),
                "second_visible_value": str(second.visible_value),
                "second_numeric_value": int(second.numeric_value),
                "composite_value": int(composite_value),
            }
        )
    if len(candidates) < 3:
        raise ValueError("not enough callouts contain both fields for composite extremum")
    numeric_values = [int(candidate["composite_value"]) for candidate in candidates]
    target_value = max(numeric_values) if direction == "highest" else min(numeric_values)
    winners = [dict(candidate) for candidate in candidates if int(candidate["composite_value"]) == int(target_value)]
    if len(winners) != 1:
        raise ValueError("composite extremum is not unique")
    winner = dict(winners[0])
    callout = spec.callouts[int(winner["callout_index"])]
    target = {
        "callout_id": str(callout.callout_id),
        "callout_title": str(callout.title),
        "first_field_label": str(field_a),
        "second_field_label": str(field_b),
        "rank_direction": str(direction),
        "rank_order_phrase": "highest to lowest" if direction == "highest" else "lowest to highest",
        "candidate_values": [dict(candidate) for candidate in candidates],
        "winner": dict(winner),
        "composite_value": int(winner["composite_value"]),
        "answer_value": str(callout.title),
    }
    pair_key = f"{field_a}+{field_b}"
    return callout, target, {pair_key: 1.0}


def _composite_prompt_slots(target: Mapping[str, Any]) -> Dict[str, str]:
    return {
        "first_field_label": f'"{target["first_field_label"]}"',
        "second_field_label": f'"{target["second_field_label"]}"',
        "rank_direction": str(target["rank_direction"]),
        "rank_order_phrase": str(target["rank_order_phrase"]),
    }


def _composite_extremum_annotation(
    *,
    ctx: _HeroSceneContext,
    callout: _Callout,
    target: Mapping[str, Any],
) -> Dict[str, List[float]]:
    callout_id = str(callout.callout_id)
    annotation: Dict[str, List[float]] = {
        "winning_callout_card": [float(value) for value in ctx.rendered.callout_bboxes_px[callout_id]],
    }
    for index, candidate in enumerate(target["candidate_values"], start=1):
        candidate_callout = str(candidate["callout_id"])
        first_field = str(candidate["first_field_id"])
        second_field = str(candidate["second_field_id"])
        annotation[f"candidate_{index}_first_field_row"] = [
            float(value) for value in ctx.rendered.field_row_bboxes_px[candidate_callout][first_field]
        ]
        annotation[f"candidate_{index}_second_field_row"] = [
            float(value) for value in ctx.rendered.field_row_bboxes_px[candidate_callout][second_field]
        ]
    return annotation


def _composite_rank_audit(target: Mapping[str, Any]) -> List[Dict[str, Any]]:
    reverse_sort = str(target["rank_direction"]) == "highest"
    rows = [dict(candidate) for candidate in target["candidate_values"]]
    rows.sort(key=lambda row: (int(row["composite_value"]), str(row["callout_title"])), reverse=reverse_sort)
    for row in rows:
        row["is_answer"] = str(row["callout_id"]) == str(target["callout_id"])
    return rows


def _condition_phrase(operator: str) -> str:
    if str(operator) == "above":
        return "above"
    if str(operator) == "below":
        return "below"
    raise ValueError(f"unsupported condition operator: {operator}")


def _condition_matches(value: int, *, operator: str, threshold: int) -> bool:
    if str(operator) == "above":
        return int(value) > int(threshold)
    if str(operator) == "below":
        return int(value) < int(threshold)
    raise ValueError(f"unsupported condition operator: {operator}")


def select_condition_target(
    *,
    gen_defaults: Mapping[str, Any],
    spec: _HeroCalloutSpec,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float]]:
    """Select a threshold predicate with a nonempty proper matching subset."""

    del gen_defaults
    operator = str(params.get("condition_operator", ""))
    if operator not in set(CONDITION_OPERATORS):
        raise ValueError(f"unsupported condition operator: {operator}")
    requested_field_label = params.get("target_field_label")
    candidates: List[Dict[str, Any]] = []
    for field_label in sorted(set(FIELD_LABELS)):
        if requested_field_label is not None and str(field_label) != str(requested_field_label):
            continue
        values = _field_values_for_label(spec, str(field_label))
        if len(values) < 4:
            continue
        unique_values = sorted({int(value["numeric_value"]) for value in values})
        thresholds = unique_values[:-1] if str(operator) == "above" else unique_values[1:]
        for threshold_index, threshold in enumerate(thresholds):
            matches = [
                dict(value)
                for value in values
                if _condition_matches(int(value["numeric_value"]), operator=str(operator), threshold=int(threshold))
            ]
            if not matches or len(matches) == len(values):
                continue
            threshold_visible = next(str(value["visible_value"]) for value in values if int(value["numeric_value"]) == int(threshold))
            candidates.append(
                {
                    "field_label": str(field_label),
                    "condition_operator": str(operator),
                    "condition_phrase": str(_condition_phrase(str(operator))),
                    "threshold_rank_index": int(threshold_index),
                    "threshold_value": int(threshold),
                    "threshold_visible": str(threshold_visible),
                    "candidate_values": [dict(value) for value in values],
                    "matching_values": [dict(value) for value in matches],
                    "answer_value": int(len(matches)),
                }
            )
    if not candidates:
        raise ValueError("no useful hero callout condition count target")
    selected = candidates[
        int(resolve_selection_index(params=params, instance_seed=int(instance_seed), namespace=f"{NAMESPACE_ROOT}.condition_target.{operator}"))
        % int(len(candidates))
    ]
    labels = [str(candidate["field_label"]) for candidate in candidates]
    threshold_keys = [
        f'{candidate["field_label"]}:{candidate["threshold_rank_index"]}'
        for candidate in candidates
        if str(candidate["field_label"]) == str(selected["field_label"])
    ]
    selected_threshold_key = f'{selected["field_label"]}:{selected["threshold_rank_index"]}'
    return (
        dict(selected),
        _uniform_probs(labels, str(selected["field_label"]), locked=requested_field_label is not None),
        _uniform_probs(threshold_keys, str(selected_threshold_key)),
    )


def render_prompt(
    *,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> Any:
    """Render the scene prompt from the v1 hero-callout prompt bundle."""

    prompt_selection = render_task_prompt_variants(
        domain="pages",
        scene_id=SCENE_ID,
        bundle_id=PROMPT_BUNDLE,
        scene_key=PROMPT_SCENE_KEY,
        task_key=PROMPT_TASK_KEY,
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


def _projected_annotation(annotation_type: str, annotation_value: Any) -> Dict[str, Any]:
    if str(annotation_type) == "bbox_map":
        keyed = {str(key): [float(value) for value in bbox] for key, bbox in dict(annotation_value).items()}
        return {
            "type": "bbox_map",
            "bbox_map": dict(keyed),
            "pixel_bbox_map": dict(keyed),
            "bbox_set": list(keyed.values()),
        }
    bbox_set = [[float(value) for value in bbox] for bbox in list(annotation_value)]
    return {"type": "bbox_set", "bbox_set": list(bbox_set), "pixel_bbox_set": list(bbox_set)}


def _trace_payload(
    *,
    ctx: _HeroSceneContext,
    prompt_artifacts: Any,
    prompt_query_key: str,
    question_format: str,
    target_payload: Mapping[str, Any],
    answer_value: Any,
    annotation_type: str,
    annotation_value: Any,
    annotation_keys: Sequence[str],
    query_params_extra: Mapping[str, Any] = {},
    execution_extra: Mapping[str, Any] = {},
) -> Dict[str, Any]:
    """Assemble task-neutral trace payload from rendered geometry and target binding."""

    target = dict(target_payload)
    query_params = {
        "query_id": str(ctx.selected_branch),
        "prompt_query_key": str(prompt_query_key),
        "scene_variant": str(ctx.scene_variant),
        "callout_count": int(ctx.callout_count),
        "target": dict(target),
        "target_answer": answer_value,
        "query_id_probabilities": dict(ctx.branch_probabilities),
        "scene_variant_probabilities": dict(ctx.scene_variant_probabilities),
        "callout_count_probabilities": dict(ctx.callout_count_probabilities),
    }
    query_params.update(dict(query_params_extra))
    execution_trace = {
        "query_id": str(ctx.selected_branch),
        "scene_variant": str(ctx.scene_variant),
        "prompt_query_key": str(prompt_query_key),
        "question_format": str(question_format),
        "callout_count": int(ctx.callout_count),
        "callout_count_support": [int(value) for value in ctx.callout_count_support],
        "field_count_support": [int(value) for value in ctx.field_count_support],
        "target": dict(target),
        "answer_value": answer_value,
        "callouts": [
            {
                "callout_id": str(callout.callout_id),
                "title": str(callout.title),
                "fields": [
                    {
                        "field_id": str(field.field_id),
                        "field_label": str(field.label),
                        "visible_value": str(field.visible_value),
                        "numeric_value": int(field.numeric_value),
                    }
                    for field in callout.fields
                ],
            }
            for callout in ctx.spec.callouts
        ],
        "page_text_resources": dict(ctx.spec.text_resource_metadata),
        "query_id_probabilities": dict(ctx.branch_probabilities),
        "scene_variant_probabilities": dict(ctx.scene_variant_probabilities),
        "callout_count_probabilities": dict(ctx.callout_count_probabilities),
    }
    execution_trace.update(dict(execution_extra))
    return {
        "scene_ir": {
            "scene_id": SCENE_ID,
            "scene_kind": "pages_hero_callout_infographic",
            "entities": [dict(entity) for entity in ctx.rendered.entities],
            "relations": {
                "query_id": str(ctx.selected_branch),
                "scene_variant": str(ctx.scene_variant),
                "target": dict(target),
                "answer_value": answer_value,
            },
        },
        "query_spec": {
            "query_id": str(ctx.selected_branch),
            "template_id": PROMPT_BUNDLE,
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": dict(query_params),
            "prompt_query_key": str(prompt_query_key),
        },
        "render_spec": {
            "canvas_width": int(ctx.render_params.canvas_width),
            "canvas_height": int(ctx.render_params.canvas_height),
            "coord_space": "pixel",
            "scene_id": SCENE_ID,
            "scene_variant": str(ctx.scene_variant),
            "background_style": dict(ctx.background_meta),
            "information_scene_style": dict(ctx.style_meta),
            "post_image_noise": dict(ctx.post_noise_meta),
            "page_bbox_px": list(ctx.rendered.page_bbox_px),
            "layout": dict(ctx.rendered.layout_meta),
            "context_text_layer": {"enabled": False},
            "page_text_resources": dict(ctx.spec.text_resource_metadata),
            "page_visual_assets": {
                "asset_version": page_visual_asset_version(),
                "asset_root": "assets/pages/visual_assets",
                "semantic_policy": "non_answer_visual_context",
                "roles": {
                    "hero_anchor": ctx.spec.hero_asset_selection.to_metadata(),
                    "section_illustration": ctx.spec.section_asset_selection.to_metadata(),
                    "callout_badges": {
                        str(callout.callout_id): callout.badge_asset_selection.to_metadata()
                        for callout in ctx.spec.callouts
                    },
                },
            },
        },
        "render_map": {
            "image_id": "img0",
            "page_bbox_px": list(ctx.rendered.page_bbox_px),
            "hero_bbox_px": list(ctx.rendered.hero_bbox_px),
            "callout_bboxes_px": dict(ctx.rendered.callout_bboxes_px),
            "callout_title_bboxes_px": dict(ctx.rendered.callout_title_bboxes_px),
            "field_row_bboxes_px": dict(ctx.rendered.field_row_bboxes_px),
            "field_label_bboxes_px": dict(ctx.rendered.field_label_bboxes_px),
            "value_cell_bboxes_px": dict(ctx.rendered.value_cell_bboxes_px),
            "badge_bboxes_px": dict(ctx.rendered.badge_bboxes_px),
            "visual_asset_bboxes_px": {
                "decorative_assets": dict(ctx.rendered.decorative_asset_bboxes_px),
                "callout_badges": dict(ctx.rendered.badge_bboxes_px),
            },
        },
        "execution_trace": dict(execution_trace),
        "witness_symbolic": {
            "type": str(question_format),
            "target": dict(target),
            "answer_value": answer_value,
            "annotation_keys": [str(key) for key in annotation_keys],
        },
        "projected_annotation": _projected_annotation(str(annotation_type), annotation_value),
    }


def build_string_bbox_map_output(
    *,
    ctx: _HeroSceneContext,
    prompt_artifacts: Any,
    answer_value: str,
    annotation: Mapping[str, Sequence[float]],
    trace_payload: Mapping[str, Any],
) -> TaskOutput:
    """Return the shared string-answer output shape for hero-callout label tasks."""

    annotation_value = {
        str(key): [float(value) for value in bbox]
        for key, bbox in dict(annotation).items()
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=TypedValue(type="string", value=str(answer_value)),
        annotation_gt=TypedValue(type="bbox_map", value=dict(annotation_value)),
        image=ctx.image,
        image_id="img0",
        trace_payload=dict(trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(ctx.selected_branch),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = [
    "CONDITION_OPERATORS",
    "FIELD_LABELS",
    "SCENE_ID",
    "SCENE_VARIANTS",
    "select_public_branch",
    "resolve_scene_context",
    "select_lookup_target",
    "select_extremum_target",
    "select_composite_extremum_target",
    "select_condition_target",
    "render_prompt",
    "_trace_payload",
    "build_string_bbox_map_output",
]
