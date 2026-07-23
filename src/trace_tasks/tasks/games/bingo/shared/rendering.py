"""Shared Bingo card rendering for games-domain Bingo tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.games.shared.layout import (
    apply_games_layout_jitter_to_bbox,
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from trace_tasks.tasks.games.shared.scene_style import (
    GamePanelSceneStyle,
    draw_panel_scene_chrome,
    make_panel_scene_background,
    resolve_game_panel_scene_style,
    game_panel_scene_style_metadata,
)
from trace_tasks.tasks.games.shared.style import BingoTheme, build_games_bingo_theme
from trace_tasks.tasks.games.shared.text import draw_game_text_traced as draw_text_traced
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_rendering import fit_font_to_box, resolve_text_stroke_fill

from .defaults import RENDER_FALLBACKS, SCENE_ID
from .state import BINGO_BOARD_SIZE, BINGO_COLUMN_LABELS, BingoCellInstance


SUPPORTED_BINGO_MARK_SHAPES: Tuple[str, ...] = ("ellipse", "cell", "ring")
SUPPORTED_BINGO_CELL_FILL_PATTERNS: Tuple[str, ...] = ("solid", "column_tint", "checker_tint")
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)
_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)


@dataclass(frozen=True)
class BingoRenderParams:
    """Resolved render controls for one bingo-card scene."""

    canvas_width: int
    canvas_height: int
    card_width_px: int
    card_height_px: int
    card_corner_radius_px: int
    panel_margin_px: int
    title_font_size_px: int
    title_band_height_px: int
    header_font_size_px: int
    header_height_px: int
    grid_gap_px: int
    number_font_size_px: int
    cell_corner_radius_px: int
    cell_gap_px: int
    mark_inset_px: int
    mark_shape: str = "ellipse"
    cell_fill_pattern: str = "solid"
    called_panel_width_px: int = 220
    called_panel_gap_px: int = 32
    called_panel_title_font_size_px: int = 26
    called_panel_number_font_size_px: int = 25
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None
    instance_seed: int = 0


@dataclass(frozen=True)
class RenderedBingoCellSpec:
    """One visible bingo cell after layout/render assignment."""

    cell_id: str
    row_index: int
    column_index: int
    column_label: str
    number: int
    is_marked: bool
    bbox_px: Tuple[float, float, float, float]


@dataclass(frozen=True)
class RenderedBingoCardScene:
    """Rendered Bingo scene plus trace-friendly cell metadata."""

    image: Image.Image
    cell_specs: Tuple[RenderedBingoCellSpec, ...]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class RenderedBingoTaskContext:
    """Rendered task context after style, layout, and noise are applied."""

    image: Image.Image
    rendered_scene: RenderedBingoCardScene
    render_params: BingoRenderParams
    panel_style_meta: Dict[str, Any]
    text_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


def resolve_bingo_render_params(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    mark_shape: str,
    cell_fill_pattern: str,
    show_called_panel: bool = False,
) -> BingoRenderParams:
    """Resolve stable render parameters for one Bingo scene."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="games.bingo.text_font",
        params=params,
    )
    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        _RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace="games.bingo.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            _RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace="games.bingo.layout",
        ),
        unit_scale_meta,
    )
    card_width_px = scale_games_px(
        params.get("card_width_px", group_default(_RENDER_DEFAULTS, "card_width_px", RENDER_FALLBACKS.card_width_px)),
        unit_scale,
        min_px=380,
    )
    card_height_px = scale_games_px(
        params.get("card_height_px", group_default(_RENDER_DEFAULTS, "card_height_px", RENDER_FALLBACKS.card_height_px)),
        unit_scale,
        min_px=310,
    )
    called_panel_width_px = scale_games_px(
        params.get("called_panel_width_px", group_default(_RENDER_DEFAULTS, "called_panel_width_px", RENDER_FALLBACKS.called_panel_width_px)),
        unit_scale,
        min_px=120,
    )
    called_panel_gap_px = scale_games_px(
        params.get("called_panel_gap_px", group_default(_RENDER_DEFAULTS, "called_panel_gap_px", RENDER_FALLBACKS.called_panel_gap_px)),
        unit_scale,
        min_px=16,
    )
    scene_footprint_width_px = int(card_width_px)
    if bool(show_called_panel):
        scene_footprint_width_px += int(called_panel_gap_px) + int(called_panel_width_px)
    dynamic_canvas_enabled = bool(
        params.get("dynamic_canvas_size_enabled", group_default(_RENDER_DEFAULTS, "dynamic_canvas_size_enabled", RENDER_FALLBACKS.dynamic_canvas_size_enabled))
    )
    base_canvas_width = int(params.get("canvas_width", group_default(_RENDER_DEFAULTS, "canvas_width", RENDER_FALLBACKS.canvas_width)))
    base_canvas_height = int(params.get("canvas_height", group_default(_RENDER_DEFAULTS, "canvas_height", RENDER_FALLBACKS.canvas_height)))
    canvas_width = base_canvas_width
    canvas_height = base_canvas_height
    if dynamic_canvas_enabled and params.get("canvas_width") is None:
        canvas_width = min(
            int(base_canvas_width),
            max(
                int(params.get("canvas_min_width_px", group_default(_RENDER_DEFAULTS, "canvas_min_width_px", RENDER_FALLBACKS.canvas_min_width_px))),
                int(
                    round(
                        float(scene_footprint_width_px)
                        + (2.0 * float(params.get("canvas_side_padding_px", group_default(_RENDER_DEFAULTS, "canvas_side_padding_px", RENDER_FALLBACKS.canvas_side_padding_px))))
                    )
                ),
            ),
        )
    if dynamic_canvas_enabled and params.get("canvas_height") is None:
        canvas_height = min(
            int(base_canvas_height),
            max(
                int(params.get("canvas_min_height_px", group_default(_RENDER_DEFAULTS, "canvas_min_height_px", RENDER_FALLBACKS.canvas_min_height_px))),
                int(
                    round(
                        float(card_height_px)
                        + (2.0 * float(params.get("canvas_vertical_padding_px", group_default(_RENDER_DEFAULTS, "canvas_vertical_padding_px", RENDER_FALLBACKS.canvas_vertical_padding_px))))
                    )
                ),
            ),
        )
    return BingoRenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        card_width_px=int(card_width_px),
        card_height_px=int(card_height_px),
        card_corner_radius_px=scale_games_px(params.get("card_corner_radius_px", group_default(_RENDER_DEFAULTS, "card_corner_radius_px", RENDER_FALLBACKS.card_corner_radius_px)), unit_scale, min_px=10),
        panel_margin_px=scale_games_px(params.get("panel_margin_px", group_default(_RENDER_DEFAULTS, "panel_margin_px", RENDER_FALLBACKS.panel_margin_px)), unit_scale, min_px=28),
        title_font_size_px=scale_games_px(params.get("title_font_size_px", group_default(_RENDER_DEFAULTS, "title_font_size_px", RENDER_FALLBACKS.title_font_size_px)), unit_scale, min_px=18),
        title_band_height_px=scale_games_px(params.get("title_band_height_px", group_default(_RENDER_DEFAULTS, "title_band_height_px", RENDER_FALLBACKS.title_band_height_px)), unit_scale, min_px=34),
        header_font_size_px=scale_games_px(params.get("header_font_size_px", group_default(_RENDER_DEFAULTS, "header_font_size_px", RENDER_FALLBACKS.header_font_size_px)), unit_scale, min_px=15),
        header_height_px=scale_games_px(params.get("header_height_px", group_default(_RENDER_DEFAULTS, "header_height_px", RENDER_FALLBACKS.header_height_px)), unit_scale, min_px=24),
        grid_gap_px=scale_games_px(params.get("grid_gap_px", group_default(_RENDER_DEFAULTS, "grid_gap_px", RENDER_FALLBACKS.grid_gap_px)), unit_scale, min_px=8),
        number_font_size_px=scale_games_px(params.get("number_font_size_px", group_default(_RENDER_DEFAULTS, "number_font_size_px", RENDER_FALLBACKS.number_font_size_px)), unit_scale, min_px=15),
        cell_corner_radius_px=scale_games_px(params.get("cell_corner_radius_px", group_default(_RENDER_DEFAULTS, "cell_corner_radius_px", RENDER_FALLBACKS.cell_corner_radius_px)), unit_scale, min_px=6),
        cell_gap_px=scale_games_px(params.get("cell_gap_px", group_default(_RENDER_DEFAULTS, "cell_gap_px", RENDER_FALLBACKS.cell_gap_px)), unit_scale, min_px=4),
        mark_inset_px=scale_games_px(params.get("mark_inset_px", group_default(_RENDER_DEFAULTS, "mark_inset_px", RENDER_FALLBACKS.mark_inset_px)), unit_scale, min_px=5),
        called_panel_width_px=int(called_panel_width_px),
        called_panel_gap_px=int(called_panel_gap_px),
        called_panel_title_font_size_px=scale_games_px(params.get("called_panel_title_font_size_px", group_default(_RENDER_DEFAULTS, "called_panel_title_font_size_px", RENDER_FALLBACKS.called_panel_title_font_size_px)), unit_scale, min_px=14),
        called_panel_number_font_size_px=scale_games_px(params.get("called_panel_number_font_size_px", group_default(_RENDER_DEFAULTS, "called_panel_number_font_size_px", RENDER_FALLBACKS.called_panel_number_font_size_px)), unit_scale, min_px=14),
        mark_shape=str(mark_shape or params.get("mark_shape", group_default(_RENDER_DEFAULTS, "mark_shape", RENDER_FALLBACKS.mark_shape))),
        cell_fill_pattern=str(cell_fill_pattern or params.get("cell_fill_pattern", group_default(_RENDER_DEFAULTS, "cell_fill_pattern", RENDER_FALLBACKS.cell_fill_pattern))),
        font_family=str(font_family),
        layout_jitter_meta=layout_jitter,
        instance_seed=int(instance_seed),
    )


def _draw_shadow(
    image: Image.Image,
    *,
    bbox_px: Tuple[float, float, float, float],
    radius_px: int,
    shadow_rgb: Tuple[int, int, int],
    shadow_alpha: int,
    shadow_offset_px: Tuple[int, int],
) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    left, top, right, bottom = bbox_px
    dx, dy = shadow_offset_px
    draw.rounded_rectangle(
        [left + dx, top + dy, right + dx, bottom + dy],
        radius=int(radius_px),
        fill=(int(shadow_rgb[0]), int(shadow_rgb[1]), int(shadow_rgb[2]), int(shadow_alpha)),
    )
    image.alpha_composite(overlay)


def render_bingo_card_scene(
    *,
    cells: Sequence[BingoCellInstance],
    called_numbers: Sequence[int] | None = None,
    background: Image.Image,
    scene_variant: str,
    style_variant: str,
    params: BingoRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedBingoCardScene:
    """Render one visible Bingo card with marked cells and traced cell boxes."""

    if str(scene_variant) != "single_card":
        raise ValueError(f"unsupported bingo scene variant: {scene_variant}")

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image)
    theme: BingoTheme = build_games_bingo_theme(style_variant=str(style_variant))
    mark_shape = str(params.mark_shape)
    if mark_shape not in SUPPORTED_BINGO_MARK_SHAPES:
        raise ValueError(f"unsupported bingo mark shape: {params.mark_shape}")
    cell_fill_pattern = str(params.cell_fill_pattern)
    if cell_fill_pattern not in SUPPORTED_BINGO_CELL_FILL_PATTERNS:
        raise ValueError(f"unsupported bingo cell fill pattern: {params.cell_fill_pattern}")

    called_values = tuple(int(value) for value in (called_numbers or ()))
    has_called_panel = bool(called_values)
    called_panel_width = int(params.called_panel_width_px) if has_called_panel else 0
    called_panel_gap = int(params.called_panel_gap_px) if has_called_panel else 0
    group_width = int(params.card_width_px) + int(called_panel_gap) + int(called_panel_width)
    group_height = int(params.card_height_px)
    group_left = float(0.5 * (int(params.canvas_width) - int(group_width)))
    group_top = float(0.5 * (int(params.canvas_height) - int(group_height)))
    group_right = float(group_left + int(group_width))
    group_bottom = float(group_top + int(group_height))
    group_bbox, _dx, _dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=(group_left, group_top, group_right, group_bottom),
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    group_left, group_top, group_right, group_bottom = [float(value) for value in group_bbox]
    card_left = float(group_left)
    card_top = float(group_top)
    card_right = float(card_left + int(params.card_width_px))
    card_bottom = float(card_top + int(params.card_height_px))
    card_bbox = (float(card_left), float(card_top), float(card_right), float(card_bottom))
    called_panel_bbox: Tuple[float, float, float, float] | None = None
    if has_called_panel:
        called_left = float(card_right + int(called_panel_gap))
        called_panel_bbox = (
            float(called_left),
            float(card_top),
            float(called_left + int(called_panel_width)),
            float(card_bottom),
        )
    card_left, card_top, card_right, card_bottom = [float(value) for value in card_bbox]

    panel_bbox: Tuple[int, int, int, int] | None = None
    if panel_style is not None:
        panel_pad_x = max(20, int(round(float(params.panel_margin_px) * 0.45)))
        panel_pad_top = max(24, int(round(float(params.title_band_height_px) * 0.34)))
        panel_pad_bottom = max(20, int(round(float(params.panel_margin_px) * 0.38)))
        panel_bbox = (
            max(4, int(round(group_left)) - panel_pad_x),
            max(4, int(round(group_top)) - panel_pad_top),
            min(int(params.canvas_width) - 4, int(round(group_right)) + panel_pad_x),
            min(int(params.canvas_height) - 4, int(round(group_bottom)) + panel_pad_bottom),
        )
        draw_panel_scene_chrome(
            draw,
            bbox=panel_bbox,
            style=panel_style,
            radius=max(22, int(params.card_corner_radius_px) + 8),
            border_width=max(2, int(theme.card_border_width_px)),
        )

    _draw_shadow(
        image,
        bbox_px=card_bbox,
        radius_px=int(params.card_corner_radius_px),
        shadow_rgb=tuple(int(value) for value in theme.shadow_rgb),
        shadow_alpha=int(theme.shadow_alpha),
        shadow_offset_px=tuple(int(value) for value in theme.shadow_offset_px),
    )
    draw.rounded_rectangle(
        card_bbox,
        radius=int(params.card_corner_radius_px),
        fill=tuple(int(value) for value in theme.card_fill_rgb),
        outline=tuple(int(value) for value in theme.card_border_rgb),
        width=int(theme.card_border_width_px),
    )

    scene_entities: List[Dict[str, Any]] = []
    called_number_bboxes: Dict[str, List[float]] = {}
    if called_panel_bbox is not None:
        _draw_shadow(
            image,
            bbox_px=called_panel_bbox,
            radius_px=int(params.card_corner_radius_px),
            shadow_rgb=tuple(int(value) for value in theme.shadow_rgb),
            shadow_alpha=max(18, int(theme.shadow_alpha * 0.72)),
            shadow_offset_px=tuple(int(value) for value in theme.shadow_offset_px),
        )
        draw.rounded_rectangle(
            called_panel_bbox,
            radius=int(params.card_corner_radius_px),
            fill=tuple(int(value) for value in theme.card_fill_rgb),
            outline=tuple(int(value) for value in theme.card_border_rgb),
            width=int(theme.card_border_width_px),
        )
        panel_left, panel_top, panel_right, panel_bottom = called_panel_bbox
        called_title = "CALLED"
        called_title_font = fit_font_to_box(
            draw,
            text=called_title,
            max_width=max(40.0, float(panel_right - panel_left) * 0.78),
            max_height=max(14.0, float(params.title_band_height_px) * 0.58),
            bold=True,
            min_size_px=11,
            max_size_px=max(11, int(params.called_panel_title_font_size_px)),
            fill_ratio=0.98,
            font_family=str(params.font_family) or None,
        )
        called_title_bbox = draw.textbbox((0, 0), called_title, font=called_title_font, stroke_width=1)
        called_title_origin = (
            float(panel_left + 0.5 * ((panel_right - panel_left) - (called_title_bbox[2] - called_title_bbox[0]))),
            float(panel_top + max(12.0, 0.28 * float(params.title_band_height_px))),
        )
        draw_text_traced(
            draw,
            called_title_origin,
            called_title,
            font=called_title_font,
            fill=tuple(int(value) for value in theme.title_rgb),
            stroke_width=1,
            stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.title_rgb)),
            role="readout",
            required=False,
        )
        list_top = float(panel_top + int(params.title_band_height_px) + int(params.grid_gap_px))
        list_bottom = float(panel_bottom - int(params.panel_margin_px) * 0.56)
        item_gap = max(6.0, float(params.cell_gap_px) * 0.65)
        item_count = max(1, len(called_values))
        item_height = max(26.0, (list_bottom - list_top - ((item_count - 1) * item_gap)) / item_count)
        item_left = float(panel_left + max(18, int(params.panel_margin_px) // 2))
        item_right = float(panel_right - max(18, int(params.panel_margin_px) // 2))
        called_number_font = fit_font_to_box(
            draw,
            text="75",
            max_width=max(22.0, float(item_right - item_left) * 0.74),
            max_height=max(16.0, float(item_height) * 0.60),
            bold=True,
            min_size_px=10,
            max_size_px=max(10, int(params.called_panel_number_font_size_px)),
            fill_ratio=0.98,
            font_family=str(params.font_family) or None,
        )
        for item_index, value in enumerate(called_values):
            item_top = float(list_top + item_index * (item_height + item_gap))
            item_bottom = float(item_top + item_height)
            item_bbox = (float(item_left), float(item_top), float(item_right), float(item_bottom))
            draw.rounded_rectangle(
                item_bbox,
                radius=max(8, int(params.cell_corner_radius_px) - 2),
                fill=tuple(int(value) for value in theme.cell_fill_rgb),
                outline=tuple(int(value) for value in theme.grid_line_rgb),
                width=2,
            )
            number_text = str(int(value))
            number_bbox = draw.textbbox((0, 0), number_text, font=called_number_font, stroke_width=1)
            number_origin = (
                float(item_left + 0.5 * ((item_right - item_left) - (number_bbox[2] - number_bbox[0]))),
                float(item_top + 0.5 * ((item_bottom - item_top) - (number_bbox[3] - number_bbox[1]))),
            )
            draw_text_traced(
                draw,
                number_origin,
                number_text,
                font=called_number_font,
                fill=tuple(int(value) for value in theme.number_rgb),
                stroke_width=1,
                stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.number_rgb)),
                role="readout",
                required=False,
            )
            rounded_item_bbox = [
                round(float(item_left), 3),
                round(float(item_top), 3),
                round(float(item_right), 3),
                round(float(item_bottom), 3),
            ]
            called_number_bboxes[f"called_{int(item_index)}"] = list(rounded_item_bbox)
            scene_entities.append(
                {
                    "entity_id": f"called_{int(item_index)}",
                    "kind": "bingo_called_number",
                    "bbox": list(rounded_item_bbox),
                    "list_index": int(item_index),
                    "number": int(value),
                }
            )

    title_text = "BINGO"
    title_font = fit_font_to_box(
        draw,
        text=title_text,
        max_width=max(40.0, float(card_right - card_left) * 0.82),
        max_height=max(12.0, float(params.title_band_height_px) * 0.78),
        bold=True,
        min_size_px=12,
        max_size_px=max(12, int(params.title_font_size_px)),
        fill_ratio=0.98,
        font_family=str(params.font_family) or None,
    )
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font, stroke_width=1)
    title_origin = (
        float(card_left + (0.5 * ((card_right - card_left) - (title_bbox[2] - title_bbox[0])))),
        float(card_top + max(10.0, 0.5 * (int(params.title_band_height_px) - (title_bbox[3] - title_bbox[1])))),
    )
    draw_text_traced(
        draw,
        title_origin,
        title_text,
        font=title_font,
        fill=tuple(int(value) for value in theme.title_rgb),
        stroke_width=1,
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.title_rgb)),
        role="readout",
        required=False,
    )

    grid_left = float(card_left + int(params.panel_margin_px))
    grid_right = float(card_right - int(params.panel_margin_px))
    header_top = float(card_top + int(params.title_band_height_px))
    header_bottom = float(header_top + int(params.header_height_px))
    grid_top = float(header_bottom + int(params.grid_gap_px))
    grid_bottom = float(card_bottom - int(params.panel_margin_px))

    total_gap_width = float((BINGO_BOARD_SIZE - 1) * int(params.cell_gap_px))
    total_gap_height = float((BINGO_BOARD_SIZE - 1) * int(params.cell_gap_px))
    cell_width = float((grid_right - grid_left - total_gap_width) / BINGO_BOARD_SIZE)
    cell_height = float((grid_bottom - grid_top - total_gap_height) / BINGO_BOARD_SIZE)
    header_font = fit_font_to_box(
        draw,
        text="W",
        max_width=max(12.0, float(cell_width) * 0.78),
        max_height=max(12.0, float(params.header_height_px) * 0.74),
        bold=True,
        min_size_px=10,
        max_size_px=max(10, int(params.header_font_size_px)),
        fill_ratio=0.98,
        font_family=str(params.font_family) or None,
    )
    number_font = fit_font_to_box(
        draw,
        text="75",
        max_width=max(14.0, float(cell_width) * 0.76),
        max_height=max(14.0, float(cell_height) * 0.68),
        bold=True,
        min_size_px=10,
        max_size_px=max(10, int(params.number_font_size_px)),
        fill_ratio=0.98,
        font_family=str(params.font_family) or None,
    )

    column_header_bboxes: Dict[str, List[float]] = {}
    for column_index, column_label in enumerate(BINGO_COLUMN_LABELS):
        header_bbox = draw.textbbox((0, 0), str(column_label), font=header_font, stroke_width=1)
        header_origin = (
            float(
                grid_left
                + column_index * (cell_width + int(params.cell_gap_px))
                + (0.5 * (cell_width - (header_bbox[2] - header_bbox[0])))
            ),
            float(header_top + (0.5 * (int(params.header_height_px) - (header_bbox[3] - header_bbox[1])))),
        )
        draw_text_traced(
            draw,
            header_origin,
            str(column_label),
            font=header_font,
            fill=tuple(int(value) for value in theme.header_rgb),
            stroke_width=1,
            stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.header_rgb)),
            role="readout",
            required=False,
        )
        column_header_bboxes[str(column_label)] = [
            round(float(header_origin[0]), 3),
            round(float(header_origin[1]), 3),
            round(float(header_origin[0] + (header_bbox[2] - header_bbox[0])), 3),
            round(float(header_origin[1] + (header_bbox[3] - header_bbox[1])), 3),
        ]

    cell_specs: List[RenderedBingoCellSpec] = []
    cell_bbox_map: Dict[str, List[float]] = {}
    cell_mark_center_map: Dict[str, List[float]] = {}
    number_draw_specs: List[Tuple[str, Tuple[float, float]]] = []
    mark_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mark_draw = ImageDraw.Draw(mark_overlay)

    from trace_tasks.tasks.games.shared.marking import (
        draw_semantic_bbox_marker,
        draw_semantic_ellipse_marker,
        resolve_semantic_marker_style,
    )

    for cell in cells:
        left = float(grid_left + cell.column_index * (cell_width + int(params.cell_gap_px)))
        top = float(grid_top + cell.row_index * (cell_height + int(params.cell_gap_px)))
        right = float(left + cell_width)
        bottom = float(top + cell_height)
        bbox = (left, top, right, bottom)

        if cell_fill_pattern == "column_tint" and int(cell.column_index) % 2 == 1:
            cell_fill_rgb = tuple(int(value) for value in theme.cell_alt_fill_rgb)
        elif cell_fill_pattern == "checker_tint" and (int(cell.row_index) + int(cell.column_index)) % 2 == 1:
            cell_fill_rgb = tuple(int(value) for value in theme.cell_alt_fill_rgb)
        else:
            cell_fill_rgb = tuple(int(value) for value in theme.cell_fill_rgb)

        draw.rounded_rectangle(
            bbox,
            radius=int(params.cell_corner_radius_px),
            fill=tuple(int(value) for value in cell_fill_rgb),
            outline=tuple(int(value) for value in theme.grid_line_rgb),
            width=2,
        )

        number_text = str(cell.number)
        number_bbox = draw.textbbox((0, 0), number_text, font=number_font, stroke_width=1)
        number_origin = (
            float(left + (0.5 * (cell_width - (number_bbox[2] - number_bbox[0])))),
            float(top + (0.5 * (cell_height - (number_bbox[3] - number_bbox[1])))),
        )
        number_draw_specs.append((number_text, number_origin))

        if bool(cell.is_marked):
            inset = int(params.mark_inset_px)
            mark_bbox = [left + inset, top + inset, right - inset, bottom - inset]
            marker_style = resolve_semantic_marker_style(
                instance_seed=int(params.instance_seed),
                namespace=f"games.bingo.marked_cell.{cell.cell_id}",
                role="bingo_marked_cell",
                surface_rgbs=(
                    tuple(int(value) for value in cell_fill_rgb),
                    tuple(int(value) for value in theme.number_rgb),
                ),
                preferred_rgbs=(tuple(int(value) for value in theme.mark_outline_rgb),),
            )
            marker_fill_rgba = (
                int(marker_style.inner_rgb[0]),
                int(marker_style.inner_rgb[1]),
                int(marker_style.inner_rgb[2]),
                max(34, min(92, int(theme.mark_fill_rgba[3]))),
            )
            if mark_shape == "cell":
                draw_semantic_bbox_marker(
                    mark_draw,
                    [left + 2, top + 2, right - 2, bottom - 2],
                    radius=max(2, int(params.cell_corner_radius_px) - 2),
                    style=marker_style,
                    width=3,
                    fill_rgba=marker_fill_rgba,
                    marker_kind="bingo_marked_cell_fill",
                    extra_metadata={"cell_id": str(cell.cell_id), "mark_shape": str(mark_shape)},
                )
            elif mark_shape == "ring":
                draw_semantic_ellipse_marker(
                    mark_draw,
                    mark_bbox,
                    style=marker_style,
                    width=5,
                    fill_rgba=(int(marker_style.inner_rgb[0]), int(marker_style.inner_rgb[1]), int(marker_style.inner_rgb[2]), 34),
                    marker_kind="bingo_marked_cell_ring",
                    extra_metadata={"cell_id": str(cell.cell_id), "mark_shape": str(mark_shape)},
                )
            else:
                draw_semantic_ellipse_marker(
                    mark_draw,
                    mark_bbox,
                    style=marker_style,
                    width=3,
                    fill_rgba=marker_fill_rgba,
                    marker_kind="bingo_marked_cell_ellipse",
                    extra_metadata={"cell_id": str(cell.cell_id), "mark_shape": str(mark_shape)},
                )

        rounded_bbox = [round(float(left), 3), round(float(top), 3), round(float(right), 3), round(float(bottom), 3)]
        cell_bbox_map[str(cell.cell_id)] = list(rounded_bbox)
        cell_mark_center_map[str(cell.cell_id)] = [
            round(float(0.5 * (left + right)), 3),
            round(float(0.5 * (top + bottom)), 3),
        ]
        cell_specs.append(
            RenderedBingoCellSpec(
                cell_id=str(cell.cell_id),
                row_index=int(cell.row_index),
                column_index=int(cell.column_index),
                column_label=str(cell.column_label),
                number=int(cell.number),
                is_marked=bool(cell.is_marked),
                bbox_px=tuple(float(value) for value in rounded_bbox),
            )
        )
        scene_entities.append(
            {
                "entity_id": str(cell.cell_id),
                "kind": "bingo_cell",
                "bbox": list(rounded_bbox),
                "row_index": int(cell.row_index),
                "column_index": int(cell.column_index),
                "column_label": str(cell.column_label),
                "number": int(cell.number),
                "is_marked": bool(cell.is_marked),
            }
        )

    image.alpha_composite(mark_overlay)
    draw = ImageDraw.Draw(image, "RGBA")
    for number_text, number_origin in number_draw_specs:
        draw_text_traced(
            draw,
            number_origin,
            number_text,
            font=number_font,
            fill=tuple(int(value) for value in theme.number_rgb),
            stroke_width=1,
            stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.number_rgb)),
            role="readout",
            required=False,
        )
    return RenderedBingoCardScene(
        image=image.convert("RGB"),
        cell_specs=tuple(cell_specs),
        scene_entities=tuple(scene_entities),
        render_map={
            "card_bbox_px": [round(float(value), 3) for value in card_bbox],
            "called_panel_bbox_px": None if called_panel_bbox is None else [round(float(value), 3) for value in called_panel_bbox],
            "panel_bbox_px": None if panel_bbox is None else [int(value) for value in panel_bbox],
            "column_header_bboxes_px": dict(column_header_bboxes),
            "cell_bboxes_px": dict(cell_bbox_map),
            "cell_mark_centers_px": dict(cell_mark_center_map),
            "called_number_bboxes_px": dict(called_number_bboxes),
            "called_numbers": [int(value) for value in called_values],
            "mark_shape": str(mark_shape),
            "cell_fill_pattern": str(cell_fill_pattern),
            "style_variant": str(style_variant),
            "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
            "layout_jitter": dict(layout_jitter),
            "effective_cell_size_px": round(float(min(cell_width, cell_height)), 3),
            "font_family": str(params.font_family),
        },
    )


def render_bingo_task_scene(
    *,
    cells: Sequence[BingoCellInstance],
    called_numbers: Sequence[int],
    params: Mapping[str, Any],
    instance_seed: int,
    scene_variant: str,
    style_variant: str,
    mark_shape: str,
    cell_fill_pattern: str,
    show_called_panel: bool,
) -> RenderedBingoTaskContext:
    """Render one Bingo task scene with panel background and post-image noise."""

    render_params = resolve_bingo_render_params(
        params,
        instance_seed=int(instance_seed),
        mark_shape=str(mark_shape),
        cell_fill_pattern=str(cell_fill_pattern),
        show_called_panel=bool(show_called_panel),
    )
    allowed_panel_treatments_raw = params.get("panel_scene_treatments", group_default(_RENDER_DEFAULTS, "panel_scene_treatments", None))
    if isinstance(allowed_panel_treatments_raw, str):
        allowed_panel_treatments = (str(allowed_panel_treatments_raw),)
    elif allowed_panel_treatments_raw is None:
        allowed_panel_treatments = None
    else:
        allowed_panel_treatments = tuple(str(item) for item in allowed_panel_treatments_raw)
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace="games.bingo_board.panel_scene_style",
        treatments=allowed_panel_treatments,
        treatment_weights=params.get("panel_scene_treatment_weights", group_default(_RENDER_DEFAULTS, "panel_scene_treatment_weights", None)),
        palette_weights=params.get("panel_scene_palette_weights", group_default(_RENDER_DEFAULTS, "panel_scene_palette_weights", None)),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered_scene = render_bingo_card_scene(
        cells=list(cells),
        called_numbers=tuple(called_numbers) if show_called_panel else (),
        background=background,
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        params=render_params,
        panel_style=panel_style,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    text_style_meta = {
        "font_family": str(render_params.font_family),
        "font_asset": get_font_family_record(str(render_params.font_family)).to_trace(),
    }
    return RenderedBingoTaskContext(
        image=image,
        rendered_scene=rendered_scene,
        render_params=render_params,
        panel_style_meta=dict(panel_style_meta),
        text_style_meta=dict(text_style_meta),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


__all__ = [
    "BingoRenderParams",
    "RenderedBingoCardScene",
    "RenderedBingoCellSpec",
    "RenderedBingoTaskContext",
    "SUPPORTED_BINGO_CELL_FILL_PATTERNS",
    "SUPPORTED_BINGO_MARK_SHAPES",
    "render_bingo_card_scene",
    "render_bingo_task_scene",
    "resolve_bingo_render_params",
]
