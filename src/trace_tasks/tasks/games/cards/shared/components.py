"""Shared face-up playing-card hand renderer for games-domain tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.text_rendering import load_font, temporary_default_font_family
from ...shared.card_rendering import SUIT_SYMBOLS, draw_playing_card_face
from ...shared.text import draw_game_text_traced as draw_text_traced
from ...shared.layout import apply_games_layout_jitter_to_bbox
from ...shared.scene_style import GamePanelSceneStyle, draw_panel_scene_chrome, game_panel_scene_style_metadata
from ...shared.style import CardTheme, build_games_card_theme
from .state import CardInstance


@dataclass(frozen=True)
class CardMoveOption:
    """One visible directed card-move option before rendering."""

    label: str
    source_label: str
    target_label: str
    is_answer: bool


@dataclass(frozen=True)
class RenderedCardSpec:
    """One face-up card after layout/render assignment."""

    card_id: str
    rank_label: str
    rank_value: int
    suit_name: str
    suit_symbol: str
    is_reference: bool
    badge_text: str | None
    group_label: str | None
    bbox_px: Tuple[float, float, float, float]
    row_index: int
    order_index: int


@dataclass(frozen=True)
class CardRenderParams:
    """Resolved render controls for one card hand scene."""

    canvas_width: int
    canvas_height: int
    card_width_px: int
    card_height_px: int
    panel_margin_px: int
    card_gap_px: int
    row_gap_px: int
    card_corner_radius_px: int
    rank_font_size_px: int
    center_symbol_font_size_px: int
    reference_banner_height_px: int
    reference_font_size_px: int
    continuation_font_size_px: int
    continuation_gap_px: int
    max_cards_per_row: int
    center_label_mode: str = "suit_symbol"
    layout_jitter_meta: Dict[str, Any] | None = None
    group_label_font_size_px: int = 22
    font_family: str = ""


@dataclass(frozen=True)
class RenderedCardHandScene:
    """Rendered card-hand scene plus trace-friendly card metadata."""

    image: Image.Image
    card_specs: Tuple[RenderedCardSpec, ...]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


def _draw_shadow(
    image: Image.Image,
    *,
    bbox_px: Tuple[float, float, float, float],
    radius_px: int,
    shadow_rgb: Tuple[int, int, int],
    shadow_alpha: int,
    shadow_offset_px: Tuple[int, int],
) -> None:
    """Draw one soft rectangular card shadow onto an RGBA image."""

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


def _draw_card_face(
    image: Image.Image,
    *,
    bbox_px: Tuple[float, float, float, float],
    card: CardInstance,
    theme: CardTheme,
    params: CardRenderParams,
) -> None:
    """Draw one face-up card inside the provided bounding box."""

    badge_text = str(card.badge_text) if card.badge_text is not None else ("REF" if bool(card.is_reference) else "")
    draw_playing_card_face(
        image,
        bbox_px=bbox_px,
        rank_label=str(card.rank_label),
        suit_name=str(card.suit_name),
        theme=theme,
        corner_radius_px=int(params.card_corner_radius_px),
        rank_font_size_px=int(params.rank_font_size_px),
        center_symbol_font_size_px=int(params.center_symbol_font_size_px),
        center_label_mode=str(params.center_label_mode),
        banner_text=str(badge_text),
        banner_height_px=int(params.reference_banner_height_px),
        banner_font_size_px=int(params.reference_font_size_px),
    )


def _row_card_positions(
    *,
    row_card_count: int,
    canvas_width: int,
    card_width_px: int,
    card_gap_px: int,
) -> List[float]:
    """Return left-edge positions for one centered row of equal-width cards."""

    total_width = (int(row_card_count) * int(card_width_px)) + (max(0, int(row_card_count) - 1) * int(card_gap_px))
    start_x = float(0.5 * (int(canvas_width) - total_width))
    return [
        float(start_x + (index * (int(card_width_px) + int(card_gap_px))))
        for index in range(int(row_card_count))
    ]


def _row_groups_for_cards(
    cards: Sequence[CardInstance],
    *,
    max_cards_per_row: int,
    row_card_counts: Sequence[int] | None = None,
) -> List[List[CardInstance]]:
    """Split cards into balanced reading-order rows with a fixed row maximum."""

    if row_card_counts is not None:
        row_sizes = [int(value) for value in row_card_counts]
        if not row_sizes or any(int(value) <= 0 for value in row_sizes):
            raise ValueError("row_card_counts must contain positive row sizes")
        if sum(row_sizes) != len(cards):
            raise ValueError("row_card_counts must sum to the number of cards")
        if any(int(value) > int(max_cards_per_row) for value in row_sizes):
            raise ValueError("row_card_counts cannot exceed max_cards_per_row")
        rows: List[List[CardInstance]] = []
        cursor = 0
        ordered_cards = [card for card in cards]
        for row_size in row_sizes:
            rows.append(ordered_cards[cursor : cursor + int(row_size)])
            cursor += int(row_size)
        return rows

    max_per_row = int(max_cards_per_row)
    if max_per_row <= 0:
        raise ValueError("max_cards_per_row must be positive")
    row_count = int(math.ceil(float(len(cards)) / float(max_per_row)))
    base_count = int(len(cards) // row_count)
    extra_count = int(len(cards) % row_count)
    row_sizes = [
        int(base_count + (1 if row_index < extra_count else 0))
        for row_index in range(row_count)
    ]

    rows: List[List[CardInstance]] = []
    cursor = 0
    ordered_cards = [card for card in cards]
    for row_size in row_sizes:
        rows.append(ordered_cards[cursor : cursor + int(row_size)])
        cursor += int(row_size)
    return rows


def _row_y_positions(
    *,
    row_count: int,
    canvas_height: int,
    card_height_px: int,
    panel_margin_px: int,
    row_gap_px: int,
) -> List[float]:
    """Return top-edge positions that keep all card rows inside the canvas."""

    if int(row_count) <= 1:
        return [float(0.5 * (int(canvas_height) - int(card_height_px)))]
    available_gap = (
        float(canvas_height)
        - (2.0 * float(panel_margin_px))
        - (float(row_count) * float(card_height_px))
    ) / float(max(1, int(row_count) - 1))
    gap_px = min(float(row_gap_px), max(0.0, float(available_gap)))
    total_height = (
        (float(row_count) * float(card_height_px))
        + ((float(row_count) - 1.0) * float(gap_px))
    )
    start_y = max(0.0, 0.5 * (float(canvas_height) - float(total_height)))
    return [
        float(start_y + (row_index * (float(card_height_px) + float(gap_px))))
        for row_index in range(int(row_count))
    ]


def _continuation_bbox(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    font_size_px: int,
    anchor_xy: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float, float, float], Any]:
    """Resolve one continuation cue origin/bbox/font tuple."""

    font = load_font(int(font_size_px), bold=True)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    origin = (
        float(anchor_xy[0] - (0.5 * width)),
        float(anchor_xy[1] - (0.5 * height)),
    )
    return (
        origin,
        (
            round(float(origin[0]), 3),
            round(float(origin[1]), 3),
            round(float(origin[0] + width), 3),
            round(float(origin[1] + height), 3),
        ),
        font,
    )


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    *,
    font,
    fill: Tuple[int, int, int],
    stroke_fill: Tuple[int, int, int] | None = None,
    stroke_width: int = 0,
) -> None:
    """Draw one text string centered on a point."""

    bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=int(stroke_width))
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    draw_text_traced(draw,
        (float(xy[0]) - (0.5 * width), float(xy[1]) - (0.5 * height)),
        str(text),
        font=font,
        fill=fill,
        stroke_width=int(stroke_width),
        stroke_fill=stroke_fill,
     role="readout", required=False,)


def _draw_move_options(
    image: Image.Image,
    *,
    move_options: Sequence[CardMoveOption],
    params: CardRenderParams,
    theme: CardTheme,
) -> Dict[str, Any]:
    """Draw a five-option directed-move panel and return option metadata."""

    if not move_options:
        return {}

    draw = ImageDraw.Draw(image)
    option_font = load_font(25, bold=True)
    panel_y0 = float(params.canvas_height - 112)
    panel_y1 = float(params.canvas_height - 34)
    left = 82.0
    gap = 18.0
    option_width = (float(params.canvas_width) - (2.0 * left) - (float(len(move_options) - 1) * gap)) / float(
        len(move_options)
    )
    option_bboxes: Dict[str, List[float]] = {}
    option_values: Dict[str, str] = {}
    answer_label = ""
    for index, option in enumerate(move_options):
        x0 = float(left + float(index) * (option_width + gap))
        x1 = float(x0 + option_width)
        bbox = [round(x0, 3), round(panel_y0, 3), round(x1, 3), round(panel_y1, 3)]
        option_bboxes[str(option.label)] = [float(v) for v in bbox]
        option_values[str(option.label)] = f"{str(option.source_label)}->{str(option.target_label)}"
        if bool(option.is_answer):
            answer_label = str(option.label)
        draw.rounded_rectangle(
            bbox,
            radius=12,
            fill=(28, 31, 35, 235),
            outline=tuple(int(v) for v in theme.reference_fill_rgb),
            width=3,
        )
        _draw_centered_text(
            draw,
            ((x0 + x1) * 0.5, (panel_y0 + panel_y1) * 0.5),
            f"{str(option.label)}: {str(option.source_label)}->{str(option.target_label)}",
            font=option_font,
            fill=(255, 255, 255),
            stroke_fill=(20, 24, 28),
            stroke_width=2,
        )
    return {
        "move_option_bboxes_px": option_bboxes,
        "move_option_values": option_values,
        "move_option_answer_label": str(answer_label),
    }


def render_cards_hand_scene(
    *,
    cards: Sequence[CardInstance],
    background: Image.Image,
    scene_variant: str,
    style_variant: str,
    params: CardRenderParams,
    show_continuation_cue: bool,
    panel_style: GamePanelSceneStyle | None = None,
    move_options: Sequence[CardMoveOption] = (),
    row_card_counts: Sequence[int] | None = None,
) -> RenderedCardHandScene:
    """Render one visible card hand and return card-level trace metadata."""

    if int(len(cards)) <= 0:
        raise ValueError("card hand renderer requires at least one visible card")

    image = background.convert("RGBA")
    theme = build_games_card_theme(style_variant=str(style_variant))
    draw = ImageDraw.Draw(image)

    with temporary_default_font_family(str(params.font_family)):
        row_groups = _row_groups_for_cards(
            cards=cards,
            max_cards_per_row=int(params.max_cards_per_row),
            row_card_counts=row_card_counts,
        )

        row_left_positions = [
            _row_card_positions(
                row_card_count=len(row_cards),
                canvas_width=int(params.canvas_width),
                card_width_px=int(params.card_width_px),
                card_gap_px=int(params.card_gap_px),
            )
            for row_cards in row_groups
        ]

        card_specs: List[RenderedCardSpec] = []
        row_ys = _row_y_positions(
            row_count=len(row_groups),
            canvas_height=int(params.canvas_height),
            card_height_px=int(params.card_height_px),
            panel_margin_px=int(params.panel_margin_px),
            row_gap_px=int(params.row_gap_px),
        )
        group_left = min(float(left) for row_lefts in row_left_positions for left in row_lefts)
        group_right = max(float(left + params.card_width_px) for row_lefts in row_left_positions for left in row_lefts)
        group_top = min(float(row_y) for row_y in row_ys)
        group_bottom = max(float(row_y + params.card_height_px) for row_y in row_ys)
        _group_bbox, dx, dy, layout_jitter = apply_games_layout_jitter_to_bbox(
            bbox_px=(group_left, group_top, group_right, group_bottom),
            canvas_width=int(params.canvas_width),
            canvas_height=int(params.canvas_height),
            jitter=params.layout_jitter_meta,
        )
        row_left_positions = [
            [float(left + dx) for left in row_lefts]
            for row_lefts in row_left_positions
        ]
        row_ys = [float(row_y + dy) for row_y in row_ys]

        if panel_style is not None:
            label_pad_px = max(38.0, float(params.group_label_font_size_px) * 3.2)
            panel_pad_px = max(22.0, float(params.panel_margin_px) * 0.55)
            panel_bbox = (
                int(max(8.0, _group_bbox[0] - panel_pad_px - label_pad_px)),
                int(max(8.0, _group_bbox[1] - panel_pad_px)),
                int(min(float(params.canvas_width - 8), _group_bbox[2] + panel_pad_px)),
                int(min(float(params.canvas_height - 8), _group_bbox[3] + panel_pad_px)),
            )
            draw_panel_scene_chrome(
                draw,
                bbox=panel_bbox,
                style=panel_style,
                radius=max(8, int(params.card_corner_radius_px) + 8),
                border_width=2,
            )

        order_index = 0
        for row_index, row_cards in enumerate(row_groups):
            row_y = float(row_ys[row_index])
            row_group_label = next(
                (str(card.group_label) for card in row_cards if card.group_label is not None and str(card.group_label).strip()),
                "",
            )
            if row_group_label:
                label_font = load_font(int(params.group_label_font_size_px), bold=True)
                label_bbox = draw.textbbox((0, 0), row_group_label, font=label_font, stroke_width=1)
                label_width = float(label_bbox[2] - label_bbox[0])
                label_height = float(label_bbox[3] - label_bbox[1])
                first_left = float(row_left_positions[row_index][0])
                label_origin = (
                    max(8.0, float(first_left - label_width - 18.0)),
                    float(row_y + (0.5 * (float(params.card_height_px) - label_height))),
                )
                draw_text_traced(draw,
                    label_origin,
                    row_group_label,
                    font=label_font,
                    fill=tuple(int(value) for value in theme.continuation_rgb),
                    stroke_width=1,
                    stroke_fill=(255, 255, 255),
                 role="readout", required=False,)
            for local_index, card in enumerate(row_cards):
                left = float(row_left_positions[row_index][local_index])
                bbox_px = (
                    round(float(left), 3),
                    round(float(row_y), 3),
                    round(float(left + params.card_width_px), 3),
                    round(float(row_y + params.card_height_px), 3),
                )
                _draw_shadow(
                    image,
                    bbox_px=bbox_px,
                    radius_px=int(params.card_corner_radius_px),
                    shadow_rgb=tuple(int(value) for value in theme.shadow_rgb),
                    shadow_alpha=int(theme.shadow_alpha),
                    shadow_offset_px=tuple(int(value) for value in theme.shadow_offset_px),
                )
                _draw_card_face(
                    image,
                    bbox_px=bbox_px,
                    card=card,
                    theme=theme,
                    params=params,
                )
                card_specs.append(
                    RenderedCardSpec(
                        card_id=str(card.card_id),
                        rank_label=str(card.rank_label),
                        rank_value=int(card.rank_value),
                        suit_name=str(card.suit_name),
                        suit_symbol=str(SUIT_SYMBOLS[str(card.suit_name)]),
                        is_reference=bool(card.is_reference),
                        badge_text=None if card.badge_text is None else str(card.badge_text),
                        group_label=None if card.group_label is None else str(card.group_label),
                        bbox_px=bbox_px,
                        row_index=int(row_index),
                        order_index=int(order_index),
                    )
                )
                order_index += 1

        continuation_bboxes_px: List[List[float]] = []
        if bool(show_continuation_cue) and len(row_groups) > 1:
            continuation_text = "continue"
            for row_index in range(len(row_groups) - 1):
                gap_center_y = float(
                    row_ys[row_index]
                    + params.card_height_px
                    + (0.5 * (row_ys[row_index + 1] - row_ys[row_index] - params.card_height_px))
                )
                cue_anchor = (
                    float(params.canvas_width - params.panel_margin_px - 92 + dx),
                    float(gap_center_y),
                )
                cue_origin, cue_bbox_px, cue_font = _continuation_bbox(
                    draw,
                    text=continuation_text,
                    font_size_px=int(params.continuation_font_size_px),
                    anchor_xy=cue_anchor,
                )
                draw_text_traced(draw,
                    cue_origin,
                    continuation_text,
                    font=cue_font,
                    fill=tuple(int(value) for value in theme.continuation_rgb),
                    stroke_width=1,
                    stroke_fill=(255, 255, 255),
                 role="readout", required=False,)
                continuation_bboxes_px.append([float(value) for value in cue_bbox_px])

    scene_entities_list: List[Dict[str, Any]] = [
        {
            "entity_id": str(spec.card_id),
            "entity_type": "playing_card",
            "bbox_px": [float(value) for value in spec.bbox_px],
            "meta": {
                "rank_label": str(spec.rank_label),
                "rank_value": int(spec.rank_value),
                "suit_name": str(spec.suit_name),
                "is_reference": bool(spec.is_reference),
                "row_index": int(spec.row_index),
                "order_index": int(spec.order_index),
                "badge_text": None if spec.badge_text is None else str(spec.badge_text),
                "group_label": None if spec.group_label is None else str(spec.group_label),
            },
        }
        for spec in card_specs
    ]
    with temporary_default_font_family(str(params.font_family)):
        option_map = _draw_move_options(
            image,
            move_options=tuple(move_options),
            params=params,
            theme=theme,
        )
    for option in move_options:
        option_bbox = option_map.get("move_option_bboxes_px", {}).get(str(option.label))
        if option_bbox is None:
            continue
        scene_entities_list.append(
            {
                "entity_id": f"move_option_{str(option.label)}",
                "entity_type": "card_move_option",
                "bbox_px": [float(v) for v in option_bbox],
                "meta": {
                    "label": str(option.label),
                    "source_label": str(option.source_label),
                    "target_label": str(option.target_label),
                    "move": f"{str(option.source_label)}->{str(option.target_label)}",
                    "is_answer": bool(option.is_answer),
                },
            }
        )

    render_map = {
        "scene_variant": str(scene_variant),
        "style_variant": str(style_variant),
        "card_bboxes_px": {str(spec.card_id): [float(value) for value in spec.bbox_px] for spec in card_specs},
        "reference_card_ids": [str(spec.card_id) for spec in card_specs if bool(spec.is_reference)],
        "card_badges": {
            str(spec.card_id): str(spec.badge_text)
            for spec in card_specs
            if spec.badge_text is not None and str(spec.badge_text).strip()
        },
        "card_group_labels": {
            str(spec.card_id): str(spec.group_label)
            for spec in card_specs
            if spec.group_label is not None and str(spec.group_label).strip()
        },
        "row_count": int(len(row_groups)),
        "max_cards_per_row": int(params.max_cards_per_row),
        "row_card_counts": [int(len(row_cards)) for row_cards in row_groups],
        "row_card_ids": [
            [str(spec.card_id) for spec in card_specs if int(spec.row_index) == int(row_index)]
            for row_index in range(len(row_groups))
        ],
        "continuation_cue_bbox_px": None if not continuation_bboxes_px else list(continuation_bboxes_px[0]),
        "continuation_cue_bboxes_px": [list(bbox_px) for bbox_px in continuation_bboxes_px],
        "center_label_mode": str(params.center_label_mode),
        "layout_jitter": dict(layout_jitter),
        "font_family": str(params.font_family),
        "suit_symbol_font_family": "system_fallback",
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
        **dict(option_map),
    }
    return RenderedCardHandScene(
        image=image.convert("RGB"),
        card_specs=tuple(card_specs),
        scene_entities=tuple(scene_entities_list),
        render_map=render_map,
    )


__all__ = [
    "CardMoveOption",
    "CardRenderParams",
    "RenderedCardHandScene",
    "RenderedCardSpec",
    "SUIT_SYMBOLS",
    "render_cards_hand_scene",
]
