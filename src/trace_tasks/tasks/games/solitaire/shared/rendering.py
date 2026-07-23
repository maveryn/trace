"""Rendering and projection helpers for solitaire tableau scenes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill
from trace_tasks.tasks.games.shared.card_rendering import (
    SUIT_SYMBOLS,
    draw_playing_card_face,
    load_playing_card_suit_symbol_font,
)
from trace_tasks.tasks.games.shared.layout import apply_games_layout_jitter_to_bbox, resolve_games_layout_jitter
from trace_tasks.tasks.games.shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style
from trace_tasks.tasks.games.shared.style import build_games_card_theme
from trace_tasks.tasks.games.shared.text import draw_game_text_traced as draw_text_traced

from .defaults import DEFAULTS, GEN_DEFAULTS, RENDER_DEFAULTS, int_default
from .rules import card_color
from .state import (
    RANK_LABEL,
    Card,
    Foundation,
    RenderedSolitaireScene,
    SolitaireSample,
    SolitaireVisualStyle,
    SUPPORTED_PANEL_STYLE_VARIANTS,
    empty_tableau_slot_id,
)


CANONICAL_SOLITAIRE_RED_RGB = (190, 24, 45)
CANONICAL_SOLITAIRE_BLACK_RGB = (24, 24, 24)

def rgb(values: Sequence[int]) -> Tuple[int, int, int]:
    return tuple(max(0, min(255, int(value))) for value in values[:3])  # type: ignore[return-value]


def _is_legal_tableau_move(source: Card, target: Card) -> bool:
    return (
        int(target.rank_value) == int(source.rank_value) + 1
        and card_color(str(target.suit_name)) != card_color(str(source.suit_name))
    )


def _is_same_suit_descending_next(upper: Card, lower: Card) -> bool:
    """Return whether `lower` continues a same-suit descending run from `upper`."""

    return (
        str(lower.suit_name) == str(upper.suit_name)
        and int(lower.rank_value) == int(upper.rank_value) - 1
    )


def _is_legal_foundation_move(source: Card, foundation: Foundation) -> bool:
    return (
        str(source.suit_name) == str(foundation.suit_name)
        and int(source.rank_value) == int(foundation.top_rank_value) + 1
    )


def _remove_card(pool: List[Tuple[int, str]], card: Tuple[int, str]) -> None:
    pool.remove((int(card[0]), str(card[1])))


def draw_text_center(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[float, float, float, float],
    text: str,
    *,
    font,
    fill: Tuple[int, int, int],
    stroke_width: int = 1,
) -> None:
    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=int(stroke_width))
    width = float(text_bbox[2] - text_bbox[0])
    height = float(text_bbox[3] - text_bbox[1])
    x0, y0, x1, y1 = bbox
    origin = (float(x0 + ((x1 - x0) - width) / 2.0), float(y0 + ((y1 - y0) - height) / 2.0))
    draw_text_traced(draw,
        origin,
        str(text),
        font=font,
        fill=tuple(int(value) for value in fill),
        stroke_width=int(stroke_width),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(fill)),
     role="readout", required=False,)


def draw_card(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[float, float, float, float],
    card: Card,
    *,
    radius_px: int,
    card_theme,
    rank_font_size_px: int,
    center_symbol_font_size_px: int,
    badge_font,
    badge_fill_rgb: Tuple[int, int, int],
    badge_text_rgb: Tuple[int, int, int],
) -> None:
    """Draw one solitaire card with the shared playing-card face and optional badge."""

    x0, y0, x1, y1 = bbox
    draw_playing_card_face(
        image,
        bbox_px=bbox,
        rank_label=str(card.rank_label),
        suit_name=str(card.suit_name),
        theme=card_theme,
        corner_radius_px=int(radius_px),
        rank_font_size_px=int(rank_font_size_px),
        center_symbol_font_size_px=int(center_symbol_font_size_px),
    )
    if card.badge_text:
        badge_w = 28
        badge_h = 22
        badge_box = (float(x0 - 6), float(y1 - 30), float(x0 - 6 + badge_w), float(y1 - 30 + badge_h))
        draw.rounded_rectangle(
            badge_box,
            radius=7,
            fill=tuple(int(value) for value in badge_fill_rgb),
            outline=tuple(int(value) for value in card_theme.card_border_rgb),
            width=1,
        )
        draw_text_center(draw, badge_box, str(card.badge_text), font=badge_font, fill=badge_text_rgb, stroke_width=0)


def draw_card_back(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[float, float, float, float],
    *,
    fill_rgb: Tuple[int, int, int],
    border_rgb: Tuple[int, int, int],
    accent_rgb: Tuple[int, int, int],
    radius_px: int,
) -> None:
    x0, y0, x1, y1 = bbox
    draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=int(radius_px),
        fill=tuple(int(value) for value in fill_rgb),
        outline=tuple(int(value) for value in border_rgb),
        width=2,
    )
    inset = 10
    draw.rounded_rectangle(
        [x0 + inset, y0 + inset, x1 - inset, y1 - inset],
        radius=max(3, int(radius_px) - 3),
        outline=tuple(int(value) for value in accent_rgb),
        width=2,
    )


def draw_foundation(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[float, float, float, float],
    foundation: Foundation,
    *,
    panel_fill_rgb: Tuple[int, int, int],
    border_rgb: Tuple[int, int, int],
    text_rgb: Tuple[int, int, int],
    label_font,
    card_theme,
    radius_px: int,
    rank_font_size_px: int,
    center_symbol_font_size_px: int,
) -> None:
    """Draw one foundation slot, preserving the same card-face style as tableau cards."""

    x0, y0, x1, y1 = bbox
    if int(foundation.top_rank_value) > 0:
        draw_playing_card_face(
            image,
            bbox_px=bbox,
            rank_label=str(RANK_LABEL[int(foundation.top_rank_value)]),
            suit_name=str(foundation.suit_name),
            theme=card_theme,
            corner_radius_px=int(radius_px),
            rank_font_size_px=int(rank_font_size_px),
            center_symbol_font_size_px=int(center_symbol_font_size_px),
        )
        return

    draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=int(radius_px),
        fill=tuple(int(value) for value in panel_fill_rgb),
        outline=tuple(int(value) for value in border_rgb),
        width=2,
    )
    suit_rgb = tuple(card_theme.rank_rgb_red) if card_color(str(foundation.suit_name)) == "red" else tuple(card_theme.rank_rgb_black)
    suit_font = load_playing_card_suit_symbol_font(max(26, int(center_symbol_font_size_px)))
    draw_text_center(
        draw,
        (x0 + 4, y0 + 14, x1 - 4, y1 - 24),
        SUIT_SYMBOLS[str(foundation.suit_name)],
        font=suit_font,
        fill=suit_rgb,
        stroke_width=1,
    )
    draw_text_center(draw, (x0 + 4, y1 - 30, x1 - 4, y1 - 6), "empty", font=label_font, fill=text_rgb, stroke_width=0)



def resolve_solitaire_visual_style(style_variant: str, panel_style) -> Tuple[SolitaireVisualStyle, Dict[str, Any]]:
    """Resolve card/tableau styling separate from the shared canvas treatment."""

    styles: Dict[str, SolitaireVisualStyle] = {
        "classic_cards": SolitaireVisualStyle(
            card_fill_rgb=(252, 250, 244),
            card_border_rgb=(82, 88, 101),
            card_back_rgb=(52, 88, 158),
            card_back_accent_rgb=(227, 235, 252),
            foundation_fill_rgb=(240, 245, 247),
            option_fill_rgb=(240, 246, 255),
            badge_fill_rgb=(238, 188, 71),
            badge_text_rgb=(34, 38, 45),
            red_suit_rgb=(172, 35, 45),
            black_suit_rgb=(34, 40, 50),
            text_rgb=rgb(panel_style.text_rgb),
        ),
        "ivory_table": SolitaireVisualStyle(
            card_fill_rgb=(255, 250, 235),
            card_border_rgb=(107, 82, 55),
            card_back_rgb=(125, 84, 55),
            card_back_accent_rgb=(241, 211, 154),
            foundation_fill_rgb=(245, 235, 209),
            option_fill_rgb=(248, 232, 190),
            badge_fill_rgb=(214, 163, 79),
            badge_text_rgb=(48, 38, 28),
            red_suit_rgb=(161, 47, 45),
            black_suit_rgb=(44, 39, 35),
            text_rgb=(59, 47, 36),
        ),
        "casino_felt": SolitaireVisualStyle(
            card_fill_rgb=(249, 251, 246),
            card_border_rgb=(30, 77, 56),
            card_back_rgb=(31, 116, 78),
            card_back_accent_rgb=(190, 238, 205),
            foundation_fill_rgb=(215, 235, 220),
            option_fill_rgb=(223, 242, 225),
            badge_fill_rgb=(246, 211, 80),
            badge_text_rgb=(25, 48, 34),
            red_suit_rgb=(184, 38, 53),
            black_suit_rgb=(24, 45, 35),
            text_rgb=(28, 62, 44),
        ),
        "slate_cards": SolitaireVisualStyle(
            card_fill_rgb=(233, 238, 243),
            card_border_rgb=(43, 54, 70),
            card_back_rgb=(75, 91, 111),
            card_back_accent_rgb=(199, 212, 226),
            foundation_fill_rgb=(218, 226, 235),
            option_fill_rgb=(225, 232, 241),
            badge_fill_rgb=(102, 149, 190),
            badge_text_rgb=(248, 250, 252),
            red_suit_rgb=(191, 60, 73),
            black_suit_rgb=(31, 38, 50),
            text_rgb=(33, 42, 55),
        ),
        "paper_tableau": SolitaireVisualStyle(
            card_fill_rgb=(254, 250, 239),
            card_border_rgb=(136, 111, 78),
            card_back_rgb=(209, 185, 139),
            card_back_accent_rgb=(119, 93, 57),
            foundation_fill_rgb=(244, 235, 211),
            option_fill_rgb=(250, 238, 205),
            badge_fill_rgb=(191, 134, 66),
            badge_text_rgb=(46, 35, 24),
            red_suit_rgb=(159, 56, 54),
            black_suit_rgb=(49, 43, 37),
            text_rgb=(66, 52, 38),
        ),
    }
    resolved_key = str(style_variant) if str(style_variant) in styles else "classic_cards"
    return styles[resolved_key], {
        "style_variant": str(resolved_key),
        "available_styles": list(SUPPORTED_PANEL_STYLE_VARIANTS),
        "card_style_policy": "scene_local_solitaire_card_tableau_palette",
    }


def solitaire_card_face_style_variant(style_variant: str) -> str:
    """Map solitaire table styles to the shared games playing-card face themes."""

    return {
        "classic_cards": "classic",
        "ivory_table": "ivory",
        "casino_felt": "soft",
        "slate_cards": "slate",
        "paper_tableau": "outlined",
    }.get(str(style_variant), "classic")


def render_solitaire_scene(
    *,
    sample: SolitaireSample,
    namespace: str,
    style_variant: str,
    instance_seed: int,
    params: Mapping[str, Any],
) -> RenderedSolitaireScene:
    """Render the complete solitaire tableau and record every card, pile, and option bbox in pixel space."""

    show_foundations = bool(sample.metadata.get("show_foundations", True))
    canvas_width = int_default(params, "canvas_width", DEFAULTS.canvas_width)
    canvas_height = int_default(params, "canvas_height", DEFAULTS.canvas_height)
    card_width = int_default(params, "card_width_px", DEFAULTS.card_width_px)
    card_height = int_default(params, "card_height_px", DEFAULTS.card_height_px)
    column_gap = int_default(params, "column_gap_px", DEFAULTS.column_gap_px)
    column_step_y = int_default(params, "column_step_y_px", DEFAULTS.column_step_y_px)
    margin = int_default(params, "panel_margin_px", DEFAULTS.panel_margin_px)
    foundation_gap = int_default(params, "foundation_gap_px", DEFAULTS.foundation_gap_px)
    radius = int_default(params, "card_corner_radius_px", DEFAULTS.card_corner_radius_px)
    if params.get("canvas_height") is None:
        max_column_len = max((len(column) for column in sample.columns), default=1)
        tableau_top_base = 194 if bool(show_foundations) else 92
        tableau_bottom = tableau_top_base + (max(0, int(max_column_len) - 1) * column_step_y) + card_height
        foundation_bottom = 58 + card_height
        if sample.move_options:
            option_height = int_default(params, "option_height_px", DEFAULTS.option_height_px)
            needed_height = max(tableau_bottom, foundation_bottom) + 64 + option_height + margin
            canvas_height = min(int(canvas_height), max(620, int(needed_height)))
        elif sample.card_options:
            needed_height = max(tableau_bottom, foundation_bottom) + 64 + card_height + 30 + margin
            canvas_height = min(int(canvas_height), max(700, int(needed_height)))
        else:
            needed_height = max(tableau_bottom, foundation_bottom if bool(show_foundations) else 0) + margin + 28
            canvas_height = min(int(canvas_height), max(560, int(needed_height)))
    style, style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.solitaire_panel_style",
        treatment_weights=group_default(GEN_DEFAULTS, "panel_scene_treatment_weights", None),
        palette_weights=group_default(GEN_DEFAULTS, "panel_scene_palette_weights", None),
    )
    image, background_meta = make_panel_scene_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=style,
    )
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    layout_jitter = resolve_games_layout_jitter(
        params,
        RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.solitaire.layout",
    )
    content_bbox = (float(margin), 20.0, float(canvas_width - margin), float(canvas_height - margin))
    _shifted_content_bbox, dx, dy, resolved_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=content_bbox,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        jitter=layout_jitter,
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.solitaire.font_family",
        params=params,
    )
    badge_font = load_font(int_default(params, "badge_font_size_px", DEFAULTS.badge_font_size_px), bold=True, font_family=str(font_family))
    label_font = load_font(int_default(params, "label_font_size_px", DEFAULTS.label_font_size_px), bold=True, font_family=str(font_family))
    option_font = load_font(int_default(params, "option_font_size_px", DEFAULTS.option_font_size_px), bold=True, font_family=str(font_family))
    solitaire_style, solitaire_style_meta = resolve_solitaire_visual_style(str(style_variant), style)
    card_face_style_variant = solitaire_card_face_style_variant(str(style_variant))
    card_theme = replace(
        build_games_card_theme(style_variant=str(card_face_style_variant)),
        center_symbol_rgb_black=CANONICAL_SOLITAIRE_BLACK_RGB,
        center_symbol_rgb_red=CANONICAL_SOLITAIRE_RED_RGB,
        rank_rgb_black=CANONICAL_SOLITAIRE_BLACK_RGB,
        rank_rgb_red=CANONICAL_SOLITAIRE_RED_RGB,
    )
    text_rgb = tuple(int(value) for value in solitaire_style.text_rgb)
    border_rgb = tuple(int(value) for value in solitaire_style.card_border_rgb)
    back_fill = tuple(int(value) for value in solitaire_style.card_back_rgb)
    badge_fill = tuple(int(value) for value in solitaire_style.badge_fill_rgb)
    badge_text = tuple(int(value) for value in solitaire_style.badge_text_rgb)

    foundation_y = 58 + int(round(dy))
    foundation_start_x = int(canvas_width - margin - (4 * card_width) - (3 * foundation_gap) + round(dx))
    foundation_bboxes: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = []
    if bool(show_foundations):
        for index, foundation in enumerate(sample.foundations):
            x0 = foundation_start_x + index * (card_width + foundation_gap)
            bbox = (float(x0), float(foundation_y), float(x0 + card_width), float(foundation_y + card_height))
            draw_foundation(
                image,
                draw,
                bbox,
                foundation,
                panel_fill_rgb=tuple(int(value) for value in solitaire_style.foundation_fill_rgb),
                border_rgb=border_rgb,
                text_rgb=text_rgb,
                label_font=label_font,
                card_theme=card_theme,
                radius_px=radius,
                rank_font_size_px=int_default(params, "rank_font_size_px", DEFAULTS.rank_font_size_px),
                center_symbol_font_size_px=int_default(params, "card_center_font_size_px", DEFAULTS.card_center_font_size_px),
            )
            foundation_bboxes[str(foundation.foundation_id)] = [float(value) for value in bbox]
            entities.append(
                {
                    "entity_id": str(foundation.foundation_id),
                    "entity_type": "foundation",
                    "suit_name": str(foundation.suit_name),
                    "suit_symbol": str(SUIT_SYMBOLS[str(foundation.suit_name)]),
                    "top_rank_value": int(foundation.top_rank_value),
                    "bbox_px": [float(value) for value in bbox],
                }
            )

    if bool(show_foundations) and str(sample.scene_variant) == "freecell_tableau":
        for index in range(4):
            x0 = margin + int(round(dx)) + index * (card_width + foundation_gap)
            bbox = (float(x0), float(foundation_y), float(x0 + card_width), float(foundation_y + card_height))
            draw.rounded_rectangle(bbox, radius=radius, fill=tuple(style.panel_fill_rgb), outline=border_rgb, width=2)
            draw_text_center(draw, bbox, f"Free {index + 1}", font=label_font, fill=text_rgb, stroke_width=0)
    elif bool(show_foundations):
        stock_x = margin + int(round(dx))
        waste_x = margin + card_width + foundation_gap
        waste_x += int(round(dx))
        stock_bbox = (float(stock_x), float(foundation_y), float(stock_x + card_width), float(foundation_y + card_height))
        waste_bbox = (float(waste_x), float(foundation_y), float(waste_x + card_width), float(foundation_y + card_height))
        draw_card_back(
            draw,
            stock_bbox,
            fill_rgb=back_fill,
            border_rgb=border_rgb,
            accent_rgb=tuple(int(value) for value in solitaire_style.card_back_accent_rgb),
            radius_px=radius,
        )
        draw.rounded_rectangle(waste_bbox, radius=radius, fill=tuple(style.panel_fill_rgb), outline=border_rgb, width=2)
        draw_text_center(draw, waste_bbox, "waste", font=label_font, fill=text_rgb, stroke_width=0)

    tableau_y = (194 if bool(show_foundations) else 92) + int(round(dy))
    column_count = len(sample.columns)
    total_columns_width = (column_count * card_width) + ((column_count - 1) * column_gap)
    start_x = int((canvas_width - total_columns_width) / 2 + round(dx))
    card_bboxes: Dict[str, List[float]] = {}
    card_visible_points: Dict[str, List[float]] = {}
    marked_card_id = str(sample.metadata.get("marked_card_id", ""))
    for col_index, column in enumerate(sample.columns):
        x0 = start_x + int(col_index) * (card_width + column_gap)
        header_bbox = (float(x0), float(tableau_y - 28), float(x0 + card_width), float(tableau_y - 4))
        if str(sample.scene_variant) == "klondike_tableau" and len(column) >= 3:
            back_bbox = (float(x0), float(tableau_y - 12), float(x0 + card_width), float(tableau_y - 12 + card_height))
            draw_card_back(
                draw,
                back_bbox,
                fill_rgb=back_fill,
                border_rgb=border_rgb,
                accent_rgb=tuple(int(value) for value in solitaire_style.card_back_accent_rgb),
                radius_px=radius,
            )
        draw.rounded_rectangle(
            header_bbox,
            radius=7,
            fill=tuple(int(value) for value in solitaire_style.option_fill_rgb),
            outline=border_rgb,
            width=1,
        )
        draw_text_center(draw, header_bbox, f"Col {col_index + 1}", font=label_font, fill=text_rgb, stroke_width=0)
        for row_index, card in enumerate(column):
            y0 = tableau_y + int(row_index) * column_step_y
            bbox = (float(x0), float(y0), float(x0 + card_width), float(y0 + card_height))
            draw_card(
                image,
                draw,
                bbox,
                card,
                radius_px=radius,
                card_theme=card_theme,
                rank_font_size_px=int_default(params, "rank_font_size_px", DEFAULTS.rank_font_size_px),
                center_symbol_font_size_px=int_default(params, "card_center_font_size_px", DEFAULTS.card_center_font_size_px),
                badge_font=badge_font,
                badge_fill_rgb=badge_fill,
                badge_text_rgb=badge_text,
            )
            if str(card.card_id) == marked_card_id:
                if row_index == len(column) - 1:
                    marker_bottom = float(y0 + card_height)
                else:
                    marker_bottom = min(float(y0 + card_height), float(y0 + column_step_y + 6))
                draw.rounded_rectangle(
                    (float(x0 - 4), float(y0 - 4), float(x0 + card_width + 4), float(marker_bottom)),
                    radius=max(4, int(radius // 2)),
                    outline=(218, 39, 49),
                    width=4,
                )
            card_bboxes[str(card.card_id)] = [float(value) for value in bbox]
            if row_index < len(column) - 1:
                visible_y = float(y0 + min(max(8.0, column_step_y / 2.0), card_height - 8.0))
            else:
                visible_y = float(y0 + (card_height / 2.0))
            card_visible_points[str(card.card_id)] = [float(x0 + (card_width / 2.0)), visible_y]
            entities.append(
                {
                    "entity_id": str(card.card_id),
                    "entity_type": "card",
                    "rank_value": int(card.rank_value),
                    "rank_label": str(card.rank_label),
                    "suit_name": str(card.suit_name),
                    "suit_symbol": str(SUIT_SYMBOLS[str(card.suit_name)]),
                    "suit_short": str(card.suit_short),
                    "badge_text": None if card.badge_text is None else str(card.badge_text),
                    "column_index": int(col_index),
                    "row_index": int(row_index),
                    "is_exposed": bool(row_index == len(column) - 1),
                    "is_marked": bool(str(card.card_id) == marked_card_id),
                    "bbox_px": [float(value) for value in bbox],
                }
            )
        if not column:
            slot_id = empty_tableau_slot_id(int(col_index))
            slot_bbox = (float(x0), float(tableau_y), float(x0 + card_width), float(tableau_y + card_height))
            draw.rounded_rectangle(
                slot_bbox,
                radius=radius,
                fill=tuple(int(value) for value in solitaire_style.foundation_fill_rgb),
                outline=border_rgb,
                width=2,
            )
            draw_text_center(draw, slot_bbox, "empty", font=label_font, fill=text_rgb, stroke_width=0)
            card_bboxes[str(slot_id)] = [float(value) for value in slot_bbox]
            entities.append(
                {
                    "entity_id": str(slot_id),
                    "entity_type": "empty_tableau_slot",
                    "column_index": int(col_index),
                    "row_index": 0,
                    "is_exposed": True,
                    "bbox_px": [float(value) for value in slot_bbox],
                }
            )

    option_bboxes: Dict[str, List[float]] = {}
    if sample.move_options:
        option_count = len(sample.move_options)
        option_height = int_default(params, "option_height_px", DEFAULTS.option_height_px)
        option_gap = int_default(params, "option_gap_px", DEFAULTS.option_gap_px)
        option_area_y = int(canvas_height - margin - option_height + round(dy))
        option_width = int((canvas_width - (2 * margin) - ((option_count - 1) * option_gap)) / option_count)
        for index, option in enumerate(sample.move_options):
            x0 = margin + int(round(dx)) + int(index) * (option_width + option_gap)
            bbox = (float(x0), float(option_area_y), float(x0 + option_width), float(option_area_y + option_height))
            draw.rounded_rectangle(
                bbox,
                radius=10,
                fill=tuple(int(value) for value in solitaire_style.option_fill_rgb),
                outline=border_rgb,
                width=2,
            )
            draw_text_center(
                draw,
                bbox,
                f"{option.label}: {option.move_text}",
                font=option_font,
                fill=text_rgb,
                stroke_width=0,
            )
            option_bboxes[str(option.option_id)] = [float(value) for value in bbox]
            entities.append(
                {
                    "entity_id": str(option.option_id),
                    "entity_type": "move_option",
                    "label": str(option.label),
                    "move": str(option.move_text),
                    "source_card_id": str(option.source_card_id),
                    "target_id": str(option.target_id),
                    "is_answer": bool(option.is_answer),
                    "bbox_px": [float(value) for value in bbox],
                }
            )
    elif sample.card_options:
        option_count = len(sample.card_options)
        option_gap = int_default(params, "option_gap_px", DEFAULTS.option_gap_px)
        total_w = (int(option_count) * card_width) + ((int(option_count) - 1) * option_gap)
        start_option_x = int((canvas_width - total_w) / 2.0 + round(dx))
        option_y = int(canvas_height - margin - card_height + round(dy))
        label_h = 24
        for index, option in enumerate(sample.card_options):
            x0 = start_option_x + int(index) * (card_width + option_gap)
            card_bbox = (float(x0), float(option_y), float(x0 + card_width), float(option_y + card_height))
            full_bbox = (float(x0), float(option_y - label_h - 4), float(x0 + card_width), float(option_y + card_height))
            draw_card(
                image,
                draw,
                card_bbox,
                option.card,
                radius_px=radius,
                card_theme=card_theme,
                rank_font_size_px=int_default(params, "rank_font_size_px", DEFAULTS.rank_font_size_px),
                center_symbol_font_size_px=int_default(params, "card_center_font_size_px", DEFAULTS.card_center_font_size_px),
                badge_font=badge_font,
                badge_fill_rgb=badge_fill,
                badge_text_rgb=badge_text,
            )
            label_bbox = (float(x0 + 14), float(option_y - label_h - 4), float(x0 + card_width - 14), float(option_y - 4))
            draw.rounded_rectangle(
                label_bbox,
                radius=8,
                fill=tuple(int(value) for value in solitaire_style.option_fill_rgb),
                outline=border_rgb,
                width=2,
            )
            draw_text_center(draw, label_bbox, str(option.label), font=option_font, fill=text_rgb, stroke_width=0)
            option_bboxes[str(option.option_id)] = [float(value) for value in full_bbox]
            entities.append(
                {
                    "entity_id": str(option.option_id),
                    "entity_type": "card_option",
                    "label": str(option.label),
                    "rank_value": int(option.card.rank_value),
                    "rank_label": str(option.card.rank_label),
                    "suit_name": str(option.card.suit_name),
                    "suit_symbol": str(SUIT_SYMBOLS[str(option.card.suit_name)]),
                    "suit_short": str(option.card.suit_short),
                    "card_label": str(option.card.label),
                    "is_answer": bool(option.is_answer),
                    "bbox_px": [float(value) for value in full_bbox],
                    "card_bbox_px": [float(value) for value in card_bbox],
                }
            )

    render_map = {
        "card_bboxes_px": dict(card_bboxes),
        "card_visible_points_px": dict(card_visible_points),
        "foundation_bboxes_px": dict(foundation_bboxes),
        "option_bboxes_px": dict(option_bboxes),
        "entity_bboxes_px": {**card_bboxes, **foundation_bboxes, **option_bboxes},
        "entity_points_px": dict(card_visible_points),
        "marked_card_id": marked_card_id or None,
        "marked_card_bbox_px": None if not marked_card_id else card_bboxes.get(marked_card_id),
        "column_count": int(column_count),
        "scene_variant": str(sample.scene_variant),
        "style": dict(style_meta),
        "panel_scene_style": dict(style_meta),
        "solitaire_tableau_style": dict(solitaire_style_meta),
        "card_face_style": {
            "style_variant": str(card_face_style_variant),
            "source": "games_shared_card_face",
            "solitaire_suit_color_policy": "canonical_red_black",
        },
        "font_family": str(font_family),
        "text_style": {"font_family": str(font_family)},
        "layout_jitter": dict(resolved_jitter),
    }
    return RenderedSolitaireScene(
        image=image.convert("RGB"),
        entities=tuple(entities),
        render_map=render_map,
        style_meta={
            "panel_scene_style": dict(style_meta),
            "solitaire_tableau_style": dict(solitaire_style_meta),
            "card_face_style": {
                "style_variant": str(card_face_style_variant),
                "source": "games_shared_card_face",
                "solitaire_suit_color_policy": "canonical_red_black",
            },
            "text_style": {
                "font_family": str(font_family),
                "font_asset": get_font_family_record(str(font_family)).to_trace(),
            },
        },
        background_meta=dict(background_meta),
    )
