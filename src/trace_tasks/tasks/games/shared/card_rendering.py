"""Scene-neutral playing-card face rendering primitives."""

from __future__ import annotations

from typing import Dict, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill, temporary_default_font_family

from .style import CardTheme, suit_color


SUIT_SYMBOLS: Dict[str, str] = {
    "spades": "♠",
    "hearts": "♥",
    "diamonds": "♦",
    "clubs": "♣",
}


def load_playing_card_suit_symbol_font(size_px: int):
    """Load a stable font for card suit glyphs independent of sampled text fonts."""

    with temporary_default_font_family(""):
        return load_font(int(size_px), bold=True)


def draw_playing_card_face(
    image: Image.Image,
    *,
    bbox_px: Tuple[float, float, float, float],
    rank_label: str,
    suit_name: str,
    theme: CardTheme,
    corner_radius_px: int,
    rank_font_size_px: int,
    center_symbol_font_size_px: int,
    center_label_mode: str = "suit_symbol",
    banner_text: str = "",
    banner_height_px: int = 0,
    banner_font_size_px: int = 18,
) -> None:
    """Draw one face-up playing card with suit glyphs and optional top banner."""

    draw = ImageDraw.Draw(image)
    left, top, right, bottom = bbox_px
    radius = int(corner_radius_px)
    draw.rounded_rectangle(
        [left, top, right, bottom],
        radius=radius,
        fill=tuple(int(value) for value in theme.card_fill_rgb),
        outline=tuple(int(value) for value in theme.card_border_rgb),
        width=int(theme.card_border_width_px),
    )

    # Optional banners are used by card-scene reference and hand labels.
    banner_height = int(banner_height_px) if str(banner_text).strip() else 0
    if str(banner_text).strip():
        banner_bottom = float(top + banner_height)
        draw.rounded_rectangle(
            [left, top, right, banner_bottom],
            radius=radius,
            fill=tuple(int(value) for value in theme.reference_fill_rgb),
        )
        banner_font = load_font(int(banner_font_size_px), bold=True)
        text_bbox = draw.textbbox((0, 0), str(banner_text), font=banner_font, stroke_width=1)
        text_width = float(text_bbox[2] - text_bbox[0])
        text_height = float(text_bbox[3] - text_bbox[1])
        draw_text_traced(
            draw,
            (
                float(left + (0.5 * ((right - left) - text_width))),
                float(top + (0.5 * (banner_height - text_height))),
            ),
            str(banner_text),
            font=banner_font,
            fill=tuple(int(value) for value in theme.reference_text_rgb),
            stroke_width=1,
            stroke_fill=(0, 0, 0),
            role="readout",
            required=False,
        )

    suit = str(suit_name)
    suit_symbol = SUIT_SYMBOLS[suit]
    pip_rgb = suit_color(theme, suit_name=suit)
    rank_rgb = (
        tuple(int(value) for value in theme.rank_rgb_red)
        if suit in {"hearts", "diamonds"}
        else tuple(int(value) for value in theme.rank_rgb_black)
    )
    rank_text = str(rank_label)
    small_font = load_font(int(rank_font_size_px), bold=True)
    small_suit_font = load_playing_card_suit_symbol_font(int(rank_font_size_px))
    label_stroke = resolve_text_stroke_fill(rank_rgb)

    # Corner labels share the same rank/suit composition at two card corners.
    def rank_suit_size(*, rank_font, suit_font, stroke_width: int) -> Tuple[float, float]:
        rank_bbox = draw.textbbox((0, 0), rank_text, font=rank_font, stroke_width=int(stroke_width))
        suit_bbox = draw.textbbox((0, 0), suit_symbol, font=suit_font, stroke_width=int(stroke_width))
        rank_width = float(rank_bbox[2] - rank_bbox[0])
        rank_height = float(rank_bbox[3] - rank_bbox[1])
        suit_width = float(suit_bbox[2] - suit_bbox[0])
        suit_height = float(suit_bbox[3] - suit_bbox[1])
        gap_px = float(max(2, int(0.12 * float(rank_font_size_px))))
        return (
            float(rank_width + gap_px + suit_width),
            float(max(rank_height, suit_height)),
        )

    def draw_rank_suit(
        origin: Tuple[float, float],
        *,
        rank_font,
        suit_font,
        stroke_width: int,
    ) -> None:
        rank_bbox = draw.textbbox((0, 0), rank_text, font=rank_font, stroke_width=int(stroke_width))
        suit_bbox = draw.textbbox((0, 0), suit_symbol, font=suit_font, stroke_width=int(stroke_width))
        rank_width = float(rank_bbox[2] - rank_bbox[0])
        rank_height = float(rank_bbox[3] - rank_bbox[1])
        suit_height = float(suit_bbox[3] - suit_bbox[1])
        total_height = max(float(rank_height), float(suit_height))
        gap_px = float(max(2, int(0.12 * float(rank_font_size_px))))
        rank_origin = (float(origin[0]), float(origin[1] + (0.5 * (total_height - rank_height))))
        suit_origin = (
            float(origin[0] + rank_width + gap_px),
            float(origin[1] + (0.5 * (total_height - suit_height))),
        )
        draw_text_traced(
            draw,
            rank_origin,
            rank_text,
            font=rank_font,
            fill=rank_rgb,
            stroke_width=int(stroke_width),
            stroke_fill=tuple(int(value) for value in label_stroke),
            role="readout",
            required=False,
        )
        draw_text_traced(
            draw,
            suit_origin,
            suit_symbol,
            font=suit_font,
            fill=rank_rgb,
            stroke_width=int(stroke_width),
            stroke_fill=tuple(int(value) for value in label_stroke),
            role="readout",
            required=False,
        )

    top_label_origin = (
        float(left + 10),
        float(top + banner_height + 10),
    )
    draw_rank_suit(top_label_origin, rank_font=small_font, suit_font=small_suit_font, stroke_width=1)

    bottom_width, bottom_height = rank_suit_size(rank_font=small_font, suit_font=small_suit_font, stroke_width=1)
    bottom_origin = (
        float(right - bottom_width - 10),
        float(bottom - bottom_height - 10),
    )
    draw_rank_suit(bottom_origin, rank_font=small_font, suit_font=small_suit_font, stroke_width=1)

    # Center rendering can be a large suit glyph or a larger rank+suit pair.
    center_font_size = int(center_symbol_font_size_px)
    if str(center_label_mode) == "rank_suit":
        center_font_size = max(int(rank_font_size_px) + 8, int(0.78 * float(center_symbol_font_size_px)))
        center_rank_font = load_font(int(center_font_size), bold=True)
        center_suit_font = load_playing_card_suit_symbol_font(int(center_font_size))
        center_width, center_height = rank_suit_size(
            rank_font=center_rank_font,
            suit_font=center_suit_font,
            stroke_width=1,
        )
    else:
        center_font = load_playing_card_suit_symbol_font(int(center_font_size))
        center_bbox = draw.textbbox((0, 0), suit_symbol, font=center_font, stroke_width=1)
        center_width = float(center_bbox[2] - center_bbox[0])
        center_height = float(center_bbox[3] - center_bbox[1])
    center_vertical_nudge_px = float(max(6, int(0.05 * float(bottom - top))))
    center_origin = (
        float(left + (0.5 * ((right - left) - center_width))),
        float(
            top
            + banner_height
            + 20
            + (0.5 * ((bottom - top - banner_height - 40) - center_height))
            - center_vertical_nudge_px
        ),
    )
    if str(center_label_mode) == "rank_suit":
        draw_rank_suit(center_origin, rank_font=center_rank_font, suit_font=center_suit_font, stroke_width=1)
    else:
        draw_text_traced(
            draw,
            center_origin,
            suit_symbol,
            font=center_font,
            fill=pip_rgb,
            stroke_width=1,
            stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(pip_rgb)),
            role="readout",
            required=False,
        )


__all__ = [
    "SUIT_SYMBOLS",
    "draw_playing_card_face",
    "load_playing_card_suit_symbol_font",
]
