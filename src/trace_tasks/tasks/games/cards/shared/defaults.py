"""Cards scene fallback defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .state import SCENE_ID

FALLBACK_PROMPT_WIRING_KEYS: Tuple[str, ...] = ("bundle_id", "scene_key", "task_key")


@dataclass(frozen=True)
class CardRenderFallbacks:
    """Stable fallback render defaults for visible card scenes."""

    canvas_width: int = 1180
    canvas_height: int = 760
    card_width_px: int = 84
    card_height_px: int = 122
    panel_margin_px: int = 42
    card_gap_px: int = 12
    row_gap_px: int = 22
    card_corner_radius_px: int = 12
    rank_font_size_px: int = 19
    center_symbol_font_size_px: int = 44
    reference_banner_height_px: int = 20
    reference_font_size_px: int = 14
    continuation_font_size_px: int = 22
    continuation_gap_px: int = 28
    max_cards_per_row: int = 8
    group_label_font_size_px: int = 22


RENDER_FALLBACKS = CardRenderFallbacks()

__all__ = ["FALLBACK_PROMPT_WIRING_KEYS", "RENDER_FALLBACKS", "SCENE_ID", "CardRenderFallbacks"]
