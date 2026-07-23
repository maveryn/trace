"""Scene-level defaults for dominoes games tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


SCENE_ID = "dominoes"
DOMINOES_NAMESPACE = "games.dominoes"
SUPPORTED_DOMINO_SCENE_VARIANTS: Tuple[str, ...] = ("single_row", "two_row")
PROMPT_WIRING_KEYS: Tuple[str, ...] = (
    "bundle_id",
    "scene_key",
    "task_key",
    "json_output_contract",
    "json_output_contract_answer_only",
)


@dataclass(frozen=True)
class DominoSceneDefaults:
    """Stable fallback defaults for visible domino-chain scenes."""

    matching_end_target_answer_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    longest_chain_length_answer_support: Tuple[int, ...] = (1, 2, 3, 4, 5)
    invalid_join_label_support: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
    higher_sum_target_answer_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    sum_to_target_answer_support: Tuple[int, ...] = (0, 1, 2, 3, 4)
    double_target_answer_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    single_row_candidate_count_support: Tuple[int, ...] = (7, 8, 9)
    two_row_candidate_count_support: Tuple[int, ...] = (10, 11, 12)
    sum_target_total_support: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10)
    chain_length: int = 3
    canvas_width: int = 1180
    canvas_height: int = 760
    panel_margin_px: int = 56
    chain_top_px: int = 104
    tile_width_px: int = 138
    tile_height_px: int = 76
    chain_gap_px: int = 6
    candidate_gap_px: int = 18
    row_gap_px: int = 34
    tile_corner_radius_px: int = 12
    pip_radius_px: int = 5
    divider_width_px: int = 4
    reference_tag_font_size_px: int = 16
    reference_tag_gap_px: int = 14
    section_label_font_size_px: int = 18
    section_separator_width_px: int = 2


DEFAULTS = DominoSceneDefaults()


__all__ = [
    "DEFAULTS",
    "DOMINOES_NAMESPACE",
    "PROMPT_WIRING_KEYS",
    "SCENE_ID",
    "SUPPORTED_DOMINO_SCENE_VARIANTS",
    "DominoSceneDefaults",
]
