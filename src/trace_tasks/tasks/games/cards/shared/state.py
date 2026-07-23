"""Cards scene state constants and dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

SCENE_ID = "cards"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("multi_row",)
RANK_VALUES: Tuple[int, ...] = tuple(range(2, 15))
RANK_LABEL_BY_VALUE: Dict[int, str] = {
    **{value: str(value) for value in range(2, 11)},
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}
SUIT_NAMES: Tuple[str, ...] = ("spades", "hearts", "diamonds", "clubs")
HAND_LABELS: Tuple[str, ...] = ("Hand A", "Hand B", "Hand C", "Hand D", "Hand E", "Hand F")
PLAYER_LABELS: Tuple[str, ...] = ("Player A", "Player B", "Player C", "Player D", "Player E", "Player F")
CANDIDATE_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
POKER_CATEGORY_LABEL_BY_KEY: Dict[str, str] = {
    "high_card": "high card",
    "one_pair": "one pair",
    "two_pair": "two pair",
    "three_of_a_kind": "three of a kind",
    "straight": "straight",
    "flush": "flush",
    "full_house": "full house",
    "four_of_a_kind": "four of a kind",
    "straight_flush": "straight flush",
}
POKER_CATEGORY_SCORE_BY_KEY: Dict[str, int] = {
    key: score
    for score, key in enumerate(
        (
            "high_card",
            "one_pair",
            "two_pair",
            "three_of_a_kind",
            "straight",
            "flush",
            "full_house",
            "four_of_a_kind",
            "straight_flush",
        )
    )
}
SUPPORTED_POKER_WINNING_CATEGORIES: Tuple[str, ...] = (
    "one_pair",
    "two_pair",
    "three_of_a_kind",
    "straight",
    "flush",
    "full_house",
    "four_of_a_kind",
    "straight_flush",
)
SUPPORTED_TRICK_PLAY_TRUMP_MODES: Tuple[str, ...] = ("no_trump", "with_trump")
SUPPORTED_POKER_DRAW_TARGET_CATEGORIES: Tuple[str, ...] = SUPPORTED_POKER_WINNING_CATEGORIES


@dataclass(frozen=True)
class CardInstance:
    """One visible face-up playing card before rendering."""

    card_id: str
    rank_label: str
    rank_value: int
    suit_name: str
    is_reference: bool = False
    badge_text: str | None = None
    group_label: str | None = None


@dataclass(frozen=True)
class SampledHand:
    """Constructed visible hand plus witness metadata."""

    cards: Tuple[CardInstance, ...]
    annotation_card_ids: Tuple[str, ...]
    reference_card_id: str | None
    reference_rank_value: int | None
    reference_rank_label: str | None
    reference_suit_name: str | None
    rank_sequence: Tuple[int, ...]
    keyed_annotation_card_ids: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class RuleSample:
    """Constructed card-game rule scene plus witness metadata."""

    pattern_kind: str
    scene_variant: str
    cards: Tuple[CardInstance, ...]
    answer: str
    annotation_card_ids: Tuple[str, ...]
    option_count: int
    cards_per_row: int
    center_label_mode: str
    render_overrides: Dict[str, int]
    prompt_slots: Dict[str, str]
    metadata: Dict[str, Any]
    row_card_counts: Tuple[int, ...] = ()


__all__ = [
    "CANDIDATE_LABELS",
    "CardInstance",
    "HAND_LABELS",
    "PLAYER_LABELS",
    "POKER_CATEGORY_LABEL_BY_KEY",
    "POKER_CATEGORY_SCORE_BY_KEY",
    "RANK_LABEL_BY_VALUE",
    "RANK_VALUES",
    "RuleSample",
    "SCENE_ID",
    "SUPPORTED_POKER_DRAW_TARGET_CATEGORIES",
    "SUPPORTED_POKER_WINNING_CATEGORIES",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_TRICK_PLAY_TRUMP_MODES",
    "SUIT_NAMES",
    "SampledHand",
]
