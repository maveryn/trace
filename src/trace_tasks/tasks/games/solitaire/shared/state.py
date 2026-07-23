"""Passive state and constants for solitaire tableau game tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


DOMAIN = "games"
SCENE_ID = "solitaire"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("klondike_tableau", "freecell_tableau")
SUPPORTED_PANEL_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic_cards",
    "ivory_table",
    "casino_felt",
    "slate_cards",
    "paper_tableau",
)
EMPTY_TABLEAU_SLOT_SUFFIX = "_empty_slot"
MOVE_OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
SUITS: Tuple[str, ...] = ("hearts", "diamonds", "spades", "clubs")
SUIT_SHORT: Dict[str, str] = {
    "hearts": "H",
    "diamonds": "D",
    "spades": "S",
    "clubs": "C",
}
SUIT_DISPLAY: Dict[str, str] = {
    "hearts": "hearts",
    "diamonds": "diamonds",
    "spades": "spades",
    "clubs": "clubs",
}
RANK_LABEL: Dict[int, str] = {
    1: "A",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "J",
    12: "Q",
    13: "K",
}


def empty_tableau_slot_id(column_index: int) -> str:
    """Return the entity id for one empty tableau column slot."""

    return f"col_{int(column_index) + 1:02d}{EMPTY_TABLEAU_SLOT_SUFFIX}"


def is_empty_tableau_slot_id(entity_id: str) -> bool:
    """Return whether an entity id denotes an empty tableau column slot."""

    return str(entity_id).endswith(EMPTY_TABLEAU_SLOT_SUFFIX)


def empty_tableau_slot_column_label(entity_id: str) -> str:
    """Return the display column label for an empty tableau slot id."""

    return f"Col {int(str(entity_id)[4:6])}"


@dataclass(frozen=True)
class Card:
    """One visible card in the solitaire tableau scene."""

    card_id: str
    rank_value: int
    suit_name: str
    badge_text: str | None = None

    @property
    def rank_label(self) -> str:
        return str(RANK_LABEL[int(self.rank_value)])

    @property
    def suit_short(self) -> str:
        return str(SUIT_SHORT[str(self.suit_name)])

    @property
    def label(self) -> str:
        return f"{self.rank_label}{self.suit_short}"


@dataclass(frozen=True)
class Foundation:
    """One visible solitaire foundation pile."""

    foundation_id: str
    suit_name: str
    top_rank_value: int

    @property
    def label(self) -> str:
        return f"{SUIT_DISPLAY[str(self.suit_name)]} pile"


@dataclass(frozen=True)
class MoveOption:
    """One drawn source-to-target move option."""

    option_id: str
    label: str
    source_card_id: str
    source_label: str
    target_id: str
    target_label: str
    is_answer: bool

    @property
    def move_text(self) -> str:
        return f"{self.source_label} -> {self.target_label}"


@dataclass(frozen=True)
class CardOption:
    """One labeled card-face option."""

    option_id: str
    label: str
    card: Card
    is_answer: bool


@dataclass(frozen=True)
class SolitaireSample:
    """Constructed solitaire scene plus objective witness metadata."""

    scene_variant: str
    columns: Tuple[Tuple[Card, ...], ...]
    foundations: Tuple[Foundation, ...]
    answer: int | str
    answer_type: str
    annotation_entity_ids: Tuple[str, ...]
    move_options: Tuple[MoveOption, ...]
    metadata: Dict[str, Any]
    card_options: Tuple[CardOption, ...] = ()


@dataclass(frozen=True)
class RenderedSolitaireScene:
    """Rendered solitaire scene and trace-friendly layout maps."""

    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]


@dataclass(frozen=True)
class SolitaireVisualStyle:
    """Scene-local nonsemantic colors for the solitaire table and cards."""

    card_fill_rgb: Tuple[int, int, int]
    card_border_rgb: Tuple[int, int, int]
    card_back_rgb: Tuple[int, int, int]
    card_back_accent_rgb: Tuple[int, int, int]
    foundation_fill_rgb: Tuple[int, int, int]
    option_fill_rgb: Tuple[int, int, int]
    badge_fill_rgb: Tuple[int, int, int]
    badge_text_rgb: Tuple[int, int, int]
    red_suit_rgb: Tuple[int, int, int]
    black_suit_rgb: Tuple[int, int, int]
    text_rgb: Tuple[int, int, int]
