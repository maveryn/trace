"""Solitaire tableau and foundation rules."""

from __future__ import annotations

from typing import List, Tuple

from .state import Card, Foundation, SUITS


def deck() -> List[Tuple[int, str]]:
    """Return a full rank/suit deck in deterministic order."""

    return [(int(rank), str(suit)) for suit in SUITS for rank in range(1, 14)]


def card_color(suit_name: str) -> str:
    """Return the solitaire color family for one suit."""

    return "red" if str(suit_name) in {"hearts", "diamonds"} else "black"


def is_legal_tableau_move(source: Card, target: Card) -> bool:
    """Return whether source may be placed on a tableau target card."""

    return (
        int(target.rank_value) == int(source.rank_value) + 1
        and card_color(str(target.suit_name)) != card_color(str(source.suit_name))
    )


def is_same_suit_descending_next(upper: Card, lower: Card) -> bool:
    """Return whether lower continues a same-suit descending run from upper."""

    return (
        str(lower.suit_name) == str(upper.suit_name)
        and int(lower.rank_value) == int(upper.rank_value) - 1
    )


def is_legal_foundation_move(source: Card, foundation: Foundation) -> bool:
    """Return whether source may move onto the foundation pile."""

    return (
        str(source.suit_name) == str(foundation.suit_name)
        and int(source.rank_value) == int(foundation.top_rank_value) + 1
    )


def remove_card(pool: List[Tuple[int, str]], card: Tuple[int, str]) -> None:
    """Remove one exact rank/suit card from a mutable deck pool."""

    pool.remove((int(card[0]), str(card[1])))
