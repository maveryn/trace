"""Simplified dartboard scoring rules for darts scene tasks."""

from __future__ import annotations

from typing import Tuple

from .defaults import BULLSEYE_SCORE, STANDARD_DART_SECTORS
from .state import DartsScoreSlot


BULLSEYE_SLOT = DartsScoreSlot(area_kind="bullseye", sector_value=None, score=BULLSEYE_SCORE)
SECTOR_SLOTS: Tuple[DartsScoreSlot, ...] = tuple(
    DartsScoreSlot(area_kind="sector", sector_value=int(value), score=int(value))
    for value in STANDARD_DART_SECTORS
)
SCORE_SLOTS: Tuple[DartsScoreSlot, ...] = (BULLSEYE_SLOT,) + SECTOR_SLOTS


def score_slot_is_bullseye(slot: DartsScoreSlot) -> bool:
    """Return whether a score slot is the center bullseye."""

    return str(slot.area_kind) == "bullseye"


def score_slot_matches_score(slot: DartsScoreSlot, *, score: int) -> bool:
    """Return whether a score slot has the requested simplified score."""

    return int(slot.score) == int(score)


def slots_for_score(score: int) -> Tuple[DartsScoreSlot, ...]:
    """Return all supported slots with one score value."""

    return tuple(slot for slot in SCORE_SLOTS if score_slot_matches_score(slot, score=int(score)))


__all__ = [
    "BULLSEYE_SLOT",
    "SCORE_SLOTS",
    "SECTOR_SLOTS",
    "score_slot_is_bullseye",
    "score_slot_matches_score",
    "slots_for_score",
]
