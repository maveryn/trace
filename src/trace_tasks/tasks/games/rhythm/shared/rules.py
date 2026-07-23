"""Rhythm lane timing rules and geometry-free mechanics."""

from __future__ import annotations

from typing import Sequence, Tuple

from .state import SUPPORTED_COLOR_KEYS, RhythmNote, SampledRhythmScene


def note_entity_id(index: int) -> str:
    """Return a stable note entity id."""

    return f"note_{int(index):03d}"


def lane_entity_id(lane: int) -> str:
    """Return a stable lane label entity id."""

    return f"lane_{int(lane)}"


def lane_label(lane: int) -> str:
    """Return the image-facing one-indexed lane label."""

    return str(int(lane) + 1)


def note_hits_in_window(note: RhythmNote, beat_window: int) -> bool:
    """Return whether one note reaches the hit line within the beat window."""

    return int(note.bottom_row) <= int(beat_window)


def occupied_cells(note: RhythmNote) -> Tuple[int, ...]:
    """Return one-indexed row cells occupied by a note body."""

    return tuple(range(int(note.bottom_row), int(note.bottom_row) + max(1, int(note.length))))


def validate_rhythm_scene_basic(sample: SampledRhythmScene) -> None:
    """Validate scene-level Rhythm invariants without objective dispatch."""

    if int(sample.lane_count) <= 0:
        raise ValueError("rhythm lane_count must be positive")
    if int(sample.row_count) <= 0:
        raise ValueError("rhythm row_count must be positive")
    if int(sample.beat_window) <= 0 or int(sample.beat_window) > int(sample.row_count):
        raise ValueError("rhythm beat_window out of range")

    note_ids = [str(note.note_id) for note in sample.notes]
    if len(note_ids) != len(set(note_ids)):
        raise ValueError("rhythm note ids must be unique")

    occupied: set[tuple[int, int]] = set()
    for note in sample.notes:
        if not (0 <= int(note.lane_index) < int(sample.lane_count)):
            raise ValueError("rhythm note lane out of range")
        if str(note.color_key) not in SUPPORTED_COLOR_KEYS:
            raise ValueError("rhythm note color out of range")
        if int(note.length) <= 0:
            raise ValueError("rhythm note length must be positive")
        for row in occupied_cells(note):
            if not (1 <= int(row) <= int(sample.row_count)):
                raise ValueError("rhythm note row out of range")
            cell = (int(note.lane_index), int(row))
            if cell in occupied:
                raise ValueError("rhythm notes overlap within a lane")
            occupied.add(cell)

    if not set(str(entity_id) for entity_id in sample.annotation_entity_ids) <= set(note_ids):
        raise ValueError("rhythm annotation ids must reference note entities")


def lane_hit_notes(notes: Sequence[RhythmNote], *, lane_index: int, beat_window: int) -> Tuple[RhythmNote, ...]:
    """Return notes in one lane that hit within the requested beat window."""

    return tuple(
        note
        for note in notes
        if int(note.lane_index) == int(lane_index) and note_hits_in_window(note, int(beat_window))
    )


__all__ = [
    "lane_entity_id",
    "lane_hit_notes",
    "lane_label",
    "note_entity_id",
    "note_hits_in_window",
    "occupied_cells",
    "validate_rhythm_scene_basic",
]
