"""Abacus value rules shared by symbolic abacus tasks."""

from __future__ import annotations


SUPPORTED_ABACUS_SCENE_VARIANTS: tuple[str, ...] = ("clean_card", "wood_frame", "worksheet")
ABACUS_COLUMN_ROLES: tuple[str, ...] = ("hundreds", "tens", "ones")
DEFAULT_OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
ABACUS_ANNOTATION_KEYS: tuple[str, ...] = (
    "hundreds_active_beads",
    "tens_active_beads",
    "ones_active_beads",
)


def digit_active_counts(digit: int) -> tuple[bool, int]:
    """Return whether the upper bead is active and how many lower beads are active."""

    if not 0 <= int(digit) <= 9:
        raise ValueError("abacus digit must be in 0..9")
    return bool(int(digit) >= 5), int(digit) % 5


def digits_for_abacus_value(value: int) -> tuple[int, int, int]:
    """Return hundreds/tens/ones digits for a three-column abacus value."""

    if not 0 <= int(value) <= 999:
        raise ValueError("abacus value must be in 0..999")
    text = f"{int(value):03d}"
    return int(text[0]), int(text[1]), int(text[2])


def value_from_digits(digits: tuple[int, int, int]) -> int:
    """Return the integer value represented by hundreds/tens/ones digits."""

    return int((100 * int(digits[0])) + (10 * int(digits[1])) + int(digits[2]))


__all__ = [
    "ABACUS_ANNOTATION_KEYS",
    "ABACUS_COLUMN_ROLES",
    "DEFAULT_OPTION_LABELS",
    "SUPPORTED_ABACUS_SCENE_VARIANTS",
    "digit_active_counts",
    "digits_for_abacus_value",
    "value_from_digits",
]
