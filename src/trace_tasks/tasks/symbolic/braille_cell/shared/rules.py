"""Braille-cell pattern utilities."""

from __future__ import annotations

from typing import Any, Sequence

from .state import BRAILLE_POSITIONS


GRADE1_LETTER_PATTERNS: dict[str, tuple[int, ...]] = {
    "a": (1,),
    "b": (1, 2),
    "c": (1, 4),
    "d": (1, 4, 5),
    "e": (1, 5),
    "f": (1, 2, 4),
    "g": (1, 2, 4, 5),
    "h": (1, 2, 5),
    "i": (2, 4),
    "j": (2, 4, 5),
    "k": (1, 3),
    "l": (1, 2, 3),
    "m": (1, 3, 4),
    "n": (1, 3, 4, 5),
    "o": (1, 3, 5),
    "p": (1, 2, 3, 4),
    "q": (1, 2, 3, 4, 5),
    "r": (1, 2, 3, 5),
    "s": (2, 3, 4),
    "t": (2, 3, 4, 5),
    "u": (1, 3, 6),
    "v": (1, 2, 3, 6),
    "w": (2, 4, 5, 6),
    "x": (1, 3, 4, 6),
    "y": (1, 3, 4, 5, 6),
    "z": (1, 3, 5, 6),
}
PATTERN_TO_GRADE1_LETTER: dict[tuple[int, ...], str] = {
    pattern: letter for letter, pattern in GRADE1_LETTER_PATTERNS.items()
}


def normalize_pattern(positions: Sequence[int]) -> tuple[int, ...]:
    """Return a sorted, validated Braille raised-dot pattern."""

    normalized = tuple(sorted({int(pos) for pos in positions}))
    if not normalized:
        raise ValueError("Braille pattern must contain at least one raised dot")
    if any(pos not in BRAILLE_POSITIONS for pos in normalized):
        raise ValueError(f"Braille dot positions must be in {BRAILLE_POSITIONS}")
    return normalized


def pattern_from_mask(mask: int) -> tuple[int, ...]:
    """Return the raised-dot pattern represented by one six-bit mask."""

    return tuple(pos for index, pos in enumerate(BRAILLE_POSITIONS) if int(mask) & (1 << int(index)))


def sample_pattern_with_count(rng: Any, count: int) -> tuple[int, ...]:
    """Sample a Braille pattern with exactly ``count`` raised dots."""

    if not 1 <= int(count) <= 6:
        raise ValueError("Braille raised-dot count must be in 1..6")
    return tuple(sorted(int(pos) for pos in rng.sample(list(BRAILLE_POSITIONS), int(count))))


def sample_any_pattern(rng: Any) -> tuple[int, ...]:
    """Sample any non-empty Braille raised-dot pattern."""

    return pattern_from_mask(int(rng.randint(1, 63)))


def pattern_key(pattern: Sequence[int]) -> str:
    """Return a compact stable key for one Braille pattern."""

    return "".join(str(pos) for pos in normalize_pattern(pattern))


def braille_pattern_for_letter(letter: str) -> tuple[int, ...]:
    """Return the Grade 1 uncontracted Braille pattern for one lowercase letter."""

    normalized = str(letter).lower().strip()
    if len(normalized) != 1 or normalized not in GRADE1_LETTER_PATTERNS:
        raise ValueError(f"unsupported Grade 1 Braille letter: {letter!r}")
    return tuple(GRADE1_LETTER_PATTERNS[normalized])


def braille_patterns_for_word(word: str) -> tuple[tuple[int, ...], ...]:
    """Return one Braille-cell pattern per lowercase letter in ``word``."""

    normalized = str(word).lower().strip()
    if not normalized.isalpha():
        raise ValueError("Braille word must contain lowercase alphabetic letters only")
    return tuple(braille_pattern_for_letter(letter) for letter in normalized)


def decode_braille_patterns(patterns: Sequence[Sequence[int]]) -> str:
    """Decode Grade 1 uncontracted Braille patterns into a lowercase word."""

    letters: list[str] = []
    for pattern in patterns:
        normalized = normalize_pattern(pattern)
        if normalized not in PATTERN_TO_GRADE1_LETTER:
            raise ValueError(f"pattern is not a supported Grade 1 letter: {pattern!r}")
        letters.append(str(PATTERN_TO_GRADE1_LETTER[normalized]))
    return "".join(letters)


__all__ = [
    "GRADE1_LETTER_PATTERNS",
    "PATTERN_TO_GRADE1_LETTER",
    "braille_pattern_for_letter",
    "braille_patterns_for_word",
    "decode_braille_patterns",
    "normalize_pattern",
    "pattern_from_mask",
    "pattern_key",
    "sample_any_pattern",
    "sample_pattern_with_count",
]
