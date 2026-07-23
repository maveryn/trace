"""Morse-code notation rules."""

from __future__ import annotations

from typing import Sequence


MORSE_LETTER_CODES: dict[str, str] = {
    "a": ".-",
    "b": "-...",
    "c": "-.-.",
    "d": "-..",
    "e": ".",
    "f": "..-.",
    "g": "--.",
    "h": "....",
    "i": "..",
    "j": ".---",
    "k": "-.-",
    "l": ".-..",
    "m": "--",
    "n": "-.",
    "o": "---",
    "p": ".--.",
    "q": "--.-",
    "r": ".-.",
    "s": "...",
    "t": "-",
    "u": "..-",
    "v": "...-",
    "w": ".--",
    "x": "-..-",
    "y": "-.--",
    "z": "--..",
}
CODE_TO_MORSE_LETTER: dict[str, str] = {code: letter for letter, code in MORSE_LETTER_CODES.items()}


def morse_code_for_letter(letter: str) -> str:
    """Return the International Morse code for one lowercase letter."""

    normalized = str(letter).lower().strip()
    if len(normalized) != 1 or normalized not in MORSE_LETTER_CODES:
        raise ValueError(f"unsupported Morse letter: {letter!r}")
    return str(MORSE_LETTER_CODES[normalized])


def morse_codes_for_word(word: str) -> tuple[str, ...]:
    """Return one Morse code string per lowercase letter in ``word``."""

    normalized = str(word).lower().strip()
    if not normalized.isalpha():
        raise ValueError("Morse word must contain alphabetic letters only")
    return tuple(morse_code_for_letter(letter) for letter in normalized)


def decode_morse_codes(codes: Sequence[str]) -> str:
    """Decode Morse code strings into one lowercase word."""

    letters: list[str] = []
    for code in codes:
        normalized = str(code).strip()
        if normalized not in CODE_TO_MORSE_LETTER:
            raise ValueError(f"unsupported Morse code: {code!r}")
        letters.append(str(CODE_TO_MORSE_LETTER[normalized]))
    return "".join(letters)


__all__ = [
    "CODE_TO_MORSE_LETTER",
    "MORSE_LETTER_CODES",
    "decode_morse_codes",
    "morse_code_for_letter",
    "morse_codes_for_word",
]
