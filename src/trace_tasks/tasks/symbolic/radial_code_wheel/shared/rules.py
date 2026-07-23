"""Pure code-wheel rules for symbolic radial scenes."""

from __future__ import annotations

from itertools import product
from typing import Sequence

from .state import CODE_SYMBOLS


def all_codes(symbols: Sequence[str] = CODE_SYMBOLS) -> tuple[str, ...]:
    """Return every three-symbol code in terminal index order."""

    return tuple("".join(parts) for parts in product(tuple(str(symbol) for symbol in symbols), repeat=3))


def code_to_index(code: str, symbols: Sequence[str] = CODE_SYMBOLS) -> int:
    """Map a three-symbol code to the matching terminal sector index."""

    symbol_tuple = tuple(str(symbol) for symbol in symbols)
    base = len(symbol_tuple)
    normalized = str(code).strip().upper()
    if len(normalized) != 3:
        raise ValueError("radial code must have exactly three symbols")
    index = 0
    for symbol in normalized:
        if symbol not in symbol_tuple:
            raise ValueError(f"unsupported radial code symbol: {symbol}")
        index = (index * base) + symbol_tuple.index(symbol)
    return int(index)


def index_to_code(index: int, symbols: Sequence[str] = CODE_SYMBOLS) -> str:
    """Map a terminal sector index back to its three-symbol code."""

    symbol_tuple = tuple(str(symbol) for symbol in symbols)
    base = len(symbol_tuple)
    value = int(index)
    if value < 0 or value >= base**3:
        raise ValueError("terminal index is outside code support")
    digits = []
    for divisor in (base**2, base, 1):
        digit = value // divisor
        value %= divisor
        digits.append(symbol_tuple[int(digit)])
    return "".join(digits)


def terminal_label_pool() -> tuple[str, ...]:
    """Return the fixed pool of short synthetic terminal labels."""

    prefixes = ("K", "M", "R", "T", "L", "N", "P", "S")
    digits = tuple(str(value) for value in range(1, 9))
    return tuple(f"{prefix}{digit}" for prefix in prefixes for digit in digits)


def validate_terminal_labels(labels: Sequence[str]) -> tuple[str, ...]:
    """Validate one complete terminal label assignment."""

    normalized = tuple(str(label).strip().upper() for label in labels)
    expected_count = len(all_codes())
    if len(normalized) != expected_count:
        raise ValueError(f"expected exactly {expected_count} terminal labels")
    if len(set(normalized)) != len(normalized):
        raise ValueError("terminal labels must be unique")
    if any((not label or len(label) > 3) for label in normalized):
        raise ValueError("terminal labels must be compact non-empty strings")
    return normalized


__all__ = ["all_codes", "code_to_index", "index_to_code", "terminal_label_pool", "validate_terminal_labels"]
