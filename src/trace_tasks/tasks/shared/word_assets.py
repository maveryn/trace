"""Shared short alphabetic word-like label helpers.

These helpers derive lowercase, single-token alphabetic strings from the
repo-wide label manifests. They are intended for visual notation scenes that
need readable text tokens without owning a scene-local word bank.
"""

from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

from .name_assets import load_label_manifest


DEFAULT_SHORT_WORD_MANIFEST = "people/first_names_ssa.txt"
DEFAULT_MAX_WORDS_PER_LENGTH = 1000


@lru_cache(maxsize=16)
def load_short_word_bank_by_length(
    *,
    manifest_name: str = DEFAULT_SHORT_WORD_MANIFEST,
    min_length: int = 3,
    max_length: int = 5,
    max_words_per_length: int = DEFAULT_MAX_WORDS_PER_LENGTH,
) -> Mapping[int, tuple[str, ...]]:
    """Return lowercase alphabetic labels grouped by visible string length."""

    min_len = int(min_length)
    max_len = int(max_length)
    if min_len < 1 or max_len < min_len:
        raise ValueError("invalid short-word length bounds")
    max_per_length = int(max_words_per_length)
    if max_per_length < 1:
        raise ValueError("max_words_per_length must be positive")

    raw_labels = load_label_manifest(
        str(manifest_name),
        min_chars=min_len,
        max_chars=max_len,
        allow_spaces=False,
        allow_punctuation=False,
        ascii_only=True,
    )
    grouped: dict[int, list[str]] = {length: [] for length in range(min_len, max_len + 1)}
    seen: set[str] = set()
    for raw_label in raw_labels:
        word = str(raw_label).strip().lower()
        if not word.isalpha() or word in seen:
            continue
        length = len(word)
        if min_len <= length <= max_len:
            if len(grouped.setdefault(length, [])) >= max_per_length:
                continue
            grouped.setdefault(length, []).append(word)
            seen.add(word)

    frozen = {length: tuple(words) for length, words in grouped.items() if words}
    if not frozen:
        raise ValueError(f"manifest {manifest_name!r} yielded no short alphabetic words")
    return MappingProxyType(frozen)


def load_short_words_for_length(
    length: int,
    *,
    manifest_name: str = DEFAULT_SHORT_WORD_MANIFEST,
    min_length: int = 3,
    max_length: int = 5,
    max_words_per_length: int = DEFAULT_MAX_WORDS_PER_LENGTH,
) -> tuple[str, ...]:
    """Return the shared short-word pool for one exact length."""

    bank = load_short_word_bank_by_length(
        manifest_name=str(manifest_name),
        min_length=int(min_length),
        max_length=int(max_length),
        max_words_per_length=int(max_words_per_length),
    )
    return tuple(bank.get(int(length), ()))


__all__ = [
    "DEFAULT_SHORT_WORD_MANIFEST",
    "DEFAULT_MAX_WORDS_PER_LENGTH",
    "load_short_word_bank_by_length",
    "load_short_words_for_length",
]
