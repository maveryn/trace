"""Common games-domain style helpers."""

from __future__ import annotations

from typing import Dict, Tuple


SUPPORTED_GAMES_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "soft",
    "outlined",
)


def style_probability_map() -> Dict[str, float]:
    """Return one stable uniform style-probability map."""

    probability = 1.0 / float(len(SUPPORTED_GAMES_STYLE_VARIANTS))
    return {str(name): float(probability) for name in SUPPORTED_GAMES_STYLE_VARIANTS}
