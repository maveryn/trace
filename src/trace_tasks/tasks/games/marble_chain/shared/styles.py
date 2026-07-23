"""Color palettes for marble-chain rendering."""

from __future__ import annotations

from typing import Dict, Tuple


COLOR_RGB: Dict[str, Tuple[int, int, int]] = {
    "red": (216, 57, 64),
    "blue": (58, 118, 218),
    "green": (54, 158, 99),
    "yellow": (238, 190, 52),
    "purple": (144, 92, 205),
    "orange": (230, 126, 51),
}

MARBLE_CHAIN_STYLE_RGB: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "classic_track": {
        "track": (177, 181, 184),
        "rail": (83, 91, 101),
        "arrow": (36, 99, 164),
        "shooter_body": (242, 244, 248),
    },
    "arcade_track": {
        "track": (75, 91, 125),
        "rail": (21, 31, 54),
        "arrow": (242, 198, 56),
        "shooter_body": (31, 43, 70),
    },
    "neon_track": {
        "track": (64, 53, 102),
        "rail": (21, 19, 42),
        "arrow": (71, 215, 205),
        "shooter_body": (42, 37, 73),
    },
    "chalk_track": {
        "track": (191, 196, 182),
        "rail": (78, 90, 76),
        "arrow": (188, 83, 69),
        "shooter_body": (235, 232, 214),
    },
    "copper_track": {
        "track": (174, 122, 82),
        "rail": (93, 57, 38),
        "arrow": (37, 112, 124),
        "shooter_body": (236, 210, 171),
    },
}


__all__ = ["COLOR_RGB", "MARBLE_CHAIN_STYLE_RGB"]
