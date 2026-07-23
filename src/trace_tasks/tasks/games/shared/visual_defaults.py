"""Games-domain visual-default loader helpers."""

from __future__ import annotations

from typing import Any, Dict

from ...shared.visual_defaults import (
    default_noise_fallback,
    load_scene_background_defaults,
    load_scene_noise_defaults,
)


def solid_table_background_fallback() -> Dict[str, Any]:
    """Return the canonical low-structure games background fallback."""

    return {
        "enabled": True,
        "styles": {
            "felt_green": {
                "kind": "solid",
                "color": [42, 92, 74],
            },
            "felt_deep_green": {
                "kind": "solid",
                "color": [34, 78, 66],
            },
            "felt_blue_green": {
                "kind": "solid",
                "color": [43, 88, 94],
            },
            "felt_olive": {
                "kind": "solid",
                "color": [68, 90, 58],
            },
            "table_slate": {
                "kind": "solid",
                "color": [58, 72, 78],
            }
        },
        "weights": {
            "felt_green": 1.0,
            "felt_deep_green": 1.0,
            "felt_blue_green": 1.0,
            "felt_olive": 1.0,
            "table_slate": 1.0,
        },
    }


def load_games_scene_background_defaults(*, scene_id: str) -> Dict[str, Any]:
    """Load games-scene background defaults."""

    return load_scene_background_defaults(
        domain="games",
        scene_id=str(scene_id),
        fallback=solid_table_background_fallback(),
        merge_with_fallback=True,
    )


def load_games_scene_noise_defaults(*, scene_id: str, apply_prob: float) -> Dict[str, Any]:
    """Load games-scene post-image noise defaults."""

    return load_scene_noise_defaults(
        domain="games",
        scene_id=str(scene_id),
        fallback=default_noise_fallback(apply_prob=float(apply_prob)),
        merge_with_fallback=False,
    )


__all__ = [
    "load_games_scene_background_defaults",
    "load_games_scene_noise_defaults",
    "solid_table_background_fallback",
]
