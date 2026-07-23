"""Puzzle-domain visual-default loader helpers."""

from __future__ import annotations

from typing import Any, Dict

from ...shared.visual_defaults import default_noise_fallback, load_scene_background_defaults, load_scene_noise_defaults


def solid_light_background_fallback() -> Dict[str, Any]:
    """Return the canonical low-structure puzzle background fallback."""

    return {
        "enabled": True,
        "styles": {
            "solid_light": {
                "kind": "solid",
                "color": [248, 248, 248],
            }
        },
        "weights": {"solid_light": 1.0},
    }


def load_puzzle_background_defaults(*, scene_id: str) -> Dict[str, Any]:
    """Load puzzle-task background config with the canonical fallback."""

    return load_scene_background_defaults(
        domain="puzzles",
        scene_id=str(scene_id),
        fallback=solid_light_background_fallback(),
        merge_with_fallback=True,
    )


def load_puzzle_noise_defaults(*, scene_id: str, apply_prob: float) -> Dict[str, Any]:
    """Load puzzle-task post-image noise config with the canonical fallback."""

    return load_scene_noise_defaults(
        domain="puzzles",
        scene_id=str(scene_id),
        fallback=default_noise_fallback(apply_prob=float(apply_prob)),
        merge_with_fallback=False,
    )


__all__ = [
    "load_puzzle_background_defaults",
    "load_puzzle_noise_defaults",
    "solid_light_background_fallback",
]
