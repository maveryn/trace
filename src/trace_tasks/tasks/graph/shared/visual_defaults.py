"""Graph-domain visual-default loader helpers."""

from __future__ import annotations

from typing import Any, Dict

from ...shared.visual_defaults import (
    default_noise_fallback,
    load_scene_background_defaults,
    load_scene_noise_defaults,
)


def solid_light_background_fallback() -> Dict[str, Any]:
    """Return the canonical low-structure graph background fallback."""

    return {
        "enabled": True,
        "styles": {
            "solid_light": {
                "kind": "solid",
                "color": [248, 248, 248],
            },
            "solid_cool": {
                "kind": "solid",
                "color": [243, 246, 252],
            },
            "solid_warm": {
                "kind": "solid",
                "color": [250, 247, 242],
            },
            "solid_mint": {
                "kind": "solid",
                "color": [244, 249, 246],
            },
            "solid_lavender": {
                "kind": "solid",
                "color": [248, 246, 253],
            }
        },
        "weights": {
            "solid_light": 1.0,
            "solid_cool": 1.0,
            "solid_warm": 1.0,
            "solid_mint": 1.0,
            "solid_lavender": 1.0,
        },
    }


def load_graph_scene_background_defaults(*, scene_id: str) -> Dict[str, Any]:
    """Load graph-domain background defaults for one scene."""

    return load_scene_background_defaults(
        domain="graph",
        scene_id=str(scene_id),
        fallback=solid_light_background_fallback(),
        merge_with_fallback=True,
    )


def load_graph_scene_noise_defaults(*, scene_id: str, apply_prob: float) -> Dict[str, Any]:
    """Load graph-domain noise defaults for one scene."""

    return load_scene_noise_defaults(
        domain="graph",
        scene_id=str(scene_id),
        fallback=default_noise_fallback(apply_prob=float(apply_prob)),
        merge_with_fallback=False,
    )


__all__ = [
    "load_graph_scene_background_defaults",
    "load_graph_scene_noise_defaults",
    "solid_light_background_fallback",
]
