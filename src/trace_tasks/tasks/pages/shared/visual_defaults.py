"""Pages-domain visual-default loader helpers."""

from __future__ import annotations

from typing import Any, Dict

from ...shared.visual_defaults import (
    default_noise_fallback,
    load_scene_background_defaults,
    load_scene_noise_defaults,
)


def paper_background_fallback() -> Dict[str, Any]:
    """Return the canonical low-structure paper-like background fallback."""

    return {
        "enabled": True,
        "styles": {
            "paper_light": {
                "kind": "solid",
                "color": [241, 241, 238],
            }
        },
        "weights": {"paper_light": 1.0},
    }


def load_pages_background_defaults(*, scene_id: str) -> Dict[str, Any]:
    """Load pages-task background config with the canonical fallback."""

    return load_scene_background_defaults(
        domain="pages",
        scene_id=str(scene_id),
        fallback=paper_background_fallback(),
        merge_with_fallback=True,
    )


def load_pages_scene_background_defaults(*, scene_id: str) -> Dict[str, Any]:
    """Load pages scene background config with the canonical fallback."""

    return load_scene_background_defaults(
        domain="pages",
        scene_id=str(scene_id),
        fallback=paper_background_fallback(),
        merge_with_fallback=True,
    )


def load_pages_noise_defaults(*, scene_id: str, apply_prob: float) -> Dict[str, Any]:
    """Load pages-task post-image noise config with the canonical fallback."""

    return load_scene_noise_defaults(
        domain="pages",
        scene_id=str(scene_id),
        fallback=default_noise_fallback(apply_prob=float(apply_prob)),
        merge_with_fallback=False,
    )


def load_pages_scene_noise_defaults(*, scene_id: str, apply_prob: float) -> Dict[str, Any]:
    """Load pages scene post-image noise config with the canonical fallback."""

    return load_scene_noise_defaults(
        domain="pages",
        scene_id=str(scene_id),
        fallback=default_noise_fallback(apply_prob=float(apply_prob)),
        merge_with_fallback=False,
    )


__all__ = [
    "load_pages_background_defaults",
    "load_pages_noise_defaults",
    "load_pages_scene_background_defaults",
    "load_pages_scene_noise_defaults",
    "paper_background_fallback",
]
