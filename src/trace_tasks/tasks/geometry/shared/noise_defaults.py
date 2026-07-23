"""Visual-variation defaults shared across geometry domain tasks."""

from __future__ import annotations

from typing import Any, Dict

from ...shared.visual_defaults import (
    default_noise_fallback,
    load_domain_noise_defaults,
    load_scene_noise_defaults,
)


def _fallback_noise_defaults() -> Dict[str, Any]:
    """Return safe fallback defaults if config is missing or invalid."""
    return default_noise_fallback(apply_prob=0.5)


def load_geometry_noise_defaults(*, scene_id: str | None = None) -> Dict[str, Any]:
    """Load geometry noise defaults with optional scene-package override support."""
    fallback = _fallback_noise_defaults()
    if scene_id is not None and str(scene_id).strip():
        return load_scene_noise_defaults(
            domain="geometry",
            scene_id=str(scene_id),
            fallback=fallback,
            merge_with_fallback=False,
        )
    return load_domain_noise_defaults(
        domain="geometry",
        fallback=fallback,
        merge_with_fallback=False,
    )


# Geometry post-image noise defaults, sourced from domain config unless a scene
# override is provided by a caller.
POST_IMAGE_NOISE_DEFAULTS: Dict[str, Any] = load_geometry_noise_defaults()
