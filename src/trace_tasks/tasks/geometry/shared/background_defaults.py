"""Background defaults shared across geometry domain tasks."""

from __future__ import annotations

from typing import Any, Dict

from ...shared.visual_defaults import (
    load_domain_background_defaults,
    load_scene_background_defaults,
)
from .graph_rendering import FALLBACK_GRAPH_STYLE


def _fallback_background_defaults() -> Dict[str, Any]:
    """Return safe fallback defaults if config is missing or invalid."""
    return {
        "enabled": True,
        "styles": {"graph_paper": dict(FALLBACK_GRAPH_STYLE)},
        "weights": {"graph_paper": 1.0},
    }


def load_geometry_background_defaults(*, scene_id: str | None = None) -> Dict[str, Any]:
    """Load geometry background defaults with optional scene-package override support."""
    fallback = _fallback_background_defaults()
    if scene_id is not None and str(scene_id).strip():
        return load_scene_background_defaults(
            domain="geometry",
            scene_id=str(scene_id),
            fallback=fallback,
            merge_with_fallback=True,
        )
    return load_domain_background_defaults(
        domain="geometry",
        fallback=fallback,
        merge_with_fallback=True,
    )


# Geometry background defaults, sourced from domain config unless a scene
# override is provided by a caller.
POST_IMAGE_BACKGROUND_DEFAULTS: Dict[str, Any] = load_geometry_background_defaults()
