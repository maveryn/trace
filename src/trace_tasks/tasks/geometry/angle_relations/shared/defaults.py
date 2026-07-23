"""Scene-local defaults for geometry angle-relations tasks."""

from __future__ import annotations

from .state import DOMAIN, SCENE_ID

PROMPT_BUNDLE_ID = "geometry_angle_relations_v1"

FALLBACK_PROMPT_WIRING_KEYS = (
    "bundle_id",
    "scene_key",
    "task_key",
)

__all__ = [
    "DOMAIN",
    "FALLBACK_PROMPT_WIRING_KEYS",
    "PROMPT_BUNDLE_ID",
    "SCENE_ID",
]
