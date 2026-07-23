"""Passive state containers for wallpaper-panel rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class WallpaperScenePayload:
    """Trace-ready scene payload returned by neutral wallpaper rendering."""

    option_count: int
    option_labels: Tuple[str, ...]
    icon_ids_by_label: Dict[str, str]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    nominal_icon_size_px: int
    panel_geometry: Dict[str, Any]
    scene_panels: Tuple[Dict[str, Any], ...]
    scene_elements: Tuple[Dict[str, Any], ...]
    scene_icon_instances: Tuple[Dict[str, Any], ...]


__all__ = ["WallpaperScenePayload"]
