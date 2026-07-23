"""Passive state records for paired-canvas icon scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


@dataclass(frozen=True)
class PairedIconSpec:
    """One icon instance to render in either paired-canvas panel."""

    instance_id: str
    identity_id: str
    icon_id: str
    panel: str
    x_frac: float
    y_frac: float
    nominal_size_px: int
    rotation_degrees: int
    tint_rgb: Tuple[int, int, int]
    noise_edits: Tuple[Any, ...] = ()
    noise_seed: int | None = None


@dataclass(frozen=True)
class RenderedPairedIcon:
    """Trace-ready rendered paired-canvas icon metadata."""

    instance_id: str
    identity_id: str
    icon_id: str
    panel: str
    bbox_xyxy: Tuple[int, int, int, int]
    center_xy: Tuple[float, float]
    normalized_center_xy: Tuple[float, float]
    nominal_size_px: int
    rotation_degrees: int
    tint_rgb: Tuple[int, int, int]
    noise_edits: Tuple[Dict[str, Any], ...]
    noise_seed: int | None
    role: str = "neutral"
    changed_attributes: Tuple[str, ...] = ()
    movement_direction: str | None = None


@dataclass(frozen=True)
class PairedCanvasRenderPayload:
    """Rendered image plus neutral trace payloads for a paired-canvas scene."""

    image: Image.Image
    panel_geometry: Dict[str, Any]
    left_icons: Tuple[Dict[str, Any], ...]
    right_icons: Tuple[Dict[str, Any], ...]


__all__ = ["PairedCanvasRenderPayload", "PairedIconSpec", "RenderedPairedIcon"]
