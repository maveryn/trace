"""Passive state containers for concept-map scene packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from PIL import Image


@dataclass(frozen=True)
class ConceptMapCase:
    """Resolved concept-map scene plus the selected objective target."""

    case_kind: str
    scene: Dict[str, Any]
    selection: Dict[str, Any]
    context_probabilities: Dict[str, float]
    layout_probabilities: Dict[str, float]
    style_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedConceptMap:
    """Rendered concept-map image and projected geometry."""

    image: Image.Image
    render_map: Dict[str, Any]
    background_meta: Mapping[str, Any]
    post_noise_meta: Mapping[str, Any]
