"""Intermediate output records for metro scene generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class MetroRouteResolvedAxes:
    """Resolved support/style axes for one metro-route graph instance."""

    target_count: int
    route_count: int
    label_variant: str
    node_color_name: str
    query_distance: int = 0
    target_count_probabilities: Dict[str, float] | None = None
    route_count_probabilities: Dict[str, float] | None = None
    label_variant_probabilities: Dict[str, float] | None = None
    node_color_name_probabilities: Dict[str, float] | None = None


@dataclass(frozen=True)
class MetroAnswerAnnotation:
    """Public answer plus pixel annotation values bound by a task file."""

    answer_value: int
    annotation_type: str
    annotation_value: list[Any]
    witness_symbolic: Dict[str, Any]
    projected_annotation: Dict[str, Any]


@dataclass(frozen=True)
class MetroPreparedAssets:
    """Prompt, image, annotation, and trace metadata after shared metro rendering."""

    prompt: str
    image: Any
    answer_annotation: MetroAnswerAnnotation
    trace_payload: Dict[str, Any]
    prompt_variants: Mapping[str, Any]


__all__ = ["MetroAnswerAnnotation", "MetroPreparedAssets", "MetroRouteResolvedAxes"]
