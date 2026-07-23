"""Intermediate records for pipe-network scene generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class PipeResolvedAxes:
    """Resolved support/style axes for one pipe-network instance."""

    node_count: int
    grid_shape_variant: str
    label_variant: str
    node_color_name: str
    target_value: int = 0
    query_distance: int = 0
    node_count_probabilities: Dict[str, float] | None = None
    grid_shape_variant_probabilities: Dict[str, float] | None = None
    label_variant_probabilities: Dict[str, float] | None = None
    node_color_name_probabilities: Dict[str, float] | None = None
    target_value_probabilities: Dict[str, float] | None = None
    query_distance_probabilities: Dict[str, float] | None = None


@dataclass(frozen=True)
class RenderedPipeAssets:
    """Rendered pipe image plus metadata needed by public task binding."""

    image: Any
    rendered_scene: Any
    render_params: Any
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


@dataclass(frozen=True)
class PipeBoundResult:
    """Task-owned answer, annotation, and semantic trace fields."""

    answer_type: str
    answer_value: Any
    annotation_type: str
    annotation_value: Any
    prompt_slots: Dict[str, Any]
    trace_params: Dict[str, Any]
    scene_relations: Dict[str, Any]
    execution_trace: Dict[str, Any]
    witness_symbolic: Dict[str, Any]
    projected_annotation: Dict[str, Any]
    entities: Any | None = None


@dataclass(frozen=True)
class PipePreparedInstance:
    """Prompt, image, and trace payload after scene-level assembly."""

    prompt: str
    image: Any
    trace_payload: Dict[str, Any]
    prompt_variants: Mapping[str, Any]


__all__ = [
    "PipeBoundResult",
    "PipePreparedInstance",
    "PipeResolvedAxes",
    "RenderedPipeAssets",
]
