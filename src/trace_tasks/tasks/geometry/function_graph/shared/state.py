"""Data records for function-graph scene construction and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

GraphPoint = Tuple[int, int]
GraphPolylinePoint = Tuple[float, float]


@dataclass(frozen=True)
class SampledFunctionGraph:
    """A plotted function before raster rendering."""

    polyline_graph: Tuple[GraphPolylinePoint, ...]
    annotation_graph_points: Tuple[GraphPoint, ...]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    execution_trace: Dict[str, Any]
    object_count: int


@dataclass(frozen=True)
class RenderedFunctionGraph:
    """A rendered function graph plus annotation payload pieces."""

    answer_value: int
    annotation_type: str
    annotation_value: List[List[float]]
    projected_annotation: Dict[str, Any]
    witness_symbolic: Dict[str, Any]
    required_annotation_labels: List[str]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    execution_trace: Dict[str, Any]
    object_count: int
