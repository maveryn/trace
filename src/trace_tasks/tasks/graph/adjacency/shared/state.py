"""State contracts and stable ids for the graph adjacency scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from PIL import Image


SCENE_ID = "adjacency"
SUPPORTED_ADJACENCY_REPRESENTATION_VARIANTS: Tuple[str, ...] = (
    "adjacency_list_panel",
    "adjacency_matrix_panel",
)


@dataclass(frozen=True)
class AdjacencyLabelSet:
    """Resolved graph labels plus provenance metadata."""

    labels: Tuple[str, ...]
    label_variant: str
    label_source_kind: str
    label_bucket: str
    label_manifest: str
    label_filter: Mapping[str, Any]
    label_bucket_probabilities: Mapping[str, float]


@dataclass(frozen=True)
class AdjacencyGraphSample:
    """Symbolic graph represented by an adjacency list or matrix."""

    labels: Tuple[str, ...]
    directed: bool
    adjacency: Mapping[str, Tuple[str, ...]]
    edges: Tuple[Tuple[str, str], ...]
    weights: Mapping[Tuple[str, str], int]
    components: Tuple[Tuple[str, ...], ...] = ()
    mst_edges: Tuple[Tuple[str, str], ...] = ()
    mst_weight: int = 0


@dataclass(frozen=True)
class AdjacencyRepresentationRender:
    """Rendered adjacency representation and pixel annotation anchors."""

    image: Image.Image
    representation_variant: str
    panel_bbox: list[float]
    node_label_bboxes: Mapping[str, list[float]]
    row_label_bboxes: Mapping[str, list[float]]
    column_label_bboxes: Mapping[str, list[float]]
    cell_bboxes: Mapping[str, list[float]]
    panel_geometry: Mapping[str, Any]
    style_meta: Mapping[str, Any]


def canonical_undirected_edge(left: str, right: str) -> Tuple[str, str]:
    """Return one deterministic undirected edge key."""

    a = str(left)
    b = str(right)
    return (a, b) if a <= b else (b, a)


def matrix_cell_key(row_label: str, column_label: str) -> str:
    """Return stable string key for a rendered matrix cell."""

    return f"{str(row_label)}||{str(column_label)}"


__all__ = [
    "SCENE_ID",
    "SUPPORTED_ADJACENCY_REPRESENTATION_VARIANTS",
    "AdjacencyGraphSample",
    "AdjacencyLabelSet",
    "AdjacencyRepresentationRender",
    "canonical_undirected_edge",
    "matrix_cell_key",
]
