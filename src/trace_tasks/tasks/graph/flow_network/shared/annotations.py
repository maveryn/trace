"""Annotation projection helpers for flow-network diagrams."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from ...shared.graph_scene import RenderedGraphScene, projected_edge_pair_annotation


def project_min_cut_segments(
    rendered_scene: RenderedGraphScene,
    edges: Sequence[Tuple[str, str]],
) -> Tuple[Mapping[str, Any], Tuple[Any, ...]]:
    """Project directed cut edges to unordered pixel-space segment witnesses."""

    annotation_edges = tuple((str(left), str(right)) for left, right in edges)
    projection = projected_edge_pair_annotation(rendered_scene, annotation_edges)
    segments = tuple([list(point) for point in segment] for segment in projection["segment_set"])
    if len(segments) != len(annotation_edges):
        raise RuntimeError("flow-network min-cut annotation projection is incomplete")
    return dict(projection), tuple(segments)


__all__ = ["project_min_cut_segments"]
