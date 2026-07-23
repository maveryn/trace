"""Annotation projection helpers for graph automaton scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ...shared.graph_scene import RenderedGraphScene, projected_node_point_annotation


def projected_state_path_points(
    rendered_scene: RenderedGraphScene,
    state_labels: Sequence[str],
) -> tuple[list[list[int]], Mapping[str, Any]]:
    """Project an ordered state-label path to public point-sequence annotation."""

    projection = projected_node_point_annotation(rendered_scene, tuple(str(label) for label in state_labels))
    annotation_path = [
        [int(round(float(point[0]))), int(round(float(point[1])))]
        for point in projection["pixel_point_sequence"]
    ]
    if len(annotation_path) != len(tuple(state_labels)):
        raise RuntimeError("automaton state-path annotation projection is incomplete")
    return annotation_path, dict(projection)


__all__ = ["projected_state_path_points"]
