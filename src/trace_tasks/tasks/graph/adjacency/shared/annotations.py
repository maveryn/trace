"""Annotation projection helpers for adjacency-scene tasks."""

from __future__ import annotations

from typing import Sequence, Tuple

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_set_annotation_artifacts,
    segment_set_annotation_artifacts,
)

from .state import AdjacencyGraphSample, AdjacencyRepresentationRender, matrix_cell_key


def component_topmost_row_labels(
    labels: Sequence[str],
    components: Sequence[Sequence[str]],
) -> Tuple[str, ...]:
    """Choose the topmost displayed row label from each component."""

    order = {str(label): int(index) for index, label in enumerate(labels)}
    sorted_components = sorted(components, key=lambda component: min(order[str(label)] for label in component))
    return tuple(
        min((str(label) for label in component), key=lambda label: order[str(label)])
        for component in sorted_components
    )


def row_label_bbox_set_artifacts(
    rendered: AdjacencyRepresentationRender,
    labels: Sequence[str],
) -> AnnotationArtifacts:
    """Project row-label ids to public bbox-set annotation artifacts."""

    return bbox_set_annotation_artifacts(
        [[round(float(value), 3) for value in rendered.row_label_bboxes[str(label)]] for label in labels]
    )


def row_label_bbox_sequence_value(
    rendered: AdjacencyRepresentationRender,
    labels: Sequence[str],
) -> list[list[float]]:
    """Project row-label ids to an ordered bbox-sequence value."""

    return [
        [round(float(value), 3) for value in rendered.row_label_bboxes[str(label)]]
        for label in labels
    ]


def mirrored_pair_cell_bbox_artifacts(
    rendered: AdjacencyRepresentationRender,
    pairs: Sequence[tuple[str, str]],
) -> tuple[AnnotationArtifacts, tuple[str, ...]]:
    """Project counted directed-pair cells to bbox-set artifacts."""

    cell_keys: list[str] = []
    bboxes: list[list[float]] = []
    for left, right in pairs:
        forward_key = matrix_cell_key(str(left), str(right))
        reverse_key = matrix_cell_key(str(right), str(left))
        cell_keys.extend([forward_key, reverse_key])
        bboxes.append([round(float(value), 3) for value in rendered.cell_bboxes[forward_key]])
        bboxes.append([round(float(value), 3) for value in rendered.cell_bboxes[reverse_key]])
    return bbox_set_annotation_artifacts(bboxes), tuple(cell_keys)


def _bbox_center(bbox: Sequence[float]) -> list[float]:
    return [
        (float(bbox[0]) + float(bbox[2])) / 2.0,
        (float(bbox[1]) + float(bbox[3])) / 2.0,
    ]


def mirrored_pair_cell_point_pair_artifacts(
    rendered: AdjacencyRepresentationRender,
    pairs: Sequence[tuple[str, str]],
) -> tuple[AnnotationArtifacts, tuple[str, ...]]:
    """Project counted mirrored matrix cells to one segment per node pair."""

    cell_keys: list[str] = []
    point_pairs: list[list[list[float]]] = []
    for left, right in pairs:
        forward_key = matrix_cell_key(str(left), str(right))
        reverse_key = matrix_cell_key(str(right), str(left))
        cell_keys.extend([forward_key, reverse_key])
        point_pairs.append(
            [
                _bbox_center(rendered.cell_bboxes[forward_key]),
                _bbox_center(rendered.cell_bboxes[reverse_key]),
            ]
        )
    return segment_set_annotation_artifacts(point_pairs), tuple(cell_keys)


def mst_cell_bbox_artifacts(
    sample: AdjacencyGraphSample,
    rendered: AdjacencyRepresentationRender,
) -> tuple[AnnotationArtifacts, Tuple[Tuple[str, str], ...]]:
    """Project one visible weighted-matrix cell for each MST edge."""

    order = {str(label): int(index) for index, label in enumerate(sample.labels)}
    cells: list[list[float]] = []
    cell_edges: list[Tuple[str, str]] = []
    for left, right in sample.mst_edges:
        if order[str(left)] <= order[str(right)]:
            row_label, column_label = str(left), str(right)
        else:
            row_label, column_label = str(right), str(left)
        cells.append([round(float(value), 3) for value in rendered.cell_bboxes[matrix_cell_key(row_label, column_label)]])
        cell_edges.append((str(row_label), str(column_label)))
    return bbox_set_annotation_artifacts(cells), tuple(cell_edges)


__all__ = [
    "component_topmost_row_labels",
    "mirrored_pair_cell_bbox_artifacts",
    "mirrored_pair_cell_point_pair_artifacts",
    "mst_cell_bbox_artifacts",
    "row_label_bbox_sequence_value",
    "row_label_bbox_set_artifacts",
]
