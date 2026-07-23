"""Annotation projection helpers for Sokoban task witnesses."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_annotation_artifacts,
    bbox_set_annotation_artifacts,
)

from .rules import cell_id, option_id
from .state import RenderedSokobanScene


def option_panel_bbox(rendered_scene: RenderedSokobanScene, *, answer_label: str) -> AnnotationArtifacts:
    """Return the scalar bbox for the selected visible option panel."""

    bbox = rendered_scene.option_panel_bbox_map[option_id(str(answer_label))]
    return bbox_annotation_artifacts(bbox)


def option_cell_bbox(
    rendered_scene: RenderedSokobanScene,
    *,
    option_specs: Sequence[Mapping[str, Any]],
    answer_label: str,
) -> AnnotationArtifacts:
    """Return the scalar bbox for the selected option-marked board cell."""

    cells = _candidate_cells_for_answer(option_specs, answer_label=answer_label)
    if len(cells) != 1:
        raise ValueError(f"expected one candidate cell for option {answer_label}, got {len(cells)}")
    bbox = rendered_scene.cell_bbox_map[cell_id(tuple(cells[0]))]
    return bbox_annotation_artifacts(bbox)


def option_pair_bbox_set(
    rendered_scene: RenderedSokobanScene,
    *,
    option_specs: Sequence[Mapping[str, Any]],
    answer_label: str,
) -> AnnotationArtifacts:
    """Return the two cell bboxes for a selected same-letter box-target pair."""

    cells = _candidate_cells_for_answer(option_specs, answer_label=answer_label)
    if len(cells) != 2:
        raise ValueError(f"expected two candidate cells for option {answer_label}, got {len(cells)}")
    return bbox_set_annotation_artifacts(
        [rendered_scene.cell_bbox_map[cell_id(tuple(cell))] for cell in cells]
    )


def _candidate_cells_for_answer(
    option_specs: Sequence[Mapping[str, Any]],
    *,
    answer_label: str,
) -> list[tuple[int, int]]:
    for option in option_specs:
        if str(option.get("option_label")) == str(answer_label):
            return [tuple(int(value) for value in cell) for cell in option.get("candidate_cells", [])]
    raise ValueError(f"missing option spec for answer label {answer_label}")


def cell_bbox_set(
    rendered_scene: RenderedSokobanScene,
    *,
    cells: Sequence[Sequence[int]],
) -> AnnotationArtifacts:
    """Return an unordered bbox-set for a homogeneous set of grid cells."""

    return bbox_set_annotation_artifacts(
        [
            rendered_scene.cell_bbox_map[cell_id(tuple(int(value) for value in cell))]
            for cell in cells
        ]
    )


__all__ = ["cell_bbox_set", "option_cell_bbox", "option_pair_bbox_set", "option_panel_bbox"]
