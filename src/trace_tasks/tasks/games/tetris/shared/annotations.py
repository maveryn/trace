"""Annotation projection helpers for Tetris board scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_annotation_artifacts,
    bbox_set_annotation_artifacts,
)

from .state import RenderedTetrisScene, TetrisSample


@dataclass(frozen=True)
class TetrisAnnotationBundle:
    """Task-bound annotation plus symbolic entity ids."""

    annotation_gt: TypedValue
    projected_annotation: Mapping[str, Any]
    witness_symbolic: Mapping[str, Any]


def _round_bbox(bbox: Sequence[float]) -> list[float]:
    return [round(float(value), 3) for value in bbox[:4]]


def _entity_bboxes(rendered: RenderedTetrisScene, entity_ids: Sequence[str]) -> list[list[float]]:
    """Project row, cell, board, preview, or option ids to rendered boxes."""

    render_map = rendered.render_map
    bboxes: list[list[float]] = []
    for entity_id in entity_ids:
        key = str(entity_id)
        if key in render_map["cell_bboxes_px"]:
            bboxes.append(_round_bbox(render_map["cell_bboxes_px"][key]))
        elif key in render_map["row_bboxes_px"]:
            bboxes.append(_round_bbox(render_map["row_bboxes_px"][key]))
        elif key in render_map["option_bboxes_px"]:
            bboxes.append(_round_bbox(render_map["option_bboxes_px"][key]))
        elif key in render_map["panels"]:
            bboxes.append(_round_bbox(render_map["panels"][key]))
        else:
            raise ValueError(f"unknown Tetris annotation entity id: {key}")
    return bboxes


def _union_bbox(bboxes: Sequence[Sequence[float]]) -> list[float]:
    """Return one bbox enclosing all projected witness boxes."""

    if not bboxes:
        raise ValueError("cannot union an empty Tetris bbox list")
    return [
        round(min(float(bbox[0]) for bbox in bboxes), 3),
        round(min(float(bbox[1]) for bbox in bboxes), 3),
        round(max(float(bbox[2]) for bbox in bboxes), 3),
        round(max(float(bbox[3]) for bbox in bboxes), 3),
    ]


def tetris_annotation_bundle(sample: TetrisSample, rendered: RenderedTetrisScene) -> TetrisAnnotationBundle:
    """Build the public annotation payload for one task-bound Tetris sample."""

    kind = str(sample.annotation_kind)
    entity_ids = tuple(str(value) for value in sample.annotation_entity_ids)
    if kind == "board_and_next_piece":
        value = {
            "board": _entity_bboxes(rendered, ("main",))[0],
            "next_piece": _entity_bboxes(rendered, ("next_piece",))[0],
        }
        projected = {
            "type": "bbox_map",
            "bbox_map": dict(value),
            "pixel_bbox_map": dict(value),
        }
        return TetrisAnnotationBundle(
            annotation_gt=TypedValue(type="bbox_map", value=dict(value)),
            projected_annotation=projected,
            witness_symbolic={"type": "bbox_map", "ids": {"board": "main", "next_piece": "next_piece"}},
        )
    if kind == "collision_keyed_cell_sets":
        raw_map = sample.metadata.get("annotation_entity_id_map", {})
        if not isinstance(raw_map, Mapping):
            raise ValueError("Tetris collision annotation requires entity id maps")
        value = {
            str(role): _entity_bboxes(rendered, tuple(str(entity_id) for entity_id in ids))
            for role, ids in raw_map.items()
        }
        projected = {
            "type": "bbox_set_map",
            "bbox_set_map": dict(value),
            "pixel_bbox_set_map": dict(value),
        }
        return TetrisAnnotationBundle(
            annotation_gt=TypedValue(type="bbox_set_map", value=dict(value)),
            projected_annotation=projected,
            witness_symbolic={
                "type": "bbox_set_map",
                "ids": {str(role): [str(entity_id) for entity_id in ids] for role, ids in raw_map.items()},
            },
        )
    if kind == "option_panel":
        if len(entity_ids) != 1:
            raise ValueError("Tetris option-panel annotation requires one selected option")
        artifacts = bbox_annotation_artifacts(_entity_bboxes(rendered, entity_ids)[0])
        return TetrisAnnotationBundle(
            annotation_gt=artifacts.annotation_gt,
            projected_annotation=dict(artifacts.projected_annotation),
            witness_symbolic={"type": artifacts.annotation_type, "ids": [entity_ids[0]]},
        )
    if kind == "active_piece_bbox":
        artifacts = bbox_annotation_artifacts(_union_bbox(_entity_bboxes(rendered, entity_ids)))
        return TetrisAnnotationBundle(
            annotation_gt=artifacts.annotation_gt,
            projected_annotation=dict(artifacts.projected_annotation),
            witness_symbolic={"type": artifacts.annotation_type, "ids": [str(entity_id) for entity_id in entity_ids]},
        )
    if kind in {"row_set", "cell_set"}:
        artifacts = bbox_set_annotation_artifacts(_entity_bboxes(rendered, entity_ids))
        return TetrisAnnotationBundle(
            annotation_gt=artifacts.annotation_gt,
            projected_annotation=dict(artifacts.projected_annotation),
            witness_symbolic={"type": artifacts.annotation_type, "ids": [str(entity_id) for entity_id in entity_ids]},
        )
    raise ValueError(f"unsupported Tetris annotation kind: {kind}")


__all__ = ["TetrisAnnotationBundle", "tetris_annotation_bundle"]
