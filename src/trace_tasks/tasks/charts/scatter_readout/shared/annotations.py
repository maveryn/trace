"""Annotation helpers for scatter-readout chart scenes."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.annotation_artifacts import point_annotation_artifacts
from trace_tasks.tasks.shared.annotation_artifacts import segment_annotation_artifacts

from .state import QueryBinding, ScatterReadoutRenderResult


def _bbox_center(bbox: list[float]) -> list[float]:
    return [
        (float(bbox[0]) + float(bbox[2])) / 2.0,
        (float(bbox[1]) + float(bbox[3])) / 2.0,
    ]


def _empty_map_annotation() -> tuple[AnnotationArtifacts, dict[str, Any]]:
    projected = {
        "type": "bbox_map",
        "bbox_map": {},
        "pixel_bbox_map": {},
        "bbox_set": [],
        "point_id": "",
        "point_ids": [],
    }
    artifacts = AnnotationArtifacts(
        annotation_type="bbox_map",
        value={},
        annotation_gt=TypedValue(type="bbox_map", value={}),
        projected_annotation=dict(projected),
    )
    witness_symbolic = {
        "type": "scatter_readout_empty_annotation",
        "annotation_bbox_map": {},
        "point_ids": [],
        "x_axis_label": "",
    }
    return artifacts, witness_symbolic


def annotation_for_binding(
    *,
    binding: QueryBinding,
    rendered: ScatterReadoutRenderResult,
    annotation_kind: str,
) -> tuple[AnnotationArtifacts, dict[str, Any]]:
    """Project a task-owned scatter-readout binding into public annotations."""

    scene = rendered.rendered_scene
    kind = str(annotation_kind)

    if kind == "target_point":
        target_point = str(binding.target_point_id)
        if not target_point:
            return _empty_map_annotation()
        point = _bbox_center(list(scene.point_bboxes[str(target_point)]))
        artifacts = point_annotation_artifacts(point)
        witness_symbolic = {
            "type": "scatter_readout_point",
            "annotation_point": list(artifacts.value),
            "point_id": str(target_point),
            "point_role": "target_point",
        }
        return artifacts, witness_symbolic

    if kind == "comparison_point":
        comparison_point = str(binding.trace.get("comparison_point_id", ""))
        if not comparison_point:
            raise ValueError("comparison_point annotation requested without comparison_point_id")
        point = _bbox_center(list(scene.point_bboxes[str(comparison_point)]))
        artifacts = point_annotation_artifacts(point)
        witness_symbolic = {
            "type": "scatter_readout_point",
            "annotation_point": list(artifacts.value),
            "point_id": str(comparison_point),
            "point_role": "comparison_point",
        }
        return artifacts, witness_symbolic

    if kind == "target_comparison_segment":
        target_point = str(binding.target_point_id)
        comparison_point = str(binding.trace.get("comparison_point_id", ""))
        if not target_point or not comparison_point:
            raise ValueError("segment annotation requested without both point ids")
        segment = [
            _bbox_center(list(scene.point_bboxes[str(target_point)])),
            _bbox_center(list(scene.point_bboxes[str(comparison_point)])),
        ]
        artifacts = segment_annotation_artifacts(segment)
        witness_symbolic = {
            "type": "scatter_readout_segment",
            "annotation_segment": [list(point) for point in artifacts.value],
            "point_ids": [str(target_point), str(comparison_point)],
        }
        return artifacts, witness_symbolic

    raise ValueError(f"unsupported scatter_readout annotation_kind: {annotation_kind}")


__all__ = ["annotation_for_binding"]
