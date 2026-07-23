"""Annotation projection helpers for scientific axis-frame chart scenes."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.scientific_axis_frame.shared.state import AxisFrameRenderResult
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, segment_annotation_artifacts


def segment_for_tick_roles(
    *,
    rendered: AxisFrameRenderResult,
    role_tick_keys: Mapping[str, str],
) -> tuple[AnnotationArtifacts, dict[str, Any]]:
    tick_points = rendered.rendered_scene.tick_points_px
    ordered_keys = [str(tick_key) for tick_key in role_tick_keys.values()]
    if len(ordered_keys) != 2:
        raise ValueError("axis-frame segment annotation requires exactly two tick roles")
    segment = [list(tick_points[str(tick_key)]) for tick_key in ordered_keys]
    artifacts = segment_annotation_artifacts(segment)
    witness_symbolic = {
        "type": "axis_tick_segment_witness",
        "tick_keys": list(ordered_keys),
        "annotation_segment": [list(point) for point in artifacts.value],
    }
    return artifacts, witness_symbolic


__all__ = ["segment_for_tick_roles"]
