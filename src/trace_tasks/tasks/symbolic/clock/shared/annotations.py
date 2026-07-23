"""Annotation helpers for symbolic clock-display scenes."""

from __future__ import annotations

from typing import Sequence

from ....shared.annotation_artifacts import AnnotationArtifacts, segment_set_annotation_artifacts

from .state import RenderedClockScene


def clock_hand_segment_annotations(
    rendered_scene: RenderedClockScene,
    *,
    include_second_hand: bool = False,
) -> AnnotationArtifacts:
    """Build unordered segment annotations for visible clock hands."""

    center = tuple(float(value) for value in rendered_scene.center_px)
    hour_tip = tuple(float(value) for value in rendered_scene.hour_hand_tip_px)
    minute_tip = tuple(float(value) for value in rendered_scene.minute_hand_tip_px)
    segments: list[Sequence[Sequence[float]]] = [
        (center, hour_tip),
        (center, minute_tip),
    ]
    if bool(include_second_hand):
        if rendered_scene.second_hand_tip_px is None:
            raise ValueError("second-hand annotation requested for a two-hand clock")
        second_tip = tuple(float(value) for value in rendered_scene.second_hand_tip_px)
        segments.append((center, second_tip))
    return segment_set_annotation_artifacts(segments)


__all__ = ["clock_hand_segment_annotations"]
