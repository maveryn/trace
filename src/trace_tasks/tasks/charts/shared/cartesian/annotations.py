"""Annotation projection helpers for rendered cartesian chart marks."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from trace_tasks.tasks.charts.shared.chart_scene_types import RenderedChartScene


def projected_mark_annotation(
    rendered_scene: RenderedChartScene,
    labels: Sequence[str],
) -> Dict[str, Any]:
    """Project an ordered label list into reusable pixel-space mark annotation."""

    requested = [str(label) for label in labels]
    by_label = {
        str(mark_trace["label"]): mark_trace
        for mark_trace in rendered_scene.mark_traces
    }
    pixel_point_map: Dict[str, list[float]] = {}
    pixel_point_set: list[list[float]] = []
    bbox_set: list[list[float]] = []
    for label in requested:
        mark_trace = by_label.get(str(label))
        if mark_trace is None:
            continue
        bbox = [float(value) for value in mark_trace["mark_bbox_px"]]
        if str(rendered_scene.scene_variant) == "bar":
            center = [
                0.5 * (float(bbox[0]) + float(bbox[2])),
                float(bbox[1]),
            ]
        elif str(rendered_scene.scene_variant) == "horizontal_bar":
            center = [
                float(bbox[2]),
                0.5 * (float(bbox[1]) + float(bbox[3])),
            ]
        else:
            center = [float(value) for value in mark_trace["mark_center_px"]]
        pixel_point_map[str(label)] = list(center)
        pixel_point_set.append(list(center))
        bbox_set.append(list(bbox))
    return {
        "pixel_point_map": pixel_point_map,
        "pixel_point_set": pixel_point_set,
        "bbox_set": bbox_set,
    }


__all__ = ["projected_mark_annotation"]
