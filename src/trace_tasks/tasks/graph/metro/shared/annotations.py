"""Annotation projection helpers for the metro graph scene."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from .state import RenderedMetroRouteScene


def projected_metro_station_point_annotation(rendered_scene: RenderedMetroRouteScene, labels: Sequence[str]) -> Dict[str, Any]:
    """Project ordered station labels into pixel point/bbox annotation."""

    station_by_label = {str(station.label): station for station in rendered_scene.stations}
    point_map: Dict[str, list[float]] = {}
    point_set: list[list[float]] = []
    bbox_set: list[list[float]] = []
    for label in [str(value) for value in labels]:
        station = station_by_label.get(str(label))
        if station is None:
            continue
        point = [float(station.center_xy[0]), float(station.center_xy[1])]
        bbox = [float(value) for value in station.bbox_xyxy]
        point_map[str(label)] = list(point)
        point_set.append(list(point))
        bbox_set.append(list(bbox))
    return {
        "pixel_point_map": point_map,
        "pixel_point_set": point_set,
        "pixel_point_sequence": list(point_set),
        "pixel_bbox_set": bbox_set,
    }


__all__ = ["projected_metro_station_point_annotation"]
