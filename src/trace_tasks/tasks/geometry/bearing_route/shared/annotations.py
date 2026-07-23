"""Annotation projection helpers for bearing-route geometry diagrams."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_to_list

from .state import Point, RenderedBearingScene


def bbox_centers(bboxes: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return center points for each bbox."""

    return [
        [
            round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
            round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
        ]
        for bbox in bboxes
    ]


def annotation_bbox_list(rendered: RenderedBearingScene) -> list[list[float]]:
    """Return annotation bboxes in pixel coordinates."""

    return [bbox_to_list(bbox) for bbox in rendered.annotation_bboxes]


def annotation_point_list(rendered: RenderedBearingScene, bboxes: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return annotation points in pixel coordinates."""

    if rendered.annotation_points:
        return [
            [round(float(point[0]), 3), round(float(point[1]), 3)]
            for point in rendered.annotation_points
        ]
    return bbox_centers(bboxes)


def keyed_bboxes(rendered: RenderedBearingScene) -> dict[str, list[float]]:
    """Return keyed bbox annotations from the rendered roles."""

    return {
        str(role): bbox_to_list(bbox)
        for role, bbox in zip(rendered.annotation_roles, rendered.annotation_bboxes, strict=True)
    }


def keyed_points(rendered: RenderedBearingScene, points: Sequence[Point | Sequence[float]]) -> dict[str, list[float]]:
    """Return keyed point annotations from the rendered roles."""

    return {
        str(role): [round(float(point[0]), 3), round(float(point[1]), 3)]
        for role, point in zip(rendered.annotation_roles, points, strict=True)
    }


__all__ = [
    "annotation_bbox_list",
    "annotation_point_list",
    "bbox_centers",
    "keyed_bboxes",
    "keyed_points",
]
