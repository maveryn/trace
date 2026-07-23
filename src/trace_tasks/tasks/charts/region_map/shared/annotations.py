"""Annotation projection helpers for region-map charts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .....core.types import TypedValue
from .rendering import MarkerMapRenderResult, RegionMapRenderResult


@dataclass(frozen=True)
class RegionMapAnnotationBundle:
    """Typed annotation payload plus projected review metadata."""

    annotation_gt: TypedValue
    annotation_type: str
    annotation_region_ids: Sequence[str]
    projected_annotation: Mapping[str, Any]
    annotation_refs: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class MarkerMapAnnotationBundle:
    """Typed marker-layer annotation payload plus projected review metadata."""

    annotation_gt: TypedValue
    annotation_type: str
    annotation_region_ids: Sequence[str]
    projected_annotation: Mapping[str, Any]
    annotation_refs: Sequence[Mapping[str, Any]]


def region_bbox_set(rendered: RegionMapRenderResult, region_ids: Sequence[str]) -> list[list[float]]:
    """Return one rendered region bbox per selected region id."""

    return [list(rendered.rendered_scene.region_bbox_map[str(region_id)]) for region_id in region_ids]


def region_point_set(rendered: RegionMapRenderResult, region_ids: Sequence[str]) -> list[list[float]]:
    """Return one rendered region center point per selected region id."""

    return [list(rendered.rendered_scene.region_center_map[str(region_id)]) for region_id in region_ids]


def projected_bbox_set(
    *,
    rendered: RegionMapRenderResult,
    region_ids: Sequence[str],
    bboxes: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Build projected annotation metadata for variable region bbox sets."""

    return {
        "type": "bbox_set",
        "bbox_set": [list(bbox) for bbox in bboxes],
        "pixel_bbox_set": [list(bbox) for bbox in bboxes],
        "bbox_map": {
            str(region_id): list(rendered.rendered_scene.region_bbox_map[str(region_id)])
            for region_id in region_ids
        },
        "center_map": {
            str(region_id): list(rendered.rendered_scene.region_center_map[str(region_id)])
            for region_id in region_ids
        },
        "region_ids": [str(region_id) for region_id in region_ids],
    }


def projected_point_set(
    *,
    rendered: RegionMapRenderResult,
    region_ids: Sequence[str],
    points: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Build projected annotation metadata for variable region point sets."""

    return {
        "type": "point_set",
        "point_set": [list(point) for point in points],
        "pixel_point_set": [list(point) for point in points],
        "point_map": {
            str(region_id): list(rendered.rendered_scene.region_center_map[str(region_id)])
            for region_id in region_ids
        },
        "bbox_map": {
            str(region_id): list(rendered.rendered_scene.region_bbox_map[str(region_id)])
            for region_id in region_ids
        },
        "region_ids": [str(region_id) for region_id in region_ids],
    }


def annotation_refs(
    *,
    region_ids: Sequence[str],
    projected_annotation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return compact region-to-bbox rows for review sidecars."""

    bboxes = list(projected_annotation.get("bbox_set", []))
    return [
        {"region_id": str(region_id), "bbox_px": list(bbox)}
        for region_id, bbox in zip(region_ids, bboxes, strict=False)
    ]


def point_annotation_refs(
    *,
    region_ids: Sequence[str],
    projected_annotation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return compact region-to-point rows for review sidecars."""

    points = list(projected_annotation.get("point_set", []))
    return [
        {"region_id": str(region_id), "point_px": list(point)}
        for region_id, point in zip(region_ids, points, strict=False)
    ]


def region_bbox_set_bundle(rendered: RegionMapRenderResult, region_ids: Sequence[str]) -> RegionMapAnnotationBundle:
    """Return the complete bbox-set annotation payload for selected map regions."""

    selected_region_ids = [str(region_id) for region_id in region_ids]
    bboxes = region_bbox_set(rendered, selected_region_ids)
    projected_annotation = projected_bbox_set(rendered=rendered, region_ids=selected_region_ids, bboxes=bboxes)
    return RegionMapAnnotationBundle(
        annotation_gt=TypedValue(type="bbox_set", value=[list(bbox) for bbox in bboxes]),
        annotation_type="bbox_set",
        annotation_region_ids=selected_region_ids,
        projected_annotation=projected_annotation,
        annotation_refs=annotation_refs(region_ids=selected_region_ids, projected_annotation=projected_annotation),
    )


def region_point_set_bundle(rendered: RegionMapRenderResult, region_ids: Sequence[str]) -> RegionMapAnnotationBundle:
    """Return the complete point-set annotation payload for selected map regions."""

    selected_region_ids = [str(region_id) for region_id in region_ids]
    points = region_point_set(rendered, selected_region_ids)
    projected_annotation = projected_point_set(rendered=rendered, region_ids=selected_region_ids, points=points)
    return RegionMapAnnotationBundle(
        annotation_gt=TypedValue(type="point_set", value=[list(point) for point in points]),
        annotation_type="point_set",
        annotation_region_ids=selected_region_ids,
        projected_annotation=projected_annotation,
        annotation_refs=point_annotation_refs(region_ids=selected_region_ids, projected_annotation=projected_annotation),
    )


def _marker_bbox_center(bbox: Sequence[float]) -> list[float]:
    """Return the center point for a rendered marker bbox."""

    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
    ]


def marker_point_set(rendered: MarkerMapRenderResult, region_ids: Sequence[str]) -> list[list[float]]:
    """Return one marker-center point per selected region id."""

    return [_marker_bbox_center(rendered.marker_group_bbox_map[str(region_id)]) for region_id in region_ids]


def marker_point(rendered: MarkerMapRenderResult, region_id: str) -> list[float]:
    """Return the scalar marker-center point for one selected region id."""

    return _marker_bbox_center(rendered.marker_group_bbox_map[str(region_id)])


def projected_marker_point_set(
    *,
    rendered: MarkerMapRenderResult,
    region_ids: Sequence[str],
    points: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Build projected annotation metadata for variable marker-region point sets."""

    return {
        "type": "point_set",
        "point_set": [list(point) for point in points],
        "pixel_point_set": [list(point) for point in points],
        "point_map": {str(region_id): marker_point(rendered, str(region_id)) for region_id in region_ids},
        "bbox_map": {str(region_id): list(rendered.marker_group_bbox_map[str(region_id)]) for region_id in region_ids},
        "region_ids": [str(region_id) for region_id in region_ids],
        "marker_bboxes_by_region": {
            str(region_id): [list(bbox) for bbox in rendered.marker_bboxes_by_region.get(str(region_id), [])]
            for region_id in region_ids
        },
    }


def projected_marker_point(
    *,
    rendered: MarkerMapRenderResult,
    region_id: str,
    point: Sequence[float],
) -> dict[str, Any]:
    """Build projected annotation metadata for one selected marker-region point."""

    return {
        "type": "point",
        "point": list(point),
        "pixel_point": list(point),
        "point_map": {str(region_id): marker_point(rendered, str(region_id))},
        "bbox_map": {str(region_id): list(rendered.marker_group_bbox_map[str(region_id)])},
        "region_id": str(region_id),
        "region_ids": [str(region_id)],
        "marker_bboxes_by_region": {
            str(region_id): [list(item) for item in rendered.marker_bboxes_by_region.get(str(region_id), [])]
        },
    }


def marker_annotation_refs(
    *,
    region_ids: Sequence[str],
    projected_annotation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return compact marker-region-to-point rows for review sidecars."""

    if str(projected_annotation.get("type")) == "point":
        return [{"region_id": str(region_ids[0]), "point_px": list(projected_annotation.get("point", []))}]
    points = list(projected_annotation.get("point_set", []))
    return [
        {"region_id": str(region_id), "point_px": list(point)}
        for region_id, point in zip(region_ids, points, strict=False)
    ]


def marker_point_set_bundle(rendered: MarkerMapRenderResult, region_ids: Sequence[str]) -> MarkerMapAnnotationBundle:
    """Return the complete point-set annotation payload for selected marker regions."""

    selected_region_ids = [str(region_id) for region_id in region_ids]
    points = marker_point_set(rendered, selected_region_ids)
    projected_annotation = projected_marker_point_set(rendered=rendered, region_ids=selected_region_ids, points=points)
    return MarkerMapAnnotationBundle(
        annotation_gt=TypedValue(type="point_set", value=[list(point) for point in points]),
        annotation_type="point_set",
        annotation_region_ids=selected_region_ids,
        projected_annotation=projected_annotation,
        annotation_refs=marker_annotation_refs(region_ids=selected_region_ids, projected_annotation=projected_annotation),
    )


def marker_point_bundle(rendered: MarkerMapRenderResult, region_id: str) -> MarkerMapAnnotationBundle:
    """Return the complete scalar point annotation payload for one marker region."""

    selected_region_id = str(region_id)
    point = marker_point(rendered, selected_region_id)
    projected_annotation = projected_marker_point(rendered=rendered, region_id=selected_region_id, point=point)
    return MarkerMapAnnotationBundle(
        annotation_gt=TypedValue(type="point", value=list(point)),
        annotation_type="point",
        annotation_region_ids=[selected_region_id],
        projected_annotation=projected_annotation,
        annotation_refs=marker_annotation_refs(region_ids=[selected_region_id], projected_annotation=projected_annotation),
    )


__all__ = [
    "MarkerMapAnnotationBundle",
    "RegionMapAnnotationBundle",
    "annotation_refs",
    "marker_annotation_refs",
    "marker_point",
    "marker_point_bundle",
    "marker_point_set",
    "marker_point_set_bundle",
    "point_annotation_refs",
    "projected_bbox_set",
    "projected_marker_point",
    "projected_marker_point_set",
    "projected_point_set",
    "region_bbox_set",
    "region_bbox_set_bundle",
    "region_point_set",
    "region_point_set_bundle",
]
