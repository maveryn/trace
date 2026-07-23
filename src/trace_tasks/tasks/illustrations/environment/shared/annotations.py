"""Annotation projection helpers for environment illustrations."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

from .rendering import EnvironmentFeature, RenderedEnvironmentObjectScene


def target_feature(scene: RenderedEnvironmentObjectScene, feature_type: str) -> EnvironmentFeature:
    """Return the unique queried road/river feature of the requested type."""

    matches = [feature for feature in scene.features if str(feature.feature_type) == str(feature_type)]
    if not matches:
        raise ValueError(f"rendered scene has no feature of type {feature_type}")
    return matches[0]


def feature_bbox_map(scene: RenderedEnvironmentObjectScene) -> Dict[str, list[float]]:
    """Return feature bbox map keyed by feature id."""

    return {str(feature.feature_id): [round(float(v), 3) for v in feature.bbox_xyxy] for feature in scene.features}


def feature_path_map(scene: RenderedEnvironmentObjectScene) -> Dict[str, list[list[float]]]:
    """Return feature path-point map keyed by feature id."""

    return {
        str(feature.feature_id): [[round(float(x), 3), round(float(y), 3)] for x, y in feature.path_points]
        for feature in scene.features
    }


def sort_bboxes_by_ids(bbox_map: Mapping[str, Sequence[float]], ids: Sequence[str]) -> list[list[float]]:
    """Return bbox values sorted top-to-bottom then left-to-right by id."""

    boxes = [(str(item_id), [round(float(v), 3) for v in bbox_map[str(item_id)]]) for item_id in ids]
    ordered = sorted(boxes, key=lambda item: (float(item[1][1]), float(item[1][0]), str(item[0])))
    return [box for _item_id, box in ordered]


def sort_bbox_centers_by_ids(bbox_map: Mapping[str, Sequence[float]], ids: Sequence[str]) -> list[list[float]]:
    """Return bbox center points sorted with the same order as sorted bboxes."""

    boxes = [(str(item_id), [float(v) for v in bbox_map[str(item_id)]]) for item_id in ids]
    ordered = sorted(boxes, key=lambda item: (float(item[1][1]), float(item[1][0]), str(item[0])))
    return [
        [
            round((float(box[0]) + float(box[2])) / 2.0, 3),
            round((float(box[1]) + float(box[3])) / 2.0, 3),
        ]
        for _item_id, box in ordered
    ]


__all__ = [
    "feature_bbox_map",
    "feature_path_map",
    "sort_bbox_centers_by_ids",
    "sort_bboxes_by_ids",
    "target_feature",
]
