"""Annotation projection helpers for symbolic music-staff scenes."""

from __future__ import annotations

from typing import Mapping, Sequence


def round_bbox(bbox: Sequence[float]) -> list[float]:
    """Round one pixel bbox for stable JSON annotation output."""

    return [round(float(value), 3) for value in bbox]


def bbox_center_point(bbox: Sequence[float]) -> list[float]:
    """Return the rounded center point for one pixel bbox."""

    values = [float(value) for value in bbox]
    return [round((values[0] + values[2]) / 2.0, 3), round((values[1] + values[3]) / 2.0, 3)]


def project_bbox_set(
    bbox_map: Mapping[str, Sequence[float]],
    item_ids: Sequence[str],
) -> list[list[float]]:
    """Project ordered notation item ids into a bbox-set annotation."""

    return [
        round_bbox(bbox_map[str(item_id)])
        for item_id in [str(item) for item in item_ids]
        if str(item_id) in bbox_map
    ]


def project_bbox_map(
    bbox_map: Mapping[str, Sequence[float]],
    role_item_ids: Mapping[str, str],
) -> dict[str, list[float]]:
    """Project role-bound notation item ids into a keyed bbox annotation."""

    projected: dict[str, list[float]] = {}
    for role, item_id in role_item_ids.items():
        key = str(item_id)
        if key not in bbox_map:
            raise RuntimeError(f"missing bbox annotation for role {role!r}: item id {key!r}")
        projected[str(role)] = round_bbox(bbox_map[key])
    return projected
