"""Annotation projection helpers for composition-panel charts."""

from __future__ import annotations

from typing import Sequence

from .rendering import bbox_center_point
from .state import AnnotationRole, RenderedCompositionPanels


def annotation_map_key(role: str, panel: str, segment: str | None = None) -> str:
    if segment is None:
        return f"{str(role)}|{str(panel)}"
    return f"{str(role)}|{str(panel)}|{str(segment)}"


def format_annotation_key_list(keys: Sequence[str]) -> str:
    return ", ".join(f'"{str(key)}"' for key in keys)


def point_map_for_roles(
    rendered: RenderedCompositionPanels,
    roles: Sequence[AnnotationRole],
) -> dict[str, list[float]]:
    points: dict[str, list[float]] = {}
    for role in roles:
        if role.segment is None:
            bbox = rendered.total_bbox_by_panel.get(str(role.panel))
        else:
            bbox = rendered.annotation_bbox_by_key.get((str(role.panel), str(role.segment)))
        if bbox is None:
            raise RuntimeError(f"missing composition-panel annotation target for {role}")
        key = str(role.key) if role.key is not None else annotation_map_key(str(role.role), str(role.panel), None if role.segment is None else str(role.segment))
        points[key] = bbox_center_point(bbox)
    if not points:
        raise RuntimeError("empty composition-panel annotation")
    return points


def point_map_payload(points: dict[str, list[float]]) -> dict[str, object]:
    point_set = [list(point) for point in points.values()]
    return {
        "type": "point_map",
        "point_map": dict(points),
        "pixel_point_map": dict(points),
        "point_set": list(point_set),
        "pixel_point_set": list(point_set),
    }


def bbox_set_for_panel_roles(
    rendered: RenderedCompositionPanels,
    roles: Sequence[AnnotationRole],
) -> list[list[float]]:
    """Return full panel boxes for panel-level annotation roles."""

    panel_bboxes = {
        str(panel_trace["panel_label"]): [round(float(value), 3) for value in panel_trace["panel_bbox_px"]]
        for panel_trace in rendered.panel_traces
    }
    bboxes: list[list[float]] = []
    for role in roles:
        if role.segment is not None:
            raise RuntimeError(f"bbox_set panel annotation role must not specify a segment: {role}")
        bbox = panel_bboxes.get(str(role.panel))
        if bbox is None:
            raise RuntimeError(f"missing composition-panel bbox for annotation role {role}")
        bboxes.append(list(bbox))
    if not bboxes:
        raise RuntimeError("empty composition-panel bbox annotation")
    return bboxes


def bbox_for_panel_role(
    rendered: RenderedCompositionPanels,
    role: AnnotationRole,
) -> list[float]:
    """Return one full panel box for a scalar panel-level annotation role."""

    bboxes = bbox_set_for_panel_roles(rendered, (role,))
    if len(bboxes) != 1:
        raise RuntimeError("single composition-panel bbox annotation must resolve exactly one panel")
    return list(bboxes[0])


def bbox_payload(bbox: Sequence[float]) -> dict[str, object]:
    value = [round(float(coord), 3) for coord in bbox]
    return {
        "type": "bbox",
        "bbox": list(value),
        "pixel_bbox": list(value),
        "bbox_set": [list(value)],
        "pixel_bbox_set": [list(value)],
    }


def bbox_set_payload(bboxes: Sequence[Sequence[float]]) -> dict[str, object]:
    values = [[round(float(value), 3) for value in bbox] for bbox in bboxes]
    return {
        "type": "bbox_set",
        "bbox_set": [list(value) for value in values],
        "pixel_bbox_set": [list(value) for value in values],
    }
