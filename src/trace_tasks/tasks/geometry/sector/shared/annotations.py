"""Annotation helpers for circular-sector formula diagrams."""

from __future__ import annotations

from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_to_list

from .state import RenderedSectorScene


def bbox_map_for_roles(rendered: RenderedSectorScene, roles: tuple[str, ...]) -> dict[str, list[float]]:
    """Return keyed pixel bboxes for the requested sector witness roles."""

    bboxes = dict(rendered.annotation_bboxes)
    missing = [role for role in roles if role not in bboxes]
    if missing:
        raise ValueError(f"missing sector annotation roles: {missing}")
    return {str(role): bbox_to_list(bboxes[str(role)]) for role in roles}


__all__ = ["bbox_map_for_roles"]
