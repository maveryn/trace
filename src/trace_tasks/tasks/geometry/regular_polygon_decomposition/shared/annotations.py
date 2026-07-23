"""Annotation projection helpers for regular-polygon decomposition diagrams."""

from __future__ import annotations

from trace_tasks.tasks.geometry.shared.vector2d import point_to_list

from .state import RenderedRegularPolygonScene


def point_map_for_roles(rendered: RenderedRegularPolygonScene, roles: tuple[str, ...]) -> dict[str, list[float]]:
    """Project the requested visible point roles into pixel coordinates."""

    return {str(role): point_to_list(rendered.annotation_points[str(role)]) for role in roles}


__all__ = ["point_map_for_roles"]
