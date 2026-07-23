"""Render-parameter helpers for radial code-wheel scenes."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

from .state import RadialRenderParams


def resolve_render_params(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> RadialRenderParams:
    """Resolve concrete render parameters from scene defaults and overrides."""

    values = {}
    for field in fields(RadialRenderParams):
        values[field.name] = int(params.get(field.name, defaults.get(field.name, field.default)))
    return RadialRenderParams(**values)


__all__ = ["resolve_render_params"]
