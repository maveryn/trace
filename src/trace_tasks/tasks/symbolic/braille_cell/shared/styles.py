"""Render-parameter helpers for Braille-cell scenes."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

from .state import BrailleRenderParams


def resolve_render_params(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> BrailleRenderParams:
    """Resolve concrete Braille render parameters from scene defaults and overrides."""

    values = {}
    for field in fields(BrailleRenderParams):
        values[field.name] = int(params.get(field.name, defaults.get(field.name, field.default)))
    return BrailleRenderParams(**values)


__all__ = ["resolve_render_params"]
