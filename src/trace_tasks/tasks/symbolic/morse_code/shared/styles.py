"""Render-parameter helpers for Morse-code scenes."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

from .state import MorseRenderParams


def resolve_render_params(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> MorseRenderParams:
    """Resolve concrete Morse render parameters from scene defaults and overrides."""

    values = {}
    for field in fields(MorseRenderParams):
        values[field.name] = int(params.get(field.name, defaults.get(field.name, field.default)))
    return MorseRenderParams(**values)


__all__ = ["resolve_render_params"]
