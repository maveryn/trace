"""Shared parameter resolvers for coordinate-geometry tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import group_default


def resolve_int_param(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: int) -> int:
    """Resolve an integer parameter from explicit params, task defaults, or fallback."""

    return int(params.get(str(key), group_default(defaults, str(key), int(fallback))))


__all__ = ["resolve_int_param"]
