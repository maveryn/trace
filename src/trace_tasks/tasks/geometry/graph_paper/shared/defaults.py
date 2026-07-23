"""Config default helpers for the graph-paper scene."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import (
    split_scene_generation_rendering_prompt_defaults,
)

from .state import SCENE_ID


def split_defaults_for(
    public_name: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Return generation, rendering, and prompt defaults for one public file."""

    scene_defaults = get_scene_defaults("geometry", SCENE_ID)
    return split_scene_generation_rendering_prompt_defaults(
        scene_defaults if isinstance(scene_defaults, Mapping) else {},
        task_id=str(public_name),
    )


def int_default(
    params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: int
) -> int:
    """Resolve one integer parameter from explicit params, defaults, then fallback."""

    value = params.get(str(key), defaults.get(str(key), fallback))
    return int(value)


def float_default(
    params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: float
) -> float:
    """Resolve one float parameter from explicit params, defaults, then fallback."""

    value = params.get(str(key), defaults.get(str(key), fallback))
    return float(value)


def string_default(
    params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: str
) -> str:
    """Resolve one string parameter from explicit params, defaults, then fallback."""

    value = params.get(str(key), defaults.get(str(key), fallback))
    return str(value)
