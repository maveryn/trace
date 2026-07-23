"""Configuration helpers for the function-panel scene."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import group_default, split_scene_generation_rendering_prompt_defaults

from .state import DEFAULT_LABEL_POOL_6, DEFAULT_LABEL_POOL_9

DOMAIN = "geometry"


def scene_defaults() -> Mapping[str, Any]:
    """Return raw scene defaults."""

    defaults = get_scene_defaults(DOMAIN, "function_panels")
    return defaults if isinstance(defaults, Mapping) else {}


def split_defaults_for(public_task_name: str) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Return task-aware defaults for one public file."""

    return split_scene_generation_rendering_prompt_defaults(scene_defaults(), task_id=str(public_task_name))


def label_pool(params: Mapping[str, Any], *, max_count: int, gen_defaults: Mapping[str, Any]) -> tuple[str, ...]:
    """Resolve candidate panel labels for four/six/nine option layouts."""

    fallback = DEFAULT_LABEL_POOL_9 if int(max_count) >= 9 else DEFAULT_LABEL_POOL_6
    raw_pool = params.get("candidate_label_pool", group_default(gen_defaults, "candidate_label_pool", fallback))
    pool = tuple(str(label).strip().upper() for label in raw_pool)[: int(max_count)]
    allowed_sizes = {4, 6, 9} if int(max_count) >= 9 else {4, 6}
    if len(pool) not in allowed_sizes or len(set(pool)) != len(pool):
        raise ValueError(f"function_panels requires {sorted(allowed_sizes)} unique candidate labels")
    return pool[: int(max_count)]


def canvas_size(params: Mapping[str, Any], *, render_defaults: Mapping[str, Any], fallback_height: int) -> tuple[int, int]:
    """Resolve canvas size with scene-safe lower bounds."""

    width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", 1024)))
    height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", int(fallback_height))))
    return max(720, width), max(520, height)
