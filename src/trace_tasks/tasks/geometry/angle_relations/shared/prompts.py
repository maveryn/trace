"""Prompt assembly for angle-relations scene tasks."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults, required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import FALLBACK_PROMPT_WIRING_KEYS
from .state import DOMAIN, SCENE_ID


_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
)


def build_angle_relation_prompt_artifacts(
    *,
    prompt_query_key: str,
    instance_seed: int,
    prompt_task_key: str | None = None,
) -> tuple[Dict[str, Any], Any]:
    """Build prompt artifacts for one angle-relations public task."""

    prompt_defaults = required_group_defaults(
        _PROMPT_DEFAULTS,
        FALLBACK_PROMPT_WIRING_KEYS,
        context="angle_relations prompt wiring defaults",
    )
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_task_key or prompt_defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={},
        instance_seed=int(instance_seed),
    )
    return dict(prompt_defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_angle_relation_prompt_artifacts"]
