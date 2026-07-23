"""Prompt assembly helpers for bearing-route geometry tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import PROMPT_WIRING_KEYS
from .state import SCENE_ID


def build_bearing_route_prompt_artifacts(
    *,
    domain: str,
    prompt_defaults: Mapping[str, Any],
    prompt_query_key: str,
    instance_seed: int,
) -> tuple[dict[str, Any], Any]:
    """Build prompt artifacts from the scene prompt bundle and query key."""

    defaults = required_group_defaults(
        prompt_defaults,
        PROMPT_WIRING_KEYS,
        context="bearing_route prompt wiring defaults",
    )
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={},
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_bearing_route_prompt_artifacts"]
