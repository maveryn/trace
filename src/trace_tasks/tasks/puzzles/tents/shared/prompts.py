"""Prompt artifact assembly for Tents puzzle tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_default
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)

from .state import DOMAIN, SCENE_ID


def render_tents_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_task_key: str,
    prompt_query_key: str,
    object_description: str,
    instance_seed: int,
) -> PromptTraceArtifacts:
    """Render Tents prompt variants from the v1 scene prompt bundle."""

    prompt_selection = render_task_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_task_key),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={"object_description": str(object_description)},
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


def object_description_for_scene_variant(
    prompt_defaults: Mapping[str, Any],
    scene_variant: str,
) -> str:
    """Return prompt-facing wording for one Tents visual variant."""

    return str(
        required_group_default(
            prompt_defaults,
            f"object_description_{str(scene_variant)}",
            context="Tents prompt defaults",
        )
    )


__all__ = ["object_description_for_scene_variant", "render_tents_prompt_artifacts"]
