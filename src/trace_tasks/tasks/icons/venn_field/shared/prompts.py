"""Prompt rendering helpers for Venn-field icon scenes."""

from __future__ import annotations

from typing import Any, Mapping

from ....shared.config_defaults import required_group_defaults
from ....shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import DOMAIN, SCENE_ID


def render_venn_prompt_artifacts(
    *,
    instance_seed: int,
    prompt_defaults: Mapping[str, Any],
    query_key: str,
    target_description: str,
):
    """Render prompt variants for one Venn-region query."""

    required = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
        ),
        context="venn_field prompt defaults",
    )
    selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(required["bundle_id"]),
        scene_key=str(required["scene_key"]),
        task_key=str(required["task_key"]),
        query_key=str(query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={"target_description": str(target_description)},
        instance_seed=int(instance_seed),
    )
    return required, build_prompt_trace_artifacts(selection)


__all__ = ["render_venn_prompt_artifacts"]
