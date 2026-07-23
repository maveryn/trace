"""Prompt rendering helpers for the sequence-strip icons scene."""

from __future__ import annotations

from typing import Any, Mapping

from ....shared.config_defaults import required_group_defaults
from ....shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)


DOMAIN = "icons"
SCENE_ID = "sequence_strip"


def render_sequence_strip_prompt_artifacts(
    *,
    instance_seed: int,
    prompt_defaults: Mapping[str, Any],
    prompt_query_key: str,
):
    """Render prompt variants for one sequence-strip task-selected query key."""

    required = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
        ),
        context="sequence_strip prompt defaults",
    )
    selection = render_task_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(required["bundle_id"]),
        scene_key=str(required["scene_key"]),
        task_key=str(required["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={},
        instance_seed=int(instance_seed),
    )
    return required, build_prompt_trace_artifacts(selection)


__all__ = ["render_sequence_strip_prompt_artifacts"]
