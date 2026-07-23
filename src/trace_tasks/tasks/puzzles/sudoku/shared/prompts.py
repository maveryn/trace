"""Prompt rendering helpers for the Sudoku scene."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)

from .state import DOMAIN, SCENE_ID


def render_sudoku_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_query_key: str,
    object_description: str,
    unit_scope_text: str = "",
    target_digit: str = "",
    instance_seed: int,
) -> PromptTraceArtifacts:
    """Render Sudoku prompt variants from the v1 scene prompt bundle."""

    prompt_selection = render_task_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(object_description),
            "unit_scope_text": str(unit_scope_text),
            "target_digit": str(target_digit),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


def object_description_for_scene_variant(
    prompt_defaults: Mapping[str, Any],
    scene_variant: str,
) -> str:
    """Return the prompt-facing object description for one board density."""

    return str(prompt_defaults[f"object_description_{str(scene_variant)}"])


def unit_scope_text(prompt_defaults: Mapping[str, Any], unit_type: str | None) -> str:
    """Return the prompt-facing phrase for one highlighted unit type."""

    if unit_type is None:
        return ""
    return str(prompt_defaults[f"unit_scope_text_{str(unit_type)}"])


__all__ = [
    "object_description_for_scene_variant",
    "render_sudoku_prompt_artifacts",
    "unit_scope_text",
]
