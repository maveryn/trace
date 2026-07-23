"""Prompt assembly for Rubik cube-net puzzles."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)

from .state import DOMAIN, PROMPT_BUNDLE_ID, PROMPT_SCENE_KEY, SCENE_ID


def _target_position_description(dataset: Mapping[str, Any]) -> str:
    return (
        f"position ({int(dataset.get('target_col', 0))}, "
        f"{int(dataset.get('target_row', 0))}) on the "
        f"{dataset.get('target_face_name', '')} face"
    )


def build_prompt(
    prompt_defaults: Mapping[str, Any],
    *,
    prompt_task_key: str,
    prompt_query_key: str,
    object_description_key: str,
    dataset: Mapping[str, Any],
    instance_seed: int,
) -> tuple[str, dict[str, str], dict[str, Any], PromptTraceArtifacts]:
    """Build prompt variants from the Rubik cube-net prompt asset."""

    required_keys = (
        "bundle_id",
        "scene_key",
        str(object_description_key),
    )
    prompt_values = required_group_defaults(
        prompt_defaults,
        required_keys,
        context="prompt defaults for rubiks_net",
    )
    dynamic_slots = {
        "object_description": str(prompt_values[str(object_description_key)]),
        "target_position_description": _target_position_description(dataset),
        "target_face_name": str(dataset.get("target_face_name", "")),
        "move_sequence_text": str(dataset.get("move_sequence_text", "")).strip(),
        "base_sequence_text": str(dataset.get("base_sequence_text", "")).strip(),
    }
    prompt_selection = render_task_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_values.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key=str(prompt_values.get("scene_key", PROMPT_SCENE_KEY)),
        task_key=str(prompt_defaults.get("task_key", prompt_task_key)),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dynamic_slots,
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
    return (
        str(prompt_artifacts.prompt),
        dict(prompt_artifacts.prompt_variants),
        {
            "bundle_id": str(prompt_values.get("bundle_id", PROMPT_BUNDLE_ID)),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(
                prompt_artifacts.prompt_variant_active_key
            ),
            "prompt_variants_for_trace": dict(
                prompt_artifacts.prompt_variants_for_trace
            ),
        },
        prompt_artifacts,
    )


__all__ = ["build_prompt"]
