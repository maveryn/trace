"""Prompt rendering helpers for mirror-grid icon scenes."""

from __future__ import annotations

from typing import Any, Mapping

from ....shared.config_defaults import required_group_defaults
from ....shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


_REQUIRED_PROMPT_DEFAULT_KEYS = (
    "bundle_id",
    "scene_key",
    "task_key",
    "json_output_contract",
    "json_output_contract_answer_only",
)


def required_mirror_grid_prompt_defaults(
    prompt_defaults: Mapping[str, Any],
    *,
    context: str,
) -> Mapping[str, Any]:
    """Return validated prompt defaults for a mirror-grid public task."""

    return required_group_defaults(
        prompt_defaults,
        _REQUIRED_PROMPT_DEFAULT_KEYS,
        context=str(context),
    )


def render_mirror_grid_prompt_artifacts(
    *,
    domain: str,
    scene_id: str,
    instance_seed: int,
    prompt_defaults: Mapping[str, Any],
):
    """Render both prompt output variants."""

    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=str(scene_id),
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


__all__ = [
    "render_mirror_grid_prompt_artifacts",
    "required_mirror_grid_prompt_defaults",
]
