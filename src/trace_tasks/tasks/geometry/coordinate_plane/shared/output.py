"""Scene-local prompt/output helpers for coordinate-plane option tasks."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

SCENE_ID = "coordinate_plane"


def build_option_letter_prompt_artifacts(
    *,
    prompt_defaults_all: Mapping[str, Any],
    config_key: str,
    scene_key_fallback: str,
    prompt_query_key: str,
    annotation_hint_key: str,
    annotation_value: Any,
    instance_seed: int,
) -> Tuple[Mapping[str, Any], Any]:
    """Build prompt assets for option-letter tasks with one annotation value."""

    prompt_defaults = required_group_defaults(
        prompt_defaults_all,
        (
            "bundle_id",
            "scene_key",
            "task_key",
        ),
        context=f"prompt defaults for {config_key}",
    )
    prompt_selection = render_scene_prompt_variants(
        domain="geometry",
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults.get("scene_key", scene_key_fallback)),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={},
        instance_seed=int(instance_seed),
    )
    return prompt_defaults, build_prompt_trace_artifacts(prompt_selection)
