"""Prompt slot helpers for indoor-room illustration tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from ....shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


INDOOR_SETTING_NAMES: Dict[str, str] = {
    "living_room": "a living room",
    "kitchen": "a kitchen",
    "study": "a study",
    "bedroom": "a bedroom",
}


def indoor_setting_name(theme_id: str) -> str:
    return INDOOR_SETTING_NAMES.get(str(theme_id), "an illustrated indoor room")


def build_indoor_prompt_artifacts(
    *,
    domain: str,
    scene_id: str,
    prompt_defaults: Mapping[str, Any],
    prompt_query_key: str,
    slots: Mapping[str, Any],
    instance_seed: int,
) -> Any:
    selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=str(scene_id),
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_query_key),
        slots=dict(slots),
        instance_seed=int(instance_seed),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        preferred_mode="answer_and_annotation",
    )
    return build_prompt_trace_artifacts(selection)


__all__ = ["INDOOR_SETTING_NAMES", "build_indoor_prompt_artifacts", "indoor_setting_name"]
