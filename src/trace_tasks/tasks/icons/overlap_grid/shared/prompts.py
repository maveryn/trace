"""Prompt rendering helpers for overlap-grid icon scenes."""

from __future__ import annotations

from typing import Any, Mapping

from ....shared.config_defaults import required_group_defaults
from ....shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import DOMAIN, SCENE_ID


def render_overlap_grid_prompt_artifacts(
    *,
    instance_seed: int,
    prompt_defaults: Mapping[str, Any],
):
    """Render prompt variants for one overlap-grid counting operation."""

    required = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "object_description",
            "question_text",
            "json_output_contract",
            "json_output_contract_answer_only",
            "annotation_hint",
            "answer_hint",
            "json_example",
            "json_example_answer_only",
        ),
        context="overlap_grid prompt defaults",
    )
    selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(required["bundle_id"]),
        scene_key=str(required["scene_key"]),
        task_key=str(required["task_key"]),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={},
        instance_seed=int(instance_seed),
    )
    return required, build_prompt_trace_artifacts(selection)


__all__ = ["render_overlap_grid_prompt_artifacts"]
