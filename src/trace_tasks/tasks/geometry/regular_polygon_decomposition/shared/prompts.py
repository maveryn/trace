"""Prompt helpers for regular-polygon decomposition objectives."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_json_example import build_keyed_point_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import DOMAIN, SCENE_ID


def regular_polygon_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_task_key: str,
    prompt_branch_key: str,
    annotation_roles: tuple[str, ...],
    target_name: str,
    answer: int | float,
    instance_seed: int,
) -> tuple[dict[str, Any], PromptTraceArtifacts]:
    """Render prompt variants for one selected regular-polygon branch."""

    defaults = required_group_defaults(
        prompt_defaults,
        ("bundle_id", "scene_key"),
        context="prompt defaults for regular_polygon_decomposition",
    )
    json_example, json_example_answer_only = build_keyed_point_prompt_json_examples(
        annotation_keys=tuple(str(role) for role in annotation_roles),
        answer=answer,
    )
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(prompt_task_key),
        query_key=str(prompt_branch_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "target_name": str(target_name),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["regular_polygon_prompt_artifacts"]
