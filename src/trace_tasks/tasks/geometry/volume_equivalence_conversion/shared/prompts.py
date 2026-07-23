"""Prompt rendering helpers for volume-equivalence conversion tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_json_example import dump_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .annotations import example_bbox_for_key
from .defaults import DOMAIN, SCENE_ID


def make_prompt_examples(
    answer: int | str,
    annotation_keys: Sequence[str],
    *,
    annotation_schema: str = "bbox_map",
) -> tuple[str, str]:
    if str(annotation_schema) == "bbox":
        if len(annotation_keys) != 1:
            raise ValueError("bbox prompt examples require exactly one annotation key")
        annotation = example_bbox_for_key(str(annotation_keys[0]))
    else:
        annotation = {str(key): example_bbox_for_key(str(key)) for key in annotation_keys}
    return dump_prompt_json_examples(annotation=annotation, answer=answer)


def volume_equivalence_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_task_key: str,
    prompt_branch_key: str,
    annotation_keys: Sequence[str],
    annotation_schema: str = "bbox_map",
    answer: int | str,
    instance_seed: int,
) -> tuple[dict[str, Any], PromptTraceArtifacts]:
    """Render v1 prompt variants from task-owned prompt keys and slots."""

    defaults = required_group_defaults(
        prompt_defaults,
        ("bundle_id", "scene_key"),
        context="prompt defaults for volume_equivalence_conversion",
    )
    json_example, json_example_answer_only = make_prompt_examples(
        answer=answer,
        annotation_keys=annotation_keys,
        annotation_schema=str(annotation_schema),
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
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["make_prompt_examples", "volume_equivalence_prompt_artifacts"]
