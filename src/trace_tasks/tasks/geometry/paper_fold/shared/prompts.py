"""Prompt rendering for paper-fold geometry tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_json_example import dump_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import DOMAIN, SCENE_ID


def paper_fold_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_task_key: str,
    object_description: str,
    annotation_keys: Sequence[str],
    answer: float,
    instance_seed: int,
) -> tuple[dict[str, Any], PromptTraceArtifacts]:
    """Render v1 paper-fold prompt variants from scene/task assets."""

    defaults = required_group_defaults(
        prompt_defaults,
        ("bundle_id", "scene_key"),
        context="prompt defaults for paper_fold",
    )
    example_annotation = {
        str(annotation_keys[0]): [260, 210, 292, 242],
        str(annotation_keys[1]): [180, 150, 245, 180],
    }
    json_example, json_example_answer_only = dump_prompt_json_examples(
        annotation=example_annotation,
        answer=float(answer),
    )
    key_list = ", ".join(f'"{key}"' for key in annotation_keys)
    annotation_hint = (
        "set \"annotation\" to a JSON object with exactly these pixel bounding-box keys: "
        f"{key_list}; each value must be the corresponding visible box [x0,y0,x1,y1]"
    )
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(prompt_task_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "annotation_hint": str(annotation_hint),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
            "object_description": str(object_description),
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


def paper_fold_segment_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_task_key: str,
    object_description: str,
    target_segment: str,
    answer: int,
    instance_seed: int,
) -> tuple[dict[str, Any], PromptTraceArtifacts]:
    """Render v1 paper-fold prompts for scalar segment annotation tasks."""

    defaults = required_group_defaults(
        prompt_defaults,
        ("bundle_id", "scene_key"),
        context="prompt defaults for paper_fold",
    )
    example_annotation = [[240, 260], [360, 315]]
    json_example, json_example_answer_only = dump_prompt_json_examples(
        annotation=example_annotation,
        answer=int(answer),
    )
    annotation_hint = (
        f"set \"annotation\" to the requested visual segment {target_segment} "
        "as [[x0,y0],[x1,y1]]"
    )
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(prompt_task_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "annotation_hint": str(annotation_hint),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
            "object_description": str(object_description),
            "target_segment": str(target_segment),
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["paper_fold_prompt_artifacts", "paper_fold_segment_prompt_artifacts"]
