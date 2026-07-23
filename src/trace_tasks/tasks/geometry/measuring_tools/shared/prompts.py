"""Prompt rendering for measuring-tool tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_json_example import (
    build_keyed_point_prompt_json_examples,
    dump_prompt_json_examples,
)
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import DOMAIN, SCENE_ID


def measuring_tool_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_task_key: str,
    object_description: str,
    annotation_type: str,
    annotation_keys: Sequence[str],
    answer: int,
    instance_seed: int,
) -> tuple[dict[str, Any], PromptTraceArtifacts]:
    """Render v1 prompt variants from scene/task prompt assets."""

    defaults = required_group_defaults(
        prompt_defaults,
        ("bundle_id", "scene_key"),
        context="prompt defaults for measuring_tools",
    )
    if str(annotation_type) == "segment":
        json_example, json_example_answer_only = dump_prompt_json_examples(
            annotation=[[140, 160], [260, 160]],
            answer=int(answer),
        )
        annotation_hint = (
            "set \"annotation\" to the marked measured segment as two pixel "
            "points [[x0,y0],[x1,y1]]"
        )
    elif str(annotation_type) == "point_map":
        json_example, json_example_answer_only = build_keyed_point_prompt_json_examples(
            annotation_keys=tuple(str(key) for key in annotation_keys),
            answer=int(answer),
        )
        annotation_key_list = ", ".join(f'"{key}"' for key in annotation_keys)
        annotation_hint = (
            "set \"annotation\" to a JSON object with exactly these pixel "
            f"point keys: {annotation_key_list}; each value must be the "
            "corresponding visible point [x,y]"
        )
    else:
        raise ValueError(f"unsupported measuring-tools annotation type: {annotation_type}")
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


__all__ = ["measuring_tool_prompt_artifacts"]
