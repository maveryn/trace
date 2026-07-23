"""Prompt artifact helpers for container volume-transfer tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_json_example import dump_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_scene_prompt_variants

from .defaults import DOMAIN, SCENE_ID
from .measurements import json_answer_value


def _example_bbox_for_key(key: str) -> list[int]:
    examples = {
        "source_container_bbox": [145, 190, 285, 420],
        "target_container_bbox": [505, 180, 685, 426],
    }
    return list(examples[str(key)])


def make_prompt_examples(answer: int | float, annotation_keys: Sequence[str]) -> tuple[str, str]:
    answer_value = json_answer_value(answer)
    annotation = {str(key): _example_bbox_for_key(str(key)) for key in annotation_keys}
    return dump_prompt_json_examples(annotation=annotation, answer=answer_value)


def container_volume_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    task_key: str,
    query_key: str,
    annotation_keys: Sequence[str],
    answer_hint_key: str,
    answer: int | float,
    instance_seed: int,
):
    """Render scene/task/query prompt layers with task-owned dynamic slots."""

    prompt_defaults = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            str(answer_hint_key),
        ),
        context="prompt defaults for container_volume_transfer",
    )
    annotation_names = ", ".join(f'"{key}"' for key in annotation_keys)
    json_example, json_example_answer_only = make_prompt_examples(answer=answer, annotation_keys=annotation_keys)
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(task_key),
        query_key=str(query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "annotation_keys": str(annotation_names),
            "answer_hint": str(prompt_defaults[str(answer_hint_key)]),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


__all__ = ["container_volume_prompt_artifacts", "make_prompt_examples"]
