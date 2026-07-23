"""Prompt assembly helpers for tower-defense scene tasks."""

from __future__ import annotations

import json
from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import DOMAIN, SCENE_ID


_PROMPT_WIRING_KEYS = (
    "bundle_id",
    "scene_key",
    "task_key",
    "json_output_contract",
    "json_output_contract_answer_only",
)


def format_json_examples(*, annotation_type: str, answer: int | str = 2) -> tuple[str, str]:
    """Return schema-valid format examples for one annotation family."""

    if str(annotation_type) == "bbox_set":
        annotation: Any = [[220, 180, 260, 220], [510, 300, 550, 340]]
    elif str(annotation_type) == "point_set":
        annotation = [[246, 314], [514, 226]]
    elif str(annotation_type) == "point":
        annotation = [420, 260]
    else:
        raise ValueError(f"unsupported tower-defense annotation type: {annotation_type}")
    return (
        json.dumps({"annotation": annotation, "answer": answer}, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
        json.dumps({"answer": answer}, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
    )


def build_tower_defense_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    scene_variant: str,
    prompt_query_key: str,
    annotation_type: str,
    example_answer: int | str,
    instance_seed: int,
) -> tuple[Mapping[str, Any], PromptTraceArtifacts]:
    """Render external tower-defense prompt assets for one task objective."""

    object_description_key = f"object_description_{str(scene_variant)}_{str(prompt_query_key)}"
    coverage_rule_key = f"coverage_rule_text_{str(prompt_query_key)}"
    answer_hint_key = f"answer_hint_{str(prompt_query_key)}"
    annotation_hint_key = f"annotation_hint_{str(prompt_query_key)}"
    defaults = required_group_defaults(
        prompt_defaults,
        (
            *_PROMPT_WIRING_KEYS,
            object_description_key,
            coverage_rule_key,
            answer_hint_key,
            annotation_hint_key,
        ),
        context=f"tower-defense prompt defaults for {prompt_query_key}",
    )
    json_example, json_example_answer_only = format_json_examples(
        annotation_type=str(annotation_type),
        answer=example_answer,
    )
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(defaults[object_description_key]),
            "coverage_rule_text": str(defaults[coverage_rule_key]),
            "json_output_contract": str(defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(defaults["json_output_contract_answer_only"]),
            "answer_hint": str(defaults[answer_hint_key]),
            "annotation_hint": str(defaults[annotation_hint_key]),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_tower_defense_prompt_artifacts", "format_json_examples"]
