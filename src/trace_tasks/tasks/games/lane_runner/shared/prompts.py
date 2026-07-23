"""Prompt asset assembly helpers for lane-runner tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import SCENE_ID


@dataclass(frozen=True)
class LaneRunnerPromptContext:
    """Dynamic prompt keys supplied by a public lane-runner task."""

    scene_variant: str
    prompt_query_key: str
    object_description_suffix: str
    rule_slot_name: str
    answer_type: str


def build_lane_runner_prompt_json_examples(*, answer_type: str) -> Tuple[str, str]:
    """Return deterministic JSON examples matching the public answer type."""

    if str(answer_type) == "option_letter":
        answer_value: str | int = "C"
        annotation_value = [314, 128, 392, 310]
    else:
        answer_value = 2
        annotation_value = [[176, 412], [176, 240]]
    return (
        json.dumps({"annotation": annotation_value, "answer": answer_value}, separators=(",", ":"), ensure_ascii=False),
        json.dumps({"answer": answer_value}, separators=(",", ":"), ensure_ascii=False),
    )


def build_lane_runner_prompt_artifacts(
    *,
    domain: str,
    prompt_defaults: Mapping[str, Any],
    context: LaneRunnerPromptContext,
    instance_seed: int,
) -> tuple[dict[str, Any], PromptTraceArtifacts]:
    """Render one lane-runner prompt from external prompt assets."""

    object_description_key = f"object_description_{str(context.scene_variant)}_{str(context.object_description_suffix)}"
    answer_hint_key = f"answer_hint_{str(context.prompt_query_key)}"
    annotation_hint_key = f"annotation_hint_{str(context.prompt_query_key)}"
    defaults = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            object_description_key,
            str(context.rule_slot_name),
            answer_hint_key,
            annotation_hint_key,
        ),
        context=f"prompt defaults for {SCENE_ID}",
    )
    json_example, json_example_answer_only = build_lane_runner_prompt_json_examples(
        answer_type=str(context.answer_type),
    )
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(defaults["task_key"]),
        query_key=str(context.prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(defaults[object_description_key]),
            str(context.rule_slot_name): str(defaults[str(context.rule_slot_name)]),
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


__all__ = [
    "LaneRunnerPromptContext",
    "build_lane_runner_prompt_artifacts",
    "build_lane_runner_prompt_json_examples",
]
