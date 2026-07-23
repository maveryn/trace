"""Prompt rendering primitives for symbolic agent-automaton tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ....shared.config_defaults import required_group_defaults
from ....shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .rules import SCENE_ID


def build_agent_prompt(
    *,
    domain: str,
    prompt_defaults: Mapping[str, Any],
    scene_variant: str,
    rule_variant: str,
    steps: int,
    instance_seed: int,
    prompt_key: str,
    question_text_key: str,
    annotation_hint_key: str,
    answer_hint_key: str,
    json_example_key: str,
    json_example_answer_only_key: str,
    output_modes: Sequence[str] = PROMPT_OUTPUT_MODES,
) -> tuple[str, dict[str, str], dict[str, Any], Any]:
    """Render one agent prompt from task-owned prompt slot key choices."""

    required_keys = (
        "bundle_id",
        "scene_key",
        "json_output_contract",
        "json_output_contract_answer_only",
        f"object_description_{scene_variant}",
        f"rule_instruction_{rule_variant}",
        str(question_text_key),
        str(annotation_hint_key),
        str(answer_hint_key),
        str(json_example_key),
        str(json_example_answer_only_key),
    )
    values = required_group_defaults(prompt_defaults, required_keys, context=f"agent automaton prompt defaults for {prompt_key}")
    selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(values["bundle_id"]),
        scene_key=str(values["scene_key"]),
        task_key=str(prompt_key),
        answer_or_annotation_keys=tuple(str(mode) for mode in output_modes),
        dynamic_slots={
            "object_description": str(values[f"object_description_{scene_variant}"]),
            "rule_instruction": str(values[f"rule_instruction_{rule_variant}"]),
            "steps": int(steps),
            "question_text": str(values[str(question_text_key)]),
            "json_output_contract": str(values["json_output_contract"]),
            "json_output_contract_answer_only": str(values["json_output_contract_answer_only"]),
            "annotation_hint": str(values[str(annotation_hint_key)]),
            "answer_hint": str(values[str(answer_hint_key)]),
            "json_example": str(values[str(json_example_key)]),
            "json_example_answer_only": str(values[str(json_example_answer_only_key)]),
        },
        instance_seed=int(instance_seed),
    )
    artifacts = build_prompt_trace_artifacts(selection)
    return str(artifacts.prompt), dict(artifacts.prompt_variants), {
        "prompt_variant": dict(artifacts.prompt_variant),
        "prompt_variant_active_key": str(artifacts.prompt_variant_active_key),
        "prompt_variants_for_trace": dict(artifacts.prompt_variants_for_trace),
        "bundle_id": str(values["bundle_id"]),
    }, artifacts
