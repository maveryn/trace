"""Prompt assembly helpers for solitaire tableau scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import PROMPT_DEFAULTS
from .state import DOMAIN, SCENE_ID


def build_solitaire_prompt(
    prompt_query_key: str,
    *,
    json_example: str,
    json_example_answer_only: str,
    instance_seed: int,
    prompt_slots: Mapping[str, Any] | None = None,
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """Render v1 solitaire prompts from task-owned prompt keys and examples."""
    prompt_defaults = required_group_defaults(
        PROMPT_DEFAULTS,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "object_description_solitaire_tableau",
            f"answer_hint_{str(prompt_query_key)}",
            f"annotation_hint_{str(prompt_query_key)}",
            "tableau_rule_text",
            "foundation_rule_text",
        ),
        context=f"prompt defaults for {str(prompt_query_key)}",
    )
    dynamic_slots = {
        "object_description": str(prompt_defaults["object_description_solitaire_tableau"]),
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(prompt_defaults[f"answer_hint_{str(prompt_query_key)}"]),
        "annotation_hint": str(prompt_defaults[f"annotation_hint_{str(prompt_query_key)}"]),
        "json_example": str(json_example),
        "json_example_answer_only": str(json_example_answer_only),
        "tableau_rule_text": str(prompt_defaults["tableau_rule_text"]),
        "foundation_rule_text": str(prompt_defaults["foundation_rule_text"]),
    }
    if prompt_slots:
        dynamic_slots.update({str(key): str(value) for key, value in prompt_slots.items()})
    prompt_selection = render_scene_prompt_variants(
        domain="games",
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dynamic_slots,
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
    return str(prompt_artifacts.prompt), dict(prompt_artifacts.prompt_variants), {
        "prompt_artifacts": prompt_artifacts,
        "bundle_id": str(prompt_defaults["bundle_id"]),
        "prompt_variant": dict(prompt_artifacts.prompt_variant),
        "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
        "prompt_variants_for_trace": dict(prompt_artifacts.prompt_variants_for_trace),
    }
