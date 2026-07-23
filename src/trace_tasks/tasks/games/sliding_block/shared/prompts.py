"""Prompt assembly helpers for the sliding-block games scene."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import PROMPT_DEFAULTS, PROMPT_WIRING_KEYS
from .state import DOMAIN, SCENE_ID


def build_sliding_block_prompt_artifacts(
    *,
    prompt_query_key: str,
    prompt_default_prefix: str,
    answer_type: str,
    instance_seed: int,
    dynamic_values: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Render the prompt bundle using task-owned semantic prompt keys and slots."""

    prompt_defaults = required_group_defaults(
        PROMPT_DEFAULTS,
        PROMPT_WIRING_KEYS,
        context="sliding-block prompt wiring defaults",
    )
    slots = {
        "object_description": str(
            prompt_defaults.get(f"object_description_{prompt_default_prefix}", prompt_defaults["object_description"])
        ),
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "annotation_hint": str(prompt_defaults[f"annotation_hint_{prompt_default_prefix}"]),
        "answer_hint": str(prompt_defaults["answer_hint_integer" if str(answer_type) == "integer" else "answer_hint_option_letter"]),
        "json_example": str(prompt_defaults[f"json_example_{prompt_default_prefix}"]),
        "json_example_answer_only": str(prompt_defaults[f"json_example_answer_only_{prompt_default_prefix}"]),
    }
    slots.update({str(key): value for key, value in dict(dynamic_values or {}).items()})
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=slots,
        instance_seed=int(instance_seed),
    )
    return dict(prompt_defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_sliding_block_prompt_artifacts"]
