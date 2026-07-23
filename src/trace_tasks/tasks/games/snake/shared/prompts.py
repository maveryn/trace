"""Prompt assembly helpers for the Snake games scene."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import PROMPT_DEFAULTS, PROMPT_WIRING_KEYS
from .rules import move_sequence_text
from .state import DOMAIN, SCENE_ID, SnakeSample


def build_snake_prompt_artifacts(
    *,
    prompt_key: str,
    sample: SnakeSample,
    scene_variant: str,
    instance_seed: int,
    json_example: str,
    json_example_answer_only: str,
) -> tuple[dict[str, Any], Any]:
    """Render the prompt bundle with task-selected semantic prompt keys."""

    prompt_defaults = required_group_defaults(
        PROMPT_DEFAULTS,
        PROMPT_WIRING_KEYS,
        context="Snake prompt wiring defaults",
    )
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(prompt_defaults[f"object_description_{str(scene_variant)}"]),
            "snake_rule_text": str(prompt_defaults["snake_rule_text"]),
            "planned_move_wall_annotation_rule_text": str(prompt_defaults["planned_move_wall_annotation_rule_text"]),
            "planned_moves": move_sequence_text(sample.planned_moves),
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "answer_hint": str(prompt_defaults[f"answer_hint_{str(prompt_key)}"]),
            "annotation_hint": str(prompt_defaults[f"annotation_hint_{str(prompt_key)}"]),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return dict(prompt_defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_snake_prompt_artifacts"]
