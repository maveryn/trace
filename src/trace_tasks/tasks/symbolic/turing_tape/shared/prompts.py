"""Prompt rendering for symbolic Turing tape tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ....shared.config_defaults import required_group_defaults
from ....shared.prompt_variants import PROMPT_OUTPUT_MODES, PromptTraceArtifacts, build_prompt_trace_artifacts, render_task_prompt_variants

from .state import SCENE_ID


def render_turing_prompt(
    *,
    prompt_defaults: Mapping[str, Any],
    scene_variant: str,
    task_prompt_key: str,
    prompt_query_key: str,
    steps: int,
    query_symbol: str,
    instance_seed: int,
) -> Tuple[str, Dict[str, str], PromptTraceArtifacts]:
    """Render the externalized Turing tape prompt bundle."""

    required_keys = (
        "bundle_id",
        f"object_description_{scene_variant}",
        f"query_instruction_{prompt_query_key}",
    )
    values = required_group_defaults(prompt_defaults, required_keys, context="prompt defaults for turing_tape")
    slots = {
        "object_description": str(values[f"object_description_{scene_variant}"]),
        "query_instruction": str(values[f"query_instruction_{prompt_query_key}"]),
        "query_symbol": str(query_symbol),
        "steps": int(steps),
    }
    selection = render_task_prompt_variants(
        domain="symbolic",
        scene_id=SCENE_ID,
        bundle_id=str(values["bundle_id"]),
        scene_key=SCENE_ID,
        task_key=str(task_prompt_key),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=slots,
        instance_seed=int(instance_seed),
    )
    artifacts = build_prompt_trace_artifacts(selection)
    return str(artifacts.prompt), dict(artifacts.prompt_variants), artifacts
