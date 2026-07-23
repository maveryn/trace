"""Prompt artifact assembly for balance-scale puzzle tasks."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import SCENE_ID

_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults("puzzles", SCENE_ID)
)


def build_balance_prompt_artifacts(
    *,
    domain: str,
    task_prompt_key: str,
    prompt_query_key: str,
    instance_seed: int,
) -> tuple[Dict[str, Any], Any]:
    """Render prompt variants from the external balance-scale prompt bundle."""

    resolved_defaults = required_group_defaults(
        _PROMPT_DEFAULTS,
        ("bundle_id", "scene_key"),
        context="balance-scale prompt wiring defaults",
    )
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(resolved_defaults["bundle_id"]),
        scene_key=str(resolved_defaults["scene_key"]),
        task_key=str(task_prompt_key),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        instance_seed=int(instance_seed),
    )
    return dict(resolved_defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_balance_prompt_artifacts"]
