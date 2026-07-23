"""Prompt artifact assembly for cell-board puzzles."""

from __future__ import annotations

from typing import Any, Dict, Mapping

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

_PROMPT_WIRING_KEYS = ("bundle_id", "scene_key", "task_key")


def build_cell_board_prompt_artifacts(
    *,
    domain: str,
    prompt_task_key: str,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> tuple[Dict[str, Any], Any]:
    """Render prompt variants from the external cell-board prompt bundle."""

    prompt_defaults = required_group_defaults(
        _PROMPT_DEFAULTS,
        _PROMPT_WIRING_KEYS,
        context="cell-board prompt wiring defaults",
    )
    selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_task_key),
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slots),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        instance_seed=int(instance_seed),
    )
    defaults = dict(prompt_defaults)
    defaults["task_key"] = str(prompt_task_key)
    return defaults, build_prompt_trace_artifacts(selection)


__all__ = ["build_cell_board_prompt_artifacts"]
