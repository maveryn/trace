"""Prompt assembly helpers for the Checkers games scene."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.games.shared.prompts import build_games_prompt_artifacts
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from .defaults import PROMPT_WIRING_KEYS
from .state import SCENE_ID


_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)


def build_checkers_prompt_artifacts(
    *,
    domain: str,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any] | None,
    instance_seed: int,
) -> tuple[Dict[str, Any], Any]:
    """Build prompt artifacts from the external Checkers prompt bundle."""

    return build_games_prompt_artifacts(
        domain=str(domain),
        scene_id=SCENE_ID,
        prompt_defaults=_PROMPT_DEFAULTS,
        prompt_wiring_keys=PROMPT_WIRING_KEYS,
        context="Checkers prompt wiring defaults",
        prompt_query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slots or {}),
        instance_seed=int(instance_seed),
    )


__all__ = ["build_checkers_prompt_artifacts"]
