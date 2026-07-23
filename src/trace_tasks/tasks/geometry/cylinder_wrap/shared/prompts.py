"""Prompt assembly helpers for cylinder-wrap tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import DOMAIN, SCENE_ID


def cylinder_wrap_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_key: str,
    instance_seed: int,
) -> Any:
    """Render prompt variants from external prompt assets."""

    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_key),
        query_key=None,
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={},
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


def resolve_cylinder_wrap_prompt(
    *,
    prompt_defaults: Mapping[str, Any],
    field_prefix: str,
    instance_seed: int,
) -> tuple[dict[str, Any], Any]:
    """Load scene prompt defaults and render the selected prompt variants."""

    defaults = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
        ),
        context="prompt defaults for cylinder_wrap",
    )
    prompt_artifacts = cylinder_wrap_prompt_artifacts(
        prompt_defaults=defaults,
        prompt_key=str(defaults["task_key"]),
        instance_seed=int(instance_seed),
    )
    return dict(defaults), prompt_artifacts


__all__ = ["cylinder_wrap_prompt_artifacts", "resolve_cylinder_wrap_prompt"]
