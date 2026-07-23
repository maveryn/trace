"""Prompt artifact assembly for color-gradient puzzle tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_default
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)

from .state import DOMAIN, SCENE_ID


def render_color_gradient_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_task_key: str,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, object],
    instance_seed: int,
) -> PromptTraceArtifacts:
    """Render prompt variants from the color-gradient prompt bundle."""

    object_description = required_group_default(
        prompt_defaults,
        "object_description",
        context="color-gradient prompt defaults",
    )
    prompt_selection = render_task_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_task_key),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(object_description),
            **{str(key): value for key, value in dict(dynamic_slots).items()},
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


def object_description_for_scene(scene_variant: str, *, completion: bool) -> str:
    """Return concise prompt-facing wording for one visual treatment."""

    if bool(completion):
        descriptions = {
            "swatch_clean": (
                "a row of color swatches with one blank swatch and labeled "
                "color options"
            ),
            "swatch_card": (
                "a card-framed row of color swatches with one blank swatch "
                "and labeled color options"
            ),
            "swatch_notebook": (
                "a notebook-style row of color swatches with one blank "
                "swatch and labeled color options"
            ),
        }
    else:
        descriptions = {
            "swatch_clean": (
                "a labeled grid of color swatches arranged as a smooth color "
                "progression"
            ),
            "swatch_card": (
                "a card-framed labeled grid of color swatches arranged as a "
                "smooth color progression"
            ),
            "swatch_notebook": (
                "a notebook-style labeled grid of color swatches arranged as "
                "a smooth color progression"
            ),
        }
    return descriptions.get(str(scene_variant), descriptions["swatch_clean"])


__all__ = [
    "object_description_for_scene",
    "render_color_gradient_prompt_artifacts",
]
