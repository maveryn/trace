"""Scene-neutral prompt assembly helpers for games tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.config_defaults import required_group_default, required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


def build_games_prompt_artifacts(
    *,
    domain: str,
    scene_id: str,
    prompt_defaults: Mapping[str, Any],
    prompt_wiring_keys: Sequence[str],
    context: str,
    prompt_query_key: str,
    instance_seed: int,
    dynamic_slots: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], PromptTraceArtifacts]:
    """Build prompt artifacts for a games scene from its external prompt bundle."""

    resolved_defaults = required_group_defaults(
        prompt_defaults,
        tuple(str(key) for key in prompt_wiring_keys),
        context=str(context),
    )
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=str(scene_id),
        bundle_id=str(resolved_defaults["bundle_id"]),
        scene_key=str(resolved_defaults["scene_key"]),
        task_key=str(resolved_defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(dynamic_slots or {}),
        instance_seed=int(instance_seed),
    )
    return dict(resolved_defaults), build_prompt_trace_artifacts(prompt_selection)


def games_prompt_output_slots(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_query_key: str,
    context: str,
    json_example: str,
    json_example_answer_only: str,
) -> dict[str, Any]:
    """Return standard answer/annotation prompt slots for one games query key."""

    return {
        "annotation_hint": str(
            required_group_default(
                prompt_defaults,
                f"annotation_hint_{str(prompt_query_key)}",
                context=str(context),
            )
        ),
        "answer_hint": str(
            required_group_default(
                prompt_defaults,
                f"answer_hint_{str(prompt_query_key)}",
                context=str(context),
            )
        ),
        "json_output_contract": str(
            prompt_defaults.get(
                "json_output_contract",
                'Return JSON with exactly two keys: "annotation" and "answer".',
            )
        ),
        "json_output_contract_answer_only": str(
            prompt_defaults.get(
                "json_output_contract_answer_only",
                'Return JSON with exactly one key: "answer".',
            )
        ),
        "json_example": str(json_example),
        "json_example_answer_only": str(json_example_answer_only),
    }


__all__ = ["build_games_prompt_artifacts", "games_prompt_output_slots"]
