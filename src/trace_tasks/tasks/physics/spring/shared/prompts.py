"""Prompt helpers for the spring physics scene."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from trace_tasks.tasks.shared.prompt_json_example import build_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import SCENE_ID, SCENE_PROMPT_KEY


PROMPT_BUNDLE_ID = "physics_spring_v1"


def build_spring_prompt_artifacts(
    *,
    domain: str,
    bundle_id: str,
    task_key: str,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    """Build prompt artifacts from v1 spring prompt assets."""

    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(bundle_id),
        scene_key=SCENE_PROMPT_KEY,
        task_key=str(task_key),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={str(key): value for key, value in dynamic_slots.items()},
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


def build_spring_prompt_examples(*, use_role_map: bool) -> Tuple[str, str]:
    """Return one stable prompt JSON example for a spring query."""

    if bool(use_role_map):
        annotation = {
            "reference_weight": [170, 342, 248, 396],
            "reference_extension": [204, 252, 256, 262],
            "query_weight": [595, 226, 647, 260],
            "query_extension": [598, 304, 650, 314],
        }
    else:
        annotation = [[210, 222, 260, 232], [620, 274, 670, 284]]
    return build_prompt_json_examples(annotation_value=annotation, answer_type="integer")


__all__ = ["PROMPT_BUNDLE_ID", "build_spring_prompt_artifacts", "build_spring_prompt_examples"]
