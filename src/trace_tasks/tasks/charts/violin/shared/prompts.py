"""Prompt assembly for violin chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.violin.shared.defaults import DOMAIN, PROMPT_BUNDLE_ID, PROMPT_DEFAULTS, SCENE_ID
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


SCENE_PROMPT_KEY = "violin_distribution"
TASK_PROMPT_KEY = "violin_label_query"
OBJECT_DESCRIPTION = (
    "a set of labeled violin plots. Wider parts of each shape show where that "
    "label's values are denser on the vertical axis"
)
JSON_OUTPUT_CONTRACT = 'Use a valid JSON object with keys "annotation" and "answer" in that order for the final answer.'
JSON_OUTPUT_CONTRACT_ANSWER_ONLY = 'Use a valid JSON object with key "answer" for the final answer.'
ANSWER_HINT = 'set "answer" to the exact visible label string of the requested violin plot'
ANNOTATION_HINT = 'set "annotation" to one [x0,y0,x1,y1] pixel box around the winning violin plot'
JSON_EXAMPLE = '{"annotation":[240,140,330,520],"answer":"H9H3"}'
JSON_EXAMPLE_ANSWER_ONLY = '{"answer":"H9H3"}'


def build_prompt_artifacts(
    *,
    prompt_query_key: str,
    instance_seed: int,
    dynamic_slots: Mapping[str, Any] | None = None,
) -> PromptTraceArtifacts:
    """Render the external violin prompt template."""

    rendered = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key=SCENE_PROMPT_KEY,
        task_key=TASK_PROMPT_KEY,
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": OBJECT_DESCRIPTION,
            "json_output_contract": JSON_OUTPUT_CONTRACT,
            "json_output_contract_answer_only": JSON_OUTPUT_CONTRACT_ANSWER_ONLY,
            "answer_hint": ANSWER_HINT,
            "annotation_hint": ANNOTATION_HINT,
            "json_example": JSON_EXAMPLE,
            "json_example_answer_only": JSON_EXAMPLE_ANSWER_ONLY,
            **dict(dynamic_slots or {}),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered)


__all__ = [
    "ANNOTATION_HINT",
    "ANSWER_HINT",
    "JSON_EXAMPLE",
    "JSON_EXAMPLE_ANSWER_ONLY",
    "OBJECT_DESCRIPTION",
    "SCENE_PROMPT_KEY",
    "TASK_PROMPT_KEY",
    "build_prompt_artifacts",
]
