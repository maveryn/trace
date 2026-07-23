"""Prompt assembly helpers for uncertainty-band charts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import prompt_bundle_id
from .state import DOMAIN, PROMPT_BUNDLE_ID, SCENE_ID


SCENE_KEY = "uncertainty_band_scene"
TASK_KEY = "uncertainty_band_query"

JSON_OUTPUT_CONTRACT = 'Use a valid JSON object with keys "annotation" and "answer" in that order for the final answer.'
JSON_OUTPUT_CONTRACT_ANSWER_ONLY = 'Use a valid JSON object with key "answer" for the final answer.'
OBJECT_DESCRIPTION = "a line chart with two labeled series, each shown with a central line and a shaded upper-to-lower uncertainty band"

ANSWER_HINT_COUNT = 'set "answer" to the requested integer count'
ANSWER_HINT_LABEL = 'set "answer" to the exact visible x-axis label as a string'
ANNOTATION_HINT_OVERLAP = 'set "annotation" to an array of [x,y] pixel points, one centered inside each counted overlap region'
ANNOTATION_HINT_WIDTH = 'set "annotation" to one segment [[x0,y0],[x1,y1]] from the lower band boundary to the upper band boundary at the answer x-axis label'
JSON_EXAMPLE_OVERLAP = '{"annotation":[[345,318],[512,286],[679,302]],"answer":3}'
JSON_EXAMPLE_WIDTH = '{"annotation":[[612,382],[612,214]],"answer":"FY24"}'
JSON_EXAMPLE_ANSWER_ONLY_OVERLAP = '{"answer":3}'
JSON_EXAMPLE_ANSWER_ONLY_WIDTH = '{"answer":"FY24"}'


def build_prompt_artifacts(
    *,
    prompt_query_key: str,
    answer_hint: str,
    annotation_hint: str,
    json_example: str,
    json_example_answer_only: str,
    dynamic_slot_values: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    rendered = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_bundle_id() or PROMPT_BUNDLE_ID),
        scene_key=SCENE_KEY,
        task_key=TASK_KEY,
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": OBJECT_DESCRIPTION,
            "json_output_contract": JSON_OUTPUT_CONTRACT,
            "json_output_contract_answer_only": JSON_OUTPUT_CONTRACT_ANSWER_ONLY,
            "answer_hint": str(answer_hint),
            "annotation_hint": str(annotation_hint),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
            **dict(dynamic_slot_values),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered)


__all__ = [
    "ANNOTATION_HINT_OVERLAP",
    "ANNOTATION_HINT_WIDTH",
    "ANSWER_HINT_COUNT",
    "ANSWER_HINT_LABEL",
    "JSON_EXAMPLE_ANSWER_ONLY_OVERLAP",
    "JSON_EXAMPLE_ANSWER_ONLY_WIDTH",
    "JSON_EXAMPLE_OVERLAP",
    "JSON_EXAMPLE_WIDTH",
    "build_prompt_artifacts",
]
