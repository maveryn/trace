"""Prompt assembly helpers for treemap charts."""

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


SCENE_KEY = "composition_treemap"
TASK_KEY = "treemap_composition_query"

JSON_OUTPUT_CONTRACT = 'Use a valid JSON object with keys "annotation" and "answer" in that order for the final answer.'
JSON_OUTPUT_CONTRACT_ANSWER_ONLY = 'Use a valid JSON object with key "answer" for the final answer.'
OBJECT_DESCRIPTION = "a treemap composition chart with parent category rectangles and child rectangles labeled with printed integer values"
ANSWER_HINT_INTEGER = 'set "answer" to the requested integer'
ANSWER_HINT_PARENT_LABEL = 'set "answer" to the exact visible parent category label as a string'
ANNOTATION_HINT_GROUP_TOTAL = 'set "annotation" to an array of [x0,y0,x1,y1] boxes around the child rectangles inside the requested parent rectangle'
ANNOTATION_HINT_PARENT_EXTREMUM = 'set "annotation" to an array of [x0,y0,x1,y1] boxes around every child rectangle inside the answer parent rectangle'
ANNOTATION_HINT_REPEATED_LEAF = 'set "annotation" to an array of [x0,y0,x1,y1] boxes around the matching child rectangles across parent rectangles'
JSON_EXAMPLE_GROUP_TOTAL = '{"annotation":[[142,150,198,210],[204,216,268,286],[300,300,370,372]],"answer":168}'
JSON_EXAMPLE_PARENT_EXTREMUM = '{"annotation":[[142,150,198,210],[204,216,268,286],[300,300,370,372]],"answer":"Housing"}'
JSON_EXAMPLE_REPEATED_LEAF = '{"annotation":[[142,150,198,210],[404,150,468,210],[672,150,738,210]],"answer":54}'
JSON_EXAMPLE_ANSWER_ONLY_GROUP_TOTAL = '{"answer":168}'
JSON_EXAMPLE_ANSWER_ONLY_PARENT_EXTREMUM = '{"answer":"Housing"}'
JSON_EXAMPLE_ANSWER_ONLY_REPEATED_LEAF = '{"answer":54}'


def render_prompt_artifacts(
    *,
    prompt_key: str,
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
        query_key=str(prompt_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": OBJECT_DESCRIPTION,
            "json_output_contract": JSON_OUTPUT_CONTRACT,
            "json_output_contract_answer_only": JSON_OUTPUT_CONTRACT_ANSWER_ONLY,
            "annotation_hint": str(annotation_hint),
            "answer_hint": ANSWER_HINT_INTEGER,
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
            **dict(dynamic_slot_values),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered)


__all__ = [
    "ANSWER_HINT_PARENT_LABEL",
    "ANNOTATION_HINT_GROUP_TOTAL",
    "ANNOTATION_HINT_PARENT_EXTREMUM",
    "ANNOTATION_HINT_REPEATED_LEAF",
    "JSON_EXAMPLE_ANSWER_ONLY_GROUP_TOTAL",
    "JSON_EXAMPLE_ANSWER_ONLY_PARENT_EXTREMUM",
    "JSON_EXAMPLE_ANSWER_ONLY_REPEATED_LEAF",
    "JSON_EXAMPLE_GROUP_TOTAL",
    "JSON_EXAMPLE_PARENT_EXTREMUM",
    "JSON_EXAMPLE_REPEATED_LEAF",
    "render_prompt_artifacts",
]
