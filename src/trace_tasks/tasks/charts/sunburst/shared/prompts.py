"""Prompt assembly for the sunburst chart scene."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import PROMPT_DEFAULTS
from .state import DOMAIN, SCENE_ID


BUNDLE_ID = "charts_sunburst_v1"
SCENE_KEY = "composition_sunburst_hierarchy"
TASK_KEY = "sunburst_hierarchy_query"
OBJECT_DESCRIPTION = (
    "a not-to-scale concentric hierarchy chart with parent categories in the inner ring, "
    "subgroups in the middle ring, and outer leaves with printed integer values"
)

JSON_OUTPUT_CONTRACT = 'Use a valid JSON object with keys "annotation" and "answer" in that order for the final answer.'
JSON_OUTPUT_CONTRACT_ANSWER_ONLY = 'Use a valid JSON object with key "answer" for the final answer.'

ANSWER_HINT_INTEGER = 'set "answer" to the requested integer'
ANSWER_HINT_LABEL = 'set "answer" to the exact parent category label'
ANNOTATION_HINT_LEAF_VALUES = 'set "annotation" to an array of [x,y] pixel points at the centers of the printed outer leaf values used to answer'

JSON_EXAMPLE_TOTAL = '{"annotation":[[885,264],[955,326],[995,404]],"answer":105}'
JSON_EXAMPLE_LABEL = '{"annotation":[[865,244],[945,334],[445,534]],"answer":"Ablation"}'
JSON_EXAMPLE_COUNT = '{"annotation":[[865,244],[945,334],[995,424]],"answer":2}'
ANSWER_ONLY_EXAMPLE_TOTAL = '{"answer":105}'
ANSWER_ONLY_EXAMPLE_LABEL = '{"answer":"Ablation"}'
ANSWER_ONLY_EXAMPLE_COUNT = '{"answer":2}'


def render_prompt_artifacts(
    *,
    prompt_key: str,
    answer_hint: str,
    json_example: str,
    json_example_answer_only: str,
    dynamic_slot_values: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    rendered = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(PROMPT_DEFAULTS.get("bundle_id", BUNDLE_ID)),
        scene_key=SCENE_KEY,
        task_key=TASK_KEY,
        query_key=str(prompt_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": OBJECT_DESCRIPTION,
            "json_output_contract": JSON_OUTPUT_CONTRACT,
            "json_output_contract_answer_only": JSON_OUTPUT_CONTRACT_ANSWER_ONLY,
            "annotation_hint": ANNOTATION_HINT_LEAF_VALUES,
            "answer_hint": str(answer_hint),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
            **dict(dynamic_slot_values),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered)


__all__ = [
    "ANSWER_HINT_INTEGER",
    "ANSWER_HINT_LABEL",
    "ANSWER_ONLY_EXAMPLE_COUNT",
    "ANSWER_ONLY_EXAMPLE_LABEL",
    "ANSWER_ONLY_EXAMPLE_TOTAL",
    "BUNDLE_ID",
    "JSON_EXAMPLE_COUNT",
    "JSON_EXAMPLE_LABEL",
    "JSON_EXAMPLE_TOTAL",
    "render_prompt_artifacts",
]
