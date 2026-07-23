"""Prompt helpers for the style-legend chart scene."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import DOMAIN, SCENE_ID


BUNDLE_ID = "charts_style_legend_v1"
SCENE_KEY = "style_legend_single_axis"
TASK_KEY = "style_legend_query"

JSON_OUTPUT_CONTRACT = 'Use a valid JSON object with keys "annotation" and "answer" in that order for the final answer.'
JSON_OUTPUT_CONTRACT_ANSWER_ONLY = 'Use a valid JSON object with key "answer" for the final answer.'
OBJECT_DESCRIPTION = "a scientific line chart where series are identified by legend style, including line pattern, marker shape, and color or grayscale tone"
ANSWER_HINT_COUNT = 'set "answer" to the requested count as an integer'
ANSWER_HINT_LABEL = 'set "answer" to the exact visible series label as a string'
ANSWER_HINT_X_LABEL = 'set "answer" to the exact visible x-axis label as a string'
POINT_HINT = 'set "annotation" to a single [x,y] pixel point at the center of the plotted point used to answer'
POINT_SET_HINT = 'set "annotation" to an array of [x,y] pixel points at the centers of the plotted points used to answer'

JSON_EXAMPLES = {
    "extremum_label": '{"annotation":[420,260],"answer":"M7"}',
    "x_label": '{"annotation":[510,230],"answer":"Q3"}',
    "threshold_count": '{"annotation":[[610,210],[610,285],[610,360]],"answer":3}',
}
ANSWER_ONLY_EXAMPLES = {
    "extremum_label": '{"answer":"M7"}',
    "x_label": '{"answer":"Q3"}',
    "threshold_count": '{"answer":3}',
}


def render_prompt_artifacts(
    *,
    prompt_key: str,
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
        bundle_id=BUNDLE_ID,
        scene_key=SCENE_KEY,
        task_key=TASK_KEY,
        query_key=str(prompt_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": OBJECT_DESCRIPTION,
            "json_output_contract": JSON_OUTPUT_CONTRACT,
            "json_output_contract_answer_only": JSON_OUTPUT_CONTRACT_ANSWER_ONLY,
            "annotation_hint": str(annotation_hint),
            "answer_hint": str(answer_hint),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
            **dict(dynamic_slot_values),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered)
