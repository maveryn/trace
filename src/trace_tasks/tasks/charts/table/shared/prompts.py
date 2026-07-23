"""Prompt assembly for styled table chart tasks."""

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


COUNTING_BUNDLE_ID = "charts_table_counting_v1"
RANKING_BUNDLE_ID = "charts_table_ranking_v1"
STATISTICS_BUNDLE_ID = "charts_table_statistics_v1"
TEMPORAL_BUNDLE_ID = "charts_table_temporal_v1"

SCENE_KEY_COUNTING = "styled_table_counting"
SCENE_KEY_RANKING = "styled_table_ranking"
SCENE_KEY_STATISTICS = "styled_table_statistics"
SCENE_KEY_TEMPORAL = "styled_table_temporal"

TASK_KEY_COUNTING = "value_count_query"
TASK_KEY_RANKING = "rank_label_query"
TASK_KEY_SUMMARY = "summary_value_query"
TASK_KEY_FILTERED = "filtered_subset_value_query"
TASK_KEY_TEMPORAL = "temporal_value_query"

JSON_OUTPUT_CONTRACT = 'Use a valid JSON object with keys "annotation" and "answer" in that order for the final answer.'
JSON_OUTPUT_CONTRACT_ANSWER_ONLY = 'Use a valid JSON object with key "answer" for the final answer.'

OBJECT_DESCRIPTIONS = {
    "spreadsheet": "a table with one Name column and several data columns",
    "zebra": "a table with one Name column and several data columns",
    "ledger": "a table with one Name column and several data columns",
    "card_table": "a table with one Name column and several data columns",
}
TEMPORAL_OBJECT_DESCRIPTIONS = {
    key: value.replace("several data columns", "several year columns in chronological order")
    for key, value in OBJECT_DESCRIPTIONS.items()
}

ANSWER_HINT_COUNT = 'set "answer" to the exact count as an integer'
ANSWER_HINT_INTEGER = 'set "answer" to the exact integer result'
ANSWER_HINT_ROW_LABEL = 'set "answer" to the exact row label as a string'

ANNOTATION_HINT_CELL_SET = 'set "annotation" to an array of [x0,y0,x1,y1] boxes around every matching table cell, or [] if none match'
ANNOTATION_HINT_COLUMN_BOX = 'set "annotation" to one [x0,y0,x1,y1] box around the queried column values'
ANNOTATION_HINT_RANK_CELL = 'set "annotation" to one [x0,y0,x1,y1] box around the answer row cell in the queried column'
ANNOTATION_HINT_FILTER_MAP = 'set "annotation" to an object with "filter_cells" and "target_cells", each mapping to arrays of [x0,y0,x1,y1] boxes for the selected rows'
ANNOTATION_HINT_TEMPORAL = 'set "annotation" to an array of [x0,y0,x1,y1] boxes for the queried year cells from both named rows across the interval'
ANNOTATION_HINT_TEMPORAL_ROW_SPAN_MAP = 'set "annotation" to an object mapping each queried row label to one [x0,y0,x1,y1] box surrounding that row span across the queried year interval'

def object_description(scene_variant: str, *, temporal: bool = False) -> str:
    descriptions = TEMPORAL_OBJECT_DESCRIPTIONS if bool(temporal) else OBJECT_DESCRIPTIONS
    return str(descriptions.get(str(scene_variant), descriptions["spreadsheet"]))


def render_prompt_artifacts(
    *,
    bundle_id: str,
    scene_key: str,
    task_key: str,
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
        bundle_id=str(bundle_id),
        scene_key=str(scene_key),
        task_key=str(task_key),
        query_key=str(prompt_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
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


__all__ = [
    "ANNOTATION_HINT_CELL_SET",
    "ANNOTATION_HINT_COLUMN_BOX",
    "ANNOTATION_HINT_FILTER_MAP",
    "ANNOTATION_HINT_RANK_CELL",
    "ANNOTATION_HINT_TEMPORAL",
    "ANNOTATION_HINT_TEMPORAL_ROW_SPAN_MAP",
    "ANSWER_HINT_COUNT",
    "ANSWER_HINT_INTEGER",
    "ANSWER_HINT_ROW_LABEL",
    "COUNTING_BUNDLE_ID",
    "RANKING_BUNDLE_ID",
    "SCENE_KEY_COUNTING",
    "SCENE_KEY_RANKING",
    "SCENE_KEY_STATISTICS",
    "SCENE_KEY_TEMPORAL",
    "STATISTICS_BUNDLE_ID",
    "TASK_KEY_COUNTING",
    "TASK_KEY_FILTERED",
    "TASK_KEY_RANKING",
    "TASK_KEY_SUMMARY",
    "TASK_KEY_TEMPORAL",
    "TEMPORAL_BUNDLE_ID",
    "object_description",
    "render_prompt_artifacts",
]
