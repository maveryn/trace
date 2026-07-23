"""Prompt constants and JSON examples for the metro graph scene."""

from __future__ import annotations

import json
from typing import Any, Tuple

PROMPT_BUNDLE_ID = "graph_metro_v1"
SCENE_PROMPT_KEY = "metro_route_map"
TASK_PROMPT_KEY = "metro_route_query"
JSON_OUTPUT_CONTRACT = 'Use a valid JSON object with keys "annotation" and "answer" in that order for the final answer.'
JSON_OUTPUT_CONTRACT_ANSWER_ONLY = 'Use a valid JSON object with key "answer" for the final answer.'
OBJECT_DESCRIPTION = "a labeled metro route map with colored routes and stations"
ANSWER_HINT = 'set "answer" to the requested count or length as an integer'


def json_examples_for_annotation(annotation_type: str) -> Tuple[str, str]:
    """Return examples for both output modes for a metro annotation type."""

    annotation: Any = [[180, 220], [310, 180]]
    if str(annotation_type) == "point_sequence":
        annotation = [[180, 220], [310, 180], [430, 260]]
    return (
        json.dumps({"annotation": annotation, "answer": 2}, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        json.dumps({"answer": 2}, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
    )


__all__ = [
    "ANSWER_HINT", "JSON_OUTPUT_CONTRACT", "JSON_OUTPUT_CONTRACT_ANSWER_ONLY", "OBJECT_DESCRIPTION",
    "PROMPT_BUNDLE_ID", "SCENE_PROMPT_KEY", "TASK_PROMPT_KEY", "json_examples_for_annotation",
]
