"""Prompt constants and JSON examples for the pipe-network graph scene."""

from __future__ import annotations

import json
from typing import Any, Tuple


PROMPT_BUNDLE_ID = "graph_pipe_network_v1"
SCENE_PROMPT_KEY = "pipe_network"
TASK_PROMPT_KEY = "pipe_network_query"
JSON_OUTPUT_CONTRACT = 'Use a valid JSON object with keys "annotation" and "answer" in that order for the final answer.'
JSON_OUTPUT_CONTRACT_ANSWER_ONLY = 'Use a valid JSON object with key "answer" for the final answer.'
OBJECT_DESCRIPTION = "a labeled pipe-junction network with open pipes and blocked pipes (blocked pipes are marked with a red X)"
ANSWER_HINT = 'set "answer" to the requested count or length as an integer'


def json_examples_for_annotation(annotation_type: str) -> Tuple[str, str]:
    """Return examples for both output modes for a pipe annotation type."""

    annotation: Any = [[180, 220], [310, 220], [430, 300]]
    answer = 3
    if str(annotation_type) == "segment_set":
        annotation = [[[180, 220], [310, 220]], [[310, 220], [430, 300]]]
        answer = 2
    elif str(annotation_type) == "point_sequence":
        annotation = [[180, 220], [310, 220], [430, 300]]
        answer = 2
    return (
        json.dumps({"annotation": annotation, "answer": answer}, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        json.dumps({"answer": answer}, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
    )


__all__ = [
    "ANSWER_HINT",
    "JSON_OUTPUT_CONTRACT",
    "JSON_OUTPUT_CONTRACT_ANSWER_ONLY",
    "OBJECT_DESCRIPTION",
    "PROMPT_BUNDLE_ID",
    "SCENE_PROMPT_KEY",
    "TASK_PROMPT_KEY",
    "json_examples_for_annotation",
]
