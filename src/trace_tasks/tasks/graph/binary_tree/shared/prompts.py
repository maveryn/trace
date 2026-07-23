"""Prompt helper fragments for graph binary-tree scene tasks."""

from __future__ import annotations

import json
from typing import Sequence, Tuple


def count_prompt_json_examples() -> Tuple[str, str]:
    """Return JSON examples for count tasks."""

    return (
        json.dumps({"annotation": [[180, 148], [495, 455]], "answer": 2}, separators=(",", ":")),
        json.dumps({"answer": 2}, separators=(",", ":")),
    )


def traversal_prompt_json_examples() -> Tuple[str, str]:
    """Return JSON examples for traversal sequence tasks."""

    return (
        json.dumps(
            {
                "annotation": [[180, 148], [274, 274], [495, 455]],
                "answer": "M",
            },
            separators=(",", ":"),
        ),
        json.dumps({"answer": "M"}, separators=(",", ":")),
    )


def keyed_node_prompt_json_examples(roles: Sequence[str]) -> Tuple[str, str]:
    """Return JSON examples for keyed node-label tasks."""

    example_points = (
        [180, 148],
        [495, 455],
        [274, 274],
    )
    annotation = {
        str(role): list(point)
        for role, point in zip(tuple(str(role) for role in roles), example_points)
    }
    return (
        json.dumps({"annotation": annotation, "answer": "M"}, separators=(",", ":")),
        json.dumps({"answer": "M"}, separators=(",", ":")),
    )


def operation_path_prompt_json_examples() -> Tuple[str, str]:
    """Return JSON examples for ordered BST path tasks."""

    return (
        json.dumps({"annotation": [[180, 148], [274, 274]], "answer": "42"}, separators=(",", ":")),
        json.dumps({"answer": "42"}, separators=(",", ":")),
    )


def heap_violation_prompt_json_examples() -> Tuple[str, str]:
    """Return JSON examples for keyed heap-violation tasks."""

    return (
        json.dumps(
            {
                "annotation": {
                    "parent": [180, 148],
                    "child": [274, 274],
                },
                "answer": "42",
            },
            separators=(",", ":"),
        ),
        json.dumps({"answer": "42"}, separators=(",", ":")),
    )


__all__ = [
    "count_prompt_json_examples",
    "heap_violation_prompt_json_examples",
    "keyed_node_prompt_json_examples",
    "operation_path_prompt_json_examples",
    "traversal_prompt_json_examples",
]
