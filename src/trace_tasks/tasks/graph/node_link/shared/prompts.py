"""Prompt helpers for the graph node-link scene."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Tuple


def resolve_prompt_slot(value: Callable[[Any], str] | str, axes: Any) -> str:
    """Resolve a prompt slot that may depend on node-link axes."""

    if callable(value):
        return str(value(axes))
    return str(value)


def build_graph_prompt_json_examples(*, annotation_value: Any, answer_value: Any) -> Tuple[str, str]:
    """Return compact JSON examples for both output modes."""

    return (
        json.dumps(
            {"annotation": annotation_value, "answer": answer_value},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ),
        json.dumps(
            {"answer": answer_value},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ),
    )


__all__ = ["build_graph_prompt_json_examples", "resolve_prompt_slot"]
