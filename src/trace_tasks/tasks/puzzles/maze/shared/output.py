"""JSON-ready trace helpers for maze puzzle tasks."""

from __future__ import annotations

from typing import Any, Mapping


def json_ready(value: Any) -> Any:
    """Convert tuples and mappings into JSON-friendly containers."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return [json_ready(inner) for inner in value]
    if isinstance(value, list):
        return [json_ready(inner) for inner in value]
    return value


def build_trace_payload(
    *,
    scene_ir: Mapping[str, Any],
    semantic_spec: Mapping[str, Any],
    render_spec: Mapping[str, Any],
    render_map: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    answer_gt: Mapping[str, Any],
    annotation_gt: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
) -> dict[str, Any]:
    """Assemble the standard trace payload for one maze task output."""

    return {
        "scene_ir": json_ready(dict(scene_ir)),
        "query_spec": json_ready(dict(semantic_spec)),
        "render_spec": json_ready(dict(render_spec)),
        "render_map": json_ready(dict(render_map)),
        "execution_trace": json_ready(dict(execution_trace)),
        "witness_symbolic": json_ready(dict(witness_symbolic)),
        "projected_annotation": json_ready(dict(projected_annotation)),
        "answer_gt": json_ready(dict(answer_gt)),
        "annotation_gt": json_ready(dict(annotation_gt)),
        "prompt_spec": {
            "defaults": json_ready(dict(prompt_defaults)),
            "active": json_ready(dict(prompt_artifacts.prompt_variant)),
        },
    }


__all__ = ["build_trace_payload", "json_ready"]
