"""JSON-ready trace helpers for voxel-cube puzzle tasks."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping


def json_ready(value: Any) -> Any:
    """Convert dataclasses and containers into JSON-friendly values."""

    if hasattr(value, "__dataclass_fields__"):
        return json_ready(asdict(value))
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
    query_spec: Mapping[str, Any],
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
    """Assemble the standard trace payload for one voxel-cube output."""

    return {
        "scene_ir": json_ready(dict(scene_ir)),
        "query_spec": json_ready(dict(query_spec)),
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
