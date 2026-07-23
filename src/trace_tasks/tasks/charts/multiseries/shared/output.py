
from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.charts.multiseries.shared.rendering import (
    render_map_payload,
    render_spec_payload,
)
from trace_tasks.tasks.charts.multiseries.shared.state import MultiseriesRenderResult


def common_trace_fields(
    *,
    answer_label: str,
    annotation_values: Sequence[int],
    queried_series_labels: Sequence[str],
    value_range: Sequence[int],
    answer_type: str,
    extra: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build common objective trace fields for public multiseries tasks."""

    normalized_extra = dict(extra)
    normalized_annotation_values = [int(value) for value in annotation_values]
    common = {
        "answer_label": str(answer_label),
        "annotation_values": list(normalized_annotation_values),
        "queried_series_labels": [str(label) for label in queried_series_labels],
        **normalized_extra,
    }
    return (
        dict(common),
        {
            "answer_label": str(answer_label),
            "value_range": [int(value) for value in value_range],
            **normalized_extra,
        },
        {
            "answer_label": str(answer_label),
            "answer_type": str(answer_type),
            "annotation_values": list(normalized_annotation_values),
            **normalized_extra,
        },
    )


def build_trace_payload(
    *,
    result: MultiseriesRenderResult,
    prompt_query_spec: Mapping[str, Any],
    relation_payload: Mapping[str, Any],
    execution_base: Mapping[str, Any],
    scene_variant: str,
    question_format: str,
    trace_extras: Mapping[str, Any],
    execution_extra: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble renderer, prompt, and objective fields into verifier metadata.

    Public task files provide the objective-specific relations and execution
    fields. This shared helper only combines those fields with the neutral
    multiseries render projection, so it must not branch on task or query
    identity.
    """

    execution_trace = {
        **dict(execution_base),
        "category_labels": list(result.category_labels),
        "series_labels": list(result.series_labels),
        "queried_series_labels": list(trace_extras.get("queried_series_labels", [])),
        "series_count": int(trace_extras["series_count"]),
        "series_count_range": list(trace_extras["series_count_range"]),
        "category_count": int(trace_extras["category_count"]),
        "category_count_range": list(trace_extras["category_count_range"]),
        "value_range": list(trace_extras.get("value_range", [])),
        "values_by_category": dict(trace_extras.get("values_by_category", {})),
        "question_format": str(question_format),
        "mark_color_sampling_policy": str(result.mark_style["sampling_policy"]),
        **dict(execution_extra),
        **{str(key): value for key, value in result.mark_style.items() if key != "sampling_policy"},
    }
    return {
        "scene_ir": {
            "scene_kind": f"chart_{str(scene_variant)}_multiseries",
            "entities": [dict(entity) for entity in result.rendered_scene.entities],
            "relations": dict(relation_payload),
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": render_spec_payload(result, scene_variant=str(scene_variant)),
        "render_map": render_map_payload(result),
        "execution_trace": execution_trace,
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
    }


__all__ = ["build_trace_payload", "common_trace_fields"]
