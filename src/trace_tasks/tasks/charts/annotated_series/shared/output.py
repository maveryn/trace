"""Trace and output scaffolding for annotated-series chart tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.charts.annotated_series.shared.defaults import (
    FALLBACK_CHART_DEFAULTS,
    generation_int,
)
from trace_tasks.tasks.charts.annotated_series.shared.rendering import (
    context_entities,
    label_maps,
    render_trace_sections,
)
from trace_tasks.tasks.charts.annotated_series.shared.state import (
    FinalRender,
    MarkupRender,
    RenderedBaseSeries,
    SeriesSample,
)


def mark_count_range_for_params(params: Mapping[str, Any]) -> list[int]:
    return [
        generation_int(params, "mark_count_min", FALLBACK_CHART_DEFAULTS.mark_count_min),
        generation_int(params, "mark_count_max", FALLBACK_CHART_DEFAULTS.mark_count_max),
    ]


def common_scene_fields(
    *,
    sample: SeriesSample,
    mark_count_range: Sequence[int],
    semantic_params: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "scene_variant": str(sample.scene_variant),
        "scene_variant_probabilities": dict(sample.scene_variant_probabilities),
        "mark_count": int(len(sample.labels)),
        "mark_count_range": [int(value) for value in mark_count_range],
        **dict(semantic_params),
    }


def build_trace_payload_scaffold(
    *,
    sample: SeriesSample,
    base: RenderedBaseSeries,
    markup: MarkupRender,
    final: FinalRender,
    answer_value: Any,
    annotation_kind: str,
    annotation_labels: Sequence[str],
    mark_count_range: Sequence[int],
    question_format: str,
    semantic_params: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build common trace sections after a task binds answer and annotation.

    The caller supplies objective-owned answer, annotation, and semantic fields.
    This helper only assembles scene/render metadata shared by annotated-series
    tasks so trace payloads stay consistent.
    """
    render_spec, render_map = render_trace_sections(base=base, markup=markup, final=final)
    render_spec["scene_variant"] = str(sample.scene_variant)
    maps = label_maps(base.rendered_scene)
    annotation_label_list = [str(label) for label in annotation_labels]
    prompt_fields = common_scene_fields(
        sample=sample,
        mark_count_range=mark_count_range,
        semantic_params=semantic_params,
    )
    mark_style_fields = {
        str(key): value
        for key, value in base.mark_style.items()
        if key not in {"sampling_policy", "mark_fill_rgb", "mark_outline_rgb"}
    }
    trace_payload = {
        "scene_ir": {
            "scene_kind": f"chart_{sample.scene_variant}_annotated_series",
            "entities": [dict(entity) for entity in base.rendered_scene.entities]
            + [dict(entity) for entity in markup.entities]
            + context_entities(final),
            "relations": {
                "scene_variant": str(sample.scene_variant),
                "annotation_kind": str(annotation_kind),
                "annotation_labels": annotation_label_list,
                **dict(semantic_params),
            },
        },
        "render_spec": render_spec,
        "render_map": render_map,
        "execution_trace": {
            "scene_variant": str(sample.scene_variant),
            "answer_value": answer_value,
            "annotation_labels": annotation_label_list,
            "labels": [str(label) for label in sample.labels],
            "values": [int(value) for value in sample.values],
            "values_by_label": dict(maps["values_by_label"]),
            "mark_count": int(len(sample.labels)),
            "mark_count_range": [int(value) for value in mark_count_range],
            "scene_variant_probabilities": dict(sample.scene_variant_probabilities),
            "question_format": str(question_format),
            "mark_color_sampling_policy": str(base.mark_style["sampling_policy"]),
            "mark_fill_rgb": list(base.mark_style["mark_fill_rgb"]),
            "mark_outline_rgb": list(base.mark_style["mark_outline_rgb"]),
            **mark_style_fields,
            **dict(semantic_params),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
    }
    return trace_payload, prompt_fields


__all__ = [
    "build_trace_payload_scaffold",
    "common_scene_fields",
    "mark_count_range_for_params",
]
