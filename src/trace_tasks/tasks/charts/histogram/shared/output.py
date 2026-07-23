"""Trace payload helpers for histogram chart scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.charts.histogram.shared.rendering import counts_by_label, render_trace_sections
from trace_tasks.tasks.charts.histogram.shared.state import HistogramRenderArtifacts


def histogram_relations(
    *,
    prompt_key: str,
    dataset_variant: str,
    trace_extras: Mapping[str, Any],
    annotation_labels: Sequence[str],
    query_probabilities: Mapping[str, float],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return trace relations supplied by a public histogram task."""

    return {
        "prompt_key": str(prompt_key),
        "histogram_dataset_variant": str(dataset_variant),
        "scene_variant": "histogram",
        "annotation_labels": [str(label) for label in annotation_labels],
        "query_id_probabilities": dict(query_probabilities),
        **dict(trace_extras),
        **dict(extra or {}),
    }


def build_trace_scaffold(
    *,
    artifacts: HistogramRenderArtifacts,
    relations: Mapping[str, Any],
    answer_value: Any,
    question_format: str,
    annotation_labels: Sequence[str],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble neutral histogram trace sections after task binding."""

    rendered_scene = artifacts.rendered_scene
    render_spec, render_map = render_trace_sections(artifacts)
    labels = [str(mark["label"]) for mark in rendered_scene.mark_traces]
    bin_counts = [int(mark["value"]) for mark in rendered_scene.mark_traces]
    return {
        "scene_ir": {
            "scene_kind": "chart_histogram_distribution",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": dict(relations),
        },
        "render_spec": render_spec,
        "render_map": render_map,
        "execution_trace": {
            "answer_value": answer_value,
            "question_format": str(question_format),
            "labels": list(labels),
            "bin_counts": list(bin_counts),
            "counts_by_label": counts_by_label(rendered_scene),
            "annotation_labels": [str(label) for label in annotation_labels],
            **dict(relations),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
    }


__all__ = ["build_trace_scaffold", "histogram_relations"]
