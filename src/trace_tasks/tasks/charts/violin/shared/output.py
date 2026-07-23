"""Trace payload helpers for violin chart tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.charts.violin.shared.annotations import label_centers
from trace_tasks.tasks.charts.violin.shared.rendering import render_spec_from_artifacts
from trace_tasks.tasks.charts.violin.shared.state import ViolinRenderArtifacts


def mode_values_by_label(support_by_label: Mapping[str, Mapping[str, Any]]) -> dict[str, list[int]]:
    """Return mode locations keyed by visible label."""

    return {
        str(label): [int(value) for value in values["mode_values"]]
        for label, values in support_by_label.items()
    }


def support_span_by_label(support_by_label: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """Return support spans keyed by visible label."""

    return {
        str(label): int(values["support_span"])
        for label, values in support_by_label.items()
    }


def build_trace_payload(
    *,
    prompt_artifacts: Any,
    artifacts: ViolinRenderArtifacts,
    support_by_label: Mapping[str, Mapping[str, Any]],
    trace_extras: Mapping[str, Any],
    answer_label: str,
    annotation_values: Sequence[int],
    relations: Mapping[str, Any],
    prompt_params: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble neutral violin trace sections after public task binding."""

    rendered_scene = artifacts.rendered_scene
    labels = [str(mark["label"]) for mark in rendered_scene.mark_traces]
    return {
        "scene_ir": {
            "scene_kind": "chart_violin_distribution",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": dict(relations),
        },
        "query_spec": {
            "template_id": str(prompt_artifacts.prompt_variant.get("bundle_id", "")),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": dict(prompt_params),
        },
        "render_spec": render_spec_from_artifacts(artifacts),
        "render_map": {
            "image_id": "img0",
            "plot_bbox_px": list(rendered_scene.plot_bbox_px),
            "label_centers_px": label_centers(rendered_scene),
        },
        "execution_trace": {
            "answer_label": str(answer_label),
            "annotation_label": str(answer_label),
            "annotation_values": [int(value) for value in annotation_values],
            "labels": list(labels),
            "mode_values_by_label": mode_values_by_label(support_by_label),
            "support_span_by_label": support_span_by_label(support_by_label),
            "support_by_label": {str(label): dict(values) for label, values in support_by_label.items()},
            "category_count": int(trace_extras["category_count"]),
            "category_count_range": list(trace_extras["category_count_range"]),
            "value_range": list(trace_extras["value_range"]),
            "mark_color_sampling_policy": str(artifacts.mark_style["sampling_policy"]),
            "mark_fill_rgb": list(artifacts.mark_style["mark_fill_rgb"]),
            "mark_outline_rgb": list(artifacts.mark_style["mark_outline_rgb"]),
            **dict(relations),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
    }


__all__ = [
    "build_trace_payload",
    "mode_values_by_label",
    "support_span_by_label",
]
