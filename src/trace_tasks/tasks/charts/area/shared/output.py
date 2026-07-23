"""Trace payload helpers for the area chart scene."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.charts.area.shared.rendering import render_trace_sections
from trace_tasks.tasks.charts.area.shared.state import AreaRenderResult


def values_by_series(
    *,
    series_labels: Sequence[str],
    series_values: Mapping[str, Sequence[int]],
) -> dict[str, list[int]]:
    """Return visible integer values keyed by rendered series label."""

    return {
        str(series): [int(value) for value in series_values[str(series)]]
        for series in series_labels
    }


def build_trace_scaffold(
    *,
    rendered: AreaRenderResult,
    scene_variant: str,
    relations: Mapping[str, Any],
    answer_value: Any,
    question_format: str,
    series_values_by_label: Mapping[str, Sequence[int]],
    annotation_pairs: Sequence[tuple[str, str]],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble scene/render trace sections after a public task binds semantics.

    Public task files own target selection, answer calculation, annotation
    pairs, and question format. This helper only records neutral area-scene
    geometry and copies the semantic fields supplied by the task.
    """

    render_spec, render_map = render_trace_sections(
        rendered=rendered,
        scene_variant=str(scene_variant),
    )
    return {
        "scene_ir": {
            "scene_kind": "chart_area_panel",
            "entities": [dict(entity) for entity in rendered.panel.entities],
            "relations": dict(relations),
        },
        "render_spec": render_spec,
        "render_map": render_map,
        "execution_trace": {
            "answer_value": answer_value,
            "question_format": str(question_format),
            "values_by_series": {
                str(label): [int(value) for value in values]
                for label, values in series_values_by_label.items()
            },
            "annotation_pairs": [
                [str(series), str(label)]
                for series, label in annotation_pairs
            ],
            **dict(relations),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
    }


__all__ = ["build_trace_scaffold", "values_by_series"]
