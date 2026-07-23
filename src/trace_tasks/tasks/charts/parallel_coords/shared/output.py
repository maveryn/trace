"""Trace payload helpers for the parallel-coordinates chart scene."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .rendering import render_map_payload, render_spec_payload
from .state import ParallelDataset, ParallelRenderResult


def profile_rows(dataset: ParallelDataset) -> list[dict[str, Any]]:
    """Return visible profile rows in JSON-stable form."""

    return [
        {
            "profile_id": str(profile.profile_id),
            "label": str(profile.label),
            "values": [int(value) for value in profile.values],
            "color_rgb": list(profile.color_rgb),
        }
        for profile in dataset.profiles
    ]


def build_trace_payload(
    *,
    dataset: ParallelDataset,
    rendered: ParallelRenderResult,
    prompt_spec: Mapping[str, Any],
    relation_payload: Mapping[str, Any],
    execution_payload: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    annotation_type: str,
    annotation_profile_ids: Sequence[str],
) -> dict[str, Any]:
    """Combine neutral scene projection and task-owned semantic trace fields."""

    rows = profile_rows(dataset)
    axis_pair = [int(dataset.query.axis_i), int(dataset.query.axis_j)]
    answer_value: int | str = (
        int(dataset.query.answer)
        if str(dataset.query.answer_type) == "integer"
        else str(dataset.query.answer)
    )
    return {
        "scene_ir": {
            "scene_kind": "chart_parallel_coords",
            "entities": [dict(entity) for entity in rendered.rendered_scene.entities],
            "relations": {
                "scene_variant": str(dataset.scene_variant),
                "answer": answer_value,
                "axis_pair": list(axis_pair),
                "annotation_profile_ids": [str(value) for value in annotation_profile_ids],
                **dict(relation_payload),
            },
        },
        "query_spec": dict(prompt_spec),
        "render_spec": render_spec_payload(rendered, dataset),
        "render_map": render_map_payload(rendered),
        "execution_trace": {
            "scene_id": "parallel_coords",
            "scene_variant": str(dataset.scene_variant),
            "question_format": "parallel_coords_query",
            "answer": answer_value,
            "answer_type": str(dataset.query.answer_type),
            "annotation_type": str(annotation_type),
            "axis_i": int(dataset.query.axis_i),
            "axis_j": int(dataset.query.axis_j),
            "axis_i_label": str(dataset.metrics[int(dataset.query.axis_i)]),
            "axis_j_label": str(dataset.metrics[int(dataset.query.axis_j)]),
            "metrics": [str(value) for value in dataset.metrics],
            "profiles": list(rows),
            "threshold": dataset.query.threshold,
            "annotation_profile_ids": [str(value) for value in annotation_profile_ids],
            "crossing_pairs": [list(pair) for pair in dataset.query.crossing_pairs],
            **dict(dataset.query.params),
            **dict(execution_payload),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
    }


__all__ = ["build_trace_payload", "profile_rows"]
