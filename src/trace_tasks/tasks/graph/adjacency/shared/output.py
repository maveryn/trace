"""Trace entity fragment helpers for adjacency-scene outputs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .state import AdjacencyGraphSample, AdjacencyLabelSet, AdjacencyRepresentationRender


def pixel_panel_frames(rendered: AdjacencyRepresentationRender) -> dict[str, Any]:
    """Return the common pixel and panel frame metadata for adjacency traces."""

    return {
        "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
        "panels": dict(rendered.panel_geometry),
    }


def render_spec_fragment(
    *,
    canvas_width: int,
    canvas_height: int,
    rendered: AdjacencyRepresentationRender,
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Return representation-level render metadata shared by adjacency tasks."""

    return {
        "canvas_size": [int(canvas_width), int(canvas_height)],
        "coord_space": "pixel",
        "panel_geometry": dict(rendered.panel_geometry),
        "style": {
            "representation_variant": str(rendered.representation_variant),
            "background_meta": dict(background_meta),
            "post_image_noise_meta": dict(post_noise_meta),
            **dict(rendered.style_meta),
        },
    }


def scene_ir_body(
    *,
    rendered: AdjacencyRepresentationRender,
    entities: Sequence[Mapping[str, Any]],
    relations: Mapping[str, Any],
) -> dict[str, Any]:
    """Return adjacency scene-ir fields that do not depend on public identity."""

    return {
        "scene_kind": "adjacency",
        "entities": [dict(entity) for entity in entities],
        "relations": dict(relations),
        "frames": pixel_panel_frames(rendered),
    }


def query_spec_body(
    *,
    prompt_bundle_id: str,
    prompt_artifacts: Any,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Return prompt/query trace fields after the task binds identity locally."""

    return {
        "template_id": str(prompt_bundle_id),
        "prompt_variant": dict(prompt_artifacts.prompt_variant),
        "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
        "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
        "params": dict(params),
    }


def execution_trace_body(
    *,
    rendered: AdjacencyRepresentationRender,
    fields: Mapping[str, Any],
) -> dict[str, Any]:
    """Return common execution fields without task or query identifiers."""

    return {
        "representation_variant": str(rendered.representation_variant),
        **dict(fields),
    }


def label_query_params(
    labels: AdjacencyLabelSet,
    *,
    label_variant_probabilities: Mapping[str, float],
) -> dict[str, Any]:
    """Return label-axis provenance fields for query-spec params."""

    return {
        "label_variant": str(labels.label_variant),
        "label_variant_probabilities": dict(label_variant_probabilities),
        "label_source_kind": str(labels.label_source_kind),
        "label_bucket": str(labels.label_bucket),
        "label_manifest": str(labels.label_manifest),
        "label_filter": dict(labels.label_filter),
        "label_bucket_probabilities": dict(labels.label_bucket_probabilities),
    }


def label_node_query_params(
    query: Any,
    labels: AdjacencyLabelSet,
) -> dict[str, Any]:
    """Return common query params for labeled adjacency graph tasks."""

    params = {
        "node_count": int(query.node_count),
        "node_count_probabilities": dict(query.node_count_probabilities),
        **label_query_params(labels, label_variant_probabilities=query.label_variant_probabilities),
    }
    if hasattr(query, "extra_edge_count"):
        params.update(
            {
                "extra_edge_count": int(query.extra_edge_count),
                "extra_edge_count_probabilities": dict(query.extra_edge_count_probabilities),
            }
        )
    return params


def component_node_entities(
    sample: AdjacencyGraphSample,
    rendered: AdjacencyRepresentationRender,
    topmost_row_labels: Sequence[str],
) -> list[dict[str, Any]]:
    """Return node row-label entities for component-count traces."""

    topmost_label_set = {str(label) for label in topmost_row_labels}
    return [
        {
            "entity_id": f"node_{label}",
            "entity_kind": "adjacency_row_label",
            "label": str(label),
            "neighbors": list(sample.adjacency.get(str(label), ())),
            "bbox_xyxy": list(rendered.row_label_bboxes[str(label)]),
            "is_component_topmost_row_label": bool(str(label) in topmost_label_set),
        }
        for label in sample.labels
    ]


def component_edge_entities(sample: AdjacencyGraphSample, *, directed: bool) -> list[dict[str, Any]]:
    """Return graph edge entities for component-count traces."""

    return [
        {
            "entity_id": f"edge_{left}_{right}",
            "entity_kind": "adjacency_edge",
            "source_label": str(left),
            "target_label": str(right),
            "directed": bool(directed),
        }
        for left, right in sample.edges
    ]


__all__ = [
    "component_edge_entities",
    "component_node_entities",
    "label_query_params",
    "label_node_query_params",
    "execution_trace_body",
    "pixel_panel_frames",
    "query_spec_body",
    "render_spec_fragment",
    "scene_ir_body",
]
