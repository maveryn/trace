"""Trace output helpers for Venn-field icon scenes."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from .rendering import serialize_venn_icon
from .spatial_primitives import venn_to_trace
from .state import VennScenePayload
from .styles import venn_style_trace


def venn_field_render_spec(
    *,
    render_params: Mapping[str, Any],
    panel_geometry: Mapping[str, Any],
    sampled_palette_rgb,
) -> dict[str, Any]:
    """Return shared render metadata for one Venn-field scene."""

    return {
        "canvas_size": [
            int(render_params["canvas_width"]),
            int(render_params["canvas_height"]),
        ],
        "coord_space": "pixel",
        "panel_geometry": dict(panel_geometry),
        "style": venn_style_trace(
            render_params=render_params,
            sampled_palette_rgb=sampled_palette_rgb,
        ),
    }


def venn_scene_summary(scene: VennScenePayload) -> dict[str, Any]:
    """Return shared trace summaries for one rendered Venn scene."""

    category_counts = dict(
        Counter(str(instance.venn_category) for instance in scene.instances)
    )
    target_category_counts = dict(
        Counter(
            str(instance.venn_category)
            for instance in scene.instances
            if instance.matches_target
        )
    )
    return {
        "serialized_instances": [
            serialize_venn_icon(instance) for instance in scene.instances
        ],
        "counted_instance_ids": [
            str(instance.instance_id)
            for instance in scene.instances
            if instance.counted
        ],
        "target_instance_ids": [
            str(instance.instance_id)
            for instance in scene.instances
            if instance.matches_target
        ],
        "category_counts": {
            str(key): int(value) for key, value in category_counts.items()
        },
        "target_category_counts": {
            str(key): int(value) for key, value in target_category_counts.items()
        },
        "venn": venn_to_trace(scene.venn),
        "object_bboxes_px": {
            str(instance.instance_id): [int(value) for value in instance.bbox_xyxy]
            for instance in scene.instances
        },
        "object_centers_px": {
            str(instance.instance_id): [float(value) for value in instance.center_xy]
            for instance in scene.instances
        },
    }


def build_venn_count_trace_payload(
    *,
    public_fields: Mapping[str, Any],
    taxonomy_fields: Mapping[str, Any],
    scene: VennScenePayload,
    render_params: Mapping[str, Any],
    query_spec: Mapping[str, Any],
    scene_kind: str,
    counting_rule: str,
    scene_variant: str,
    question_format: str,
    target_metadata: Mapping[str, Any],
    count_metadata: Mapping[str, Any],
    counted_venn_categories: Any,
    projected_annotation: Mapping[str, Any],
    extra_relation_fields: Mapping[str, Any] | None = None,
    extra_render_map_fields: Mapping[str, Any] | None = None,
    extra_execution_fields: Mapping[str, Any] | None = None,
    extra_witness_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared Venn count trace envelope from task-owned fields."""

    summary = venn_scene_summary(scene)
    categories = [str(value) for value in counted_venn_categories]
    common_fields = dict(public_fields)
    relation_fields = {
        "counting_rule": str(counting_rule),
        **dict(target_metadata),
        "target_count": int(count_metadata["target_count"]),
        "counted_venn_categories": list(categories),
        "category_counts": dict(summary["category_counts"]),
        "target_category_counts": dict(summary["target_category_counts"]),
        "venn": dict(summary["venn"]),
        **dict(extra_relation_fields or {}),
    }
    execution_fields = {
        "scene_variant": str(scene_variant),
        "question_format": str(question_format),
        **dict(target_metadata),
        **dict(count_metadata),
        "object_count": int(scene.object_count),
        "counted_venn_categories": list(categories),
        "category_counts": dict(summary["category_counts"]),
        "target_category_counts": dict(summary["target_category_counts"]),
        "target_instance_ids": list(summary["target_instance_ids"]),
        "counted_instance_ids": list(summary["counted_instance_ids"]),
        "venn": dict(summary["venn"]),
        **dict(extra_execution_fields or {}),
    }
    witness_fields = {
        **dict(target_metadata),
        "answer": int(count_metadata["target_count"]),
        "counted_instance_ids": list(summary["counted_instance_ids"]),
        "counted_venn_categories": list(categories),
        "venn": dict(summary["venn"]),
        **dict(extra_witness_fields or {}),
    }
    return {
        "taxonomy": {
            **dict(taxonomy_fields),
        },
        "scene_ir": {
            **common_fields,
            "scene_kind": str(scene_kind),
            "entities": list(summary["serialized_instances"]),
            "relations": relation_fields,
            "frames": {
                "pixel": {
                    "origin": [0.0, 0.0],
                    "x_positive": "right",
                    "y_positive": "down",
                },
                "panels": dict(scene.panel_geometry),
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            **common_fields,
            **venn_field_render_spec(
                render_params=render_params,
                panel_geometry=scene.panel_geometry,
                sampled_palette_rgb=scene.sampled_palette_rgb,
            ),
        },
        "render_map": {
            "image_id": "img0",
            "object_bboxes_px": dict(summary["object_bboxes_px"]),
            "object_centers_px": dict(summary["object_centers_px"]),
            "target_instance_ids": list(summary["target_instance_ids"]),
            "counted_instance_ids": list(summary["counted_instance_ids"]),
            "venn": dict(summary["venn"]),
            **dict(extra_render_map_fields or {}),
        },
        "execution_trace": {
            **common_fields,
            **execution_fields,
        },
        "witness_symbolic": witness_fields,
        "projected_annotation": {
            **dict(projected_annotation),
            "counted_instance_ids": list(summary["counted_instance_ids"]),
        },
    }


__all__ = [
    "build_venn_count_trace_payload",
    "venn_field_render_spec",
    "venn_scene_summary",
]
