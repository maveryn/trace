"""Identity-free trace-payload primitives for angle-relations scene tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.geometry.shared.metadata_serialization import geometry_json_ready

from .state import SCENE_ID


def angle_relation_trace_payload(
    *,
    rendered_context: Any,
    annotation_artifacts: Any,
    query_spec: Mapping[str, Any],
    scene_relations: Mapping[str, Any],
    execution_fields: Mapping[str, Any],
    witness_fields: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble query-neutral render sections plus task-owned trace fields."""

    rendered = rendered_context.rendered_scene
    rendered_entities = geometry_json_ready(rendered.scene_entities, round_floats=False)
    rendered_map = geometry_json_ready(rendered.render_map, round_floats=False)
    rendered_witness = geometry_json_ready(rendered.witness, round_floats=False)
    return {
        "scene_ir": {
            "scene_kind": "geometry_angle_relation",
            "scene_id": SCENE_ID,
            "entities": rendered_entities,
            "relations": dict(scene_relations),
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            "canvas_size": [int(rendered_context.image.size[0]), int(rendered_context.image.size[1])],
            "coord_space": "pixel",
            "post_image_noise": dict(rendered_context.post_noise_meta),
            **dict(rendered_context.render_meta),
        },
        "render_map": {"coord_space": "pixel", **dict(rendered_map)},
        "execution_trace": {
            **dict(execution_fields),
            "reasoning_steps": int(rendered.reasoning_steps),
            **dict(rendered_witness),
        },
        "witness_symbolic": {
            **dict(witness_fields),
            "source_witness_type": str(annotation_artifacts.annotation_type),
            "original_annotation_value": dict(annotation_artifacts.value),
            **dict(rendered_witness),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
    }


__all__ = ["angle_relation_trace_payload"]
