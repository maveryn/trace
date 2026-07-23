"""Output helpers for cone-net scene primitives."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.geometry.shared.annotation_values import PixelAnnotationArtifacts
from .defaults import SCENE_ID, SCENE_KIND, SCENE_VARIANT
from .state import RenderedConeNetScene


def cone_net_trace_payload(
    *,
    rendered: RenderedConeNetScene,
    annotation_artifacts: PixelAnnotationArtifacts,
    query_spec: Mapping[str, Any],
    render_meta: Mapping[str, Any],
    noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    relations: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose scene-neutral trace sections from task-bound semantic fields."""

    return {
        "scene_ir": {
            "scene_kind": SCENE_KIND,
            "scene_id": SCENE_ID,
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "scene_variant": SCENE_VARIANT,
                **dict(relations),
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            "canvas_size": [int(image_size[0]), int(image_size[1])],
            "coord_space": "pixel",
            "post_image_noise": dict(noise_meta),
            **dict(render_meta),
        },
        "render_map": {"coord_space": "pixel", **dict(rendered.render_map)},
        "execution_trace": {
            "scene_id": SCENE_ID,
            "scene_variant": SCENE_VARIANT,
            **dict(execution_trace),
        },
        "witness_symbolic": {
            "type": "cone_sector_net_formula",
            "scene_id": SCENE_ID,
            "source_witness_type": annotation_artifacts.annotation_type,
            "original_annotation_value": annotation_artifacts.value,
            **dict(witness_symbolic),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
    }


__all__ = ["cone_net_trace_payload"]
