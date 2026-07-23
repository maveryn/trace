"""Trace payload helpers for lane-runner scene tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts

from .rendering import RenderedLaneRunnerScene
from .state import LaneRunnerSceneAxes


def build_lane_runner_common_trace_params(
    *,
    axes: LaneRunnerSceneAxes,
    branch_probabilities: Mapping[str, float],
    extra_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build prompt-query metadata shared by lane-runner outputs."""

    params: dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "row_count": int(axes.row_count),
        "lane_count": int(axes.lane_count),
        "start_lane": int(axes.start_lane),
        "row_count_support": [int(value) for value in axes.row_count_support],
        "start_lane_support": [int(value) for value in axes.start_lane_support],
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "row_count_probabilities": dict(axes.row_count_probabilities),
        "start_lane_probabilities": dict(axes.start_lane_probabilities),
        "query_id_probabilities": {str(key): float(value) for key, value in dict(branch_probabilities).items()},
    }
    params.update(dict(extra_params or {}))
    return params


def build_lane_runner_trace_payload(
    *,
    annotation_artifacts: AnnotationArtifacts,
    annotation_entity_ids: Sequence[str],
    axes: LaneRunnerSceneAxes,
    rendered_scene: RenderedLaneRunnerScene,
    prompt_defaults: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    scene_kind: str,
    selected_branch: str,
    branch_field_name: str,
    execution_trace: Mapping[str, Any],
    relations_extra: Mapping[str, Any] | None = None,
    render_spec_extra: Mapping[str, Any] | None = None,
    witness_type: str = "object_set",
) -> dict[str, Any]:
    """Build the verifier payload from one rendered lane-runner instance."""

    branch_key = str(branch_field_name)
    annotation_ids = tuple(str(value) for value in annotation_entity_ids)
    relations = {
        "scene_variant": str(axes.scene_variant),
        branch_key: str(selected_branch),
        "style_variant": str(axes.style_variant),
        "row_count": int(axes.row_count),
        "lane_count": int(axes.lane_count),
        "start_lane": int(axes.start_lane),
        "annotation_entity_ids": [str(entity_id) for entity_id in annotation_ids],
    }
    relations.update(dict(relations_extra or {}))
    render_spec = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "canvas_width": int(image_size[0]),
        "canvas_height": int(image_size[1]),
        "row_count": int(axes.row_count),
        "lane_count": int(axes.lane_count),
        "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
        "panel_scene_style": dict(rendered_scene.render_map.get("panel_scene_style", {})),
        "lane_runner_style": dict(rendered_scene.render_map.get("lane_runner_style", {})),
    }
    render_spec.update(dict(render_spec_extra or {}))
    execution = {
        "scene_variant": str(axes.scene_variant),
        branch_key: str(selected_branch),
        "style_variant": str(axes.style_variant),
        "row_count": int(axes.row_count),
        "lane_count": int(axes.lane_count),
        "start_lane": int(axes.start_lane),
        **dict(execution_trace),
    }
    return {
        "scene_ir": {
            "scene_kind": str(scene_kind),
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": relations,
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": render_spec,
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": execution,
        "witness_symbolic": {
            "type": str(witness_type),
            "ids": [str(entity_id) for entity_id in annotation_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
        "prompt_metadata": {"bundle_id": str(prompt_defaults["bundle_id"])},
    }


__all__ = [
    "build_lane_runner_common_trace_params",
    "build_lane_runner_trace_payload",
]
