"""Trace payload helpers for symbolic Turing tape scenes."""

from __future__ import annotations

from typing import Any, Mapping

from ....shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec
from ...shared.unit_size_jitter import with_symbolic_unit_size_jitter

from .state import RenderedTuringScene, TuringRenderParams


def build_turing_trace_payload(
    *,
    scene_name: str,
    prompt_artifacts: PromptTraceArtifacts,
    public_query_id: str,
    params_payload: Mapping[str, Any],
    render_params: TuringRenderParams,
    rendered_scene: RenderedTuringScene,
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    answer_value: Any,
    execution_record: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the shared trace shell after task-owned bindings exist."""

    render_map = {
        "image_id": "img0",
        "scene_bbox_px": [int(value) for value in rendered_scene.scene_bbox_px],
        "item_bboxes_px": {str(key): list(value) for key, value in rendered_scene.item_bboxes.items()},
        "annotation_source": "item_bboxes_px",
        "layout_jitter": dict(rendered_scene.layout_jitter),
        "scene_style": dict(rendered_scene.style_metadata),
    }
    relation_params = dict(params_payload)
    return {
        "scene_ir": {
            "scene_kind": str(scene_name),
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": dict(relation_params),
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(public_query_id),
            params=dict(relation_params),
        ),
        "render_spec": {
            "scene_id": str(scene_name),
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_variant": str(params_payload.get("scene_variant", "")),
            "background_style": dict(background_meta),
            "post_image_noise": dict(post_noise_meta),
            "scene_bbox_px": [int(value) for value in rendered_scene.scene_bbox_px],
            "render_params": {
                "cell_size_px": int(render_params.cell_size_px),
                "grid_gap_px": int(render_params.grid_gap_px),
                "option_card_width_px": int(render_params.option_card_width_px),
                "option_card_height_px": int(render_params.option_card_height_px),
                "option_grid_cell_px": int(render_params.option_grid_cell_px),
            },
            "unit_size_jitter": dict(render_params.unit_size_jitter),
            "layout_jitter": dict(rendered_scene.layout_jitter),
            "scene_style": dict(rendered_scene.style_metadata),
        },
        "render_map": with_symbolic_unit_size_jitter(render_map, render_params.unit_size_jitter),
        "execution_trace": {
            **dict(relation_params),
            "answer_value": answer_value,
            **dict(execution_record),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
    }
