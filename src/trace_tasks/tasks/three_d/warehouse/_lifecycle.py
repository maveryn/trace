"""Scene-private output assembly for warehouse option-label tasks."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence

from ....core.types import TypedValue
from ....core.visual.background import make_background_canvas
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.annotation_artifacts import bbox_annotation_artifacts
from ...shared.config_defaults import required_group_defaults
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)
from ..shared.object_scene import POINT_LABELS
from ..shared.canvas import render_params_canvas_metadata
from ..shared.option_panel import build_text_option_choices
from .shared.state import _WarehouseRenderParams


def build_warehouse_option_label_task_output(
    *,
    domain: str,
    scene_id: str,
    prompt_defaults: Mapping[str, Any],
    prompt_query_key: str,
    dynamic_prompt_slots: Mapping[str, Any],
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    query_id: str,
    query_probabilities: Mapping[str, float],
    scene_variant: str,
    scene_probabilities: Mapping[str, float],
    candidate_count: int,
    candidate_count_probabilities: Mapping[str, float],
    context_object_count: int,
    context_object_count_probabilities: Mapping[str, float],
    camera_yaw_band_index: int,
    camera_yaw_probabilities: Mapping[str, float],
    render_params: _WarehouseRenderParams,
    dataset: Mapping[str, Any],
    render_scene: Callable[..., Any],
    candidate_specs: Sequence[Mapping[str, Any]],
    scene_kind: str,
    view_family: str,
    scene_relation_fields: Mapping[str, Any] | None = None,
    query_param_fields: Mapping[str, Any] | None = None,
    render_spec_fields: Mapping[str, Any] | None = None,
    execution_trace_fields: Mapping[str, Any] | None = None,
) -> TaskOutput:
    """Render a warehouse option task and assemble shared verifier sections."""

    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=dict(background_defaults),
    )
    option_choices = build_text_option_choices(candidate_specs)
    rendered_scene = render_scene(
        background,
        dataset=dataset,
        render_params=render_params,
        option_choices=option_choices,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=dict(noise_defaults),
    )

    prompt_default_values = required_group_defaults(
        prompt_defaults,
        ("bundle_id", "scene_key", "task_key"),
        context=f"prompt defaults for warehouse option task",
    )
    prompt_selection = render_task_prompt_variants(
        domain=str(domain),
        scene_id=str(scene_id),
        bundle_id=str(prompt_default_values["bundle_id"]),
        scene_key=str(prompt_default_values["scene_key"]),
        task_key=str(prompt_default_values["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(dynamic_prompt_slots),
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

    answer_label = str(dataset["answer_label"])
    answer_gt = TypedValue(type="option_letter", value=str(answer_label))
    annotation_bboxes = [
        [round(float(value), 3) for value in bbox]
        for bbox in rendered_scene.annotation_bboxes
    ]
    if len(annotation_bboxes) != 1:
        raise RuntimeError("warehouse option-label task expected exactly one annotation bbox")
    annotation_payload = bbox_annotation_artifacts(annotation_bboxes[0])
    solver_trace = dict(dataset["solver_trace"])
    option_descriptor_by_label = {
        str(choice["label"]): str(choice["descriptor"])
        for choice in rendered_scene.option_choices
    }

    relation_fields: Dict[str, Any] = {
        "scene_variant": str(scene_variant),
        "candidate_count": int(dataset["candidate_count"]),
        "context_object_count": int(dataset["context_object_count"]),
        "answer_label": str(answer_label),
        "answer_object_id": str(dataset["answer_object_id"]),
        "view_family": str(view_family),
    }
    relation_fields.update(dict(scene_relation_fields or {}))

    query_params: Dict[str, Any] = {
        "query_id": str(query_id),
        "query_id_probabilities": dict(query_probabilities),
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": dict(scene_probabilities),
        "candidate_count": int(candidate_count),
        "candidate_count_probabilities": dict(candidate_count_probabilities),
        "context_object_count": int(context_object_count),
        "context_object_count_probabilities": dict(context_object_count_probabilities),
        "camera_yaw_band_index": int(camera_yaw_band_index),
        "camera_yaw_band_probabilities": dict(camera_yaw_probabilities),
        "answer_label_probabilities": {
            str(label): round(1.0 / float(candidate_count), 8)
            for label in POINT_LABELS[: int(candidate_count)]
        },
    }
    query_params.update(dict(query_param_fields or {}))

    render_spec: Dict[str, Any] = {
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(image.height),
        "scene_canvas_preset": str(render_params.canvas_preset),
        "scene_canvas_width": int(render_params.canvas_width),
        "scene_canvas_height": int(render_params.canvas_height),
        "scene_canvas_policy": str(render_params.canvas_policy),
        **render_params_canvas_metadata(render_params),
        "final_canvas_width": int(image.width),
        "final_canvas_height": int(image.height),
        "final_canvas_pixels": int(image.width) * int(image.height),
        "option_panel_height_px": int(rendered_scene.option_panel_height_px),
        "coord_space": "pixel",
        "scene_variant": str(scene_variant),
        "background_style": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
        "camera": dict(dataset["camera"]),
        "projection_frame": dict(dataset["projection_frame"]),
        "label_font_size_px": int(render_params.label_font_size_px),
    }
    render_spec.update(dict(render_spec_fields or {}))

    render_map = {
        "image_id": "img0",
        "scene_bbox_px": list(rendered_scene.scene_bbox_px),
        "warehouse_bbox_px": list(rendered_scene.warehouse_bbox_px),
        "object_bboxes_px": {
            str(key): list(value)
            for key, value in rendered_scene.object_bboxes_px.items()
        },
        "object_centers_px": {
            str(key): list(value)
            for key, value in rendered_scene.object_centers_px.items()
        },
        "candidate_bboxes_px": {
            str(key): list(value)
            for key, value in rendered_scene.candidate_bboxes_px.items()
        },
        "candidate_centers_px": {
            str(key): list(value)
            for key, value in rendered_scene.candidate_centers_px.items()
        },
        "option_panel_bbox_px": list(rendered_scene.option_panel_bbox_px),
        "option_panel_height_px": int(rendered_scene.option_panel_height_px),
        "option_choice_bboxes_px": {
            str(key): list(value)
            for key, value in rendered_scene.option_choice_bboxes_px.items()
        },
        "option_choices": [dict(choice) for choice in rendered_scene.option_choices],
        "context_object_bboxes_px": {
            str(key): list(value)
            for key, value in rendered_scene.context_object_bboxes_px.items()
        },
        "context_object_centers_px": {
            str(key): list(value)
            for key, value in rendered_scene.context_object_centers_px.items()
        },
        "reference_object_bboxes_px": {
            str(key): list(value)
            for key, value in rendered_scene.reference_object_bboxes_px.items()
        },
        "reference_object_centers_px": {
            str(key): list(value)
            for key, value in rendered_scene.reference_object_centers_px.items()
        },
        "target_object_bboxes_px": {
            str(key): list(rendered_scene.object_bboxes_px[str(key)])
            for key in dataset["target_object_ids"]
        },
    }

    execution_trace: Dict[str, Any] = {
        "query_id": str(query_id),
        "scene_id": str(scene_id),
        "scene_variant": str(scene_variant),
        "candidate_count": int(dataset["candidate_count"]),
        "context_object_count": int(dataset["context_object_count"]),
        "object_count": int(dataset["object_count"]),
        "answer_label": str(answer_label),
        "answer_object_id": str(dataset["answer_object_id"]),
        "answer_object_type": str(dataset["answer_object_type"]),
        "target_object_ids": [str(value) for value in dataset["target_object_ids"]],
        "option_choices": [dict(choice) for choice in rendered_scene.option_choices],
        "option_descriptor_by_label": dict(option_descriptor_by_label),
        "context_object_specs": [dict(spec) for spec in dataset["context_object_specs"]],
        "object_specs": [dict(spec) for spec in dataset["object_specs"]],
        "object_type_counts": dict(dataset["object_type_counts"]),
        "camera": dict(dataset["camera"]),
        "projection_frame": dict(dataset["projection_frame"]),
        "question_format": str(query_id),
        "view_family": str(view_family),
        "solver_trace": dict(solver_trace),
    }
    execution_trace.update(dict(execution_trace_fields or {}))

    trace_payload = {
        "scene_ir": {
            "scene_kind": str(scene_kind),
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": relation_fields,
        },
        "query_spec": {
            "query_id": str(query_id),
            "template_id": str(prompt_default_values["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": query_params,
        },
        "render_spec": render_spec,
        "render_map": render_map,
        "execution_trace": execution_trace,
        "witness_symbolic": {
            "type": "object",
            "id": str(dataset["answer_object_id"]),
            "answer": str(answer_label),
        },
        "projected_annotation": dict(annotation_payload.projected_annotation),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=answer_gt,
        annotation_gt=annotation_payload.annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=str(scene_id),
        query_id=str(query_id),
    )
