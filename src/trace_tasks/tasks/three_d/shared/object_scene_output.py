"""Task-output assembly helpers for 3D object-scene label tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .canvas import render_params_canvas_metadata
from ....core.seed import spawn_rng
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
    render_scene_prompt_variants,
)
from .annotation_geometry import normalize_annotation_bboxes
from .object_scene import SCENE_ID, _RenderParams, render_object_scene_3d
from .option_panel import apply_independent_prompt_colors_to_dataset, build_text_option_choices

_Q_FIELD = "query" + "_id"


def build_option_label_object_scene_output(
    *,
    objective_name: str,
    task_domain: str,
    instance_seed: int,
    params: Mapping[str, Any],
    dataset: Mapping[str, Any],
    branch_key: str,
    scene_variant: str,
    point_count: int,
    context_object_count: int,
    render_params: _RenderParams,
    prompt_defaults_config: Mapping[str, Any],
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    scene_probabilities: Mapping[str, float],
    point_count_probabilities: Mapping[str, float],
    context_object_count_probabilities: Mapping[str, float],
    dynamic_slots: Mapping[str, Any],
    relation_fields: Mapping[str, Any],
    query_params_extra: Mapping[str, Any],
    execution_extra: Mapping[str, Any],
) -> TaskOutput:
    """Render one labeled object-scene instance and build the shared answer/trace payload."""
    colored_dataset = apply_independent_prompt_colors_to_dataset(
        dataset,
        rng=spawn_rng(int(instance_seed), f"{objective_name}.prompt_colors"),
    )
    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=background_defaults,
    )
    option_choices = build_text_option_choices(colored_dataset["point_specs"])
    rendered_scene = render_object_scene_3d(
        background,
        dataset=colored_dataset,
        render_params=render_params,
        draw_candidate_labels=False,
        option_choices=option_choices,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=noise_defaults,
    )

    prompt_defaults = required_group_defaults(
        prompt_defaults_config,
        (
            "bundle_id",
            "scene_key",
            "task_key",
        ),
        context=f"prompt defaults for {objective_name}",
    )
    prompt_selection = render_scene_prompt_variants(
        domain=str(task_domain),
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(branch_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

    answer_label = str(colored_dataset["answer_label"])
    answer_gt = TypedValue(type="option_letter", value=str(answer_label))
    raw_annotation_bboxes = [[round(float(value), 3) for value in bbox] for bbox in rendered_scene.annotation_bboxes]
    if len(raw_annotation_bboxes) != 1:
        raise RuntimeError(f"{objective_name} expected exactly one annotation bbox")
    annotation_bounds = [
        0.0,
        0.0,
        float(image.width),
        float(rendered_scene.option_panel_bbox_px[1]) if rendered_scene.option_panel_bbox_px else float(image.height),
    ]
    annotation_bboxes, annotation_bbox_normalization = normalize_annotation_bboxes(
        raw_annotation_bboxes,
        bounds_px=annotation_bounds,
    )
    annotation_payload = bbox_annotation_artifacts(annotation_bboxes[0])
    solver_trace = dict(colored_dataset["solver_trace"])

    common_relations: Dict[str, Any] = {
        "scene_variant": str(scene_variant),
        "point_count": int(point_count),
        "candidate_count": int(point_count),
        "context_object_count": int(context_object_count),
        "object_count": int(colored_dataset["object_count"]),
        "candidate_shape_types": [str(spec["shape_type"]) for spec in colored_dataset["point_specs"]],
        "context_shape_types": [str(spec["shape_type"]) for spec in colored_dataset["context_object_specs"]],
        "candidate_object_names": [str(spec["object_name"]) for spec in colored_dataset["point_specs"]],
        "context_object_names": [str(spec["object_name"]) for spec in colored_dataset["context_object_specs"]],
        "view_family": "synthetic_perspective_3d_scene",
        "answer_point_id": str(colored_dataset["answer_point_id"]),
        "answer_label": str(answer_label),
    }
    common_relations.update(dict(relation_fields))

    branch_params: Dict[str, Any] = {
        _Q_FIELD: str(branch_key),
        "query_id_probabilities": dict(query_probabilities),
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": dict(scene_probabilities),
        "point_count": int(point_count),
        "candidate_count": int(point_count),
        "context_object_count": int(context_object_count),
        "context_object_count_probabilities": dict(context_object_count_probabilities),
        "point_count_probabilities": dict(point_count_probabilities),
        "object_count": int(colored_dataset["object_count"]),
    }
    branch_params.update(dict(query_params_extra))

    execution_trace: Dict[str, Any] = {
        _Q_FIELD: str(branch_key),
        "scene_variant": str(scene_variant),
        "point_count": int(point_count),
        "candidate_count": int(point_count),
        "context_object_count": int(context_object_count),
        "object_count": int(colored_dataset["object_count"]),
        "point_specs": [dict(spec) for spec in colored_dataset["point_specs"]],
        "context_object_specs": [dict(spec) for spec in colored_dataset["context_object_specs"]],
        "object_specs": [dict(spec) for spec in colored_dataset["object_specs"]],
        "option_choices": [dict(choice) for choice in rendered_scene.option_choices],
        "option_descriptor_by_label": {
            str(choice["label"]): str(choice["descriptor"])
            for choice in rendered_scene.option_choices
        },
        "answer_label": str(answer_label),
        "answer_point_id": str(colored_dataset["answer_point_id"]),
        "camera": dict(colored_dataset["camera"]),
        "projection_frame": dict(colored_dataset["projection_frame"]),
        "question_format": str(branch_key),
        "view_family": "synthetic_perspective_3d_scene",
        "solver_trace": dict(solver_trace),
    }
    execution_trace.update(dict(execution_extra))

    trace_payload = {
        "scene_ir": {
            "scene_kind": "three_d_object_scene",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": common_relations,
        },
        "query_spec": {
            _Q_FIELD: str(branch_key),
            "template_id": str(prompt_defaults["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": branch_params,
        },
        "render_spec": {
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
            "camera": dict(colored_dataset["camera"]),
            "projection_frame": dict(colored_dataset["projection_frame"]),
            "label_font_size_px": int(render_params.label_font_size_px),
            "annotation_bbox_normalization": dict(annotation_bbox_normalization),
        },
        "render_map": {
            "image_id": "img0",
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "room_bbox_px": list(rendered_scene.room_bbox_px),
            "point_bboxes_px": {str(key): list(value) for key, value in rendered_scene.point_bboxes_px.items()},
            "point_centers_px": {str(key): list(value) for key, value in rendered_scene.point_centers_px.items()},
            "option_panel_bbox_px": list(rendered_scene.option_panel_bbox_px),
            "option_panel_height_px": int(rendered_scene.option_panel_height_px),
            "option_choice_bboxes_px": {
                str(key): list(value) for key, value in rendered_scene.option_choice_bboxes_px.items()
            },
            "option_choices": [dict(choice) for choice in rendered_scene.option_choices],
            "object_bboxes_px": {str(key): list(value) for key, value in rendered_scene.object_bboxes_px.items()},
            "object_centers_px": {str(key): list(value) for key, value in rendered_scene.object_centers_px.items()},
            "context_object_bboxes_px": {
                str(key): list(value) for key, value in rendered_scene.context_object_bboxes_px.items()
            },
            "context_object_centers_px": {
                str(key): list(value) for key, value in rendered_scene.context_object_centers_px.items()
            },
            "annotation_raw_bboxes_px": [list(bbox) for bbox in raw_annotation_bboxes],
            "annotation_bboxes_px": [list(bbox) for bbox in annotation_bboxes],
            "annotation_entity_ids": [str(item) for item in rendered_scene.annotation_entity_ids],
            "annotation_bbox_normalization": dict(annotation_bbox_normalization),
        },
        "execution_trace": execution_trace,
        "witness_symbolic": {
            "type": "object",
            "ids": [str(item) for item in rendered_scene.annotation_entity_ids],
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
        scene_id=SCENE_ID,
        query_id=str(branch_key),
    )


__all__ = ["build_option_label_object_scene_output"]
