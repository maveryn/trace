"""Domain-level output helpers for object-scene marked-point ordering tasks."""

from __future__ import annotations

import math
from itertools import permutations
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from PIL import Image

from ....core.types import TypedValue
from ....core.visual.background import make_background_canvas
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.config_defaults import required_group_defaults
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from .canvas import (
    bbox_dict_transform,
    bbox_transform,
    entities_transform,
    final_canvas_metadata,
    point_dict_transform,
    render_params_canvas_metadata,
    resize_image_to_fit_pixel_cap,
)
from .object_scene import SCENE_ID, SUPPORTED_SCENE_VARIANTS, _RenderParams, _resolve_render_params, render_object_scene_3d
from .option_panel import append_text_option_panel
from .task_support import resolve_axis_variant as _shared_resolve_axis_variant
from .task_support import resolve_count as _shared_resolve_count


ORDER_POINT_LABELS: Tuple[str, ...] = ("P", "Q", "R")
ORDER_OPTION_LABELS: Tuple[str, ...] = tuple("ABCDEF")
_Q_FIELD = "query" + "_id"


def label_order_descriptor(labels: Sequence[str]) -> str:
    return " < ".join(str(label) for label in labels)


def order_option_choices(*, answer_order: Sequence[str]) -> Tuple[List[Dict[str, Any]], str]:
    """Return the six permutation options and the answer option label."""

    answer_descriptor = label_order_descriptor(answer_order)
    choices: List[Dict[str, Any]] = []
    answer_label = ""
    for option_label, labels in zip(ORDER_OPTION_LABELS, permutations(ORDER_POINT_LABELS)):
        descriptor = label_order_descriptor(labels)
        if descriptor == answer_descriptor:
            answer_label = str(option_label)
        choices.append(
            {
                "label": str(option_label),
                "option_id": f"order_option_{option_label}",
                "object_id": f"order_option_{option_label}",
                "descriptor": str(descriptor),
                "object_name": "point order",
                "color_name": None,
            }
        )
    if not answer_label:
        raise ValueError(f"answer order is not a permutation of {ORDER_POINT_LABELS}: {answer_order}")
    return choices, str(answer_label)


def demote_context_spec(spec: Mapping[str, Any], *, index: int, prefix: str) -> Dict[str, Any]:
    """Turn a sampled small candidate object into unlettered context."""

    updated = dict(spec)
    shape_type = str(updated["shape_type"])
    updated["object_id"] = f"{prefix}_context_{int(index):02d}_{shape_type}"
    updated["object_role"] = "small_context"
    updated["is_answer_candidate"] = False
    for key in ("point_id", "point_label", "object_label"):
        updated.pop(key, None)
    return updated


def finalize_object_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    camera,
    frame,
    project_screen_fn: Callable[..., Sequence[float]],
) -> List[Dict[str, Any]]:
    finalized_specs: List[Dict[str, Any]] = []
    for spec in specs:
        screen = project_screen_fn(spec["world_xyz"], camera, frame)
        finalized = dict(spec)
        finalized.update(
            {
                "screen_xy": [round(float(screen[0]), 3), round(float(screen[1]), 3)],
                "camera_xyz": [
                    round(float(screen[5]), 4),
                    round(float(screen[6]), 4),
                    round(float(screen[4]), 4),
                ],
                "camera_distance": round(float(screen[7]), 4),
            }
        )
        finalized_specs.append(finalized)
    return finalized_specs


def relabel_order_points(
    records: Sequence[Mapping[str, Any]],
    *,
    rng,
) -> List[Dict[str, Any]]:
    labels = list(ORDER_POINT_LABELS)
    rng.shuffle(labels)
    relabeled: List[Dict[str, Any]] = []
    for record, label in zip(records, labels):
        updated = dict(record)
        updated["point_label"] = str(label)
        updated["point_id"] = f"marked_point_{label}"
        relabeled.append(updated)
    return sorted(relabeled, key=lambda item: str(item["point_label"]))


def point_map_by_label(
    *,
    marker_render_map: Mapping[str, Any],
    labels: Sequence[str] = ORDER_POINT_LABELS,
) -> Dict[str, List[float]]:
    centers = marker_render_map["marked_point_centers_px"]
    return {
        str(label): [round(float(value), 3) for value in centers[str(label)]]
        for label in labels
    }


def _bbox_union(*bboxes: Sequence[float]) -> List[float]:
    clean = [list(bbox) for bbox in bboxes if bbox]
    return [
        round(float(min(bbox[0] for bbox in clean)), 3),
        round(float(min(bbox[1] for bbox in clean)), 3),
        round(float(max(bbox[2] for bbox in clean)), 3),
        round(float(max(bbox[3] for bbox in clean)), 3),
    ]


def build_point_order_object_scene_output(
    *,
    objective_name: str,
    task_domain: str,
    instance_seed: int,
    params: Mapping[str, Any],
    dataset: Mapping[str, Any],
    branch_key: str,
    scene_variant: str,
    point_count: int,
    render_params: _RenderParams,
    prompt_defaults_config: Mapping[str, Any],
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    scene_probabilities: Mapping[str, float],
    dynamic_slots: Mapping[str, Any],
    scene_kind: str,
    order_axis: str,
    draw_marked_points_fn: Callable[..., tuple[Image.Image, Dict[str, Any], list[Dict[str, Any]]]],
    count_params: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> TaskOutput:
    """Render a three-point ordering task with six visual permutation options."""

    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=background_defaults,
    )
    rendered_scene = render_object_scene_3d(
        background,
        dataset=dataset,
        render_params=render_params,
        draw_candidate_labels=False,
        compute_single_annotation=False,
    )
    marked_image, marker_render_map, marker_entities = draw_marked_points_fn(
        rendered_scene.image,
        marked_points=dataset["marked_points"],
        render_params=render_params,
    )
    option_choices = [dict(choice) for choice in dataset["option_choices"]]
    option_image, option_metadata, option_entities = append_text_option_panel(
        marked_image,
        option_choices=option_choices,
        font_size_px=int(render_params.label_font_size_px),
        text_rgb=render_params.text_rgb,
        stroke_rgb=render_params.text_stroke_rgb,
    )
    scaled_image, image_scale = resize_image_to_fit_pixel_cap(option_image)
    if image_scale.changed:
        scale_x = float(image_scale.scale_x)
        scale_y = float(image_scale.scale_y)
        marker_render_map = dict(marker_render_map)
        marker_render_map["marked_point_centers_px"] = point_dict_transform(
            marker_render_map["marked_point_centers_px"],
            scale_x=scale_x,
            scale_y=scale_y,
        )
        for key in ("marked_point_glyph_bboxes_px", "marked_point_circle_bboxes_px", "marked_point_label_bboxes_px", "marked_point_bboxes_px"):
            marker_render_map[key] = bbox_dict_transform(
                marker_render_map.get(key, {}),
                scale_x=scale_x,
                scale_y=scale_y,
            )
        option_metadata = dict(option_metadata)
        option_metadata["option_panel_bbox_px"] = bbox_transform(
            option_metadata["option_panel_bbox_px"],
            scale_x=scale_x,
            scale_y=scale_y,
        )
        option_metadata["option_choice_bboxes_px"] = bbox_dict_transform(
            option_metadata["option_choice_bboxes_px"],
            scale_x=scale_x,
            scale_y=scale_y,
        )
        option_metadata["option_panel_height_px"] = int(round(float(option_metadata["option_panel_height_px"]) * scale_y))
        rendered_scene_entities = entities_transform(rendered_scene.entities, scale_x=scale_x, scale_y=scale_y)
        marker_entities = entities_transform(marker_entities, scale_x=scale_x, scale_y=scale_y)
        option_entities = entities_transform(option_entities, scale_x=scale_x, scale_y=scale_y)
        scene_bbox = bbox_transform(rendered_scene.scene_bbox_px, scale_x=scale_x, scale_y=scale_y)
        room_bbox = bbox_transform(rendered_scene.room_bbox_px, scale_x=scale_x, scale_y=scale_y)
        object_bboxes = bbox_dict_transform(rendered_scene.object_bboxes_px, scale_x=scale_x, scale_y=scale_y)
        object_centers = point_dict_transform(rendered_scene.object_centers_px, scale_x=scale_x, scale_y=scale_y)
        context_object_bboxes = bbox_dict_transform(rendered_scene.context_object_bboxes_px, scale_x=scale_x, scale_y=scale_y)
        context_object_centers = point_dict_transform(rendered_scene.context_object_centers_px, scale_x=scale_x, scale_y=scale_y)
    else:
        rendered_scene_entities = list(rendered_scene.entities)
        scene_bbox = list(rendered_scene.scene_bbox_px)
        room_bbox = list(rendered_scene.room_bbox_px)
        object_bboxes = dict(rendered_scene.object_bboxes_px)
        object_centers = dict(rendered_scene.object_centers_px)
        context_object_bboxes = dict(rendered_scene.context_object_bboxes_px)
        context_object_centers = dict(rendered_scene.context_object_centers_px)

    image, post_noise_meta = apply_post_image_noise(
        scaled_image,
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

    annotation_points = point_map_by_label(marker_render_map=marker_render_map)
    annotation_gt = TypedValue(type="point_map", value=dict(annotation_points))
    answer_label = str(dataset["answer_label"])
    answer_gt = TypedValue(type="option_letter", value=str(answer_label))
    all_marker_bboxes = [bbox for bbox in marker_render_map["marked_point_bboxes_px"].values()]
    scene_bbox = _bbox_union(scene_bbox, *all_marker_bboxes)
    scene_entities = [*rendered_scene_entities, *marker_entities, *option_entities]
    count_params = dict(count_params or {})
    execution_extra = dict(execution_extra or {})

    branch_params: Dict[str, Any] = {
        _Q_FIELD: str(branch_key),
        _Q_FIELD + "_probabilities": dict(query_probabilities),
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": dict(scene_probabilities),
        "point_count": int(point_count),
        "point_labels": list(ORDER_POINT_LABELS),
        "answer_support": list(ORDER_OPTION_LABELS),
        "order_axis": str(order_axis),
    }
    branch_params.update(count_params)

    render_map: Dict[str, Any] = {
        "image_id": "img0",
        "scene_bbox_px": list(scene_bbox),
        "room_bbox_px": list(room_bbox),
        "object_bboxes_px": {str(key): list(value) for key, value in object_bboxes.items()},
        "object_centers_px": {str(key): list(value) for key, value in object_centers.items()},
        "context_object_bboxes_px": {str(key): list(value) for key, value in context_object_bboxes.items()},
        "context_object_centers_px": {str(key): list(value) for key, value in context_object_centers.items()},
        **dict(marker_render_map),
        "option_panel_bbox_px": list(option_metadata["option_panel_bbox_px"]),
        "option_choice_bboxes_px": {
            str(key): list(value) for key, value in option_metadata["option_choice_bboxes_px"].items()
        },
        "option_choices": [dict(choice) for choice in option_metadata["option_choices"]],
        "option_panel_height_px": int(option_metadata["option_panel_height_px"]),
        "annotation_point_map_px": dict(annotation_points),
    }

    execution_trace: Dict[str, Any] = {
        _Q_FIELD: str(branch_key),
        "scene_variant": str(scene_variant),
        "point_count": int(point_count),
        "point_labels": list(ORDER_POINT_LABELS),
        "marked_points": [dict(point) for point in dataset["marked_points"]],
        "context_object_specs": [dict(spec) for spec in dataset["context_object_specs"]],
        "object_specs": [dict(spec) for spec in dataset["object_specs"]],
        "option_choices": [dict(choice) for choice in option_choices],
        "answer_label": str(answer_label),
        "answer_order": list(dataset["answer_order"]),
        "answer_descriptor": str(dataset["answer_descriptor"]),
        "camera": dict(dataset["camera"]),
        "projection_frame": dict(dataset["projection_frame"]),
        "question_format": str(branch_key),
        "view_family": "synthetic_perspective_3d_marked_point_order",
        "solver_trace": dict(dataset["solver_trace"]),
    }
    execution_trace.update(execution_extra)

    trace_payload = {
        "scene_ir": {
            "scene_kind": str(scene_kind),
            "entities": [dict(entity) for entity in scene_entities],
            "relations": {
                "scene_variant": str(scene_variant),
                "point_count": int(point_count),
                "point_labels": list(ORDER_POINT_LABELS),
                "answer_label": str(answer_label),
                "answer_order": list(dataset["answer_order"]),
                "order_axis": str(order_axis),
                "view_family": "synthetic_perspective_3d_marked_point_order",
            },
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
            "canvas_height": int(render_params.canvas_height),
            **render_params_canvas_metadata(render_params),
            **final_canvas_metadata(image),
            "coord_space": "pixel",
            "scene_variant": str(scene_variant),
            "background_style": dict(background_meta),
            "post_image_noise": dict(post_noise_meta),
            "camera": dict(dataset["camera"]),
            "projection_frame": dict(dataset["projection_frame"]),
            "label_font_size_px": int(render_params.label_font_size_px),
            "marker_radius_px": int(max(12.0, float(render_params.marker_radius_px) * 0.66)),
            "option_panel_height_px": int(option_metadata["option_panel_height_px"]),
        },
        "render_map": render_map,
        "execution_trace": execution_trace,
        "witness_symbolic": {
            "type": "marked_point_order",
            "point_ids_by_label": {
                str(point["point_label"]): str(point["point_id"]) for point in dataset["marked_points"]
            },
            "answer_label": str(answer_label),
            "answer_order": list(dataset["answer_order"]),
        },
        "projected_annotation": {
            "type": "point_map",
            "point_map": dict(annotation_points),
            "pixel_point_map": dict(annotation_points),
        },
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }

    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(branch_key),
    )


def generate_point_order_once(
    *,
    owner_domain: str,
    objective_name: str,
    branch_options: Sequence[str],
    dataset_builder: Callable[..., Dict[str, Any]],
    order_axis: str,
    scene_kind: str,
    instance_seed: int,
    params: Dict[str, Any],
    camera_yaw_band: Tuple[float, float] | None,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    draw_marked_points_fn: Callable[..., tuple[Image.Image, Dict[str, Any], list[Dict[str, Any]]]],
) -> TaskOutput:
    """Resolve shared order-task axes and render one point-order instance."""

    branch_key, branch_probabilities = _shared_resolve_axis_variant(
        params,
        task_id=str(objective_name),
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=branch_options,
        explicit_key=_Q_FIELD,
        weights_key=_Q_FIELD + "_weights",
        balance_flag_key="balanced_query_id_sampling",
        axis_namespace=_Q_FIELD,
    )
    scene_variant, scene_probabilities = _shared_resolve_axis_variant(
        params,
        task_id=str(objective_name),
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )
    point_count, point_count_probabilities = _shared_resolve_count(
        params,
        task_id=str(objective_name),
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        key="point_count",
        default_min=3,
        default_max=3,
        lower=3,
        upper=3,
    )
    context_object_count, context_object_count_probabilities = _shared_resolve_count(
        params,
        task_id=str(objective_name),
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        key="context_object_count",
        default_min=4,
        default_max=4,
        lower=2,
        upper=5,
    )
    render_params = _resolve_render_params(
        params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{objective_name}.canvas",
    )
    dataset = dataset_builder(
        branch_key=str(branch_key),
        scene_variant=str(scene_variant),
        point_count=int(point_count),
        context_object_count=int(context_object_count),
        render_params=render_params,
        instance_seed=int(instance_seed),
        camera_yaw_band=camera_yaw_band,
    )
    return build_point_order_object_scene_output(
        objective_name=str(objective_name),
        task_domain=str(owner_domain),
        instance_seed=int(instance_seed),
        params=params,
        dataset=dataset,
        branch_key=str(branch_key),
        scene_variant=str(scene_variant),
        point_count=int(point_count),
        render_params=render_params,
        prompt_defaults_config=prompt_defaults,
        background_defaults=background_defaults,
        noise_defaults=noise_defaults,
        query_probabilities=branch_probabilities,
        scene_probabilities=scene_probabilities,
        dynamic_slots={},
        scene_kind=str(scene_kind),
        order_axis=str(order_axis),
        draw_marked_points_fn=draw_marked_points_fn,
        count_params={
            "context_object_count": int(context_object_count),
            "context_object_count_probabilities": dict(context_object_count_probabilities),
            "point_count_probabilities": dict(point_count_probabilities),
        },
        execution_extra={
            "context_object_count": int(context_object_count),
            "small_context_object_count": int(dataset["small_context_object_count"]),
            "large_context_object_count": int(dataset["large_context_object_count"]),
            "object_count": int(len(dataset["object_specs"])),
        },
    )


def min_pairwise(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("inf")
    return min(abs(float(a) - float(b)) for index, a in enumerate(values) for b in values[index + 1 :])


def screen_separation_ok(points: Sequence[Mapping[str, Any]], *, min_px: float) -> bool:
    centers = [(float(point["screen_xy"][0]), float(point["screen_xy"][1])) for point in points]
    return all(
        math.hypot(float(a[0] - b[0]), float(a[1] - b[1])) >= float(min_px)
        for index, a in enumerate(centers)
        for b in centers[index + 1 :]
    )


__all__ = [
    "ORDER_OPTION_LABELS",
    "ORDER_POINT_LABELS",
    "build_point_order_object_scene_output",
    "demote_context_spec",
    "finalize_object_specs",
    "generate_point_order_once",
    "label_order_descriptor",
    "min_pairwise",
    "order_option_choices",
    "point_map_by_label",
    "relabel_order_points",
    "screen_separation_ok",
]
