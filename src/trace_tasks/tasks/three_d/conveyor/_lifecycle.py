"""Scene-private lifecycle orchestration for straight conveyor public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from trace_tasks.core.scene_config import get_domain_defaults
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.background import make_background_canvas
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.three_d.shared.canvas import render_params_canvas_metadata
from trace_tasks.tasks.three_d.shared.object_scene import _resolve_render_params

from .shared.annotations import (
    bbox_set_annotation_for_objects,
    bbox_set_map_annotation_for_object_groups,
    segment_set_annotation_for_object_pairs,
)
from .shared.prompts import build_prompt_artifacts, dynamic_slots_for_conveyor
from .shared.rendering import RenderedConveyor, render_conveyor
from .shared.sampling import (
    PREDICATE_BELT_TOTAL,
    PREDICATE_BETWEEN_OBJECT_ANCHORS,
    PREDICATE_COLOR,
    PREDICATE_COLOR_TYPE,
    PREDICATE_COLOR_TRANSFER,
    PREDICATE_ORDERED_COLOR_PAIR,
    PREDICATE_ORDERED_OBJECT_PAIR,
    PREDICATE_OBJECT_TYPE,
    PREDICATE_OBJECT_TYPE_ARITHMETIC,
    PREDICATE_OBJECT_TYPE_TRANSFER,
    ResolvedConveyorAxes,
    build_between_marked_items_count_dataset,
    build_belt_total_count_dataset,
    build_lane_count_arithmetic_dataset,
    build_ordered_pair_count_dataset,
    build_scoped_belt_count_dataset,
    build_transfer_count_dataset,
    resolve_conveyor_axes,
)
from .shared.state import SCENE_ID


@dataclass(frozen=True)
class ConveyorTaskPlan:
    """Task-owned objective data for one straight conveyor instance."""

    dataset: Mapping[str, Any]
    answer_gt: TypedValue
    target_object_ids: tuple[str, ...]
    objective_params: Mapping[str, Any]
    target_object_ids_by_annotation_key: Mapping[str, tuple[str, ...]] | None = None
    target_object_id_pairs: tuple[tuple[str, str], ...] | None = None


_DOMAIN_DEFAULTS = get_domain_defaults("three_d")
_VISUAL_DEFAULTS = _DOMAIN_DEFAULTS.get("visual", {}) if isinstance(_DOMAIN_DEFAULTS, Mapping) else {}
_BACKGROUND_DEFAULTS = _VISUAL_DEFAULTS.get("background", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}
_NOISE_DEFAULTS = _VISUAL_DEFAULTS.get("noise", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}


def _attempt_seed(instance_seed: int, *, public_name: str, attempt_index: int) -> int:
    if int(attempt_index) == 0:
        return int(instance_seed)
    return int(spawn_rng(int(instance_seed), f"{public_name}.attempt_seed.{attempt_index}").randrange(1, 2**62))


def _bbox_inside_image(bbox: Sequence[float], *, width: int, height: int) -> bool:
    x0, y0, x1, y1 = (float(value) for value in bbox[:4])
    return x0 >= 0.0 and y0 >= 0.0 and x1 <= float(width) and y1 <= float(height) and x1 > x0 and y1 > y0


def _rendered_bboxes_are_readable(
    rendered: RenderedConveyor,
    *,
    min_side_px: float,
) -> bool:
    width, height = int(rendered.image.width), int(rendered.image.height)
    for bbox in rendered.object_bboxes_px.values():
        if not _bbox_inside_image(bbox, width=width, height=height):
            return False
        box_width = float(bbox[2]) - float(bbox[0])
        box_height = float(bbox[3]) - float(bbox[1])
        if max(box_width, box_height) < float(min_side_px):
            return False
        if box_width * box_height < float(min_side_px) ** 2:
            return False
    for bbox in rendered.target_object_bboxes_px.values():
        side = min(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))
        if side < float(min_side_px):
            return False
    return True


def _build_trace_payload(
    *,
    public_name: str,
    selected_branch: str,
    axes: ResolvedConveyorAxes,
    plan: ConveyorTaskPlan,
    rendered: RenderedConveyor,
    annotation_artifacts: Any,
    prompt_artifacts: Any,
    query_spec: Mapping[str, Any],
    render_params: Any,
    image: Any,
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble public trace sections after rendering and annotation binding."""

    dataset = dict(plan.dataset)
    target_ids = [str(object_id) for object_id in plan.target_object_ids]
    target_bboxes = {
        str(object_id): list(rendered.object_bboxes_px[str(object_id)])
        for object_id in target_ids
    }
    target_centers = {
        str(object_id): list(rendered.object_centers_px[str(object_id)])
        for object_id in target_ids
    }
    target_pair_ids = [
        [str(pair[0]), str(pair[1])]
        for pair in dataset.get("target_pair_object_id_pairs", [])
    ]
    target_pair_segments = [
        [
            list(rendered.object_centers_px[str(pair[0])]),
            list(rendered.object_centers_px[str(pair[1])]),
        ]
        for pair in target_pair_ids
    ]
    marked_anchor_ids = [str(object_id) for object_id in dataset.get("marked_anchor_object_ids", [])]
    marked_anchor_bboxes = {
        str(object_id): list(rendered.object_bboxes_px[str(object_id)])
        for object_id in marked_anchor_ids
    }
    marked_anchor_centers = {
        str(object_id): list(rendered.object_centers_px[str(object_id)])
        for object_id in marked_anchor_ids
    }
    solver_trace = dict(dataset["solver_trace"])
    solver_trace.update(
        {
            "answer_value": int(dataset["answer_value"]),
            "target_object_ids": list(target_ids),
            "target_object_bboxes_px": dict(target_bboxes),
            "target_object_centers_px": dict(target_centers),
        }
    )
    if target_pair_ids:
        solver_trace.update(
            {
                "target_pair_object_id_pairs": [list(pair) for pair in target_pair_ids],
                "target_pair_segments_px": [[list(point) for point in segment] for segment in target_pair_segments],
            }
        )
    if marked_anchor_ids:
        solver_trace.update(
            {
                "marked_anchor_object_ids": list(marked_anchor_ids),
                "marked_anchor_object_bboxes_px": dict(marked_anchor_bboxes),
                "marked_anchor_object_centers_px": dict(marked_anchor_centers),
            }
        )
    query_params = query_spec.get("params", {}) if isinstance(query_spec, Mapping) else {}
    internal_query_id = str(query_params.get("internal_query_id", selected_branch)) if isinstance(query_params, Mapping) else str(selected_branch)
    execution_trace = {
        "query_id": str(selected_branch),
        "internal_query_id": str(internal_query_id),
        "scene_id": SCENE_ID,
        "scene_variant": str(axes.scene_variant),
        "layout_family": str(dataset.get("layout_family", "")),
        "layout_orientation": str(dataset.get("layout_orientation", "")),
        "predicate_kind": str(dataset["predicate_kind"]),
        "answer_value": int(dataset["answer_value"]),
        "target_lane_key": str(dataset["target_lane_key"]),
        "target_lane_label": str(dataset["target_lane_label"]),
        "target_belt_key": str(dataset["target_belt_key"]),
        "target_belt_label": str(dataset["target_belt_label"]),
        "target_shape_type": str(dataset.get("target_shape_type", "")),
        "target_object_name": str(dataset.get("target_object_name", "")),
        "target_object_plural": str(dataset.get("target_object_plural", "")),
        "target_shape_pair": [str(shape) for shape in dataset.get("target_shape_pair", [])],
        "target_object_name_pair": [str(name) for name in dataset.get("target_object_name_pair", [])],
        "target_object_plural_pair": [str(name) for name in dataset.get("target_object_plural_pair", [])],
        "target_color_name": str(dataset.get("target_color_name", "")),
        "second_target_color_name": str(dataset.get("second_target_color_name", "")),
        "target_color_label": str(dataset.get("target_color_label", "")),
        "second_target_color_label": str(dataset.get("second_target_color_label", "")),
        "target_object_ids": list(target_ids),
        "target_object_bboxes_px": dict(target_bboxes),
        "target_object_centers_px": dict(target_centers),
        "target_pair_object_id_pairs": [list(pair) for pair in target_pair_ids],
        "target_pair_segments_px": [[list(point) for point in segment] for segment in target_pair_segments],
        "start_anchor_object_id": str(dataset.get("start_anchor_object_id", "")),
        "end_anchor_object_id": str(dataset.get("end_anchor_object_id", "")),
        "marked_anchor_object_ids": list(marked_anchor_ids),
        "marked_anchor_object_bboxes_px": dict(marked_anchor_bboxes),
        "marked_anchor_object_centers_px": dict(marked_anchor_centers),
        "marked_anchor_records": [dict(record) for record in dataset.get("marked_anchor_records", [])],
        "between_object_ids": [str(object_id) for object_id in dataset.get("between_object_ids", [])],
        "start_anchor_index": int(dataset.get("start_anchor_index", -1)),
        "end_anchor_index": int(dataset.get("end_anchor_index", -1)),
        "target_lane_object_ids": list(dataset["target_lane_object_ids"]),
        "target_belt_object_ids": list(dataset["target_belt_object_ids"]),
        "object_sequences_by_lane": {
            str(key): [str(object_id) for object_id in value]
            for key, value in dict(dataset.get("object_sequences_by_lane", {})).items()
        },
        "object_count": int(dataset["object_count"]),
        "lane_records": [dict(record) for record in dataset["lane_records"]],
        "object_specs": [dict(spec) for spec in dataset["object_specs"]],
        "shape_counts": dict(dataset["shape_counts"]),
        "color_counts": dict(dataset["color_counts"]),
        "lane_counts": dict(dataset["lane_counts"]),
        "belt_counts": dict(dataset["belt_counts"]),
        "camera": dict(dataset["camera"]),
        "projection_frame": dict(dataset["projection_frame"]),
        "question_format": str(selected_branch),
        "solver_trace": dict(solver_trace),
    }
    if "target_object_ids_by_annotation_key" in dataset:
        target_ids_by_key = {
            str(key): [str(object_id) for object_id in object_ids]
            for key, object_ids in dict(dataset.get("target_object_ids_by_annotation_key", {})).items()
        }
        execution_trace.update(
            {
                "scope_keys": [str(key) for key in dataset.get("scope_keys", [])],
                "scope_labels": dict(dataset.get("scope_labels", {})),
                "annotation_key_by_scope": dict(dataset.get("annotation_key_by_scope", {})),
                "operand_counts_by_scope": dict(dataset.get("operand_counts_by_scope", {})),
                "target_object_ids_by_annotation_key": dict(target_ids_by_key),
            }
        )
    if "arithmetic_operation" in dataset:
        execution_trace.update({"arithmetic_operation": str(dataset["arithmetic_operation"])})
    if "transfer_operation" in dataset:
        execution_trace.update(
            {
                "transfer_operation": str(dataset["transfer_operation"]),
                "source_lane_key": str(dataset.get("source_lane_key", "")),
                "source_lane_label": str(dataset.get("source_lane_label", "")),
                "source_belt_key": str(dataset.get("source_belt_key", "")),
                "source_belt_label": str(dataset.get("source_belt_label", "")),
                "destination_lane_key": str(dataset.get("destination_lane_key", "")),
                "destination_lane_label": str(dataset.get("destination_lane_label", "")),
                "destination_belt_key": str(dataset.get("destination_belt_key", "")),
                "destination_belt_label": str(dataset.get("destination_belt_label", "")),
                "moved_count": int(dataset.get("moved_count", 0)),
                "destination_existing_count": int(dataset.get("destination_existing_count", 0)),
                "source_moved_object_ids": [str(object_id) for object_id in dataset.get("source_moved_object_ids", [])],
                "destination_existing_object_ids": [str(object_id) for object_id in dataset.get("destination_existing_object_ids", [])],
            }
        )
    witness_symbolic = {
        "type": "conveyor_lane_object_set",
        "object_ids": list(target_ids),
        "count": int(dataset["answer_value"]),
        "scope": {
            "lane_key": str(dataset["target_lane_key"]),
            "lane_label": str(dataset["target_lane_label"]),
        },
    }
    if "arithmetic_operation" in dataset:
        witness_symbolic = {
            "type": "conveyor_count_arithmetic_object_sets",
            "operation": str(dataset["arithmetic_operation"]),
            "object_ids_by_annotation_key": {
                str(key): [str(object_id) for object_id in object_ids]
                for key, object_ids in dict(dataset.get("target_object_ids_by_annotation_key", {})).items()
            },
            "operand_counts_by_scope": dict(dataset.get("operand_counts_by_scope", {})),
            "answer_value": int(dataset["answer_value"]),
        }
    if "transfer_operation" in dataset:
        witness_symbolic = {
            "type": "conveyor_transfer_count_object_sets",
            "operation": str(dataset["transfer_operation"]),
            "source_moved_object_ids": [str(object_id) for object_id in dataset.get("source_moved_object_ids", [])],
            "destination_existing_object_ids": [str(object_id) for object_id in dataset.get("destination_existing_object_ids", [])],
            "operand_counts_by_scope": dict(dataset.get("operand_counts_by_scope", {})),
            "answer_value": int(dataset["answer_value"]),
            "source": {
                "lane_key": str(dataset.get("source_lane_key", "")),
                "lane_label": str(dataset.get("source_lane_label", "")),
            },
            "destination": {
                "lane_key": str(dataset.get("destination_lane_key", "")),
                "lane_label": str(dataset.get("destination_lane_label", "")),
            },
        }
    if target_pair_ids:
        witness_symbolic = {
            "type": "conveyor_ordered_pair_set",
            "ordered_pair_object_id_pairs": [list(pair) for pair in target_pair_ids],
            "segments_px": [[list(point) for point in segment] for segment in target_pair_segments],
            "count": int(dataset["answer_value"]),
            "scope": {
                "lane_key": str(dataset["target_lane_key"]),
                "lane_label": str(dataset["target_lane_label"]),
            },
        }
    if marked_anchor_ids:
        witness_symbolic = {
            "type": "conveyor_between_marked_item_set",
            "object_ids": list(target_ids),
            "count": int(dataset["answer_value"]),
            "start_anchor_object_id": str(dataset.get("start_anchor_object_id", "")),
            "end_anchor_object_id": str(dataset.get("end_anchor_object_id", "")),
            "marked_anchor_object_ids": list(marked_anchor_ids),
            "scope": {
                "lane_key": str(dataset["target_lane_key"]),
                "lane_label": str(dataset["target_lane_label"]),
            },
        }
    return {
        "scene_ir": {
            "scene_kind": f"three_d_conveyor_{public_name.rsplit('__', 1)[-1]}",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "layout_family": str(dataset.get("layout_family", "")),
                "layout_orientation": str(dataset.get("layout_orientation", "")),
                "target_lane_label": str(dataset["target_lane_label"]),
                "target_object_ids": list(target_ids),
                "answer_value": int(dataset["answer_value"]),
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "scene_canvas_preset": str(render_params.canvas_preset),
            "scene_canvas_width": int(render_params.canvas_width),
            "scene_canvas_height": int(render_params.canvas_height),
            "scene_canvas_policy": str(render_params.canvas_policy),
            **render_params_canvas_metadata(render_params),
            "final_canvas_width": int(image.width),
            "final_canvas_height": int(image.height),
            "final_canvas_pixels": int(image.width) * int(image.height),
            "coord_space": "pixel",
            "scene_variant": str(axes.scene_variant),
            "layout_family": str(dataset.get("layout_family", "")),
            "layout_orientation": str(dataset.get("layout_orientation", "")),
            "background_style": dict(background_meta),
            "post_image_noise": dict(post_noise_meta),
            "camera": dict(dataset["camera"]),
            "projection_frame": dict(dataset["projection_frame"]),
            "semantic_color_palette": dict(dataset["semantic_color_palette"]),
        },
        "render_map": {
            "image_id": "img0",
            "scene_bbox_px": list(rendered.scene_bbox_px),
            "conveyor_bbox_px": list(rendered.conveyor_bbox_px),
            "object_bboxes_px": dict(rendered.object_bboxes_px),
            "object_centers_px": dict(rendered.object_centers_px),
            "target_object_bboxes_px": dict(target_bboxes),
            "target_object_centers_px": dict(target_centers),
            "target_pair_segments_px": [[list(point) for point in segment] for segment in target_pair_segments],
            "marked_anchor_object_bboxes_px": dict(marked_anchor_bboxes),
            "marked_anchor_object_centers_px": dict(marked_anchor_centers),
            "belt_bboxes_px": dict(rendered.belt_bboxes_px),
        },
        "execution_trace": execution_trace,
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }


def _trace_params(
    *,
    axes: ResolvedConveyorAxes,
    dataset: Mapping[str, Any],
    branch_probabilities: Mapping[str, float],
    internal_query_id: str,
) -> Dict[str, Any]:
    """Record resolved sampling axes for replay without routing public task behavior."""

    params = {
        "predicate_kind": str(dataset["predicate_kind"]),
        "internal_query_id": str(internal_query_id),
        "query_id_probabilities": dict(branch_probabilities),
        "scene_variant": str(axes.scene_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "layout_family": str(dataset.get("layout_family", "")),
        "layout_orientation": str(dataset.get("layout_orientation", "")),
        "layout_orientation_probabilities": dict(dataset["layout_orientation_probabilities"]),
        "target_lane_key": str(dataset["target_lane_key"]),
        "target_lane_label": str(dataset["target_lane_label"]),
        "target_lane_key_probabilities": dict(dataset["target_lane_key_probabilities"]),
        "target_belt_key": str(dataset["target_belt_key"]),
        "target_belt_label": str(dataset["target_belt_label"]),
        "target_belt_key_probabilities": dict(dataset["target_belt_key_probabilities"]),
        "target_count": int(dataset["target_count"]),
        "target_count_probabilities": dict(dataset["target_count_probabilities"]),
        "object_count": int(dataset["object_count"]),
        "target_shape_type": str(dataset.get("target_shape_type", "")),
        "target_shape_type_probabilities": dict(dataset["target_shape_type_probabilities"]),
        "target_shape_pair": [str(shape) for shape in dataset.get("target_shape_pair", [])],
        "target_object_name_pair": [str(name) for name in dataset.get("target_object_name_pair", [])],
        "target_object_plural_pair": [str(name) for name in dataset.get("target_object_plural_pair", [])],
        "target_color_name": str(dataset.get("target_color_name", "")),
        "target_color_label": str(dataset.get("target_color_label", "")),
        "second_target_color_name": str(dataset.get("second_target_color_name", "")),
        "second_target_color_label": str(dataset.get("second_target_color_label", "")),
        "target_color_name_probabilities": dict(dataset.get("target_color_name_probabilities", {})),
    }
    if "arithmetic_operation" in dataset:
        params.update(
            {
                "arithmetic_operation": str(dataset["arithmetic_operation"]),
                "scope_keys": [str(key) for key in dataset.get("scope_keys", [])],
                "annotation_key_by_scope": dict(dataset.get("annotation_key_by_scope", {})),
                "operand_counts_by_scope": dict(dataset.get("operand_counts_by_scope", {})),
            }
        )
    if "transfer_operation" in dataset:
        params.update(
            {
                "transfer_operation": str(dataset["transfer_operation"]),
                "scope_keys": [str(key) for key in dataset.get("scope_keys", [])],
                "annotation_key_by_scope": dict(dataset.get("annotation_key_by_scope", {})),
                "operand_counts_by_scope": dict(dataset.get("operand_counts_by_scope", {})),
                "source_lane_key": str(dataset.get("source_lane_key", "")),
                "source_lane_label": str(dataset.get("source_lane_label", "")),
                "destination_lane_key": str(dataset.get("destination_lane_key", "")),
                "destination_lane_label": str(dataset.get("destination_lane_label", "")),
                "moved_count": int(dataset.get("moved_count", 0)),
                "destination_existing_count": int(dataset.get("destination_existing_count", 0)),
            }
        )
    return params


def run_conveyor_transfer_count_lifecycle(
    *,
    public_name: str,
    domain_name: str,
    prompt_query_key_by_branch: Mapping[str, str],
    predicate_kind_by_branch: Mapping[str, str],
    supported_branches: Sequence[str],
    default_branch: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run the counterfactual source-to-destination transfer count lifecycle."""

    from trace_tasks.core.scene_config import get_scene_defaults
    from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

    scene_defaults = get_scene_defaults(str(domain_name), SCENE_ID)
    gen_defaults, render_defaults, _prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        scene_defaults if isinstance(scene_defaults, Mapping) else {},
        task_id=str(public_name),
    )
    selected_branch, branch_probabilities, clean_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(branch) for branch in supported_branches),
        default_query_id=str(default_branch),
        task_id=str(public_name),
        namespace=f"{public_name}.query",
    )
    axes = resolve_conveyor_axes(
        params=clean_params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=str(public_name),
    )
    prompt_query_key = str(prompt_query_key_by_branch[str(selected_branch)])
    predicate_kind = str(predicate_kind_by_branch[str(selected_branch)])
    if predicate_kind not in {PREDICATE_OBJECT_TYPE_TRANSFER, PREDICATE_COLOR_TRANSFER}:
        raise ValueError(f"unsupported straight conveyor transfer predicate: {predicate_kind}")
    min_bbox_side_px = float(clean_params.get("min_rendered_bbox_side_px", gen_defaults.get("min_rendered_bbox_side_px", 24.0)))
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = _attempt_seed(int(instance_seed), public_name=str(public_name), attempt_index=int(attempt_index))
        try:
            render_params = _resolve_render_params(
                clean_params,
                render_defaults=render_defaults,
                instance_seed=int(attempt_seed),
                namespace=f"{public_name}.canvas",
            )
            dataset = build_transfer_count_dataset(
                instance_seed=int(attempt_seed),
                params=clean_params,
                gen_defaults=gen_defaults,
                render_params=render_params,
                axes=axes,
                predicate_kind=str(predicate_kind),
                namespace=str(public_name),
            )
            target_ids_by_key = {
                str(key): tuple(str(object_id) for object_id in object_ids)
                for key, object_ids in dict(dataset["target_object_ids_by_annotation_key"]).items()
            }
            plan = ConveyorTaskPlan(
                dataset=dict(dataset),
                answer_gt=TypedValue(type="integer", value=int(dataset["answer_value"])),
                target_object_ids=tuple(str(object_id) for object_id in dataset["target_object_ids"]),
                objective_params={},
                target_object_ids_by_annotation_key=dict(target_ids_by_key),
            )
            background, background_meta = make_background_canvas(
                canvas_width=int(render_params.canvas_width),
                canvas_height=int(render_params.canvas_height),
                instance_seed=int(attempt_seed),
                params=clean_params,
                default_config=_BACKGROUND_DEFAULTS,
            )
            rendered = render_conveyor(background, dataset=plan.dataset, render_params=render_params)
            if not _rendered_bboxes_are_readable(rendered, min_side_px=float(min_bbox_side_px)):
                raise ValueError("rendered conveyor object boxes failed readability constraints")
            image, post_noise_meta = apply_post_image_noise(
                rendered.image,
                instance_seed=int(attempt_seed),
                params=clean_params,
                default_config=_NOISE_DEFAULTS,
            )
            annotation_artifacts = bbox_set_map_annotation_for_object_groups(rendered, target_ids_by_key)
            _prompt_defaults, prompt_artifacts = build_prompt_artifacts(
                prompt_query_key=str(prompt_query_key),
                dynamic_slot_values=dynamic_slots_for_conveyor(plan.dataset),
                instance_seed=int(attempt_seed),
            )
            query_spec = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_branch),
                params=_trace_params(
                    axes=axes,
                    dataset=plan.dataset,
                    branch_probabilities=branch_probabilities,
                    internal_query_id=str(prompt_query_key),
                ),
            )
            trace_payload = _build_trace_payload(
                public_name=str(public_name),
                selected_branch=str(selected_branch),
                axes=axes,
                plan=plan,
                rendered=rendered,
                annotation_artifacts=annotation_artifacts,
                prompt_artifacts=prompt_artifacts,
                query_spec=query_spec,
                render_params=render_params,
                image=image,
                background_meta=background_meta,
                post_noise_meta=post_noise_meta,
            )
            return TaskOutput(
                prompt=str(prompt_artifacts.prompt),
                prompt_variants=dict(prompt_artifacts.prompt_variants),
                answer_gt=plan.answer_gt,
                annotation_gt=annotation_artifacts.annotation_gt,
                image=image,
                image_id="img0",
                trace_payload=trace_payload,
                task_versions=default_task_versions(),
                scene_id=SCENE_ID,
                query_id=str(selected_branch),
            )
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"{public_name} failed to generate a valid straight conveyor transfer scene after {max_attempts} attempts: {last_error}")


def run_conveyor_ordered_pair_count_lifecycle(
    *,
    public_name: str,
    domain_name: str,
    prompt_query_key_by_branch: Mapping[str, str],
    predicate_kind_by_branch: Mapping[str, str],
    supported_branches: Sequence[str],
    default_branch: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run the ordered adjacent-pair count lifecycle for straight conveyor tasks."""

    from trace_tasks.core.scene_config import get_scene_defaults
    from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

    scene_defaults = get_scene_defaults(str(domain_name), SCENE_ID)
    gen_defaults, render_defaults, _prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        scene_defaults if isinstance(scene_defaults, Mapping) else {},
        task_id=str(public_name),
    )
    selected_branch, branch_probabilities, clean_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(branch) for branch in supported_branches),
        default_query_id=str(default_branch),
        task_id=str(public_name),
        namespace=f"{public_name}.query",
    )
    axes = resolve_conveyor_axes(
        params=clean_params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=str(public_name),
    )
    prompt_query_key = str(prompt_query_key_by_branch[str(selected_branch)])
    predicate_kind = str(predicate_kind_by_branch[str(selected_branch)])
    if predicate_kind not in {PREDICATE_ORDERED_OBJECT_PAIR, PREDICATE_ORDERED_COLOR_PAIR}:
        raise ValueError(f"unsupported straight conveyor ordered-pair predicate: {predicate_kind}")
    min_bbox_side_px = float(clean_params.get("min_rendered_bbox_side_px", gen_defaults.get("min_rendered_bbox_side_px", 24.0)))
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = _attempt_seed(int(instance_seed), public_name=str(public_name), attempt_index=int(attempt_index))
        try:
            render_params = _resolve_render_params(
                clean_params,
                render_defaults=render_defaults,
                instance_seed=int(attempt_seed),
                namespace=f"{public_name}.canvas",
            )
            dataset = build_ordered_pair_count_dataset(
                instance_seed=int(attempt_seed),
                params=clean_params,
                gen_defaults=gen_defaults,
                render_params=render_params,
                axes=axes,
                predicate_kind=str(predicate_kind),
                namespace=str(public_name),
            )
            target_pair_ids = tuple(
                (str(pair[0]), str(pair[1]))
                for pair in dataset["target_pair_object_id_pairs"]
            )
            plan = ConveyorTaskPlan(
                dataset=dict(dataset),
                answer_gt=TypedValue(type="integer", value=int(dataset["answer_value"])),
                target_object_ids=tuple(str(object_id) for object_id in dataset["target_object_ids"]),
                objective_params={},
                target_object_id_pairs=target_pair_ids,
            )
            background, background_meta = make_background_canvas(
                canvas_width=int(render_params.canvas_width),
                canvas_height=int(render_params.canvas_height),
                instance_seed=int(attempt_seed),
                params=clean_params,
                default_config=_BACKGROUND_DEFAULTS,
            )
            rendered = render_conveyor(background, dataset=plan.dataset, render_params=render_params)
            if not _rendered_bboxes_are_readable(rendered, min_side_px=float(min_bbox_side_px)):
                raise ValueError("rendered conveyor object boxes failed readability constraints")
            image, post_noise_meta = apply_post_image_noise(
                rendered.image,
                instance_seed=int(attempt_seed),
                params=clean_params,
                default_config=_NOISE_DEFAULTS,
            )
            annotation_artifacts = segment_set_annotation_for_object_pairs(rendered, target_pair_ids)
            _prompt_defaults, prompt_artifacts = build_prompt_artifacts(
                prompt_query_key=str(prompt_query_key),
                dynamic_slot_values=dynamic_slots_for_conveyor(plan.dataset),
                instance_seed=int(attempt_seed),
            )
            query_spec = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_branch),
                params=_trace_params(
                    axes=axes,
                    dataset=plan.dataset,
                    branch_probabilities=branch_probabilities,
                    internal_query_id=str(prompt_query_key),
                ),
            )
            trace_payload = _build_trace_payload(
                public_name=str(public_name),
                selected_branch=str(selected_branch),
                axes=axes,
                plan=plan,
                rendered=rendered,
                annotation_artifacts=annotation_artifacts,
                prompt_artifacts=prompt_artifacts,
                query_spec=query_spec,
                render_params=render_params,
                image=image,
                background_meta=background_meta,
                post_noise_meta=post_noise_meta,
            )
            return TaskOutput(
                prompt=str(prompt_artifacts.prompt),
                prompt_variants=dict(prompt_artifacts.prompt_variants),
                answer_gt=plan.answer_gt,
                annotation_gt=annotation_artifacts.annotation_gt,
                image=image,
                image_id="img0",
                trace_payload=trace_payload,
                task_versions=default_task_versions(),
                scene_id=SCENE_ID,
                query_id=str(selected_branch),
            )
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"{public_name} failed to generate a valid straight conveyor ordered-pair scene after {max_attempts} attempts: {last_error}")


def run_conveyor_between_marked_items_count_lifecycle(
    *,
    public_name: str,
    domain_name: str,
    prompt_query_key_by_branch: Mapping[str, str],
    predicate_kind_by_branch: Mapping[str, str],
    supported_branches: Sequence[str],
    default_branch: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run the marked-anchor between-items count lifecycle for straight conveyor tasks."""

    from trace_tasks.core.scene_config import get_scene_defaults
    from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

    scene_defaults = get_scene_defaults(str(domain_name), SCENE_ID)
    gen_defaults, render_defaults, _prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        scene_defaults if isinstance(scene_defaults, Mapping) else {},
        task_id=str(public_name),
    )
    selected_branch, branch_probabilities, clean_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(branch) for branch in supported_branches),
        default_query_id=str(default_branch),
        task_id=str(public_name),
        namespace=f"{public_name}.query",
    )
    axes = resolve_conveyor_axes(
        params=clean_params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=str(public_name),
    )
    prompt_query_key = str(prompt_query_key_by_branch[str(selected_branch)])
    predicate_kind = str(predicate_kind_by_branch[str(selected_branch)])
    if predicate_kind != PREDICATE_BETWEEN_OBJECT_ANCHORS:
        raise ValueError(f"unsupported straight conveyor between-anchor predicate: {predicate_kind}")
    min_bbox_side_px = float(clean_params.get("min_rendered_bbox_side_px", gen_defaults.get("min_rendered_bbox_side_px", 24.0)))
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = _attempt_seed(int(instance_seed), public_name=str(public_name), attempt_index=int(attempt_index))
        try:
            render_params = _resolve_render_params(
                clean_params,
                render_defaults=render_defaults,
                instance_seed=int(attempt_seed),
                namespace=f"{public_name}.canvas",
            )
            dataset = build_between_marked_items_count_dataset(
                instance_seed=int(attempt_seed),
                params=clean_params,
                gen_defaults=gen_defaults,
                render_params=render_params,
                axes=axes,
                predicate_kind=str(predicate_kind),
                namespace=str(public_name),
            )
            plan = ConveyorTaskPlan(
                dataset=dict(dataset),
                answer_gt=TypedValue(type="integer", value=int(dataset["answer_value"])),
                target_object_ids=tuple(str(object_id) for object_id in dataset["target_object_ids"]),
                objective_params={},
            )
            background, background_meta = make_background_canvas(
                canvas_width=int(render_params.canvas_width),
                canvas_height=int(render_params.canvas_height),
                instance_seed=int(attempt_seed),
                params=clean_params,
                default_config=_BACKGROUND_DEFAULTS,
            )
            rendered = render_conveyor(background, dataset=plan.dataset, render_params=render_params)
            if not _rendered_bboxes_are_readable(rendered, min_side_px=float(min_bbox_side_px)):
                raise ValueError("rendered conveyor object boxes failed readability constraints")
            image, post_noise_meta = apply_post_image_noise(
                rendered.image,
                instance_seed=int(attempt_seed),
                params=clean_params,
                default_config=_NOISE_DEFAULTS,
            )
            annotation_artifacts = bbox_set_annotation_for_objects(rendered, plan.target_object_ids)
            _prompt_defaults, prompt_artifacts = build_prompt_artifacts(
                prompt_query_key=str(prompt_query_key),
                dynamic_slot_values=dynamic_slots_for_conveyor(plan.dataset),
                instance_seed=int(attempt_seed),
            )
            query_spec = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_branch),
                params=_trace_params(
                    axes=axes,
                    dataset=plan.dataset,
                    branch_probabilities=branch_probabilities,
                    internal_query_id=str(prompt_query_key),
                ),
            )
            trace_payload = _build_trace_payload(
                public_name=str(public_name),
                selected_branch=str(selected_branch),
                axes=axes,
                plan=plan,
                rendered=rendered,
                annotation_artifacts=annotation_artifacts,
                prompt_artifacts=prompt_artifacts,
                query_spec=query_spec,
                render_params=render_params,
                image=image,
                background_meta=background_meta,
                post_noise_meta=post_noise_meta,
            )
            return TaskOutput(
                prompt=str(prompt_artifacts.prompt),
                prompt_variants=dict(prompt_artifacts.prompt_variants),
                answer_gt=plan.answer_gt,
                annotation_gt=annotation_artifacts.annotation_gt,
                image=image,
                image_id="img0",
                trace_payload=trace_payload,
                task_versions=default_task_versions(),
                scene_id=SCENE_ID,
                query_id=str(selected_branch),
            )
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"{public_name} failed to generate a valid straight conveyor between-anchor scene after {max_attempts} attempts: {last_error}")


def run_conveyor_lifecycle(
    *,
    public_name: str,
    domain_name: str,
    prompt_query_key_by_branch: Mapping[str, str],
    predicate_kind_by_branch: Mapping[str, str],
    supported_branches: Sequence[str],
    default_branch: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run query selection, rendering, prompt, annotation, and output assembly."""

    from trace_tasks.core.scene_config import get_scene_defaults
    from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

    scene_defaults = get_scene_defaults(str(domain_name), SCENE_ID)
    gen_defaults, render_defaults, _prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        scene_defaults if isinstance(scene_defaults, Mapping) else {},
        task_id=str(public_name),
    )
    selected_branch, branch_probabilities, clean_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(branch) for branch in supported_branches),
        default_query_id=str(default_branch),
        task_id=str(public_name),
        namespace=f"{public_name}.query",
    )
    axes = resolve_conveyor_axes(
        params=clean_params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=str(public_name),
    )
    prompt_query_key = str(prompt_query_key_by_branch[str(selected_branch)])
    predicate_kind = str(predicate_kind_by_branch[str(selected_branch)])
    if predicate_kind not in {PREDICATE_BELT_TOTAL, PREDICATE_OBJECT_TYPE, PREDICATE_COLOR, PREDICATE_COLOR_TYPE}:
        raise ValueError(f"unsupported straight conveyor predicate: {predicate_kind}")
    min_bbox_side_px = float(clean_params.get("min_rendered_bbox_side_px", gen_defaults.get("min_rendered_bbox_side_px", 24.0)))
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = _attempt_seed(int(instance_seed), public_name=str(public_name), attempt_index=int(attempt_index))
        try:
            render_params = _resolve_render_params(
                clean_params,
                render_defaults=render_defaults,
                instance_seed=int(attempt_seed),
                namespace=f"{public_name}.canvas",
            )
            if predicate_kind == PREDICATE_BELT_TOTAL:
                dataset = build_belt_total_count_dataset(
                    instance_seed=int(attempt_seed),
                    params=clean_params,
                    gen_defaults=gen_defaults,
                    render_params=render_params,
                    axes=axes,
                    namespace=str(public_name),
                )
            else:
                dataset = build_scoped_belt_count_dataset(
                    instance_seed=int(attempt_seed),
                    params=clean_params,
                    gen_defaults=gen_defaults,
                    render_params=render_params,
                    axes=axes,
                    predicate_kind=str(predicate_kind),
                    namespace=str(public_name),
                )
            plan = ConveyorTaskPlan(
                dataset=dict(dataset),
                answer_gt=TypedValue(type="integer", value=int(dataset["answer_value"])),
                target_object_ids=tuple(str(object_id) for object_id in dataset["target_object_ids"]),
                objective_params={},
            )
            background, background_meta = make_background_canvas(
                canvas_width=int(render_params.canvas_width),
                canvas_height=int(render_params.canvas_height),
                instance_seed=int(attempt_seed),
                params=clean_params,
                default_config=_BACKGROUND_DEFAULTS,
            )
            rendered = render_conveyor(background, dataset=plan.dataset, render_params=render_params)
            if not _rendered_bboxes_are_readable(rendered, min_side_px=float(min_bbox_side_px)):
                raise ValueError("rendered conveyor object boxes failed readability constraints")
            image, post_noise_meta = apply_post_image_noise(
                rendered.image,
                instance_seed=int(attempt_seed),
                params=clean_params,
                default_config=_NOISE_DEFAULTS,
            )
            annotation_artifacts = bbox_set_annotation_for_objects(rendered, plan.target_object_ids)
            _prompt_defaults, prompt_artifacts = build_prompt_artifacts(
                prompt_query_key=str(prompt_query_key),
                dynamic_slot_values=dynamic_slots_for_conveyor(plan.dataset),
                instance_seed=int(attempt_seed),
            )
            query_spec = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_branch),
                params=_trace_params(
                    axes=axes,
                    dataset=plan.dataset,
                    branch_probabilities=branch_probabilities,
                    internal_query_id=str(prompt_query_key),
                ),
            )
            trace_payload = _build_trace_payload(
                public_name=str(public_name),
                selected_branch=str(selected_branch),
                axes=axes,
                plan=plan,
                rendered=rendered,
                annotation_artifacts=annotation_artifacts,
                prompt_artifacts=prompt_artifacts,
                query_spec=query_spec,
                render_params=render_params,
                image=image,
                background_meta=background_meta,
                post_noise_meta=post_noise_meta,
            )
            return TaskOutput(
                prompt=str(prompt_artifacts.prompt),
                prompt_variants=dict(prompt_artifacts.prompt_variants),
                answer_gt=plan.answer_gt,
                annotation_gt=annotation_artifacts.annotation_gt,
                image=image,
                image_id="img0",
                trace_payload=trace_payload,
                task_versions=default_task_versions(),
                scene_id=SCENE_ID,
                query_id=str(selected_branch),
            )
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"{public_name} failed to generate a valid straight conveyor scene after {max_attempts} attempts: {last_error}")


def run_conveyor_count_arithmetic_lifecycle(
    *,
    public_name: str,
    domain_name: str,
    prompt_query_key_by_branch: Mapping[str, str],
    predicate_kind_by_branch: Mapping[str, str],
    operation_by_branch: Mapping[str, str],
    supported_branches: Sequence[str],
    default_branch: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run the two-lane count-arithmetic lifecycle for straight conveyor tasks."""

    from trace_tasks.core.scene_config import get_scene_defaults
    from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

    scene_defaults = get_scene_defaults(str(domain_name), SCENE_ID)
    gen_defaults, render_defaults, _prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        scene_defaults if isinstance(scene_defaults, Mapping) else {},
        task_id=str(public_name),
    )
    selected_branch, branch_probabilities, clean_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(branch) for branch in supported_branches),
        default_query_id=str(default_branch),
        task_id=str(public_name),
        namespace=f"{public_name}.query",
    )
    axes = resolve_conveyor_axes(
        params=clean_params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=str(public_name),
    )
    prompt_query_key = str(prompt_query_key_by_branch[str(selected_branch)])
    predicate_kind = str(predicate_kind_by_branch[str(selected_branch)])
    if predicate_kind != PREDICATE_OBJECT_TYPE_ARITHMETIC:
        raise ValueError(f"unsupported straight conveyor arithmetic predicate: {predicate_kind}")
    operation = str(operation_by_branch[str(selected_branch)])
    min_bbox_side_px = float(clean_params.get("min_rendered_bbox_side_px", gen_defaults.get("min_rendered_bbox_side_px", 24.0)))
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = _attempt_seed(int(instance_seed), public_name=str(public_name), attempt_index=int(attempt_index))
        try:
            render_params = _resolve_render_params(
                clean_params,
                render_defaults=render_defaults,
                instance_seed=int(attempt_seed),
                namespace=f"{public_name}.canvas",
            )
            dataset = build_lane_count_arithmetic_dataset(
                instance_seed=int(attempt_seed),
                params=clean_params,
                gen_defaults=gen_defaults,
                render_params=render_params,
                axes=axes,
                predicate_kind=str(predicate_kind),
                operation=str(operation),
                namespace=str(public_name),
            )
            target_ids_by_key = {
                str(key): tuple(str(object_id) for object_id in object_ids)
                for key, object_ids in dict(dataset["target_object_ids_by_annotation_key"]).items()
            }
            plan = ConveyorTaskPlan(
                dataset=dict(dataset),
                answer_gt=TypedValue(type="integer", value=int(dataset["answer_value"])),
                target_object_ids=tuple(str(object_id) for object_id in dataset["target_object_ids"]),
                objective_params={},
                target_object_ids_by_annotation_key=dict(target_ids_by_key),
            )
            background, background_meta = make_background_canvas(
                canvas_width=int(render_params.canvas_width),
                canvas_height=int(render_params.canvas_height),
                instance_seed=int(attempt_seed),
                params=clean_params,
                default_config=_BACKGROUND_DEFAULTS,
            )
            rendered = render_conveyor(background, dataset=plan.dataset, render_params=render_params)
            if not _rendered_bboxes_are_readable(rendered, min_side_px=float(min_bbox_side_px)):
                raise ValueError("rendered conveyor object boxes failed readability constraints")
            image, post_noise_meta = apply_post_image_noise(
                rendered.image,
                instance_seed=int(attempt_seed),
                params=clean_params,
                default_config=_NOISE_DEFAULTS,
            )
            annotation_artifacts = bbox_set_map_annotation_for_object_groups(rendered, target_ids_by_key)
            _prompt_defaults, prompt_artifacts = build_prompt_artifacts(
                prompt_query_key=str(prompt_query_key),
                dynamic_slot_values=dynamic_slots_for_conveyor(plan.dataset),
                instance_seed=int(attempt_seed),
            )
            query_spec = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_branch),
                params=_trace_params(
                    axes=axes,
                    dataset=plan.dataset,
                    branch_probabilities=branch_probabilities,
                    internal_query_id=str(prompt_query_key),
                ),
            )
            trace_payload = _build_trace_payload(
                public_name=str(public_name),
                selected_branch=str(selected_branch),
                axes=axes,
                plan=plan,
                rendered=rendered,
                annotation_artifacts=annotation_artifacts,
                prompt_artifacts=prompt_artifacts,
                query_spec=query_spec,
                render_params=render_params,
                image=image,
                background_meta=background_meta,
                post_noise_meta=post_noise_meta,
            )
            return TaskOutput(
                prompt=str(prompt_artifacts.prompt),
                prompt_variants=dict(prompt_artifacts.prompt_variants),
                answer_gt=plan.answer_gt,
                annotation_gt=annotation_artifacts.annotation_gt,
                image=image,
                image_id="img0",
                trace_payload=trace_payload,
                task_versions=default_task_versions(),
                scene_id=SCENE_ID,
                query_id=str(selected_branch),
            )
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"{public_name} failed to generate a valid straight conveyor arithmetic scene after {max_attempts} attempts: {last_error}")


__all__ = [
    "ConveyorTaskPlan",
    "run_conveyor_between_marked_items_count_lifecycle",
    "run_conveyor_count_arithmetic_lifecycle",
    "run_conveyor_lifecycle",
    "run_conveyor_ordered_pair_count_lifecycle",
    "run_conveyor_transfer_count_lifecycle",
]
