"""Trace payload scaffolds for named-field icon tasks.

Public task files own target selection, answer binding, annotation binding, and
final ``TaskOutput`` construction. These helpers only serialize scene state and
task-supplied semantic fields into Trace payload sections.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence, Tuple

from ....shared.color_format import format_named_color_with_hex
from ....shared.config_defaults import group_default
from ....shared.named_colors import available_named_colors
from ...shared.icon_task_rendering import icon_render_style_trace
from ...shared.procedural_named_icon_field_scene import SCENE_ID, serialize_named_icon_instance

from .metrics import (
    BOOLEAN_PREDICATE_AND,
    BOOLEAN_PREDICATE_ATTRIBUTE_WITHOUT_SHAPE,
    BOOLEAN_PREDICATE_NEITHER,
    BOOLEAN_PREDICATE_OR,
    BOOLEAN_PREDICATE_SHAPE_WITHOUT_ATTRIBUTE,
    BOOLEAN_PREDICATE_XOR,
    COUNTERFACTUAL_SHAPE_REMOVAL,
    COUNTERFACTUAL_SHAPE_REPLACEMENT,
    boolean_entity_partition,
    counterfactual_counted_shape_ids_after_edit,
    counterfactual_role_by_instance_id,
    trace_key,
)
from .state import (
    CloserReferenceRenderedIcon,
    DistanceRankRenderedIcon,
    RegionSpec,
    RenderedRegionIcon,
)


def render_slot_params(params: Mapping[str, Any], render_defaults: Mapping[str, Any], fallback_defaults: Any) -> tuple[int, int, int]:
    """Resolve shared named-icon slot rendering knobs."""

    return (
        int(
            params.get(
                "named_icon_slot_padding_px",
                group_default(render_defaults, "named_icon_slot_padding_px", fallback_defaults.named_icon_slot_padding_px),
            )
        ),
        int(
            params.get(
                "named_icon_slot_jitter_px",
                group_default(render_defaults, "named_icon_slot_jitter_px", fallback_defaults.named_icon_slot_jitter_px),
            )
        ),
        int(
            params.get(
                "named_icon_stack_gap_px",
                group_default(render_defaults, "named_icon_stack_gap_px", fallback_defaults.named_icon_stack_gap_px),
            )
        ),
    )


def object_bboxes(instances: Sequence[Any]) -> dict[str, list[int]]:
    """Return visible object bboxes keyed by rendered instance id."""

    return {str(instance.instance_id): [int(value) for value in instance.bbox_xyxy] for instance in instances}


def shape_counted_instance_ids(sample: Any, instances: Sequence[Any]) -> tuple[str, ...]:
    """Return rendered instance ids matching a direct shape-count target."""

    return tuple(str(instance.instance_id) for instance in instances if str(instance.shape_id) == str(sample.target_shape_id))


def semantic_color_palette() -> list[dict[str, Any]]:
    """Serialize the named-color support visible to semantic color tasks."""

    return [
        {
            "name": str(name),
            "rgb": [int(channel) for channel in rgb],
            "label": format_named_color_with_hex(str(name), rgb),
        }
        for name, rgb in available_named_colors()
    ]


def serialize_closer_reference_icon(icon: CloserReferenceRenderedIcon) -> Dict[str, Any]:
    """Serialize one closer-reference icon for trace payloads."""

    return {
        "entity_kind": "procedural_named_icon",
        "instance_id": str(icon.instance_id),
        "role": str(icon.role),
        "label": str(icon.label),
        "shape_id": str(icon.shape_id),
        "shape_name": str(icon.shape_name),
        "color_name": str(icon.color_name),
        "tint_rgb": [int(value) for value in icon.tint_rgb],
        "fill_style": str(icon.fill_style),
        "bbox_xyxy": [int(value) for value in icon.bbox_xyxy],
        "center_xy": [float(icon.center_xy[0]), float(icon.center_xy[1])],
        "nominal_size_px": int(icon.nominal_size_px),
        "rotation_degrees": int(icon.rotation_degrees),
        "distance_to_reference_a_px": None if icon.distance_to_reference_a_px is None else float(icon.distance_to_reference_a_px),
        "distance_to_reference_b_px": None if icon.distance_to_reference_b_px is None else float(icon.distance_to_reference_b_px),
        "closer_reference_label": str(icon.closer_reference_label),
        "counted": bool(icon.counted),
        "label_bbox_xyxy": None if icon.label_bbox_xyxy is None else [int(value) for value in icon.label_bbox_xyxy],
        "noise_edits": [dict(edit) for edit in icon.noise_edits],
        "noise_seed": None if icon.noise_seed is None else int(icon.noise_seed),
    }


def serialize_distance_rank_icon(icon: DistanceRankRenderedIcon) -> Dict[str, Any]:
    """Serialize one distance-rank icon for trace payloads."""

    return {
        "entity_kind": "procedural_named_icon",
        "instance_id": str(icon.instance_id),
        "role": str(icon.role),
        "label": str(icon.label),
        "shape_id": str(icon.shape_id),
        "shape_name": str(icon.shape_name),
        "color_name": str(icon.color_name),
        "tint_rgb": [int(value) for value in icon.tint_rgb],
        "fill_style": str(icon.fill_style),
        "bbox_xyxy": [int(value) for value in icon.bbox_xyxy],
        "center_xy": [float(icon.center_xy[0]), float(icon.center_xy[1])],
        "nominal_size_px": int(icon.nominal_size_px),
        "rotation_degrees": int(icon.rotation_degrees),
        "distance_to_reference_px": None if icon.distance_to_reference_px is None else float(icon.distance_to_reference_px),
        "distance_rank": None if icon.distance_rank is None else int(icon.distance_rank),
        "noise_edits": [dict(edit) for edit in icon.noise_edits],
        "noise_seed": None if icon.noise_seed is None else int(icon.noise_seed),
    }


def region_to_trace(region: RegionSpec) -> Dict[str, Any]:
    """Serialize a visible region without hardcoding public identity keys."""

    payload: Dict[str, Any] = {
        trace_key("query", "id"): str(region.query_key),
        "region_kind": str(region.region_kind),
        "counts_inside": bool(region.counts_inside),
        "shape_kind": str(region.shape_kind),
        "band_kind": str(region.band_kind),
        "quadrant_id": str(region.quadrant_id),
        "shelf_index": int(region.shelf_index),
        "shelf_count": int(region.shelf_count),
    }
    if region.bbox_xyxy is not None:
        payload["bbox_xyxy"] = [int(value) for value in region.bbox_xyxy]
    if region.ellipse_center_xy is not None:
        payload["ellipse_center_xy"] = [float(value) for value in region.ellipse_center_xy]
    if region.ellipse_radii_xy is not None:
        payload["ellipse_radii_xy"] = [float(value) for value in region.ellipse_radii_xy]
    if region.band_normal_xy is not None:
        payload["band_normal_xy"] = [float(value) for value in region.band_normal_xy]
    if region.band_center_distance is not None:
        payload["band_center_distance"] = float(region.band_center_distance)
    if region.band_half_width_px is not None:
        payload["band_half_width_px"] = float(region.band_half_width_px)
    if region.band_polygon_xy:
        payload["band_polygon_xy"] = [[float(x), float(y)] for x, y in region.band_polygon_xy]
    return payload


def serialize_region_icon(instance: RenderedRegionIcon) -> Dict[str, Any]:
    """Serialize one scoped-region icon for trace payloads."""

    return {
        "entity_kind": "procedural_named_icon",
        "instance_id": str(instance.instance_id),
        "shape_id": str(instance.shape_id),
        "shape_name": str(instance.shape_name),
        "bbox_xyxy": [int(value) for value in instance.bbox_xyxy],
        "center_xy": [float(value) for value in instance.center_xy],
        "nominal_size_px": int(instance.nominal_size_px),
        "rotation_degrees": int(instance.rotation_degrees),
        "tint_rgb": [int(value) for value in instance.tint_rgb],
        "fill_style": str(instance.fill_style),
        "inside_region": bool(instance.inside_region),
        "counted": bool(instance.counted),
        "noise_edits": [dict(edit) for edit in instance.noise_edits],
        "noise_seed": None if instance.noise_seed is None else int(instance.noise_seed),
    }


def build_shape_count_query_metadata(
    *,
    sample: Any,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    shape_support: Sequence[str],
) -> dict[str, Any]:
    """Serialize sampled non-public axes for direct shape counting."""

    return {
        "target_shape_id": str(sample.target_shape_id),
        "target_shape_name": str(sample.target_shape_name),
        "target_count": int(sample.target_count),
        "object_count": int(sample.object_count),
        "arrangement_mode": str(sample.arrangement_mode),
        "arrangement_mode_probabilities": dict(sample.arrangement_mode_probabilities),
        "arrangement_details": dict(sample.arrangement_details),
        "shape_id_support": list(shape_support),
        "shape_probabilities": dict(sample.shape_probabilities),
        "target_count_probabilities": dict(sample.target_count_probabilities),
        "object_count_probabilities": dict(sample.object_count_probabilities),
        "named_icon_fill_style_support": list(sample.fill_style_support),
        "fill_style_probabilities": dict(sample.fill_style_probabilities),
    }


def build_boolean_query_metadata(
    *,
    sample: Any,
    query_expression: str,
    public_query_id: str,
    public_query_probabilities: Mapping[str, float],
    shape_support: Sequence[str],
    color_support: Sequence[Any],
    fill_style_support: Sequence[str],
) -> dict[str, Any]:
    """Serialize sampled non-public axes for a Boolean count task."""

    return {
        trace_key("query", "id"): str(public_query_id),
        "prompt_query_key": str(sample.prompt_query_key),
        "internal_query_id": str(sample.prompt_query_key),
        "target_shape_id": str(sample.target_shape_id),
        "target_shape_name": str(sample.target_shape_name),
        "target_attribute_axis": str(sample.target_attribute_axis),
        "target_attribute_value": str(sample.target_attribute_value),
        "target_attribute_label": str(sample.target_attribute_label),
        "target_color_name": str(sample.target_color.name) if sample.target_color is not None else "",
        "target_color_rgb": [int(channel) for channel in sample.target_color.rgb] if sample.target_color is not None else [],
        "target_color_label": str(sample.target_color.label) if sample.target_color is not None else "",
        "target_answer": int(sample.target_answer),
        "object_count": int(sample.object_count),
        "object_count_max_answer_offset": int(sample.object_count_max_answer_offset),
        "boolean_expression": str(query_expression),
        "partition_counts": {str(key): int(value) for key, value in sample.partition_counts.items()},
        "arrangement_mode": str(sample.arrangement_mode),
        "arrangement_mode_probabilities": dict(sample.arrangement_mode_probabilities),
        "shape_id_support": list(shape_support),
        "named_color_support": [str(entry.name) for entry in color_support],
        "named_icon_fill_style_support": list(fill_style_support),
        "query_probabilities": {str(key): float(value) for key, value in public_query_probabilities.items()},
        "query_id_probabilities": {str(key): float(value) for key, value in public_query_probabilities.items()},
        "shape_probabilities": dict(sample.shape_probabilities),
        "color_probabilities": dict(sample.color_probabilities),
        "fill_style_probabilities": dict(sample.fill_style_probabilities),
        "attribute_axis_probabilities": dict(sample.attribute_axis_probabilities),
        "target_count_probabilities": dict(sample.target_count_probabilities),
        "object_count_probabilities": dict(sample.object_count_probabilities),
    }


def build_counterfactual_query_metadata(
    *,
    sample: Any,
    public_query_id: str,
    public_query_probabilities: Mapping[str, float],
    shape_support: Sequence[str],
) -> dict[str, Any]:
    """Serialize sampled non-public axes for a counterfactual count task."""

    return {
        trace_key("query", "id"): str(public_query_id),
        "prompt_query_key": str(sample.prompt_query_key),
        "internal_query_id": str(sample.prompt_query_key),
        "target_answer": int(sample.target_answer),
        "object_count": int(sample.object_count),
        "target_shape_id": str(sample.target_shape_id),
        "target_shape_name": str(sample.target_shape_name),
        "source_shape_id": str(sample.source_shape_id),
        "source_shape_name": str(sample.source_shape_name),
        "remove_shape_id": str(sample.remove_shape_id),
        "remove_shape_name": str(sample.remove_shape_name),
        "source_count": int(sample.source_count),
        "existing_target_count": int(sample.existing_target_count),
        "removal_count": int(sample.removal_count),
        "distractor_count": int(sample.distractor_count),
        "shape_id_support": list(shape_support),
        "query_probabilities": {str(key): float(value) for key, value in public_query_probabilities.items()},
        "query_id_probabilities": {str(key): float(value) for key, value in public_query_probabilities.items()},
        "shape_probabilities": dict(sample.shape_probabilities),
        "target_count_probabilities": dict(sample.target_count_probabilities),
        "removal_count_probabilities": dict(sample.removal_count_probabilities),
        "distractor_count_probabilities": dict(sample.distractor_count_probabilities),
        "arrangement_mode": str(sample.arrangement_mode),
        "arrangement_mode_probabilities": dict(sample.arrangement_mode_probabilities),
        "named_icon_fill_style_support": list(sample.fill_style_support),
        "fill_style_probabilities": dict(sample.fill_style_probabilities),
    }


def build_shape_count_trace_payload(
    *,
    sample: Any,
    scene: Any,
    render_params: Mapping[str, Any],
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...],
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    annotation_artifacts: Mapping[str, Any],
    counted_instance_ids: Tuple[str, ...],
    query_metadata: Mapping[str, Any],
    public_query_id: str,
    slot_padding_px: int,
    slot_jitter_px: int,
    stack_gap_px: int,
) -> dict[str, Any]:
    """Build trace sections for a direct named-shape count."""

    serialized_instances = [serialize_named_icon_instance(instance) for instance in scene.instances]
    shape_counts = dict(Counter(str(instance.shape_id) for instance in scene.instances))
    return {
        "scene_ir": {
            "scene_kind": "icons_named_shape_field",
            "scene_id": SCENE_ID,
            "entities": list(serialized_instances),
            "relations": {
                "counting_rule": "shape_id_equals_target",
                "target_shape_id": str(sample.target_shape_id),
                "target_shape_name": str(sample.target_shape_name),
                "shape_counts": {str(key): int(value) for key, value in shape_counts.items()},
                "arrangement_mode": str(sample.arrangement_mode),
                "arrangement_details": dict(sample.arrangement_details),
            },
            "frames": {"pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"}, "panels": dict(scene.panel_geometry)},
        },
        "query_spec": {
            trace_key("query", "id"): str(public_query_id),
            "template_id": str(prompt_defaults["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": dict(query_metadata),
        },
        "render_spec": {
            "canvas_size": list(scene.panel_geometry["canvas_size"]),
            "coord_space": "pixel",
            "scene_id": SCENE_ID,
            "panel_geometry": dict(scene.panel_geometry),
            "style": {
                **icon_render_style_trace(render_params=render_params, sampled_palette_rgb=sampled_palette_rgb),
                "layout_mode": str(scene.layout_mode),
                "named_icon_fill_style_support": list(sample.fill_style_support),
                "named_icon_slot_padding_px": int(slot_padding_px),
                "named_icon_slot_jitter_px": int(slot_jitter_px),
                "named_icon_stack_gap_px": int(stack_gap_px),
            },
        },
        "render_map": {"image_id": "img0", "object_bboxes_px": object_bboxes(scene.instances), "counted_instance_ids": list(counted_instance_ids)},
        "execution_trace": {
            "scene_variant": "single_panel_named_shape_field",
            "arrangement_mode": str(sample.arrangement_mode),
            trace_key("query", "id"): str(public_query_id),
            "question_format": "count_named_shape_icons",
            "target_shape_id": str(sample.target_shape_id),
            "target_shape_name": str(sample.target_shape_name),
            "target_count": int(sample.target_count),
            "object_count": int(sample.object_count),
            "shape_counts": {str(key): int(value) for key, value in shape_counts.items()},
            "arrangement_details": dict(sample.arrangement_details),
            "scene_shape_ids": [str(instance.shape_id) for instance in scene.instances],
            "counted_instance_ids": list(counted_instance_ids),
        },
        "witness_symbolic": {
            "target_shape_id": str(sample.target_shape_id),
            "target_shape_name": str(sample.target_shape_name),
            "answer": int(sample.target_count),
            "counted_instance_ids": list(counted_instance_ids),
        },
        "projected_annotation": {**dict(annotation_artifacts["projected_annotation"])},
    }


def build_boolean_trace_payload(
    *,
    sample: Any,
    scene: Any,
    render_params: Mapping[str, Any],
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...],
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    annotation_artifacts: Mapping[str, Any],
    counted_instance_ids: Tuple[str, ...],
    query_expression: str,
    query_metadata: Mapping[str, Any],
    public_query_id: str,
    slot_padding_px: int,
    slot_jitter_px: int,
    stack_gap_px: int,
    fill_style_support: Tuple[str, ...],
) -> dict[str, Any]:
    """Build trace sections for a Boolean named-field count."""

    serialized_instances = [serialize_named_icon_instance(instance) for instance in scene.instances]
    shape_counts = Counter(str(instance.shape_id) for instance in scene.instances)
    color_counts = Counter(str(instance.color_name) for instance in scene.instances)
    fill_style_counts = Counter(str(instance.fill_style) for instance in scene.instances)
    attribute_counts = color_counts
    shape_attribute_counts = Counter(f"{instance.shape_id}|{instance.color_name}" for instance in scene.instances)
    entity_partition = boolean_entity_partition(sample, scene.instances)
    return {
        "scene_ir": {
            "scene_kind": "icons_named_shape_color_field",
            "scene_id": SCENE_ID,
            "entities": list(serialized_instances),
            "relations": {
                "counting_rule": str(query_expression),
                "target_shape_id": str(sample.target_shape_id),
                "target_shape_name": str(sample.target_shape_name),
                "target_attribute_axis": str(sample.target_attribute_axis),
                "target_attribute_value": str(sample.target_attribute_value),
                "target_attribute_label": str(sample.target_attribute_label),
                "target_color_name": str(sample.target_color.name) if sample.target_color is not None else "",
                "target_color_rgb": [int(channel) for channel in sample.target_color.rgb] if sample.target_color is not None else [],
                "target_color_label": str(sample.target_color.label) if sample.target_color is not None else "",
                "shape_counts": {str(key): int(value) for key, value in shape_counts.items()},
                "color_counts": {str(key): int(value) for key, value in color_counts.items()},
                "fill_style_counts": {str(key): int(value) for key, value in fill_style_counts.items()},
                "attribute_counts": {str(key): int(value) for key, value in attribute_counts.items()},
                "shape_attribute_counts": {str(key): int(value) for key, value in shape_attribute_counts.items()},
                "partition_counts": {str(key): int(value) for key, value in sample.partition_counts.items()},
                "arrangement_mode": str(sample.arrangement_mode),
            },
            "frames": {"pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"}, "panels": dict(scene.panel_geometry)},
        },
        "query_spec": {
            trace_key("query", "id"): str(public_query_id),
            "prompt_query_key": str(sample.prompt_query_key),
            "internal_query_id": str(sample.prompt_query_key),
            "template_id": str(prompt_defaults["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": dict(query_metadata),
        },
        "render_spec": {
            "canvas_size": list(scene.panel_geometry["canvas_size"]),
            "coord_space": "pixel",
            "scene_id": SCENE_ID,
            "panel_geometry": dict(scene.panel_geometry),
            "style": {
                **icon_render_style_trace(render_params=render_params, sampled_palette_rgb=sampled_palette_rgb),
                "layout_mode": str(scene.layout_mode),
                "named_icon_slot_padding_px": int(slot_padding_px),
                "named_icon_slot_jitter_px": int(slot_jitter_px),
                "named_icon_stack_gap_px": int(stack_gap_px),
                "semantic_color_palette": semantic_color_palette(),
                "named_icon_fill_style_support": list(fill_style_support),
            },
        },
        "render_map": {
            "image_id": "img0",
            "object_bboxes_px": object_bboxes(scene.instances),
            "counted_instance_ids": list(counted_instance_ids),
            "entity_partition": dict(entity_partition),
        },
        "execution_trace": {
            "scene_variant": "single_panel_named_shape_color_field",
            "arrangement_mode": str(sample.arrangement_mode),
            trace_key("query", "id"): str(public_query_id),
            "prompt_query_key": str(sample.prompt_query_key),
            "internal_query_id": str(sample.prompt_query_key),
            "question_format": "count_named_shape_color_boolean_icons",
            "target_shape_id": str(sample.target_shape_id),
            "target_shape_name": str(sample.target_shape_name),
            "target_attribute_axis": str(sample.target_attribute_axis),
            "target_attribute_value": str(sample.target_attribute_value),
            "target_attribute_label": str(sample.target_attribute_label),
            "target_color_name": str(sample.target_color.name) if sample.target_color is not None else "",
            "target_color_rgb": [int(channel) for channel in sample.target_color.rgb] if sample.target_color is not None else [],
            "target_color_label": str(sample.target_color.label) if sample.target_color is not None else "",
            "target_answer": int(sample.target_answer),
            "object_count": int(sample.object_count),
            "boolean_expression": str(query_expression),
            "partition_counts": {str(key): int(value) for key, value in sample.partition_counts.items()},
            "shape_counts": {str(key): int(value) for key, value in shape_counts.items()},
            "color_counts": {str(key): int(value) for key, value in color_counts.items()},
            "fill_style_counts": {str(key): int(value) for key, value in fill_style_counts.items()},
            "attribute_counts": {str(key): int(value) for key, value in attribute_counts.items()},
            "shape_attribute_counts": {str(key): int(value) for key, value in shape_attribute_counts.items()},
            "scene_shape_ids": [str(instance.shape_id) for instance in scene.instances],
            "scene_color_names": [str(instance.color_name) for instance in scene.instances],
            "scene_fill_styles": [str(instance.fill_style) for instance in scene.instances],
            "counted_instance_ids": list(counted_instance_ids),
        },
        "witness_symbolic": {
            "target_shape_id": str(sample.target_shape_id),
            "target_shape_name": str(sample.target_shape_name),
            "target_attribute_axis": str(sample.target_attribute_axis),
            "target_attribute_value": str(sample.target_attribute_value),
            "target_attribute_label": str(sample.target_attribute_label),
            "target_color_name": str(sample.target_color.name) if sample.target_color is not None else "",
            "target_color_label": str(sample.target_color.label) if sample.target_color is not None else "",
            "answer": int(sample.target_answer),
            "counted_instance_ids": list(counted_instance_ids),
        },
        "projected_annotation": {**dict(annotation_artifacts["projected_annotation"])},
    }


def build_counterfactual_trace_payload(
    *,
    sample: Any,
    scene: Any,
    render_params: Mapping[str, Any],
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...],
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    annotation_artifacts: Mapping[str, Any],
    counted_instance_ids: Tuple[str, ...],
    query_metadata: Mapping[str, Any],
    public_query_id: str,
    slot_padding_px: int,
    slot_jitter_px: int,
    stack_gap_px: int,
) -> dict[str, Any]:
    """Build trace sections for a counterfactual named-field count."""

    serialized_instances = [serialize_named_icon_instance(instance) for instance in scene.instances]
    shape_counts = Counter(str(instance.shape_id) for instance in scene.instances)
    role_by_instance_id = counterfactual_role_by_instance_id(sample)
    counted_shape_ids_after_edit = counterfactual_counted_shape_ids_after_edit(sample)
    return {
        "scene_ir": {
            "scene_kind": "icons_named_shape_counterfactual_field",
            "scene_id": SCENE_ID,
            "entities": list(serialized_instances),
            "relations": {
                "counting_rule": "apply_hypothetical_icon_removal_or_replacement_then_count",
                trace_key("query", "id"): str(public_query_id),
                "prompt_query_key": str(sample.prompt_query_key),
                "target_shape_id": str(sample.target_shape_id),
                "target_shape_name": str(sample.target_shape_name),
                "source_shape_id": str(sample.source_shape_id),
                "source_shape_name": str(sample.source_shape_name),
                "remove_shape_id": str(sample.remove_shape_id),
                "remove_shape_name": str(sample.remove_shape_name),
                "source_count": int(sample.source_count),
                "existing_target_count": int(sample.existing_target_count),
                "removal_count": int(sample.removal_count),
                "distractor_count": int(sample.distractor_count),
                "shape_counts": {str(key): int(value) for key, value in shape_counts.items()},
                "role_by_instance_id": dict(role_by_instance_id),
                "arrangement_mode": str(sample.arrangement_mode),
            },
            "frames": {"pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"}, "panels": dict(scene.panel_geometry)},
        },
        "query_spec": {
            trace_key("query", "id"): str(public_query_id),
            "prompt_query_key": str(sample.prompt_query_key),
            "internal_query_id": str(sample.prompt_query_key),
            "template_id": str(prompt_defaults["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": dict(query_metadata),
        },
        "render_spec": {
            "canvas_size": list(scene.panel_geometry["canvas_size"]),
            "coord_space": "pixel",
            "scene_id": SCENE_ID,
            "panel_geometry": dict(scene.panel_geometry),
            "style": {
                **icon_render_style_trace(render_params=render_params, sampled_palette_rgb=sampled_palette_rgb),
                "layout_mode": str(scene.layout_mode),
                "named_icon_fill_style_support": list(sample.fill_style_support),
                "named_icon_slot_padding_px": int(slot_padding_px),
                "named_icon_slot_jitter_px": int(slot_jitter_px),
                "named_icon_stack_gap_px": int(stack_gap_px),
            },
        },
        "render_map": {
            "image_id": "img0",
            "object_bboxes_px": object_bboxes(scene.instances),
            "counted_instance_ids": list(counted_instance_ids),
            "role_by_instance_id": dict(role_by_instance_id),
        },
        "execution_trace": {
            "scene_variant": "single_panel_named_shape_counterfactual_field",
            "arrangement_mode": str(sample.arrangement_mode),
            trace_key("query", "id"): str(public_query_id),
            "prompt_query_key": str(sample.prompt_query_key),
            "internal_query_id": str(sample.prompt_query_key),
            "question_format": "count_named_shape_icons_after_hypothetical_edit",
            "target_answer": int(sample.target_answer),
            "object_count": int(sample.object_count),
            "target_shape_id": str(sample.target_shape_id),
            "target_shape_name": str(sample.target_shape_name),
            "source_shape_id": str(sample.source_shape_id),
            "source_shape_name": str(sample.source_shape_name),
            "remove_shape_id": str(sample.remove_shape_id),
            "remove_shape_name": str(sample.remove_shape_name),
            "source_count": int(sample.source_count),
            "existing_target_count": int(sample.existing_target_count),
            "removal_count": int(sample.removal_count),
            "distractor_count": int(sample.distractor_count),
            "shape_counts": {str(key): int(value) for key, value in shape_counts.items()},
            "final_counted_shape_ids_after_edit": list(counted_shape_ids_after_edit),
            "final_target_shape_count": int(sample.target_answer),
            "counted_instance_ids": list(counted_instance_ids),
            "role_by_instance_id": dict(role_by_instance_id),
        },
        "witness_symbolic": {
            "answer": int(sample.target_answer),
            "counted_instance_ids": list(counted_instance_ids),
            trace_key("query", "id"): str(public_query_id),
            "prompt_query_key": str(sample.prompt_query_key),
            "internal_query_id": str(sample.prompt_query_key),
            "target_shape_id": str(sample.target_shape_id),
            "target_shape_name": str(sample.target_shape_name),
            "source_shape_id": str(sample.source_shape_id),
            "source_shape_name": str(sample.source_shape_name),
            "remove_shape_id": str(sample.remove_shape_id),
            "remove_shape_name": str(sample.remove_shape_name),
        },
        "projected_annotation": {**dict(annotation_artifacts["projected_annotation"])},
    }


def build_pair_arithmetic_trace_payload(
    *,
    sample: Any,
    scene: Any,
    render_params: Mapping[str, Any],
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...],
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    annotation_artifacts: Mapping[str, Any],
    counted_instance_ids: Tuple[str, ...],
    left_instance_ids: Tuple[str, ...],
    right_instance_ids: Tuple[str, ...],
    role_by_instance_id: Mapping[str, str],
    annotation_instance_ids: Tuple[str, ...],
    query_ids: Sequence[str],
    shape_support: Sequence[str],
    color_support: Sequence[Any],
    fill_style_support: Sequence[str],
    slot_padding_px: int,
    slot_jitter_px: int,
    stack_gap_px: int,
) -> dict[str, Any]:
    """Build trace sections for arithmetic over two named-icon groups."""

    serialized_instances = []
    for instance in scene.instances:
        entity = serialize_named_icon_instance(instance)
        entity["operand_role"] = str(role_by_instance_id.get(str(instance.instance_id), ""))
        serialized_instances.append(entity)

    shape_counts = Counter(str(instance.shape_id) for instance in scene.instances)
    color_counts = Counter(str(instance.color_name) for instance in scene.instances)
    shape_color_counts = Counter(f"{instance.shape_id}|{instance.color_name}" for instance in scene.instances)

    left_operand = {
        "shape_id": str(sample.left_operand.shape_id),
        "shape_name": str(sample.left_operand.shape_name),
        "color_name": str(sample.left_operand.color_name),
        "color_label": str(sample.left_operand.color_label),
        "label": str(sample.left_operand.label),
    }
    right_operand = {
        "shape_id": str(sample.right_operand.shape_id),
        "shape_name": str(sample.right_operand.shape_name),
        "color_name": str(sample.right_operand.color_name),
        "color_label": str(sample.right_operand.color_label),
        "label": str(sample.right_operand.label),
    }
    count_maps = {
        "shape_counts": {str(key): int(value) for key, value in shape_counts.items()},
        "color_counts": {str(key): int(value) for key, value in color_counts.items()},
        "shape_color_counts": {str(key): int(value) for key, value in shape_color_counts.items()},
    }
    return {
        "scene_ir": {
            "scene_kind": "icons_named_shape_pair_arithmetic_field",
            "scene_id": SCENE_ID,
            "entities": list(serialized_instances),
            "relations": {
                "counting_rule": "two_operand_total_or_absolute_difference",
                "operation": str(sample.operation),
                "uses_color_binding": bool(sample.uses_color_binding),
                "left_operand": dict(left_operand),
                "right_operand": dict(right_operand),
                "left_count": int(sample.left_count),
                "right_count": int(sample.right_count),
                "target_answer": int(sample.target_answer),
                **count_maps,
                "arrangement_mode": str(sample.arrangement_mode),
            },
            "frames": {
                "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                "panels": dict(scene.panel_geometry),
            },
        },
        "query_spec": {
            trace_key("query", "id"): str(sample.query_key),
            "template_id": str(prompt_defaults["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": {
                trace_key("query", "id"): str(sample.query_key),
                "operation": str(sample.operation),
                "uses_color_binding": bool(sample.uses_color_binding),
                "left_operand": dict(left_operand),
                "right_operand": dict(right_operand),
                "left_count": int(sample.left_count),
                "right_count": int(sample.right_count),
                "target_answer": int(sample.target_answer),
                "distractor_count": int(sample.distractor_count),
                "object_count": int(sample.object_count),
                "arrangement_mode": str(sample.arrangement_mode),
                "pair_arithmetic_query_ids": list(tuple(query_ids)),
                "query_probabilities": dict(sample.query_probabilities),
                "shape_id_support": list(shape_support),
                "named_color_support": [str(entry.name) for entry in color_support],
                "shape_probabilities": dict(sample.shape_probabilities),
                "color_probabilities": dict(sample.color_probabilities),
                "answer_probabilities": dict(sample.answer_probabilities),
                "operand_count_probabilities": dict(sample.operand_count_probabilities),
                "distractor_count_probabilities": dict(sample.distractor_count_probabilities),
                "named_icon_fill_style_support": list(sample.fill_style_support),
                "fill_style_probabilities": dict(sample.fill_style_probabilities),
                "arrangement_mode_probabilities": dict(sample.arrangement_mode_probabilities),
            },
        },
        "render_spec": {
            "canvas_size": list(scene.panel_geometry["canvas_size"]),
            "coord_space": "pixel",
            "scene_id": SCENE_ID,
            "panel_geometry": dict(scene.panel_geometry),
            "style": {
                **icon_render_style_trace(render_params=render_params, sampled_palette_rgb=sampled_palette_rgb),
                "layout_mode": str(scene.layout_mode),
                "named_icon_slot_padding_px": int(slot_padding_px),
                "named_icon_slot_jitter_px": int(slot_jitter_px),
                "named_icon_stack_gap_px": int(stack_gap_px),
                "semantic_color_palette": semantic_color_palette(),
                "semantic_fill_style_support": list(fill_style_support),
            },
        },
        "render_map": {
            "image_id": "img0",
            "object_bboxes_px": object_bboxes(scene.instances),
            "counted_instance_ids": list(counted_instance_ids),
            "left_operand_instance_ids": list(left_instance_ids),
            "right_operand_instance_ids": list(right_instance_ids),
            "entity_partition": {
                str(instance.instance_id): str(role_by_instance_id.get(str(instance.instance_id), ""))
                for instance in scene.instances
            },
        },
        "execution_trace": {
            "scene_variant": "single_panel_named_shape_pair_arithmetic_field",
            "arrangement_mode": str(sample.arrangement_mode),
            trace_key("query", "id"): str(sample.query_key),
            "question_format": "count_named_shape_pair_arithmetic_icons",
            "operation": str(sample.operation),
            "uses_color_binding": bool(sample.uses_color_binding),
            "left_operand": dict(left_operand),
            "right_operand": dict(right_operand),
            "left_count": int(sample.left_count),
            "right_count": int(sample.right_count),
            "target_answer": int(sample.target_answer),
            "distractor_count": int(sample.distractor_count),
            "object_count": int(sample.object_count),
            **count_maps,
            "scene_shape_ids": [str(instance.shape_id) for instance in scene.instances],
            "scene_color_names": [str(instance.color_name) for instance in scene.instances],
            "scene_fill_styles": [str(instance.fill_style) for instance in scene.instances],
            "counted_instance_ids": list(counted_instance_ids),
            "left_operand_instance_ids": list(left_instance_ids),
            "right_operand_instance_ids": list(right_instance_ids),
        },
        "witness_symbolic": {
            "operation": str(sample.operation),
            "uses_color_binding": bool(sample.uses_color_binding),
            "left_operand": dict(left_operand),
            "right_operand": dict(right_operand),
            "left_count": int(sample.left_count),
            "right_count": int(sample.right_count),
            "answer": int(sample.target_answer),
            "counted_instance_ids": list(counted_instance_ids),
            "left_operand_instance_ids": list(left_instance_ids),
            "right_operand_instance_ids": list(right_instance_ids),
            "annotation_instance_ids": list(annotation_instance_ids),
            "operand_role_instance_ids": {
                "left_operand": list(left_instance_ids),
                "right_operand": list(right_instance_ids),
            },
        },
        "projected_annotation": {**dict(annotation_artifacts["projected_annotation"])},
    }


def build_scoped_region_trace_payload(
    *,
    scene: Any,
    render_params: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    annotation_artifacts: Mapping[str, Any],
    counted_instance_ids: Tuple[str, ...],
    shape_support: Sequence[str],
) -> dict[str, Any]:
    """Build trace sections for named-icon counts inside or outside a region."""

    serialized_instances = [serialize_region_icon(instance) for instance in scene.instances]
    shape_counts = Counter(str(instance.shape_id) for instance in scene.instances)
    inside_shape_counts = Counter(str(instance.shape_id) for instance in scene.instances if instance.inside_region)
    outside_shape_counts = Counter(str(instance.shape_id) for instance in scene.instances if not instance.inside_region)
    region_payload = region_to_trace(scene.region)
    count_maps = {
        "shape_counts": {str(key): int(value) for key, value in shape_counts.items()},
        "inside_shape_counts": {str(key): int(value) for key, value in inside_shape_counts.items()},
        "outside_shape_counts": {str(key): int(value) for key, value in outside_shape_counts.items()},
    }
    return {
        "scene_ir": {
            "scene_kind": "icons_named_shape_region_field",
            "scene_id": SCENE_ID,
            "entities": list(serialized_instances),
            "relations": {
                "counting_rule": "target_shape_center_membership_in_visible_region",
                "target_shape_id": str(scene.target_shape_id),
                "target_shape_name": str(scene.target_shape_name),
                "target_count": int(scene.target_count),
                **count_maps,
                "region": dict(region_payload),
            },
            "frames": {
                "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                "panels": dict(scene.panel_geometry),
            },
        },
        "query_spec": {
            trace_key("query", "id"): str(scene.region.query_key),
            "template_id": str(prompt_defaults["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": {
                "target_shape_id": str(scene.target_shape_id),
                "target_shape_name": str(scene.target_shape_name),
                "target_count": int(scene.target_count),
                "object_count": int(scene.object_count),
                trace_key("query", "id"): str(scene.region.query_key),
                "region": dict(region_payload),
                "shape_id_support": list(shape_support),
                "query_probabilities": dict(scene.query_probabilities),
                "shape_probabilities": dict(scene.shape_probabilities),
                "target_count_probabilities": dict(scene.target_count_probabilities),
                "object_count_probabilities": dict(scene.object_count_probabilities),
                "named_icon_fill_style_support": list(scene.fill_style_support),
                "fill_style_probabilities": dict(scene.fill_style_probabilities),
            },
        },
        "render_spec": {
            "canvas_size": list(scene.panel_geometry["canvas_size"]),
            "coord_space": "pixel",
            "scene_id": SCENE_ID,
            "panel_geometry": dict(scene.panel_geometry),
            "style": {
                **icon_render_style_trace(render_params=render_params, sampled_palette_rgb=scene.sampled_palette_rgb),
                "named_icon_fill_style_support": list(scene.fill_style_support),
                "region_fill_rgb": [int(value) for value in render_params["region_fill_rgb"]],
                "region_outline_rgb": [int(value) for value in render_params["region_outline_rgb"]],
                "region_guide_rgb": [int(value) for value in render_params["region_guide_rgb"]],
                "region_fill_alpha": int(render_params["region_fill_alpha"]),
                "region_outline_width_px": int(render_params["region_outline_width_px"]),
                "region_boundary_margin_px": int(render_params["region_boundary_margin_px"]),
            },
        },
        "render_map": {
            "image_id": "img0",
            "object_bboxes_px": object_bboxes(scene.instances),
            "object_centers_px": {
                str(instance.instance_id): [float(value) for value in instance.center_xy]
                for instance in scene.instances
            },
            "counted_instance_ids": list(counted_instance_ids),
            "region": dict(region_payload),
        },
        "execution_trace": {
            "scene_variant": "single_panel_named_shape_region_field",
            trace_key("query", "id"): str(scene.region.query_key),
            "question_format": "count_named_shape_icons_by_visible_region_membership",
            "target_shape_id": str(scene.target_shape_id),
            "target_shape_name": str(scene.target_shape_name),
            "target_count": int(scene.target_count),
            "object_count": int(scene.object_count),
            "region": dict(region_payload),
            **count_maps,
            "counted_instance_ids": list(counted_instance_ids),
        },
        "witness_symbolic": {
            "target_shape_id": str(scene.target_shape_id),
            "target_shape_name": str(scene.target_shape_name),
            "answer": int(scene.target_count),
            "counted_instance_ids": list(counted_instance_ids),
            "region": dict(region_payload),
        },
        "projected_annotation": {
            **dict(annotation_artifacts["projected_annotation"]),
            "counted_instance_ids": list(counted_instance_ids),
        },
    }


def build_distance_rank_trace_payload(
    *,
    scene_payload: Any,
    render_params: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    annotation_artifacts: Mapping[str, Any],
    answer_icon: DistanceRankRenderedIcon,
    query_probabilities: Mapping[str, float],
    answer_label_probabilities: Mapping[str, float],
    distractor_count_probabilities: Mapping[str, float],
    option_labels: Sequence[str],
) -> dict[str, Any]:
    """Build trace sections for distance-rank selection from a reference icon."""

    serialized_reference = serialize_distance_rank_icon(scene_payload.reference_icon)
    serialized_candidates = [serialize_distance_rank_icon(icon) for icon in scene_payload.candidate_icons]
    serialized_distractors = [serialize_distance_rank_icon(icon) for icon in scene_payload.distractor_icons]
    serialized_answer = serialize_distance_rank_icon(answer_icon)

    return {
        "scene_ir": {
            "scene_kind": "icons_named_field_distance_rank",
            "scene_id": SCENE_ID,
            "entities": [dict(serialized_reference), *serialized_candidates, *serialized_distractors],
            "relations": {
                "target": "labeled_candidate_distance_rank_from_named_reference",
                trace_key("query", "id"): str(scene_payload.query_key),
                "reference_instance_id": str(scene_payload.reference_icon.instance_id),
                "reference_description": str(scene_payload.reference_description),
                "candidate_labels": [str(label) for label in option_labels],
                "answer_label": str(scene_payload.answer_label),
                "answer_rank": int(scene_payload.answer_rank),
                "sorted_candidate_labels_by_distance": list(scene_payload.sorted_candidate_labels_by_distance),
            },
            "frames": {
                "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                "panels": dict(scene_payload.panel_geometry),
            },
        },
        "query_spec": {
            trace_key("query", "id"): str(scene_payload.query_key),
            "template_id": str(prompt_defaults["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": {
                trace_key("query", "id"): str(scene_payload.query_key),
                "query_id_probabilities": dict(query_probabilities),
                "distance_rank_query": str(scene_payload.query_key),
                "distance_rank_query_probabilities": dict(query_probabilities),
                "answer_label": str(scene_payload.answer_label),
                "answer_label_probabilities": dict(answer_label_probabilities),
                "candidate_count": int(len(scene_payload.candidate_icons)),
                "distractor_count": int(scene_payload.distractor_count),
                "distractor_count_probabilities": dict(distractor_count_probabilities),
                "reference_description": str(scene_payload.reference_description),
            },
        },
        "render_spec": {
            "canvas_size": [int(render_params["canvas_width"]), int(render_params["canvas_height"])],
            "coord_space": "pixel",
            "panel_geometry": dict(scene_payload.panel_geometry),
            "style": {
                **icon_render_style_trace(
                    render_params=render_params,
                    sampled_palette_rgb=tuple(scene_payload.sampled_palette_rgb),
                ),
                "candidate_label_font_size_px": int(render_params["candidate_label_font_size_px"]),
                "candidate_label_color_rgb": [int(value) for value in render_params["candidate_label_color_rgb"]],
                "candidate_label_stroke_rgb": [int(value) for value in render_params["candidate_label_stroke_rgb"]],
                "candidate_label_background_rgb": [
                    int(value) for value in render_params["candidate_label_background_rgb"]
                ],
                "candidate_label_border_rgb": [int(value) for value in render_params["candidate_label_border_rgb"]],
            },
        },
        "render_map": {
            "image_id": "img0",
            "anchors": {
                "reference": dict(serialized_reference),
                "candidate_icons": list(serialized_candidates),
                "distractor_icons": list(serialized_distractors),
                "answer_label": str(scene_payload.answer_label),
                "answer_candidate": dict(serialized_answer),
            },
        },
        "execution_trace": {
            "scene_variant": "single_panel_named_field_distance_rank",
            trace_key("query", "id"): str(scene_payload.query_key),
            "distance_rank_query": str(scene_payload.query_key),
            "distance_rank_query_probabilities": dict(query_probabilities),
            "answer_label": str(scene_payload.answer_label),
            "answer_label_probabilities": dict(answer_label_probabilities),
            "answer_rank": int(scene_payload.answer_rank),
            "candidate_count": int(len(scene_payload.candidate_icons)),
            "distractor_count": int(scene_payload.distractor_count),
            "distractor_count_probabilities": dict(distractor_count_probabilities),
            "reference_description": str(scene_payload.reference_description),
            "reference_instance_id": str(scene_payload.reference_icon.instance_id),
            "candidate_labels": [str(label) for label in option_labels],
            "distance_by_label_px": {str(key): float(value) for key, value in scene_payload.distance_by_label.items()},
            "sorted_candidate_labels_by_distance": list(scene_payload.sorted_candidate_labels_by_distance),
            "question_format": "select_labeled_named_icon_by_distance_rank_from_unique_named_reference",
        },
        "witness_symbolic": {
            trace_key("query", "id"): str(scene_payload.query_key),
            "reference_instance_id": str(scene_payload.reference_icon.instance_id),
            "reference_description": str(scene_payload.reference_description),
            "answer_label": str(scene_payload.answer_label),
            "answer_instance_id": str(answer_icon.instance_id),
            "answer_rank": int(scene_payload.answer_rank),
            "sorted_candidate_labels_by_distance": list(scene_payload.sorted_candidate_labels_by_distance),
            "annotation_roles": {
                "reference_icon": str(scene_payload.reference_icon.instance_id),
                "selected_candidate": str(answer_icon.instance_id),
            },
        },
        "projected_annotation": {
            **dict(annotation_artifacts["projected_annotation"]),
            "items": [
                {
                    "role": "reference_icon",
                    "instance_id": str(scene_payload.reference_icon.instance_id),
                    "bbox_xyxy": list(scene_payload.reference_icon.bbox_xyxy),
                },
                {
                    "role": "selected_candidate",
                    "instance_id": str(answer_icon.instance_id),
                    "bbox_xyxy": list(answer_icon.bbox_xyxy),
                },
            ],
        },
    }


__all__ = [
    "BOOLEAN_PREDICATE_AND",
    "BOOLEAN_PREDICATE_ATTRIBUTE_WITHOUT_SHAPE",
    "BOOLEAN_PREDICATE_NEITHER",
    "BOOLEAN_PREDICATE_OR",
    "BOOLEAN_PREDICATE_SHAPE_WITHOUT_ATTRIBUTE",
    "BOOLEAN_PREDICATE_XOR",
    "COUNTERFACTUAL_SHAPE_REMOVAL",
    "COUNTERFACTUAL_SHAPE_REPLACEMENT",
    "build_boolean_query_metadata",
    "build_boolean_trace_payload",
    "build_counterfactual_query_metadata",
    "build_counterfactual_trace_payload",
    "build_distance_rank_trace_payload",
    "build_pair_arithmetic_trace_payload",
    "build_shape_count_query_metadata",
    "build_shape_count_trace_payload",
    "build_scoped_region_trace_payload",
    "object_bboxes",
    "region_to_trace",
    "render_slot_params",
    "semantic_color_palette",
    "serialize_closer_reference_icon",
    "serialize_distance_rank_icon",
    "serialize_region_icon",
    "shape_counted_instance_ids",
]
