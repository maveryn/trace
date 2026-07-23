"""Neutral render and trace scaffolding for region-map charts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .rendering import MarkerMapRenderResult, RegionMapRenderResult, font_assets_payload


def render_spec_payload(
    *,
    dataset: Mapping[str, Any],
    rendered: RegionMapRenderResult,
    scene_variant: str,
    show_region_value_labels: bool,
) -> dict[str, Any]:
    """Return scene render metadata independent of public task identity."""

    render_params = rendered.render_params
    return {
        "scene_variant": str(scene_variant),
        "map_asset_id": str(dataset.get("map_asset_id") or ""),
        "geographic_map_variant": str(dataset.get("geographic_map_variant") or ""),
        "geographic_map_variant_probabilities": dict(dataset.get("geographic_map_variant_probabilities", {})),
        "map_display_name": str(dataset.get("map_display_name") or ""),
        "map_region_noun": str(dataset.get("map_region_noun") or ""),
        "map_source": dict(dataset.get("map_source", {})),
        "geographic_component_constraints": dict(dataset.get("geographic_component_constraints", {})),
        "geographic_eligible_region_count_after_component_filter": int(
            dataset.get("geographic_eligible_region_count_after_component_filter", 0) or 0
        ),
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "rows": int(dataset["rows"]),
        "cols": int(dataset["cols"]),
        "active_cells": [[int(row), int(col)] for row, col in dataset["active_cells"]],
        "selected_region_ids": [str(region["region_id"]) for region in dataset["regions"]],
        "legend_position": str(render_params.legend_position),
        "legend_position_probabilities": dict(render_params.legend_position_probabilities),
        "map_palette_rgb": [[int(channel) for channel in color] for color in render_params.map_palette_rgb],
        "map_palette_variant": str(render_params.map_palette_variant),
        "map_palette_variant_probabilities": dict(render_params.map_palette_variant_probabilities),
        "background_style": dict(rendered.background_meta),
        "information_scene_style": dict(rendered.background_meta.get("information_scene_style", {})),
        "font_assets": font_assets_payload(rendered.chart_font_family),
        "region_gap_px": int(render_params.region_gap_px),
        "layout_jitter": dict(render_params.layout_jitter_meta),
        "map_render_style": dict(rendered.rendered_scene.render_meta),
        "show_region_value_labels": bool(show_region_value_labels),
        "show_region_reference_labels": bool(dataset.get("show_region_reference_labels")),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def render_map_payload(*, rendered: RegionMapRenderResult) -> dict[str, Any]:
    """Return pixel lookup maps for rendered region-map elements."""

    render_map = {
        "image_id": "img0",
        "panel_bbox_px": list(rendered.rendered_scene.panel_bbox_px),
        "title_bbox_px": list(rendered.rendered_scene.title_bbox_px),
        "map_bbox_px": list(rendered.rendered_scene.map_bbox_px),
        "legend_bbox_px": list(rendered.rendered_scene.legend_bbox_px),
        "region_bboxes_px": dict(rendered.rendered_scene.region_bbox_map),
        "region_centers_px": dict(rendered.rendered_scene.region_center_map),
        "legend_entry_bboxes_px": dict(rendered.rendered_scene.legend_entry_bbox_map),
    }
    if "world_projection_bbox_px" in rendered.rendered_scene.render_meta:
        render_map["world_projection_bbox_px"] = list(rendered.rendered_scene.render_meta["world_projection_bbox_px"])
    return render_map


def base_scene_params(
    *,
    dataset: Mapping[str, Any],
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
) -> dict[str, Any]:
    """Return common semantic metadata for region-map objectives."""

    return {
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "geographic_map_variant": str(dataset.get("geographic_map_variant") or ""),
        "geographic_map_variant_probabilities": dict(dataset.get("geographic_map_variant_probabilities", {})),
        "region_count": int(dataset["region_count"]),
        "target_count": int(dataset["target_count"]),
        "target_count_probabilities": dict(dataset["target_count_probabilities"]),
        **dict(dataset["question_params"]),
    }


def marker_render_spec_payload(
    *,
    dataset: Mapping[str, Any],
    rendered: MarkerMapRenderResult,
    scene_variant: str,
) -> dict[str, Any]:
    """Return marker-layer render metadata independent of public task identity."""

    render_params = rendered.render_params
    return {
        "scene_variant": str(scene_variant),
        "map_asset_id": str(dataset.get("map_asset_id") or ""),
        "geographic_map_variant": str(dataset.get("geographic_map_variant") or ""),
        "geographic_map_variant_probabilities": dict(dataset.get("geographic_map_variant_probabilities", {})),
        "map_display_name": str(dataset.get("map_display_name") or ""),
        "map_region_noun": str(dataset.get("map_region_noun") or ""),
        "map_source": dict(dataset.get("map_source", {})),
        "geographic_component_constraints": dict(dataset.get("geographic_component_constraints", {})),
        "geographic_eligible_region_count_after_component_filter": int(
            dataset.get("geographic_eligible_region_count_after_component_filter", 0) or 0
        ),
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "rows": int(dataset["rows"]),
        "cols": int(dataset["cols"]),
        "active_cells": [[int(row), int(col)] for row, col in dataset["active_cells"]],
        "selected_region_ids": [str(region["region_id"]) for region in dataset["regions"]],
        "legend_position": str(render_params.legend_position),
        "legend_position_probabilities": dict(render_params.legend_position_probabilities),
        "marker_render": dict(rendered.marker_render_meta),
        "map_palette_rgb": [[int(channel) for channel in color] for color in render_params.map_palette_rgb],
        "map_palette_variant": str(render_params.map_palette_variant),
        "map_palette_variant_probabilities": dict(render_params.map_palette_variant_probabilities),
        "background_style": dict(rendered.background_meta),
        "information_scene_style": dict(rendered.background_meta.get("information_scene_style", {})),
        "font_assets": font_assets_payload(rendered.chart_font_family),
        "region_gap_px": int(render_params.region_gap_px),
        "layout_jitter": dict(render_params.layout_jitter_meta),
        "map_render_style": dict(rendered.rendered_scene.render_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def marker_render_map_payload(*, rendered: MarkerMapRenderResult) -> dict[str, Any]:
    """Return pixel lookup maps for rendered region-map marker elements."""

    marker_points_by_region = {
        str(region_id): [
            round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
            round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
        ]
        for region_id, bbox in rendered.marker_group_bbox_map.items()
    }
    render_map = {
        "image_id": "img0",
        "panel_bbox_px": list(rendered.rendered_scene.panel_bbox_px),
        "title_bbox_px": list(rendered.rendered_scene.title_bbox_px),
        "map_bbox_px": list(rendered.rendered_scene.map_bbox_px),
        "legend_bbox_px": list(rendered.rendered_scene.legend_bbox_px),
        "region_bboxes_px": dict(rendered.rendered_scene.region_bbox_map),
        "region_centers_px": dict(rendered.rendered_scene.region_center_map),
        "legend_entry_bboxes_px": dict(rendered.rendered_scene.legend_entry_bbox_map),
        "marker_group_bboxes_px": dict(rendered.marker_group_bbox_map),
        "marker_bboxes_px": dict(rendered.marker_bboxes_by_region),
        "marker_points_px": dict(marker_points_by_region),
    }
    if "world_projection_bbox_px" in rendered.rendered_scene.render_meta:
        render_map["world_projection_bbox_px"] = list(rendered.rendered_scene.render_meta["world_projection_bbox_px"])
    return render_map


def base_marker_query_params(
    *,
    dataset: Mapping[str, Any],
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
) -> dict[str, Any]:
    """Return common query metadata for region-map marker objectives."""

    return {
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "geographic_map_variant": str(dataset.get("geographic_map_variant") or ""),
        "geographic_map_variant_probabilities": dict(dataset.get("geographic_map_variant_probabilities", {})),
        "region_count": int(dataset["region_count"]),
        "target_count": int(dataset["target_count"]),
        "target_count_probabilities": dict(dataset["target_count_probabilities"]),
        **dict(dataset["question_params"]),
    }


def build_trace_scaffold(
    *,
    dataset: Mapping[str, Any],
    rendered: RegionMapRenderResult,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    query_spec: Mapping[str, Any],
    relations: Mapping[str, Any],
    question_format: str,
    answer_value: int | str,
    answer_type: str,
    annotation_type: str,
    annotation_region_ids: Sequence[str],
    projected_annotation: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    annotation_refs: Sequence[Mapping[str, Any]],
    show_region_value_labels: bool,
) -> dict[str, Any]:
    """Build common trace sections around task-bound answer and annotation."""

    return {
        "scene_ir": {
            "scene_kind": "chart_region_map",
            "entities": [dict(entity) for entity in rendered.rendered_scene.entities],
            "relations": {
                "scene_variant": str(scene_variant),
                "answer_value": answer_value,
                "annotation_region_ids": [str(region_id) for region_id in annotation_region_ids],
                **dict(relations),
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": render_spec_payload(
            dataset=dataset,
            rendered=rendered,
            scene_variant=str(scene_variant),
            show_region_value_labels=bool(show_region_value_labels),
        ),
        "render_map": render_map_payload(rendered=rendered),
        "execution_trace": {
            "scene_id": "region_map",
            "scene_variant": str(scene_variant),
            "question_format": str(question_format),
            "scene_title": str(dataset["scene_title"]),
            "map_asset_id": str(dataset.get("map_asset_id") or ""),
            "geographic_map_variant": str(dataset.get("geographic_map_variant") or ""),
            "map_display_name": str(dataset.get("map_display_name") or ""),
            "map_region_noun": str(dataset.get("map_region_noun") or ""),
            "geographic_component_constraints": dict(dataset.get("geographic_component_constraints", {})),
            "geographic_eligible_region_count_after_component_filter": int(
                dataset.get("geographic_eligible_region_count_after_component_filter", 0) or 0
            ),
            "rows": int(dataset["rows"]),
            "cols": int(dataset["cols"]),
            "region_count": int(dataset["region_count"]),
            "active_cells": [[int(row), int(col)] for row, col in dataset["active_cells"]],
            "legend_bins": [dict(item) for item in dataset["legend_bins"]],
            "regions": [dict(item) for item in dataset["regions"]],
            "regions_by_id": {str(key): dict(value) for key, value in dict(dataset["regions_by_id"]).items()},
            "answer_value": answer_value,
            "answer_type": str(answer_type),
            "annotation_type": str(annotation_type),
            "annotation_region_ids": [str(region_id) for region_id in annotation_region_ids],
            "show_region_reference_labels": bool(dataset.get("show_region_reference_labels")),
            "target_bin_indices": [int(value) for value in dataset.get("target_bin_indices", [])],
            "nonmatching_bin_indices": [int(value) for value in dataset.get("nonmatching_bin_indices", [])],
            "threshold_direction": str(dataset.get("threshold_direction") or ""),
            **dict(relations),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
        "annotation_refs": [dict(ref) for ref in annotation_refs],
    }


def build_marker_trace_scaffold(
    *,
    dataset: Mapping[str, Any],
    rendered: MarkerMapRenderResult,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    query_spec: Mapping[str, Any],
    relations: Mapping[str, Any],
    question_format: str,
    answer_value: int | str,
    answer_type: str,
    annotation_type: str,
    annotation_region_ids: Sequence[str],
    projected_annotation: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    annotation_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build common trace sections around marker-task-bound answer and annotation."""

    return {
        "scene_ir": {
            "scene_kind": "chart_region_map",
            "entities": [dict(entity) for entity in rendered.rendered_scene.entities],
            "relations": {
                "scene_variant": str(scene_variant),
                "answer_value": answer_value,
                "annotation_region_ids": [str(region_id) for region_id in annotation_region_ids],
                **dict(relations),
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": marker_render_spec_payload(dataset=dataset, rendered=rendered, scene_variant=str(scene_variant)),
        "render_map": marker_render_map_payload(rendered=rendered),
        "execution_trace": {
            "scene_id": "region_map",
            "scene_variant": str(scene_variant),
            "question_format": str(question_format),
            "scene_title": str(dataset["scene_title"]),
            "map_asset_id": str(dataset.get("map_asset_id") or ""),
            "geographic_map_variant": str(dataset.get("geographic_map_variant") or ""),
            "map_display_name": str(dataset.get("map_display_name") or ""),
            "map_region_noun": str(dataset.get("map_region_noun") or ""),
            "geographic_component_constraints": dict(dataset.get("geographic_component_constraints", {})),
            "geographic_eligible_region_count_after_component_filter": int(
                dataset.get("geographic_eligible_region_count_after_component_filter", 0) or 0
            ),
            "rows": int(dataset["rows"]),
            "cols": int(dataset["cols"]),
            "region_count": int(dataset["region_count"]),
            "active_cells": [[int(row), int(col)] for row, col in dataset["active_cells"]],
            "regions": [dict(item) for item in dataset["regions"]],
            "regions_by_id": {str(key): dict(value) for key, value in dict(dataset["regions_by_id"]).items()},
            "answer_value": answer_value,
            "answer_type": str(answer_type),
            "annotation_type": str(annotation_type),
            "annotation_region_ids": [str(region_id) for region_id in annotation_region_ids],
            **dict(relations),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
        "annotation_refs": [dict(ref) for ref in annotation_refs],
    }


__all__ = [
    "base_marker_query_params",
    "base_scene_params",
    "build_marker_trace_scaffold",
    "build_trace_scaffold",
    "marker_render_map_payload",
    "marker_render_spec_payload",
    "render_map_payload",
    "render_spec_payload",
]
