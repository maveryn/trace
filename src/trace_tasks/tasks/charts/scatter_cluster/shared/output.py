"""Trace-output primitives for scatter-cluster chart scenes."""

from __future__ import annotations

from typing import Any

from .defaults import font_assets_payload
from .state import ScatterClusterDataset, ScatterClusterInputs, ScatterClusterRenderResult


def values_by_cluster(dataset: ScatterClusterDataset) -> dict[str, Any]:
    return {
        str(cluster.cluster_label): {
            "center": [round(float(cluster.center_x), 3), round(float(cluster.center_y), 3)],
            "slope": round(float(cluster.slope), 4),
            "spread_x": round(float(cluster.spread_x), 3),
            "spread_y": round(float(cluster.spread_y), 3),
            "area_envelope": (
                {
                    "center": [
                        round(float(cluster.area_envelope.center_x), 3),
                        round(float(cluster.area_envelope.center_y), 3),
                    ],
                    "radius_x": round(float(cluster.area_envelope.radius_x), 3),
                    "radius_y": round(float(cluster.area_envelope.radius_y), 3),
                    "angle_degrees": round(float(cluster.area_envelope.angle_degrees), 3),
                    "area_value": round(float(cluster.area_envelope.area_value), 4),
                }
                if cluster.area_envelope is not None
                else None
            ),
            "points": [
                {
                    "point_id": str(point.point_id),
                    "x_value": round(float(point.x_value), 3),
                    "y_value": round(float(point.y_value), 3),
                }
                for point in cluster.points
            ],
        }
        for cluster in dataset.clusters
    }


def render_spec(rendered: ScatterClusterRenderResult) -> dict[str, Any]:
    return {
        "canvas_width": int(rendered.render_params.canvas_width),
        "canvas_height": int(rendered.render_params.canvas_height),
        "coord_space": "pixel",
        "plot_bbox_px": list(rendered.rendered_scene.plot_bbox_px),
        "point_radius_px": int(rendered.render_params.point_radius_px),
        "layout_jitter": dict(rendered.render_params.layout_jitter_meta),
        "font_assets": font_assets_payload(chart_font_family=rendered.chart_font_family),
        "background_style": dict(rendered.background_meta),
        "information_scene_style": dict(rendered.background_meta.get("information_scene_style", {})),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def render_map(rendered: ScatterClusterRenderResult) -> dict[str, Any]:
    rendered_scene = rendered.rendered_scene
    return {
        "image_id": "img0",
        "plot_bbox_px": list(rendered_scene.plot_bbox_px),
        "point_bboxes_px": dict(rendered_scene.point_bboxes),
        "cluster_bboxes_px": dict(rendered_scene.cluster_bboxes),
        "cluster_envelope_bboxes_px": dict(rendered_scene.cluster_envelope_bboxes),
        "cluster_label_bboxes_px": dict(rendered_scene.cluster_label_bboxes),
        "legend_bboxes_px": dict(rendered_scene.legend_bboxes),
        "option_bboxes_px": dict(rendered_scene.option_bboxes),
        "option_centers_px": dict(rendered_scene.option_centers_px),
    }


def base_execution_record(
    *,
    dataset: ScatterClusterDataset,
    inputs: ScatterClusterInputs,
) -> dict[str, Any]:
    total_points = int(sum(len(cluster.points) for cluster in dataset.clusters))
    return {
        "scene_variant": str(dataset.scene_variant),
        "scene_variant_probabilities": {str(dataset.scene_variant): 1.0},
        "question_format": "scatter_cluster_query",
        "answer": str(dataset.question.answer),
        "answer_type": str(dataset.question.answer_type),
        "annotation_type": str(dataset.question.annotation_type),
        "cluster_labels": list(inputs.labels),
        "cluster_count": int(inputs.cluster_count),
        "points_per_cluster": int(inputs.points_per_cluster),
        "total_point_count": int(total_points),
        "option_labels": [str(marker.option_label) for marker in dataset.option_markers],
        "values_by_cluster": values_by_cluster(dataset),
        **dict(dataset.question.params),
    }
