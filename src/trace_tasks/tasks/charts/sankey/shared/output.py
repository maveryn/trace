"""Neutral render-map scaffolding for standard Sankey charts."""

from __future__ import annotations

from typing import Any

from .defaults import font_assets_payload
from .sampling import node_dict, path_dict
from .state import SankeyDataset, SankeyRenderResult


def render_spec(dataset: SankeyDataset, rendered: SankeyRenderResult) -> dict[str, Any]:
    frame = dataset.frame
    params = rendered.render_params
    return {
        "scene_variant": str(frame.scene_variant),
        "canvas_width": int(params.canvas_width),
        "canvas_height": int(params.canvas_height),
        "source_count": int(len(frame.sources)),
        "middle_count": int(len(frame.middles)),
        "target_count": int(len(frame.targets)),
        "path_count": int(len(frame.paths)),
        "max_paths_per_node_side": int(frame.max_paths_per_node_side),
        "value_min": int(frame.value_min),
        "value_max": int(frame.value_max),
        "min_flow_width_px": int(params.min_flow_width_px),
        "max_flow_width_px": int(params.max_flow_width_px),
        "layout_jitter": dict(params.layout_jitter_meta),
        "background_style": dict(rendered.background_meta),
        "information_scene_style": dict(rendered.background_meta.get("information_scene_style", {})),
        "font_assets": font_assets_payload(chart_font_family=rendered.chart_font_family),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def render_map(rendered: SankeyRenderResult) -> dict[str, Any]:
    scene = rendered.rendered_scene
    return {
        "panel_bbox_px": list(scene.panel_bbox_px),
        "title_bbox_px": list(scene.title_bbox_px),
        "plot_bbox_px": list(scene.plot_bbox_px),
        "node_bboxes_px": dict(scene.node_bbox_map),
        "node_label_bboxes_px": dict(scene.node_label_bbox_map),
        "segment_bboxes_px": dict(scene.segment_bbox_map),
        "segment_label_bboxes_px": dict(scene.segment_label_bbox_map),
        "segment_centers_px": dict(scene.segment_center_map),
    }


def scene_records(dataset: SankeyDataset) -> dict[str, Any]:
    frame = dataset.frame
    return {
        "scene_title": str(frame.scene_title),
        "sources": [node_dict(node) for node in frame.sources],
        "middles": [node_dict(node) for node in frame.middles],
        "targets": [node_dict(node) for node in frame.targets],
        "paths": [path_dict(path) for path in frame.paths],
        "paths_by_id": {str(path.path_id): path_dict(path) for path in frame.paths},
        "source_count": int(len(frame.sources)),
        "middle_count": int(len(frame.middles)),
        "target_count": int(len(frame.targets)),
        "path_count": int(len(frame.paths)),
        "max_paths_per_node_side": int(frame.max_paths_per_node_side),
        "path_side_counts": dict(frame.path_side_counts),
        "value_min": int(frame.value_min),
        "value_max": int(frame.value_max),
    }


__all__ = ["render_map", "render_spec", "scene_records"]
