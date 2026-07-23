"""Annotation projection helpers for radial Sankey charts."""

from __future__ import annotations

from typing import Any

from .state import RadialSankeyDataset, RadialSankeyRenderResult


def annotation_payload(*, dataset: RadialSankeyDataset, rendered: RadialSankeyRenderResult) -> dict[str, Any]:
    """Project task-owned link/node references into public annotation values."""

    rendered_scene = rendered.rendered_scene
    link_refs = [str(value) for value in dataset.question.annotation_link_ids]
    node_refs = [str(value) for value in dataset.question.annotation_node_ids]
    annotation_type = str(dataset.question.annotation_type)

    link_boxes = [list(rendered_scene.link_label_bbox_map[str(ref)]) for ref in link_refs]
    node_boxes = [list(rendered_scene.node_bbox_map[str(ref)]) for ref in node_refs]
    if annotation_type == "bbox":
        if link_boxes or len(node_boxes) != 1:
            raise ValueError("radial Sankey scalar bbox annotation requires exactly one endpoint node")
        value = list(node_boxes[0])
        return {
            "type": "bbox",
            "value": list(value),
            "link_refs": list(link_refs),
            "node_refs": list(node_refs),
            "projected_annotation": {
                "type": "bbox",
                "bbox": list(value),
                "pixel_bbox": list(value),
                "node_refs": list(node_refs),
                "node_bbox_map": {
                    str(ref): list(rendered_scene.node_bbox_map[str(ref)])
                    for ref in node_refs
                },
            },
        }
    if annotation_type == "bbox_set":
        boxes = [*link_boxes, *node_boxes]
        return {
            "type": "bbox_set",
            "value": [list(box) for box in boxes],
            "link_refs": list(link_refs),
            "node_refs": list(node_refs),
            "projected_annotation": {
                "type": "bbox_set",
                "bbox_set": [list(box) for box in boxes],
                "pixel_bbox_set": [list(box) for box in boxes],
                "link_refs": list(link_refs),
                "node_refs": list(node_refs),
                "link_label_bbox_map": {
                    str(ref): list(rendered_scene.link_label_bbox_map[str(ref)])
                    for ref in link_refs
                },
                "node_bbox_map": {
                    str(ref): list(rendered_scene.node_bbox_map[str(ref)])
                    for ref in node_refs
                },
            },
        }
    raise ValueError(f"unsupported radial Sankey annotation type: {annotation_type}")


__all__ = ["annotation_payload"]
