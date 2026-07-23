"""Trace serialization and output helpers for graph node-link tasks."""

from __future__ import annotations

from typing import Any

def node_entities(rendered_scene: Any, sample: Any | None = None) -> list[dict[str, Any]]:
    """Return generic rendered-node entities for trace inspection."""

    removed_node_label = str(getattr(sample, "removed_node_label", ""))
    post_removal_isolated_labels = {
        str(label)
        for label in getattr(sample, "target_labels", ())
    } if hasattr(sample, "removed_node_label") else set()
    return [
        {
            "entity_id": f"node_{node.label}",
            "entity_kind": "graph_node",
            "label": str(node.label),
            "degree": int(node.degree),
            "neighbors": list(node.neighbors),
            "successors": list(node.successors),
            "predecessors": list(node.predecessors),
            "center_px": list(node.center_xy),
            "bbox_xyxy": list(node.bbox_xyxy),
            "node_color_name": str(node.color_name) if node.color_name is not None else None,
            "fill_rgb": list(node.fill_rgb) if node.fill_rgb is not None else None,
            "border_rgb": list(node.border_rgb) if node.border_rgb is not None else None,
            "is_removed_query_node": bool(removed_node_label and str(node.label) == removed_node_label),
            "is_post_removal_isolated": bool(str(node.label) in post_removal_isolated_labels),
        }
        for node in rendered_scene.nodes
    ]


def edge_entities(rendered_scene: Any, sample: Any) -> list[dict[str, Any]]:
    """Return generic rendered-edge entities for trace inspection."""

    labels_by_edge = getattr(sample, "edge_attribute_labels_by_label", {})
    colors_by_edge = getattr(sample, "edge_color_names_by_label", {})
    weights_by_edge = getattr(sample, "edge_weights_by_label", {})
    return [
        {
            "entity_id": str(edge.edge_id),
            "entity_kind": "graph_edge",
            "node_u_label": str(edge.node_u_label),
            "node_v_label": str(edge.node_v_label),
            "directed": bool(edge.directed),
            "segment_px": [list(edge.segment_px[0]), list(edge.segment_px[1])],
            "route_variant": str(edge.route_variant),
            "control_px": list(edge.control_px) if edge.control_px is not None else None,
            "edge_text_label": labels_by_edge.get((str(edge.node_u_label), str(edge.node_v_label))),
            "edge_color_name": str(edge.color_name)
            if edge.color_name is not None
            else colors_by_edge.get((str(edge.node_u_label), str(edge.node_v_label))),
            "edge_color_rgb": list(edge.edge_color_rgb) if edge.edge_color_rgb is not None else None,
            "label_bbox_xyxy": list(edge.edge_label_bbox_xyxy) if edge.edge_label_bbox_xyxy is not None else None,
            "label_center_px": [
                int(round((float(edge.edge_label_bbox_xyxy[0]) + float(edge.edge_label_bbox_xyxy[2])) / 2.0)),
                int(round((float(edge.edge_label_bbox_xyxy[1]) + float(edge.edge_label_bbox_xyxy[3])) / 2.0)),
            ] if edge.edge_label_bbox_xyxy is not None else None,
            "edge_weight": int(edge.weight)
            if edge.weight is not None
            else weights_by_edge.get((str(edge.node_u_label), str(edge.node_v_label))),
            "weight_label_bbox_xyxy": list(edge.weight_label_bbox_xyxy)
            if edge.weight_label_bbox_xyxy is not None
            else None,
            "weight_label_center_px": [
                int(round((float(edge.weight_label_bbox_xyxy[0]) + float(edge.weight_label_bbox_xyxy[2])) / 2.0)),
                int(round((float(edge.weight_label_bbox_xyxy[1]) + float(edge.weight_label_bbox_xyxy[3])) / 2.0)),
            ] if edge.weight_label_bbox_xyxy is not None else None,
        }
        for edge in rendered_scene.edges
    ]


__all__ = [
    "edge_entities",
    "node_entities",
]
