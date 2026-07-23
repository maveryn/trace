"""Annotation and trace payload helpers for concept-map scenes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Sequence

from trace_tasks.tasks.pages.shared.diagram.common import projected_diagram_bbox_annotation
from trace_tasks.tasks.pages.shared.page_semantic_assets import page_semantic_asset_manifest_metadata

from .state import ConceptMapCase, RenderedConceptMap


def _round_box(bbox: Sequence[float]) -> List[float]:
    return [round(float(value), 3) for value in bbox]


def node_entities(case: ConceptMapCase) -> List[Dict[str, Any]]:
    """Return concept nodes as metadata entities without public task identity."""

    scene = case.scene
    entities: List[Dict[str, Any]] = [
        {
            "entity_id": "central",
            "entity_type": "concept_topic",
            "label": str(scene["central_label"]),
            "shape": str(scene["branches"][0]["central"].get("shape", "rounded_rect")),
            "bbox_id": "node:central",
        }
    ]
    for branch in scene["branches"]:
        entities.append(
            {
                "entity_id": str(branch["branch_id"]),
                "entity_type": "concept_branch",
                "label": str(branch["label"]),
                "shape": str(branch.get("shape", "rounded_rect")),
                "bbox_id": f"node:{branch['branch_id']}",
            }
        )
        for child in branch["children"]:
            entities.append(
                {
                    "entity_id": str(child["node_id"]),
                    "entity_type": "concept_child",
                    "branch_id": str(branch["branch_id"]),
                    "label": str(child["label"]),
                    "marker_id": str(child["marker_id"]),
                    "shape": str(child.get("shape", "rounded_rect")),
                    "bbox_id": f"node:{child['node_id']}",
                }
            )
    return entities


def branch_payload(case: ConceptMapCase) -> List[Dict[str, Any]]:
    """Return branch records with child ids and rendered node shapes."""

    return [
        {
            "branch_id": str(branch["branch_id"]),
            "label": str(branch["label"]),
            "shape": str(branch.get("shape", "rounded_rect")),
            "bbox_id": f"node:{branch['branch_id']}",
            "child_node_ids": [str(child["node_id"]) for child in branch["children"]],
            "child_node_shapes": {
                str(child["node_id"]): str(child.get("shape", "rounded_rect"))
                for child in branch["children"]
            },
        }
        for branch in case.scene["branches"]
    ]


def common_trace_fields(case: ConceptMapCase) -> Dict[str, Any]:
    """Return scene fields shared by all concept-map objective contracts."""

    scene = case.scene
    return {
        "view_family": "concept_map",
        "scene_title": str(scene["scene_title"]),
        "central_label": str(scene["central_label"]),
        "context_id": str(scene["context_id"]),
        "layout_variant": str(scene["layout_variant"]),
        "style_variant": str(scene["style_variant"]),
        "node_shape_profile": str(scene["node_shape_profile"]),
        "branch_count": int(scene["branch_count"]),
        "child_count": int(scene["child_count"]),
        "branches": branch_payload(case),
        "page_semantic_assets": page_semantic_asset_manifest_metadata(
            semantic_role="marker",
            allowed_use="filter",
        ),
        "context_probabilities": dict(case.context_probabilities),
        "layout_probabilities": dict(case.layout_probabilities),
        "style_probabilities": dict(case.style_probabilities),
    }


def target_payload(case: ConceptMapCase) -> Dict[str, Any]:
    """Return selected objective target metadata."""

    return deepcopy(dict(case.selection))


def count_annotation(case: ConceptMapCase, rendered: RenderedConceptMap) -> List[List[float]]:
    """Return homogeneous node boxes for a concept-map count objective."""

    node_bboxes = rendered.render_map["node_bboxes_px"]
    return [
        _round_box(node_bboxes[str(node_id)])
        for node_id in [str(item) for item in case.selection.get("annotation_node_ids", [])]
    ]


def selected_child_annotation(case: ConceptMapCase, rendered: RenderedConceptMap) -> List[float]:
    """Return the scalar bbox for the selected answer child."""

    node_bboxes = rendered.render_map["node_bboxes_px"]
    answer_node_id = str(case.selection["answer_node_id"])
    return _round_box(node_bboxes[str(answer_node_id)])


def projected_annotation_payload(case: ConceptMapCase, rendered: RenderedConceptMap, annotation_kind: str) -> Dict[str, Any]:
    """Build projected annotation metadata from selected rendered nodes."""

    if str(annotation_kind) == "bbox":
        bbox = selected_child_annotation(case, rendered)
        annotation_ids = selection_annotation_ids(case)
        payload: Dict[str, Any] = {
            "type": "bbox",
            "bbox": list(bbox),
            "pixel_bbox": list(bbox),
        }
        if annotation_ids:
            payload["annotation_id"] = str(annotation_ids[0])
        return payload

    node_bboxes = rendered.render_map["node_bboxes_px"]
    annotation_ids: list[str] = []
    bbox_source_map: Dict[str, Sequence[float]] = {}
    for node_id in [str(item) for item in case.selection.get("annotation_node_ids", [])]:
        annotation_id = f"node:{node_id}"
        annotation_ids.append(str(annotation_id))
        bbox_source_map[str(annotation_id)] = node_bboxes[str(node_id)]
    projection = projected_diagram_bbox_annotation(bbox_source_map, annotation_ids)
    return {
        **dict(projection),
        "bbox_set": [_round_box(bbox) for bbox in projection.get("bbox_set", [])],
        "pixel_bbox_set": [_round_box(bbox) for bbox in projection.get("pixel_bbox_set", projection.get("bbox_set", []))],
    }


def selection_annotation_ids(case: ConceptMapCase) -> List[str]:
    """Return bbox ids used as visual witnesses for the selected target."""

    return [
        f"node:{node_id}"
        for node_id in [str(item) for item in case.selection.get("annotation_node_ids", [])]
    ]


def selection_role_node_ids(case: ConceptMapCase) -> Dict[str, str]:
    """Return role-to-node mapping when the selected target has named roles."""

    return {
        str(role): str(node_id)
        for role, node_id in dict(case.selection.get("annotation_role_node_ids", {})).items()
    }
