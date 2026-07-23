"""Trace-fragment helpers for graph binary-tree tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .state import BinaryTreeSample, RenderedBinaryTreeScene, SCENE_ID


def _label_key(prefix: str) -> str:
    return "".join((str(prefix), "_label"))


def node_entities(
    *,
    rendered_scene: RenderedBinaryTreeScene,
    sample: BinaryTreeSample,
    counted_labels: Sequence[str] = (),
    query_node_ids: Sequence[str] = (),
    answer_node_id: str | None = None,
    annotation_labels: Sequence[str] = (),
    annotation_roles_by_label: Mapping[str, Sequence[str]] | None = None,
    numeric_labels: bool = False,
    path_node_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Return rendered binary-tree node entities with optional task-local flags."""

    sample_by_label = {str(node.label): node for node in sample.nodes}
    counted = {str(label) for label in counted_labels}
    query_ids = {str(node_id) for node_id in query_node_ids}
    annotation_label_set = {str(label) for label in annotation_labels}
    path_ids = {str(node_id) for node_id in path_node_ids}
    roles_by_label = {
        str(label): [str(role) for role in roles]
        for label, roles in dict(annotation_roles_by_label or {}).items()
    }
    entities: list[dict[str, Any]] = []
    for node in rendered_scene.nodes:
        sample_node = sample_by_label[str(node.label)]
        child_count = int(node.left_label is not None) + int(node.right_label is not None)
        entity = {
            "entity_id": f"node_{node.label}",
            "entity_kind": "binary_tree_node",
            "label": str(node.label),
            "left_label": node.left_label,
            "right_label": node.right_label,
            "depth": int(node.depth),
            "child_count": int(child_count),
            "center_px": list(node.center_xy),
            "bbox_xyxy": list(node.bbox_xyxy),
            "is_counted": bool(str(node.label) in counted),
            "is_query_node": bool(str(sample_node.node_id) in query_ids),
            "is_query_path_node": bool(str(sample_node.node_id) in path_ids),
            "is_answer_node": bool(answer_node_id is not None and str(sample_node.node_id) == str(answer_node_id)),
            "is_annotation_node": bool(str(node.label) in annotation_label_set),
            "annotation_roles": list(roles_by_label.get(str(node.label), [])),
        }
        entity[_label_key("parent")] = node.parent_label
        if bool(numeric_labels):
            entity["numeric_key"] = int(node.label)
        entities.append(entity)
    return entities


def edge_entities(rendered_scene: RenderedBinaryTreeScene) -> list[dict[str, Any]]:
    """Return rendered binary-tree edge entities."""

    return [
        {
            "entity_id": str(edge.edge_id),
            "entity_kind": "binary_tree_edge",
            "child_label": str(edge.child_label),
            "child_side": str(edge.child_side),
            "segment_px": [list(edge.segment_px[0]), list(edge.segment_px[1])],
            "connector_path_px": [list(point) for point in edge.connector_path_px],
            "connector_style_variant": str(edge.connector_style_variant),
            _label_key("parent"): str(edge.parent_label),
        }
        for edge in rendered_scene.edges
    ]


def scene_frames(rendered_scene: RenderedBinaryTreeScene) -> dict[str, Any]:
    """Return scene frame metadata shared by binary-tree trace payloads."""

    return {
        "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
        "panels": dict(rendered_scene.panel_geometry),
    }


def render_style_spec(
    *,
    rendered_scene: RenderedBinaryTreeScene,
    render_params: Any,
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    extra_style: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return style metadata shared by binary-tree render specs."""

    style = {
        "scene_variant": str(rendered_scene.scene_variant),
        "connector_style_variant": str(rendered_scene.connector_style_variant),
        "theme_tone": str(render_params.theme_tone),
        "panel_style_variant": str(render_params.panel_style_variant),
        "background_color_rgb": list(render_params.background_color_rgb),
        "panel_fill_rgb": list(render_params.panel_fill_rgb),
        "panel_border_rgb": list(render_params.panel_border_rgb),
        "title_color_rgb": list(render_params.title_color_rgb),
        "edge_color_rgb": list(render_params.edge_color_rgb),
        "node_fill_rgb": list(render_params.node_fill_rgb),
        "node_border_rgb": list(render_params.node_border_rgb),
        "label_text_rgb": list(render_params.label_text_rgb),
        "label_stroke_rgb": list(render_params.label_stroke_rgb),
        "node_shape_variant": str(render_params.node_shape_variant),
        "node_radius_px": int(render_params.node_radius_px),
        "edge_width_px": int(render_params.edge_width_px),
        "node_border_width_px": int(render_params.node_border_width_px),
        "label_font_size_px": int(render_params.label_font_size_px),
        "resolved_label_font_size_px": int(rendered_scene.resolved_label_font_size_px),
        "label_stroke_width_px": int(rendered_scene.resolved_label_stroke_width_px),
        "background_meta": dict(background_meta),
        "post_image_noise_meta": dict(post_noise_meta),
    }
    style.update(dict(extra_style or {}))
    return style


def render_spec(
    *,
    rendered_scene: RenderedBinaryTreeScene,
    render_params: Any,
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    extra_style: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return render metadata shared by binary-tree trace payloads."""

    return {
        "canvas_size": list(rendered_scene.panel_geometry["canvas_size"]),
        "coord_space": "pixel",
        "panel_geometry": dict(rendered_scene.panel_geometry),
        "style": render_style_spec(
            rendered_scene=rendered_scene,
            render_params=render_params,
            background_meta=background_meta,
            post_noise_meta=post_noise_meta,
            extra_style=extra_style,
        ),
    }


def common_query_params(*, sample: BinaryTreeSample, visual_axes: Any) -> dict[str, Any]:
    """Return scene-level query params that are independent of objective identity."""

    return {
        "scene_variant": str(visual_axes.scene_variant),
        "scene_variant_probabilities": dict(visual_axes.scene_variant_probabilities),
        "node_count": int(sample.node_count),
        "max_depth": int(sample.max_depth),
        "label_variant": str(visual_axes.label_variant),
        "label_variant_probabilities": dict(visual_axes.label_variant_probabilities),
        "node_shape_variant": str(visual_axes.node_shape_variant),
        "node_shape_variant_probabilities": dict(visual_axes.node_shape_variant_probabilities),
        "node_color_name": str(visual_axes.node_color_name),
        "node_color_name_probabilities": dict(visual_axes.node_color_name_probabilities),
        "label_source_kind": str(sample.label_source_kind),
        "label_bucket": str(sample.label_bucket),
        "label_manifest": str(sample.label_manifest),
        "label_filter": dict(sample.label_filter),
        "label_bucket_probabilities": dict(sample.label_bucket_probabilities),
    }


def common_execution_fields(
    *,
    sample: BinaryTreeSample,
    rendered_scene: RenderedBinaryTreeScene,
    visual_axes: Any,
) -> dict[str, Any]:
    """Return reusable execution trace fields for any binary-tree objective."""

    return {
        "scene_variant": str(visual_axes.scene_variant),
        "connector_style_variant": str(rendered_scene.connector_style_variant),
        "node_count": int(sample.node_count),
        "max_depth": int(sample.max_depth),
        "label_variant": str(sample.label_variant),
    }


def rendered_trace_sections(
    *,
    rendered_scene: RenderedBinaryTreeScene,
    render_params: Any,
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    extra_style: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the common render sections embedded in binary-tree traces."""

    return {
        "render_spec": render_spec(
            rendered_scene=rendered_scene,
            render_params=render_params,
            background_meta=background_meta,
            post_noise_meta=post_noise_meta,
            extra_style=extra_style,
        ),
        "render_map": {"image_id": "img0", "anchors": {}},
    }


def scene_ir(
    *,
    scene_kind: str,
    rendered_scene: RenderedBinaryTreeScene,
    sample: BinaryTreeSample,
    entities: Sequence[Mapping[str, Any]],
    relations: Mapping[str, Any],
) -> dict[str, Any]:
    """Return scene IR without public task/query identity."""

    node_by_id = {str(node.node_id): node for node in sample.nodes}
    base_relations = {
        "root_label": str(node_by_id[""].label),
        "preorder_labels": list(sample.preorder_labels),
        "inorder_labels": list(sample.inorder_labels),
        "postorder_labels": list(sample.postorder_labels),
        "level_order_labels": list(sample.level_order_labels),
    }
    base_relations.update(dict(relations))
    return {
        "scene_id": SCENE_ID,
        "scene_kind": str(scene_kind),
        "entities": [dict(entity) for entity in entities],
        "relations": base_relations,
        "frames": scene_frames(rendered_scene),
    }


__all__ = [
    "edge_entities",
    "common_execution_fields",
    "common_query_params",
    "node_entities",
    "render_spec",
    "render_style_spec",
    "rendered_trace_sections",
    "scene_frames",
    "scene_ir",
]
