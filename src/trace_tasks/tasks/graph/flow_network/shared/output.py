"""Neutral output primitives for flow-network graph scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default
from ...shared.graph_sample_types import graph_label_sort_key
from ...shared.task_support import resolve_graph_render_params
from .annotations import project_min_cut_segments
from .rendering import render_flow_network_scene
from .sampling import sample_flow_network
from .state import (
    FlowNetworkAxes,
    FlowNetworkDefaults,
    FlowNetworkSceneBundle,
    FlowNetworkSample,
    ResolvedFlowNetworkAxes,
    SCENE_ID,
)


def capacity_entries(flow_sample: FlowNetworkSample) -> list[Dict[str, Any]]:
    """Return capacity labels in deterministic trace order."""

    return [
        {"edge": [str(left), str(right)], "capacity": int(capacity)}
        for (left, right), capacity in sorted(
            flow_sample.capacity_by_edge_label.items(),
            key=lambda item: (graph_label_sort_key(item[0][0]), graph_label_sort_key(item[0][1])),
        )
    ]


def create_flow_network_scene_bundle(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    axes: ResolvedFlowNetworkAxes,
    namespace: str,
    max_attempts: int,
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    defaults: FlowNetworkDefaults,
) -> FlowNetworkSceneBundle:
    """Sample, render, and project one neutral capacitated network scene."""

    render_params = resolve_graph_render_params(
        params,
        instance_seed=int(instance_seed),
        task_id=str(namespace),
        render_defaults=render_defaults,
        fallback_defaults=defaults,
        node_color_name=str(axes.node_color_name),
        node_shape_variant="circle",
        edge_routing_variant=str(axes.edge_routing_variant),
    )
    capacity_label_font_size_px = int(
        params.get(
            "capacity_label_font_size_px",
            group_default(
                render_defaults,
                "capacity_label_font_size_px",
                int(defaults.capacity_label_font_size_px),
            ),
        )
    )
    capacity_label_offset_px = int(
        params.get(
            "capacity_label_offset_px",
            group_default(
                render_defaults,
                "capacity_label_offset_px",
                int(defaults.capacity_label_offset_px),
            ),
        )
    )
    capacity_label_padding_px = int(
        params.get(
            "capacity_label_padding_px",
            group_default(
                render_defaults,
                "capacity_label_padding_px",
                int(defaults.capacity_label_padding_px),
            ),
        )
    )
    sample_axes = FlowNetworkAxes(
        node_count=int(axes.node_count),
        target_cut_edge_count=int(axes.target_cut_edge_count),
        target_flow_value=int(axes.target_flow_value),
        distractor_edge_count=int(axes.distractor_edge_count),
    )
    graph_rng = spawn_rng(int(instance_seed), f"{namespace}.graph")
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            flow_sample = sample_flow_network(graph_rng, axes=sample_axes, defaults=defaults)
            render = render_flow_network_scene(
                flow_sample=flow_sample,
                render_params=render_params,
                layout_variant=str(axes.layout_variant),
                layout_transform_variant=str(axes.layout_transform_variant),
                capacity_label_font_size_px=int(capacity_label_font_size_px),
                capacity_label_offset_px=int(capacity_label_offset_px),
                capacity_label_padding_px=int(capacity_label_padding_px),
                instance_seed=int(instance_seed),
                attempt=int(attempt),
                params=params,
                background_defaults=background_defaults,
                noise_defaults=noise_defaults,
            )
            if int(render.rendered_scene.crossing_count) > int(axes.max_crossing_count):
                raise ValueError("flow-network render exceeds crossing-count limit")
            projection, segments = project_min_cut_segments(
                render.rendered_scene,
                flow_sample.original_min_cut_edges,
            )
            return FlowNetworkSceneBundle(
                axes=axes,
                render_params=render_params,
                flow_sample=flow_sample,
                render=render,
                annotation_edges=tuple(flow_sample.original_min_cut_edges),
                annotation_projection=dict(projection),
                annotation_segments=tuple(segments),
                capacity_label_font_size_px=int(capacity_label_font_size_px),
                capacity_label_offset_px=int(capacity_label_offset_px),
                capacity_label_padding_px=int(capacity_label_padding_px),
            )
        except Exception as exc:  # pragma: no cover - retry path depends on sampled geometry
            last_error = exc
            continue
    raise RuntimeError("failed to generate flow-network scene") from last_error


def node_entities(bundle: FlowNetworkSceneBundle) -> list[Dict[str, Any]]:
    """Serialize rendered flow-network nodes."""

    return [
        {
            "entity_id": f"node_{node.label}",
            "entity_kind": "graph_node",
            "label": str(node.label),
            "role": "source" if str(node.label) == "S" else ("sink" if str(node.label) == "T" else "intermediate"),
            "degree": int(node.degree),
            "neighbors": list(node.neighbors),
            "successors": list(node.successors),
            "predecessors": list(node.predecessors),
            "center_px": list(node.center_xy),
            "bbox_xyxy": list(node.bbox_xyxy),
        }
        for node in bundle.render.rendered_scene.nodes
    ]


def edge_entities(bundle: FlowNetworkSceneBundle) -> list[Dict[str, Any]]:
    """Serialize rendered flow-network edges and capacity-label boxes."""

    annotation_edge_set = set(tuple(edge) for edge in bundle.annotation_edges)
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
            "capacity": int(edge.weight) if edge.weight is not None else None,
            "capacity_label_bbox_xyxy": list(edge.weight_label_bbox_xyxy) if edge.weight_label_bbox_xyxy is not None else None,
            "is_minimum_cut_edge": bool((str(edge.node_u_label), str(edge.node_v_label)) in annotation_edge_set),
        }
        for edge in bundle.render.rendered_scene.edges
    ]


def scene_relations(bundle: FlowNetworkSceneBundle) -> Dict[str, Any]:
    """Return task-neutral semantic relations for one rendered capacity graph."""

    flow_sample = bundle.flow_sample
    return {
        "graph_directionality": "directed",
        "source_label": "S",
        "sink_label": "T",
        "capacity_by_edge": list(capacity_entries(flow_sample)),
        "original_max_flow_value": int(flow_sample.original_max_flow_value),
        "minimum_cut_edges": [list(edge) for edge in flow_sample.original_min_cut_edges],
    }


def scene_ir(bundle: FlowNetworkSceneBundle) -> Dict[str, Any]:
    """Return scene-level IR without public task metadata."""

    return {
        "scene_id": SCENE_ID,
        "scene_kind": "graph_capacity_flow_network",
        "entities": [*node_entities(bundle), *edge_entities(bundle)],
        "relations": scene_relations(bundle),
        "frames": {
            "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
            "panels": dict(bundle.render.rendered_scene.panel_geometry),
        },
    }


def render_spec(bundle: FlowNetworkSceneBundle) -> Dict[str, Any]:
    """Return non-semantic rendering metadata for one flow-network scene."""

    render_params = bundle.render_params
    rendered = bundle.render.rendered_scene
    return {
        "canvas_size": list(rendered.panel_geometry["canvas_size"]),
        "coord_space": "pixel",
        "panel_geometry": dict(rendered.panel_geometry),
        "style": {
            "node_color_name": str(bundle.axes.node_color_name),
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
            "node_shape_variant": "circle",
            "node_radius_px": int(render_params.node_radius_px),
            "edge_width_px": int(render_params.edge_width_px),
            "edge_routing_variant": str(rendered.edge_routing_variant),
            "arrow_length_px": int(render_params.arrow_length_px),
            "arrow_width_px": int(render_params.arrow_width_px),
            "node_border_width_px": int(render_params.node_border_width_px),
            "label_font_size_px": int(render_params.label_font_size_px),
            "resolved_label_font_size_px": int(rendered.resolved_label_font_size_px),
            "label_stroke_width_px": int(rendered.resolved_label_stroke_width_px),
            "capacity_label_font_size_px": int(bundle.capacity_label_font_size_px),
            "capacity_label_offset_px": int(bundle.capacity_label_offset_px),
            "capacity_label_padding_px": int(bundle.capacity_label_padding_px),
            "font_family": str(render_params.font_family or ""),
            "font_asset": dict(render_params.font_asset) if isinstance(render_params.font_asset, Mapping) else {},
            "font_asset_version": str(render_params.font_asset_version or ""),
            "font_exclusion_reason": str(render_params.font_exclusion_reason),
            "context_text_elements": list(rendered.panel_geometry.get("context_text_elements", [])),
            "background_meta": dict(bundle.render.background_meta),
            "post_image_noise_meta": dict(bundle.render.post_noise_meta),
        },
    }


def axis_parameter_fields(bundle: FlowNetworkSceneBundle) -> Dict[str, Any]:
    """Return task-neutral sampled-axis fields for trace metadata."""

    axes = bundle.axes
    return {
        "node_count": int(axes.node_count),
        "node_count_probabilities": dict(axes.node_count_probabilities),
        "target_cut_edge_count": int(axes.target_cut_edge_count),
        "target_cut_edge_count_probabilities": dict(axes.target_cut_edge_count_probabilities),
        "target_flow_value": int(axes.target_flow_value),
        "target_flow_value_probabilities": dict(axes.target_flow_value_probabilities),
        "distractor_edge_count": int(axes.distractor_edge_count),
        "distractor_edge_count_probabilities": dict(axes.distractor_edge_count_probabilities),
        "layout_variant": str(axes.layout_variant),
        "layout_variant_probabilities": dict(axes.layout_variant_probabilities),
        "layout_transform_variant": str(axes.layout_transform_variant),
        "layout_transform_variant_probabilities": dict(axes.layout_transform_variant_probabilities),
        "edge_routing_variant": str(axes.edge_routing_variant),
        "edge_routing_variant_probabilities": dict(axes.edge_routing_variant_probabilities),
        "node_color_name": str(axes.node_color_name),
        "node_color_name_probabilities": dict(axes.node_color_name_probabilities),
        "max_crossing_count": int(axes.max_crossing_count),
    }


def execution_fields(bundle: FlowNetworkSceneBundle) -> Dict[str, Any]:
    """Return common execution facts for flow-network tasks."""

    flow_sample = bundle.flow_sample
    rendered = bundle.render.rendered_scene
    return {
        "graph_directionality": "directed",
        "node_count": int(bundle.axes.node_count),
        "edge_count": int(flow_sample.graph_sample.edge_count),
        "source_label": "S",
        "sink_label": "T",
        "original_max_flow_value": int(flow_sample.original_max_flow_value),
        "minimum_cut_edge_count": int(len(flow_sample.original_min_cut_edges)),
        "minimum_cut_edges": [list(edge) for edge in flow_sample.original_min_cut_edges],
        "original_min_cut_edges": [list(edge) for edge in flow_sample.original_min_cut_edges],
        "minimum_cut_partition": [list(flow_sample.original_min_cut_partition[0]), list(flow_sample.original_min_cut_partition[1])],
        "capacity_by_edge": list(capacity_entries(flow_sample)),
        "successors_by_label": {str(key): list(values) for key, values in flow_sample.graph_sample.successors_by_label.items()},
        "predecessors_by_label": {str(key): list(values) for key, values in flow_sample.graph_sample.predecessors_by_label.items()},
        "layout_variant_requested": str(bundle.axes.layout_variant),
        "layout_variant_used": str(rendered.layout_variant),
        "layout_transform_variant": str(rendered.layout_transform_variant),
        "edge_routing_variant": str(rendered.edge_routing_variant),
        "node_color_name": str(bundle.axes.node_color_name),
        "crossing_count": int(rendered.crossing_count),
    }


def symbolic_witness(bundle: FlowNetworkSceneBundle) -> Dict[str, Any]:
    """Return symbolic witness metadata for the annotated cut edges."""

    return {
        "type": "directed_edge_segment_set",
        "edges": [list(edge) for edge in bundle.annotation_edges],
    }


def projected_annotation(bundle: FlowNetworkSceneBundle) -> Dict[str, Any]:
    """Return projected public annotation metadata."""

    return {
        "type": "segment_set",
        **dict(bundle.annotation_projection),
    }


def trace_sections(bundle: FlowNetworkSceneBundle) -> Dict[str, Any]:
    """Return reusable trace sections before public task metadata is attached."""

    return {
        "scene_ir": scene_ir(bundle),
        "render_spec": render_spec(bundle),
        "render_map": {"image_id": "img0", "anchors": {}},
        "execution_trace": execution_fields(bundle),
        "witness_symbolic": symbolic_witness(bundle),
        "projected_annotation": projected_annotation(bundle),
        "axis_parameter_fields": axis_parameter_fields(bundle),
    }


__all__ = [
    "axis_parameter_fields",
    "capacity_entries",
    "create_flow_network_scene_bundle",
    "edge_entities",
    "execution_fields",
    "node_entities",
    "projected_annotation",
    "render_spec",
    "scene_ir",
    "scene_relations",
    "symbolic_witness",
    "trace_sections",
]
