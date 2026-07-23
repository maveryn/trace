"""Rendering helpers for graph flow-network scenes."""

from __future__ import annotations

from typing import Any, Mapping

from .....core.visual.background import make_background_canvas
from .....core.visual.noise import apply_post_image_noise
from ...shared.graph_scene import GraphRenderParams, render_graph_scene
from .state import FlowNetworkRender, FlowNetworkSample


def render_flow_network_scene(
    *,
    flow_sample: FlowNetworkSample,
    render_params: GraphRenderParams,
    layout_variant: str,
    layout_transform_variant: str,
    capacity_label_font_size_px: int,
    capacity_label_offset_px: int,
    capacity_label_padding_px: int,
    instance_seed: int,
    attempt: int,
    params: Mapping[str, Any],
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
) -> FlowNetworkRender:
    """Render one capacitated source-sink network with highlighted endpoints."""

    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=params,
        default_config=background_defaults,
    )
    node_style_by_label = {
        "S": {
            "fill_rgb": (42, 157, 88),
            "border_rgb": (20, 100, 58),
            "label_text_rgb": (255, 255, 255),
            "label_stroke_rgb": (20, 100, 58),
            "halo_rgb": (20, 100, 58),
            "halo_width_px": 3,
            "halo_pad_px": 6,
        },
        "T": {
            "fill_rgb": (126, 87, 194),
            "border_rgb": (80, 48, 140),
            "label_text_rgb": (255, 255, 255),
            "label_stroke_rgb": (80, 48, 140),
            "halo_rgb": (80, 48, 140),
            "halo_width_px": 3,
            "halo_pad_px": 6,
        },
    }
    rendered_scene = render_graph_scene(
        graph_sample=flow_sample.graph_sample,
        layout_variant=str(layout_variant),
        layout_transform_variant=str(layout_transform_variant),
        render_params=render_params,
        layout_seed=int(instance_seed + attempt),
        scene_title="Capacity Network",
        directed=True,
        base_image=background,
        edge_weights_by_label=dict(flow_sample.capacity_by_edge_label),
        edge_weight_label_font_size_px=int(capacity_label_font_size_px),
        edge_weight_label_offset_px=int(capacity_label_offset_px),
        edge_weight_label_padding_px=int(capacity_label_padding_px),
        node_style_by_label=node_style_by_label,
        edge_style_by_label={},
        layout_fallback_variants=("layered",),
    )
    if any(edge.weight_label_bbox_xyxy is None for edge in rendered_scene.edges):
        raise ValueError("not all capacity labels were rendered")
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=noise_defaults,
    )
    return FlowNetworkRender(
        image=image,
        rendered_scene=rendered_scene,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


__all__ = ["render_flow_network_scene"]
