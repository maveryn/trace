"""Rendering orchestration for the graph node-link scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.core.visual.background import make_background_canvas
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.named_colors import darken_color, named_color
from trace_tasks.tasks.graph.shared.graph_scene import render_graph_scene
from trace_tasks.tasks.graph.shared.visual_defaults import load_graph_scene_background_defaults, load_graph_scene_noise_defaults

from .state import SCENE_ID


@dataclass(frozen=True)
class NodeLinkRenderedSample:
    """Rendered node-link sample plus non-semantic rendering metadata."""

    rendered_scene: Any
    image: Any
    background_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]


def _semantic_node_styles(sample: Any) -> dict[str, dict[str, Any]]:
    """Return visible node styles for semantic node-color graph samples."""

    color_names = getattr(sample, "node_color_names_by_label", None)
    if not isinstance(color_names, Mapping):
        return {}
    styles: dict[str, dict[str, Any]] = {}
    for label, color_name in color_names.items():
        try:
            fill_rgb = tuple(int(value) for value in named_color(str(color_name)))
        except Exception:
            continue
        styles[str(label)] = {
            "color_name": str(color_name),
            "fill_rgb": fill_rgb,
            "border_rgb": tuple(int(value) for value in darken_color(fill_rgb, factor=0.55)),
        }
    return styles


def _semantic_edge_styles(sample: Any) -> dict[tuple[str, str], dict[str, Any]]:
    """Return visible edge styles for semantic edge-color graph samples."""

    color_names = getattr(sample, "edge_color_names_by_label", None)
    if not isinstance(color_names, Mapping):
        return {}
    styles: dict[tuple[str, str], dict[str, Any]] = {}
    for edge, color_name in color_names.items():
        try:
            left, right = edge
            rgb = tuple(int(value) for value in named_color(str(color_name)))
        except Exception:
            continue
        styles[(str(left), str(right))] = {
            "color_name": str(color_name),
            "edge_color_rgb": tuple(int(value) for value in darken_color(rgb, factor=0.82)),
        }
    return styles


def _semantic_edge_text_labels(sample: Any) -> dict[tuple[str, str], str]:
    """Return visible text labels for edge-label graph samples."""

    labels = getattr(sample, "edge_attribute_labels_by_label", None)
    if not isinstance(labels, Mapping):
        return {}
    visible_labels: dict[tuple[str, str], str] = {}
    for edge, label in labels.items():
        try:
            left, right = edge
        except Exception:
            continue
        visible_labels[(str(left), str(right))] = str(label)
    return visible_labels


def _semantic_edge_weights(sample: Any) -> dict[tuple[str, str], int]:
    """Return visible numeric edge weights for weighted graph samples."""

    weights = getattr(sample, "edge_weights_by_label", None)
    if not isinstance(weights, Mapping):
        return {}
    visible_weights: dict[tuple[str, str], int] = {}
    for edge, weight in weights.items():
        try:
            left, right = edge
            visible_weights[(str(left), str(right))] = int(weight)
        except Exception:
            continue
    return visible_weights


def render_node_link_sample(
    *,
    sample: Any,
    layout_variant: str,
    layout_transform_variant: str,
    render_params: Any,
    layout_seed: int,
    directed: bool,
    params: Mapping[str, Any],
    instance_seed: int,
    scene_id: str = SCENE_ID,
    strict_edge_label_placement: bool = False,
    edge_text_label_font_size_px: int | None = None,
) -> NodeLinkRenderedSample:
    """Render one graph sample and apply scene-local background/noise policy."""

    background_defaults = load_graph_scene_background_defaults(scene_id=str(scene_id))
    noise_defaults = load_graph_scene_noise_defaults(scene_id=str(scene_id), apply_prob=0.5)
    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=params,
        default_config=background_defaults,
    )
    rendered_scene = render_graph_scene(
        graph_sample=sample,
        layout_variant=str(layout_variant),
        layout_transform_variant=str(layout_transform_variant),
        render_params=render_params,
        layout_seed=int(layout_seed),
        scene_title="Graph",
        directed=bool(directed),
        base_image=background,
        node_style_by_label=_semantic_node_styles(sample),
        edge_style_by_label=_semantic_edge_styles(sample),
        edge_weights_by_label=_semantic_edge_weights(sample),
        edge_text_labels_by_label=_semantic_edge_text_labels(sample),
        edge_text_label_font_size_px=edge_text_label_font_size_px,
        edge_text_label_strict_placement=bool(strict_edge_label_placement),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=noise_defaults,
    )
    return NodeLinkRenderedSample(
        rendered_scene=rendered_scene,
        image=image,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


__all__ = ["NodeLinkRenderedSample", "render_node_link_sample"]
