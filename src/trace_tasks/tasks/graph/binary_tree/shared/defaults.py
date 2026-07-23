"""Fallback defaults for graph binary-tree scene tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from .....core.seed import spawn_rng
from ...shared.graph_scene import SUPPORTED_NODE_SHAPE_VARIANTS
from ...shared.graph_sample_types import SUPPORTED_NODE_LINK_LABEL_VARIANTS
from ...shared.style import SUPPORTED_NODE_COLOR_NAMES
from ...shared.task_support import resolve_graph_named_variant


@dataclass(frozen=True)
class BinaryTreeDefaults:
    """Stable scene fallback defaults shared by binary-tree objectives."""

    node_count_min: int = 7
    node_count_max: int = 13
    target_count_min: int = 1
    target_count_max: int = 6
    internal_count_min: int = 3
    internal_count_max: int = 7
    depth_min: int = 1
    depth_max: int = 4
    traversal_position_min: int = 2
    traversal_position_max: int = 10
    max_depth: int = 4
    key_min: int = 10
    key_max: int = 99
    heap_violation_gap_min: int = 3
    heap_violation_gap_max: int = 3
    label_max_chars: int = 3
    canvas_width: int = 900
    canvas_height: int = 660
    outer_margin_px: int = 28
    panel_padding_px: int = 24
    panel_corner_radius_px: int = 20
    panel_title_font_size_px: int = 24
    node_shape_variant: str = "circle"
    node_radius_min_px: int = 20
    node_radius_max_px: int = 26
    edge_width_px: int = 4
    arrow_length_px: int = 12
    arrow_width_px: int = 7
    node_border_width_px: int = 2
    label_font_size_px: int = 20
    label_variant: str = "letters"
    layout_transform_variant: str = "identity"
    node_color_name: str = "blue"
    background_color_rgb: Tuple[int, int, int] = (247, 248, 251)
    panel_fill_rgb: Tuple[int, int, int] = (255, 255, 255)
    panel_border_rgb: Tuple[int, int, int] = (205, 212, 224)
    title_color_rgb: Tuple[int, int, int] = (70, 78, 96)
    edge_color_rgb: Tuple[int, int, int] = (118, 128, 145)
    node_fill_rgb: Tuple[int, int, int] = (92, 124, 250)
    node_border_rgb: Tuple[int, int, int] = (52, 73, 144)
    label_text_rgb: Tuple[int, int, int] = (255, 255, 255)
    label_stroke_rgb: Tuple[int, int, int] = (52, 73, 144)


@dataclass(frozen=True)
class BinaryTreeVisualAxes:
    """Resolved non-semantic scene/render axes for a binary-tree instance."""

    scene_variant: str
    label_variant: str
    node_shape_variant: str
    node_color_name: str
    scene_variant_probabilities: Dict[str, float]
    label_variant_probabilities: Dict[str, float]
    node_shape_variant_probabilities: Dict[str, float]
    node_color_name_probabilities: Dict[str, float]


def resolve_visual_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    sampling_namespace: str,
    include_label_variant: bool = True,
    fallback_label_variant: str = "numeric_keys",
) -> BinaryTreeVisualAxes:
    """Resolve balanced visual axes without knowing any public objective identity."""

    scene_rng = spawn_rng(int(instance_seed), f"{sampling_namespace}.scene_variant")
    scene_variant, scene_probs = resolve_graph_named_variant(
        scene_rng,
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=("classic_tree", "paper_tree", "boxed_tree"),
        instance_seed=int(instance_seed),
        task_id=str(sampling_namespace),
        namespace="scene_variant",
    )
    if bool(include_label_variant):
        label_rng = spawn_rng(int(instance_seed), f"{sampling_namespace}.label_variant")
        label_variant, label_probs = resolve_graph_named_variant(
            label_rng,
            params=params,
            gen_defaults=gen_defaults,
            explicit_key="label_variant",
            weights_key="label_variant_weights",
            balance_flag_key="balanced_label_variant_sampling",
            supported=SUPPORTED_NODE_LINK_LABEL_VARIANTS,
            instance_seed=int(instance_seed),
            task_id=str(sampling_namespace),
            namespace="label_variant",
        )
    else:
        label_variant = str(fallback_label_variant)
        label_probs = {str(label_variant): 1.0}
    shape_rng = spawn_rng(int(instance_seed), f"{sampling_namespace}.node_shape_variant")
    node_shape_variant, shape_probs = resolve_graph_named_variant(
        shape_rng,
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="node_shape_variant",
        weights_key="node_shape_variant_weights",
        balance_flag_key="balanced_node_shape_variant_sampling",
        supported=SUPPORTED_NODE_SHAPE_VARIANTS,
        instance_seed=int(instance_seed),
        task_id=str(sampling_namespace),
        namespace="node_shape_variant",
    )
    color_rng = spawn_rng(int(instance_seed), f"{sampling_namespace}.node_color_name")
    node_color_name, color_probs = resolve_graph_named_variant(
        color_rng,
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="node_color_name",
        weights_key="node_color_name_weights",
        balance_flag_key="balanced_node_color_name_sampling",
        supported=SUPPORTED_NODE_COLOR_NAMES,
        instance_seed=int(instance_seed),
        task_id=str(sampling_namespace),
        namespace="node_color_name",
    )
    return BinaryTreeVisualAxes(
        scene_variant=str(scene_variant),
        label_variant=str(label_variant),
        node_shape_variant=str(node_shape_variant),
        node_color_name=str(node_color_name),
        scene_variant_probabilities=dict(scene_probs),
        label_variant_probabilities=dict(label_probs),
        node_shape_variant_probabilities=dict(shape_probs),
        node_color_name_probabilities=dict(color_probs),
    )


__all__ = [
    "BinaryTreeDefaults",
    "BinaryTreeVisualAxes",
    "resolve_visual_axes",
]
