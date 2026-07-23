"""Neutral trace-section helpers for graph option-panel scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from ....shared.prompt_variants import PromptTraceArtifacts
from .sampling import canonical_spec
from .state import GraphOptionsDataset, GraphOptionsRenderedScene, SCENE_ID


def scene_ir(dataset: GraphOptionsDataset, rendered: GraphOptionsRenderedScene) -> Dict[str, Any]:
    """Return scene-level IR without public task metadata."""

    return {
        "scene_id": SCENE_ID,
        "scene_kind": "graph_structure_options",
        "entities": [dict(entity) for entity in rendered.entities],
        "relations": {
            "edge_mode": str(dataset.edge_mode),
            "answer_option_label": str(dataset.answer_option_label),
            "correct_option_panel_id": str(dataset.correct_option_panel_id),
            "source_structure_signature": canonical_spec(dataset.source_structure_spec),
            "answer_structure_signature": canonical_spec(dataset.answer_structure_spec),
        },
    }


def axis_parameter_fields(
    *,
    dataset: GraphOptionsDataset,
    edge_mode_probabilities: Mapping[str, float],
    scene_variant_probabilities: Mapping[str, float],
) -> Dict[str, Any]:
    """Return sampled generation axes for trace metadata."""

    return {
        "edge_mode": str(dataset.edge_mode),
        "edge_mode_probabilities": dict(edge_mode_probabilities),
        "scene_variant": str(dataset.solver_trace.get("scene_variant", "")),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "option_count": int(dataset.option_count),
        "node_count": int(dataset.node_count),
        "node_count_range": list(dataset.node_count_range),
        "node_count_probabilities": dict(dataset.node_count_probabilities),
        "pattern_node_count": dataset.pattern_node_count,
        "pattern_node_count_range": list(dataset.pattern_node_count_range or []),
        "pattern_node_count_probability_map": dict(dataset.pattern_node_count_probability_map or {}),
        "subgraph_node_count": dataset.subgraph_node_count,
        "subgraph_node_count_range": list(dataset.subgraph_node_count_range or []),
        "subgraph_node_count_probability_map": dict(dataset.subgraph_node_count_probability_map or {}),
    }


def render_spec(rendered: GraphOptionsRenderedScene) -> Dict[str, Any]:
    """Return non-semantic rendering metadata."""

    render_params = rendered.render_params
    return {
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "coord_space": "pixel",
        "scene_id": SCENE_ID,
        "scene_variant": str(rendered.scene_variant),
        "background_style": dict(rendered.background_meta),
        "scene_style": dict(rendered.scene_style_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
        "scene_bbox_px": list(rendered.scene_bbox_px),
        "style": {
            "panel_fill_rgb": list(render_params.panel_fill_rgb),
            "option_fill_rgb": list(render_params.option_fill_rgb),
            "border_rgb": list(render_params.border_rgb),
            "edge_rgb": list(render_params.edge_rgb),
            "node_fill_rgb": list(render_params.node_fill_rgb),
            "node_outline_rgb": list(render_params.node_outline_rgb),
            "text_rgb": list(render_params.text_rgb),
            "text_stroke_rgb": list(render_params.text_stroke_rgb),
            "node_radius_px": int(render_params.node_radius_px),
            "edge_width_px": int(render_params.edge_width_px),
        },
        "font_family": str(render_params.font_family or ""),
        "font_asset": dict(render_params.font_asset) if isinstance(render_params.font_asset, Mapping) else {},
        "font_asset_version": str(render_params.font_asset_version),
        "font_exclusion_reason": str(render_params.font_exclusion_reason),
    }


def render_map(rendered: GraphOptionsRenderedScene) -> Dict[str, Any]:
    """Return pixel-space geometry maps for review and verification."""

    return {
        "image_id": "img0",
        "scene_bbox_px": list(rendered.scene_bbox_px),
        "item_bboxes_px": {str(key): list(value) for key, value in rendered.bbox_map.items()},
        "option_panel_bboxes_px": {
            str(key): list(value) for key, value in rendered.option_panel_bbox_map.items()
        },
    }


def execution_fields(
    *,
    dataset: GraphOptionsDataset,
    edge_mode_probabilities: Mapping[str, float],
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
) -> Dict[str, Any]:
    """Return task-neutral execution fields."""

    return {
        "edge_mode": str(dataset.edge_mode),
        "edge_mode_probabilities": dict(edge_mode_probabilities),
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "panel_title": str(dataset.panel_title),
        "source_structure_spec": dict(dataset.source_structure_spec),
        "answer_structure_spec": dict(dataset.answer_structure_spec),
        "answer_option_label": str(dataset.answer_option_label),
        "correct_option_index": int(dataset.correct_option_index),
        "correct_option_panel_id": str(dataset.correct_option_panel_id),
        "option_count": int(dataset.option_count),
        "option_specs": [dict(option) for option in dataset.option_specs],
        "node_count": int(dataset.node_count),
        "node_count_range": list(dataset.node_count_range),
        "node_count_probabilities": dict(dataset.node_count_probabilities),
        "pattern_node_count": dataset.pattern_node_count,
        "pattern_node_count_range": list(dataset.pattern_node_count_range or []),
        "pattern_node_count_probability_map": dict(dataset.pattern_node_count_probability_map or {}),
        "subgraph_node_count": dataset.subgraph_node_count,
        "subgraph_node_count_range": list(dataset.subgraph_node_count_range or []),
        "subgraph_node_count_probability_map": dict(dataset.subgraph_node_count_probability_map or {}),
        "solver_trace": dict(dataset.solver_trace),
        "supporting_option_panel_ids": [str(dataset.correct_option_panel_id)],
    }


def trace_sections(
    *,
    dataset: GraphOptionsDataset,
    rendered: GraphOptionsRenderedScene,
    prompt_artifacts: PromptTraceArtifacts,
    prompt_bundle_id: str,
    prompt_key: str,
    edge_mode_probabilities: Mapping[str, float],
    scene_variant_probabilities: Mapping[str, float],
    annotation_projection: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return neutral trace sections before public task metadata is attached."""

    axis_fields = axis_parameter_fields(
        dataset=dataset,
        edge_mode_probabilities=edge_mode_probabilities,
        scene_variant_probabilities=scene_variant_probabilities,
    )
    axis_fields["scene_variant"] = str(rendered.scene_variant)
    return {
        "scene_ir": scene_ir(dataset, rendered),
        "prompt_fields": {
            "template_id": str(prompt_bundle_id),
            "prompt_key": str(prompt_key),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
        },
        "axis_parameter_fields": axis_fields,
        "render_spec": render_spec(rendered),
        "render_map": render_map(rendered),
        "execution_trace": execution_fields(
            dataset=dataset,
            edge_mode_probabilities=edge_mode_probabilities,
            scene_variant=str(rendered.scene_variant),
            scene_variant_probabilities=scene_variant_probabilities,
        ),
        "witness_symbolic": {
            "type": "option_panel_bbox",
            "correct_option_panel_id": str(dataset.correct_option_panel_id),
        },
        "projected_annotation": dict(annotation_projection),
    }


__all__ = [
    "axis_parameter_fields",
    "execution_fields",
    "render_map",
    "render_spec",
    "scene_ir",
    "trace_sections",
]
