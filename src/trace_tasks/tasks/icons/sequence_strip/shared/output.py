"""Trace payload helpers for sequence-strip icon scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from ...shared.icon_scene import serialize_rendered_icon_instance, single_panel_geometry_to_trace
from ...shared.icon_task_rendering import icon_render_style_trace
from .rendering import render_sequence_completion_scene_from_params
from .sampling import SequenceCompletionPlan


def sequence_render_spec(
    *,
    common_ids: Mapping[str, Any],
    panel_geometry: Mapping[str, Any],
    render_params: Mapping[str, Any],
    sampled_palette_rgb: Sequence[Sequence[int]],
    cell_box_width_px: int,
    cell_box_height_px: int,
    extra_style: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize shared render metadata for a horizontal sequence row."""

    style = {
        **icon_render_style_trace(
            render_params=render_params,
            sampled_palette_rgb=tuple(tuple(int(channel) for channel in color) for color in sampled_palette_rgb),
        ),
        "cell_padding_px": int(render_params["cell_padding_px"]),
        "cell_icon_padding_px": int(render_params["cell_icon_padding_px"]),
        "cell_corner_radius_px": int(render_params["cell_corner_radius_px"]),
        "cell_box_width_range_px": [
            int(render_params["cell_box_width_min_px"]),
            int(render_params["cell_box_width_max_px"]),
        ],
        "cell_box_height_range_px": [
            int(render_params["cell_box_height_min_px"]),
            int(render_params["cell_box_height_max_px"]),
        ],
        "sampled_cell_box_size_px": [
            int(cell_box_width_px),
            int(cell_box_height_px),
        ],
    }
    if extra_style:
        style.update({str(key): value for key, value in extra_style.items()})
    return {
        **dict(common_ids),
        "canvas_size": list(panel_geometry["canvas_size"]),
        "coord_space": "pixel",
        "panel_geometry": dict(panel_geometry),
        "style": style,
    }


def bbox_anchor_render_map(*, anchor_name: str, bbox_xyxy: Sequence[int]) -> dict[str, Any]:
    """Serialize a scalar bbox anchor for the rendered image."""

    return {
        "image_id": "img0",
        "anchors": {
            str(anchor_name): [int(value) for value in bbox_xyxy],
        },
    }


def build_completion_trace_payload(
    *,
    common_ids: Mapping[str, Any],
    scene_kind: str,
    question_format: str,
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    plan: SequenceCompletionPlan,
    rendered: "CompletionRenderArtifacts",
    annotation_payload: Any,
    render_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the common trace scaffold for sequence-completion tasks."""

    plan_relations = {
        "sequence_rule": str(plan.sequence_rule),
        "attribute_id": str(plan.attribute_id),
        "sequence_icon_id": str(plan.sequence_icon_id),
        "full_sequence_values": list(plan.full_sequence_values),
        "missing_index": int(plan.missing_index),
        "correct_option_label": str(plan.correct_option_label),
        "correct_option_value": int(plan.correct_option_value),
        "option_values_by_label": dict(plan.option_values_by_label),
    }
    trace_params = {
        "scene_id": str(common_ids["scene_id"]),
        "sequence_rule": str(plan.sequence_rule),
        "attribute_id": str(plan.attribute_id),
        "missing_index": int(plan.missing_index),
        "option_labels": ["A", "B", "C", "D"],
        "support_probabilities": dict(plan.support_probabilities),
        **dict(plan.extra_trace),
    }
    execution_trace = {
        **dict(common_ids),
        "scene_variant": "two_row_sequence_completion_options",
        "question_format": str(question_format),
        **plan_relations,
        "support_probabilities": dict(plan.support_probabilities),
        **dict(plan.extra_trace),
    }
    return {
        "taxonomy": {
            **dict(common_ids),
            "source_domain": str(common_ids["domain"]),
            "source_scene_id": str(common_ids["scene_id"]),
        },
        "scene_ir": {
            **dict(common_ids),
            "scene_kind": str(scene_kind),
            "entities": [
                *rendered.sequence_entities,
                *rendered.option_entities,
                *rendered.icon_entities,
            ],
            "relations": plan_relations,
            "frames": {
                "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                "panels": dict(rendered.panel_geometry),
            },
        },
        "query_spec": {
            **dict(common_ids),
            "template_id": str(prompt_defaults["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": trace_params,
        },
        "render_spec": sequence_render_spec(
            common_ids=common_ids,
            panel_geometry=rendered.panel_geometry,
            render_params=render_params,
            sampled_palette_rgb=plan.sampled_palette_rgb,
            cell_box_width_px=int(render_params["cell_box_width_px"]),
            cell_box_height_px=int(render_params["cell_box_height_px"]),
            extra_style={
                "row_gap_px": int(render_params["row_gap_px"]),
                "cell_label_font_size_px": int(render_params["cell_label_font_size_px"]),
                "cell_label_color_rgb": list(render_params["cell_label_color_rgb"]),
                "missing_mark_font_size_px": int(render_params["missing_mark_font_size_px"]),
                "missing_mark_color_rgb": list(render_params["missing_mark_color_rgb"]),
            },
        ),
        "render_map": bbox_anchor_render_map(
            anchor_name="correct_option_bbox",
            bbox_xyxy=annotation_payload.value,
        ),
        "execution_trace": execution_trace,
        "witness_symbolic": {
            "sequence_rule": str(plan.sequence_rule),
            "attribute_id": str(plan.attribute_id),
            "full_sequence_values": list(plan.full_sequence_values),
            "missing_index": int(plan.missing_index),
            "correct_option_label": str(plan.correct_option_label),
            "correct_option_value": int(plan.correct_option_value),
        },
        "projected_annotation": dict(annotation_payload.projected_annotation),
    }


@dataclass(frozen=True)
class CompletionRenderArtifacts:
    """Rendered image plus identity-free scene payload for a completion scene."""

    image: Any
    panel_geometry: dict[str, Any]
    sequence_entities: Tuple[dict[str, Any], ...]
    option_entities: Tuple[dict[str, Any], ...]
    icon_entities: Tuple[dict[str, Any], ...]
    correct_option_bbox: Tuple[int, int, int, int]


def render_completion_artifacts(
    *,
    plan: SequenceCompletionPlan,
    render_params: Mapping[str, Any],
) -> CompletionRenderArtifacts:
    """Render a completion plan and return annotation-ready visual witnesses."""

    rendered = render_sequence_completion_scene_from_params(
        sequence_cells=plan.sequence_cells,
        option_cells=plan.option_cells,
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        render_params=render_params,
    )
    option_entities = []
    sequence_entities = []
    icon_entities = []
    correct_option_bbox = None
    for cell in rendered.sequence_cells:
        sequence_entities.append(
            {
                "entity_kind": "sequence_cell",
                "panel": "scene",
                "row_id": "sequence",
                "cell_index": int(cell.cell_index),
                "cell_bbox_xyxy": [int(value) for value in cell.cell_bbox_xyxy],
                "is_missing": bool(cell.is_missing),
                "target_value": plan.full_sequence_values[int(cell.cell_index)],
                "rendered_icon_count": int(len(cell.icon_instances)),
            }
        )
        for instance in cell.icon_instances:
            icon_entities.append(
                serialize_rendered_icon_instance(
                    instance,
                    entity_kind="scene_icon",
                    extra_fields={
                        "row_id": "sequence",
                        "cell_index": int(cell.cell_index),
                        "cell_bbox_xyxy": [int(value) for value in cell.cell_bbox_xyxy],
                    },
                )
            )
    for cell in rendered.option_cells:
        label = str(cell.cell_label_text or ("A", "B", "C", "D")[int(cell.cell_index)])
        value = plan.option_values_by_label[str(label)]
        is_correct = str(label) == str(plan.correct_option_label)
        if is_correct:
            correct_option_bbox = tuple(int(value) for value in cell.cell_bbox_xyxy)
        option_entities.append(
            {
                "entity_kind": "option_cell",
                "panel": "scene",
                "row_id": "option",
                "cell_index": int(cell.cell_index),
                "option_label": str(label),
                "cell_bbox_xyxy": [int(value) for value in cell.cell_bbox_xyxy],
                "option_value": value,
                "is_correct": bool(is_correct),
                "rendered_icon_count": int(len(cell.icon_instances)),
            }
        )
        for instance in cell.icon_instances:
            icon_entities.append(
                serialize_rendered_icon_instance(
                    instance,
                    entity_kind="scene_icon",
                    extra_fields={
                        "row_id": "option",
                        "cell_index": int(cell.cell_index),
                        "option_label": str(label),
                        "cell_bbox_xyxy": [int(value) for value in cell.cell_bbox_xyxy],
                    },
                )
            )
    if correct_option_bbox is None:
        raise RuntimeError("completion render did not resolve a correct option bbox")
    return CompletionRenderArtifacts(
        image=rendered.image,
        panel_geometry=single_panel_geometry_to_trace(rendered.layout),
        sequence_entities=tuple(sequence_entities),
        option_entities=tuple(option_entities),
        icon_entities=tuple(icon_entities),
        correct_option_bbox=tuple(int(value) for value in correct_option_bbox),
    )


__all__ = [
    "CompletionRenderArtifacts",
    "bbox_anchor_render_map",
    "build_completion_trace_payload",
    "render_completion_artifacts",
    "sequence_render_spec",
]
