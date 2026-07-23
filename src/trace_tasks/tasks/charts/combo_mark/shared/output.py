"""Trace payload helpers for combo-mark chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.combo_mark.shared.state import ComboRenderArtifacts
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, annotation_cardinality


def build_trace_scaffold(
    *,
    artifacts: ComboRenderArtifacts,
    annotation: AnnotationArtifacts,
    relations: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble neutral combo scene trace sections after task-owned binding."""

    scene = artifacts.scene
    p = artifacts.render_params
    return {
        "scene_ir": {
            "scene_kind": "chart_combo_panel",
            "entities": [dict(entity) for entity in scene.entities],
            "relations": dict(relations),
        },
        "render_spec": {
            "scene_id": "combo_mark",
            "scene_variant": str(scene.scene_variant),
            "plot_bbox": list(scene.plot_bbox),
            "primary_axis_max": int(scene.primary_axis_max),
            "line_axis_max": int(scene.line_axis_max),
            "layout_jitter": dict(p.layout_jitter_meta),
            "text_style": {
                "tick_font_size_px": int(p.tick_font_size),
                "label_font_size_px": int(p.label_font_size),
                "value_font_size_px": int(p.value_font_size),
                "legend_font_size_px": int(p.legend_font_size),
                "font_assets": dict(artifacts.font_assets),
            },
            "background": dict(artifacts.background_style),
            "information_scene_style": dict(artifacts.background_style["information_scene_style"]),
            "post_image_noise": dict(artifacts.post_image_noise),
        },
        "render_map": {
            "image_id": "img0",
            "plot_bbox_px": list(scene.plot_bbox),
            "legend_bbox_px": list(scene.legend_bbox),
            "context_protected_bboxes_px": {
                "plot": list(scene.plot_bbox),
                "legend": list(scene.legend_bbox),
            },
            "primary_points_px": [list(point) for point in scene.primary_points],
            "line_points_px": [list(point) for point in scene.line_points],
            "entities": [dict(entity) for entity in scene.entities],
        },
        "execution_trace": {
            "answer": relations.get("answer"),
            "answer_type": relations.get("answer_type"),
            **dict(relations),
        },
        "witness_symbolic": {
            "type": str(annotation.annotation_type),
            "count": int(
                annotation_cardinality(
                    annotation.value,
                    annotation_type=str(annotation.annotation_type),
                )
            ),
        },
        "projected_annotation": dict(annotation.projected_annotation),
        "background": dict(artifacts.background_style),
        "post_image_noise": dict(artifacts.post_image_noise),
    }


__all__ = ["build_trace_scaffold"]
