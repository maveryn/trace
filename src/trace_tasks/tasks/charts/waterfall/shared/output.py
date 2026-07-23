"""Trace payload assembly for waterfall chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.shared.visual_defaults import chart_font_asset_metadata
from trace_tasks.tasks.charts.waterfall.shared.defaults import SCENE_VARIANT
from trace_tasks.tasks.charts.waterfall.shared.state import WaterfallDataset, WaterfallRenderArtifacts


def step_rows(dataset: WaterfallDataset) -> list[dict[str, Any]]:
    """Return JSON-stable rows for sampled waterfall contributions."""

    return [
        {
            "step_id": str(step.step_id),
            "label": str(step.label),
            "delta": int(step.delta),
            "running_before": int(step.running_before),
            "running_after": int(step.running_after),
        }
        for step in dataset.steps
    ]


def build_trace_payload(
    *,
    artifacts: WaterfallRenderArtifacts,
    dataset: WaterfallDataset,
    answer_value: int | str,
    answer_type: str,
    question_format: str,
    relations: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the standard trace sections after task-owned binding."""

    rendered = artifacts.rendered_scene
    return {
        "scene_ir": {
            "scene_kind": "chart_waterfall",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "answer": answer_value,
                **dict(relations),
            },
        },
        "render_spec": {
            "scene_variant": SCENE_VARIANT,
            "canvas_width": int(artifacts.render_params.canvas_width),
            "canvas_height": int(artifacts.render_params.canvas_height),
            "coord_space": "pixel",
            "plot_bbox_px": list(rendered.plot_bbox_px),
            "background_style": dict(artifacts.background_style),
            "font_assets": chart_font_asset_metadata(str(artifacts.chart_font_family)),
            "y_axis_max": int(rendered.y_axis_max),
            "threshold_value": rendered.threshold_value,
            "layout_jitter": dict(artifacts.render_params.layout_jitter_meta),
            "post_image_noise": dict(artifacts.post_image_noise),
        },
        "render_map": {
            "image_id": "img0",
            "plot_bbox_px": list(rendered.plot_bbox_px),
            "bar_bboxes_px": dict(rendered.bar_bboxes_px),
            "value_label_bboxes_px": dict(rendered.value_label_bboxes_px),
            "x_label_bboxes_px": dict(rendered.x_label_bboxes_px),
            "connector_bboxes_px": dict(rendered.connector_bboxes_px),
            "extra_bboxes_px": dict(rendered.extra_bboxes_px),
        },
        "execution_trace": {
            "question_format": str(question_format),
            "answer": answer_value,
            "answer_type": str(answer_type),
            "start_value": int(dataset.start_value),
            "final_value": int(dataset.final_value),
            "step_count": int(len(dataset.steps)),
            "steps": step_rows(dataset),
            **dict(relations),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
        "background": dict(artifacts.background_style),
        "post_image_noise": dict(artifacts.post_image_noise),
    }


__all__ = ["build_trace_payload", "step_rows"]
