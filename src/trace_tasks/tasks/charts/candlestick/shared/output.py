"""Trace-payload scaffolding for candlestick chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.candlestick.shared.state import Dataset, RenderArtifacts


def candle_rows(dataset: Dataset) -> list[dict[str, Any]]:
    return [
        {
            "candle_id": str(candle.candle_id),
            "label": str(candle.label),
            "open": int(candle.open_value),
            "high": int(candle.high_value),
            "low": int(candle.low_value),
            "close": int(candle.close_value),
            "direction": str(candle.direction),
            "body_size": int(candle.body_size),
            "wick_range": int(candle.wick_range),
        }
        for candle in dataset.candles
    ]


def build_trace_scaffold(
    *,
    dataset: Dataset,
    artifacts: RenderArtifacts,
    relations: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the common trace payload after a public task has bound its answer."""

    rendered = artifacts.rendered
    render_params = artifacts.render_params
    return {
        "scene_ir": {
            "scene_kind": "chart_candlestick_ohlc",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": dict(relations),
        },
        "render_spec": {
            "scene_variant": "candlestick",
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "plot_bbox_px": list(rendered.plot_bbox_px),
            "y_axis_min": int(render_params.y_axis_min),
            "y_axis_max": int(render_params.y_axis_max),
            "layout_jitter": dict(render_params.layout_jitter_meta),
            "background_style": dict(artifacts.background_style),
            "information_scene_style": dict(artifacts.background_style["information_scene_style"]),
            "font_assets": dict(artifacts.font_assets),
            "post_image_noise": dict(artifacts.post_image_noise),
        },
        "render_map": {
            "image_id": "img0",
            "plot_bbox_px": list(rendered.plot_bbox_px),
            "candle_bboxes_px": dict(rendered.candle_bboxes_px),
            "body_bboxes_px": dict(rendered.body_bboxes_px),
            "wick_bboxes_px": dict(rendered.wick_bboxes_px),
            "value_label_bboxes_px": dict(rendered.value_label_bboxes_px),
            "x_label_bboxes_px": dict(rendered.x_label_bboxes_px),
        },
        "execution_trace": {
            "question_format": "candlestick_ohlc_query",
            "answer": dataset.selection.answer,
            "answer_type": str(dataset.selection.answer_type),
            "candle_count": int(len(dataset.candles)),
            "candles": candle_rows(dataset),
            "annotation_candle_ids": list(dataset.selection.annotation_candle_ids),
            "support_label_ids": list(dataset.selection.annotation_label_ids),
            "annotation_roles": list(dataset.selection.annotation_roles),
            **dict(relations),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
    }


__all__ = ["build_trace_scaffold", "candle_rows"]
