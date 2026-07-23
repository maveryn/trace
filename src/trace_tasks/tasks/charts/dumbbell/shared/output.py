"""Neutral output helpers for dumbbell chart scenes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.dumbbell.shared.defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    SCENE_ID,
    SCENE_LOAD_BY_VARIANT,
)
from trace_tasks.tasks.charts.dumbbell.shared.rendering import (
    render_dumbbell_chart,
    render_metadata,
    resolve_dumbbell_render_params,
)
from trace_tasks.tasks.charts.dumbbell.shared.state import DumbbellDataset, RenderedDumbbell


def render_dataset(
    dataset: DumbbellDataset,
    *,
    params: Mapping[str, Any],
    instance_seed: int,
) -> tuple[RenderedDumbbell, dict[str, Any], dict[str, Any]]:
    """Render one task-owned dumbbell dataset on the configured background."""

    render_params = resolve_dumbbell_render_params(params, instance_seed=int(instance_seed))
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=dict(params),
        scene_id=SCENE_ID,
        render_params=render_params,
        protected_colors=(render_params.series_a_rgb, render_params.series_b_rgb),
    )
    rendered = render_dumbbell_chart(
        background,
        dataset=dataset,
        render_params=render_params,
        instance_seed=int(instance_seed),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    rendered = replace(rendered, image=image)
    render_meta = render_metadata(render_params, rendered, dataset)
    render_meta["background_style"] = dict(background_meta)
    render_meta["information_scene_style"] = dict(information_style_meta)
    render_meta["post_image_noise"] = dict(post_noise_meta)
    return rendered, dict(render_meta), {
        "background": dict(background_meta),
        "information_scene_style": dict(information_style_meta),
        "post_image_noise": dict(post_noise_meta),
    }


def row_records(dataset: DumbbellDataset) -> list[dict[str, Any]]:
    """Return serializable row records for execution traces."""

    return [
        {
            "row_id": str(row.row_id),
            "label": str(row.label),
            "value_a": int(row.value_a),
            "value_b": int(row.value_b),
            "gap": int(row.gap),
            "signed_delta_a_minus_b": int(row.signed_delta_a_minus_b),
        }
        for row in dataset.rows
    ]


def build_trace_scaffold(
    *,
    dataset: DumbbellDataset,
    rendered: RenderedDumbbell,
    render_meta: Mapping[str, Any],
    sidecar_meta: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    annotation_refs: list[dict[str, Any]],
    answer_value: int | str,
    reasoning_load: float,
    question_format: str,
) -> dict[str, Any]:
    """Build task-neutral trace sections for a rendered dumbbell sample."""

    return {
        "scene_ir": {
            "scene_kind": "chart_dumbbell",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "scene_variant": str(dataset.scene_variant),
                "series_a_name": str(dataset.series_a_name),
                "series_b_name": str(dataset.series_b_name),
                "answer": answer_value,
                "annotation_row_ids": [str(row_id) for row_id in dataset.query.annotation_row_ids],
                "reasoning_load": float(reasoning_load),
                "scene_load": float(SCENE_LOAD_BY_VARIANT.get(str(dataset.scene_variant), 0.58)),
            },
        },
        "render_spec": dict(render_meta),
        "render_map": {
            "image_id": "img0",
            "plot_bbox_px": list(rendered.plot_bbox_px),
            "row_label_bboxes_px": dict(rendered.row_label_bboxes_px),
            "point_bboxes_px": dict(rendered.point_bboxes_px),
            "row_pair_bboxes_px": dict(rendered.row_pair_bboxes_px),
            "connector_bboxes_px": dict(rendered.connector_bboxes_px),
            "legend_bboxes_px": dict(rendered.legend_bboxes_px),
        },
        "execution_trace": {
            "scene_id": SCENE_ID,
            "scene_variant": str(dataset.scene_variant),
            "series_a_name": str(dataset.series_a_name),
            "series_b_name": str(dataset.series_b_name),
            "row_count": int(len(dataset.rows)),
            "row_labels": [str(row.label) for row in dataset.rows],
            "row_records": row_records(dataset),
            "rows": row_records(dataset),
            "query_params": dict(dataset.query.params),
            "annotation_row_ids": [str(row_id) for row_id in dataset.query.annotation_row_ids],
            "answer": answer_value,
            "question_format": str(question_format),
        },
        "witness_symbolic": {
            "type": "row_set",
            "value": [str(row_id) for row_id in dataset.query.annotation_row_ids],
        },
        "projected_annotation": dict(projected_annotation),
        "sidecar": dict(sidecar_meta),
        "annotation_refs": [dict(record) for record in annotation_refs],
    }


__all__ = [
    "build_trace_scaffold",
    "render_dataset",
    "row_records",
]
