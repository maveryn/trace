"""Neutral render and trace assembly helpers for error-interval charts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Mapping

from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.error_interval.shared.defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    SCENE_ID,
)
from trace_tasks.tasks.charts.error_interval.shared.rendering import (
    render_error_interval_chart,
    resolve_error_interval_render_params,
)
from trace_tasks.tasks.charts.error_interval.shared.state import _Dataset, _Rendered
from trace_tasks.tasks.shared.font_assets import font_asset_version


def render_dataset(
    dataset: _Dataset,
    *,
    params: Mapping[str, Any],
    instance_seed: int,
) -> tuple[_Rendered, Dict[str, Any], Dict[str, Any]]:
    """Render one sampled dataset and return render/sidecar metadata."""

    render_params = resolve_error_interval_render_params(params, instance_seed=int(instance_seed))
    protected_colors = tuple(tuple(int(channel) for channel in item.color_rgb) for item in dataset.items)
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=dict(params),
        scene_id=SCENE_ID,
        render_params=render_params,
        protected_colors=protected_colors,
    )
    rendered = render_error_interval_chart(
        background,
        dataset=dataset,
        params=params,
        instance_seed=int(instance_seed),
        render_params=render_params,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    rendered = replace(rendered, image=image)
    render_meta = {
        "scene_variant": str(dataset.scene_variant),
        "canvas_width": int(image.size[0]),
        "canvas_height": int(image.size[1]),
        "coord_space": "pixel",
        "category_labels": [str(item.label) for item in dataset.items],
        "render_meta": dict(rendered.render_meta),
        "layout_jitter": dict(rendered.render_meta.get("layout_jitter", {})),
        "font_assets": dict(rendered.render_meta.get("font_assets", {}))
        or {"asset_version": str(font_asset_version())},
        "background_style": dict(background_meta),
        "information_scene_style": dict(information_style_meta),
        "post_image_noise": dict(post_noise_meta),
    }
    return rendered, dict(render_meta), {
        "background": dict(background_meta),
        "information_scene_style": dict(information_style_meta),
        "post_image_noise": dict(post_noise_meta),
    }


def interval_records(dataset: _Dataset) -> list[dict[str, Any]]:
    """Return symbolic interval rows for trace/audit payloads."""

    return [
        {
            "item_id": str(item.item_id),
            "label": str(item.label),
            "lower": int(item.lower),
            "midpoint": int(item.midpoint),
            "upper": int(item.upper),
            "interval_width": int(item.upper) - int(item.lower),
        }
        for item in dataset.items
    ]


def build_trace_scaffold(
    *,
    dataset: _Dataset,
    rendered: _Rendered,
    render_meta: Mapping[str, Any],
    sidecar_meta: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    annotation_refs: list[dict[str, Any]],
    answer_value: int | str,
) -> Dict[str, Any]:
    """Build the scene trace around already-bound answer and annotation rows."""

    label_to_interval = {
        item.label: {"lower": int(item.lower), "midpoint": int(item.midpoint), "upper": int(item.upper)}
        for item in dataset.items
    }
    return {
        "scene_ir": {
            "scene_kind": SCENE_ID,
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "scene_variant": str(dataset.scene_variant),
                "reference_value": dataset.reference_value,
                "answer_value": answer_value,
                "annotation_item_ids": [str(value) for value in dataset.query.annotation_item_ids],
            },
        },
        "render_spec": dict(render_meta),
        "render_map": {
            "plot_bbox_px": list(rendered.plot_bbox_px),
            "item_bboxes_px": dict(rendered.item_bboxes_px),
            "interval_bboxes_px": dict(rendered.interval_bboxes_px),
            "interval_center_points_px": dict(rendered.interval_center_points_px),
        },
        "execution_trace": {
            "scene_id": SCENE_ID,
            "question_format": "error_interval",
            "scene_variant": str(dataset.scene_variant),
            "category_count": int(len(dataset.items)),
            "reference_value": dataset.reference_value,
            "items": interval_records(dataset),
            "label_to_interval": dict(label_to_interval),
            "answer_value": answer_value,
            "answer_type": str(dataset.query.answer_type),
            "annotation_item_ids": [str(value) for value in dataset.query.annotation_item_ids],
            "annotation_labels": [str(record["item_label"]) for record in annotation_refs],
            **dict(dataset.query.params),
        },
        "witness_symbolic": {
            "type": "error_interval_witness",
            "answer_value": answer_value,
            "annotation_item_ids": [str(value) for value in dataset.query.annotation_item_ids],
        },
        "projected_annotation": dict(projected_annotation),
        "sidecar": dict(sidecar_meta),
        "annotation_refs": [dict(record) for record in annotation_refs],
    }


__all__ = ["build_trace_scaffold", "interval_records", "render_dataset"]
