"""Neutral render and trace scaffolding for dashboard chart tasks."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Mapping, Sequence

from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.font_assets import font_asset_version
from trace_tasks.tasks.shared.visual_style.context_layer import context_text_layer_metadata

from .annotations import AnnotationRef
from .defaults import POST_IMAGE_NOISE_DEFAULTS, resolve_render_params
from .rendering import _bbox_map_to_json, _nested_bbox_map_to_json, _nested_point_map_to_json, render_dashboard
from .state import DashboardDataset, RenderedDashboard, SCENE_ID


def _context_layer_mode(layout: Mapping[str, Any]) -> str:
    mode = str(layout.get("chart_context_mode", ""))
    if mode in {"clean", "minimal", "paragraph_box"}:
        return f"chart_context:{mode}"
    return f"{layout.get('layout_mode', 'reserved_context')}:{layout.get('placement', 'none')}"


def render_dataset(dataset: DashboardDataset, *, params: Mapping[str, Any], instance_seed: int) -> tuple[RenderedDashboard, Dict[str, Any], Dict[str, Any]]:
    """Render one dashboard scene and return synchronized render metadata.

    This shared helper is task-neutral: public task files bind answer and
    annotation semantics, while this function only resolves visual style,
    draws the scene, applies post-noise, and records render-side metadata from
    the same image trace.
    """

    render_style_params = {**dict(params), "_render_style_seed": int(instance_seed)}
    render_params = resolve_render_params(render_style_params)
    protected_colors = tuple(tuple(int(channel) for channel in category.color_rgb) for category in dataset.categories)
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=dict(params),
        scene_id=SCENE_ID,
        render_params=render_params,
        protected_colors=protected_colors,
    )
    rendered = render_dashboard(background, dataset=dataset, render_params=render_params, params=dict(params), instance_seed=int(instance_seed))
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    rendered = replace(rendered, image=image)
    render_meta = {
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "coord_space": "pixel",
        "panel_count": int(len(dataset.panels)),
        "category_count": int(len(dataset.categories)),
        "layout_jitter": dict(render_params.layout_jitter_meta),
        "font_assets": {"asset_version": font_asset_version(), "chart_font_family": str(render_params.font_family)},
        "background_style": dict(background_meta),
        "information_scene_style": dict(information_style_meta),
        "context_text_layer": context_text_layer_metadata(
            [],
            enabled=bool(rendered.context_text_layout.get("enabled", True)),
            layout_mode=_context_layer_mode(rendered.context_text_layout),
            layout_spec=dict(rendered.context_text_layout),
        )
        | {"element_count": int(len(rendered.context_text_elements)), "elements": [dict(element) for element in rendered.context_text_elements]},
        "post_image_noise": dict(post_noise_meta),
    }
    return rendered, render_meta, {
        "background": dict(background_meta),
        "information_scene_style": dict(information_style_meta),
        "post_image_noise": dict(post_noise_meta),
    }


def _category_records(dataset: DashboardDataset) -> list[dict[str, Any]]:
    return [{"category_id": str(category.category_id), "label": str(category.label), "color_rgb": list(category.color_rgb)} for category in dataset.categories]


def _panel_records(dataset: DashboardDataset) -> list[dict[str, Any]]:
    return [{"panel_id": str(panel.panel_id), "panel_kind": str(panel.kind), "panel_name": str(panel.name)} for panel in dataset.panels]


def _values_by_panel(dataset: DashboardDataset) -> dict[str, Any]:
    categories_by_id = {str(category.category_id): category for category in dataset.categories}
    return {
        str(panel.panel_id): {
            "panel_name": str(panel.name),
            "panel_kind": str(panel.kind),
            "values_by_category_id": {str(category_id): int(value) for category_id, value in panel.values_by_category_id.items()},
            "values_by_category_label": {str(categories_by_id[str(category_id)].label): int(value) for category_id, value in panel.values_by_category_id.items()},
        }
        for panel in dataset.panels
    }


def build_trace_scaffold(
    *,
    dataset: DashboardDataset,
    rendered: RenderedDashboard,
    render_meta: Mapping[str, Any],
    sidecar_meta: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    annotation_refs: Sequence[AnnotationRef],
    answer_value: int | str,
    relations: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build trace sections that are independent of public task identity."""

    answerability = str(relations.get("answerability", "answerable"))
    absence = {"absence_proof": dict(relations["absence_proof"])} if answerability == "unanswerable" and "absence_proof" in relations else {}
    return {
        "scene_ir": {
            "scene_kind": "chart_mixed_dashboard",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "scene_variant": str(dataset.scene_variant),
                "answer": answer_value,
                "annotation_refs": [list(ref) for ref in annotation_refs],
                "answerability": answerability,
                **absence,
            },
        },
        "render_spec": dict(render_meta),
        "render_map": {
            "image_id": "img0",
            "panel_bboxes_px": _bbox_map_to_json(rendered.panel_bboxes_px),
            "support_bboxes_px": _nested_bbox_map_to_json(rendered.support_bboxes_px),
            "support_points_px": _nested_point_map_to_json(rendered.support_points_px),
            "value_label_bboxes_px": _nested_bbox_map_to_json(rendered.value_label_bboxes_px),
            "option_statement_bboxes_px": _bbox_map_to_json(rendered.option_statement_bboxes_px),
            "context_text_bboxes_px": {str(element["context_id"]): [int(value) for value in element["bbox_xyxy"]] for element in rendered.context_text_elements},
        },
        "execution_trace": {
            "scene_id": SCENE_ID,
            "scene_variant": str(dataset.scene_variant),
            "question_format": "dashboard_cross_panel_query",
            "answer": answer_value,
            "answer_type": str(dataset.query.answer_type),
            "category_count": int(len(dataset.categories)),
            "panel_count": int(len(dataset.panels)),
            "categories": _category_records(dataset),
            "panels": _panel_records(dataset),
            "panel_order": [str(panel.panel_id) for panel in dataset.panels],
            "panel_kinds": [str(panel.kind) for panel in dataset.panels],
            "values_by_panel": _values_by_panel(dataset),
            "annotation_refs": [list(ref) for ref in annotation_refs],
            **dict(relations),
        },
        "witness_symbolic": {**dict(witness_symbolic), "answer": answer_value, "answerability": answerability, **absence},
        "projected_annotation": dict(projected_annotation),
        **dict(sidecar_meta),
    }


__all__ = ["build_trace_scaffold", "render_dataset"]
