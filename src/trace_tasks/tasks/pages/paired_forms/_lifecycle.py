"""Scene-private response assembly for paired-form public tasks."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Mapping, Sequence

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.background import make_background_canvas
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)

from ..shared.visual_defaults import load_pages_background_defaults, load_pages_noise_defaults
from .shared.rendering import render_reconciliation_scene
from .shared.sampling import (
    AnswerFunction,
    ReconciliationDefaults,
    build_paired_forms_reconciliation_dataset,
    resolve_reconciliation_render_params,
    resolve_reconciliation_scene_variant,
)


DOMAIN = "pages"
SCENE = "paired_forms"
PROMPT_BUNDLE = "pages_paired_forms_v1"
PROMPT_SCENE_KEY = "paired_forms_reconciliation"
PROMPT_TASK_KEY = "paired_forms_reconciliation_value_query"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)

_DEFAULTS = ReconciliationDefaults()
_SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE)
_BACKGROUND_DEFAULTS = load_pages_background_defaults(scene_id=SCENE)
_NOISE_DEFAULTS = load_pages_noise_defaults(scene_id=SCENE, apply_prob=0.0)
_SCENE_LOAD_BY_VARIANT = {"purchase_receipt_pair": 0.34}


def _normalize_int_with_bounds(value: int, bounds: Sequence[int]) -> float:
    """Normalize an integer into [0, 1] using inclusive bounds."""

    low, high = int(bounds[0]), int(bounds[1])
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (float(value) - float(low)) / float(high - low)))


def _clamp_unit_interval(value: float) -> float:
    """Clamp a float to [0, 1]."""

    return max(0.0, min(1.0, float(value)))


def _build_prompt_json_examples(*, answer_value: int) -> tuple[str, str]:
    """Return stable JSON examples for one paired-form task prompt."""

    annotation: list[list[int]] = []
    for index in range(1, 4):
        y0 = 300 + (index * 48)
        y1 = y0 + 22
        annotation.append([850, y0 - 8, 1268, y1 + 8])
    answer_and_annotation = {"annotation": annotation, "answer": int(answer_value)}
    answer_only = {"answer": int(answer_value)}
    return (
        json.dumps(answer_and_annotation, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        json.dumps(answer_only, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
    )


def select_single_query(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    task_id: str,
) -> tuple[str, Dict[str, float], Dict[str, Any]]:
    """Validate the public single-query contract and strip query selector params."""

    selected_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id=SINGLE_QUERY_ID,
        task_id=str(task_id),
    )
    return str(selected_query_id), dict(query_probabilities), dict(task_params)


def build_paired_forms_response(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    task_id: str,
    prompt_query_key: str,
    question_format: str,
    answer_fn: AnswerFunction,
    include_unit_value_support: bool,
    reasoning_load_base: float,
    example_answer: int,
) -> TaskOutput:
    """Generate one paired-form numeric reconciliation task instance."""

    selected_query_id, query_probabilities, task_params = select_single_query(
        instance_seed=int(instance_seed),
        params=params,
        task_id=str(task_id),
    )
    gen_defaults, render_defaults, prompt_defaults = split_generation_rendering_prompt_defaults(
        _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
        task_id=str(task_id),
    )
    scene_variant, scene_variant_probabilities = resolve_reconciliation_scene_variant(
        task_params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        sampling_namespace=str(task_id),
    )
    dataset = build_paired_forms_reconciliation_dataset(
        operation_key=str(prompt_query_key),
        answer_fn=answer_fn,
        include_unit_value_support=bool(include_unit_value_support),
        scene_variant=str(scene_variant),
        params=task_params,
        instance_seed=int(instance_seed),
        gen_defaults=gen_defaults,
        defaults=_DEFAULTS,
        sampling_namespace=str(task_id),
    )
    render_params = resolve_reconciliation_render_params(
        task_params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
    )
    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=_BACKGROUND_DEFAULTS,
    )
    rendered_scene = render_reconciliation_scene(
        background,
        scene_title=str(dataset["scene_title"]),
        purchase_title=str(dataset["purchase_title"]),
        receiving_title=str(dataset["receiving_title"]),
        purchase_header_specs=list(dataset["purchase_header_specs"]),
        receiving_header_specs=list(dataset["receiving_header_specs"]),
        item_specs=list(dataset["item_specs"]),
        receiving_item_specs=list(dataset["receiving_item_specs"]),
        render_params=render_params,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=_NOISE_DEFAULTS,
    )

    bundle_id = str(prompt_defaults.get("bundle_id", PROMPT_BUNDLE)).strip() or PROMPT_BUNDLE
    json_example, json_example_answer_only = _build_prompt_json_examples(answer_value=int(example_answer))
    prompt_selection = render_task_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE,
        bundle_id=bundle_id,
        scene_key=PROMPT_SCENE_KEY,
        task_key=PROMPT_TASK_KEY,
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

    annotation_bbox_ids = [str(value) for value in dataset["annotation_bbox_ids"]]
    annotation_bboxes = [
        [round(float(value), 3) for value in rendered_scene.row_bbox_map[str(bbox_id)]]
        for bbox_id in annotation_bbox_ids
    ]
    supporting_cell_bbox_ids = {
        str(key): str(value)
        for key, value in dict(dataset["supporting_cell_bbox_ids"]).items()
    }
    answer_value = int(dataset["answer_value"])
    answer_gt = TypedValue(type="integer", value=int(answer_value))
    annotation_gt = TypedValue(type="bbox_set", value=list(annotation_bboxes))

    item_scan = _normalize_int_with_bounds(int(dataset["item_count"]), list(dataset["item_count_range"]))
    annotation_scan = _normalize_int_with_bounds(len(annotation_bbox_ids), [4, 60])
    mismatch_scan = _normalize_int_with_bounds(
        len(dataset["mismatch_item_ids"]),
        [4, int(dataset["item_count"])],
    )
    reasoning_load = _clamp_unit_interval(
        (0.60 * float(reasoning_load_base))
        + (0.25 * float(annotation_scan))
        + (0.15 * float(mismatch_scan))
    )
    common_params = {
        "query_id": str(selected_query_id),
        "prompt_query_key": str(prompt_query_key),
        "source_query_id": str(prompt_query_key),
        "operation_key": str(prompt_query_key),
        "scene_variant": str(scene_variant),
        "query_id_probabilities": dict(query_probabilities),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "item_count": int(dataset["item_count"]),
        "mismatch_count_range": list(dataset["mismatch_count_range"]),
        "direction_count_min": int(dataset["direction_count_min"]),
        "annotation_bbox_count": int(len(annotation_bbox_ids)),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query_id),
        params=common_params,
    )
    query_spec["scene_id"] = SCENE

    trace_payload = {
        "scene_ir": {
            "scene_id": SCENE,
            "scene_kind": f"paired_forms_reconciliation_{str(scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "query_id": str(selected_query_id),
                "prompt_query_key": str(prompt_query_key),
                "source_query_id": str(prompt_query_key),
                "operation_key": str(prompt_query_key),
                "scene_variant": str(scene_variant),
                "answer_value": int(answer_value),
                "view_family": str(dataset["view_family"]),
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "scene_variant": str(scene_variant),
            "geometry_seed": int(instance_seed),
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "outer_margin_px": int(render_params.outer_margin_px),
            "panel_gap_px": int(render_params.panel_gap_px),
            "row_min_height_px": int(render_params.row_min_height_px),
            "row_max_height_px": int(render_params.row_max_height_px),
            "layout_jitter": dict(rendered_scene.layout_jitter_meta),
            "background_style": dict(background_meta),
            "post_image_noise": dict(post_noise_meta),
        },
        "render_map": {
            "panel_bboxes_px": dict(rendered_scene.panel_bboxes_px),
            "title_bboxes_px": dict(rendered_scene.title_bboxes_px),
            "header_value_bboxes_px": dict(rendered_scene.header_value_bbox_map),
            "cell_value_bboxes_px": dict(rendered_scene.cell_value_bbox_map),
            "row_bboxes_px": dict(rendered_scene.row_bbox_map),
        },
        "execution_trace": {
            "query_id": str(selected_query_id),
            "prompt_query_key": str(prompt_query_key),
            "source_query_id": str(prompt_query_key),
            "operation_key": str(prompt_query_key),
            "scene_variant": str(scene_variant),
            "question_format": str(question_format),
            "view_family": str(dataset["view_family"]),
            "scene_title": str(dataset["scene_title"]),
            "purchase_title": str(dataset["purchase_title"]),
            "receiving_title": str(dataset["receiving_title"]),
            "purchase_header_specs": [dict(spec) for spec in dataset["purchase_header_specs"]],
            "receiving_header_specs": [dict(spec) for spec in dataset["receiving_header_specs"]],
            "item_specs": [dict(spec) for spec in dataset["item_specs"]],
            "receiving_item_order_ids": [str(item) for item in dataset["receiving_item_order_ids"]],
            "item_count": int(dataset["item_count"]),
            "item_count_range": list(dataset["item_count_range"]),
            "quantity_range": list(dataset["quantity_range"]),
            "unit_value_range": list(dataset["unit_value_range"]),
            "discrepancy_range": list(dataset["discrepancy_range"]),
            "mismatch_count_range": list(dataset["mismatch_count_range"]),
            "direction_count_min": int(dataset["direction_count_min"]),
            "shortfall_item_ids": [str(item) for item in dataset["shortfall_item_ids"]],
            "overage_item_ids": [str(item) for item in dataset["overage_item_ids"]],
            "mismatch_item_ids": [str(item) for item in dataset["mismatch_item_ids"]],
            "answer_value": int(answer_value),
            "annotation_bbox_ids": list(annotation_bbox_ids),
            "supporting_cell_bbox_ids": dict(supporting_cell_bbox_ids),
            "supporting_bbox_ids": list(annotation_bbox_ids),
            "annotation_semantics": str(prompt_query_key),
        },
        "witness_symbolic": {
            "type": "receiving_row_bbox_ids",
            "ids": list(annotation_bbox_ids),
            "supporting_cell_bbox_ids": dict(supporting_cell_bbox_ids),
        },
        "projected_annotation": {
            "type": "bbox_set",
            "bbox_set": list(annotation_bboxes),
            "pixel_bbox_set": list(annotation_bboxes),
            "annotation_bbox_ids": list(annotation_bbox_ids),
        },
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
        "answer_gt": answer_gt.to_dict(),
        "annotation_gt": annotation_gt.to_dict(),
        "difficulty": {
            "visual_scan": _clamp_unit_interval(
                (0.50 * float(item_scan)) + (0.50 * float(annotation_scan))
            ),
            "reasoning_load": float(reasoning_load),
            "answer_noise": 0.0,
            "scene_load": float(_SCENE_LOAD_BY_VARIANT.get(str(scene_variant), 0.25)),
        },
    }

    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        query_id=str(selected_query_id),
    )


__all__ = [
    "DOMAIN",
    "PROMPT_BUNDLE",
    "PROMPT_SCENE_KEY",
    "PROMPT_TASK_KEY",
    "SCENE",
    "SUPPORTED_QUERY_IDS",
    "build_paired_forms_response",
]
