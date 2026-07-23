"""Neutral render and trace helpers for spinner tasks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from PIL import Image

from ....core.seed import spawn_rng
from ....core.query_ids import SINGLE_QUERY_ID
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...shared.config_defaults import group_default
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import PROMPT_OUTPUT_MODES
from ..shared.scene_style import make_symbolic_scene_background, resolve_symbolic_scene_style

from .shared.annotations import scalar_panel_bbox
from .shared.defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    SCENE_ID,
    load_spinner_defaults,
    resolve_spinner_render_params,
    resolve_spinner_scene_variant,
    with_spinner_style_overrides,
)
from .shared.prompts import render_spinner_prompt
from .shared.rendering import (
    RenderedSpinnerScene,
    SpinnerRenderParams,
    draw_probability_option_cards,
    option_cards_y_for_scene,
    render_spinner_scene,
)
from .shared.rules import (
    PROBABILITY_OPTION_LABELS,
    build_probability_option_set,
    build_single_spinner_dataset,
    color_shape_event_candidates,
    normalize_int_with_bounds,
    select_event_candidate,
)


_SCENE_LOAD = {
    "spinner_clean": 0.16,
    "spinner_card": 0.22,
    "spinner_notebook": 0.24,
}


@dataclass(frozen=True)
class SpinnerRenderBundle:
    """Rendered spinner scene and common prompt/render metadata."""

    image: Image.Image
    rendered_scene: RenderedSpinnerScene
    render_params: SpinnerRenderParams
    prompt: str
    prompt_variants: Dict[str, str]
    prompt_meta: Dict[str, Any]
    scene_variant: str
    scene_variant_probabilities: Dict[str, float]
    background_metadata: Dict[str, Any]
    scene_style_metadata: Dict[str, Any]
    post_noise_metadata: Dict[str, Any]
    answer_options: Dict[str, Any]
    option_bboxes_px: Dict[str, list[float]]
    option_y0_px: int
    task_versions: Dict[str, str]


@dataclass(frozen=True)
class SpinnerTaskOutputParts:
    """Common generated payloads used by public tasks to build TaskOutput."""

    prompt: str
    image: Image.Image
    image_id: str
    trace_payload: Dict[str, Any]
    task_versions: Dict[str, str]
    prompt_variants: Dict[str, str]


@dataclass(frozen=True)
class SingleColorShapeOutputParts:
    """Answer, annotation, and common output parts for fixed color/shape tasks."""

    query_id: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    output_parts: SpinnerTaskOutputParts


def _union_bbox(*bboxes) -> list[float]:
    if not bboxes:
        raise ValueError("at least one bbox is required")
    return [
        round(min(float(bbox[0]) for bbox in bboxes), 3),
        round(min(float(bbox[1]) for bbox in bboxes), 3),
        round(max(float(bbox[2]) for bbox in bboxes), 3),
        round(max(float(bbox[3]) for bbox in bboxes), 3),
    ]


def _resolve_probability_option_labels(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
) -> tuple[str, ...]:
    raw_labels = params.get(
        "option_label_support",
        group_default(gen_defaults, "option_label_support", PROBABILITY_OPTION_LABELS),
    )
    labels = tuple(str(label) for label in raw_labels)
    option_count = int(params.get("option_count", group_default(gen_defaults, "option_count", 6)))
    if int(option_count) != 6:
        raise ValueError("spinner probability tasks require exactly six visible options")
    if len(labels) < int(option_count):
        raise ValueError("spinner probability option label support must contain at least six labels")
    labels = labels[: int(option_count)]
    if len(set(labels)) != len(labels):
        raise ValueError("spinner probability option labels must be unique")
    return tuple(str(label) for label in labels)


def render_spinner_bundle(
    *,
    public_task_id: str,
    prompt_query_key: str,
    dataset: Mapping[str, Any],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    event_description: str,
) -> SpinnerRenderBundle:
    """Render one spinner scene after a public task has built its dataset."""

    scene_variant, scene_variant_probabilities = resolve_spinner_scene_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        public_task_id=str(public_task_id),
    )
    render_params = resolve_spinner_render_params(render_defaults)
    scene_style, scene_style_meta = resolve_symbolic_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{public_task_id}.spinner_background",
    )
    render_params = with_spinner_style_overrides(render_params, scene_style)
    background, background_meta = make_symbolic_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=scene_style,
    )
    rendered_scene = render_spinner_scene(
        background,
        scene_variant=str(scene_variant),
        mode=str(dataset["mode"]),
        spinner_specs=list(dataset["spinner_specs"]),
        render_params=render_params,
    )
    event = dict(dataset["event"])
    option_labels = _resolve_probability_option_labels(params, gen_defaults=gen_defaults)
    option_rng = spawn_rng(int(instance_seed), f"{public_task_id}.spinner_probability_options")
    answer_options = build_probability_option_set(
        favorable=int(event["favorable_outcome_count"]),
        total=int(event["total_outcome_count"]),
        rng=option_rng,
        labels=option_labels,
        correct_label=(
            str(params.get("answer_label", params.get("correct_label")))
            if params.get("answer_label", params.get("correct_label")) is not None
            else None
        ),
    )
    option_y0_px = option_cards_y_for_scene(
        rendered_scene.scene_bbox_px,
        canvas_height=int(render_params.canvas_height),
    )
    raw_option_bboxes, option_entities = draw_probability_option_cards(
        rendered_scene.image,
        text_by_label=dict(answer_options["text_by_label"]),
        correct_label=str(answer_options["correct_label"]),
        y0_px=int(option_y0_px),
    )
    option_bboxes_px = {
        str(label): [round(float(value), 3) for value in bbox]
        for label, bbox in raw_option_bboxes.items()
    }
    rendered_scene = RenderedSpinnerScene(
        image=rendered_scene.image,
        entities=[*rendered_scene.entities, *[dict(entity) for entity in option_entities]],
        item_bbox_map=dict(rendered_scene.item_bbox_map),
        sector_bbox_map=dict(rendered_scene.sector_bbox_map),
        panel_bbox_map=dict(rendered_scene.panel_bbox_map),
        scene_bbox_px=_union_bbox(
            rendered_scene.scene_bbox_px,
            *[tuple(float(value) for value in bbox) for bbox in raw_option_bboxes.values()],
        ),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    prompt, prompt_variants, prompt_meta = render_spinner_prompt(
        prompt_query_key=str(prompt_query_key),
        scene_variant=str(scene_variant),
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
        event_description=str(event_description),
    )
    return SpinnerRenderBundle(
        image=image,
        rendered_scene=rendered_scene,
        render_params=render_params,
        prompt=str(prompt),
        prompt_variants=dict(prompt_variants),
        prompt_meta=dict(prompt_meta),
        scene_variant=str(scene_variant),
        scene_variant_probabilities={str(key): float(value) for key, value in scene_variant_probabilities.items()},
        background_metadata=dict(background_meta),
        scene_style_metadata=dict(scene_style_meta),
        post_noise_metadata=dict(post_noise_meta),
        answer_options=dict(answer_options),
        option_bboxes_px=dict(option_bboxes_px),
        option_y0_px=int(option_y0_px),
        task_versions=default_task_versions(),
    )


def _spinner_outcome_bounds(dataset: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    mode = str(dataset["mode"])
    if mode == "single":
        sector_min, sector_max = (int(value) for value in dataset["sector_count_range"])
        return [sector_min, sector_max], [sector_min, sector_max]
    sector_min, sector_max = (int(value) for value in dataset["sector_count_range"])
    return [2 * sector_min, 2 * sector_max], [sector_min * sector_min, sector_max * sector_max]


def build_spinner_trace_payload(
    *,
    public_query_id: str,
    prompt_query_key: str,
    query_probabilities: Mapping[str, float],
    dataset: Mapping[str, Any],
    bundle: SpinnerRenderBundle,
    answer_gt: TypedValue,
    annotation_gt: TypedValue,
    projected_annotation: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    annotation_source: str,
    reasoning_load_base: float,
    annotation_role_item_ids: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """Build common trace payload fields from task-owned answer and annotation."""

    event = dict(dataset["event"])
    answer_options = dict(bundle.answer_options)
    spinner_specs = [dict(spinner) for spinner in dataset["spinner_specs"]]
    all_sectors = [dict(sector) for spinner in spinner_specs for sector in spinner["sectors"]]
    visual_scan_bounds, outcome_bounds = _spinner_outcome_bounds(dataset)
    visual_scan = normalize_int_with_bounds(len(all_sectors), visual_scan_bounds)
    outcome_load = normalize_int_with_bounds(int(event["total_outcome_count"]), outcome_bounds)
    reasoning_load = min(1.0, float(reasoning_load_base) + (0.12 * float(outcome_load)))

    payload: Dict[str, Any] = {
        "scene_ir": {
            "scene_kind": "symbolic_probability_spinner_panel",
            "entities": [dict(entity) for entity in bundle.rendered_scene.entities],
            "relations": {
                "scene_id": SCENE_ID,
                "query_id": str(public_query_id),
                "event_key": str(prompt_query_key),
                "scene_variant": str(bundle.scene_variant),
                "answer_value": str(answer_gt.value),
                "probability_fraction": str(dataset["answer_value"]),
                "event_description": str(event["event_description"]),
            },
        },
        "query_spec": {
            "query_id": str(public_query_id),
            "template_id": str(bundle.prompt_meta["bundle_id"]),
            "prompt_variant": dict(bundle.prompt_meta["prompt_variant"]),
            "prompt_variant_active_key": str(bundle.prompt_meta["prompt_variant_active_key"]),
            "prompt_variants": dict(bundle.prompt_meta["prompt_variants_for_trace"]),
            "params": {
                "scene_id": SCENE_ID,
                "query_id": str(public_query_id),
                "query_id_probabilities": {str(key): float(value) for key, value in query_probabilities.items()},
                "event_key": str(prompt_query_key),
                "scene_variant": str(bundle.scene_variant),
                "scene_variant_probabilities": dict(bundle.scene_variant_probabilities),
                "mode": str(dataset["mode"]),
                "event_description": str(event["event_description"]),
                "favorable_outcome_count": int(event["favorable_outcome_count"]),
                "total_outcome_count": int(event["total_outcome_count"]),
                "probability_fraction": str(dataset["answer_value"]),
                "option_labels": [str(label) for label in answer_options["labels"]],
                "correct_label": str(answer_options["correct_label"]),
            },
        },
        "render_spec": {
            "scene_id": SCENE_ID,
            "canvas_width": int(bundle.render_params.canvas_width),
            "canvas_height": int(bundle.render_params.canvas_height),
            "coord_space": "pixel",
            "scene_variant": str(bundle.scene_variant),
            "background_style": dict(bundle.background_metadata),
            "scene_style": dict(bundle.scene_style_metadata),
            "post_image_noise": dict(bundle.post_noise_metadata),
            "post_image_noise_policy": {
                "apply_prob": 0.15,
                "reason": "spinner_color_and_shape_readability",
                "scope": "semantic sector colors and visible shape markers",
            },
            "scene_bbox_px": list(bundle.rendered_scene.scene_bbox_px),
            "layout": str(dataset["mode"]),
            "option_card_layout": {
                "option_labels": [str(label) for label in answer_options["labels"]],
                "option_y0_px": int(bundle.option_y0_px),
                "option_count": int(len(answer_options["labels"])),
            },
        },
        "render_map": {
            "image_id": "img0",
            "scene_bbox_px": list(bundle.rendered_scene.scene_bbox_px),
            "sector_bboxes_px": {str(key): list(value) for key, value in bundle.rendered_scene.sector_bbox_map.items()},
            "panel_bboxes_px": {str(key): list(value) for key, value in bundle.rendered_scene.panel_bbox_map.items()},
            "option_bboxes_px": dict(bundle.option_bboxes_px),
            "selected_option_label": str(answer_options["correct_label"]),
            "selected_option_bbox_px": list(bundle.option_bboxes_px[str(answer_options["correct_label"])]),
            "item_bboxes_px": {str(key): list(value) for key, value in bundle.rendered_scene.item_bbox_map.items()},
            "annotation_source": str(annotation_source),
        },
        "execution_trace": {
            "query_id": str(public_query_id),
            "query_id_probabilities": {str(key): float(value) for key, value in query_probabilities.items()},
            "event_key": str(prompt_query_key),
            "scene_id": SCENE_ID,
            "scene_variant": str(bundle.scene_variant),
            "scene_variant_probabilities": dict(bundle.scene_variant_probabilities),
            "mode": str(dataset["mode"]),
            "spinner_specs": spinner_specs,
            "sector_specs": all_sectors,
            "sector_attribute_counts": {
                "color": dict(Counter(str(sector["color_name"]) for sector in all_sectors)),
                "shape": dict(Counter(str(sector["shape"]) for sector in all_sectors)),
            },
            "event": dict(event),
            "event_description": str(event["event_description"]),
            "favorable_outcome_count": int(event["favorable_outcome_count"]),
            "total_outcome_count": int(event["total_outcome_count"]),
            "answer_value": str(answer_gt.value),
            "answer_label": str(answer_options["correct_label"]),
            "answer_type": "option_letter",
            "probability_fraction": str(dataset["answer_value"]),
            "option_labels": [str(label) for label in answer_options["labels"]],
            "option_text_by_label": {
                str(label): str(text)
                for label, text in dict(answer_options["text_by_label"]).items()
            },
            "option_values_by_label": {
                str(label): str(text)
                for label, text in dict(answer_options["value_by_label"]).items()
            },
            "correct_label_probabilities": {
                str(label): float(probability)
                for label, probability in dict(answer_options["correct_label_probabilities"]).items()
            },
            "annotation_item_ids": [str(item_id) for item_id in dataset["annotation_item_ids"]],
            "calculation_supporting_item_ids": [
                str(item_id) for item_id in dataset["calculation_supporting_item_ids"]
            ],
            "question_format": str(prompt_query_key),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
        "answer_gt": answer_gt.to_dict(),
        "annotation_gt": annotation_gt.to_dict(),
        "analysis": {
            "complexity_components": {
                "scene_load": float(_SCENE_LOAD[str(bundle.scene_variant)]),
                "visual_scan": float(visual_scan),
                "outcome_load": float(outcome_load),
                "reasoning_load": float(reasoning_load),
            }
        },
    }
    if annotation_role_item_ids is not None:
        payload["execution_trace"]["annotation_role_item_ids"] = {
            str(role): str(item_id) for role, item_id in annotation_role_item_ids.items()
        }
    if str(dataset["mode"]) == "single":
        payload["execution_trace"]["sector_count"] = int(dataset["sector_count"])
        payload["execution_trace"]["sector_count_range"] = list(dataset["sector_count_range"])
    else:
        payload["execution_trace"]["sector_count_a"] = int(dataset["sector_count_a"])
        payload["execution_trace"]["sector_count_b"] = int(dataset["sector_count_b"])
        payload["execution_trace"]["sector_count_range"] = list(dataset["sector_count_range"])
    return payload


def prepare_spinner_task_output_parts(
    *,
    public_query_id: str,
    prompt_query_key: str,
    query_probabilities: Mapping[str, float],
    dataset: Mapping[str, Any],
    bundle: SpinnerRenderBundle,
    answer_gt: TypedValue,
    annotation_gt: TypedValue,
    projected_annotation: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    annotation_source: str,
    reasoning_load_base: float,
    annotation_role_item_ids: Mapping[str, str] | None = None,
) -> SpinnerTaskOutputParts:
    """Build common trace/output parts after a public task binds annotation."""

    trace_payload = build_spinner_trace_payload(
        public_query_id=str(public_query_id),
        prompt_query_key=str(prompt_query_key),
        query_probabilities=query_probabilities,
        dataset=dataset,
        bundle=bundle,
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        projected_annotation=projected_annotation,
        witness_symbolic=witness_symbolic,
        annotation_source=str(annotation_source),
        annotation_role_item_ids=annotation_role_item_ids,
        reasoning_load_base=float(reasoning_load_base),
    )
    return SpinnerTaskOutputParts(
        prompt=str(bundle.prompt),
        image=bundle.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=dict(bundle.task_versions),
        prompt_variants=dict(bundle.prompt_variants),
    )


def spinner_task_output_fields(
    *,
    output_parts: SpinnerTaskOutputParts,
    answer_gt: TypedValue,
    annotation_gt: TypedValue,
    query_id: str,
) -> Dict[str, Any]:
    """Return final output fields after the public task binds answer/annotation."""

    return {
        "prompt": str(output_parts.prompt),
        "answer_gt": answer_gt,
        "annotation_gt": annotation_gt,
        "image": output_parts.image,
        "image_id": str(output_parts.image_id),
        "trace_payload": dict(output_parts.trace_payload),
        "task_versions": dict(output_parts.task_versions),
        "scene_id": SCENE_ID,
        "query_id": str(query_id),
        "prompt_variants": dict(output_parts.prompt_variants),
    }


def prepare_single_color_shape_probability_parts(
    *,
    public_task_id: str,
    operator: str,
    prompt_query_key: str,
    reasoning_load_base: float,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> SingleColorShapeOutputParts:
    """Prepare generated parts for a fixed color/shape spinner objective."""

    del max_attempts
    gen_defaults, render_defaults, prompt_defaults = load_spinner_defaults(str(public_task_id))
    public_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=(SINGLE_QUERY_ID,),
        task_id=str(public_task_id),
        namespace=f"{public_task_id}.query",
    )

    def _event_builder(**kwargs: Any) -> Mapping[str, Any]:
        candidates = color_shape_event_candidates(
            sectors=kwargs["sectors"],
            params=kwargs["params"],
            gen_defaults=kwargs["gen_defaults"],
            operator=str(operator),
        )
        return select_event_candidate(
            candidates,
            rng=kwargs["rng"],
            favorable_key="favorable_sector_ids",
            total_outcome_count=len(kwargs["sectors"]),
        )

    dataset = build_single_spinner_dataset(
        params=task_params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        rng_namespace=f"{public_task_id}.dataset",
        event_builder=_event_builder,
    )
    event = dict(dataset["event"])
    bundle = render_spinner_bundle(
        public_task_id=str(public_task_id),
        prompt_query_key=str(prompt_query_key),
        dataset=dataset,
        params=task_params,
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
        event_description=str(event["event_description"]),
    )
    annotation_gt, projected_annotation, witness_symbolic, annotation_source = scalar_panel_bbox(
        bundle.rendered_scene.item_bbox_map,
        item_id="spinner_panel",
    )
    answer_gt = TypedValue(type="option_letter", value=str(bundle.answer_options["correct_label"]))
    output_parts = prepare_spinner_task_output_parts(
        public_query_id=str(public_query_id),
        prompt_query_key=str(prompt_query_key),
        query_probabilities=query_probabilities,
        dataset=dataset,
        bundle=bundle,
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        projected_annotation=projected_annotation,
        witness_symbolic=witness_symbolic,
        annotation_source=str(annotation_source),
        reasoning_load_base=float(reasoning_load_base),
    )
    return SingleColorShapeOutputParts(
        query_id=str(public_query_id),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        output_parts=output_parts,
    )


__all__ = [
    "PROMPT_OUTPUT_MODES",
    "SingleColorShapeOutputParts",
    "SpinnerRenderBundle",
    "SpinnerTaskOutputParts",
    "build_spinner_trace_payload",
    "prepare_spinner_task_output_parts",
    "prepare_single_color_shape_probability_parts",
    "render_spinner_bundle",
    "spinner_task_output_fields",
]
