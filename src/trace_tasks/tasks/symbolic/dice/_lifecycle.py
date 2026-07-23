"""Scene-private rendering/output lifecycle for symbolic dice probability."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.scene_config import get_scene_defaults
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.config_defaults import group_default, required_group_defaults
from ...shared.font_assets import font_asset_version, sample_font_family
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_scene_prompt_variants
from ...shared.text_rendering import temporary_default_font_family
from ..shared.common import (
    load_symbolic_task_defaults,
    projected_symbolic_keyed_bbox_annotation,
    resolve_symbolic_axis_variant,
)
from ..shared.scene_style import make_symbolic_scene_background, resolve_symbolic_scene_style
from ..shared.visual_defaults import load_symbolic_background_defaults, load_symbolic_noise_defaults
from .shared.rendering import (
    SUPPORTED_DICE_SCENE_VARIANTS,
    SUPPORTED_DICE_VISUAL_STYLES,
    DiceRenderParams,
    RenderedDiceScene,
    draw_probability_option_cards,
    option_cards_y_for_scene,
    render_dice_probability_scene,
)
from .shared.rules import PROBABILITY_OPTION_LABELS, build_probability_option_set, normalize_int_with_bounds


SCENE_ID = "dice"

_SCENE_LOAD = {
    "dice_tray_clean": 0.16,
    "dice_tray_felt": 0.22,
    "dice_tray_notebook": 0.24,
}

_SCENE_DEFAULTS = get_scene_defaults("symbolic", SCENE_ID)
POST_IMAGE_BACKGROUND_DEFAULTS = load_symbolic_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_symbolic_noise_defaults(scene_id=SCENE_ID, apply_prob=0.15)


def load_dice_probability_defaults(public_task_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load scene defaults for one public dice-probability task."""

    gen_defaults, render_defaults, prompt_defaults, _axis_probabilities = load_symbolic_task_defaults(
        _SCENE_DEFAULTS,
        task_id=str(public_task_id),
    )
    return gen_defaults, render_defaults, prompt_defaults


def resolve_dice_query(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    public_task_id: str,
    supported_queries: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    """Resolve a public query id from the task-owned support set."""

    effective_params = dict(params)
    if effective_params.get("query_id") is None and effective_params.get("query_variant") is not None:
        effective_params["query_id"] = str(effective_params["query_variant"])
    return resolve_symbolic_axis_variant(
        params=effective_params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=[str(query) for query in supported_queries],
        task_id=str(public_task_id),
        explicit_key="query_id",
        weights_key="query_id_weights",
        balance_flag_key="balanced_query_id_sampling",
        axis_namespace="query_id",
    )


def _sample_dice_probability_font(
    *,
    public_task_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> str:
    """Sample one role-aware font family for tray labels in a dice probability scene."""

    return sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{public_task_id}.dice.label_font",
        params={**dict(render_defaults), **dict(params)},
    )


def _font_trace_record(font_family: str) -> Dict[str, Any]:
    """Build trace metadata for the sampled dice scene label font."""

    return {
        "source": "global_font_pool",
        "font_family": str(font_family),
        "font_asset_version": font_asset_version(),
        "scope": "dice_probability_tray_labels",
    }


def _round_bbox_map(projected: Mapping[str, Any]) -> Dict[str, List[float]]:
    keyed = projected.get("bbox_map", {})
    if not isinstance(keyed, Mapping):
        raise ValueError("keyed bbox projection missing bbox_map")
    return {
        str(role): [round(float(value), 3) for value in bbox]
        for role, bbox in keyed.items()
    }


def _resolve_scene_variant(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    public_task_id: str,
) -> Tuple[str, Dict[str, float]]:
    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_DICE_SCENE_VARIANTS,
        task_id=str(public_task_id),
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def _resolve_dice_visual_style(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    public_task_id: str,
) -> Tuple[str, Dict[str, float]]:
    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_DICE_VISUAL_STYLES,
        task_id=str(public_task_id),
        explicit_key="dice_visual_style",
        weights_key="dice_visual_style_weights",
        balance_flag_key="balanced_dice_visual_style_sampling",
        axis_namespace="dice_visual_style",
    )


def _dice_visual_style_metadata(style_id: str, probabilities: Mapping[str, float]) -> Dict[str, Any]:
    return {
        "style_id": str(style_id),
        "style_probabilities": {str(key): float(value) for key, value in probabilities.items()},
        "semantic_color_policy": {
            "die_face_colors_preserved": True,
            "pip_count_and_positions_preserved": True,
            "style_is_non_semantic": True,
        },
    }


def _bbox_tuple(raw: Any, fallback: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 4:
        return tuple(int(value) for value in raw[:4])  # type: ignore[return-value]
    return tuple(int(value) for value in fallback)


def _resolve_render_params(render_defaults: Mapping[str, Any]) -> DiceRenderParams:
    return DiceRenderParams(
        canvas_width=int(render_defaults.get("canvas_width", 1100)),
        canvas_height=int(render_defaults.get("canvas_height", 780)),
        single_tray_bbox_px=_bbox_tuple(render_defaults.get("single_tray_bbox_px"), (145, 112, 955, 650)),
        pair_left_tray_bbox_px=_bbox_tuple(render_defaults.get("pair_left_tray_bbox_px"), (68, 142, 522, 628)),
        pair_right_tray_bbox_px=_bbox_tuple(render_defaults.get("pair_right_tray_bbox_px"), (578, 142, 1032, 628)),
        die_size_px=int(render_defaults.get("die_size_px", 72)),
        die_gap_px=int(render_defaults.get("die_gap_px", 18)),
        tray_corner_radius_px=int(render_defaults.get("tray_corner_radius_px", 24)),
        tray_outline_width_px=int(render_defaults.get("tray_outline_width_px", 3)),
        die_corner_radius_px=int(render_defaults.get("die_corner_radius_px", 14)),
        die_outline_width_px=int(render_defaults.get("die_outline_width_px", 3)),
        pip_radius_px=int(render_defaults.get("pip_radius_px", 6)),
        title_font_size_px=int(render_defaults.get("title_font_size_px", 28)),
    )


def _resolve_probability_option_labels(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
) -> Tuple[str, ...]:
    raw_labels = params.get(
        "option_label_support",
        group_default(gen_defaults, "option_label_support", PROBABILITY_OPTION_LABELS),
    )
    labels = tuple(str(label) for label in raw_labels)
    option_count = int(params.get("option_count", group_default(gen_defaults, "option_count", 6)))
    if int(option_count) != 6:
        raise ValueError("dice probability tasks require exactly six visible options")
    if len(labels) < int(option_count):
        raise ValueError("dice probability option label support must contain at least six labels")
    labels = labels[: int(option_count)]
    if len(set(labels)) != len(labels):
        raise ValueError("dice probability option labels must be unique")
    return tuple(str(label) for label in labels)


def _union_bbox(*bboxes: Sequence[float]) -> List[float]:
    if not bboxes:
        raise ValueError("at least one bbox is required")
    return [
        round(min(float(bbox[0]) for bbox in bboxes), 3),
        round(min(float(bbox[1]) for bbox in bboxes), 3),
        round(max(float(bbox[2]) for bbox in bboxes), 3),
        round(max(float(bbox[3]) for bbox in bboxes), 3),
    ]


def _build_prompt(
    *,
    prompt_query_key: str,
    scene_variant: str,
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    event_description: str,
    given_description: str | None = None,
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """Render the scene/task/query/output prompt path for one dice event.

    The public task chooses the event key and the sampled event description;
    this helper only binds those task-owned values into the shared v1 prompt
    bundle and records prompt metadata.
    """

    raw_task_key = str(prompt_defaults.get("task_key", ""))
    object_description_key = f"object_description_{raw_task_key}_{scene_variant}"
    required = [
        "bundle_id",
        "scene_key",
        "task_key",
        object_description_key,
    ]
    prompt_values = required_group_defaults(
        prompt_defaults,
        tuple(required),
        context="prompt defaults for dice probability",
    )
    dynamic_slots: Dict[str, Any] = {
        "object_description": str(prompt_values[object_description_key]),
        "event_description": str(event_description),
    }
    if given_description is not None:
        dynamic_slots["given_description"] = str(given_description)
    prompt_selection = render_scene_prompt_variants(
        domain="symbolic",
        scene_id=SCENE_ID,
        bundle_id=str(prompt_values["bundle_id"]),
        scene_key=str(prompt_values["scene_key"]),
        task_key=str(prompt_values["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dynamic_slots,
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
    return str(prompt_artifacts.prompt), dict(prompt_artifacts.prompt_variants), {
        "prompt_variant": dict(prompt_artifacts.prompt_variant),
        "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
        "prompt_variants_for_trace": dict(prompt_artifacts.prompt_variants_for_trace),
        "bundle_id": str(prompt_values["bundle_id"]),
    }


def build_dice_probability_output(
    *,
    public_task_id: str,
    public_query_id: str,
    prompt_query_key: str,
    query_probabilities: Mapping[str, float],
    dataset: Mapping[str, Any],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    reasoning_load_base: float,
) -> TaskOutput:
    """Render one dice-probability scene and bind answer plus annotation."""

    scene_variant, scene_variant_probabilities = _resolve_scene_variant(
        params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        public_task_id=str(public_task_id),
    )
    dice_visual_style, dice_visual_style_probabilities = _resolve_dice_visual_style(
        params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        public_task_id=str(public_task_id),
    )
    render_params = _resolve_render_params(render_defaults)
    font_family = _sample_dice_probability_font(
        public_task_id=str(public_task_id),
        instance_seed=int(instance_seed),
        params=params,
        render_defaults=render_defaults,
    )
    scene_style, scene_style_meta = resolve_symbolic_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{public_task_id}.dice_background",
    )
    background, background_meta = make_symbolic_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=scene_style,
    )
    with temporary_default_font_family(str(font_family)):
        rendered_scene = render_dice_probability_scene(
            background,
            scene_variant=str(scene_variant),
            mode=str(dataset["mode"]),
            tray_specs=list(dataset["tray_specs"]),
            render_params=render_params,
            scene_style=scene_style,
            visual_style=str(dice_visual_style),
        )
    event = dict(dataset["event"])
    option_labels = _resolve_probability_option_labels(params, gen_defaults=gen_defaults)
    option_rng = spawn_rng(int(instance_seed), f"{public_task_id}.dice_probability_options")
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
    rendered_scene = RenderedDiceScene(
        image=rendered_scene.image,
        entities=[*rendered_scene.entities, *[dict(entity) for entity in option_entities]],
        item_bbox_map=dict(rendered_scene.item_bbox_map),
        die_bbox_map=dict(rendered_scene.die_bbox_map),
        tray_bbox_map=dict(rendered_scene.tray_bbox_map),
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
    prompt, prompt_variants, prompt_meta = _build_prompt(
        prompt_query_key=str(prompt_query_key),
        scene_variant=str(scene_variant),
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
        event_description=str(event["event_description"]),
        given_description=str(event["given_description"]) if "given_description" in event else None,
    )
    annotation_item_ids = [str(item_id) for item_id in dataset["annotation_item_ids"]]
    calculation_supporting_item_ids = [str(item_id) for item_id in dataset["calculation_supporting_item_ids"]]
    annotation_role_item_ids = (
        {"tray_a": "tray_a", "tray_b": "tray_b"}
        if str(dataset["mode"]) == "pair"
        else {"dice_tray": "tray"}
    )
    annotation_projection = projected_symbolic_keyed_bbox_annotation(rendered_scene.item_bbox_map, annotation_role_item_ids)
    annotation_bboxes = _round_bbox_map(annotation_projection)
    if len(annotation_bboxes) != len(annotation_role_item_ids):
        raise ValueError("dice probability annotation projection dropped tray boxes")

    probability_fraction = str(dataset["answer_value"])
    answer_value = str(answer_options["correct_label"])
    answer_gt = TypedValue(type="option_letter", value=str(answer_value))
    if str(dataset["mode"]) == "pair":
        annotation_gt = TypedValue(type="bbox_map", value=dict(annotation_bboxes))
        projected_annotation = {
            "type": "bbox_map",
            "bbox_map": dict(annotation_bboxes),
            "pixel_bbox_map": dict(annotation_bboxes),
            "value": dict(annotation_bboxes),
        }
        witness_symbolic = {"type": "bbox_map", "value": dict(annotation_bboxes)}
        annotation_source = "keyed_tray_bboxes_px"
    else:
        tray_bbox = list(annotation_bboxes["dice_tray"])
        annotation_gt = TypedValue(type="bbox", value=list(tray_bbox))
        projected_annotation = {
            "type": "bbox",
            "bbox": list(tray_bbox),
            "pixel_bbox": list(tray_bbox),
            "value": list(tray_bbox),
        }
        witness_symbolic = {"type": "bbox", "value": list(tray_bbox)}
        annotation_source = "tray_bbox_px"
    tray_specs = [dict(tray) for tray in dataset["tray_specs"]]
    all_dice = [dict(die) for tray in tray_specs for die in tray["dice"]]
    visual_scan_count = len(all_dice)
    mode = str(dataset["mode"])
    visual_scan_bounds = [8, 18] if mode != "pair" else [8, 12]
    outcome_count = int(event["total_outcome_count"])
    outcome_bounds = [8, 22] if mode in {"single", "conditional"} else [16, 36]
    visual_scan = normalize_int_with_bounds(int(visual_scan_count), visual_scan_bounds)
    outcome_load = normalize_int_with_bounds(int(outcome_count), outcome_bounds)
    reasoning_load = min(1.0, float(reasoning_load_base) + (0.12 * float(outcome_load)))

    trace_payload = {
        "scene_ir": {
            "scene_kind": "symbolic_probability_dice_panel",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "scene_id": SCENE_ID,
                "query_id": str(public_query_id),
                "event_key": str(prompt_query_key),
                "scene_variant": str(scene_variant),
                "answer_value": str(answer_value),
                "event_description": str(event["event_description"]),
            },
        },
        "query_spec": {
            "query_id": str(public_query_id),
            "template_id": str(prompt_meta["bundle_id"]),
            "prompt_variant": dict(prompt_meta["prompt_variant"]),
            "prompt_variant_active_key": str(prompt_meta["prompt_variant_active_key"]),
            "prompt_variants": dict(prompt_meta["prompt_variants_for_trace"]),
            "params": {
                "scene_id": SCENE_ID,
                "query_id": str(public_query_id),
                "query_id_probabilities": dict(query_probabilities),
                "event_key": str(prompt_query_key),
                "scene_variant": str(scene_variant),
                "scene_variant_probabilities": dict(scene_variant_probabilities),
                "dice_visual_style": str(dice_visual_style),
                "dice_visual_style_probabilities": dict(dice_visual_style_probabilities),
                "mode": str(mode),
                "event_description": str(event["event_description"]),
                "given_description": str(event.get("given_description", "")),
                "favorable_outcome_count": int(event["favorable_outcome_count"]),
                "total_outcome_count": int(event["total_outcome_count"]),
                "probability_fraction": str(probability_fraction),
                "option_labels": [str(label) for label in answer_options["labels"]],
                "correct_label": str(answer_options["correct_label"]),
            },
        },
        "render_spec": {
            "scene_id": SCENE_ID,
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_variant": str(scene_variant),
            "background_style": dict(background_meta),
            "scene_style": dict(scene_style_meta),
            "dice_visual_style": _dice_visual_style_metadata(
                str(dice_visual_style),
                dice_visual_style_probabilities,
            ),
            "post_image_noise": dict(post_noise_meta),
            "post_image_noise_policy": {
                "apply_prob": 0.15,
                "reason": "dice_color_and_pip_readability",
                "scope": "semantic die colors and visible pip counts",
            },
            "label_style": {
                "font": _font_trace_record(str(font_family)),
            },
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "layout": str(mode),
            "option_card_layout": {
                "option_labels": [str(label) for label in answer_options["labels"]],
                "option_y0_px": int(option_y0_px),
                "option_count": int(len(answer_options["labels"])),
            },
        },
        "render_map": {
            "image_id": "img0",
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "die_bboxes_px": {str(key): list(value) for key, value in rendered_scene.die_bbox_map.items()},
            "tray_bboxes_px": {str(key): list(value) for key, value in rendered_scene.tray_bbox_map.items()},
            "option_bboxes_px": dict(option_bboxes_px),
            "selected_option_label": str(answer_options["correct_label"]),
            "selected_option_bbox_px": list(option_bboxes_px[str(answer_options["correct_label"])]),
            "item_bboxes_px": {str(key): list(value) for key, value in rendered_scene.item_bbox_map.items()},
            "annotation_source": str(annotation_source),
        },
        "execution_trace": {
            "query_id": str(public_query_id),
            "query_id_probabilities": dict(query_probabilities),
            "event_key": str(prompt_query_key),
            "scene_id": SCENE_ID,
            "scene_variant": str(scene_variant),
            "scene_variant_probabilities": dict(scene_variant_probabilities),
            "dice_visual_style": str(dice_visual_style),
            "dice_visual_style_probabilities": dict(dice_visual_style_probabilities),
            "mode": str(mode),
            "tray_specs": tray_specs,
            "dice_specs": all_dice,
            "dice_attribute_counts": {
                "color": dict(Counter(str(die["color_name"]) for die in all_dice)),
                "value": {str(key): int(value) for key, value in Counter(int(die["value"]) for die in all_dice).items()},
            },
            "event": dict(event),
            "event_description": str(event["event_description"]),
            "given_description": str(event.get("given_description", "")),
            "favorable_outcome_count": int(event["favorable_outcome_count"]),
            "total_outcome_count": int(event["total_outcome_count"]),
            "answer_value": str(answer_value),
            "answer_label": str(answer_options["correct_label"]),
            "answer_type": "option_letter",
            "probability_fraction": str(probability_fraction),
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
            "annotation_item_ids": list(annotation_item_ids),
            "annotation_role_item_ids": dict(annotation_role_item_ids),
            "calculation_supporting_item_ids": list(calculation_supporting_item_ids),
            "question_format": str(prompt_query_key),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
        "answer_gt": answer_gt.to_dict(),
        "annotation_gt": annotation_gt.to_dict(),
    }
    if mode == "pair":
        trace_payload["execution_trace"]["dice_count_a"] = int(dataset["dice_count_a"])
        trace_payload["execution_trace"]["dice_count_b"] = int(dataset["dice_count_b"])
        trace_payload["execution_trace"]["dice_count_range"] = list(dataset["dice_count_range"])
    else:
        trace_payload["execution_trace"]["dice_count"] = int(dataset["dice_count"])
        trace_payload["execution_trace"]["dice_count_range"] = list(dataset["dice_count_range"])
    if mode == "conditional":
        trace_payload["execution_trace"]["denominator_supporting_item_ids"] = list(dataset["denominator_supporting_item_ids"])

    trace_payload["analysis"] = {
        "complexity_components": {
            "scene_load": float(_SCENE_LOAD[str(scene_variant)]),
            "visual_scan": float(visual_scan),
            "outcome_load": float(outcome_load),
            "reasoning_load": float(reasoning_load),
        }
    }

    return TaskOutput(
        prompt=str(prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(public_query_id),
        prompt_variants=dict(prompt_variants),
    )


def run_dice_probability_lifecycle(
    *,
    public_task_id: str,
    supported_queries: Sequence[str],
    prompt_query_key: str | None,
    dataset_builder: Callable[..., Mapping[str, Any]],
    reasoning_load_base: float | Mapping[str, float],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run shared rendering plumbing after a public task supplies its dataset hook.

    Public files still own query support, event construction, and the dataset
    builder. This helper only removes duplicated defaults/query/render glue for
    single-branch dice-probability tasks.
    """

    del max_attempts
    gen_defaults, render_defaults, prompt_defaults = load_dice_probability_defaults(str(public_task_id))
    public_query_id, query_probabilities = resolve_dice_query(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        public_task_id=str(public_task_id),
        supported_queries=supported_queries,
    )
    dataset = dataset_builder(
        public_query_id=str(public_query_id),
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
    )
    resolved_prompt_query_key = str(public_query_id) if prompt_query_key is None else str(prompt_query_key)
    resolved_reasoning_load = (
        float(reasoning_load_base[str(public_query_id)])
        if isinstance(reasoning_load_base, Mapping)
        else float(reasoning_load_base)
    )
    return build_dice_probability_output(
        public_task_id=str(public_task_id),
        public_query_id=str(public_query_id),
        prompt_query_key=str(resolved_prompt_query_key),
        query_probabilities=query_probabilities,
        dataset=dataset,
        params=params,
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
        reasoning_load_base=float(resolved_reasoning_load),
    )


__all__ = [
    "SCENE_ID",
    "build_dice_probability_output",
    "load_dice_probability_defaults",
    "resolve_dice_query",
    "run_dice_probability_lifecycle",
]
