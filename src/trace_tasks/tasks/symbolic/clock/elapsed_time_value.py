"""Elapsed-time value task for two labeled analog clocks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Tuple

from PIL import ImageDraw

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.sampling import uniform_choice_with_probabilities
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.annotation_artifacts import bbox_annotation_artifacts
from ...registry import register_task
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults, required_group_defaults
from ...shared.fixed_query import select_task_query_id
from ...shared.font_assets import font_asset_version, sample_font_family
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_task_prompt_variants
from ...shared.text_rendering import draw_text_centered, load_font, temporary_default_font_family
from ...shared.time_artifact_style import (
    SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES,
    SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS,
    build_time_artifact_clock_theme,
)
from ...shared.time_artifact_task_support import resolve_time_artifact_named_variant
from ...shared.time_format import add_clock_minutes, clock_total_minutes, format_clock_hhmm, split_clock_total_minutes
from ..shared.scene_style import make_symbolic_scene_background, resolve_symbolic_scene_style
from ..shared.visual_defaults import load_symbolic_background_defaults, load_symbolic_noise_defaults
from .shared.rendering import draw_clock_geometry, draw_text_option_cards, option_cards_y_below_bbox
from .shared.sampling import (
    feasible_clock_times,
    nearby_integer_distractors,
    option_value_map,
    resolve_clock_time_support,
    resolve_text_option_labels,
    sample_correct_option_label,
)
from .shared.state import SUPPORTED_SYMBOLIC_CLOCK_SCENE_VARIANTS, ClockRenderParams
from .shared.styles import resolve_clock_render_params, scale_clock_render_params_for_radius


DOMAIN = "symbolic"
SCENE_ID = "clock"
TASK_ID = "task_symbolic__clock__elapsed_time_value"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "elapsed_time_value"


@dataclass(frozen=True)
class _TaskDefaults:
    """Stable fallback defaults for two-clock elapsed-time scenes."""

    hour_min: int = 1
    hour_max: int = 12
    minute_min: int = 0
    minute_max: int = 55
    minute_step: int = 5
    min_hand_angle_gap_deg: float = 10.0
    elapsed_minutes_min: int = 15
    elapsed_minutes_max: int = 360
    elapsed_minutes_step: int = 15
    canvas_width: int = 880
    canvas_height: int = 520
    outer_margin_px: int = 40
    face_radius_px: int = 118
    bezel_width_px: int = 8
    numeral_font_size_px: int = 18
    major_tick_length_px: int = 14
    minor_tick_length_px: int = 6
    major_tick_width_px: int = 3
    minor_tick_width_px: int = 2
    minor_tick_dot_radius_px: int = 2
    hour_hand_width_px: int = 8
    minute_hand_width_px: int = 6
    second_hand_width_px: int = 2
    hand_bbox_padding_px: int = 5
    center_dot_radius_px: int = 6
    inner_ring_inset_px: int = 14
    inner_ring_width_px: int = 3
    card_width_px: int = 300
    card_height_px: int = 330
    card_gap_px: int = 80
    label_font_size_px: int = 34
    caption_font_size_px: int = 18


@dataclass(frozen=True)
class _ResolvedQuery:
    """Resolved semantic and visual support for one elapsed-time instance."""

    query_id: str
    scene_variant: str
    style_variant: str
    accent_color_name: str
    start_total_minutes: int
    end_total_minutes: int
    elapsed_minutes: int
    elapsed_minutes_support: Tuple[int, ...]
    hour_support: Tuple[int, int]
    minute_support: Tuple[int, int, int]
    min_hand_angle_gap_deg: float
    query_id_probabilities: Dict[str, float]
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]


_DEFAULTS = _TaskDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_symbolic_background_defaults(scene_id="clock")
POST_IMAGE_NOISE_DEFAULTS = {
    **load_symbolic_noise_defaults(scene_id="clock", apply_prob=0.45),
    "apply_prob": 0.45,
}


def _resolve_named_variant(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Tuple[str, ...],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced named symbolic-clock axis."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.{namespace}")
    return resolve_time_artifact_named_variant(
        rng,
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported=supported,
        instance_seed=int(instance_seed),
        task_id=TASK_ID,
        namespace=str(namespace),
    )


def _range_support(*, min_value: int, max_value: int, step: int) -> Tuple[int, ...]:
    """Return an inclusive integer support."""

    if int(step) <= 0:
        raise ValueError("clock elapsed-time support step must be positive")
    if int(min_value) > int(max_value):
        raise ValueError("clock elapsed-time support min must be <= max")
    return tuple(range(int(min_value), int(max_value) + 1, int(step)))


def _resolve_elapsed_support(params: Mapping[str, Any]) -> Tuple[int, ...]:
    """Resolve supported forward elapsed-minute values."""

    raw = params.get("elapsed_minutes_support", group_default(_GEN_DEFAULTS, "elapsed_minutes_support", None))
    if isinstance(raw, Mapping):
        min_value = int(raw.get("min", _DEFAULTS.elapsed_minutes_min))
        max_value = int(raw.get("max", _DEFAULTS.elapsed_minutes_max))
        step = int(raw.get("step", _DEFAULTS.elapsed_minutes_step))
        support = _range_support(min_value=min_value, max_value=max_value, step=step)
    elif raw is not None:
        support = tuple(int(value) for value in raw)
    else:
        support = _range_support(
            min_value=_DEFAULTS.elapsed_minutes_min,
            max_value=_DEFAULTS.elapsed_minutes_max,
            step=_DEFAULTS.elapsed_minutes_step,
        )
    support = tuple(value for value in dict.fromkeys(int(value) for value in support) if 0 < int(value) < 720)
    if len(support) < 4:
        raise ValueError("symbolic clock elapsed-time support must contain at least four nonzero values below 720")
    return support


def _resolve_query(instance_seed: int, *, params: Mapping[str, Any]) -> _ResolvedQuery:
    """Resolve the elapsed-time scene and answer."""

    query_id, query_probs, _ = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id=SINGLE_QUERY_ID,
        task_id=TASK_ID,
        namespace=f"{TASK_ID}.query",
    )
    scene_variant, scene_probs = _resolve_named_variant(
        instance_seed=int(instance_seed),
        params=params,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_SYMBOLIC_CLOCK_SCENE_VARIANTS,
        namespace="scene_variant",
    )
    style_variant, style_probs = _resolve_named_variant(
        instance_seed=int(instance_seed),
        params=params,
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS,
        namespace="style_variant",
    )
    accent_color_name, accent_probs = _resolve_named_variant(
        instance_seed=int(instance_seed),
        params=params,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        balance_flag_key="balanced_accent_color_name_sampling",
        supported=SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES,
        namespace="accent_color_name",
    )
    hour_support, minute_support, raw_times = resolve_clock_time_support(
        params,
        gen_defaults=_GEN_DEFAULTS,
        fallback_hour_min=_DEFAULTS.hour_min,
        fallback_hour_max=_DEFAULTS.hour_max,
        fallback_minute_min=_DEFAULTS.minute_min,
        fallback_minute_max=_DEFAULTS.minute_max,
        fallback_minute_step=_DEFAULTS.minute_step,
        context="symbolic clock elapsed-time",
    )
    min_gap = float(params.get("min_hand_angle_gap_deg", group_default(_GEN_DEFAULTS, "min_hand_angle_gap_deg", _DEFAULTS.min_hand_angle_gap_deg)))
    time_support = feasible_clock_times(raw_times, min_hand_angle_gap_deg=float(min_gap))
    if len(time_support) < 8:
        raise ValueError("symbolic clock elapsed-time requires at least eight feasible shown times")
    elapsed_support = _resolve_elapsed_support(params)
    explicit_duration = params.get("elapsed_minutes", params.get("duration_minutes"))
    if explicit_duration is not None:
        elapsed_minutes = int(explicit_duration)
        if elapsed_minutes not in elapsed_support:
            raise ValueError("explicit elapsed_minutes is outside elapsed_minutes_support")
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.elapsed_minutes")
        elapsed_minutes, _elapsed_probs = uniform_choice_with_probabilities(
            rng,
            elapsed_support,
            sort_keys=True,
        )
        elapsed_minutes = int(elapsed_minutes)

    explicit_start_total = params.get("start_total_minutes")
    if explicit_start_total is None and (params.get("start_hour") is not None or params.get("start_minute") is not None):
        explicit_start_total = clock_total_minutes(int(params.get("start_hour", 1)), int(params.get("start_minute", 0)))
    if explicit_start_total is not None:
        start_total = int(explicit_start_total) % 720
        end_total = add_clock_minutes(start_total, int(elapsed_minutes))
        if start_total not in time_support or end_total not in time_support:
            raise ValueError("explicit start time and elapsed duration do not produce feasible displayed clocks")
    else:
        start_candidates = tuple(total for total in time_support if add_clock_minutes(int(total), int(elapsed_minutes)) in time_support)
        if not start_candidates:
            raise ValueError("elapsed_minutes_support produced no feasible start/end clock pair")
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.start_time")
        start_total, _start_probs = uniform_choice_with_probabilities(
            rng,
            start_candidates,
            sort_keys=True,
        )
        start_total = int(start_total)
        end_total = int(add_clock_minutes(start_total, int(elapsed_minutes)))

    return _ResolvedQuery(
        query_id=str(query_id),
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        accent_color_name=str(accent_color_name),
        start_total_minutes=int(start_total),
        end_total_minutes=int(end_total),
        elapsed_minutes=int(elapsed_minutes),
        elapsed_minutes_support=tuple(int(value) for value in elapsed_support),
        hour_support=hour_support,
        minute_support=minute_support,
        min_hand_angle_gap_deg=float(min_gap),
        query_id_probabilities=dict(query_probs),
        scene_variant_probabilities=dict(scene_probs),
        style_variant_probabilities=dict(style_probs),
        accent_color_name_probabilities=dict(accent_probs),
    )


def _clock_card_bboxes(render_params: ClockRenderParams, *, card_width: int, card_height: int, card_gap: int) -> Dict[str, Tuple[float, float, float, float]]:
    """Return fixed side-by-side A/B card bboxes."""

    total_width = (2 * float(card_width)) + float(card_gap)
    start_x = 0.5 * (float(render_params.canvas_width) - total_width)
    y0 = 105.0
    return {
        "A": (float(start_x), y0, float(start_x + card_width), float(y0 + card_height)),
        "B": (float(start_x + card_width + card_gap), y0, float(start_x + (2 * card_width) + card_gap), float(y0 + card_height)),
    }


def _round_bbox(bbox: Tuple[float, float, float, float]) -> List[float]:
    """Return one rounded bbox."""

    return [round(float(value), 3) for value in bbox]


def _union_bbox(boxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    """Return the union of non-empty bbox list."""

    return (
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    )


def _build_prompt_json_examples() -> tuple[str, str]:
    """Return prompt JSON examples for elapsed-time value."""

    answer_and_annotation = {
        "annotation": [304, 670, 436, 736],
        "answer": "C",
    }
    answer_only = {"answer": "C"}
    return (
        json.dumps(answer_and_annotation, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        json.dumps(answer_only, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
    )


@register_task
class SymbolicClockElapsedTimeValueTask:
    """Compute the forward elapsed minutes between two labeled analog clocks."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Render two clocks and bind one integer elapsed-minute answer.

        The key invariant is that answer, annotation, prompt, and trace all use
        the same finalized A/B clock pair after objective sampling.
        """

        del max_attempts
        query = _resolve_query(int(instance_seed), params=params)
        render_params = resolve_clock_render_params(
            params,
            render_defaults=_RENDER_DEFAULTS,
            fallback_values=asdict(_DEFAULTS),
            instance_seed=int(instance_seed),
        )
        face_radius_px = int(params.get("face_radius_px", group_default(_RENDER_DEFAULTS, "face_radius_px", _DEFAULTS.face_radius_px)))
        card_width_px = int(params.get("card_width_px", group_default(_RENDER_DEFAULTS, "card_width_px", _DEFAULTS.card_width_px)))
        card_height_px = int(params.get("card_height_px", group_default(_RENDER_DEFAULTS, "card_height_px", _DEFAULTS.card_height_px)))
        card_gap_px = int(params.get("card_gap_px", group_default(_RENDER_DEFAULTS, "card_gap_px", _DEFAULTS.card_gap_px)))
        label_font_size_px = int(params.get("label_font_size_px", group_default(_RENDER_DEFAULTS, "label_font_size_px", _DEFAULTS.label_font_size_px)))
        caption_font_size_px = int(params.get("caption_font_size_px", group_default(_RENDER_DEFAULTS, "caption_font_size_px", _DEFAULTS.caption_font_size_px)))

        clock_theme = build_time_artifact_clock_theme(
            accent_color_name=str(query.accent_color_name),
            style_variant=str(query.style_variant),
        )
        font_family = sample_font_family(
            role="readout",
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.font",
            params={**dict(_RENDER_DEFAULTS), **dict(params)},
        )
        scene_style, scene_style_meta = resolve_symbolic_scene_style(
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.background",
        )
        background, background_meta = make_symbolic_scene_background(
            canvas_width=int(render_params.canvas_width),
            canvas_height=int(render_params.canvas_height),
            style=scene_style,
        )
        image = background.copy().convert("RGB")
        draw = ImageDraw.Draw(image)

        card_bboxes = _clock_card_bboxes(
            render_params,
            card_width=int(card_width_px),
            card_height=int(card_height_px),
            card_gap=int(card_gap_px),
        )
        face_bboxes: Dict[str, List[float]] = {}
        scene_entities: List[Dict[str, Any]] = []
        scaled_params = scale_clock_render_params_for_radius(render_params, radius_px=int(face_radius_px))
        with temporary_default_font_family(str(font_family)):
            label_font = load_font(int(label_font_size_px), bold=True)
            caption_font = load_font(int(caption_font_size_px), bold=False)
            for label, shown_total in (("A", query.start_total_minutes), ("B", query.end_total_minutes)):
                card = card_bboxes[str(label)]
                draw.rounded_rectangle(card, radius=20, fill=(252, 252, 250), outline=(166, 174, 184), width=2)
                draw_text_centered(
                    draw,
                    text=str(label),
                    center=((float(card[0]) + float(card[2])) / 2.0, float(card[1] + 36.0)),
                    font=label_font,
                    fill=(34, 40, 49),
                )
                draw_text_centered(
                    draw,
                    text="start" if label == "A" else "end",
                    center=((float(card[0]) + float(card[2])) / 2.0, float(card[3] - 24.0)),
                    font=caption_font,
                    fill=(82, 90, 102),
                )
                center = ((float(card[0]) + float(card[2])) / 2.0, float(card[1] + 198.0))
                geometry = draw_clock_geometry(
                    image,
                    center_px=center,
                    face_radius_px=float(face_radius_px),
                    scene_variant=str(query.scene_variant),
                    shown_total_minutes=int(shown_total),
                    render_params=scaled_params,
                    visual_theme=clock_theme,
                    entity_prefix=f"clock_{str(label).lower()}_",
                    extra_face_attrs={
                        "clock_label": str(label),
                        "role": "start_clock" if label == "A" else "end_clock",
                        "shown_time_text": str(format_clock_hhmm(int(shown_total))),
                    },
                )
                face_bboxes[str(label)] = _round_bbox(tuple(float(value) for value in geometry.face_bbox_px))
                scene_entities.extend([dict(entity) for entity in geometry.entities])

            option_labels = resolve_text_option_labels(params, gen_defaults=_GEN_DEFAULTS)
            correct_label, label_probs = sample_correct_option_label(
                params=params,
                gen_defaults=_GEN_DEFAULTS,
                instance_seed=int(instance_seed),
                seed_namespace=TASK_ID,
                labels=option_labels,
            )
            distractors = nearby_integer_distractors(
                correct_value=int(query.elapsed_minutes),
                support_values=query.elapsed_minutes_support,
                preferred_offsets=(15, 30, 45, 60, 90, 120),
                min_value=min(int(value) for value in query.elapsed_minutes_support),
                max_value=max(int(value) for value in query.elapsed_minutes_support),
            )
            option_values = option_value_map(
                labels=option_labels,
                correct_label=str(correct_label),
                correct_value=int(query.elapsed_minutes),
                distractors=distractors,
            )
            option_text = {str(label): str(value) for label, value in option_values.items()}
            clock_panel_bbox = _union_bbox(
                [tuple(float(value) for value in box) for box in card_bboxes.values()]
            )
            raw_option_bboxes, option_entities = draw_text_option_cards(
                image,
                text_by_label=option_text,
                correct_label=str(correct_label),
                y0_px=option_cards_y_below_bbox(
                    clock_panel_bbox,
                    canvas_height=int(render_params.canvas_height),
                ),
            )
            option_bboxes_px = {
                str(label): _round_bbox(tuple(float(value) for value in bbox))
                for label, bbox in raw_option_bboxes.items()
            }
            selected_option_bbox_px = list(option_bboxes_px[str(correct_label)])
            scene_entities.extend([dict(entity) for entity in option_entities])

        image, post_noise_meta = apply_post_image_noise(
            image,
            instance_seed=int(instance_seed),
            params=params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )

        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "object_description_elapsed_time_value_classic",
                "object_description_elapsed_time_value_minimal",
                "object_description_elapsed_time_value_outline",
                "annotation_hint_elapsed_time_value",
                "answer_hint_elapsed_time_value",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        object_description = str(prompt_defaults[f"object_description_elapsed_time_value_{str(query.scene_variant)}"])
        json_example, json_example_answer_only = _build_prompt_json_examples()
        prompt_selection = render_task_prompt_variants(
            domain=DOMAIN,
            scene_id=SCENE_ID,
            bundle_id=str(prompt_defaults["bundle_id"]),
            scene_key=str(prompt_defaults["scene_key"]),
            task_key=str(prompt_defaults["task_key"]),
            query_key=PROMPT_QUERY_KEY,
            answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
            dynamic_slots={
                "object_description": str(object_description),
                "json_output_contract": str(prompt_defaults["json_output_contract"]),
                "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
                "annotation_hint": str(prompt_defaults["annotation_hint_elapsed_time_value"]),
                "answer_hint": str(prompt_defaults["answer_hint_elapsed_time_value"]),
                "json_example": str(json_example),
                "json_example_answer_only": str(json_example_answer_only),
            },
            instance_seed=int(instance_seed),
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

        annotation_bboxes = {
            "start_clock": list(face_bboxes["A"]),
            "end_clock": list(face_bboxes["B"]),
        }
        answer_gt = TypedValue(type="option_letter", value=str(correct_label))
        annotation_payload = bbox_annotation_artifacts(selected_option_bbox_px)
        annotation_gt = annotation_payload.annotation_gt
        scene_bbox = _union_bbox(
            [
                *[tuple(float(value) for value in box) for box in card_bboxes.values()],
                *[tuple(float(value) for value in box) for box in raw_option_bboxes.values()],
            ]
        )
        start_hour, start_minute = split_clock_total_minutes(int(query.start_total_minutes))
        end_hour, end_minute = split_clock_total_minutes(int(query.end_total_minutes))

        trace_payload = {
            "scene_ir": {
                "scene_kind": "symbolic_clock_elapsed_time_pair",
                "entities": [dict(entity) for entity in scene_entities],
                "relations": {
                    "query_id": str(query.query_id),
                    "start_clock_label": "A",
                    "end_clock_label": "B",
                    "elapsed_minutes": int(query.elapsed_minutes),
                    "answer_label": str(correct_label),
                },
            },
            "query_spec": {
                "query_id": str(query.query_id),
                "template_id": str(prompt_defaults["bundle_id"]),
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "query_id": str(query.query_id),
                    "prompt_query_key": PROMPT_QUERY_KEY,
                    "scene_variant": str(query.scene_variant),
                    "style_variant": str(query.style_variant),
                    "accent_color_name": str(query.accent_color_name),
                    "elapsed_minutes_support": [int(value) for value in query.elapsed_minutes_support],
                    "option_labels": [str(label) for label in option_labels],
                    "correct_label": str(correct_label),
                    "correct_label_probabilities": {str(key): float(value) for key, value in label_probs.items()},
                    "hour_support": [int(query.hour_support[0]), int(query.hour_support[1])],
                    "minute_support": [int(value) for value in query.minute_support],
                    "min_hand_angle_gap_deg": float(query.min_hand_angle_gap_deg),
                    "query_id_probabilities": dict(query.query_id_probabilities),
                    "scene_variant_probabilities": dict(query.scene_variant_probabilities),
                    "style_variant_probabilities": dict(query.style_variant_probabilities),
                    "accent_color_name_probabilities": dict(query.accent_color_name_probabilities),
                },
            },
            "render_spec": {
                "canvas_width": int(render_params.canvas_width),
                "canvas_height": int(render_params.canvas_height),
                "coord_space": "pixel",
                "scene_variant": str(query.scene_variant),
                "background_style": dict(background_meta),
                "scene_style": dict(scene_style_meta),
                "post_image_noise": dict(post_noise_meta),
                "scene_bbox_px": _round_bbox(scene_bbox),
                "clock_style": {
                    "accent_color_name": str(query.accent_color_name),
                    "style_variant": str(query.style_variant),
                    "face_radius_px": int(face_radius_px),
                    "font": {
                        "source": "global_font_pool",
                        "font_family": str(font_family),
                        "font_asset_version": font_asset_version(),
                        "scope": "clock_elapsed_time_pair",
                    },
                    "resolved_colors_rgb": {
                        "face_fill": [int(value) for value in clock_theme.face_fill_rgb],
                        "face_outline": [int(value) for value in clock_theme.face_outline_rgb],
                        "numerals": [int(value) for value in clock_theme.numeral_color_rgb],
                        "ticks": [int(value) for value in clock_theme.tick_color_rgb],
                        "hour_hand": [int(value) for value in clock_theme.hour_hand_color_rgb],
                        "minute_hand": [int(value) for value in clock_theme.minute_hand_color_rgb],
                        "center_dot": [int(value) for value in clock_theme.center_dot_color_rgb],
                    },
                    "minor_tick_mode": str(clock_theme.minor_tick_mode),
                },
            },
            "render_map": {
                "image_id": "img0",
                "scene_bbox_px": _round_bbox(scene_bbox),
                "card_bboxes_px": {str(label): _round_bbox(tuple(float(value) for value in bbox)) for label, bbox in card_bboxes.items()},
                "clock_face_bboxes_px": dict(face_bboxes),
                "start_clock_bbox_px": list(annotation_bboxes["start_clock"]),
                "end_clock_bbox_px": list(annotation_bboxes["end_clock"]),
                "option_bboxes_px": dict(option_bboxes_px),
                "selected_option_label": str(correct_label),
                "selected_option_bbox_px": list(selected_option_bbox_px),
                "annotation_source": "selected_answer_option_bbox_px",
            },
            "execution_trace": {
                "query_id": str(query.query_id),
                "prompt_query_key": PROMPT_QUERY_KEY,
                "scene_variant": str(query.scene_variant),
                "style_variant": str(query.style_variant),
                "accent_color_name": str(query.accent_color_name),
                "start_total_minutes": int(query.start_total_minutes),
                "start_hour": int(start_hour),
                "start_minute": int(start_minute),
                "start_time_text": str(format_clock_hhmm(int(query.start_total_minutes))),
                "end_total_minutes": int(query.end_total_minutes),
                "end_hour": int(end_hour),
                "end_minute": int(end_minute),
                "end_time_text": str(format_clock_hhmm(int(query.end_total_minutes))),
                "elapsed_minutes": int(query.elapsed_minutes),
                "answer_value": int(query.elapsed_minutes),
                "answer_label": str(correct_label),
                "option_values_by_label": {str(key): int(value) for key, value in option_values.items()},
                "option_text_by_label": dict(option_text),
                "answer_type": "option_letter",
                "elapsed_minutes_support": [int(value) for value in query.elapsed_minutes_support],
                "hour_support": [int(query.hour_support[0]), int(query.hour_support[1])],
                "minute_support": [int(value) for value in query.minute_support],
                "min_hand_angle_gap_deg": float(query.min_hand_angle_gap_deg),
                "question_format": PROMPT_QUERY_KEY,
                "supporting_bbox_roles": ["selected_answer_option"],
                "source_clock_bboxes_px": dict(annotation_bboxes),
            },
            "witness_symbolic": {
                "type": str(annotation_payload.annotation_type),
                "value": list(annotation_payload.value),
            },
            "projected_annotation": dict(annotation_payload.projected_annotation),
            "answer_gt": answer_gt.to_dict(),
            "annotation_gt": annotation_gt.to_dict(),
        }

        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(query.query_id),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


__all__ = ["SymbolicClockElapsedTimeValueTask"]
