"""Apply a minute offset to the time shown on one clock display."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Tuple

from ....core.seed import spawn_rng
from ....core.sampling import uniform_choice_with_probabilities
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.annotation_artifacts import bbox_annotation_artifacts
from ...registry import register_task
from ...shared.config_defaults import (
    group_default,
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from ...shared.fixed_query import select_task_query_id
from ...shared.font_assets import font_asset_version, sample_font_family
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ...shared.text_rendering import temporary_default_font_family
from ...shared.time_artifact_style import (
    SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES,
    SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS,
    build_time_artifact_clock_theme,
)
from ...shared.time_format import (
    add_clock_minutes,
    clock_hand_angle_gap_deg,
    clock_total_minutes,
    format_clock_hhmm,
    split_clock_total_minutes,
)
from ..shared.common import resolve_symbolic_axis_variant
from ..shared.scene_style import make_symbolic_scene_background, resolve_symbolic_scene_style

from .shared.annotations import clock_hand_segment_annotations
from .shared.defaults import DEFAULTS, POST_IMAGE_NOISE_DEFAULTS
from .shared.rendering import draw_text_option_cards, option_cards_y_below_bbox, render_clock_scene
from .shared.sampling import (
    option_value_map,
    resolve_text_option_labels,
    sample_correct_option_label,
)
from .shared.state import SUPPORTED_SYMBOLIC_CLOCK_SCENE_VARIANTS, ClockStyleResolution
from .shared.styles import resolve_clock_render_params


DOMAIN = "symbolic"
SCENE_ID = "clock"
TASK_ID = "task_symbolic__clock__offset_readout"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("minutes_after", "minutes_before")
TASK_PROMPT_KEY = "clock_offset_readout_query"
QUESTION_FORMAT = "minute_offset_readout"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


@dataclass(frozen=True)
class _ResolvedQuery:
    """Resolved objective and visual support for one clock sample."""

    query_id: str
    offset_direction: str
    shown_total_minutes: int
    shown_hour: int
    shown_minute: int
    delta_minutes: int
    answer_total_minutes: int
    answer_time_text: str
    hour_support: Tuple[int, int]
    minute_support: Tuple[int, int, int]
    delta_minutes_support: Tuple[int, int, int]
    min_hand_angle_gap_deg: float
    query_id_probabilities: Dict[str, float]


def _resolve_style_axis(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    supported_variants: Tuple[str, ...],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    axis_namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one non-query visual axis."""

    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=supported_variants,
        task_id=TASK_ID,
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        axis_namespace=str(axis_namespace),
    )


def _select_public_query(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
) -> tuple[str, Dict[str, float], Dict[str, Any]]:
    """Select the before/after public query with the shared policy helper."""

    params_with_direction = dict(params)
    explicit_direction = params_with_direction.get("offset_direction", params_with_direction.get("direction"))
    if explicit_direction is not None and "query_id" not in params_with_direction and "query_variant" not in params_with_direction:
        direction_text = str(explicit_direction).strip().lower()
        if direction_text not in {"after", "before"}:
            raise ValueError(f"unsupported offset_direction: {explicit_direction}")
        params_with_direction["query_id"] = f"minutes_{direction_text}"

    return select_task_query_id(
        instance_seed=int(instance_seed),
        params=params_with_direction,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id="minutes_after",
        task_id=TASK_ID,
        namespace=f"{TASK_ID}.query",
    )


def _offset_support_from_config(raw_support: Any) -> Tuple[int, int, int]:
    """Resolve one compact positive minute-offset support."""

    if not isinstance(raw_support, Mapping):
        raise ValueError("delta_minutes_support must be a mapping with min, max, and step")
    support_min = int(raw_support.get("min", raw_support.get("start", 0)))
    support_max = int(raw_support.get("max", raw_support.get("stop", support_min)))
    support_step = int(raw_support.get("step", 1))
    if support_step <= 0:
        raise ValueError("delta_minutes_support step must be positive")
    if support_min <= 0 or support_max < support_min:
        raise ValueError("delta_minutes_support must have positive min <= max")
    if (support_max - support_min) % support_step != 0:
        raise ValueError("delta_minutes_support max must align to min + n*step")
    return int(support_min), int(support_max), int(support_step)


def _offset_support_count(support: Tuple[int, int, int]) -> int:
    """Return the number of possible values in a compact offset support."""

    return int(((int(support[1]) - int(support[0])) // int(support[2])) + 1)


def _offset_support_values(support: Tuple[int, int, int]) -> Tuple[int, ...]:
    """Expand one compact offset support into explicit values."""

    return tuple(
        int(support[0]) + (index * int(support[2]))
        for index in range(_offset_support_count(support))
    )


def _offset_support_contains(support: Tuple[int, int, int], value: int) -> bool:
    """Return whether one minute offset is in the configured support."""

    candidate = int(value)
    return int(support[0]) <= candidate <= int(support[1]) and ((candidate - int(support[0])) % int(support[2])) == 0


def _resolve_clock_style(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
) -> ClockStyleResolution:
    """Resolve visual clock axes for one sample."""

    scene_variant, scene_variant_probabilities = _resolve_style_axis(
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SYMBOLIC_CLOCK_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )
    style_variant, style_variant_probabilities = _resolve_style_axis(
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS,
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        axis_namespace="style_variant",
    )
    accent_color_name, accent_color_name_probabilities = _resolve_style_axis(
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        balance_flag_key="balanced_accent_color_name_sampling",
        axis_namespace="accent_color_name",
    )
    return ClockStyleResolution(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        accent_color_name=str(accent_color_name),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        accent_color_name_probabilities=dict(accent_color_name_probabilities),
    )


def _resolve_query(instance_seed: int, *, params: Mapping[str, Any]) -> tuple[_ResolvedQuery, Dict[str, Any]]:
    """Resolve the public query, shown time, offset, and answer."""

    query_id, query_id_probabilities, task_params = _select_public_query(
        instance_seed=int(instance_seed),
        params=params,
    )
    offset_unit = task_params.get("offset_unit", task_params.get("unit", "minutes"))
    if str(offset_unit).strip().lower() not in {"minute", "minutes"}:
        raise ValueError("clock offset_readout supports minute offsets only")
    offset_direction = "after" if str(query_id) == "minutes_after" else "before"

    hour_min = int(task_params.get("hour_min", group_default(_GEN_DEFAULTS, "hour_min", DEFAULTS.hour_min)))
    hour_max = int(task_params.get("hour_max", group_default(_GEN_DEFAULTS, "hour_max", DEFAULTS.hour_max)))
    minute_min = int(task_params.get("minute_min", group_default(_GEN_DEFAULTS, "minute_min", DEFAULTS.minute_min)))
    minute_max = int(task_params.get("minute_max", group_default(_GEN_DEFAULTS, "minute_max", DEFAULTS.minute_max)))
    minute_step = int(task_params.get("minute_step", group_default(_GEN_DEFAULTS, "minute_step", DEFAULTS.minute_step)))
    min_hand_angle_gap_deg = float(
        task_params.get(
            "min_hand_angle_gap_deg",
            group_default(_GEN_DEFAULTS, "min_hand_angle_gap_deg", DEFAULTS.min_hand_angle_gap_deg),
        )
    )
    if minute_step <= 0:
        raise ValueError("minute_step must be positive for symbolic clock tasks")
    if float(min_hand_angle_gap_deg) < 0.0:
        raise ValueError("min_hand_angle_gap_deg must be non-negative for symbolic clock tasks")

    minute_support = tuple(range(int(minute_min), int(minute_max) + 1, int(minute_step)))
    if not minute_support:
        raise ValueError("minute support is empty for symbolic clock tasks")
    if minute_support[0] < 0 or minute_support[-1] > 59:
        raise ValueError("minute support must stay within 0..59 for symbolic clock tasks")
    hour_support = tuple(range(int(hour_min), int(hour_max) + 1))
    if not hour_support:
        raise ValueError("hour support is empty for symbolic clock tasks")
    if hour_support[0] < 1 or hour_support[-1] > 12:
        raise ValueError("hour support must stay within 1..12 for symbolic clock tasks")

    shown_total_minute_support = tuple(
        clock_total_minutes(int(hour), int(minute))
        for hour in hour_support
        for minute in minute_support
        if float(clock_hand_angle_gap_deg(clock_total_minutes(int(hour), int(minute)))) >= float(min_hand_angle_gap_deg)
    )
    if not shown_total_minute_support:
        raise ValueError("shown time support is empty after clock-hand angle-gap filtering")

    explicit_hour = task_params.get("shown_hour")
    explicit_minute = task_params.get("shown_minute")
    explicit_total = task_params.get("shown_total_minutes")
    if explicit_total is not None:
        shown_total_minutes = int(explicit_total)
    elif explicit_hour is not None or explicit_minute is not None:
        if explicit_hour is None or explicit_minute is None:
            raise ValueError("shown_hour and shown_minute must be provided together")
        shown_total_minutes = clock_total_minutes(int(explicit_hour), int(explicit_minute))
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.shown_total_minutes")
        shown_total_minutes, _shown_probs = uniform_choice_with_probabilities(
            rng,
            shown_total_minute_support,
            sort_keys=True,
        )
        shown_total_minutes = int(shown_total_minutes)
    if int(shown_total_minutes) not in shown_total_minute_support:
        raise ValueError("shown time is outside configured support for symbolic clock tasks")

    delta_support_raw = task_params.get("delta_minutes_support", group_default(_GEN_DEFAULTS, "delta_minutes_support", ()))
    delta_support = _offset_support_from_config(delta_support_raw)
    explicit_delta = task_params.get("delta_minutes")
    if explicit_delta is not None:
        delta_minutes = int(explicit_delta)
        if not _offset_support_contains(delta_support, int(delta_minutes)):
            raise ValueError("delta_minutes is outside configured support for symbolic clock tasks")
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.delta_minutes")
        delta_minutes, _delta_probs = uniform_choice_with_probabilities(
            rng,
            _offset_support_values(delta_support),
            sort_keys=True,
        )
        delta_minutes = int(delta_minutes)

    signed_delta = int(delta_minutes) if offset_direction == "after" else -int(delta_minutes)
    answer_total_minutes = add_clock_minutes(int(shown_total_minutes), int(signed_delta))
    shown_hour, shown_minute = split_clock_total_minutes(int(shown_total_minutes))

    return (
        _ResolvedQuery(
            query_id=str(query_id),
            offset_direction=str(offset_direction),
            shown_total_minutes=int(shown_total_minutes),
            shown_hour=int(shown_hour),
            shown_minute=int(shown_minute),
            delta_minutes=int(delta_minutes),
            answer_total_minutes=int(answer_total_minutes),
            answer_time_text=str(format_clock_hhmm(int(answer_total_minutes))),
            hour_support=(int(hour_support[0]), int(hour_support[-1])),
            minute_support=(int(minute_support[0]), int(minute_support[-1]), int(minute_step)),
            delta_minutes_support=tuple(int(value) for value in delta_support),
            min_hand_angle_gap_deg=float(min_hand_angle_gap_deg),
            query_id_probabilities=dict(query_id_probabilities),
        ),
        dict(task_params),
    )


def _prompt_examples(*, offset_direction: str, delta_minutes: int) -> tuple[str, str]:
    """Return prompt JSON examples that match the selected offset direction."""

    answer_and_annotation = {
        "annotation": [224, 770, 316, 836],
        "answer": "C",
    }
    answer_only = {"answer": "C"}
    return (
        json.dumps(answer_and_annotation, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        json.dumps(answer_only, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
    )


def _build_prompt(
    *,
    prompt_defaults: Mapping[str, Any],
    style: ClockStyleResolution,
    query: _ResolvedQuery,
    instance_seed: int,
) -> tuple[str, dict[str, str], dict[str, Any], Any]:
    """Render the v1 prompt bundle for the offset task."""

    prompt_values = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            f"object_description_offset_readout_{style.scene_variant}",
            "json_output_contract",
            "json_output_contract_answer_only",
            "annotation_hint_offset_readout",
            "answer_hint_offset_readout",
        ),
        context=f"prompt defaults for {TASK_ID}",
    )
    json_example, json_example_answer_only = _prompt_examples(
        offset_direction=str(query.offset_direction),
        delta_minutes=int(query.delta_minutes),
    )
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_values["bundle_id"]),
        scene_key=str(prompt_values["scene_key"]),
        task_key=str(prompt_values["task_key"]),
        query_key=str(query.query_id),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(prompt_values[f"object_description_offset_readout_{style.scene_variant}"]),
            "delta_minutes": int(query.delta_minutes),
            "json_output_contract": str(prompt_values["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_values["json_output_contract_answer_only"]),
            "annotation_hint": str(prompt_values["annotation_hint_offset_readout"]),
            "answer_hint": str(prompt_values["answer_hint_offset_readout"]),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
    return str(prompt_artifacts.prompt), dict(prompt_artifacts.prompt_variants), {
        "prompt_variant": dict(prompt_artifacts.prompt_variant),
        "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
        "prompt_variants_for_trace": dict(prompt_artifacts.prompt_variants_for_trace),
        "bundle_id": str(prompt_values["bundle_id"]),
    }, prompt_artifacts


@register_task
class SymbolicClockOffsetReadoutTask:
    """Apply a minute offset to a single analog clock."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one minute-offset clock readout task."""

        del max_attempts
        query, task_params = _resolve_query(int(instance_seed), params=params)
        style = _resolve_clock_style(params=task_params, instance_seed=int(instance_seed))
        render_params = resolve_clock_render_params(
            task_params,
            render_defaults=_RENDER_DEFAULTS,
            fallback_values=asdict(DEFAULTS),
            instance_seed=int(instance_seed),
        )
        font_family = sample_font_family(
            role="readout",
            instance_seed=int(instance_seed),
            namespace="symbolic.clock.offset_readout.font",
            params={**dict(_RENDER_DEFAULTS), **dict(task_params)},
        )
        clock_theme = build_time_artifact_clock_theme(
            accent_color_name=str(style.accent_color_name),
            style_variant=str(style.style_variant),
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
        with temporary_default_font_family(str(font_family)):
            rendered_scene = render_clock_scene(
                background,
                scene_variant=str(style.scene_variant),
                shown_total_minutes=int(query.shown_total_minutes),
                render_params=render_params,
                visual_theme=clock_theme,
                center_px=(0.5 * float(render_params.canvas_width), 300.0),
            )
            option_labels = resolve_text_option_labels(task_params, gen_defaults=_GEN_DEFAULTS)
            correct_label, label_probs = sample_correct_option_label(
                params=task_params,
                gen_defaults=_GEN_DEFAULTS,
                instance_seed=int(instance_seed),
                seed_namespace=TASK_ID,
                labels=option_labels,
            )
            candidate_totals = [
                add_clock_minutes(int(query.answer_total_minutes), int(offset))
                for offset in (5, -5, 10, -10, 15, -15, 30, -30, 60, -60, int(query.delta_minutes), -int(query.delta_minutes))
            ]
            option_values = option_value_map(
                labels=option_labels,
                correct_label=str(correct_label),
                correct_value=int(query.answer_total_minutes),
                distractors=candidate_totals,
            )
            option_text = {
                str(label): str(format_clock_hhmm(int(value)))
                for label, value in option_values.items()
            }
            raw_option_bboxes, option_entities = draw_text_option_cards(
                rendered_scene.image,
                text_by_label=option_text,
                correct_label=str(correct_label),
                y0_px=option_cards_y_below_bbox(
                    rendered_scene.scene_bbox_px,
                    canvas_height=int(render_params.canvas_height),
                ),
            )
            option_bboxes_px = {
                str(label): [round(float(value), 3) for value in bbox]
                for label, bbox in raw_option_bboxes.items()
            }
            selected_option_bbox_px = list(option_bboxes_px[str(correct_label)])
            rendered_scene = rendered_scene.__class__(
                image=rendered_scene.image,
                scene_bbox_px=(
                    min(float(rendered_scene.scene_bbox_px[0]), min(float(b[0]) for b in raw_option_bboxes.values())),
                    min(float(rendered_scene.scene_bbox_px[1]), min(float(b[1]) for b in raw_option_bboxes.values())),
                    max(float(rendered_scene.scene_bbox_px[2]), max(float(b[2]) for b in raw_option_bboxes.values())),
                    max(float(rendered_scene.scene_bbox_px[3]), max(float(b[3]) for b in raw_option_bboxes.values())),
                ),
                face_bbox_px=rendered_scene.face_bbox_px,
                center_px=rendered_scene.center_px,
                hour_hand_bbox_px=rendered_scene.hour_hand_bbox_px,
                minute_hand_bbox_px=rendered_scene.minute_hand_bbox_px,
                second_hand_bbox_px=rendered_scene.second_hand_bbox_px,
                alarm_hand_bbox_px=rendered_scene.alarm_hand_bbox_px,
                hour_hand_tip_px=rendered_scene.hour_hand_tip_px,
                minute_hand_tip_px=rendered_scene.minute_hand_tip_px,
                second_hand_tip_px=rendered_scene.second_hand_tip_px,
                alarm_hand_tip_px=rendered_scene.alarm_hand_tip_px,
                entities=[*rendered_scene.entities, *option_entities],
            )
        image, post_noise_meta = apply_post_image_noise(
            rendered_scene.image,
            instance_seed=int(instance_seed),
            params=task_params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )

        prompt, prompt_variants, prompt_meta, prompt_artifacts = _build_prompt(
            prompt_defaults=_PROMPT_DEFAULTS,
            style=style,
            query=query,
            instance_seed=int(instance_seed),
        )
        hand_annotation_artifacts = clock_hand_segment_annotations(rendered_scene)
        annotation_artifacts = bbox_annotation_artifacts(selected_option_bbox_px)
        answer_gt = TypedValue(type="option_letter", value=str(correct_label))
        shown_time_text = str(format_clock_hhmm(int(query.shown_total_minutes)))
        hand_bboxes_px = {
            "hour": [round(float(value), 3) for value in rendered_scene.hour_hand_bbox_px],
            "minute": [round(float(value), 3) for value in rendered_scene.minute_hand_bbox_px],
        }
        hand_tips_px = {
            "hour": [round(float(value), 3) for value in rendered_scene.hour_hand_tip_px],
            "minute": [round(float(value), 3) for value in rendered_scene.minute_hand_tip_px],
        }
        query_params = {
            "query_id": str(query.query_id),
            "query_id_probabilities": dict(query.query_id_probabilities),
            "question_format": QUESTION_FORMAT,
            "scene_id": SCENE_ID,
            "offset_unit": "minutes",
            "offset_direction": str(query.offset_direction),
            "scene_variant": str(style.scene_variant),
            "style_variant": str(style.style_variant),
            "accent_color_name": str(style.accent_color_name),
            "scene_variant_probabilities": dict(style.scene_variant_probabilities),
            "style_variant_probabilities": dict(style.style_variant_probabilities),
            "accent_color_name_probabilities": dict(style.accent_color_name_probabilities),
            "hour_support": [int(query.hour_support[0]), int(query.hour_support[1])],
            "minute_support": [int(value) for value in query.minute_support],
            "delta_minutes_support": [int(value) for value in query.delta_minutes_support],
            "min_hand_angle_gap_deg": float(query.min_hand_angle_gap_deg),
            "option_labels": [str(label) for label in option_labels],
            "correct_label": str(correct_label),
            "correct_label_probabilities": {str(key): float(value) for key, value in label_probs.items()},
        }
        prompt_query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query.query_id),
            params=query_params,
        )

        trace_payload = {
            "scene_ir": {
                "scene_kind": "symbolic_clock_single",
                "entities": [dict(entity) for entity in rendered_scene.entities],
                "relations": {
                    "query_id": str(query.query_id),
                    "scene_id": SCENE_ID,
                    "offset_unit": "minutes",
                    "offset_direction": str(query.offset_direction),
                    "scene_variant": str(style.scene_variant),
                    "shown_total_minutes": int(query.shown_total_minutes),
                    "shown_time_text": str(shown_time_text),
                    "delta_minutes": int(query.delta_minutes),
                    "answer_total_minutes": int(query.answer_total_minutes),
                    "answer_time_text": str(query.answer_time_text),
                    "answer_label": str(correct_label),
                },
            },
            "query_spec": {
                **dict(prompt_query_spec),
                "template_id": str(prompt_meta["bundle_id"]),
            },
            "render_spec": {
                "scene_id": SCENE_ID,
                "canvas_width": int(render_params.canvas_width),
                "canvas_height": int(render_params.canvas_height),
                "coord_space": "pixel",
                "scene_variant": str(style.scene_variant),
                "background_style": dict(background_meta),
                "scene_style": dict(scene_style_meta),
                "post_image_noise": dict(post_noise_meta),
                "scene_bbox_px": [round(float(value), 3) for value in rendered_scene.scene_bbox_px],
                "clock_style": {
                    "accent_color_name": str(style.accent_color_name),
                    "style_variant": str(style.style_variant),
                    "face_radius_px": int(render_params.face_radius_px),
                    "bezel_width_px": int(render_params.bezel_width_px),
                    "numeral_font_size_px": int(render_params.numeral_font_size_px),
                    "hour_hand_width_px": int(render_params.hour_hand_width_px),
                    "minute_hand_width_px": int(render_params.minute_hand_width_px),
                    "minor_tick_dot_radius_px": int(render_params.minor_tick_dot_radius_px),
                    "inner_ring_inset_px": int(render_params.inner_ring_inset_px),
                    "inner_ring_width_px": int(render_params.inner_ring_width_px),
                    "font": {
                        "source": "global_font_pool",
                        "font_family": str(font_family),
                        "font_asset_version": font_asset_version(),
                        "scope": "single_clock_face",
                    },
                    "resolved_colors_rgb": {
                        "face_fill": [int(value) for value in clock_theme.face_fill_rgb],
                        "face_outline": [int(value) for value in clock_theme.face_outline_rgb],
                        "numerals": [int(value) for value in clock_theme.numeral_color_rgb],
                        "ticks": [int(value) for value in clock_theme.tick_color_rgb],
                        "hour_hand": [int(value) for value in clock_theme.hour_hand_color_rgb],
                        "minute_hand": [int(value) for value in clock_theme.minute_hand_color_rgb],
                        "center_dot": [int(value) for value in clock_theme.center_dot_color_rgb],
                        "inner_ring": (
                            [int(value) for value in clock_theme.inner_ring_rgb]
                            if clock_theme.inner_ring_rgb is not None
                            else None
                        ),
                    },
                    "minor_tick_mode": str(clock_theme.minor_tick_mode),
                },
            },
            "render_map": {
                "image_id": "img0",
                "scene_bbox_px": [round(float(value), 3) for value in rendered_scene.scene_bbox_px],
                "face_bbox_px": [round(float(value), 3) for value in rendered_scene.face_bbox_px],
                "center_px": [round(float(value), 3) for value in rendered_scene.center_px],
                "hand_bboxes_px": dict(hand_bboxes_px),
                "hand_tips_px": dict(hand_tips_px),
                "annotation_source": "selected_answer_option_bbox_px",
                "option_bboxes_px": dict(option_bboxes_px),
                "selected_option_label": str(correct_label),
                "selected_option_bbox_px": list(selected_option_bbox_px),
            },
            "execution_trace": {
                **dict(query_params),
                "shown_total_minutes": int(query.shown_total_minutes),
                "shown_hour": int(query.shown_hour),
                "shown_minute": int(query.shown_minute),
                "shown_time_text": str(shown_time_text),
                "delta_minutes": int(query.delta_minutes),
                "answer_total_minutes": int(query.answer_total_minutes),
                "answer_time_text": str(query.answer_time_text),
                "answer_value": str(query.answer_time_text),
                "answer_label": str(correct_label),
                "option_values_by_label": {str(key): int(value) for key, value in option_values.items()},
                "option_text_by_label": dict(option_text),
                "answer_type": "option_letter",
                "supporting_parts": ["selected_answer_option"],
                "supporting_segments": list(hand_annotation_artifacts.value),
                "selected_option_bbox_px": list(selected_option_bbox_px),
            },
            "witness_symbolic": {
                "type": str(annotation_artifacts.annotation_type),
                "value": list(annotation_artifacts.value),
            },
            "projected_annotation": dict(annotation_artifacts.projected_annotation),
            "answer_gt": answer_gt.to_dict(),
            "annotation_gt": annotation_artifacts.annotation_gt.to_dict(),
        }
        return TaskOutput(
            prompt=str(prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_artifacts.annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(query.query_id),
            prompt_variants=dict(prompt_variants),
        )


__all__ = [
    "SymbolicClockOffsetReadoutTask",
    "TASK_ID",
]
