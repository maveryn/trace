"""Compute minutes until the next analog alarm-clock hour."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping, Tuple

from ....core.seed import spawn_rng
from ....core.sampling import uniform_choice_with_probabilities
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...registry import register_task
from ...shared.annotation_artifacts import bbox_annotation_artifacts, segment_set_annotation_artifacts
from ...shared.config_defaults import (
    group_default,
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
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
    MINUTES_PER_CLOCK_CYCLE,
    clock_hand_angle_gap_deg,
    clock_total_minutes,
    format_clock_hhmm,
    split_clock_total_minutes,
)
from ..shared.common import resolve_symbolic_axis_variant
from ..shared.scene_style import make_symbolic_scene_background, resolve_symbolic_scene_style

from .shared.defaults import DEFAULTS, POST_IMAGE_NOISE_DEFAULTS
from .shared.rendering import draw_text_option_cards, option_cards_y_below_bbox, render_clock_scene
from .shared.sampling import (
    feasible_clock_times,
    nearby_integer_distractors,
    option_value_map,
    resolve_clock_time_support,
    resolve_text_option_labels,
    sample_correct_option_label,
)
from .shared.state import SUPPORTED_SYMBOLIC_CLOCK_SCENE_VARIANTS, ClockStyleResolution
from .shared.styles import resolve_clock_render_params


DOMAIN = "symbolic"
SCENE_ID = "clock"
TASK_ID = "task_symbolic__clock__alarm_wait_time_value"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("single",)
TASK_PROMPT_KEY = "clock_alarm_wait_time_value_query"
PROMPT_QUERY_KEY = "alarm_wait_time_value"
QUESTION_FORMAT = "alarm_wait_time_value"

ALARM_HAND_COLOR_RGB: Tuple[int, int, int] = (214, 38, 38)
CURRENT_HOUR_HAND_RGB: Tuple[int, int, int] = (42, 48, 58)
CURRENT_MINUTE_HAND_RGB: Tuple[int, int, int] = (34, 78, 132)
CURRENT_CENTER_DOT_RGB: Tuple[int, int, int] = (42, 48, 58)
DEFAULT_MIN_ALARM_HAND_GAP_DEG = 20.0

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


@dataclass(frozen=True)
class _ResolvedAlarmQuery:
    """Resolved current time, alarm hour, and answer for one sample."""

    shown_total_minutes: int
    shown_hour: int
    shown_minute: int
    shown_time_text: str
    alarm_hour: int
    alarm_time_text: str
    wait_minutes: int
    hour_support: Tuple[int, int]
    minute_support: Tuple[int, int, int]
    alarm_hour_support: Tuple[int, int]
    min_hand_angle_gap_deg: float
    min_alarm_hand_gap_deg: float
    alarm_hand_angle_gaps_deg: Tuple[float, float]


def _angle_gap_deg(first_angle: float, second_angle: float) -> float:
    """Return the smaller angular gap between two clockwise clock-scale angles."""

    raw_gap = abs(float(first_angle) - float(second_angle)) % 360.0
    return float(min(raw_gap, 360.0 - raw_gap))


def _current_hand_angles_deg(total_minutes: int) -> Tuple[float, float]:
    """Return current hour/minute hand angles where 12 o'clock is 0 degrees."""

    hour, minute = split_clock_total_minutes(int(total_minutes))
    hour_angle = (30.0 * float(hour % 12)) + (0.5 * float(minute))
    minute_angle = 6.0 * float(minute)
    return float(hour_angle % 360.0), float(minute_angle % 360.0)


def _alarm_hand_angle_deg(alarm_hour: int) -> float:
    """Return the alarm hand angle on the hour scale."""

    hour = int(alarm_hour)
    if not 1 <= hour <= 12:
        raise ValueError("alarm_hour must be in 1..12")
    return float((30.0 * float(hour % 12)) % 360.0)


def _alarm_hand_angle_gaps_deg(total_minutes: int, alarm_hour: int) -> Tuple[float, float]:
    """Return alarm-to-hour and alarm-to-minute hand gaps."""

    hour_angle, minute_angle = _current_hand_angles_deg(int(total_minutes))
    alarm_angle = _alarm_hand_angle_deg(int(alarm_hour))
    return (
        _angle_gap_deg(float(alarm_angle), float(hour_angle)),
        _angle_gap_deg(float(alarm_angle), float(minute_angle)),
    )


def _wait_minutes_until_alarm(total_minutes: int, alarm_hour: int) -> int:
    """Return forward minutes until the next alarm occurrence at alarm_hour:00."""

    alarm_total = clock_total_minutes(int(alarm_hour), 0)
    delta = (int(alarm_total) - int(total_minutes)) % int(MINUTES_PER_CLOCK_CYCLE)
    return int(delta if int(delta) != 0 else MINUTES_PER_CLOCK_CYCLE)


def _resolve_alarm_hour_support(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> Tuple[int, int]:
    """Resolve the inclusive alarm-hour support."""

    alarm_hour_min = int(params.get("alarm_hour_min", group_default(gen_defaults, "alarm_hour_min", 1)))
    alarm_hour_max = int(params.get("alarm_hour_max", group_default(gen_defaults, "alarm_hour_max", 12)))
    if not (1 <= alarm_hour_min <= alarm_hour_max <= 12):
        raise ValueError("alarm-hour support must satisfy 1 <= min <= max <= 12")
    return int(alarm_hour_min), int(alarm_hour_max)


def _explicit_shown_total_minutes(params: Mapping[str, Any]) -> int | None:
    """Return explicitly requested current time, if supplied."""

    if params.get("shown_total_minutes") is not None:
        return int(params["shown_total_minutes"]) % int(MINUTES_PER_CLOCK_CYCLE)
    if params.get("shown_hour") is not None or params.get("shown_minute") is not None:
        if params.get("shown_hour") is None or params.get("shown_minute") is None:
            raise ValueError("shown_hour and shown_minute must be provided together")
        return clock_total_minutes(int(params["shown_hour"]), int(params["shown_minute"]))
    return None


def _explicit_alarm_hour(params: Mapping[str, Any]) -> int | None:
    """Return explicitly requested alarm hour, if supplied."""

    if params.get("alarm_hour") is not None:
        return int(params["alarm_hour"])
    if params.get("alarm_hour_12") is not None:
        return int(params["alarm_hour_12"])
    return None


def _is_feasible_alarm_pair(
    *,
    shown_total_minutes: int,
    alarm_hour: int,
    min_alarm_hand_gap_deg: float,
) -> bool:
    """Return whether the red alarm hand is separated from both current-time hands."""

    gaps = _alarm_hand_angle_gaps_deg(int(shown_total_minutes), int(alarm_hour))
    return min(float(gap) for gap in gaps) >= float(min_alarm_hand_gap_deg)


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
    """Resolve one non-semantic clock style axis."""

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


def _resolve_clock_style(*, params: Mapping[str, Any], instance_seed: int) -> ClockStyleResolution:
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


def _alarm_safe_clock_theme(*, accent_color_name: str, style_variant: str):
    """Return a clock theme whose current-time hands cannot be confused with red alarm hand."""

    base_theme = build_time_artifact_clock_theme(
        accent_color_name=str(accent_color_name),
        style_variant=str(style_variant),
    )
    return replace(
        base_theme,
        hour_hand_color_rgb=tuple(int(value) for value in CURRENT_HOUR_HAND_RGB),
        minute_hand_color_rgb=tuple(int(value) for value in CURRENT_MINUTE_HAND_RGB),
        second_hand_color_rgb=tuple(int(value) for value in CURRENT_MINUTE_HAND_RGB),
        center_dot_color_rgb=tuple(int(value) for value in CURRENT_CENTER_DOT_RGB),
    )


def _resolve_query(instance_seed: int, *, params: Mapping[str, Any]) -> _ResolvedAlarmQuery:
    """Resolve the current time, red alarm hour, and wait-minute answer."""

    hour_support, minute_support, all_times = resolve_clock_time_support(
        params,
        gen_defaults=_GEN_DEFAULTS,
        fallback_hour_min=DEFAULTS.hour_min,
        fallback_hour_max=DEFAULTS.hour_max,
        fallback_minute_min=DEFAULTS.minute_min,
        fallback_minute_max=DEFAULTS.minute_max,
        fallback_minute_step=DEFAULTS.minute_step,
        context="clock alarm wait-time task",
    )
    min_hand_gap = float(
        params.get(
            "min_hand_angle_gap_deg",
            group_default(_GEN_DEFAULTS, "min_hand_angle_gap_deg", DEFAULTS.min_hand_angle_gap_deg),
        )
    )
    min_alarm_gap = float(
        params.get(
            "min_alarm_hand_gap_deg",
            group_default(_GEN_DEFAULTS, "min_alarm_hand_gap_deg", DEFAULT_MIN_ALARM_HAND_GAP_DEG),
        )
    )
    if min_hand_gap < 0.0 or min_alarm_gap < 0.0:
        raise ValueError("clock hand angle-gap constraints must be non-negative")

    shown_times = feasible_clock_times(tuple(int(value) for value in all_times), min_hand_angle_gap_deg=float(min_hand_gap))
    alarm_hour_support = _resolve_alarm_hour_support(params, _GEN_DEFAULTS)
    alarm_hours = tuple(range(int(alarm_hour_support[0]), int(alarm_hour_support[1]) + 1))

    explicit_total = _explicit_shown_total_minutes(params)
    explicit_alarm = _explicit_alarm_hour(params)
    feasible_pairs: list[Tuple[int, int]] = []
    for shown_total in shown_times:
        if explicit_total is not None and int(shown_total) != int(explicit_total):
            continue
        for alarm_hour in alarm_hours:
            if explicit_alarm is not None and int(alarm_hour) != int(explicit_alarm):
                continue
            if _is_feasible_alarm_pair(
                shown_total_minutes=int(shown_total),
                alarm_hour=int(alarm_hour),
                min_alarm_hand_gap_deg=float(min_alarm_gap),
            ):
                feasible_pairs.append((int(shown_total), int(alarm_hour)))
    if not feasible_pairs:
        raise ValueError("no feasible current-time/alarm-hour pairs for configured alarm clock task")

    if explicit_total is not None or explicit_alarm is not None:
        shown_total_minutes, alarm_hour = feasible_pairs[0]
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.current_time_and_alarm_hour")
        selected_pair, _pair_probs = uniform_choice_with_probabilities(
            rng,
            tuple(feasible_pairs),
            sort_keys=True,
        )
        shown_total_minutes, alarm_hour = int(selected_pair[0]), int(selected_pair[1])

    shown_hour, shown_minute = split_clock_total_minutes(int(shown_total_minutes))
    wait_minutes = _wait_minutes_until_alarm(int(shown_total_minutes), int(alarm_hour))
    if not 1 <= int(wait_minutes) <= int(MINUTES_PER_CLOCK_CYCLE):
        raise ValueError("alarm wait-minute answer must be in 1..720")
    alarm_time_text = format_clock_hhmm(clock_total_minutes(int(alarm_hour), 0))
    return _ResolvedAlarmQuery(
        shown_total_minutes=int(shown_total_minutes),
        shown_hour=int(shown_hour),
        shown_minute=int(shown_minute),
        shown_time_text=str(format_clock_hhmm(int(shown_total_minutes))),
        alarm_hour=int(alarm_hour),
        alarm_time_text=str(alarm_time_text),
        wait_minutes=int(wait_minutes),
        hour_support=(int(hour_support[0]), int(hour_support[1])),
        minute_support=tuple(int(value) for value in minute_support),
        alarm_hour_support=tuple(int(value) for value in alarm_hour_support),
        min_hand_angle_gap_deg=float(min_hand_gap),
        min_alarm_hand_gap_deg=float(min_alarm_gap),
        alarm_hand_angle_gaps_deg=tuple(
            round(float(value), 6)
            for value in _alarm_hand_angle_gaps_deg(int(shown_total_minutes), int(alarm_hour))
        ),
    )


def _prompt_examples() -> Tuple[str, str]:
    """Return stable JSON examples for the alarm wait-time task."""

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
    instance_seed: int,
) -> tuple[str, dict[str, str], dict[str, Any], Any]:
    """Render the v1 prompt bundle for the alarm wait-time task."""

    prompt_values = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            f"object_description_alarm_wait_time_value_{style.scene_variant}",
            "json_output_contract",
            "json_output_contract_answer_only",
            "annotation_hint_alarm_wait_time_value",
            "answer_hint_alarm_wait_time_value",
        ),
        context=f"prompt defaults for {TASK_ID}",
    )
    json_example, json_example_answer_only = _prompt_examples()
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_values["bundle_id"]),
        scene_key=str(prompt_values["scene_key"]),
        task_key=str(prompt_values["task_key"]),
        query_key=PROMPT_QUERY_KEY,
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(prompt_values[f"object_description_alarm_wait_time_value_{style.scene_variant}"]),
            "json_output_contract": str(prompt_values["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_values["json_output_contract_answer_only"]),
            "annotation_hint": str(prompt_values["annotation_hint_alarm_wait_time_value"]),
            "answer_hint": str(prompt_values["answer_hint_alarm_wait_time_value"]),
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


def _alarm_segment_annotations(rendered_scene) -> Any:
    """Build segment annotation for current hour, current minute, and red alarm hand."""

    if rendered_scene.alarm_hand_tip_px is None:
        raise ValueError("alarm-hand annotation requested for a clock without an alarm hand")
    center = tuple(float(value) for value in rendered_scene.center_px)
    segments = [
        (center, tuple(float(value) for value in rendered_scene.hour_hand_tip_px)),
        (center, tuple(float(value) for value in rendered_scene.minute_hand_tip_px)),
        (center, tuple(float(value) for value in rendered_scene.alarm_hand_tip_px)),
    ]
    return segment_set_annotation_artifacts(segments)


@register_task
class SymbolicClockAlarmWaitTimeValueTask:
    """Compute minutes until the next alarm hour on an analog alarm clock."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one analog alarm-clock wait-time task."""

        del max_attempts
        task_params = dict(params)
        query = _resolve_query(int(instance_seed), params=task_params)
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
            namespace=f"{TASK_ID}.font",
            params={**dict(_RENDER_DEFAULTS), **dict(task_params)},
        )
        clock_theme = _alarm_safe_clock_theme(
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
        alarm_hand_width_px = max(4, int(round(0.55 * float(render_params.hour_hand_width_px))))
        with temporary_default_font_family(str(font_family)):
            rendered_scene = render_clock_scene(
                background,
                scene_variant=str(style.scene_variant),
                shown_total_minutes=int(query.shown_total_minutes),
                render_params=render_params,
                visual_theme=clock_theme,
                alarm_hour_12=int(query.alarm_hour),
                alarm_hand_color_rgb=tuple(int(value) for value in ALARM_HAND_COLOR_RGB),
                alarm_hand_width_px=int(alarm_hand_width_px),
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
            distractors = nearby_integer_distractors(
                correct_value=int(query.wait_minutes),
                support_values=range(5, int(MINUTES_PER_CLOCK_CYCLE) + 1, 5),
                preferred_offsets=(5, 10, 15, 30, 60, 120, 180),
                min_value=1,
                max_value=int(MINUTES_PER_CLOCK_CYCLE),
            )
            option_values = option_value_map(
                labels=option_labels,
                correct_label=str(correct_label),
                correct_value=int(query.wait_minutes),
                distractors=distractors,
            )
            option_text = {str(label): str(value) for label, value in option_values.items()}
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
            instance_seed=int(instance_seed),
        )
        hand_annotation_artifacts = _alarm_segment_annotations(rendered_scene)
        annotation_artifacts = bbox_annotation_artifacts(selected_option_bbox_px)
        answer_gt = TypedValue(type="option_letter", value=str(correct_label))
        hand_bboxes_px = {
            "hour": [round(float(value), 3) for value in rendered_scene.hour_hand_bbox_px],
            "minute": [round(float(value), 3) for value in rendered_scene.minute_hand_bbox_px],
            "alarm": [round(float(value), 3) for value in rendered_scene.alarm_hand_bbox_px],
        }
        hand_tips_px = {
            "hour": [round(float(value), 3) for value in rendered_scene.hour_hand_tip_px],
            "minute": [round(float(value), 3) for value in rendered_scene.minute_hand_tip_px],
            "alarm": [round(float(value), 3) for value in rendered_scene.alarm_hand_tip_px],
        }
        query_params = {
            "query_id": "single",
            "query_id_probabilities": {"single": 1.0},
            "question_format": QUESTION_FORMAT,
            "scene_id": SCENE_ID,
            "scene_variant": str(style.scene_variant),
            "style_variant": str(style.style_variant),
            "accent_color_name": str(style.accent_color_name),
            "scene_variant_probabilities": dict(style.scene_variant_probabilities),
            "style_variant_probabilities": dict(style.style_variant_probabilities),
            "accent_color_name_probabilities": dict(style.accent_color_name_probabilities),
            "hour_support": [int(query.hour_support[0]), int(query.hour_support[1])],
            "minute_support": [int(value) for value in query.minute_support],
            "alarm_hour_support": [int(query.alarm_hour_support[0]), int(query.alarm_hour_support[1])],
            "alarm_minute": 0,
            "alarm_hand_scale": "hour",
            "alarm_hand_color": "red",
            "min_hand_angle_gap_deg": float(query.min_hand_angle_gap_deg),
            "min_alarm_hand_gap_deg": float(query.min_alarm_hand_gap_deg),
            "option_labels": [str(label) for label in option_labels],
            "correct_label": str(correct_label),
            "correct_label_probabilities": {str(key): float(value) for key, value in label_probs.items()},
        }
        prompt_query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id="single",
            params=query_params,
        )

        trace_payload = {
            "scene_ir": {
                "scene_kind": "symbolic_clock_alarm",
                "entities": [dict(entity) for entity in rendered_scene.entities],
                "relations": {
                    "query_id": "single",
                    "scene_id": SCENE_ID,
                    "scene_variant": str(style.scene_variant),
                    "shown_total_minutes": int(query.shown_total_minutes),
                    "shown_time_text": str(query.shown_time_text),
                    "alarm_hour": int(query.alarm_hour),
                    "alarm_minute": 0,
                    "alarm_time_text": str(query.alarm_time_text),
                    "wait_minutes": int(query.wait_minutes),
                    "answer_label": str(correct_label),
                    "alarm_hand_scale": "hour",
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
                    "alarm_hand_width_px": int(alarm_hand_width_px),
                    "alarm_hand_scale": "hour",
                    "font": {
                        "source": "global_font_pool",
                        "font_family": str(font_family),
                        "font_asset_version": font_asset_version(),
                        "scope": "single_alarm_clock_face",
                    },
                    "resolved_colors_rgb": {
                        "face_fill": [int(value) for value in clock_theme.face_fill_rgb],
                        "face_outline": [int(value) for value in clock_theme.face_outline_rgb],
                        "numerals": [int(value) for value in clock_theme.numeral_color_rgb],
                        "ticks": [int(value) for value in clock_theme.tick_color_rgb],
                        "hour_hand": [int(value) for value in clock_theme.hour_hand_color_rgb],
                        "minute_hand": [int(value) for value in clock_theme.minute_hand_color_rgb],
                        "alarm_hand": [int(value) for value in ALARM_HAND_COLOR_RGB],
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
                "shown_time_text": str(query.shown_time_text),
                "alarm_hour": int(query.alarm_hour),
                "alarm_minute": 0,
                "alarm_time_text": str(query.alarm_time_text),
                "wait_minutes": int(query.wait_minutes),
                "answer_value": int(query.wait_minutes),
                "answer_label": str(correct_label),
                "option_values_by_label": {str(key): int(value) for key, value in option_values.items()},
                "option_text_by_label": dict(option_text),
                "answer_type": "option_letter",
                "alarm_hand_angle_gaps_deg": [float(value) for value in query.alarm_hand_angle_gaps_deg],
                "current_hand_angle_gap_deg": round(float(clock_hand_angle_gap_deg(int(query.shown_total_minutes))), 6),
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
            query_id="single",
            prompt_variants=dict(prompt_variants),
        )


__all__ = [
    "ALARM_HAND_COLOR_RGB",
    "SymbolicClockAlarmWaitTimeValueTask",
    "TASK_ID",
]
