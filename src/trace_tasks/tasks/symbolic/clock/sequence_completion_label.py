"""Clock sequence-completion option task."""

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
from .shared.rendering import draw_clock_geometry
from .shared.sampling import feasible_clock_times, resolve_clock_time_support
from .shared.state import SUPPORTED_SYMBOLIC_CLOCK_SCENE_VARIANTS
from .shared.styles import resolve_clock_render_params, scale_clock_render_params_for_radius


DOMAIN = "symbolic"
SCENE_ID = "clock"
TASK_ID = "task_symbolic__clock__sequence_completion_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "sequence_completion_label"


@dataclass(frozen=True)
class _TaskDefaults:
    """Stable fallback defaults for clock sequence-completion scenes."""

    hour_min: int = 1
    hour_max: int = 12
    minute_min: int = 0
    minute_max: int = 55
    minute_step: int = 5
    min_hand_angle_gap_deg: float = 10.0
    sequence_step_minutes_support: Tuple[int, ...] = (15, 30, 45, 60, 90)
    missing_slot_support: Tuple[int, ...] = (0, 1, 2, 3)
    option_label_support: Tuple[str, ...] = ("A", "B", "C", "D")
    option_count: int = 4
    min_option_gap_minutes: int = 10
    canvas_width: int = 980
    canvas_height: int = 760
    outer_margin_px: int = 40
    face_radius_px: int = 78
    bezel_width_px: int = 7
    numeral_font_size_px: int = 15
    major_tick_length_px: int = 12
    minor_tick_length_px: int = 5
    major_tick_width_px: int = 3
    minor_tick_width_px: int = 2
    minor_tick_dot_radius_px: int = 2
    hour_hand_width_px: int = 7
    minute_hand_width_px: int = 5
    second_hand_width_px: int = 2
    hand_bbox_padding_px: int = 5
    center_dot_radius_px: int = 5
    inner_ring_inset_px: int = 12
    inner_ring_width_px: int = 2
    top_card_width_px: int = 205
    top_card_height_px: int = 220
    option_card_width_px: int = 205
    option_card_height_px: int = 220
    card_gap_px: int = 20
    top_row_y_px: int = 70
    option_row_y_px: int = 430
    label_font_size_px: int = 28
    missing_mark_font_size_px: int = 78


@dataclass(frozen=True)
class _ResolvedQuery:
    """Resolved semantic and visual support for one sequence-completion instance."""

    query_id: str
    scene_variant: str
    style_variant: str
    accent_color_name: str
    sequence_start_total_minutes: int
    sequence_step_minutes: int
    missing_slot_index: int
    sequence_total_minutes: Tuple[int, int, int, int]
    option_labels: Tuple[str, ...]
    correct_label: str
    option_total_minutes_by_label: Dict[str, int]
    sequence_step_minutes_support: Tuple[int, ...]
    missing_slot_support: Tuple[int, ...]
    hour_support: Tuple[int, int]
    minute_support: Tuple[int, int, int]
    min_hand_angle_gap_deg: float
    min_option_gap_minutes: int
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


def _resolve_int_tuple(
    params: Mapping[str, Any],
    *,
    key: str,
    fallback: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Resolve one integer support tuple."""

    raw = params.get(str(key), group_default(_GEN_DEFAULTS, str(key), fallback))
    values = tuple(dict.fromkeys(int(value) for value in raw))
    if not values:
        raise ValueError(f"{key} must not be empty")
    return values


def _resolve_option_labels(params: Mapping[str, Any]) -> Tuple[str, ...]:
    """Resolve fixed option-label support."""

    raw = params.get("option_label_support", group_default(_GEN_DEFAULTS, "option_label_support", _DEFAULTS.option_label_support))
    labels = tuple(str(label).strip() for label in raw if str(label).strip())
    option_count = int(params.get("option_count", group_default(_GEN_DEFAULTS, "option_count", _DEFAULTS.option_count)))
    if option_count != 4 or len(labels) < 4:
        raise ValueError("symbolic clock sequence completion requires exactly four option labels")
    labels = labels[:4]
    if len(set(labels)) != len(labels):
        raise ValueError("symbolic clock sequence option labels must be unique")
    return labels


def _circular_gap_minutes(first: int, second: int) -> int:
    """Return the smaller distance between two 12-hour clock times."""

    diff = abs(int(first) - int(second)) % 720
    return int(min(diff, 720 - diff))


def _distractor_pool(
    *,
    feasible_times: Tuple[int, ...],
    sequence_times: Tuple[int, int, int, int],
    correct_time: int,
    min_gap_minutes: int,
) -> Tuple[int, ...]:
    """Return valid distractor time values."""

    blocked = {int(value) for value in sequence_times}
    return tuple(
        int(value)
        for value in feasible_times
        if int(value) not in blocked and _circular_gap_minutes(int(value), int(correct_time)) >= int(min_gap_minutes)
    )


def _resolve_query(instance_seed: int, *, params: Mapping[str, Any]) -> _ResolvedQuery:
    """Resolve the sequence-completion scene and answer."""

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
        context="symbolic clock sequence",
    )
    min_gap = float(params.get("min_hand_angle_gap_deg", group_default(_GEN_DEFAULTS, "min_hand_angle_gap_deg", _DEFAULTS.min_hand_angle_gap_deg)))
    feasible = feasible_clock_times(raw_times, min_hand_angle_gap_deg=float(min_gap))
    if len(feasible) < 12:
        raise ValueError("symbolic clock sequence completion requires at least twelve feasible shown times")
    step_support = _resolve_int_tuple(params, key="sequence_step_minutes_support", fallback=_DEFAULTS.sequence_step_minutes_support)
    missing_support = _resolve_int_tuple(params, key="missing_slot_support", fallback=_DEFAULTS.missing_slot_support)
    missing_support = tuple(value for value in missing_support if 0 <= int(value) <= 3)
    if len(missing_support) < 1:
        raise ValueError("symbolic clock sequence missing_slot_support must contain slot indices in 0..3")
    option_labels = _resolve_option_labels(params)
    min_option_gap = int(params.get("min_option_gap_minutes", group_default(_GEN_DEFAULTS, "min_option_gap_minutes", _DEFAULTS.min_option_gap_minutes)))

    explicit_step = params.get("sequence_step_minutes")
    if explicit_step is not None:
        step = int(explicit_step)
        if step not in step_support:
            raise ValueError("explicit sequence_step_minutes is outside sequence_step_minutes_support")
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.sequence_step")
        step, _step_probs = uniform_choice_with_probabilities(
            rng,
            step_support,
            sort_keys=True,
        )
        step = int(step)
    explicit_missing_slot = params.get("missing_slot_index")
    if explicit_missing_slot is not None:
        missing_slot = int(explicit_missing_slot)
        if missing_slot not in missing_support:
            raise ValueError("explicit missing_slot_index is outside missing_slot_support")
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.missing_slot")
        missing_slot, _missing_probs = uniform_choice_with_probabilities(
            rng,
            missing_support,
            sort_keys=True,
        )
        missing_slot = int(missing_slot)

    def _sequence_for_start(start_total: int) -> Tuple[int, int, int, int]:
        return tuple(int(add_clock_minutes(int(start_total), int(step) * offset)) for offset in range(4))  # type: ignore[return-value]

    explicit_start = params.get("sequence_start_total_minutes")
    if explicit_start is None and (params.get("start_hour") is not None or params.get("start_minute") is not None):
        explicit_start = clock_total_minutes(int(params.get("start_hour", 1)), int(params.get("start_minute", 0)))
    if explicit_start is not None:
        start_total = int(explicit_start) % 720
        sequence_times = _sequence_for_start(start_total)
        if not all(total in feasible for total in sequence_times):
            raise ValueError("explicit sequence start/step produces an infeasible clock sequence")
    else:
        starts = tuple(total for total in feasible if all(seq_total in feasible for seq_total in _sequence_for_start(int(total))))
        if not starts:
            raise ValueError("sequence_step_minutes_support produced no feasible four-clock sequence")
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.sequence_start")
        start_total, _start_probs = uniform_choice_with_probabilities(
            rng,
            starts,
            sort_keys=True,
        )
        start_total = int(start_total)
        sequence_times = _sequence_for_start(start_total)

    correct_time = int(sequence_times[int(missing_slot)])
    explicit_answer_label = params.get("answer_label")
    if explicit_answer_label is not None:
        correct_label = str(explicit_answer_label).strip()
        if correct_label not in option_labels:
            raise ValueError("explicit answer_label is outside option_label_support")
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.answer_label")
        correct_label, _label_probs = uniform_choice_with_probabilities(
            rng,
            option_labels,
            sort_keys=False,
        )
        correct_label = str(correct_label)
    distractors = list(
        _distractor_pool(
            feasible_times=feasible,
            sequence_times=sequence_times,
            correct_time=int(correct_time),
            min_gap_minutes=int(min_option_gap),
        )
    )
    if len(distractors) < 3:
        raise ValueError("symbolic clock sequence completion could not build three unique distractors")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.distractors")
    rng.shuffle(distractors)
    option_times: Dict[str, int] = {}
    distractor_iter = iter(distractors)
    for label in option_labels:
        if str(label) == str(correct_label):
            option_times[str(label)] = int(correct_time)
        else:
            option_times[str(label)] = int(next(distractor_iter))

    return _ResolvedQuery(
        query_id=str(query_id),
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        accent_color_name=str(accent_color_name),
        sequence_start_total_minutes=int(start_total),
        sequence_step_minutes=int(step),
        missing_slot_index=int(missing_slot),
        sequence_total_minutes=tuple(int(value) for value in sequence_times),
        option_labels=tuple(str(label) for label in option_labels),
        correct_label=str(correct_label),
        option_total_minutes_by_label=dict(option_times),
        sequence_step_minutes_support=tuple(int(value) for value in step_support),
        missing_slot_support=tuple(int(value) for value in missing_support),
        hour_support=hour_support,
        minute_support=minute_support,
        min_hand_angle_gap_deg=float(min_gap),
        min_option_gap_minutes=int(min_option_gap),
        query_id_probabilities=dict(query_probs),
        scene_variant_probabilities=dict(scene_probs),
        style_variant_probabilities=dict(style_probs),
        accent_color_name_probabilities=dict(accent_probs),
    )


def _row_card_bboxes(
    *,
    canvas_width: int,
    y0: int,
    card_width: int,
    card_height: int,
    card_gap: int,
    labels: Tuple[str, ...],
) -> Dict[str, Tuple[float, float, float, float]]:
    """Return fixed row card bboxes keyed by label."""

    total_width = (len(labels) * float(card_width)) + ((len(labels) - 1) * float(card_gap))
    start_x = 0.5 * (float(canvas_width) - total_width)
    return {
        str(label): (
            float(start_x + (index * (float(card_width) + float(card_gap)))),
            float(y0),
            float(start_x + (index * (float(card_width) + float(card_gap))) + float(card_width)),
            float(y0 + card_height),
        )
        for index, label in enumerate(labels)
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
    """Return prompt JSON examples for clock sequence completion."""

    answer_and_annotation = {
        "annotation": {
            "sequence_panel": [50, 70, 930, 290],
            "correct_option": [275, 430, 480, 650],
        },
        "answer": "B",
    }
    answer_only = {"answer": "B"}
    return (
        json.dumps(answer_and_annotation, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        json.dumps(answer_only, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
    )


@register_task
class SymbolicClockSequenceCompletionLabelTask:
    """Choose the option clock that completes a four-slot time sequence."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation', 'matching')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Render the four-slot sequence and bind one visual option answer.

        The key invariant is that the hidden slot time, option clocks, selected
        option annotation, and trace are all derived from one fixed sequence.
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
        top_card_width_px = int(params.get("top_card_width_px", group_default(_RENDER_DEFAULTS, "top_card_width_px", _DEFAULTS.top_card_width_px)))
        top_card_height_px = int(params.get("top_card_height_px", group_default(_RENDER_DEFAULTS, "top_card_height_px", _DEFAULTS.top_card_height_px)))
        option_card_width_px = int(params.get("option_card_width_px", group_default(_RENDER_DEFAULTS, "option_card_width_px", _DEFAULTS.option_card_width_px)))
        option_card_height_px = int(params.get("option_card_height_px", group_default(_RENDER_DEFAULTS, "option_card_height_px", _DEFAULTS.option_card_height_px)))
        card_gap_px = int(params.get("card_gap_px", group_default(_RENDER_DEFAULTS, "card_gap_px", _DEFAULTS.card_gap_px)))
        top_row_y_px = int(params.get("top_row_y_px", group_default(_RENDER_DEFAULTS, "top_row_y_px", _DEFAULTS.top_row_y_px)))
        option_row_y_px = int(params.get("option_row_y_px", group_default(_RENDER_DEFAULTS, "option_row_y_px", _DEFAULTS.option_row_y_px)))
        label_font_size_px = int(params.get("label_font_size_px", group_default(_RENDER_DEFAULTS, "label_font_size_px", _DEFAULTS.label_font_size_px)))
        missing_mark_font_size_px = int(params.get("missing_mark_font_size_px", group_default(_RENDER_DEFAULTS, "missing_mark_font_size_px", _DEFAULTS.missing_mark_font_size_px)))

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
        top_labels = ("slot_1", "slot_2", "slot_3", "slot_4")
        top_card_bboxes = _row_card_bboxes(
            canvas_width=int(render_params.canvas_width),
            y0=int(top_row_y_px),
            card_width=int(top_card_width_px),
            card_height=int(top_card_height_px),
            card_gap=int(card_gap_px),
            labels=top_labels,
        )
        option_card_bboxes = _row_card_bboxes(
            canvas_width=int(render_params.canvas_width),
            y0=int(option_row_y_px),
            card_width=int(option_card_width_px),
            card_height=int(option_card_height_px),
            card_gap=int(card_gap_px),
            labels=query.option_labels,
        )
        scaled_params = scale_clock_render_params_for_radius(render_params, radius_px=int(face_radius_px))
        top_clock_bboxes: Dict[str, List[float]] = {}
        option_clock_bboxes: Dict[str, List[float]] = {}
        scene_entities: List[Dict[str, Any]] = []

        with temporary_default_font_family(str(font_family)):
            label_font = load_font(int(label_font_size_px), bold=True)
            missing_font = load_font(int(missing_mark_font_size_px), bold=True)
            for slot_index, slot_label in enumerate(top_labels):
                card = top_card_bboxes[str(slot_label)]
                is_missing = int(slot_index) == int(query.missing_slot_index)
                fill = (252, 252, 250) if not is_missing else (245, 247, 250)
                outline = (166, 174, 184) if not is_missing else (114, 123, 138)
                draw.rounded_rectangle(card, radius=18, fill=fill, outline=outline, width=2)
                if is_missing:
                    draw_text_centered(
                        draw,
                        text="?",
                        center=((float(card[0]) + float(card[2])) / 2.0, (float(card[1]) + float(card[3])) / 2.0),
                        font=missing_font,
                        fill=(102, 112, 128),
                    )
                    continue
                shown_total = int(query.sequence_total_minutes[int(slot_index)])
                center = ((float(card[0]) + float(card[2])) / 2.0, (float(card[1]) + float(card[3])) / 2.0)
                geometry = draw_clock_geometry(
                    image,
                    center_px=center,
                    face_radius_px=float(face_radius_px),
                    scene_variant=str(query.scene_variant),
                    shown_total_minutes=int(shown_total),
                    render_params=scaled_params,
                    visual_theme=clock_theme,
                    entity_prefix=f"sequence_slot_{slot_index + 1}_",
                    extra_face_attrs={
                        "role": "sequence_slot",
                        "slot_index": int(slot_index),
                        "shown_time_text": str(format_clock_hhmm(int(shown_total))),
                    },
                )
                top_clock_bboxes[str(slot_label)] = _round_bbox(tuple(float(value) for value in geometry.face_bbox_px))
                scene_entities.extend([dict(entity) for entity in geometry.entities])

            for label in query.option_labels:
                card = option_card_bboxes[str(label)]
                draw.rounded_rectangle(card, radius=18, fill=(252, 252, 250), outline=(166, 174, 184), width=2)
                draw.rounded_rectangle((card[0] + 8, card[1] + 8, card[0] + 42, card[1] + 42), radius=10, fill=(42, 48, 56))
                draw_text_centered(
                    draw,
                    text=str(label),
                    center=(float(card[0] + 25), float(card[1] + 25)),
                    font=label_font,
                    fill=(248, 250, 252),
                )
                shown_total = int(query.option_total_minutes_by_label[str(label)])
                center = ((float(card[0]) + float(card[2])) / 2.0, float(card[1] + 116.0))
                geometry = draw_clock_geometry(
                    image,
                    center_px=center,
                    face_radius_px=float(face_radius_px),
                    scene_variant=str(query.scene_variant),
                    shown_total_minutes=int(shown_total),
                    render_params=scaled_params,
                    visual_theme=clock_theme,
                    entity_prefix=f"option_{str(label).lower()}_",
                    extra_face_attrs={
                        "role": "option",
                        "option_label": str(label),
                        "shown_time_text": str(format_clock_hhmm(int(shown_total))),
                        "is_correct": bool(str(label) == str(query.correct_label)),
                    },
                )
                option_clock_bboxes[str(label)] = _round_bbox(tuple(float(value) for value in geometry.face_bbox_px))
                scene_entities.extend([dict(entity) for entity in geometry.entities])

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
                "object_description_sequence_completion_label_classic",
                "object_description_sequence_completion_label_minimal",
                "object_description_sequence_completion_label_outline",
                "annotation_hint_sequence_completion_label",
                "answer_hint_sequence_completion_label",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        object_description = str(prompt_defaults[f"object_description_sequence_completion_label_{str(query.scene_variant)}"])
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
                "annotation_hint": str(prompt_defaults["annotation_hint_sequence_completion_label"]),
                "answer_hint": str(prompt_defaults["answer_hint_sequence_completion_label"]),
                "json_example": str(json_example),
                "json_example_answer_only": str(json_example_answer_only),
            },
            instance_seed=int(instance_seed),
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

        sequence_panel_bbox = _round_bbox(_union_bbox([tuple(float(value) for value in bbox) for bbox in top_card_bboxes.values()]))
        correct_option_bbox = _round_bbox(tuple(float(value) for value in option_card_bboxes[str(query.correct_label)]))
        annotation_bboxes = {
            "sequence_panel": list(sequence_panel_bbox),
            "correct_option": list(correct_option_bbox),
        }
        answer_gt = TypedValue(type="string", value=str(query.correct_label))
        annotation_gt = TypedValue(type="bbox_map", value=dict(annotation_bboxes))
        scene_bbox = _union_bbox([tuple(float(value) for value in bbox) for bbox in [*top_card_bboxes.values(), *option_card_bboxes.values()]])
        missing_hour, missing_minute = split_clock_total_minutes(int(query.sequence_total_minutes[int(query.missing_slot_index)]))

        trace_payload = {
            "scene_ir": {
                "scene_kind": "symbolic_clock_sequence_completion_panel",
                "entities": [dict(entity) for entity in scene_entities],
                "relations": {
                    "query_id": str(query.query_id),
                    "missing_slot_index": int(query.missing_slot_index),
                    "sequence_step_minutes": int(query.sequence_step_minutes),
                    "correct_label": str(query.correct_label),
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
                    "missing_slot_index": int(query.missing_slot_index),
                    "sequence_step_minutes": int(query.sequence_step_minutes),
                    "sequence_step_minutes_support": [int(value) for value in query.sequence_step_minutes_support],
                    "missing_slot_support": [int(value) for value in query.missing_slot_support],
                    "option_labels": [str(label) for label in query.option_labels],
                    "hour_support": [int(query.hour_support[0]), int(query.hour_support[1])],
                    "minute_support": [int(value) for value in query.minute_support],
                    "min_hand_angle_gap_deg": float(query.min_hand_angle_gap_deg),
                    "min_option_gap_minutes": int(query.min_option_gap_minutes),
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
                        "scope": "clock_sequence_completion_panel",
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
                "top_card_bboxes_px": {str(label): _round_bbox(tuple(float(value) for value in bbox)) for label, bbox in top_card_bboxes.items()},
                "option_card_bboxes_px": {str(label): _round_bbox(tuple(float(value) for value in bbox)) for label, bbox in option_card_bboxes.items()},
                "top_clock_bboxes_px": dict(top_clock_bboxes),
                "option_clock_bboxes_px": dict(option_clock_bboxes),
                "sequence_panel_bbox_px": list(sequence_panel_bbox),
                "correct_option_bbox_px": list(correct_option_bbox),
                "correct_label": str(query.correct_label),
            },
            "execution_trace": {
                "query_id": str(query.query_id),
                "prompt_query_key": PROMPT_QUERY_KEY,
                "scene_variant": str(query.scene_variant),
                "style_variant": str(query.style_variant),
                "accent_color_name": str(query.accent_color_name),
                "sequence_start_total_minutes": int(query.sequence_start_total_minutes),
                "sequence_step_minutes": int(query.sequence_step_minutes),
                "missing_slot_index": int(query.missing_slot_index),
                "missing_total_minutes": int(query.sequence_total_minutes[int(query.missing_slot_index)]),
                "missing_hour": int(missing_hour),
                "missing_minute": int(missing_minute),
                "missing_time_text": str(format_clock_hhmm(int(query.sequence_total_minutes[int(query.missing_slot_index)]))),
                "sequence_total_minutes": [int(value) for value in query.sequence_total_minutes],
                "sequence_time_texts": [str(format_clock_hhmm(int(value))) for value in query.sequence_total_minutes],
                "visible_sequence_time_texts": [
                    None if int(index) == int(query.missing_slot_index) else str(format_clock_hhmm(int(value)))
                    for index, value in enumerate(query.sequence_total_minutes)
                ],
                "option_labels": [str(label) for label in query.option_labels],
                "correct_label": str(query.correct_label),
                "option_total_minutes_by_label": {str(key): int(value) for key, value in query.option_total_minutes_by_label.items()},
                "option_time_text_by_label": {
                    str(label): str(format_clock_hhmm(int(query.option_total_minutes_by_label[str(label)])))
                    for label in query.option_labels
                },
                "sequence_step_minutes_support": [int(value) for value in query.sequence_step_minutes_support],
                "missing_slot_support": [int(value) for value in query.missing_slot_support],
                "hour_support": [int(query.hour_support[0]), int(query.hour_support[1])],
                "minute_support": [int(value) for value in query.minute_support],
                "min_hand_angle_gap_deg": float(query.min_hand_angle_gap_deg),
                "min_option_gap_minutes": int(query.min_option_gap_minutes),
                "question_format": PROMPT_QUERY_KEY,
                "supporting_bbox_roles": ["sequence_panel", "correct_option"],
            },
            "witness_symbolic": {
                "type": "bbox_map",
                "value": dict(annotation_bboxes),
            },
            "projected_annotation": {
                "type": "bbox_map",
                "bbox_map": dict(annotation_bboxes),
                "pixel_bbox_map": dict(annotation_bboxes),
                "value": dict(annotation_bboxes),
            },
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


__all__ = ["SymbolicClockSequenceCompletionLabelTask"]
