"""Clock time-ordering option task."""

from __future__ import annotations

import itertools
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
from ...shared.annotation_artifacts import bbox_annotation_artifacts
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
from ...shared.time_format import clock_hand_angle_gap_deg, clock_total_minutes, format_clock_hhmm, split_clock_total_minutes
from ..shared.scene_style import make_symbolic_scene_background, resolve_symbolic_scene_style
from ..shared.visual_defaults import load_symbolic_background_defaults, load_symbolic_noise_defaults
from .shared.rendering import draw_clock_geometry
from .shared.state import SUPPORTED_SYMBOLIC_CLOCK_SCENE_VARIANTS, ClockRenderParams
from .shared.styles import resolve_clock_render_params


DOMAIN = "symbolic"
SCENE_ID = "clock"
TASK_ID = "task_symbolic__clock__time_order_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "time_order_label"


@dataclass(frozen=True)
class _TaskDefaults:
    """Stable fallback defaults for clock time-order option panels."""

    hour_min: int = 1
    hour_max: int = 12
    minute_min: int = 0
    minute_max: int = 55
    minute_step: int = 5
    min_hand_angle_gap_deg: float = 10.0
    min_compare_gap_minutes: int = 20
    clock_label_support: Tuple[str, ...] = ("A", "B", "C", "D")
    option_label_support: Tuple[str, ...] = ("1", "2", "3", "4", "5", "6")
    option_count: int = 6
    canvas_width: int = 980
    canvas_height: int = 760
    outer_margin_px: int = 40
    face_radius_px: int = 76
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
    clock_row_y_px: int = 184
    clock_label_y_px: int = 326
    clock_gap_px: int = 66
    clock_label_font_size_px: int = 30
    option_grid_y_px: int = 450
    option_card_width_px: int = 280
    option_card_height_px: int = 88
    option_card_gap_x_px: int = 22
    option_card_gap_y_px: int = 22
    option_card_radius_px: int = 14
    option_label_font_size_px: int = 23
    option_order_font_size_px: int = 28


@dataclass(frozen=True)
class _ResolvedQuery:
    """Resolved semantic and visual support for one ordering-option instance."""

    query_id: str
    scene_variant: str
    style_variant: str
    accent_color_name: str
    clock_labels: Tuple[str, ...]
    option_labels: Tuple[str, ...]
    correct_label: str
    shown_total_minutes_by_label: Dict[str, int]
    true_order_labels: Tuple[str, ...]
    option_order_labels_by_label: Dict[str, Tuple[str, ...]]
    min_compare_gap_minutes: int
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


def _resolve_clock_labels(params: Mapping[str, Any]) -> Tuple[str, ...]:
    """Resolve the four visible clock labels."""

    raw_labels = params.get("clock_label_support", group_default(_GEN_DEFAULTS, "clock_label_support", _DEFAULTS.clock_label_support))
    labels = tuple(str(label).strip() for label in raw_labels if str(label).strip())
    if len(labels) != 4:
        raise ValueError("symbolic clock time ordering requires exactly four clock labels")
    if len(set(labels)) != len(labels):
        raise ValueError("symbolic clock time-order clock labels must be unique")
    return tuple(str(label) for label in labels)


def _resolve_option_labels(params: Mapping[str, Any]) -> Tuple[str, ...]:
    """Resolve the six visible option-card labels."""

    raw_labels = params.get("option_label_support", group_default(_GEN_DEFAULTS, "option_label_support", _DEFAULTS.option_label_support))
    labels = tuple(str(label).strip() for label in raw_labels if str(label).strip())
    option_count = int(params.get("option_count", group_default(_GEN_DEFAULTS, "option_count", _DEFAULTS.option_count)))
    if option_count != 6:
        raise ValueError("symbolic clock time ordering requires exactly six options")
    if len(labels) != int(option_count):
        raise ValueError("symbolic clock time ordering requires six option labels")
    if len(set(labels)) != len(labels):
        raise ValueError("symbolic clock time-order option labels must be unique")
    return tuple(str(label) for label in labels)


def _feasible_time_support(
    *,
    hour_support: Tuple[int, ...],
    minute_support: Tuple[int, ...],
    min_hand_angle_gap_deg: float,
) -> Tuple[int, ...]:
    """Return all feasible clock times after the hand-angle filter."""

    return tuple(
        clock_total_minutes(int(hour), int(minute))
        for hour in hour_support
        for minute in minute_support
        if float(clock_hand_angle_gap_deg(clock_total_minutes(int(hour), int(minute)))) >= float(min_hand_angle_gap_deg)
    )


def _linear_gap_ok(times: Tuple[int, ...], *, min_gap_minutes: int) -> bool:
    """Return whether sorted 12-hour-cycle times are visibly separated."""

    ordered = sorted(int(value) for value in times)
    return all((int(b) - int(a)) >= int(min_gap_minutes) for a, b in zip(ordered, ordered[1:]))


def _resolve_explicit_clock_times(
    params: Mapping[str, Any],
    *,
    labels: Tuple[str, ...],
    time_support: Tuple[int, ...],
    min_compare_gap_minutes: int,
) -> Dict[str, int] | None:
    """Resolve caller-provided clock times when supplied."""

    raw_mapping = params.get("shown_total_minutes_by_label")
    if raw_mapping is None:
        return None
    if not isinstance(raw_mapping, Mapping):
        raise ValueError("shown_total_minutes_by_label must be a label-to-minutes mapping")
    missing = [str(label) for label in labels if str(label) not in raw_mapping]
    if missing:
        raise ValueError(f"shown_total_minutes_by_label missing labels: {missing}")
    resolved = {str(label): int(raw_mapping[str(label)]) % 720 for label in labels}
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("symbolic clock time ordering requires four distinct explicit times")
    if not set(resolved.values()).issubset(set(int(value) for value in time_support)):
        raise ValueError("explicit clock times must be inside the configured feasible support")
    if not _linear_gap_ok(tuple(resolved.values()), min_gap_minutes=int(min_compare_gap_minutes)):
        raise ValueError("explicit clock times violate min_compare_gap_minutes")
    return resolved


def _sample_clock_times(
    *,
    rng,
    labels: Tuple[str, ...],
    time_support: Tuple[int, ...],
    min_compare_gap_minutes: int,
) -> Dict[str, int]:
    """Sample four distinct clock times with a unique visible order."""

    if len(time_support) < len(labels):
        raise ValueError("symbolic clock time ordering has too few feasible times")
    for _attempt in range(4000):
        sampled = tuple(int(value) for value in rng.sample(tuple(int(value) for value in time_support), k=len(labels)))
        if _linear_gap_ok(sampled, min_gap_minutes=int(min_compare_gap_minutes)):
            return {str(label): int(total) for label, total in zip(labels, sampled)}
    raise ValueError("symbolic clock time ordering could not sample separated clock times")


def _true_order(shown_total_minutes_by_label: Mapping[str, int]) -> Tuple[str, ...]:
    """Return labels sorted from earliest to latest in the 12-hour cycle."""

    return tuple(
        str(label)
        for label, _total in sorted(
            ((str(label), int(total)) for label, total in shown_total_minutes_by_label.items()),
            key=lambda item: (int(item[1]), str(item[0])),
        )
    )


def _swap_adjacent(order: Tuple[str, ...], index: int) -> Tuple[str, ...]:
    """Return one adjacent-swap distractor."""

    values = list(order)
    values[int(index)], values[int(index) + 1] = values[int(index) + 1], values[int(index)]
    return tuple(str(value) for value in values)


def _distractor_orders(*, rng, true_order: Tuple[str, ...], labels: Tuple[str, ...], count: int) -> Tuple[Tuple[str, ...], ...]:
    """Return unique candidate orderings excluding the correct order."""

    true_order = tuple(str(label) for label in true_order)
    candidates: List[Tuple[str, ...]] = []

    def _add(candidate: Tuple[str, ...]) -> None:
        normalized = tuple(str(label) for label in candidate)
        if normalized == true_order:
            return
        if normalized not in candidates:
            candidates.append(normalized)

    for index in range(len(true_order) - 1):
        _add(_swap_adjacent(true_order, int(index)))
    _add(tuple(reversed(true_order)))
    _add((true_order[1], true_order[0], true_order[3], true_order[2]))
    _add((true_order[2], true_order[0], true_order[1], true_order[3]))

    remaining = [tuple(str(label) for label in order) for order in itertools.permutations(tuple(str(label) for label in labels))]
    rng.shuffle(remaining)
    for candidate in remaining:
        _add(candidate)
        if len(candidates) >= int(count):
            break
    if len(candidates) < int(count):
        raise ValueError("symbolic clock time ordering could not build enough unique option distractors")
    return tuple(tuple(str(label) for label in order) for order in candidates[: int(count)])


def _resolve_query(instance_seed: int, *, params: Mapping[str, Any]) -> _ResolvedQuery:
    """Resolve one concrete time-ordering option panel."""

    query_id, query_id_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id=SINGLE_QUERY_ID,
        task_id=TASK_ID,
        namespace=f"{TASK_ID}.query",
    )
    scene_variant, scene_variant_probabilities = _resolve_named_variant(
        instance_seed=int(instance_seed),
        params=task_params,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_SYMBOLIC_CLOCK_SCENE_VARIANTS,
        namespace="scene_variant",
    )
    style_variant, style_variant_probabilities = _resolve_named_variant(
        instance_seed=int(instance_seed),
        params=task_params,
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS,
        namespace="style_variant",
    )
    accent_color_name, accent_color_name_probabilities = _resolve_named_variant(
        instance_seed=int(instance_seed),
        params=task_params,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        balance_flag_key="balanced_accent_color_name_sampling",
        supported=SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES,
        namespace="accent_color_name",
    )

    hour_min = int(task_params.get("hour_min", group_default(_GEN_DEFAULTS, "hour_min", _DEFAULTS.hour_min)))
    hour_max = int(task_params.get("hour_max", group_default(_GEN_DEFAULTS, "hour_max", _DEFAULTS.hour_max)))
    minute_min = int(task_params.get("minute_min", group_default(_GEN_DEFAULTS, "minute_min", _DEFAULTS.minute_min)))
    minute_max = int(task_params.get("minute_max", group_default(_GEN_DEFAULTS, "minute_max", _DEFAULTS.minute_max)))
    minute_step = int(task_params.get("minute_step", group_default(_GEN_DEFAULTS, "minute_step", _DEFAULTS.minute_step)))
    min_hand_angle_gap_deg = float(
        task_params.get(
            "min_hand_angle_gap_deg",
            group_default(_GEN_DEFAULTS, "min_hand_angle_gap_deg", _DEFAULTS.min_hand_angle_gap_deg),
        )
    )
    min_compare_gap_minutes = int(
        task_params.get(
            "min_compare_gap_minutes",
            group_default(_GEN_DEFAULTS, "min_compare_gap_minutes", _DEFAULTS.min_compare_gap_minutes),
        )
    )
    if minute_step <= 0:
        raise ValueError("minute_step must be positive for symbolic clock time ordering")
    if int(min_compare_gap_minutes) <= 0:
        raise ValueError("min_compare_gap_minutes must be positive for symbolic clock time ordering")

    hour_support = tuple(range(int(hour_min), int(hour_max) + 1))
    minute_support = tuple(range(int(minute_min), int(minute_max) + 1, int(minute_step)))
    if not hour_support or not minute_support:
        raise ValueError("symbolic clock time-order support is empty")
    if hour_support[0] < 1 or hour_support[-1] > 12:
        raise ValueError("hour support must stay within 1..12 for symbolic clock time ordering")
    if minute_support[0] < 0 or minute_support[-1] > 59:
        raise ValueError("minute support must stay within 0..59 for symbolic clock time ordering")

    clock_labels = _resolve_clock_labels(task_params)
    option_labels = _resolve_option_labels(task_params)
    time_support = _feasible_time_support(
        hour_support=hour_support,
        minute_support=minute_support,
        min_hand_angle_gap_deg=float(min_hand_angle_gap_deg),
    )
    if len(time_support) < 12:
        raise ValueError("symbolic clock time ordering requires at least twelve feasible times")

    explicit_times = _resolve_explicit_clock_times(
        task_params,
        labels=clock_labels,
        time_support=tuple(int(value) for value in time_support),
        min_compare_gap_minutes=int(min_compare_gap_minutes),
    )
    if explicit_times is not None:
        shown_total_minutes_by_label = dict(explicit_times)
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.clock_times")
        shown_total_minutes_by_label = _sample_clock_times(
            rng=rng,
            labels=clock_labels,
            time_support=tuple(int(value) for value in time_support),
            min_compare_gap_minutes=int(min_compare_gap_minutes),
        )

    true_order = _true_order(shown_total_minutes_by_label)
    explicit_answer_label = task_params.get("answer_label", task_params.get("correct_label"))
    if explicit_answer_label is not None:
        correct_label = str(explicit_answer_label).strip()
        if correct_label not in option_labels:
            raise ValueError("answer_label is outside symbolic clock time-order option labels")
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.answer_label")
        correct_label, _answer_probs = uniform_choice_with_probabilities(
            rng,
            option_labels,
            sort_keys=False,
        )
        correct_label = str(correct_label)

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.option_orders")
    distractors = list(
        _distractor_orders(
            rng=rng,
            true_order=true_order,
            labels=clock_labels,
            count=len(option_labels) - 1,
        )
    )
    option_orders: Dict[str, Tuple[str, ...]] = {}
    distractor_iter = iter(distractors)
    for option_label in option_labels:
        if str(option_label) == str(correct_label):
            option_orders[str(option_label)] = tuple(str(label) for label in true_order)
        else:
            option_orders[str(option_label)] = tuple(str(label) for label in next(distractor_iter))

    return _ResolvedQuery(
        query_id=str(query_id),
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        accent_color_name=str(accent_color_name),
        clock_labels=tuple(str(label) for label in clock_labels),
        option_labels=tuple(str(label) for label in option_labels),
        correct_label=str(correct_label),
        shown_total_minutes_by_label={str(key): int(value) for key, value in shown_total_minutes_by_label.items()},
        true_order_labels=tuple(str(label) for label in true_order),
        option_order_labels_by_label={str(key): tuple(str(label) for label in value) for key, value in option_orders.items()},
        min_compare_gap_minutes=int(min_compare_gap_minutes),
        hour_support=(int(hour_support[0]), int(hour_support[-1])),
        minute_support=(int(minute_support[0]), int(minute_support[-1]), int(minute_step)),
        min_hand_angle_gap_deg=float(min_hand_angle_gap_deg),
        query_id_probabilities=dict(query_id_probabilities),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        accent_color_name_probabilities=dict(accent_color_name_probabilities),
    )


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


def _clock_centers(
    *,
    render_params: ClockRenderParams,
    labels: Tuple[str, ...],
    clock_row_y_px: int,
    clock_gap_px: int,
) -> Dict[str, Tuple[float, float]]:
    """Return top-row clock centers keyed by clock label."""

    face_diameter = 2.0 * float(render_params.face_radius_px)
    total_width = (float(len(labels)) * face_diameter) + (float(max(0, len(labels) - 1)) * float(clock_gap_px))
    start_x = 0.5 * (float(render_params.canvas_width) - float(total_width))
    return {
        str(label): (
            float(start_x + (float(index) * (face_diameter + float(clock_gap_px))) + float(render_params.face_radius_px)),
            float(clock_row_y_px),
        )
        for index, label in enumerate(labels)
    }


def _option_card_bboxes(
    *,
    canvas_width: int,
    start_y_px: int,
    card_width_px: int,
    card_height_px: int,
    gap_x_px: int,
    gap_y_px: int,
    option_labels: Tuple[str, ...],
) -> Dict[str, Tuple[float, float, float, float]]:
    """Return a fixed 3x2 option-card layout keyed by option label."""

    if len(option_labels) != 6:
        raise ValueError("time-order option layout requires six labels")
    total_width = (3.0 * float(card_width_px)) + (2.0 * float(gap_x_px))
    start_x = 0.5 * (float(canvas_width) - total_width)
    bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    for index, label in enumerate(option_labels):
        row = int(index // 3)
        col = int(index % 3)
        x0 = float(start_x + (float(col) * (float(card_width_px) + float(gap_x_px))))
        y0 = float(start_y_px + (float(row) * (float(card_height_px) + float(gap_y_px))))
        bboxes[str(label)] = (float(x0), float(y0), float(x0 + float(card_width_px)), float(y0 + float(card_height_px)))
    return bboxes


def _order_text(order: Tuple[str, ...]) -> str:
    """Return visible option text for one label ordering."""

    return " < ".join(str(label) for label in order)


def _build_prompt_json_examples() -> tuple[str, str]:
    """Return prompt JSON examples for clock time ordering."""

    answer_and_annotation = {
        "annotation": [350, 450, 630, 538],
        "answer": "2",
    }
    answer_only = {"answer": "2"}
    return (
        json.dumps(answer_and_annotation, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        json.dumps(answer_only, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
    )


@register_task
class SymbolicClockTimeOrderLabelTask:
    """Choose the option card that lists four clocks from earliest to latest."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Render four clocks and six ordering options with one correct card."""

        del max_attempts
        query = _resolve_query(int(instance_seed), params=params)
        render_params = resolve_clock_render_params(
            params,
            render_defaults=_RENDER_DEFAULTS,
            fallback_values=asdict(_DEFAULTS),
            instance_seed=int(instance_seed),
        )
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

        clock_row_y_px = int(params.get("clock_row_y_px", group_default(_RENDER_DEFAULTS, "clock_row_y_px", _DEFAULTS.clock_row_y_px)))
        clock_label_y_px = int(params.get("clock_label_y_px", group_default(_RENDER_DEFAULTS, "clock_label_y_px", _DEFAULTS.clock_label_y_px)))
        clock_gap_px = int(params.get("clock_gap_px", group_default(_RENDER_DEFAULTS, "clock_gap_px", _DEFAULTS.clock_gap_px)))
        clock_label_font_size_px = int(params.get("clock_label_font_size_px", group_default(_RENDER_DEFAULTS, "clock_label_font_size_px", _DEFAULTS.clock_label_font_size_px)))
        option_grid_y_px = int(params.get("option_grid_y_px", group_default(_RENDER_DEFAULTS, "option_grid_y_px", _DEFAULTS.option_grid_y_px)))
        option_card_width_px = int(params.get("option_card_width_px", group_default(_RENDER_DEFAULTS, "option_card_width_px", _DEFAULTS.option_card_width_px)))
        option_card_height_px = int(params.get("option_card_height_px", group_default(_RENDER_DEFAULTS, "option_card_height_px", _DEFAULTS.option_card_height_px)))
        option_card_gap_x_px = int(params.get("option_card_gap_x_px", group_default(_RENDER_DEFAULTS, "option_card_gap_x_px", _DEFAULTS.option_card_gap_x_px)))
        option_card_gap_y_px = int(params.get("option_card_gap_y_px", group_default(_RENDER_DEFAULTS, "option_card_gap_y_px", _DEFAULTS.option_card_gap_y_px)))
        option_card_radius_px = int(params.get("option_card_radius_px", group_default(_RENDER_DEFAULTS, "option_card_radius_px", _DEFAULTS.option_card_radius_px)))
        option_label_font_size_px = int(params.get("option_label_font_size_px", group_default(_RENDER_DEFAULTS, "option_label_font_size_px", _DEFAULTS.option_label_font_size_px)))
        option_order_font_size_px = int(params.get("option_order_font_size_px", group_default(_RENDER_DEFAULTS, "option_order_font_size_px", _DEFAULTS.option_order_font_size_px)))

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

        clock_centers = _clock_centers(
            render_params=render_params,
            labels=query.clock_labels,
            clock_row_y_px=int(clock_row_y_px),
            clock_gap_px=int(clock_gap_px),
        )
        option_card_bboxes = _option_card_bboxes(
            canvas_width=int(render_params.canvas_width),
            start_y_px=int(option_grid_y_px),
            card_width_px=int(option_card_width_px),
            card_height_px=int(option_card_height_px),
            gap_x_px=int(option_card_gap_x_px),
            gap_y_px=int(option_card_gap_y_px),
            option_labels=query.option_labels,
        )

        scene_entities: List[Dict[str, Any]] = []
        clocks_by_label: Dict[str, Dict[str, Any]] = {}
        option_cards_by_label: Dict[str, Dict[str, Any]] = {}
        scene_bbox_values: List[Tuple[float, float, float, float]] = []

        with temporary_default_font_family(str(font_family)):
            clock_label_font = load_font(int(clock_label_font_size_px), bold=True)
            option_label_font = load_font(int(option_label_font_size_px), bold=False)
            option_order_font = load_font(int(option_order_font_size_px), bold=False)

            for label in query.clock_labels:
                center = clock_centers[str(label)]
                shown_total = int(query.shown_total_minutes_by_label[str(label)])
                geometry = draw_clock_geometry(
                    image,
                    center_px=center,
                    face_radius_px=float(render_params.face_radius_px),
                    scene_variant=str(query.scene_variant),
                    shown_total_minutes=int(shown_total),
                    render_params=render_params,
                    visual_theme=clock_theme,
                    entity_prefix=f"clock_{str(label).lower()}",
                    extra_face_attrs={
                        "role": "ordered_clock",
                        "clock_label": str(label),
                        "shown_time_text": str(format_clock_hhmm(int(shown_total))),
                    },
                )
                draw_text_centered(
                    draw,
                    text=str(label),
                    center=(float(center[0]), float(clock_label_y_px)),
                    font=clock_label_font,
                    fill=tuple(int(value) for value in clock_theme.numeral_color_rgb),
                )
                scene_entities.extend([dict(entity) for entity in geometry.entities])
                clocks_by_label[str(label)] = {
                    "face_bbox_px": _round_bbox(tuple(float(value) for value in geometry.face_bbox_px)),
                    "center_px": [round(float(value), 3) for value in geometry.center_px],
                    "hour_hand_bbox_px": _round_bbox(tuple(float(value) for value in geometry.hour_hand_bbox_px)),
                    "minute_hand_bbox_px": _round_bbox(tuple(float(value) for value in geometry.minute_hand_bbox_px)),
                    "hour_hand_tip_px": [round(float(value), 3) for value in geometry.hour_hand_tip_px],
                    "minute_hand_tip_px": [round(float(value), 3) for value in geometry.minute_hand_tip_px],
                    "shown_total_minutes": int(shown_total),
                    "shown_time_text": str(format_clock_hhmm(int(shown_total))),
                    "clock_label": str(label),
                }
                scene_bbox_values.append(tuple(float(value) for value in geometry.face_bbox_px))

            for option_label in query.option_labels:
                card = option_card_bboxes[str(option_label)]
                order_labels = tuple(str(label) for label in query.option_order_labels_by_label[str(option_label)])
                is_correct = bool(str(option_label) == str(query.correct_label))
                shadow = (float(card[0] + 3), float(card[1] + 4), float(card[2] + 3), float(card[3] + 4))
                draw.rounded_rectangle(shadow, radius=int(option_card_radius_px), fill=(214, 218, 224))
                draw.rounded_rectangle(
                    card,
                    radius=int(option_card_radius_px),
                    fill=(252, 252, 250),
                    outline=(148, 158, 170),
                    width=2,
                )
                badge = (
                    float(card[0] + 12),
                    float(card[1] + 18),
                    float(card[0] + 48),
                    float(card[1] + 54),
                )
                draw.rounded_rectangle(badge, radius=10, fill=(42, 48, 56))
                draw_text_centered(
                    draw,
                    text=str(option_label),
                    center=(float(card[0] + 30), float(card[1] + 36)),
                    font=option_label_font,
                    fill=(248, 250, 252),
                    stroke_width=1,
                )
                draw_text_centered(
                    draw,
                    text=_order_text(order_labels),
                    center=(float(card[0] + 165), float((card[1] + card[3]) / 2.0) + 2.0),
                    font=option_order_font,
                    fill=(31, 41, 55),
                    stroke_width=1,
                )
                scene_entities.append(
                    {
                        "entity_id": f"option_{str(option_label).lower()}_card",
                        "entity_kind": "option_card",
                        "bbox_px": _round_bbox(tuple(float(value) for value in card)),
                        "attrs": {
                            "option_label": str(option_label),
                            "order_labels": [str(label) for label in order_labels],
                            "order_text": _order_text(order_labels),
                            "is_correct": bool(is_correct),
                        },
                    }
                )
                option_cards_by_label[str(option_label)] = {
                    "card_bbox_px": _round_bbox(tuple(float(value) for value in card)),
                    "order_labels": [str(label) for label in order_labels],
                    "order_text": _order_text(order_labels),
                    "is_correct": bool(is_correct),
                }
                scene_bbox_values.append(tuple(float(value) for value in card))

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
                "object_description_time_order_label_classic",
                "object_description_time_order_label_minimal",
                "object_description_time_order_label_outline",
                "annotation_hint_time_order_label",
                "answer_hint_time_order_label",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        object_description = str(prompt_defaults[f"object_description_time_order_label_{str(query.scene_variant)}"])
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
                "annotation_hint": str(prompt_defaults["annotation_hint_time_order_label"]),
                "answer_hint": str(prompt_defaults["answer_hint_time_order_label"]),
                "json_example": str(json_example),
                "json_example_answer_only": str(json_example_answer_only),
            },
            instance_seed=int(instance_seed),
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

        correct_option_bbox = list(option_cards_by_label[str(query.correct_label)]["card_bbox_px"])
        annotation_payload = bbox_annotation_artifacts(correct_option_bbox)
        answer_gt = TypedValue(type="string", value=str(query.correct_label))
        annotation_gt = annotation_payload.annotation_gt
        scene_bbox = _union_bbox([tuple(float(value) for value in bbox) for bbox in scene_bbox_values])
        true_order_text = _order_text(tuple(str(label) for label in query.true_order_labels))
        true_order_times = [
            int(query.shown_total_minutes_by_label[str(label)])
            for label in query.true_order_labels
        ]
        correct_hour, correct_minute = split_clock_total_minutes(true_order_times[0])

        trace_payload = {
            "scene_ir": {
                "scene_kind": "symbolic_clock_time_order_panel",
                "entities": [dict(entity) for entity in scene_entities],
                "relations": {
                    "query_id": str(query.query_id),
                    "prompt_query_key": PROMPT_QUERY_KEY,
                    "clock_labels": [str(label) for label in query.clock_labels],
                    "option_labels": [str(label) for label in query.option_labels],
                    "correct_label": str(query.correct_label),
                    "true_order_labels": [str(label) for label in query.true_order_labels],
                    "true_order_text": str(true_order_text),
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
                    "clock_labels": [str(label) for label in query.clock_labels],
                    "option_labels": [str(label) for label in query.option_labels],
                    "hour_support": [int(query.hour_support[0]), int(query.hour_support[1])],
                    "minute_support": [int(value) for value in query.minute_support],
                    "min_hand_angle_gap_deg": float(query.min_hand_angle_gap_deg),
                    "min_compare_gap_minutes": int(query.min_compare_gap_minutes),
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
                "scene_bbox_px": _round_bbox(tuple(float(value) for value in scene_bbox)),
                "clock_style": {
                    "accent_color_name": str(query.accent_color_name),
                    "style_variant": str(query.style_variant),
                    "face_radius_px": int(render_params.face_radius_px),
                    "bezel_width_px": int(render_params.bezel_width_px),
                    "numeral_font_size_px": int(render_params.numeral_font_size_px),
                    "hour_hand_width_px": int(render_params.hour_hand_width_px),
                    "minute_hand_width_px": int(render_params.minute_hand_width_px),
                    "clock_label_font_size_px": int(clock_label_font_size_px),
                    "font": {
                        "source": "global_font_pool",
                        "font_family": str(font_family),
                        "font_asset_version": font_asset_version(),
                        "scope": "clock_time_order_panel",
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
                "option_style": {
                    "option_card_width_px": int(option_card_width_px),
                    "option_card_height_px": int(option_card_height_px),
                    "option_card_gap_x_px": int(option_card_gap_x_px),
                    "option_card_gap_y_px": int(option_card_gap_y_px),
                    "option_label_font_size_px": int(option_label_font_size_px),
                    "option_order_font_size_px": int(option_order_font_size_px),
                },
            },
            "render_map": {
                "image_id": "img0",
                "scene_bbox_px": _round_bbox(tuple(float(value) for value in scene_bbox)),
                "clocks_by_label": dict(clocks_by_label),
                "option_cards_by_label": dict(option_cards_by_label),
                "option_card_bboxes_px": {
                    str(label): list(option_cards_by_label[str(label)]["card_bbox_px"])
                    for label in query.option_labels
                },
                "correct_label": str(query.correct_label),
                "correct_option_bbox_px": list(correct_option_bbox),
            },
            "execution_trace": {
                "query_id": str(query.query_id),
                "prompt_query_key": PROMPT_QUERY_KEY,
                "scene_variant": str(query.scene_variant),
                "style_variant": str(query.style_variant),
                "accent_color_name": str(query.accent_color_name),
                "clock_labels": [str(label) for label in query.clock_labels],
                "option_labels": [str(label) for label in query.option_labels],
                "shown_total_minutes_by_label": {str(key): int(value) for key, value in query.shown_total_minutes_by_label.items()},
                "shown_time_text_by_label": {
                    str(label): str(format_clock_hhmm(int(query.shown_total_minutes_by_label[str(label)])))
                    for label in query.clock_labels
                },
                "true_order_labels": [str(label) for label in query.true_order_labels],
                "true_order_text": str(true_order_text),
                "true_order_total_minutes": [int(value) for value in true_order_times],
                "true_order_time_texts": [str(format_clock_hhmm(int(value))) for value in true_order_times],
                "option_order_labels_by_label": {
                    str(label): [str(value) for value in query.option_order_labels_by_label[str(label)]]
                    for label in query.option_labels
                },
                "option_order_text_by_label": {
                    str(label): _order_text(tuple(str(value) for value in query.option_order_labels_by_label[str(label)]))
                    for label in query.option_labels
                },
                "correct_label": str(query.correct_label),
                "correct_order_text": str(true_order_text),
                "earliest_clock_label": str(query.true_order_labels[0]),
                "earliest_total_minutes": int(true_order_times[0]),
                "earliest_hour": int(correct_hour),
                "earliest_minute": int(correct_minute),
                "hour_support": [int(query.hour_support[0]), int(query.hour_support[1])],
                "minute_support": [int(value) for value in query.minute_support],
                "min_hand_angle_gap_deg": float(query.min_hand_angle_gap_deg),
                "min_compare_gap_minutes": int(query.min_compare_gap_minutes),
                "query_id_probabilities": dict(query.query_id_probabilities),
                "scene_variant_probabilities": dict(query.scene_variant_probabilities),
                "style_variant_probabilities": dict(query.style_variant_probabilities),
                "accent_color_name_probabilities": dict(query.accent_color_name_probabilities),
                "question_format": PROMPT_QUERY_KEY,
                "supporting_bbox_roles": ["correct_option"],
            },
            "witness_symbolic": {
                "type": "bbox",
                "value": list(annotation_payload.value),
            },
            "projected_annotation": dict(annotation_payload.projected_annotation),
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


__all__ = ["SymbolicClockTimeOrderLabelTask"]
