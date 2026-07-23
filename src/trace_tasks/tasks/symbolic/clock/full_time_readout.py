"""Read the full hour, minute, and second time from one analog clock."""

from __future__ import annotations

import json
from typing import Any, Mapping, Tuple

from ....core.seed import spawn_rng
from ....core.sampling import uniform_choice_with_probabilities
from ....core.types import TypedValue
from ...registry import register_task
from ...shared.config_defaults import group_default
from ...shared.time_format import (
    clock_hand_pair_angle_gaps_deg,
    clock_total_seconds,
    format_clock_hhmmss,
    split_clock_total_seconds,
)

from ._lifecycle import (
    SingleClockBinding,
    SingleClockObjective,
    SingleClockPlan,
    run_single_clock_task,
)
from .shared.annotations import clock_hand_segment_annotations
from .shared.defaults import DEFAULTS
from .shared.sampling import (
    option_value_map,
    resolve_clock_time_support,
    resolve_text_option_labels,
    sample_correct_option_label,
)
from .shared.state import ClockTextOptionSpec


TASK_ID = "task_symbolic__clock__full_time_readout"
SUPPORTED_QUERY_IDS = ("single",)

_DEFAULT_SECOND_MIN = 0
_DEFAULT_SECOND_MAX = 55
_DEFAULT_SECOND_STEP = 5


def _resolve_second_support(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
) -> Tuple[int, int, int]:
    """Resolve compact second-hand support from task params and config."""

    second_min = int(
        params.get(
            "second_min",
            group_default(gen_defaults, "second_min", _DEFAULT_SECOND_MIN),
        )
    )
    second_max = int(
        params.get(
            "second_max",
            group_default(gen_defaults, "second_max", _DEFAULT_SECOND_MAX),
        )
    )
    second_step = int(
        params.get(
            "second_step",
            group_default(gen_defaults, "second_step", _DEFAULT_SECOND_STEP),
        )
    )
    if not (0 <= second_min <= second_max <= 59):
        raise ValueError("clock full-time seconds must satisfy 0 <= min <= max <= 59")
    if second_step <= 0:
        raise ValueError("clock full-time second_step must be positive")
    values = tuple(range(second_min, second_max + 1, second_step))
    if not values:
        raise ValueError("clock full-time second support is empty")
    return int(second_min), int(second_max), int(second_step)


def _feasible_full_times(
    minute_times: Tuple[int, ...],
    second_values: Tuple[int, ...],
    *,
    min_hand_angle_gap_deg: float,
) -> Tuple[int, ...]:
    """Return canonical second-level times with all three hands separated."""

    feasible: list[int] = []
    for total_minutes in minute_times:
        for second in second_values:
            total_seconds = int(total_minutes * 60) + int(second)
            gaps = clock_hand_pair_angle_gaps_deg(int(total_seconds))
            if min(float(gap) for gap in gaps) >= float(min_hand_angle_gap_deg):
                feasible.append(int(total_seconds))
    return tuple(feasible)


def _explicit_total_seconds(params: Mapping[str, Any]) -> int | None:
    """Return an explicitly requested time if the caller supplied one."""

    if params.get("shown_total_seconds") is not None:
        return int(params["shown_total_seconds"])
    if params.get("shown_total_minutes") is not None:
        shown_second = int(params.get("shown_second", 0))
        if not 0 <= shown_second <= 59:
            raise ValueError("shown_second must be within 0..59")
        return (int(params["shown_total_minutes"]) * 60) + int(shown_second)
    if (
        params.get("shown_hour") is not None
        or params.get("shown_minute") is not None
        or params.get("shown_second") is not None
    ):
        return clock_total_seconds(
            int(params.get("shown_hour", 1)),
            int(params.get("shown_minute", 0)),
            int(params.get("shown_second", 0)),
        )
    return None


def _build_full_time_plan(*, params, gen_defaults, instance_seed):
    """Sample a three-hand clock time and bind the HH:MM:SS answer."""

    hour_support, minute_support, minute_times = resolve_clock_time_support(
        params,
        gen_defaults=gen_defaults,
        fallback_hour_min=DEFAULTS.hour_min,
        fallback_hour_max=DEFAULTS.hour_max,
        fallback_minute_min=DEFAULTS.minute_min,
        fallback_minute_max=DEFAULTS.minute_max,
        fallback_minute_step=DEFAULTS.minute_step,
        context="clock full-time readout task",
    )
    second_support = _resolve_second_support(params, gen_defaults=gen_defaults)
    second_values = tuple(
        range(int(second_support[0]), int(second_support[1]) + 1, int(second_support[2]))
    )
    min_gap = float(
        params.get(
            "min_hand_angle_gap_deg",
            group_default(
                gen_defaults,
                "min_hand_angle_gap_deg",
                DEFAULTS.min_hand_angle_gap_deg,
            ),
        )
    )
    if min_gap < 0.0:
        raise ValueError("min_hand_angle_gap_deg must be non-negative")

    feasible = _feasible_full_times(
        minute_times,
        second_values,
        min_hand_angle_gap_deg=float(min_gap),
    )
    explicit_total = _explicit_total_seconds(params)
    if explicit_total is not None:
        shown_total_seconds = int(explicit_total) % (12 * 60 * 60)
        if shown_total_seconds not in feasible:
            raise ValueError("explicit shown time is outside feasible support")
    else:
        if not feasible:
            raise ValueError("clock full-time support has no feasible separated hands")
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.time")
        shown_total_seconds, _time_probs = uniform_choice_with_probabilities(
            rng,
            feasible,
            sort_keys=True,
        )
        shown_total_seconds = int(shown_total_seconds)

    shown_hour, shown_minute, shown_second = split_clock_total_seconds(
        int(shown_total_seconds)
    )
    answer_text = format_clock_hhmmss(int(shown_total_seconds))
    option_labels = resolve_text_option_labels(params, gen_defaults=gen_defaults)
    correct_label, label_probs = sample_correct_option_label(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        seed_namespace=TASK_ID,
        labels=option_labels,
    )
    cycle_seconds = 12 * 60 * 60
    preferred_offsets = (5, -5, 10, -10, 15, -15, 30, -30, 60, -60, 300, -300, 3600, -3600)
    candidate_values = [
        int((int(shown_total_seconds) + int(offset)) % int(cycle_seconds))
        for offset in preferred_offsets
    ]
    candidate_values.extend(int(value) for value in feasible if int(value) != int(shown_total_seconds))
    option_values = option_value_map(
        labels=option_labels,
        correct_label=str(correct_label),
        correct_value=int(shown_total_seconds),
        distractors=candidate_values,
    )
    json_example = json.dumps(
        {
            "annotation": [224, 770, 316, 836],
            "answer": "C",
        },
        separators=(",", ":"),
    )
    json_example_answer_only = json.dumps(
        {"answer": "C"},
        separators=(",", ":"),
    )
    return SingleClockPlan(
        shown_total_minutes=int(shown_total_seconds // 60),
        shown_total_seconds=int(shown_total_seconds),
        show_second_hand=True,
        answer_gt=TypedValue(type="option_letter", value=str(correct_label)),
        query_id="single",
        question_format="full_time_readout",
        query_params={
            "prompt_query_key": "full_time_readout",
            "hour_support": [int(hour_support[0]), int(hour_support[1])],
            "minute_support": [int(value) for value in minute_support],
            "second_support": [int(value) for value in second_support],
            "min_hand_angle_gap_deg": float(min_gap),
            "option_labels": [str(label) for label in option_labels],
            "correct_label_probabilities": {str(key): float(value) for key, value in label_probs.items()},
        },
        execution_fields={
            "shown_total_seconds": int(shown_total_seconds),
            "shown_total_minutes": int(shown_total_seconds // 60),
            "shown_hour": int(shown_hour),
            "shown_minute": int(shown_minute),
            "shown_second": int(shown_second),
            "shown_time_text": str(answer_text),
            "answer_value": str(answer_text),
            "answer_total_seconds": int(shown_total_seconds),
            "answer_label": str(correct_label),
            "option_values_by_label": {str(key): int(value) for key, value in option_values.items()},
            "option_text_by_label": {
                str(key): str(format_clock_hhmmss(int(value)))
                for key, value in option_values.items()
            },
            "hand_angle_gaps_deg": [
                round(float(value), 6)
                for value in clock_hand_pair_angle_gaps_deg(int(shown_total_seconds))
            ],
            "answer_type": "option_letter",
        },
        json_example=str(json_example),
        json_example_answer_only=str(json_example_answer_only),
        answer_options=ClockTextOptionSpec(
            labels=tuple(str(label) for label in option_labels),
            correct_label=str(correct_label),
            text_by_label={
                str(key): str(format_clock_hhmmss(int(value)))
                for key, value in option_values.items()
            },
            value_by_label={str(key): int(value) for key, value in option_values.items()},
        ),
    )


def _build_full_time_objective(
    plan: SingleClockPlan,
    rendered_scene,
) -> SingleClockObjective:
    artifacts = clock_hand_segment_annotations(
        rendered_scene,
        include_second_hand=True,
    )
    return SingleClockObjective(
        annotation_gt=artifacts.annotation_gt,
        witness_symbolic={
            "type": str(artifacts.annotation_type),
            "value": list(artifacts.value),
        },
        projected_annotation=dict(artifacts.projected_annotation),
        execution_fields={
            "supporting_parts": ["hour_hand", "minute_hand", "second_hand"],
            "supporting_segments": list(artifacts.value),
        },
        render_map_extra={},
    )


_BINDING = SingleClockBinding(
    domain="symbolic",
    task_identifier=TASK_ID,
    supported_query_ids=SUPPORTED_QUERY_IDS,
    task_prompt_key="clock_full_time_readout_query",
    prompt_query_key="full_time_readout",
    object_description_prefix="object_description_full_time_readout",
    annotation_hint_key="annotation_hint_full_time_readout",
    answer_hint_key="answer_hint_full_time_readout",
    build_plan=_build_full_time_plan,
    build_objective=_build_full_time_objective,
)


@register_task
class SymbolicClockFullTimeReadoutTask:
    """Read the exact HH:MM:SS value from a three-hand analog clock."""

    task_id = TASK_ID
    reasoning_operations = ('direct_retrieval',)
    domain = "symbolic"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return run_single_clock_task(
            _BINDING,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


__all__ = ["SymbolicClockFullTimeReadoutTask"]
