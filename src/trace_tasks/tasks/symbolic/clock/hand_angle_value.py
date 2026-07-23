"""Compute the smaller angle between analog clock hands."""

from __future__ import annotations

import json

from ....core.seed import spawn_rng
from ....core.sampling import uniform_choice_with_probabilities
from ....core.types import TypedValue
from ...registry import register_task
from ...shared.config_defaults import group_default
from ...shared.time_format import (
    clock_hand_angle_gap_deg,
    clock_total_minutes,
    format_clock_hhmm,
    split_clock_total_minutes,
)
from ..shared.common import get_int_param as _get_int

from ._lifecycle import (
    SingleClockBinding,
    SingleClockObjective,
    SingleClockPlan,
    run_single_clock_task,
)
from .shared.annotations import clock_hand_segment_annotations
from .shared.defaults import DEFAULTS
from .shared.sampling import (
    feasible_clock_times,
    nearby_integer_distractors,
    option_value_map,
    resolve_clock_time_support,
    resolve_text_option_labels,
    sample_correct_option_label,
)
from .shared.state import ClockTextOptionSpec


TASK_ID = "task_symbolic__clock__hand_angle_value"
SUPPORTED_QUERY_IDS = ("single",)


def _integer_angle(total_minutes: int) -> int:
    angle = float(clock_hand_angle_gap_deg(int(total_minutes)))
    rounded = int(round(angle))
    if abs(angle - float(rounded)) > 1e-9:
        raise ValueError("clock hand angle is not an integer number of degrees")
    return int(rounded)


def _build_hand_angle_plan(*, params, gen_defaults, instance_seed):
    """Choose a valid shown time while keeping angle sampling explicit."""

    hour_support, minute_support, all_times = resolve_clock_time_support(
        params,
        gen_defaults=gen_defaults,
        fallback_hour_min=DEFAULTS.hour_min,
        fallback_hour_max=DEFAULTS.hour_max,
        fallback_minute_min=DEFAULTS.minute_min,
        fallback_minute_max=DEFAULTS.minute_max,
        fallback_minute_step=DEFAULTS.minute_step,
        context="clock hand-angle task",
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
    feasible = feasible_clock_times(all_times, min_hand_angle_gap_deg=float(min_gap))
    explicit_total = params.get("shown_total_minutes")
    if explicit_total is None and (
        params.get("shown_hour") is not None or params.get("shown_minute") is not None
    ):
        explicit_total = clock_total_minutes(
            int(params.get("shown_hour", 1)),
            int(params.get("shown_minute", 0)),
        )
    if explicit_total is not None:
        shown_total = int(explicit_total) % 720
        if shown_total not in feasible:
            raise ValueError("explicit shown time is outside feasible support")
    else:
        by_angle: dict[int, list[int]] = {}
        answer_min = _get_int(params, gen_defaults, "hand_angle_answer_min", 15)
        answer_max = _get_int(params, gen_defaults, "hand_angle_answer_max", 180)
        answer_step = _get_int(params, gen_defaults, "hand_angle_answer_step", 5)
        for total in feasible:
            try:
                angle = _integer_angle(int(total))
            except ValueError:
                continue
            if (
                answer_min <= angle <= answer_max
                and ((angle - answer_min) % answer_step) == 0
            ):
                by_angle.setdefault(int(angle), []).append(int(total))
        if not by_angle:
            raise ValueError("no feasible integer hand angles for configured support")
        explicit_angle = params.get("answer_angle_deg", params.get("target_angle_deg"))
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.angle")
        if explicit_angle is None:
            angle, _angle_probs = uniform_choice_with_probabilities(
                rng,
                tuple(sorted(by_angle)),
                sort_keys=True,
            )
            angle = int(angle)
        else:
            angle = int(explicit_angle)
            if angle not in by_angle:
                raise ValueError("explicit hand angle is outside feasible support")
        times = tuple(sorted(by_angle[int(angle)]))
        shown_total, _time_probs = uniform_choice_with_probabilities(
            rng,
            times,
            sort_keys=True,
        )
        shown_total = int(shown_total)

    answer_angle = _integer_angle(int(shown_total))
    shown_hour, shown_minute = split_clock_total_minutes(int(shown_total))
    option_labels = resolve_text_option_labels(params, gen_defaults=gen_defaults)
    correct_label, label_probs = sample_correct_option_label(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        seed_namespace=TASK_ID,
        labels=option_labels,
    )
    answer_min = _get_int(params, gen_defaults, "hand_angle_answer_min", 15)
    answer_max = _get_int(params, gen_defaults, "hand_angle_answer_max", 180)
    answer_step = _get_int(params, gen_defaults, "hand_angle_answer_step", 5)
    angle_support = tuple(range(int(answer_min), int(answer_max) + 1, int(answer_step)))
    distractors = nearby_integer_distractors(
        correct_value=int(answer_angle),
        support_values=angle_support,
        preferred_offsets=(5, 10, 15, 20, 30, 45, 60, 90),
        min_value=int(answer_min),
        max_value=int(answer_max),
    )
    option_values = option_value_map(
        labels=option_labels,
        correct_label=str(correct_label),
        correct_value=int(answer_angle),
        distractors=distractors,
    )
    json_example = json.dumps(
        {
            "annotation": [224, 770, 316, 836],
            "answer": "C",
        },
        separators=(",", ":"),
    )
    json_example_answer_only = json.dumps({"answer": "C"}, separators=(",", ":"))
    return SingleClockPlan(
        shown_total_minutes=int(shown_total),
        answer_gt=TypedValue(type="option_letter", value=str(correct_label)),
        query_id="single",
        question_format="hand_angle_value",
        query_params={
            "prompt_query_key": "hand_angle_value",
            "hour_support": [int(hour_support[0]), int(hour_support[1])],
            "minute_support": [int(value) for value in minute_support],
            "min_hand_angle_gap_deg": float(min_gap),
            "option_labels": [str(label) for label in option_labels],
            "correct_label_probabilities": {str(key): float(value) for key, value in label_probs.items()},
        },
        execution_fields={
            "shown_total_minutes": int(shown_total),
            "shown_hour": int(shown_hour),
            "shown_minute": int(shown_minute),
            "shown_time_text": str(format_clock_hhmm(int(shown_total))),
            "hand_angle_deg": int(answer_angle),
            "answer_value": int(answer_angle),
            "answer_label": str(correct_label),
            "option_values_by_label": {str(key): int(value) for key, value in option_values.items()},
            "answer_type": "option_letter",
        },
        json_example=str(json_example),
        json_example_answer_only=str(json_example_answer_only),
        answer_options=ClockTextOptionSpec(
            labels=tuple(str(label) for label in option_labels),
            correct_label=str(correct_label),
            text_by_label={str(key): str(value) for key, value in option_values.items()},
            value_by_label={str(key): int(value) for key, value in option_values.items()},
        ),
    )


def _build_hand_angle_objective(plan: SingleClockPlan, rendered_scene) -> SingleClockObjective:
    artifacts = clock_hand_segment_annotations(rendered_scene)
    return SingleClockObjective(
        annotation_gt=artifacts.annotation_gt,
        witness_symbolic={"type": str(artifacts.annotation_type), "value": list(artifacts.value)},
        projected_annotation=dict(artifacts.projected_annotation),
        execution_fields={
            "supporting_parts": ["hour_hand", "minute_hand"],
            "supporting_segments": list(artifacts.value),
        },
        render_map_extra={},
    )


_BINDING = SingleClockBinding(
    domain="symbolic",
    task_identifier=TASK_ID,
    supported_query_ids=SUPPORTED_QUERY_IDS,
    task_prompt_key="clock_hand_angle_value_query",
    prompt_query_key="hand_angle_value",
    object_description_prefix="object_description_hand_angle_value",
    annotation_hint_key="annotation_hint_hand_angle_value",
    answer_hint_key="answer_hint_hand_angle_value",
    build_plan=_build_hand_angle_plan,
    build_objective=_build_hand_angle_objective,
)


@register_task
class SymbolicClockHandAngleValueTask:
    """Compute the smaller angle between the hour and minute hands."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
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
