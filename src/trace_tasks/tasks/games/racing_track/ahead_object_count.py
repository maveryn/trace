"""Count cars ahead of a marked car on a racing track."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from ._lifecycle import AttemptRacingTrackResult, ObjectiveRacingTrackPlan, run_racing_track_lifecycle
from .shared.annotations import bbox_set_for_entity_ids
from .shared.defaults import DEFAULTS, SCENE_ID
from .shared.rendering import RacingTrackRenderParams
from .shared.rules import circular_progress_gap, progress_is_ahead_of_reference
from .shared.sampling import (
    RacingTrackVisualAxes,
    build_racing_track_state,
    make_racing_track_car,
    shuffled_car_labels,
)


TASK_ID = "task_games__racing_track__ahead_object_count"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "car_ahead_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
FORBIDDEN_PROGRESS_BANDS: Tuple[Tuple[float, float], ...] = ((0.49, 0.56),)


@dataclass(frozen=True)
class _AheadTaskDefaults:
    """Stable fallback defaults for racing-track ahead-count task axes."""

    car_count_support: Tuple[int, ...] = (5, 6, 7)
    target_answer_support: Tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class _AheadAxes:
    """Task-owned count target axes for one racing-track ahead instance."""

    car_count: int
    target_answer: int
    car_count_support: Tuple[int, ...]
    target_answer_support: Tuple[int, ...]
    car_count_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]


_DEFAULTS = _AheadTaskDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _resolve_ahead_axes(instance_seed: int, params: Mapping[str, Any]) -> _AheadAxes:
    """Resolve task-owned car-count and target-answer axes."""

    car_count, car_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="ahead_car_count_support",
        explicit_key="car_count",
        fallback_support=_DEFAULTS.car_count_support,
        namespace=f"{TASK_ID}.car_count",
        balanced_flag_key="balanced_car_count_sampling",
        namespace_support_permutation=True,
    )
    target_answer, target_answer_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="target_answer_support",
        explicit_key="target_answer",
        fallback_support=_DEFAULTS.target_answer_support,
        namespace=f"{TASK_ID}.target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    car_count = max(int(car_count), int(target_answer) + 1)
    return _AheadAxes(
        car_count=int(car_count),
        target_answer=int(target_answer),
        car_count_support=resolve_integer_support(
            params,
            gen_defaults=_GEN_DEFAULTS,
            key="ahead_car_count_support",
            fallback=_DEFAULTS.car_count_support,
        ),
        target_answer_support=resolve_integer_support(
            params,
            gen_defaults=_GEN_DEFAULTS,
            key="target_answer_support",
            fallback=_DEFAULTS.target_answer_support,
        ),
        car_count_probabilities=dict(car_count_probabilities),
        target_answer_probabilities=dict(target_answer_probabilities),
    )


def _progress_allowed(progress: float, *, existing: Sequence[float], min_gap: float) -> bool:
    """Return whether one progress value is visually separated from existing cars."""

    value = float(progress)
    if not (0.045 <= value <= 0.955):
        return False
    for low, high in FORBIDDEN_PROGRESS_BANDS:
        if float(low) <= value <= float(high):
            return False
    return all(circular_progress_gap(value, other) >= float(min_gap) for other in existing)


def _sample_progress_values(
    *,
    rng: Any,
    count: int,
    existing: Sequence[float],
    min_gap: float,
) -> Tuple[float, ...]:
    """Sample distinct progress values away from finish and direction-arrow clutter."""

    values = [float(value) for value in existing]
    sampled: list[float] = []
    for _slot in range(int(count)):
        for _attempt in range(500):
            candidate = round(rng.uniform(0.055, 0.945), 6)
            if _progress_allowed(candidate, existing=values, min_gap=float(min_gap)):
                sampled.append(float(candidate))
                values.append(float(candidate))
                break
        else:
            raise ValueError("failed to sample separated racing-track progress values")
    return tuple(sampled)


def _sample_ranked_progress_values(
    *,
    rng: Any,
    count: int,
    min_gap: float,
) -> Tuple[float, ...]:
    """Sample sorted progress values so rank encodes the target ahead count."""

    for _ in range(300):
        values = _sample_progress_values(rng=rng, count=int(count), existing=(), min_gap=float(min_gap))
        ordered = tuple(sorted(float(value) for value in values))
        if len(ordered) == int(count):
            return ordered
    raise ValueError("failed to sample ranked racing-track progress values")


def _build_ahead_cars(
    *,
    rng: Any,
    axes: _AheadAxes,
    visual_axes: RacingTrackVisualAxes,
    render_params: RacingTrackRenderParams,
    min_gap: float,
) -> tuple[tuple[Any, ...], str, tuple[str, ...]]:
    """Build cars whose progress ranks encode the requested ahead-count answer."""

    ordered_progress = _sample_ranked_progress_values(
        rng=rng,
        count=int(axes.car_count),
        min_gap=float(min_gap),
    )
    reference_rank = int(axes.car_count) - 1 - int(axes.target_answer)
    reference_progress = float(ordered_progress[int(reference_rank)])
    labels = shuffled_car_labels(rng, int(axes.car_count))
    cars = tuple(
        make_racing_track_car(
            index=index,
            label=str(label),
            progress=float(progress),
            scene_variant=str(visual_axes.scene_variant),
            track_width_px=int(render_params.track_width_px),
            track_height_px=int(render_params.track_height_px),
        )
        for index, (label, progress) in enumerate(zip(labels, ordered_progress))
    )
    reference_car = next(car for car in cars if abs(float(car.progress) - float(reference_progress)) <= 1e-9)
    annotation_ids = tuple(
        str(car.car_id)
        for car in sorted(cars, key=lambda item: float(item.progress))
        if str(car.car_id) != str(reference_car.car_id)
        and progress_is_ahead_of_reference(
            reference_progress=reference_car.progress,
            object_progress=car.progress,
        )
    )
    if len(annotation_ids) != int(axes.target_answer):
        raise ValueError("racing-track ahead count construction did not match target")
    return cars, str(reference_car.car_id), annotation_ids


def _json_examples() -> tuple[str, str]:
    """Return valid format examples for racing-track ahead-count output."""

    return (
        json.dumps({"annotation": [[462, 200, 510, 228], [588, 317, 636, 345]], "answer": 2}, separators=(",", ":"), ensure_ascii=False),
        json.dumps({"answer": 2}, separators=(",", ":"), ensure_ascii=False),
    )


def _prepare_ahead_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    _query_id: str,
    visual_axes: RacingTrackVisualAxes,
    render_params: RacingTrackRenderParams,
) -> ObjectiveRacingTrackPlan:
    """Resolve ahead-count axes and bind the exact-count scene constructor."""

    axes = _resolve_ahead_axes(int(instance_seed), params)
    min_gap = float(params.get("min_progress_gap", group_default(_GEN_DEFAULTS, "min_progress_gap", DEFAULTS.min_progress_gap)))

    def construct_attempt(rng: Any) -> AttemptRacingTrackResult:
        """Build a track whose selected reference car has the requested count ahead."""

        cars, reference_car_id, annotation_ids = _build_ahead_cars(
            rng=rng,
            axes=axes,
            visual_axes=visual_axes,
            render_params=render_params,
            min_gap=float(min_gap),
        )
        state = build_racing_track_state(
            cars=cars,
            axes=visual_axes,
            track_width_px=int(render_params.track_width_px),
            track_height_px=int(render_params.track_height_px),
            construction_mode="sample_forward_interval_count_from_marked_car",
        )
        return AttemptRacingTrackResult(
            state=state,
            answer_gt=TypedValue(type="integer", value=len(annotation_ids)),
            annotation_entity_ids=tuple(annotation_ids),
            build_annotation=lambda rendered: bbox_set_for_entity_ids(rendered.rendered_scene, annotation_ids),
            witness_type="object_set",
            marked_car_id=str(reference_car_id),
            query_params={
                "target_answer": int(axes.target_answer),
                "car_count_support": [int(value) for value in axes.car_count_support],
                "target_answer_support": [int(value) for value in axes.target_answer_support],
                "car_count_probabilities": dict(axes.car_count_probabilities),
                "target_answer_probabilities": dict(axes.target_answer_probabilities),
            },
            relations_extra={"reference_car_id": str(reference_car_id)},
            execution_extra={
                "reference_car_id": str(reference_car_id),
                "target_answer": int(axes.target_answer),
            },
        )

    json_example, json_example_answer_only = _json_examples()
    return ObjectiveRacingTrackPlan(
        attempt_namespace="games.racing_track.ahead_count",
        prompt_query_key=PROMPT_QUERY_KEY,
        object_description_key=f"object_description_{str(visual_axes.scene_variant)}_ahead_count",
        rule_text_key="ahead_rule_text",
        answer_hint_key="answer_hint_car_ahead_count",
        annotation_hint_key="annotation_hint_car_ahead_count",
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
        query_params={"query_id_probabilities": dict(query_probabilities)},
        construct_attempt=construct_attempt,
    )


@register_task
class GamesRacingTrackAheadObjectCountTask:
    """Count cars ahead of a marked car before the finish line."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_racing_track_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_ahead_objective,
        )


__all__ = ["GamesRacingTrackAheadObjectCountTask"]
