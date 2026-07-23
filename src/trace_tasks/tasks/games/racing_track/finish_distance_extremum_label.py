"""Select the racing car closest or farthest from the finish line."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from ._lifecycle import AttemptRacingTrackResult, ObjectiveRacingTrackPlan, run_racing_track_lifecycle
from .shared.annotations import point_annotation_for_entity_id
from .shared.defaults import DEFAULTS, SCENE_ID
from .shared.rendering import RacingTrackRenderParams
from .shared.rules import circular_progress_gap, remaining_distance_to_finish
from .shared.sampling import (
    RacingTrackVisualAxes,
    build_racing_track_state,
    make_racing_track_car,
    shuffled_car_labels,
)


TASK_ID = "task_games__racing_track__finish_distance_extremum_label"
CLOSEST_QUERY_ID = "closest_to_finish_label"
FARTHEST_QUERY_ID = "farthest_from_finish_label"
SUPPORTED_QUERY_IDS = (CLOSEST_QUERY_ID, FARTHEST_QUERY_ID)


@dataclass(frozen=True)
class _DistanceTaskDefaults:
    """Stable fallback defaults for finish-distance extremum task axes."""

    car_count_support: Tuple[int, ...] = (4, 5, 6, 7)
    min_remaining_gap: float = 0.08


@dataclass(frozen=True)
class _DistanceAxes:
    """Task-owned car-count axis for one finish-distance instance."""

    car_count: int
    car_count_support: Tuple[int, ...]
    car_count_probabilities: Dict[str, float]


_DEFAULTS = _DistanceTaskDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _resolve_distance_axes(instance_seed: int, params: Mapping[str, Any]) -> _DistanceAxes:
    """Resolve task-owned car-count axis."""

    car_count, car_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="car_count_support",
        explicit_key="car_count",
        fallback_support=_DEFAULTS.car_count_support,
        namespace=f"{TASK_ID}.car_count",
        balanced_flag_key="balanced_car_count_sampling",
        namespace_support_permutation=True,
    )
    car_count_support = resolve_integer_support(
        params,
        gen_defaults=_GEN_DEFAULTS,
        key="car_count_support",
        fallback=_DEFAULTS.car_count_support,
    )
    return _DistanceAxes(
        car_count=int(car_count),
        car_count_support=tuple(int(value) for value in car_count_support),
        car_count_probabilities=dict(car_count_probabilities),
    )


def _sample_progresses(
    *,
    rng: Any,
    query_id: str,
    car_count: int,
    min_progress_gap: float,
    min_remaining_gap: float,
) -> Tuple[float, ...]:
    """Sample car progress values with a unique closest/farthest answer."""

    for _ in range(500):
        if str(query_id) == CLOSEST_QUERY_ID:
            answer_progress = rng.uniform(0.88, 0.96)
            trap_progress = rng.uniform(0.04, 0.12)
        else:
            answer_progress = rng.uniform(0.04, 0.12)
            trap_progress = rng.uniform(0.88, 0.96)
        progress_values = [round(float(answer_progress), 6), round(float(trap_progress), 6)]
        candidate_bands = (
            (0.20, 0.32),
            (0.38, 0.46),
            (0.60, 0.72),
            (0.76, 0.82),
        )
        for _slot in range(max(0, int(car_count) - 2)):
            placed = False
            for _attempt in range(120):
                low, high = rng.choice(candidate_bands)
                candidate = round(rng.uniform(float(low), float(high)), 6)
                if all(circular_progress_gap(candidate, existing) >= float(min_progress_gap) for existing in progress_values):
                    progress_values.append(float(candidate))
                    placed = True
                    break
            if not placed:
                break
        if len(progress_values) != int(car_count):
            continue
        remaining = [remaining_distance_to_finish(value) for value in progress_values]
        sorted_remaining = sorted(float(value) for value in remaining)
        min_gap = min(abs(float(a) - float(b)) for a, b in zip(sorted_remaining, sorted_remaining[1:]))
        if float(min_gap) < float(min_remaining_gap):
            continue
        if str(query_id) == CLOSEST_QUERY_ID and remaining.index(min(remaining)) != 0:
            continue
        if str(query_id) == FARTHEST_QUERY_ID and remaining.index(max(remaining)) != 0:
            continue
        return tuple(float(value) for value in progress_values)
    raise ValueError("failed to sample racing-track car positions")


def _build_distance_cars(
    *,
    rng: Any,
    query_id: str,
    axes: _DistanceAxes,
    visual_axes: RacingTrackVisualAxes,
    render_params: RacingTrackRenderParams,
    min_progress_gap: float,
    min_remaining_gap: float,
) -> tuple[tuple[Any, ...], str, str]:
    """Build cars and return the extremal answer entity and label."""

    progress_values = list(
        _sample_progresses(
            rng=rng,
            query_id=str(query_id),
            car_count=int(axes.car_count),
            min_progress_gap=float(min_progress_gap),
            min_remaining_gap=float(min_remaining_gap),
        )
    )
    labels = list(shuffled_car_labels(rng, int(axes.car_count)))
    paired = list(zip(labels, progress_values))
    rng.shuffle(paired)
    cars = tuple(
        make_racing_track_car(
            index=index,
            label=str(label),
            progress=float(progress),
            scene_variant=str(visual_axes.scene_variant),
            track_width_px=int(render_params.track_width_px),
            track_height_px=int(render_params.track_height_px),
        )
        for index, (label, progress) in enumerate(paired)
    )
    if str(query_id) == CLOSEST_QUERY_ID:
        answer_car = min(cars, key=lambda car: float(car.remaining_distance))
    else:
        answer_car = max(cars, key=lambda car: float(car.remaining_distance))
    return cars, str(answer_car.car_id), str(answer_car.label)


def _json_examples() -> tuple[str, str]:
    """Return valid format examples for racing-track extremum-label output."""

    return (
        json.dumps({"annotation": [486, 214], "answer": "C"}, separators=(",", ":"), ensure_ascii=False),
        json.dumps({"answer": "C"}, separators=(",", ":"), ensure_ascii=False),
    )


def _prepare_distance_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    query_id: str,
    visual_axes: RacingTrackVisualAxes,
    render_params: RacingTrackRenderParams,
) -> ObjectiveRacingTrackPlan:
    """Resolve distance-extremum axes and bind the scene constructor."""

    axes = _resolve_distance_axes(int(instance_seed), params)
    min_progress_gap = float(params.get("min_progress_gap", group_default(_GEN_DEFAULTS, "min_progress_gap", DEFAULTS.min_progress_gap)))
    min_remaining_gap = float(params.get("min_remaining_gap", group_default(_GEN_DEFAULTS, "min_remaining_gap", _DEFAULTS.min_remaining_gap)))

    def construct_attempt(rng: Any) -> AttemptRacingTrackResult:
        """Build a track with one unique closest or farthest car answer."""

        cars, answer_entity_id, answer_label = _build_distance_cars(
            rng=rng,
            query_id=str(query_id),
            axes=axes,
            visual_axes=visual_axes,
            render_params=render_params,
            min_progress_gap=float(min_progress_gap),
            min_remaining_gap=float(min_remaining_gap),
        )
        state = build_racing_track_state(
            cars=cars,
            axes=visual_axes,
            track_width_px=int(render_params.track_width_px),
            track_height_px=int(render_params.track_height_px),
            construction_mode="sample_closed_loop_extremal_finish_distance",
        )
        return AttemptRacingTrackResult(
            state=state,
            answer_gt=TypedValue(type="string", value=str(answer_label)),
            annotation_entity_ids=(str(answer_entity_id),),
            build_annotation=lambda rendered: point_annotation_for_entity_id(rendered.rendered_scene, answer_entity_id),
            witness_type="object",
            query_params={
                "car_count_support": [int(value) for value in axes.car_count_support],
                "car_count_probabilities": dict(axes.car_count_probabilities),
            },
            relations_extra={
                "finish_point_px_local": [
                    round(float(state.finish_point_px[0]), 3),
                    round(float(state.finish_point_px[1]), 3),
                ],
            },
            execution_extra={
                "answer_label": str(answer_label),
                "answer_entity_id": str(answer_entity_id),
            },
        )

    json_example, json_example_answer_only = _json_examples()
    return ObjectiveRacingTrackPlan(
        attempt_namespace="games.racing_track.finish_distance_extremum",
        prompt_query_key=str(query_id),
        object_description_key=f"object_description_{str(visual_axes.scene_variant)}",
        rule_text_key="distance_rule_text",
        answer_hint_key=f"answer_hint_{str(query_id)}",
        annotation_hint_key=f"annotation_hint_{str(query_id)}",
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
        query_params={"query_id_probabilities": dict(query_probabilities)},
        construct_attempt=construct_attempt,
    )


@register_task
class GamesRacingTrackFinishDistanceExtremumTask:
    """Select the car closest or farthest from the finish along the track."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'spatial_relations')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_racing_track_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=CLOSEST_QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_distance_objective,
        )


__all__ = ["GamesRacingTrackFinishDistanceExtremumTask"]
