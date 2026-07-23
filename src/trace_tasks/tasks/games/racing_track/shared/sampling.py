"""Scene-neutral sampling helpers for racing-track scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis

from .defaults import CAR_LABELS, SCENE_ID, SUPPORTED_SCENE_VARIANTS, SUPPORTED_STYLE_VARIANTS
from .rules import car_entity_id, centerline_points, remaining_distance_to_finish, track_point_and_tangent
from .state import RacingTrackCar, RacingTrackSceneState


@dataclass(frozen=True)
class RacingTrackVisualAxes:
    """Resolved scene/style axes shared by racing-track tasks."""

    scene_variant: str
    style_variant: str
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]


def _resolve_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace_root: str,
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced racing-track visual axis."""

    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported_variants=[str(value) for value in supported],
    )


def resolve_racing_track_visual_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace_root: str = f"games.{SCENE_ID}",
) -> RacingTrackVisualAxes:
    """Resolve scene-level visual axes without task/objective branching."""

    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace_root=str(namespace_root),
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace_root=str(namespace_root),
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_STYLE_VARIANTS,
    )
    return RacingTrackVisualAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
    )


def make_racing_track_car(
    *,
    index: int,
    label: str,
    progress: float,
    scene_variant: str,
    track_width_px: int,
    track_height_px: int,
) -> RacingTrackCar:
    """Build one symbolic car from a progress value on the track."""

    center, tangent = track_point_and_tangent(
        scene_variant=str(scene_variant),
        progress=float(progress),
        track_width_px=int(track_width_px),
        track_height_px=int(track_height_px),
    )
    return RacingTrackCar(
        car_id=car_entity_id(int(index)),
        label=str(label),
        progress=float(progress),
        center_px=center,
        tangent_px=tangent,
        remaining_distance=remaining_distance_to_finish(float(progress)),
    )


def build_racing_track_state(
    *,
    cars: Sequence[RacingTrackCar],
    axes: RacingTrackVisualAxes,
    track_width_px: int,
    track_height_px: int,
    construction_mode: str,
) -> RacingTrackSceneState:
    """Build neutral racing-track state from task-selected car placements."""

    finish_point, finish_tangent = track_point_and_tangent(
        scene_variant=str(axes.scene_variant),
        progress=0.0,
        track_width_px=int(track_width_px),
        track_height_px=int(track_height_px),
    )
    return RacingTrackSceneState(
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        track_width_px=int(track_width_px),
        track_height_px=int(track_height_px),
        centerline_points_px=centerline_points(
            scene_variant=str(axes.scene_variant),
            track_width_px=int(track_width_px),
            track_height_px=int(track_height_px),
        ),
        finish_point_px=finish_point,
        finish_tangent_px=finish_tangent,
        cars=tuple(cars),
        construction_mode=str(construction_mode),
    )


def shuffled_car_labels(rng: Any, count: int) -> Tuple[str, ...]:
    """Return unique shuffled car labels for one racing-track state."""

    labels = list(CAR_LABELS[: int(count)])
    rng.shuffle(labels)
    return tuple(str(label) for label in labels)


__all__ = [
    "RacingTrackVisualAxes",
    "build_racing_track_state",
    "make_racing_track_car",
    "resolve_racing_track_visual_axes",
    "shuffled_car_labels",
]
