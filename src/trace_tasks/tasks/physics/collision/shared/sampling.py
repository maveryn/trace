"""Sampling helpers for physics collision tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.physics.shared.style import SUPPORTED_PHYSICS_COLOR_NAMES
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .mechanics import aftermath_option_directions, sample_sticky_scene_spec
from .state import (
    AftermathAxes,
    CollisionAftermathSpec,
    DIRECTION_ANGLE_DEGREES,
    DIRECTION_NAMES,
    OPTION_LETTERS,
    STICKY_DIRECTION_OPTION_LETTERS,
    SUPPORTED_AFTER_EFFECT_VARIANTS,
    SUPPORTED_STICKY_VARIANTS,
    AftermathRenderDefaults,
    StickyAxes,
    StickyRenderDefaults,
)


def probability_map(values: Sequence[int | str], selected: int | str | None = None) -> Dict[str, float]:
    """Return a string-keyed probability map over a finite support."""

    support = tuple(str(value) for value in values)
    if not support:
        return {}
    if selected is not None:
        selected_text = str(selected)
        return {value: (1.0 if value == selected_text else 0.0) for value in support}
    probability = 1.0 / float(len(support))
    return {value: float(probability) for value in support}


def resolve_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    supported: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one named sampling axis without public branch routing."""

    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=defaults,
        supported_variants=tuple(str(value) for value in supported),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=tuple(str(value) for value in supported),
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=str(namespace),
    )
    return str(selected), {str(key): float(value) for key, value in probabilities.items()}


def resolve_accent_color(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Select one visible accent color for the collision diagram."""

    return resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        supported=SUPPORTED_PHYSICS_COLOR_NAMES,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        balance_flag_key="balanced_accent_color_name_sampling",
        namespace=f"{namespace}.accent_color_name",
    )


def resolve_correct_option_letter(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
    option_letters: Sequence[str] = OPTION_LETTERS,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the visible option letter that carries the correct arrow."""

    option_letters = tuple(str(letter) for letter in option_letters)
    option_params = dict(params)
    if option_params.get("correct_option_letter") is None:
        raw_target = option_params.get("target_answer")
        if raw_target is not None and str(raw_target).strip().upper() in set(option_letters):
            option_params["correct_option_letter"] = str(raw_target).strip().upper()
    return resolve_named_axis(
        instance_seed=int(instance_seed),
        params=option_params,
        defaults=defaults,
        supported=option_letters,
        explicit_key="correct_option_letter",
        weights_key="correct_option_letter_weights",
        balance_flag_key="balanced_correct_option_letter_sampling",
        namespace=f"{namespace}.correct_option_letter",
    )


def resolve_aftermath_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> AftermathAxes:
    """Resolve all symbolic axes for one collision-aftermath instance."""

    scene_variant, scene_probs = resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        supported=SUPPORTED_AFTER_EFFECT_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace=f"{namespace}.scene_variant",
    )
    direction_params = dict(params)
    if params.get("aftermath_direction") is not None and params.get("final_motion_direction") is None:
        direction_params["final_motion_direction"] = str(params["aftermath_direction"])
    final_motion_direction, direction_probs = resolve_named_axis(
        instance_seed=int(instance_seed),
        params=direction_params,
        defaults=defaults,
        supported=DIRECTION_NAMES,
        explicit_key="final_motion_direction",
        weights_key="final_motion_direction_weights",
        balance_flag_key="balanced_final_motion_direction_sampling",
        namespace=f"{namespace}.final_motion_direction",
    )
    correct_option_letter, option_probs = resolve_correct_option_letter(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    accent_color_name, accent_probs = resolve_accent_color(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    return AftermathAxes(
        scene_variant=str(scene_variant),
        final_motion_direction=str(final_motion_direction),
        correct_option_letter=str(correct_option_letter),
        accent_color_name=str(accent_color_name),
        scene_variant_probabilities=dict(scene_probs),
        final_motion_direction_probabilities=dict(direction_probs),
        correct_option_letter_probabilities=dict(option_probs),
        accent_color_name_probabilities=dict(accent_probs),
    )


def make_aftermath_spec(instance_seed: int, axes: AftermathAxes) -> CollisionAftermathSpec:
    """Construct option arrows around one selected aftermath direction."""

    option_directions = aftermath_option_directions(
        instance_seed=int(instance_seed),
        final_motion_direction=str(axes.final_motion_direction),
        correct_option_letter=str(axes.correct_option_letter),
    )
    return CollisionAftermathSpec(
        scene_variant=str(axes.scene_variant),
        final_motion_direction=str(axes.final_motion_direction),
        correct_option_letter=str(axes.correct_option_letter),
        option_directions=dict(option_directions),
        option_angles_degrees={
            str(letter): float(DIRECTION_ANGLE_DEGREES[str(direction)])
            for letter, direction in option_directions.items()
        },
    )


def component_support(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> Tuple[int, ...]:
    """Return configured signed final-component answer support."""

    fallback = StickyRenderDefaults().component_answer_support
    return resolve_integer_support(
        params,
        gen_defaults=defaults,
        key="component_answer_support",
        fallback=fallback,
    )


def speed_answer_tenths_support(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> Tuple[int, ...]:
    """Return configured final-speed answer support in tenths of m/s."""

    fallback = StickyRenderDefaults().speed_answer_tenths_support
    return resolve_integer_support(
        params,
        gen_defaults=defaults,
        key="speed_answer_tenths_support",
        fallback=fallback,
    )


def mass_support(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[int, ...]:
    """Return configured puck mass support."""

    return resolve_integer_support(
        params,
        gen_defaults=defaults,
        key="mass_support",
        fallback=StickyRenderDefaults().mass_support,
    )


def speed_support(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[int, ...]:
    """Return configured input speed support."""

    return resolve_integer_support(
        params,
        gen_defaults=defaults,
        key="speed_support",
        fallback=StickyRenderDefaults().speed_support,
    )


def _coerce_speed_tenths(value: Any) -> int:
    """Resolve a visible speed answer into an integer tenths support key."""

    return int(round(float(value) * 10.0))


def resolve_speed_target_answer(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[int, Dict[str, float]]:
    """Resolve a one-decimal speed target as an integer number of tenths."""

    speed_params = dict(params)
    if speed_params.get("target_speed_tenths") is None and speed_params.get("target_answer") is not None:
        speed_params["target_speed_tenths"] = _coerce_speed_tenths(speed_params["target_answer"])
    support = speed_answer_tenths_support(speed_params, defaults)
    selected, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=speed_params,
        gen_defaults=defaults,
        support_key="speed_answer_tenths_support",
        explicit_key="target_speed_tenths",
        fallback_support=support,
        namespace=f"{namespace}.target_speed_tenths",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    return int(selected), {
        f"{float(key) / 10.0:.1f}": float(value)
        for key, value in sorted(probabilities.items())
    }


def resolve_sticky_direction_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> StickyAxes:
    """Resolve axes for the sticky-collision direction-option task."""

    scene_variant, scene_probs = resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        supported=SUPPORTED_STICKY_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace=f"{namespace}.scene_variant",
    )
    accent_color_name, accent_probs = resolve_accent_color(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    correct_option_letter, option_probs = resolve_correct_option_letter(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
        option_letters=STICKY_DIRECTION_OPTION_LETTERS,
    )
    return StickyAxes(
        scene_variant=str(scene_variant),
        component_axis=None,
        target_speed_tenths=None,
        accent_color_name=str(accent_color_name),
        target_answer=str(correct_option_letter),
        correct_option_letter=str(correct_option_letter),
        option_letters=tuple(STICKY_DIRECTION_OPTION_LETTERS),
        scene_variant_probabilities=dict(scene_probs),
        component_axis_probabilities={},
        accent_color_name_probabilities=dict(accent_probs),
        target_answer_probabilities=dict(option_probs),
        correct_option_letter_probabilities=dict(option_probs),
    )


def resolve_sticky_speed_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> StickyAxes:
    """Resolve axes for the sticky-collision final-speed task."""

    scene_variant, scene_probs = resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        supported=SUPPORTED_STICKY_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace=f"{namespace}.scene_variant",
    )
    accent_color_name, accent_probs = resolve_accent_color(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    correct_option_letter, option_probs = resolve_correct_option_letter(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    target_speed_tenths, target_probs = resolve_speed_target_answer(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    return StickyAxes(
        scene_variant=str(scene_variant),
        component_axis=None,
        target_speed_tenths=int(target_speed_tenths),
        accent_color_name=str(accent_color_name),
        target_answer=float(int(target_speed_tenths) / 10.0),
        correct_option_letter=str(correct_option_letter),
        option_letters=tuple(OPTION_LETTERS),
        scene_variant_probabilities=dict(scene_probs),
        component_axis_probabilities={},
        accent_color_name_probabilities=dict(accent_probs),
        target_answer_probabilities=dict(target_probs),
        correct_option_letter_probabilities=dict(option_probs),
    )


def sample_sticky_spec(
    *,
    instance_seed: int,
    attempt_index: int,
    axes: StickyAxes,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
):
    """Sample one sticky-collision scene from resolved semantic axes."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.attempt.{int(attempt_index)}")
    return sample_sticky_scene_spec(
        rng,
        scene_variant=str(axes.scene_variant),
        component_axis=axes.component_axis,
        target_answer=axes.target_answer,
        correct_option_letter=str(axes.correct_option_letter),
        params=params,
        masses=mass_support(params, defaults),
        speeds=speed_support(params, defaults),
        component_values=component_support(params, defaults),
        option_letters=tuple(axes.option_letters),
        target_speed_tenths=getattr(axes, "target_speed_tenths", None),
    )


def sample_sticky_spec_with_retries(
    *,
    instance_seed: int,
    max_attempts: int,
    axes: StickyAxes,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
):
    """Try deterministic sticky-scene candidates until one satisfies constraints."""

    for attempt_index in range(max(1, int(max_attempts))):
        try:
            return sample_sticky_spec(
                instance_seed=int(instance_seed),
                attempt_index=int(attempt_index),
                axes=axes,
                params=params,
                defaults=defaults,
                namespace=str(namespace),
            )
        except ValueError:
            continue
    raise ValueError("no feasible sticky collision scene after configured attempts")
