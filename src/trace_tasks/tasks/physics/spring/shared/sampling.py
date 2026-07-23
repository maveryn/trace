"""Sampling and symbolic construction for the spring physics scene."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.physics.shared.style import SUPPORTED_PHYSICS_COLOR_NAMES
from trace_tasks.tasks.physics.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .state import (
    SPRING_MODE_DIFFERENCE,
    SPRING_MODE_MISSING_EXTENSION,
    SPRING_MODE_MISSING_WEIGHT,
    SUPPORTED_SCENE_VARIANTS,
    SpringAxes,
    SpringColumnSpec,
    SpringSceneSpec,
    SpringTaskDefaults,
)


def scale_factor_support(
    params: Mapping[str, Any],
    *,
    generation_defaults: Mapping[str, Any],
    defaults: SpringTaskDefaults,
    spring_mode: str,
) -> Tuple[int, ...]:
    """Return the active integer scale-factor support."""

    base_support = resolve_integer_support(
        params,
        gen_defaults=generation_defaults,
        key="scale_factor_support",
        fallback=defaults.scale_factor_support,
    )
    if str(spring_mode) == SPRING_MODE_DIFFERENCE:
        return resolve_integer_support(
            params,
            gen_defaults=generation_defaults,
            key="extension_difference_scale_factor_support",
            fallback=base_support,
        )
    return base_support


def resolve_spring_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    defaults: SpringTaskDefaults,
    spring_mode: str,
    public_branch: str,
    internal_branch: str,
    public_branch_probabilities: Mapping[str, float],
    solve_for: str | None,
    target_support_key: str,
    target_support_fallback: Sequence[int],
    namespace: str,
) -> SpringAxes:
    """Resolve scene variant, accent color, and answer target for one instance."""

    axis_rng = spawn_rng(int(instance_seed), f"{namespace}.axes")
    scene_variant, scene_probs = resolve_variant(
        axis_rng,
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )
    scene_variant = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(scene_variant),
        variant_probabilities=scene_probs,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        balance_flag_key="balanced_scene_variant_sampling",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        sampling_namespace=f"{namespace}.scene_variant",
    )
    target_answer, target_answer_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        support_key=str(target_support_key),
        explicit_key="target_answer",
        fallback_support=tuple(int(value) for value in target_support_fallback),
        namespace=f"{namespace}.target_answer.{spring_mode}",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    color_rng = spawn_rng(int(instance_seed), f"{namespace}.accent_color_name")
    accent_color_name, accent_probs = resolve_variant(
        color_rng,
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=SUPPORTED_PHYSICS_COLOR_NAMES,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
    )
    accent_color_name = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(accent_color_name),
        variant_probabilities=accent_probs,
        supported_variants=SUPPORTED_PHYSICS_COLOR_NAMES,
        balance_flag_key="balanced_accent_color_name_sampling",
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        sampling_namespace=f"{namespace}.accent_color_name",
    )
    return SpringAxes(
        scene_variant=str(scene_variant),
        spring_mode=str(spring_mode),
        public_branch=str(public_branch),
        internal_branch=str(internal_branch),
        solve_for=None if solve_for is None else str(solve_for),
        accent_color_name=str(accent_color_name),
        target_answer=int(target_answer),
        scene_variant_probabilities=dict(scene_probs),
        public_branch_probabilities={str(key): float(value) for key, value in public_branch_probabilities.items()},
        accent_color_name_probabilities=dict(accent_probs),
        target_answer_probabilities=dict(target_answer_probabilities),
    )


def sample_spring_scene_spec(
    rng,
    *,
    scene_variant: str,
    spring_mode: str,
    target_answer: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    defaults: SpringTaskDefaults,
) -> SpringSceneSpec:
    """Sample one spring-extension symbolic scene that realizes the target answer."""

    scale_support = scale_factor_support(
        params,
        generation_defaults=generation_defaults,
        defaults=defaults,
        spring_mode=str(spring_mode),
    )
    weight_min = int(group_default(generation_defaults, "weight_value_min", defaults.weight_value_min))
    weight_max = int(group_default(generation_defaults, "weight_value_max", defaults.weight_value_max))
    extension_max = int(group_default(generation_defaults, "extension_value_max", defaults.extension_value_max))
    target_answer = int(target_answer)

    if str(spring_mode) == SPRING_MODE_MISSING_WEIGHT:
        scale_candidates = [
            int(scale)
            for scale in scale_support
            if int(scale) * int(target_answer) <= int(extension_max)
        ]
        if not scale_candidates:
            raise ValueError("no feasible scale factor for missing spring weight")
        scale_factor = int(rng.choice(scale_candidates))
        reference_candidates = [
            int(weight)
            for weight in range(int(weight_min), int(weight_max) + 1)
            if int(weight) != int(target_answer)
            and (int(weight) * int(scale_factor)) <= int(extension_max)
        ]
        if not reference_candidates:
            raise ValueError("no feasible reference weight for missing spring weight")
        left_weight = int(rng.choice(reference_candidates))
        left_extension = int(left_weight * scale_factor)
        right_extension = int(target_answer * scale_factor)
        left = SpringColumnSpec(
            column_id="left",
            shown_weight_value=int(left_weight),
            true_weight_value=int(left_weight),
            shown_extension_value=int(left_extension),
            true_extension_value=int(left_extension),
            missing_weight=False,
            missing_extension=False,
            detached_weight=False,
        )
        right = SpringColumnSpec(
            column_id="right",
            shown_weight_value=None,
            true_weight_value=int(target_answer),
            shown_extension_value=int(right_extension),
            true_extension_value=int(right_extension),
            missing_weight=True,
            missing_extension=False,
            detached_weight=False,
        )
        annotation_entity_ids = (
            "left_weight_block",
            "left_extension_marker",
            "right_weight_block",
            "right_extension_marker",
        )
    elif str(spring_mode) == SPRING_MODE_MISSING_EXTENSION:
        scale_candidates = [
            int(scale)
            for scale in scale_support
            if int(target_answer) % int(scale) == 0
            and weight_min <= int(target_answer // int(scale)) <= weight_max
        ]
        if not scale_candidates:
            raise ValueError("no feasible scale factor for missing spring extension")
        scale_factor = int(rng.choice(scale_candidates))
        right_weight = int(target_answer // scale_factor)
        reference_candidates = [
            int(weight)
            for weight in range(int(weight_min), int(weight_max) + 1)
            if int(weight) != int(right_weight)
            and (int(weight) * int(scale_factor)) <= int(extension_max)
        ]
        if not reference_candidates:
            raise ValueError("no feasible reference weight for missing spring extension")
        left_weight = int(rng.choice(reference_candidates))
        left_extension = int(left_weight * scale_factor)
        left = SpringColumnSpec(
            column_id="left",
            shown_weight_value=int(left_weight),
            true_weight_value=int(left_weight),
            shown_extension_value=int(left_extension),
            true_extension_value=int(left_extension),
            missing_weight=False,
            missing_extension=False,
            detached_weight=False,
        )
        right = SpringColumnSpec(
            column_id="right",
            shown_weight_value=int(right_weight),
            true_weight_value=int(right_weight),
            shown_extension_value=None,
            true_extension_value=int(target_answer),
            missing_weight=False,
            missing_extension=True,
            detached_weight=True,
        )
        annotation_entity_ids = (
            "left_weight_block",
            "left_extension_marker",
            "right_weight_block",
            "right_extension_marker",
        )
    else:
        feasible_pairs: List[Tuple[int, int, int, int]] = []
        for scale_factor in scale_support:
            scale_value = int(scale_factor)
            if int(target_answer) % int(scale_value) != 0:
                continue
            delta = int(target_answer) // int(scale_value)
            if int(delta) <= 0:
                continue
            max_weight_for_scale = min(
                int(weight_max),
                int(extension_max) // int(scale_value),
            )
            for lower_weight in range(int(weight_min), int(max_weight_for_scale - delta) + 1):
                upper_weight = int(lower_weight + delta)
                if int(upper_weight) > int(max_weight_for_scale):
                    continue
                feasible_pairs.append((int(scale_value), int(lower_weight), int(upper_weight), int(delta)))
        if not feasible_pairs:
            raise ValueError("no feasible weight pair for spring extension difference")
        scale_factor, lower_weight, upper_weight, _ = feasible_pairs[int(rng.randrange(len(feasible_pairs)))]
        if bool(rng.randrange(2)):
            left_weight, right_weight = int(lower_weight), int(upper_weight)
        else:
            left_weight, right_weight = int(upper_weight), int(lower_weight)
        left_extension = int(left_weight * scale_factor)
        right_extension = int(right_weight * scale_factor)
        left = SpringColumnSpec(
            column_id="left",
            shown_weight_value=int(left_weight),
            true_weight_value=int(left_weight),
            shown_extension_value=int(left_extension),
            true_extension_value=int(left_extension),
            missing_weight=False,
            missing_extension=False,
            detached_weight=False,
        )
        right = SpringColumnSpec(
            column_id="right",
            shown_weight_value=int(right_weight),
            true_weight_value=int(right_weight),
            shown_extension_value=int(right_extension),
            true_extension_value=int(right_extension),
            missing_weight=False,
            missing_extension=False,
            detached_weight=False,
        )
        annotation_entity_ids = ("left_extension_marker", "right_extension_marker")

    return SpringSceneSpec(
        scene_variant=str(scene_variant),
        spring_mode=str(spring_mode),
        scale_factor=int(scale_factor),
        left=left,
        right=right,
        target_answer=int(target_answer),
        annotation_entity_ids=tuple(str(item) for item in annotation_entity_ids),
    )


__all__ = ["resolve_spring_axes", "sample_spring_scene_spec", "scale_factor_support"]
