"""Sampling helpers for ray-optics scene packages."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.physics.shared.style import SUPPORTED_PHYSICS_COLOR_NAMES
from trace_tasks.tasks.physics.shared.support_sampling import (
    resolve_integer_choice,
    resolve_integer_support,
)
from trace_tasks.tasks.shared.variant_sampling import (
    apply_balanced_variant_sampling,
    resolve_variant,
)

from .state import (
    BOUNCE_SCENE_VARIANTS,
    RAY_EVENT_BOUNCE,
    RAY_EVENT_TARGET_HIT,
    RayAxes,
    RayOpticsTaskDefaults,
    TARGET_SCENE_VARIANTS,
)


def target_support_key(*, scene_variant: str, ray_event_kind: str) -> str:
    """Return the active integer-answer support key."""

    if str(ray_event_kind) == RAY_EVENT_TARGET_HIT:
        return "target_hit_count_support"
    return f"bounce_count_support_{str(scene_variant)}"


def resolve_target_answer(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    fallback_defaults: RayOpticsTaskDefaults,
    scene_variant: str,
    ray_event_kind: str,
    namespace: str,
) -> Tuple[int, Dict[str, float]]:
    """Resolve one count answer with deterministic support balancing."""

    support_key = target_support_key(
        scene_variant=str(scene_variant),
        ray_event_kind=str(ray_event_kind),
    )
    fallback = getattr(fallback_defaults, support_key)
    return resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=fallback,
        namespace=f"{namespace}.target_answer.{scene_variant}.{ray_event_kind}",
        balanced_flag_key="balanced_target_answer_sampling",
        use_instance_seed_cycle=True,
        namespace_support_permutation=True,
    )


def _resolve_scene_variant(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    supported_scene_variants: Sequence[str],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the visible mirror-layout variant for one objective."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.scene_variant")
    scene_variant, scene_probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=tuple(str(value) for value in supported_scene_variants),
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )
    scene_variant = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        selected_variant=str(scene_variant),
        variant_probabilities=scene_probabilities,
        supported_variants=tuple(str(value) for value in supported_scene_variants),
        balance_flag_key="balanced_scene_variant_sampling",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        sampling_namespace=f"{namespace}.scene_variant",
    )
    return str(scene_variant), dict(scene_probabilities)


def resolve_ray_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    fallback_defaults: RayOpticsTaskDefaults,
    ray_event_kind: str,
    namespace: str,
) -> RayAxes:
    """Resolve scene, answer, and accent-color axes for one ray objective."""

    if str(ray_event_kind) == RAY_EVENT_BOUNCE:
        supported_scene_variants = BOUNCE_SCENE_VARIANTS
    elif str(ray_event_kind) == RAY_EVENT_TARGET_HIT:
        supported_scene_variants = TARGET_SCENE_VARIANTS
    else:
        raise ValueError(f"unsupported ray event kind: {ray_event_kind}")

    scene_variant, scene_probabilities = _resolve_scene_variant(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        supported_scene_variants=supported_scene_variants,
        namespace=str(namespace),
    )
    target_answer, target_answer_probabilities = resolve_target_answer(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        fallback_defaults=fallback_defaults,
        scene_variant=str(scene_variant),
        ray_event_kind=str(ray_event_kind),
        namespace=str(namespace),
    )
    color_rng = spawn_rng(int(instance_seed), f"{namespace}.accent_color_name")
    accent_color_name, accent_color_name_probabilities = resolve_variant(
        color_rng,
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=SUPPORTED_PHYSICS_COLOR_NAMES,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
    )
    accent_color_name = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        selected_variant=str(accent_color_name),
        variant_probabilities=accent_color_name_probabilities,
        supported_variants=SUPPORTED_PHYSICS_COLOR_NAMES,
        balance_flag_key="balanced_accent_color_name_sampling",
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        sampling_namespace=f"{namespace}.accent_color_name",
    )
    return RayAxes(
        scene_variant=str(scene_variant),
        ray_event_kind=str(ray_event_kind),
        accent_color_name=str(accent_color_name),
        target_answer=int(target_answer),
        scene_variant_probabilities=dict(scene_probabilities),
        accent_color_name_probabilities=dict(accent_color_name_probabilities),
        target_answer_probabilities=dict(target_answer_probabilities),
    )


def answer_support(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    fallback_defaults: RayOpticsTaskDefaults,
    scene_variant: str,
    ray_event_kind: str,
) -> list[int]:
    """Return the active answer support for trace metadata."""

    support_key = target_support_key(
        scene_variant=str(scene_variant),
        ray_event_kind=str(ray_event_kind),
    )
    return [
        int(value)
        for value in resolve_integer_support(
            params,
            gen_defaults=gen_defaults,
            key=str(support_key),
            fallback=getattr(fallback_defaults, str(support_key)),
        )
    ]


__all__ = [
    "answer_support",
    "resolve_ray_axes",
    "resolve_target_answer",
    "target_support_key",
]
