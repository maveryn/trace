"""Identity-free sampling helpers for Mancala pit-board tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .state import (
    DEFAULTS,
    LABELS,
    SCENE_VARIANTS,
    STYLE_VARIANTS,
    IntegerAxisSelection,
    LabelAxisSelection,
    MancalaSceneAxes,
)


def resolve_mancala_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    axis_name: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> tuple[str, dict[str, float]]:
    """Resolve a named scene axis using the shared games sampler."""

    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace}.{axis_name}",
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported_variants=tuple(str(value) for value in supported),
    )


def resolve_mancala_scene_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
) -> MancalaSceneAxes:
    """Resolve scene-level visual and seed-count axes common to both tasks."""

    scene_variant, scene_probs = resolve_mancala_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        axis_name="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SCENE_VARIANTS,
    )
    style_variant, style_probs = resolve_mancala_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        axis_name="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=STYLE_VARIANTS,
    )
    return MancalaSceneAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        scene_variant_probabilities=dict(scene_probs),
        style_variant_probabilities=dict(style_probs),
        min_seed_count_per_pit=int(group_default(gen_defaults, "min_seed_count_per_pit", DEFAULTS.min_seed_count_per_pit)),
        max_seed_count_per_pit=int(group_default(gen_defaults, "max_seed_count_per_pit", DEFAULTS.max_seed_count_per_pit)),
        min_source_seed_count=int(group_default(gen_defaults, "min_source_seed_count", DEFAULTS.min_source_seed_count)),
        max_source_seed_count=int(group_default(gen_defaults, "max_source_seed_count", DEFAULTS.max_source_seed_count)),
    )


def resolve_mancala_label_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    support_key: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    fallback_support: Sequence[str],
) -> LabelAxisSelection:
    """Resolve one visible pit-label target axis."""

    support = tuple(
        str(value)
        for value in group_default(
            gen_defaults,
            str(support_key),
            tuple(str(item) for item in fallback_support),
        )
    )
    value, probabilities = resolve_mancala_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        axis_name=str(explicit_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported=support,
    )
    return LabelAxisSelection(value=str(value), support=tuple(support), probabilities=dict(probabilities))


def resolve_mancala_integer_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> IntegerAxisSelection:
    """Resolve one integer target axis and preserve support metadata."""

    value, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(item) for item in fallback_support),
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(item) for item in fallback_support),
    )
    return IntegerAxisSelection(
        value=int(value),
        support=tuple(int(item) for item in support),
        probabilities=dict(probabilities),
    )


def seed_count_bounds(scene_variant: str) -> Tuple[int, int]:
    """Return the seed-count range implied by one visual density variant."""

    if str(scene_variant) == "low_seed":
        return (0, 5)
    if str(scene_variant) == "busy_seed":
        return (2, 8)
    return (0, 8)


def random_initial_counts(*, rng: Any, axes: MancalaSceneAxes) -> list[int]:
    """Sample per-pit seed counts within scene and visual-density bounds."""

    low, high = seed_count_bounds(str(axes.scene_variant))
    low = int(max(int(axes.min_seed_count_per_pit), int(low)))
    high = int(min(int(axes.max_seed_count_per_pit), int(high)))
    return [int(rng.randint(low, high)) for _ in range(len(LABELS))]


__all__ = [
    "random_initial_counts",
    "resolve_mancala_integer_axis",
    "resolve_mancala_label_axis",
    "resolve_mancala_named_axis",
    "resolve_mancala_scene_axes",
    "seed_count_bounds",
]
