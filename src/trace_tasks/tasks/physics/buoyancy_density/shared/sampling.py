"""Sampling helpers for buoyancy-density scene parameters."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.variant_sampling import (
    apply_balanced_variant_sampling,
    resolve_variant,
)

from .formulas import object_density_tenths, parse_fraction
from .state import (
    DEFAULT_FRACTIONS,
    OBJECT_SHAPES,
    SCENE_NAMESPACE,
    SCENE_VARIANTS,
    BuoyancyScenario,
)


def fraction_support(
    *,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> Tuple[Tuple[int, int], ...]:
    """Resolve supported submerged fractions."""

    raw = params.get(
        "submerged_fraction_support",
        group_default(defaults, "submerged_fraction_support", None),
    )
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        out: List[Tuple[int, int]] = []
        for item in raw:
            if isinstance(item, str) and "/" in item:
                out.append(parse_fraction(item))
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                out.append(parse_fraction(item))
        if out:
            return tuple(out)
    return DEFAULT_FRACTIONS


def liquid_density_support_tenths(
    *,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> Tuple[int, ...]:
    """Resolve supported liquid-density values stored in tenths."""

    raw = params.get(
        "liquid_density_tenths_support",
        group_default(defaults, "liquid_density_tenths_support", list(range(8, 31))),
    )
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple(int(value) for value in raw)
    return tuple(range(8, 31))


def feasible_scenarios(
    *,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> Tuple[Tuple[int, int, int, int], ...]:
    """Return feasible fraction/liquid/object-density tuples."""

    scenarios: List[Tuple[int, int, int, int]] = []
    explicit_fraction = params.get("submerged_fraction")
    explicit_liquid = params.get("liquid_density_tenths")
    fraction_values = fraction_support(params=params, defaults=defaults)
    if explicit_fraction is not None:
        fraction_values = (parse_fraction(explicit_fraction),)
    liquid_values = liquid_density_support_tenths(params=params, defaults=defaults)
    if explicit_liquid is not None:
        liquid_values = (int(explicit_liquid),)
    for num, den in fraction_values:
        for liquid_tenths in liquid_values:
            object_tenths = object_density_tenths(
                liquid_density_tenths=int(liquid_tenths),
                fraction_num=int(num),
                fraction_den=int(den),
            )
            if object_tenths is None:
                continue
            scenarios.append((int(num), int(den), int(liquid_tenths), int(object_tenths)))
    if not scenarios:
        raise ValueError("no feasible buoyancy-density scenarios for configured supports")
    return tuple(scenarios)


def answer_support_tenths(
    *,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> Tuple[int, ...]:
    """Resolve feasible object-density answer support in tenths."""

    explicit_support = params.get("target_answer_tenths_support")
    if isinstance(explicit_support, Sequence) and not isinstance(explicit_support, (str, bytes)):
        return tuple(sorted({int(value) for value in explicit_support}))
    return tuple(
        sorted(
            {
                int(item[3])
                for item in feasible_scenarios(params=params, defaults=defaults)
            }
        )
    )


def resolve_scene_variant(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the tank/container scene variant."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.scene_variant")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=defaults,
        supported_variants=SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=SCENE_VARIANTS,
        balance_flag_key="balanced_scene_variant_sampling",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        sampling_namespace=f"{namespace}.scene_variant",
    )
    return str(selected), {str(key): float(value) for key, value in probabilities.items()}


def resolve_object_shape(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the floating object shape."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.object_shape")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=defaults,
        supported_variants=OBJECT_SHAPES,
        explicit_key="object_shape",
        weights_key="object_shape_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=OBJECT_SHAPES,
        balance_flag_key="balanced_object_shape_sampling",
        explicit_key="object_shape",
        weights_key="object_shape_weights",
        sampling_namespace=f"{namespace}.object_shape",
    )
    return str(selected), {str(key): float(value) for key, value in probabilities.items()}


def resolve_target_answer_tenths(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[int, Dict[str, float]]:
    """Resolve the target object-density answer in tenths."""

    scenarios = feasible_scenarios(params=params, defaults=defaults)
    support = answer_support_tenths(params=params, defaults=defaults)
    explicit_answer = params.get("target_answer", params.get("object_density"))
    if explicit_answer is not None:
        selected = int(round(float(explicit_answer) * 10.0))
        if selected not in support:
            raise ValueError(
                f"target_answer {float(explicit_answer)} is not feasible "
                "for buoyancy-density supports"
            )
        return selected, {
            str(float(value) / 10.0): (1.0 if int(value) == selected else 0.0)
            for value in support
        }
    feasible_answers = sorted(
        {int(item[3]) for item in scenarios if int(item[3]) in set(support)}
    )
    if not feasible_answers:
        raise ValueError("no feasible target answers for configured buoyancy-density supports")
    if bool(
        params.get(
            "balanced_target_answer_sampling",
            group_default(defaults, "balanced_target_answer_sampling", True),
        )
    ):
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_answer")
        selected = int(rng.choice(feasible_answers))
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_answer")
        selected = int(rng.choice(feasible_answers))
    probability = 1.0 / float(len(feasible_answers))
    return selected, {
        str(float(value) / 10.0): float(probability)
        for value in feasible_answers
    }


def resolve_buoyancy_scenario(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> BuoyancyScenario:
    """Resolve all semantic state for one buoyancy-density instance."""

    scene_variant, scene_probs = resolve_scene_variant(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    object_shape, shape_probs = resolve_object_shape(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    target_tenths, target_probs = resolve_target_answer_tenths(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    matching = [
        item
        for item in feasible_scenarios(params=params, defaults=defaults)
        if int(item[3]) == int(target_tenths)
    ]
    if not matching:
        raise ValueError("target answer has no matching buoyancy scenario")
    rng = spawn_rng(int(instance_seed), f"{namespace}.scenario")
    num, den, liquid_tenths, object_tenths = rng.choice(matching)
    answer_support = tuple(
        float(value) / 10.0
        for value in answer_support_tenths(params=params, defaults=defaults)
    )
    return BuoyancyScenario(
        scene_variant=str(scene_variant),
        object_shape=str(object_shape),
        fraction_num=int(num),
        fraction_den=int(den),
        liquid_density_tenths=int(liquid_tenths),
        object_density_tenths=int(object_tenths),
        target_answer=float(object_tenths) / 10.0,
        answer_support=answer_support,
        target_answer_probabilities=dict(target_probs),
        scene_variant_probabilities=dict(scene_probs),
        object_shape_probabilities=dict(shape_probs),
    )


__all__ = [
    "answer_support_tenths",
    "feasible_scenarios",
    "fraction_support",
    "liquid_density_support_tenths",
    "resolve_buoyancy_scenario",
    "resolve_object_shape",
    "resolve_scene_variant",
    "resolve_target_answer_tenths",
]
