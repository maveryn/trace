"""Sampling helpers for equivalent circuit diagrams."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.physics.shared.style import SUPPORTED_PHYSICS_COLOR_NAMES
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from trace_tasks.tasks.shared.variant_sampling import (
    apply_balanced_variant_sampling,
    resolve_variant,
)

from .formulas import parallel_resistance, series_capacitance
from .state import (
    COMPONENT_KINDS,
    DEFAULT_RENDERING,
    SCENE_VARIANT,
    SCENE_VARIANTS,
    EquivalentCircuitLayout,
    EquivalentCircuitScenario,
)


def probability_map(values: Sequence[str], selected: str | None = None) -> Dict[str, float]:
    """Return a string-keyed probability map over a finite support."""

    resolved = tuple(str(value) for value in values)
    if selected is not None:
        return {value: (1.0 if value == str(selected) else 0.0) for value in resolved}
    if not resolved:
        return {}
    probability = 1.0 / float(len(resolved))
    return {value: float(probability) for value in resolved}


def _resolve_int_options(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    key: str,
    fallback: Sequence[int],
) -> Tuple[int, ...]:
    """Resolve a positive integer option list from params/config defaults."""

    raw = params.get(str(key), group_default(defaults, str(key), fallback))
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{key} must be a sequence of positive integers")
    values = tuple(int(value) for value in raw)
    if not values or any(int(value) < 1 for value in values):
        raise ValueError(f"{key} must contain at least one positive integer")
    return tuple(dict.fromkeys(values))


def value_bounds(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> Tuple[int, int]:
    """Resolve positive component value bounds."""

    min_value = int(
        params.get(
            "component_value_min",
            group_default(defaults, "component_value_min", DEFAULT_RENDERING.component_value_min),
        )
    )
    max_value = int(
        params.get(
            "component_value_max",
            group_default(defaults, "component_value_max", DEFAULT_RENDERING.component_value_max),
        )
    )
    if min_value < 1 or max_value < min_value:
        raise ValueError("component value bounds must be positive and ordered")
    return int(min_value), int(max_value)


def component_count_options(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> Tuple[int, ...]:
    """Resolve branch component counts for one parallel block."""

    return _resolve_int_options(
        params,
        defaults,
        key="parallel_component_count_options",
        fallback=DEFAULT_RENDERING.parallel_component_count_options,
    )


def branch_count_options(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> Tuple[int, ...]:
    """Resolve branch counts used inside a parallel block."""

    return _resolve_int_options(
        params,
        defaults,
        key="series_parallel_branch_count_options",
        fallback=DEFAULT_RENDERING.series_parallel_branch_count_options,
    )


def parallel_block_count_options(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> Tuple[int, ...]:
    """Resolve number of parallel blocks in the mixed network."""

    return _resolve_int_options(
        params,
        defaults,
        key="parallel_block_count_options",
        fallback=DEFAULT_RENDERING.parallel_block_count_options,
    )


def resolve_scene_variant(params: Mapping[str, Any]) -> Tuple[str, Dict[str, float]]:
    """Resolve the single equivalent-circuit topology variant."""

    explicit = str(params.get("scene_variant") or SCENE_VARIANT).strip()
    if explicit != SCENE_VARIANT:
        raise ValueError(f"unsupported scene_variant: {explicit}")
    return SCENE_VARIANT, probability_map(SCENE_VARIANTS, selected=SCENE_VARIANT)


def resolve_accent_color(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Select one diagram accent color."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.accent_color_name")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=defaults,
        supported_variants=SUPPORTED_PHYSICS_COLOR_NAMES,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_PHYSICS_COLOR_NAMES,
        balance_flag_key="balanced_accent_color_name_sampling",
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        sampling_namespace=f"{namespace}.accent_color_name",
    )
    return str(selected), {str(key): float(value) for key, value in probabilities.items()}


def _can_realize_target(
    *,
    component_kind: str,
    target: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> bool:
    """Return true when the sampler has an exact construction for this target."""

    if int(target) < 1:
        return False
    _min_value, max_value = value_bounds(params, defaults)
    if str(component_kind) == "resistor":
        max_parallel = max(
            int(max_value) // int(count)
            for count in branch_count_options(params, defaults)
        )
        return any(
            any(
                int(block_count) <= (int(target) - int(series_sum)) <= int(block_count) * int(max_parallel)
                for series_sum in range(1, int(target))
            )
            for block_count in parallel_block_count_options(params, defaults)
        )
    for block_count in parallel_block_count_options(params, defaults):
        block_equivalent = int(block_count + 1) * int(target)
        if int(block_equivalent) > int(max_value):
            continue
        if any(
            int(count) <= int(block_equivalent) <= int(count) * int(max_value)
            for count in component_count_options(params, defaults)
        ):
            return True
    return False


def resolve_target_answer(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    component_kind: str,
    support_key: str,
    namespace: str,
) -> Tuple[int, Tuple[int, ...], Dict[str, float]]:
    """Resolve a feasible integer target answer."""

    fallback = getattr(DEFAULT_RENDERING, str(support_key))
    support = resolve_integer_support(
        params,
        gen_defaults=defaults,
        key=str(support_key),
        fallback=fallback,
    )
    feasible = tuple(
        int(value)
        for value in support
        if _can_realize_target(
            component_kind=str(component_kind),
            target=int(value),
            params=params,
            defaults=defaults,
        )
    )
    if not feasible:
        raise ValueError(f"no feasible {component_kind} target_answer values remain")
    resolved_params = dict(params)
    resolved_params[str(support_key)] = [int(value) for value in feasible]
    target_answer, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=resolved_params,
        gen_defaults=defaults,
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=feasible,
        namespace=f"{namespace}.target_answer.{component_kind}.{SCENE_VARIANT}",
        balanced_flag_key="balanced_target_answer_sampling",
    )
    if int(target_answer) not in set(feasible):
        raise ValueError(f"unsupported target_answer: {target_answer}")
    return int(target_answer), tuple(int(value) for value in feasible), dict(probabilities)


def resolve_equivalent_circuit_scenario(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    component_kind: str,
    support_key: str,
    namespace: str,
) -> EquivalentCircuitScenario:
    """Resolve the symbolic apparatus axes for one public objective."""

    if str(component_kind) not in set(COMPONENT_KINDS):
        raise ValueError(f"unsupported component_kind: {component_kind}")
    scene_variant, scene_probabilities = resolve_scene_variant(params)
    accent_color_name, accent_probabilities = resolve_accent_color(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    target_answer, target_support, target_probabilities = resolve_target_answer(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        component_kind=str(component_kind),
        support_key=str(support_key),
        namespace=str(namespace),
    )
    return EquivalentCircuitScenario(
        scene_variant=str(scene_variant),
        component_kind=str(component_kind),
        accent_color_name=str(accent_color_name),
        target_answer=int(target_answer),
        target_answer_support=tuple(int(value) for value in target_support),
        scene_variant_probabilities=dict(scene_probabilities),
        accent_color_name_probabilities=dict(accent_probabilities),
        target_answer_probabilities=dict(target_probabilities),
    )


def _choose_from_options(rng: Any, options: Sequence[int]) -> int:
    values = tuple(int(value) for value in options)
    if not values:
        raise ValueError("empty option list")
    return int(values[int(rng.randrange(len(values)))])


def _compose_positive_sum(
    rng: Any,
    *,
    total: int,
    count: int,
    max_value: int,
) -> Tuple[int, ...]:
    """Compose one positive integer sum with bounded parts."""

    if int(count) <= 1:
        if int(total) > int(max_value):
            raise ValueError("sum component exceeds max value")
        return (int(total),)
    count = min(int(count), int(total))
    if int(total) > int(count) * int(max_value):
        raise ValueError("cannot compose target within max value")
    values = [1 for _ in range(int(count))]
    remaining = int(total) - int(count)
    guard = 0
    while int(remaining) > 0:
        guard += 1
        if guard > 10000:
            raise ValueError("failed to compose bounded sum")
        index = int(rng.randrange(len(values)))
        room = int(max_value) - int(values[index])
        if room <= 0:
            continue
        addition = int(rng.randint(1, min(int(room), int(remaining))))
        values[index] += int(addition)
        remaining -= int(addition)
    rng.shuffle(values)
    return tuple(int(value) for value in values)


def _compose_nonnegative_sum(
    rng: Any,
    *,
    total: int,
    count: int,
    max_value: int,
) -> Tuple[int, ...]:
    """Compose one nonnegative integer sum with bounded parts."""

    if int(count) < 1:
        raise ValueError("nonnegative composition requires at least one slot")
    if int(total) < 0 or int(total) > int(count) * int(max_value):
        raise ValueError("cannot compose bounded nonnegative sum")
    values = [0 for _ in range(int(count))]
    remaining = int(total)
    guard = 0
    while int(remaining) > 0:
        guard += 1
        if guard > 10000:
            raise ValueError("failed to compose bounded nonnegative sum")
        index = int(rng.randrange(len(values)))
        room = int(max_value) - int(values[index])
        if room <= 0:
            continue
        addition = int(rng.randint(1, min(int(room), int(remaining))))
        values[index] += int(addition)
        remaining -= int(addition)
    return tuple(int(value) for value in values)


def _sample_resistor_layout(
    rng: Any,
    *,
    target_answer: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> EquivalentCircuitLayout:
    """Sample a mixed resistor network with exact integer equivalent value."""

    _min_value, max_value = value_bounds(params, defaults)
    max_parallel = max(
        int(max_value) // int(count)
        for count in branch_count_options(params, defaults)
    )
    feasible_pairs: List[Tuple[int, int]] = []
    for block_count in parallel_block_count_options(params, defaults):
        for series_sum in range(1, int(target_answer)):
            parallel_equivalent_sum = int(target_answer) - int(series_sum)
            if (
                int(block_count) <= int(parallel_equivalent_sum) <= int(block_count) * int(max_parallel)
                and int(series_sum) <= (int(block_count) + 1) * int(max_value)
            ):
                feasible_pairs.append((int(block_count), int(series_sum)))
    if not feasible_pairs:
        raise ValueError("no feasible mixed resistor layout")
    block_count, series_sum = feasible_pairs[int(rng.randrange(len(feasible_pairs)))]
    parallel_equivalent_sum = int(target_answer) - int(series_sum)
    parallel_equivalents = _compose_positive_sum(
        rng,
        total=int(parallel_equivalent_sum),
        count=int(block_count),
        max_value=int(max_parallel),
    )
    blocks: List[Tuple[int, ...]] = []
    for parallel_equivalent in parallel_equivalents:
        feasible_branch_counts = tuple(
            int(count)
            for count in branch_count_options(params, defaults)
            if int(count) * int(parallel_equivalent) <= int(max_value)
        )
        branch_count = _choose_from_options(rng, feasible_branch_counts)
        blocks.append(tuple(int(branch_count) * int(parallel_equivalent) for _ in range(int(branch_count))))
    series_slots = _compose_nonnegative_sum(
        rng,
        total=int(series_sum),
        count=int(block_count) + 1,
        max_value=int(max_value),
    )
    equivalent = sum(parallel_resistance(block) for block in blocks) + sum(series_slots)
    return EquivalentCircuitLayout(
        scene_variant=SCENE_VARIANT,
        component_kind="resistor",
        series_values=tuple(),
        parallel_values=tuple(),
        parallel_blocks=tuple(tuple(block) for block in blocks),
        inter_block_series_values=tuple(int(value) for value in series_slots[1:-1]),
        outer_series_values=(int(series_slots[0]), int(series_slots[-1])),
        target_answer=int(target_answer),
        equivalent_value=equivalent,
    )


def _sample_capacitor_layout(
    rng: Any,
    *,
    target_answer: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> EquivalentCircuitLayout:
    """Sample a mixed capacitor network with exact integer equivalent value."""

    _min_value, max_value = value_bounds(params, defaults)
    feasible_block_counts: List[int] = []
    for block_count in parallel_block_count_options(params, defaults):
        block_equivalent = int(block_count + 1) * int(target_answer)
        if int(block_equivalent) <= int(max_value) and any(
            int(count) <= int(block_equivalent) <= int(count) * int(max_value)
            for count in component_count_options(params, defaults)
        ):
            feasible_block_counts.append(int(block_count))
    if not feasible_block_counts:
        raise ValueError("no feasible mixed capacitor layout")
    block_count = _choose_from_options(rng, feasible_block_counts)
    block_equivalent = int(block_count + 1) * int(target_answer)
    blocks: List[Tuple[int, ...]] = []
    for _block_index in range(int(block_count)):
        count_options = tuple(
            int(count)
            for count in component_count_options(params, defaults)
            if int(count) <= int(block_equivalent) <= int(count) * int(max_value)
        )
        count = _choose_from_options(rng, count_options)
        blocks.append(
            _compose_positive_sum(
                rng,
                total=int(block_equivalent),
                count=int(count),
                max_value=int(max_value),
            )
        )
    series_slots = [0 for _ in range(int(block_count) + 1)]
    series_slots[int(rng.randrange(len(series_slots)))] = int(block_equivalent)
    equivalent = series_capacitance([int(block_equivalent)] + [int(sum(block)) for block in blocks])
    return EquivalentCircuitLayout(
        scene_variant=SCENE_VARIANT,
        component_kind="capacitor",
        series_values=tuple(),
        parallel_values=tuple(),
        parallel_blocks=tuple(tuple(block) for block in blocks),
        inter_block_series_values=tuple(int(value) for value in series_slots[1:-1]),
        outer_series_values=(int(series_slots[0]), int(series_slots[-1])),
        target_answer=int(target_answer),
        equivalent_value=equivalent,
    )


def sample_equivalent_circuit_layout(
    rng: Any,
    *,
    scenario: EquivalentCircuitScenario,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> EquivalentCircuitLayout:
    """Sample one topology and check that it matches the selected answer."""

    if str(scenario.scene_variant) != SCENE_VARIANT:
        raise ValueError(f"unsupported scene_variant: {scenario.scene_variant}")
    if str(scenario.component_kind) == "resistor":
        layout = _sample_resistor_layout(
            rng,
            target_answer=int(scenario.target_answer),
            params=params,
            defaults=defaults,
        )
    else:
        layout = _sample_capacitor_layout(
            rng,
            target_answer=int(scenario.target_answer),
            params=params,
            defaults=defaults,
        )
    if layout.equivalent_value.denominator != 1:
        raise ValueError("sampled equivalent value is not integral")
    if int(layout.equivalent_value) != int(scenario.target_answer):
        raise ValueError("sampled equivalent value does not match target")
    return layout


__all__ = [
    "branch_count_options",
    "component_count_options",
    "parallel_block_count_options",
    "probability_map",
    "resolve_equivalent_circuit_scenario",
    "resolve_scene_variant",
    "resolve_target_answer",
    "sample_equivalent_circuit_layout",
    "value_bounds",
]
