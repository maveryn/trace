"""Sampling primitives for piston-cylinder boundary-work diagrams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import uniform_probability_map
from trace_tasks.tasks.shared.support_sampling import resolve_integer_support
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .state import PistonGenerationDefaults, PistonScenario, SCENE_NAMESPACE, SUPPORTED_ORIENTATIONS


DEFAULTS = PistonGenerationDefaults()


def integer_support(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    key: str,
    fallback: Sequence[int],
) -> tuple[int, ...]:
    """Resolve one configured integer support."""

    return resolve_integer_support(
        params,
        gen_defaults=generation_defaults,
        key=str(key),
        fallback=fallback,
    )


def feasible_scenarios(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> tuple[tuple[int, int, int, int], ...]:
    """Enumerate pressure/volume tuples with nonzero supported boundary work."""

    pressures = integer_support(
        params,
        generation_defaults,
        key="pressure_mpa_support",
        fallback=DEFAULTS.pressure_mpa_support,
    )
    volumes = integer_support(
        params,
        generation_defaults,
        key="volume_l_support",
        fallback=DEFAULTS.volume_l_support,
    )
    configured_answers = set(
        integer_support(
            params,
            generation_defaults,
            key="boundary_work_answer_support",
            fallback=DEFAULTS.boundary_work_answer_support,
        )
    )
    min_delta = int(
        params.get(
            "min_volume_delta",
            group_default(generation_defaults, "min_volume_delta", DEFAULTS.min_volume_delta),
        )
    )
    max_delta = int(
        params.get(
            "max_volume_delta",
            group_default(generation_defaults, "max_volume_delta", DEFAULTS.max_volume_delta),
        )
    )
    scenarios: list[tuple[int, int, int, int]] = []
    for pressure in pressures:
        for initial_volume in volumes:
            for final_volume in volumes:
                delta = int(final_volume) - int(initial_volume)
                if delta == 0:
                    continue
                if abs(delta) < min_delta or abs(delta) > max_delta:
                    continue
                answer = int(pressure) * delta
                if int(answer) not in configured_answers:
                    continue
                scenarios.append((int(pressure), int(initial_volume), int(final_volume), int(answer)))
    if not scenarios:
        raise ValueError("no feasible piston-cylinder boundary-work scenarios for configured supports")
    return tuple(scenarios)


def probability_map(values: Sequence[str], selected: str | None = None) -> dict[str, float]:
    """Return a probability map over a finite string support."""

    if selected is not None:
        return {str(value): (1.0 if str(value) == str(selected) else 0.0) for value in values}
    probability = 1.0 / float(len(values)) if values else 0.0
    return {str(value): float(probability) for value in values}


def uniform_int_probability_map(values: Sequence[int], selected: int | None = None) -> dict[str, float]:
    """Return a JSON-stable probability map over integer values."""

    return {str(key): float(value) for key, value in uniform_probability_map(values, selected=selected).items()}


def resolve_orientation(
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    namespace: str = SCENE_NAMESPACE,
) -> tuple[str, dict[str, float]]:
    """Resolve the vertical or horizontal apparatus layout."""

    explicit = str(params.get("orientation") or "").strip()
    if explicit:
        if explicit not in SUPPORTED_ORIENTATIONS:
            raise ValueError(f"unsupported piston-cylinder orientation: {explicit}")
        return explicit, probability_map(SUPPORTED_ORIENTATIONS, selected=explicit)
    rng = spawn_rng(int(instance_seed), f"{namespace}.orientation")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=SUPPORTED_ORIENTATIONS,
        explicit_key="orientation",
        weights_key="orientation_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_ORIENTATIONS,
        balance_flag_key="balanced_orientation_sampling",
        explicit_key="orientation",
        weights_key="orientation_weights",
        sampling_namespace=f"{namespace}.orientation",
    )
    return str(selected), {str(key): float(value) for key, value in sorted(probabilities.items())}


def resolve_scenario(
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    namespace: str = SCENE_NAMESPACE,
) -> PistonScenario:
    """Resolve one exact constant-pressure boundary-work scenario."""

    orientation, orientation_probabilities = resolve_orientation(
        int(instance_seed),
        params,
        generation_defaults,
        namespace=str(namespace),
    )
    pressure_support = integer_support(
        params,
        generation_defaults,
        key="pressure_mpa_support",
        fallback=DEFAULTS.pressure_mpa_support,
    )
    volume_support = integer_support(
        params,
        generation_defaults,
        key="volume_l_support",
        fallback=DEFAULTS.volume_l_support,
    )
    feasible = feasible_scenarios(params, generation_defaults)
    explicit_pressure = params.get("pressure_mpa", params.get("pressure"))
    explicit_initial = params.get("initial_volume_l", params.get("volume_start"))
    explicit_final = params.get("final_volume_l", params.get("volume_end"))
    explicit_answer = params.get("target_answer", params.get("boundary_work_kj"))

    candidates = list(feasible)
    if explicit_pressure is not None:
        candidates = [item for item in candidates if int(item[0]) == int(explicit_pressure)]
    if explicit_initial is not None:
        candidates = [item for item in candidates if int(item[1]) == int(explicit_initial)]
    if explicit_final is not None:
        candidates = [item for item in candidates if int(item[2]) == int(explicit_final)]
    if explicit_answer is not None:
        candidates = [item for item in candidates if int(item[3]) == int(explicit_answer)]
    if not candidates:
        raise ValueError("explicit piston-cylinder parameters do not define a feasible scenario")

    if explicit_pressure is not None and explicit_initial is not None and explicit_final is not None:
        selected_tuple = candidates[0]
    elif explicit_answer is not None:
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_answer_tuple")
        selected_tuple = rng.choice(candidates)
    else:
        answers = sorted(
            {int(answer) for _, _, _, answer in candidates},
            key=lambda value: (abs(int(value)), int(value) < 0),
        )
        if bool(
            params.get(
                "balanced_target_answer_sampling",
                group_default(generation_defaults, "balanced_target_answer_sampling", True),
            )
        ):
            rng = spawn_rng(int(instance_seed), f"{namespace}.target_answer")
            target_answer = int(rng.choice(answers))
        else:
            rng = spawn_rng(int(instance_seed), f"{namespace}.target_answer")
            target_answer = int(answers[int(rng.randrange(len(answers)))])
        answer_candidates = [item for item in candidates if int(item[3]) == int(target_answer)]
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_answer_tuple")
        selected_tuple = rng.choice(answer_candidates)

    pressure, initial_volume, final_volume, answer = selected_tuple
    if explicit_answer is not None and int(answer) != int(explicit_answer):
        raise ValueError("target_answer does not match resolved piston-cylinder scenario")

    return PistonScenario(
        pressure_mpa=int(pressure),
        initial_volume_l=int(initial_volume),
        final_volume_l=int(final_volume),
        boundary_work_kj=int(answer),
        orientation=str(orientation),
        orientation_probabilities=orientation_probabilities,
        pressure_probabilities=uniform_int_probability_map(
            pressure_support,
            selected=int(pressure) if explicit_pressure is not None else None,
        ),
        initial_volume_probabilities=uniform_int_probability_map(
            volume_support,
            selected=int(initial_volume) if explicit_initial is not None else None,
        ),
        final_volume_probabilities=uniform_int_probability_map(
            volume_support,
            selected=int(final_volume) if explicit_final is not None else None,
        ),
        target_answer_probabilities=uniform_int_probability_map(
            sorted({int(item[3]) for item in feasible}),
            selected=int(answer) if explicit_answer is not None else None,
        ),
    )


__all__ = [
    "feasible_scenarios",
    "integer_support",
    "probability_map",
    "resolve_orientation",
    "resolve_scenario",
    "uniform_int_probability_map",
]
