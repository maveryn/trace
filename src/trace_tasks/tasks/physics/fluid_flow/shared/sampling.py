"""Sampling primitives for fluid-flow continuity diagrams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import uniform_probability_map
from trace_tasks.tasks.shared.support_sampling import resolve_integer_support
from trace_tasks.tasks.shared.variant_sampling import (
    apply_balanced_variant_sampling,
    resolve_variant,
)

from .state import (
    DEFAULT_AREA_CM2_SUPPORT,
    DEFAULT_SPEED_M_S_SUPPORT,
    SCENE_NAMESPACE,
    SUPPORTED_MISSING_STATIONS,
    SUPPORTED_ORIENTATIONS,
    FlowScenario,
)


def _probability_map(values: Sequence[str], selected: str | None = None) -> dict[str, float]:
    if selected is not None:
        return {str(value): (1.0 if str(value) == str(selected) else 0.0) for value in values}
    probability = 1.0 / float(len(values)) if values else 0.0
    return {str(value): float(probability) for value in values}


def _uniform_int_probability_map(
    values: Sequence[int],
    selected: int | None = None,
) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in uniform_probability_map(values, selected=selected).items()
    }


def integer_support(
    params: Mapping[str, Any],
    *,
    generation_defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
) -> tuple[int, ...]:
    """Resolve a positive integer support from params and scene defaults."""

    return resolve_integer_support(
        params,
        gen_defaults=generation_defaults,
        key=str(key),
        fallback=tuple(int(value) for value in fallback),
    )


def resolve_orientation(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> tuple[str, dict[str, float]]:
    """Resolve horizontal vs vertical pipe orientation."""

    explicit = str(params.get("orientation") or "").strip()
    if explicit:
        if explicit not in SUPPORTED_ORIENTATIONS:
            raise ValueError(f"unsupported fluid-flow orientation: {explicit}")
        return explicit, _probability_map(SUPPORTED_ORIENTATIONS, selected=explicit)

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


def normalize_missing_station(value: Any) -> str | None:
    """Normalize station/speed aliases onto v1 or v2."""

    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "1": "v1",
        "station_1": "v1",
        "speed_1": "v1",
        "v1": "v1",
        "2": "v2",
        "station_2": "v2",
        "speed_2": "v2",
        "v2": "v2",
    }
    if text not in aliases:
        raise ValueError(f"unsupported missing fluid-flow speed station: {value}")
    return aliases[text]


def feasible_scenarios(
    params: Mapping[str, Any],
    *,
    generation_defaults: Mapping[str, Any],
) -> tuple[tuple[str, int, int, int, int, int], ...]:
    """Return feasible continuity tuples with integer missing speeds."""

    areas = integer_support(
        params,
        generation_defaults=generation_defaults,
        key="area_cm2_support",
        fallback=DEFAULT_AREA_CM2_SUPPORT,
    )
    speeds = integer_support(
        params,
        generation_defaults=generation_defaults,
        key="speed_m_s_support",
        fallback=DEFAULT_SPEED_M_S_SUPPORT,
    )
    speed_set = {int(value) for value in speeds}
    scenarios: list[tuple[str, int, int, int, int, int]] = []
    for area_1 in areas:
        for area_2 in areas:
            if int(area_1) == int(area_2):
                continue
            for speed_1 in speeds:
                numerator = int(area_1) * int(speed_1)
                if numerator % int(area_2) != 0:
                    continue
                speed_2 = numerator // int(area_2)
                if int(speed_2) not in speed_set:
                    continue
                scenarios.append(
                    ("v2", int(area_1), int(area_2), int(speed_1), int(speed_2), int(speed_2))
                )
                scenarios.append(
                    ("v1", int(area_1), int(area_2), int(speed_1), int(speed_2), int(speed_1))
                )
    if not scenarios:
        raise ValueError("no feasible fluid-flow continuity scenarios for configured supports")
    return tuple(scenarios)


def resolve_flow_scenario(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> FlowScenario:
    """Resolve one continuity scenario from explicit or sampled operands."""

    orientation, orientation_probabilities = resolve_orientation(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        namespace=str(namespace),
    )
    area_support = integer_support(
        params,
        generation_defaults=generation_defaults,
        key="area_cm2_support",
        fallback=DEFAULT_AREA_CM2_SUPPORT,
    )
    speed_support = integer_support(
        params,
        generation_defaults=generation_defaults,
        key="speed_m_s_support",
        fallback=DEFAULT_SPEED_M_S_SUPPORT,
    )
    feasible = list(feasible_scenarios(params, generation_defaults=generation_defaults))
    explicit_missing = normalize_missing_station(
        params.get("missing_speed_station", params.get("missing_station"))
    )
    explicit_area_1 = params.get("area_1_cm2", params.get("area1_cm2"))
    explicit_area_2 = params.get("area_2_cm2", params.get("area2_cm2"))
    explicit_speed_1 = params.get("speed_1_m_s", params.get("v1_m_s"))
    explicit_speed_2 = params.get("speed_2_m_s", params.get("v2_m_s"))
    explicit_answer = params.get("target_answer")

    candidates = list(feasible)
    if explicit_missing is not None:
        candidates = [item for item in candidates if str(item[0]) == str(explicit_missing)]
    if explicit_area_1 is not None:
        candidates = [item for item in candidates if int(item[1]) == int(explicit_area_1)]
    if explicit_area_2 is not None:
        candidates = [item for item in candidates if int(item[2]) == int(explicit_area_2)]
    if explicit_speed_1 is not None:
        candidates = [item for item in candidates if int(item[3]) == int(explicit_speed_1)]
    if explicit_speed_2 is not None:
        candidates = [item for item in candidates if int(item[4]) == int(explicit_speed_2)]
    if explicit_answer is not None:
        candidates = [item for item in candidates if int(item[5]) == int(explicit_answer)]
    if not candidates:
        raise ValueError("explicit fluid-flow parameters do not define a feasible continuity scenario")

    fully_bound_forward = (
        explicit_missing is not None
        and explicit_area_1 is not None
        and explicit_area_2 is not None
        and (
            (explicit_speed_1 is not None and str(explicit_missing) == "v2")
            or (explicit_speed_2 is not None and str(explicit_missing) == "v1")
        )
    )
    if fully_bound_forward:
        selected_tuple = candidates[0]
    else:
        missing_values = sorted({str(item[0]) for item in candidates})
        if explicit_missing is None and bool(
            params.get(
                "balanced_missing_station_sampling",
                group_default(generation_defaults, "balanced_missing_station_sampling", True),
            )
        ):
            rng = spawn_rng(int(instance_seed), f"{namespace}.missing_station")
            selected_missing = str(rng.choice(missing_values))
            candidates = [item for item in candidates if str(item[0]) == selected_missing]

        answers = sorted({int(item[5]) for item in candidates})
        if explicit_answer is None and bool(
            params.get(
                "balanced_target_answer_sampling",
                group_default(generation_defaults, "balanced_target_answer_sampling", True),
            )
        ):
            rng = spawn_rng(int(instance_seed), f"{namespace}.target_answer")
            selected_answer = int(rng.choice(answers))
            candidates = [item for item in candidates if int(item[5]) == int(selected_answer)]
        elif explicit_answer is None:
            rng = spawn_rng(int(instance_seed), f"{namespace}.target_answer")
            selected_answer = int(answers[int(rng.randrange(len(answers)))])
            candidates = [item for item in candidates if int(item[5]) == int(selected_answer)]

        rng = spawn_rng(int(instance_seed), f"{namespace}.scenario_tuple")
        selected_tuple = rng.choice(candidates)

    missing_station, area_1, area_2, speed_1, speed_2, answer = selected_tuple
    return FlowScenario(
        orientation=str(orientation),
        missing_station=str(missing_station),
        area_1_cm2=int(area_1),
        area_2_cm2=int(area_2),
        speed_1_m_s=int(speed_1),
        speed_2_m_s=int(speed_2),
        target_answer=int(answer),
        orientation_probabilities=orientation_probabilities,
        missing_station_probabilities=_probability_map(
            SUPPORTED_MISSING_STATIONS,
            selected=explicit_missing,
        ),
        area_1_cm2_probabilities=_uniform_int_probability_map(
            area_support,
            selected=int(area_1) if explicit_area_1 is not None else None,
        ),
        area_2_cm2_probabilities=_uniform_int_probability_map(
            area_support,
            selected=int(area_2) if explicit_area_2 is not None else None,
        ),
        speed_1_m_s_probabilities=_uniform_int_probability_map(
            speed_support,
            selected=int(speed_1) if explicit_speed_1 is not None else None,
        ),
        speed_2_m_s_probabilities=_uniform_int_probability_map(
            speed_support,
            selected=int(speed_2) if explicit_speed_2 is not None else None,
        ),
        target_answer_probabilities=_uniform_int_probability_map(
            sorted({int(item[5]) for item in feasible}),
            selected=int(answer) if explicit_answer is not None else None,
        ),
    )
