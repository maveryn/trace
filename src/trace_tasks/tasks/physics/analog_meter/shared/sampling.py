"""Sampling helpers for analog-meter apparatus parameters."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import uniform_probability_map

from .state import METER_PROFILES, PROFILE_IDS_BY_METER_KIND, MeterProfile, MeterScenario, SCENE_NAMESPACE


def probability_map(values: Sequence[str], selected: str | None = None) -> Dict[str, float]:
    """Return a string-keyed probability map for a support set."""

    resolved = tuple(str(value) for value in values)
    if selected is not None:
        return {value: (1.0 if value == str(selected) else 0.0) for value in resolved}
    if not resolved:
        return {}
    probability = 1.0 / float(len(resolved))
    return {value: float(probability) for value in resolved}


def support_from_defaults(
    defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
) -> Tuple[int, ...]:
    """Resolve a non-empty sorted integer support from scene defaults."""

    raw = group_default(defaults, str(key), tuple(int(value) for value in fallback))
    support: List[int] = []
    for value in raw:
        selected = int(value)
        if selected not in support:
            support.append(selected)
    if not support:
        raise ValueError(f"{key} must contain at least one integer")
    return tuple(sorted(support))


def resolve_meter_profile(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    meter_kind: str,
    namespace: str = SCENE_NAMESPACE,
) -> Tuple[MeterProfile, Dict[str, float]]:
    """Select one supported analog meter profile for a semantic meter kind."""

    supported = PROFILE_IDS_BY_METER_KIND[str(meter_kind)]
    explicit_profile = str(params.get("meter_profile") or "").strip()
    explicit_unit = str(params.get("unit") or "").strip()
    if explicit_profile:
        if explicit_profile not in supported:
            raise ValueError(f"unsupported meter_profile for {meter_kind}: {explicit_profile}")
        return METER_PROFILES[explicit_profile], probability_map(supported, selected=explicit_profile)
    if explicit_unit:
        matches = [profile_id for profile_id in supported if str(METER_PROFILES[profile_id].unit) == explicit_unit]
        if not matches:
            raise ValueError(f"unsupported unit for {meter_kind}: {explicit_unit}")
        return METER_PROFILES[matches[0]], probability_map(supported, selected=matches[0])
    if len(supported) == 1:
        return METER_PROFILES[supported[0]], probability_map(supported, selected=supported[0])

    weights = group_default(defaults, "meter_profile_weights", {})
    enabled = bool(params.get("balanced_meter_profile_sampling", group_default(defaults, "balanced_meter_profile_sampling", True)))
    if enabled:
        rng = spawn_rng(int(instance_seed), f"{namespace}.meter_profile.{meter_kind}")
        selected = str(rng.choice(supported))
    else:
        raw_weights = params.get("meter_profile_weights", weights)
        weighted: List[str] = []
        for profile_id in supported:
            weight = float(raw_weights.get(profile_id, 1.0)) if isinstance(raw_weights, Mapping) else 1.0
            weighted.extend([profile_id] * max(0, int(round(weight))))
        if not weighted:
            weighted = list(supported)
        rng = spawn_rng(int(instance_seed), f"{namespace}.meter_profile.{meter_kind}")
        selected = weighted[int(rng.randrange(len(weighted)))]
    return METER_PROFILES[str(selected)], probability_map(supported)


def resolve_readout_value(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    profile: MeterProfile,
    namespace: str = SCENE_NAMESPACE,
) -> Tuple[int, Dict[str, float]]:
    """Select the tick-aligned integer readout for one meter profile."""

    support = support_from_defaults(defaults, f"{profile.profile_id}_answer_support", profile.answer_support)
    explicit = params.get("readout_value", params.get("target_answer"))
    if explicit is not None:
        selected = int(explicit)
        if selected not in set(support):
            raise ValueError(f"unsupported readout_value for {profile.profile_id}: {selected}")
        return int(selected), uniform_probability_map(support, selected=int(selected))
    if bool(params.get("balanced_target_answer_sampling", group_default(defaults, "balanced_target_answer_sampling", True))):
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_answer.{profile.profile_id}")
        selected = int(rng.choice(support))
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_answer.{profile.profile_id}")
        selected = int(support[int(rng.randrange(len(support)))])
    return int(selected), uniform_probability_map(support)


def resolve_meter_scenario(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    meter_kind: str,
    namespace: str = SCENE_NAMESPACE,
) -> MeterScenario:
    """Resolve a complete meter profile and readout without public task routing."""

    profile, profile_probabilities = resolve_meter_profile(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        meter_kind=str(meter_kind),
        namespace=str(namespace),
    )
    readout_value, answer_probabilities = resolve_readout_value(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        profile=profile,
        namespace=str(namespace),
    )
    return MeterScenario(
        profile=profile,
        readout_value=int(readout_value),
        meter_profile_probabilities=profile_probabilities,
        target_answer_probabilities=answer_probabilities,
    )


__all__ = [
    "probability_map",
    "resolve_meter_profile",
    "resolve_meter_scenario",
    "resolve_readout_value",
    "support_from_defaults",
]
