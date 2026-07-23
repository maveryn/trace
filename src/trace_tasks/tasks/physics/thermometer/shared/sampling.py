"""Sampling helpers for thermometer scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default

from .formulas import convert_temperature, source_temperature_from_target
from .state import PROFILES, SCENE_NAMESPACE, ThermometerProfile, ThermometerScenario


def probability_map(values: Sequence[str], *, selected: str | None = None) -> Dict[str, float]:
    """Return a JSON-stable probability map."""

    resolved = tuple(str(value) for value in values)
    if selected is not None:
        return {str(value): (1.0 if str(value) == str(selected) else 0.0) for value in resolved}
    if not resolved:
        return {}
    probability = 1.0 / float(len(resolved))
    return {str(value): float(probability) for value in resolved}


def profile_ids_for_units(source_unit: str, target_unit: str) -> Tuple[str, ...]:
    """Return profile ids matching the requested source and target units."""

    supported = tuple(
        str(profile_id)
        for profile_id, profile in sorted(PROFILES.items())
        if profile.source_unit == str(source_unit) and profile.target_unit == str(target_unit)
    )
    if not supported:
        raise ValueError(f"no thermometer profiles for {source_unit} to {target_unit}")
    return supported


def resolve_profile(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    source_unit: str,
    target_unit: str,
) -> Tuple[ThermometerProfile, Dict[str, float]]:
    """Resolve one visible thermometer scale profile."""

    supported = profile_ids_for_units(str(source_unit), str(target_unit))
    explicit = str(params.get("scale_profile") or params.get("thermometer_profile") or "").strip()
    if explicit:
        if explicit not in supported:
            raise ValueError(f"unsupported scale_profile for {source_unit} to {target_unit}: {explicit}")
        return PROFILES[explicit], probability_map(supported, selected=explicit)
    balanced = bool(
        params.get(
            "balanced_scale_profile_sampling",
            group_default(generation_defaults, "balanced_scale_profile_sampling", True),
        )
    )
    if balanced:
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.scale_profile.{source_unit}.{target_unit}")
        selected = str(rng.choice(supported))
    else:
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.scale_profile.{source_unit}.{target_unit}")
        weights = group_default(generation_defaults, "scale_profile_weights", {})
        weighted = [max(0.0, float(weights.get(profile_id, 1.0))) for profile_id in supported]
        total = sum(weighted) or float(len(supported))
        threshold = rng.random() * total
        cumulative = 0.0
        selected = supported[-1]
        for profile_id, weight in zip(supported, weighted):
            cumulative += weight if sum(weighted) else 1.0
            if threshold <= cumulative:
                selected = profile_id
                break
    return PROFILES[str(selected)], probability_map(supported, selected=str(selected))


def resolve_source_temperature(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    profile: ThermometerProfile,
) -> Tuple[int, int, Dict[str, float]]:
    """Resolve a source temperature and its converted target answer."""

    support = tuple(int(value) for value in profile.source_support)
    target_support = tuple(
        convert_temperature(str(profile.source_unit), str(profile.target_unit), int(value))
        for value in support
    )
    explicit_source_raw = params.get("source_temperature", params.get("source_value"))
    explicit_target_raw = params.get("target_answer")
    if explicit_source_raw is not None:
        source = int(explicit_source_raw)
        if source not in support:
            raise ValueError(f"unsupported source_temperature for {profile.profile_id}: {source}")
        target = convert_temperature(str(profile.source_unit), str(profile.target_unit), source)
        return source, target, probability_map([str(value) for value in target_support], selected=str(target))
    if explicit_target_raw is not None:
        target = int(explicit_target_raw)
        source = source_temperature_from_target(str(profile.source_unit), str(profile.target_unit), target)
        if source not in support:
            raise ValueError(f"unsupported target_answer for {profile.profile_id}: {target}")
        return source, target, probability_map([str(value) for value in target_support], selected=str(target))
    balanced = bool(
        params.get(
            "balanced_target_answer_sampling",
            group_default(generation_defaults, "balanced_target_answer_sampling", True),
        )
    )
    if balanced:
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.target_answer.{profile.profile_id}")
        source = int(rng.choice(support))
    else:
        source = int(spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.source_temperature.{profile.profile_id}").choice(support))
    target = convert_temperature(str(profile.source_unit), str(profile.target_unit), source)
    return int(source), int(target), probability_map([str(value) for value in target_support])


def make_thermometer_scenario(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    source_unit: str,
    target_unit: str,
) -> ThermometerScenario:
    """Build a thermometer conversion scenario for semantic units."""

    profile, profile_probabilities = resolve_profile(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        source_unit=str(source_unit),
        target_unit=str(target_unit),
    )
    source_temperature, target_temperature, target_probabilities = resolve_source_temperature(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        profile=profile,
    )
    return ThermometerScenario(
        profile=profile,
        source_temperature=int(source_temperature),
        target_temperature=int(target_temperature),
        scale_profile_probabilities=dict(profile_probabilities),
        target_answer_probabilities=dict(target_probabilities),
    )


__all__ = [
    "make_thermometer_scenario",
    "probability_map",
    "profile_ids_for_units",
    "resolve_profile",
    "resolve_source_temperature",
]
