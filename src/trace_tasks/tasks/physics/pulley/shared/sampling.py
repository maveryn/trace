"""Sampling and pulley-force mechanics for pulley diagrams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.physics.shared.style import SUPPORTED_PHYSICS_COLOR_NAMES
from trace_tasks.tasks.physics.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .state import (
    CutSegmentSpec,
    PulleyResolvedAxes,
    PulleySceneSpec,
    PulleyTaskDefaults,
    SCENE_NAMESPACE,
    SUPPORTED_SCENE_VARIANTS,
    SUPPORTED_SOLVE_FOR_TARGETS,
)


DEFAULTS = PulleyTaskDefaults()


def normalize_solve_for(value: Any) -> str:
    """Normalize the hidden force slot requested by this scene."""

    normalized = str(value).strip().lower()
    if normalized in {"effort", "effort_force"}:
        return "effort_force"
    if normalized in {"load", "load_force"}:
        return "load_force"
    raise ValueError(f"unsupported solve_for: {value}")


def answer_support_key(solve_for: str) -> str:
    """Return the configured answer-support key for one force target."""

    if str(solve_for) == "effort_force":
        return "effort_force_support"
    return "load_force_support"


def fallback_support(solve_for: str) -> tuple[int, ...]:
    """Return fallback answer support for one force target."""

    if str(solve_for) == "effort_force":
        return DEFAULTS.effort_force_support
    return DEFAULTS.load_force_support


def probability_map(values: Sequence[str], selected: str | None = None) -> dict[str, float]:
    """Return a probability map over a finite string support."""

    if selected is not None:
        return {str(value): (1.0 if str(value) == str(selected) else 0.0) for value in values}
    probability = 1.0 / float(len(values)) if values else 0.0
    return {str(value): float(probability) for value in values}


def resolve_solve_for(
    rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> tuple[str, dict[str, float]]:
    """Resolve whether the unknown pulley force is effort or load."""

    explicit_solve_for = params.get("solve_for")
    if explicit_solve_for is not None:
        selected = normalize_solve_for(explicit_solve_for)
        return selected, probability_map(SUPPORTED_SOLVE_FOR_TARGETS, selected=selected)

    selected = str(rng.choice(SUPPORTED_SOLVE_FOR_TARGETS))
    return str(selected), probability_map(SUPPORTED_SOLVE_FOR_TARGETS)


def resolve_target_answer(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    solve_for: str,
    generation_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> tuple[int, dict[str, float]]:
    """Resolve one balanced answer target for the active force target."""

    return resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=dict(params),
        gen_defaults=generation_defaults,
        support_key=answer_support_key(str(solve_for)),
        explicit_key="target_answer",
        fallback_support=fallback_support(str(solve_for)),
        namespace=f"{namespace}.target_answer.{str(solve_for)}",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )


def resolve_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> PulleyResolvedAxes:
    """Resolve one scene, force target, color, and answer target."""

    axis_rng = spawn_rng(int(instance_seed), f"{namespace}.axes")
    solve_for, solve_for_probabilities = resolve_solve_for(
        axis_rng,
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        namespace=str(namespace),
    )
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
    target_answer, target_answer_probabilities = resolve_target_answer(
        instance_seed=int(instance_seed),
        params=params,
        solve_for=str(solve_for),
        generation_defaults=generation_defaults,
        namespace=str(namespace),
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
    return PulleyResolvedAxes(
        scene_variant=str(scene_variant),
        solve_for=str(solve_for),
        accent_color_name=str(accent_color_name),
        target_answer=int(target_answer),
        scene_variant_probabilities=dict(scene_probs),
        solve_for_probabilities=dict(solve_for_probabilities),
        accent_color_name_probabilities=dict(accent_probs),
        target_answer_probabilities=dict(target_answer_probabilities),
    )


def connected_support_count_support(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> tuple[int, ...]:
    """Return the active support for connected full-length strands."""

    return resolve_integer_support(
        params,
        gen_defaults=generation_defaults,
        key="connected_support_count_support",
        fallback=DEFAULTS.connected_support_count_support,
    )


def disconnected_segment_count_support(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> tuple[int, ...]:
    """Return the active support for cut non-supporting strands."""

    return resolve_integer_support(
        params,
        gen_defaults=generation_defaults,
        key="disconnected_segment_count_support",
        fallback=DEFAULTS.disconnected_segment_count_support,
    )


def effort_force_bounds(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> tuple[int, int]:
    """Return inclusive effort-force bounds used for feasible pulley scenes."""

    effort_min = int(
        params.get(
            "effort_force_min",
            group_default(generation_defaults, "effort_force_min", DEFAULTS.effort_force_min),
        )
    )
    effort_max = int(
        params.get(
            "effort_force_max",
            group_default(generation_defaults, "effort_force_max", DEFAULTS.effort_force_max),
        )
    )
    if int(effort_min) > int(effort_max):
        raise ValueError("effort_force_min must be <= effort_force_max")
    return int(effort_min), int(effort_max)


def choose_count_from_support(
    rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    support: Sequence[int],
    explicit_keys: Sequence[str],
    namespace: str,
    balanced_flag_key: str,
    feasible_values: Sequence[int] | None = None,
) -> int:
    """Choose one integer count from support while respecting explicit overrides."""

    support_set = {int(value) for value in support}
    feasible_set = set(support_set if feasible_values is None else {int(value) for value in feasible_values})
    feasible = tuple(sorted(int(value) for value in support_set.intersection(feasible_set)))
    if not feasible:
        raise ValueError(f"no feasible values for {namespace}")
    for explicit_key in explicit_keys:
        raw_value = params.get(str(explicit_key))
        if raw_value is None:
            continue
        selected = int(raw_value)
        if int(selected) not in support_set:
            raise ValueError(f"unsupported {explicit_key}: {selected}")
        if int(selected) not in feasible:
            raise ValueError(f"infeasible {explicit_key}: {selected}")
        return int(selected)

    balanced_enabled = bool(
        params.get(
            str(balanced_flag_key),
            group_default(generation_defaults, str(balanced_flag_key), True),
        )
    )
    if bool(balanced_enabled):
        return int(rng.choice(feasible))
    return int(feasible[int(rng.randrange(len(feasible)))])


def resolve_disconnected_segment_count(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> int:
    """Resolve how many cut non-supporting strands to draw."""

    support = disconnected_segment_count_support(params, generation_defaults)
    selected, _ = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        support_key="disconnected_segment_count_support",
        explicit_key="disconnected_segment_count",
        fallback_support=support,
        namespace=f"{namespace}.disconnected_segment_count",
        balanced_flag_key="balanced_disconnected_segment_count_sampling",
        namespace_support_permutation=True,
    )
    return int(selected)


def resolve_support_count_for_solve_for(
    rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    solve_for: str,
    target_answer: int,
    effort_min: int,
    effort_max: int,
    namespace: str = SCENE_NAMESPACE,
) -> int:
    """Resolve a connected strand count that can realize the queried force."""

    support = connected_support_count_support(params, generation_defaults)
    if str(solve_for) == "load_force":
        feasible = tuple(
            int(count)
            for count in support
            if int(count) > 0
            and int(target_answer) % int(count) == 0
            and int(effort_min) <= int(target_answer) // int(count) <= int(effort_max)
        )
    else:
        feasible = tuple(int(value) for value in support)
    return choose_count_from_support(
        rng,
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        support=support,
        explicit_keys=("connected_support_count", "support_segment_count"),
        namespace=f"{namespace}.connected_support_count.{str(solve_for)}",
        balanced_flag_key="balanced_connected_support_count_sampling",
        feasible_values=feasible,
    )


def sample_cut_segments(
    rng,
    *,
    disconnected_segment_count: int,
    cut_slot_indices: Sequence[int],
) -> tuple[CutSegmentSpec, ...]:
    """Sample cut-strand attachment sides and cut fractions."""

    side_cycle = ["top" if index % 2 == 0 else "bottom" for index in range(int(disconnected_segment_count))]
    rng.shuffle(side_cycle)
    segments: list[CutSegmentSpec] = []
    for segment_index, (slot_index, attach_side) in enumerate(zip(cut_slot_indices, side_cycle), start=1):
        segments.append(
            CutSegmentSpec(
                segment_id=f"cut_segment_{int(segment_index)}",
                attach_side=str(attach_side),
                x_order=int(slot_index),
                cut_fraction=round(float(rng.uniform(0.25, 0.75)), 3),
            )
        )
    return tuple(segments)


def sample_scene_spec(
    rng,
    *,
    instance_seed: int,
    scene_variant: str,
    solve_for: str,
    target_answer: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> PulleySceneSpec:
    """Sample one symbolic pulley system that realizes the target answer."""

    solve_for = normalize_solve_for(solve_for)
    target_answer = int(target_answer)
    effort_min, effort_max = effort_force_bounds(params, generation_defaults)
    support_count = resolve_support_count_for_solve_for(
        rng,
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        solve_for=str(solve_for),
        target_answer=int(target_answer),
        effort_min=int(effort_min),
        effort_max=int(effort_max),
        namespace=str(namespace),
    )
    disconnected_count = resolve_disconnected_segment_count(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        namespace=str(namespace),
    )

    if solve_for == "effort_force":
        effort_value = int(target_answer)
        load_value = int(effort_value) * int(support_count)
        shown_effort = None
        shown_load = int(load_value)
    else:
        if int(target_answer) % int(support_count) != 0:
            raise ValueError("load-force target is not divisible by connected support count")
        effort_value = int(target_answer) // int(support_count)
        if int(effort_value) < int(effort_min) or int(effort_value) > int(effort_max):
            raise ValueError("load-force target implies effort outside configured bounds")
        load_value = int(target_answer)
        shown_effort = int(effort_value)
        shown_load = None

    total_slots = int(support_count) + int(disconnected_count)
    slot_indices = list(range(int(total_slots)))
    rng.shuffle(slot_indices)
    connected_slot_indices = tuple(sorted(int(slot_index) for slot_index in slot_indices[: int(support_count)]))
    cut_slot_indices = tuple(sorted(int(slot_index) for slot_index in slot_indices[int(support_count) :]))
    cut_segments = sample_cut_segments(
        rng,
        disconnected_segment_count=int(disconnected_count),
        cut_slot_indices=cut_slot_indices,
    )
    annotation_entity_ids: list[str] = [
        f"support_segment_{index}" for index in range(1, int(support_count) + 1)
    ]
    if solve_for == "effort_force":
        annotation_entity_ids.extend(["load_force_label", "effort_force_label"])
    else:
        annotation_entity_ids.extend(["effort_force_label", "load_force_label"])

    return PulleySceneSpec(
        scene_variant=str(scene_variant),
        solve_for=str(solve_for),
        support_segment_count=int(support_count),
        disconnected_segment_count=int(disconnected_count),
        connected_slot_indices=tuple(int(slot_index) for slot_index in connected_slot_indices),
        cut_segments=tuple(cut_segments),
        effort_force_value=int(effort_value),
        load_force_value=int(load_value),
        shown_effort_force_value=None if shown_effort is None else int(shown_effort),
        shown_load_force_value=None if shown_load is None else int(shown_load),
        target_answer=int(target_answer),
        annotation_entity_ids=tuple(str(entity_id) for entity_id in annotation_entity_ids),
    )


__all__ = [
    "answer_support_key",
    "connected_support_count_support",
    "disconnected_segment_count_support",
    "effort_force_bounds",
    "fallback_support",
    "normalize_solve_for",
    "resolve_axes",
    "sample_scene_spec",
]
