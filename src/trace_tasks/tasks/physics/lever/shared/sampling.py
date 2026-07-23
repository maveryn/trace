"""Deterministic symbolic sampling helpers for lever-balance scenes."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.physics.shared.style import SUPPORTED_PHYSICS_COLOR_NAMES
from trace_tasks.tasks.physics.shared.support_sampling import resolve_integer_choice
from trace_tasks.tasks.physics.shared.support_sampling import resolve_integer_support
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .state import (
    MISSING_WEIGHT_SCENE_VARIANTS,
    SUPPORTED_SCENE_VARIANTS,
    TORQUE_SIDES,
    LeverMissingWeightAxes,
    LeverSideTorqueAxes,
    LeverTaskDefaults,
    LeverWeightSlot,
)


def distance_support(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    defaults: LeverTaskDefaults,
) -> Tuple[int, ...]:
    """Return the active integer distance slots available on each side."""

    return resolve_integer_support(
        params,
        gen_defaults=generation_defaults,
        key="distance_support",
        fallback=defaults.distance_support,
    )


def generation_int(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    key: str,
    fallback: int,
) -> int:
    """Resolve one integer generation parameter from params or scene defaults."""

    return int(params.get(str(key), group_default(generation_defaults, str(key), int(fallback))))


def resolve_lever_axis_choice(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    supported: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> tuple[str, Dict[str, float]]:
    """Resolve one visual/semantic axis with optional balanced sampling."""

    selected, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), str(namespace)),
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=tuple(str(item) for item in supported),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=tuple(str(item) for item in supported),
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=str(namespace),
    )
    return str(selected), {str(key): float(value) for key, value in probabilities.items()}


def select_lever_public_branch(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    public_name: str,
    namespace: str,
) -> tuple[str, Mapping[str, float], Dict[str, Any]]:
    """Resolve the single public replay branch shared by lever objectives."""

    selected, probabilities, resolved_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=(SINGLE_QUERY_ID,),
        default_query_id=SINGLE_QUERY_ID,
        task_id=str(public_name),
        namespace=f"{namespace}.public_branch",
    )
    return str(selected), {str(key): float(value) for key, value in dict(probabilities).items()}, dict(resolved_params)


def resolve_side_torque_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    public_branch: str,
    public_branch_probabilities: Mapping[str, float],
    generation_defaults: Mapping[str, Any],
    defaults: LeverTaskDefaults,
    namespace: str,
) -> LeverSideTorqueAxes:
    """Resolve scene, side, answer, and style axes for side-torque generation."""

    scene_variant, scene_probs = resolve_lever_axis_choice(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        supported=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace=f"{namespace}.scene_variant",
    )
    torque_side, torque_side_probs = resolve_lever_axis_choice(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        supported=TORQUE_SIDES,
        explicit_key="torque_side",
        weights_key="torque_side_weights",
        balance_flag_key="balanced_torque_side_sampling",
        namespace=f"{namespace}.torque_side",
    )
    accent_color_name, accent_probs = resolve_lever_axis_choice(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        supported=SUPPORTED_PHYSICS_COLOR_NAMES,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        balance_flag_key="balanced_accent_color_name_sampling",
        namespace=f"{namespace}.accent_color_name",
    )
    target_answer, target_probs = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        support_key="torque_answer_support",
        explicit_key="target_answer",
        fallback_support=defaults.torque_answer_support,
        namespace=f"{namespace}.target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    return LeverSideTorqueAxes(
        public_branch=str(public_branch),
        scene_variant=str(scene_variant),
        torque_side=str(torque_side),
        accent_color_name=str(accent_color_name),
        target_answer=int(target_answer),
        public_branch_probabilities={str(key): float(value) for key, value in dict(public_branch_probabilities).items()},
        scene_variant_probabilities=dict(scene_probs),
        torque_side_probabilities=dict(torque_side_probs),
        accent_color_name_probabilities=dict(accent_probs),
        target_answer_probabilities=dict(target_probs),
    )


def resolve_missing_weight_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    public_branch: str,
    public_branch_probabilities: Mapping[str, float],
    generation_defaults: Mapping[str, Any],
    defaults: LeverTaskDefaults,
    namespace: str,
) -> LeverMissingWeightAxes:
    """Resolve scene, answer, and style axes for missing-weight generation."""

    scene_weights_key = "scene_variant_weights" if params.get("scene_variant") is not None else "missing_weight_scene_variant_weights"
    scene_variant, scene_probs = resolve_lever_axis_choice(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        supported=MISSING_WEIGHT_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key=scene_weights_key,
        balance_flag_key="balanced_scene_variant_sampling",
        namespace=f"{namespace}.scene_variant",
    )
    accent_color_name, accent_probs = resolve_lever_axis_choice(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        supported=SUPPORTED_PHYSICS_COLOR_NAMES,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        balance_flag_key="balanced_accent_color_name_sampling",
        namespace=f"{namespace}.accent_color_name",
    )
    target_answer, target_probs = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        support_key="missing_weight_support",
        explicit_key="target_answer",
        fallback_support=defaults.missing_weight_support,
        namespace=f"{namespace}.target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    return LeverMissingWeightAxes(
        public_branch=str(public_branch),
        scene_variant=str(scene_variant),
        accent_color_name=str(accent_color_name),
        target_answer=int(target_answer),
        public_branch_probabilities={str(key): float(value) for key, value in dict(public_branch_probabilities).items()},
        scene_variant_probabilities=dict(scene_probs),
        accent_color_name_probabilities=dict(accent_probs),
        target_answer_probabilities=dict(target_probs),
    )


def candidate_side_configs(
    *,
    target_torque: int,
    distances: Sequence[int],
    weight_min: int,
    weight_max: int,
    max_weights: int,
) -> List[List[Tuple[int, int]]]:
    """Enumerate feasible ``(distance, weight)`` sets matching one torque."""

    resolved_distances = [int(value) for value in distances]
    candidates: List[List[Tuple[int, int]]] = []
    max_count = min(int(max_weights), len(resolved_distances))

    def _search(
        *,
        distance_combo: Tuple[int, ...],
        combo_index: int,
        remaining_torque: int,
        prefix: List[Tuple[int, int]],
    ) -> None:
        distance = int(distance_combo[int(combo_index)])
        remaining_slots = int(len(distance_combo) - combo_index - 1)
        if remaining_slots == 0:
            if int(remaining_torque) % int(distance) != 0:
                return
            weight = int(remaining_torque) // int(distance)
            if int(weight_min) <= int(weight) <= int(weight_max):
                candidates.append(list(prefix) + [(int(distance), int(weight))])
            return

        tail_distances = [int(value) for value in distance_combo[int(combo_index) + 1 :]]
        min_tail_torque = sum(int(value) * int(weight_min) for value in tail_distances)
        max_tail_torque = sum(int(value) * int(weight_max) for value in tail_distances)
        for weight in range(int(weight_min), int(weight_max) + 1):
            next_remaining = int(remaining_torque) - (int(distance) * int(weight))
            if next_remaining < int(min_tail_torque) or next_remaining > int(max_tail_torque):
                continue
            _search(
                distance_combo=distance_combo,
                combo_index=int(combo_index) + 1,
                remaining_torque=int(next_remaining),
                prefix=list(prefix) + [(int(distance), int(weight))],
            )

    for count in range(1, int(max_count) + 1):
        for distance_combo in combinations(resolved_distances, int(count)):
            min_possible = sum(int(distance) * int(weight_min) for distance in distance_combo)
            max_possible = sum(int(distance) * int(weight_max) for distance in distance_combo)
            if int(target_torque) < int(min_possible) or int(target_torque) > int(max_possible):
                continue
            _search(
                distance_combo=tuple(int(value) for value in distance_combo),
                combo_index=0,
                remaining_torque=int(target_torque),
                prefix=[],
            )

    deduped: List[List[Tuple[int, int]]] = []
    seen: set[Tuple[Tuple[int, int], ...]] = set()
    for candidate in candidates:
        canonical = tuple(sorted((int(distance), int(weight)) for distance, weight in candidate))
        if canonical in seen:
            continue
        seen.add(canonical)
        deduped.append([(int(distance), int(weight)) for distance, weight in canonical])
    return deduped


def sample_random_side_config(
    rng,
    *,
    distances: Sequence[int],
    weight_min: int,
    weight_max: int,
    max_weights: int,
) -> List[Tuple[int, int]]:
    """Sample one random visible weight set for a non-queried side."""

    resolved_distances = [int(value) for value in distances]
    count = int(rng.randint(1, min(int(max_weights), len(resolved_distances))))
    chosen_distances = sorted(rng.sample(resolved_distances, count))
    return [
        (int(distance), int(rng.randint(int(weight_min), int(weight_max))))
        for distance in chosen_distances
    ]


def sample_side_torque_layout(
    rng,
    *,
    target_torque: int,
    torque_side: str,
    distances: Sequence[int],
    weight_min: int,
    weight_max: int,
    max_weights: int,
) -> Tuple[List[LeverWeightSlot], Dict[str, Any]]:
    """Sample a lever layout whose queried side has the requested torque."""

    relevant_side = str(torque_side)
    distractor_side = "right" if relevant_side == "left" else "left"
    candidates = candidate_side_configs(
        target_torque=int(target_torque),
        distances=distances,
        weight_min=int(weight_min),
        weight_max=int(weight_max),
        max_weights=int(max_weights),
    )
    if not candidates:
        raise ValueError(f"unable to build side config for torque target {target_torque}")
    relevant_config = candidates[int(rng.randrange(len(candidates)))]
    distractor_config = sample_random_side_config(
        rng,
        distances=distances,
        weight_min=int(weight_min),
        weight_max=int(weight_max),
        max_weights=int(max_weights),
    )
    placements = [
        LeverWeightSlot(str(relevant_side), int(distance), int(weight), False, True)
        for distance, weight in relevant_config
    ] + [
        LeverWeightSlot(str(distractor_side), int(distance), int(weight), False, False)
        for distance, weight in distractor_config
    ]
    metadata = {
        "query_side": str(relevant_side),
        "known_torque_left": sum(
            int(slot.distance_units) * int(slot.value or 0)
            for slot in placements
            if slot.side == "left" and slot.value is not None
        ),
        "known_torque_right": sum(
            int(slot.distance_units) * int(slot.value or 0)
            for slot in placements
            if slot.side == "right" and slot.value is not None
        ),
        "placeholder_side": None,
    }
    return placements, metadata


def sample_same_side_known_weights(
    rng,
    *,
    placeholder_distance: int,
    distances: Sequence[int],
    weight_min: int,
    weight_max: int,
    max_side_weights: int,
) -> List[Tuple[int, int]]:
    """Sample extra known weights on the placeholder side."""

    remaining_distances = [int(value) for value in distances if int(value) != int(placeholder_distance)]
    max_extra_weights = min(max(0, int(max_side_weights) - 1), len(remaining_distances))
    if not remaining_distances or int(max_extra_weights) <= 0:
        return []
    count = int(rng.randint(0, int(max_extra_weights)))
    if count <= 0:
        return []
    chosen_distances = sorted(rng.sample(remaining_distances, int(count)))
    return [
        (int(distance), int(rng.randint(int(weight_min), int(weight_max))))
        for distance in chosen_distances
    ]


def sample_missing_weight_layout(
    rng,
    *,
    target_weight: int,
    distances: Sequence[int],
    weight_min: int,
    weight_max: int,
    max_side_weights: int,
) -> Tuple[List[LeverWeightSlot], Dict[str, Any]]:
    """Sample a layout where one marked placeholder weight balances the lever."""

    placeholder_side = "left" if rng.random() < 0.5 else "right"
    opposite_side = "right" if placeholder_side == "left" else "left"
    resolved_distances = [int(value) for value in distances]
    for _ in range(120):
        placeholder_distance = int(resolved_distances[int(rng.randrange(len(resolved_distances)))])
        same_side_known = sample_same_side_known_weights(
            rng,
            placeholder_distance=int(placeholder_distance),
            distances=resolved_distances,
            weight_min=int(weight_min),
            weight_max=int(weight_max),
            max_side_weights=int(max_side_weights),
        )
        same_side_torque = sum(int(distance) * int(weight) for distance, weight in same_side_known)
        opposite_target_torque = int(same_side_torque + (int(target_weight) * int(placeholder_distance)))
        opposite_candidates = candidate_side_configs(
            target_torque=int(opposite_target_torque),
            distances=resolved_distances,
            weight_min=int(weight_min),
            weight_max=int(weight_max),
            max_weights=int(max_side_weights),
        )
        if not opposite_candidates:
            continue
        opposite_config = opposite_candidates[int(rng.randrange(len(opposite_candidates)))]
        placements = [
            LeverWeightSlot(str(placeholder_side), int(placeholder_distance), None, True, True),
        ] + [
            LeverWeightSlot(str(placeholder_side), int(distance), int(weight), False, True)
            for distance, weight in same_side_known
        ] + [
            LeverWeightSlot(str(opposite_side), int(distance), int(weight), False, True)
            for distance, weight in opposite_config
        ]
        metadata = {
            "query_side": None,
            "known_torque_left": sum(
                int(slot.distance_units) * int(slot.value or 0)
                for slot in placements
                if slot.side == "left" and slot.value is not None and not slot.missing
            ),
            "known_torque_right": sum(
                int(slot.distance_units) * int(slot.value or 0)
                for slot in placements
                if slot.side == "right" and slot.value is not None and not slot.missing
            ),
            "placeholder_side": str(placeholder_side),
            "placeholder_distance_units": int(placeholder_distance),
        }
        return placements, metadata
    raise ValueError("unable to sample missing-weight lever layout")
