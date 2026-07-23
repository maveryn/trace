"""Sampling primitives for free-body-force diagrams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.physics.shared.style import SUPPORTED_PHYSICS_COLOR_NAMES
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.variant_sampling import (
    apply_balanced_variant_sampling,
    resolve_variant,
)

from .formulas import sum_force_vectors, vector_direction
from .state import (
    CARDINAL_VECTORS,
    DIRECTION_NAMES,
    DIRECTION_VECTORS,
    OPTION_LETTERS,
    SCENE_NAMESPACE,
    SUPPORTED_SCENE_VARIANTS,
    ForceScenario,
    ForceSpec,
    SamplingAxes,
)


def resolve_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    supported: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve one non-query sampling axis from params and scene defaults."""

    selected, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), namespace),
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=supported,
        explicit_key=explicit_key,
        weights_key=weights_key,
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=supported,
        balance_flag_key=balance_flag_key,
        explicit_key=explicit_key,
        weights_key=weights_key,
        sampling_namespace=namespace,
    )
    return str(selected), {str(key): float(value) for key, value in probabilities.items()}


def resolve_sampling_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> SamplingAxes:
    """Resolve scene/style/answer axes for one force diagram."""

    scene_variant, scene_probs = resolve_axis(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        supported=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace=f"{namespace}.scene_variant",
    )
    net_direction, net_probs = resolve_axis(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        supported=DIRECTION_NAMES,
        explicit_key="net_force_direction",
        weights_key="net_force_direction_weights",
        balance_flag_key="balanced_net_force_direction_sampling",
        namespace=f"{namespace}.net_force_direction",
    )
    correct_letter, letter_probs = resolve_axis(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        supported=OPTION_LETTERS,
        explicit_key="correct_option_letter",
        weights_key="correct_option_letter_weights",
        balance_flag_key="balanced_correct_option_letter_sampling",
        namespace=f"{namespace}.correct_option_letter",
    )
    accent_color, accent_probs = resolve_axis(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        supported=SUPPORTED_PHYSICS_COLOR_NAMES,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        balance_flag_key="balanced_accent_color_name_sampling",
        namespace=f"{namespace}.accent_color_name",
    )
    return SamplingAxes(
        scene_variant=str(scene_variant),
        net_force_direction=str(net_direction),
        correct_option_letter=str(correct_letter),
        accent_color_name=str(accent_color),
        scene_variant_probabilities=dict(scene_probs),
        net_force_direction_probabilities=dict(net_probs),
        correct_option_letter_probabilities=dict(letter_probs),
        accent_color_name_probabilities=dict(accent_probs),
    )


def option_directions(
    *,
    instance_seed: int,
    net_force_direction: str,
    correct_option_letter: str,
    namespace: str = SCENE_NAMESPACE,
) -> dict[str, str]:
    """Assign the correct and distractor directions to visible option letters."""

    remaining_directions = [
        direction for direction in DIRECTION_NAMES if str(direction) != str(net_force_direction)
    ]
    rng = spawn_rng(int(instance_seed), f"{namespace}.option_directions")
    rng.shuffle(remaining_directions)
    mapping: dict[str, str] = {}
    cursor = 0
    for letter in OPTION_LETTERS:
        if str(letter) == str(correct_option_letter):
            mapping[str(letter)] = str(net_force_direction)
        else:
            mapping[str(letter)] = str(remaining_directions[cursor])
            cursor += 1
    return dict(mapping)


def make_force_specs(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    net_force_direction: str,
    namespace: str = SCENE_NAMESPACE,
) -> tuple[ForceSpec, ...]:
    """Construct visible cardinal force arrows whose sum has the requested direction."""

    explicit = params.get("force_specs")
    if explicit is not None:
        if isinstance(explicit, (str, bytes)) or not isinstance(explicit, Sequence):
            raise ValueError("force_specs must be a sequence of mappings")
        specs: list[ForceSpec] = []
        for index, raw_spec in enumerate(explicit):
            if not isinstance(raw_spec, Mapping):
                raise ValueError("each force spec must be a mapping")
            direction = str(raw_spec["direction"])
            magnitude = int(raw_spec["magnitude_n"])
            if direction not in CARDINAL_VECTORS:
                raise ValueError("force_specs support cardinal directions only")
            if magnitude <= 0:
                raise ValueError("force magnitudes must be positive")
            unit = CARDINAL_VECTORS[direction]
            specs.append(
                ForceSpec(
                    force_id=f"F{index + 1}",
                    direction=str(direction),
                    magnitude_n=int(magnitude),
                    vector=(
                        int(unit[0]) * int(magnitude),
                        int(unit[1]) * int(magnitude),
                    ),
                )
            )
        resultant = sum_force_vectors(tuple(spec.vector for spec in specs))
        if vector_direction(resultant) != str(net_force_direction):
            raise ValueError("explicit force_specs do not match net_force_direction")
        return tuple(specs)

    rng = spawn_rng(int(instance_seed), f"{namespace}.force_specs")
    dx_sign, dy_sign = DIRECTION_VECTORS[str(net_force_direction)]
    residual_options = tuple(
        int(value) for value in group_default(generation_defaults, "resultant_component_support", (2, 3, 4, 5, 6))
    )
    base_options = tuple(
        int(value) for value in group_default(generation_defaults, "base_force_support", (2, 3, 4, 5, 6, 7, 8))
    )
    pair_options = tuple(
        int(value) for value in group_default(generation_defaults, "canceling_pair_support", (1, 2, 3, 4))
    )
    residual = int(rng.choice(residual_options))
    if int(dx_sign) == 0:
        east = west = int(rng.choice(base_options))
    elif int(dx_sign) > 0:
        west = int(rng.choice(base_options))
        east = int(west + residual)
    else:
        east = int(rng.choice(base_options))
        west = int(east + residual)
    if int(dy_sign) == 0:
        north = south = int(rng.choice(base_options))
    elif int(dy_sign) > 0:
        south = int(rng.choice(base_options))
        north = int(south + residual)
    else:
        north = int(rng.choice(base_options))
        south = int(north + residual)

    magnitudes = {
        "east": int(east),
        "north": int(north),
        "west": int(west),
        "south": int(south),
    }
    specs = [
        ForceSpec(
            force_id=f"F{index + 1}",
            direction=str(direction),
            magnitude_n=int(magnitudes[str(direction)]),
            vector=(
                int(CARDINAL_VECTORS[str(direction)][0]) * int(magnitudes[str(direction)]),
                int(CARDINAL_VECTORS[str(direction)][1]) * int(magnitudes[str(direction)]),
            ),
        )
        for index, direction in enumerate(("east", "north", "west", "south"))
    ]
    if bool(params.get("include_extra_canceling_pair", rng.random() < 0.65)):
        pair_direction = str(rng.choice(("east_west", "north_south")))
        magnitude = int(rng.choice(pair_options))
        for direction in (("east", "west") if pair_direction == "east_west" else ("north", "south")):
            specs.append(
                ForceSpec(
                    force_id=f"F{len(specs) + 1}",
                    direction=str(direction),
                    magnitude_n=int(magnitude),
                    vector=(
                        int(CARDINAL_VECTORS[str(direction)][0]) * int(magnitude),
                        int(CARDINAL_VECTORS[str(direction)][1]) * int(magnitude),
                    ),
                )
            )
    rng.shuffle(specs)
    relabeled = tuple(
        ForceSpec(
            force_id=f"F{index + 1}",
            direction=str(spec.direction),
            magnitude_n=int(spec.magnitude_n),
            vector=tuple(spec.vector),
        )
        for index, spec in enumerate(specs)
    )
    resultant = sum_force_vectors(tuple(spec.vector for spec in relabeled))
    if vector_direction(resultant) != str(net_force_direction):
        raise RuntimeError("constructed force specs do not match requested net direction")
    return tuple(relabeled)


def make_force_scenario(
    *,
    instance_seed: int,
    axes: SamplingAxes,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> ForceScenario:
    """Resolve one force scenario from axes and generated/applied force vectors."""

    forces = make_force_specs(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        net_force_direction=str(axes.net_force_direction),
        namespace=str(namespace),
    )
    resultant = sum_force_vectors(tuple(spec.vector for spec in forces))
    resolved_direction = vector_direction(resultant)
    return ForceScenario(
        scene_variant=str(axes.scene_variant),
        net_force_direction=str(resolved_direction),
        correct_option_letter=str(axes.correct_option_letter),
        option_directions=option_directions(
            instance_seed=int(instance_seed),
            net_force_direction=str(resolved_direction),
            correct_option_letter=str(axes.correct_option_letter),
            namespace=str(namespace),
        ),
        force_specs=tuple(forces),
        resultant_vector=tuple(resultant),
    )
