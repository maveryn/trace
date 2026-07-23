"""Sampling helpers for gear-train diagrams."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_support
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .formulas import normalize_direction, opposite_direction, propagate_output_direction, radius_from_teeth, speed_relation
from .state import (
    DIRECTION_OPTION_LETTERS,
    GearDirectionChoiceScenario,
    GearDirectionScenario,
    GearSpeedScenario,
    GearTrainDefaults,
    SUPPORTED_DIRECTIONS,
    SUPPORTED_SCENE_VARIANTS,
    SUPPORTED_SPEED_RELATIONS,
)


DEFAULTS = GearTrainDefaults()


def probability_map(values: Sequence[str], selected: str | None = None) -> Dict[str, float]:
    """Return a string-keyed probability map over a finite support."""

    if selected is not None:
        return {str(value): (1.0 if str(value) == str(selected) else 0.0) for value in values}
    probability = 1.0 / float(len(values)) if values else 0.0
    return {str(value): float(probability) for value in values}


def integer_probability_map(values: Sequence[int], selected: int | None = None) -> Dict[str, float]:
    """Return an integer-support probability map with string keys."""

    if selected is not None:
        return {str(int(value)): (1.0 if int(value) == int(selected) else 0.0) for value in values}
    probability = 1.0 / float(len(values)) if values else 0.0
    return {str(int(value)): float(probability) for value in values}


def integer_support(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
) -> Tuple[int, ...]:
    """Resolve one positive integer support from params/defaults."""

    support = resolve_integer_support(
        params,
        gen_defaults=defaults,
        key=str(key),
        fallback=fallback,
    )
    if not support:
        raise ValueError(f"{key} must not be empty for gear-train sampling")
    return tuple(int(value) for value in support)


def balanced_axis_choice(values: Sequence[Any], *, params: Mapping[str, Any], instance_seed: int, namespace: str) -> Any:
    """Return one balanced finite-support choice with cursor support."""

    support = tuple(values)
    if not support:
        raise ValueError(f"cannot sample empty support for {namespace}")
    sample_cursor = params.get("_sample_cursor")
    if sample_cursor is not None:
        return support[abs(int(sample_cursor)) % len(support)]
    choice_namespace = str(namespace)
    rng = spawn_rng(int(instance_seed), choice_namespace)
    return rng.choice(support)


def resolve_scene_variant(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the gear layout variant."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.scene_variant")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=defaults,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        balance_flag_key="balanced_scene_variant_sampling",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        sampling_namespace=f"{namespace}.scene_variant",
    )
    return str(selected), {str(key): float(value) for key, value in sorted(probabilities.items())}


def resolve_direction_scenario(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> GearDirectionScenario:
    """Resolve one gear-direction scenario without public branch routing."""

    scene_variant, scene_variant_probabilities = resolve_scene_variant(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    explicit_input = normalize_direction(params.get("input_direction"))
    explicit_target = normalize_direction(params.get("target_answer", params.get("output_direction")))
    gear_count, gear_count_probabilities = resolve_direction_gear_count(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
        input_direction=explicit_input,
        target_answer=explicit_target,
    )

    if explicit_input is not None:
        input_direction = str(explicit_input)
    elif explicit_target is not None:
        input_direction = str(explicit_target) if (int(gear_count) - 1) % 2 == 0 else opposite_direction(str(explicit_target))
    elif bool(params.get("balanced_target_answer_sampling", group_default(defaults, "balanced_target_answer_sampling", True))):
        selected_target = str(balanced_axis_choice(
            SUPPORTED_DIRECTIONS,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.target_answer",
        ))
        input_direction = str(selected_target) if (int(gear_count) - 1) % 2 == 0 else opposite_direction(str(selected_target))
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.input_direction")
        input_direction = str(rng.choice(SUPPORTED_DIRECTIONS))

    output_direction = propagate_output_direction(str(input_direction), int(gear_count))
    if explicit_target is not None and str(output_direction) != str(explicit_target):
        raise ValueError("explicit gear train parameters do not produce the requested output direction")

    radius_support = integer_support(params, defaults, "gear_radius_px_support", DEFAULTS.gear_radius_px_support)
    radius_rng = spawn_rng(int(instance_seed), f"{namespace}.radii")
    radii = tuple(float(radius_rng.choice(radius_support)) for _ in range(int(gear_count)))

    return GearDirectionScenario(
        scene_variant=str(scene_variant),
        gear_count=int(gear_count),
        input_direction=str(input_direction),
        output_direction=str(output_direction),
        radii_px=tuple(float(value) for value in radii),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        gear_count_probabilities=dict(gear_count_probabilities),
        input_direction_probabilities=probability_map(SUPPORTED_DIRECTIONS, selected=explicit_input),
        target_answer_probabilities=probability_map(SUPPORTED_DIRECTIONS, selected=explicit_target),
    )


def resolve_direction_choice_scenario(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> GearDirectionChoiceScenario:
    """Resolve one four-panel direction-choice scenario with a unique answer."""

    explicit_target = normalize_direction(
        params.get("target_direction", params.get("target_answer", params.get("output_direction")))
    )
    if explicit_target is not None:
        target_direction = str(explicit_target)
    elif bool(params.get("balanced_target_answer_sampling", group_default(defaults, "balanced_target_answer_sampling", True))):
        target_direction = str(balanced_axis_choice(
            SUPPORTED_DIRECTIONS,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.target_direction",
        ))
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_direction")
        target_direction = str(rng.choice(SUPPORTED_DIRECTIONS))

    explicit_letter = params.get("correct_option_letter")
    if explicit_letter is not None:
        correct_option_letter = str(explicit_letter).strip().upper()
        if correct_option_letter not in DIRECTION_OPTION_LETTERS:
            raise ValueError(f"correct_option_letter must be one of {DIRECTION_OPTION_LETTERS}")
    else:
        correct_option_letter = str(balanced_axis_choice(
            DIRECTION_OPTION_LETTERS,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.correct_option_letter",
        ))

    panel_scenarios: Dict[str, GearDirectionScenario] = {}
    distractor_direction = opposite_direction(str(target_direction))
    for option_index, letter in enumerate(DIRECTION_OPTION_LETTERS):
        desired_output = str(target_direction) if str(letter) == str(correct_option_letter) else str(distractor_direction)
        panel_params = dict(params)
        panel_params["target_answer"] = str(desired_output)
        panel_params.pop("target_direction", None)
        panel_params.pop("output_direction", None)
        panel_params.pop("correct_option_letter", None)
        panel_scenarios[str(letter)] = resolve_direction_scenario(
            instance_seed=int(instance_seed) + 7919 * (int(option_index) + 1),
            params=panel_params,
            defaults=defaults,
            namespace=f"{namespace}.panel_{letter}",
        )

    gear_count_support = integer_support(params, defaults, "gear_count_support", DEFAULTS.gear_count_support)
    gear_count_support = tuple(int(value) for value in gear_count_support if 2 <= int(value) <= 6)
    scene_variant_probabilities = {
        variant: 1.0 / float(len(SUPPORTED_SCENE_VARIANTS))
        for variant in SUPPORTED_SCENE_VARIANTS
    }
    return GearDirectionChoiceScenario(
        target_direction=str(target_direction),
        correct_option_letter=str(correct_option_letter),
        panel_scenarios={str(key): value for key, value in panel_scenarios.items()},
        target_direction_probabilities=probability_map(SUPPORTED_DIRECTIONS, selected=explicit_target),
        correct_option_letter_probabilities=probability_map(DIRECTION_OPTION_LETTERS, selected=str(correct_option_letter) if explicit_letter is not None else None),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        gear_count_probabilities=integer_probability_map(gear_count_support),
    )


def resolve_direction_gear_count(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
    input_direction: str | None,
    target_answer: str | None,
) -> Tuple[int, Dict[str, float]]:
    """Resolve gear count for direction propagation."""

    support = integer_support(params, defaults, "gear_count_support", DEFAULTS.gear_count_support)
    support = tuple(int(value) for value in support if 2 <= int(value) <= 6)
    if not support:
        raise ValueError("gear_count_support for direction must include values in 2..6")
    explicit = params.get("gear_count")
    if explicit is not None:
        gear_count = int(explicit)
        if int(gear_count) not in set(support):
            raise ValueError(f"unsupported gear_count for gear direction: {gear_count}")
        return int(gear_count), integer_probability_map(support, selected=int(gear_count))

    candidates = list(support)
    if input_direction is not None and target_answer is not None:
        candidates = [
            int(value)
            for value in candidates
            if propagate_output_direction(str(input_direction), int(value)) == str(target_answer)
        ]
        if not candidates:
            raise ValueError("explicit input_direction and target_answer are incompatible with gear_count_support")

    if bool(params.get("balanced_gear_count_sampling", group_default(defaults, "balanced_gear_count_sampling", True))):
        rng = spawn_rng(int(instance_seed), f"{namespace}.gear_count")
        gear_count = int(rng.choice(candidates))
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.gear_count")
        gear_count = int(candidates[int(rng.randrange(len(candidates)))])
    return int(gear_count), integer_probability_map(support)


def normalize_speed_relation(value: Any) -> str | None:
    """Normalize speed-up/speed-down aliases."""

    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "faster": "faster",
        "speed_up": "faster",
        "higher": "faster",
        "slower": "slower",
        "speed_down": "slower",
        "lower": "slower",
    }
    if text not in aliases:
        raise ValueError(f"unsupported speed_relation for gear-train speed: {value}")
    return str(aliases[text])


def feasible_speed_scenarios(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> Tuple[Tuple[int, int, int, int, int, str], ...]:
    """Return feasible exact integer speed scenarios."""

    gear_counts = tuple(value for value in integer_support(params, defaults, "gear_count_support", (2, 3, 4)) if 2 <= int(value) <= 4)
    if not gear_counts:
        raise ValueError("gear_count_support for speed must include values in 2..4")
    teeth_support = integer_support(params, defaults, "tooth_count_support", DEFAULTS.tooth_count_support)
    input_rpm_support = integer_support(params, defaults, "input_rpm_support", DEFAULTS.input_rpm_support)
    min_output = int(params.get("output_rpm_min", group_default(defaults, "output_rpm_min", 20)))
    max_output = int(params.get("output_rpm_max", group_default(defaults, "output_rpm_max", 240)))
    scenarios: List[Tuple[int, int, int, int, int, str]] = []
    for gear_count in gear_counts:
        for input_teeth in teeth_support:
            for output_teeth in teeth_support:
                if int(input_teeth) == int(output_teeth):
                    continue
                for input_rpm in input_rpm_support:
                    numerator = int(input_rpm) * int(input_teeth)
                    if numerator % int(output_teeth) != 0:
                        continue
                    output_rpm = int(numerator // int(output_teeth))
                    relation = speed_relation(int(input_rpm), int(output_rpm))
                    if relation not in SUPPORTED_SPEED_RELATIONS:
                        continue
                    if not (int(min_output) <= int(output_rpm) <= int(max_output)):
                        continue
                    scenarios.append((int(gear_count), int(input_teeth), int(output_teeth), int(input_rpm), int(output_rpm), str(relation)))
    if not scenarios:
        raise ValueError("no feasible gear-train speed scenarios for configured supports")
    return tuple(scenarios)


def resolve_speed_scenario(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> GearSpeedScenario:
    """Resolve one gear-speed scenario without public branch routing."""

    scene_variant, scene_variant_probabilities = resolve_scene_variant(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    feasible = list(feasible_speed_scenarios(params, defaults))
    gear_count_support = tuple(value for value in integer_support(params, defaults, "gear_count_support", (2, 3, 4)) if 2 <= int(value) <= 4)
    input_rpm_support = integer_support(params, defaults, "input_rpm_support", DEFAULTS.input_rpm_support)

    explicit_gear_count = params.get("gear_count")
    explicit_input_teeth = params.get("input_teeth", params.get("input_tooth_count"))
    explicit_output_teeth = params.get("output_teeth", params.get("output_tooth_count"))
    explicit_input_rpm = params.get("input_rpm", params.get("input_speed_rpm"))
    explicit_answer = params.get("target_answer", params.get("output_rpm"))
    explicit_relation = normalize_speed_relation(params.get("speed_relation"))

    candidates = list(feasible)
    if explicit_gear_count is not None:
        candidates = [item for item in candidates if int(item[0]) == int(explicit_gear_count)]
    if explicit_input_teeth is not None:
        candidates = [item for item in candidates if int(item[1]) == int(explicit_input_teeth)]
    if explicit_output_teeth is not None:
        candidates = [item for item in candidates if int(item[2]) == int(explicit_output_teeth)]
    if explicit_input_rpm is not None:
        candidates = [item for item in candidates if int(item[3]) == int(explicit_input_rpm)]
    if explicit_answer is not None:
        candidates = [item for item in candidates if int(item[4]) == int(explicit_answer)]
    if explicit_relation is not None:
        candidates = [item for item in candidates if str(item[5]) == str(explicit_relation)]
    if not candidates:
        raise ValueError("explicit gear-train speed parameters do not define a feasible ratio scenario")

    if explicit_relation is None and bool(params.get("balanced_speed_relation_sampling", group_default(defaults, "balanced_speed_relation_sampling", True))):
        relations = [relation for relation in SUPPORTED_SPEED_RELATIONS if any(str(item[5]) == relation for item in candidates)]
        rng = spawn_rng(int(instance_seed), f"{namespace}.speed_relation")
        selected_relation = str(rng.choice(relations))
        candidates = [item for item in candidates if str(item[5]) == selected_relation]

    if explicit_answer is None and bool(params.get("balanced_target_answer_sampling", group_default(defaults, "balanced_target_answer_sampling", True))):
        answers = sorted({int(item[4]) for item in candidates})
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_answer")
        selected_answer = int(rng.choice(answers))
        candidates = [item for item in candidates if int(item[4]) == int(selected_answer)]

    rng = spawn_rng(int(instance_seed), f"{namespace}.scenario_tuple")
    gear_count, input_teeth, output_teeth, input_rpm, output_rpm, relation = rng.choice(candidates)
    teeth_support = integer_support(params, defaults, "tooth_count_support", DEFAULTS.tooth_count_support)
    idler_rng = spawn_rng(int(instance_seed), f"{namespace}.idler_teeth")
    idler_teeth = tuple(int(idler_rng.choice(teeth_support)) for _ in range(max(0, int(gear_count) - 2)))
    all_teeth = (int(input_teeth),) + idler_teeth + (int(output_teeth),)
    radii = tuple(radius_from_teeth(value) for value in all_teeth)

    return GearSpeedScenario(
        scene_variant=str(scene_variant),
        gear_count=int(gear_count),
        input_teeth=int(input_teeth),
        output_teeth=int(output_teeth),
        idler_teeth=tuple(int(value) for value in idler_teeth),
        input_rpm=int(input_rpm),
        output_rpm=int(output_rpm),
        speed_relation=str(relation),
        radii_px=tuple(float(value) for value in radii),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        gear_count_probabilities=integer_probability_map(gear_count_support, selected=int(gear_count) if explicit_gear_count is not None else None),
        speed_relation_probabilities=probability_map(SUPPORTED_SPEED_RELATIONS, selected=explicit_relation),
        input_rpm_probabilities=integer_probability_map(input_rpm_support, selected=int(input_rpm) if explicit_input_rpm is not None else None),
        target_answer_probabilities=integer_probability_map(
            sorted({int(item[4]) for item in feasible}),
            selected=int(output_rpm) if explicit_answer is not None else None,
        ),
    )


__all__ = [
    "feasible_speed_scenarios",
    "integer_probability_map",
    "integer_support",
    "probability_map",
    "resolve_direction_choice_scenario",
    "resolve_direction_scenario",
    "resolve_scene_variant",
    "resolve_speed_scenario",
]
