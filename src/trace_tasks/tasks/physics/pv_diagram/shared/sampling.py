"""Sampling and symbolic mechanics for pressure-volume diagrams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.physics.shared.style import SUPPORTED_PHYSICS_COLOR_NAMES
from trace_tasks.tasks.physics.shared.support_sampling import resolve_integer_support
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import uniform_probability_map
from trace_tasks.tasks.shared.variant_sampling import (
    apply_balanced_variant_sampling,
    is_uniform_probability_map,
    resolve_variant,
)

from .state import (
    OPTION_LETTERS,
    PVDiagramSceneSpec,
    PVDiagramSignChoiceAxes,
    PVDiagramTaskDefaults,
    PVDiagramWorkAxes,
    PVProcessCandidate,
    PVWorkScenario,
    SCENE_NAMESPACE,
    SUPPORTED_SCENE_VARIANTS,
    SUPPORTED_TARGET_SIGNS,
    SUPPORTED_WORK_MODES,
)


DEFAULTS = PVDiagramTaskDefaults()


def _string_probability_map(values: Sequence[str], selected: str | None = None) -> dict[str, float]:
    support = tuple(str(value) for value in values)
    if selected is not None:
        return {str(value): (1.0 if str(value) == str(selected) else 0.0) for value in support}
    probability = 1.0 / float(len(support)) if support else 0.0
    return {str(value): float(probability) for value in support}


def pressure_support(params: Mapping[str, Any], generation_defaults: Mapping[str, Any]) -> tuple[int, ...]:
    """Return configured pressure support in kPa."""

    return resolve_integer_support(
        params,
        gen_defaults=generation_defaults,
        key="pressure_support",
        fallback=DEFAULTS.pressure_support,
    )


def volume_support(params: Mapping[str, Any], generation_defaults: Mapping[str, Any]) -> tuple[int, ...]:
    """Return configured volume support in liters."""

    return resolve_integer_support(
        params,
        gen_defaults=generation_defaults,
        key="volume_support",
        fallback=DEFAULTS.volume_support,
    )


def work_answer_support(params: Mapping[str, Any], generation_defaults: Mapping[str, Any]) -> tuple[int, ...]:
    """Return configured signed PV-work answer support in joules."""

    return resolve_integer_support(
        params,
        gen_defaults=generation_defaults,
        key="work_answer_support",
        fallback=DEFAULTS.work_answer_support,
    )


def sign_for_work(work_value: int) -> str:
    """Return positive/negative/zero for one signed work value."""

    if int(work_value) > 0:
        return "positive"
    if int(work_value) < 0:
        return "negative"
    return "zero"


def feasible_work_scenarios(
    params: Mapping[str, Any],
    *,
    generation_defaults: Mapping[str, Any],
    work_mode: str | None = None,
) -> tuple[PVWorkScenario, ...]:
    """Enumerate PV constructions satisfying supports and explicit overrides."""

    pressures = pressure_support(params, generation_defaults)
    volumes = volume_support(params, generation_defaults)
    configured_answers = set(work_answer_support(params, generation_defaults))
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
    explicit_pressure = params.get("pressure")
    explicit_volume_start = params.get("volume_start")
    explicit_volume_end = params.get("volume_end")
    explicit_pressure_low = params.get("pressure_low")
    explicit_pressure_high = params.get("pressure_high")
    explicit_volume_left = params.get("volume_left")
    explicit_volume_right = params.get("volume_right")
    explicit_cycle_direction = params.get("cycle_direction")

    modes = [str(work_mode)] if work_mode is not None else list(SUPPORTED_WORK_MODES)
    scenarios: list[PVWorkScenario] = []
    if "single_process" in modes:
        for pressure in pressures:
            if explicit_pressure is not None and int(pressure) != int(explicit_pressure):
                continue
            for volume_start in volumes:
                if explicit_volume_start is not None and int(volume_start) != int(explicit_volume_start):
                    continue
                for volume_end in volumes:
                    if explicit_volume_end is not None and int(volume_end) != int(explicit_volume_end):
                        continue
                    delta_v = int(volume_end) - int(volume_start)
                    if int(delta_v) == 0:
                        continue
                    if abs(int(delta_v)) < int(min_delta) or abs(int(delta_v)) > int(max_delta):
                        continue
                    work_value = int(pressure) * int(delta_v)
                    if int(work_value) not in configured_answers:
                        continue
                    scenarios.append(
                        PVWorkScenario(
                            work_mode="single_process",
                            work_value=int(work_value),
                            pressure=int(pressure),
                            volume_start=int(volume_start),
                            volume_end=int(volume_end),
                            pressure_low=None,
                            pressure_high=None,
                            volume_left=None,
                            volume_right=None,
                            cycle_direction=None,
                        )
                    )

    if "rectangular_cycle" in modes:
        directions = ("clockwise", "counterclockwise")
        if explicit_cycle_direction is not None:
            direction_text = str(explicit_cycle_direction)
            if direction_text not in set(directions):
                raise ValueError(f"unsupported cycle_direction: {explicit_cycle_direction}")
            directions = (direction_text,)
        for pressure_low in pressures:
            if explicit_pressure_low is not None and int(pressure_low) != int(explicit_pressure_low):
                continue
            for pressure_high in pressures:
                if explicit_pressure_high is not None and int(pressure_high) != int(explicit_pressure_high):
                    continue
                if int(pressure_high) <= int(pressure_low):
                    continue
                delta_p = int(pressure_high) - int(pressure_low)
                for volume_left in volumes:
                    if explicit_volume_left is not None and int(volume_left) != int(explicit_volume_left):
                        continue
                    for volume_right in volumes:
                        if explicit_volume_right is not None and int(volume_right) != int(explicit_volume_right):
                            continue
                        if int(volume_right) <= int(volume_left):
                            continue
                        delta_v = int(volume_right) - int(volume_left)
                        if int(delta_v) < int(min_delta) or int(delta_v) > int(max_delta):
                            continue
                        area = int(delta_p) * int(delta_v)
                        for direction in directions:
                            work_value = int(area) if str(direction) == "clockwise" else -int(area)
                            if int(work_value) not in configured_answers:
                                continue
                            scenarios.append(
                                PVWorkScenario(
                                    work_mode="rectangular_cycle",
                                    work_value=int(work_value),
                                    pressure=None,
                                    volume_start=None,
                                    volume_end=None,
                                    pressure_low=int(pressure_low),
                                    pressure_high=int(pressure_high),
                                    volume_left=int(volume_left),
                                    volume_right=int(volume_right),
                                    cycle_direction=str(direction),
                                )
                            )

    if not scenarios:
        raise ValueError("no feasible PV-work scenarios for configured supports")
    return tuple(scenarios)


def feasible_work_answers(
    params: Mapping[str, Any],
    *,
    generation_defaults: Mapping[str, Any],
    work_mode: str,
) -> tuple[int, ...]:
    """Return configured work answers that have at least one construction."""

    feasible = sorted(
        {
            int(scenario.work_value)
            for scenario in feasible_work_scenarios(
                params,
                generation_defaults=generation_defaults,
                work_mode=str(work_mode),
            )
        },
        key=lambda value: (abs(int(value)), 0 if int(value) > 0 else 1),
    )
    if not feasible:
        raise ValueError(f"no feasible PV-work answers for {work_mode}")
    return tuple(int(value) for value in feasible)


def _resolve_scene_variant(
    axis_rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> tuple[str, dict[str, float]]:
    selected, probabilities = resolve_variant(
        axis_rng,
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        balance_flag_key="balanced_scene_variant_sampling",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        sampling_namespace=f"{namespace}.scene_variant",
    )
    return str(selected), {str(key): float(value) for key, value in sorted(probabilities.items())}


def _resolve_accent_color(
    axis_rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> tuple[str, dict[str, float]]:
    selected, probabilities = resolve_variant(
        axis_rng,
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=SUPPORTED_PHYSICS_COLOR_NAMES,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_PHYSICS_COLOR_NAMES,
        balance_flag_key="balanced_accent_color_name_sampling",
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        sampling_namespace=f"{namespace}.accent_color_name",
    )
    return str(selected), {str(key): float(value) for key, value in sorted(probabilities.items())}


def _resolve_work_mode(
    axis_rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> tuple[str, dict[str, float]]:
    selected, probabilities = resolve_variant(
        axis_rng,
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=SUPPORTED_WORK_MODES,
        explicit_key="work_mode",
        weights_key="work_mode_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_WORK_MODES,
        balance_flag_key="balanced_work_mode_sampling",
        explicit_key="work_mode",
        weights_key="work_mode_weights",
        sampling_namespace=f"{namespace}.work_mode",
    )
    return str(selected), {str(key): float(value) for key, value in sorted(probabilities.items())}


def _resolve_target_sign(
    axis_rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> tuple[str, dict[str, float]]:
    selected, probabilities = resolve_variant(
        axis_rng,
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=SUPPORTED_TARGET_SIGNS,
        explicit_key="target_sign",
        weights_key="target_sign_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_TARGET_SIGNS,
        balance_flag_key="balanced_target_sign_sampling",
        explicit_key="target_sign",
        weights_key="target_sign_weights",
        sampling_namespace=f"{namespace}.target_sign",
    )
    return str(selected), {str(key): float(value) for key, value in sorted(probabilities.items())}


def _resolve_correct_option_letter(
    axis_rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve the unique correct option letter while preserving balanced coverage."""

    option_params = dict(params)
    if option_params.get("correct_option_letter") is None and option_params.get("target_answer") is not None:
        option_params["correct_option_letter"] = str(option_params["target_answer"]).strip().upper()
    selected, probabilities = resolve_variant(
        axis_rng,
        params=option_params,
        gen_defaults=generation_defaults,
        supported_variants=OPTION_LETTERS,
        explicit_key="correct_option_letter",
        weights_key="correct_option_letter_weights",
    )
    balanced_enabled = bool(
        option_params.get(
            "balanced_correct_option_letter_sampling",
            group_default(generation_defaults, "balanced_correct_option_letter_sampling", True),
        )
    )
    has_override = any(
        option_params.get(str(key)) is not None
        for key in ("correct_option_letter", "correct_option_letter_weights")
    )
    if bool(balanced_enabled) and not bool(has_override) and is_uniform_probability_map(probabilities):
        selected = apply_balanced_variant_sampling(
            instance_seed=int(instance_seed),
            params=option_params,
            gen_defaults=generation_defaults,
            selected_variant=str(selected),
            variant_probabilities=probabilities,
            supported_variants=OPTION_LETTERS,
            balance_flag_key="balanced_correct_option_letter_sampling",
            explicit_key="correct_option_letter",
            weights_key="correct_option_letter_weights",
            sampling_namespace=f"{namespace}.correct_option_letter",
        )
    return str(selected), {str(key): float(value) for key, value in sorted(probabilities.items())}


def _resolve_work_target_answer(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    work_mode: str,
    namespace: str,
) -> tuple[int, dict[str, float]]:
    support = feasible_work_answers(
        params,
        generation_defaults=generation_defaults,
        work_mode=str(work_mode),
    )
    explicit = params.get("target_answer")
    if explicit is not None:
        selected = int(explicit)
        if int(selected) not in set(support):
            raise ValueError(f"unsupported target_answer for {work_mode}: {selected}")
        return int(selected), uniform_probability_map(support, selected=int(selected))

    balanced_enabled = bool(
        params.get(
            "balanced_target_answer_sampling",
            group_default(generation_defaults, "balanced_target_answer_sampling", True),
        )
    )
    if bool(balanced_enabled):
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_answer.{str(work_mode)}")
        selected = int(rng.choice(support))
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_answer.{str(work_mode)}")
        selected = int(support[int(rng.randrange(len(support)))])
    return int(selected), uniform_probability_map(support)


def resolve_work_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> PVDiagramWorkAxes:
    """Resolve scene, construction mode, color, and target answer for work scenes."""

    axis_rng = spawn_rng(int(instance_seed), f"{namespace}.work.axes")
    scene_variant, scene_probs = _resolve_scene_variant(
        axis_rng,
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        namespace=str(namespace),
    )
    work_mode, work_mode_probs = _resolve_work_mode(
        axis_rng,
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        namespace=str(namespace),
    )
    accent_name, accent_probs = _resolve_accent_color(
        axis_rng,
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        namespace=str(namespace),
    )
    target_answer, target_probs = _resolve_work_target_answer(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        work_mode=str(work_mode),
        namespace=str(namespace),
    )
    return PVDiagramWorkAxes(
        scene_variant=str(scene_variant),
        work_mode=str(work_mode),
        accent_color_name=str(accent_name),
        target_answer=int(target_answer),
        scene_variant_probabilities=dict(scene_probs),
        work_mode_probabilities=dict(work_mode_probs),
        accent_color_name_probabilities=dict(accent_probs),
        target_answer_probabilities=dict(target_probs),
    )


def resolve_sign_choice_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> PVDiagramSignChoiceAxes:
    """Resolve scene, target sign, correct letter, and color for sign-choice scenes."""

    axis_rng = spawn_rng(int(instance_seed), f"{namespace}.sign_choice.axes")
    scene_variant, scene_probs = _resolve_scene_variant(
        axis_rng,
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        namespace=str(namespace),
    )
    target_sign, target_sign_probs = _resolve_target_sign(
        axis_rng,
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        namespace=str(namespace),
    )
    option_letter, option_probs = _resolve_correct_option_letter(
        axis_rng,
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        namespace=str(namespace),
    )
    accent_name, accent_probs = _resolve_accent_color(
        axis_rng,
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        namespace=str(namespace),
    )
    return PVDiagramSignChoiceAxes(
        scene_variant=str(scene_variant),
        target_sign=str(target_sign),
        correct_option_letter=str(option_letter),
        accent_color_name=str(accent_name),
        target_answer=str(option_letter),
        scene_variant_probabilities=dict(scene_probs),
        target_sign_probabilities=dict(target_sign_probs),
        correct_option_letter_probabilities=dict(option_probs),
        accent_color_name_probabilities=dict(accent_probs),
        target_answer_probabilities=dict(option_probs),
    )


def sample_work_scenario(
    rng,
    *,
    work_mode: str,
    target_answer: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> PVWorkScenario:
    """Sample one symbolic PV-work scenario that realizes the target answer."""

    scenarios = [
        scenario
        for scenario in feasible_work_scenarios(
            params,
            generation_defaults=generation_defaults,
            work_mode=str(work_mode),
        )
        if int(scenario.work_value) == int(target_answer)
    ]
    if not scenarios:
        raise ValueError(f"no feasible PV-work scenario for {work_mode} target {target_answer}")
    return scenarios[int(rng.randrange(len(scenarios)))]


def _candidate_values_for_sign(rng, *, sign: str) -> tuple[int, int, int, int]:
    """Return start/end pressure and volume for one sign-choice mini process."""

    if str(sign) == "positive":
        pressure = int(rng.choice((3, 4, 5, 6, 7, 8)))
        left = int(rng.choice((2, 3, 4)))
        right = int(rng.choice((8, 9, 10)))
        return pressure, pressure, left, right
    if str(sign) == "negative":
        pressure = int(rng.choice((3, 4, 5, 6, 7, 8)))
        left = int(rng.choice((2, 3, 4)))
        right = int(rng.choice((8, 9, 10)))
        return pressure, pressure, right, left
    volume = int(rng.choice((4, 5, 6, 7)))
    p0 = int(rng.choice((2, 3, 4)))
    p1 = int(rng.choice((7, 8, 9)))
    if int(rng.randrange(2)) == 0:
        return p0, p1, volume, volume
    return p1, p0, volume, volume


def sample_process_candidates(
    rng,
    *,
    target_sign: str,
    correct_option_letter: str,
) -> tuple[PVProcessCandidate, ...]:
    """Sample labeled process candidates with exactly one target-sign match."""

    candidates: list[PVProcessCandidate] = []
    non_target_signs = [sign for sign in SUPPORTED_TARGET_SIGNS if str(sign) != str(target_sign)]
    for letter in OPTION_LETTERS:
        sign = str(target_sign) if str(letter) == str(correct_option_letter) else str(rng.choice(non_target_signs))
        pressure_start, pressure_end, volume_start, volume_end = _candidate_values_for_sign(rng, sign=str(sign))
        candidates.append(
            PVProcessCandidate(
                letter=str(letter),
                sign=str(sign),
                pressure_start=int(pressure_start),
                pressure_end=int(pressure_end),
                volume_start=int(volume_start),
                volume_end=int(volume_end),
            )
        )
    return tuple(candidates)


def sample_work_scene_spec(
    rng,
    *,
    axes: PVDiagramWorkAxes,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> PVDiagramSceneSpec:
    """Sample one symbolic work-value PV scene."""

    scenario = sample_work_scenario(
        rng,
        work_mode=str(axes.work_mode),
        target_answer=int(axes.target_answer),
        params=params,
        generation_defaults=generation_defaults,
    )
    return PVDiagramSceneSpec(
        scene_variant=str(axes.scene_variant),
        diagram_kind="work_plot",
        work_mode=str(axes.work_mode),
        target_sign=None,
        correct_option_letter=None,
        target_answer=int(scenario.work_value),
        work_scenario=scenario,
        process_candidates=(),
        annotation_entity_ids=("work_witness_region",),
    )


def sample_sign_choice_scene_spec(
    rng,
    *,
    axes: PVDiagramSignChoiceAxes,
) -> PVDiagramSceneSpec:
    """Sample one symbolic work-sign choice PV scene."""

    candidates = sample_process_candidates(
        rng,
        target_sign=str(axes.target_sign),
        correct_option_letter=str(axes.correct_option_letter),
    )
    return PVDiagramSceneSpec(
        scene_variant=str(axes.scene_variant),
        diagram_kind="sign_choice_panel",
        work_mode=None,
        target_sign=str(axes.target_sign),
        correct_option_letter=str(axes.correct_option_letter),
        target_answer=str(axes.correct_option_letter),
        work_scenario=None,
        process_candidates=tuple(candidates),
        annotation_entity_ids=(f"option_{str(axes.correct_option_letter)}_process",),
    )


__all__ = [
    "feasible_work_answers",
    "feasible_work_scenarios",
    "pressure_support",
    "resolve_sign_choice_axes",
    "resolve_work_axes",
    "sample_sign_choice_scene_spec",
    "sample_work_scene_spec",
    "sign_for_work",
    "volume_support",
    "work_answer_support",
]
