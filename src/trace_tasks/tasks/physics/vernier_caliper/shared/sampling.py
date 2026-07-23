"""Sampling helpers for Vernier-caliper apparatus parameters."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import uniform_probability_map
from trace_tasks.tasks.shared.variant_sampling import (
    apply_balanced_variant_sampling,
    resolve_variant,
)

from .state import (
    DEFAULTS,
    OPTION_LETTERS,
    SCENE_NAMESPACE,
    VERNIER_DIVISIONS,
    CaliperScenario,
)


def probability_map(values: Sequence[int], selected: int | None = None) -> Dict[str, float]:
    """Return a string-keyed probability map for an integer support."""

    resolved = tuple(int(value) for value in values)
    if selected is not None:
        return {
            str(int(value)): (1.0 if int(value) == int(selected) else 0.0)
            for value in resolved
        }
    if not resolved:
        return {}
    probability = 1.0 / float(len(resolved))
    return {str(int(value)): float(probability) for value in resolved}


def integer_support(
    defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
) -> Tuple[int, ...]:
    """Resolve a non-empty sorted integer support from scene defaults."""

    raw = group_default(defaults, str(key), tuple(int(value) for value in fallback))
    support = tuple(sorted({int(value) for value in raw}))
    if not support:
        raise ValueError(f"{key} must contain at least one integer")
    return support


def main_mm_support(defaults: Mapping[str, Any]) -> Tuple[int, ...]:
    """Return supported main-scale millimeter readings."""

    return integer_support(defaults, "main_mm_support", DEFAULTS.main_mm_support)


def aligned_tick_support(defaults: Mapping[str, Any]) -> Tuple[int, ...]:
    """Return supported aligned Vernier tick indices."""

    support = integer_support(
        defaults,
        "aligned_vernier_tick_support",
        DEFAULTS.aligned_vernier_tick_support,
    )
    if any(value < 0 or value >= VERNIER_DIVISIONS for value in support):
        raise ValueError("aligned_vernier_tick_support must be in 0..9")
    return support


def _resolve_correct_option_letter(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the visual MCQ option letter while preserving balanced sampling."""

    selected, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), f"{namespace}.correct_option_letter.raw"),
        params=params,
        gen_defaults=defaults,
        supported_variants=OPTION_LETTERS,
        explicit_key="correct_option_letter",
        weights_key="correct_option_letter_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=OPTION_LETTERS,
        balance_flag_key="balanced_correct_option_letter_sampling",
        explicit_key="correct_option_letter",
        weights_key="correct_option_letter_weights",
        sampling_namespace=f"{namespace}.correct_option_letter",
    )
    return str(selected), {str(key): float(value) for key, value in probabilities.items()}


def _option_values_mm(
    *,
    instance_seed: int,
    namespace: str,
    correct_tenths: int,
    main_support: Sequence[int],
    tick_support: Sequence[int],
    correct_option_letter: str,
) -> Dict[str, float]:
    """Return one correct and five plausible nearby numeric options."""

    support_tenths = sorted(
        {
            int(main_mm) * 10 + int(tick)
            for main_mm in main_support
            for tick in tick_support
        }
    )
    support_set = set(support_tenths)
    if int(correct_tenths) not in support_set:
        raise ValueError(f"correct_tenths={correct_tenths} is outside caliper support")

    rng = spawn_rng(int(instance_seed), f"{namespace}.option_values")
    nearby_offsets = [
        -1,
        1,
        -2,
        2,
        -3,
        3,
        -4,
        4,
        -5,
        5,
        -9,
        9,
        -10,
        10,
        -11,
        11,
        -19,
        19,
        -20,
        20,
    ]
    rng.shuffle(nearby_offsets)
    distractors: list[int] = []
    for offset in nearby_offsets:
        value = int(correct_tenths) + int(offset)
        if value in support_set and value != int(correct_tenths) and value not in distractors:
            distractors.append(int(value))
        if len(distractors) >= len(OPTION_LETTERS) - 1:
            break
    if len(distractors) < len(OPTION_LETTERS) - 1:
        fallback_pool = [
            int(value)
            for value in support_tenths
            if int(value) != int(correct_tenths) and int(value) not in set(distractors)
        ]
        rng.shuffle(fallback_pool)
        distractors.extend(fallback_pool[: len(OPTION_LETTERS) - 1 - len(distractors)])
    if len(distractors) < len(OPTION_LETTERS) - 1:
        raise ValueError("not enough Vernier option distractors for six-option MCQ")

    option_values: Dict[str, float] = {}
    distractor_iter = iter(distractors)
    for letter in OPTION_LETTERS:
        if str(letter) == str(correct_option_letter):
            value_tenths = int(correct_tenths)
        else:
            value_tenths = int(next(distractor_iter))
        option_values[str(letter)] = round(float(value_tenths) / 10.0, 1)
    return dict(option_values)


def resolve_caliper_scenario(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> CaliperScenario:
    """Resolve operands while preserving answer-balanced caliper sampling invariants."""

    main_support = main_mm_support(defaults)
    tick_support = aligned_tick_support(defaults)
    effective_params = dict(params)
    raw_target_answer = effective_params.get("target_answer")
    target_answer_is_option_letter = (
        isinstance(raw_target_answer, str)
        and str(raw_target_answer).strip().upper() in set(OPTION_LETTERS)
    )
    if target_answer_is_option_letter:
        effective_params.setdefault(
            "correct_option_letter",
            str(raw_target_answer).strip().upper(),
        )
    explicit_main = params.get("main_mm")
    explicit_tick = params.get("aligned_vernier_tick", params.get("vernier_tick"))
    explicit_answer = (
        params.get("answer_mm")
        if target_answer_is_option_letter
        else params.get("target_answer", params.get("answer_mm"))
    )
    if explicit_answer is not None:
        answer_tenths = int(round(float(explicit_answer) * 10.0))
        inferred_main_mm = int(answer_tenths // 10)
        inferred_aligned_tick = int(answer_tenths % 10)
        main_mm = int(explicit_main) if explicit_main is not None else inferred_main_mm
        aligned_tick = (
            int(explicit_tick) if explicit_tick is not None else inferred_aligned_tick
        )
        if main_mm not in set(main_support) or aligned_tick not in set(tick_support):
            raise ValueError(f"unsupported target_answer for Vernier caliper: {explicit_answer}")
        if int(main_mm * 10 + aligned_tick) != int(answer_tenths):
            raise ValueError(
                "target_answer is inconsistent with explicit main_mm or "
                "aligned_vernier_tick"
            )
    else:
        if explicit_main is not None:
            main_mm = int(explicit_main)
            if main_mm not in set(main_support):
                raise ValueError(f"main_mm={main_mm} is outside configured support")
        else:
            rng = spawn_rng(int(instance_seed), f"{namespace}.main_mm")
            main_mm = int(rng.choice(main_support))

        if explicit_tick is not None:
            aligned_tick = int(explicit_tick)
            if aligned_tick not in set(tick_support):
                raise ValueError(
                    f"aligned_vernier_tick={aligned_tick} is outside configured support"
                )
        else:
            rng = spawn_rng(int(instance_seed), f"{namespace}.aligned_vernier_tick")
            aligned_tick = int(rng.choice(tick_support))

    answer_tenths = int(main_mm * 10 + aligned_tick)
    answer_mm = round(float(answer_tenths) / 10.0, 1)
    correct_option_letter, correct_option_probs = _resolve_correct_option_letter(
        instance_seed=int(instance_seed),
        params=effective_params,
        defaults=defaults,
        namespace=str(namespace),
    )
    option_values = _option_values_mm(
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        correct_tenths=int(answer_tenths),
        main_support=main_support,
        tick_support=tick_support,
        correct_option_letter=str(correct_option_letter),
    )
    answer_support = [
        int(main * 10 + tick)
        for main in main_support
        for tick in tick_support
    ]
    selected = int(answer_tenths)
    return CaliperScenario(
        main_mm=int(main_mm),
        aligned_vernier_tick=int(aligned_tick),
        answer_mm=float(answer_mm),
        option_values_mm=dict(option_values),
        correct_option_letter=str(correct_option_letter),
        correct_option_letter_probabilities=dict(correct_option_probs),
        target_answer_probabilities={
            f"{value / 10.0:.1f}": (1.0 if int(value) == selected else 0.0)
            for value in answer_support
        },
        main_mm_probabilities=probability_map(main_support, selected=int(main_mm)),
        aligned_vernier_tick_probabilities=uniform_probability_map(
            tick_support,
            selected=int(aligned_tick),
        ),
    )


__all__ = [
    "aligned_tick_support",
    "integer_support",
    "main_mm_support",
    "OPTION_LETTERS",
    "probability_map",
    "resolve_caliper_scenario",
]
