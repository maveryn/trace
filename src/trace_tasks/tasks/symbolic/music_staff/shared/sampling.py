"""Neutral sampling helpers for symbolic music-staff scenes."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, Tuple

from ....shared.mcq import option_label_for_index
from ...shared.common import resolve_symbolic_axis_variant

from .components import OptionCard
from .state import SCENE_VARIANTS


def resolve_scene_variant(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> Tuple[str, dict[str, float]]:
    """Resolve a music-staff visual scene variant."""

    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=defaults,
        instance_seed=int(instance_seed),
        supported_variants=SCENE_VARIANTS,
        task_id=str(namespace),
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def resolve_count_target(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
    minimum: int = 0,
    maximum: int = 4,
    context: str,
) -> Tuple[int, Tuple[int, ...], dict[str, float]]:
    """Resolve an integer target count from scene defaults and task params."""

    min_value = int(params.get("target_answer_min", defaults.get("target_answer_min", int(minimum))))
    max_value = int(params.get("target_answer_max", defaults.get("target_answer_max", int(maximum))))
    if int(min_value) > int(max_value):
        raise ValueError(f"target_answer_min must be <= target_answer_max for {context}")
    support = tuple(range(int(min_value), int(max_value) + 1))
    selected, probabilities = resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=defaults,
        instance_seed=int(instance_seed),
        supported_variants=[str(value) for value in support],
        task_id=str(namespace),
        explicit_key="target_answer",
        weights_key="target_answer_weights",
        balance_flag_key="balanced_target_answer_sampling",
        axis_namespace="target_answer",
    )
    return int(selected), tuple(int(value) for value in support), {str(key): float(value) for key, value in probabilities.items()}


def resolve_named_choice(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
    values: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    axis_namespace: str,
) -> Tuple[str, dict[str, float]]:
    """Resolve one finite named semantic generation choice."""

    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=defaults,
        instance_seed=int(instance_seed),
        supported_variants=tuple(str(value) for value in values),
        task_id=str(namespace),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        axis_namespace=str(axis_namespace),
    )


def build_text_option_cards(
    rng,
    *,
    correct_text: str,
    candidate_texts: Iterable[str],
) -> tuple[tuple[OptionCard, ...], str, tuple[str, ...]]:
    """Build visible text option cards for music-staff readout tasks."""

    correct = str(correct_text)
    candidates = []
    seen = set()
    for value in (correct, *tuple(str(item) for item in candidate_texts)):
        if value in seen:
            continue
        seen.add(value)
        candidates.append(value)

    option_count = 4 if len(candidates) <= 5 else 6
    if len(candidates) < option_count:
        raise ValueError("insufficient music-staff option candidates")

    distractors = [value for value in candidates if value != correct]
    rng.shuffle(distractors)
    option_texts = [correct, *distractors[: option_count - 1]]
    rng.shuffle(option_texts)

    cards = []
    answer_label = ""
    for index, text in enumerate(option_texts):
        label = option_label_for_index(index)
        is_correct = str(text) == correct
        if is_correct:
            answer_label = str(label)
        cards.append(OptionCard(f"option_{label}", str(label), text=str(text), is_correct=bool(is_correct)))
    if not answer_label:
        raise RuntimeError("failed to bind music-staff correct option label")
    return tuple(cards), str(answer_label), tuple(str(text) for text in option_texts)
