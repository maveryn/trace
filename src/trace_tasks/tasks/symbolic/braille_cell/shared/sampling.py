"""Sampling helpers for Braille-cell scene packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, TypeVar

from .....core.sampling import uniform_choice
from ....shared.word_assets import load_short_word_bank_by_length
from ...shared.common import get_int_range, resolve_symbolic_axis_variant
from ...shared.word_option_sampling import (
    PrefixWordOptionSet,
    sample_prefix_word_option_set,
    validate_prefix_word_options,
)

from .rules import braille_patterns_for_word, pattern_key
from .state import SUPPORTED_BRAILLE_SCENE_VARIANTS, BrailleCellSpec, BraillePlateSpec


T = TypeVar("T")

BRAILLE_WORD_BANK_BY_LENGTH: Mapping[int, tuple[str, ...]] = load_short_word_bank_by_length(min_length=3, max_length=5)


@dataclass(frozen=True)
class BrailleWordOptionBinding:
    """Same-length word candidates bound to fixed visible option labels."""

    target_word: str
    labels: tuple[str, ...]
    correct_label: str
    words_by_label: tuple[tuple[str, str], ...]
    shared_prefix: str
    shared_prefix_length: int
    prefix_candidate_count: int
    word_option_sampling_strategy: str


@dataclass(frozen=True)
class BrailleWordChoiceDataset:
    """Resolved word-choice scene data used by Braille word tasks."""

    scene_variant: str
    answer_value: str
    target_answer_support: tuple[str, ...]
    annotation_item_ids: tuple[str, str]
    source_word: str
    source_plate: BraillePlateSpec | None
    word_options: tuple[tuple[str, str], ...]
    braille_options: tuple[BraillePlateSpec, ...]
    metadata: dict[str, Any]
    scene_variant_probabilities: dict[str, float]


def resolve_braille_scene_variant(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve the non-semantic Braille card style axis."""

    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_BRAILLE_SCENE_VARIANTS,
        task_id=str(namespace),
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def build_with_retries(
    factory: Callable[[int], T],
    *,
    instance_seed: int,
    max_attempts: int,
    failure_message: str,
) -> T:
    """Call a deterministic dataset factory with sequential retry seeds."""

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            return factory(int(instance_seed) + int(attempt_index))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(str(failure_message)) from last_error


def available_word_lengths(*, min_length: int, max_length: int, option_count: int) -> tuple[int, ...]:
    """Return word lengths with enough shared words for an option set."""

    lengths = []
    for length in range(int(min_length), int(max_length) + 1):
        words = BRAILLE_WORD_BANK_BY_LENGTH.get(int(length), ())
        if len(words) >= int(option_count):
            lengths.append(int(length))
    if not lengths:
        raise ValueError("no Braille word lengths have enough shared option candidates")
    return tuple(lengths)


def sample_word_options(
    rng: Any,
    *,
    min_length: int,
    max_length: int,
    option_count: int,
) -> PrefixWordOptionSet:
    """Sample one target word and same-prefix distractor options."""

    return sample_prefix_word_option_set(
        rng,
        word_bank_by_length=BRAILLE_WORD_BANK_BY_LENGTH,
        min_length=int(min_length),
        max_length=int(max_length),
        option_count=int(option_count),
        params={},
        gen_defaults={},
        min_shared_prefix_length=1,
    )


def sample_word_option_set(
    rng: Any,
    *,
    min_length: int,
    max_length: int,
    option_count: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    target_word: str | None = None,
    word_length: int | None = None,
) -> PrefixWordOptionSet:
    """Sample a target word and same-prefix unique options containing it."""

    return sample_prefix_word_option_set(
        rng,
        word_bank_by_length=BRAILLE_WORD_BANK_BY_LENGTH,
        min_length=int(min_length),
        max_length=int(max_length),
        option_count=int(option_count),
        params=params,
        gen_defaults=gen_defaults,
        target_word=target_word,
        word_length=word_length,
        min_shared_prefix_length=1,
    )


def validate_word_options(words: Sequence[str], *, option_count: int) -> tuple[str, ...]:
    """Validate a same-length Braille word option set."""

    normalized = tuple(str(word).lower().strip() for word in words)
    if len(normalized) != int(option_count):
        raise ValueError(f"expected exactly {option_count} Braille word options")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Braille word options must be unique")
    lengths = {len(word) for word in normalized}
    if len(lengths) != 1:
        raise ValueError("Braille word options must all have the same length")
    if any((not word.isalpha()) for word in normalized):
        raise ValueError("Braille word options must contain alphabetic letters only")
    return normalized


def resolve_word_length_bounds(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> tuple[int, int]:
    """Return configured Braille word length support."""

    word_min, word_max = get_int_range(
        params,
        gen_defaults,
        min_key="word_length_min",
        max_key="word_length_max",
        fallback_min=3,
        fallback_max=5,
    )
    if int(word_min) < 1 or int(word_max) < int(word_min):
        raise ValueError("invalid Braille word length support")
    return int(word_min), int(word_max)


def resolve_word_option_count(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> int:
    """Return the fixed Braille word option count."""

    option_count = int(params.get("word_option_count", gen_defaults.get("word_option_count", 4)))
    if int(option_count) != 4:
        raise ValueError("Braille word tasks require exactly four visual options")
    return int(option_count)


def make_word_plate(*, item_id: str, word: str, label: str, role: str, marked: bool) -> BraillePlateSpec:
    """Create a Braille plate spec for a lowercase word."""

    cells = tuple(
        BrailleCellSpec(
            item_id=f"{item_id}_cell_{index + 1}",
            raised_positions=tuple(pattern),
            label="",
            role=f"{role}_cell",
            marked=False,
        )
        for index, pattern in enumerate(braille_patterns_for_word(word))
    )
    return BraillePlateSpec(
        item_id=str(item_id),
        cells=cells,
        label=str(label),
        role=str(role),
        marked=bool(marked),
    )


def bind_words_to_option_labels(
    *,
    rng: Any,
    labels: Sequence[str],
    correct_label: str,
    target_word: str,
    option_words: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    """Bind a target word and distractor words to fixed visible option labels."""

    label_tuple = tuple(str(label) for label in labels)
    if str(correct_label) not in label_tuple:
        raise ValueError(f"correct_label must be one of {label_tuple}")
    distractors = [str(word) for word in option_words if str(word) != str(target_word)]
    rng.shuffle(distractors)
    bound: list[tuple[str, str]] = []
    cursor = 0
    for label in label_tuple:
        if str(label) == str(correct_label):
            word = str(target_word)
        else:
            word = str(distractors[int(cursor)])
            cursor += 1
        bound.append((str(label), str(word)))
    return tuple(bound)


def sample_labeled_word_options(
    rng: Any,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    labels: Sequence[str],
) -> BrailleWordOptionBinding:
    """Sample same-length word options and bind the target to one label."""

    option_count = resolve_word_option_count(params, gen_defaults)
    word_min, word_max = resolve_word_length_bounds(params, gen_defaults)
    word_length = params.get("word_length")
    if "word_options" in params:
        option_words = validate_word_options(params["word_options"], option_count=int(option_count))
        prefix_set = validate_prefix_word_options(option_words, option_count=int(option_count), min_shared_prefix_length=1)
        target_word = str(params.get("target_word", option_words[0])).lower().strip()
        if target_word not in option_words:
            raise ValueError("target_word must be included in word_options")
    else:
        prefix_set = sample_word_option_set(
            rng,
            min_length=int(word_min),
            max_length=int(word_max),
            option_count=int(option_count),
            params=params,
            gen_defaults=gen_defaults,
            target_word=params.get("target_word"),
            word_length=int(word_length) if word_length is not None else None,
        )
        target_word = str(prefix_set.target_word)
        option_words = tuple(prefix_set.option_words)

    label_tuple = tuple(str(label) for label in labels)
    correct_label = str(params.get("correct_label", uniform_choice(rng, label_tuple, sort_keys=False)))
    words_by_label = bind_words_to_option_labels(
        rng=rng,
        labels=label_tuple,
        correct_label=str(correct_label),
        target_word=str(target_word),
        option_words=tuple(option_words),
    )
    return BrailleWordOptionBinding(
        target_word=str(target_word),
        labels=label_tuple,
        correct_label=str(correct_label),
        words_by_label=tuple(words_by_label),
        shared_prefix=str(prefix_set.shared_prefix),
        shared_prefix_length=int(prefix_set.shared_prefix_length),
        prefix_candidate_count=int(prefix_set.prefix_candidate_count),
        word_option_sampling_strategy=str(prefix_set.sampling_strategy),
    )


def build_plate_source_word_choice(
    *,
    rng: Any,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    labels: Sequence[str],
) -> BrailleWordChoiceDataset:
    """Build a choice dataset where the source is a Braille word plate."""

    binding = sample_labeled_word_options(
        rng=rng,
        params=params,
        gen_defaults=gen_defaults,
        labels=labels,
    )
    source_plate = make_word_plate(
        item_id="source_plate",
        word=str(binding.target_word),
        label="",
        role="source_plate",
        marked=True,
    )
    target_patterns = braille_patterns_for_word(str(binding.target_word))
    option_word_map = {str(label): str(word) for label, word in binding.words_by_label}
    return BrailleWordChoiceDataset(
        scene_variant=str(scene_variant),
        answer_value=str(binding.correct_label),
        target_answer_support=tuple(binding.labels),
        annotation_item_ids=("source_plate", f"option_{binding.correct_label}"),
        source_word=str(binding.target_word),
        source_plate=source_plate,
        word_options=tuple(binding.words_by_label),
        braille_options=(),
        metadata={
            "target_word": str(binding.target_word),
            "word_length": int(len(binding.target_word)),
            "target_patterns": [pattern_key(pattern) for pattern in target_patterns],
            "target_raised_positions": [[int(pos) for pos in pattern] for pattern in target_patterns],
            "correct_option_label": str(binding.correct_label),
            "correct_option_id": f"option_{binding.correct_label}",
            "option_words": dict(option_word_map),
            "shared_prefix": str(binding.shared_prefix),
            "shared_prefix_length": int(binding.shared_prefix_length),
            "prefix_candidate_count": int(binding.prefix_candidate_count),
            "word_option_sampling_strategy": str(binding.word_option_sampling_strategy),
        },
        scene_variant_probabilities={str(key): float(value) for key, value in scene_variant_probabilities.items()},
    )


def build_word_source_braille_choice(
    *,
    rng: Any,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    labels: Sequence[str],
) -> BrailleWordChoiceDataset:
    """Build a choice dataset where the source is a visible text word."""

    binding = sample_labeled_word_options(
        rng=rng,
        params=params,
        gen_defaults=gen_defaults,
        labels=labels,
    )
    braille_options = tuple(
        make_word_plate(
            item_id=f"option_{label}",
            word=str(word),
            label=str(label),
            role="correct_option" if str(label) == str(binding.correct_label) else "distractor_option",
            marked=False,
        )
        for label, word in binding.words_by_label
    )
    source_patterns = braille_patterns_for_word(str(binding.target_word))
    option_patterns = {
        str(label): [pattern_key(pattern) for pattern in braille_patterns_for_word(str(word))]
        for label, word in binding.words_by_label
    }
    option_word_map = {str(label): str(word) for label, word in binding.words_by_label}
    return BrailleWordChoiceDataset(
        scene_variant=str(scene_variant),
        answer_value=str(binding.correct_label),
        target_answer_support=tuple(binding.labels),
        annotation_item_ids=("source_word", f"option_{binding.correct_label}"),
        source_word=str(binding.target_word),
        source_plate=None,
        word_options=(),
        braille_options=tuple(braille_options),
        metadata={
            "source_word": str(binding.target_word),
            "word_length": int(len(binding.target_word)),
            "source_patterns": [pattern_key(pattern) for pattern in source_patterns],
            "source_raised_positions": [[int(pos) for pos in pattern] for pattern in source_patterns],
            "correct_option_label": str(binding.correct_label),
            "correct_option_id": f"option_{binding.correct_label}",
            "option_words": dict(option_word_map),
            "option_patterns": dict(option_patterns),
            "shared_prefix": str(binding.shared_prefix),
            "shared_prefix_length": int(binding.shared_prefix_length),
            "prefix_candidate_count": int(binding.prefix_candidate_count),
            "word_option_sampling_strategy": str(binding.word_option_sampling_strategy),
        },
        scene_variant_probabilities={str(key): float(value) for key, value in scene_variant_probabilities.items()},
    )


__all__ = [
    "BRAILLE_WORD_BANK_BY_LENGTH",
    "BrailleWordChoiceDataset",
    "BrailleWordOptionBinding",
    "available_word_lengths",
    "bind_words_to_option_labels",
    "build_plate_source_word_choice",
    "build_word_source_braille_choice",
    "build_with_retries",
    "make_word_plate",
    "resolve_braille_scene_variant",
    "resolve_word_length_bounds",
    "resolve_word_option_count",
    "sample_labeled_word_options",
    "sample_word_option_set",
    "sample_word_options",
    "validate_word_options",
]
