"""Sampling helpers for Morse-code scene packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, TypeVar

from .....core.sampling import uniform_choice
from ....shared.word_assets import load_short_word_bank_by_length
from ...shared.word_option_sampling import (
    PrefixWordOptionSet,
    sample_prefix_word_option_set,
    validate_prefix_word_options,
)
from ...shared.common import get_int_range, resolve_symbolic_axis_variant

from .rules import morse_codes_for_word
from .state import (
    SUPPORTED_MORSE_SCENE_VARIANTS,
    MorseLetterSpec,
    MorseSymbolSpec,
    MorseWordSpec,
)


T = TypeVar("T")

MORSE_WORD_BANK_BY_LENGTH: Mapping[int, tuple[str, ...]] = load_short_word_bank_by_length(min_length=3, max_length=5)


@dataclass(frozen=True)
class MorseWordOptionBinding:
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
class MorseWordChoiceDataset:
    """Resolved word-choice scene data used by Morse word tasks."""

    scene_variant: str
    answer_value: str
    target_answer_support: tuple[str, ...]
    annotation_item_ids: tuple[str, str]
    source_word: str
    source_code: MorseWordSpec | None
    word_options: tuple[tuple[str, str], ...]
    code_options: tuple[MorseWordSpec, ...]
    metadata: dict[str, Any]
    scene_variant_probabilities: dict[str, float]


def resolve_morse_scene_variant(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve the non-semantic Morse card style axis."""

    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_MORSE_SCENE_VARIANTS,
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


def resolve_word_length_bounds(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> tuple[int, int]:
    """Return configured Morse word length support."""

    word_min, word_max = get_int_range(
        params,
        gen_defaults,
        min_key="word_length_min",
        max_key="word_length_max",
        fallback_min=3,
        fallback_max=5,
    )
    if int(word_min) < 1 or int(word_max) < int(word_min):
        raise ValueError("invalid Morse word length support")
    return int(word_min), int(word_max)


def resolve_word_option_count(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> int:
    """Return the fixed Morse word option count."""

    option_count = int(params.get("word_option_count", gen_defaults.get("word_option_count", 4)))
    if int(option_count) != 4:
        raise ValueError("Morse word tasks require exactly four visual options")
    return int(option_count)


def available_word_lengths(*, min_length: int, max_length: int, option_count: int) -> tuple[int, ...]:
    """Return word lengths with enough shared words for an option set."""

    lengths = []
    for length in range(int(min_length), int(max_length) + 1):
        words = MORSE_WORD_BANK_BY_LENGTH.get(int(length), ())
        if len(words) >= int(option_count):
            lengths.append(int(length))
    if not lengths:
        raise ValueError("no Morse word lengths have enough shared option candidates")
    return tuple(lengths)


def validate_word_options(words: Sequence[str], *, option_count: int) -> tuple[str, ...]:
    """Validate a same-length Morse word option set."""

    normalized = tuple(str(word).lower().strip() for word in words)
    if len(normalized) != int(option_count):
        raise ValueError(f"expected exactly {option_count} Morse word options")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Morse word options must be unique")
    if len({len(word) for word in normalized}) != 1:
        raise ValueError("Morse word options must all have the same length")
    if any((not word.isalpha()) for word in normalized):
        raise ValueError("Morse word options must contain alphabetic letters only")
    return normalized


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
        word_bank_by_length=MORSE_WORD_BANK_BY_LENGTH,
        min_length=int(min_length),
        max_length=int(max_length),
        option_count=int(option_count),
        params=params,
        gen_defaults=gen_defaults,
        target_word=target_word,
        word_length=word_length,
        min_shared_prefix_length=1,
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
) -> MorseWordOptionBinding:
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
    return MorseWordOptionBinding(
        target_word=str(target_word),
        labels=label_tuple,
        correct_label=str(correct_label),
        words_by_label=tuple(words_by_label),
        shared_prefix=str(prefix_set.shared_prefix),
        shared_prefix_length=int(prefix_set.shared_prefix_length),
        prefix_candidate_count=int(prefix_set.prefix_candidate_count),
        word_option_sampling_strategy=str(prefix_set.sampling_strategy),
    )


def make_morse_word_spec(*, item_id: str, word: str, label: str, role: str, marked: bool) -> MorseWordSpec:
    """Create a Morse word spec for one lowercase word."""

    letters: list[MorseLetterSpec] = []
    for letter_index, (letter, code) in enumerate(zip(str(word).lower().strip(), morse_codes_for_word(word))):
        symbols = tuple(
            MorseSymbolSpec(
                item_id=f"{item_id}_letter_{letter_index + 1}_symbol_{symbol_index + 1}",
                symbol=str(symbol),
                role=f"{role}_symbol",
            )
            for symbol_index, symbol in enumerate(str(code))
        )
        letters.append(
            MorseLetterSpec(
                item_id=f"{item_id}_letter_{letter_index + 1}",
                letter=str(letter),
                symbols=symbols,
                role=f"{role}_letter",
            )
        )
    return MorseWordSpec(
        item_id=str(item_id),
        word=str(word).lower().strip(),
        letters=tuple(letters),
        label=str(label),
        role=str(role),
        marked=bool(marked),
    )


def build_code_source_word_choice(
    *,
    rng: Any,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    labels: Sequence[str],
) -> MorseWordChoiceDataset:
    """Build a choice dataset where the source is a Morse word code."""

    binding = sample_labeled_word_options(rng=rng, params=params, gen_defaults=gen_defaults, labels=labels)
    source_code = make_morse_word_spec(
        item_id="source_code",
        word=str(binding.target_word),
        label="",
        role="source_code",
        marked=True,
    )
    option_word_map = {str(label): str(word) for label, word in binding.words_by_label}
    target_codes = morse_codes_for_word(str(binding.target_word))
    return MorseWordChoiceDataset(
        scene_variant=str(scene_variant),
        answer_value=str(binding.correct_label),
        target_answer_support=tuple(binding.labels),
        annotation_item_ids=("source_code", f"option_{binding.correct_label}"),
        source_word=str(binding.target_word),
        source_code=source_code,
        word_options=tuple(binding.words_by_label),
        code_options=(),
        metadata={
            "target_word": str(binding.target_word),
            "word_length": int(len(binding.target_word)),
            "target_codes": [str(code) for code in target_codes],
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


def build_word_source_code_choice(
    *,
    rng: Any,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    labels: Sequence[str],
) -> MorseWordChoiceDataset:
    """Build a choice dataset where the source is a visible text word."""

    binding = sample_labeled_word_options(rng=rng, params=params, gen_defaults=gen_defaults, labels=labels)
    code_options = tuple(
        make_morse_word_spec(
            item_id=f"option_{label}",
            word=str(word),
            label=str(label),
            role="correct_option" if str(label) == str(binding.correct_label) else "distractor_option",
            marked=False,
        )
        for label, word in binding.words_by_label
    )
    option_words = {str(label): str(word) for label, word in binding.words_by_label}
    option_codes = {
        str(label): [str(code) for code in morse_codes_for_word(str(word))]
        for label, word in binding.words_by_label
    }
    source_codes = morse_codes_for_word(str(binding.target_word))
    return MorseWordChoiceDataset(
        scene_variant=str(scene_variant),
        answer_value=str(binding.correct_label),
        target_answer_support=tuple(binding.labels),
        annotation_item_ids=("source_word", f"option_{binding.correct_label}"),
        source_word=str(binding.target_word),
        source_code=None,
        word_options=(),
        code_options=tuple(code_options),
        metadata={
            "source_word": str(binding.target_word),
            "word_length": int(len(binding.target_word)),
            "source_codes": [str(code) for code in source_codes],
            "correct_option_label": str(binding.correct_label),
            "correct_option_id": f"option_{binding.correct_label}",
            "option_words": dict(option_words),
            "option_codes": dict(option_codes),
            "shared_prefix": str(binding.shared_prefix),
            "shared_prefix_length": int(binding.shared_prefix_length),
            "prefix_candidate_count": int(binding.prefix_candidate_count),
            "word_option_sampling_strategy": str(binding.word_option_sampling_strategy),
        },
        scene_variant_probabilities={str(key): float(value) for key, value in scene_variant_probabilities.items()},
    )


__all__ = [
    "MORSE_WORD_BANK_BY_LENGTH",
    "MorseWordChoiceDataset",
    "MorseWordOptionBinding",
    "available_word_lengths",
    "bind_words_to_option_labels",
    "build_code_source_word_choice",
    "build_with_retries",
    "build_word_source_code_choice",
    "make_morse_word_spec",
    "resolve_morse_scene_variant",
    "resolve_word_length_bounds",
    "resolve_word_option_count",
    "sample_labeled_word_options",
    "sample_word_option_set",
    "validate_word_options",
]
