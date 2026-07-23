"""Scene-neutral word-option sampling helpers for symbolic readout tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PrefixWordOptionSet:
    """Same-length word options sharing a non-empty prefix."""

    target_word: str
    option_words: tuple[str, ...]
    shared_prefix: str
    shared_prefix_length: int
    prefix_candidate_count: int
    sampling_strategy: str


def longest_common_prefix(words: Sequence[str]) -> str:
    """Return the longest shared prefix across non-empty words."""

    normalized = tuple(str(word).lower().strip() for word in words)
    if not normalized:
        return ""
    prefix = normalized[0]
    for word in normalized[1:]:
        while prefix and not str(word).startswith(prefix):
            prefix = prefix[:-1]
    return str(prefix)


def validate_prefix_word_options(
    words: Sequence[str],
    *,
    option_count: int,
    min_shared_prefix_length: int = 1,
) -> PrefixWordOptionSet:
    """Validate caller-supplied word options and report their shared prefix."""

    normalized = tuple(str(word).lower().strip() for word in words)
    if len(normalized) != int(option_count):
        raise ValueError(f"expected exactly {option_count} word options")
    if len(set(normalized)) != len(normalized):
        raise ValueError("word options must be unique")
    if len({len(word) for word in normalized}) != 1:
        raise ValueError("word options must all have the same length")
    if any((not word.isalpha()) for word in normalized):
        raise ValueError("word options must contain alphabetic letters only")
    prefix = longest_common_prefix(normalized)
    if len(prefix) < int(min_shared_prefix_length):
        raise ValueError(f"word options must share at least a {min_shared_prefix_length}-letter prefix")
    return PrefixWordOptionSet(
        target_word=str(normalized[0]),
        option_words=tuple(normalized),
        shared_prefix=str(prefix),
        shared_prefix_length=int(len(prefix)),
        prefix_candidate_count=int(len(normalized)),
        sampling_strategy="caller_supplied_shared_prefix",
    )


def _prefix_length_weights(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    max_prefix_length: int,
) -> dict[int, float]:
    raw = params.get("word_option_shared_prefix_length_weights")
    if raw is None:
        raw = gen_defaults.get("word_option_shared_prefix_length_weights", {1: 1.0, 2: 1.0, 3: 0.5})
    if not isinstance(raw, Mapping):
        raise ValueError("word_option_shared_prefix_length_weights must be a mapping")
    weights: dict[int, float] = {}
    for key, value in raw.items():
        prefix_length = int(key)
        if 1 <= int(prefix_length) <= int(max_prefix_length):
            weight = float(value)
            if weight > 0.0:
                weights[int(prefix_length)] = float(weight)
    if not weights:
        raise ValueError("no positive word option shared-prefix weights are available")
    return dict(weights)


def _weighted_choice(rng: Any, weighted_items: Sequence[tuple[int, float]]) -> int:
    total = sum(float(weight) for _item, weight in weighted_items)
    if total <= 0.0:
        raise ValueError("weighted choice requires positive total weight")
    threshold = float(rng.random()) * float(total)
    cursor = 0.0
    for item, weight in weighted_items:
        cursor += float(weight)
        if threshold <= cursor:
            return int(item)
    return int(weighted_items[-1][0])


def _candidate_groups_by_prefix(
    word_bank: Sequence[str],
    *,
    prefix_length: int,
    option_count: int,
) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for word in word_bank:
        normalized = str(word).lower().strip()
        if len(normalized) >= int(prefix_length) and normalized.isalpha():
            groups.setdefault(normalized[: int(prefix_length)], []).append(normalized)
    return {
        str(prefix): tuple(words)
        for prefix, words in groups.items()
        if len(set(words)) >= int(option_count)
    }


def sample_prefix_word_option_set(
    rng: Any,
    *,
    word_bank_by_length: Mapping[int, Sequence[str]],
    min_length: int,
    max_length: int,
    option_count: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    target_word: str | None = None,
    word_length: int | None = None,
    min_shared_prefix_length: int = 1,
) -> PrefixWordOptionSet:
    """Sample same-length word options that share a visible encoding prefix."""

    if int(option_count) < 2:
        raise ValueError("prefix word option sets require at least two options")
    if int(min_shared_prefix_length) < 1:
        raise ValueError("min_shared_prefix_length must be positive")

    requested_prefix_length = params.get("word_option_shared_prefix_length")
    if requested_prefix_length is not None:
        requested_prefix_length = int(requested_prefix_length)
        if int(requested_prefix_length) < int(min_shared_prefix_length):
            raise ValueError("requested shared prefix length is below the minimum")

    if target_word is not None:
        normalized_target = str(target_word).lower().strip()
        if not normalized_target.isalpha():
            raise ValueError("target_word must contain alphabetic letters only")
        if word_length is not None and len(normalized_target) != int(word_length):
            raise ValueError("target_word length does not match requested word_length")
        if not int(min_length) <= len(normalized_target) <= int(max_length):
            raise ValueError("target_word length is outside configured word support")
        word_bank = tuple(str(word).lower().strip() for word in word_bank_by_length.get(len(normalized_target), ()))
        viable: list[tuple[int, float, tuple[str, ...]]] = []
        weights = _prefix_length_weights(
            params,
            gen_defaults,
            max_prefix_length=len(normalized_target),
        )
        for prefix_length, weight in weights.items():
            if requested_prefix_length is not None and int(prefix_length) != int(requested_prefix_length):
                continue
            prefix = normalized_target[: int(prefix_length)]
            candidates = tuple(word for word in word_bank if word.startswith(prefix))
            if normalized_target not in candidates:
                candidates = (normalized_target, *candidates)
            candidates = tuple(dict.fromkeys(candidates))
            if len(candidates) >= int(option_count):
                viable.append((int(prefix_length), float(weight), candidates))
        if not viable:
            raise ValueError("not enough same-prefix word candidates for target_word")
        selected_prefix_length = _weighted_choice(rng, [(length, weight) for length, weight, _candidates in viable])
        selected_candidates = next(candidates for length, _weight, candidates in viable if int(length) == int(selected_prefix_length))
        distractors = [word for word in selected_candidates if word != normalized_target]
        selected = (normalized_target, *tuple(str(word) for word in rng.sample(distractors, int(option_count) - 1)))
        return PrefixWordOptionSet(
            target_word=str(normalized_target),
            option_words=tuple(selected),
            shared_prefix=str(normalized_target[: int(selected_prefix_length)]),
            shared_prefix_length=int(selected_prefix_length),
            prefix_candidate_count=int(len(selected_candidates)),
            sampling_strategy="same_prefix_target_word",
        )

    length_min = int(word_length) if word_length is not None else int(min_length)
    length_max = int(word_length) if word_length is not None else int(max_length)
    if int(length_min) > int(length_max):
        raise ValueError("invalid word length range")

    viable_groups: list[tuple[int, float, str, tuple[str, ...]]] = []
    max_prefix_length = int(length_max)
    weights = _prefix_length_weights(params, gen_defaults, max_prefix_length=int(max_prefix_length))
    for length in range(int(length_min), int(length_max) + 1):
        word_bank = tuple(str(word).lower().strip() for word in word_bank_by_length.get(int(length), ()))
        if not word_bank:
            continue
        for prefix_length, weight in weights.items():
            if requested_prefix_length is not None and int(prefix_length) != int(requested_prefix_length):
                continue
            if int(prefix_length) > int(length):
                continue
            groups = _candidate_groups_by_prefix(
                word_bank,
                prefix_length=int(prefix_length),
                option_count=int(option_count),
            )
            for prefix, candidates in groups.items():
                viable_groups.append((int(prefix_length), float(weight), str(prefix), tuple(candidates)))
    if not viable_groups:
        raise ValueError("no same-prefix word option groups are available")

    selected_prefix_length = _weighted_choice(rng, [(length, weight) for length, weight, _prefix, _candidates in viable_groups])
    matching_groups = [
        (prefix, candidates)
        for length, _weight, prefix, candidates in viable_groups
        if int(length) == int(selected_prefix_length)
    ]
    selected_prefix, selected_candidates = rng.choice(matching_groups)
    selected = tuple(str(word) for word in rng.sample(list(selected_candidates), int(option_count)))
    return PrefixWordOptionSet(
        target_word=str(selected[0]),
        option_words=tuple(selected),
        shared_prefix=str(selected_prefix),
        shared_prefix_length=int(selected_prefix_length),
        prefix_candidate_count=int(len(selected_candidates)),
        sampling_strategy="same_prefix_random_word",
    )


__all__ = [
    "PrefixWordOptionSet",
    "longest_common_prefix",
    "sample_prefix_word_option_set",
    "validate_prefix_word_options",
]
