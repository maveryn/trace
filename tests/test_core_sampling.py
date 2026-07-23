from __future__ import annotations

import random

import pytest

from trace_tasks.core.sampling import (
    integer_range_choice,
    sample_without_replacement,
    shuffled_support,
    support_probability_map,
    uniform_choice,
    uniform_choice_with_probabilities,
    weighted_support_choice,
)


def test_uniform_choice_with_probabilities_uses_equal_weights() -> None:
    rng = random.Random(17)

    selected, probabilities = uniform_choice_with_probabilities(
        rng,
        ["left", "right", "balanced", "not_determined"],
    )

    assert selected in probabilities
    assert probabilities == {
        "left": 0.25,
        "right": 0.25,
        "balanced": 0.25,
        "not_determined": 0.25,
    }


def test_uniform_choice_is_deterministic_for_seeded_rng() -> None:
    support = ("red", "green", "blue")

    first = [uniform_choice(random.Random(31), support) for _ in range(5)]
    second = [uniform_choice(random.Random(31), support) for _ in range(5)]

    assert first == second


def test_uniform_choice_preserves_typed_values() -> None:
    rng = random.Random(2)

    selected = uniform_choice(rng, [10, 20, 30])

    assert selected in {10, 20, 30}
    assert isinstance(selected, int)


def test_uniform_choice_rejects_empty_support() -> None:
    with pytest.raises(ValueError, match="empty support"):
        uniform_choice(random.Random(0), [])


def test_uniform_choice_rejects_duplicate_string_keys() -> None:
    with pytest.raises(ValueError, match="unique string keys"):
        uniform_choice(random.Random(0), [1, "1"])


def test_support_probability_map_can_be_one_hot() -> None:
    assert support_probability_map(["A", "B", "C"], selected="B") == {
        "A": 0.0,
        "B": 1.0,
        "C": 0.0,
    }


def test_weighted_support_choice_preserves_value_type() -> None:
    selected, probabilities = weighted_support_choice(
        random.Random(4),
        [1, 2, 3],
        weights={"1": 0.0, "2": 0.0, "3": 2.0},
    )

    assert selected == 3
    assert isinstance(selected, int)
    assert probabilities == {"3": 1.0}


def test_integer_range_choice_samples_from_inclusive_range() -> None:
    selected, probabilities = integer_range_choice(random.Random(7), 2, 4)

    assert selected in {2, 3, 4}
    assert probabilities == {"2": 1 / 3, "3": 1 / 3, "4": 1 / 3}


def test_sample_without_replacement_uses_unique_values() -> None:
    selected = sample_without_replacement(random.Random(8), ["a", "b", "c"], 2)

    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert set(selected).issubset({"a", "b", "c"})


def test_shuffled_support_returns_all_values() -> None:
    shuffled = shuffled_support(random.Random(9), ("a", "b", "c"))

    assert set(shuffled) == {"a", "b", "c"}
