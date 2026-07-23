"""Tests for shared finite-support sampling helpers."""

from __future__ import annotations

from collections import Counter

from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice


def test_integer_choice_balances_uniform_support_with_sample_cursor() -> None:
    """Review cursors cover balanced integer supports evenly."""

    counts: Counter[int] = Counter()
    for cursor in range(40):
        selected, probabilities = resolve_integer_choice(
            instance_seed=123,
            params={"_sample_cursor": int(cursor)},
            gen_defaults={"balanced_target_answer_sampling": True},
            support_key="target_answer_support",
            explicit_key="target_answer",
            fallback_support=(0, 1, 2, 3),
            namespace="tests.integer_choice.balance",
            balanced_flag_key="balanced_target_answer_sampling",
            namespace_support_permutation=True,
        )
        counts[int(selected)] += 1
        assert probabilities[str(selected)] == 1.0

    assert counts == {0: 10, 1: 10, 2: 10, 3: 10}


def test_integer_choice_uses_seeded_rng_without_sample_cursor() -> None:
    """Normal generation remains deterministic seeded RNG sampling."""

    first, first_probabilities = resolve_integer_choice(
        instance_seed=456,
        params={},
        gen_defaults={"balanced_target_answer_sampling": True},
        support_key="target_answer_support",
        explicit_key="target_answer",
        fallback_support=(1, 2, 3),
        namespace="tests.integer_choice.rng",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    second, second_probabilities = resolve_integer_choice(
        instance_seed=456,
        params={},
        gen_defaults={"balanced_target_answer_sampling": True},
        support_key="target_answer_support",
        explicit_key="target_answer",
        fallback_support=(1, 2, 3),
        namespace="tests.integer_choice.rng",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )

    assert first == second
    assert first_probabilities == second_probabilities
    assert set(first_probabilities) == {"1", "2", "3"}
