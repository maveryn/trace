"""Tests for shared chart balanced-sampling helpers."""

from __future__ import annotations

from trace_tasks.tasks.charts.shared.balanced_sampling import (
    balanced_int_from_support,
    decouple_sample_cursor_for_axis_lengths,
    support_sampling_params_for_uniform_query_cycle,
    uses_uniform_query_id_cycle,
)
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index


def test_uniform_query_cycle_detection_respects_overrides_and_weights() -> None:
    probabilities = {"a": 0.5, "b": 0.5}
    supported = ("a", "b")

    assert uses_uniform_query_id_cycle(
        {},
        gen_defaults={"balanced_query_id_sampling": True},
        query_id_probabilities=probabilities,
        supported_query_ids=supported,
    )
    assert not uses_uniform_query_id_cycle(
        {"query_id": "a"},
        gen_defaults={"balanced_query_id_sampling": True},
        query_id_probabilities=probabilities,
        supported_query_ids=supported,
    )
    assert not uses_uniform_query_id_cycle(
        {},
        gen_defaults={"balanced_query_id_sampling": False},
        query_id_probabilities=probabilities,
        supported_query_ids=supported,
    )
    assert not uses_uniform_query_id_cycle(
        {},
        gen_defaults={"balanced_query_id_sampling": True},
        query_id_probabilities={"a": 1.0, "b": 0.0},
        supported_query_ids=supported,
    )


def test_support_sampling_params_decouples_sample_cursor_only_for_uniform_query_cycle() -> None:
    params = {"_sample_cursor": 9}
    probabilities = {"a": 0.5, "b": 0.5}
    supported = ("a", "b")

    assert support_sampling_params_for_uniform_query_cycle(
        params,
        gen_defaults={"balanced_query_id_sampling": True},
        query_id_probabilities=probabilities,
        supported_query_ids=supported,
    )["_sample_cursor"] == 4
    assert support_sampling_params_for_uniform_query_cycle(
        {**params, "query_id": "a"},
        gen_defaults={"balanced_query_id_sampling": True},
        query_id_probabilities=probabilities,
        supported_query_ids=supported,
    )["_sample_cursor"] == 9


def test_decouple_sample_cursor_for_axis_lengths_preserves_explicit_policy() -> None:
    axes = (
        (2, ("query_id", "query_id_weights")),
        (3, ("statistic_kind", "statistic_kind_weights")),
    )

    assert decouple_sample_cursor_for_axis_lengths({"_sample_cursor": 23}, axes=axes)["_sample_cursor"] == 3
    assert decouple_sample_cursor_for_axis_lengths({"_sample_cursor": 23, "query_id": None}, axes=axes)[
        "_sample_cursor"
    ] == 7
    assert decouple_sample_cursor_for_axis_lengths(
        {"_sample_cursor": 23, "query_id": None},
        axes=axes,
        explicit_policy="non_null",
    )["_sample_cursor"] == 3
    assert decouple_sample_cursor_for_axis_lengths(
        {"_sample_cursor": -23},
        axes=axes,
        use_abs=True,
    )["_sample_cursor"] == 3


def test_balanced_int_from_support_preserves_ordered_support_selection() -> None:
    support = (30, 10, 20)
    namespace = "charts.test.support"
    seed = 987
    expected_index = resolve_selection_index(params={}, instance_seed=seed, namespace=namespace) % len(support)

    assert balanced_int_from_support(support, params={}, instance_seed=seed, namespace=namespace) == support[expected_index]
