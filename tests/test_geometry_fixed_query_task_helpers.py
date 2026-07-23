"""Tests for geometry fixed-query probability helpers."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.shared.fixed_query import (
    geometry_probability_map,
    geometry_query_ids_for_task,
    geometry_selected_probability_map,
    select_indexed_geometry_query_id,
)


def test_geometry_probability_map_preserves_ordered_support() -> None:
    assert geometry_probability_map(("first", "second")) == {
        "first": 0.5,
        "second": 0.5,
    }


def test_geometry_probability_map_can_sort_and_deduplicate_keys() -> None:
    assert geometry_probability_map(("b", "a", "b"), sort_unique=True) == {
        "a": 0.5,
        "b": 0.5,
    }


def test_geometry_selected_probability_map_supports_value_formatters() -> None:
    probabilities = geometry_selected_probability_map(
        (1.0, 2.0, 3.0),
        2.0,
        key_fn=lambda value: str(int(value)),
        is_selected=lambda value, selected: int(value) == int(selected),
    )

    assert probabilities == {"1": 0.0, "2": 1.0, "3": 0.0}


def test_geometry_selected_probability_map_supports_rounded_keys() -> None:
    probabilities = geometry_selected_probability_map(
        (1.24, 2.45),
        2.46,
        key_fn=lambda value: f"{float(value):.1f}",
    )

    assert probabilities == {"1.2": 0.0, "2.5": 1.0}


def test_geometry_query_ids_for_task_returns_validated_tuple() -> None:
    assert geometry_query_ids_for_task(
        "task_geometry__example__value",
        {"task_geometry__example__value": ("first", "second")},
    ) == ("first", "second")


def test_geometry_query_ids_for_task_rejects_missing_task() -> None:
    with pytest.raises(ValueError, match="unsupported example context task_id"):
        geometry_query_ids_for_task(
            "task_geometry__example__missing",
            {"task_geometry__example__value": ("first",)},
            context="example context",
        )


def test_geometry_query_ids_for_task_rejects_duplicate_query_ids() -> None:
    with pytest.raises(ValueError, match="must not contain duplicate"):
        geometry_query_ids_for_task(
            "task_geometry__example__value",
            {"task_geometry__example__value": ("first", "first")},
        )


def test_select_indexed_geometry_query_id_explicit_query_is_one_hot() -> None:
    selected, probabilities = select_indexed_geometry_query_id(
        {"query_id": "second"},
        query_ids=("first", "second"),
        task_id="task_geometry__example__value",
        instance_seed=17,
    )

    assert selected == "second"
    assert probabilities == {"first": 0.0, "second": 1.0}


def test_select_indexed_geometry_query_id_samples_default_by_default() -> None:
    selected, probabilities = select_indexed_geometry_query_id(
        {"query_id": "default"},
        query_ids=("first", "second"),
        task_id="task_geometry__example__value",
        instance_seed=17,
    )

    assert selected in {"first", "second"}
    assert probabilities == {"first": 0.5, "second": 0.5}


def test_select_indexed_geometry_query_id_can_reject_default_explicit_query() -> None:
    with pytest.raises(ValueError, match="query_id='default' is not valid"):
        select_indexed_geometry_query_id(
            {"query_id": "default"},
            query_ids=("first", "second"),
            task_id="task_geometry__example__value",
            instance_seed=17,
            default_means_sample=False,
        )


def test_select_indexed_geometry_query_id_rejects_invalid_explicit_query() -> None:
    with pytest.raises(ValueError, match="query_id='third' is not valid"):
        select_indexed_geometry_query_id(
            {"query_id": "third"},
            query_ids=("first", "second"),
            task_id="task_geometry__example__value",
            instance_seed=17,
        )
