"""Regression tests for canonical query_id and legacy query_variant inputs."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.shared.fixed_query import (
    DEFAULT_QUERY_ID,
    force_query_id_params,
    merged_query_params,
    normalize_query_id_params,
    resolve_task_query_id_param,
    select_task_query_id,
    strip_query_id_params,
)
from trace_tasks.tasks.registry import _ensure_supported_query_ids
from trace_tasks.tasks.three_d.shared.task_support import resolve_axis_variant


def test_normalize_query_id_params_accepts_legacy_query_variant() -> None:
    params = normalize_query_id_params(
        {
            "query_variant": "second",
            "query_variant_weights": {"first": 0.0, "second": 1.0},
        }
    )

    assert params["query_id"] == "second"
    assert params["query_id_weights"] == {"first": 0.0, "second": 1.0}
    assert "query_variant" not in params
    assert "query_variant_weights" not in params


def test_normalize_query_id_params_rejects_conflicting_aliases() -> None:
    with pytest.raises(ValueError, match="query_id conflicts with query_variant"):
        normalize_query_id_params({"query_id": "first", "query_variant": "second"})


def test_force_query_id_params_checks_legacy_alias_conflicts() -> None:
    forced = force_query_id_params(
        {"query_variant": "second", "query_variant_weights": {"second": 1.0}},
        query_id="second",
    )

    assert forced["query_id"] == "second"
    assert forced["query_id_weights"] == {"second": 1.0}
    assert "query_variant" not in forced
    assert "query_variant_weights" not in forced
    with pytest.raises(ValueError, match="public task query_id must match"):
        force_query_id_params({"query_variant": "first"}, query_id="second")


def test_resolve_task_query_id_param_validates_supported_ids_and_strips_aliases() -> None:
    assert DEFAULT_QUERY_ID == "single"
    assert (
        resolve_task_query_id_param(
            {},
            supported_query_ids=("single",),
            default_query_id="single",
            task_id="task_test__scene__fixed",
        )
        == "single"
    )
    assert (
        resolve_task_query_id_param(
            {"query_variant": "default"},
            supported_query_ids=("single",),
            default_query_id="single",
            task_id="task_test__scene__fixed",
        )
        == "single"
    )
    assert (
        resolve_task_query_id_param(
            {"query_id": "right"},
            supported_query_ids=("left", "right"),
            default_query_id="left",
            task_id="task_test__scene__multi",
        )
        == "right"
    )
    with pytest.raises(ValueError, match="unsupported query_id"):
        resolve_task_query_id_param(
            {"query_id": "other"},
            supported_query_ids=("single",),
            default_query_id="single",
            task_id="task_test__scene__fixed",
        )

    assert strip_query_id_params({"query_id": "default", "query_variant": "default", "keep": 1}) == {"keep": 1}


def test_select_task_query_id_handles_explicit_query_and_strips_selector() -> None:
    selected, probabilities, task_params = select_task_query_id(
        instance_seed=123,
        params={"query_variant": "right", "keep": 7},
        supported_query_ids=("left", "right"),
        default_query_id="left",
        task_id="task_test__scene__multi",
    )

    assert selected == "right"
    assert probabilities == {"left": 0.0, "right": 1.0}
    assert task_params == {"keep": 7}


def test_select_task_query_id_uses_single_for_no_branch_tasks() -> None:
    selected, probabilities, task_params = select_task_query_id(
        instance_seed=123,
        params={"query_id": "single", "keep": 7},
        supported_query_ids=("single",),
        default_query_id="single",
        task_id="task_test__scene__fixed",
    )

    assert selected == "single"
    assert probabilities == {"single": 1.0}
    assert task_params == {"keep": 7}

    legacy_selected, legacy_probabilities, legacy_task_params = select_task_query_id(
        instance_seed=123,
        params={"query_id": "default", "keep": 7},
        supported_query_ids=("single",),
        default_query_id="single",
        task_id="task_test__scene__fixed",
    )

    assert legacy_selected == "single"
    assert legacy_probabilities == {"single": 1.0}
    assert legacy_task_params == {"keep": 7}


def test_registry_normalizes_one_declared_query_to_single() -> None:
    class SingleQueryTask:
        supported_query_ids = ("objective_named_query",)

    assert (
        _ensure_supported_query_ids(
            SingleQueryTask,
            task_id="task_test__scene__objective",
        )
        == ("single",)
    )
    assert SingleQueryTask.supported_query_ids == ("single",)
    assert SingleQueryTask._trace_internal_supported_query_ids == ("objective_named_query",)

    class MultiQueryTask:
        supported_query_ids = ("above_threshold", "below_threshold")

    assert (
        _ensure_supported_query_ids(
            MultiQueryTask,
            task_id="task_test__scene__objective",
        )
        == ("above_threshold", "below_threshold")
    )


def test_select_task_query_id_cycles_sample_cursor_and_reduces_for_lower_axes() -> None:
    selected, probabilities, task_params = select_task_query_id(
        instance_seed=123,
        params={"_sample_cursor": 5, "keep": "x"},
        supported_query_ids=("a", "b", "c"),
        default_query_id="a",
        task_id="task_test__scene__multi",
    )

    assert selected == "c"
    assert probabilities == {"a": 1.0 / 3.0, "b": 1.0 / 3.0, "c": 1.0 / 3.0}
    assert task_params == {"_sample_cursor": 1, "keep": "x"}


def test_select_task_query_id_random_fallback_is_uniform_and_deterministic() -> None:
    first = select_task_query_id(
        instance_seed=123,
        params={"keep": 1},
        supported_query_ids=("a", "b", "c"),
        default_query_id="a",
        task_id="task_test__scene__multi",
        namespace="unit.query",
    )
    second = select_task_query_id(
        instance_seed=123,
        params={"keep": 1},
        supported_query_ids=("a", "b", "c"),
        default_query_id="a",
        task_id="task_test__scene__multi",
        namespace="unit.query",
    )

    assert first == second
    assert first[0] in {"a", "b", "c"}
    assert first[1] == {"a": 1.0 / 3.0, "b": 1.0 / 3.0, "c": 1.0 / 3.0}
    assert first[2] == {"keep": 1}


def test_select_task_query_id_rejects_conflicting_aliases() -> None:
    with pytest.raises(ValueError, match="query_id conflicts with query_variant"):
        select_task_query_id(
            instance_seed=123,
            params={"query_id": "a", "query_variant": "b"},
            supported_query_ids=("a", "b"),
            default_query_id="a",
            task_id="task_test__scene__multi",
        )


def test_merged_chart_params_honor_legacy_query_variant_without_leaking_it() -> None:
    params = merged_query_params(
        {"query_variant": "second"},
        allowed_query_ids=("first", "second"),
    )

    assert params["query_id"] == "second"
    assert "query_variant" not in params
    with pytest.raises(ValueError, match="query_id conflicts with query_variant"):
        merged_query_params(
            {"query_id": "first", "query_variant": "second"},
            allowed_query_ids=("first", "second"),
        )


def test_three_d_axis_resolver_honors_legacy_query_variant() -> None:
    query_id, probabilities = resolve_axis_variant(
        {"query_variant": "second"},
        task_id="test_three_d_query_alias",
        gen_defaults={},
        instance_seed=123,
        supported_variants=("first", "second"),
        explicit_key="query_id",
        weights_key="query_id_weights",
        balance_flag_key="balanced_query_id_sampling",
        axis_namespace="query_id",
    )

    assert query_id == "second"
    assert probabilities == {"first": 0.0, "second": 1.0}
