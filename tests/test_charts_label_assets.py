"""Tests for chart-domain label asset adapters."""

from __future__ import annotations

import random

from trace_tasks.tasks.charts.shared.label_assets import (
    CHART_CATEGORY_LABEL_BUCKET_MANIFESTS,
    CHART_PANEL_LABEL_BUCKET_MANIFESTS,
    resolve_chart_category_labels,
    resolve_chart_compact_axis_labels,
    resolve_chart_entity_labels,
    resolve_chart_panel_labels,
    resolve_chart_text_labels,
    validate_chart_label_namespaces,
)
from trace_tasks.tasks.shared.name_assets import load_label_manifest


def test_chart_category_labels_use_category_manifests() -> None:
    resolved = resolve_chart_category_labels(random.Random(17), count=6, max_chars=14)

    assert len(resolved.labels) == 6
    assert len(set(resolved.labels)) == 6
    assert resolved.label_pool_kind == "category"
    assert resolved.label_bucket in CHART_CATEGORY_LABEL_BUCKET_MANIFESTS
    assert resolved.label_manifest == CHART_CATEGORY_LABEL_BUCKET_MANIFESTS[resolved.label_bucket]
    assert all(label.isascii() for label in resolved.labels)
    assert all(len(label.replace(" ", "")) <= 14 for label in resolved.labels)


def test_chart_entity_labels_can_force_one_shared_manifest_bucket() -> None:
    resolved = resolve_chart_entity_labels(
        random.Random(23),
        count=5,
        max_chars=10,
        allow_spaces=False,
        bucket_weights={"places_countries": 1.0},
    )

    countries = set(
        load_label_manifest(
            "places/countries_natural_earth.txt",
            min_chars=2,
            max_chars=10,
            allow_spaces=False,
            allow_punctuation=False,
            compact_length=True,
        )
    )
    assert resolved.label_bucket == "places_countries"
    assert set(resolved.labels).issubset(countries)


def test_chart_text_labels_still_support_compact_letters() -> None:
    resolved = resolve_chart_text_labels(random.Random(31), count=12, label_variant="letters")

    assert len(resolved.labels) == 12
    assert len(set(resolved.labels)) == 12
    assert resolved.label_source_kind == "letters"
    assert all(len(label) == 1 and label.isalpha() for label in resolved.labels)


def test_chart_compact_axis_labels_use_large_synthetic_pool() -> None:
    resolved = resolve_chart_compact_axis_labels(random.Random(37), count=40)

    assert len(resolved.labels) == 40
    assert len(set(resolved.labels)) == 40
    assert resolved.label_source_kind == "synthetic_compact_id"
    assert resolved.label_bucket == "dense_axis_compact_id"
    assert all(2 <= len(label) <= 4 for label in resolved.labels)
    assert all(label.isascii() and label.isalnum() for label in resolved.labels)


def test_chart_panel_labels_use_technical_manifest_and_reserved_filter() -> None:
    resolved = resolve_chart_panel_labels(
        random.Random(41),
        count=8,
        min_chars=3,
        max_chars=14,
        allow_spaces=False,
        variant_weights={"technical_topics": 1.0},
        reserved_labels=("Ablation", "Latency", "Recall"),
    )

    technical_terms = set(
        load_label_manifest(
            CHART_PANEL_LABEL_BUCKET_MANIFESTS["technical_topics"],
            min_chars=3,
            max_chars=14,
            allow_spaces=False,
            allow_punctuation=False,
            compact_length=True,
        )
    )
    assert resolved.label_pool_kind == "panel"
    assert resolved.label_bucket == "technical_topics"
    assert resolved.label_manifest == CHART_PANEL_LABEL_BUCKET_MANIFESTS["technical_topics"]
    assert len(resolved.labels) == 8
    assert len(set(resolved.labels)) == 8
    assert set(resolved.labels).issubset(technical_terms)
    assert {"Ablation", "Latency", "Recall"}.isdisjoint(set(resolved.labels))


def test_chart_panel_labels_can_emit_ordered_temporal_sequence() -> None:
    resolved = resolve_chart_panel_labels(
        random.Random(43),
        count=5,
        min_chars=4,
        max_chars=6,
        allow_spaces=False,
        variant_weights={"temporal_sequence": 1.0},
    )

    assert resolved.label_bucket == "temporal_sequence"
    assert resolved.label_source_kind == "synthetic_temporal_sequence"
    assert len(resolved.labels) == 5
    assert len(set(resolved.labels)) == 5


def test_chart_panel_label_namespace_validator_rejects_collisions() -> None:
    validate_chart_label_namespaces(
        panel_labels=("Alpha", "Beta"),
        other_label_groups={"series": ("Gamma", "Delta")},
        context="unit_test",
    )

    try:
        validate_chart_label_namespaces(
            panel_labels=("Alpha", "Beta"),
            other_label_groups={"series": (" alpha ",)},
            context="unit_test",
        )
    except ValueError as exc:
        assert "collide" in str(exc)
    else:
        raise AssertionError("expected panel-label collision to raise")
