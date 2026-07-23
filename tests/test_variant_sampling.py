"""Tests for shared variant-sampling helpers."""

from __future__ import annotations

from collections import Counter
import random

from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_compatible_scene_query_ids


def test_balanced_variant_sampling_ignores_zero_weight_variants() -> None:
    outputs = [
        apply_balanced_variant_sampling(
            instance_seed=index,
            params={},
            gen_defaults={"balanced_variant_sampling": True},
            selected_variant="missing_resistor_value",
            variant_probabilities={
                "total_resistance": 0.0,
                "missing_resistor_value": 1.0,
            },
            supported_variants=["total_resistance", "missing_resistor_value"],
        )
        for index in range(4)
    ]

    assert outputs == ["missing_resistor_value"] * 4


def test_balanced_variant_sampling_uses_seeded_public_sampler_for_positive_uniform_variants() -> None:
    outputs = [
        apply_balanced_variant_sampling(
            instance_seed=index,
            params={},
            gen_defaults={"balanced_variant_sampling": True},
            selected_variant="b",
            variant_probabilities={"a": 0.0, "b": 0.5, "c": 0.5},
            supported_variants=["a", "b", "c"],
        )
        for index in range(4)
    ]

    assert set(outputs).issubset({"b", "c"})
    assert "a" not in outputs
    assert apply_balanced_variant_sampling(
        instance_seed=2,
        params={},
        gen_defaults={"balanced_variant_sampling": True},
        selected_variant="b",
        variant_probabilities={"a": 0.0, "b": 0.5, "c": 0.5},
        supported_variants=["a", "b", "c"],
    ) == outputs[2]


def test_compatible_scene_query_sampling_uses_separate_seeded_namespaces() -> None:
    combos: Counter[tuple[str, str]] = Counter()
    for index in range(30):
        scene, _, query, _ = resolve_compatible_scene_query_ids(
            random.Random(index),
            instance_seed=index,
            params={},
            gen_defaults={
                "balanced_scene_variant_sampling": True,
                "balanced_query_id_sampling": True,
            },
            supported_scene_variants=("scene_a", "scene_b", "scene_c"),
            supported_query_ids=("query_a", "query_b", "query_c"),
            compatibility={
                "scene_a": ("query_a", "query_b", "query_c"),
                "scene_b": ("query_a", "query_b", "query_c"),
                "scene_c": ("query_a", "query_b", "query_c"),
            },
            scene_sampling_namespace="test.scene",
            query_sampling_namespace="test.query",
            decouple_scene_sampling=True,
        )
        combos[(scene, query)] += 1

    assert len(combos) > 3
    assert {scene for scene, _ in combos} == {"scene_a", "scene_b", "scene_c"}
    assert {query for _, query in combos} == {"query_a", "query_b", "query_c"}
