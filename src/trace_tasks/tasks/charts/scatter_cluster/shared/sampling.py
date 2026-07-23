"""Sampling helpers for scatter-cluster chart scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.label_assets import resolve_chart_entity_labels
from trace_tasks.tasks.shared.config_defaults import group_default

from .defaults import GEN_DEFAULTS, gen_int
from .state import OPTION_LABELS, SCENE_NAMESPACE, ScatterClusterInputs


def sample_cluster_labels(*, cluster_count: int, instance_seed: int) -> tuple[str, ...]:
    label_rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.cluster_labels")
    labels = resolve_chart_entity_labels(
        label_rng,
        count=int(cluster_count),
        min_chars=2,
        max_chars=6,
        allow_spaces=False,
    ).labels
    return tuple(str(label) for label in labels)


def target_answer_label(params: Mapping[str, Any], *, instance_seed: int, labels: Sequence[str]) -> str:
    sampling_index = params.get("_sample_cursor")
    occurrence = int(sampling_index) if sampling_index is not None else abs(int(instance_seed))
    return str(labels[int(occurrence) % max(1, len(labels))])


def sample_cluster_inputs(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
) -> ScatterClusterInputs:
    cluster_min = gen_int(params, "cluster_count_min", 5)
    cluster_max = gen_int(params, "cluster_count_max", 8)
    cluster_count_rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.cluster_count")
    cluster_count = int(cluster_count_rng.randint(int(cluster_min), int(cluster_max)))
    cluster_count = max(4, min(8, int(cluster_count)))
    labels = sample_cluster_labels(cluster_count=int(cluster_count), instance_seed=int(instance_seed))
    points_min = gen_int(params, "points_per_cluster_min", 8)
    points_max = gen_int(params, "points_per_cluster_max", 12)
    point_count_rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.point_count")
    points_per_cluster = int(point_count_rng.randint(int(points_min), int(points_max)))
    answer_label = target_answer_label(params, instance_seed=int(instance_seed), labels=labels)
    return ScatterClusterInputs(
        cluster_count=int(cluster_count),
        labels=tuple(str(label) for label in labels),
        points_per_cluster=int(points_per_cluster),
        answer_label=str(answer_label),
    )


def option_count_support(params: Mapping[str, Any]) -> tuple[int, ...]:
    explicit = params.get("option_count")
    if explicit is not None:
        count = int(explicit)
        if count not in {4, 6}:
            raise ValueError("option_count must be either 4 or 6 for scatter centroid option tasks")
        return (int(count),)
    raw_support = params.get(
        "centroid_option_count_support",
        group_default(GEN_DEFAULTS, "centroid_option_count_support", (4, 6)),
    )
    if isinstance(raw_support, Sequence) and not isinstance(raw_support, (str, bytes)):
        support = tuple(sorted({int(value) for value in raw_support if int(value) in {4, 6}}))
    else:
        support = ()
    if not support:
        raise ValueError("centroid_option_count_support must contain 4 and/or 6")
    return support


def target_option_count(params: Mapping[str, Any], *, instance_seed: int) -> int:
    support = option_count_support(params)
    return int(
        uniform_choice(
            spawn_rng(
                int(instance_seed),
                f"{SCENE_NAMESPACE}.centroid_option.option_count",
            ),
            support,
            sort_keys=True,
        )
    )


def option_labels_for_count(option_count: int) -> tuple[str, ...]:
    if int(option_count) not in {4, 6}:
        raise ValueError("option_count must be either 4 or 6")
    return tuple(OPTION_LABELS[: int(option_count)])


def target_option_label(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    cluster_count: int,
    option_labels: Sequence[str],
) -> str:
    return str(
        uniform_choice(
            spawn_rng(
                int(instance_seed),
                f"{SCENE_NAMESPACE}.centroid_option.label",
            ),
            tuple(str(label) for label in option_labels),
        )
    )
