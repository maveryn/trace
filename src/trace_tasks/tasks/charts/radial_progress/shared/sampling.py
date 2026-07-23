"""Scene-neutral sampling primitives for radial-progress charts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import cycle, islice
from typing import Any

from trace_tasks.core.sampling import shuffled_support, uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.label_assets import resolve_chart_entity_labels
from trace_tasks.tasks.charts.shared.labeled_chart_variants import resolve_chart_axis_variant

from .defaults import GEN_DEFAULTS, RENDER_DEFAULTS, generation_default, int_bounds, support_probability_map
from .state import ProgressFrame, ProgressItem, RGB, SCENE_NAMESPACE, SUPPORTED_SCENE_VARIANTS


def sample_scene_variant(params: Mapping[str, Any], *, instance_seed: int) -> tuple[str, dict[str, float]]:
    return resolve_chart_axis_variant(
        params=params,
        gen_defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        task_id="charts_radial_progress_scene",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def sample_item_count(params: Mapping[str, Any], *, instance_seed: int) -> tuple[int, dict[str, float]]:
    low, high = int_bounds(
        params,
        min_key="item_count_min",
        max_key="item_count_max",
        fallback_min=6,
        fallback_max=10,
    )
    support = list(range(int(low), int(high) + 1))
    selected = uniform_choice(
        spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.item_count"),
        tuple(support),
        sort_keys=True,
    )
    return int(selected), support_probability_map(support)


def sample_answer_count(
    params: Mapping[str, Any],
    *,
    item_count: int,
    instance_seed: int,
    namespace: str,
) -> tuple[int, list[int], dict[str, float]]:
    max_answer = min(
        int(item_count) - 1,
        int(params.get("answer_count_max", generation_default("answer_count_max", 5))),
    )
    min_answer = int(params.get("answer_count_min", generation_default("answer_count_min", 1)))
    support = list(range(max(1, int(min_answer)), max(1, int(max_answer)) + 1))
    selected = uniform_choice(
        spawn_rng(int(instance_seed), str(namespace)),
        tuple(support),
        sort_keys=True,
    )
    return int(selected), list(support), support_probability_map(support)


def choose_labels(*, count: int, instance_seed: int) -> list[str]:
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.labels")
    labels = resolve_chart_entity_labels(
        rng,
        count=int(count),
        min_chars=2,
        max_chars=7,
        allow_spaces=False,
    ).labels
    return [str(label) for label in labels]


def palette(params: Mapping[str, Any], *, count: int, instance_seed: int) -> list[RGB]:
    raw_palette = params.get("progress_palette_rgb", RENDER_DEFAULTS.get("progress_palette_rgb", []))
    colors: list[RGB] = []
    if isinstance(raw_palette, Sequence):
        for raw in raw_palette:
            if isinstance(raw, Sequence) and len(raw) == 3:
                colors.append(tuple(int(channel) for channel in raw))
    if not colors:
        colors = [
            (37, 99, 235),
            (220, 84, 45),
            (16, 132, 96),
            (139, 92, 246),
            (202, 138, 4),
            (14, 116, 144),
        ]
    shuffled = shuffled_support(
        spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.palette"),
        tuple(colors),
    )
    return list(islice(cycle(shuffled), int(count)))


def value_support(params: Mapping[str, Any]) -> list[int]:
    low, high = int_bounds(
        params,
        min_key="value_min",
        max_key="value_max",
        fallback_min=10,
        fallback_max=95,
    )
    step = max(1, int(params.get("value_step", generation_default("value_step", 5))))
    return [int(value) for value in range(int(low), int(high) + 1, int(step))]


def sample_threshold(params: Mapping[str, Any], *, instance_seed: int, namespace: str) -> int:
    threshold_values = [int(v) for v in params.get("threshold_values", generation_default("threshold_values", [30, 40, 50, 60, 70]))]
    if not threshold_values:
        threshold_values = [30, 40, 50, 60, 70]
    return int(
        uniform_choice(
            spawn_rng(int(instance_seed), str(namespace)),
            tuple(threshold_values),
            sort_keys=True,
        )
    )


def sample_range_pair(params: Mapping[str, Any], *, instance_seed: int, namespace: str) -> tuple[int, int]:
    raw_pairs = params.get(
        "range_pairs",
        generation_default("range_pairs", [[25, 55], [30, 60], [35, 70], [40, 75]]),
    )
    pairs: list[tuple[int, int]] = []
    if isinstance(raw_pairs, Sequence):
        for raw in raw_pairs:
            if isinstance(raw, Sequence) and len(raw) == 2:
                pairs.append((int(raw[0]), int(raw[1])))
    if not pairs:
        pairs = [(25, 55), (30, 60), (35, 70), (40, 75)]
    lower, upper = uniform_choice(
        spawn_rng(int(instance_seed), str(namespace)),
        tuple(pairs),
        sort_keys=True,
    )
    return (int(lower), int(upper)) if int(lower) <= int(upper) else (int(upper), int(lower))


def sample_condition_values(
    *,
    item_count: int,
    answer_count: int,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    target_predicate: Callable[[int], bool],
) -> tuple[list[int], tuple[str, ...]]:
    """Construct values with an exact target count for a task-owned predicate."""

    rng = spawn_rng(int(instance_seed), str(namespace))
    support = value_support(params)
    target_support = [int(value) for value in support if target_predicate(int(value))]
    other_support = [int(value) for value in support if not target_predicate(int(value))]
    if not target_support or not other_support:
        raise ValueError("radial progress predicate support must include matching and non-matching values")

    target_indices = tuple(sorted(rng.sample(range(int(item_count)), k=int(answer_count))))
    target_set = set(int(index) for index in target_indices)
    values: list[int] = []
    for index in range(int(item_count)):
        source = target_support if int(index) in target_set else other_support
        values.append(int(source[int(rng.randrange(0, len(source)))]))
    return values, tuple(str(f"i{index}") for index in target_indices)


def sample_distinct_values(params: Mapping[str, Any], *, item_count: int, instance_seed: int, namespace: str) -> list[int]:
    rng = spawn_rng(int(instance_seed), str(namespace))
    support = list(value_support(params))
    if len(support) < int(item_count):
        raise ValueError("radial progress extremum queries require enough distinct values to avoid ties")
    return [int(value) for value in rng.sample(support, k=int(item_count))]


def build_progress_items(
    *,
    labels: Sequence[str],
    values: Sequence[int],
    colors: Sequence[RGB],
) -> tuple[ProgressItem, ...]:
    return tuple(
        ProgressItem(
            item_id=f"i{index}",
            label=str(labels[index]),
            value=int(values[index]),
            color_rgb=tuple(int(channel) for channel in colors[index]),
        )
        for index in range(len(values))
    )


def sample_progress_frame(params: Mapping[str, Any], *, instance_seed: int) -> ProgressFrame:
    scene_variant, scene_probabilities = sample_scene_variant(params, instance_seed=int(instance_seed))
    item_count, item_count_probabilities = sample_item_count(params, instance_seed=int(instance_seed))
    labels = choose_labels(count=int(item_count), instance_seed=int(instance_seed))
    colors = palette(params, count=int(item_count), instance_seed=int(instance_seed))
    return ProgressFrame(
        scene_variant=str(scene_variant),
        scene_probabilities=dict(scene_probabilities),
        item_count=int(item_count),
        item_count_probabilities=dict(item_count_probabilities),
        labels=tuple(str(label) for label in labels),
        colors=tuple(colors),
        title=sample_title(params, instance_seed=int(instance_seed)),
    )


def sample_title(params: Mapping[str, Any], *, instance_seed: int) -> str:
    title_options = [
        str(value)
        for value in params.get("title_options", RENDER_DEFAULTS.get("title_options", ["Progress Summary"]))
    ] or ["Progress Summary"]
    return str(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.title"),
            tuple(title_options),
        )
    )
