"""Scene-neutral sampling primitives for standard Sankey charts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.flow import (
    sample_flow_count,
    sample_flow_scene_variant,
    sample_flow_title,
)
from trace_tasks.tasks.charts.shared.balanced_sampling import balanced_int_from_support
from trace_tasks.tasks.charts.shared.label_assets import resolve_chart_entity_labels

from .defaults import GEN_DEFAULTS, TITLE_OPTIONS, gen_int_param, required_int_bounds
from .state import (
    SCENE_NAMESPACE,
    SUPPORTED_SCENE_VARIANTS,
    THREE_COLUMN_SANKEY,
    FlowNode,
    FlowPath,
    SankeyFrame,
)


def sample_scene_variant(params: Mapping[str, Any], *, instance_seed: int) -> tuple[str, dict[str, float]]:
    return sample_flow_scene_variant(
        params=params,
        gen_defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        namespace=f"{SCENE_NAMESPACE}.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
    )


def sample_count(
    params: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    explicit_key: str,
    fallback_min: int,
    fallback_max: int,
    instance_seed: int,
    namespace: str,
) -> tuple[int, tuple[int, int]]:
    return sample_flow_count(
        params,
        GEN_DEFAULTS,
        min_key=str(min_key),
        max_key=str(max_key),
        explicit_key=str(explicit_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        context=f"Sankey {explicit_key}",
    )


def sample_title(params: Mapping[str, Any], *, instance_seed: int) -> str:
    return sample_flow_title(
        params,
        title_options=TITLE_OPTIONS,
        instance_seed=int(instance_seed),
        namespace=SCENE_NAMESPACE,
        selection="index",
    )


def node_dict(node: FlowNode) -> dict[str, Any]:
    return {
        "node_id": str(node.node_id),
        "label": str(node.label),
        "column": str(node.column),
        "index": int(node.index),
    }


def path_dict(path: FlowPath) -> dict[str, Any]:
    return {
        "path_id": str(path.path_id),
        "source_id": str(path.source_id),
        "source_label": str(path.source_label),
        "middle_id": str(path.middle_id),
        "middle_label": str(path.middle_label),
        "target_id": str(path.target_id),
        "target_label": str(path.target_label),
        "first_value": int(path.first_value),
        "second_value": int(path.second_value),
        "bottleneck_value": int(path.bottleneck_value),
        "absolute_difference": int(path.absolute_difference),
    }


def bottleneck_segment_ref(path: FlowPath) -> str:
    """Return the unique lower-valued segment ref for a two-band path."""

    if int(path.first_value) == int(path.second_value):
        raise ValueError("Sankey bottleneck segment is ambiguous when path values tie")
    segment_kind = "source_middle" if int(path.first_value) < int(path.second_value) else "middle_target"
    return f"{path.path_id}:{segment_kind}"


def join_quoted(labels: Sequence[str]) -> str:
    return ", ".join(f'"{label}"' for label in labels)


def node_specs(labels: Sequence[str], *, prefix: str, column: str) -> tuple[FlowNode, ...]:
    return tuple(
        FlowNode(
            node_id=f"{prefix}_{index}",
            label=str(label),
            column=str(column),
            index=int(index),
        )
        for index, label in enumerate(labels)
    )


def sample_nodes(
    rng: Any,
    *,
    source_count: int,
    middle_count: int,
    target_count: int,
) -> tuple[tuple[FlowNode, ...], tuple[FlowNode, ...], tuple[FlowNode, ...]]:
    labels = list(
        resolve_chart_entity_labels(
            rng,
            count=int(source_count) + int(middle_count) + int(target_count),
            min_chars=2,
            max_chars=6,
            allow_spaces=False,
        ).labels
    )
    middle_start = int(source_count)
    target_start = int(source_count) + int(middle_count)
    return (
        node_specs(labels[: int(source_count)], prefix="source", column="source"),
        node_specs(labels[middle_start:target_start], prefix="middle", column="middle"),
        node_specs(labels[target_start : target_start + int(target_count)], prefix="target", column="target"),
    )


def build_path(
    *,
    path_id: str,
    source: FlowNode,
    middle: FlowNode,
    target: FlowNode,
    first_value: int,
    second_value: int,
) -> FlowPath:
    bottleneck = min(int(first_value), int(second_value))
    difference = abs(int(first_value) - int(second_value))
    return FlowPath(
        path_id=str(path_id),
        source_id=str(source.node_id),
        source_label=str(source.label),
        middle_id=str(middle.node_id),
        middle_label=str(middle.label),
        target_id=str(target.node_id),
        target_label=str(target.label),
        first_value=int(first_value),
        second_value=int(second_value),
        bottleneck_value=int(bottleneck),
        absolute_difference=int(difference),
    )


def path_side_counts(paths: Sequence[FlowPath]) -> dict[str, dict[str, int]]:
    source_out: Counter[str] = Counter()
    middle_in: Counter[str] = Counter()
    middle_out: Counter[str] = Counter()
    target_in: Counter[str] = Counter()
    for path in paths:
        source_out[str(path.source_id)] += 1
        middle_in[str(path.middle_id)] += 1
        middle_out[str(path.middle_id)] += 1
        target_in[str(path.target_id)] += 1
    return {
        "source_out": dict(source_out),
        "middle_in": dict(middle_in),
        "middle_out": dict(middle_out),
        "target_in": dict(target_in),
    }


def paths_respect_side_limit(paths: Sequence[FlowPath], *, max_paths_per_node_side: int) -> bool:
    if int(max_paths_per_node_side) <= 0:
        return True
    for counts in path_side_counts(paths).values():
        if counts and max(int(value) for value in counts.values()) > int(max_paths_per_node_side):
            return False
    return True


def paths_by_source(paths: Sequence[FlowPath]) -> dict[str, list[FlowPath]]:
    grouped: dict[str, list[FlowPath]] = {}
    for path in paths:
        grouped.setdefault(str(path.source_id), []).append(path)
    return grouped


def paths_by_target(paths: Sequence[FlowPath]) -> dict[str, list[FlowPath]]:
    grouped: dict[str, list[FlowPath]] = {}
    for path in paths:
        grouped.setdefault(str(path.target_id), []).append(path)
    return grouped


def sorted_by_middle_target(paths: Sequence[FlowPath]) -> list[FlowPath]:
    return sorted(paths, key=lambda item: (str(item.middle_label), str(item.target_label), str(item.path_id)))


def sorted_by_source_middle(paths: Sequence[FlowPath]) -> list[FlowPath]:
    return sorted(paths, key=lambda item: (str(item.source_label), str(item.middle_label), str(item.path_id)))


def sorted_by_middle_label(paths: Sequence[FlowPath]) -> list[FlowPath]:
    return sorted(paths, key=lambda item: (str(item.middle_label), str(item.path_id)))


def sample_route_count(params: Mapping[str, Any], *, instance_seed: int) -> int:
    lower, upper = required_int_bounds(
        params,
        min_key="source_target_route_count_min",
        max_key="source_target_route_count_max",
        fallback_min=2,
        fallback_max=3,
        context="Sankey source-target route count",
    )
    return balanced_int_from_support(
        list(range(int(lower), int(upper) + 1)),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.source_target_route_count",
    )


def answer_value_bounds(
    params: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
    context: str,
) -> tuple[int, int]:
    return required_int_bounds(
        params,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context=str(context),
    )


def _select_path_triples(
    *,
    rng: Any,
    params: Mapping[str, Any],
    instance_seed: int,
    sources: Sequence[FlowNode],
    middles: Sequence[FlowNode],
    targets: Sequence[FlowNode],
    path_count: int,
    reserved_route_count: int | None,
) -> tuple[list[tuple[FlowNode, FlowNode, FlowNode]], dict[str, Any]]:
    """Choose visible path triples, optionally reserving a multi-route source-target family."""

    all_triples = [(source, middle, target) for source in sources for middle in middles for target in targets]
    if reserved_route_count is None:
        return list(rng.sample(list(all_triples), k=int(path_count))), {}

    source_target_pairs = [(source, target) for source in sources for target in targets]
    pair_index = balanced_int_from_support(
        list(range(len(source_target_pairs))),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.source_target_pair",
    )
    selected_source, selected_target = source_target_pairs[int(pair_index)]
    route_count = min(int(reserved_route_count), len(middles), int(path_count))
    if int(route_count) < 2:
        raise ValueError("reserved Sankey source-target route count must be at least two")
    selected_middles = [middles[index] for index in rng.sample(list(range(len(middles))), int(route_count))]
    triples = [(selected_source, middle, selected_target) for middle in selected_middles]
    excluded_pair = (str(selected_source.node_id), str(selected_target.node_id))
    remaining = [
        triple
        for triple in all_triples
        if (str(triple[0].node_id), str(triple[2].node_id)) != excluded_pair
        and str(triple[0].node_id) != str(selected_source.node_id)
        and str(triple[2].node_id) != str(selected_target.node_id)
    ]
    rng.shuffle(remaining)
    needed = max(0, int(path_count) - len(triples))
    if len(remaining) < int(needed):
        raise ValueError("not enough uncrowded distractor paths for Sankey source-target route")
    triples.extend(remaining[: int(needed)])
    return triples, {
        "source_id": str(selected_source.node_id),
        "source_label": str(selected_source.label),
        "target_id": str(selected_target.node_id),
        "target_label": str(selected_target.label),
        "route_count": int(route_count),
    }


def sample_frame(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    reserved_route_count: int | None = None,
) -> SankeyFrame:
    """Sample the visible Sankey diagram without binding a public objective."""

    scene_variant, scene_probabilities = sample_scene_variant(params, instance_seed=int(instance_seed))
    if str(scene_variant) != THREE_COLUMN_SANKEY:
        raise ValueError(f"unsupported Sankey scene variant: {scene_variant}")

    source_count, source_count_bounds = sample_count(
        params,
        min_key="source_count_min",
        max_key="source_count_max",
        explicit_key="source_count",
        fallback_min=2,
        fallback_max=3,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.source_count",
    )
    middle_count, middle_count_bounds = sample_count(
        params,
        min_key="middle_count_min",
        max_key="middle_count_max",
        explicit_key="middle_count",
        fallback_min=2,
        fallback_max=3,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.middle_count",
    )
    target_count, target_count_bounds = sample_count(
        params,
        min_key="target_count_min",
        max_key="target_count_max",
        explicit_key="target_count",
        fallback_min=2,
        fallback_max=3,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.target_count",
    )
    path_count, path_count_bounds = sample_count(
        params,
        min_key="path_count_min",
        max_key="path_count_max",
        explicit_key="path_count",
        fallback_min=3,
        fallback_max=4,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.path_count",
    )
    value_min, value_max = required_int_bounds(
        params,
        min_key="link_value_min",
        max_key="link_value_max",
        fallback_min=5,
        fallback_max=35,
        context="Sankey link values",
    )
    if int(path_count) > int(source_count) * int(middle_count) * int(target_count):
        raise ValueError("Sankey path count exceeds possible unique paths")
    max_paths_per_node_side = max(0, gen_int_param(params, "max_paths_per_node_side", 3))

    for attempt in range(100):
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.frame", int(attempt))
        sources, middles, targets = sample_nodes(
            rng,
            source_count=int(source_count),
            middle_count=int(middle_count),
            target_count=int(target_count),
        )
        triples, route_focus = _select_path_triples(
            rng=rng,
            params=params,
            instance_seed=int(instance_seed),
            sources=sources,
            middles=middles,
            targets=targets,
            path_count=int(path_count),
            reserved_route_count=reserved_route_count,
        )
        paths = tuple(
            build_path(
                path_id=f"path_{index}",
                source=source,
                middle=middle,
                target=target,
                first_value=int(rng.randint(int(value_min), int(value_max))),
                second_value=int(rng.randint(int(value_min), int(value_max))),
            )
            for index, (source, middle, target) in enumerate(triples)
        )
        if not paths_respect_side_limit(paths, max_paths_per_node_side=int(max_paths_per_node_side)):
            continue
        return SankeyFrame(
            scene_variant=str(scene_variant),
            scene_probabilities=dict(scene_probabilities),
            scene_title=sample_title(params, instance_seed=int(instance_seed) + int(attempt)),
            sources=tuple(sources),
            middles=tuple(middles),
            targets=tuple(targets),
            paths=tuple(paths),
            source_count_bounds=tuple(int(value) for value in source_count_bounds),
            middle_count_bounds=tuple(int(value) for value in middle_count_bounds),
            target_count_bounds=tuple(int(value) for value in target_count_bounds),
            path_count_bounds=tuple(int(value) for value in path_count_bounds),
            max_paths_per_node_side=int(max_paths_per_node_side),
            path_side_counts=path_side_counts(paths),
            value_min=int(value_min),
            value_max=int(value_max),
            route_focus=dict(route_focus),
        )
    raise ValueError("failed to sample a Sankey frame satisfying path-side limits")
