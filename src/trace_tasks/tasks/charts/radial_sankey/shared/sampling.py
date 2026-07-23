"""Scene-neutral sampling primitives for radial Sankey charts."""

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
    RADIAL_CHORD_SANKEY,
    SCENE_NAMESPACE,
    SUPPORTED_SCENE_VARIANTS,
    FlowLink,
    FlowNode,
    RadialSankeyFrame,
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
        context=f"radial Sankey {explicit_key}",
    )


def sample_title(params: Mapping[str, Any], *, instance_seed: int) -> str:
    return sample_flow_title(
        params,
        title_options=TITLE_OPTIONS,
        instance_seed=int(instance_seed),
        namespace=SCENE_NAMESPACE,
        selection="uniform_choice",
    )


def sample_nodes(
    *,
    rng: Any,
    source_count: int,
    target_count: int,
) -> tuple[tuple[FlowNode, ...], tuple[FlowNode, ...]]:
    labels = list(
        resolve_chart_entity_labels(
            rng,
            count=int(source_count) + int(target_count),
            min_chars=2,
            max_chars=6,
            allow_spaces=False,
        ).labels
    )
    sources = tuple(
        FlowNode(node_id=f"source_{index}", label=str(label), role="source", index=int(index))
        for index, label in enumerate(labels[: int(source_count)])
    )
    targets = tuple(
        FlowNode(node_id=f"target_{index}", label=str(label), role="target", index=int(index))
        for index, label in enumerate(labels[int(source_count) : int(source_count) + int(target_count)])
    )
    return sources, targets


def sample_links(
    *,
    rng: Any,
    sources: Sequence[FlowNode],
    targets: Sequence[FlowNode],
    link_count: int,
    value_min: int,
    value_max: int,
) -> tuple[FlowLink, ...]:
    all_pairs = [(source, target) for source in sources for target in targets]
    if int(link_count) > len(all_pairs):
        raise ValueError("radial Sankey link count exceeds unique source-target pairs")
    selected_pairs = rng.sample(list(all_pairs), k=int(link_count))
    return tuple(
        FlowLink(
            link_id=f"link_{index}",
            source_id=str(source.node_id),
            source_label=str(source.label),
            target_id=str(target.node_id),
            target_label=str(target.label),
            value=int(rng.randint(int(value_min), int(value_max))),
        )
        for index, (source, target) in enumerate(selected_pairs)
    )


def link_side_counts(links: Sequence[FlowLink]) -> dict[str, dict[str, int]]:
    source_out: Counter[str] = Counter()
    target_in: Counter[str] = Counter()
    for link in links:
        source_out[str(link.source_id)] += 1
        target_in[str(link.target_id)] += 1
    return {"source_out": dict(source_out), "target_in": dict(target_in)}


def links_respect_side_limit(links: Sequence[FlowLink], *, max_links_per_node_side: int) -> bool:
    if int(max_links_per_node_side) <= 0:
        return True
    for counts in link_side_counts(links).values():
        if counts and max(int(value) for value in counts.values()) > int(max_links_per_node_side):
            return False
    return True


def sample_frame(params: Mapping[str, Any], *, instance_seed: int) -> RadialSankeyFrame:
    """Sample the visible radial Sankey graph without binding any public objective."""

    scene_variant, scene_probabilities = sample_scene_variant(params, instance_seed=int(instance_seed))
    if str(scene_variant) != RADIAL_CHORD_SANKEY:
        raise ValueError(f"unsupported radial Sankey scene variant: {scene_variant}")

    source_count, source_count_bounds = sample_count(
        params,
        min_key="radial_source_count_min",
        max_key="radial_source_count_max",
        explicit_key="radial_source_count",
        fallback_min=4,
        fallback_max=5,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.source_count",
    )
    target_count, target_count_bounds = sample_count(
        params,
        min_key="radial_target_count_min",
        max_key="radial_target_count_max",
        explicit_key="radial_target_count",
        fallback_min=4,
        fallback_max=5,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.target_count",
    )
    link_count, link_count_bounds = sample_count(
        params,
        min_key="radial_link_count_min",
        max_key="radial_link_count_max",
        explicit_key="radial_link_count",
        fallback_min=7,
        fallback_max=9,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.link_count",
    )
    value_min, value_max = required_int_bounds(
        params,
        min_key="radial_link_value_min",
        max_key="radial_link_value_max",
        fallback_min=8,
        fallback_max=35,
        context="radial Sankey link values",
    )
    max_links_per_node_side = max(0, gen_int_param(params, "radial_max_links_per_node_side", 4))

    for attempt in range(120):
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.frame", int(attempt))
        sources, targets = sample_nodes(
            rng=rng,
            source_count=int(source_count),
            target_count=int(target_count),
        )
        links = sample_links(
            rng=rng,
            sources=sources,
            targets=targets,
            link_count=int(link_count),
            value_min=int(value_min),
            value_max=int(value_max),
        )
        if not links_respect_side_limit(links, max_links_per_node_side=int(max_links_per_node_side)):
            continue
        return RadialSankeyFrame(
            scene_variant=str(scene_variant),
            scene_probabilities=dict(scene_probabilities),
            scene_title=sample_title(params, instance_seed=int(instance_seed) + int(attempt)),
            sources=tuple(sources),
            targets=tuple(targets),
            links=tuple(links),
            source_count_bounds=tuple(int(value) for value in source_count_bounds),
            target_count_bounds=tuple(int(value) for value in target_count_bounds),
            link_count_bounds=tuple(int(value) for value in link_count_bounds),
            max_links_per_node_side=int(max_links_per_node_side),
            link_side_counts=link_side_counts(links),
            value_min=int(value_min),
            value_max=int(value_max),
        )
    raise ValueError("failed to sample a radial Sankey frame satisfying link-side limits")


def links_by_source(links: Sequence[FlowLink]) -> dict[str, list[FlowLink]]:
    grouped: dict[str, list[FlowLink]] = {}
    for link in links:
        grouped.setdefault(str(link.source_id), []).append(link)
    return grouped


def links_by_target(links: Sequence[FlowLink]) -> dict[str, list[FlowLink]]:
    grouped: dict[str, list[FlowLink]] = {}
    for link in links:
        grouped.setdefault(str(link.target_id), []).append(link)
    return grouped


def sorted_by_target_label(links: Sequence[FlowLink]) -> list[FlowLink]:
    return sorted(links, key=lambda item: (str(item.target_label), str(item.link_id)))


def sorted_by_source_label(links: Sequence[FlowLink]) -> list[FlowLink]:
    return sorted(links, key=lambda item: (str(item.source_label), str(item.link_id)))


def group_size_bounds(params: Mapping[str, Any]) -> tuple[int, int]:
    return required_int_bounds(
        params,
        min_key="radial_group_size_min",
        max_key="radial_group_size_max",
        fallback_min=2,
        fallback_max=3,
        context="radial Sankey grouped endpoint count",
    )


def sample_group_size(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> int:
    lower, upper = group_size_bounds(params)
    return balanced_int_from_support(
        list(range(int(lower), int(upper) + 1)),
        params=params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def answer_value_bounds(params: Mapping[str, Any]) -> tuple[int, int]:
    return required_int_bounds(
        params,
        min_key="radial_transfer_answer_min",
        max_key="radial_transfer_answer_max",
        fallback_min=16,
        fallback_max=100,
        context="radial Sankey transfer answer",
    )


def join_quoted(labels: Sequence[str]) -> str:
    quoted = [f'"{str(label)}"' for label in labels]
    if len(quoted) <= 1:
        return quoted[0] if quoted else ""
    if len(quoted) == 2:
        return f"{quoted[0]} and {quoted[1]}"
    return f"{', '.join(quoted[:-1])}, and {quoted[-1]}"


def link_dict(link: FlowLink) -> dict[str, Any]:
    return {
        "link_id": str(link.link_id),
        "source_id": str(link.source_id),
        "source_label": str(link.source_label),
        "target_id": str(link.target_id),
        "target_label": str(link.target_label),
        "value": int(link.value),
    }


def node_dict(node: FlowNode) -> dict[str, Any]:
    return {
        "node_id": str(node.node_id),
        "label": str(node.label),
        "role": str(node.role),
        "index": int(node.index),
    }


def links_by_id(links: Sequence[FlowLink]) -> dict[str, FlowLink]:
    return {str(link.link_id): link for link in links}


__all__ = [
    "answer_value_bounds",
    "group_size_bounds",
    "join_quoted",
    "link_dict",
    "link_side_counts",
    "links_by_id",
    "links_by_source",
    "links_by_target",
    "node_dict",
    "sample_frame",
    "sample_group_size",
    "sample_scene_variant",
    "sorted_by_source_label",
    "sorted_by_target_label",
]
