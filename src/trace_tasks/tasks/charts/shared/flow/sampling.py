"""Neutral sampling helpers for flow-style chart renderers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.labeled_chart_variants import resolve_chart_axis_variant_for_namespace
from trace_tasks.tasks.charts.shared.balanced_sampling import balanced_int_from_support
from trace_tasks.tasks.shared.config_defaults import resolve_required_int_bounds


def resolve_flow_required_int_bounds(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
    context: str,
) -> tuple[int, int]:
    """Resolve required integer bounds for a flow scene generation parameter."""

    return resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context=str(context),
    )


def sample_flow_scene_variant(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    supported_variants: Sequence[str],
    namespace: str,
    explicit_key: str = "scene_variant",
    weights_key: str = "scene_variant_weights",
    balance_flag_key: str = "balanced_scene_variant_sampling",
) -> tuple[str, dict[str, float]]:
    """Sample a scene variant from caller-provided flow scene support."""

    return resolve_chart_axis_variant_for_namespace(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=tuple(str(value) for value in supported_variants),
        namespace=str(namespace),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
    )


def sample_flow_count(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    explicit_key: str,
    fallback_min: int,
    fallback_max: int,
    instance_seed: int,
    namespace: str,
    context: str,
) -> tuple[int, tuple[int, int]]:
    """Sample one bounded count with explicit override validation."""

    lower, upper = resolve_flow_required_int_bounds(
        params,
        gen_defaults,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context=str(context),
    )
    support = [int(value) for value in range(int(lower), int(upper) + 1)]
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        selected = int(explicit)
        if selected not in set(support):
            raise ValueError(f"{explicit_key} must be in {lower}..{upper}")
        return int(selected), (int(lower), int(upper))
    return (
        balanced_int_from_support(
            support,
            params=params,
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        ),
        (int(lower), int(upper)),
    )


def sample_flow_title(
    params: Mapping[str, Any],
    *,
    title_options: Sequence[str],
    instance_seed: int,
    namespace: str,
    selection: Literal["index", "uniform_choice"],
) -> str:
    """Sample a title from caller-provided flow scene options."""

    options = [str(value) for value in params.get("title_options", title_options) if str(value)] or [
        str(value) for value in title_options
    ]
    rng = spawn_rng(int(instance_seed), f"{namespace}.title")
    if selection == "index":
        return str(options[int(rng.randrange(len(options)))])
    if selection == "uniform_choice":
        return str(uniform_choice(rng, tuple(options)))
    raise ValueError(f"unsupported flow title selection method: {selection}")
