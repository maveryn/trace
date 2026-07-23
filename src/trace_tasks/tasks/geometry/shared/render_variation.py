"""Shared deterministic render-parameter variation helpers for geometry tasks."""

from __future__ import annotations

from typing import Any, Mapping

from ...shared.config_defaults import group_default


def sample_int_render_param(
    rng,
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    key: str,
    fallback: int,
    min_key: str | None = None,
    max_key: str | None = None,
    minimum_value: int = 1,
) -> int:
    """Resolve one integer render parameter with optional deterministic range sampling.

    Resolution order:
    1) explicit `params[key]`
    2) sampled integer in `[min_key, max_key]` if either bound is provided
       in params or render defaults
    3) `render_defaults[key]`
    4) function fallback
    """
    name = str(key)
    explicit_value = params.get(name)
    if explicit_value is not None:
        return max(int(minimum_value), int(explicit_value))

    min_name = str(min_key) if min_key is not None else f"{name}_min"
    max_name = str(max_key) if max_key is not None else f"{name}_max"
    min_raw = params.get(min_name, group_default(render_defaults, min_name, None))
    max_raw = params.get(max_name, group_default(render_defaults, max_name, None))
    if min_raw is not None or max_raw is not None:
        default_value = group_default(render_defaults, name, int(fallback))
        low = int(default_value if min_raw is None else min_raw)
        high = int(default_value if max_raw is None else max_raw)
        if int(low) > int(high):
            raise ValueError(f"{min_name} must be <= {max_name}")
        sampled = int(rng.randint(int(low), int(high)))
        return max(int(minimum_value), int(sampled))

    resolved = int(group_default(render_defaults, name, int(fallback)))
    return max(int(minimum_value), int(resolved))

