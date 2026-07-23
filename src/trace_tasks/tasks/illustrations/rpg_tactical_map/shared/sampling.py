"""Sampling helpers for RPG tactical map tasks."""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.config_defaults import group_default


def select_int_from_support(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
) -> tuple[int, dict[str, float]]:
    """Select one integer from an explicit/default support set."""

    support_raw = params.get(str(support_key), group_default(defaults, str(support_key), fallback_support))
    if isinstance(support_raw, str):
        support = (int(support_raw),)
    else:
        support = tuple(int(value) for value in support_raw)
    if not support:
        raise ValueError(f"{support_key} must contain at least one value")
    explicit = params.get(str(explicit_key), None)
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError(f"{explicit_key} must be in {support}")
        return value, {str(value): 1.0}
    rng = random.Random(f"{int(instance_seed)}:{namespace}")
    value = int(support[int(rng.randrange(len(support)))])
    probability = 1.0 / float(len(support))
    return value, {str(item): probability for item in support}


def select_string_from_support(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[str],
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Select one string from an explicit/default support set."""

    support_raw = params.get(str(support_key), group_default(defaults, str(support_key), fallback_support))
    if isinstance(support_raw, str):
        support = (str(support_raw),)
    else:
        support = tuple(str(value) for value in support_raw)
    support = tuple(value for value in support if value)
    if not support:
        raise ValueError(f"{support_key} must contain at least one value")
    explicit = params.get(str(explicit_key), None)
    if explicit is not None:
        value = str(explicit)
        if value not in set(support):
            raise ValueError(f"{explicit_key} must be in {support}")
        return value, {str(value): 1.0}
    rng = random.Random(f"{int(instance_seed)}:{namespace}")
    value = str(support[int(rng.randrange(len(support)))])
    probability = 1.0 / float(len(support))
    return value, {str(item): probability for item in support}


__all__ = ["select_int_from_support", "select_string_from_support"]
