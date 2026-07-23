"""Shared label-axis helpers for Bubble-shooter tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default


def resolve_bubble_shooter_label_support(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[str],
    allowed_labels: Sequence[str] | None = None,
) -> Tuple[str, ...]:
    """Resolve a task-owned Bubble-shooter label support."""

    raw = params.get(str(key), group_default(gen_defaults, str(key), tuple(fallback)))
    if raw is None:
        raw = tuple(fallback)
    if isinstance(raw, str):
        values = (raw,)
    else:
        values = tuple(str(value) for value in raw)
    values = tuple(value for value in values if value)
    if not values:
        raise ValueError(f"{key} must contain at least one label")
    if allowed_labels is not None:
        invalid = sorted(set(values) - {str(label) for label in allowed_labels})
        if invalid:
            raise ValueError(
                f"{key} contains labels outside displayed options: {invalid}"
            )
    return values


def resolve_bubble_shooter_label_choice(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[str],
    namespace: str,
    allowed_labels: Sequence[str] | None = None,
) -> tuple[str, Dict[str, float], Tuple[str, ...]]:
    """Resolve one task-owned Bubble-shooter answer label."""

    support = resolve_bubble_shooter_label_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=fallback_support,
        allowed_labels=allowed_labels,
    )
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        value = str(explicit)
        if value not in support:
            raise ValueError(f"{explicit_key}={value!r} is not in {support_key}")
        return (
            value,
            {str(item): (1.0 if str(item) == value else 0.0) for item in support},
            support,
        )

    probabilities = {str(item): 1.0 / float(len(support)) for item in support}
    rng = spawn_rng(int(instance_seed), str(namespace))
    return str(uniform_choice(rng, support)), probabilities, support


__all__ = [
    "resolve_bubble_shooter_label_choice",
    "resolve_bubble_shooter_label_support",
]
