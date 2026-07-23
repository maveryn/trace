"""Shared option-count selection helpers for geometry option-label tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.sampling import normalize_positive_weights, weighted_choice
from ....core.seed import spawn_rng


def resolve_geometry_option_count(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    field_name: str,
    supported_counts: Sequence[int],
    task_id: str,
    instance_seed: int,
) -> Tuple[int, Dict[str, float]]:
    """Resolve a visible option/candidate/panel count for geometry MCQ-style tasks."""

    counts = tuple(int(value) for value in supported_counts)
    if not counts:
        raise ValueError(f"{field_name} supported_counts must not be empty")
    if len(set(counts)) != len(counts):
        raise ValueError(f"{field_name} supported_counts must not contain duplicates")
    if min(counts) < 2:
        raise ValueError(f"{field_name} supported_counts must be at least 2")
    count_set = set(counts)

    explicit = params.get(str(field_name))
    if explicit is not None:
        selected = int(explicit)
        if selected not in count_set:
            raise ValueError(f"{field_name}={selected} is outside supported counts {counts}")
        return selected, {str(value): (1.0 if int(value) == selected else 0.0) for value in counts}

    weights_key = f"{field_name}_weights"
    raw_weights = params.get(str(weights_key), gen_defaults.get(str(weights_key), {str(value): 1.0 for value in counts}))
    if not isinstance(raw_weights, Mapping):
        raise ValueError(f"{weights_key} must be a mapping when provided")
    weights: Dict[str, float] = {}
    for key, value in raw_weights.items():
        try:
            count_key = int(key)
        except (TypeError, ValueError):
            continue
        if int(count_key) in count_set:
            weights[str(count_key)] = float(value)
    probabilities = normalize_positive_weights(weights, default_keys=[str(value) for value in counts])
    rng = spawn_rng(int(instance_seed), f"{str(task_id)}.{field_name}")
    selected = int(weighted_choice(rng, probabilities, sort_keys=True))
    return selected, {str(key): float(value) for key, value in sorted(probabilities.items(), key=lambda item: int(item[0]))}


def panel_grid_shape_for_option_count(option_count: int) -> Tuple[int, int]:
    """Return a compact panel-grid shape for visible MCQ option counts."""

    count = int(option_count)
    if count == 4:
        return (2, 2)
    if count == 6:
        return (3, 2)
    if count == 9:
        return (3, 3)
    raise ValueError(f"unsupported panel-grid option_count={count}; expected 4, 6, or 9")
