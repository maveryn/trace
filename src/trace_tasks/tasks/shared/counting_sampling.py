"""Cross-domain sampling helpers for counting-style tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ...core.sampling import normalize_positive_weights, weighted_choice


def resolve_counting_object_count(
    rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    fallback_min: int,
    fallback_max: int,
) -> Tuple[int, Dict[str, float]]:
    """Resolve how many objects appear in one counting scene."""

    min_count = int(params.get("object_count_min", gen_defaults.get("object_count_min", int(fallback_min))))
    max_count = int(params.get("object_count_max", gen_defaults.get("object_count_max", int(fallback_max))))
    if min_count < 2 or min_count > max_count:
        raise ValueError("invalid object_count_min/object_count_max for counting task")
    supported_counts = [int(value) for value in range(int(min_count), int(max_count) + 1)]

    explicit = params.get("object_count")
    if explicit is not None:
        selected = int(explicit)
        if int(selected) not in set(supported_counts):
            raise ValueError("object_count is outside configured supported range")
        return int(selected), {
            str(value): (1.0 if int(value) == int(selected) else 0.0)
            for value in supported_counts
        }

    raw_weights = params.get(
        "object_count_weights",
        gen_defaults.get("object_count_weights", {str(value): 1.0 for value in supported_counts}),
    )
    if not isinstance(raw_weights, Mapping):
        raise ValueError("object_count_weights must be a mapping when provided")
    supported_keys = {str(value) for value in supported_counts}
    weights = {str(key): float(value) for key, value in raw_weights.items() if str(key) in supported_keys}
    probabilities = normalize_positive_weights(weights, default_keys=[str(value) for value in supported_counts])
    selected = int(weighted_choice(rng, probabilities, sort_keys=True))

    return int(selected), {
        str(key): float(value)
        for key, value in sorted(probabilities.items(), key=lambda item: int(item[0]))
    }


def resolve_counting_target_count(
    rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    object_count: int,
    default_min: int = 1,
    default_margin_from_total: int = 1,
) -> Tuple[int, Dict[str, float]]:
    """Resolve how many objects match the queried class in one counting scene."""

    min_count = int(params.get("target_count_min", gen_defaults.get("target_count_min", int(default_min))))
    max_default = max(int(default_min), int(object_count) - int(default_margin_from_total))
    max_count = int(params.get("target_count_max", gen_defaults.get("target_count_max", int(max_default))))
    min_count = max(0, int(min_count))
    max_count = min(int(object_count), int(max_count))
    if min_count > max_count:
        raise ValueError("invalid target_count_min/target_count_max for counting task")
    supported_counts = [int(value) for value in range(int(min_count), int(max_count) + 1)]
    if not supported_counts:
        raise ValueError("counting task resolved no supported target counts")

    explicit = params.get("target_count")
    if explicit is not None:
        selected = int(explicit)
        if int(selected) not in set(supported_counts):
            raise ValueError("target_count is outside configured supported range")
        return int(selected), {
            str(value): (1.0 if int(value) == int(selected) else 0.0)
            for value in supported_counts
        }

    raw_weights = params.get(
        "target_count_weights",
        gen_defaults.get("target_count_weights", {str(value): 1.0 for value in supported_counts}),
    )
    if not isinstance(raw_weights, Mapping):
        raise ValueError("target_count_weights must be a mapping when provided")
    supported_keys = {str(value) for value in supported_counts}
    weights = {str(key): float(value) for key, value in raw_weights.items() if str(key) in supported_keys}
    probabilities = normalize_positive_weights(weights, default_keys=[str(value) for value in supported_counts])
    selected = int(weighted_choice(rng, probabilities, sort_keys=True))

    return int(selected), {
        str(key): float(value)
        for key, value in sorted(probabilities.items(), key=lambda item: int(item[0]))
    }


def resolve_counting_cardinality_pair(
    rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    fallback_object_min: int,
    fallback_object_max: int,
    default_target_min: int = 1,
    default_margin_from_total: int = 1,
) -> Tuple[int, Dict[str, float], int, Dict[str, float]]:
    """Resolve one `(object_count, target_count)` pair with flatter count answers."""

    min_object = int(params.get("object_count_min", gen_defaults.get("object_count_min", int(fallback_object_min))))
    max_object = int(params.get("object_count_max", gen_defaults.get("object_count_max", int(fallback_object_max))))
    if min_object < 2 or min_object > max_object:
        raise ValueError("invalid object_count_min/object_count_max for counting task")
    supported_object_counts = [int(value) for value in range(int(min_object), int(max_object) + 1)]

    min_target = int(params.get("target_count_min", gen_defaults.get("target_count_min", int(default_target_min))))
    max_default = max(int(default_target_min), int(max_object) - int(default_margin_from_total))
    max_target = int(params.get("target_count_max", gen_defaults.get("target_count_max", int(max_default))))
    min_target = max(0, int(min_target))
    max_target = min(int(max_object), int(max_target))
    supported_target_counts = [
        int(value)
        for value in range(int(min_target), int(max_target) + 1)
        if any(int(obj) - int(default_margin_from_total) >= int(value) for obj in supported_object_counts)
    ]
    if not supported_target_counts:
        raise ValueError("counting task resolved no globally feasible target counts")

    explicit_object = params.get("object_count")
    explicit_target = params.get("target_count")
    if explicit_object is not None and explicit_target is not None:
        object_count = int(explicit_object)
        target_count = int(explicit_target)
        if int(object_count) not in set(supported_object_counts):
            raise ValueError("object_count is outside configured supported range")
        feasible_targets = [
            int(value)
            for value in supported_target_counts
            if int(value) <= int(object_count) - int(default_margin_from_total)
        ]
        if int(target_count) not in set(feasible_targets):
            raise ValueError("target_count is outside configured supported range")
        return (
            int(object_count),
            {
                str(value): (1.0 if int(value) == int(object_count) else 0.0)
                for value in supported_object_counts
            },
            int(target_count),
            {
                str(value): (1.0 if int(value) == int(target_count) else 0.0)
                for value in supported_target_counts
            },
        )

    if explicit_object is not None:
        object_count = int(explicit_object)
        if int(object_count) not in set(supported_object_counts):
            raise ValueError("object_count is outside configured supported range")
        target_count, target_probabilities = resolve_counting_target_count(
            rng,
            instance_seed=int(instance_seed),
            params=params,
            gen_defaults=gen_defaults,
            object_count=int(object_count),
            default_min=int(default_target_min),
            default_margin_from_total=int(default_margin_from_total),
        )
        return (
            int(object_count),
            {
                str(value): (1.0 if int(value) == int(object_count) else 0.0)
                for value in supported_object_counts
            },
            int(target_count),
            dict(target_probabilities),
        )

    target_weights_raw = params.get(
        "target_count_weights",
        gen_defaults.get("target_count_weights", {str(value): 1.0 for value in supported_target_counts}),
    )
    if not isinstance(target_weights_raw, Mapping):
        raise ValueError("target_count_weights must be a mapping when provided")
    supported_target_keys = {str(value) for value in supported_target_counts}
    target_weights = {
        str(key): float(value)
        for key, value in target_weights_raw.items()
        if str(key) in supported_target_keys
    }
    target_probabilities = normalize_positive_weights(
        target_weights,
        default_keys=[str(value) for value in supported_target_counts],
    )

    if explicit_target is not None:
        target_count = int(explicit_target)
        if int(target_count) not in set(supported_target_counts):
            raise ValueError("target_count is outside configured supported range")
    else:
        target_count = int(weighted_choice(rng, target_probabilities, sort_keys=True))

    feasible_object_counts = [
        int(value)
        for value in supported_object_counts
        if int(value) - int(default_margin_from_total) >= int(target_count)
    ]
    if not feasible_object_counts:
        raise ValueError("counting task resolved no feasible object counts for selected target_count")
    object_weights_raw = params.get(
        "object_count_weights",
        gen_defaults.get("object_count_weights", {str(value): 1.0 for value in supported_object_counts}),
    )
    if not isinstance(object_weights_raw, Mapping):
        raise ValueError("object_count_weights must be a mapping when provided")
    supported_object_keys = {str(value) for value in feasible_object_counts}
    object_weights = {
        str(key): float(value)
        for key, value in object_weights_raw.items()
        if str(key) in supported_object_keys
    }
    object_probabilities = normalize_positive_weights(
        object_weights,
        default_keys=[str(value) for value in feasible_object_counts],
    )
    object_count = int(weighted_choice(rng, object_probabilities, sort_keys=True))

    return (
        int(object_count),
        {
            str(key): float(value)
            for key, value in sorted(object_probabilities.items(), key=lambda item: int(item[0]))
        },
        int(target_count),
        {
            str(key): float(value)
            for key, value in sorted(target_probabilities.items(), key=lambda item: int(item[0]))
        },
    )


def _sorted_probability_map(probabilities: Mapping[str, float]) -> Dict[str, float]:
    """Return one integer-key-sorted probability mapping."""

    return {
        str(key): float(value)
        for key, value in sorted(probabilities.items(), key=lambda item: int(item[0]))
    }


def _one_hot_probability_map(selected: int, supported_counts: Mapping[int, Any] | list[int]) -> Dict[str, float]:
    """Return one deterministic one-hot probability mapping."""

    return {
        str(int(value)): (1.0 if int(value) == int(selected) else 0.0)
        for value in supported_counts
    }


def _distractor_probability_map_for_target(
    *,
    object_probabilities: Mapping[str, float],
    target_count: int,
) -> Dict[str, float]:
    """Project total-count probabilities into distractor-count probabilities."""

    distractor_probabilities: Dict[str, float] = {}
    for object_count_key, probability in object_probabilities.items():
        distractor_count = int(object_count_key) - int(target_count)
        distractor_probabilities[str(int(distractor_count))] = (
            float(distractor_probabilities.get(str(int(distractor_count)), 0.0)) + float(probability)
        )
    return _sorted_probability_map(distractor_probabilities)


def resolve_counting_target_first_cardinality_triplet(
    rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    fallback_object_min: int,
    fallback_object_max: int,
    fallback_target_min: int = 0,
    fallback_target_max: int = 10,
) -> Tuple[int, Dict[str, float], int, Dict[str, float], int, Dict[str, float]]:
    """Resolve `(object_count, target_count, distractor_count)` via target-first sampling.

    This matches the target-first policy used by reference-scene icon counting:
    sample the answer count first, then sample the total scene cardinality from
    the feasible support, and derive distractors as the remaining icons.
    """

    min_object = int(params.get("object_count_min", gen_defaults.get("object_count_min", int(fallback_object_min))))
    max_object = int(params.get("object_count_max", gen_defaults.get("object_count_max", int(fallback_object_max))))
    if min_object < 2 or min_object > max_object:
        raise ValueError("invalid object_count_min/object_count_max for counting task")
    supported_object_counts = [int(value) for value in range(int(min_object), int(max_object) + 1)]

    min_target = int(params.get("target_count_min", gen_defaults.get("target_count_min", int(fallback_target_min))))
    max_target = int(params.get("target_count_max", gen_defaults.get("target_count_max", int(fallback_target_max))))
    min_target = max(0, int(min_target))
    max_target = min(int(max_target), int(max_object))
    supported_target_counts = [int(value) for value in range(int(min_target), int(max_target) + 1)]
    if not supported_target_counts:
        raise ValueError("counting task resolved no supported target counts")

    explicit_object = params.get("object_count")
    explicit_target = params.get("target_count")
    explicit_distractor = params.get("distractor_count")

    if explicit_object is not None and explicit_target is not None:
        object_count = int(explicit_object)
        target_count = int(explicit_target)
        if int(object_count) not in set(supported_object_counts):
            raise ValueError("object_count is outside configured supported range")
        if int(target_count) not in set(supported_target_counts):
            raise ValueError("target_count is outside configured supported range")
        if int(target_count) > int(object_count):
            raise ValueError("target_count cannot exceed object_count")
        distractor_count = int(object_count) - int(target_count)
        if explicit_distractor is not None and int(explicit_distractor) != int(distractor_count):
            raise ValueError("distractor_count must equal object_count - target_count")
        object_probabilities = _one_hot_probability_map(int(object_count), supported_object_counts)
        target_probabilities = _one_hot_probability_map(int(target_count), supported_target_counts)
        distractor_probabilities = _distractor_probability_map_for_target(
            object_probabilities=object_probabilities,
            target_count=int(target_count),
        )
        return (
            int(object_count),
            object_probabilities,
            int(target_count),
            target_probabilities,
            int(distractor_count),
            distractor_probabilities,
        )

    if explicit_target is not None and explicit_distractor is not None:
        target_count = int(explicit_target)
        distractor_count = int(explicit_distractor)
        object_count = int(target_count) + int(distractor_count)
        if int(target_count) not in set(supported_target_counts):
            raise ValueError("target_count is outside configured supported range")
        if int(object_count) not in set(supported_object_counts):
            raise ValueError("target_count + distractor_count is outside configured supported range")
        object_probabilities = _one_hot_probability_map(int(object_count), supported_object_counts)
        target_probabilities = _one_hot_probability_map(int(target_count), supported_target_counts)
        distractor_probabilities = _distractor_probability_map_for_target(
            object_probabilities=object_probabilities,
            target_count=int(target_count),
        )
        return (
            int(object_count),
            object_probabilities,
            int(target_count),
            target_probabilities,
            int(distractor_count),
            distractor_probabilities,
        )

    target_weights_raw = params.get(
        "target_count_weights",
        gen_defaults.get("target_count_weights", {str(value): 1.0 for value in supported_target_counts}),
    )
    if not isinstance(target_weights_raw, Mapping):
        raise ValueError("target_count_weights must be a mapping when provided")
    supported_target_keys = {str(value) for value in supported_target_counts}
    target_weights = {
        str(key): float(value)
        for key, value in target_weights_raw.items()
        if str(key) in supported_target_keys
    }
    target_probabilities = normalize_positive_weights(
        target_weights,
        default_keys=[str(value) for value in supported_target_counts],
    )

    if explicit_target is not None:
        target_count = int(explicit_target)
        if int(target_count) not in set(supported_target_counts):
            raise ValueError("target_count is outside configured supported range")
    else:
        target_count = int(weighted_choice(rng, target_probabilities, sort_keys=True))

    feasible_object_counts = [int(value) for value in supported_object_counts if int(value) >= int(target_count)]
    if not feasible_object_counts:
        raise ValueError("counting task resolved no feasible object counts for selected target_count")

    if explicit_distractor is not None:
        distractor_count = int(explicit_distractor)
        object_count = int(target_count) + int(distractor_count)
        if int(object_count) not in set(feasible_object_counts):
            raise ValueError("distractor_count is outside configured supported range")
        object_probabilities = _one_hot_probability_map(int(object_count), feasible_object_counts)
        distractor_probabilities = _distractor_probability_map_for_target(
            object_probabilities=object_probabilities,
            target_count=int(target_count),
        )
        return (
            int(object_count),
            _sorted_probability_map(object_probabilities),
            int(target_count),
            _sorted_probability_map(target_probabilities),
            int(distractor_count),
            distractor_probabilities,
        )

    if explicit_object is not None:
        object_count = int(explicit_object)
        if int(object_count) not in set(feasible_object_counts):
            raise ValueError("object_count is outside configured supported range")
        object_probabilities = _one_hot_probability_map(int(object_count), feasible_object_counts)
    else:
        object_weights_raw = params.get(
            "object_count_weights",
            gen_defaults.get("object_count_weights", {str(value): 1.0 for value in supported_object_counts}),
        )
        if not isinstance(object_weights_raw, Mapping):
            raise ValueError("object_count_weights must be a mapping when provided")
        supported_object_keys = {str(value) for value in feasible_object_counts}
        object_weights = {
            str(key): float(value)
            for key, value in object_weights_raw.items()
            if str(key) in supported_object_keys
        }
        object_probabilities = normalize_positive_weights(
            object_weights,
            default_keys=[str(value) for value in feasible_object_counts],
        )
        object_count = int(weighted_choice(rng, object_probabilities, sort_keys=True))

    distractor_count = int(object_count) - int(target_count)
    distractor_probabilities = _distractor_probability_map_for_target(
        object_probabilities=object_probabilities,
        target_count=int(target_count),
    )
    return (
        int(object_count),
        _sorted_probability_map(object_probabilities),
        int(target_count),
        _sorted_probability_map(target_probabilities),
        int(distractor_count),
        distractor_probabilities,
    )


def resolve_counting_target_and_distractor_triplet(
    rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    fallback_total_min: int,
    fallback_total_max: int,
    fallback_target_min: int = 0,
    fallback_target_max: int = 10,
    fallback_distractor_min: int = 1,
    fallback_distractor_max: int = 10,
) -> Tuple[int, Dict[str, float], int, Dict[str, float], int, Dict[str, float]]:
    """Resolve `(object_count, target_count, distractor_count)` from independent supports.

    This is useful for families like icon counting where we want direct control
    over both the answer count and the number of distractors while still
    respecting a total scene-capacity range.
    """

    total_min = int(params.get("object_count_min", gen_defaults.get("object_count_min", int(fallback_total_min))))
    total_max = int(params.get("object_count_max", gen_defaults.get("object_count_max", int(fallback_total_max))))
    if total_min < 1 or total_min > total_max:
        raise ValueError("invalid object_count_min/object_count_max for counting task")

    target_min = int(params.get("target_count_min", gen_defaults.get("target_count_min", int(fallback_target_min))))
    target_max = int(params.get("target_count_max", gen_defaults.get("target_count_max", int(fallback_target_max))))
    target_min = max(0, int(target_min))
    target_max = max(target_min, int(target_max))
    supported_targets = [int(value) for value in range(int(target_min), int(target_max) + 1)]
    if not supported_targets:
        raise ValueError("counting task resolved no supported target counts")

    distractor_min = int(
        params.get("distractor_count_min", gen_defaults.get("distractor_count_min", int(fallback_distractor_min)))
    )
    distractor_max = int(
        params.get("distractor_count_max", gen_defaults.get("distractor_count_max", int(fallback_distractor_max)))
    )
    distractor_min = max(0, int(distractor_min))
    distractor_max = max(distractor_min, int(distractor_max))
    supported_distractors = [int(value) for value in range(int(distractor_min), int(distractor_max) + 1)]
    if not supported_distractors:
        raise ValueError("counting task resolved no supported distractor counts")

    distractor_margin_over_target = int(
        params.get(
            "distractor_margin_over_target",
            gen_defaults.get("distractor_margin_over_target", 0),
        )
    )
    distractor_margin_over_target = max(0, int(distractor_margin_over_target))

    explicit_total = params.get("object_count")
    explicit_target = params.get("target_count")
    explicit_distractor = params.get("distractor_count")

    def _feasible_distractors_for_target(target_count: int) -> list[int]:
        min_distractor_for_target = int(distractor_min)
        if int(distractor_margin_over_target) > 0:
            min_distractor_for_target = max(
                int(min_distractor_for_target),
                int(target_count) + int(distractor_margin_over_target),
            )
        return [
            int(value)
            for value in supported_distractors
            if int(value) >= int(min_distractor_for_target) and total_min <= int(target_count) + int(value) <= total_max
        ]

    def _feasible_targets_for_distractor(distractor_count: int) -> list[int]:
        return [
            int(value)
            for value in supported_targets
            if total_min <= int(value) + int(distractor_count) <= total_max
        ]

    if explicit_total is not None and explicit_target is not None:
        object_count = int(explicit_total)
        target_count = int(explicit_target)
        distractor_count = int(object_count) - int(target_count)
        if not (total_min <= int(object_count) <= total_max):
            raise ValueError("object_count is outside configured supported range")
        if int(target_count) not in set(supported_targets):
            raise ValueError("target_count is outside configured supported range")
        if int(distractor_count) not in set(supported_distractors):
            raise ValueError("derived distractor_count is outside configured supported range")
        if int(distractor_margin_over_target) > 0 and int(distractor_count) < int(target_count) + int(distractor_margin_over_target):
            raise ValueError("derived distractor_count violates configured distractor margin over target")
        if explicit_distractor is not None and int(explicit_distractor) != int(distractor_count):
            raise ValueError("distractor_count must equal object_count - target_count")
        return (
            int(object_count),
            {str(int(object_count)): 1.0},
            int(target_count),
            _one_hot_probability_map(int(target_count), supported_targets),
            int(distractor_count),
            _one_hot_probability_map(int(distractor_count), supported_distractors),
        )

    if explicit_total is not None and explicit_distractor is not None:
        object_count = int(explicit_total)
        distractor_count = int(explicit_distractor)
        target_count = int(object_count) - int(distractor_count)
        if not (total_min <= int(object_count) <= total_max):
            raise ValueError("object_count is outside configured supported range")
        if int(distractor_count) not in set(supported_distractors):
            raise ValueError("distractor_count is outside configured supported range")
        if int(target_count) not in set(supported_targets):
            raise ValueError("derived target_count is outside configured supported range")
        if int(distractor_margin_over_target) > 0 and int(distractor_count) < int(target_count) + int(distractor_margin_over_target):
            raise ValueError("distractor_count violates configured distractor margin over target")
        return (
            int(object_count),
            {str(int(object_count)): 1.0},
            int(target_count),
            _one_hot_probability_map(int(target_count), supported_targets),
            int(distractor_count),
            _one_hot_probability_map(int(distractor_count), supported_distractors),
        )

    if explicit_target is not None and explicit_distractor is not None:
        target_count = int(explicit_target)
        distractor_count = int(explicit_distractor)
        object_count = int(target_count) + int(distractor_count)
        if int(target_count) not in set(supported_targets):
            raise ValueError("target_count is outside configured supported range")
        if int(distractor_count) not in set(supported_distractors):
            raise ValueError("distractor_count is outside configured supported range")
        if int(distractor_margin_over_target) > 0 and int(distractor_count) < int(target_count) + int(distractor_margin_over_target):
            raise ValueError("distractor_count violates configured distractor margin over target")
        if not (total_min <= int(object_count) <= total_max):
            raise ValueError("target_count + distractor_count is outside configured supported range")
        return (
            int(object_count),
            {str(int(object_count)): 1.0},
            int(target_count),
            _one_hot_probability_map(int(target_count), supported_targets),
            int(distractor_count),
            _one_hot_probability_map(int(distractor_count), supported_distractors),
        )

    target_weights_raw = params.get(
        "target_count_weights",
        gen_defaults.get("target_count_weights", {str(value): 1.0 for value in supported_targets}),
    )
    if not isinstance(target_weights_raw, Mapping):
        raise ValueError("target_count_weights must be a mapping when provided")
    target_weights = {
        str(key): float(value)
        for key, value in target_weights_raw.items()
        if str(key) in {str(value) for value in supported_targets}
    }
    target_probabilities = normalize_positive_weights(
        target_weights,
        default_keys=[str(value) for value in supported_targets],
    )
    if explicit_target is not None:
        target_count = int(explicit_target)
        if int(target_count) not in set(supported_targets):
            raise ValueError("target_count is outside configured supported range")
    else:
        target_count = int(weighted_choice(rng, target_probabilities, sort_keys=True))

    feasible_distractors = _feasible_distractors_for_target(int(target_count))
    if not feasible_distractors:
        raise ValueError("counting task resolved no feasible distractor counts for selected target_count")

    distractor_weights_raw = params.get(
        "distractor_count_weights",
        gen_defaults.get("distractor_count_weights", {str(value): 1.0 for value in supported_distractors}),
    )
    if not isinstance(distractor_weights_raw, Mapping):
        raise ValueError("distractor_count_weights must be a mapping when provided")
    distractor_weights = {
        str(key): float(value)
        for key, value in distractor_weights_raw.items()
        if str(key) in {str(value) for value in feasible_distractors}
    }
    distractor_probabilities = normalize_positive_weights(
        distractor_weights,
        default_keys=[str(value) for value in feasible_distractors],
    )
    if explicit_distractor is not None:
        distractor_count = int(explicit_distractor)
        if int(distractor_count) not in set(feasible_distractors):
            raise ValueError("distractor_count is outside configured supported range")
    else:
        distractor_count = int(weighted_choice(rng, distractor_probabilities, sort_keys=True))

    object_count = int(target_count) + int(distractor_count)
    object_probabilities = {
        str(int(target_count) + int(distractor)): float(probability)
        for distractor, probability in (
            (int(key), float(value))
            for key, value in distractor_probabilities.items()
        )
    }

    return (
        int(object_count),
        _sorted_probability_map(object_probabilities),
        int(target_count),
        _sorted_probability_map(target_probabilities),
        int(distractor_count),
        _sorted_probability_map(distractor_probabilities),
    )


__all__ = [
    "resolve_counting_cardinality_pair",
    "resolve_counting_target_and_distractor_triplet",
    "resolve_counting_object_count",
    "resolve_counting_target_first_cardinality_triplet",
    "resolve_counting_target_count",
]
