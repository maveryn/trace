"""Sampling helpers for mirror-grid icon scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from ....shared.config_defaults import group_default
from ....shared.labeling import LABEL_POOL_A_L


SYMMETRY_KINDS: Tuple[str, ...] = (
    "vertical",
    "horizontal",
    "diagonal_main",
    "diagonal_anti",
    "both_axes",
)
NONSYMMETRIC_KIND = "none"


def fixed_grid_labels(object_count: int) -> Tuple[str, ...]:
    """Return fixed row-major labels for the visible scene cells."""

    if int(object_count) <= 0 or int(object_count) > len(LABEL_POOL_A_L):
        raise ValueError("mirror-grid object_count is outside the label support")
    return tuple(str(value) for value in LABEL_POOL_A_L[: int(object_count)])


def probability_map(values: Sequence[str | int], *, selected: str | int | None = None) -> dict[str, float]:
    """Return a JSON-stable probability map for one finite support."""

    support = tuple(str(value) for value in values)
    if not support:
        return {}
    if selected is not None:
        selected_text = str(selected)
        return {str(value): (1.0 if str(value) == selected_text else 0.0) for value in support}
    probability = 1.0 / float(len(support))
    return {str(value): float(probability) for value in support}


def requested_answer_label(params: Mapping[str, Any]) -> str:
    """Return the normalized explicit answer label, if one was requested."""

    return str(params.get("answer_label", "") or params.get("correct_option_label", "")).strip().upper()


def resolve_pool_manifest(
    params: Mapping[str, Any],
    *,
    generation_defaults: Mapping[str, Any],
    fallback: str,
) -> str:
    """Resolve the icon-pool manifest for one mirror-grid scene."""

    return str(params.get("pool_manifest", group_default(generation_defaults, "pool_manifest", str(fallback))))


def option_count_choices(
    params: Mapping[str, Any],
    *,
    generation_defaults: Mapping[str, Any],
    fallback_choices: Sequence[int],
) -> Tuple[int, ...]:
    """Resolve supported mirror-grid option counts."""

    raw = params.get(
        "option_count_choices",
        group_default(generation_defaults, "option_count_choices", list(fallback_choices)),
    )
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("option_count_choices must be a sequence of integers")
    choices = tuple(dict.fromkeys(int(value) for value in raw))
    if not choices:
        raise ValueError("option_count_choices must not be empty")
    if any(int(value) not in (4, 6) for value in choices):
        raise ValueError("mirror-grid option_count_choices currently supports only 4 or 6")
    return choices


def resolve_option_count(
    rng,
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    fallback_choices: Sequence[int],
    explicit_answer_label: str,
    context: str,
) -> Tuple[int, dict[str, float]]:
    """Select 4 or 6 visible option cells."""

    choices = option_count_choices(
        params,
        generation_defaults=generation_defaults,
        fallback_choices=fallback_choices,
    )
    explicit_count = params.get("option_count")
    if explicit_count is not None:
        count = int(explicit_count)
        if count not in choices:
            raise ValueError(f"unsupported option_count for {context}: {count}; supported: {choices}")
        if explicit_answer_label and explicit_answer_label not in fixed_grid_labels(int(count)):
            raise ValueError(f"answer_label {explicit_answer_label!r} is not visible with option_count={count}")
        return count, probability_map(choices, selected=count)

    feasible_choices = choices
    if explicit_answer_label:
        feasible_choices = tuple(count for count in choices if explicit_answer_label in fixed_grid_labels(int(count)))
        if not feasible_choices:
            raise ValueError(f"answer_label {explicit_answer_label!r} is not visible in any supported option count")

    count = int(rng.choice(feasible_choices))
    return count, probability_map(feasible_choices)


def resolve_answer_label(
    rng,
    *,
    params: Mapping[str, Any],
    option_labels: Sequence[str],
    context: str,
) -> Tuple[str, int, dict[str, float]]:
    """Select the single correct option label."""

    labels = tuple(str(label) for label in option_labels)
    explicit_label = requested_answer_label(params)
    if explicit_label:
        if explicit_label not in labels:
            raise ValueError(f"unsupported answer_label for visible options in {context} {labels}: {explicit_label}")
        return explicit_label, int(labels.index(explicit_label)), probability_map(labels, selected=explicit_label)
    answer_label = str(rng.choice(labels))
    return answer_label, int(labels.index(answer_label)), probability_map(labels)


def sample_matching_indices(rng, *, object_count: int, target_count: int) -> Tuple[int, ...]:
    """Sample the scene-cell indices that satisfy the target predicate."""

    if int(target_count) < 0 or int(target_count) > int(object_count):
        raise ValueError("target_count must be within the object_count range")
    return tuple(sorted(int(index) for index in rng.sample(list(range(int(object_count))), int(target_count))))


def sample_distractor_symmetry_kinds(
    rng,
    *,
    reference_symmetry_kind: str,
    distractor_count: int,
) -> Tuple[str, ...]:
    """Sample exact-other or nonsymmetric distractor cell kinds."""

    reference_kind = str(reference_symmetry_kind)
    if reference_kind not in set(SYMMETRY_KINDS):
        raise ValueError(f"unsupported reference symmetry kind: {reference_kind}")
    other_symmetries = [str(value) for value in SYMMETRY_KINDS if str(value) != reference_kind]
    variants: list[str] = []
    if int(distractor_count) >= 1:
        variants.append(str(rng.choice(other_symmetries)))
    if int(distractor_count) >= 2:
        variants.append(NONSYMMETRIC_KIND)
    while len(variants) < int(distractor_count):
        variants.append(str(rng.choice(tuple(other_symmetries) + (NONSYMMETRIC_KIND,))))
    rng.shuffle(variants)
    return tuple(str(value) for value in variants)


__all__ = [
    "NONSYMMETRIC_KIND",
    "SYMMETRY_KINDS",
    "fixed_grid_labels",
    "option_count_choices",
    "probability_map",
    "requested_answer_label",
    "resolve_answer_label",
    "resolve_option_count",
    "resolve_pool_manifest",
    "sample_distractor_symmetry_kinds",
    "sample_matching_indices",
]
