"""Graph-domain label pools backed by shared repo-wide label assets."""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence, Tuple

from ....core.sampling import normalize_positive_weights, weighted_choice
from ...shared.labeling import LABEL_POOL_SAFE_UPPER, assign_random_shuffled_labels, assign_shuffled_labels
from ...shared.name_assets import load_label_manifest


LABEL_POOL_1_20: Tuple[str, ...] = tuple(str(value) for value in range(1, 21))
SUPPORTED_GRAPH_LABEL_VARIANTS: Tuple[str, ...] = ("letters", "numbers", "named")
GRAPH_NAMED_LABEL_MIN_BUCKET_SIZE = 26

GRAPH_LABEL_BUCKET_MANIFESTS: Mapping[str, str] = {
    "people_first": "people/first_names_ssa.txt",
    "people_surnames": "people/surnames_census_2010.txt",
    "places_countries": "places/countries_natural_earth.txt",
    "places_cities": "places/cities_natural_earth.txt",
    "organizations_tickers": "organizations/company_tickers_sec.txt",
    "organizations_terms": "organizations/company_terms_sec.txt",
    "occupations": "occupations/occupations_bls_oews.txt",
    "industries": "industries/industries_bls_qcew.txt",
    "mixed_proper": "mixed/proper_labels.txt",
    "mixed_compact": "mixed/compact_labels.txt",
}


@dataclass(frozen=True)
class ResolvedGraphLabels:
    """Resolved visible labels plus traceable asset metadata."""

    labels: Tuple[str, ...]
    label_variant: str
    label_source_kind: str
    label_bucket: str
    label_manifest: str
    label_filter: Mapping[str, Any]
    label_bucket_probabilities: Mapping[str, float]


def default_graph_label_bucket_weights() -> Mapping[str, float]:
    """Return equal default weights over all graph-safe label buckets."""

    return {str(bucket): 1.0 for bucket in GRAPH_LABEL_BUCKET_MANIFESTS}


def _dedupe_preserve_order(values: Sequence[str], *, lowercase: bool) -> Tuple[str, ...]:
    """Return unique labels after optional lowercasing."""

    labels: list[str] = []
    seen: set[str] = set()
    for value in values:
        label = str(value).strip()
        if bool(lowercase):
            label = str(label).lower()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return tuple(labels)


@lru_cache(maxsize=512)
def _graph_label_bucket_pool(
    manifest_name: str,
    *,
    min_chars: int | None,
    max_chars: int | None,
    lowercase: bool,
) -> Tuple[str, ...]:
    """Load one graph-safe manifest pool under compact label constraints."""

    labels = load_label_manifest(
        str(manifest_name),
        min_chars=min_chars,
        max_chars=max_chars,
        allow_spaces=False,
        allow_punctuation=False,
        ascii_only=True,
    )
    return _dedupe_preserve_order(labels, lowercase=bool(lowercase))


def _eligible_named_label_buckets(
    *,
    object_count: int,
    min_chars: int | None,
    max_chars: int | None,
    lowercase: bool,
    min_bucket_size: int,
) -> Mapping[str, Tuple[str, ...]]:
    """Return buckets with enough filtered labels for one instance."""

    required_count = max(int(object_count), int(min_bucket_size))
    eligible: dict[str, Tuple[str, ...]] = {}
    for bucket, manifest_name in GRAPH_LABEL_BUCKET_MANIFESTS.items():
        try:
            labels = _graph_label_bucket_pool(
                str(manifest_name),
                min_chars=min_chars,
                max_chars=max_chars,
                lowercase=bool(lowercase),
            )
        except Exception:
            continue
        if len(labels) >= int(required_count):
            eligible[str(bucket)] = tuple(str(label) for label in labels)
    return dict(eligible)


def _resolve_bucket_probabilities(
    *,
    eligible_buckets: Sequence[str],
    bucket_weights: Mapping[str, float] | None,
) -> Mapping[str, float]:
    """Normalize configured bucket weights over currently eligible buckets."""

    eligible = tuple(str(bucket) for bucket in eligible_buckets)
    raw_weights = (
        {str(key): float(value) for key, value in bucket_weights.items()}
        if isinstance(bucket_weights, Mapping)
        else default_graph_label_bucket_weights()
    )
    candidate_weights = {str(bucket): float(raw_weights.get(str(bucket), 1.0)) for bucket in eligible}
    return normalize_positive_weights(candidate_weights, default_keys=eligible)


def resolve_graph_node_labels(
    rng: random.Random,
    *,
    label_variant: str,
    object_count: int,
    max_chars: int,
    min_chars: int | None = None,
    bucket_weights: Mapping[str, float] | None = None,
    min_bucket_size: int = GRAPH_NAMED_LABEL_MIN_BUCKET_SIZE,
    sequential_numbers: bool = False,
) -> ResolvedGraphLabels:
    """Resolve graph node/station labels for letters, numbers, or named labels."""

    variant = str(label_variant).strip()
    if variant not in set(SUPPORTED_GRAPH_LABEL_VARIANTS):
        raise ValueError(f"unsupported graph label_variant: {label_variant}")
    count = int(object_count)
    if count <= 0:
        raise ValueError("object_count must be positive")
    label_filter = {
        "min_chars": min_chars,
        "max_chars": int(max_chars),
        "allow_spaces": False,
        "allow_punctuation": False,
        "ascii_only": True,
        "min_bucket_size": int(min_bucket_size),
    }
    if variant == "letters":
        labels = assign_random_shuffled_labels(rng, object_count=count, label_pool=LABEL_POOL_SAFE_UPPER)
        return ResolvedGraphLabels(
            labels=tuple(str(label) for label in labels),
            label_variant="letters",
            label_source_kind="letters",
            label_bucket="",
            label_manifest="",
            label_filter=dict(label_filter),
            label_bucket_probabilities={},
        )
    if variant == "numbers":
        assign = assign_shuffled_labels if bool(sequential_numbers) else assign_random_shuffled_labels
        labels = assign(rng, object_count=count, label_pool=LABEL_POOL_1_20)
        return ResolvedGraphLabels(
            labels=tuple(str(label) for label in labels),
            label_variant="numbers",
            label_source_kind="numbers",
            label_bucket="",
            label_manifest="",
            label_filter=dict(label_filter),
            label_bucket_probabilities={},
        )

    eligible = _eligible_named_label_buckets(
        object_count=count,
        min_chars=min_chars,
        max_chars=int(max_chars),
        lowercase=False,
        min_bucket_size=int(min_bucket_size),
    )
    if not eligible:
        raise ValueError("no named graph label bucket satisfies the requested constraints")
    probabilities = _resolve_bucket_probabilities(
        eligible_buckets=tuple(eligible.keys()),
        bucket_weights=bucket_weights,
    )
    bucket = weighted_choice(rng, probabilities, sort_keys=True)
    pool = tuple(str(label) for label in eligible[str(bucket)])
    labels = tuple(str(label) for label in rng.sample(list(pool), k=count))
    return ResolvedGraphLabels(
        labels=tuple(str(label) for label in labels),
        label_variant="named",
        label_source_kind="shared_label_manifest",
        label_bucket=str(bucket),
        label_manifest=str(GRAPH_LABEL_BUCKET_MANIFESTS[str(bucket)]),
        label_filter=dict(label_filter),
        label_bucket_probabilities=dict(probabilities),
    )


def resolve_graph_edge_label_support(
    rng: random.Random,
    *,
    support_size: int,
    min_chars: int = 3,
    max_chars: int = 12,
    bucket_weights: Mapping[str, float] | None = None,
    min_bucket_size: int = GRAPH_NAMED_LABEL_MIN_BUCKET_SIZE,
    exclude_labels: Sequence[str] = (),
) -> ResolvedGraphLabels:
    """Resolve visible text labels for edge-attribute graph tasks."""

    size = int(support_size)
    if size < 2:
        raise ValueError("edge label support_size must be at least 2")
    excluded = {
        str(label).strip().lower()
        for label in exclude_labels
        if str(label).strip()
    }
    eligible = _eligible_named_label_buckets(
        object_count=size,
        min_chars=int(min_chars),
        max_chars=int(max_chars),
        lowercase=True,
        min_bucket_size=int(min_bucket_size),
    )
    if excluded:
        required_count = max(int(size), int(min_bucket_size))
        eligible = {
            str(bucket): tuple(str(label) for label in labels if str(label).strip().lower() not in excluded)
            for bucket, labels in eligible.items()
        }
        eligible = {
            str(bucket): tuple(labels)
            for bucket, labels in eligible.items()
            if len(labels) >= int(required_count)
        }
    if not eligible:
        raise ValueError("no edge label bucket satisfies the requested constraints")
    probabilities = _resolve_bucket_probabilities(
        eligible_buckets=tuple(eligible.keys()),
        bucket_weights=bucket_weights,
    )
    bucket = weighted_choice(rng, probabilities, sort_keys=True)
    pool = tuple(str(label) for label in eligible[str(bucket)])
    labels = tuple(str(label) for label in rng.sample(list(pool), k=size))
    return ResolvedGraphLabels(
        labels=tuple(str(label) for label in labels),
        label_variant="named",
        label_source_kind="shared_label_manifest",
        label_bucket=str(bucket),
        label_manifest=str(GRAPH_LABEL_BUCKET_MANIFESTS[str(bucket)]),
        label_filter={
            "min_chars": int(min_chars),
            "max_chars": int(max_chars),
            "allow_spaces": False,
            "allow_punctuation": False,
            "ascii_only": True,
            "lowercase": True,
            "min_bucket_size": int(min_bucket_size),
        },
        label_bucket_probabilities=dict(probabilities),
    )


__all__ = [
    "GRAPH_LABEL_BUCKET_MANIFESTS",
    "GRAPH_NAMED_LABEL_MIN_BUCKET_SIZE",
    "LABEL_POOL_1_20",
    "ResolvedGraphLabels",
    "SUPPORTED_GRAPH_LABEL_VARIANTS",
    "default_graph_label_bucket_weights",
    "resolve_graph_edge_label_support",
    "resolve_graph_node_labels",
]
