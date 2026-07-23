"""Chart-domain label pools backed by shared repo-wide label assets."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from typing import Any, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.sampling import normalize_positive_weights, weighted_choice
from ...shared.labeling import LABEL_POOL_SAFE_UPPER, assign_random_shuffled_labels
from ...shared.name_assets import load_label_manifest


CHART_LABEL_POOL_UP_TO_25: Tuple[str, ...] = tuple("ABCDEFGHIJKLMNPQRSTUVWXYZ")
CHART_NAMED_LABEL_MIN_BUCKET_SIZE = 20
CHART_COMPACT_ID_FIRST_CHARS: Tuple[str, ...] = tuple("ABCDEFGHJKLMNPQRSTUVWXYZ")
CHART_COMPACT_ID_DIGITS: Tuple[str, ...] = tuple("23456789")
CHART_COMPACT_ID_ALPHANUM: Tuple[str, ...] = CHART_COMPACT_ID_FIRST_CHARS + CHART_COMPACT_ID_DIGITS

CHART_ENTITY_LABEL_BUCKET_MANIFESTS: Mapping[str, str] = {
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

CHART_GENERATED_ENTITY_LABEL_BUCKETS: Tuple[str, ...] = ("temporal",)

CHART_CATEGORY_LABEL_BUCKET_MANIFESTS: Mapping[str, str] = {
    "categories_abstract": "categories/abstract_group_labels.txt",
    "categories_priority": "categories/priority_labels.txt",
    "categories_product": "categories/product_labels.txt",
    "categories_status": "categories/status_labels.txt",
}

CHART_PANEL_LABEL_BUCKET_MANIFESTS: Mapping[str, str] = {
    "named_compact": "mixed/compact_labels.txt",
    "report_topics": "categories/abstract_group_labels.txt",
    "technical_topics": "panel_titles/technical_topics.txt",
}

CHART_ALL_LABEL_BUCKET_MANIFESTS: Mapping[str, str] = {
    **CHART_ENTITY_LABEL_BUCKET_MANIFESTS,
    **CHART_CATEGORY_LABEL_BUCKET_MANIFESTS,
}

SUPPORTED_CHART_LABEL_VARIANTS: Tuple[str, ...] = ("letters", "named")
SUPPORTED_CHART_LABEL_POOL_KINDS: Tuple[str, ...] = ("all", "entity", "category")
SUPPORTED_CHART_PANEL_LABEL_VARIANTS: Tuple[str, ...] = (
    "temporal_sequence",
    "subplot_letters",
    "named_compact",
    "report_topics",
    "condition_labels",
    "technical_topics",
)
CHART_TEMPORAL_LABEL_SUBTYPES: Tuple[str, ...] = (
    "year",
    "fiscal_year",
    "quarter",
    "month",
    "month_day",
)
CHART_CONDITION_PANEL_LABEL_POOL: Tuple[str, ...] = (
    "Baseline",
    "Control",
    "Reference",
    "Target",
    "TrialA",
    "TrialB",
    "TrialC",
    "RunA",
    "RunB",
    "RunC",
    "GroupA",
    "GroupB",
    "GroupC",
    "VariantA",
    "VariantB",
    "VariantC",
    "Phase1",
    "Phase2",
    "Phase3",
    "BatchA",
    "BatchB",
    "BatchC",
    "CohortA",
    "CohortB",
    "CohortC",
    "CaseA",
    "CaseB",
    "CaseC",
    "SampleA",
    "SampleB",
    "SampleC",
    "ModelA",
    "ModelB",
    "ModelC",
    "SiteA",
    "SiteB",
    "SiteC",
    "UnitA",
    "UnitB",
    "UnitC",
)
DEFAULT_CHART_PANEL_LABEL_VARIANT_WEIGHTS: Mapping[str, float] = {
    "temporal_sequence": 0.5,
    "subplot_letters": 0.75,
    "named_compact": 1.0,
    "report_topics": 1.0,
    "condition_labels": 0.75,
    "technical_topics": 1.0,
}


@dataclass(frozen=True)
class ResolvedChartLabels:
    """Resolved visible chart labels plus traceable asset metadata."""

    labels: Tuple[str, ...]
    label_variant: str
    label_pool_kind: str
    label_source_kind: str
    label_bucket: str
    label_manifest: str
    label_filter: Mapping[str, Any]
    label_bucket_probabilities: Mapping[str, float]


def _bucket_manifests_for_kind(label_pool_kind: str) -> Mapping[str, str]:
    kind = str(label_pool_kind).strip().lower()
    if kind == "all":
        return CHART_ALL_LABEL_BUCKET_MANIFESTS
    if kind == "entity":
        return CHART_ENTITY_LABEL_BUCKET_MANIFESTS
    if kind == "category":
        return CHART_CATEGORY_LABEL_BUCKET_MANIFESTS
    raise ValueError(f"unsupported chart label_pool_kind: {label_pool_kind}")


def _bucket_names_for_kind(label_pool_kind: str) -> Tuple[str, ...]:
    """Return manifest-backed and generated bucket names for one label kind."""

    manifest_names = tuple(str(bucket) for bucket in _bucket_manifests_for_kind(str(label_pool_kind)))
    kind = str(label_pool_kind).strip().lower()
    if kind in {"all", "entity"}:
        return tuple(manifest_names) + tuple(CHART_GENERATED_ENTITY_LABEL_BUCKETS)
    return tuple(manifest_names)


def default_chart_label_bucket_weights(label_pool_kind: str = "all") -> Mapping[str, float]:
    """Return equal default weights over eligible chart label buckets."""

    return {str(bucket): 1.0 for bucket in _bucket_names_for_kind(str(label_pool_kind))}


def _dedupe_preserve_order(values: Sequence[str], *, lowercase: bool) -> Tuple[str, ...]:
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


def normalize_chart_label_for_collision(label: str) -> str:
    """Normalize a visible chart label for cross-role collision checks."""

    return " ".join(str(label).strip().casefold().split())


def _filter_reserved_chart_labels(labels: Sequence[str], reserved_labels: Sequence[str] | None) -> Tuple[str, ...]:
    reserved = {
        normalize_chart_label_for_collision(str(label))
        for label in (reserved_labels or ())
        if normalize_chart_label_for_collision(str(label))
    }
    if not reserved:
        return tuple(str(label) for label in labels)
    return tuple(
        str(label)
        for label in labels
        if normalize_chart_label_for_collision(str(label)) not in reserved
    )


def validate_chart_label_namespaces(
    *,
    panel_labels: Sequence[str],
    other_label_groups: Mapping[str, Sequence[str]] | None = None,
    context: str = "chart_panel_labels",
) -> Mapping[str, Any]:
    """Validate that panel labels do not reuse legend/category/series labels."""

    groups = other_label_groups or {}
    panel_norms: dict[str, str] = {}
    for label in panel_labels:
        normalized = normalize_chart_label_for_collision(str(label))
        if not normalized:
            continue
        if normalized in panel_norms:
            raise ValueError(f"{context} has duplicate panel label after normalization: {label!r}")
        panel_norms[normalized] = str(label)

    overlaps: list[dict[str, str]] = []
    for group_name, labels in groups.items():
        for label in labels:
            normalized = normalize_chart_label_for_collision(str(label))
            if not normalized:
                continue
            if normalized in panel_norms:
                overlaps.append(
                    {
                        "panel_label": str(panel_norms[normalized]),
                        "group": str(group_name),
                        "other_label": str(label),
                    }
                )
    if overlaps:
        raise ValueError(f"{context} panel labels collide with non-panel labels: {overlaps!r}")
    return {
        "checked_scope": str(context),
        "normalization": "strip_casefold_collapse_whitespace",
        "panel_count": int(len(tuple(panel_labels))),
        "other_label_count": int(sum(len(tuple(labels)) for labels in groups.values())),
        "overlap_count": 0,
        "other_label_groups": sorted(str(group) for group in groups),
    }


@lru_cache(maxsize=16)
def _compact_chart_id_pool(*, min_chars: int, max_chars: int) -> Tuple[str, ...]:
    """Return a large synthetic pool for dense chart identifiers.

    Dense categorical axes are layout-constrained and do not need semantic
    names. Use short unambiguous IDs instead of repeatedly sampling from the
    tiny short-word category buckets.
    """

    lower = max(1, int(min_chars))
    upper = max(lower, min(4, int(max_chars)))
    labels: list[str] = []
    if lower <= 1 <= upper:
        labels.extend(str(char) for char in CHART_COMPACT_ID_FIRST_CHARS)
    if lower <= 2 <= upper:
        labels.extend(
            f"{first}{digit}"
            for first in CHART_COMPACT_ID_FIRST_CHARS
            for digit in CHART_COMPACT_ID_DIGITS
        )
    if lower <= 3 <= upper:
        labels.extend(
            f"{first}{second}{digit}"
            for first in CHART_COMPACT_ID_FIRST_CHARS
            for second in CHART_COMPACT_ID_FIRST_CHARS
            for digit in CHART_COMPACT_ID_DIGITS
        )
    if lower <= 4 <= upper:
        labels.extend(
            f"{first}{digit_a}{second}{digit_b}"
            for first in CHART_COMPACT_ID_FIRST_CHARS
            for digit_a in CHART_COMPACT_ID_DIGITS
            for second in CHART_COMPACT_ID_FIRST_CHARS
            for digit_b in CHART_COMPACT_ID_DIGITS
        )
    return tuple(labels)


def resolve_chart_compact_axis_labels(
    rng: random.Random,
    *,
    count: int,
    min_chars: int = 2,
    max_chars: int = 4,
) -> ResolvedChartLabels:
    """Resolve short synthetic identifiers for dense chart axes and marks."""

    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    pool = _compact_chart_id_pool(min_chars=int(min_chars), max_chars=int(max_chars))
    if len(pool) < int(count):
        raise ValueError("compact chart id pool is too small for requested count")
    labels = tuple(str(label) for label in rng.sample(list(pool), k=count))
    return ResolvedChartLabels(
        labels=tuple(labels),
        label_variant="compact_id",
        label_pool_kind="synthetic",
        label_source_kind="synthetic_compact_id",
        label_bucket="dense_axis_compact_id",
        label_manifest="",
        label_filter={
            "min_chars": int(min_chars),
            "max_chars": int(max_chars),
            "allow_spaces": False,
            "allow_punctuation": False,
            "ascii_only": True,
            "compact_length": True,
            "lowercase": False,
            "min_bucket_size": int(count),
        },
        label_bucket_probabilities={"dense_axis_compact_id": 1.0},
    )


@lru_cache(maxsize=512)
def _chart_label_bucket_pool(
    manifest_name: str,
    *,
    min_chars: int | None,
    max_chars: int | None,
    allow_spaces: bool,
    allow_punctuation: bool,
    compact_length: bool,
    lowercase: bool,
) -> Tuple[str, ...]:
    labels = load_label_manifest(
        str(manifest_name),
        min_chars=min_chars,
        max_chars=max_chars,
        allow_spaces=bool(allow_spaces),
        allow_punctuation=bool(allow_punctuation),
        ascii_only=True,
        compact_length=bool(compact_length),
    )
    return _dedupe_preserve_order(labels, lowercase=bool(lowercase))


def _eligible_chart_label_buckets(
    *,
    bucket_manifests: Mapping[str, str],
    object_count: int,
    min_chars: int | None,
    max_chars: int | None,
    allow_spaces: bool,
    allow_punctuation: bool,
    compact_length: bool,
    lowercase: bool,
    min_bucket_size: int,
) -> Mapping[str, Tuple[str, ...]]:
    required_count = max(int(object_count), int(min_bucket_size))
    eligible: dict[str, Tuple[str, ...]] = {}
    for bucket, manifest_name in bucket_manifests.items():
        try:
            labels = _chart_label_bucket_pool(
                str(manifest_name),
                min_chars=min_chars,
                max_chars=max_chars,
                allow_spaces=bool(allow_spaces),
                allow_punctuation=bool(allow_punctuation),
                compact_length=bool(compact_length),
                lowercase=bool(lowercase),
            )
        except Exception:
            continue
        if len(labels) >= int(required_count):
            eligible[str(bucket)] = tuple(str(label) for label in labels)
    return dict(eligible)


def _label_satisfies_temporal_filter(
    label: str,
    *,
    min_chars: int | None,
    max_chars: int | None,
    allow_spaces: bool,
    allow_punctuation: bool,
) -> bool:
    value = str(label)
    if min_chars is not None and len(value) < int(min_chars):
        return False
    if max_chars is not None and len(value) > int(max_chars):
        return False
    if not bool(allow_spaces) and any(char.isspace() for char in value):
        return False
    if not bool(allow_punctuation) and any(not char.isalnum() and not char.isspace() for char in value):
        return False
    return True


def _format_temporal_label(value: str, *, lowercase: bool) -> str:
    label = str(value)
    return label.lower() if bool(lowercase) else label


def _temporal_labels_for_subtype(
    rng: random.Random,
    *,
    count: int,
    subtype: str,
    allow_punctuation: bool,
    lowercase: bool,
) -> Tuple[str, ...]:
    count = int(count)
    kind = str(subtype)
    if kind == "year":
        start_max = max(1995, 2035 - int(count) + 1)
        start = int(rng.randint(1995, start_max))
        labels = [str(start + index) for index in range(count)]
    elif kind == "fiscal_year":
        start_max = max(18, 35 - int(count) + 1)
        start = int(rng.randint(18, start_max))
        labels = [f"FY{start + index:02d}" for index in range(count)]
    elif kind == "quarter":
        start = int(rng.randint(20 * 4, 30 * 4))
        labels = []
        for offset in range(count):
            absolute = int(start + offset)
            year = int(absolute // 4)
            quarter = int(absolute % 4) + 1
            labels.append(f"{year:02d}Q{quarter}")
    elif kind == "month":
        month_names = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
        if count <= len(month_names):
            start = int(rng.randint(0, len(month_names) - count))
            labels = [str(month_names[start + index]) for index in range(count)]
        else:
            start = int(rng.randint(20 * 12, 28 * 12))
            labels = []
            for offset in range(count):
                absolute = int(start + offset)
                year = int(absolute // 12)
                month = int(absolute % 12)
                labels.append(f"{month_names[month]}{year:02d}")
    elif kind == "month_day":
        start_day = date(2024, 1, 1) + timedelta(days=int(rng.randint(0, 220)))
        labels = []
        for offset in range(count):
            current = start_day + timedelta(days=7 * int(offset))
            if bool(allow_punctuation):
                labels.append(current.strftime("%m-%d"))
            else:
                labels.append(current.strftime("%b%d"))
    else:
        raise ValueError(f"unsupported temporal chart label subtype: {subtype}")
    return tuple(_format_temporal_label(str(label), lowercase=bool(lowercase)) for label in labels)


def _eligible_temporal_label_subtypes(
    *,
    object_count: int,
    min_chars: int | None,
    max_chars: int | None,
    allow_spaces: bool,
    allow_punctuation: bool,
    lowercase: bool,
) -> Tuple[str, ...]:
    # Fixed-seed dry runs are enough here because temporal subtypes have stable
    # format lengths regardless of the sampled start point.
    probe_rng = random.Random(173)
    eligible: list[str] = []
    for subtype in CHART_TEMPORAL_LABEL_SUBTYPES:
        labels = _temporal_labels_for_subtype(
            probe_rng,
            count=int(object_count),
            subtype=str(subtype),
            allow_punctuation=bool(allow_punctuation),
            lowercase=bool(lowercase),
        )
        if len(set(labels)) != int(object_count):
            continue
        if all(
            _label_satisfies_temporal_filter(
                str(label),
                min_chars=min_chars,
                max_chars=max_chars,
                allow_spaces=bool(allow_spaces),
                allow_punctuation=bool(allow_punctuation),
            )
            for label in labels
        ):
            eligible.append(str(subtype))
    return tuple(eligible)


def _eligible_generated_chart_label_buckets(
    *,
    label_pool_kind: str,
    object_count: int,
    min_chars: int | None,
    max_chars: int | None,
    allow_spaces: bool,
    allow_punctuation: bool,
    lowercase: bool,
) -> Mapping[str, Tuple[str, ...]]:
    kind = str(label_pool_kind).strip().lower()
    if kind not in {"all", "entity"}:
        return {}
    temporal_subtypes = _eligible_temporal_label_subtypes(
        object_count=int(object_count),
        min_chars=min_chars,
        max_chars=max_chars,
        allow_spaces=bool(allow_spaces),
        allow_punctuation=bool(allow_punctuation),
        lowercase=bool(lowercase),
    )
    if not temporal_subtypes:
        return {}
    return {"temporal": tuple(temporal_subtypes)}


def _resolve_generated_chart_labels(
    rng: random.Random,
    *,
    bucket: str,
    count: int,
    eligible_subtypes: Sequence[str],
    allow_punctuation: bool,
    lowercase: bool,
) -> tuple[Tuple[str, ...], Mapping[str, Any]]:
    if str(bucket) != "temporal":
        raise ValueError(f"unsupported generated chart label bucket: {bucket}")
    subtype = weighted_choice(
        rng,
        {str(value): 1.0 / float(len(eligible_subtypes)) for value in eligible_subtypes},
        sort_keys=True,
    )
    labels = _temporal_labels_for_subtype(
        rng,
        count=int(count),
        subtype=str(subtype),
        allow_punctuation=bool(allow_punctuation),
        lowercase=bool(lowercase),
    )
    return tuple(str(label) for label in labels), {"temporal_subtype": str(subtype)}


def _resolve_bucket_probabilities(
    *,
    eligible_buckets: Sequence[str],
    bucket_weights: Mapping[str, float] | None,
    label_pool_kind: str,
) -> Mapping[str, float]:
    eligible = tuple(str(bucket) for bucket in eligible_buckets)
    raw_weights = (
        {str(key): float(value) for key, value in bucket_weights.items()}
        if isinstance(bucket_weights, Mapping)
        else default_chart_label_bucket_weights(str(label_pool_kind))
    )
    candidate_weights = {
        str(bucket): float(raw_weights.get(str(bucket), 0.0 if isinstance(bucket_weights, Mapping) else 1.0))
        for bucket in eligible
    }
    return normalize_positive_weights(candidate_weights, default_keys=eligible)


def resolve_chart_text_labels(
    rng: random.Random,
    *,
    count: int,
    label_variant: str = "named",
    label_pool_kind: str = "all",
    min_chars: int | None = None,
    max_chars: int = 12,
    allow_spaces: bool = True,
    allow_punctuation: bool = False,
    compact_length: bool = True,
    lowercase: bool = False,
    bucket_weights: Mapping[str, float] | None = None,
    min_bucket_size: int = CHART_NAMED_LABEL_MIN_BUCKET_SIZE,
) -> ResolvedChartLabels:
    """Resolve visible chart labels from letters or shared named-label assets."""

    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    variant = str(label_variant).strip().lower()
    if variant not in set(SUPPORTED_CHART_LABEL_VARIANTS):
        raise ValueError(f"unsupported chart label_variant: {label_variant}")
    pool_kind = str(label_pool_kind).strip().lower()
    label_filter = {
        "min_chars": min_chars,
        "max_chars": int(max_chars),
        "allow_spaces": bool(allow_spaces),
        "allow_punctuation": bool(allow_punctuation),
        "ascii_only": True,
        "compact_length": bool(compact_length),
        "lowercase": bool(lowercase),
        "min_bucket_size": int(min_bucket_size),
    }
    if variant == "letters":
        label_pool = LABEL_POOL_SAFE_UPPER if count <= len(LABEL_POOL_SAFE_UPPER) else CHART_LABEL_POOL_UP_TO_25
        labels = assign_random_shuffled_labels(rng, object_count=count, label_pool=label_pool)
        return ResolvedChartLabels(
            labels=tuple(str(label) for label in labels),
            label_variant="letters",
            label_pool_kind=pool_kind,
            label_source_kind="letters",
            label_bucket="",
            label_manifest="",
            label_filter=dict(label_filter),
            label_bucket_probabilities={},
        )

    bucket_manifests = _bucket_manifests_for_kind(pool_kind)
    eligible = _eligible_chart_label_buckets(
        bucket_manifests=bucket_manifests,
        object_count=count,
        min_chars=min_chars,
        max_chars=int(max_chars),
        allow_spaces=bool(allow_spaces),
        allow_punctuation=bool(allow_punctuation),
        compact_length=bool(compact_length),
        lowercase=bool(lowercase),
        min_bucket_size=int(min_bucket_size),
    )
    generated_eligible = _eligible_generated_chart_label_buckets(
        label_pool_kind=pool_kind,
        object_count=count,
        min_chars=min_chars,
        max_chars=int(max_chars),
        allow_spaces=bool(allow_spaces),
        allow_punctuation=bool(allow_punctuation),
        lowercase=bool(lowercase),
    )
    all_eligible = {**eligible, **generated_eligible}
    if not all_eligible:
        raise ValueError("no chart label bucket satisfies the requested constraints")
    probabilities = _resolve_bucket_probabilities(
        eligible_buckets=tuple(all_eligible.keys()),
        bucket_weights=bucket_weights,
        label_pool_kind=pool_kind,
    )
    bucket = weighted_choice(rng, probabilities, sort_keys=True)
    if str(bucket) in generated_eligible:
        labels, generated_filter = _resolve_generated_chart_labels(
            rng,
            bucket=str(bucket),
            count=int(count),
            eligible_subtypes=generated_eligible[str(bucket)],
            allow_punctuation=bool(allow_punctuation),
            lowercase=bool(lowercase),
        )
        return ResolvedChartLabels(
            labels=tuple(str(label) for label in labels),
            label_variant="named",
            label_pool_kind=pool_kind,
            label_source_kind="synthetic_temporal_sequence",
            label_bucket=str(bucket),
            label_manifest="",
            label_filter={**dict(label_filter), **dict(generated_filter)},
            label_bucket_probabilities=dict(probabilities),
        )
    pool = tuple(str(label) for label in eligible[str(bucket)])
    labels = tuple(str(label) for label in rng.sample(list(pool), k=count))
    return ResolvedChartLabels(
        labels=tuple(str(label) for label in labels),
        label_variant="named",
        label_pool_kind=pool_kind,
        label_source_kind="shared_label_manifest",
        label_bucket=str(bucket),
        label_manifest=str(bucket_manifests[str(bucket)]),
        label_filter=dict(label_filter),
        label_bucket_probabilities=dict(probabilities),
    )


def resolve_chart_axis_labels(
    rng: random.Random,
    *,
    count: int,
    min_chars: int = 2,
    max_chars: int = 4,
    bucket_weights: Mapping[str, float] | None = None,
) -> ResolvedChartLabels:
    """Resolve compact ordered-axis labels with optional semantic buckets."""

    weights = (
        {str(key): float(value) for key, value in bucket_weights.items()}
        if isinstance(bucket_weights, Mapping)
        else {"dense_axis_compact_id": 1.0, **default_chart_label_bucket_weights("entity")}
    )
    eligible_buckets: dict[str, Tuple[str, ...]] = {}
    compact_pool = _compact_chart_id_pool(min_chars=int(min_chars), max_chars=int(max_chars))
    if len(compact_pool) >= int(count):
        eligible_buckets["dense_axis_compact_id"] = tuple(compact_pool)
    entity_manifests = _bucket_manifests_for_kind("entity")
    entity_eligible = _eligible_chart_label_buckets(
        bucket_manifests=entity_manifests,
        object_count=int(count),
        min_chars=int(min_chars),
        max_chars=int(max_chars),
        allow_spaces=False,
        allow_punctuation=False,
        compact_length=True,
        lowercase=False,
        min_bucket_size=max(CHART_NAMED_LABEL_MIN_BUCKET_SIZE, int(count)),
    )
    generated_eligible = _eligible_generated_chart_label_buckets(
        label_pool_kind="entity",
        object_count=int(count),
        min_chars=int(min_chars),
        max_chars=int(max_chars),
        allow_spaces=False,
        allow_punctuation=False,
        lowercase=False,
    )
    eligible_buckets.update(entity_eligible)
    eligible_buckets.update(generated_eligible)
    probabilities = normalize_positive_weights(
        {bucket: float(weights.get(str(bucket), 1.0)) for bucket in eligible_buckets},
        default_keys=tuple(eligible_buckets.keys()),
    )
    bucket = weighted_choice(rng, probabilities, sort_keys=True)
    if str(bucket) == "dense_axis_compact_id":
        resolved = resolve_chart_compact_axis_labels(
            rng,
            count=int(count),
            min_chars=int(min_chars),
            max_chars=int(max_chars),
        )
        return ResolvedChartLabels(
            labels=tuple(resolved.labels),
            label_variant="axis",
            label_pool_kind="axis",
            label_source_kind=str(resolved.label_source_kind),
            label_bucket=str(resolved.label_bucket),
            label_manifest=str(resolved.label_manifest),
            label_filter=dict(resolved.label_filter),
            label_bucket_probabilities=dict(probabilities),
        )
    if str(bucket) in generated_eligible:
        labels, generated_filter = _resolve_generated_chart_labels(
            rng,
            bucket=str(bucket),
            count=int(count),
            eligible_subtypes=generated_eligible[str(bucket)],
            allow_punctuation=False,
            lowercase=False,
        )
        return ResolvedChartLabels(
            labels=tuple(str(label) for label in labels),
            label_variant="axis",
            label_pool_kind="axis",
            label_source_kind="synthetic_temporal_sequence",
            label_bucket=str(bucket),
            label_manifest="",
            label_filter={
                "min_chars": int(min_chars),
                "max_chars": int(max_chars),
                "allow_spaces": False,
                "allow_punctuation": False,
                "ascii_only": True,
                "compact_length": True,
                "lowercase": False,
                "min_bucket_size": int(count),
                **dict(generated_filter),
            },
            label_bucket_probabilities=dict(probabilities),
        )
    pool = tuple(str(label) for label in entity_eligible[str(bucket)])
    labels = tuple(str(label) for label in rng.sample(list(pool), k=int(count)))
    return ResolvedChartLabels(
        labels=tuple(str(label) for label in labels),
        label_variant="axis",
        label_pool_kind="axis",
        label_source_kind="shared_label_manifest",
        label_bucket=str(bucket),
        label_manifest=str(entity_manifests[str(bucket)]),
        label_filter={
            "min_chars": int(min_chars),
            "max_chars": int(max_chars),
            "allow_spaces": False,
            "allow_punctuation": False,
            "ascii_only": True,
            "compact_length": True,
            "lowercase": False,
            "min_bucket_size": max(CHART_NAMED_LABEL_MIN_BUCKET_SIZE, int(count)),
        },
        label_bucket_probabilities=dict(probabilities),
    )


def _condition_panel_label_pool(
    *,
    min_chars: int | None,
    max_chars: int | None,
    allow_spaces: bool,
    allow_punctuation: bool,
    lowercase: bool,
    reserved_labels: Sequence[str] | None,
) -> Tuple[str, ...]:
    labels = [
        _format_temporal_label(str(label), lowercase=bool(lowercase))
        for label in CHART_CONDITION_PANEL_LABEL_POOL
        if _label_satisfies_temporal_filter(
            str(label),
            min_chars=min_chars,
            max_chars=max_chars,
            allow_spaces=bool(allow_spaces),
            allow_punctuation=bool(allow_punctuation),
        )
    ]
    return _filter_reserved_chart_labels(labels, reserved_labels)


def _subplot_letter_panel_label_pool(
    *,
    count: int,
    min_chars: int | None,
    max_chars: int | None,
    allow_spaces: bool,
    allow_punctuation: bool,
    lowercase: bool,
    reserved_labels: Sequence[str] | None,
) -> Tuple[str, ...]:
    del count
    labels = [
        _format_temporal_label(str(label), lowercase=bool(lowercase))
        for label in CHART_LABEL_POOL_UP_TO_25
        if _label_satisfies_temporal_filter(
            str(label),
            min_chars=min_chars,
            max_chars=max_chars,
            allow_spaces=bool(allow_spaces),
            allow_punctuation=bool(allow_punctuation),
        )
    ]
    return _filter_reserved_chart_labels(labels, reserved_labels)


def _temporal_panel_labels(
    rng: random.Random,
    *,
    count: int,
    min_chars: int | None,
    max_chars: int | None,
    allow_spaces: bool,
    allow_punctuation: bool,
    lowercase: bool,
    reserved_labels: Sequence[str] | None,
) -> tuple[Tuple[str, ...], Mapping[str, Any]] | None:
    eligible_subtypes = _eligible_temporal_label_subtypes(
        object_count=int(count),
        min_chars=min_chars,
        max_chars=max_chars,
        allow_spaces=bool(allow_spaces),
        allow_punctuation=bool(allow_punctuation),
        lowercase=bool(lowercase),
    )
    if not eligible_subtypes:
        return None
    shuffled = list(eligible_subtypes)
    rng.shuffle(shuffled)
    for subtype in shuffled:
        labels = _temporal_labels_for_subtype(
            rng,
            count=int(count),
            subtype=str(subtype),
            allow_punctuation=bool(allow_punctuation),
            lowercase=bool(lowercase),
        )
        filtered = _filter_reserved_chart_labels(labels, reserved_labels)
        if len(filtered) == int(count):
            return tuple(str(label) for label in filtered), {"temporal_subtype": str(subtype)}
    return None


def _eligible_panel_label_variants(
    *,
    count: int,
    min_chars: int | None,
    max_chars: int | None,
    allow_spaces: bool,
    allow_punctuation: bool,
    compact_length: bool,
    lowercase: bool,
    min_bucket_size: int,
    reserved_labels: Sequence[str] | None,
) -> Mapping[str, Tuple[str, ...]]:
    required_count = max(int(count), int(min_bucket_size))
    eligible: dict[str, Tuple[str, ...]] = {}
    for variant, manifest_name in CHART_PANEL_LABEL_BUCKET_MANIFESTS.items():
        try:
            labels = _chart_label_bucket_pool(
                str(manifest_name),
                min_chars=min_chars,
                max_chars=max_chars,
                allow_spaces=bool(allow_spaces),
                allow_punctuation=bool(allow_punctuation),
                compact_length=bool(compact_length),
                lowercase=bool(lowercase),
            )
        except Exception:
            continue
        labels = _filter_reserved_chart_labels(labels, reserved_labels)
        if len(labels) >= int(required_count):
            eligible[str(variant)] = tuple(str(label) for label in labels)

    subplot_pool = _subplot_letter_panel_label_pool(
        count=int(count),
        min_chars=min_chars,
        max_chars=max_chars,
        allow_spaces=bool(allow_spaces),
        allow_punctuation=bool(allow_punctuation),
        lowercase=bool(lowercase),
        reserved_labels=reserved_labels,
    )
    if len(subplot_pool) >= int(count):
        eligible["subplot_letters"] = tuple(subplot_pool)

    condition_pool = _condition_panel_label_pool(
        min_chars=min_chars,
        max_chars=max_chars,
        allow_spaces=bool(allow_spaces),
        allow_punctuation=bool(allow_punctuation),
        lowercase=bool(lowercase),
        reserved_labels=reserved_labels,
    )
    if len(condition_pool) >= int(count):
        eligible["condition_labels"] = tuple(condition_pool)

    if _eligible_temporal_label_subtypes(
        object_count=int(count),
        min_chars=min_chars,
        max_chars=max_chars,
        allow_spaces=bool(allow_spaces),
        allow_punctuation=bool(allow_punctuation),
        lowercase=bool(lowercase),
    ):
        eligible["temporal_sequence"] = tuple(CHART_TEMPORAL_LABEL_SUBTYPES)
    return dict(eligible)


def _resolve_panel_label_variant_probabilities(
    *,
    eligible_variants: Sequence[str],
    variant_weights: Mapping[str, float] | None,
) -> Mapping[str, float]:
    eligible = tuple(str(variant) for variant in eligible_variants)
    raw_weights = (
        {str(key): float(value) for key, value in variant_weights.items()}
        if isinstance(variant_weights, Mapping)
        else {str(key): float(value) for key, value in DEFAULT_CHART_PANEL_LABEL_VARIANT_WEIGHTS.items()}
    )
    candidate_weights = {
        str(variant): float(raw_weights.get(str(variant), 0.0 if isinstance(variant_weights, Mapping) else 1.0))
        for variant in eligible
    }
    return normalize_positive_weights(candidate_weights, default_keys=eligible)


def resolve_chart_panel_labels(
    rng: random.Random,
    *,
    count: int,
    min_chars: int | None = 1,
    max_chars: int = 14,
    allow_spaces: bool = True,
    allow_punctuation: bool = False,
    compact_length: bool = True,
    lowercase: bool = False,
    variant_weights: Mapping[str, float] | None = None,
    reserved_labels: Sequence[str] | None = None,
    min_bucket_size: int = CHART_NAMED_LABEL_MIN_BUCKET_SIZE,
) -> ResolvedChartLabels:
    """Resolve panel/subplot labels while avoiding visible label collisions."""

    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    label_filter = {
        "min_chars": min_chars,
        "max_chars": int(max_chars),
        "allow_spaces": bool(allow_spaces),
        "allow_punctuation": bool(allow_punctuation),
        "ascii_only": True,
        "compact_length": bool(compact_length),
        "lowercase": bool(lowercase),
        "min_bucket_size": int(min_bucket_size),
        "reserved_label_count": int(len(tuple(reserved_labels or ()))),
    }
    eligible = _eligible_panel_label_variants(
        count=int(count),
        min_chars=min_chars,
        max_chars=int(max_chars),
        allow_spaces=bool(allow_spaces),
        allow_punctuation=bool(allow_punctuation),
        compact_length=bool(compact_length),
        lowercase=bool(lowercase),
        min_bucket_size=int(min_bucket_size),
        reserved_labels=reserved_labels,
    )
    if not eligible:
        raise ValueError("no chart panel label variant satisfies the requested constraints")
    probabilities = _resolve_panel_label_variant_probabilities(
        eligible_variants=tuple(eligible.keys()),
        variant_weights=variant_weights,
    )
    variant = weighted_choice(rng, probabilities, sort_keys=True)
    if str(variant) == "temporal_sequence":
        resolved_temporal = _temporal_panel_labels(
            rng,
            count=int(count),
            min_chars=min_chars,
            max_chars=int(max_chars),
            allow_spaces=bool(allow_spaces),
            allow_punctuation=bool(allow_punctuation),
            lowercase=bool(lowercase),
            reserved_labels=reserved_labels,
        )
        if resolved_temporal is None:
            raise ValueError("temporal panel labels could not avoid reserved labels")
        labels, generated_filter = resolved_temporal
        return ResolvedChartLabels(
            labels=tuple(str(label) for label in labels),
            label_variant="panel",
            label_pool_kind="panel",
            label_source_kind="synthetic_temporal_sequence",
            label_bucket=str(variant),
            label_manifest="",
            label_filter={**dict(label_filter), **dict(generated_filter)},
            label_bucket_probabilities=dict(probabilities),
        )
    pool = tuple(str(label) for label in eligible[str(variant)])
    if str(variant) == "subplot_letters":
        labels = tuple(str(label) for label in pool[: int(count)])
        source_kind = "synthetic_subplot_letters"
        manifest_name = ""
    elif str(variant) == "condition_labels":
        labels = tuple(str(label) for label in rng.sample(list(pool), k=int(count)))
        source_kind = "synthetic_condition_labels"
        manifest_name = ""
    else:
        labels = tuple(str(label) for label in rng.sample(list(pool), k=int(count)))
        source_kind = "shared_label_manifest"
        manifest_name = str(CHART_PANEL_LABEL_BUCKET_MANIFESTS[str(variant)])
    return ResolvedChartLabels(
        labels=tuple(str(label) for label in labels),
        label_variant="panel",
        label_pool_kind="panel",
        label_source_kind=str(source_kind),
        label_bucket=str(variant),
        label_manifest=str(manifest_name),
        label_filter=dict(label_filter),
        label_bucket_probabilities=dict(probabilities),
    )


def resolve_chart_category_labels(
    rng: random.Random,
    *,
    count: int,
    max_chars: int = 14,
    min_chars: int | None = 3,
    allow_spaces: bool = True,
    bucket_weights: Mapping[str, float] | None = None,
) -> ResolvedChartLabels:
    """Resolve semantic category labels for chart legends and class bins."""

    return resolve_chart_text_labels(
        rng,
        count=int(count),
        label_variant="named",
        label_pool_kind="category",
        min_chars=min_chars,
        max_chars=int(max_chars),
        allow_spaces=bool(allow_spaces),
        allow_punctuation=False,
        compact_length=True,
        bucket_weights=bucket_weights,
    )


def resolve_chart_entity_labels(
    rng: random.Random,
    *,
    count: int,
    max_chars: int = 12,
    min_chars: int | None = 2,
    allow_spaces: bool = True,
    bucket_weights: Mapping[str, float] | None = None,
) -> ResolvedChartLabels:
    """Resolve entity-style labels for chart axes, row labels, and series names."""

    return resolve_chart_text_labels(
        rng,
        count=int(count),
        label_variant="named",
        label_pool_kind="entity",
        min_chars=min_chars,
        max_chars=int(max_chars),
        allow_spaces=bool(allow_spaces),
        allow_punctuation=False,
        compact_length=True,
        bucket_weights=bucket_weights,
    )


def sample_chart_labels(
    *,
    count: int,
    instance_seed: int,
    namespace: str = "charts.labels",
    max_chars: int = 4,
) -> Tuple[str, ...]:
    """Sample one randomized short label list for chart marks."""

    label_rng = spawn_rng(int(instance_seed), str(namespace))
    resolved = resolve_chart_axis_labels(
        label_rng,
        count=int(count),
        min_chars=2,
        max_chars=int(max_chars),
    )
    return tuple(str(label) for label in resolved.labels)


__all__ = [
    "CHART_ALL_LABEL_BUCKET_MANIFESTS",
    "CHART_CATEGORY_LABEL_BUCKET_MANIFESTS",
    "CHART_COMPACT_ID_ALPHANUM",
    "CHART_COMPACT_ID_DIGITS",
    "CHART_COMPACT_ID_FIRST_CHARS",
    "CHART_ENTITY_LABEL_BUCKET_MANIFESTS",
    "CHART_GENERATED_ENTITY_LABEL_BUCKETS",
    "CHART_LABEL_POOL_UP_TO_25",
    "CHART_NAMED_LABEL_MIN_BUCKET_SIZE",
    "CHART_PANEL_LABEL_BUCKET_MANIFESTS",
    "CHART_TEMPORAL_LABEL_SUBTYPES",
    "DEFAULT_CHART_PANEL_LABEL_VARIANT_WEIGHTS",
    "ResolvedChartLabels",
    "SUPPORTED_CHART_LABEL_POOL_KINDS",
    "SUPPORTED_CHART_LABEL_VARIANTS",
    "SUPPORTED_CHART_PANEL_LABEL_VARIANTS",
    "default_chart_label_bucket_weights",
    "normalize_chart_label_for_collision",
    "resolve_chart_axis_labels",
    "resolve_chart_category_labels",
    "resolve_chart_compact_axis_labels",
    "resolve_chart_entity_labels",
    "resolve_chart_panel_labels",
    "resolve_chart_text_labels",
    "sample_chart_labels",
    "validate_chart_label_namespaces",
]
