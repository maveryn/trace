"""Scene-local data sampling helpers for styled table chart tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import integer_range_choice, uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default, resolve_required_int_bounds
from trace_tasks.tasks.shared.name_assets import load_short_name_manifest
from ...shared.label_assets import resolve_chart_category_labels, resolve_chart_text_labels
from .state import TableDefaults

_TEMPORAL_YEAR_MIN: int = 2000
_TEMPORAL_YEAR_MAX: int = 2026


def table_value_cell_id(*, data_row_index: int, numeric_column_index: int) -> str:
    """Return the rendered cell id for one numeric-value table cell."""

    return f"cell_r{int(data_row_index) + 1}_c{int(numeric_column_index) + 1}"


def _build_values_by_row(
    *,
    row_labels: Sequence[str],
    column_headers: Sequence[str],
    rng,
    value_min: int,
    value_max: int,
    query_column: str | None = None,
    query_values_by_row: Mapping[str, int] | None = None,
) -> Dict[str, Dict[str, int]]:
    """Populate one deterministic numeric table, optionally fixing one queried column."""

    if (query_column is None) != (query_values_by_row is None):
        raise ValueError("query_column and query_values_by_row must be provided together")

    values_by_row: Dict[str, Dict[str, int]] = {}
    for row_label in row_labels:
        resolved_row_label = str(row_label)
        row_values: Dict[str, int] = {}
        for header in column_headers:
            resolved_header = str(header)
            if query_values_by_row is not None and resolved_header == str(query_column):
                row_values[resolved_header] = int(query_values_by_row[resolved_row_label])
            else:
                row_values[resolved_header] = int(rng.randint(int(value_min), int(value_max)))
        values_by_row[resolved_row_label] = dict(row_values)
    return values_by_row


def _sample_values_with_total(*, count: int, target_total: int, min_value: int, max_value: int, rng) -> List[int]:
    """Sample `count` bounded integers whose sum equals `target_total`."""

    values: List[int] = []
    remaining_total = int(target_total)
    for index in range(int(count)):
        slots_left = int(count) - int(index) - 1
        min_here = max(int(min_value), int(remaining_total - (slots_left * int(max_value))))
        max_here = min(int(max_value), int(remaining_total - (slots_left * int(min_value))))
        if int(min_here) > int(max_here):
            raise ValueError("failed to construct bounded values for requested total")
        values.append(int(rng.randint(int(min_here), int(max_here))))
        remaining_total -= int(values[-1])
    rng.shuffle(values)
    return [int(value) for value in values]


def _resolve_base_table_schema(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: TableDefaults,
    namespace: str,
    odd_row_count_required: bool = False,
    sample_generic_column_headers: bool = True,
) -> Dict[str, Any]:
    """Resolve one reusable base table schema and queried numeric column."""

    row_count_min, row_count_max = resolve_row_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        namespace=namespace,
    )
    numeric_col_count_min, numeric_col_count_max = resolve_numeric_column_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        namespace=namespace,
    )
    value_min, value_max = resolve_value_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        namespace=namespace,
    )

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    if bool(odd_row_count_required):
        odd_counts = [
            int(value)
            for value in range(int(row_count_min), int(row_count_max) + 1)
            if int(value) % 2 == 1
        ]
        if not odd_counts:
            raise ValueError("table row-count bounds must include an odd count for this variant")
        row_count = int(odd_counts[int(rng.randint(0, len(odd_counts) - 1))])
    else:
        row_count = int(rng.randint(int(row_count_min), int(row_count_max)))
    numeric_column_count = int(rng.randint(int(numeric_col_count_min), int(numeric_col_count_max)))
    row_labels = list(sample_table_row_labels(count=int(row_count), instance_seed=int(instance_seed)))
    if bool(sample_generic_column_headers):
        column_headers = list(sample_numeric_column_headers(count=int(numeric_column_count), instance_seed=int(instance_seed)))
    else:
        column_headers = [f"col_{index}" for index in range(int(numeric_column_count))]

    query_col_index = int(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{namespace}:query_column"),
            tuple(range(int(numeric_column_count))),
        )
    )
    query_column = str(column_headers[int(query_col_index)])
    return {
        "rng": rng,
        "row_count": int(row_count),
        "numeric_column_count": int(numeric_column_count),
        "row_count_range": [int(row_count_min), int(row_count_max)],
        "numeric_column_count_range": [int(numeric_col_count_min), int(numeric_col_count_max)],
        "value_range": [int(value_min), int(value_max)],
        "value_min": int(value_min),
        "value_max": int(value_max),
        "row_labels": [str(label) for label in row_labels],
        "column_headers": [str(header) for header in column_headers],
        "query_column": str(query_column),
        "query_column_index": int(query_col_index),
    }


def _resolve_distinct_secondary_numeric_column(
    *,
    column_headers: Sequence[str],
    numeric_column_count: int,
    primary_column_index: int,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> Tuple[int, str]:
    """Resolve one numeric column distinct from the primary queried column."""

    if int(numeric_column_count) < 2:
        raise ValueError("task requires at least two numeric columns")
    del params
    second_index = int(
        uniform_choice(
            spawn_rng(int(instance_seed), str(namespace)),
            tuple(
                index
                for index in range(int(numeric_column_count))
                if int(index) != int(primary_column_index)
            ),
        )
    )
    return int(second_index), str(column_headers[int(second_index)])


def _build_column_filter_query_values(
    *,
    filter_variant: str,
    row_count: int,
    target_count: int,
    value_min: int,
    value_max: int,
    rng,
    threshold_boundary_distractor_radius: int = 0,
    interval_boundary_distractor_radius: int = 0,
) -> Dict[str, Any]:
    """Construct one queried-column value list with a controlled matching-count filter."""

    supported_variants = {"above_threshold", "below_threshold", "in_interval"}
    if str(filter_variant) not in supported_variants:
        raise ValueError(f"unsupported table column-filter variant: {filter_variant}")
    if not (0 <= int(target_count) <= int(row_count)):
        raise ValueError("target_count must lie within the row-count support")

    def _sample_values(*, count: int, low: int, high: int) -> List[int]:
        if int(count) <= 0:
            return []
        if int(low) > int(high):
            raise ValueError("invalid bounded sampling range")
        return [int(rng.randint(int(low), int(high))) for _ in range(int(count))]

    def _sample_from_pool(*, count: int, values: Sequence[int]) -> List[int]:
        if int(count) <= 0:
            return []
        if not values:
            raise ValueError("cannot sample from an empty value pool")
        return [
            int(values[int(rng.randint(0, len(values) - 1))])
            for _ in range(int(count))
        ]

    def _bounded_values_near(
        *,
        center: int,
        offsets: Sequence[int],
    ) -> List[int]:
        values: List[int] = []
        for offset in offsets:
            value = int(center) + int(offset)
            if int(value_min) <= int(value) <= int(value_max):
                values.append(int(value))
        return list(values)

    metadata: Dict[str, Any] = {"filter_variant": str(filter_variant)}
    if str(filter_variant) == "above_threshold":
        if int(target_count) == int(row_count):
            threshold_value = int(value_min - 1)
            query_values = _sample_values(count=int(row_count), low=int(value_min), high=int(value_max))
        elif int(target_count) == 0:
            threshold_value = int(value_max)
            query_values = _sample_values(count=int(row_count), low=int(value_min), high=int(value_max))
        else:
            threshold_value = int(rng.randint(int(value_min), int(value_max - 1)))
            boundary_radius = max(0, int(threshold_boundary_distractor_radius))
            matching_pool = _bounded_values_near(
                center=int(threshold_value),
                offsets=range(1, int(boundary_radius) + 1),
            )
            non_matching_pool = _bounded_values_near(
                center=int(threshold_value),
                offsets=range(-int(boundary_radius), 1),
            )
            query_values = [
                *(
                    _sample_from_pool(count=int(target_count), values=matching_pool)
                    if matching_pool
                    else _sample_values(count=int(target_count), low=int(threshold_value + 1), high=int(value_max))
                ),
                *(
                    _sample_from_pool(count=int(row_count - target_count), values=non_matching_pool)
                    if non_matching_pool
                    else _sample_values(count=int(row_count - target_count), low=int(value_min), high=int(threshold_value))
                ),
            ]
        rng.shuffle(query_values)
        metadata["threshold_value"] = int(threshold_value)

        def _matches(value: int) -> bool:
            return int(value) > int(threshold_value)

    elif str(filter_variant) == "below_threshold":
        if int(target_count) == int(row_count):
            threshold_value = int(value_max + 1)
            query_values = _sample_values(count=int(row_count), low=int(value_min), high=int(value_max))
        elif int(target_count) == 0:
            threshold_value = int(value_min)
            query_values = _sample_values(count=int(row_count), low=int(value_min), high=int(value_max))
        else:
            threshold_value = int(rng.randint(int(value_min + 1), int(value_max)))
            boundary_radius = max(0, int(threshold_boundary_distractor_radius))
            matching_pool = _bounded_values_near(
                center=int(threshold_value),
                offsets=range(-int(boundary_radius), 0),
            )
            non_matching_pool = _bounded_values_near(
                center=int(threshold_value),
                offsets=range(0, int(boundary_radius) + 1),
            )
            query_values = [
                *(
                    _sample_from_pool(count=int(target_count), values=matching_pool)
                    if matching_pool
                    else _sample_values(count=int(target_count), low=int(value_min), high=int(threshold_value - 1))
                ),
                *(
                    _sample_from_pool(count=int(row_count - target_count), values=non_matching_pool)
                    if non_matching_pool
                    else _sample_values(count=int(row_count - target_count), low=int(threshold_value), high=int(value_max))
                ),
            ]
        rng.shuffle(query_values)
        metadata["threshold_value"] = int(threshold_value)

        def _matches(value: int) -> bool:
            return int(value) < int(threshold_value)

    else:
        if int(target_count) == int(row_count):
            interval_min = int(value_min)
            interval_max = int(value_max)
            query_values = _sample_values(count=int(row_count), low=int(value_min), high=int(value_max))
        else:
            while True:
                interval_min = int(rng.randint(int(value_min), int(value_max)))
                interval_max = int(rng.randint(int(interval_min), int(value_max)))
                inside_values = list(range(int(interval_min), int(interval_max) + 1))
                outside_values = [
                    int(value)
                    for value in range(int(value_min), int(value_max) + 1)
                    if int(value) < int(interval_min) or int(value) > int(interval_max)
                ]
                boundary_radius = max(0, int(interval_boundary_distractor_radius))
                boundary_outside_values: List[int] = []
                if boundary_radius > 0:
                    for offset in range(1, int(boundary_radius) + 1):
                        lower_value = int(interval_min - offset)
                        upper_value = int(interval_max + offset)
                        if int(value_min) <= int(lower_value) <= int(value_max):
                            boundary_outside_values.append(int(lower_value))
                        if int(value_min) <= int(upper_value) <= int(value_max):
                            boundary_outside_values.append(int(upper_value))
                outside_sample_values = list(boundary_outside_values or outside_values)
                if int(target_count) == 0 and outside_values:
                    query_values = _sample_from_pool(count=int(row_count), values=outside_sample_values)
                    break
                if int(target_count) > 0 and inside_values and outside_values:
                    query_values = [
                        *_sample_from_pool(count=int(target_count), values=inside_values),
                        *_sample_from_pool(
                            count=int(row_count - target_count),
                            values=outside_sample_values,
                        ),
                    ]
                    break
        rng.shuffle(query_values)
        metadata["interval_min"] = int(interval_min)
        metadata["interval_max"] = int(interval_max)

        def _matches(value: int) -> bool:
            return int(interval_min) <= int(value) <= int(interval_max)

    matching_row_indices = [
        int(row_index)
        for row_index, value in enumerate(query_values)
        if _matches(int(value))
    ]
    return {
        "query_values": [int(value) for value in query_values],
        "matching_row_indices": [int(row_index) for row_index in matching_row_indices],
        **metadata,
    }


def _resolve_counting_target_count(
    *,
    operation: str,
    row_count: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> int:
    """Resolve the target answer support for one table-counting variant."""

    target_min = 0
    target_max = int(row_count)
    if str(operation) in {"above_threshold", "below_threshold"}:
        target_min = int(gen_defaults.get("threshold_count_target_count_min", 0))
        target_max = int(row_count) - int(gen_defaults.get("threshold_count_target_count_max_row_offset", 0))
    elif str(operation) == "in_interval":
        target_min = int(gen_defaults.get("in_interval_target_count_min", 0))
        target_max = int(row_count) - int(gen_defaults.get("in_interval_target_count_max_row_offset", 0))
    if int(target_min) < 0:
        raise ValueError("target count minimum must be non-negative")
    if int(target_max) > int(row_count):
        raise ValueError("target count maximum cannot exceed row_count")
    if int(target_min) > int(target_max):
        raise ValueError(
            f"invalid target count support for {namespace}/{operation}: "
            f"{int(target_min)}..{int(target_max)} with row_count={int(row_count)}"
        )
    del params
    rng = spawn_rng(int(instance_seed), f"{namespace}:target_count")
    selected, _probabilities = integer_range_choice(rng, int(target_min), int(target_max))
    return int(selected)


def _decouple_sampling_after_operation(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
) -> Mapping[str, Any]:
    """No-op hook for query-id axis call sites."""

    _ = gen_defaults
    return params


def _resolve_balanced_integer_support_value(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    support_namespace: str,
    support_min: int,
    support_max: int,
) -> int:
    """Resolve one integer target from a bounded support."""

    if int(support_min) > int(support_max):
        raise ValueError("integer support must be non-empty")
    del gen_defaults
    del params
    rng = spawn_rng(int(instance_seed), f"{namespace}:{support_namespace}")
    selected, _probabilities = integer_range_choice(rng, int(support_min), int(support_max))
    return int(selected)


def resolve_row_count_bounds(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    defaults: TableDefaults,
    namespace: str,
) -> Tuple[int, int]:
    """Resolve inclusive row-count bounds."""

    return resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="row_count_min",
        max_key="row_count_max",
        fallback_min=int(defaults.row_count_min),
        fallback_max=int(defaults.row_count_max),
        context=f"generation defaults for {namespace}",
    )


def resolve_numeric_column_count_bounds(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    defaults: TableDefaults,
    namespace: str,
) -> Tuple[int, int]:
    """Resolve inclusive numeric-column-count bounds."""

    return resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="numeric_column_count_min",
        max_key="numeric_column_count_max",
        fallback_min=int(defaults.numeric_column_count_min),
        fallback_max=int(defaults.numeric_column_count_max),
        context=f"generation defaults for {namespace}",
    )


def resolve_value_bounds(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    defaults: TableDefaults,
    namespace: str,
) -> Tuple[int, int]:
    """Resolve inclusive numeric cell-value bounds."""

    return resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="value_min",
        max_key="value_max",
        fallback_min=int(defaults.value_min),
        fallback_max=int(defaults.value_max),
        context=f"generation defaults for {namespace}",
    )


def sample_table_row_labels(*, count: int, instance_seed: int, namespace: str = "tables.row_labels") -> Tuple[str, ...]:
    """Sample one tuple of unique short row-label names."""

    pool = load_short_name_manifest()
    if int(count) <= 0:
        raise ValueError("row count must be positive")
    if int(count) > len(pool):
        raise ValueError("row count exceeds the supported name pool")
    rng = spawn_rng(int(instance_seed), str(namespace))
    candidates = list(pool)
    rng.shuffle(candidates)
    return tuple(str(value) for value in candidates[: int(count)])


def sample_numeric_column_headers(
    *,
    count: int,
    instance_seed: int,
    namespace: str = "tables.numeric_headers",
) -> Tuple[str, ...]:
    """Sample one tuple of unique short numeric-column headers."""

    if int(count) <= 0:
        raise ValueError("numeric column count must be positive")
    rng = spawn_rng(int(instance_seed), str(namespace))
    resolved = resolve_chart_text_labels(
        rng,
        count=int(count),
        label_pool_kind="all",
        min_chars=2,
        max_chars=6,
        allow_spaces=False,
    )
    return tuple(str(value) for value in resolved.labels)


def sample_category_column_header(
    *,
    instance_seed: int,
    namespace: str = "tables.category_header",
) -> str:
    """Sample one short categorical-column header."""

    rng = spawn_rng(int(instance_seed), str(namespace))
    resolved = resolve_chart_category_labels(
        rng,
        count=1,
        min_chars=2,
        max_chars=8,
        allow_spaces=False,
    )
    return str(resolved.labels[0])


def sample_temporal_year_headers(
    *,
    count: int,
    instance_seed: int,
    namespace: str = "tables.temporal_year_headers",
) -> Tuple[str, ...]:
    """Sample one contiguous tuple of visible year headers."""

    if int(count) <= 0:
        raise ValueError("temporal year-header count must be positive")
    feasible_start_max = int(_TEMPORAL_YEAR_MAX) - int(count) + 1
    if int(feasible_start_max) < int(_TEMPORAL_YEAR_MIN):
        raise ValueError("temporal year-header count exceeds the supported year span")
    rng = spawn_rng(int(instance_seed), str(namespace))
    start_year = int(rng.randint(int(_TEMPORAL_YEAR_MIN), int(feasible_start_max)))
    return tuple(str(int(start_year) + int(offset)) for offset in range(int(count)))


def build_ranking_label_dataset_for_variant(
    *,
    operation: str,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: TableDefaults,
    namespace: str,
) -> Dict[str, Any]:
    """Construct one deterministic table dataset for kth-order row-label ranking queries."""

    if str(operation) not in {"descending", "ascending"}:
        raise ValueError(f"unsupported table ranking-label variant: {operation}")

    base = _resolve_base_table_schema(
        params=params,
        instance_seed=int(instance_seed),
        gen_defaults=gen_defaults,
        defaults=defaults,
        namespace=namespace,
    )
    rng = base["rng"]
    row_count = int(base["row_count"])
    numeric_column_count = int(base["numeric_column_count"])
    row_labels = list(base["row_labels"])
    column_headers = list(base["column_headers"])
    query_col_index = int(base["query_column_index"])
    query_column = str(base["query_column"])
    value_min = int(base["value_min"])
    value_max = int(base["value_max"])

    if int(row_count) < 3:
        raise ValueError("table ranking tasks require at least three rows")
    if int(value_max) - int(value_min) + 1 < int(row_count):
        raise ValueError("table ranking tasks require enough value range for unique ordered ranks")

    allowed_ranks = [int(rank) for rank in range(2, min(4, int(row_count) - 1) + 1)]
    rank_k = int(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{namespace}:query_rank"),
            tuple(int(rank) for rank in allowed_ranks),
        )
    )

    unique_query_values = list(rng.sample(range(int(value_min), int(value_max) + 1), int(row_count)))
    rng.shuffle(unique_query_values)
    query_values_by_row = {
        str(row_label): int(unique_query_values[int(row_index)])
        for row_index, row_label in enumerate(row_labels)
    }
    values_by_row = _build_values_by_row(
        row_labels=row_labels,
        column_headers=column_headers,
        rng=rng,
        value_min=int(value_min),
        value_max=int(value_max),
        query_column=str(query_column),
        query_values_by_row=query_values_by_row,
    )

    sorted_rows = sorted(
        (
            {
                "row_label": str(row_label),
                "row_index": int(row_index),
                "value": int(values_by_row[str(row_label)][str(query_column)]),
            }
            for row_index, row_label in enumerate(row_labels)
        ),
        key=lambda item: int(item["value"]),
        reverse=(str(operation) == "descending"),
    )
    answer_row = dict(sorted_rows[int(rank_k) - 1])
    return {
        "row_count": int(row_count),
        "numeric_column_count": int(numeric_column_count),
        "row_count_range": list(base["row_count_range"]),
        "numeric_column_count_range": list(base["numeric_column_count_range"]),
        "value_range": list(base["value_range"]),
        "row_labels": [str(label) for label in row_labels],
        "column_headers": [str(header) for header in column_headers],
        "query_column": str(query_column),
        "query_column_index": int(query_col_index),
        "query_rank": int(rank_k),
        "values_by_row": dict(values_by_row),
        "answer_row_label": str(answer_row["row_label"]),
        "answer_row_index": int(answer_row["row_index"]),
        "answer_value": int(answer_row["value"]),
    }


def build_summary_value_dataset_for_variant(
    *,
    operation: str,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: TableDefaults,
    namespace: str,
) -> Dict[str, Any]:
    """Construct one deterministic table dataset for column-summary numeric queries."""

    if str(operation) not in {"sum", "mean", "median"}:
        raise ValueError(f"unsupported table summary-value variant: {operation}")

    base = _resolve_base_table_schema(
        params=params,
        instance_seed=int(instance_seed),
        gen_defaults=gen_defaults,
        defaults=defaults,
        namespace=namespace,
        odd_row_count_required=(str(operation) == "median"),
    )
    rng = base["rng"]
    row_count = int(base["row_count"])
    numeric_column_count = int(base["numeric_column_count"])
    row_labels = list(base["row_labels"])
    column_headers = list(base["column_headers"])
    query_col_index = int(base["query_column_index"])
    query_column = str(base["query_column"])
    value_min = int(base["value_min"])
    value_max = int(base["value_max"])

    if str(operation) == "sum":
        target_sum = int(rng.randint(int(row_count * value_min), int(row_count * value_max)))
        query_values = _sample_values_with_total(
            count=int(row_count),
            target_total=int(target_sum),
            min_value=int(value_min),
            max_value=int(value_max),
            rng=rng,
        )
        answer_value = int(target_sum)
    elif str(operation) == "mean":
        target_mean = _resolve_balanced_integer_support_value(
            params=params,
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
            namespace=namespace,
            support_namespace="column_mean_answer",
            support_min=int(value_min),
            support_max=int(value_max),
        )
        query_values = _sample_values_with_total(
            count=int(row_count),
            target_total=int(row_count * target_mean),
            min_value=int(value_min),
            max_value=int(value_max),
            rng=rng,
        )
        answer_value = int(target_mean)
    else:
        if int(value_max) - int(value_min) < 2:
            raise ValueError("table value range must support strict lower/upper values around the median")
        target_median = _resolve_balanced_integer_support_value(
            params=params,
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
            namespace=namespace,
            support_namespace="column_median_answer",
            support_min=int(value_min + 1),
            support_max=int(value_max - 1),
        )
        lower_count = int(row_count // 2)
        upper_count = int(row_count // 2)
        lower_values = [int(rng.randint(int(value_min), int(target_median - 1))) for _ in range(int(lower_count))]
        upper_values = [int(rng.randint(int(target_median + 1), int(value_max))) for _ in range(int(upper_count))]
        query_values = [*lower_values, int(target_median), *upper_values]
        rng.shuffle(query_values)
        answer_value = int(target_median)

    query_values_by_row = {
        str(row_label): int(query_values[int(row_index)])
        for row_index, row_label in enumerate(row_labels)
    }
    values_by_row = _build_values_by_row(
        row_labels=row_labels,
        column_headers=column_headers,
        rng=rng,
        value_min=int(value_min),
        value_max=int(value_max),
        query_column=str(query_column),
        query_values_by_row=query_values_by_row,
    )

    return {
        "row_count": int(row_count),
        "numeric_column_count": int(numeric_column_count),
        "row_count_range": list(base["row_count_range"]),
        "numeric_column_count_range": list(base["numeric_column_count_range"]),
        "value_range": list(base["value_range"]),
        "row_labels": [str(label) for label in row_labels],
        "column_headers": [str(header) for header in column_headers],
        "query_column": str(query_column),
        "values_by_row": dict(values_by_row),
        "answer_value": int(answer_value),
        "query_column_index": int(query_col_index),
    }


def build_counting_value_dataset_for_variant(
    *,
    operation: str,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: TableDefaults,
    namespace: str,
) -> Dict[str, Any]:
    """Construct one deterministic table dataset for column-filter counting queries."""

    supported_variants = {"above_threshold", "below_threshold", "in_interval", "category_membership"}
    if str(operation) not in supported_variants:
        raise ValueError(f"unsupported table counting variant: {operation}")

    base = _resolve_base_table_schema(
        params=params,
        instance_seed=int(instance_seed),
        gen_defaults=gen_defaults,
        defaults=defaults,
        namespace=namespace,
    )
    rng = base["rng"]
    row_count = int(base["row_count"])
    numeric_column_count = int(base["numeric_column_count"])
    row_labels = list(base["row_labels"])
    column_headers = list(base["column_headers"])
    query_col_index = int(base["query_column_index"])
    query_column = str(base["query_column"])
    value_min = int(base["value_min"])
    value_max = int(base["value_max"])
    support_params = _decouple_sampling_after_operation(params, gen_defaults=gen_defaults)
    target_count = _resolve_counting_target_count(
        operation=str(operation),
        row_count=int(row_count),
        params=support_params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=namespace,
    )

    if str(operation) == "category_membership":
        category_column = sample_category_column_header(instance_seed=int(instance_seed))
        category_value_labels = tuple(
            str(label)
            for label in resolve_chart_category_labels(
                spawn_rng(int(instance_seed), f"{namespace}.category_values"),
                count=6,
                min_chars=2,
                max_chars=8,
                allow_spaces=False,
            ).labels
        )
        target_category = str(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{namespace}:target_category"),
                tuple(str(label) for label in category_value_labels),
            )
        )
        non_target_categories = [
            str(value)
            for value in category_value_labels
            if str(value) != str(target_category)
        ]
        if not non_target_categories:
            raise ValueError("categorical count requires at least one non-target category")
        matching_row_indices = sorted(int(index) for index in rng.sample(range(int(row_count)), int(target_count)))
        matching_index_set = set(int(index) for index in matching_row_indices)
        category_values = []
        for row_index in range(int(row_count)):
            if int(row_index) in matching_index_set:
                category_values.append(str(target_category))
            else:
                category_values.append(str(non_target_categories[int(rng.randint(0, len(non_target_categories) - 1))]))

        rendered_column_headers = [str(category_column), *[str(header) for header in column_headers]]
        values_by_row: Dict[str, Dict[str, Any]] = {}
        for row_index, row_label in enumerate(row_labels):
            row_values: Dict[str, Any] = {str(category_column): str(category_values[int(row_index)])}
            for header in column_headers:
                row_values[str(header)] = int(rng.randint(int(value_min), int(value_max)))
            values_by_row[str(row_label)] = dict(row_values)
        matching_row_labels = [str(row_labels[int(row_index)]) for row_index in matching_row_indices]
        return {
            "row_count": int(row_count),
            "numeric_column_count": int(numeric_column_count),
            "column_count": int(len(rendered_column_headers)),
            "row_count_range": list(base["row_count_range"]),
            "numeric_column_count_range": list(base["numeric_column_count_range"]),
            "value_range": list(base["value_range"]),
            "row_labels": [str(label) for label in row_labels],
            "column_headers": [str(header) for header in rendered_column_headers],
            "numeric_column_headers": [str(header) for header in column_headers],
            "query_column": str(category_column),
            "category_column": str(category_column),
            "target_category": str(target_category),
            "category_values_by_row": {
                str(row_label): str(category_values[int(row_index)])
                for row_index, row_label in enumerate(row_labels)
            },
            "values_by_row": dict(values_by_row),
            "answer_value": int(len(matching_row_indices)),
            "query_column_index": 0,
            "category_column_index": 0,
            "matching_row_indices": [int(row_index) for row_index in matching_row_indices],
            "matching_row_labels": [str(label) for label in matching_row_labels],
            "filter_variant": "category_membership",
        }

    filter_query = _build_column_filter_query_values(
        filter_variant=str(operation),
        row_count=int(row_count),
        target_count=int(target_count),
        value_min=int(value_min),
        value_max=int(value_max),
        rng=rng,
        threshold_boundary_distractor_radius=int(
            gen_defaults.get("threshold_count_boundary_distractor_radius", 0)
        ),
        interval_boundary_distractor_radius=int(
            gen_defaults.get("in_interval_boundary_distractor_radius", 0)
        ),
    )
    query_values = [int(value) for value in filter_query["query_values"]]

    query_values_by_row = {
        str(row_label): int(query_values[int(row_index)])
        for row_index, row_label in enumerate(row_labels)
    }
    values_by_row = _build_values_by_row(
        row_labels=row_labels,
        column_headers=column_headers,
        rng=rng,
        value_min=int(value_min),
        value_max=int(value_max),
        query_column=str(query_column),
        query_values_by_row=query_values_by_row,
    )
    matching_row_indices = [int(row_index) for row_index in filter_query["matching_row_indices"]]
    matching_row_labels = [str(row_labels[int(row_index)]) for row_index in matching_row_indices]
    return {
        "row_count": int(row_count),
        "numeric_column_count": int(numeric_column_count),
        "row_count_range": list(base["row_count_range"]),
        "numeric_column_count_range": list(base["numeric_column_count_range"]),
        "value_range": list(base["value_range"]),
        "row_labels": [str(label) for label in row_labels],
        "column_headers": [str(header) for header in column_headers],
        "query_column": str(query_column),
        "values_by_row": dict(values_by_row),
        "answer_value": int(len(matching_row_indices)),
        "query_column_index": int(query_col_index),
        "matching_row_indices": [int(row_index) for row_index in matching_row_indices],
        "matching_row_labels": [str(label) for label in matching_row_labels],
        **{
            str(key): (int(value) if isinstance(value, int) else value)
            for key, value in filter_query.items()
            if str(key) not in {"query_values", "matching_row_indices"}
        },
    }


def build_statistics_filtered_subset_dataset_for_variant(
    *,
    operation: str,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: TableDefaults,
    namespace: str,
) -> Dict[str, Any]:
    """Construct one deterministic table dataset for filtered column aggregation queries."""

    if str(operation) != "mean":
        raise ValueError(f"unsupported filtered table statistics variant: {operation}")

    base = _resolve_base_table_schema(
        params=params,
        instance_seed=int(instance_seed),
        gen_defaults=gen_defaults,
        defaults=defaults,
        namespace=namespace,
    )
    rng = base["rng"]
    row_count = int(base["row_count"])
    numeric_column_count = int(base["numeric_column_count"])
    row_labels = list(base["row_labels"])
    column_headers = list(base["column_headers"])
    filter_column_index = int(base["query_column_index"])
    filter_column = str(base["query_column"])
    target_column_index, target_column = _resolve_distinct_secondary_numeric_column(
        column_headers=column_headers,
        numeric_column_count=int(numeric_column_count),
        primary_column_index=int(filter_column_index),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}:target_column",
    )
    value_min = int(base["value_min"])
    value_max = int(base["value_max"])

    if int(row_count) < 2:
        raise ValueError("filtered subset tasks require at least two rows")
    selected_count_min = max(
        1,
        int(params.get("selected_row_count_min", group_default(gen_defaults, "selected_row_count_min", 1))),
    )
    selected_count_max = min(
        int(row_count) - 1,
        int(params.get("selected_row_count_max", group_default(gen_defaults, "selected_row_count_max", int(row_count) - 1))),
    )
    if int(selected_count_min) > int(selected_count_max):
        raise ValueError("selected_row_count_min must be <= selected_row_count_max and row_count - 1")
    selected_count_params = _decouple_sampling_after_operation(params, gen_defaults=gen_defaults)
    del selected_count_params
    target_count, _probabilities = integer_range_choice(
        spawn_rng(int(instance_seed), f"{namespace}:selected_row_count"),
        int(selected_count_min),
        int(selected_count_max),
    )
    target_count = int(target_count)
    supported_filter_variants = ("above_threshold", "below_threshold", "in_interval")
    explicit_filter_variant = params.get("filter_variant")
    if explicit_filter_variant is None:
        filter_variant = str(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{namespace}:filter_variant"),
                supported_filter_variants,
            )
        )
    else:
        filter_variant = str(explicit_filter_variant)
        if filter_variant not in set(supported_filter_variants):
            raise ValueError(f"unsupported filtered table variant: {filter_variant}")
    filter_query = _build_column_filter_query_values(
        filter_variant=str(filter_variant),
        row_count=int(row_count),
        target_count=int(target_count),
        value_min=int(value_min),
        value_max=int(value_max),
        rng=rng,
    )

    query_values_by_row = {
        str(row_label): int(filter_query["query_values"][int(row_index)])
        for row_index, row_label in enumerate(row_labels)
    }
    values_by_row = _build_values_by_row(
        row_labels=row_labels,
        column_headers=column_headers,
        rng=rng,
        value_min=int(value_min),
        value_max=int(value_max),
        query_column=str(filter_column),
        query_values_by_row=query_values_by_row,
    )

    selected_row_indices = [int(row_index) for row_index in filter_query["matching_row_indices"]]
    selected_row_labels = [str(row_labels[int(row_index)]) for row_index in selected_row_indices]
    selected_count = int(len(selected_row_indices))
    if int(selected_count) <= 0:
        raise ValueError("filtered subset tasks require at least one selected row")
    target_mean = int(rng.randint(int(value_min), int(value_max)))
    target_values = _sample_values_with_total(
        count=int(selected_count),
        target_total=int(selected_count * target_mean),
        min_value=int(value_min),
        max_value=int(value_max),
        rng=rng,
    )
    answer_value = int(target_mean)

    for offset, row_index in enumerate(selected_row_indices):
        row_label = str(row_labels[int(row_index)])
        values_by_row[str(row_label)][str(target_column)] = int(target_values[int(offset)])

    supporting_cell_ids: List[str] = []
    for row_index in selected_row_indices:
        supporting_cell_ids.append(
            table_value_cell_id(
                data_row_index=int(row_index),
                numeric_column_index=int(filter_column_index),
            )
        )
        supporting_cell_ids.append(
            table_value_cell_id(
                data_row_index=int(row_index),
                numeric_column_index=int(target_column_index),
            )
        )

    return {
        "row_count": int(row_count),
        "numeric_column_count": int(numeric_column_count),
        "row_count_range": list(base["row_count_range"]),
        "numeric_column_count_range": list(base["numeric_column_count_range"]),
        "value_range": list(base["value_range"]),
        "row_labels": [str(label) for label in row_labels],
        "column_headers": [str(header) for header in column_headers],
        "filter_variant": str(filter_variant),
        "filter_column": str(filter_column),
        "filter_column_index": int(filter_column_index),
        "target_column": str(target_column),
        "target_column_index": int(target_column_index),
        "values_by_row": dict(values_by_row),
        "selected_row_indices": [int(row_index) for row_index in selected_row_indices],
        "selected_row_labels": [str(label) for label in selected_row_labels],
        "selected_row_count_range": [int(selected_count_min), int(selected_count_max)],
        "supporting_cell_ids": [str(cell_id) for cell_id in supporting_cell_ids],
        "answer_value": int(answer_value),
        **{
            str(key): (int(value) if isinstance(value, int) else value)
            for key, value in filter_query.items()
            if str(key) not in {"query_values", "matching_row_indices"}
        },
    }


def render_table_filter_condition(dataset: Mapping[str, Any]) -> str:
    """Render one concise human-readable filter condition phrase."""

    filter_variant = str(dataset["filter_variant"])
    filter_column = str(dataset["filter_column"])
    if filter_variant == "above_threshold":
        return f"\"{filter_column}\" is greater than {int(dataset['threshold_value'])}"
    if filter_variant == "below_threshold":
        return f"\"{filter_column}\" is less than {int(dataset['threshold_value'])}"
    return (
        f"\"{filter_column}\" is from {int(dataset['interval_min'])} "
        f"to {int(dataset['interval_max'])} inclusive"
    )


def build_temporal_value_dataset_for_variant(
    *,
    operation: str,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: TableDefaults,
    namespace: str,
) -> Dict[str, Any]:
    """Construct one deterministic year-column table dataset for temporal queries."""

    supported_variants = {"row_interval_sum_difference_abs", "yearwise_abs_difference_sum"}
    if str(operation) not in supported_variants:
        raise ValueError(f"unsupported table temporal variant: {operation}")

    base = _resolve_base_table_schema(
        params=params,
        instance_seed=int(instance_seed),
        gen_defaults=gen_defaults,
        defaults=defaults,
        namespace=namespace,
        sample_generic_column_headers=False,
    )
    rng = base["rng"]
    row_count = int(base["row_count"])
    numeric_column_count = int(base["numeric_column_count"])
    row_labels = list(base["row_labels"])
    column_headers = list(
        sample_temporal_year_headers(
            count=int(numeric_column_count),
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.temporal_headers",
        )
    )
    value_min = int(base["value_min"])
    value_max = int(base["value_max"])

    values_by_row = _build_values_by_row(
        row_labels=row_labels,
        column_headers=column_headers,
        rng=rng,
        value_min=int(value_min),
        value_max=int(value_max),
    )

    query_row_index = int(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{namespace}:query_row"),
            tuple(range(int(row_count))),
        )
    )
    query_row_label = str(row_labels[int(query_row_index)])
    query_cells: List[Dict[str, Any]] = []

    def _append_query_cell(
        *,
        row_index: int,
        row_label: str,
        column_index: int,
        value: int,
        row_role: str,
    ) -> None:
        column_header = str(column_headers[int(column_index)])
        values_by_row[str(row_label)][str(column_header)] = int(value)
        query_cells.append(
            {
                "row_label": str(row_label),
                "row_index": int(row_index),
                "row_role": str(row_role),
                "column": str(column_header),
                "column_index": int(column_index),
                "cell_id": table_value_cell_id(
                    data_row_index=int(row_index),
                    numeric_column_index=int(column_index),
                ),
                "value": int(value),
            }
        )

    if int(numeric_column_count) < 2:
        raise ValueError("temporal interval variants require at least two year columns")
    interval_len_min, interval_len_max = resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="interval_length_min",
        max_key="interval_length_max",
        fallback_min=2,
        fallback_max=int(numeric_column_count),
        context=f"generation defaults for {namespace}",
    )
    interval_len_min = max(2, int(interval_len_min))
    interval_len_max = min(int(numeric_column_count), int(interval_len_max))
    if int(interval_len_min) > int(interval_len_max):
        raise ValueError("temporal interval length bounds must overlap available year columns")
    interval_len, _probabilities = integer_range_choice(
        spawn_rng(int(instance_seed), f"{namespace}:interval_length"),
        int(interval_len_min),
        int(interval_len_max),
    )
    interval_len = int(interval_len)
    start_index = int(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{namespace}:interval_start"),
            tuple(range(int(numeric_column_count - interval_len + 1))),
        )
    )
    end_index = int(start_index + interval_len - 1)
    interval_indices = list(range(int(start_index), int(end_index) + 1))
    query_years = [str(column_headers[int(column_index)]) for column_index in interval_indices]
    query_row_label_b = ""
    query_row_index_b = -1
    row_interval_sums: Dict[str, int] = {}
    paired_absolute_differences: List[int] = []

    if int(row_count) < 2:
        raise ValueError("two-row temporal variants require at least two data rows")
    query_row_index_b = int(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{namespace}:query_row_b"),
            tuple(index for index in range(int(row_count)) if int(index) != int(query_row_index)),
        )
    )
    query_row_label_b = str(row_labels[int(query_row_index_b)])
    interval_values_a = [
        int(rng.randint(int(value_min), int(value_max)))
        for _ in range(int(interval_len))
    ]
    interval_values_b = [
        int(rng.randint(int(value_min), int(value_max)))
        for _ in range(int(interval_len))
    ]
    sum_a = int(sum(int(value) for value in interval_values_a))
    sum_b = int(sum(int(value) for value in interval_values_b))
    if int(sum_a) == int(sum_b):
        if int(interval_values_b[0]) < int(value_max):
            interval_values_b[0] = int(interval_values_b[0]) + 1
            sum_b += 1
        else:
            interval_values_b[0] = int(interval_values_b[0]) - 1
            sum_b -= 1
    row_interval_sums[str(query_row_label)] = int(sum_a)
    row_interval_sums[str(query_row_label_b)] = int(sum_b)
    if str(operation) == "row_interval_sum_difference_abs":
        answer_value = int(abs(int(sum_a) - int(sum_b)))
    else:
        paired_absolute_differences = [
            int(abs(int(value_a) - int(value_b)))
            for value_a, value_b in zip(interval_values_a, interval_values_b)
        ]
        answer_value = int(sum(int(value) for value in paired_absolute_differences))
    for offset, cell_value in enumerate(interval_values_a):
        _append_query_cell(
            row_index=int(query_row_index),
            row_label=str(query_row_label),
            column_index=int(start_index + offset),
            value=int(cell_value),
            row_role="row_a",
        )
    for offset, cell_value in enumerate(interval_values_b):
        _append_query_cell(
            row_index=int(query_row_index_b),
            row_label=str(query_row_label_b),
            column_index=int(start_index + offset),
            value=int(cell_value),
            row_role="row_b",
        )

    return {
        "row_count": int(row_count),
        "numeric_column_count": int(numeric_column_count),
        "row_count_range": list(base["row_count_range"]),
        "numeric_column_count_range": list(base["numeric_column_count_range"]),
        "interval_length_range": [int(interval_len_min), int(interval_len_max)],
        "value_range": list(base["value_range"]),
        "row_labels": [str(label) for label in row_labels],
        "column_headers": [str(header) for header in column_headers],
        "values_by_row": dict(values_by_row),
        "query_row_label": str(query_row_label),
        "query_row_index": int(query_row_index),
        "query_row_label_a": str(query_row_label),
        "query_row_index_a": int(query_row_index),
        "query_row_label_b": str(query_row_label_b),
        "query_row_index_b": int(query_row_index_b),
        "query_row_labels": (
            [str(query_row_label), str(query_row_label_b)]
            if str(query_row_label_b)
            else [str(query_row_label)]
        ),
        "query_cells": [dict(cell) for cell in query_cells],
        "query_years": list(query_years),
        "query_year_start": str(query_years[0]),
        "query_year_end": str(query_years[-1]),
        "row_interval_sums": dict(row_interval_sums),
        "paired_absolute_differences": [int(value) for value in paired_absolute_differences],
        "answer_value": int(answer_value),
    }



__all__ = [
    "build_counting_value_dataset_for_variant",
    "build_ranking_label_dataset_for_variant",
    "build_temporal_value_dataset_for_variant",
    "build_statistics_filtered_subset_dataset_for_variant",
    "build_summary_value_dataset_for_variant",
    "render_table_filter_condition",
    "resolve_numeric_column_count_bounds",
    "resolve_row_count_bounds",
    "sample_category_column_header",
    "sample_numeric_column_headers",
    "sample_table_row_labels",
    "sample_temporal_year_headers",
    "table_value_cell_id",
]
