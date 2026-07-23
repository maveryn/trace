"""Neutral data helpers for region-map chart tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....shared.config_defaults import resolve_required_int_bounds
from ....shared.deterministic_sampling import uniform_probability_map
from ...shared.label_assets import resolve_chart_category_labels
from .defaults import _GEN_DEFAULTS, SCENE_NAMESPACE
from .spatial_primitives import (
    _balanced_int,
    _choose_random,
    _grid_pair_support,
    _sample_connected_cells,
)


def _make_numeric_bins(bin_count: int) -> List[Dict[str, Any]]:
    bins: List[Dict[str, Any]] = []
    for index in range(int(bin_count)):
        lower = int(math.floor(index * 100 / int(bin_count)))
        upper = int(math.floor((index + 1) * 100 / int(bin_count)) - 1)
        bins.append(
            {
                "bin_id": f"legend_bin_{index}",
                "bin_index": int(index),
                "bin_label": f"{lower}-{upper}",
                "lower": int(lower),
                "upper": int(upper),
                "category": "",
            }
        )
    return bins


def _marker_label_for_index(index: int) -> str:
    """Return spreadsheet-style labels A, B, ..., Z, AA, AB for marker regions."""

    n = int(index)
    letters: List[str] = []
    while True:
        letters.append(chr(ord("A") + (n % 26)))
        n = (n // 26) - 1
        if n < 0:
            break
    return "".join(reversed(letters))


def _make_category_bins(bin_count: int, *, rng) -> List[Dict[str, Any]]:
    resolved_labels = resolve_chart_category_labels(
        rng,
        count=int(bin_count),
        max_chars=14,
        allow_spaces=True,
    )
    label_source = {
        "label_source_kind": str(resolved_labels.label_source_kind),
        "label_pool_kind": str(resolved_labels.label_pool_kind),
        "label_bucket": str(resolved_labels.label_bucket),
        "label_manifest": str(resolved_labels.label_manifest),
        "label_filter": dict(resolved_labels.label_filter),
        "label_bucket_probabilities": dict(resolved_labels.label_bucket_probabilities),
    }
    return [
        {
            "bin_id": f"legend_bin_{index}",
            "bin_index": int(index),
            "bin_label": str(label),
            "lower": None,
            "upper": None,
            "category": str(label),
            "label_source": dict(label_source),
        }
        for index, label in enumerate(str(value) for value in resolved_labels.labels)
    ]


def build_synthetic_regions(
    *,
    categorical: bool,
    params: Mapping[str, Any],
    instance_seed: int,
    rng,
    namespace_suffix: str,
) -> Tuple[int, int, List[Tuple[int, int]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Sample a connected synthetic map and legend bins."""

    row_min, row_max = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key="region_row_count_min",
        max_key="region_row_count_max",
        fallback_min=5,
        fallback_max=7,
        context=f"{SCENE_NAMESPACE} region rows",
    )
    col_min, col_max = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key="region_col_count_min",
        max_key="region_col_count_max",
        fallback_min=5,
        fallback_max=7,
        context=f"{SCENE_NAMESPACE} region cols",
    )
    region_min, region_max = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key="region_count_min",
        max_key="region_count_max",
        fallback_min=14,
        fallback_max=28,
        context=f"{SCENE_NAMESPACE} total regions",
    )
    grid_pairs = _grid_pair_support(
        row_min=int(row_min),
        row_max=int(row_max),
        col_min=int(col_min),
        col_max=int(col_max),
        region_min=int(region_min),
    )
    if not grid_pairs:
        raise ValueError(f"no valid grid pairs for {SCENE_NAMESPACE} with current row/col/region bounds")
    pair_index = _balanced_int(
        list(range(len(grid_pairs))),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{namespace_suffix}.grid_pair",
    )
    rows, cols = grid_pairs[int(pair_index)]
    region_support = list(range(int(region_min), min(int(region_max), int(rows) * int(cols)) + 1))
    region_count = _balanced_int(
        region_support,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{namespace_suffix}.region_count",
    )
    active_cells = _sample_connected_cells(rows=int(rows), cols=int(cols), target_count=int(region_count), rng=rng)

    if "legend_bin_count" in params:
        bin_min = bin_max = int(params["legend_bin_count"])
    else:
        bin_min, bin_max = resolve_required_int_bounds(
            params,
            _GEN_DEFAULTS,
            min_key="legend_bin_count_min",
            max_key="legend_bin_count_max",
            fallback_min=4,
            fallback_max=8,
            context=f"{SCENE_NAMESPACE} legend bins",
        )
    bin_count = _balanced_int(
        list(range(int(bin_min), int(bin_max) + 1)),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{namespace_suffix}.legend_bin_count",
    )
    legend_bins = _make_category_bins(int(bin_count), rng=rng) if bool(categorical) else _make_numeric_bins(int(bin_count))

    regions: List[Dict[str, Any]] = []
    for index, (row, col) in enumerate(active_cells):
        regions.append(
            {
                "region_id": f"region_{index}",
                "row": int(row),
                "col": int(col),
                "bin_index": int(rng.randrange(int(bin_count))),
                "bin_label": str(legend_bins[0]["bin_label"]),
            }
        )
    return int(rows), int(cols), list(active_cells), list(regions), list(legend_bins)


def build_synthetic_marker_regions(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    rng,
    namespace_suffix: str,
) -> Tuple[int, int, List[Tuple[int, int]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Sample a connected synthetic grid-map shape and neutral legend bins for marker overlays."""

    row_min, row_max = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key="region_row_count_min",
        max_key="region_row_count_max",
        fallback_min=4,
        fallback_max=7,
        context=f"{SCENE_NAMESPACE} marker-region rows",
    )
    col_min, col_max = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key="region_col_count_min",
        max_key="region_col_count_max",
        fallback_min=4,
        fallback_max=7,
        context=f"{SCENE_NAMESPACE} marker-region cols",
    )
    region_min, region_max = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key="region_count_min",
        max_key="region_count_max",
        fallback_min=9,
        fallback_max=14,
        context=f"{SCENE_NAMESPACE} marker-region count",
    )
    grid_pairs = _grid_pair_support(
        row_min=int(row_min),
        row_max=int(row_max),
        col_min=int(col_min),
        col_max=int(col_max),
        region_min=int(region_min),
    )
    if not grid_pairs:
        raise ValueError(f"no valid grid pairs for {SCENE_NAMESPACE} marker regions with current bounds")
    pair_index = _balanced_int(
        list(range(len(grid_pairs))),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{namespace_suffix}.grid_pair",
    )
    rows, cols = grid_pairs[int(pair_index)]
    region_support = list(range(int(region_min), min(int(region_max), int(rows) * int(cols)) + 1))
    region_count = _balanced_int(
        region_support,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{namespace_suffix}.region_count",
    )
    active_cells = _sample_connected_cells(rows=int(rows), cols=int(cols), target_count=int(region_count), rng=rng)

    bin_count = int(params.get("legend_bin_count", _GEN_DEFAULTS.get("legend_bin_count", 3)))
    legend_bins = _make_numeric_bins(int(bin_count))
    regions: List[Dict[str, Any]] = []
    for index, (row, col) in enumerate(active_cells):
        regions.append(
            {
                "region_id": f"region_{index}",
                "row": int(row),
                "col": int(col),
                "bin_index": int(rng.randrange(int(bin_count))),
                "bin_label": str(legend_bins[0]["bin_label"]),
            }
        )
    return int(rows), int(cols), list(active_cells), list(regions), list(legend_bins)


def apply_bin(
    regions_by_id: Dict[str, Dict[str, Any]],
    *,
    region_id: str,
    bin_index: int,
    legend_bins: Sequence[Mapping[str, Any]],
) -> None:
    bin_spec = dict(legend_bins[int(bin_index)])
    regions_by_id[str(region_id)]["bin_index"] = int(bin_index)
    regions_by_id[str(region_id)]["bin_label"] = str(bin_spec["bin_label"])
    regions_by_id[str(region_id)]["bin_lower"] = bin_spec.get("lower")
    regions_by_id[str(region_id)]["bin_upper"] = bin_spec.get("upper")
    regions_by_id[str(region_id)]["category"] = str(bin_spec.get("category") or "")


def target_count_support(params: Mapping[str, Any], *, region_count: int) -> List[int]:
    answer_min, answer_max = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key="count_answer_min",
        max_key="count_answer_max",
        fallback_min=3,
        fallback_max=14,
        context=f"{SCENE_NAMESPACE} answer count",
    )
    max_supported = min(int(answer_max), max(1, int(region_count) - 1))
    min_supported = min(int(answer_min), int(max_supported))
    return list(range(int(min_supported), int(max_supported) + 1))


def sample_target_count(
    params: Mapping[str, Any],
    *,
    region_count: int,
    instance_seed: int,
    namespace_suffix: str,
) -> Tuple[int, List[int], Dict[str, float]]:
    support = target_count_support(params, region_count=int(region_count))
    target_count = _balanced_int(
        support,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{namespace_suffix}.target_count",
    )
    return int(target_count), list(support), uniform_probability_map(tuple(support))


def marker_target_count_support(params: Mapping[str, Any], *, region_count: int) -> List[int]:
    answer_min, answer_max = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key="count_answer_min",
        max_key="count_answer_max",
        fallback_min=1,
        fallback_max=5,
        context=f"{SCENE_NAMESPACE} marker answer count",
    )
    max_supported = min(int(answer_max), max(1, int(region_count) - 1))
    min_supported = min(int(answer_min), int(max_supported))
    return list(range(int(min_supported), int(max_supported) + 1))


def sample_marker_target_count(
    params: Mapping[str, Any],
    *,
    region_count: int,
    instance_seed: int,
    namespace_suffix: str,
) -> Tuple[int, List[int], Dict[str, float]]:
    support = marker_target_count_support(params, region_count=int(region_count))
    target_count = _balanced_int(
        support,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{namespace_suffix}.target_count",
    )
    return int(target_count), list(support), uniform_probability_map(tuple(support))


def sample_count_from_param_range(
    params: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
    max_supported: int,
    instance_seed: int,
    namespace_suffix: str,
) -> Tuple[int, List[int], Dict[str, float]]:
    low, high = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context=f"{SCENE_NAMESPACE} {namespace_suffix}",
    )
    support = list(range(int(low), min(int(high), int(max_supported)) + 1))
    if not support:
        raise ValueError(f"no feasible count support for {namespace_suffix}")
    count = _balanced_int(
        support,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{namespace_suffix}",
    )
    return int(count), list(support), uniform_probability_map(tuple(support))


def sample_numeric_value_for_bin(bin_index: int, legend_bins: Sequence[Mapping[str, Any]], *, rng) -> int:
    bin_spec = dict(legend_bins[int(bin_index)])
    lower = int(bin_spec.get("lower") if bin_spec.get("lower") is not None else 0)
    upper = int(bin_spec.get("upper") if bin_spec.get("upper") is not None else lower)
    return int(rng.randrange(int(lower), int(upper) + 1))


def value_bin_index(value: int, legend_bins: Sequence[Mapping[str, Any]]) -> int:
    for bin_spec in legend_bins:
        lower = bin_spec.get("lower")
        upper = bin_spec.get("upper")
        if lower is None or upper is None:
            continue
        if int(lower) <= int(value) <= int(upper):
            return int(bin_spec["bin_index"])
    raise ValueError(f"value {value} does not fall in any legend bin")


def apply_numeric_value(
    regions_by_id: Dict[str, Dict[str, Any]],
    *,
    region_id: str,
    value: int,
    legend_bins: Sequence[Mapping[str, Any]],
) -> None:
    bin_index = value_bin_index(int(value), legend_bins)
    apply_bin(regions_by_id, region_id=str(region_id), bin_index=int(bin_index), legend_bins=legend_bins)
    regions_by_id[str(region_id)]["region_value"] = int(value)


def assign_values_from_bins(
    *,
    regions_by_id: Dict[str, Dict[str, Any]],
    region_ids: Sequence[str],
    legend_bins: Sequence[Mapping[str, Any]],
    rng,
) -> None:
    for region_id in region_ids:
        bin_index = int(regions_by_id[str(region_id)]["bin_index"])
        value = sample_numeric_value_for_bin(int(bin_index), legend_bins, rng=rng)
        apply_numeric_value(
            regions_by_id,
            region_id=str(region_id),
            value=int(value),
            legend_bins=legend_bins,
        )


def assign_matching_bins(
    *,
    regions_by_id: Dict[str, Dict[str, Any]],
    selected_ids: Sequence[str],
    matching_bins: Sequence[int],
    nonmatching_bins: Sequence[int],
    legend_bins: Sequence[Mapping[str, Any]],
    rng,
) -> None:
    selected = {str(region_id) for region_id in selected_ids}
    matching = [int(value) for value in matching_bins]
    nonmatching = [int(value) for value in nonmatching_bins]
    if not matching or not nonmatching:
        raise ValueError("matching and nonmatching bin supports must be non-empty")
    for region_id in sorted(regions_by_id):
        support = matching if str(region_id) in selected else nonmatching
        apply_bin(
            regions_by_id,
            region_id=str(region_id),
            bin_index=int(_choose_random(support, rng=rng)),
            legend_bins=legend_bins,
        )


def assign_region_display_labels(regions_by_id: Dict[str, Dict[str, Any]]) -> None:
    for index, region_id in enumerate(sorted(regions_by_id)):
        regions_by_id[str(region_id)]["region_label"] = chr(ord("A") + int(index))


def assign_region_reference_labels(
    regions_by_id: Dict[str, Dict[str, Any]],
    *,
    rng,
    reserved_labels: Sequence[str] = (),
) -> Dict[str, str]:
    """Assign short text-only map labels that do not collide with legend labels."""

    reserved = {str(label).strip().casefold() for label in reserved_labels if str(label).strip()}
    candidates: List[str] = []
    candidates.extend(chr(ord("A") + index) for index in range(26))
    candidates.extend(f"{chr(ord('A') + letter)}{number}" for letter in range(26) for number in range(1, 10))
    candidates.extend(f"{prefix}{number:02d}" for prefix in ("P", "R", "Q", "N", "Z") for number in range(1, 100))
    usable = [
        label
        for label in candidates
        if 1 <= len(label) <= 4 and any(character.isalpha() for character in label) and label.casefold() not in reserved
    ]
    if len(usable) < len(regions_by_id):
        raise ValueError("not enough distinct region reference labels")
    region_ids = list(sorted(regions_by_id))
    rng.shuffle(region_ids)
    rng.shuffle(usable)
    region_label_map: Dict[str, str] = {}
    for region_id, label in zip(region_ids, usable, strict=False):
        regions_by_id[str(region_id)]["region_label"] = str(label)
        region_label_map[str(region_id)] = str(label)
    return dict(region_label_map)


__all__ = [
    "_make_category_bins",
    "_marker_label_for_index",
    "_make_numeric_bins",
    "apply_bin",
    "apply_numeric_value",
    "assign_matching_bins",
    "assign_region_display_labels",
    "assign_region_reference_labels",
    "build_synthetic_marker_regions",
    "assign_values_from_bins",
    "build_synthetic_regions",
    "marker_target_count_support",
    "sample_marker_target_count",
    "sample_count_from_param_range",
    "sample_numeric_value_for_bin",
    "sample_target_count",
    "target_count_support",
    "value_bin_index",
]
