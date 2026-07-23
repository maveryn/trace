"""Dataset sampling for region-map chart tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from ....shared.deterministic_sampling import uniform_probability_map
from ....shared.config_defaults import resolve_required_int_bounds
from .assets import GEOGRAPHIC_MAP_ASSETS, WORLD_CATEGORY_TITLE_OPTIONS, WORLD_TITLE_OPTIONS, load_geographic_map_asset
from .data import (
    _make_category_bins,
    _marker_label_for_index,
    _make_numeric_bins,
    apply_bin,
    apply_numeric_value,
    assign_matching_bins,
    assign_region_reference_labels,
    assign_region_display_labels,
    build_synthetic_marker_regions,
    build_synthetic_regions,
    sample_marker_target_count,
    sample_count_from_param_range,
    sample_numeric_value_for_bin,
    sample_target_count,
)
from .defaults import SCENE_NAMESPACE, _GEN_DEFAULTS, resolve_geographic_map_variant, resolve_marker_render_variant
from .projection import (
    WORLD_FILTERED_CONTINENTS,
    _centroid_lonlat_from_rings,
    _selected_geographic_region_adjacency,
    _synthetic_region_adjacency,
    _world_filtered_region_candidates,
)
from .rendering import _resolve_render_params, geographic_visible_component_metadata
from .spatial_primitives import _balanced_int, _choose_random, _reading_order_region_ids


def _world_selected_count_support(params: Mapping[str, Any], *, eligible_count: int) -> List[int]:
    count_min = int(
        params.get(
            "geographic_selected_region_count_min",
            params.get(
                "world_selected_region_count_min",
                _GEN_DEFAULTS.get("geographic_selected_region_count_min", 14),
            ),
        )
    )
    count_max = int(
        params.get(
            "geographic_selected_region_count_max",
            params.get(
                "world_selected_region_count_max",
                _GEN_DEFAULTS.get("geographic_selected_region_count_max", 21),
            ),
        )
    )
    if int(count_min) > int(count_max):
        raise ValueError(f"{SCENE_NAMESPACE} geographic selected region min cannot exceed max")
    high = min(int(count_max), int(eligible_count))
    low = min(int(count_min), int(high))
    return list(range(int(low), int(high) + 1))


def _legend_bin_count(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace_suffix: str,
) -> int:
    bin_count = int(params.get("legend_bin_count", params.get("legend_bin_count_min", _GEN_DEFAULTS.get("legend_bin_count_min", 4))))
    if "legend_bin_count" not in params and ("legend_bin_count_max" in params or "legend_bin_count_max" in _GEN_DEFAULTS):
        bin_min = int(params.get("legend_bin_count_min", _GEN_DEFAULTS.get("legend_bin_count_min", 4)))
        bin_max = int(params.get("legend_bin_count_max", _GEN_DEFAULTS.get("legend_bin_count_max", 6)))
        bin_count = _balanced_int(
            list(range(int(bin_min), int(bin_max) + 1)),
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.{namespace_suffix}.legend_bin_count",
        )
    return int(bin_count)


def _geographic_component_constraints(params: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "min_area_px": float(
            params.get(
                "geographic_min_target_component_area_px",
                _GEN_DEFAULTS.get("geographic_min_target_component_area_px", 400),
            )
        ),
        "min_width_px": float(
            params.get(
                "geographic_min_target_component_width_px",
                _GEN_DEFAULTS.get("geographic_min_target_component_width_px", 12),
            )
        ),
        "min_height_px": float(
            params.get(
                "geographic_min_target_component_height_px",
                _GEN_DEFAULTS.get("geographic_min_target_component_height_px", 10),
            )
        ),
    }


def _filter_geographic_regions_by_visible_component(
    asset_regions: Sequence[Mapping[str, Any]],
    *,
    asset: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    namespace_suffix: str,
    legend_bin_count: int,
    categorical: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Keep geographic regions whose largest projected connected component is visible enough."""

    constraints = _geographic_component_constraints(params)
    render_params = _resolve_render_params(
        {**dict(params), "_render_style_seed": int(instance_seed)},
        legend_count=int(legend_bin_count),
        categorical=bool(categorical),
    )
    eligible_regions: List[Dict[str, Any]] = []
    for asset_region in asset_regions:
        component_meta = geographic_visible_component_metadata(
            asset_region,
            asset=asset,
            render_params=render_params,
        )
        bbox = [float(value) for value in component_meta.get("visible_component_bbox_px", [0.0, 0.0, 0.0, 0.0])]
        width = max(0.0, float(bbox[2] - bbox[0]))
        height = max(0.0, float(bbox[3] - bbox[1]))
        if (
            float(component_meta.get("visible_component_area_px", 0.0)) >= float(constraints["min_area_px"])
            and float(width) >= float(constraints["min_width_px"])
            and float(height) >= float(constraints["min_height_px"])
        ):
            row = dict(asset_region)
            row["_visible_component_meta"] = dict(component_meta)
            eligible_regions.append(row)
    if not eligible_regions:
        raise ValueError(
            f"{SCENE_NAMESPACE} no geographic regions satisfy projected component constraints for {namespace_suffix}"
        )
    return eligible_regions, dict(constraints)


def _build_geographic_regions(
    *,
    categorical: bool,
    params: Mapping[str, Any],
    instance_seed: int,
    rng,
    namespace_suffix: str,
    force_world: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[int], Dict[str, Any]]:
    """Sample visible geographic regions and legend bins without task identity routing."""

    if bool(force_world):
        geographic_map_variant = "world_countries"
        geographic_map_variant_probabilities = {"world_countries": 1.0}
    else:
        geographic_map_variant, geographic_map_variant_probabilities = resolve_geographic_map_variant(
            params,
            instance_seed=int(instance_seed),
        )
    asset = load_geographic_map_asset(str(geographic_map_variant))
    asset_regions = [dict(region) for region in asset.get("regions", []) if isinstance(region, Mapping)]
    bin_count = _legend_bin_count(
        params,
        instance_seed=int(instance_seed),
        namespace_suffix=str(namespace_suffix),
    )
    eligible_regions, component_constraints = _filter_geographic_regions_by_visible_component(
        [dict(region) for region in asset_regions if bool(region.get("question_eligible"))],
        asset=asset,
        params=params,
        instance_seed=int(instance_seed),
        namespace_suffix=str(namespace_suffix),
        legend_bin_count=int(bin_count),
        categorical=bool(categorical),
    )
    if not eligible_regions:
        raise ValueError("geographic map asset has no question-eligible regions")

    selected_count_support = _world_selected_count_support(params, eligible_count=len(eligible_regions))
    selected_count = _balanced_int(
        selected_count_support,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{namespace_suffix}.selected_region_count",
    )
    selected_asset_regions = rng.sample(
        sorted(eligible_regions, key=lambda item: str(item["region_id"])),
        int(selected_count),
    )

    legend_bins = _make_category_bins(int(bin_count), rng=rng) if bool(categorical) else _make_numeric_bins(int(bin_count))
    regions = _adapt_asset_regions(selected_asset_regions, asset=asset, legend_bins=legend_bins, rng=rng)
    asset_meta = _asset_meta(asset, map_variant=str(geographic_map_variant), probabilities=geographic_map_variant_probabilities)
    asset_meta["component_constraints"] = dict(component_constraints)
    asset_meta["eligible_region_count_after_component_filter"] = int(len(eligible_regions))
    return list(regions), list(legend_bins), [str(region["region_id"]) for region in regions], list(selected_count_support), dict(asset_meta)


def _adapt_asset_regions(
    asset_regions: Sequence[Mapping[str, Any]],
    *,
    asset: Mapping[str, Any],
    legend_bins: Sequence[Mapping[str, Any]],
    rng,
) -> List[Dict[str, Any]]:
    region_prefix = str(asset.get("region_prefix") or "geo_region")
    regions: List[Dict[str, Any]] = []
    for asset_region in asset_regions:
        region_id = f"{region_prefix}_{asset_region['region_id']}"
        bbox_lonlat = [float(value) for value in asset_region.get("bbox_lonlat", [0.0, 0.0, 0.0, 0.0])]
        component_meta = dict(asset_region.get("_visible_component_meta", {})) if isinstance(asset_region.get("_visible_component_meta"), Mapping) else {}
        regions.append(
            {
                "region_id": str(region_id),
                "asset_region_id": str(asset_region["region_id"]),
                "display_name": str(asset_region.get("display_name") or asset_region["region_id"]),
                "continent": str(asset_region.get("continent") or ""),
                "admin0_a3": str(asset_region.get("admin0_a3") or ""),
                "subregion": str(asset_region.get("subregion") or ""),
                "bbox_lonlat": [round(float(value), 3) for value in bbox_lonlat],
                "centroid_lonlat": _centroid_lonlat_from_rings(asset_region.get("rings", [])),
                "bin_index": int(rng.randrange(int(len(legend_bins)))),
                "bin_label": str(legend_bins[0]["bin_label"]),
                "category": "",
                "is_reference_region": False,
                "visible_component_index": int(component_meta.get("visible_component_index", -1)),
                "visible_component_area_px": float(component_meta.get("visible_component_area_px", 0.0)),
                "visible_component_bbox_px": list(component_meta.get("visible_component_bbox_px", [])),
                "visible_component_center_px": list(component_meta.get("visible_component_center_px", [])),
                "component_count": int(component_meta.get("component_count", 0)),
            }
        )
    return list(regions)


def _asset_meta(asset: Mapping[str, Any], *, map_variant: str, probabilities: Mapping[str, float]) -> Dict[str, Any]:
    return {
        "asset_id": str(asset.get("asset_id") or GEOGRAPHIC_MAP_ASSETS[str(map_variant)]["asset_id"]),
        "map_variant": str(map_variant),
        "map_variant_probabilities": dict(probabilities),
        "display_name": str(asset.get("display_name") or ""),
        "region_noun": str(asset.get("region_noun") or "regions"),
        "object_description": str(asset.get("object_description") or ""),
        "title_options": [str(item) for item in asset.get("title_options", [])],
        "category_title_options": [str(item) for item in asset.get("category_title_options", [])],
        "source": dict(asset.get("source", {})) if isinstance(asset.get("source"), Mapping) else {},
    }


def _base_region_map_dataset(
    *,
    scene_variant: str,
    categorical: bool,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace_suffix: str,
    force_world: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], List[str], Any]:
    """Build a synthetic or geographic map base shared by semantic constructors."""

    if str(scene_variant) not in {"synthetic_region_map", "geographic_region_map"}:
        raise ValueError(f"unsupported region-map scene variant: {scene_variant}")
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.{namespace_suffix}.dataset")
    if str(scene_variant) == "geographic_region_map":
        rows = 0
        cols = 0
        active_cells: List[Tuple[int, int]] = []
        regions, legend_bins, _selected_region_ids, _selected_count_support, map_asset_meta = _build_geographic_regions(
            categorical=bool(categorical),
            params=params,
            instance_seed=int(instance_seed),
            rng=rng,
            namespace_suffix=str(namespace_suffix),
            force_world=bool(force_world),
        )
    else:
        map_asset_meta: Dict[str, Any] = {}
        rows, cols, active_cells, regions, legend_bins = build_synthetic_regions(
            categorical=bool(categorical),
            params=params,
            instance_seed=int(instance_seed),
            rng=rng,
            namespace_suffix=str(namespace_suffix),
        )
    regions_by_id = {str(region["region_id"]): dict(region) for region in regions}
    region_ids = _reading_order_region_ids([str(region["region_id"]) for region in regions], regions_by_id)
    world_scene = str(scene_variant) == "geographic_region_map"
    if world_scene and bool(categorical):
        title_options = list(map_asset_meta.get("category_title_options", [])) or list(WORLD_CATEGORY_TITLE_OPTIONS)
    else:
        title_options = list(map_asset_meta.get("title_options", [])) if world_scene else []
        if not title_options:
            title_options = list(WORLD_TITLE_OPTIONS) if world_scene else [
                "Regional Value Map",
                "District Value Map",
                "Service Area Map",
                "Planning Region Map",
                "County Indicator Map",
            ]
    region_noun = str(map_asset_meta.get("region_noun") or ("countries" if world_scene else "regions"))
    return (
        {
            "scene_title": str(_choose_random(title_options, rng=rng)),
            "scene_variant": str(scene_variant),
            "map_asset_id": str(map_asset_meta.get("asset_id") or ""),
            "geographic_map_variant": str(map_asset_meta.get("map_variant") or ""),
            "geographic_map_variant_probabilities": dict(map_asset_meta.get("map_variant_probabilities", {})),
            "map_display_name": str(map_asset_meta.get("display_name") or ""),
            "map_region_noun": str(region_noun),
            "map_object_description": str(map_asset_meta.get("object_description") or ""),
            "map_source": dict(map_asset_meta.get("source", {})),
            "geographic_component_constraints": dict(map_asset_meta.get("component_constraints", {})),
            "geographic_eligible_region_count_after_component_filter": int(map_asset_meta.get("eligible_region_count_after_component_filter", 0)),
            "rows": int(rows),
            "cols": int(cols),
            "active_cells": [[int(row), int(col)] for row, col in active_cells],
            "legend_bins": [dict(item) for item in legend_bins],
            "question_params": {},
        },
        dict(regions_by_id),
        list(region_ids),
        rng,
    )


def _finalize_region_dataset(
    *,
    base_dataset: Mapping[str, Any],
    regions_by_id: Mapping[str, Mapping[str, Any]],
    region_ids: Sequence[str],
    annotation_region_ids: Sequence[str],
    answer_value: int,
    target_count: int,
    target_count_probabilities: Mapping[str, float],
    question_params: Mapping[str, Any],
    target_bin_indices: Sequence[int],
    nonmatching_bin_indices: Sequence[int],
    threshold_direction: str = "",
) -> Dict[str, Any]:
    final_regions = [dict(regions_by_id[str(region_id)]) for region_id in region_ids]
    return {
        **dict(base_dataset),
        "region_count": int(len(final_regions)),
        "regions": final_regions,
        "regions_by_id": {str(region["region_id"]): dict(region) for region in final_regions},
        "annotation_region_ids": [str(region_id) for region_id in annotation_region_ids],
        "answer_value": int(answer_value),
        "answer_type": "integer",
        "target_count": int(target_count),
        "target_count_probabilities": dict(target_count_probabilities),
        "question_params": {**dict(base_dataset.get("question_params", {})), **dict(question_params)},
        "target_bin_indices": [int(value) for value in target_bin_indices],
        "nonmatching_bin_indices": [int(value) for value in nonmatching_bin_indices],
        "threshold_direction": str(threshold_direction),
    }


def _threshold_bins(
    *,
    threshold_direction: str,
    legend_bins: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    instance_seed: int,
    namespace_suffix: str,
) -> Tuple[int, str, List[int], List[int]]:
    bin_count = int(len(legend_bins))
    if str(threshold_direction) == "greater_than":
        threshold_bin = _balanced_int(
            list(range(0, bin_count - 1)),
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.{namespace_suffix}.threshold.greater_than",
        )
        threshold_value = int(legend_bins[int(threshold_bin)]["upper"])
        return (
            int(threshold_value),
            f"greater than {threshold_value}",
            list(range(int(threshold_bin) + 1, bin_count)),
            list(range(0, int(threshold_bin) + 1)),
        )
    if str(threshold_direction) == "less_than":
        threshold_bin = _balanced_int(
            list(range(1, bin_count)),
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.{namespace_suffix}.threshold.less_than",
        )
        threshold_value = int(legend_bins[int(threshold_bin)]["lower"])
        return (
            int(threshold_value),
            f"less than {threshold_value}",
            list(range(0, int(threshold_bin))),
            list(range(int(threshold_bin), bin_count)),
        )
    raise ValueError(f"unsupported threshold direction: {threshold_direction}")


def construct_numeric_threshold_dataset(
    *,
    scene_variant: str,
    threshold_direction: str,
    threshold_direction_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    """Assign bins so exactly the selected regions satisfy one numeric threshold."""

    base, regions_by_id, region_ids, rng = _base_region_map_dataset(
        scene_variant=str(scene_variant),
        categorical=False,
        params=params,
        instance_seed=int(instance_seed),
        namespace_suffix="numeric_threshold_regions",
    )
    target_count, target_support, target_probabilities = sample_target_count(
        params,
        region_count=len(region_ids),
        instance_seed=int(instance_seed),
        namespace_suffix="numeric_threshold_regions",
    )
    selected_ids = set(rng.sample(list(region_ids), int(target_count)))
    annotation_region_ids = _reading_order_region_ids(list(selected_ids), regions_by_id)
    threshold_value, threshold_phrase, target_bin_indices, nonmatching_bin_indices = _threshold_bins(
        threshold_direction=str(threshold_direction),
        legend_bins=base["legend_bins"],
        params=params,
        instance_seed=int(instance_seed),
        namespace_suffix="numeric_threshold_regions",
    )
    assign_matching_bins(
        regions_by_id=regions_by_id,
        selected_ids=annotation_region_ids,
        matching_bins=target_bin_indices,
        nonmatching_bins=nonmatching_bin_indices,
        legend_bins=base["legend_bins"],
        rng=rng,
    )
    return _finalize_region_dataset(
        base_dataset=base,
        regions_by_id=regions_by_id,
        region_ids=region_ids,
        annotation_region_ids=annotation_region_ids,
        answer_value=int(target_count),
        target_count=int(target_count),
        target_count_probabilities=target_probabilities,
        question_params={
            "threshold_direction": str(threshold_direction),
            "threshold_direction_probabilities": dict(threshold_direction_probabilities),
            "threshold_value": int(threshold_value),
            "threshold_phrase": str(threshold_phrase),
            "target_bin_indices": [int(value) for value in target_bin_indices],
        },
        target_bin_indices=target_bin_indices,
        nonmatching_bin_indices=nonmatching_bin_indices,
        threshold_direction=str(threshold_direction),
    )


def construct_numeric_interval_dataset(
    *,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    """Assign bins so exactly the selected regions fall inside one numeric interval."""

    base, regions_by_id, region_ids, rng = _base_region_map_dataset(
        scene_variant=str(scene_variant),
        categorical=False,
        params=params,
        instance_seed=int(instance_seed),
        namespace_suffix="numeric_interval_regions",
    )
    target_count, _target_support, target_probabilities = sample_target_count(
        params,
        region_count=len(region_ids),
        instance_seed=int(instance_seed),
        namespace_suffix="numeric_interval_regions",
    )
    selected_ids = set(rng.sample(list(region_ids), int(target_count)))
    annotation_region_ids = _reading_order_region_ids(list(selected_ids), regions_by_id)
    bin_count = int(len(base["legend_bins"]))
    max_span = min(4, max(2, bin_count - 1))
    span = _balanced_int(
        list(range(2, max_span + 1)),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.numeric_interval_regions.interval_span",
    )
    low_bin = _balanced_int(
        list(range(0, bin_count - int(span) + 1)),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.numeric_interval_regions.interval_low_bin",
    )
    high_bin = int(low_bin) + int(span) - 1
    target_bin_indices = list(range(int(low_bin), int(high_bin) + 1))
    nonmatching_bin_indices = [index for index in range(bin_count) if index not in set(target_bin_indices)]
    lower = int(base["legend_bins"][int(low_bin)]["lower"])
    upper = int(base["legend_bins"][int(high_bin)]["upper"])
    assign_matching_bins(
        regions_by_id=regions_by_id,
        selected_ids=annotation_region_ids,
        matching_bins=target_bin_indices,
        nonmatching_bins=nonmatching_bin_indices,
        legend_bins=base["legend_bins"],
        rng=rng,
    )
    return _finalize_region_dataset(
        base_dataset=base,
        regions_by_id=regions_by_id,
        region_ids=region_ids,
        annotation_region_ids=annotation_region_ids,
        answer_value=int(target_count),
        target_count=int(target_count),
        target_count_probabilities=target_probabilities,
        question_params={
            "interval_lower": int(lower),
            "interval_upper": int(upper),
            "interval_phrase": f"between {lower} and {upper}, inclusive",
            "interval_bin_span": int(span),
            "target_bin_indices": [int(value) for value in target_bin_indices],
        },
        target_bin_indices=target_bin_indices,
        nonmatching_bin_indices=nonmatching_bin_indices,
    )


def construct_categorical_count_dataset(
    *,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    """Assign categories so exactly the selected regions match the target category."""

    base, regions_by_id, region_ids, rng = _base_region_map_dataset(
        scene_variant=str(scene_variant),
        categorical=True,
        params=params,
        instance_seed=int(instance_seed),
        namespace_suffix="category_regions",
    )
    target_count, _target_support, target_probabilities = sample_target_count(
        params,
        region_count=len(region_ids),
        instance_seed=int(instance_seed),
        namespace_suffix="category_regions",
    )
    selected_ids = set(rng.sample(list(region_ids), int(target_count)))
    annotation_region_ids = _reading_order_region_ids(list(selected_ids), regions_by_id)
    bin_count = int(len(base["legend_bins"]))
    category_index = _balanced_int(
        list(range(bin_count)),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.category_regions.category_index",
    )
    target_bin_indices = [int(category_index)]
    nonmatching_bin_indices = [index for index in range(bin_count) if int(index) != int(category_index)]
    assign_matching_bins(
        regions_by_id=regions_by_id,
        selected_ids=annotation_region_ids,
        matching_bins=target_bin_indices,
        nonmatching_bins=nonmatching_bin_indices,
        legend_bins=base["legend_bins"],
        rng=rng,
    )
    return _finalize_region_dataset(
        base_dataset=base,
        regions_by_id=regions_by_id,
        region_ids=region_ids,
        annotation_region_ids=annotation_region_ids,
        answer_value=int(target_count),
        target_count=int(target_count),
        target_count_probabilities=target_probabilities,
        question_params={
            "category_index": int(category_index),
            "category_label": str(base["legend_bins"][int(category_index)]["bin_label"]),
            "target_bin_indices": [int(category_index)],
        },
        target_bin_indices=target_bin_indices,
        nonmatching_bin_indices=nonmatching_bin_indices,
    )


def _world_filtered_base(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    rng,
    namespace_suffix: str,
    target_count: int,
    same_group_count: int = 0,
    categorical: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], List[str], List[str], str, List[str]]:
    """Create a world-map subset with a feasible target continent and distractors."""

    asset = load_geographic_map_asset("world_countries")
    asset_regions = [dict(region) for region in asset.get("regions", []) if isinstance(region, Mapping)]
    bin_count = _legend_bin_count(
        params,
        instance_seed=int(instance_seed),
        namespace_suffix=str(namespace_suffix),
    )
    eligible_regions, component_constraints = _filter_geographic_regions_by_visible_component(
        _world_filtered_region_candidates([dict(region) for region in asset_regions if bool(region.get("question_eligible"))]),
        asset=asset,
        params=params,
        instance_seed=int(instance_seed),
        namespace_suffix=str(namespace_suffix),
        legend_bin_count=int(bin_count),
        categorical=bool(categorical),
    )
    selected_count_support = _world_selected_count_support(params, eligible_count=len(eligible_regions))
    selected_count = _balanced_int(
        selected_count_support,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{namespace_suffix}.selected_region_count",
    )
    regions_by_continent = {
        continent: [dict(region) for region in eligible_regions if str(region.get("continent") or "") == str(continent)]
        for continent in WORLD_FILTERED_CONTINENTS
    }
    feasible_continents = [
        str(continent)
        for continent, continent_regions in regions_by_continent.items()
        if len(continent_regions) >= int(target_count) + int(same_group_count)
        and (len(eligible_regions) - len(continent_regions)) >= int(selected_count) - int(target_count) - int(same_group_count)
    ]
    if not feasible_continents:
        raise ValueError("no feasible continent can support the requested filtered world map")
    continent_index = _balanced_int(
        list(range(len(feasible_continents))),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{namespace_suffix}.target_continent",
    )
    target_continent = str(feasible_continents[int(continent_index)])
    same_continent_pool = sorted(regions_by_continent[str(target_continent)], key=lambda item: str(item["region_id"]))
    target_regions = rng.sample(same_continent_pool, int(target_count))
    remaining_same_pool = [dict(region) for region in same_continent_pool if str(region["region_id"]) not in {str(item["region_id"]) for item in target_regions}]
    same_group_distractors = rng.sample(remaining_same_pool, int(same_group_count)) if int(same_group_count) > 0 else []
    other_pool = sorted(
        [dict(region) for region in eligible_regions if str(region.get("continent") or "") != str(target_continent)],
        key=lambda item: str(item["region_id"]),
    )
    other_count = int(selected_count) - int(target_count) - int(same_group_count)
    distractor_regions = rng.sample(other_pool, int(other_count))
    selected_asset_regions = list(target_regions) + list(same_group_distractors) + list(distractor_regions)
    rng.shuffle(selected_asset_regions)
    legend_bins = _make_category_bins(int(bin_count), rng=rng) if bool(categorical) else _make_numeric_bins(int(bin_count))
    regions = _adapt_asset_regions(selected_asset_regions, asset=asset, legend_bins=legend_bins, rng=rng)
    regions_by_id = {str(region["region_id"]): dict(region) for region in regions}
    region_ids = _reading_order_region_ids([str(region["region_id"]) for region in regions], regions_by_id)
    asset_meta = _asset_meta(asset, map_variant="world_countries", probabilities={"world_countries": 1.0})
    base = {
        "scene_title": str(_choose_random(asset_meta.get("title_options") or WORLD_TITLE_OPTIONS, rng=rng)),
        "scene_variant": "geographic_region_map",
        "map_asset_id": str(asset_meta["asset_id"]),
        "geographic_map_variant": "world_countries",
        "geographic_map_variant_probabilities": {"world_countries": 1.0},
        "map_display_name": str(asset_meta.get("display_name") or "World"),
        "map_region_noun": str(asset_meta.get("region_noun") or "countries"),
        "map_object_description": str(asset_meta.get("object_description") or ""),
        "map_source": dict(asset_meta.get("source", {})),
        "geographic_component_constraints": dict(component_constraints),
        "geographic_eligible_region_count_after_component_filter": int(len(eligible_regions)),
        "rows": 0,
        "cols": 0,
        "active_cells": [],
        "legend_bins": [dict(item) for item in legend_bins],
        "question_params": {},
    }
    target_asset_ids = {str(item["region_id"]) for item in target_regions}
    annotation_region_ids = _reading_order_region_ids(
        [
            str(region_id)
            for region_id, region in regions_by_id.items()
            if str(region.get("asset_region_id") or "") in target_asset_ids
        ],
        regions_by_id,
    )
    same_group_ids = [
        f"country_{region['region_id']}"
        for region in same_group_distractors
    ]
    return dict(base), dict(regions_by_id), list(region_ids), list(annotation_region_ids), str(target_continent), list(same_group_ids)


def construct_named_region_set_total_dataset(
    *,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    """Label selected regions and expose visible values whose sum is the answer."""

    base, regions_by_id, region_ids, rng = _base_region_map_dataset(
        scene_variant=str(scene_variant),
        categorical=False,
        params=params,
        instance_seed=int(instance_seed),
        namespace_suffix="named_region_set_total",
    )
    assign_region_display_labels(regions_by_id)
    for region_id in region_ids:
        value = sample_numeric_value_for_bin(int(rng.randrange(len(base["legend_bins"]))), base["legend_bins"], rng=rng)
        apply_numeric_value(regions_by_id, region_id=str(region_id), value=int(value), legend_bins=base["legend_bins"])
    target_count, _support, target_probabilities = sample_count_from_param_range(
        params,
        min_key="region_set_size_min",
        max_key="region_set_size_max",
        fallback_min=3,
        fallback_max=5,
        max_supported=max(1, len(region_ids) - 1),
        instance_seed=int(instance_seed),
        namespace_suffix="named_region_set_total.region_set_size",
    )
    annotation_region_ids = _reading_order_region_ids(rng.sample(list(region_ids), int(target_count)), regions_by_id)
    labels = [str(regions_by_id[str(region_id)].get("region_label") or "") for region_id in annotation_region_ids]
    target_bin_indices = sorted({int(regions_by_id[str(region_id)]["bin_index"]) for region_id in annotation_region_ids})
    answer_value = int(sum(int(regions_by_id[str(region_id)]["region_value"]) for region_id in annotation_region_ids))
    set_name = str(_choose_random(("Focus set", "Review set", "Priority set", "Audit set", "Target set"), rng=rng))
    return _finalize_region_dataset(
        base_dataset=base,
        regions_by_id=regions_by_id,
        region_ids=region_ids,
        annotation_region_ids=annotation_region_ids,
        answer_value=int(answer_value),
        target_count=int(target_count),
        target_count_probabilities=target_probabilities,
        question_params={
            "region_set_name": str(set_name),
            "region_set_region_ids": [str(region_id) for region_id in annotation_region_ids],
            "region_set_labels": labels,
            "region_set_label_list": ", ".join(f'"{label}"' for label in labels),
            "target_region_values": {
                str(region_id): int(regions_by_id[str(region_id)]["region_value"])
                for region_id in annotation_region_ids
            },
            "target_bin_indices": [int(value) for value in target_bin_indices],
        },
        target_bin_indices=target_bin_indices,
        nonmatching_bin_indices=[index for index in range(len(base["legend_bins"])) if index not in set(target_bin_indices)],
    )


def construct_group_category_count_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    """Build a world-map task counting countries in one continent and category."""

    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.group_category_count.dataset")
    target_count, _support, target_probabilities = sample_count_from_param_range(
        params,
        min_key="group_category_target_count_min",
        max_key="group_category_target_count_max",
        fallback_min=1,
        fallback_max=5,
        max_supported=8,
        instance_seed=int(instance_seed),
        namespace_suffix="group_category_count.target_count",
    )
    same_group_count, _same_support, _same_probabilities = sample_count_from_param_range(
        params,
        min_key="group_category_same_group_distractor_count_min",
        max_key="group_category_same_group_distractor_count_max",
        fallback_min=1,
        fallback_max=3,
        max_supported=6,
        instance_seed=int(instance_seed),
        namespace_suffix="group_category_count.same_group_distractor_count",
    )
    base, regions_by_id, region_ids, annotation_region_ids, continent, same_group_ids = _world_filtered_base(
        params=params,
        instance_seed=int(instance_seed),
        rng=rng,
        namespace_suffix="group_category_count",
        target_count=int(target_count),
        same_group_count=int(same_group_count),
        categorical=True,
    )
    category_index = _balanced_int(
        list(range(len(base["legend_bins"]))),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.group_category_count.category_index",
    )
    target_bin_indices = [int(category_index)]
    nonmatching_bin_indices = [index for index in range(len(base["legend_bins"])) if int(index) != int(category_index)]
    outside_group_ids = [
        str(region_id)
        for region_id, region in regions_by_id.items()
        if str(region.get("continent") or "") != str(continent)
    ]
    outside_target_count, _outside_support, _outside_probabilities = sample_count_from_param_range(
        params,
        min_key="group_category_outside_group_target_count_min",
        max_key="group_category_outside_group_target_count_max",
        fallback_min=1,
        fallback_max=3,
        max_supported=max(1, len(outside_group_ids)),
        instance_seed=int(instance_seed),
        namespace_suffix="group_category_count.outside_group_target_count",
    )
    outside_group_target_ids = (
        rng.sample(sorted(outside_group_ids), int(outside_target_count))
        if outside_group_ids
        else []
    )
    assign_matching_bins(
        regions_by_id=regions_by_id,
        selected_ids=[*annotation_region_ids, *outside_group_target_ids],
        matching_bins=target_bin_indices,
        nonmatching_bins=nonmatching_bin_indices,
        legend_bins=base["legend_bins"],
        rng=rng,
    )
    qparams = {
        "continent_label": str(continent),
        "target_continent": str(continent),
        "category_index": int(category_index),
        "category_label": str(base["legend_bins"][int(category_index)]["bin_label"]),
        "target_bin_indices": [int(category_index)],
        "same_group_distractor_region_ids": [str(region_id) for region_id in same_group_ids],
        "outside_group_target_region_ids": [str(region_id) for region_id in outside_group_target_ids],
    }
    return _finalize_region_dataset(
        base_dataset=base,
        regions_by_id=regions_by_id,
        region_ids=region_ids,
        annotation_region_ids=annotation_region_ids,
        answer_value=int(len(annotation_region_ids)),
        target_count=int(len(annotation_region_ids)),
        target_count_probabilities=target_probabilities,
        question_params=qparams,
        target_bin_indices=target_bin_indices,
        nonmatching_bin_indices=nonmatching_bin_indices,
    )


def _synthetic_adjacent_base(
    *,
    categorical: bool,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace_suffix: str,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], List[str], List[str], str, Any, Dict[str, float]]:
    """Choose a highlighted synthetic region with enough neighboring answer candidates."""

    base, regions_by_id, region_ids, rng = _base_region_map_dataset(
        scene_variant="synthetic_region_map",
        categorical=bool(categorical),
        params=params,
        instance_seed=int(instance_seed),
        namespace_suffix=str(namespace_suffix),
    )
    adjacency = _synthetic_region_adjacency(regions_by_id)
    answer_min, answer_max = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key="count_answer_min",
        max_key="count_answer_max",
        fallback_min=1,
        fallback_max=6,
        context=f"{SCENE_NAMESPACE} adjacent answer count",
    )
    feasible_counts = sorted(
        {
            count
            for neighbor_ids in adjacency.values()
            for count in range(int(answer_min), min(int(answer_max), len(neighbor_ids)) + 1)
        }
    )
    if not feasible_counts:
        raise ValueError("no feasible adjacent-region count in selected map")
    target_count = _balanced_int(
        feasible_counts,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{namespace_suffix}.adjacent_target_count",
    )
    target_count_probabilities = uniform_probability_map(tuple(feasible_counts))
    reference_candidates = [
        str(region_id)
        for region_id, neighbor_ids in adjacency.items()
        if len(neighbor_ids) >= int(target_count)
    ]
    reference_index = _balanced_int(
        list(range(len(reference_candidates))),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{namespace_suffix}.reference_region",
    )
    reference_region_id = str(reference_candidates[int(reference_index)])
    regions_by_id[str(reference_region_id)]["is_reference_region"] = True
    legend_labels = [str(bin_spec.get("bin_label") or "") for bin_spec in base["legend_bins"]]
    label_rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.{namespace_suffix}.reference_labels")
    reference_label_map = assign_region_reference_labels(
        regions_by_id,
        rng=label_rng,
        reserved_labels=legend_labels,
    )
    neighbor_ids = list(adjacency.get(str(reference_region_id), []))
    annotation_region_ids = _reading_order_region_ids(rng.sample(neighbor_ids, int(target_count)), regions_by_id)
    base["question_params"] = {
        "reference_region_id": str(reference_region_id),
        "reference_region_label": str(reference_label_map[str(reference_region_id)]),
        "region_reference_label_map": dict(reference_label_map),
        "adjacent_neighbor_region_ids": [str(value) for value in neighbor_ids],
    }
    base["show_region_reference_labels"] = True
    return (
        dict(base),
        dict(regions_by_id),
        list(region_ids),
        list(annotation_region_ids),
        str(reference_region_id),
        rng,
        dict(target_count_probabilities),
    )


def construct_adjacent_same_category_dataset(*, params: Mapping[str, Any], instance_seed: int) -> Dict[str, Any]:
    """Color neighbors so selected adjacent regions match the highlighted category."""

    base, regions_by_id, region_ids, annotation_region_ids, reference_region_id, rng, target_probabilities = _synthetic_adjacent_base(
        categorical=True,
        params=params,
        instance_seed=int(instance_seed),
        namespace_suffix="same_category_neighbors",
    )
    category_index = _balanced_int(
        list(range(len(base["legend_bins"]))),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.same_category_neighbors.category_index",
    )
    target_bin_indices = [int(category_index)]
    nonmatching_bin_indices = [index for index in range(len(base["legend_bins"])) if int(index) != int(category_index)]
    assign_matching_bins(
        regions_by_id=regions_by_id,
        selected_ids=[*annotation_region_ids, str(reference_region_id)],
        matching_bins=target_bin_indices,
        nonmatching_bins=nonmatching_bin_indices,
        legend_bins=base["legend_bins"],
        rng=rng,
    )
    qparams = {
        **dict(base["question_params"]),
        "category_index": int(category_index),
        "category_label": str(base["legend_bins"][int(category_index)]["bin_label"]),
        "target_bin_indices": [int(category_index)],
    }
    return _finalize_region_dataset(
        base_dataset=base,
        regions_by_id=regions_by_id,
        region_ids=region_ids,
        annotation_region_ids=annotation_region_ids,
        answer_value=int(len(annotation_region_ids)),
        target_count=int(len(annotation_region_ids)),
        target_count_probabilities=target_probabilities,
        question_params=qparams,
        target_bin_indices=target_bin_indices,
        nonmatching_bin_indices=nonmatching_bin_indices,
    )


def construct_adjacent_category_dataset(*, params: Mapping[str, Any], instance_seed: int) -> Dict[str, Any]:
    """Color neighbors so selected adjacent regions match a prompted category."""

    base, regions_by_id, region_ids, annotation_region_ids, _reference_region_id, rng, target_probabilities = _synthetic_adjacent_base(
        categorical=True,
        params=params,
        instance_seed=int(instance_seed),
        namespace_suffix="category_neighbors",
    )
    category_index = _balanced_int(
        list(range(len(base["legend_bins"]))),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.category_neighbors.category_index",
    )
    target_bin_indices = [int(category_index)]
    nonmatching_bin_indices = [index for index in range(len(base["legend_bins"])) if int(index) != int(category_index)]
    assign_matching_bins(
        regions_by_id=regions_by_id,
        selected_ids=annotation_region_ids,
        matching_bins=target_bin_indices,
        nonmatching_bins=nonmatching_bin_indices,
        legend_bins=base["legend_bins"],
        rng=rng,
    )
    qparams = {
        **dict(base["question_params"]),
        "category_index": int(category_index),
        "category_label": str(base["legend_bins"][int(category_index)]["bin_label"]),
        "target_bin_indices": [int(category_index)],
    }
    return _finalize_region_dataset(
        base_dataset=base,
        regions_by_id=regions_by_id,
        region_ids=region_ids,
        annotation_region_ids=annotation_region_ids,
        answer_value=int(len(annotation_region_ids)),
        target_count=int(len(annotation_region_ids)),
        target_count_probabilities=target_probabilities,
        question_params=qparams,
        target_bin_indices=target_bin_indices,
        nonmatching_bin_indices=nonmatching_bin_indices,
    )


def construct_adjacent_numeric_threshold_dataset(
    *,
    threshold_direction: str,
    threshold_direction_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    """Assign neighbor bins so selected adjacent regions satisfy one threshold."""

    base, regions_by_id, region_ids, annotation_region_ids, _reference_region_id, rng, target_probabilities = _synthetic_adjacent_base(
        categorical=False,
        params=params,
        instance_seed=int(instance_seed),
        namespace_suffix="threshold_neighbors",
    )
    threshold_value, threshold_phrase, target_bin_indices, nonmatching_bin_indices = _threshold_bins(
        threshold_direction=str(threshold_direction),
        legend_bins=base["legend_bins"],
        params=params,
        instance_seed=int(instance_seed),
        namespace_suffix="threshold_neighbors",
    )
    assign_matching_bins(
        regions_by_id=regions_by_id,
        selected_ids=annotation_region_ids,
        matching_bins=target_bin_indices,
        nonmatching_bins=nonmatching_bin_indices,
        legend_bins=base["legend_bins"],
        rng=rng,
    )
    qparams = {
        **dict(base["question_params"]),
        "threshold_direction": str(threshold_direction),
        "threshold_direction_probabilities": dict(threshold_direction_probabilities),
        "threshold_value": int(threshold_value),
        "threshold_phrase": str(threshold_phrase),
        "target_bin_indices": [int(value) for value in target_bin_indices],
    }
    return _finalize_region_dataset(
        base_dataset=base,
        regions_by_id=regions_by_id,
        region_ids=region_ids,
        annotation_region_ids=annotation_region_ids,
        answer_value=int(len(annotation_region_ids)),
        target_count=int(len(annotation_region_ids)),
        target_count_probabilities=target_probabilities,
        question_params=qparams,
        target_bin_indices=target_bin_indices,
        nonmatching_bin_indices=nonmatching_bin_indices,
        threshold_direction=str(threshold_direction),
    )


def _marker_value_bounds(params: Mapping[str, Any]) -> Tuple[int, int]:
    low, high = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key="marker_value_min",
        max_key="marker_value_max",
        fallback_min=1,
        fallback_max=5,
        context=f"{SCENE_NAMESPACE} marker values",
    )
    if int(high) - int(low) < 2:
        raise ValueError("marker value support must contain at least three values")
    return int(low), int(high)


def _marker_title_options() -> Tuple[str, ...]:
    return (
        "Regional Marker Map",
        "Bubble Indicator Map",
        "Marker Value Map",
        "Area Marker Overview",
        "Region Bubble Map",
    )


def _base_marker_layer_dataset(
    *,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace_suffix: str,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], List[str], Any]:
    """Build the scene-neutral marker-layer dataset before objective-specific constraints."""

    if str(scene_variant) not in {"synthetic_region_map", "geographic_region_map"}:
        raise ValueError(f"unsupported region-map marker scene variant: {scene_variant}")
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.{namespace_suffix}.dataset")
    if str(scene_variant) == "geographic_region_map":
        rows = 0
        cols = 0
        active_cells: List[Tuple[int, int]] = []
        regions, legend_bins, _selected_region_ids, _selected_count_support, map_asset_meta = _build_geographic_regions(
            categorical=False,
            params=params,
            instance_seed=int(instance_seed),
            rng=rng,
            namespace_suffix=str(namespace_suffix),
        )
    else:
        map_asset_meta: Dict[str, Any] = {}
        rows, cols, active_cells, regions, legend_bins = build_synthetic_marker_regions(
            params=params,
            instance_seed=int(instance_seed),
            rng=rng,
            namespace_suffix=str(namespace_suffix),
        )

    regions_by_id: Dict[str, Dict[str, Any]] = {str(region["region_id"]): dict(region) for region in regions}
    region_ids = _reading_order_region_ids([str(region["region_id"]) for region in regions], regions_by_id)
    if len(region_ids) < 3:
        raise ValueError("region-map marker layer needs at least three visible regions")

    marker_render_variant, marker_render_variant_probabilities = resolve_marker_render_variant(
        params,
        instance_seed=int(instance_seed),
    )
    value_min, value_max = _marker_value_bounds(params)
    value_support = list(range(int(value_min), int(value_max) + 1))
    for index, region_id in enumerate(region_ids):
        regions_by_id[str(region_id)]["marker_label"] = _marker_label_for_index(int(index))
        regions_by_id[str(region_id)]["marker_value"] = int(_choose_random(value_support, rng=rng))
        regions_by_id[str(region_id)]["has_marker"] = True
        regions_by_id[str(region_id)]["bin_index"] = 0
        regions_by_id[str(region_id)]["bin_label"] = ""
        regions_by_id[str(region_id)]["category"] = ""

    geographic_scene = str(scene_variant) == "geographic_region_map"
    region_noun = str(map_asset_meta.get("region_noun") or ("countries" if geographic_scene else "regions"))
    map_display = str(map_asset_meta.get("display_name") or "geographic")
    object_description = (
        f"a {map_display.lower() if map_display.lower() == 'world' else map_display} map with marker bubbles over selected {region_noun}"
        if geographic_scene
        else "a synthetic map with marker bubbles over selected regions"
    )
    base_dataset = {
        "scene_title": str(_choose_random(_marker_title_options(), rng=rng)),
        "scene_variant": str(scene_variant),
        "map_asset_id": str(map_asset_meta.get("asset_id") or ("world_countries" if geographic_scene else "")),
        "geographic_map_variant": str(map_asset_meta.get("map_variant") or ""),
        "geographic_map_variant_probabilities": dict(map_asset_meta.get("map_variant_probabilities", {})),
        "map_display_name": str(map_asset_meta.get("display_name") or ""),
        "map_region_noun": str(region_noun),
        "map_object_description": str(object_description),
        "map_source": dict(map_asset_meta.get("source", {})),
        "geographic_component_constraints": dict(map_asset_meta.get("component_constraints", {})),
        "geographic_eligible_region_count_after_component_filter": int(
            map_asset_meta.get("eligible_region_count_after_component_filter", 0) or 0
        ),
        "rows": int(rows),
        "cols": int(cols),
        "active_cells": [[int(row), int(col)] for row, col in active_cells],
        "legend_bins": [dict(item) for item in legend_bins],
        "question_params": {
            "marker_render_variant": str(marker_render_variant),
            "marker_render_variant_probabilities": dict(marker_render_variant_probabilities),
            "marker_region_ids": [str(region_id) for region_id in region_ids],
        },
        "marker_render_variant": str(marker_render_variant),
        "marker_render_variant_probabilities": dict(marker_render_variant_probabilities),
        "marker_value_min": int(value_min),
        "marker_value_max": int(value_max),
    }
    return dict(base_dataset), regions_by_id, list(region_ids), rng


def _finalize_marker_dataset(
    *,
    base_dataset: Mapping[str, Any],
    regions_by_id: Mapping[str, Mapping[str, Any]],
    region_ids: Sequence[str],
    annotation_region_ids: Sequence[str],
    answer_type: str,
    answer_value: Any,
    target_count: int,
    target_count_probabilities: Mapping[str, float],
    question_params: Mapping[str, Any],
    show_marker_labels: bool,
) -> Dict[str, Any]:
    final_regions = [dict(regions_by_id[str(region_id)]) for region_id in region_ids]
    return {
        **dict(base_dataset),
        "region_count": int(len(final_regions)),
        "regions": final_regions,
        "regions_by_id": {str(region["region_id"]): dict(region) for region in final_regions},
        "annotation_region_ids": [str(region_id) for region_id in annotation_region_ids],
        "answer_value": answer_value,
        "answer_type": str(answer_type),
        "target_count": int(target_count),
        "target_count_probabilities": dict(target_count_probabilities),
        "question_params": {**dict(base_dataset.get("question_params", {})), **dict(question_params)},
        "target_bin_indices": [],
        "nonmatching_bin_indices": [],
        "show_marker_labels": bool(show_marker_labels),
    }


def construct_marker_threshold_dataset(
    *,
    scene_variant: str,
    threshold_direction: str,
    threshold_direction_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    """Constrain marker values so exactly the sampled target regions satisfy the threshold."""

    base, regions_by_id, region_ids, rng = _base_marker_layer_dataset(
        scene_variant=str(scene_variant),
        params=params,
        instance_seed=int(instance_seed),
        namespace_suffix="marker_threshold_count",
    )
    value_min = int(base["marker_value_min"])
    value_max = int(base["marker_value_max"])
    target_count, target_support, target_probabilities = sample_marker_target_count(
        params,
        region_count=len(region_ids),
        instance_seed=int(instance_seed),
        namespace_suffix="marker_threshold_count",
    )
    target_ids = set(rng.sample(list(region_ids), int(target_count)))
    annotation_region_ids = _reading_order_region_ids(list(target_ids), dict(regions_by_id))
    if str(threshold_direction) == "greater_than":
        threshold_value = _balanced_int(
            list(range(int(value_min), int(value_max))),
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.marker_threshold_count.threshold.greater_than",
        )
        matching_values = list(range(int(threshold_value) + 1, int(value_max) + 1))
        nonmatching_values = list(range(int(value_min), int(threshold_value) + 1))
        threshold_phrase = f"greater than {threshold_value}"
    elif str(threshold_direction) == "less_than":
        threshold_value = _balanced_int(
            list(range(int(value_min) + 1, int(value_max) + 1)),
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.marker_threshold_count.threshold.less_than",
        )
        matching_values = list(range(int(value_min), int(threshold_value)))
        nonmatching_values = list(range(int(threshold_value), int(value_max) + 1))
        threshold_phrase = f"less than {threshold_value}"
    else:
        raise ValueError(f"unsupported marker threshold direction: {threshold_direction}")

    for region_id in region_ids:
        support = matching_values if str(region_id) in target_ids else nonmatching_values
        regions_by_id[str(region_id)]["marker_value"] = int(_choose_random(support, rng=rng))
        regions_by_id[str(region_id)]["is_target_marker_region"] = bool(str(region_id) in target_ids)

    return _finalize_marker_dataset(
        base_dataset=base,
        regions_by_id=regions_by_id,
        region_ids=region_ids,
        annotation_region_ids=annotation_region_ids,
        answer_type="integer",
        answer_value=int(target_count),
        target_count=int(target_count),
        target_count_probabilities=target_probabilities,
        question_params={
            "threshold_direction": str(threshold_direction),
            "threshold_direction_probabilities": dict(threshold_direction_probabilities),
            "threshold_value": int(threshold_value),
            "threshold_phrase": str(threshold_phrase),
            "marker_value_min": int(value_min),
            "marker_value_max": int(value_max),
            "target_count_support": [int(value) for value in target_support],
        },
        show_marker_labels=False,
    )


def construct_marker_extremum_dataset(
    *,
    scene_variant: str,
    extremum_direction: str,
    extremum_direction_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    """Constrain marker values so one visible region has the unique selected extremum."""

    base, regions_by_id, region_ids, rng = _base_marker_layer_dataset(
        scene_variant=str(scene_variant),
        params=params,
        instance_seed=int(instance_seed),
        namespace_suffix="marker_extremum_label",
    )
    value_min = int(base["marker_value_min"])
    value_max = int(base["marker_value_max"])
    marked_region_count_max = max(1, min(10, int(params.get("marked_region_count_max", 10))))
    marked_region_count = min(int(marked_region_count_max), len(region_ids))
    if marked_region_count < len(region_ids):
        marked_region_id_set = set(rng.sample(list(region_ids), int(marked_region_count)))
        marked_region_ids = _reading_order_region_ids(list(marked_region_id_set), dict(regions_by_id))
    else:
        marked_region_ids = list(region_ids)
    for index, marked_region_id in enumerate(marked_region_ids):
        regions_by_id[str(marked_region_id)]["marker_label"] = _marker_label_for_index(int(index))
        regions_by_id[str(marked_region_id)]["has_marker"] = True
    for region_id in region_ids:
        if str(region_id) not in set(marked_region_ids):
            regions_by_id[str(region_id)]["marker_label"] = ""
            regions_by_id[str(region_id)]["has_marker"] = False

    answer_index = _balanced_int(
        list(range(len(marked_region_ids))),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.marker_extremum_label.answer_region_index",
    )
    answer_region_id = str(marked_region_ids[int(answer_index)])
    if str(extremum_direction) == "largest":
        answer_marker_value = int(value_max)
        distractor_values = list(range(int(value_min), int(value_max)))
        extremum_word = "largest"
    elif str(extremum_direction) == "smallest":
        answer_marker_value = int(value_min)
        distractor_values = list(range(int(value_min) + 1, int(value_max) + 1))
        extremum_word = "smallest"
    else:
        raise ValueError(f"unsupported marker extremum direction: {extremum_direction}")

    marked_region_id_set = set(marked_region_ids)
    for region_id in region_ids:
        if str(region_id) not in marked_region_id_set:
            regions_by_id[str(region_id)]["marker_value"] = int(_choose_random(distractor_values, rng=rng))
            regions_by_id[str(region_id)]["is_target_marker_region"] = False
        elif str(region_id) == str(answer_region_id):
            regions_by_id[str(region_id)]["marker_value"] = int(answer_marker_value)
            regions_by_id[str(region_id)]["is_target_marker_region"] = True
        else:
            regions_by_id[str(region_id)]["marker_value"] = int(_choose_random(distractor_values, rng=rng))
            regions_by_id[str(region_id)]["is_target_marker_region"] = False
    answer_value = str(regions_by_id[str(answer_region_id)]["marker_label"])
    return _finalize_marker_dataset(
        base_dataset=base,
        regions_by_id=regions_by_id,
        region_ids=region_ids,
        annotation_region_ids=[str(answer_region_id)],
        answer_type="string",
        answer_value=str(answer_value),
        target_count=1,
        target_count_probabilities=uniform_probability_map((1,)),
        question_params={
            "extremum_direction": str(extremum_direction),
            "extremum_direction_probabilities": dict(extremum_direction_probabilities),
            "extremum_word": str(extremum_word),
            "answer_region_id": str(answer_region_id),
            "answer_marker_label": str(answer_value),
            "answer_marker_value": int(answer_marker_value),
            "marked_region_ids": [str(region_id) for region_id in marked_region_ids],
            "marked_region_count": int(len(marked_region_ids)),
            "marked_region_count_max": int(marked_region_count_max),
            "marker_value_min": int(value_min),
            "marker_value_max": int(value_max),
        },
        show_marker_labels=True,
    )


__all__ = [
    "construct_adjacent_category_dataset",
    "construct_adjacent_numeric_threshold_dataset",
    "construct_adjacent_same_category_dataset",
    "construct_categorical_count_dataset",
    "construct_group_category_count_dataset",
    "construct_marker_extremum_dataset",
    "construct_marker_threshold_dataset",
    "construct_named_region_set_total_dataset",
    "construct_numeric_interval_dataset",
    "construct_numeric_threshold_dataset",
]
