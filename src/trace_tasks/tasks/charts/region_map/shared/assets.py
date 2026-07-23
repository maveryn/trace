"""Bundled geographic map assets for region-map tasks."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from .....resources import resource_path

SUPPORTED_GEOGRAPHIC_MAP_VARIANTS: Tuple[str, ...] = (
    "world_countries",
    "eu_countries",
    "usa_states",
    "china_provinces",
)
WORLD_TITLE_OPTIONS: Tuple[str, ...] = (
    "World Indicator Map",
    "Global Country Map",
    "World Regional Value Map",
    "Global Metric Map",
    "Country Value Overview",
)
WORLD_CATEGORY_TITLE_OPTIONS: Tuple[str, ...] = (
    "World Category Map",
    "Regional Category Map",
    "Country Category Overview",
    "Area Category Map",
    "Regional Class Map",
)
def _map_asset_root() -> Path:
    return resource_path("assets", "charts", "maps")


MAP_ASSET_ROOT = _map_asset_root()
WORLD_MAP_ASSET_ID = "natural_earth_admin0_world_110m_v0"
GEOGRAPHIC_MAP_ASSETS: Dict[str, Dict[str, str]] = {
    "world_countries": {
        "asset_id": WORLD_MAP_ASSET_ID,
        "path": "natural_earth_admin0_world_110m_v0.json",
    },
    "eu_countries": {
        "asset_id": "natural_earth_admin0_eu_110m_v0",
        "path": "natural_earth_admin0_eu_110m_v0.json",
    },
    "usa_states": {
        "asset_id": "natural_earth_admin1_usa_contiguous_110m_v0",
        "path": "natural_earth_admin1_usa_contiguous_110m_v0.json",
    },
    "china_provinces": {
        "asset_id": "natural_earth_admin1_china_50m_v0",
        "path": "natural_earth_admin1_china_50m_v0.json",
    },
}
GEOGRAPHIC_MAP_VARIANT_BY_ASSET_ID: Dict[str, str] = {
    str(spec["asset_id"]): str(variant)
    for variant, spec in GEOGRAPHIC_MAP_ASSETS.items()
}

def normalize_geographic_map_variant(value: Any) -> str:
    text = str(value or "").strip()
    if text in GEOGRAPHIC_MAP_ASSETS:
        return text
    if text in GEOGRAPHIC_MAP_VARIANT_BY_ASSET_ID:
        return GEOGRAPHIC_MAP_VARIANT_BY_ASSET_ID[text]
    if not text:
        return "world_countries"
    raise ValueError(f"unsupported geographic_map_variant: {text}")


@lru_cache(maxsize=None)
def load_geographic_map_asset(map_variant_or_asset_id: str = "world_countries") -> Dict[str, Any]:
    variant = normalize_geographic_map_variant(map_variant_or_asset_id)
    asset_path = MAP_ASSET_ROOT / str(GEOGRAPHIC_MAP_ASSETS[variant]["path"])
    if not asset_path.exists():
        raise FileNotFoundError(f"missing bundled geographic map asset: {asset_path}")
    asset = json.loads(asset_path.read_text(encoding="utf-8"))
    if not isinstance(asset, Mapping):
        raise ValueError("geographic map asset must be a JSON object")
    regions = asset.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("geographic map asset must contain non-empty regions")
    normalized = dict(asset)
    normalized.setdefault("asset_id", str(GEOGRAPHIC_MAP_ASSETS[variant]["asset_id"]))
    normalized.setdefault("map_variant", str(variant))
    normalized.setdefault("display_name", "World" if variant == "world_countries" else str(variant).replace("_", " ").title())
    normalized.setdefault("region_noun", "countries" if variant in {"world_countries", "eu_countries"} else "regions")
    normalized.setdefault("region_prefix", "country" if variant == "world_countries" else "geo_region")
    normalized.setdefault(
        "title_options",
        list(WORLD_TITLE_OPTIONS) if variant == "world_countries" else [str(normalized["display_name"]) + " Value Map"],
    )
    normalized.setdefault(
        "object_description",
        "a world map with selected countries colored by value and a color legend"
        if variant == "world_countries"
        else "a geographic map with selected regions colored by value and a color legend",
    )
    return normalized


def load_world_map_asset() -> Dict[str, Any]:
    return load_geographic_map_asset("world_countries")



__all__ = [
    "GEOGRAPHIC_MAP_ASSETS",
    "GEOGRAPHIC_MAP_VARIANT_BY_ASSET_ID",
    "MAP_ASSET_ROOT",
    "SUPPORTED_GEOGRAPHIC_MAP_VARIANTS",
    "WORLD_CATEGORY_TITLE_OPTIONS",
    "WORLD_MAP_ASSET_ID",
    "WORLD_TITLE_OPTIONS",
    "load_geographic_map_asset",
    "load_world_map_asset",
    "normalize_geographic_map_variant",
]
