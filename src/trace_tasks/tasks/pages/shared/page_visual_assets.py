"""Shared pages-domain visual asset loader.

The assets in ``assets/pages/visual_assets`` are decorative, non-answer
resources for page scenes. They are intentionally separate from the icon-domain
pool so page renderers can use illustration anchors, section art, and badges
without depending on icon task semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from PIL import Image, ImageOps

from ....resources import resource_root, safe_resource_join

_REPO_ROOT = resource_root()
_ASSET_ROOT = _REPO_ROOT / "assets" / "pages" / "visual_assets"
_MANIFEST_PATH = _ASSET_ROOT / "manifest.jsonl"
_SOURCES_PATH = _ASSET_ROOT / "sources.json"
_SUPPORTED_ROLES = ("hero_anchor", "section_illustration", "badge_spot")


@dataclass(frozen=True)
class PageVisualAsset:
    """One vendored page visual asset manifest row."""

    asset_id: str
    source_id: str
    source_label: str
    source_url: str
    source_revision: str
    license_spdx: str
    local_license: str
    raw_path: str
    normalized_path: str
    categories: Tuple[str, ...]
    allowed_roles: Tuple[str, ...]
    style: str
    render_mode: str
    semantic_policy: str
    width_px: int
    height_px: int
    alpha_bbox_px: Tuple[int, int, int, int]
    alpha_coverage: float
    recommended_min_px: int
    recommended_max_px: int
    checksum_sha256: str
    normalized_checksum_sha256: str

    def to_metadata(self) -> dict[str, Any]:
        """Return trace-safe provenance metadata for this asset."""

        return {
            "asset_id": str(self.asset_id),
            "source_id": str(self.source_id),
            "source_label": str(self.source_label),
            "source_url": str(self.source_url),
            "source_revision": str(self.source_revision),
            "license_spdx": str(self.license_spdx),
            "local_license": str(self.local_license),
            "raw_path": str(self.raw_path),
            "normalized_path": str(self.normalized_path),
            "categories": [str(category) for category in self.categories],
            "allowed_roles": [str(role) for role in self.allowed_roles],
            "style": str(self.style),
            "render_mode": str(self.render_mode),
            "semantic_policy": str(self.semantic_policy),
            "width_px": int(self.width_px),
            "height_px": int(self.height_px),
            "alpha_bbox_px": [int(value) for value in self.alpha_bbox_px],
            "alpha_coverage": float(self.alpha_coverage),
            "recommended_min_px": int(self.recommended_min_px),
            "recommended_max_px": int(self.recommended_max_px),
            "checksum_sha256": str(self.checksum_sha256),
            "normalized_checksum_sha256": str(self.normalized_checksum_sha256),
        }


@dataclass(frozen=True)
class PageVisualAssetSelection:
    """One deterministic visual asset selection for a page-render role."""

    asset: PageVisualAsset
    role: str
    sampled_index: int
    candidate_count: int
    manifest_path: str

    def to_metadata(self) -> dict[str, Any]:
        """Return trace-safe metadata for one sampled asset selection."""

        return {
            "role": str(self.role),
            "sampled_index": int(self.sampled_index),
            "candidate_count": int(self.candidate_count),
            "manifest_path": str(self.manifest_path),
            "asset": self.asset.to_metadata(),
        }


def page_visual_asset_root() -> Path:
    """Return the Trace-side page visual asset root."""

    return _ASSET_ROOT


def available_page_visual_asset_roles() -> Tuple[str, ...]:
    """Return supported page visual asset placement roles."""

    return _SUPPORTED_ROLES


def _relative_asset_path(path_value: str) -> Path:
    relative = Path(str(path_value).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid page visual asset path: {path_value!r}")
    return relative


def _coerce_asset_row(row: Mapping[str, Any]) -> PageVisualAsset:
    alpha_bbox = tuple(int(value) for value in row.get("alpha_bbox_px", ()))
    if len(alpha_bbox) != 4:
        raise ValueError(f"page visual asset alpha bbox must have four values: {row!r}")
    categories = tuple(str(category) for category in row.get("categories", ()))
    allowed_roles = tuple(str(role) for role in row.get("allowed_roles", ()))
    if not allowed_roles:
        raise ValueError(f"page visual asset has no allowed roles: {row!r}")
    unsupported_roles = sorted(set(allowed_roles).difference(_SUPPORTED_ROLES))
    if unsupported_roles:
        raise ValueError(f"unsupported page visual asset roles: {unsupported_roles}")
    raw_path = str(_relative_asset_path(str(row["raw_path"])))
    normalized_path = str(_relative_asset_path(str(row["normalized_path"])))
    local_license = str(_relative_asset_path(str(row["local_license"])))
    return PageVisualAsset(
        asset_id=str(row["asset_id"]),
        source_id=str(row["source_id"]),
        source_label=str(row.get("source_label", "")),
        source_url=str(row.get("source_url", "")),
        source_revision=str(row.get("source_revision", "")),
        license_spdx=str(row["license_spdx"]),
        local_license=local_license,
        raw_path=raw_path,
        normalized_path=normalized_path,
        categories=categories,
        allowed_roles=allowed_roles,
        style=str(row.get("style", "")),
        render_mode=str(row.get("render_mode", "")),
        semantic_policy=str(row.get("semantic_policy", "")),
        width_px=int(row["width_px"]),
        height_px=int(row["height_px"]),
        alpha_bbox_px=alpha_bbox,
        alpha_coverage=float(row["alpha_coverage"]),
        recommended_min_px=int(row.get("recommended_min_px", 1)),
        recommended_max_px=int(row.get("recommended_max_px", max(row["width_px"], row["height_px"]))),
        checksum_sha256=str(row.get("checksum_sha256", "")),
        normalized_checksum_sha256=str(row.get("normalized_checksum_sha256", "")),
    )


@lru_cache(maxsize=1)
def load_page_visual_asset_manifest() -> Tuple[PageVisualAsset, ...]:
    """Load all curated page visual asset manifest rows."""

    if not _MANIFEST_PATH.exists():
        raise FileNotFoundError(_MANIFEST_PATH)
    assets: list[PageVisualAsset] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(_MANIFEST_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"page visual asset manifest line {line_no} must be a mapping")
        asset = _coerce_asset_row(payload)
        if asset.asset_id in seen_ids:
            raise ValueError(f"duplicate page visual asset id: {asset.asset_id}")
        seen_ids.add(asset.asset_id)
        assets.append(asset)
    if not assets:
        raise ValueError(f"page visual asset manifest is empty: {_MANIFEST_PATH}")
    return tuple(assets)


@lru_cache(maxsize=1)
def _page_visual_asset_by_id() -> Mapping[str, PageVisualAsset]:
    return {asset.asset_id: asset for asset in load_page_visual_asset_manifest()}


@lru_cache(maxsize=1)
def load_page_visual_asset_sources() -> Mapping[str, Any]:
    """Return source and build metadata for the page visual asset pool."""

    payload = json.loads(_SOURCES_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"page visual asset source payload must be a mapping: {_SOURCES_PATH}")
    return payload


def resolve_page_visual_asset(asset_id: str) -> PageVisualAsset:
    """Resolve one page visual asset by id."""

    asset_key = str(asset_id).strip()
    if not asset_key:
        raise ValueError("asset_id must be non-empty")
    asset = _page_visual_asset_by_id().get(asset_key)
    if asset is None:
        raise KeyError(f"unknown page visual asset: {asset_id!r}")
    return asset


def page_visual_asset_path(asset: str | PageVisualAsset) -> Path:
    """Return the normalized PNG path for one page visual asset."""

    resolved = resolve_page_visual_asset(asset) if isinstance(asset, str) else asset
    return safe_resource_join(_ASSET_ROOT, _relative_asset_path(resolved.normalized_path))


def filter_page_visual_assets(
    *,
    role: str | None = None,
    categories: Sequence[str] | None = None,
    styles: Sequence[str] | None = None,
    render_modes: Sequence[str] | None = None,
) -> Tuple[PageVisualAsset, ...]:
    """Return assets matching optional role, category, style, and render-mode filters."""

    category_set = {str(category) for category in categories} if categories is not None else None
    style_set = {str(style) for style in styles} if styles is not None else None
    mode_set = {str(mode) for mode in render_modes} if render_modes is not None else None
    role_key = str(role) if role is not None else None
    if role_key is not None and role_key not in _SUPPORTED_ROLES:
        raise ValueError(f"unsupported page visual asset role: {role!r}")

    matches: list[PageVisualAsset] = []
    for asset in load_page_visual_asset_manifest():
        if role_key is not None and role_key not in asset.allowed_roles:
            continue
        if category_set is not None and not category_set.intersection(asset.categories):
            continue
        if style_set is not None and asset.style not in style_set:
            continue
        if mode_set is not None and asset.render_mode not in mode_set:
            continue
        matches.append(asset)
    return tuple(matches)


def sample_page_visual_asset(
    rng: Any,
    *,
    role: str,
    categories: Sequence[str] | None = None,
    styles: Sequence[str] | None = None,
    render_modes: Sequence[str] | None = None,
) -> PageVisualAssetSelection:
    """Sample one asset for a placement role using an explicit RNG."""

    candidates = filter_page_visual_assets(
        role=str(role),
        categories=categories,
        styles=styles,
        render_modes=render_modes,
    )
    if not candidates:
        raise ValueError(
            "no page visual assets match "
            f"role={role!r}, categories={categories!r}, styles={styles!r}, render_modes={render_modes!r}"
        )
    sampled_index = int(rng.randrange(len(candidates)))
    return PageVisualAssetSelection(
        asset=candidates[sampled_index],
        role=str(role),
        sampled_index=sampled_index,
        candidate_count=len(candidates),
        manifest_path=str(_MANIFEST_PATH.relative_to(_REPO_ROOT)),
    )


@lru_cache(maxsize=1024)
def _load_normalized_asset_rgba(asset_id: str) -> Image.Image:
    path = page_visual_asset_path(str(asset_id))
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return image.convert("RGBA").copy()


def _apply_monochrome_tint(image: Image.Image, tint_rgb: tuple[int, int, int]) -> Image.Image:
    alpha = image.getchannel("A")
    tinted = Image.new("RGBA", image.size, tuple(int(value) for value in tint_rgb) + (0,))
    tinted.putalpha(alpha)
    return tinted


def render_page_visual_asset_rgba(
    asset: str | PageVisualAsset,
    *,
    size_px: int | tuple[int, int],
    tint_rgb: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Render one normalized asset as an RGBA image contained within ``size_px``.

    Color assets preserve their source colors. Monochrome assets may be tinted
    by passing ``tint_rgb``; otherwise they keep the normalized source color.
    """

    resolved = resolve_page_visual_asset(asset) if isinstance(asset, str) else asset
    if isinstance(size_px, tuple):
        target_size = (max(1, int(size_px[0])), max(1, int(size_px[1])))
    else:
        size = max(1, int(size_px))
        target_size = (size, size)
    image = _load_normalized_asset_rgba(resolved.asset_id).copy()
    if tint_rgb is not None and resolved.render_mode == "monochrome":
        image = _apply_monochrome_tint(image, tint_rgb)
    return ImageOps.contain(image, target_size, method=Image.Resampling.LANCZOS)


def page_visual_asset_version() -> str:
    """Return the current pages visual asset build version string."""

    sources = load_page_visual_asset_sources()
    return str(sources.get("asset_version", ""))


__all__ = [
    "PageVisualAsset",
    "PageVisualAssetSelection",
    "available_page_visual_asset_roles",
    "filter_page_visual_assets",
    "load_page_visual_asset_manifest",
    "load_page_visual_asset_sources",
    "page_visual_asset_path",
    "page_visual_asset_root",
    "page_visual_asset_version",
    "render_page_visual_asset_rgba",
    "resolve_page_visual_asset",
    "sample_page_visual_asset",
]
