"""Pages-domain semantic visual asset overlay.

The base pages visual asset pool is decorative by default. This module owns the
small curated overlay that promotes selected pages-owned assets into stable
answer-bearing symbols for page tasks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from PIL import Image

from ....resources import resource_root
from .page_visual_assets import (
    PageVisualAsset,
    page_visual_asset_root,
    page_visual_asset_version,
    render_page_visual_asset_rgba,
    resolve_page_visual_asset,
)


_REPO_ROOT = resource_root()
_OVERLAY_PATH = page_visual_asset_root() / "semantic_overlay.jsonl"


@dataclass(frozen=True)
class PageSemanticAsset:
    """One stable semantic symbol backed by a pages visual asset."""

    semantic_id: str
    display_label: str
    asset_id: str
    semantic_role: str
    families: Tuple[str, ...]
    allowed_uses: Tuple[str, ...]
    semantic_policy: str
    asset: PageVisualAsset

    def to_metadata(self) -> dict[str, Any]:
        return {
            "semantic_id": str(self.semantic_id),
            "display_label": str(self.display_label),
            "asset_id": str(self.asset_id),
            "semantic_role": str(self.semantic_role),
            "families": [str(value) for value in self.families],
            "allowed_uses": [str(value) for value in self.allowed_uses],
            "semantic_policy": str(self.semantic_policy),
            "overlay_manifest_path": str(_OVERLAY_PATH.relative_to(_REPO_ROOT)),
            "asset_version": page_visual_asset_version(),
            "asset": self.asset.to_metadata(),
        }


def _coerce_row(row: Mapping[str, Any]) -> PageSemanticAsset:
    asset_id = str(row["asset_id"])
    asset = resolve_page_visual_asset(asset_id)
    semantic_id = str(row["semantic_id"]).strip()
    display_label = str(row["display_label"]).strip()
    semantic_role = str(row["semantic_role"]).strip()
    families = tuple(str(value).strip() for value in row.get("families", ()) if str(value).strip())
    allowed_uses = tuple(str(value).strip() for value in row.get("allowed_uses", ()) if str(value).strip())
    semantic_policy = str(row.get("semantic_policy", "answer_bearing_visual_symbol")).strip()
    if not semantic_id or not display_label or not semantic_role:
        raise ValueError(f"invalid page semantic asset row: {row!r}")
    if not allowed_uses:
        raise ValueError(f"page semantic asset has no allowed uses: {row!r}")
    return PageSemanticAsset(
        semantic_id=semantic_id,
        display_label=display_label,
        asset_id=asset_id,
        semantic_role=semantic_role,
        families=families,
        allowed_uses=allowed_uses,
        semantic_policy=semantic_policy,
        asset=asset,
    )


@lru_cache(maxsize=1)
def load_page_semantic_asset_manifest() -> Tuple[PageSemanticAsset, ...]:
    """Load the curated pages semantic visual overlay."""

    if not _OVERLAY_PATH.exists():
        raise FileNotFoundError(_OVERLAY_PATH)
    assets: list[PageSemanticAsset] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(_OVERLAY_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"page semantic overlay line {line_no} must be a mapping")
        asset = _coerce_row(payload)
        if asset.semantic_id in seen_ids:
            raise ValueError(f"duplicate page semantic asset id: {asset.semantic_id}")
        seen_ids.add(asset.semantic_id)
        assets.append(asset)
    if not assets:
        raise ValueError(f"page semantic overlay is empty: {_OVERLAY_PATH}")
    return tuple(assets)


@lru_cache(maxsize=1)
def _semantic_asset_by_id() -> Mapping[str, PageSemanticAsset]:
    return {asset.semantic_id: asset for asset in load_page_semantic_asset_manifest()}


def resolve_page_semantic_asset(semantic_id: str) -> PageSemanticAsset:
    """Resolve one semantic pages visual asset by stable id."""

    key = str(semantic_id).strip()
    if not key:
        raise ValueError("semantic_id must be non-empty")
    asset = _semantic_asset_by_id().get(key)
    if asset is None:
        raise KeyError(f"unknown page semantic asset: {semantic_id!r}")
    return asset


def filter_page_semantic_assets(
    *,
    semantic_role: str | None = None,
    families: Sequence[str] | None = None,
    allowed_uses: Sequence[str] | None = None,
) -> Tuple[PageSemanticAsset, ...]:
    """Return semantic assets matching optional role, family, and allowed-use filters."""

    role_key = str(semantic_role) if semantic_role is not None else None
    family_set = {str(value) for value in families} if families is not None else None
    use_set = {str(value) for value in allowed_uses} if allowed_uses is not None else None
    matches: list[PageSemanticAsset] = []
    for asset in load_page_semantic_asset_manifest():
        if role_key is not None and asset.semantic_role != role_key:
            continue
        if family_set is not None and not family_set.intersection(asset.families):
            continue
        if use_set is not None and not use_set.intersection(asset.allowed_uses):
            continue
        matches.append(asset)
    return tuple(matches)


def page_semantic_asset_ids(*, semantic_role: str, allowed_use: str | None = None) -> Tuple[str, ...]:
    """Return stable semantic ids for one pages symbol role."""

    allowed_uses = (str(allowed_use),) if allowed_use is not None else None
    return tuple(
        str(asset.semantic_id)
        for asset in filter_page_semantic_assets(semantic_role=str(semantic_role), allowed_uses=allowed_uses)
    )


def page_semantic_asset_label(semantic_id: str) -> str:
    """Return the prompt-facing label for one semantic symbol."""

    return resolve_page_semantic_asset(str(semantic_id)).display_label


def render_page_semantic_asset_rgba(
    semantic_id: str,
    *,
    size_px: int | tuple[int, int],
    tint_rgb: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Render a semantic pages visual asset as RGBA."""

    asset = resolve_page_semantic_asset(str(semantic_id))
    return render_page_visual_asset_rgba(asset.asset, size_px=size_px, tint_rgb=tint_rgb)


def page_semantic_asset_manifest_metadata(
    *,
    semantic_role: str | None = None,
    allowed_use: str | None = None,
) -> dict[str, Any]:
    """Return trace-safe metadata for a semantic asset subset."""

    assets = filter_page_semantic_assets(
        semantic_role=semantic_role,
        allowed_uses=(str(allowed_use),) if allowed_use is not None else None,
    )
    return {
        "asset_root": "assets/pages/visual_assets",
        "overlay_manifest_path": str(_OVERLAY_PATH.relative_to(_REPO_ROOT)),
        "asset_version": page_visual_asset_version(),
        "semantic_policy": "answer_bearing_visual_overlay",
        "semantic_role": str(semantic_role) if semantic_role is not None else None,
        "allowed_use": str(allowed_use) if allowed_use is not None else None,
        "semantic_ids": [str(asset.semantic_id) for asset in assets],
        "assets": {str(asset.semantic_id): asset.to_metadata() for asset in assets},
    }


__all__ = [
    "PageSemanticAsset",
    "filter_page_semantic_assets",
    "load_page_semantic_asset_manifest",
    "page_semantic_asset_ids",
    "page_semantic_asset_label",
    "page_semantic_asset_manifest_metadata",
    "render_page_semantic_asset_rgba",
    "resolve_page_semantic_asset",
]
