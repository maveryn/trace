"""Shared loader for vendored context/distractor text manifests."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Tuple

from ...resources import resource_root, safe_resource_join

REPO_ROOT = resource_root()
CONTEXT_TEXT_ROOT = REPO_ROOT / "assets" / "context_text"
SOURCES_PATH = CONTEXT_TEXT_ROOT / "sources.json"


@dataclass(frozen=True)
class ContextTextManifest:
    """One loaded context-text manifest plus its source metadata."""

    relative_path: str
    values: Tuple[str, ...]
    source_ids: Tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ContextTextSelection:
    """One deterministic context-text selection from a manifest."""

    text: str
    manifest_path: str
    row_index: int
    source_ids: Tuple[str, ...]

    def to_trace(self) -> dict[str, Any]:
        return {
            "text": str(self.text),
            "manifest_path": str(self.manifest_path),
            "row_index": int(self.row_index),
            "source_ids": [str(source_id) for source_id in self.source_ids],
        }


def _normalize_manifest_path(relative_path: str) -> str:
    normalized = str(relative_path).replace("\\", "/").strip().lstrip("/")
    if not normalized or normalized.startswith("../") or "/../" in normalized:
        raise ValueError(f"invalid context text manifest path: {relative_path!r}")
    return normalized


@lru_cache(maxsize=1)
def load_context_text_sources() -> Mapping[str, Any]:
    """Return the context-text source manifest payload."""

    payload = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"context text sources payload must be a mapping: {SOURCES_PATH}")
    return payload


@lru_cache(maxsize=64)
def load_context_text_manifest(relative_path: str) -> ContextTextManifest:
    """Load one context-text manifest by repo-relative path under assets/context_text."""

    manifest_path = _normalize_manifest_path(str(relative_path))
    sources_payload = load_context_text_sources()
    manifests = sources_payload.get("manifests", {})
    if not isinstance(manifests, Mapping) or manifest_path not in manifests:
        raise KeyError(f"unknown context text manifest: {manifest_path}")
    manifest_meta = manifests[manifest_path]
    if not isinstance(manifest_meta, Mapping):
        raise ValueError(f"context text manifest metadata must be a mapping: {manifest_path}")
    full_path = safe_resource_join(CONTEXT_TEXT_ROOT, manifest_path)
    values = tuple(
        line.strip()
        for line in full_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not values:
        raise ValueError(f"context text manifest is empty: {manifest_path}")
    source_ids = tuple(str(source_id) for source_id in manifest_meta.get("sources", ()))
    return ContextTextManifest(
        relative_path=str(manifest_path),
        values=values,
        source_ids=source_ids,
        description=str(manifest_meta.get("description", "")),
    )


def sample_context_text(
    relative_path: str,
    *,
    rng: random.Random,
) -> ContextTextSelection:
    """Sample one text string from a vendored context-text manifest."""

    manifest = load_context_text_manifest(str(relative_path))
    index = int(rng.randrange(len(manifest.values)))
    return ContextTextSelection(
        text=str(manifest.values[index]),
        manifest_path=str(manifest.relative_path),
        row_index=int(index),
        source_ids=tuple(manifest.source_ids),
    )


def context_text_asset_version() -> str:
    """Return the current context-text asset version string."""

    payload = load_context_text_sources()
    return str(payload.get("asset_version", ""))


__all__ = [
    "CONTEXT_TEXT_ROOT",
    "ContextTextManifest",
    "ContextTextSelection",
    "context_text_asset_version",
    "load_context_text_manifest",
    "load_context_text_sources",
    "sample_context_text",
]
