"""Shared label/name asset helpers reused across domains."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from ...resources import resource_path, safe_resource_join

_ASSET_ROOT = resource_path("assets")
_DEFAULT_SHORT_NAME_MANIFEST = "series_legend_names_random_name_2to4.txt"
_DEFAULT_LABEL_MANIFEST = "mixed/compact_labels.txt"


def asset_root() -> Path:
    """Return the repo-local shared asset root."""

    return _ASSET_ROOT


@lru_cache(maxsize=32)
def load_name_manifest(*, asset_group: str, manifest_name: str) -> Tuple[str, ...]:
    """Load one vendored name manifest as a deterministic tuple."""

    path = safe_resource_join(asset_root(), str(asset_group), str(manifest_name))
    if not path.exists():
        raise FileNotFoundError(path)
    names = tuple(
        str(line).strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if str(line).strip()
    )
    if not names:
        raise ValueError(f"name manifest {path} resolved no names")
    return names


def load_short_name_manifest(manifest_name: str = _DEFAULT_SHORT_NAME_MANIFEST) -> Tuple[str, ...]:
    """Load the canonical short-name manifest used for visible person-style labels."""

    return load_name_manifest(asset_group="charts", manifest_name=str(manifest_name))


@lru_cache(maxsize=8)
def load_label_sources() -> Mapping[str, Any]:
    """Load metadata for the shared repo-wide label/name manifests."""

    path = asset_root() / "labels" / "sources.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"label sources metadata {path} must be a JSON object")
    return payload


def filter_label_values(
    labels: Sequence[str],
    *,
    min_chars: int | None = None,
    max_chars: int | None = None,
    allow_spaces: bool = True,
    allow_punctuation: bool = True,
    ascii_only: bool = True,
    compact_length: bool = False,
) -> Tuple[str, ...]:
    """Filter labels for task-local render constraints.

    Character bounds apply to the visible string after stripping outer
    whitespace by default. Set `compact_length=True` when a task wants to
    ignore whitespace for support constraints but still allow labels with spaces.
    """

    filtered: list[str] = []
    for raw_label in labels:
        label = str(raw_label).strip()
        if not label:
            continue
        if ascii_only and not label.isascii():
            continue
        if not allow_spaces and any(ch.isspace() for ch in label):
            continue
        if not allow_punctuation and not all(ch.isalnum() or ch.isspace() for ch in label):
            continue
        length_value = "".join(ch for ch in label if not ch.isspace()) if compact_length else label
        if min_chars is not None and len(length_value) < int(min_chars):
            continue
        if max_chars is not None and len(length_value) > int(max_chars):
            continue
        filtered.append(label)
    return tuple(filtered)


@lru_cache(maxsize=128)
def load_label_manifest(
    manifest_name: str = _DEFAULT_LABEL_MANIFEST,
    *,
    min_chars: int | None = None,
    max_chars: int | None = None,
    allow_spaces: bool = True,
    allow_punctuation: bool = True,
    ascii_only: bool = True,
    compact_length: bool = False,
) -> Tuple[str, ...]:
    """Load one shared label manifest from `assets/labels/`.

    `manifest_name` is a path relative to `assets/labels`, for example
    `people/first_names_ssa.txt` or `places/cities_natural_earth.txt`.
    Optional filters let tasks constrain the pool without creating new
    repo-wide files for every render-size requirement.
    """

    labels = load_name_manifest(asset_group="labels", manifest_name=str(manifest_name))
    filtered = filter_label_values(
        labels,
        min_chars=min_chars,
        max_chars=max_chars,
        allow_spaces=allow_spaces,
        allow_punctuation=allow_punctuation,
        ascii_only=ascii_only,
        compact_length=compact_length,
    )
    if not filtered:
        raise ValueError(f"label manifest {manifest_name!r} resolved no labels after filtering")
    return filtered


__all__ = [
    "asset_root",
    "filter_label_values",
    "load_label_manifest",
    "load_label_sources",
    "load_name_manifest",
    "load_short_name_manifest",
]
