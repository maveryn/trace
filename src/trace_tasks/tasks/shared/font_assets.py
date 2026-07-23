"""Shared loader and deterministic sampler for vendored Trace font assets."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, Tuple

from ...resources import resource_root, safe_resource_join
from ...core.seed import spawn_rng


REPO_ROOT = resource_root()
FONT_ASSET_ROOT = REPO_ROOT / "assets" / "fonts"
SOURCES_PATH = FONT_ASSET_ROOT / "sources.json"
READOUT_POOL_PATH = FONT_ASSET_ROOT / "readout_pool_v0.json"
FontRole = Literal["readout", "context", "decorative"]


@dataclass(frozen=True)
class FontFamilyRecord:
    """One vendored font family and its regular/bold paths."""

    key: str
    family_name: str
    regular_path: str
    bold_path: str
    license: str
    license_path: str
    source_id: str
    source_url: str
    tags: Tuple[str, ...]

    def to_trace(self) -> dict[str, Any]:
        trace = {
            "font_family": str(self.key),
            "family_name": str(self.family_name),
            "font_asset_version": font_asset_version(),
            "license": str(self.license),
            "source_id": str(self.source_id),
            "source_url": str(self.source_url),
            "tags": [str(tag) for tag in self.tags],
        }
        memberships: list[dict[str, Any]] = []
        try:
            readout_families = set(list_font_families_for_role("readout"))
            if str(self.key) in readout_families:
                memberships.append(
                    {
                        "font_role": "readout",
                        "font_pool_id": font_pool_id_for_role("readout"),
                        "font_pool_size": font_pool_size_for_role("readout"),
                    }
                )
        except Exception:
            pass
        if memberships:
            trace["font_pool_memberships"] = memberships
        return trace


def _normalize_family_key(value: str) -> str:
    key = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key.strip("_")


@lru_cache(maxsize=1)
def load_font_sources() -> Mapping[str, Any]:
    """Return the font source metadata payload."""

    payload = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"font source metadata must be a mapping: {SOURCES_PATH}")
    return payload


@lru_cache(maxsize=1)
def load_readout_font_pool() -> Mapping[str, Any]:
    """Return the shared readout font-pool payload."""

    payload = json.loads(READOUT_POOL_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"readout font pool must be a mapping: {READOUT_POOL_PATH}")
    families = payload.get("font_families", ())
    if not isinstance(families, Sequence) or isinstance(families, (str, bytes)):
        raise ValueError(f"readout font pool must contain a font_families list: {READOUT_POOL_PATH}")
    keys = tuple(_normalize_family_key(str(key)) for key in families)
    target_count = int(payload.get("target_count", len(keys)))
    if len(keys) != target_count:
        raise ValueError(
            f"readout font pool target_count={target_count} but contains {len(keys)} families"
        )
    if len(set(keys)) != len(keys):
        raise ValueError(f"readout font pool contains duplicate family keys: {READOUT_POOL_PATH}")
    records = load_font_family_records()
    missing = [key for key in keys if key not in records]
    if missing:
        raise ValueError(f"readout font pool contains unknown family keys: {missing}")
    return payload


@lru_cache(maxsize=1)
def load_font_family_records() -> Mapping[str, FontFamilyRecord]:
    """Return all vendored font family records keyed by normalized family id."""

    payload = load_font_sources()
    families = payload.get("families", {})
    if not isinstance(families, Mapping):
        raise ValueError("font sources payload must contain a family mapping")
    records: dict[str, FontFamilyRecord] = {}
    for raw_key, raw_record in families.items():
        if not isinstance(raw_record, Mapping):
            continue
        key = _normalize_family_key(str(raw_key))
        regular_path = str(raw_record.get("regular_path", ""))
        bold_path = str(raw_record.get("bold_path", regular_path))
        if not regular_path:
            continue
        records[key] = FontFamilyRecord(
            key=str(key),
            family_name=str(raw_record.get("family_name", key)),
            regular_path=str(regular_path),
            bold_path=str(bold_path or regular_path),
            license=str(raw_record.get("license", "")),
            license_path=str(raw_record.get("license_path", "")),
            source_id=str(raw_record.get("source_id", "")),
            source_url=str(raw_record.get("source_url", "")),
            tags=tuple(str(tag) for tag in raw_record.get("tags", ())),
        )
    if not records:
        raise ValueError(f"no font family records found in {SOURCES_PATH}")
    return records


def list_font_families(
    *,
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
) -> Tuple[str, ...]:
    """Return font family keys filtered by optional tag constraints."""

    include = {str(tag) for tag in include_tags or ()}
    exclude = {str(tag) for tag in exclude_tags or ()}
    candidates: list[str] = []
    for key, record in load_font_family_records().items():
        tags = set(record.tags)
        if include and not bool(tags & include):
            continue
        if exclude and bool(tags & exclude):
            continue
        candidates.append(str(key))
    return tuple(sorted(candidates))


def _normalize_font_role(role: FontRole | str) -> FontRole:
    normalized = str(role).strip().casefold().replace("-", "_")
    if normalized not in {"readout", "context", "decorative"}:
        raise ValueError(f"unknown font role: {role!r}")
    return normalized  # type: ignore[return-value]


def list_font_families_for_role(role: FontRole | str) -> Tuple[str, ...]:
    """Return the candidate family keys for one semantic text role."""

    normalized = _normalize_font_role(role)
    if normalized == "readout":
        payload = load_readout_font_pool()
        return tuple(_normalize_family_key(str(key)) for key in payload.get("font_families", ()))
    return list_font_families()


def font_pool_id_for_role(role: FontRole | str) -> str:
    """Return the public pool id used for one role."""

    normalized = _normalize_font_role(role)
    if normalized == "readout":
        return str(load_readout_font_pool().get("pool_id", "readout_v0"))
    return f"{font_asset_version()}:full_vendored"


def font_pool_size_for_role(role: FontRole | str) -> int:
    """Return the number of candidate families available to one role."""

    return int(len(list_font_families_for_role(role)))


def font_role_trace(font_family: str, *, role: FontRole | str) -> dict[str, Any]:
    """Return compact trace metadata for a sampled family/role pair."""

    normalized = _normalize_font_role(role)
    return {
        "font_family": _normalize_family_key(str(font_family)),
        "font_role": str(normalized),
        "font_pool_id": font_pool_id_for_role(normalized),
        "font_pool_size": font_pool_size_for_role(normalized),
        "font_asset_version": font_asset_version(),
    }


def get_font_family_record(font_family: str) -> FontFamilyRecord:
    """Return one font family record by normalized key."""

    key = _normalize_family_key(str(font_family))
    records = load_font_family_records()
    if key not in records:
        raise KeyError(f"unknown font family: {font_family!r}")
    return records[key]


def resolve_font_paths(font_family: str, *, bold: bool) -> Tuple[Path, ...]:
    """Return preferred local font paths for one family/style."""

    record = get_font_family_record(str(font_family))
    primary = record.bold_path if bool(bold) else record.regular_path
    fallback = record.regular_path if bool(bold) else record.bold_path
    paths = []
    for relative_path in (primary, fallback):
        if not relative_path:
            continue
        full_path = safe_resource_join(FONT_ASSET_ROOT, str(relative_path))
        if full_path.exists() and full_path not in paths:
            paths.append(full_path)
    return tuple(paths)


def _weighted_choice(rng: random.Random, weighted: Mapping[str, float], *, fallback: Sequence[str]) -> str:
    candidates: list[tuple[str, float]] = []
    for key, value in weighted.items():
        try:
            weight = float(value)
        except Exception:
            continue
        if weight > 0:
            candidates.append((str(key), float(weight)))
    if not candidates:
        candidates = [(str(key), 1.0) for key in fallback]
    total = sum(weight for _, weight in candidates)
    cursor = rng.random() * float(total)
    running = 0.0
    for key, weight in candidates:
        running += float(weight)
        if cursor <= running:
            return str(key)
    return str(candidates[-1][0])


def sample_font_family(
    *,
    role: FontRole,
    instance_seed: int,
    namespace: str,
    params: Mapping[str, Any] | None = None,
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    explicit_key: str = "font_family",
    weights_key: str = "font_family_weights",
) -> str:
    """Sample one vendored font family deterministically from seed/namespace.

    The role chooses the candidate pool. Use ``readout`` for answer-bearing or
    read-required text, ``context`` for non-answer chrome/context text, and
    ``decorative`` only for non-semantic visual dressing.
    """

    resolved_params = params or {}
    normalized_role = _normalize_font_role(role)
    explicit = resolved_params.get(str(explicit_key))
    candidates = tuple(list_font_families_for_role(normalized_role))
    if include_tags or exclude_tags:
        role_candidates = set(candidates)
        candidates = tuple(
            key
            for key in list_font_families(include_tags=include_tags, exclude_tags=exclude_tags)
            if key in role_candidates
        )
    if not candidates:
        candidates = tuple(list_font_families_for_role(normalized_role))
    candidate_set = set(candidates)
    if explicit is not None:
        key = _normalize_family_key(str(explicit))
        if key not in candidate_set:
            raise ValueError(f"font family {explicit!r} is not available for this text role")
        return str(key)
    raw_weights = resolved_params.get(str(weights_key), {})
    weights: dict[str, float] = {}
    if isinstance(raw_weights, Mapping):
        for key, value in raw_weights.items():
            normalized = _normalize_family_key(str(key))
            if normalized in candidate_set:
                try:
                    weights[str(normalized)] = float(value)
                except Exception:
                    continue
    rng = spawn_rng(int(instance_seed), str(namespace))
    return _weighted_choice(rng, weights, fallback=tuple(candidates))


def font_asset_version() -> str:
    """Return the current font asset version string."""

    payload = load_font_sources()
    return str(payload.get("asset_version", ""))


__all__ = [
    "FONT_ASSET_ROOT",
    "FontRole",
    "FontFamilyRecord",
    "font_asset_version",
    "font_pool_id_for_role",
    "font_pool_size_for_role",
    "font_role_trace",
    "get_font_family_record",
    "list_font_families",
    "list_font_families_for_role",
    "load_font_family_records",
    "load_font_sources",
    "load_readout_font_pool",
    "READOUT_POOL_PATH",
    "resolve_font_paths",
    "sample_font_family",
]
