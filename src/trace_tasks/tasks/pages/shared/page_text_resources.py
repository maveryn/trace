"""Shared pages-domain text resource helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, Tuple

from ...shared.context_text_assets import (
    context_text_asset_version,
    load_context_text_manifest,
)
from ...shared.name_assets import load_label_manifest, load_label_sources


@dataclass(frozen=True)
class PageTextSelection:
    """One visible page text selection from a shared manifest."""

    text: str
    role: str
    source_kind: str
    manifest_path: str
    sampled_index: int
    candidate_count: int
    asset_version: str
    source_ids: Tuple[str, ...]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "text": str(self.text),
            "role": str(self.role),
            "source_kind": str(self.source_kind),
            "manifest_path": str(self.manifest_path),
            "sampled_index": int(self.sampled_index),
            "candidate_count": int(self.candidate_count),
            "asset_version": str(self.asset_version),
            "source_ids": [str(source_id) for source_id in self.source_ids],
        }


@dataclass(frozen=True)
class PageTextBatch:
    """A deterministic batch of page text selections."""

    role: str
    selections: Tuple[PageTextSelection, ...]

    @property
    def values(self) -> Tuple[str, ...]:
        return tuple(str(selection.text) for selection in self.selections)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "role": str(self.role),
            "count": len(self.selections),
            "values": [str(value) for value in self.values],
            "selections": [selection.to_metadata() for selection in self.selections],
        }


def _unique_preserve_order(values: Iterable[str]) -> Tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for raw_value in values:
        value = str(raw_value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _label_manifest_source_ids(manifest_name: str) -> Tuple[str, ...]:
    sources = load_label_sources()
    manifests = sources.get("manifests", {}) if isinstance(sources, dict) else {}
    meta = manifests.get(str(manifest_name), {}) if isinstance(manifests, dict) else {}
    return tuple(str(source_id) for source_id in meta.get("sources", ())) if isinstance(meta, dict) else tuple()


def _label_asset_version() -> str:
    sources = load_label_sources()
    return str(sources.get("asset_version", "")) if isinstance(sources, dict) else ""


def sample_page_label_batch(
    rng: random.Random,
    *,
    role: str,
    count: int,
    manifest_name: str = "mixed/compact_labels.txt",
    min_chars: int | None = None,
    max_chars: int | None = None,
    allow_spaces: bool = True,
    allow_punctuation: bool = True,
    ascii_only: bool = True,
    compact_length: bool = False,
    exclude: Sequence[str] = (),
) -> PageTextBatch:
    """Sample unique answer-bearing page labels from shared label manifests."""

    values = load_label_manifest(
        str(manifest_name),
        min_chars=min_chars,
        max_chars=max_chars,
        allow_spaces=allow_spaces,
        allow_punctuation=allow_punctuation,
        ascii_only=ascii_only,
        compact_length=compact_length,
    )
    excluded = {str(value) for value in exclude}
    candidates = [str(value) for value in _unique_preserve_order(values) if str(value) not in excluded]
    if int(count) > len(candidates):
        raise ValueError(
            f"not enough shared page labels for role={role!r}, manifest={manifest_name!r}, "
            f"requested={count}, available={len(candidates)}"
        )
    indices = list(range(len(candidates)))
    rng.shuffle(indices)
    selected_indices = indices[: int(count)]
    source_ids = _label_manifest_source_ids(str(manifest_name))
    selections = tuple(
        PageTextSelection(
            text=str(candidates[index]),
            role=str(role),
            source_kind="label",
            manifest_path=f"assets/labels/{manifest_name}",
            sampled_index=int(index),
            candidate_count=len(candidates),
            asset_version=_label_asset_version(),
            source_ids=source_ids,
        )
        for index in selected_indices
    )
    return PageTextBatch(role=str(role), selections=selections)


def sample_page_context_batch(
    rng: random.Random,
    *,
    role: str,
    count: int,
    manifest_names: Sequence[str],
    min_chars: int | None = None,
    max_chars: int | None = None,
) -> PageTextBatch:
    """Sample non-answer page context text from shared context-text manifests."""

    manifests = tuple(str(value) for value in manifest_names if str(value).strip())
    if not manifests:
        raise ValueError("manifest_names must not be empty")
    manifest_payloads = tuple(load_context_text_manifest(manifest_name) for manifest_name in manifests)
    candidates: list[tuple[str, int]] = []
    for manifest in manifest_payloads:
        for index, text in enumerate(manifest.values):
            if min_chars is not None and len(str(text)) < int(min_chars):
                continue
            if max_chars is not None and len(str(text)) > int(max_chars):
                continue
            candidates.append((str(manifest.relative_path), int(index)))
    if int(count) > len({load_context_text_manifest(path).values[index] for path, index in candidates}):
        raise ValueError(
            f"not enough shared page context text for role={role!r}, requested={count}, "
            f"available={len(candidates)}, min_chars={min_chars}, max_chars={max_chars}"
        )
    selections: list[PageTextSelection] = []
    used_text: set[str] = set()
    attempts = 0
    while len(selections) < int(count) and attempts < int(count) * max(32, len(candidates) * 2):
        attempts += 1
        manifest_name, row_index = candidates[int(rng.randrange(len(candidates)))]
        manifest = load_context_text_manifest(manifest_name)
        text = str(manifest.values[int(row_index)])
        if text in used_text:
            continue
        used_text.add(str(text))
        selections.append(
            PageTextSelection(
                text=str(text),
                role=str(role),
                source_kind="context_text",
                manifest_path=f"assets/context_text/{manifest.relative_path}",
                sampled_index=int(row_index),
                candidate_count=len(candidates),
                asset_version=context_text_asset_version(),
                source_ids=tuple(str(value) for value in manifest.source_ids),
            )
        )
    if len(selections) < int(count):
        raise ValueError(f"could not sample {count} unique context text values for role={role!r}")
    return PageTextBatch(role=str(role), selections=tuple(selections))


def page_text_resource_metadata(*batches: PageTextBatch) -> dict[str, Any]:
    """Return trace-safe metadata for one or more page text batches."""

    return {
        "policy": "shared_pages_text_resources",
        "batches": {str(batch.role): batch.to_metadata() for batch in batches},
    }


__all__ = [
    "PageTextBatch",
    "PageTextSelection",
    "page_text_resource_metadata",
    "sample_page_context_batch",
    "sample_page_label_batch",
]
