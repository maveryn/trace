"""Public reusable-object inventories and materialized preview assets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from .index import is_path_within
from ..provenance import sha256_file

ASSET_PREVIEW_SCHEMA = "trace-review-asset-preview-v1"
_PREVIEW_MANIFEST_FIELDS = frozenset({"schema", "kind", "asset_id", "image"})
_PREVIEW_IMAGE_FIELDS = frozenset(
    {"path", "format", "mode", "width", "height", "sha256"}
)


@dataclass(frozen=True)
class AssetRecord:
    id: str
    kind: str
    name: str
    family: str
    source: str
    metadata: dict[str, Any]
    media_id: str = ""
    rel_path: str = ""


@dataclass
class AssetIndex:
    records: dict[str, list[AssetRecord]]
    media: dict[str, Path]


def build_asset_index(review_root: Path | str) -> AssetIndex:
    """Build illustration and 3D catalogs plus optional materialized previews."""

    root = Path(review_root).expanduser().resolve()
    media: dict[str, Path] = {}
    preview_by_key: dict[tuple[str, str], tuple[str, str]] = {}
    duplicate_keys: set[tuple[str, str]] = set()
    assets_root = root / "assets"
    if (
        assets_root.is_dir()
        and not assets_root.is_symlink()
        and is_path_within(root, assets_root)
    ):
        for manifest_path in sorted(assets_root.rglob("*.json")):
            preview = _load_preview_manifest(root, assets_root, manifest_path)
            if preview is None:
                continue
            kind, asset_id, image_path = preview
            key = (kind, asset_id)
            if key in duplicate_keys:
                continue
            if key in preview_by_key:
                previous_media_id = preview_by_key.pop(key)[0]
                media.pop(previous_media_id, None)
                duplicate_keys.add(key)
                continue
            rel = image_path.relative_to(root).as_posix()
            media_id = _stable_id("asset-media", f"{rel}\0{sha256_file(image_path)}")
            media[media_id] = image_path.resolve()
            preview_by_key[key] = (media_id, rel)

    records = {
        "illustrations": _illustration_records(preview_by_key),
        "three_d": _three_d_records(preview_by_key),
    }
    referenced_media = {
        record.media_id
        for kind_records in records.values()
        for record in kind_records
        if record.media_id
    }
    media = {key: value for key, value in media.items() if key in referenced_media}
    return AssetIndex(records=records, media=media)


def _load_preview_manifest(
    root: Path, assets_root: Path, manifest_path: Path
) -> tuple[str, str, Path] | None:
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not is_path_within(root, manifest_path)
        or _has_symlink_component(root, manifest_path)
    ):
        return None
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or set(raw) != _PREVIEW_MANIFEST_FIELDS
            or raw.get("schema") != ASSET_PREVIEW_SCHEMA
        ):
            return None
        kind = raw.get("kind")
        asset_id = raw.get("asset_id")
        image = raw.get("image")
        manifest_relative = manifest_path.relative_to(assets_root)
        if (
            kind not in {"illustrations", "three_d"}
            or len(manifest_relative.parts) != 2
            or manifest_relative.parts[0] != kind
            or not isinstance(asset_id, str)
            or not asset_id
            or asset_id != asset_id.strip()
        ):
            return None
        if (
            not isinstance(image, dict)
            or set(image) != _PREVIEW_IMAGE_FIELDS
            or image.get("format") != "png"
        ):
            return None
        raw_path = image.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return None
        if (
            not isinstance(image.get("mode"), str)
            or not image.get("mode")
            or any(
                isinstance(image.get(field), bool)
                or not isinstance(image.get(field), int)
                or image.get(field) <= 0
                for field in ("width", "height")
            )
        ):
            return None
        relative = PurePosixPath(raw_path)
        if (
            relative.is_absolute()
            or "\\" in raw_path
            or relative.as_posix() != raw_path
            or len(relative.parts) != 3
            or relative.parts[:2] != ("assets", kind)
            or relative.suffix != ".png"
            or relative.stem != manifest_path.stem
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            return None
        image_path_input = root.joinpath(*relative.parts)
        image_path = image_path_input.resolve(strict=True)
        expected_parent = (assets_root / kind).resolve()
        if (
            not image_path.is_file()
            or not is_path_within(expected_parent, image_path)
            or _has_symlink_component(root, image_path_input)
        ):
            return None
        expected_hash = image.get("sha256")
        if (
            not isinstance(expected_hash, str)
            or sha256_file(image_path) != expected_hash
        ):
            return None
        with Image.open(image_path) as decoded:
            if decoded.format != "PNG":
                return None
            decoded.load()
            actual = {
                "mode": str(decoded.mode),
                "width": int(decoded.width),
                "height": int(decoded.height),
            }
        if any(image.get(field) != value for field, value in actual.items()):
            return None
        return kind, asset_id.strip(), image_path
    except (OSError, ValueError, json.JSONDecodeError, Image.DecompressionBombError):
        return None


def _has_symlink_component(root: Path, path: Path) -> bool:
    try:
        relative = path.absolute().relative_to(root.absolute())
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _illustration_records(
    preview_by_key: dict[tuple[str, str], tuple[str, str]],
) -> list[AssetRecord]:
    try:
        from trace_tasks.tasks.illustrations.shared.object_catalog import (
            CATALOG_ENTRIES,
        )
    except Exception:
        return []
    records: list[AssetRecord] = []
    for entry in CATALOG_ENTRIES:
        raw = asdict(entry)
        item_id = str(entry.catalog_id)
        media_id, rel_path = preview_by_key.get(("illustrations", item_id), ("", ""))
        records.append(
            AssetRecord(
                id=item_id,
                kind="illustrations",
                name=str(entry.public_name),
                family=str(entry.family),
                source="trace_tasks.tasks.illustrations.shared.object_catalog",
                metadata=raw,
                media_id=media_id,
                rel_path=rel_path,
            )
        )
    return sorted(records, key=lambda item: (item.family, item.name, item.id))


def _three_d_records(
    preview_by_key: dict[tuple[str, str], tuple[str, str]],
) -> list[AssetRecord]:
    try:
        from trace_tasks.tasks.three_d.shared.object_resources import (
            THREE_D_OBJECT_PROFILES,
        )
    except Exception:
        return []
    records: list[AssetRecord] = []
    for profile in THREE_D_OBJECT_PROFILES:
        raw = asdict(profile)
        item_id = str(profile.profile_id)
        media_id, rel_path = preview_by_key.get(("three_d", item_id), ("", ""))
        records.append(
            AssetRecord(
                id=item_id,
                kind="three_d",
                name=str(profile.display_name),
                family=str(profile.source_scene),
                source="trace_tasks.tasks.three_d.shared.object_resources",
                metadata=raw,
                media_id=media_id,
                rel_path=rel_path,
            )
        )
    return sorted(records, key=lambda item: (item.family, item.name, item.id))


def _normalize_kind(value: str) -> str | None:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"illustration", "illustrations"}:
        return "illustrations"
    if normalized in {"3d", "three_d", "three-d"}:
        return "three_d"
    return None


def _stable_id(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode("utf-8")).hexdigest()[:24]


__all__ = ["AssetIndex", "AssetRecord", "build_asset_index"]
