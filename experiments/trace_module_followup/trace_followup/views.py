"""Inference views for the seven cells under P1 (B's interface files) and P2 (V's interface files).

Same contract as ``trace_merge.views``: weight shards and index are symlinks to the weight source (a pinned parent
snapshot or a finished hybrid), the interface files are byte copies from one declared view parent, and
``view_manifest.json`` records both. The donor of a weight group never chooses the tokenizer or template. A view is built
only after its weight files match the recorded hashes (``audit/parent_weight_sha256.json`` or the hybrid manifest).
"""

from __future__ import annotations

import fnmatch
import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

from .common import ROLES, FollowupConfig, FollowupError, sha256_file, write_json

from trace_merge.compatibility import DOC_FILES  # noqa: E402
from trace_merge.views import MARKER_NAME, VIEW_MANIFEST, VIEW_PARENT, WEIGHT_GLOBS, _check_existing  # noqa: E402


def view_specs(cfg: FollowupConfig) -> dict[str, tuple[str, str, str]]:
    """label (e.g. 'TF@P1') -> (served slug, weight source, protocol id)."""

    out = {}
    for cell, stem in {**cfg.cells, **cfg.extension_cells}.items():
        for short, protocol_id in cfg.raw["protocols"].items():
            out[f"{cell}@{short.upper()}"] = (f"{stem}-{short}", cell, protocol_id)
    return out


def weight_source_dir(cfg: FollowupConfig, source: str) -> Path:
    if source in ROLES:
        return cfg.parent_dir(source)
    built = cfg.constructed(source)
    if built is None or not built[1].is_file():
        raise FollowupError(f"{source} is not built ({built[0] if built else 'unknown source'})")
    return built[0]


@lru_cache(maxsize=None)
def _sha(path: str) -> str:
    return sha256_file(Path(path))


def recorded_weight_hashes(cfg: FollowupConfig, source: str, wdir: Path) -> dict[str, str]:
    if source in ROLES:
        path = cfg.work("audit", "parent_weight_sha256.json")
        if not path.is_file():
            raise FollowupError(f"{path} is missing; run inventory first")
        return dict(json.loads(path.read_text(encoding="utf-8"))[source])
    return dict(json.loads(cfg.constructed(source)[1].read_text(encoding="utf-8"))["output_file_hashes"])


def build_view(cfg: FollowupConfig, label: str) -> dict[str, Any]:
    specs = view_specs(cfg)
    if label not in specs:
        raise FollowupError(f"unknown view label {label!r}")
    slug, source, protocol_id = specs[label]
    spec = cfg.base.raw["protocols"][protocol_id]
    view_parent = VIEW_PARENT[protocol_id]
    if spec.get("view") != view_parent:
        raise FollowupError(f"{protocol_id}: config view {spec.get('view')} != {view_parent}")
    wdir = weight_source_dir(cfg, source)
    vdir = cfg.parent_dir(view_parent)
    out = cfg.work("views", slug)
    weights = sorted({p for g in WEIGHT_GLOBS for p in wdir.glob(g)})
    meta = {name: vdir / name for name in spec["files"]}
    missing = [n for n, p in meta.items() if not p.is_file()]
    if missing:
        raise FollowupError(f"view parent {view_parent} lacks {missing}")
    unlisted = sorted(p.name for p in vdir.iterdir() if p.name not in meta and p.name not in DOC_FILES
                      and not any(fnmatch.fnmatchcase(p.name, g) for g in WEIGHT_GLOBS))
    if unlisted:
        raise FollowupError(f"view parent {view_parent} has unlisted runtime files {unlisted}")
    revision = (cfg.revision(source) if source in ROLES
                else json.loads(cfg.constructed(source)[1].read_text(encoding="utf-8"))["content_revision"])
    manifest = {
        "schema": "trace-followup-view-v1",
        "label": label,
        "slug": slug,
        "protocol_id": protocol_id,
        "weight_source": source,
        "weight_source_revision": revision,
        "view_parent": view_parent,
        "view_parent_revision": cfg.revision(view_parent),
        "weights": {p.name: p.resolve().name for p in weights},
        "view_files_sha256": {n: sha256_file(p) for n, p in meta.items()},
    }
    if (out / VIEW_MANIFEST).is_file():
        old = json.loads((out / VIEW_MANIFEST).read_text(encoding="utf-8"))
        if {k: old.get(k) for k in manifest} != manifest:
            raise FollowupError(f"{out} exists with a different view manifest")
        _check_existing(out, manifest, weights)
        return old
    if out.exists():
        raise FollowupError(f"{out} exists without a view manifest; inspect it")
    recorded = recorded_weight_hashes(cfg, source, wdir)
    covered = {p.name for p in weights if source not in ROLES or p.suffix == ".safetensors"}
    if set(recorded) != covered:
        raise FollowupError(f"{wdir}: weight files {sorted(covered)} differ from the recorded set {sorted(recorded)}")
    for p in weights:
        if p.name in recorded and _sha(str(p.resolve())) != recorded[p.name]:
            raise FollowupError(f"{p}: sha256 differs from the hash recorded for {source}")
    tmp = out.with_name(f".{out.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for p in weights:
        (tmp / p.name).symlink_to(p.resolve())
    for n, p in meta.items():
        shutil.copyfile(p.resolve(), tmp / n)
        if sha256_file(tmp / n) != manifest["view_files_sha256"][n]:
            raise FollowupError(f"copy of {n} does not match its source")
    write_json(tmp / VIEW_MANIFEST, manifest)
    tmp.rename(out)
    return manifest


def build_views(cfg: FollowupConfig, labels: list[str] | None = None) -> dict[str, Any]:
    labels = labels or list(view_specs(cfg))
    return {label: {k: v for k, v in build_view(cfg, label).items() if k != "view_files_sha256"} for label in labels}


__all__ = ["MARKER_NAME", "build_view", "build_views", "view_specs"]
