"""Inference views: weights from one source, non-weight files (config, tokenizer, chat template, processor, generation
config) from a prespecified view parent (DESIGN.md section 3).

A view directory holds symlinks to the weight shards and index of its weight source and byte copies of the view
parent's non-weight files, plus ``view_manifest.json``. The evaluation runner registers it with the protocol's own
``register-local`` marker, whose content revision hashes every file in the directory (weights included), so the
served bytes are content-bound. A new view is built only after its weight files match the recorded hashes of their
source (the merge manifest, or ``audit/parent_weight_sha256.json`` for a parent) and only when the view parent holds no
runtime file outside the protocol's list; an existing view is reused only while its files still match its manifest.
"""

from __future__ import annotations

import fnmatch
import json
import shutil
from pathlib import Path
from typing import Any

from .compatibility import DOC_FILES
from .config import StudyConfig, sha256_file, write_json

VIEW_MANIFEST = "view_manifest.json"
MARKER_NAME = ".trace_model_revision.json"  # written into the view by the protocol's register-local
WEIGHT_GLOBS = ("*.safetensors", "model.safetensors.index.json")

# label -> (served slug, weight source, protocol id). Weight sources: a role (pinned HF snapshot) or a merge id.
VIEW_SPECS: dict[str, tuple[str, str, str]] = {
    "V@P1": ("vero-p1", "V", "P1_backbone"),
    "M050@P1": ("m050-p1", "M050", "P1_backbone"),
    "CV050@P1": ("cv050-p1", "CV050", "P1_backbone"),
    "CT050@P1": ("ct050-p1", "CT050", "P1_backbone"),
    "B@P2": ("qwen25vl7b-p2", "B", "P2_vero"),
    "T@P2": ("trace7b-p2", "T", "P2_vero"),
    "M050@P2": ("m050-p2", "M050", "P2_vero"),
    "CV050@P2": ("cv050-p2", "CV050", "P2_vero"),
    "CT050@P2": ("ct050-p2", "CT050", "P2_vero"),
}
VIEW_PARENT = {"P1_backbone": "B", "P2_vero": "V"}


class ViewError(RuntimeError):
    pass


def weight_source_dir(cfg: StudyConfig, source: str) -> Path:
    if source in ("B", "T", "V"):
        return cfg.hf_snapshot(source)
    d = cfg.work("merges", source)
    if not (d / "merge_manifest.json").is_file():
        raise ViewError(f"merge {source} is not built ({d})")
    return d


def recorded_weight_hashes(cfg: StudyConfig, source: str, wdir: Path) -> dict[str, str]:
    """sha256 of the weight files as recorded when they were made: the merge manifest's output files, or a parent's
    safetensors shards from ``audit/parent_weight_sha256.json`` (written by ``parent-hashes``)."""

    if source in ("B", "T", "V"):
        path = cfg.work("audit", "parent_weight_sha256.json")
        if not path.is_file():
            raise ViewError(f"{path} is missing; run parent-hashes before building a view of {source}")
        return dict(json.loads(path.read_text(encoding="utf-8"))[source])
    return dict(json.loads((wdir / "merge_manifest.json").read_text(encoding="utf-8"))["output_file_hashes"])


def _check_existing(out: Path, manifest: dict[str, Any], weights: list[Path]) -> None:
    """Cheap re-check of a finished view: weight links still point at the source files, copied files still have the
    manifest's hashes, and nothing else was added (besides the protocol's registration marker)."""

    for p in weights:
        link = out / p.name
        if not link.is_symlink() or link.resolve() != p.resolve():
            raise ViewError(f"{link} no longer links to {p.resolve()}")
    for n, digest in manifest["view_files_sha256"].items():
        if not (out / n).is_file() or (out / n).is_symlink() or sha256_file(out / n) != digest:
            raise ViewError(f"{out / n} no longer matches its view manifest")
    allowed = {p.name for p in weights} | set(manifest["view_files_sha256"]) | {VIEW_MANIFEST, MARKER_NAME}
    extra = sorted({p.name for p in out.iterdir()} - allowed)
    if extra:
        raise ViewError(f"{out} holds files outside its view manifest: {extra}")


def build_view(cfg: StudyConfig, label: str, *, force: bool = False) -> dict[str, Any]:
    if label not in VIEW_SPECS:
        raise ViewError(f"unknown view label {label!r}")
    slug, source, protocol_id = VIEW_SPECS[label]
    spec = cfg.raw["protocols"][protocol_id]
    view_parent = VIEW_PARENT[protocol_id]
    if spec.get("view") != view_parent:
        raise ViewError(f"{protocol_id}: config view {spec.get('view')} != {view_parent}")
    files = list(spec["files"])
    wdir = weight_source_dir(cfg, source)
    vdir = cfg.hf_snapshot(view_parent)
    out = cfg.work("views", slug)
    weights = sorted({p for g in WEIGHT_GLOBS for p in wdir.glob(g)})
    if not any(p.suffix == ".safetensors" for p in weights):
        raise ViewError(f"{wdir}: no weights")
    meta = {name: vdir / name for name in files}
    missing = [n for n, p in meta.items() if not p.is_file()]
    if missing:
        raise ViewError(f"view parent {view_parent} lacks {missing}")
    unlisted = sorted(p.name for p in vdir.iterdir() if p.name not in meta and p.name not in DOC_FILES
                      and not any(fnmatch.fnmatchcase(p.name, g) for g in WEIGHT_GLOBS))
    if unlisted:
        raise ViewError(f"view parent {view_parent} has files outside {protocol_id}'s list that the view would drop: "
                        f"{unlisted}; add them to the protocol's files or to DOC_FILES after review")
    manifest = {
        "schema": "trace-merge-view-v1",
        "label": label,
        "slug": slug,
        "protocol_id": protocol_id,
        "weight_source": source,
        "weight_source_revision": (cfg.role(source)["source_revision"] if source in ("B", "T", "V")
                                   else json.loads((wdir / "merge_manifest.json").read_text())["content_revision"]),
        "view_parent": view_parent,
        "view_parent_revision": cfg.role(view_parent)["source_revision"],
        "weights": {p.name: p.resolve().name for p in weights},
        "view_files_sha256": {n: sha256_file(p) for n, p in meta.items()},
    }
    if out.exists():
        existing = out / VIEW_MANIFEST
        if existing.is_file():
            old = json.loads(existing.read_text(encoding="utf-8"))
            same = {k: old.get(k) for k in manifest} == manifest
            if same and not force:
                _check_existing(out, manifest, weights)
                return old
            if not force:
                raise ViewError(f"{out} exists with a different view manifest; pass force after inspecting it")
        if not force:
            raise ViewError(f"{out} exists without a view manifest")
    recorded = recorded_weight_hashes(cfg, source, wdir)
    covered = {p.name for p in weights if source not in ("B", "T", "V") or p.suffix == ".safetensors"}
    if set(recorded) != covered:
        raise ViewError(f"{wdir}: weight files {sorted(covered)} differ from the recorded set {sorted(recorded)}")
    for p in weights:
        if p.name in recorded and sha256_file(p) != recorded[p.name]:
            raise ViewError(f"{p}: sha256 differs from the hash recorded for {source}")
    if out.exists():
        shutil.rmtree(out)
    tmp = out.with_name(f".{out.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for p in weights:
        (tmp / p.name).symlink_to(p.resolve())
    for n, p in meta.items():
        shutil.copyfile(p.resolve(), tmp / n)
        if sha256_file(tmp / n) != manifest["view_files_sha256"][n]:
            raise ViewError(f"copy of {n} does not match its source")
    write_json(tmp / VIEW_MANIFEST, manifest)
    tmp.rename(out)
    return manifest


def build_all_views(cfg: StudyConfig, labels: list[str] | None = None) -> dict[str, Any]:
    return {label: build_view(cfg, label) for label in (labels or list(VIEW_SPECS))}
