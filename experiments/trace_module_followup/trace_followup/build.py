"""Exact-copy component-swap hybrids (design/followup_v1.md, construction algorithm).

For each output tensor the stored bytes of its assigned donor are copied unchanged (no dtype conversion, no
arithmetic). The output keeps B's tensor-to-shard layout through ``trace_merge``'s streaming writer. A build happens in a
staging directory under an exclusive lock and is finalized by an atomic rename only after these checks on the staged
files: every expected tensor present with the base name/shape/dtype, index consistent with the shard headers, every
tensor's sha256 equal to its donor tensor's sha256 from the parent inventory, and no non-finite value. Parent files are
checked unchanged before and after. The manifest records the per-tensor source, parent pins and hashes, code hashes and
output hashes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .common import FollowupConfig, FollowupError, canonical_json, sha256_bytes, sha256_file, write_json
from .components import donor_map, ownership, partition_def

from trace_merge.merge import _build_lock, _fsync, _git_commit, _git_dirty, _layout, file_identity  # noqa: E402
from trace_merge.safetensors_io import StreamingShardWriter, TensorInfo, index_checkpoint  # noqa: E402

MANIFEST_NAME = "manifest.json"
RECIPE_NAME = "recipe.json"
SCHEMA = "trace-followup-hybrid-manifest-v1"
CHUNK = 1 << 26  # 64 MiB, even, so bf16 words never straddle a chunk
CODE_FILES = ("build.py", "components.py", "inventory.py", "common.py")


def code_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    out = {f"trace_followup/{n}": sha256_file(here / n) for n in CODE_FILES}
    merge_pkg = here.parents[1] / "trace_merge" / "trace_merge"
    for n in ("safetensors_io.py", "merge.py", "compatibility.py"):
        out[f"trace_merge/{n}"] = sha256_file(merge_pkg / n)
    return out


def nonfinite_count(buf: bytes, dtype: str) -> int:
    """Number of inf/nan values in a little-endian buffer (bf16/f16/f32); other dtypes have none."""

    if dtype == "BF16":
        a = np.frombuffer(buf, dtype="<u2")
        return int(np.count_nonzero((a & 0x7F80) == 0x7F80))
    if dtype == "F16":
        a = np.frombuffer(buf, dtype="<u2")
        return int(np.count_nonzero((a & 0x7C00) == 0x7C00))
    if dtype == "F32":
        a = np.frombuffer(buf, dtype="<u4")
        return int(np.count_nonzero((a & 0x7F800000) == 0x7F800000))
    return 0


def copy_tensor_exact(info: TensorInfo, write: Callable[[bytes], None]) -> str:
    """Stream one stored tensor's bytes from its file to ``write``; returns the sha256 of the bytes copied."""

    h = hashlib.sha256()
    with open(info.file, "rb") as f:
        f.seek(info.begin)
        left = info.nbytes
        while left:
            buf = f.read(min(CHUNK, left))
            if not buf:
                raise FollowupError(f"{info.file}: short read for {info.name}")
            h.update(buf)
            write(buf)
            left -= len(buf)
    return h.hexdigest()


def scan_checkpoint(model_dir: Path) -> dict[str, Any]:
    """One sequential pass per shard: file sha256, and per tensor its sha256 and non-finite count."""

    index = index_checkpoint(model_dir)
    by_file: dict[Path, list[TensorInfo]] = {}
    for info in index.values():
        by_file.setdefault(info.file, []).append(info)
    files: dict[str, str] = {}
    tensors: dict[str, dict[str, Any]] = {}
    for path, infos in sorted(by_file.items()):
        infos = sorted(infos, key=lambda t: t.begin)
        fh = hashlib.sha256()
        th = {t.name: hashlib.sha256() for t in infos}
        bad = {t.name: 0 for t in infos}
        pos, k = 0, 0
        with open(path, "rb") as f:
            while True:
                buf = f.read(CHUNK)
                if not buf:
                    break
                fh.update(buf)
                lo, hi = pos, pos + len(buf)
                while k < len(infos) and infos[k].end <= lo:
                    k += 1
                j = k
                while j < len(infos) and infos[j].begin < hi:
                    t = infos[j]
                    a, b = max(t.begin, lo), min(t.end, hi)
                    if a < b:
                        part = buf[a - lo:b - lo]
                        th[t.name].update(part)
                        bad[t.name] += nonfinite_count(part, t.dtype)
                    j += 1
                pos = hi
        name = infos[0].shard or path.name
        files[name] = fh.hexdigest()
        for t in infos:
            tensors[t.name] = {"sha256": th[t.name].hexdigest(), "nonfinite": bad[t.name], "dtype": t.dtype,
                               "shape": list(t.shape), "shard": t.shard, "numel": t.numel}
    return {"files": files, "tensors": tensors}


def recipe_for(cfg: FollowupConfig, hybrid_id: str, names: list[str]) -> dict[str, Any]:
    spec = cfg.hybrids[hybrid_id]
    owner = ownership(cfg, names, spec["partition"])
    parts, partition = partition_def(cfg, spec["partition"])
    return {
        "hybrid_id": hybrid_id,
        "partition": spec["partition"],
        "partition_groups": partition,
        "parts": parts,
        "donors": dict(sorted(spec["donors"].items())),
        "donor_revisions": {r: cfg.revision(r) for r in sorted(set(spec["donors"].values()) | {"B"})},
        "layout_role": "B",
        "mask_sha256": sha256_bytes(canonical_json(dict(sorted(owner.items()))).encode("utf-8")),
        "operation": "exact copy of the donor's stored tensor bytes; no averaging, no dtype conversion",
        "no_training": True,
    }


def build_hybrid(cfg: FollowupConfig, hybrid_id: str, parent_inventory: dict[str, Any], *,
                 log: Callable[[str], None] = print, dry_run: bool = False) -> dict[str, Any]:
    out_root = cfg.work("hybrids")
    out_root.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return _build(cfg, hybrid_id, parent_inventory, out_root, log=log, dry_run=True)
    with _build_lock(out_root / f".{hybrid_id}.lock"):
        return _build(cfg, hybrid_id, parent_inventory, out_root, log=log, dry_run=False)


def _build(cfg: FollowupConfig, hybrid_id: str, inv: dict[str, Any], out_root: Path, *, log: Callable[[str], None],
           dry_run: bool) -> dict[str, Any]:
    parents = {r: cfg.parent_dir(r) for r in ("B", "T", "V")}
    indexes = {r: index_checkpoint(p) for r, p in parents.items()}
    names = sorted(indexes["B"])
    for r, idx in indexes.items():
        if sorted(idx) != names:
            raise FollowupError(f"{r}: tensor names differ from B")
        for n in names:
            if (idx[n].dtype, idx[n].shape) != (indexes["B"][n].dtype, indexes["B"][n].shape):
                raise FollowupError(f"{r}: {n} dtype/shape differs from B")
    recipe = recipe_for(cfg, hybrid_id, names)
    rsha = sha256_bytes(canonical_json(recipe).encode("utf-8"))
    dm = donor_map(cfg, hybrid_id, names)
    final = out_root / hybrid_id
    staging = out_root / f".{hybrid_id}.staging"
    plan = {"hybrid_id": hybrid_id, "recipe_sha256": rsha, "out": str(final),
            "tensors_by_donor": {r: sum(1 for v in dm.values() if v == r) for r in sorted(set(dm.values()))}}
    if dry_run:
        log(f"[build] dry run: {json.dumps(plan)}")
        return {"dry_run": True, **plan}
    if (final / MANIFEST_NAME).is_file():
        existing = json.loads((final / MANIFEST_NAME).read_text(encoding="utf-8"))
        if existing.get("recipe_sha256") != rsha:
            raise FollowupError(f"{final} exists with a different recipe; refusing to overwrite")
        log(f"[build] {hybrid_id}: already built (recipe {rsha[:12]}); run verify-builds to re-check")
        return existing
    if final.exists():
        raise FollowupError(f"{final} exists without a manifest; inspect and remove it by hand")
    if staging.exists():
        rec = staging / RECIPE_NAME
        if not rec.is_file() or json.loads(rec.read_text(encoding="utf-8")).get("recipe_sha256") != rsha:
            raise FollowupError(f"{staging} holds a different or unknown recipe; remove it by hand")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    write_json(staging / RECIPE_NAME, {"recipe": recipe, "recipe_sha256": rsha})

    files_before = {r: {f.name: file_identity(f) for f in sorted(p.glob("*.safetensors"))} for r, p in parents.items()}
    for r, p in parents.items():  # parent weight files must still be the inventoried bytes (cheap identity check)
        if {k: v for k, v in files_before[r].items()} != inv["parents"][r]["file_identities"]:
            raise FollowupError(f"{r}: parent weight files changed since the inventory; rerun inventory")
    layout = _layout(parents["B"], set(names))
    started = dt.datetime.now(dt.timezone.utc)
    weight_map: dict[str, str] = {}
    copied: dict[str, str] = {}
    total = 0
    for shard, tensor_names in layout.items():
        spec = [(n, indexes["B"][n].dtype, indexes["B"][n].shape) for n in tensor_names]
        writer = StreamingShardWriter(staging / shard, spec, {"format": "pt"})
        for n in writer.order:
            writer.begin(n)
            copied[n] = copy_tensor_exact(indexes[dm[n]][n], writer.write)
            writer.end()
            weight_map[n] = shard
        writer.close()
        total += writer.total - len(writer.header_bytes)
        log(f"[build] {hybrid_id}: wrote {shard} ({len(tensor_names)} tensors)")
    write_json(staging / "model.safetensors.index.json", {"metadata": {"total_size": total}, "weight_map": weight_map})

    # ---- checks on the staged files (completeness, index, tensor content, finiteness, parents unchanged)
    out_index = index_checkpoint(staging)
    if sorted(out_index) != names:
        raise FollowupError(f"{hybrid_id}: staged tensors differ from the base inventory")
    for n, t in out_index.items():
        if (t.dtype, t.shape) != (indexes["B"][n].dtype, indexes["B"][n].shape):
            raise FollowupError(f"{hybrid_id}: staged {n} has dtype/shape {t.dtype} {t.shape}")
    scan = scan_checkpoint(staging)
    mismatch = [n for n in names if not (scan["tensors"][n]["sha256"] == copied[n]
                                         == inv["parents"][dm[n]]["tensors"][n]["sha256"])]
    if mismatch:
        raise FollowupError(f"{hybrid_id}: {len(mismatch)} tensors differ from their donor, e.g. {mismatch[:3]}")
    nonfinite = {n: v["nonfinite"] for n, v in scan["tensors"].items() if v["nonfinite"]}
    if nonfinite:
        raise FollowupError(f"{hybrid_id}: non-finite values in {list(nonfinite)[:3]}")
    files_after = {r: {f.name: file_identity(f) for f in sorted(p.glob("*.safetensors"))} for r, p in parents.items()}
    if files_after != files_before:
        raise FollowupError("a parent weight file changed during the build")
    output_hashes = dict(scan["files"])
    output_hashes["model.safetensors.index.json"] = sha256_file(staging / "model.safetensors.index.json")
    # which tensors differ from B (the hybrid's realized change set; used by the ownership tests)
    differs_from_b = sorted(n for n in names if copied[n] != inv["parents"]["B"]["tensors"][n]["sha256"])
    impl_root = Path(__file__).resolve().parents[3]
    manifest = {
        "schema_version": SCHEMA,
        "hybrid_id": hybrid_id,
        "recipe": recipe,
        "recipe_sha256": rsha,
        "parent_repo_ids": {r: cfg.base.role(r)["repo_id"] for r in ("B", "T", "V")},
        "parent_revisions": {r: cfg.revision(r) for r in ("B", "T", "V")},
        "parent_weight_file_sha256": {r: inv["parents"][r]["files"] for r in ("B", "T", "V")},
        "parent_inventory_sha256": inv["inventory_sha256"],
        "layout_parent": {"role": "B", "index_sha256": sha256_file(parents["B"] / "model.safetensors.index.json")},
        "tensor_count": len(names),
        "tensors_by_donor": plan["tensors_by_donor"],
        "tensor_sources": {n: {"donor": dm[n], "sha256": copied[n]} for n in names},
        "tensors_differing_from_B": len(differs_from_b),
        "tensors_differing_from_B_sha256": sha256_bytes(canonical_json(differs_from_b).encode("utf-8")),
        "checks": {"complete": True, "index_consistent": True, "tensor_bytes_equal_donor": True, "finite": True,
                   "parents_unchanged": True},
        "implementation_commit": _git_commit(impl_root),
        "implementation_dirty": _git_dirty(impl_root),
        "implementation_code_sha256": code_hashes(),
        "environment_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                 "platform": platform.platform()},
        "output_file_hashes": output_hashes,
        "content_revision": "sha256set:" + sha256_bytes(canonical_json(output_hashes).encode("utf-8")),
        "content_revision_note": "local derived provenance (hash of the output weight files); not a Hugging Face commit",
        "started": started.isoformat(timespec="seconds"),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "no_training": True,
        "inference_assets": "weights only; inference views (tokenizer/template/processor/config) are separate directories",
    }
    write_json(staging / MANIFEST_NAME, manifest)
    for p in sorted(staging.iterdir()):
        os.chmod(p, 0o444)
    _fsync(*sorted(staging.iterdir()), staging)
    os.replace(staging, final)
    _fsync(out_root)
    log(f"[build] {hybrid_id}: finalized {final} ({manifest['content_revision'][:24]}...)")
    return manifest


def verify_hybrid(cfg: FollowupConfig, hybrid_id: str, inv: dict[str, Any]) -> dict[str, Any]:
    """Re-read a finished hybrid: files match the manifest, every tensor equals its intended donor's tensor."""

    final = cfg.work("hybrids", hybrid_id)
    manifest = json.loads((final / MANIFEST_NAME).read_text(encoding="utf-8"))
    present = {p.name for p in final.iterdir()}
    expected = set(manifest["output_file_hashes"]) | {MANIFEST_NAME, RECIPE_NAME}
    if present != expected:
        raise FollowupError(f"{final}: files {sorted(present ^ expected)} disagree with the manifest")
    scan = scan_checkpoint(final)
    for name, digest in scan["files"].items():
        if manifest["output_file_hashes"].get(name) != digest:
            raise FollowupError(f"{final / name}: sha256 differs from the manifest")
    names = sorted(scan["tensors"])
    dm = donor_map(cfg, hybrid_id, names)
    bad = [n for n in names if scan["tensors"][n]["sha256"] != inv["parents"][dm[n]]["tensors"][n]["sha256"]]
    if bad:
        raise FollowupError(f"{hybrid_id}: {len(bad)} tensors do not equal their donor, e.g. {bad[:3]}")
    if any(v["nonfinite"] for v in scan["tensors"].values()):
        raise FollowupError(f"{hybrid_id}: non-finite values")
    return {"hybrid_id": hybrid_id, "tensors": len(names), "donor_equal": True,
            "content_revision": manifest["content_revision"]}
