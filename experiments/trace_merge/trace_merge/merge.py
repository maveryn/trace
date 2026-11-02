"""Safe, deterministic, training-free linear weight interpolation (handoff 7.2-7.4).

``out = sum_i c_i * float32(parent_i)`` for every floating tensor, accumulated in float32 from the bf16 parents and
cast once to bf16 (round-to-nearest-even). Tensors are processed in bounded row slices, so peak memory is a few hundred
MB regardless of tensor size. Non-floating tensors must be bitwise identical in all parents and are copied (never
averaged). The output keeps the layout parent's tensor-to-shard assignment. Builds hold an exclusive per-merge lock,
happen in a staging directory and are finalized by an atomic rename after an fsync; a finished directory whose manifest
has a different recipe is never overwritten, and one with the same recipe is re-verified before it is reused.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import math
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

from .compatibility import audit, require_compatible, tensor_group
from .config import StudyConfig, canonical_json, sha256_bytes, sha256_file, write_json
from .safetensors_io import (FLOAT_DTYPES, TORCH_DTYPES, StreamingShardWriter, TensorInfo, index_checkpoint,
                             iter_row_slices, read_header, read_rows, tensor_bytes)

MANIFEST_NAME = "merge_manifest.json"
RECIPE_NAME = "recipe.json"
SCHEMA = "trace-merge-manifest-v1"
CODE_FILES = ("merge.py", "safetensors_io.py", "compatibility.py", "config.py")


class MergeError(RuntimeError):
    pass


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def _git_dirty(path: Path) -> bool | None:
    try:
        out = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--", "experiments/trace_merge"],
                             capture_output=True, text=True, check=True).stdout
        return bool(out.strip())
    except Exception:
        return None


def file_identity(path: Path) -> dict[str, Any]:
    """Cheap change detector for a parent file (resolved target, size, mtime, inode); full hashes are separate."""

    real = path.resolve()
    st = real.stat()
    return {"name": path.name, "resolved": real.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "inode": st.st_ino}


def implementation_code_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    return {f"trace_merge/{name}": sha256_file(here / name) for name in CODE_FILES}


def recipe_for(merge_id: str, coefficients: dict[str, float], parents: dict[str, Path], layout_role: str,
               source_revisions: dict[str, str], *, accumulation_dtype: str = "float32",
               output_dtype: str = "bfloat16", max_slice_elements: int = 1 << 25) -> dict[str, Any]:
    if set(coefficients) - set(parents):
        raise MergeError(f"{merge_id}: coefficients for unknown parents {set(coefficients) - set(parents)}")
    bad = {k: v for k, v in coefficients.items() if not (math.isfinite(float(v)) and 0.0 <= float(v) <= 1.0)}
    if bad:
        raise MergeError(f"{merge_id}: coefficients must be finite and in [0, 1] (interpolation only; got {bad})")
    if abs(sum(coefficients.values()) - 1.0) > 1e-12:
        raise MergeError(f"{merge_id}: coefficients must sum to 1 (got {coefficients})")
    if accumulation_dtype != "float32" or output_dtype != "bfloat16":
        raise MergeError("only float32 accumulation and bfloat16 output are supported")
    return {
        "merge_id": merge_id,
        "model_role_coefficients": {k: float(coefficients[k]) for k in sorted(coefficients)},
        "parent_source_revisions": {k: source_revisions[k] for k in sorted(coefficients)},
        "layout_role": layout_role,
        "accumulation_dtype": accumulation_dtype,
        "output_dtype": output_dtype,
        "rounding": "torch float32 -> bfloat16 round-to-nearest-even",
        "component_mask": None,
        "component_mask_hash": None,
        "max_slice_elements": int(max_slice_elements),
        "nonfloating_buffer_policy": "must be bitwise identical in all parents; copied, never averaged",
        "tied_weight_policy": "Qwen2.5-VL-7B stores untied embed_tokens and lm_head; both are interpolated as separate tensors",
        "no_training": True,
    }


def recipe_sha256(recipe: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(recipe).encode("utf-8"))


def interpolate_slice(slices: list[torch.Tensor], coeffs: list[float]) -> torch.Tensor:
    """sum_i c_i * float32(x_i), cast once to bf16. Fails on non-finite inputs or outputs.

    A term with an exact zero coefficient is skipped (it contributes nothing), so the endpoints copy their parent
    bitwise, signed zeros included."""

    acc: torch.Tensor | None = None
    for x, c in zip(slices, coeffs):
        if c == 0.0:
            continue
        xf = x.to(torch.float32)
        if not torch.isfinite(xf).all():
            raise MergeError("non-finite parent values")
        term = xf * c if c != 1.0 else xf
        acc = term if acc is None else acc + term
    assert acc is not None
    if not torch.isfinite(acc).all():
        raise MergeError("non-finite merged values")
    return acc.to(torch.bfloat16)


def merge_tensor(name: str, infos: list[TensorInfo], coeffs: list[float], write: Callable[[bytes], None],
                 max_elements: int) -> None:
    ref = infos[0]
    if any(i.shape != ref.shape or i.dtype != ref.dtype for i in infos):
        raise MergeError(f"{name}: shape/dtype mismatch across parents")
    if ref.dtype not in FLOAT_DTYPES:
        blobs = []
        for info in infos:
            with open(info.file, "rb") as f:
                f.seek(info.begin)
                blobs.append(f.read(info.nbytes))
        if any(b != blobs[0] for b in blobs[1:]):
            raise MergeError(f"{name}: non-floating tensor differs across parents (needs explicit review)")
        write(blobs[0])
        return
    if ref.dtype != "BF16":
        raise MergeError(f"{name}: parents are {ref.dtype}; only bf16 parents are supported in this study")
    for r0, r1 in iter_row_slices(ref, max_elements):
        out = interpolate_slice([read_rows(i, r0, r1) for i in infos], coeffs)
        write(tensor_bytes(out))


def _layout(layout_dir: Path, names: set[str]) -> dict[str, list[str]]:
    """shard file name -> ordered tensor names, from the layout parent's index (single-file checkpoints: one shard)."""

    index_path = layout_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    else:
        files = sorted(layout_dir.glob("*.safetensors"))
        if len(files) != 1:
            raise MergeError(f"{layout_dir}: no index and {len(files)} shards")
        weight_map = {n: files[0].name for n in names}
    if set(weight_map) != names:
        raise MergeError("layout parent's weight map does not cover exactly the merged tensors")
    shards: dict[str, list[str]] = {}
    for n, shard in weight_map.items():
        shards.setdefault(Path(shard).name, []).append(n)
    return {k: sorted(v) for k, v in sorted(shards.items())}


def verify_finished(final: Path, manifest: dict[str, Any], *, deep: bool = True) -> None:
    """A finished build holds exactly its manifest's files, every shard's size matches its header and, when ``deep``,
    every output file still has the sha256 recorded in the manifest."""

    hashes = manifest.get("output_file_hashes") or {}
    present = {p.name for p in final.iterdir()}
    expected = set(hashes) | {MANIFEST_NAME, RECIPE_NAME}
    if present != expected:
        raise MergeError(f"{final}: files {sorted(present ^ expected)} disagree with its manifest")
    try:
        index_checkpoint(final)
        for f in sorted(final.glob("*.safetensors")):
            header, _, start = read_header(f)
            end = max((int(s["data_offsets"][1]) for s in header.values()), default=0)
            if f.stat().st_size != start + end:
                raise MergeError(f"{f}: {f.stat().st_size} bytes but its header declares {start + end}")
    except ValueError as exc:
        raise MergeError(f"{final}: {exc}") from exc
    if deep:
        for name, digest in sorted(hashes.items()):
            if sha256_file(final / name) != digest:
                raise MergeError(f"{final / name}: sha256 differs from {MANIFEST_NAME}")


def _verify_windows(info: TensorInfo, max_elements: int, full: bool) -> list[tuple[int, int]]:
    """Row windows of an output tensor that the build recomputes and compares: the first rows, the last rows and one
    interior window seeded by the tensor name, for every tensor (every slice when ``full``)."""

    rows = info.shape[0] if info.shape else 1
    if rows == 0:
        return []
    if full:
        return list(iter_row_slices(info, max_elements))
    k = min(rows, 2)
    mid = int(sha256_bytes(info.name.encode("utf-8"))[:8], 16) % (rows - k + 1)
    return sorted({(0, k), (mid, mid + k), (rows - k, rows)})


@contextlib.contextmanager
def _build_lock(path: Path) -> Iterator[None]:
    """Exclusive per-merge lock (flock on a lock file); a second builder of the same merge id fails at once."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise MergeError(f"{path}: lock held by another build of this merge ({exc})") from exc
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _fsync(*paths: Path) -> None:
    """Flush files or directories to stable storage (before and after the atomic rename of a build)."""

    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def build_merge(merge_id: str, coefficients: dict[str, float], parents: dict[str, Path], out_root: Path,
                **kwargs: Any) -> dict[str, Any]:
    """Build (or verify an existing build of) one merged checkpoint; returns its manifest. Everything except a dry run
    holds an exclusive per-merge lock, so two builders never share a staging directory (arguments: ``_build_merge``)."""

    if kwargs.get("dry_run"):
        return _build_merge(merge_id, coefficients, parents, out_root, **kwargs)
    with _build_lock(out_root / f".{merge_id}.lock"):
        return _build_merge(merge_id, coefficients, parents, out_root, **kwargs)


def _build_merge(merge_id: str, coefficients: dict[str, float], parents: dict[str, Path], out_root: Path, *,
                 layout_role: str, source_revisions: dict[str, str], lineage: dict[str, Any] | None = None,
                 max_slice_elements: int = 1 << 25, implementation_root: Path | None = None,
                 parent_sha256: dict[str, dict[str, str]] | None = None, log: Callable[[str], None] = print,
                 dry_run: bool = False, full_verify: bool = False) -> dict[str, Any]:
    used = {r: parents[r] for r in coefficients}
    recipe = recipe_for(merge_id, coefficients, used, layout_role, source_revisions,
                        max_slice_elements=max_slice_elements)
    rsha = recipe_sha256(recipe)
    final = out_root / merge_id
    staging = out_root / f".{merge_id}.staging"
    if (final / MANIFEST_NAME).is_file():
        existing = json.loads((final / MANIFEST_NAME).read_text(encoding="utf-8"))
        if existing.get("recipe_sha256") != rsha:
            raise MergeError(f"{final} exists with a different recipe ({existing.get('recipe_sha256')} != {rsha}); "
                             "refusing to overwrite")
        # the recipe does not name the layout parent, so its tensor-to-shard map is compared directly
        verify_finished(final, existing, deep=not dry_run)
        names = set(json.loads((final / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"])
        if _layout(final, names) != _layout(parents[layout_role], names):
            raise MergeError(f"{final} has a different tensor-to-shard layout than layout parent {layout_role}")
        log(f"[merge] {merge_id}: already built with recipe {rsha[:12]}"
            f"{' (structure checked)' if dry_run else ' (output files re-hashed)'}")
        return existing
    if final.exists():
        raise MergeError(f"{final} exists without a manifest; inspect and remove it by hand")

    report = audit({layout_role: parents[layout_role], **used}, lineage)
    require_compatible(report)
    plan = {"merge_id": merge_id, "recipe_sha256": rsha, "out": str(final), "staging": str(staging),
            "parents": {r: str(p) for r, p in used.items()}, "tensors": report["tensors"]["tensor_count"][layout_role],
            "output_bytes_estimate": sum(f.stat().st_size for f in parents[layout_role].glob("*.safetensors"))}
    if dry_run:
        log(f"[merge] dry run: {json.dumps(plan)}")
        return {"dry_run": True, **plan}

    if staging.exists():
        rec = staging / RECIPE_NAME
        if not rec.is_file() or json.loads(rec.read_text(encoding="utf-8")).get("recipe_sha256") != rsha:
            raise MergeError(f"{staging} holds a different or unknown recipe; remove it by hand")
        log(f"[merge] {merge_id}: discarding incomplete staging build with the same recipe")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    write_json(staging / RECIPE_NAME, {"recipe": recipe, "recipe_sha256": rsha})

    code_hashes = implementation_code_hashes()
    layout_index = parents[layout_role] / "model.safetensors.index.json"
    parent_files_before = {r: {f.name: file_identity(f) for f in sorted(p.glob("*.safetensors"))} for r, p in used.items()}
    indexes = {r: index_checkpoint(p) for r, p in used.items()}
    roles = sorted(coefficients)
    coeffs = [float(coefficients[r]) for r in roles]
    names = set(indexes[roles[0]])
    layout = _layout(parents[layout_role], names)
    ref_index = index_checkpoint(parents[layout_role])
    weight_map: dict[str, str] = {}
    total = 0
    started = dt.datetime.now(dt.timezone.utc)
    for shard, tensor_names in layout.items():
        spec = [(n, ref_index[n].dtype if ref_index[n].dtype not in FLOAT_DTYPES else "BF16", ref_index[n].shape)
                for n in tensor_names]
        writer = StreamingShardWriter(staging / shard, spec, {"format": "pt"})
        for n in writer.order:
            writer.begin(n)
            merge_tensor(n, [indexes[r][n] for r in roles], coeffs, writer.write, max_slice_elements)
            writer.end()
            weight_map[n] = shard
        writer.close()
        total += writer.total - len(writer.header_bytes)
        log(f"[merge] {merge_id}: wrote {shard} ({len(tensor_names)} tensors)")
    write_json(staging / "model.safetensors.index.json", {"metadata": {"total_size": total}, "weight_map": weight_map})

    # ---- verification of the staged output
    out_index = index_checkpoint(staging)
    if set(out_index) != names:
        raise MergeError("staged output does not contain exactly the parent tensors")
    for n, t in out_index.items():
        if t.shape != ref_index[n].shape:
            raise MergeError(f"staged {n} has shape {t.shape}")
    for n, info in sorted(out_index.items()):
        for r0, r1 in _verify_windows(info, max_slice_elements, full_verify):
            got = read_rows(info, r0, r1)
            if info.dtype not in FLOAT_DTYPES:
                ok = torch.equal(got, read_rows(indexes[roles[0]][n], r0, r1))
            else:
                expect = interpolate_slice([read_rows(indexes[r][n], r0, r1) for r in roles], coeffs)
                ok = torch.equal(expect.view(torch.int16), got.view(torch.int16))
            if not ok:
                raise MergeError(f"verification failed for {n} rows {r0}:{r1}")
    parent_files_after = {r: {f.name: file_identity(f) for f in sorted(p.glob("*.safetensors"))} for r, p in used.items()}
    if parent_files_after != parent_files_before:
        raise MergeError("a parent weight file changed during the build")
    if parent_sha256:  # the recorded (pre-build) parent hashes, checked again against the files after the build
        for r in roles:
            now = {f.name: sha256_file(f) for f in sorted(used[r].glob("*.safetensors"))}
            if now != parent_sha256.get(r):
                raise MergeError(f"{r}: parent weight sha256 after the build differs from the recorded hashes")
    output_hashes = {f.name: sha256_file(f) for f in sorted(staging.glob("*.safetensors"))}
    output_hashes["model.safetensors.index.json"] = sha256_file(staging / "model.safetensors.index.json")
    shape_digest = sha256_bytes(canonical_json(sorted((n, out_index[n].dtype, list(out_index[n].shape))
                                                      for n in out_index)).encode("utf-8"))
    finished = dt.datetime.now(dt.timezone.utc)
    impl_root = implementation_root or Path(__file__).resolve().parents[3]
    manifest = {
        "schema_version": SCHEMA,
        "merge_id": merge_id,
        "recipe": recipe,
        "recipe_sha256": rsha,
        "model_role_coefficients": recipe["model_role_coefficients"],
        "component_mask_hash": None,
        "parent_repo_ids": {r: (lineage or {}).get(r, {}).get("repo_id") for r in roles},
        "source_revisions": {r: source_revisions[r] for r in roles},
        "runtime_revisions": {r: lineage.get(r, {}).get("runtime_revision") for r in roles} if lineage else None,
        "layout_parent": {"role": layout_role, "source_revision": source_revisions.get(layout_role),
                          "index_sha256": sha256_file(layout_index) if layout_index.is_file() else None},
        "parent_weight_file_identities": parent_files_before,
        "parent_weight_file_sha256": parent_sha256 or None,
        "parent_weight_file_sha256_rechecked_after_build": bool(parent_sha256),
        "parent_config_tokenizer_processor_hashes": {r: report["non_weight_files"][r] for r in roles},
        "compatibility_status": report["status"],
        "accumulation_dtype": "float32",
        "output_dtype": "bfloat16",
        "tensor_count": len(out_index),
        "tensor_shape_digest": shape_digest,
        "tensor_groups": {g: sum(1 for n in out_index if tensor_group(n) == g)
                          for g in sorted({tensor_group(n) for n in out_index})},
        "nonfloating_buffer_policy": recipe["nonfloating_buffer_policy"],
        "tied_weight_policy": recipe["tied_weight_policy"],
        "layout_role": layout_role,
        "implementation_commit": _git_commit(impl_root),
        "implementation_dirty": _git_dirty(impl_root),
        "implementation_code_sha256": code_hashes,
        "environment_versions": {"python": platform.python_version(), "torch": torch.__version__,
                                 "platform": platform.platform()},
        "output_file_hashes": output_hashes,
        "content_revision": "sha256set:" + sha256_bytes(canonical_json(output_hashes).encode("utf-8")),
        "content_revision_note": "local derived provenance (hash of the output weight files); not a Hugging Face commit",
        "started": started.isoformat(timespec="seconds"),
        "timestamp": finished.isoformat(timespec="seconds"),
        "no_training": True,
        "inference_assets": "weights only; inference views (tokenizer/template/processor/config) are separate directories",
    }
    write_json(staging / MANIFEST_NAME, manifest)
    _fsync(*sorted(staging.iterdir()), staging)
    os.replace(staging, final)
    _fsync(out_root)
    log(f"[merge] {merge_id}: finalized {final} ({manifest['content_revision'][:24]}…)")
    return manifest


def build_configured_merges(cfg: StudyConfig, only: list[str] | None = None, dry_run: bool = False,
                            log: Callable[[str], None] = print) -> dict[str, Any]:
    parents = {r: cfg.hf_snapshot(r) for r in ("B", "T", "V")}
    for r, p in parents.items():
        if not p.is_dir():
            raise MergeError(f"{r}: pinned snapshot missing at {p}")
    revisions = {r: cfg.role(r)["source_revision"] for r in parents}
    lineage = {r: {"repo_id": cfg.role(r)["repo_id"], "source_revision": revisions[r],
                   "runtime_revision": cfg.role(r)["archived_model_revision"]} for r in parents}
    mb = cfg.raw.get("merge_build") or {}
    sha_path = cfg.work("audit", "parent_weight_sha256.json")
    parent_sha = json.loads(sha_path.read_text(encoding="utf-8")) if sha_path.is_file() else None
    out = {}
    for mid, spec in cfg.merges.items():
        if only and mid not in only:
            continue
        coeffs = {k: float(v) for k, v in spec["coefficients"].items()}
        out[mid] = build_merge(mid, coeffs, parents, cfg.work("merges"), layout_role=mb.get("shard_layout", "B"),
                               source_revisions=revisions, lineage=lineage,
                               max_slice_elements=int(mb.get("max_slice_elements", 1 << 25)),
                               parent_sha256={r: parent_sha[r] for r in coeffs} if parent_sha else None,
                               log=log, dry_run=dry_run)
    return out


def parent_update_statistics(parents: dict[str, Path], base_role: str = "B", max_slice_elements: int = 1 << 25,
                             log: Callable[[str], None] = print) -> dict[str, Any]:
    """Diagnostic only: per tensor group, ||P - B||, ||B||, cos(T - B, V - B) and how many tensors are bitwise equal
    to B. These norms do not prove independent skills; they show which components each recipe changed at all."""

    idx = {r: index_checkpoint(p) for r, p in parents.items()}
    others = [r for r in parents if r != base_role]
    groups: dict[str, dict[str, float]] = {}
    per_tensor_equal: dict[str, list[str]] = {r: [] for r in others}
    for n in sorted(idx[base_role]):
        g = tensor_group(n)
        acc = groups.setdefault(g, {"tensors": 0, "base_sq": 0.0, **{f"{r}_delta_sq": 0.0 for r in others},
                                    **{f"{r}_equal_tensors": 0 for r in others}, "dot": 0.0})
        acc["tensors"] += 1
        eq = {r: True for r in others}
        for r0, r1 in iter_row_slices(idx[base_role][n], max_slice_elements):
            b = read_rows(idx[base_role][n], r0, r1)
            bf = b.to(torch.float64)
            acc["base_sq"] += float((bf * bf).sum())
            deltas = {}
            for r in others:
                x = read_rows(idx[r][n], r0, r1)
                if eq[r] and not torch.equal(x.view(torch.int16), b.view(torch.int16)):
                    eq[r] = False
                d = x.to(torch.float64) - bf
                deltas[r] = d
                acc[f"{r}_delta_sq"] += float((d * d).sum())
            if len(others) == 2:
                acc["dot"] += float((deltas[others[0]] * deltas[others[1]]).sum())
        for r in others:
            if eq[r]:
                acc[f"{r}_equal_tensors"] += 1
                per_tensor_equal[r].append(n)
    out_groups = {}
    for g, a in groups.items():
        row = {"tensors": a["tensors"], "base_norm": a["base_sq"] ** 0.5}
        for r in others:
            row[f"{r}_delta_norm"] = a[f"{r}_delta_sq"] ** 0.5
            row[f"{r}_relative_update"] = (a[f"{r}_delta_sq"] ** 0.5) / max(a["base_sq"] ** 0.5, 1e-30)
            row[f"{r}_tensors_bitwise_equal_to_{base_role}"] = a[f"{r}_equal_tensors"]
        if len(others) == 2:
            na, nb = a[f"{others[0]}_delta_sq"] ** 0.5, a[f"{others[1]}_delta_sq"] ** 0.5
            row[f"cos_{others[0]}_{others[1]}_updates"] = a["dot"] / (na * nb) if na > 0 and nb > 0 else None
        out_groups[g] = row
        log(f"[update-stats] {g}: {json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})}")
    return {"schema": "trace-merge-update-stats-v1", "base_role": base_role, "groups": out_groups,
            "bitwise_equal_tensor_counts": {r: len(v) for r, v in per_tensor_equal.items()},
            "note": "diagnostic only; update norms and cosines do not establish independent skills"}
