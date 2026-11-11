"""Inventory of the study's inputs: archive receipts and files, frozen release inputs, dataset manifest, model snapshots.

Why: Stage A may not read a single archived row before the archive's bytes are proven to be the ones the release
receipts pin, and no merge may be built before the parent snapshots are proven to be the pinned revisions. This module
checks, without network access:

* the archive manifest sha256 against the config and against ``release_receipts.v1.json``; the results file pinned by
  ``result_sha256`` (it is ``metadata/results/benchmark_scores.json``); every listed part (parquet and part manifest)
  by sha256 and size; the exact expected artifact grid (config x model x seed x benchmark); no unlisted files;
* the frozen release inputs of this checkout (suite, results, provenance, ...) against the receipts' canonical files;
* the prepared dataset manifest's self-consistency (per-dataset and whole-snapshot digests recompute);
* each role's local HF snapshot: file sizes, HF blob names (for LFS files the blob name is the sha256, so the 16 GB of
  weights are not re-hashed unless asked), sha256 of every small file, and the runtime views the archive used (T's
  archived ``sha256set:`` revision is recomputed from the file set that T's own repository marker records).

Outputs: ``manifests/input_manifest.json`` and ``manifests/model_roles.json``. Paths are recorded relative to their
roots (archive run root, HF hub, LMUData), never as machine paths.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pyarrow.parquet as pq

from .config import ROLES, StudyConfig, sha256_file, write_json
from .ingest import (ARCHIVE_LABELS, MEDIA_CONTRACT, MODEL_MARKER, NATIVE_PROTOCOL_SLUGS, IngestError,
                     content_revision, protocol_code_provenance, protocol_module, valid_protocols_for)

Log = Callable[[str], None]
SMALL_FILE_BYTES = 64 * 1024 * 1024
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
# served slugs of the prepared protocol clone (rlvr/evaluation/.work/models/<slug>); V is not prepared there
PROTOCOL_MODEL_SLUGS = NATIVE_PROTOCOL_SLUGS


class InventoryError(RuntimeError):
    """An input does not match its receipt, or the archive file set is not exactly the manifest's."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------------- receipts
def _receipts(cfg: StudyConfig) -> dict[str, Any]:
    return json.loads(cfg.repo_path(cfg.raw["archive_receipts"]).read_text(encoding="utf-8"))


def archive_receipt(cfg: StudyConfig) -> dict[str, Any]:
    arch = cfg.raw["archive"]
    hits = [r for r in _receipts(cfg).get("hugging_face_source_archives") or []
            if r.get("repository_id") == arch["repository_id"] and r.get("revision") == arch["revision"]
            and r.get("run_id") == arch["run_id"]]
    if len(hits) != 1:
        raise InventoryError(f"release receipts list {len(hits)} entries for {arch['run_id']}@{arch['revision']}")
    return hits[0]


def verify_release_inputs(cfg: StudyConfig) -> dict[str, Any]:
    """This checkout's frozen release files against ``release_receipts.v1.json`` canonical_files."""

    rows = []
    problems = []
    for entry in _receipts(cfg).get("canonical_files") or []:
        path = cfg.repo_path(entry["path"])
        actual = sha256_file(path) if path.is_file() else None
        ok = actual == entry["sha256"]
        rows.append({"path": entry["path"], "sha256": actual, "expected": entry["sha256"], "ok": ok})
        if not ok:
            problems.append(f"{entry['path']}: {actual} != {entry['sha256']}")
    return {"files": rows, "problems": problems, "status": "verified" if not problems else "failed"}


# --------------------------------------------------------------------------------------------- archive
def verify_archive(cfg: StudyConfig, *, strict_release_files: bool = False, benchmarks: Sequence[str] | None = None,
                   receipt: Mapping[str, Any] | None = None, log: Log = print) -> dict[str, Any]:
    """Verify the local archive snapshot file-by-file against its manifest and the release receipts.

    ``benchmarks`` (default: the suite) defines the expected artifact grid; ``receipt`` overrides the entry looked up in
    ``release_receipts.v1.json`` (synthetic fixtures only).
    """

    arch = cfg.raw["archive"]
    root = cfg.archive_run_root
    problems: list[str] = []
    warnings: list[str] = []
    mpath = root / "metadata" / "manifest.json"
    if not mpath.is_file():
        raise InventoryError("archive manifest.json not found under the archive run root")
    manifest_sha = sha256_file(mpath)
    receipt = dict(receipt) if receipt is not None else archive_receipt(cfg)
    if manifest_sha != arch["manifest_sha256"]:
        problems.append(f"manifest.json sha256 {manifest_sha} != config {arch['manifest_sha256']}")
    if receipt["manifest_sha256"] != arch["manifest_sha256"] or receipt["result_sha256"] != arch["result_sha256"]:
        problems.append("config archive digests differ from release_receipts.v1.json")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("run_ids") != [arch["run_id"]]:
        problems.append(f"manifest run_ids {manifest.get('run_ids')} != [{arch['run_id']}]")

    listed: set[str] = {"metadata/manifest.json"}
    verified_bytes = 0

    def _check_file(rel: str, sha: str, size: int | None, what: str) -> bool:
        nonlocal verified_bytes
        listed.add(rel)
        path = root / rel
        if not path.is_file():
            problems.append(f"missing {what}: {rel}")
            return False
        actual_size = path.stat().st_size
        if size is not None and actual_size != int(size):
            problems.append(f"size mismatch {what}: {rel} ({actual_size} != {size})")
            return False
        if sha256_file(path) != sha:
            problems.append(f"sha256 mismatch {what}: {rel}")
            return False
        verified_bytes += actual_size
        return True

    # results file pinned by result_sha256
    results = [m for m in manifest.get("metadata_files") or [] if m.get("sha256") == arch["result_sha256"]]
    result_rel = results[0]["path"] if len(results) == 1 else None
    if result_rel is None:
        problems.append(f"result_sha256 matches {len(results)} metadata files")
    for meta in manifest.get("metadata_files") or []:
        _check_file(str(meta["path"]), str(meta["sha256"]), meta.get("size"), "metadata file")
    release_files = []
    for rel in manifest.get("release_files") or []:
        path = root / str(rel["path"])
        present = path.is_file()
        ok = present and sha256_file(path) == rel["sha256"]
        release_files.append({"path": rel["path"], "present": present, "ok": ok})
        if not present:
            (problems if strict_release_files else warnings).append(
                f"release file {rel['path']} (repository-level documentation) is not in the local snapshot")
        elif not ok:
            problems.append(f"release file {rel['path']} does not match its sha256")
        listed.add(str(rel["path"]))

    # artifacts: exact grid, digests, part manifests
    model_ids = [cfg.role(r)["archived_model_id"] for r in ROLES]
    seeds = [int(s) for s in arch["seeds"]]
    benchmarks = list(benchmarks or cfg.benchmark_keys())
    config_meta = {c["config_name"]: c for c in manifest.get("configs") or []}
    expected_grid = {(c, m, s, b) for c in ("responses", "extractions", "scores") for m in model_ids for s in seeds
                     for b in benchmarks}
    seen: Counter[tuple[str, str, int, str]] = Counter()
    rows_by_config: Counter[str] = Counter()
    parts_by_config: Counter[str] = Counter()
    for entry in manifest.get("artifacts") or []:
        key = (str(entry["config_name"]), str(entry["model_id"]), int(entry["seed"]), str(entry["benchmark_id"]))
        seen[key] += 1
        rows_by_config[key[0]] += int(entry["rows"])
        parts_by_config[key[0]] += 1
        ok_p = _check_file(str(entry["parquet_path"]), str(entry["parquet_sha256"]), entry.get("parquet_size"),
                           "parquet part")
        ok_m = _check_file(str(entry["manifest_path"]), str(entry["manifest_sha256"]), entry.get("manifest_size"),
                           "part manifest")
        expected_dir = (f"run={arch['run_id']}/model={key[1]}/seed={key[2]}/benchmark={key[3]}/")
        if expected_dir not in str(entry["parquet_path"]):
            problems.append(f"part path does not encode its identity: {entry['parquet_path']}")
        if ok_m:
            part = json.loads((root / str(entry["manifest_path"])).read_text(encoding="utf-8"))
            ident = part.get("identity") or {}
            if (part.get("config_name") != key[0] or part.get("parquet_sha256") != entry["parquet_sha256"]
                    or part.get("parquet_path") != entry["parquet_path"]
                    or int(part.get("parquet_size", -1)) != int(entry["parquet_size"])
                    or int(part.get("rows", -1)) != int(entry["rows"]) or ident.get("model_id") != key[1]
                    or int(ident.get("seed", -1)) != key[2] or ident.get("benchmark_id") != key[3]
                    or ident.get("run_id") != arch["run_id"]
                    or part.get("contract_version") != config_meta.get(key[0], {}).get("contract_version")):
                problems.append(f"part manifest disagrees with the archive manifest: {entry['manifest_path']}")
        if ok_p:
            n = pq.ParquetFile(root / str(entry["parquet_path"])).metadata.num_rows
            if n != int(entry["rows"]):
                problems.append(f"{entry['parquet_path']}: {n} rows != manifest {entry['rows']}")
    dupes = [k for k, v in seen.items() if v > 1]
    missing = sorted(expected_grid - set(seen))
    extra = sorted(set(seen) - expected_grid)
    if dupes:
        problems.append(f"{len(dupes)} artifact identities listed twice, e.g. {dupes[:2]}")
    if missing:
        problems.append(f"{len(missing)} expected artifacts missing from the manifest, e.g. {missing[:2]}")
    if extra:
        problems.append(f"{len(extra)} unexpected artifacts in the manifest, e.g. {extra[:2]}")
    for name, meta in config_meta.items():
        if int(meta.get("rows", -1)) != rows_by_config[name] or int(meta.get("parts", -1)) != parts_by_config[name]:
            problems.append(f"config {name}: manifest totals {meta.get('rows')}/{meta.get('parts')} != "
                            f"{rows_by_config[name]}/{parts_by_config[name]} summed over parts")

    # no unlisted files under data/ and metadata/
    unlisted = []
    for sub in ("data", "metadata"):
        for path in sorted((root / sub).rglob("*")):
            if path.is_file() and path.relative_to(root).as_posix() not in listed:
                unlisted.append(path.relative_to(root).as_posix())
    if unlisted:
        problems.append(f"{len(unlisted)} files are not listed in the manifest, e.g. {unlisted[:3]}")

    # model and run metadata agree with the config's role pins
    models_meta = {}
    for role in ROLES:
        spec = cfg.role(role)
        path = root / "metadata" / "models" / f"{spec['archived_model_id']}.json"
        meta = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        models_meta[role] = meta
        if (meta.get("model_revision") != spec["archived_model_revision"]
                or meta.get("repository_id") != spec["repo_id"]
                or meta.get("repository_revision") != spec["source_revision"]):
            problems.append(f"archive model metadata for {role} disagrees with the config pins")
    run_meta_path = root / "metadata" / "runs" / f"{arch['run_id']}.json"
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8")) if run_meta_path.is_file() else {}
    judge = run_meta.get("judge_model") or {}
    if (judge.get("model_id") != cfg.raw["protocol"]["judge"]["repo_id"]
            or judge.get("model_revision") != cfg.raw["protocol"]["judge"]["revision"]):
        problems.append("archive run metadata names a different judge than the config")
    gen = cfg.raw["protocol"]["generation"]
    for item in run_meta.get("generation_settings") or []:
        mismatch = {k: (item.get(k), v) for k, v in (("temperature", gen["temperature"]), ("top_p", gen["top_p"]),
                                                      ("top_k", gen["top_k"]), ("max_output_tokens", gen["max_tokens"]),
                                                      ("presence_penalty", gen["presence_penalty"]),
                                                      ("repetition_penalty", gen["repetition_penalty"]),
                                                      ("min_image_pixels", gen["min_image_pixels"]),
                                                      ("max_image_pixels", gen["max_image_pixels"]),
                                                      ("media_transport", gen["media_transport"]))
                    if item.get(k) != v}
        if mismatch:
            problems.append(f"archive generation settings for {item.get('model_id')}/{item.get('seed')} differ "
                            f"from the configured protocol: {mismatch}")
    report = {
        "repository_id": arch["repository_id"], "revision": arch["revision"], "run_id": arch["run_id"],
        "manifest_path": "metadata/manifest.json", "manifest_sha256": manifest_sha,
        "manifest_sha256_expected": arch["manifest_sha256"], "receipt": receipt,
        "result_file": result_rel, "result_sha256": arch["result_sha256"],
        "artifacts": len(manifest.get("artifacts") or []), "rows_by_config": dict(rows_by_config),
        "parts_by_config": dict(parts_by_config), "configs": list(config_meta.values()),
        "verified_files": len(listed) - 1, "verified_bytes": verified_bytes, "release_files": release_files,
        "source_selection_sha256": manifest.get("source_selection_sha256"),
        "source_slice_set_sha256": manifest.get("source_slice_set_sha256"),
        "model_metadata": models_meta, "judge": judge,
        "generation_settings": run_meta.get("generation_settings"), "warnings": warnings, "problems": problems,
        "status": "verified" if not problems else "failed",
    }
    log(f"[inventory] archive: {report['verified_files']} files, {verified_bytes} bytes, status {report['status']}")
    return report


# --------------------------------------------------------------------------------------------- dataset manifest
def inventory_dataset_manifest(cfg: StudyConfig, *, hash_metadata_files: bool = True) -> dict[str, Any]:
    """Summarize and self-check the prepared dataset manifest (licensed inputs stay local; only digests recorded).

    The licensed TSVs are hashed by default: metrics.py re-derives aggregate-only scores and WeMath groups from them,
    so a size match alone is not enough.
    """

    path = cfg.dataset_manifest
    if not path.is_file():
        raise InventoryError("prepared dataset manifest not found in the protocol clone")
    data = json.loads(path.read_bytes())
    mc = protocol_module(MEDIA_CONTRACT)
    suite_path = cfg.repo_path(cfg.raw["external_suite"])
    suite_sha = sha256_file(suite_path)
    problems = []
    lmu_root = Path(str(data.get("lmu_data_root", "")))
    suite_rows = {b["key"]: int(b["rows"]) for b in cfg.suite()["benchmarks"]}
    per = {}
    for key in cfg.benchmark_keys():
        rec = (data.get("datasets") or {}).get(key) or {}
        rows = rec.get("row_media") or []
        snap = mc.dataset_snapshot_sha256(rows) if rows else None
        files = []
        for m in rec.get("metadata_files") or []:
            fpath = Path(str(m.get("path", "")))
            rel = fpath.relative_to(lmu_root).as_posix() if lmu_root.parts and fpath.is_relative_to(lmu_root) \
                else fpath.name
            entry = {"path": rel, "sha256": m.get("sha256"), "size_bytes": m.get("size_bytes"),
                     "present": fpath.is_file()}
            if fpath.is_file():
                entry["size_ok"] = fpath.stat().st_size == int(m.get("size_bytes", -1))
                if hash_metadata_files:
                    entry["sha256_ok"] = sha256_file(fpath) == m.get("sha256")
                if not entry["size_ok"] or entry.get("sha256_ok") is False:
                    problems.append(f"{key}: metadata file {rel} does not match its recorded size/digest")
            else:
                problems.append(f"{key}: metadata file {rel} is missing")
            files.append(entry)
        if rec.get("status") != "ready" or len(rows) != suite_rows[key] or snap != rec.get("dataset_snapshot_sha256"):
            problems.append(f"{key}: status {rec.get('status')}, rows {len(rows)} (suite {suite_rows[key]}), "
                            f"snapshot recomputes {snap == rec.get('dataset_snapshot_sha256')}")
        per[key] = {"alias": rec.get("alias"), "rows": len(rows), "dataset_snapshot_sha256": rec.get(
            "dataset_snapshot_sha256"), "unique_media": rec.get("unique_media"), "metadata_files": files}
    recomputed = mc.manifest_snapshot_sha256(suite_sha256=data.get("suite_sha256"),
                                             vlmevalkit_commit=data.get("vlmevalkit_commit"),
                                             datasets=data.get("datasets") or {}, keys=cfg.benchmark_keys())
    if recomputed != data.get("dataset_snapshot_sha256"):
        problems.append("whole-manifest dataset_snapshot_sha256 does not recompute")
    if data.get("suite_sha256") != suite_sha:
        problems.append("dataset manifest was prepared for a different suite file")
    if data.get("vlmevalkit_commit") != cfg.raw["protocol"]["vlmevalkit_revision"]:
        problems.append("dataset manifest was prepared with a different VLMEvalKit commit")
    return {"path": "rlvr/evaluation/.work/LMUData/trace_eval_v1_dataset_manifest.json (protocol clone)",
            "sha256": sha256_file(path), "schema_version": data.get("schema_version"),
            "dataset_snapshot_sha256": data.get("dataset_snapshot_sha256"),
            "dataset_snapshot_recomputed": recomputed, "suite_sha256": data.get("suite_sha256"),
            "vlmevalkit_commit": data.get("vlmevalkit_commit"), "benchmarks": per,
            "metadata_files_hashed": hash_metadata_files, "problems": problems,
            "status": "verified" if not problems else "failed"}


# --------------------------------------------------------------------------------------------- model snapshots
def _hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def snapshot_inventory(snapshot: Path, *, hash_weights: bool = False) -> dict[str, Any]:
    """Files of one HF snapshot: size, blob name and kind, sha256 (LFS blob name, or computed for small files)."""

    files = []
    problems = []
    for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        rel = path.relative_to(snapshot).as_posix()
        # the HF blob name is the snapshot symlink's immediate target; the blob may itself point into a
        # deduplicated store under another name, so resolve() would report the wrong identity
        name = Path(os.readlink(path)).name if path.is_symlink() else path.name
        kind = "lfs_sha256" if HEX64.match(name) else ("git_blob_sha1" if HEX40.match(name) else "regular_file")
        size = path.stat().st_size
        entry: dict[str, Any] = {"name": rel, "size": size, "blob": name, "blob_kind": kind}
        if kind == "lfs_sha256" and not hash_weights and size > SMALL_FILE_BYTES:
            entry.update(sha256=name, sha256_source="lfs_blob_name")
        else:
            digest = sha256_file(path)
            entry.update(sha256=digest, sha256_source="computed")
            if kind == "lfs_sha256" and digest != name:
                problems.append(f"{rel}: sha256 {digest} != LFS blob name {name}")
        files.append(entry)
    weights = [f for f in files if f["name"].endswith(".safetensors")]
    return {"files": files, "weight_files": len(weights), "weight_bytes": sum(f["size"] for f in weights),
            "problems": problems}


def inventory_models(cfg: StudyConfig, *, hash_weights: bool = False, log: Log = print) -> dict[str, Any]:
    """Local snapshots of B/T/V, prepared protocol model markers, and the runtime views used by the archive."""

    hub = _hf_home() / "hub"
    weight_hashes_path = cfg.work("audit", "parent_weight_sha256.json") if cfg.work_root else None
    full_hashes = (json.loads(weight_hashes_path.read_text(encoding="utf-8"))
                   if weight_hashes_path is not None and weight_hashes_path.is_file() else {})
    out: dict[str, Any] = {}
    for role in ROLES:
        spec = cfg.role(role)
        snap = cfg.hf_snapshot(role)
        entry: dict[str, Any] = {
            "repo_id": spec["repo_id"], "source_revision": spec["source_revision"],
            "archived_model_id": spec["archived_model_id"],
            "archived_model_revision": spec["archived_model_revision"],
            "archive_label": ARCHIVE_LABELS[role],
            "archive_valid_protocol_ids": valid_protocols_for(role, "P0_native").split(","),
            "hf_snapshot": snap.relative_to(hub).as_posix() if snap.is_relative_to(hub) else snap.name,
            "available": snap.is_dir(), "problems": []}
        if not snap.is_dir():
            entry["problems"].append("local HF snapshot is missing")
            out[role] = entry
            continue
        inv = snapshot_inventory(snap, hash_weights=hash_weights)
        entry.update(files=inv["files"], weight_files=inv["weight_files"], weight_bytes=inv["weight_bytes"])
        entry["problems"].extend(inv["problems"])
        by_name = {f["name"]: f["sha256"] for f in inv["files"]}
        # cross-check against full weight hashes recorded by the parent-hashes command, when present
        recorded = full_hashes.get(role) or {}
        if recorded:
            bad = [n for n, h in recorded.items() if by_name.get(n) != h]
            entry["full_weight_hashes"] = {"files": len(recorded), "agree": not bad}
            if bad:
                entry["problems"].append(f"recorded full weight hashes disagree for {bad}")
        # runtime view used by the archive
        if spec["archived_model_revision"] == spec["source_revision"]:
            entry["archived_runtime_view"] = {"kind": "hf_commit", "revision": spec["source_revision"],
                                              "status": "identical_to_source_revision"}
        else:
            marker_path = snap / MODEL_MARKER
            view: dict[str, Any] = {"kind": "sha256set", "revision": spec["archived_model_revision"]}
            if marker_path.is_file():
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                listed = {str(k): str(v) for k, v in (marker.get("file_sha256") or {}).items()}
                recomputed = content_revision(listed)
                disagree = sorted(n for n, h in listed.items() if by_name.get(n) != h)
                view.update(marker_file=MODEL_MARKER, marker_revision=marker.get("immutable_revision"),
                            files=sorted(listed), recomputed=recomputed,
                            files_agree_with_snapshot=not disagree, disagreeing_files=disagree,
                            status=("reproduced_from_snapshot_files" if recomputed == spec["archived_model_revision"]
                                    and not disagree else "not_reproduced"))
                if view["status"] != "reproduced_from_snapshot_files":
                    entry["problems"].append("archived runtime view is not reproduced by the snapshot's files")
            else:
                view["status"] = "no_marker_in_snapshot"
                entry["problems"].append("archived runtime view cannot be checked (no marker in snapshot)")
            entry["archived_runtime_view"] = view
        # prepared model directory in the protocol clone (used by the existing local reruns)
        slug = PROTOCOL_MODEL_SLUGS.get(role)
        if slug and cfg.protocol_clone is not None:
            mpath = cfg.protocol_clone / "rlvr/evaluation/.work/models" / slug / MODEL_MARKER
            if mpath.is_file():
                marker = json.loads(mpath.read_text(encoding="utf-8"))
                listed = {str(k): str(v) for k, v in (marker.get("file_sha256") or {}).items()}
                disagree = sorted(n for n, h in listed.items() if by_name.get(n) != h)
                entry["protocol_prepared_view"] = {
                    "slug": slug, "resolved_commit": marker.get("resolved_commit"),
                    "content_revision": content_revision(listed), "files": len(listed),
                    "files_agree_with_snapshot": not disagree, "disagreeing_files": disagree}
                if marker.get("resolved_commit") != spec["source_revision"] or disagree:
                    entry["problems"].append("prepared protocol model directory is not the pinned snapshot")
        out[role] = entry
        log(f"[inventory] {role}: {len(inv['files'])} files, {inv['weight_bytes']} weight bytes, "
            f"problems {len(entry['problems'])}")
    return out


# --------------------------------------------------------------------------------------------- manifests
def build_input_manifest(cfg: StudyConfig, *, hash_weights: bool = False, hash_metadata_files: bool = True,
                         strict_release_files: bool = False, log: Log = print) -> dict[str, Path]:
    """Write manifests/input_manifest.json and manifests/model_roles.json; raise if any input fails verification."""

    archive = verify_archive(cfg, strict_release_files=strict_release_files, log=log)
    release = verify_release_inputs(cfg)
    dataset = inventory_dataset_manifest(cfg, hash_metadata_files=hash_metadata_files)
    models = inventory_models(cfg, hash_weights=hash_weights, log=log)
    try:
        code = protocol_code_provenance(cfg)
    except IngestError as error:
        code = {"problems": [str(error)]}
    problems = (archive["problems"] + release["problems"] + dataset["problems"]
                + [f"{r}: {p}" for r, m in models.items() for p in m["problems"]] + list(code.get("problems", [])))
    input_manifest = {
        "schema": "trace_merge_input_manifest_v1", "study_id": cfg.study_id, "config_sha256": cfg.sha256,
        "created_at": _now(), "archive": archive, "release_inputs": release, "dataset_manifest": dataset,
        "protocol_code": code,
        "protocol": {k: cfg.raw["protocol"][k] for k in ("code_revision", "vlmevalkit_revision", "judge",
                                                          "generation")},
        "rl_baselines_archive": "not used: holds no B/T/V run (DESIGN.md section 1)",
        "problems": problems, "status": "verified" if not problems else "failed"}
    roles = {
        "schema": "trace_merge_model_roles_v1", "study_id": cfg.study_id, "config_sha256": cfg.sha256,
        "created_at": _now(), "hash_weights": hash_weights, "roles": models,
        "merges": cfg.merges, "protocols": cfg.raw["protocols"],
        "labels": {"archive": {r: ARCHIVE_LABELS[r] for r in ROLES},
                   "evaluation_matrix": cfg.raw["evaluation_matrix"]},
        "archive_role_map": {cfg.role(r)["archived_model_id"]: r for r in ROLES}}
    p_in = cfg.work("manifests", "input_manifest.json")
    p_roles = cfg.work("manifests", "model_roles.json")
    write_json(p_in, input_manifest)
    write_json(p_roles, roles)
    log(f"[inventory] input manifest status {input_manifest['status']} ({len(problems)} problem(s))")
    if problems:
        raise InventoryError("input verification failed: " + " | ".join(problems[:8]))
    return {"input_manifest": p_in, "model_roles": p_roles}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from .config import ensure_work_tree, load_config

    p = argparse.ArgumentParser(prog="trace_merge.inventory", description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--work-root", type=Path, default=None)
    p.add_argument("--protocol-clone", type=Path, default=None)
    p.add_argument("--hash-weights", action="store_true")
    p.add_argument("--no-hash-metadata-files", action="store_true", help="size-check the licensed TSVs only")
    args = p.parse_args(argv)
    cfg = load_config(args.config, work_root=args.work_root, protocol_clone=args.protocol_clone)
    ensure_work_tree(cfg)
    try:
        build_input_manifest(cfg, hash_weights=args.hash_weights, hash_metadata_files=not args.no_hash_metadata_files)
    except InventoryError as error:
        print(f"FATAL: {error}")
        return 4
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
