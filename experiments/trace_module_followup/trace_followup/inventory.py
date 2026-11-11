"""Parent inventory and pre-build checks (design/followup_v1.md, construction steps 1-2; required tests).

``inventory`` resolves the three pinned parent snapshots, hashes every weight file and every stored tensor in one
sequential pass per shard, compares the file hashes with the reviewed study's recorded parent hashes, reruns the
architecture / token identity / tensor name-shape-dtype / effective processor audit, and writes the component map.
``identity_checks`` replays the builder's copy path with every group assigned to one parent and requires the copied
tensor bytes to equal that parent tensor for tensor. ``ownership_checks`` requires TF/TL (and VF/VL) to partition their
parent's changes relative to B.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from .build import copy_tensor_exact, scan_checkpoint
from .common import ROLES, FollowupConfig, FollowupError, canonical_json, sha256_bytes, write_json
from .components import component_map, donor_map

from trace_merge.compatibility import audit, require_compatible  # noqa: E402
from trace_merge.merge import file_identity  # noqa: E402
from trace_merge.safetensors_io import index_checkpoint  # noqa: E402

INVENTORY = "parent_inventory.json"


def _scan(path: str) -> dict[str, Any]:
    return scan_checkpoint(Path(path))


def _replay(path: str, names: list[str]) -> dict[str, str]:
    idx = index_checkpoint(Path(path))
    return {n: copy_tensor_exact(idx[n], lambda _b: None) for n in names}


def run_inventory(cfg: FollowupConfig, reference_hashes: Path | None = None) -> dict[str, Any]:
    parents = {r: cfg.parent_dir(r) for r in ROLES}
    for r, p in parents.items():
        if not p.is_dir():
            raise FollowupError(f"{r}: pinned snapshot missing at {p}")
    lineage = {r: {"repo_id": cfg.base.role(r)["repo_id"], "source_revision": cfg.revision(r),
                   "runtime_revision": cfg.base.role(r)["archived_model_revision"]} for r in ROLES}
    report = audit(parents, lineage)
    require_compatible(report)
    write_json(cfg.work("audit", "compatibility.json"), report)
    identities = {r: {f.name: file_identity(f) for f in sorted(p.glob("*.safetensors"))} for r, p in parents.items()}
    with ProcessPoolExecutor(max_workers=3) as ex:
        scans = dict(zip(ROLES, ex.map(_scan, [str(parents[r]) for r in ROLES])))
    for r in ROLES:
        bad = {n: v["nonfinite"] for n, v in scans[r]["tensors"].items() if v["nonfinite"]}
        if bad:
            raise FollowupError(f"{r}: non-finite stored values in {list(bad)[:3]}")
    names = sorted(scans["B"]["tensors"])
    for r in ROLES:
        if sorted(scans[r]["tensors"]) != names:
            raise FollowupError(f"{r}: stored tensor names differ from B")
    reference = None
    if reference_hashes is not None:
        reference = json.loads(reference_hashes.read_text(encoding="utf-8"))
        for r in ROLES:
            if reference.get(r) != scans[r]["files"]:
                raise FollowupError(f"{r}: weight-file sha256 differ from the reviewed study's record {reference_hashes}")
    numels = {n: scans["B"]["tensors"][n]["numel"] for n in names}
    cmap = component_map(cfg, names, numels)
    write_json(cfg.work("audit", "component_map.json"), cmap)
    core = {r: {"files": scans[r]["files"], "tensors": {n: scans[r]["tensors"][n]["sha256"] for n in names}}
            for r in ROLES}
    inv = {
        "schema": "trace-followup-parent-inventory-v1",
        "revisions": {r: cfg.revision(r) for r in ROLES},
        "inventory_sha256": sha256_bytes(canonical_json(core).encode("utf-8")),
        "reference_hashes_matched": reference is not None,
        "compatibility_status": report["status"],
        "component_counts": cmap["counts"],
        "parents": {r: {"dir_name": parents[r].name, "files": scans[r]["files"], "file_identities": identities[r],
                        "tensors": {n: {"sha256": scans[r]["tensors"][n]["sha256"]} for n in names},
                        "tensors_differing_from_B": sum(1 for n in names if core[r]["tensors"][n] != core["B"]["tensors"][n])}
                    for r in ROLES},
    }
    write_json(cfg.work("audit", INVENTORY), inv)
    write_json(cfg.work("audit", "parent_weight_sha256.json"), {r: scans[r]["files"] for r in ROLES})
    return inv


def load_inventory(cfg: FollowupConfig) -> dict[str, Any]:
    path = cfg.work("audit", INVENTORY)
    if not path.is_file():
        raise FollowupError(f"{path} is missing; run inventory first")
    return json.loads(path.read_text(encoding="utf-8"))


def identity_checks(cfg: FollowupConfig, inv: dict[str, Any]) -> dict[str, Any]:
    """Assigning one parent to every group reconstructs that parent tensor for tensor (through the builder's copy)."""

    names = sorted(inv["parents"]["B"]["tensors"])
    with ProcessPoolExecutor(max_workers=3) as ex:
        replays = dict(zip(ROLES, ex.map(_replay, [str(cfg.parent_dir(r)) for r in ROLES], [names] * 3)))
    out = {}
    for r in ROLES:
        bad = [n for n in names if replays[r][n] != inv["parents"][r]["tensors"][n]["sha256"]]
        if bad:
            raise FollowupError(f"identity {r}: {len(bad)} tensors differ, e.g. {bad[:3]}")
        out[r] = {"tensors": len(names), "equal": True}
    return out


def ownership_checks(cfg: FollowupConfig, inv: dict[str, Any]) -> dict[str, Any]:
    """Every key has exactly one owner; for each trained parent P the tensors in which PF and PL differ from B are
    disjoint and their union is exactly the set in which P differs from B."""

    names = sorted(inv["parents"]["B"]["tensors"])
    h = {r: {n: inv["parents"][r]["tensors"][n]["sha256"] for n in names} for r in ROLES}
    out: dict[str, Any] = {}
    for parent, (f_id, l_id) in {"T": ("TF", "TL"), "V": ("VF", "VL")}.items():
        changed = {n for n in names if h[parent][n] != h["B"][n]}
        sets = {}
        for hid in (f_id, l_id):
            dm = donor_map(cfg, hid, names)
            sets[hid] = {n for n in names if h[dm[n]][n] != h["B"][n]}
        if sets[f_id] & sets[l_id]:
            raise FollowupError(f"{f_id}/{l_id}: change sets overlap")
        if sets[f_id] | sets[l_id] != changed:
            raise FollowupError(f"{f_id}/{l_id}: change sets do not partition {parent}'s changes")
        out[parent] = {"changed_vs_B": len(changed), f_id: len(sets[f_id]), l_id: len(sets[l_id]), "partition": True}
    return out
