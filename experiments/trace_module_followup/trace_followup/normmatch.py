"""F3 (design/decision_F3.md): one norm-matched VERO checkpoint, V-NM = θ_B + λ (θ_V − θ_B), λ = ‖Δ_T‖_F / ‖Δ_V‖_F.

``update_norms`` recomputes the parents' update norms in float64 over every stored tensor (the reviewed
``parent_update_statistics``) and derives λ at full precision. ``build_vnm`` builds ``(1 − λ) B + λ V`` with the
reviewed streaming interpolator (float32 accumulation, one BF16 cast, B's layout, atomic finalize, parent hashes
checked) under its own recipe id in ``<work root>/scaled/``. ``realized`` measures what serialization produced: global
and per-group ‖V-NM − B‖, the residual against λ‖Δ_V‖, the fraction of stored values bitwise equal to B, and finiteness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch

from .common import ROLES, FollowupConfig, FollowupError, write_json

from trace_merge.compatibility import tensor_group  # noqa: E402
from trace_merge.merge import build_merge, parent_update_statistics  # noqa: E402
from trace_merge.safetensors_io import index_checkpoint, iter_row_slices, read_rows  # noqa: E402

VNM = "VNM"
NORMS = "f3_update_norms.json"
VERIFICATION = "build_verification_F3.json"


def update_norms(cfg: FollowupConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    parents = {r: cfg.parent_dir(r) for r in ROLES}
    stats = parent_update_statistics(parents, base_role="B", log=log)
    groups = stats["groups"]
    total = {r: sum(g[f"{r}_delta_norm"] ** 2 for g in groups.values()) ** 0.5 for r in ("T", "V")}
    base = sum(g["base_norm"] ** 2 for g in groups.values()) ** 0.5
    out = {"schema": "trace-followup-f3-norms-v1", "dtype": "float64 over every stored BF16 tensor",
           "tensors": sum(g["tensors"] for g in groups.values()), "global_delta_norm": total, "base_norm": base,
           "relative_update": {r: total[r] / base for r in total},
           "lambda": total["T"] / total["V"], "groups": groups,
           "reference_rounded": {"T": 1.994, "V": 8.800, "lambda": 0.227}}
    write_json(cfg.work("audit", NORMS), out)
    return out


def build_vnm(cfg: FollowupConfig, lam: float, log: Callable[[str], None] = print) -> dict[str, Any]:
    if not 0.0 < lam < 1.0:
        raise FollowupError(f"lambda {lam} outside (0, 1)")
    parents = {r: cfg.parent_dir(r) for r in ("B", "V")}
    recorded = json.loads(cfg.work("audit", "parent_weight_sha256.json").read_text(encoding="utf-8"))
    lineage = {r: {"repo_id": cfg.base.role(r)["repo_id"], "source_revision": cfg.revision(r),
                   "runtime_revision": cfg.base.role(r)["archived_model_revision"]} for r in parents}
    out_root = cfg.work("scaled")
    out_root.mkdir(parents=True, exist_ok=True)
    return build_merge(VNM, {"B": 1.0 - lam, "V": lam}, parents, out_root, layout_role="B",
                       source_revisions={r: cfg.revision(r) for r in parents}, lineage=lineage,
                       parent_sha256={r: recorded[r] for r in parents}, log=log)


def realized(cfg: FollowupConfig, lam: float, norms: dict[str, Any], max_slice_elements: int = 1 << 25) -> dict[str, Any]:
    b_idx = index_checkpoint(cfg.parent_dir("B"))
    m_dir = cfg.work("scaled", VNM)
    m_idx = index_checkpoint(m_dir)
    if sorted(b_idx) != sorted(m_idx):
        raise FollowupError("V-NM tensors differ from B's")
    groups: dict[str, dict[str, float]] = {}
    equal = values = nonfinite = 0
    for n in sorted(b_idx):
        g = groups.setdefault(tensor_group(n), {"delta_sq": 0.0, "values": 0, "equal": 0})
        for r0, r1 in iter_row_slices(b_idx[n], max_slice_elements):
            b, m = read_rows(b_idx[n], r0, r1), read_rows(m_idx[n], r0, r1)
            d = m.to(torch.float64) - b.to(torch.float64)
            g["delta_sq"] += float((d * d).sum())
            eq = int((m.view(torch.int16) == b.view(torch.int16)).sum())
            g["equal"] += eq
            g["values"] += b.numel()
            nonfinite += int((~torch.isfinite(m.to(torch.float32))).sum())
    for g in groups.values():
        equal += g["equal"]
        values += g["values"]
    realized_norm = sum(g["delta_sq"] for g in groups.values()) ** 0.5
    target = lam * norms["global_delta_norm"]["V"]
    manifest = json.loads((m_dir / "merge_manifest.json").read_text(encoding="utf-8"))
    out = {"VNM": {
        "lambda": lam, "target_norm": target, "realized_norm": realized_norm, "residual": realized_norm - target,
        "relative_residual": (realized_norm - target) / target, "trace_norm": norms["global_delta_norm"]["T"],
        "realized_over_trace_norm": realized_norm / norms["global_delta_norm"]["T"],
        "fraction_values_bitwise_equal_to_B": equal / values, "nonfinite_values": nonfinite,
        "groups": {k: {"delta_norm": v["delta_sq"] ** 0.5, "fraction_equal_to_B": v["equal"] / v["values"]}
                   for k, v in sorted(groups.items())},
        "verified": nonfinite == 0, "content_revision": manifest["content_revision"]}}
    if nonfinite:
        raise FollowupError(f"V-NM holds {nonfinite} non-finite values")
    write_json(cfg.work("audit", VERIFICATION), out)
    return out


def run_f3_build(cfg: FollowupConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    norms_path = cfg.work("audit", NORMS)
    norms = json.loads(norms_path.read_text()) if norms_path.is_file() else update_norms(cfg, log=log)
    lam = float(norms["lambda"])
    log(f"[f3] lambda = {lam!r} (||dT|| {norms['global_delta_norm']['T']:.6f}, ||dV|| {norms['global_delta_norm']['V']:.6f})")
    manifest = build_vnm(cfg, lam, log=log)
    log(f"[f3] built {manifest['content_revision'][:24]}")
    return realized(cfg, lam, norms)


def scaled_manifest_path(cfg: FollowupConfig, source: str) -> Path:
    return cfg.work("scaled", source, "merge_manifest.json")
