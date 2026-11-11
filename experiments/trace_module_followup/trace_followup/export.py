"""Evaluation manifest and the sanitized public bundle (design/followup_v1.md §8; guide "Required output bundle").

``evaluation_manifest`` writes ``manifests/evaluation.json``: every checkpoint x interface x subset x seed run with its
served revision, view, attempts (host, GPUs, rc), runner code hashes and protocol code/evaluator hashes, plus the
stack, processor, decoding and judge pins. ``export_bundle`` copies aggregate files only into
``experiments/trace_module_followup/results/`` with the reviewed study's sanitizer: JSON has whole machine paths
replaced by ``<local>``, any other text must already be clean, and nothing is written unless every file passes.
Row-level files (row-change ledgers, blinded-check samples, aligned rows, run directories) stay in the work root.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .common import PACKAGE_ROOT, FollowupConfig, sha256_bytes, sha256_file, write_json

from trace_merge.reports import ABS_PATH, sanitization_problems  # noqa: E402

PROVENANCE = re.compile(r"provenance: run_tag (\S+) code (\w+) evaluator (\w+) dataset (\w+) campaign (\w+)")
# work-root files of the public bundle, as globs relative to the work root (aggregates only)
BUNDLE_GLOBS = ("audit/component_map.json", "audit/compatibility.json", "audit/build_verification*.json",
                "audit/parent_weight_sha256.json", "audit/scoring_spec.*.json", "audit/scoring_changes.md",
                "audit/tablevqa_diagnosis.md", "audit/f3_update_norms.json", "hybrids/*/manifest.json",
                "hybrids/*/recipe.json", "scaled/*/merge_manifest.json", "scaled/*/recipe.json", "manifests/*.json",
                "tables/*.csv", "tables/*.json", "reports/*.md", "figures/*.png", "dispatch/protocol_overlay/*.diff")


def _run_records(cfg: FollowupConfig) -> list[dict[str, Any]]:
    out = []
    for rd in sorted(p for p in cfg.work("runs").glob("*/*") if p.is_dir()):
        slug, subset = rd.parent.name, rd.name
        for plan in sorted(rd.glob("plan.seed*.json")):
            seed = int(plan.name.split("seed")[1].split(".")[0])
            val = rd / f"validation.seed{seed}.json"
            summ = rd / f"summary.seed{seed}.json"
            timing = rd / f"timing.seed{seed}.json"
            rec: dict[str, Any] = {
                "slug": slug, "subset": subset, "seed": seed,
                "complete": bool(json.loads(val.read_text()).get("complete")) if val.is_file() else False,
                "row_list_sha256": json.loads(plan.read_text()).get("row_list_sha256"),
            }
            if summ.is_file():
                s = json.loads(summ.read_text())
                rec["served_model_revision"] = s.get("model_revision")
                rec["benchmarks_scored"] = s.get("benchmarks_scored")
            vm = cfg.work("views", slug, "view_manifest.json")
            if vm.is_file():
                v = json.loads(vm.read_text())
                rec.update({"label": v["label"], "protocol_id": v["protocol_id"], "weight_source": v["weight_source"],
                            "weight_source_revision": v["weight_source_revision"],
                            "view_manifest_sha256": sha256_file(vm)})
            if timing.is_file():
                rec["attempts"] = [{"host": a.get("host"), "gpus": a.get("gpus"), "rc": a.get("rc"),
                                    "jobs_sha256": a.get("jobs_sha256"),
                                    "serving": {k: v for k, v in a.items() if k.startswith(("gen_", "judge_"))}}
                                   for a in json.loads(timing.read_text()).get("attempts", [])]
            prov = set()
            for log in sorted((rd / "logs").glob(f"eval_subset.seed{seed}.*.log")):
                for m in PROVENANCE.finditer(log.read_text(errors="replace")):
                    prov.add((m.group(2), m.group(3)))
            rec["protocol_code_evaluator_hashes"] = sorted(f"code {c} evaluator {e}" for c, e in prov)
            quarantine = sorted(q.name for q in rd.glob("quarantine_*") if q.is_dir())
            if quarantine:
                rec["quarantined_outputs"] = quarantine
            out.append(rec)
    return out


def evaluation_manifest(cfg: FollowupConfig) -> Path:
    base = cfg.base.raw
    manifest = {
        "schema": "trace-followup-evaluation-manifest-v1",
        "study_id": cfg.raw["study_id"],
        "config_sha256": cfg.sha256,
        "pins": {"protocol": base["protocol"], "model_roles": base["model_roles"],
                 "interfaces": {k: v for k, v in base["protocols"].items() if k in ("P1_backbone", "P2_vero")},
                 "evaluation_tiers": cfg.raw["evaluation"],
                 "split_manifest_sha256": sha256_file(cfg.work("splits", "split_manifest.json")),
                 "scorer": "official adapters (reviewed metric registry) + format-normalized-v2 "
                           "(audit/scoring_spec.format-normalized-v2.json)",
                 "protocol_overlay": "two P2 screen rescores (tf-p2, vl-p2) used the private overlay with fix-4/fix-5 "
                                     "(design A1); every other run used the shared clone unchanged"},
        "coverage": json.loads(cfg.work("manifests", "coverage.json").read_text()),
        "runs": _run_records(cfg),
    }
    path = cfg.work("manifests", "evaluation.json")
    write_json(path, manifest)
    return path


def bundle_files(cfg: FollowupConfig) -> dict[str, Path]:
    root = cfg.require_work_root()
    files = {}
    for pattern in BUNDLE_GLOBS:
        for src in sorted(root.glob(pattern)):
            if src.is_file():
                files[src.relative_to(root).as_posix()] = src
    return files


def export_bundle(cfg: FollowupConfig, dest: Path | None = None) -> dict[str, str]:
    dest = dest or PACKAGE_ROOT / "results"
    staged: dict[str, bytes] = {}
    bad = []
    for rel, src in bundle_files(cfg).items():
        if src.suffix == ".png":                      # figures: aggregate plots, copied as bytes
            staged[rel] = src.read_bytes()
            continue
        text = src.read_text(encoding="utf-8")
        text = ABS_PATH.sub("<local>", text) if src.suffix == ".json" else text
        problems = sanitization_problems(text)
        if problems:
            bad.append(f"{rel}: {' and '.join(problems)}")
        staged[rel] = text.encode("utf-8")
    if bad:
        raise RuntimeError("export refused, nothing written: " + "; ".join(bad))
    hashes = {}
    for rel, data in staged.items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[rel] = sha256_bytes(data)
    write_json(dest / "MANIFEST.json", {"files": hashes, "study_id": cfg.raw["study_id"],
                                        "note": "aggregates only; written by trace_followup.export.export_bundle"})
    return hashes
