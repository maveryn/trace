"""Evaluation planning and run status for the follow-up matrix (design/followup_v1.md, execution stages).

Task lines run the reviewed study's runner (``experiments/trace_merge/jobs/run_tiers.sh`` -> ``eval_subset.sh``) with
``TVM_WORK`` pointing at the follow-up work root, so the protocol, generation settings, judge and row lists are the
reviewed study's. One line per (view, seed): the long analysis runs parallelize across cards. This module writes task
files and the coverage matrix; it never starts GPU work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import FollowupConfig, write_json
from .views import view_specs


def tier_runs(cfg: FollowupConfig, tier: str, cells: list[str] | None = None) -> list[tuple[str, str, str, int]]:
    """(label, slug, subset, seed) for every run of a tier."""

    spec = cfg.raw["evaluation"][tier]
    specs = view_specs(cfg)
    out = []
    for cell in cells or list(cfg.cells):
        for short in spec["protocols"]:
            label = f"{cell}@{short.upper()}"
            for seed in spec["seeds"]:
                out.append((label, specs[label][0], spec["subset"], int(seed)))
    return out


def task_lines(cfg: FollowupConfig, tier: str, env_file: Path, jobs_dir: Path, cells: list[str] | None = None,
               with_smoke: bool = False) -> list[str]:
    """One line per (view, seed). ``with_smoke`` puts the view's own smoke run first on the same line (run_tiers.sh
    stops at the first tier that does not complete, so a broken view never reaches the larger tier)."""

    smoke = {slug: f"{subset}:{seed}" for _l, slug, subset, seed in tier_runs(cfg, "smoke", cells)} if with_smoke else {}
    return [f"source {env_file} && bash {jobs_dir}/run_tiers.sh {slug} "
            + (f"{smoke[slug]} " if slug in smoke and tier != "smoke" else "") + f"{subset}:{seed}"
            for _label, slug, subset, seed in tier_runs(cfg, tier, cells)]


def write_tasks(cfg: FollowupConfig, tier: str, env_file: Path, jobs_dir: Path, cells: list[str] | None = None,
                out: Path | None = None, with_smoke: bool = False) -> tuple[Path, list[str]]:
    lines = task_lines(cfg, tier, env_file, jobs_dir, cells, with_smoke)
    path = out or cfg.work("dispatch", f"{tier}.tasks")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    coverage = cfg.work("manifests", "coverage.json")
    cov = json.loads(coverage.read_text(encoding="utf-8")) if coverage.is_file() else {}
    key = tier if cells is None else f"{tier}[{','.join(cells)}]"   # an extension's cells never replace the F1 entry
    cov[key] = [{"label": l, "slug": s, "subset": sub, "seed": seed} for l, s, sub, seed in tier_runs(cfg, tier, cells)]
    write_json(coverage, cov)
    return path, lines


def run_status(cfg: FollowupConfig, slug: str, subset: str, seed: int) -> dict[str, Any]:
    root = cfg.work("runs", slug, subset)
    v = root / f"validation.seed{seed}.json"
    val = json.loads(v.read_text(encoding="utf-8")) if v.is_file() else {}
    return {"slug": slug, "subset": subset, "seed": seed, "complete": bool(val.get("complete")),
            "pending_judge": (root / f"PENDING_JUDGE.seed{seed}").is_file(),
            "generation_finalized": (root / f"validation.generation.seed{seed}.json").is_file()}


def status(cfg: FollowupConfig, tiers: list[str] | None = None) -> list[dict[str, Any]]:
    out = []
    for tier in tiers or list(cfg.raw["evaluation"]):
        for label, slug, subset, seed in tier_runs(cfg, tier):
            out.append({"tier": tier, "label": label, **run_status(cfg, slug, subset, seed)})
    return out
