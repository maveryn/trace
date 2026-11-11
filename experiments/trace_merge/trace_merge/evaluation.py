"""Evaluation planning, status and the pre-confirmation lock (handoff 6.3, 8.1, 10).

New generations run only through ``jobs/eval_subset.sh`` (the protocol's own generation, extraction and scoring
components) on held GPU allocations; this module writes the task lines for the cluster's queue runner, reports which
model x tier x seed runs are complete, and freezes every hash the confirmation analysis depends on before any
confirmation output exists. It never launches GPU work itself.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, StudyConfig, canonical_json, sha256_bytes, sha256_file, write_json
from .grouping import split_rule
from .views import VIEW_SPECS

TIER_SUBSET = {"smoke": "smoke", "development": "development_screen", "confirmation": "confirmation"}
LOCK_NAME = "selection_lock.json"
AMENDMENTS_HEADING = "## Amendments"
_AMENDMENT = re.compile(r"^\*\*(A\d+)\b", re.M)
# lock fields verify_lock checks by their own rule instead of by equality
_VERIFY_SEPARATELY = {"design_sha256", "design_amendments_at_lock", "runner_code_sha256"}


def tier_runs(cfg: StudyConfig, tier: str) -> list[tuple[str, str, str, int]]:
    """(label, slug, subset, seed) for every run of a tier in the configured matrix."""

    spec = cfg.raw["evaluation_matrix"][tier]
    subset = TIER_SUBSET[tier]
    out = []
    for label in spec["models"]:
        slug = VIEW_SPECS[label][0]
        for seed in spec["seeds"]:
            out.append((label, slug, subset, int(seed)))
    return out


def task_lines(cfg: StudyConfig, tier: str, env_file: Path, jobs_dir: Path, with_smoke: bool = False) -> list[str]:
    """One line per model (its seeds run in sequence on the same allocation, in seed order).

    ``with_smoke`` first runs the model's own runs of the smoke matrix (a model outside it gets none); run_tiers.sh
    stops at the first tier that does not complete, so a broken view never reaches the larger tier.
    """

    smoke: dict[str, list[str]] = {}
    if with_smoke and tier != "smoke":
        for _, slug, subset, seed in tier_runs(cfg, "smoke"):
            smoke.setdefault(slug, []).append(f"{subset}:{seed}")
    by_model: dict[str, list[str]] = {}
    for label, slug, subset, seed in tier_runs(cfg, tier):
        by_model.setdefault(slug, list(smoke.get(slug, [])))
        by_model[slug].append(f"{subset}:{seed}")
    return [f"source {env_file} && bash {jobs_dir}/run_tiers.sh {slug} {' '.join(specs)}"
            for slug, specs in by_model.items()]


def run_dir(cfg: StudyConfig, slug: str, subset: str) -> Path:
    return cfg.work("runs", slug, subset)


def run_status(cfg: StudyConfig, slug: str, subset: str, seed: int) -> dict[str, Any]:
    root = run_dir(cfg, slug, subset)
    st: dict[str, Any] = {"slug": slug, "subset": subset, "seed": seed, "dir_exists": root.is_dir()}
    val = root / f"validation.seed{seed}.json"
    gen = root / f"validation.generation.seed{seed}.json"
    if (root / f"PENDING_JUDGE.seed{seed}").is_file():
        st["state"] = "judge_pending"
    if val.is_file():
        v = json.loads(val.read_text(encoding="utf-8"))
        st.update({"scored_complete": bool(v.get("complete")), "rows": v.get("generated_total"),
                   "expected": v.get("expected_total"), "problems": v.get("problems", [])[:5]})
        st.setdefault("state", "complete" if v.get("complete") else "incomplete")
    elif gen.is_file():
        v = json.loads(gen.read_text(encoding="utf-8"))
        st.update({"generated_complete": bool(v.get("complete")), "rows": v.get("generated_total"),
                   "expected": v.get("expected_total")})
        st.setdefault("state", "generated" if v.get("complete") else "generating")
    else:
        st.setdefault("state", "not_started" if not root.is_dir() else "started")
    return st


def status(cfg: StudyConfig, tiers: list[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for tier in tiers or list(TIER_SUBSET):
        for label, slug, subset, seed in tier_runs(cfg, tier):
            rows.append({"tier": tier, "label": label, **run_status(cfg, slug, subset, seed)})
    return rows


def confirmation_outputs(cfg: StudyConfig) -> dict[str, str]:
    """Every run directory on the confirmation partition that holds any file (row list, plan, campaign, predictions,
    validation), for any slug, under ``<work>/runs`` and under the runner's ``TVM_RUNS`` override -> UTC time of its
    earliest file."""

    roots = [("", cfg.work("runs"))]
    extra = os.environ.get("TVM_RUNS")
    if extra and Path(extra).resolve() != roots[0][1].resolve():
        roots.append(("TVM_RUNS:", Path(extra)))
    subset = TIER_SUBSET["confirmation"]
    out: dict[str, str] = {}
    for tag, root in roots:
        if not root.is_dir():
            continue
        for run in sorted(root.glob(f"*/{subset}")):
            first = None
            for dirpath, _, files in os.walk(run):
                for name in files:
                    try:
                        t = os.lstat(os.path.join(dirpath, name)).st_mtime
                    except FileNotFoundError:  # a live run renames its temp files
                        continue
                    first = t if first is None else min(first, t)
            if first is not None:
                out[f"{tag}{run.parent.name}/{subset}"] = dt.datetime.fromtimestamp(first, dt.timezone.utc).isoformat(
                    timespec="seconds")
    return out


def design_amendments(text: str) -> list[tuple[str, int]]:
    """(id, offset) of every amendment paragraph (``**A<n>``) below the DESIGN.md Amendments heading."""

    head = text.find(AMENDMENTS_HEADING)
    return [] if head < 0 else [(m.group(1), m.start()) for m in _AMENDMENT.finditer(text, head)]


def amendments_after(text: str, locked_sha256: str) -> list[str] | None:
    """Amendments appended since the lock when the locked DESIGN.md is the current text cut before one of them (an
    append-only change); [] when unchanged; None when anything else changed (frozen body or an earlier amendment)."""

    if sha256_bytes(text.encode("utf-8")) == locked_sha256:
        return []
    amends = design_amendments(text)
    for i, (_, start) in enumerate(amends):
        head = text[:start]
        if locked_sha256 in {sha256_bytes(c.encode("utf-8")) for c in (head, head.rstrip("\n") + "\n")}:
            return [a for a, _ in amends[i:]]
    return None


def runner_code(jobs_dir: Path = PROJECT_ROOT / "jobs") -> dict[str, str]:
    """sha256 of every runner file the queue tasks execute (``jobs/``, caches excluded)."""

    return {p.relative_to(jobs_dir).as_posix(): sha256_file(p) for p in sorted(jobs_dir.rglob("*"))
            if p.is_file() and "__pycache__" not in p.relative_to(jobs_dir).parts}


def _lock_content(cfg: StudyConfig, design_path: Path) -> dict[str, Any]:
    """Every frozen field of the lock, from the current inputs; fails when a confirmation input is missing."""

    conf = tier_runs(cfg, "confirmation")
    split_manifest = cfg.work("splits", "split_manifest.json")
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    if split.get("rule") != split_rule(cfg.raw["split"]):
        raise RuntimeError("the config's split section no longer gives the frozen split manifest's rule")
    registry_path = cfg.work("manifests", "metric_registry.json")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    merges = {mid: json.loads((cfg.work("merges", mid) / "merge_manifest.json").read_text(encoding="utf-8"))
              for mid in cfg.merges}
    views = {}
    for label, slug, _, _ in conf:
        vm = cfg.work("views", slug, "view_manifest.json")
        marker = cfg.work("views", slug, ".trace_model_revision.json")
        revision = (json.loads(marker.read_text(encoding="utf-8")).get("immutable_revision")
                    if marker.is_file() else None)
        if not vm.is_file() or not revision:
            raise RuntimeError(f"{label}: view {slug} is not built and registered (view_manifest.json, "
                               ".trace_model_revision.json); lock only after every confirmation view exists")
        views[label] = {"view_manifest_sha256": sha256_file(vm), "registered_revision": revision}
    return {
        "schema": "trace-merge-selection-lock-v1",
        "study_id": cfg.study_id,
        "rule": cfg.raw["selection"]["rule"],
        "selected_candidates": sorted({label for label, *_ in conf}),
        "confirmation_runs": [{"label": l, "slug": s, "subset": sub, "seed": seed} for l, s, sub, seed in conf],
        "primary_family": list(cfg.raw["statistics"]["primary_family"]),
        "multiplicity": cfg.raw["statistics"]["multiplicity"],
        "category_regression_margin_pp": cfg.raw["selection"].get("category_regression_margin_pp"),
        "tie_tolerance_pp": cfg.raw["selection"].get("tie_tolerance_pp"),
        "config_sha256": cfg.sha256,
        "design_sha256": sha256_file(design_path),
        "design_amendments_at_lock": [a for a, _ in design_amendments(design_path.read_bytes().decode("utf-8"))],
        "split_manifest_sha256": sha256_file(split_manifest),
        "split_config_sha256": sha256_bytes(canonical_json(cfg.raw["split"]).encode("utf-8")),
        "confirmation_row_list_sha256": split["row_lists"]["confirmation"]["row_list_sha256"],
        "metric_registry_sha256": sha256_file(registry_path),
        "accepted_benchmarks": sorted(registry["accepted_benchmarks"]),
        "merge_content_revisions": {mid: m["content_revision"] for mid, m in merges.items()},
        "merge_recipe_sha256": {mid: m["recipe_sha256"] for mid, m in merges.items()},
        "views": views,
        "runner_code_sha256": runner_code(),
    }


def lock_selection(cfg: StudyConfig, design_path: Path, force: bool = False) -> dict[str, Any]:
    """Freeze candidates, comparisons and every input hash before any confirmation output exists.

    Refuses when any run directory on the confirmation partition already holds a file (any slug, either run root;
    ``force`` does not bypass this), when a confirmation view, the metric registry or the split manifest is missing or
    disagrees with the config, or when an existing lock disagrees with the current inputs. ``force`` replaces a
    disagreeing lock only before confirmation starts, and keeps the replaced one as ``selection_lock.<locked_at>.json``.
    """

    lock_path = cfg.work("reports", LOCK_NAME)
    lock = _lock_content(cfg, design_path)
    old = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.is_file() else None
    if old is not None:
        if {k: old.get(k) for k in lock} == lock:
            return old
        if not force:
            raise RuntimeError(f"{lock_path} exists with different contents; confirmation inputs changed after locking")
    started = confirmation_outputs(cfg)
    if started:
        raise RuntimeError(f"confirmation runs have already started ({sorted(started)[:3]}); the lock must predate "
                           "every confirmation output, so it cannot be written or replaced now (use verify_lock)")
    if old is not None:
        archived = lock_path.with_name(f"selection_lock.{str(old.get('locked_at')).replace(':', '')}.json")
        write_json(archived, old)
        lock["previous_locks"] = [*old.get("previous_locks", []),
                                  {"locked_at": old.get("locked_at"), "file": archived.name,
                                   "design_sha256": old.get("design_sha256"),
                                   "config_sha256": old.get("config_sha256")}]
    lock["locked_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    lock["confirmation_runs_scored_at_lock"] = []
    write_json(lock_path, lock)
    return lock


def verify_lock(cfg: StudyConfig, design_path: Path, strict: bool = True) -> dict[str, Any]:
    """Check the written lock against the current inputs before confirmation is planned or analysed (writes nothing).

    Every frozen field must still hold, and no confirmation output may predate the lock. DESIGN.md may only have grown
    by amendments appended after the lock (``post_lock_amendments``, to be reported as such). Runner files changed since
    the lock are listed in ``runner_code_changed`` (each needs a DESIGN amendment; every run records its own code hash).
    Fields an older lock did not record are listed in ``not_recorded``. ``strict`` raises on any mismatch.
    """

    lock_path = cfg.work("reports", LOCK_NAME)
    if not lock_path.is_file():
        raise RuntimeError(f"{lock_path} is missing; run lock-selection before the confirmation tier")
    old = json.loads(lock_path.read_text(encoding="utf-8"))
    cur = _lock_content(cfg, design_path)
    mismatches = [k for k, v in cur.items() if k in old and k not in _VERIFY_SEPARATELY and old[k] != v]
    text = design_path.read_bytes().decode("utf-8")
    post = amendments_after(text, str(old.get("design_sha256")))
    if post is None:
        mismatches.append("design_sha256")
    runner_changed: list[str] = []
    if "runner_code_sha256" in old:
        was, now = old["runner_code_sha256"], cur["runner_code_sha256"]
        runner_changed = sorted(k for k in set(was) | set(now) if was.get(k) != now.get(k))
    locked_at = dt.datetime.fromisoformat(old["locked_at"])
    early = {run: t for run, t in confirmation_outputs(cfg).items() if dt.datetime.fromisoformat(t) < locked_at}
    if early:
        mismatches.append("confirmation_outputs_before_lock")
    report = {"ok": not mismatches, "locked_at": old["locked_at"], "mismatches": mismatches,
              "amendments_at_lock": [a for a, _ in design_amendments(text) if a not in (post or [])],
              "post_lock_amendments": post or [], "runner_code_changed": runner_changed,
              "not_recorded": sorted(k for k in cur if k not in old), "confirmation_outputs_before_lock": early}
    if strict and mismatches:
        raise RuntimeError(f"{lock_path} no longer holds for {mismatches}: {json.dumps(report, sort_keys=True)}")
    return report
