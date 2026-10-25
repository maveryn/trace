"""Paired comparison of conditions against a baseline on the same rows.

Example:
    trace-tool-compare runs/full/qwen2.5-vl-7b-instruct --baseline no_tool --out runs/full/qwen2.5-vl-7b-instruct/compare.md

For each (model, seed) and condition C: accuracy delta vs the baseline on the
intersection of scored rows, a paired bootstrap 95% CI, and an exact McNemar
p-value. Seeds are then pooled by averaging each row's correctness over the
seeds present in both conditions before bootstrapping. Run trace-tool-score
first.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .score_runs import find_run_dirs


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b (baseline-only correct) vs c (condition-only correct)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def paired_bootstrap(diff: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.concatenate(
        [diff[rng.integers(0, len(diff), size=(min(1000, n_boot - start), len(diff)))].mean(axis=1) for start in range(0, n_boot, 1000)]
    )
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_scores(paths: list[str], metric: str) -> dict[tuple[str, str, int], dict[str, dict[str, Any]]]:
    out = {}
    sources: dict[tuple[str, str, int], Path] = {}
    for run_dir in find_run_dirs(paths):
        scores_path = run_dir / "scores.jsonl"
        if not scores_path.exists():
            continue
        config = json.loads((run_dir / "run_config.json").read_text())
        rows = {}
        for line in scores_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                rows[record["instance_id"]] = {"y": float(record[metric]), "domain": record["domain"], "pending": record.get("official_source") == "judge_pending"}
        key = (config["model"], config["condition"], int(config["seed"]))
        if key in out:
            raise SystemExit(
                f"two run directories hold {key}: {sources[key]} and {run_dir}. "
                "Compare one --out directory at a time (e.g. runs/full/<model>, not runs/)."
            )
        sources[key] = run_dir
        out[key] = rows
    return out


def _fmt(x: float) -> str:
    return f"{100 * x:+.2f}"


def compare(scores: dict, baseline: str, n_boot: int, only: list[str] | None = None) -> str:
    lines = []
    models = sorted({m for m, _, _ in scores})
    pending = sum(r["pending"] for rows in scores.values() for r in rows.values())
    if pending:
        lines += [f"> {pending} scored rows still await the judge pass and count as wrong.", ""]
    for model in models:
        conds = sorted({c for m, c, _ in scores if m == model and c != baseline and (not only or c in only)})
        seeds = sorted({s for m, c, s in scores if m == model and c == baseline})
        if not seeds:
            lines.append(f"## {model}: no `{baseline}` runs; skipped\n")
            continue
        lines += [f"## {model} (baseline `{baseline}`)", "", "Holm adjusts across the compared conditions within each seed.", "",
                  "| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |"]
        per_seed: dict[int, list[tuple[str, str, float]]] = defaultdict(list)
        pooled: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
        domains: dict[str, str] = {}
        for cond in conds:
            for seed in seeds:
                base_rows = scores.get((model, baseline, seed))
                cond_rows = scores.get((model, cond, seed))
                if not base_rows or not cond_rows:
                    continue
                ids = sorted(set(base_rows) & set(cond_rows))
                y0 = np.array([base_rows[i]["y"] for i in ids])
                y1 = np.array([cond_rows[i]["y"] for i in ids])
                for i, a, b_ in zip(ids, y0, y1):
                    pooled[cond][i].append((a, b_))
                    domains[i] = base_rows[i]["domain"]
                lo, hi = paired_bootstrap(y1 - y0, n_boot)
                b = int(np.sum((y0 == 1) & (y1 == 0)))
                c = int(np.sum((y0 == 0) & (y1 == 1)))
                row = (f"| {cond} | {seed} | {len(ids)} | {100 * y0.mean():.2f} | {100 * y1.mean():.2f} | {_fmt(y1.mean() - y0.mean())} | "
                       f"[{_fmt(lo)}, {_fmt(hi)}] | {b} / {c} | {mcnemar_exact(b, c):.3g} |")
                per_seed[seed].append((cond, row, mcnemar_exact(b, c)))
        for seed in sorted(per_seed):
            entries = per_seed[seed]
            for (cond, row, _), adj in zip(entries, holm([p for _, _, p in entries])):
                lines.append(f"{row} {adj:.3g} |")
        lines += ["", "Seeds pooled (row correctness averaged over shared seeds):", "",
                  "| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |", "| --- | ---: | ---: | ---: | ---: | --- |"]
        domain_deltas: dict[str, dict[str, float]] = {}
        for cond, per_row in pooled.items():
            ids = sorted(per_row)
            y0 = np.array([np.mean([a for a, _ in per_row[i]]) for i in ids])
            y1 = np.array([np.mean([b for _, b in per_row[i]]) for i in ids])
            lo, hi = paired_bootstrap(y1 - y0, n_boot)
            lines.append(f"| {cond} | {len(ids)} | {100 * y0.mean():.2f} | {100 * y1.mean():.2f} | {_fmt(y1.mean() - y0.mean())} | [{_fmt(lo)}, {_fmt(hi)}] |")
            by_domain: dict[str, list[float]] = defaultdict(list)
            for i, a, b_ in zip(ids, y0, y1):
                by_domain[domains[i]].append(b_ - a)
            domain_deltas[cond] = {d: float(np.mean(v)) for d, v in by_domain.items()}
        all_domains = sorted({d for v in domain_deltas.values() for d in v})
        if all_domains:
            lines += ["", "Δ accuracy (pp) by domain, seeds pooled:", "", "| condition | " + " | ".join(all_domains) + " |",
                      "| --- | " + " | ".join("---:" for _ in all_domains) + " |"]
            for cond, deltas in domain_deltas.items():
                lines.append(f"| {cond} | " + " | ".join(_fmt(deltas[d]) if d in deltas else "–" for d in all_domains) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+")
    p.add_argument("--baseline", default="no_tool")
    p.add_argument("--metric", default="official_correct", choices=["official_correct", "deterministic_correct", "strict_correct"])
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--conditions", help="comma-separated conditions to compare (the Holm family); default: all present")
    p.add_argument("--out")
    args = p.parse_args(argv)
    only = [c.strip() for c in args.conditions.split(",") if c.strip()] if args.conditions else None
    scores = load_scores(args.paths, args.metric)
    missing = sorted(set(only or []) - {c for _, c, _ in scores})
    if missing:
        print(f"warning: --conditions lists {missing}, which have no scored runs; the Holm family is smaller", file=sys.stderr)
    report = compare(scores, args.baseline, args.n_boot, only)
    if args.out:
        Path(args.out).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
