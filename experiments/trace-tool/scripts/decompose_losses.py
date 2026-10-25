#!/usr/bin/env python
"""Split each condition's accuracy change vs. a baseline into rows that ended without an answer and rows that answered.

Pairs every row with the same row of the baseline in the same seed (``scores.jsonl`` from trace-tool-score),
over every seed both conditions have. For each condition it prints:

- the share of episodes that stopped without a final answer (``stop_reason`` other than ``answer``:
  ``max_turns``, ``token_budget``, ``length``);
- the change in official accuracy (pp of all rows), split into the part from those no-answer rows and the
  part from rows that answered;
- on the answered rows only, the condition's accuracy next to the baseline's accuracy on the same rows.
  Which rows answer is decided by the condition's own run, so this comparison is descriptive.

``--by-domain`` adds the seeds-pooled change per domain with a normal-approximation 95% CI over rows
(uncorrected for the number of domains).

    python scripts/decompose_losses.py runs/full/qwen2.5-vl-7b-instruct
    python scripts/decompose_losses.py runs/full/qwen2.5-vl-3b-instruct --conditions zoom,ocr --by-domain
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def load_scores(path: Path) -> dict[str, dict]:
    with path.open() as handle:
        return {row["instance_id"]: row for row in map(json.loads, handle) if row}


def seeds_of(out: Path, condition: str) -> set[str]:
    return {p.parent.name for p in (out / condition).glob("seed*/scores.jsonl")}


def decompose(out: Path, baseline: str, condition: str) -> dict:
    seeds = sorted(seeds_of(out, condition) & seeds_of(out, baseline))
    if not seeds:
        raise SystemExit(f"{out}: no seed with scores.jsonl for both {condition} and {baseline}")
    n = n_noans = n_ans = 0
    d_noans = d_ans = acc_ans = base_ans = 0.0
    by_domain: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for seed in seeds:
        base = load_scores(out / baseline / seed / "scores.jsonl")
        rows = load_scores(out / condition / seed / "scores.jsonl")
        for iid, row in rows.items():
            if iid not in base:
                continue
            correct, base_correct = row["official_correct"], base[iid]["official_correct"]
            n += 1
            by_domain[row["domain"]][iid].append(correct - base_correct)
            if row["stop_reason"] == "answer":
                n_ans += 1
                d_ans += correct - base_correct
                acc_ans += correct
                base_ans += base_correct
            else:
                n_noans += 1
                d_noans += correct - base_correct
    domains = {}
    for domain, per_row in sorted(by_domain.items()):
        xs = [sum(v) / len(v) for v in per_row.values()]  # seeds pooled per row
        mean = sum(xs) / len(xs)
        se = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1) / len(xs)) if len(xs) > 1 else float("nan")
        domains[domain] = (len(xs), mean, mean - 1.96 * se, mean + 1.96 * se)
    return {
        "condition": condition, "seeds": seeds, "n": n, "no_answer": n_noans / n,
        "delta": (d_noans + d_ans) / n, "delta_no_answer": d_noans / n, "delta_answered": d_ans / n,
        "acc_answered": acc_ans / n_ans if n_ans else float("nan"),
        "baseline_acc_answered": base_ans / n_ans if n_ans else float("nan"),
        "domains": domains,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("out", type=Path, help="a trace-tool-eval --out directory, e.g. runs/full/qwen2.5-vl-7b-instruct")
    p.add_argument("--baseline", default="no_tool")
    p.add_argument("--conditions", help="comma-separated (default: every other condition in OUT)")
    p.add_argument("--by-domain", action="store_true", help="also print the per-domain change with a 95%% CI")
    args = p.parse_args(argv)

    if args.conditions:
        conditions = [c for c in args.conditions.split(",") if c]
    else:
        conditions = sorted(d.name for d in args.out.iterdir() if d.is_dir() and d.name != args.baseline and seeds_of(args.out, d.name))
    results = [decompose(args.out, args.baseline, c) for c in conditions]

    print(f"## {args.out.name} (baseline `{args.baseline}`, paired rows, seeds pooled)\n")
    print("| condition | seeds | rows | no answer | Δ (pp) | Δ from no-answer rows | Δ from answered rows | acc on answered rows | baseline acc, same rows |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        print(f"| {r['condition']} | {','.join(s[4:] for s in r['seeds'])} | {r['n']} | {100 * r['no_answer']:.1f}% | {100 * r['delta']:+.2f} | "
              f"{100 * r['delta_no_answer']:+.2f} | {100 * r['delta_answered']:+.2f} | {100 * r['acc_answered']:.1f} | {100 * r['baseline_acc_answered']:.1f} |")
    if args.by_domain:
        print("\nΔ accuracy (pp) by domain, seeds pooled per row, with a normal-approximation 95% CI (uncorrected):\n")
        print("| condition | domain | rows | Δ (pp) | 95% CI |")
        print("| --- | --- | ---: | ---: | --- |")
        for r in results:
            for domain, (rows, mean, lo, hi) in r["domains"].items():
                print(f"| {r['condition']} | {domain} | {rows} | {100 * mean:+.2f} | [{100 * lo:+.2f}, {100 * hi:+.2f}] |")


if __name__ == "__main__":
    sys.exit(main())
