"""Scan the per-rollout reward records of every RL run (DESIGN 7.3 Matching; RQ3 tokens).

    nice -n 10 $TRACE_CERT_PYTHON experiments/certificates/analysis/training_rollout_scan.py \\
        --rl-root "$TRACE_CERT_RUNS/rl" --out "$TRACE_CERT_RUNS/results/final/training/rollout_scan.json"

Reads ``rl/<model>/<arm>/s<seed>/reward_details/{train,val}_<pid>_<call:06d>.jsonl`` (one line per
rollout, written by ``reward_entry.compute_score``; one train call per optimizer step, one val call
per ``val_freq`` steps) and writes, per run and per call:

- ``n_rows``, ``n_prompts`` (distinct ``instance_id``), ``rollouts_per_prompt`` (min/max),
  ``prompt_multiset_sha256`` (sha256 of the sorted instance ids: the data order check of the
  matching audit compares it across the arms of one training seed, step by step);
- ``response_multiset_sha256`` (sha256 of the sorted per-rollout ``response_sha256``) and, for the
  train calls in ``KEEP_RESPONSE_CALLS``, the sorted 16-hex prefixes of the response hashes: the
  matching audit uses them to measure how many rollouts the arms of one seed share at a step
  (identical at step 1 when init, prompts, rollout seed and hardware class agree);
- ``response_tokens`` (sum of ``response_length``: exact generated rollout tokens of the step);
- means of the arm's own reward (``metrics.overall``), A, G, M, F, joint, and of every arm's reward
  formula on the same rollouts (``overall_by_formula``: A, G, P, P_gated, G_gated);
- GRPO signal: ``zero_var_groups`` = prompts whose rollouts all received the same own reward
  (no advantage signal), and ``zero_var_groups_A`` (all rollouts equal on A);
- ``error_class`` counts and per-adapter means of A, G, M, F and the own reward.

Read-only on the run directories. Each file is streamed line by line, so memory stays small.
The script re-runs from raw files; a run that is still training is scanned as far as it got.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

VERSION = "training_rollout_scan-1.1.0"
# train calls (0-based; call k = optimizer step k+1) whose response hashes are kept for the rollout-overlap check
KEEP_RESPONSE_CALLS = (0, 1, 2, 4, 9, 19, 49, 99, 199, 299)
FILE_RE = re.compile(r"^(train|val)_(\d+)_(\d{6})\.jsonl$")
FORMULAS = ("A", "G", "P", "P_gated", "G_gated")
SIGNALS = ("A", "G", "M", "F", "joint")


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def scan_file(path: Path, keep_responses: bool = False) -> Dict[str, Any]:
    ids: List[str] = []
    resp: List[str] = []
    own: List[float] = []
    sig: Dict[str, List[float]] = {k: [] for k in SIGNALS}
    form: Dict[str, List[float]] = {k: [] for k in FORMULAS}
    tokens = 0
    errors: Counter = Counter()
    by_prompt_own: Dict[str, List[float]] = defaultdict(list)
    by_prompt_a: Dict[str, List[float]] = defaultdict(list)
    by_adapter: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    arms = set()
    scorer = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            iid = row["instance_id"]
            ids.append(iid)
            resp.append(str(row.get("response_sha256")))
            arms.add(row.get("arm"))
            scorer.add(row.get("scorer_version"))
            o = float(row["metrics"]["overall"])
            own.append(o)
            by_prompt_own[iid].append(o)
            by_prompt_a[iid].append(float(row["A"]))
            for k in SIGNALS:
                sig[k].append(float(row[k]))
            for k in FORMULAS:
                form[k].append(float(row["overall"][k]))
            tokens += int(row["response_length"])
            errors[str(row.get("error_class"))] += 1
            ad = by_adapter[row["adapter_id"]]
            for k in SIGNALS:
                ad[k].append(float(row[k]))
            ad["overall"].append(o)
    per_prompt = [len(v) for v in by_prompt_own.values()]
    return {
        "file": path.name,
        "n_rows": len(ids),
        "n_prompts": len(by_prompt_own),
        "rollouts_per_prompt": [min(per_prompt), max(per_prompt)] if per_prompt else None,
        "prompt_multiset_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
        "response_multiset_sha256": hashlib.sha256("\n".join(sorted(resp)).encode()).hexdigest(),
        "response_sha_prefixes": sorted(r[:16] for r in resp) if keep_responses else None,
        "response_tokens": tokens,
        "overall": _mean(own),
        "signals": {k: _mean(v) for k, v in sig.items()},
        "overall_by_formula": {k: _mean(v) for k, v in form.items()},
        "zero_var_groups": sum(1 for v in by_prompt_own.values() if max(v) - min(v) == 0.0),
        "zero_var_groups_A": sum(1 for v in by_prompt_a.values() if max(v) - min(v) == 0.0),
        "error_class": dict(sorted(errors.items())),
        "by_adapter": {a: {k: _mean(v) for k, v in d.items()} | {"n": len(d["A"])} for a, d in sorted(by_adapter.items())},
        "arms_seen": sorted(a for a in arms if a is not None),
        "scorer_versions": sorted(s for s in scorer if s is not None),
    }


def scan_run(run_dir: str) -> Dict[str, Any]:
    run = Path(run_dir)
    det = run / "reward_details"
    calls: Dict[str, Dict[int, Dict[str, Any]]] = {"train": {}, "val": {}}
    pids: Dict[str, set] = {"train": set(), "val": set()}
    unknown = []
    t0 = time.time()
    for path in sorted(det.iterdir()) if det.is_dir() else []:
        m = FILE_RE.match(path.name)
        if not m:
            unknown.append(path.name)
            continue
        phase, pid, call = m.group(1), m.group(2), int(m.group(3))
        pids[phase].add(pid)
        rec = scan_file(path, keep_responses=(phase == "train" and call in KEEP_RESPONSE_CALLS))
        key = call
        if key in calls[phase]:
            # several reward-actor pids (a resumed run) restart the call counter; keep them apart
            key = f"{pid}:{call}"
        calls[phase][key] = rec
    return {
        "run_dir": str(run),
        "reward_details": str(det),
        "pids": {k: sorted(v) for k, v in pids.items()},
        "unknown_files": unknown,
        "n_calls": {k: len(v) for k, v in calls.items()},
        "calls": {k: {str(c): v[c] for c in sorted(v, key=lambda x: (isinstance(x, str), x))} for k, v in calls.items()},
        "seconds": round(time.time() - t0, 1),
    }


def find_runs(rl_root: Path) -> List[Path]:
    return sorted(p for p in rl_root.glob("*/*/s*") if (p / "run_receipt.json").is_file())


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rl-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None, help="regex on '<model>/<arm>/s<seed>' to scan a subset")
    args = ap.parse_args(argv)
    rl_root = Path(args.rl_root)
    runs = find_runs(rl_root)
    if args.only:
        runs = [r for r in runs if re.search(args.only, str(r.relative_to(rl_root)))]
    if not runs:
        print("no runs found", file=sys.stderr)
        return 2
    out: Dict[str, Any] = {"version": VERSION, "rl_root": str(rl_root), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": {}}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = pool.map(scan_run, [str(r) for r in runs])
        for run, res in zip(runs, results):
            key = str(run.relative_to(rl_root))
            out["runs"][key] = res
            print(f"{key}: train calls {res['n_calls']['train']}, val calls {res['n_calls']['val']}, {res['seconds']} s", flush=True)
    tmp = Path(args.out + f".tmp.{os.getpid()}")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
