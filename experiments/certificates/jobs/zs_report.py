"""Bookkeeping for the zero-shot certificate runs (jobs/zs_eval.sh); CPU only.

    zs_report.py count       --responses R --expected N
    zs_report.py check-data  --data D --split S --cond C [--cond C ...] [--cache F]
    zs_report.py validate    --model-root M --data D --splits ... --conds ... --seeds ... --frame px|rel1000
                             [--expect-rows N] [--out validation.json]
    zs_report.py report      --model-root M --splits ... [--out summary.md]
    zs_report.py frame-diag  --run RUN_DIR --store STORE [--out frame_diag.json]

``count``, ``check-data``, ``validate`` and ``report`` use the standard library only (they also run
in trace-tool's env); ``frame-diag`` imports ``trace_tasks`` (the scoring env). ``validate`` prints
``FLAG <kind> <detail>`` lines (read by the host watcher) and exits 1 on any gap: missing rows,
rows whose only records are endpoint errors, unscored or stale scores, a dataset that is not the
frozen public parquet, or a non-finite summary value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PUBLIC = {"ans_v1": "ans", "ann_v1": "ann"}


def public_cond(cond: str, frame: str) -> str:
    return PUBLIC.get(cond) or f"cert_{frame}"


# ------------------------------------------------------------------------------------- count


def response_counts(path: Path) -> Dict[str, int]:
    """trace-tool resume rule: a row is done when it has a record whose stop_reason != "error"."""
    ok: set = set()
    seen: set = set()
    lines = 0
    if path.exists():
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(raw):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                if i == len(raw) - 1:
                    continue  # crash-torn last line (trace-tool repairs it on resume)
                raise
            lines += 1
            if not isinstance(rec, dict) or "instance_id" not in rec:
                continue
            seen.add(rec["instance_id"])
            if rec.get("stop_reason") != "error":
                ok.add(rec["instance_id"])
    return {"done": len(ok), "error_only": len(seen - ok), "lines": lines}


def cmd_count(args: argparse.Namespace) -> int:
    c = response_counts(Path(args.responses))
    complete = int(c["done"] >= args.expected and c["error_only"] == 0)
    print(f"done={c['done']} expected={args.expected} error_only={c['error_only']} lines={c['lines']} complete={complete}")
    return 0


# ------------------------------------------------------------------------------------- data


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def receipt(data: Path, split: str) -> Dict[str, Any]:
    return json.loads((data / "receipts" / f"{split}.json").read_text(encoding="utf-8"))


def expected_rows(data: Path, split: str) -> int:
    return int(receipt(data, split)["totals"]["accepted"])


def public_entry(data: Path, split: str, cond: str) -> Dict[str, Any]:
    entry = (receipt(data, split).get("outputs") or {}).get(f"public/{split}.{cond}.parquet")
    if not entry or entry.get("kind") != "public_parquet":
        raise SystemExit(f"receipts/{split}.json lists no public/{split}.{cond}.parquet")
    return entry


def cmd_check_data(args: argparse.Namespace) -> int:
    data = Path(args.data)
    cache: Dict[str, Any] = {}
    cache_path = Path(args.cache) if args.cache else None
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except (OSError, ValueError):
            cache = {}
    bad = 0
    for cond in args.cond:
        path = data / "public" / f"{args.split}.{cond}.parquet"
        entry = public_entry(data, args.split, cond)
        if not path.exists():
            print(f"FLAG data_missing {path}")
            bad += 1
            continue
        st = path.stat()
        key = f"{path}|{st.st_size}|{st.st_mtime_ns}"
        got = cache.get(key) or sha256_file(path)
        cache[key] = got
        if got != entry["sha256"]:
            print(f"FLAG data_mismatch {path.name} sha256 {got} != receipt {entry['sha256']}")
            bad += 1
        else:
            print(f"data ok {path.name} rows={entry['rows']} sha256={got[:12]}")
    if cache_path:
        tmp = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(cache, indent=1, sort_keys=True))
        os.replace(tmp, cache_path)
    return 4 if bad else 0


# ------------------------------------------------------------------------------------- validate


def non_finite(obj: Any, path: str = "") -> List[str]:
    out: List[str] = []
    if isinstance(obj, float) and not math.isfinite(obj):
        out.append(path or "<root>")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(non_finite(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(non_finite(v, f"{path}[{i}]"))
    return out


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for line in f if line.strip())


def validate_run(root: Path, data: Path, split: str, cond: str, seed: int, frame: str, expect: Optional[int]) -> Dict[str, Any]:
    run = root / split / cond / f"seed{seed}"
    want = expect if expect is not None else expected_rows(data, split)
    res: Dict[str, Any] = {"run": f"{split}/{cond}/seed{seed}", "expected": want, "problems": []}
    responses = run / "responses.jsonl"
    if not responses.exists():
        res["problems"].append("missing")
        return res
    c = response_counts(responses)
    res.update(c)
    if c["done"] < want:
        res["problems"].append(f"gap {want - c['done']} rows")
    if c["error_only"]:
        res["problems"].append(f"endpoint_errors {c['error_only']} rows")
    try:
        cfg = json.loads((run / "run_config.json").read_text())
        want_sha = public_entry(data, split, public_cond(cond, frame))["sha256"]
        if (cfg.get("dataset") or {}).get("sha256") != want_sha:
            res["problems"].append("dataset is not the frozen public parquet")
    except (OSError, ValueError, SystemExit) as exc:
        res["problems"].append(f"run_config: {exc}")
    scores = run / "cert" / "cert_scores.jsonl"
    summary = run / "cert" / "summary.json"
    if not scores.exists() or not (run / "cert" / "score_receipt.json").exists():
        res["problems"].append("unscored")
    else:
        res["scored"] = count_lines(scores)
        if scores.stat().st_mtime < responses.stat().st_mtime:
            res["problems"].append("scores older than responses")
        if res["scored"] != c["done"]:
            res["problems"].append(f"scored {res['scored']} != done {c['done']}")
        try:
            bad = non_finite(json.loads(summary.read_text()))
            if bad:
                res["problems"].append(f"nan in summary: {bad[:3]}")
        except (OSError, ValueError) as exc:
            res["problems"].append(f"summary: {exc}")
    return res


def cmd_validate(args: argparse.Namespace) -> int:
    root, data = Path(args.model_root), Path(args.data)
    runs = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        for split in args.splits.split(","):
            for cond in args.conds.split(","):
                runs.append(validate_run(root, data, split, cond, seed, args.frame, args.expect_rows))
    for summary in sorted(root.glob("summary*.json")):
        try:
            bad = non_finite(json.loads(summary.read_text()))
        except (OSError, ValueError):
            bad = ["unreadable"]
        if bad:
            runs.append({"run": summary.name, "problems": [f"nan in summary: {bad[:3]}"]})
    failed = [r for r in runs if r["problems"]]
    for r in failed:
        for problem in r["problems"]:
            kind = ("endpoint_errors" if problem.startswith("endpoint_errors") else
                    "nan" if problem.startswith("nan") else "gap")
            print(f"FLAG {kind} {r['run']}: {problem}")
    total = sum(r.get("done", 0) for r in runs)
    want = sum(r.get("expected", 0) for r in runs)
    out = {"model_root": str(root), "runs": runs, "done": total, "expected": want, "ok": not failed}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"VALIDATION {'OK' if not failed else 'FAILED'}: {total}/{want} rows, {len(runs) - len(failed)}/{len(runs)} runs clean")
    return 0 if not failed else 1


# ------------------------------------------------------------------------------------- report

COLUMNS = [
    ("n", "n", "{:d}"), ("A", "A", "{:.3f}"), ("G", "G", "{:.3f}"), ("G_shown", "G_shown", "{:.3f}"),
    ("G_px", "G_px", "{:.3f}"), ("M", "M", "{:.3f}"), ("F", "F", "{:.3f}"), ("joint", "joint", "{:.3f}"),
    ("valid", "valid_schema_rate", "{:.3f}"), ("complete", "completion_rate", "{:.3f}"),
    ("trunc", "truncation_rate", "{:.3f}"), ("<=2048", "within_2048_rate", "{:.3f}"),
]


def fmt(value: Any, spec: str) -> str:
    if value is None:
        return "-"
    try:
        return spec.format(value)
    except (ValueError, TypeError):
        return str(value)


def report_table(groups: Iterable[Dict[str, Any]]) -> List[str]:
    head = ["cond"] + [c[0] for c in COLUMNS] + ["tok mean", "A1M<1", "A0M1", "verifier ms"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for g in sorted(groups, key=lambda g: str(g.get("cond"))):
        cells = [str(g.get("cond"))] + [fmt(g.get(key), spec) for _, key, spec in COLUMNS]
        axm = g.get("A_x_M1") or {}
        cells += [fmt((g.get("tokens") or {}).get("mean"), "{:.0f}"), fmt(axm.get("A1_Mlt1"), "{:d}"),
                  fmt(axm.get("A0_M1"), "{:d}"), fmt((g.get("verifier_ms") or {}).get("mean"), "{:.2f}")]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def cmd_report(args: argparse.Namespace) -> int:
    root = Path(args.model_root)
    out = [f"# Zero-shot certificate evaluation: {root.name}", ""]
    for split in args.splits.split(","):
        path = root / f"summary.{split}.json"
        if not path.exists():
            out += [f"## {split}", "", "(no summary)", ""]
            continue
        summary = json.loads(path.read_text())
        out += [f"## {split}", ""] + report_table(summary.get("all") or []) + [""]
        classes = {str(g.get("cond")): g.get("error_classes") for g in summary.get("all") or [] if g.get("error_classes")}
        for cond, cls in sorted(classes.items()):
            out.append(f"- error classes ({cond}): " + ", ".join(f"{k} {v}" for k, v in cls.items()))
        out.append("")
    out += ["Columns: A answer, G annotation (cert: declared frame; ann: family-native frame), G_shown/G_px",
            "frame diagnostics, M certificate truth, F strict format, joint, valid schema, completion",
            "(stop_reason answer), truncation, completes within 2048 tokens, mean completion tokens,",
            "A=1 with M<1, A=0 with M=1, mean verifier time. Seeds pooled; see summary.<split>.json."]
    text = "\n".join(out) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


# ------------------------------------------------------------------------------------- frame-diag


def flatten_numbers(value: Any) -> List[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)] if math.isfinite(float(value)) else []
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for v in value:
            out.extend(flatten_numbers(v))
        return out
    if isinstance(value, dict):
        out = []
        for v in value.values():
            out.extend(flatten_numbers(v))
        return out
    return []


def xy_pairs(value: Any) -> List[Tuple[float, float]]:
    nums = flatten_numbers(value)
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]


def mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def cmd_frame_diag(args: argparse.Namespace) -> int:
    from trace_tasks.experimental.certificates import registry
    from trace_tasks.experimental.certificates.evaluate import load_store, read_responses
    from trace_tasks.experimental.certificates.geometry import Frame, source_frame
    from trace_tasks.experimental.certificates.scoring import legacy_prediction, score_grounding

    run = Path(args.run)
    store = load_store(args.store)
    scores = {}
    for line in (run / "cert" / "cert_scores.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            scores[row["instance_id"]] = row
    records, _, _ = read_responses(run / "responses.jsonl")
    rows = []
    for rec in records:
        ext = store.for_instance(rec["instance_id"])
        adapter = registry.adapter_for_extension(ext)
        sc = scores.get(rec["instance_id"], {})
        w, h = (int(v) for v in ext.image_size)
        sw, sh = (int(v) for v in (rec.get("shown_image_size") or [w, h]))
        _, annotation, _ = legacy_prediction(str(rec.get("final_response") or ""))
        frames = {
            "px_source": source_frame("px_source", (w, h)),
            "px_shown": Frame("px_shown", w / sw, h / sh),
            "rel1000": source_frame("rel1000", (w, h)),
        }
        g = {}
        for name, frame in frames.items():
            try:
                g[name] = score_grounding(annotation, ext.labels, adapter.annotation_type, frame, (w, h))[0]
            except Exception:  # a malformed annotation scores 0 under every frame
                g[name] = 0.0
        pts = xy_pairs(annotation)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        row = {
            "instance_id": rec["instance_id"],
            "adapter": ext.adapter_id.split(".")[0],
            "source": [w, h],
            "shown": [sw, sh],
            "n_coords": len(pts),
            "max_x": max(xs) if xs else None,
            "max_y": max(ys) if ys else None,
            "le_1000": bool(pts) and max(xs + ys) <= 1000,
            "in_source": bool(pts) and max(xs) <= w and max(ys) <= h,
            "in_shown": bool(pts) and max(xs) <= sw and max(ys) <= sh,
            "G_px_source": g["px_source"], "G_px_shown": g["px_shown"], "G_rel1000": g["rel1000"],
            "best_frame": max(g, key=lambda k: g[k]) if any(g.values()) else None,
        }
        for key in ("A", "G", "G_shown", "M", "M_shown", "F", "error_class", "coord_frame", "family", "stop_reason", "completion_tokens"):
            row[key] = sc.get(key)
        rows.append(row)
    declared = rows[0]["coord_frame"] if rows else None
    agg: Dict[str, Any] = {"n": len(rows), "declared_frame": declared, "family": rows[0]["family"] if rows else None}
    for key in ("A", "G", "G_shown", "M", "M_shown", "F", "G_px_source", "G_px_shown", "G_rel1000"):
        agg[key] = mean([r[key] for r in rows if isinstance(r.get(key), (int, float))])
    with_coords = [r for r in rows if r["n_coords"]]
    agg["rows_with_coords"] = len(with_coords)
    for key in ("le_1000", "in_source", "in_shown"):
        agg[f"frac_{key}"] = mean([float(r[key]) for r in with_coords])
    best: Dict[str, int] = {}
    for r in rows:
        best[str(r["best_frame"])] = best.get(str(r["best_frame"]), 0) + 1
    agg["best_frame_counts"] = best
    hdr = ("adapter", "A", "G", "G_shown", "M", "F", "G@px", "G@shown", "G@rel", "max_x", "max_y", "src WxH", "err")
    print("frame diagnostics (G re-scored on the legacy-extracted annotation under each frame)")
    print("  ".join(f"{h:>9}" for h in hdr))
    for r in rows:
        cells = [r["adapter"][:9], r["A"], r["G"], r["G_shown"], r["M"], r["F"], r["G_px_source"], r["G_px_shown"],
                 r["G_rel1000"], r["max_x"], r["max_y"], f"{r['source'][0]}x{r['source'][1]}", str(r["error_class"])[:9]]
        print("  ".join(f"{c:>9.3f}" if isinstance(c, float) else f"{str(c if c is not None else '-'):>9}" for c in cells))
    print("aggregate: " + json.dumps(agg, sort_keys=True))
    best_g = max(agg["G_px_source"] or 0, agg["G_px_shown"] or 0, agg["G_rel1000"] or 0)
    if not with_coords or best_g == 0:
        hint = "INCONCLUSIVE: no annotation scores above 0 in any frame (" + (
            f"{len(with_coords)}/{len(rows)} rows give coordinates)" if rows else "no rows)")
    elif declared == "rel1000":
        hint = ("obeys rel1000" if (agg["frac_le_1000"] or 0) >= 0.9 and (agg["G_rel1000"] or 0) >= (agg["G_px_source"] or 0)
                else "CHECK: coordinates look like pixels, not rel1000")
    else:
        hint = ("obeys source px" if (agg["G_px_source"] or 0) >= max(agg["G_px_shown"] or 0, agg["G_rel1000"] or 0)
                else "CHECK: G is higher in another frame than the declared source px")
    print(f"frame obedience: {hint}")
    if args.out:
        Path(args.out).write_text(json.dumps({"aggregate": agg, "hint": hint, "rows": rows}, indent=1, sort_keys=True) + "\n")
    return 0


# ------------------------------------------------------------------------------------- main


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("count")
    c.add_argument("--responses", required=True)
    c.add_argument("--expected", type=int, required=True)
    c.set_defaults(func=cmd_count)
    d = sub.add_parser("check-data")
    d.add_argument("--data", required=True)
    d.add_argument("--split", required=True)
    d.add_argument("--cond", action="append", required=True, help="public condition (ans, ann, cert_px, cert_rel1000)")
    d.add_argument("--cache")
    d.set_defaults(func=cmd_check_data)
    v = sub.add_parser("validate")
    v.add_argument("--model-root", required=True)
    v.add_argument("--data", required=True)
    v.add_argument("--splits", required=True)
    v.add_argument("--conds", required=True)
    v.add_argument("--seeds", required=True)
    v.add_argument("--frame", required=True, choices=["px", "rel1000"])
    v.add_argument("--expect-rows", type=int)
    v.add_argument("--out")
    v.set_defaults(func=cmd_validate)
    r = sub.add_parser("report")
    r.add_argument("--model-root", required=True)
    r.add_argument("--splits", required=True)
    r.add_argument("--out")
    r.set_defaults(func=cmd_report)
    f = sub.add_parser("frame-diag")
    f.add_argument("--run", required=True)
    f.add_argument("--store", required=True)
    f.add_argument("--out")
    f.set_defaults(func=cmd_frame_diag)
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
