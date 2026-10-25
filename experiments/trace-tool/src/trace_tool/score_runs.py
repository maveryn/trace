"""Score generated runs and write per-run summaries plus a cross-run report.

Example:
    trace-tool-score runs/full/qwen2.5-vl-7b-instruct --report runs/full/qwen2.5-vl-7b-instruct/report.md
    trace-tool-judge --pending runs/full/qwen2.5-vl-7b-instruct/judge_pending.jsonl ...   # optional
    trace-tool-score runs/full/qwen2.5-vl-7b-instruct --judge-results runs/full/qwen2.5-vl-7b-instruct/judge_results.jsonl \
        --report runs/full/qwen2.5-vl-7b-instruct/report.md

A run directory is any directory containing responses.jsonl + run_config.json
(normally <out>/<condition>/seed<seed>).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from .data import TraceRow, load_split
from .jsonl import read_jsonl
from .scoring import apply_judge, score_row


def response_key(response: str, answer_type: str) -> str:
    return hashlib.sha256(f"{answer_type}\x00{response}".encode()).hexdigest()


def find_run_dirs(paths: Iterable[str]) -> list[Path]:
    found = []
    for raw in paths:
        path = Path(raw)
        if (path / "responses.jsonl").exists():
            found.append(path)
        else:
            found.extend(sorted(p.parent for p in path.rglob("responses.jsonl")))
    return sorted(set(found))


def load_responses(run_dir: Path) -> list[dict[str, Any]]:
    """One record per instance_id: the last non-error record, else the last error record.

    Same rule as run_eval's resume (a row with any non-error record is done), so
    an endpoint error appended after a success never hides the success.
    """
    chosen: dict[str, dict[str, Any]] = {}
    records, _ = read_jsonl(run_dir / "responses.jsonl")
    for record in records:
        if "instance_id" not in record:
            continue
        previous = chosen.get(record["instance_id"])
        if previous is None or record.get("stop_reason") != "error" or previous.get("stop_reason") == "error":
            chosen[record["instance_id"]] = record
    return sorted(chosen.values(), key=lambda r: r["row_index"])


def load_judge_results(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    if not Path(path).exists():
        raise SystemExit(f"--judge-results {path} does not exist")
    records, _ = read_jsonl(Path(path))
    return {r["key"]: r for r in records if "key" in r}


def score_run(run_dir: Path, rows_by_id: dict[str, TraceRow], judge: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Returns (scored rows, rows needing the judge, number of rows skipped for endpoint errors).

    Rows whose last attempt still ended in an endpoint error (``stop_reason ==
    "error"``) are infrastructure failures, not model failures: they are left
    out of the scores (and so out of paired comparisons) until a rerun of
    trace-tool-eval retries them.
    """
    scored, pending, endpoint_errors = [], [], 0
    for record in load_responses(run_dir):
        if record.get("stop_reason") == "error":
            endpoint_errors += 1
            continue
        row = rows_by_id[record["instance_id"]]
        response = record.get("final_response") or ""
        scores = score_row(response, row.answer_gt, row.annotation_gt, row.reward_contract)
        key = response_key(response, row.answer_type)
        if scores["judge_needed"] and key not in judge:
            pending.append({"key": key, "answer_type": row.answer_type, "raw_response": response})
        scores = apply_judge(scores, judge.get(key), row.answer_gt)
        calls = record.get("tool_calls", [])
        executed = [c for c in calls if c.get("executed", c.get("ok", False))]
        scored.append(
            {
                "instance_id": row.instance_id,
                "row_index": row.row_index,
                "domain": row.domain,
                "task": row.task,
                "answer_type": row.answer_type,
                "answer_gt": row.answer_gt["value"],
                **scores,
                "stop_reason": record.get("stop_reason"),
                "turns": record.get("turns", 0),
                # requested calls, including malformed, unknown-tool and skipped ones
                "n_tool_calls": len(calls),
                # calls to an available tool that actually ran (successfully or not)
                "n_executed_calls": len(executed),
                "n_ok_calls": sum(1 for c in executed if c.get("ok")),
                "tools_used": sorted({c["name"] for c in executed if c.get("ok") and c.get("name")}),
                "tool_errors": sum(1 for c in executed if not c.get("ok")),
                "completion_tokens": record.get("completion_tokens", 0),
                "latency_s": record.get("latency_s"),
            }
        )
    return scored, pending, endpoint_errors


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(scored)
    if not n:
        return {"n": 0}
    calls = Counter(t for s in scored for t in s["tools_used"])
    with_tool = [s for s in scored if s["n_ok_calls"] > 0]
    without_tool = [s for s in scored if s["n_ok_calls"] == 0]
    executed_calls = sum(s["n_executed_calls"] for s in scored)
    latencies = sorted(s["latency_s"] for s in scored if s.get("latency_s") is not None)
    summary = {
        "n": n,
        "official_acc": fmean(s["official_correct"] for s in scored),
        "deterministic_acc": fmean(s["deterministic_correct"] for s in scored),
        "strict_acc": fmean(s["strict_correct"] for s in scored),
        "strict_format_rate": fmean(s["strict_format"] for s in scored),
        "judge_pending": sum(s["official_source"] == "judge_pending" for s in scored),
        "judge_resolved": sum(s["official_source"] == "judge" for s in scored),
        "unresolved": sum(s["official_source"] == "unresolved" for s in scored),
        # share of episodes with >= 1 successful call to an available tool (decision rule (b))
        "tool_use_rate": len(with_tool) / n,
        # share of episodes that requested any call, valid or not
        "tool_attempt_rate": sum(s["n_tool_calls"] > 0 for s in scored) / n,
        "mean_tool_calls": sum(s["n_tool_calls"] for s in scored) / n,
        "mean_executed_calls": executed_calls / n,
        "tool_error_rate": (sum(s["tool_errors"] for s in scored) / executed_calls) if executed_calls else 0.0,
        "acc_when_tool_used": fmean(s["official_correct"] for s in with_tool) if with_tool else None,
        "acc_when_no_tool": fmean(s["official_correct"] for s in without_tool) if without_tool else None,
        "episodes_using_tool": dict(calls),
        "mean_latency_s": fmean(latencies) if latencies else None,
        "median_latency_s": latencies[len(latencies) // 2] if latencies else None,
        "mean_turns": fmean(s["turns"] for s in scored),
        "mean_completion_tokens": fmean(s["completion_tokens"] for s in scored),
        "stop_reasons": dict(Counter(s["stop_reason"] for s in scored)),
    }
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in scored:
        by_domain[s["domain"]].append(s)
        by_type[s["answer_type"]].append(s)
    summary["by_domain"] = {
        d: {
            "n": len(v),
            "official_acc": fmean(x["official_correct"] for x in v),
            "tool_use_rate": fmean(x["n_ok_calls"] > 0 for x in v),
            "tool_attempt_rate": fmean(x["n_tool_calls"] > 0 for x in v),
        }
        for d, v in sorted(by_domain.items())
    }
    summary["by_answer_type"] = {t: {"n": len(v), "official_acc": fmean(x["official_correct"] for x in v)} for t, v in sorted(by_type.items())}
    return summary


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.2f}"


def markdown_report(runs: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    lines = [
        "| run | model | condition | seed | n | official acc | det. acc | strict acc | format | tool use | calls/ep | tool err | turns | gen tok | median s | pending judge | endpoint errors |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for config, s in runs:
        if not s.get("n"):
            continue
        median = "–" if s.get("median_latency_s") is None else f"{s['median_latency_s']:.1f}"
        lines.append(
            f"| {config.get('_run', '')} | {config['model']} | {config['condition']} | {config['seed']} | {s['n']} | {_pct(s['official_acc'])} | "
            f"{_pct(s['deterministic_acc'])} | {_pct(s['strict_acc'])} | {_pct(s['strict_format_rate'])} | "
            f"{_pct(s['tool_use_rate'])} | {s['mean_tool_calls']:.2f} | {_pct(s['tool_error_rate'])} | "
            f"{s['mean_turns']:.2f} | {s['mean_completion_tokens']:.0f} | {median} | {s['judge_pending']} | {s.get('excluded_endpoint_errors', 0)} |"
        )
    domains = sorted({d for _, s in runs for d in s.get("by_domain", {})})
    if domains:
        lines += ["", "Official accuracy (%) by domain", "", "| model | condition | seed | " + " | ".join(domains) + " |",
                  "| --- | --- | ---: | " + " | ".join("---:" for _ in domains) + " |"]
        for config, s in runs:
            if s.get("n"):
                cells = [_pct(s["by_domain"].get(d, {}).get("official_acc")) for d in domains]
                lines.append(f"| {config['model']} | {config['condition']} | {config['seed']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="run directories or roots to search")
    p.add_argument("--split", default="validation")
    p.add_argument("--dataset-path")
    p.add_argument("--judge-results", help="judge_results.jsonl from trace-tool-judge")
    p.add_argument("--pending-out", help="where to write judge_pending.jsonl (default: <first path>/judge_pending.jsonl)")
    p.add_argument("--report", help="write a markdown summary table here")
    args = p.parse_args(argv)

    run_dirs = find_run_dirs(args.paths)
    if not run_dirs:
        raise SystemExit("no responses.jsonl found")
    rows_by_id = {r.instance_id: r for r in load_split(args.split, local_path=args.dataset_path)}
    judge = load_judge_results(args.judge_results)

    all_pending: dict[str, dict[str, Any]] = {}
    report_rows = []
    for run_dir in run_dirs:
        config = json.loads((run_dir / "run_config.json").read_text())
        config["_run"] = str(run_dir.parent.parent)  # the --out directory
        scored, pending, endpoint_errors = score_run(run_dir, rows_by_id, judge)
        summary = summarize(scored)
        summary["excluded_endpoint_errors"] = endpoint_errors
        if endpoint_errors:
            print(f"warning: {run_dir}: {endpoint_errors} rows still end in an endpoint error and are excluded; "
                  "rerun trace-tool-eval with the same arguments to retry them", file=sys.stderr)
        with (run_dir / "scores.jsonl").open("w") as handle:
            for s in scored:
                handle.write(json.dumps(s, ensure_ascii=False) + "\n")
        (run_dir / "summary.json").write_text(json.dumps({"config": {k: config.get(k) for k in ("model", "condition", "seed")}, **summary}, indent=2) + "\n")
        for item in pending:
            all_pending[item["key"]] = item
        report_rows.append((config, summary))
        print(f"{run_dir}: n={summary.get('n')} official={_pct(summary.get('official_acc'))} "
              f"strict={_pct(summary.get('strict_acc'))} tool_use={_pct(summary.get('tool_use_rate'))} pending_judge={summary.get('judge_pending')}")

    pending_path = Path(args.pending_out) if args.pending_out else Path(args.paths[0]) / "judge_pending.jsonl"
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    with pending_path.open("w") as handle:
        for item in all_pending.values():
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{len(all_pending)} unique responses need the judge -> {pending_path}")

    report_rows.sort(key=lambda cs: (cs[0]["_run"], cs[0]["model"], cs[0]["condition"], cs[0]["seed"]))
    report = markdown_report(report_rows)
    if args.report:
        Path(args.report).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
