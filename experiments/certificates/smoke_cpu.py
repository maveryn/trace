"""Reproducible M1 CPU smoke of the certificate extension (DESIGN 1.1), no GPU.

    python experiments/certificates/smoke_cpu.py --out "$TRACE_CERT_RUNS/scratch/smoke"

1. ``audit`` of the five supported tasks (manifest written next to the output).
2. ``export --split fixtures`` with ``--count`` rows per adapter.
3. Synthetic trace-tool runs (``responses.jsonl`` + ``run_config.json``) for the ``cert_px``,
   ``ans`` and ``ann`` conditions, derived from the freshly built private store: the SFT target,
   the oracle with prose before it, the oracle with trailing prose, one operand corruption per
   adapter, a truncated response, an endpoint error and a missing row. Instance ids hash the
   rendered PNGs, so a frozen ``responses.jsonl`` would only fit one environment; deriving the
   responses from the store keeps the smoke reproducible anywhere.
4. ``score`` of each run, then checks of the expected aggregates (exit 1 on a mismatch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from trace_tasks.experimental.certificates import export, prompts, registry
from trace_tasks.experimental.certificates.__main__ import main as cli
from trace_tasks.experimental.certificates.types import CertificateExtension

SUPPORTED_TASKS = sorted(registry.TASK_ADAPTERS)


def operand_corruption(ext: CertificateExtension) -> Dict[str, Any]:
    """The first corruption that keeps G and lowers M (wrong operand, consistent arithmetic)."""
    adapter = registry.adapter_for_extension(ext)
    sft = {key: ext.sft_target[key] for key in ("annotation", "certificate", "answer")}
    for corruption in adapter.corruptions(ext.oracle, sft, 0):
        m = corruption.expected.get("M")
        lowers_m = m == "fail" or (isinstance(m, float) and m < 1.0)
        if corruption.expected.get("G") in ("unchanged", "pass") and lowers_m:
            return dict(corruption.response_obj)
    raise AssertionError(f"{adapter.adapter_id} declares no operand corruption")


def fake_run(data: Path, cond: str, run: Path) -> Dict[str, str]:
    """Write a trace-tool-shaped run of the fixtures split; returns instance_id -> response kind."""
    import pyarrow.parquet as pq

    parquet = data / "public" / f"fixtures.{cond}.parquet"
    rows = pq.read_table(parquet, columns=["instance_id", "task", "domain", "prompt_answer"]).to_pylist()
    store = {r["base"]["instance_id"]: CertificateExtension.from_dict(r)
             for r in export.read_store(data / "private" / f"fixtures.{export.STORE_NAME}")}
    run.mkdir(parents=True, exist_ok=True)
    # ans uses TRACE's own system prompt (DESIGN 7.1), which lives in trace-tool; the smoke
    # stands in a placeholder text and records the matching sha, then scores ans with
    # --allow-settings-mismatch (the override is recorded in the score receipt).
    system = prompts.system_prompt() if cond != "ans" else "TRACE answer-only system prompt (smoke stand-in)"
    settings = {"system_prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(), "oracle_crop": False,
                "tools": [], "decoding": "trace"}
    config = {"model": "smoke", "model_id": "Qwen/Qwen2.5-VL-3B-Instruct", "condition": f"{cond}_smoke", "seed": 42,
              "settings": settings, "dataset": {"split": "cert_fixtures", "sha256": export.file_sha256(parquet)}}
    (run / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    kinds: Dict[str, str] = {}
    lines: List[str] = []
    for i, row in enumerate(rows):
        ext = store[row["instance_id"]]
        target = json.loads(export.make_sft_target(None, ext))
        kind = ("sft", "prose_before", "trailing_prose", "operand", "truncated", "error", "missing")[i % 7]
        if cond == "ans":
            text = "Counting carefully.\n" + json.dumps({"answer": target["answer"]})
        elif cond == "ann":
            text = json.dumps({"annotation": target["annotation"], "answer": target["answer"]})
        elif kind == "prose_before":
            text = "Reading the image first.\n" + json.dumps(target)
        elif kind == "trailing_prose":
            text = json.dumps(target) + "\nDone."
        elif kind == "operand":
            text = json.dumps(operand_corruption(ext))
        else:
            text = json.dumps(target, separators=(",", ":"))
        stop, tokens = "answer", 64 + i
        if kind == "truncated":
            text, stop, tokens = text[: len(text) // 2], "length", 8192
        kinds[row["instance_id"]] = kind
        if kind == "missing":
            continue
        before, after = row["prompt_answer"].split("<image>", 1)
        user = ([{"type": "text", "text": before}] if before else []) + [{"type": "image", "index": 0}]
        user += [{"type": "text", "text": after}] if after else []
        record = {"row_index": i, "instance_id": row["instance_id"], "task": row["task"], "domain": row["domain"],
                  "final_response": "" if kind == "error" else text, "stop_reason": "error" if kind == "error" else stop,
                  "completion_tokens": 0 if kind == "error" else tokens, "seed": 42, "model": "smoke",
                  "transcript": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        lines.append(json.dumps(record))
    (run / "responses.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return kinds


def check_scores(run: Path, cond: str, kinds: Dict[str, str]) -> List[str]:
    problems = []
    rows = [json.loads(line) for line in (run / "cert" / "cert_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    expected_rows = sum(1 for kind in kinds.values() if kind not in ("error", "missing"))
    if len(rows) != expected_rows:
        problems.append(f"{cond}: {len(rows)} scored rows, expected {expected_rows}")
    for row in rows:
        kind = kinds[row["instance_id"]]
        got = (row["A"], row["M"], row["F"])
        want = {
            "sft": (1.0, 1.0, 1.0), "prose_before": (1.0, 1.0, 1.0), "trailing_prose": (1.0, 1.0, 0.0),
        }.get(kind) if cond == "cert_px" else None
        if want is not None and got != want:
            problems.append(f"{cond} {kind} {row['adapter_id']}: (A, M, F) = {got}, expected {want}")
        if cond == "cert_px" and kind == "operand" and not row["M"] < 1.0:
            problems.append(f"{cond} operand {row['adapter_id']}: M = {row['M']}, expected < 1")
        if cond != "cert_px" and kind != "truncated" and row["A"] != 1.0:
            problems.append(f"{cond} {kind} {row['adapter_id']}: A = {row['A']}")
    summary = json.loads((run / "cert" / "summary.json").read_text(encoding="utf-8"))
    counts = [c for c in summary["denominators"].values() if isinstance(c, dict) and "missing" in c]
    for key, kind in (("missing", "missing"), ("endpoint_errors", "error")):
        want = sum(1 for k in kinds.values() if k == kind)
        if sum(c[key] for c in counts) != want:
            problems.append(f"{cond}: {key} = {sum(c[key] for c in counts)}, expected {want}")
    return problems


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="scratch directory (created)")
    parser.add_argument("--count", type=int, default=8, help="fixture rows per adapter (>= 7 covers every response kind)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    data = out / "data"
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps({"tasks": SUPPORTED_TASKS}, indent=2) + "\n", encoding="utf-8")
    steps = [
        ["audit", "--manifest", str(manifest), "--out", str(out / "coverage.v1.json")],
        ["export", "--split", "fixtures", "--out", str(data), "--count", str(args.count), "--workers", str(args.workers)],
    ]
    for step in steps:
        if cli(step) != 0:
            print(f"smoke: {step[0]} failed", file=sys.stderr)
            return 1
    problems: List[str] = []
    for cond in ("cert_px", "ans", "ann"):
        run = out / "runs" / cond / "seed42"
        kinds = fake_run(data, cond, run)
        code = cli(["score", "--responses", str(run / "responses.jsonl"),
                    "--store", str(data / "private" / f"fixtures.{export.STORE_NAME}"),
                    "--receipt", str(data / "receipts" / "fixtures.json"), "--out", str(run / "cert")]
                   + (["--allow-settings-mismatch"] if cond == "ans" else []))
        if code != 0:
            problems.append(f"{cond}: score exited {code}")
            continue
        problems += check_scores(run, cond, kinds)
    for problem in problems:
        print(f"smoke: {problem}", file=sys.stderr)
    print(json.dumps({"smoke": "ok" if not problems else "failed", "out": str(out), "problems": len(problems)}))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
