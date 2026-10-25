"""Section 5 CPU experiments on the frozen evaluation splits (DESIGN 7.1 CPU part, 8, 9).

    python experiments/certificates/analysis/section5_cpu.py \\
        --data "$TRACE_CERT_RUNS/data/v1" --out "$TRACE_CERT_RUNS/results/section5_cpu" \\
        --splits e1_dev,e1_test,e2_test --workers 12

For every extension of the given splits (read from ``private/<split>.extensions_v1.jsonl.zst``,
checked against ``receipts/<split>.json``):

(a) **Oracle round trip.** The unrounded oracle object must give A = G = M = F = 1; the
    serialized SFT target (boxes rounded half-up) must give A = M = F = 1 with every box
    resolved; for rpg rows every enumerated optimal path (cap ``--path-cap``) must give
    A = G = M = F = 1. The SFT object is validated against the public schema of its kind and
    the extension against the private schema.
(b) **Corruption suite.** Every ``Adapter.corruptions`` case is scored; per adapter x corruption
    the output has mean A/G/M/F, the fraction of rows meeting every declared expectation, the
    fraction where every intended failing check fails, and the go/no-go 1(b) count of corruption
    types per numeric adapter with |dG| < 0.01 and M lowered.
(c) **Verifier time.** Wall ``verifier_ms`` and process CPU ms of the oracle-object score, with
    and without the Qwen2.5-VL shown-frame diagnostics (diagnostic only: scores never depend on it).

Writes ``rows.jsonl``, ``corruptions.jsonl``, ``summary.json`` and ``receipt.json`` under ``--out``
and prints the Markdown tables (``--md`` also writes them to a file).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from trace_tasks.experimental.certificates import export, prompts, registry
from trace_tasks.experimental.certificates.scoring import score_certificate_response
from trace_tasks.experimental.certificates.types import SCORER_VERSION, CertificateExtension, ScoreResult

NUMERIC_ADAPTERS = ("filtered_mean.table_filtered_column_mean", "weighted_sum.lever_side_torque", "path.rpg_movement_cost")
SIGNALS = {"A": "answer", "G": "grounding", "M": "semantics", "F": "format"}
GNG_MIN_TYPES = 3  # DESIGN 8 go/no-go 1(b)
# Typing / reference corruptions (element failures) also keep G and lower M, but say nothing about
# reading the scene; go/no-go 1(b) is judged on the content corruptions only.
TYPING_CORRUPTIONS = ("negative_index", "index_out_of_range", "total_missing")
TYPING_PREFIXES = ("bool_", "float_", "integral_float_")


def is_content_corruption(name: str) -> bool:
    return name not in TYPING_CORRUPTIONS and not name.startswith(TYPING_PREFIXES)


def _validator():
    try:
        import jsonschema

        def validate(value: Any, schema: Mapping[str, Any]) -> List[str]:
            return [error.message for error in jsonschema.Draft202012Validator(schema).iter_errors(value)]

        return validate, "jsonschema"
    except ImportError:  # the trace_cert env has no jsonschema: reuse the tested subset validator
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from tests.test_certificate_prompts import validate

        return validate, "tests.test_certificate_prompts.validate"


VALIDATE, VALIDATOR = _validator()


def _score(text: str, ext: CertificateExtension, adapter, *, shown: bool = False) -> Tuple[ScoreResult, float]:
    cpu0 = time.process_time()
    result = score_certificate_response(text, {"instance_id": ext.instance_id}, ext, coord_frame="px_source",
                                        adapter=adapter, shown_frame_diagnostics=shown)
    return result, (time.process_time() - cpu0) * 1000.0


def _signals(result: ScoreResult) -> List[float]:
    return [result.answer, result.grounding, result.semantics, result.format]


def _value(result: ScoreResult, key: str) -> Optional[float]:
    if key in SIGNALS:
        return float(getattr(result, SIGNALS[key]))
    found = [check.score for check in result.checks if check.check_id == key]
    return float(found[0]) if len(found) == 1 else None


def _expectations(result: ScoreResult, baseline: ScoreResult, expected: Mapping[str, Any]) -> Tuple[List[str], bool]:
    """``(problems, intended_fail)``: unmet expectations, and whether every check expected to
    fail (``"fail"`` or an exact score below 1) did fail."""
    problems: List[str] = []
    intended = True
    for key, want in expected.items():
        got = _value(result, key)
        if got is None:
            problems.append(f"{key}: not reported")
            intended = intended and not (want == "fail" or (isinstance(want, float) and want < 1.0))
            continue
        if want == "pass":
            ok = got == 1.0
        elif want == "fail":
            ok = got < 1.0
            intended = intended and ok
        elif want == "unchanged":
            base = _value(baseline, key)
            ok = base is not None and abs(got - base) <= 0.01
        else:
            ok = abs(got - float(want)) <= 1e-9
            if float(want) < 1.0:
                intended = intended and got < 1.0
        if not ok:
            problems.append(f"{key}: want {want!r} got {got:.4f}")
    return problems, intended


def _all_resolved(result: ScoreResult) -> bool:
    return all(item["status"] == "resolved" for items in result.diagnostics["identity"].values() for item in items)


def score_row(payload: Mapping[str, Any], path_cap: int = 64) -> Dict[str, Any]:
    """Every Section 5 measurement of one extension (runs in a worker)."""
    ext = CertificateExtension.from_dict(payload)
    adapter = registry.adapter_for_extension(ext)
    base = ext.base
    oracle_obj = {key: ext.sft_target[key] for key in ("annotation", "certificate", "answer")}
    oracle_text = json.dumps(oracle_obj, allow_nan=False)
    sft_text = export.make_sft_target(None, ext)

    oracle, cpu_ms = _score(oracle_text, ext, adapter)
    sft, _ = _score(sft_text, ext, adapter)
    shown, cpu_shown_ms = _score(oracle_text, ext, adapter, shown=True)
    schema_problems = VALIDATE(json.loads(sft_text), prompts.public_schema(adapter.kind))
    schema_problems += VALIDATE(json.loads(json.dumps(ext.to_dict())), prompts.private_schema())
    row: Dict[str, Any] = {
        "instance_id": ext.instance_id,
        "adapter_id": adapter.adapter_id,
        "split": base.get("split"),
        "cell": base.get("cell"),
        "holdout": base.get("holdout", ""),
        "oracle": _signals(oracle),
        "oracle_joint": oracle.joint,
        "oracle_resolved": _all_resolved(oracle),
        "sft": _signals(sft),
        "sft_resolved": _all_resolved(sft),
        "schema_problems": schema_problems,
        "verifier_ms": oracle.diagnostics["verifier_ms"],
        "cpu_ms": cpu_ms,
        "verifier_ms_shown": shown.diagnostics["verifier_ms"],
        "cpu_ms_shown": cpu_shown_ms,
        "M_shown": shown.diagnostics.get("M_shown"),
        "G_shown": shown.diagnostics.get("G_shown"),
    }
    if adapter.kind == "path":
        from trace_tasks.experimental.certificates.adapters.path import Grid, oracle_paths, steps_for

        grid = Grid.from_oracle(ext.oracle)
        routes = oracle_paths(ext.oracle, path_cap)
        bad = 0
        for route in routes:
            cert = {"kind": "path", "reachable": True,
                    "steps": steps_for(grid, [grid.index(r, c) for r, c in route]), "total": ext.oracle["opt"]}
            obj = {"annotation": oracle_obj["annotation"], "certificate": cert, "answer": ext.oracle["opt"]}
            result, _ = _score(json.dumps(obj), ext, adapter)
            bad += _signals(result) != [1.0, 1.0, 1.0, 1.0]
        row.update(n_optimal=ext.oracle["n_optimal"], optimal_checked=len(routes), optimal_failed=bad)
    row["roundtrip_ok"] = (
        row["oracle"] == [1.0, 1.0, 1.0, 1.0] and row["oracle_resolved"]
        and [row["sft"][i] for i in (0, 2, 3)] == [1.0, 1.0, 1.0] and row["sft_resolved"]
        and not schema_problems and not row.get("optimal_failed", 0)
    )
    corruptions = []
    for corruption in adapter.corruptions(ext.oracle, json.loads(oracle_text), int(base["instance_seed"])):
        result, _ = _score(json.dumps(corruption.response_obj), ext, adapter)
        problems, intended = _expectations(result, oracle, corruption.expected)
        corruptions.append({
            "name": corruption.name,
            "signals": _signals(result),
            "dG": result.grounding - oracle.grounding,
            "dM": result.semantics - oracle.semantics,
            "expect_ok": not problems,
            "intended_fail": intended,
            "problems": problems,
            "error_class": result.error_class,
        })
    row["corruptions"] = corruptions
    return row


def _job(args: Tuple[Mapping[str, Any], int]) -> Dict[str, Any]:
    return score_row(*args)


# ------------------------------------------------------------------------------------------
# aggregation


def _q(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))])


def _timing(values: Sequence[float]) -> Dict[str, Optional[float]]:
    return {"median": float(statistics.median(values)) if values else None, "p99": _q(values, 0.99),
            "max": max(values) if values else None}


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_adapter: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_adapter[row["adapter_id"]].append(row)
    roundtrip: Dict[str, Any] = {}
    timing: Dict[str, Any] = {}
    corruption_table: Dict[str, Dict[str, Any]] = {}
    go_no_go: Dict[str, Any] = {}
    for adapter_id, members in sorted(by_adapter.items()):
        failed = [r for r in members if not r["roundtrip_ok"]]
        entry = {
            "rows": len(members),
            "oracle_all_one": sum(r["oracle"] == [1.0, 1.0, 1.0, 1.0] for r in members),
            "oracle_all_resolved": sum(bool(r["oracle_resolved"]) for r in members),
            "sft_AMF_one": sum([r["sft"][i] for i in (0, 2, 3)] == [1.0, 1.0, 1.0] for r in members),
            "sft_all_resolved": sum(bool(r["sft_resolved"]) for r in members),
            "sft_G_mean": statistics.fmean(r["sft"][1] for r in members),
            "sft_G_min": min(r["sft"][1] for r in members),
            "schema_ok": sum(not r["schema_problems"] for r in members),
            "M_shown_lt1": sum(1 for r in members if r.get("M_shown") is not None and r["M_shown"] < 1.0),
            "G_shown_mean": statistics.fmean(r["G_shown"] for r in members if r.get("G_shown") is not None),
            "roundtrip_ok": len(members) - len(failed),
            "failed_instances": [r["instance_id"] for r in failed][:50],
        }
        if any("optimal_checked" in r for r in members):
            entry["optimal_paths_checked"] = sum(r.get("optimal_checked", 0) for r in members)
            entry["optimal_paths_failed"] = sum(r.get("optimal_failed", 0) for r in members)
            entry["rows_with_capped_optima"] = sum(1 for r in members if r.get("n_optimal", 0) > r.get("optimal_checked", 0))
            entry["rows_nonunique_optimum"] = sum(1 for r in members if r.get("n_optimal", 0) > 1)
        roundtrip[adapter_id] = entry
        timing[adapter_id] = {key: _timing([float(r[key]) for r in members])
                              for key in ("verifier_ms", "cpu_ms", "verifier_ms_shown", "cpu_ms_shown")}
        table: Dict[str, Dict[str, Any]] = {}
        for row in members:
            for c in row["corruptions"]:
                cell = table.setdefault(c["name"], {"rows": 0, "sum": [0.0] * 4, "expect_ok": 0, "intended_fail": 0,
                                                     "keeps_g_lowers_m": 0, "dG": 0.0, "dM": 0.0, "problems": defaultdict(int)})
                cell["rows"] += 1
                cell["sum"] = [a + b for a, b in zip(cell["sum"], c["signals"])]
                cell["expect_ok"] += bool(c["expect_ok"])
                cell["intended_fail"] += bool(c["intended_fail"])
                cell["keeps_g_lowers_m"] += abs(c["dG"]) < 0.01 and c["dM"] < 0.0
                cell["dG"] += c["dG"]
                cell["dM"] += c["dM"]
                for problem in c["problems"]:
                    cell["problems"][problem.split(":")[0]] += 1
        out_table = {}
        for name, cell in table.items():
            n = cell["rows"]
            out_table[name] = {
                "rows": n,
                "A": cell["sum"][0] / n, "G": cell["sum"][1] / n, "M": cell["sum"][2] / n, "F": cell["sum"][3] / n,
                "dG_mean": cell["dG"] / n, "dM_mean": cell["dM"] / n,
                "expect_ok_rate": cell["expect_ok"] / n,
                "intended_fail_rate": cell["intended_fail"] / n,
                "keeps_g_lowers_m_rate": cell["keeps_g_lowers_m"] / n,
                "unmet_by_key": dict(cell["problems"]),
            }
        corruption_table[adapter_id] = out_table
        if adapter_id in NUMERIC_ADAPTERS:
            strict = sorted(name for name, c in out_table.items() if c["keeps_g_lowers_m_rate"] == 1.0)
            loose = sorted(name for name, c in out_table.items() if c["keeps_g_lowers_m_rate"] >= 0.95)
            content = [name for name in strict if is_content_corruption(name)]
            go_no_go[adapter_id] = {"types_all_rows": strict, "count_all_rows": len(strict),
                                    "content_types_all_rows": content, "count_content_all_rows": len(content),
                                    "types_95pct_rows": loose, "count_95pct_rows": len(loose),
                                    "pass": len(content) >= GNG_MIN_TYPES}
    all_ms = [float(r["verifier_ms"]) for r in rows]
    all_cpu = [float(r["cpu_ms"]) for r in rows]
    return {
        "rows": len(rows),
        "roundtrip": roundtrip,
        "roundtrip_ok_total": sum(bool(r["roundtrip_ok"]) for r in rows),
        "corruptions": corruption_table,
        "corruptions_scored": sum(len(r["corruptions"]) for r in rows),
        "corruption_expectations_unmet": sum(not c["expect_ok"] for r in rows for c in r["corruptions"]),
        "go_no_go_1b": go_no_go,
        "timing": timing,
        "timing_all": {"verifier_ms": _timing(all_ms), "cpu_ms": _timing(all_cpu)},
        "validator": VALIDATOR,
    }


def markdown(summary: Mapping[str, Any], header: Mapping[str, Any]) -> str:
    short = {a: a.split(".", 1)[0] for a in summary["roundtrip"]}
    out = [f"Splits: {', '.join(header['splits'])} ({summary['rows']} rows). Scorer {header['scorer_version']}, "
           f"cert code sha256 `{header['cert_code_sha256'][:12]}`, store sha256 by split in `receipt.json`.", ""]
    out += ["**(a) Oracle round trip**", "",
            "| adapter | rows | oracle A=G=M=F=1 | all boxes resolved | SFT A=M=F=1 | SFT boxes resolved | SFT G mean (min) | schema ok | optimal paths ok |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for adapter_id, r in summary["roundtrip"].items():
        opt = f"{r['optimal_paths_checked'] - r['optimal_paths_failed']}/{r['optimal_paths_checked']}" if "optimal_paths_checked" in r else "n/a"
        out.append(f"| {short[adapter_id]} | {r['rows']} | {r['oracle_all_one']} | {r['oracle_all_resolved']} | {r['sft_AMF_one']} | "
                   f"{r['sft_all_resolved']} | {r['sft_G_mean']:.4f} ({r['sft_G_min']:.4f}) | {r['schema_ok']} | {opt} |")
    out += ["", "**(b) Corruption suite** (means over applicable rows; `expect` = every declared expectation met; "
            "`intended` = every check expected to fail did fail; `G kept, M down` = |dG| < 0.01 and M lowered)", ""]
    for adapter_id, table in summary["corruptions"].items():
        out += [f"*{short[adapter_id]}*", "",
                "| corruption | rows | A | G | M | F | expect | intended | G kept, M down |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, c in table.items():
            out.append(f"| {name} | {c['rows']} | {c['A']:.3f} | {c['G']:.3f} | {c['M']:.3f} | {c['F']:.3f} | "
                       f"{c['expect_ok_rate']:.3f} | {c['intended_fail_rate']:.3f} | {c['keeps_g_lowers_m_rate']:.3f} |")
        out.append("")
    out += ["**Go/no-go 1(b)** (>= 3 content corruption types per numeric adapter keep G (|dG| < 0.01) and lower M on "
            "every applicable row; typing/index corruptions are counted separately)", "",
            "| adapter | content types (all rows) | all types (all rows) | all types (>= 95% rows) | pass |", "|---|---|---:|---:|---|"]
    for adapter_id, g in summary["go_no_go_1b"].items():
        out.append(f"| {short[adapter_id]} | {g['count_content_all_rows']}: {', '.join(g['content_types_all_rows'])} | "
                   f"{g['count_all_rows']} | {g['count_95pct_rows']} | {'yes' if g['pass'] else 'no'} |")
    out += ["", f"**(c) Verifier time per row** (ms; oracle object; workers: {header['workers']}, shared login node; "
            "wall ms from `verifier_ms`, CPU ms from `time.process_time`)", "",
            "| adapter | wall median | wall p99 | CPU median | CPU p99 | + shown frame wall median | + shown frame wall p99 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for adapter_id, t in summary["timing"].items():
        out.append(f"| {short[adapter_id]} | {t['verifier_ms']['median']:.2f} | {t['verifier_ms']['p99']:.2f} | "
                   f"{t['cpu_ms']['median']:.2f} | {t['cpu_ms']['p99']:.2f} | {t['verifier_ms_shown']['median']:.2f} | "
                   f"{t['verifier_ms_shown']['p99']:.2f} |")
    return "\n".join(out) + "\n"


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    text = "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="export root (data/v1)")
    parser.add_argument("--out", required=True, help="results directory")
    parser.add_argument("--splits", default="e1_dev,e1_test,e2_test")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--path-cap", type=int, default=64, help="optimal paths checked per rpg row")
    parser.add_argument("--md", help="also write the Markdown tables here")
    args = parser.parse_args(argv)
    data, out = Path(args.data), Path(args.out)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    payloads: List[Dict[str, Any]] = []
    stores: Dict[str, Any] = {}
    for split in splits:
        receipt = json.loads((data / "receipts" / f"{split}.json").read_text(encoding="utf-8"))
        store = data / "private" / f"{split}.{export.STORE_NAME}"
        sha = export.file_sha256(store)
        if sha != receipt["outputs"][f"private/{split}.{export.STORE_NAME}"]["sha256"]:
            raise SystemExit(f"{store} does not match receipts/{split}.json")
        stores[split] = {"sha256": sha, "receipt_sha256": export.file_sha256(data / "receipts" / f"{split}.json")}
        payloads.extend(export.read_store(store))
    started = time.time()
    jobs = [(payload, args.path_cap) for payload in payloads]
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            rows = list(pool.imap(_job, jobs, chunksize=8))
    else:
        rows = [_job(job) for job in jobs]
    elapsed = time.time() - started
    summary = summarize(rows)
    out.mkdir(parents=True, exist_ok=True)
    corruption_rows = [{"instance_id": r["instance_id"], "adapter_id": r["adapter_id"], "split": r["split"], "cell": r["cell"], **c}
                       for r in rows for c in r["corruptions"]]
    row_records = [{k: v for k, v in r.items() if k != "corruptions"} for r in rows]
    header = {
        "splits": splits,
        "stores": stores,
        "scorer_version": SCORER_VERSION,
        "cert_code_sha256": export.cert_code_sha256(),
        "workers": args.workers,
        "path_cap": args.path_cap,
        "elapsed_s": round(elapsed, 1),
    }
    outputs = {
        "rows.jsonl": _write_jsonl(out / "rows.jsonl", row_records),
        "corruptions.jsonl": _write_jsonl(out / "corruptions.jsonl", corruption_rows),
    }
    (out / "summary.json").write_text(json.dumps({**header, **summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs["summary.json"] = export.file_sha256(out / "summary.json")
    table = markdown(summary, header)
    (out / "tables.md").write_text(table, encoding="utf-8")
    if args.md:
        Path(args.md).write_text(table, encoding="utf-8")
    receipt = {"command": "section5_cpu", **header, "outputs": outputs, "provenance": export.code_provenance()}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(table)
    ok = summary["roundtrip_ok_total"] == len(rows) and summary["corruption_expectations_unmet"] == 0
    print(json.dumps({"rows": len(rows), "roundtrip_ok": summary["roundtrip_ok_total"],
                      "corruptions_scored": summary["corruptions_scored"],
                      "expectations_unmet": summary["corruption_expectations_unmet"], "elapsed_s": round(elapsed, 1)}))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
