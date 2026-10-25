#!/usr/bin/env python3
"""experiments/certificates/jobs/ext_eval_lib.py -- helpers for jobs/ext_eval.sh.

DESIGN 6 "External suite", 7.4, 7.5 W5: TRACE's own 24-benchmark ``trace_eval_v1`` evaluation, run from a separate clone
of the rlvr branch (af32fe0). Every subcommand runs with the trace_eval env's python and the protocol's PYTHONPATH
(``.work/eval_deps``, ``<clone>/src``, ``rlvr/evaluation/scripts``, ``external/VLMEvalKit``, ``VLMEvalKit/scripts``),
i.e. what ``run_trace_eval.sh`` exports. Nothing here edits a protocol file: the subset modes wrap the protocol's
generation queue in-process and leave prompts, decoding, media handling and scoring untouched.

Subcommands
  subset      q25 / smoke row list from the prepared dataset manifest (blake3 rank within each benchmark)
  gen-queue   run_external_benchmark_generation_api_queue.py, optionally restricted to a row list
  model-link  materialize .work/models/<slug> from a pinned HF-cache snapshot + the protocol's provenance marker
  model-local register (or re-verify) a local merged checkpoint with the protocol's register-local marker
  judge-plan  Qwen3-32B judge GPU groups for this node, or why it cannot host the judge
  extract     TRACE deterministic answer-extraction statistics of finalized predictions (no judge)
  summary     per-benchmark and macro summaries, natural vs chart/diagram/synthetic (DESIGN 6), JSON + Markdown
  validate    completeness of generation (and scores) against the plan
  archive-code-hash  the TRACE_EVAL_CODE_HASH one stage of a run was produced under (protocol's local hf_archive)
  verify-generation  the protocol's verify_trace_eval.py --phase generation against that recorded code hash, for
              a generation made before a scoring-only post-release fix (README, external evaluation)

Subset rule (DESIGN 6, frozen): within each benchmark, every manifest row gets
``rank = blake3(json.dumps({"benchmark": <suite key>, "row_id": <row_id>}, sort_keys=True,
separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()``, where ``row_id`` is the VLMEvalKit ``index``
as a string (``"<ordinal>:<index>"`` only if a benchmark has duplicate indices; recorded per benchmark). q25 keeps the
``ceil(n / 4)`` smallest ranks of every benchmark; smoke keeps the ``k`` smallest (a prefix of q25). Two exceptions keep
the pinned scorers meaningful on a subset:
  * WeMath's official metric is defined over its 525 problem groups (a 2- or 3-step problem plus its one-step
    sub-problems, column ``ID``) with a fixed /525 denominator, so WeMath samples whole groups, ranked by
    ``row_id = "group:<ID>"`` (q25: ceil(525/4) = 132 groups); the summary rescales its score by 525/groups;
  * TableVQABench (``split``), PuzzleVQA (``category``) and VisualPuzzles (``category``, ``difficulty``) are scored with
    fixed per-stratum lookups/divisions, so every value of those columns is covered: a missing value adds its
    best-ranked row (a no-op for q25; it matters for the smoke).
q25 = 8,207 of 32,805 rows. The row list hash is sha256 over ``"<key>\\t<row_id>\\n"`` lines in suite order, then
manifest ordinal order.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SUBSET_SCHEMA = "trace-cert-ext-eval-subset-v1"
SUMMARY_SCHEMA = "trace-cert-ext-eval-summary-v1"
SUBSET_RULE = (
    "per benchmark: rank = blake3(json.dumps({'benchmark': key, 'row_id': row_id}, sort_keys=True, "
    "separators=(',', ':'), ensure_ascii=False)); q25 keeps ceil(n/4) smallest ranks, smoke keeps k smallest; "
    "row_id = str(index) (ordinal:index if a benchmark has duplicate indices)"
)

# DESIGN 6 (frozen): natural-image benchmarks; the other 15 are chart/diagram/synthetic. MMStar (mixed content) is in
# the natural macro and is also reported alone.
NATURAL_KEYS = (
    "realworldqa", "embspatial", "cvbench_3d", "erqa", "blink", "countbenchqa", "countqa", "treebench", "mmstar",
)
MIXED_ALONE_KEYS = ("mmstar",)
# Routes whose answer extraction or correctness can call the Qwen3-32B judge (suite.v1.json answer/score contracts).
# Official VLMEvalKit routes may also fall back to the judge inside dataset.evaluate (MCQ option matching).
JUDGE_EXTRACTION_KEYS = ("charxivreason", "mathvision", "mathvista", "mathverse", "mme_reasoning")

BARE_ANSWER_CHARS = 32
QWEN3_32B_WEIGHT_GIB_FALLBACK = 61.04  # 65.5 GB of bf16 safetensors at 9216db57
# Subset exceptions (see the module docstring): whole-group sampling and per-stratum coverage.
GROUP_UNITS = {"wemath": "ID"}
COVER_STRATA = {"tablevqabench": ("split",), "puzzlevqa": ("category",), "visualpuzzles": ("category", "difficulty")}
WEMATH_GROUPS = 525


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_suite():
    from trace_eval_suite import load_trace_eval_suite

    return load_trace_eval_suite()


# ============================================================================== subset


def rank_hex(benchmark: str, row_id: str) -> str:
    from blake3 import blake3

    return blake3(_canonical({"benchmark": benchmark, "row_id": row_id}).encode("utf-8")).hexdigest()


def subset_size(mode: str, n: int, per_bench: int) -> int:
    if mode == "q25":
        return math.ceil(n / 4)
    if mode == "smoke":
        return min(int(per_bench), n)
    if mode == "full":
        return n
    raise ValueError(f"unknown subset mode {mode!r}")


def load_columns(manifest: dict[str, Any], key: str, columns: Iterable[str]) -> list[dict[str, str]]:
    """Per-ordinal values of `columns` from the benchmark's prepared TSV (row order checked against the manifest)."""

    import pandas as pd

    receipt = manifest["datasets"][key]
    path = Path(receipt["metadata_files"][0]["path"])
    cols = list(columns)
    table = pd.read_csv(path, sep="\t", usecols=["index", *cols], dtype=str, keep_default_na=False)
    rows = receipt["row_media"]
    if len(table) != len(rows) or [str(i) for i in table["index"]] != [str(r["index"]) for r in rows]:
        raise ValueError(f"{key}: {path.name} row order does not match the dataset manifest")
    return [{c: str(v[c]) for c in cols} for v in table.to_dict("records")]


def build_subset(manifest: dict[str, Any], suite: Any, mode: str, per_bench: int = 4,
                 columns: dict[str, list[dict[str, str]]] | None = None) -> dict[str, Any]:
    view = suite.dataset_manifest_view
    snapshot = str((manifest.get("view_snapshot_sha256") or {}).get(view) or "")
    if len(snapshot) != 64:
        raise ValueError("dataset manifest has no complete trace_eval_v1 snapshot; finish data preparation first")
    datasets = manifest.get("datasets") or {}
    lines: list[str] = []
    out: dict[str, Any] = {
        "schema": SUBSET_SCHEMA,
        "mode": mode,
        "per_bench": int(per_bench) if mode == "smoke" else None,
        "rule": SUBSET_RULE,
        "suite_id": suite.suite_id,
        "suite_sha256": suite.manifest_sha256,
        "dataset_snapshot_sha256": snapshot,
        "benchmarks": {},
    }
    total = 0
    for bench in suite.benchmarks:
        receipt = datasets.get(bench.key) or {}
        rows = receipt.get("row_media") or []
        if receipt.get("status") != "ready" or len(rows) != bench.rows:
            raise ValueError(f"{bench.key}: manifest receipt not ready or has {len(rows)} != {bench.rows} rows")
        ids = [str(r["index"]) for r in rows]
        unique = len(set(ids)) == len(ids)
        ranked = []
        for r in rows:
            row_id = str(r["index"]) if unique else f"{int(r['ordinal'])}:{r['index']}"
            ranked.append((rank_hex(bench.key, row_id), int(r["ordinal"]), str(r["index"]),
                           str(r["source_row_hash"]), row_id))
        ranked.sort()
        entry: dict[str, Any] = {"total": len(rows), "row_id_rule": "index" if unique else "ordinal:index"}
        cols = (columns or {}).get(bench.key)
        if bench.key in GROUP_UNITS and cols is not None:
            gcol = GROUP_UNITS[bench.key]
            members: dict[str, list[int]] = {}
            for o, values in enumerate(cols):
                members.setdefault(values[gcol], []).append(o)
            units = sorted((rank_hex(bench.key, f"group:{gid}"), gid) for gid in members)
            k_units = math.ceil(len(units) / 4) if mode == "q25" else min(2 * int(per_bench), len(units))
            keep = {o for _, gid in units[:k_units] for o in members[gid]}
            chosen = [item for item in ranked if item[1] in keep]
            entry.update({"unit": f"group:{gcol}", "units_total": len(units), "units_selected": k_units})
        else:
            chosen = ranked[: subset_size(mode, len(rows), per_bench)]
        strata = COVER_STRATA.get(bench.key, ())
        if (strata or bench.key in GROUP_UNITS) and cols is None:
            raise ValueError(f"{bench.key}: the subset rule needs its TSV columns (pass columns=...)")
        if strata:
            added = 0
            for col in strata:
                have = {cols[item[1]][col] for item in chosen}
                for item in ranked:
                    value = cols[item[1]][col]
                    if value not in have:
                        chosen.append(item)
                        have.add(value)
                        added += 1
            entry.update({"cover_strata": list(strata), "added_for_coverage": added})
        chosen = sorted(set(chosen), key=lambda item: item[1])
        k = len(chosen)
        entry["selected"] = k
        entry["rows"] = [
            {"ordinal": o, "index": i, "source_row_hash": h, "row_id": rid, "rank": rk}
            for rk, o, i, h, rid in chosen
        ]
        out["benchmarks"][bench.key] = entry
        lines.extend(f"{bench.key}\t{rid}\n" for _, _, _, _, rid in chosen)
        total += k
    out["total_selected"] = total
    out["row_list_sha256"] = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    return out


def load_selection(path: Path) -> dict[str, set[tuple[int, str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SUBSET_SCHEMA:
        raise ValueError(f"{path}: not a {SUBSET_SCHEMA} row list")
    return {
        key: {(int(r["ordinal"]), str(r["index"]), str(r["source_row_hash"])) for r in entry["rows"]}
        for key, entry in payload["benchmarks"].items()
    }


def cmd_subset(args: argparse.Namespace) -> int:
    suite = _load_suite()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    needed = {**{k: [c] for k, c in GROUP_UNITS.items()}, **{k: list(v) for k, v in COVER_STRATA.items()}}
    columns = {key: load_columns(manifest, key, cols) for key, cols in needed.items()}
    sub = build_subset(manifest, suite, args.mode, args.per_bench, columns)
    out = Path(args.out)
    if args.frozen and Path(args.frozen).is_file():
        frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
        if frozen.get("row_list_sha256") != sub["row_list_sha256"]:
            print(f"FATAL: {args.mode} row list sha {sub['row_list_sha256']} != frozen {frozen.get('row_list_sha256')} "
                  f"({args.frozen}); the prepared data changed", file=sys.stderr)
            return 4
    sub["created"] = _now()
    if out.is_file():
        old = json.loads(out.read_text(encoding="utf-8"))
        if old.get("row_list_sha256") == sub["row_list_sha256"]:
            sub["created"] = old.get("created", sub["created"])
    _write_json(out, sub)
    if args.frozen and not Path(args.frozen).is_file():
        _write_json(Path(args.frozen), sub)
    print(json.dumps({"mode": args.mode, "total_selected": sub["total_selected"],
                      "row_list_sha256": sub["row_list_sha256"], "out": str(out)}))
    return 0


# ============================================================================== generation queue wrapper


def _install_row_filter(gen: Any, selection: dict[str, set[tuple[int, str, str]]]) -> None:
    """Keep the protocol's full-dataset handles (ranks, ordinals, row keys and request hashes stay exactly those of a
    full run, so subset rows are byte-compatible with full-run rows) and enqueue only the selected rows. Under
    --finalize-only the handles are then cut to the selected rows, so workbooks, summaries and archives describe exactly
    the subset; finalization finds each row through the protocol's order-free row identity."""

    original = gen._prepare_handles_and_jobs

    def prepare(args: argparse.Namespace, specs: list[Any]):
        handles, jobs = original(args, specs)
        kept = []
        for job in jobs:
            ident = (int(job.source_ordinal if job.source_ordinal is not None else job.rank),
                     str(job.row.get("index")), gen._row_hash(job.row))
            if ident in selection.get(job.spec.key, set()):
                kept.append(job)
        for key, handle in handles.items():
            wanted = selection.get(key, set())
            data = handle.dataset.data
            positions = []
            for ordinal, idx, row_hash in sorted(wanted):
                if ordinal >= len(data):
                    raise RuntimeError(f"{key}: selected ordinal {ordinal} beyond {len(data)} rows")
                row = data.iloc[ordinal].to_dict()
                if str(row.get("index")) != idx or gen._row_hash(row) != row_hash:
                    raise RuntimeError(f"{key}: selected row {ordinal}/{idx} does not match the current dataset")
                positions.append(ordinal)
            if getattr(args, "finalize_only", False):
                handle.dataset.data = data.iloc[positions].copy()
                handle.source_ordinals = positions
        print(f"[ext-eval:subset] pending rows kept {len(kept)} of {len(jobs)}; "
              f"selected {sum(len(v) for v in selection.values())} rows in {len(selection)} benchmarks", flush=True)
        return handles, kept

    gen._prepare_handles_and_jobs = prepare


def cmd_gen_queue(args: argparse.Namespace, rest: list[str]) -> int:
    import run_external_benchmark_generation_api_queue as gen

    if args.rows:
        _install_row_filter(gen, load_selection(Path(args.rows)))
    sys.argv = [gen.__file__, *rest]
    gen.main()
    return 0


# ============================================================================== models


def _hf_snapshot(hf_home: Path, repo: str, revision: str) -> Path:
    return hf_home / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / revision


def cmd_model_link(args: argparse.Namespace) -> int:
    import prepare_trace_eval_models as models

    hf_home = Path(args.hf_home)
    snap = _hf_snapshot(hf_home, args.repo, args.revision)
    if not snap.is_dir():
        print(f"FATAL: {args.repo}@{args.revision} is not in the HF cache ({snap})", file=sys.stderr)
        return 4
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    marker_path = out / models.MARKER_NAME
    def is_marker(name: str) -> bool:  # TRACE's own model repos ship the protocol marker of their registration
        return name == models.MARKER_NAME or name.startswith(".trace_model_revision.")

    src_files = sorted(p for p in snap.rglob("*") if p.is_file() and not is_marker(p.name))
    upstream_marker = (snap / models.MARKER_NAME).is_file()
    for src in src_files:
        rel = src.relative_to(snap)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() and os.readlink(dst) == str(src):
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    current = {p.relative_to(out).as_posix() for p in models._snapshot_files(out)}
    if marker_path.is_file() and not args.force:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (marker.get("slug") == args.slug and marker.get("immutable_revision") == args.revision
                and set((marker.get("file_sha256") or {})) == current):
            print(json.dumps({"path": str(out), "revision": args.revision, "source": f"{args.repo}@{args.revision}",
                              "marker": "kept"}))
            return 0
    if args.online:
        from huggingface_hub import HfApi

        info = HfApi().model_info(args.repo, revision=args.revision)
        if str(info.sha) != args.revision:
            print(f"FATAL: {args.repo}@{args.revision} resolved to {info.sha}", file=sys.stderr)
            return 4
        expected = {name for name in models._repository_file_names(info, args.repo)
                    if not is_marker(Path(name).name)}
        if expected != current:
            print(f"FATAL: cached snapshot differs from the pinned repository: missing={sorted(expected - current)} "
                  f"extra={sorted(current - expected)}", file=sys.stderr)
            return 4
    models._validate_model_shape(out)
    hashes = models._relative_hashes(out, models._snapshot_files(out))
    models._write_marker(out, {
        "schema_version": "trace-model-revision-v1",
        "slug": args.slug,
        "source": args.repo,
        "immutable_revision": args.revision,
        "resolved_commit": args.revision,
        "model_origin": "public_download",
        "materialization": "symlinks into the pinned Hugging Face cache snapshot (no second download)",
        "inventory_checked_online": bool(args.online),
        "upstream_marker_ignored": bool(upstream_marker),
        "file_count": len(hashes),
        "file_sha256": hashes,
        "registered_at": _now(),
    })
    print(json.dumps({"path": str(out), "revision": args.revision, "source": f"{args.repo}@{args.revision}",
                      "marker": "written"}))
    return 0


def cmd_model_local(args: argparse.Namespace) -> int:
    import prepare_trace_eval_models as models

    path = Path(args.path).resolve()
    marker_path = path / models.MARKER_NAME
    revision = None
    from contextlib import redirect_stdout

    with redirect_stdout(sys.stderr):  # stdout carries only the JSON result
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            rev = str(marker.get("immutable_revision") or "")
            if marker.get("slug") == args.slug and rev.startswith("sha256set:"):
                try:
                    models.verify([(args.slug, path, rev)], deep=True)
                    revision = rev
                except Exception as exc:  # stale marker: register again
                    print(f"[ext-eval:model] re-registering {path}: {exc}")
        if revision is None:
            revision = models.register_local(args.slug, path, f"trace-cert/{args.slug}@{'0' * 64}")
    source = f"trace-cert/{args.slug}@{revision.split(':', 1)[1]}"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("source") != source:
        marker["source"] = source
        models._write_marker(path, marker)
    print(json.dumps({"path": str(path), "revision": revision, "source": source}))
    return 0


# ============================================================================== judge placement


def judge_weight_gib(judge_dir: Path | None) -> float:
    if judge_dir is not None and judge_dir.is_dir():
        total = sum(p.stat().st_size for p in judge_dir.glob("*.safetensors"))
        if total > 0:
            return total / 2**30
    return QWEN3_32B_WEIGHT_GIB_FALLBACK


def judge_plan(devices: list[str], mib: list[int], weight_gib: float, util: float = 0.90,
               overhead_gib: float = 1.5, comfortable_gib: float = 16.0, minimum_gib: float = 6.0) -> dict[str, Any]:
    """Smallest tensor-parallel group (1, 2, 4, 8; Qwen3-32B has 8 KV heads) whose cards hold the bf16 weights with
    KV/activation headroom; prefers >= comfortable_gib headroom, accepts >= minimum_gib with fewer batched tokens."""

    n = len(devices)
    if n == 0 or len(mib) != n:
        return {"feasible": False, "reason": "no GPUs"}
    card = min(mib) / 1024.0
    usable = card * util - overhead_gib
    options = []
    for g in (1, 2, 4, 8):
        if g > n:
            break
        options.append((g, g * usable - weight_gib))
    for floor in (comfortable_gib, minimum_gib):
        for g, headroom in options:
            if headroom >= floor:
                groups = [devices[i * g:(i + 1) * g] for i in range(n // g)]
                return {
                    "feasible": True,
                    "tp": g,
                    "endpoints": len(groups),
                    "gpu_groups": " ".join(",".join(grp) for grp in groups),
                    "headroom_gib_per_endpoint": round(headroom, 1),
                    "tight": floor == minimum_gib,
                    "max_num_batched_tokens": 32768 if floor == comfortable_gib else 8192,
                    "weight_gib": round(weight_gib, 2),
                    "card_gib": round(card, 1),
                    "total_gib": round(card * n, 1),
                }
    return {
        "feasible": False,
        "reason": (f"{n} x {card:.0f} GiB cannot hold Qwen3-32B bf16 ({weight_gib:.1f} GiB) with >= {minimum_gib} GiB "
                   f"headroom at util {util} (best: {max((h for _, h in options), default=float('nan')):.1f} GiB)"),
        "weight_gib": round(weight_gib, 2),
        "card_gib": round(card, 1),
        "total_gib": round(card * n, 1),
    }


def cmd_judge_plan(args: argparse.Namespace) -> int:
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    mib = [int(float(x)) for x in args.mib.split(",") if x.strip()]
    plan = judge_plan(devices, mib, judge_weight_gib(Path(args.judge_dir) if args.judge_dir else None))
    print(json.dumps(plan))
    return 0


# ============================================================================== predictions / extraction


def _spec_dirs(run_root: Path, slug: str) -> dict[str, Path]:
    from benchmark_queue_lib import run_dir, spec_by_key

    suite = _load_suite()
    return {key: run_dir(spec_by_key(key), slug, run_root) for key in suite.benchmark_keys}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _cert_shaped(text: str) -> bool:
    return bool(re.search(r'"certificate"\s*:', text)) and bool(re.search(r'"answer"\s*:', text))


def extraction_stats(run_root: Path, slug: str) -> dict[str, Any]:
    from trace_benchmark_answer_parsing import extract_final_answer

    per: dict[str, Any] = {}
    totals: Counter[str] = Counter()
    for key, directory in _spec_dirs(run_root, slug).items():
        records = _read_jsonl(directory / "predictions.jsonl")
        methods: Counter[str] = Counter()
        finish: Counter[str] = Counter()
        tokens = []
        cert = 0
        for rec in records:
            text = str(rec.get("prediction") or "")
            _, method = extract_final_answer(text)
            if method == "raw" and len(text.strip()) <= BARE_ANSWER_CHARS and "\n" not in text.strip():
                method = "bare"  # the whole response is a short answer ("B", "12", "yes")
            methods[method] += 1
            finish[str(rec.get("finish_reason"))] += 1
            if rec.get("output_token_count") is not None:
                tokens.append(int(rec["output_token_count"]))
            cert += _cert_shaped(text)
        n = len(records)
        fail = methods.get("raw", 0) + methods.get("empty", 0)
        per[key] = {
            "rows": n,
            "methods": dict(methods),
            "det_fail": fail,
            "det_fail_rate": fail / n if n else None,
            "json_answer_rate": methods.get("json_answer", 0) / n if n else None,
            "cert_json": cert,
            "cert_json_rate": cert / n if n else None,
            "length_cap_rate": finish.get("length", 0) / n if n else None,
            "mean_output_tokens": statistics.fmean(tokens) if tokens else None,
            "judge_extraction_route": key in JUDGE_EXTRACTION_KEYS,
        }
        totals["rows"] += n
        totals["det_fail"] += fail
        totals["cert_json"] += cert
        totals["json_answer"] += methods.get("json_answer", 0)
        totals["length"] += finish.get("length", 0)
        if key in JUDGE_EXTRACTION_KEYS:
            totals["judge_route_rows"] += n
            totals["judge_route_det_fail"] += fail
    rows = totals["rows"]
    return {
        "extractor": "trace_benchmark_answer_parsing.extract_final_answer (answer tag, boxed, JSON 'answer', "
                     f"'answer is/:' marker) plus bare short responses (<= {BARE_ANSWER_CHARS} chars, one line); "
                     "'raw' (longer unwrapped text) and 'empty' = no deterministic answer",
        "rows": rows,
        "det_fail_rate": totals["det_fail"] / rows if rows else None,
        "json_answer_rate": totals["json_answer"] / rows if rows else None,
        "cert_json_rate": totals["cert_json"] / rows if rows else None,
        "length_cap_rate": totals["length"] / rows if rows else None,
        "judge_needed_rate_proxy": (totals["judge_route_det_fail"] / totals["judge_route_rows"]
                                    if totals["judge_route_rows"] else None),
        "judge_needed_rate_proxy_definition": (
            "rows of the judge-extraction routes (" + ", ".join(JUDGE_EXTRACTION_KEYS)
            + ") without a deterministic answer / rows of those routes"),
        "benchmarks": per,
    }


def cmd_extract(args: argparse.Namespace) -> int:
    stats = extraction_stats(Path(args.run_root), args.slug)
    stats["created"] = _now()
    _write_json(Path(args.out), stats)
    print(json.dumps({k: stats[k] for k in ("rows", "det_fail_rate", "cert_json_rate", "judge_needed_rate_proxy",
                                            "length_cap_rate")}))
    return 0


# ============================================================================== validation


def _expected_rows(plan: dict[str, Any]) -> dict[str, int]:
    return {k: int(v) for k, v in plan["expected_rows"].items()}


def validate(plan: dict[str, Any], run_root: Path, score_root: Path | None, phase: str) -> dict[str, Any]:
    from benchmark_queue_lib import extract_score_and_rows, score_path, spec_by_key

    slug = plan["slug"]
    expected = _expected_rows(plan)
    dirs = _spec_dirs(run_root, slug)
    problems: list[str] = []
    bench: dict[str, Any] = {}
    for key, want in expected.items():
        entry: dict[str, Any] = {"expected": want}
        summary_path = dirs[key] / "generation_summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            entry["generated"] = int(summary.get("rows", -1))
            ok = (int(summary.get("rows", -1)) == want and int(summary.get("expected_rows", -1)) == want
                  and "warning" not in summary)
            gen = summary.get("generation") or {}
            if int(gen.get("seed", -1)) != int(plan["seed"]) or int(gen.get("max_tokens", -1)) != 4096:
                ok = False
                problems.append(f"{key}: generation seed/max_tokens {gen.get('seed')}/{gen.get('max_tokens')}")
            preds = _read_jsonl(dirs[key] / "predictions.jsonl")
            entry["predictions"] = len(preds)
            if len(preds) != want:
                ok = False
            entry["generation_ok"] = ok
            if not ok:
                problems.append(f"{key}: generation {entry.get('generated')}/{want} rows, predictions {len(preds)}")
        except (OSError, ValueError) as exc:
            entry["generation_ok"] = False
            problems.append(f"{key}: no finalized generation ({exc.__class__.__name__})")
        if phase == "score" and score_root is not None:
            path = score_path(spec_by_key(key), slug, score_root / f"seed_{plan['seed']}" / "benchmark")
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                score, rows = extract_score_and_rows(payload)
                rows = int(payload.get("rows", rows if rows is not None else -1))
                entry.update({"score": score, "scored_rows": rows})
                entry["score_ok"] = score is not None and math.isfinite(float(score)) and rows == want
                if not entry["score_ok"]:
                    problems.append(f"{key}: score {score} rows {rows}/{want}")
            except (OSError, ValueError) as exc:
                entry["score_ok"] = False
                problems.append(f"{key}: no scores.json ({exc.__class__.__name__})")
        bench[key] = entry
    return {
        "phase": phase,
        "complete": not problems,
        "expected_total": sum(expected.values()),
        "generated_total": sum(int(b.get("predictions") or 0) for b in bench.values()),
        "problems": problems,
        "benchmarks": bench,
        "checked": _now(),
    }


def cmd_validate(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    res = validate(plan, Path(args.run_root), Path(args.score_root) if args.score_root else None, args.phase)
    if args.out:
        _write_json(Path(args.out), res)
    for line in res["problems"][:30]:
        print(f"  problem: {line}")
    print(f"validation {args.phase}: complete={res['complete']} rows {res['generated_total']}/{res['expected_total']}")
    return 0 if res["complete"] else 1


# ============================================================================== protocol verification across post-release fixes
# The protocol hashes generation and scoring code together (TRACE_EVAL_CODE_HASH enters every generation contract), so
# a scoring-only post-release fix applied after a run's generation makes verify_trace_eval.py --phase generation reject
# that generation ("generation code hash does not match the current trace_eval_v1 code contract"). The generation is
# then verified against the code hash it was produced under, which the protocol's own local archive records.


def archive_code_hashes(archive: Path, *, stage: str, slug: str, seed: int, run_id: str | None = None) -> set[str]:
    """``final25_code_hash`` values of the protocol's hf_archive descriptors of one stage, model, seed (and run id)."""
    found: set[str] = set()
    for path in sorted((archive / "ready").glob("*.ready.json")):
        try:
            desc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        ident, prov = desc.get("identity") or {}, desc.get("provenance") or {}
        if (ident.get("stage") != stage or ident.get("model_slug") != slug or str(ident.get("seed")) != str(seed)
                or (run_id is not None and ident.get("run_id") != run_id)):
            continue
        found.add(str(prov.get("final25_code_hash") or ""))
    return found


def cmd_archive_code_hash(args: argparse.Namespace) -> int:
    hashes = archive_code_hashes(Path(args.archive), stage=args.stage, slug=args.slug, seed=args.seed,
                                 run_id=args.run_id)
    only = next(iter(hashes)) if len(hashes) == 1 else ""
    if not re.fullmatch(r"[0-9a-f]{64}", only):
        print(f"no single {args.stage} code hash in {args.archive}: {sorted(hashes)}", file=sys.stderr)
        return 1
    print(only)
    return 0


def cmd_verify_generation(args: argparse.Namespace, rest: list[str]) -> int:
    """The protocol's verify_trace_eval.py --phase generation with ``--code-hash`` taken as the code the generation was
    produced under (its "matches the current code" check is pinned to that hash; every per-benchmark generation
    contract, row count, model revision, seed and dataset check runs unchanged against it)."""
    if "--phase" not in rest or rest[rest.index("--phase") + 1:][:1] != ["generation"]:
        print("verify-generation passes --phase generation to verify_trace_eval.py", file=sys.stderr)
        return 2
    import verify_trace_eval

    code_hash = args.code_hash
    verify_trace_eval.trace_eval_code_sha256 = lambda **_: code_hash
    try:
        verify_trace_eval.main([*rest, "--code-hash", code_hash])
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0


# ============================================================================== summary


def evaluator_exception_count(payload: dict[str, Any]) -> int:
    """Rows of one scores.json scored incorrect because the official per-row evaluator raised (post-release fix-2;
    only the MME-Reasoning scorer records them). 0 when the field is absent: before fix-2 such a row failed the
    slice, and the other routes have no such policy."""
    ev = payload.get("evaluation")
    return int((ev.get("evaluator_exception_scored_incorrect") if isinstance(ev, dict) else 0) or 0)


# post-release fix-3: rows scored incorrect because the judge's output was still unusable after the official 5 attempts,
# summary key -> (scores.json block, field): MME-Reasoning answer extractions and MathVerse score judgements
JUDGE_UNRESOLVED_FIELDS = {
    "extraction_failed_scored_incorrect": ("evaluation", "extraction_failed_scored_incorrect"),
    "mathverse_all_retries_failed_scored_incorrect": ("artifacts", "mathverse_all_retries_failed_scored_incorrect"),
}


def judge_unresolved_counts(payload: dict[str, Any]) -> dict[str, int]:
    """Rows of one scores.json scored incorrect under post-release fix-3, per JUDGE_UNRESOLVED_FIELDS key. 0 when the
    field is absent: before fix-3 such a row failed the slice, and the other benchmarks record none."""
    counts = {}
    for key, (block, field) in JUDGE_UNRESOLVED_FIELDS.items():
        sub = payload.get(block)
        counts[key] = int((sub.get(field) if isinstance(sub, dict) else 0) or 0)
    return counts


def summarize(plan: dict[str, Any], score_root: Path, extract: dict[str, Any] | None) -> dict[str, Any]:
    from benchmark_queue_lib import extract_score_and_rows, score_path, spec_by_key

    suite = _load_suite()
    slug, seed = plan["slug"], int(plan["seed"])
    category_of = {key: cat for cat, keys in suite.categories.items() for key in keys}
    root = score_root / f"seed_{seed}" / "benchmark"
    sub_meta: dict[str, Any] = {}
    if plan.get("rows_file") and Path(plan["rows_file"]).is_file():
        sub_meta = json.loads(Path(plan["rows_file"]).read_text(encoding="utf-8"))["benchmarks"]
    rows_out = []
    scores: dict[str, float] = {}
    for bench in suite.benchmarks:
        path = score_path(spec_by_key(bench.key), slug, root)
        entry: dict[str, Any] = {
            "key": bench.key,
            "display": bench.display,
            "category": category_of[bench.key],
            "group": "natural" if bench.key in NATURAL_KEYS else "chart_diagram_synthetic",
            "route": next((r for r, keys in suite.routes.items() if bench.key in keys), None),
            "expected_rows": int(plan["expected_rows"][bench.key]),
            "suite_rows": bench.rows,
        }
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            score, rows = extract_score_and_rows(payload)
            entry.update({"score": score, "rows": int(payload.get("rows", rows or 0)), "scores_path": str(path),
                          "evaluator_exception_scored_incorrect": evaluator_exception_count(payload),
                          **judge_unresolved_counts(payload)})
            meta = sub_meta.get(bench.key) or {}
            if (score is not None and bench.key in GROUP_UNITS and meta.get("units_selected")
                    and meta["units_selected"] != meta["units_total"]):
                # the official WeMath score divides by all 525 groups; on k sampled groups rescale the penalty
                entry["score_protocol"] = score
                score = 100.0 - (100.0 - float(score)) * meta["units_total"] / meta["units_selected"]
                entry["score"] = score
                entry["score_note"] = f"rescaled from {meta['units_selected']} of {meta['units_total']} problem groups"
            if score is not None and math.isfinite(float(score)):
                scores[bench.key] = float(score)
            table = (payload.get("scores") or {}).get("table") if isinstance(payload.get("scores"), dict) else None
            for row in table or []:  # MathVision / MathVista: rows answered by the official prefetch (no judge)
                if isinstance(row, dict) and "prefetch" in row and str(row.get("Subject", row.get("Task&Skill"))) == "Overall":
                    tot = int(row.get("tot") or 0)
                    if tot:
                        entry["judge_needed_protocol"] = (tot - int(row["prefetch"])) / tot
        else:
            entry.update({"score": None, "rows": None, "evaluator_exception_scored_incorrect": None,
                          **{key: None for key in JUDGE_UNRESOLVED_FIELDS}})
        if extract:
            ex = (extract.get("benchmarks") or {}).get(bench.key) or {}
            entry.update({k: ex.get(k) for k in ("det_fail_rate", "cert_json_rate", "length_cap_rate",
                                                 "mean_output_tokens")})
        rows_out.append(entry)

    def macro(keys: Iterable[str]) -> float | None:
        keys = list(keys)
        return statistics.fmean(scores[k] for k in keys) if keys and all(k in scores for k in keys) else None

    natural = [k for k in suite.benchmark_keys if k in NATURAL_KEYS]
    synthetic = [k for k in suite.benchmark_keys if k not in NATURAL_KEYS]
    categories = {cat: macro(keys) for cat, keys in suite.categories.items()}
    cat_vals = [v for v in categories.values() if v is not None]
    return {
        "schema": SUMMARY_SCHEMA,
        "suite": suite.suite_id,
        "model": plan["model"],
        "served": plan["slug"],
        "model_revision": plan.get("revision"),
        "seed": seed,
        "subset": plan["subset"],
        "row_list_sha256": plan.get("row_list_sha256"),
        "rows_expected": sum(int(v) for v in plan["expected_rows"].values()),
        "benchmarks_scored": len(scores),
        "complete": len(scores) == len(suite.benchmark_keys),
        "macro": {
            "overall_24": macro(suite.benchmark_keys),
            "overall_category_mean": statistics.fmean(cat_vals) if len(cat_vals) == len(categories) else None,
            "natural_9": macro(natural),
            "chart_diagram_synthetic_15": macro(synthetic),
            "mmstar": scores.get("mmstar"),
            "categories": categories,
        },
        "groups": {"natural": natural, "chart_diagram_synthetic": synthetic, "reported_alone": list(MIXED_ALONE_KEYS)},
        "extraction": ({k: extract.get(k) for k in ("extractor", "det_fail_rate", "json_answer_rate", "cert_json_rate",
                                                    "length_cap_rate", "judge_needed_rate_proxy",
                                                    "judge_needed_rate_proxy_definition")} if extract else None),
        "judge_needed_protocol": {b["key"]: b["judge_needed_protocol"] for b in rows_out if "judge_needed_protocol" in b},
        # post-release fix-2 (README, external evaluation): rows whose official per-row evaluator raised on the
        # extracted response, scored incorrect as upstream VLMEvalKit's 'Failed to evaluate' (MME-Reasoning only)
        "evaluator_exception_scored_incorrect": {
            "total": sum(int(b.get("evaluator_exception_scored_incorrect") or 0) for b in rows_out),
            "by_benchmark": {b["key"]: b.get("evaluator_exception_scored_incorrect") for b in rows_out},
        },
        # post-release fix-3 (README, external evaluation): rows whose judge output was still unusable after the
        # official 5 attempts, scored incorrect as upstream VLMEvalKit does (MME-Reasoning answer extraction,
        # MathVerse score judgement)
        **{
            key: {
                "total": sum(int(b.get(key) or 0) for b in rows_out),
                "by_benchmark": {b["key"]: b.get(key) for b in rows_out},
            }
            for key in JUDGE_UNRESOLVED_FIELDS
        },
        "benchmarks": rows_out,
        "created": _now(),
    }


def _fmt(value: Any, pct: bool = False) -> str:
    if value is None:
        return "-"
    return f"{100 * value:.1f}%" if pct else f"{float(value):.2f}"


def summary_markdown(s: dict[str, Any]) -> str:
    m = s["macro"]
    lines = [
        f"# trace_eval_v1 {s['subset']}: {s['served']} seed {s['seed']}",
        "",
        f"model `{s['model']}` revision `{s.get('model_revision')}`; rows expected {s['rows_expected']}; "
        f"benchmarks scored {s['benchmarks_scored']}/24; complete={s['complete']}"
        + (f"; row list sha256 `{s['row_list_sha256']}`" if s.get("row_list_sha256") else ""),
        "",
        "| macro | score |",
        "|---|---|",
        f"| overall (24) | {_fmt(m['overall_24'])} |",
        f"| natural (9) | {_fmt(m['natural_9'])} |",
        f"| chart/diagram/synthetic (15) | {_fmt(m['chart_diagram_synthetic_15'])} |",
        f"| MMStar (mixed, alone) | {_fmt(m['mmstar'])} |",
    ]
    for cat, val in m["categories"].items():
        lines.append(f"| {cat} | {_fmt(val)} |")
    ex = s.get("extraction") or {}
    jn = s.get("judge_needed_protocol") or {}
    if jn:
        lines += ["", "Judge-needed (protocol prefetch failed): " + ", ".join(f"{k} {_fmt(v, True)}" for k, v in jn.items()) + "."]
    if ex:
        lines += ["", f"Deterministic extraction failure {_fmt(ex.get('det_fail_rate'), True)}; judge-needed (proxy) "
                      f"{_fmt(ex.get('judge_needed_rate_proxy'), True)}; certificate-shaped JSON "
                      f"{_fmt(ex.get('cert_json_rate'), True)}; length cap {_fmt(ex.get('length_cap_rate'), True)}."]
    exc = s.get("evaluator_exception_scored_incorrect") or {}
    if exc:
        nonzero = {k: v for k, v in (exc.get("by_benchmark") or {}).items() if v}
        lines += ["", f"Official-evaluator exceptions scored incorrect ('Failed to evaluate', post-release fix-2): "
                      f"{exc.get('total', 0)}" + (" (" + ", ".join(f"{k} {v}" for k, v in nonzero.items()) + ")"
                                                  if nonzero else "") + "."]
    unresolved = [s[key] for key in JUDGE_UNRESOLVED_FIELDS if isinstance(s.get(key), dict)]
    if unresolved:  # summaries written before fix-3 have neither block
        nonzero = {}
        for block in unresolved:
            for k, v in (block.get("by_benchmark") or {}).items():
                if v:
                    nonzero[k] = nonzero.get(k, 0) + int(v)
        total = sum(int(b.get("total") or 0) for b in unresolved)
        lines += ["", "Judge output still unusable after 5 attempts, scored incorrect (MME-Reasoning extraction, "
                      f"MathVerse score judgement; post-release fix-3): {total}"
                      + (" (" + ", ".join(f"{k} {v}" for k, v in nonzero.items()) + ")" if nonzero else "") + "."]
    lines += ["", "| benchmark | group | rows | score | det. fail | cert JSON | mean tokens |", "|---|---|---|---|---|---|---|"]
    for b in s["benchmarks"]:
        lines.append(f"| {b['display']} | {'nat' if b['group'] == 'natural' else 'syn'} | {b.get('rows') or '-'}/"
                     f"{b['expected_rows']} | {_fmt(b.get('score'))} | {_fmt(b.get('det_fail_rate'), True)} | "
                     f"{_fmt(b.get('cert_json_rate'), True)} | "
                     f"{'-' if b.get('mean_output_tokens') is None else round(b['mean_output_tokens'])} |")
    return "\n".join(lines) + "\n"


def cmd_summary(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    extract = json.loads(Path(args.extract).read_text(encoding="utf-8")) if args.extract and Path(args.extract).is_file() else None
    s = summarize(plan, Path(args.score_root), extract)
    _write_json(Path(args.out_json), s)
    Path(args.out_md).write_text(summary_markdown(s), encoding="utf-8")
    m = s["macro"]
    print(f"summary: overall {_fmt(m['overall_24'])} natural {_fmt(m['natural_9'])} "
          f"synthetic {_fmt(m['chart_diagram_synthetic_15'])} mmstar {_fmt(m['mmstar'])} "
          f"({s['benchmarks_scored']}/24 scored)")
    return 0 if s["complete"] else 1


# ============================================================================== plan


def cmd_plan(args: argparse.Namespace) -> int:
    suite = _load_suite()
    if args.rows:
        sub = json.loads(Path(args.rows).read_text(encoding="utf-8"))
        expected = {k: int(v["selected"]) for k, v in sub["benchmarks"].items()}
        sha = sub["row_list_sha256"]
    else:
        expected = dict(suite.rows_by_benchmark)
        sha = None
    plan = {
        "schema": "trace-cert-ext-eval-plan-v1",
        "model": args.model,
        "slug": args.slug,
        "model_path": args.model_path,
        "revision": args.revision,
        "source": args.source,
        "seed": int(args.seed),
        "subset": args.subset,
        "rows_file": args.rows,
        "row_list_sha256": sha,
        "expected_rows": expected,
        "expected_total": sum(expected.values()),
        "updated": _now(),
    }
    _write_json(Path(args.out), plan)
    print(json.dumps({"expected_total": plan["expected_total"], "row_list_sha256": sha}))
    return 0


# ============================================================================== CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset")
    s.add_argument("--manifest", required=True)
    s.add_argument("--mode", choices=("q25", "smoke"), required=True)
    s.add_argument("--per-bench", type=int, default=4)
    s.add_argument("--out", required=True)
    s.add_argument("--frozen", help="frozen row list: written on first use, compared afterwards")
    g = sub.add_parser("gen-queue", help="args after -- go to the protocol's generation queue")
    g.add_argument("--rows")
    ml = sub.add_parser("model-link")
    ml.add_argument("--repo", required=True)
    ml.add_argument("--revision", required=True)
    ml.add_argument("--slug", required=True)
    ml.add_argument("--out", required=True)
    ml.add_argument("--hf-home", required=True)
    ml.add_argument("--online", action="store_true", help="check the file inventory against the Hub (login node)")
    ml.add_argument("--force", action="store_true")
    lo = sub.add_parser("model-local")
    lo.add_argument("--path", required=True)
    lo.add_argument("--slug", required=True)
    jp = sub.add_parser("judge-plan")
    jp.add_argument("--devices", required=True, help="comma list of CUDA_VISIBLE_DEVICES tokens")
    jp.add_argument("--mib", required=True, help="comma list of per-card memory.total MiB")
    jp.add_argument("--judge-dir")
    ex = sub.add_parser("extract")
    ex.add_argument("--run-root", required=True)
    ex.add_argument("--slug", required=True)
    ex.add_argument("--out", required=True)
    va = sub.add_parser("validate")
    va.add_argument("--plan", required=True)
    va.add_argument("--run-root", required=True)
    va.add_argument("--score-root")
    va.add_argument("--phase", choices=("generation", "score"), required=True)
    va.add_argument("--out")
    su = sub.add_parser("summary")
    su.add_argument("--plan", required=True)
    su.add_argument("--score-root", required=True)
    su.add_argument("--extract")
    su.add_argument("--out-json", required=True)
    su.add_argument("--out-md", required=True)
    ah = sub.add_parser("archive-code-hash")
    ah.add_argument("--archive", required=True)
    ah.add_argument("--stage", default="generation")
    ah.add_argument("--slug", required=True)
    ah.add_argument("--seed", type=int, required=True)
    ah.add_argument("--run-id")
    vg = sub.add_parser("verify-generation", help="args after -- go to the protocol's verify_trace_eval.py")
    vg.add_argument("--code-hash", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--model", required=True)
    pl.add_argument("--slug", required=True)
    pl.add_argument("--model-path", required=True)
    pl.add_argument("--revision", required=True)
    pl.add_argument("--source", required=True)
    pl.add_argument("--seed", type=int, required=True)
    pl.add_argument("--subset", choices=("full", "q25", "smoke"), required=True)
    pl.add_argument("--rows")
    pl.add_argument("--out", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, rest = argv[:cut], argv[cut + 1:]
    args = build_parser().parse_args(argv)
    if args.cmd == "gen-queue":
        return cmd_gen_queue(args, rest)
    if args.cmd == "verify-generation":
        return cmd_verify_generation(args, rest)
    handlers = {
        "subset": cmd_subset, "model-link": cmd_model_link, "model-local": cmd_model_local,
        "judge-plan": cmd_judge_plan, "extract": cmd_extract, "validate": cmd_validate, "summary": cmd_summary,
        "plan": cmd_plan, "archive-code-hash": cmd_archive_code_hash,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
