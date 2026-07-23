#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from benchmark_queue_lib import (  # noqa: E402
    TRACE_EVAL_V1_BENCHMARK_CATEGORIES,
    TRACE_EVAL_V1_BENCHMARKS,
    extract_score_and_rows,
    score_path,
    spec_by_key,
)


SUITE_BENCHMARKS: dict[str, tuple[str, ...]] = {
    "trace_eval_v1": TRACE_EVAL_V1_BENCHMARKS,
}
SUITE_IDS = {"trace_eval_v1": "trace_eval_v1"}
SUITE_DISPLAY_NAMES = {"trace_eval_v1": "Eval v1"}
TRACE_EVAL_SELECTION_PATH = REPO_ROOT / "rlvr" / "evaluation" / "trace_eval" / "suite.v1.json"


def _categories_for_suite(suite: str) -> dict[str, tuple[str, ...]]:
    if suite != "trace_eval_v1":
        raise ValueError(f"unsupported suite: {suite}")
    categories = {
        category: tuple(keys)
        for category, keys in TRACE_EVAL_V1_BENCHMARK_CATEGORIES.items()
    }

    selected = SUITE_BENCHMARKS[suite]
    categorized = tuple(key for keys in categories.values() for key in keys)
    if len(categorized) != len(set(categorized)) or set(categorized) != set(selected):
        raise RuntimeError(f"Category coverage does not match the {suite!r} suite")
    return categories


def _benchmark_rows_for_suite(suite: str) -> list[tuple[str, str]]:
    categories = _categories_for_suite(suite)
    category_by_key = {
        key: category
        for category, keys in categories.items()
        for key in keys
    }
    return [(category_by_key[key], key) for key in SUITE_BENCHMARKS[suite]]


def _default_title(suite: str, seeds: list[int]) -> str:
    if len(seeds) == 1:
        seed_label = "Single-Seed"
    elif len(seeds) == 3:
        seed_label = "Three-Seed"
    else:
        seed_label = f"{len(seeds)}-Seed"
    return f"TRACE {SUITE_DISPLAY_NAMES[suite]} Temp0.6 {seed_label} Results"


def _parse_model_entry(value: str) -> tuple[str, str]:
    slug, separator, label = value.partition("=")
    if not separator or not slug.strip() or not label.strip():
        raise argparse.ArgumentTypeError("Expected MODEL_SLUG=DISPLAY_LABEL")
    return slug.strip(), label.strip()


def _parse_delta(value: str) -> tuple[str, str, str]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError("Expected LABEL=MINUEND_MODEL_SLUG=SUBTRAHEND_MODEL_SLUG")
    return tuple(part.strip() for part in parts)  # type: ignore[return-value]


def _write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a strict multi-seed TRACE evaluation campaign.")
    parser.add_argument("--score-root-base", type=Path, required=True)
    parser.add_argument("--model-entry", action="append", type=_parse_model_entry, required=True)
    parser.add_argument(
        "--delta",
        action="append",
        type=_parse_delta,
        default=[],
        metavar="LABEL=MINUEND_SLUG=SUBTRAHEND_SLUG",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--suite",
        choices=tuple(SUITE_BENCHMARKS),
        default="trace_eval_v1",
        help="Benchmark coverage to summarize (default: canonical trace_eval_v1).",
    )
    parser.add_argument("--excel", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--title", help="Report title (defaults to a suite-specific title).")
    args = parser.parse_args()

    model_entries = list(dict(args.model_entry).items())
    model_labels = dict(model_entries)
    for label, minuend, subtrahend in args.delta:
        missing = [slug for slug in (minuend, subtrahend) if slug not in model_labels]
        if missing:
            parser.error(f"delta {label!r} references unknown model slug(s): {missing}")
    categories_for_suite = _categories_for_suite(args.suite)
    benchmark_rows = _benchmark_rows_for_suite(args.suite)
    suite_title = args.title or _default_title(args.suite, args.seeds)
    suite_display = SUITE_DISPLAY_NAMES[args.suite]
    seed_records: list[dict[str, Any]] = []
    missing: list[str] = []
    for seed in args.seeds:
        benchmark_root = args.score_root_base / f"seed_{seed}" / "benchmark"
        for category, benchmark_key in benchmark_rows:
            spec = spec_by_key(benchmark_key)
            for model_slug, model_label in model_entries:
                path = score_path(spec, model_slug, benchmark_root)
                if not path.exists():
                    missing.append(str(path))
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                score, rows = extract_score_and_rows(payload)
                if score is None or not math.isfinite(float(score)):
                    raise ValueError(f"Missing finite primary score in {path}")
                seed_records.append(
                    {
                        "category": category,
                        "benchmark_key": benchmark_key,
                        "benchmark": spec.display,
                        "model_slug": model_slug,
                        "model": model_label,
                        "seed": seed,
                        "rows": rows or payload.get("rows"),
                        "score": float(score),
                        "score_path": str(path),
                    }
                )
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} {suite_display} score files; first paths:\n{preview}")

    seed_values = pd.DataFrame(seed_records)
    summary_rows: list[dict[str, Any]] = []
    for category, benchmark_key in benchmark_rows:
        spec = spec_by_key(benchmark_key)
        row: dict[str, Any] = {
            "Category": category,
            "Benchmark": spec.display,
            "Rows": int(seed_values[seed_values["benchmark_key"] == benchmark_key]["rows"].dropna().max()),
        }
        for model_slug, model_label in model_entries:
            values = seed_values[
                (seed_values["benchmark_key"] == benchmark_key)
                & (seed_values["model_slug"] == model_slug)
            ].sort_values("seed")["score"].tolist()
            if len(values) != len(args.seeds):
                raise ValueError(f"Expected {len(args.seeds)} scores for {benchmark_key}/{model_slug}, found {len(values)}")
            row[f"{model_label} mean"] = statistics.fmean(values)
            row[f"{model_label} std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        for label, minuend, subtrahend in args.delta:
            row[label] = row[f"{model_labels[minuend]} mean"] - row[f"{model_labels[subtrahend]} mean"]
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    average: dict[str, Any] = {"Category": "Overall", "Benchmark": "Average", "Rows": None}
    for model_slug, model_label in model_entries:
        per_seed = (
            seed_values[seed_values["model_slug"] == model_slug]
            .groupby("seed", sort=True)["score"]
            .mean()
            .tolist()
        )
        average[f"{model_label} mean"] = statistics.fmean(per_seed)
        average[f"{model_label} std"] = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
    for label, minuend, subtrahend in args.delta:
        average[label] = average[f"{model_labels[minuend]} mean"] - average[f"{model_labels[subtrahend]} mean"]
    summary = pd.concat([summary, pd.DataFrame([average])], ignore_index=True)

    category_rows: list[dict[str, Any]] = []
    for category, keys in categories_for_suite.items():
        record: dict[str, Any] = {"Category": category, "Benchmarks": len(keys)}
        for model_slug, model_label in model_entries:
            per_seed = (
                seed_values[
                    (seed_values["category"] == category)
                    & (seed_values["model_slug"] == model_slug)
                ]
                .groupby("seed", sort=True)["score"]
                .mean()
                .tolist()
            )
            record[f"{model_label} mean"] = statistics.fmean(per_seed)
            record[f"{model_label} std"] = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
        for label, minuend, subtrahend in args.delta:
            record[label] = (
                record[f"{model_labels[minuend]} mean"]
                - record[f"{model_labels[subtrahend]} mean"]
            )
        category_rows.append(record)
    category_summary = pd.DataFrame(category_rows)

    metadata_rows = [
        {"key": "suite", "value": SUITE_IDS[args.suite]},
        {"key": "suite_view", "value": args.suite},
        {"key": "benchmark_count", "value": len(benchmark_rows)},
        {"key": "rows_per_model_seed", "value": int(summary.iloc[:-1]["Rows"].sum())},
        {"key": "seeds", "value": ",".join(map(str, args.seeds))},
        {
            "key": "decoding",
            "value": (
                "temperature=0.6, top_p=1, top_k=-1, no penalties, max_tokens=4096"
            ),
        },
        {"key": "judge", "value": "Qwen/Qwen3-32B, temperature=0"},
        {"key": "score_root_base", "value": str(args.score_root_base)},
    ]
    selection_manifest_sha256 = ""
    selection_path: Path | None = None
    source_contract_sha256 = ""
    if args.suite == "trace_eval_v1":
        selection_path = TRACE_EVAL_SELECTION_PATH
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection_manifest_sha256 = hashlib.sha256(selection_path.read_bytes()).hexdigest()
        source_contract = selection.get("source_contract")
        source_contract_sha256 = (
            str(source_contract["sha256"])
            if isinstance(source_contract, dict)
            else ""
        )
        metadata_rows.extend(
            [
                {"key": "selection_manifest", "value": str(selection_path)},
                {"key": "selection_manifest_sha256", "value": selection_manifest_sha256},
                *(
                    [{"key": "source_contract_sha256", "value": source_contract_sha256}]
                    if source_contract_sha256
                    else []
                ),
            ]
        )
    metadata = pd.DataFrame(
        metadata_rows
        + [
            {"key": f"model/{slug}", "value": label}
            for slug, label in model_entries
        ]
    )
    categories = pd.DataFrame(
        [
            {"Category": category, "Benchmarks": ", ".join(spec_by_key(key).display for key in keys), "Count": len(keys)}
            for category, keys in categories_for_suite.items()
        ]
    )
    _write_excel(
        args.excel,
        {
            "mean_std": summary,
            "seed_values": seed_values,
            "category_summary": category_summary,
            "categories": categories,
            "metadata": metadata,
        },
    )

    lines = [f"# {suite_title}", "", f"Seeds: `{', '.join(map(str, args.seeds))}`", ""]
    category_headers = ["Category", "Benchmarks"] + [label for _, label in model_entries] + [
        label for label, _, _ in args.delta
    ]
    lines.append("## Category Means")
    lines.append("")
    lines.append("| " + " | ".join(category_headers) + " |")
    lines.append("|" + "|".join(["---"] * len(category_headers)) + "|")
    for _, row in category_summary.iterrows():
        values = [str(row["Category"]), str(int(row["Benchmarks"]))]
        for _, label in model_entries:
            values.append(f"{_fmt(float(row[f'{label} mean']))} +/- {_fmt(float(row[f'{label} std']))}")
        for label, _, _ in args.delta:
            values.append(_fmt(float(row[label])))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Benchmark Means", ""])

    headers = ["Category", "Benchmark", "Rows"] + [label for _, label in model_entries] + [
        label for label, _, _ in args.delta
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for _, row in summary.iterrows():
        values = [str(row["Category"]), str(row["Benchmark"]), "" if pd.isna(row["Rows"]) else str(int(row["Rows"]))]
        for _, label in model_entries:
            values.append(f"{_fmt(float(row[f'{label} mean']))} +/- {_fmt(float(row[f'{label} std']))}")
        for label, _, _ in args.delta:
            values.append(_fmt(float(row[label])))
        lines.append("| " + " | ".join(values) + " |")
    scoring_note = "Scoring: the pinned benchmark routes selected by canonical trace_eval_v1."
    decoding_note = "Decoding: temperature 0.6, top-p 1, top-k -1, no penalties, maximum 4096 generated tokens."
    selection_note = (
        f"Selection manifest SHA-256: `{selection_manifest_sha256}`."
        if selection_manifest_sha256
        else ""
    )
    lines.extend(
        [
            "",
            decoding_note,
            scoring_note,
            *([selection_note] if selection_note else []),
        ]
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log_tag = "trace-eval"
    print(f"[{log_tag}-summary:done] rows={len(summary_rows)} models={len(model_entries)} excel={args.excel} markdown={args.markdown}")


if __name__ == "__main__":
    main()
