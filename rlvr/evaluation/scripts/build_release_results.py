#!/usr/bin/env python3
"""Build the single TRACE paper-release result source from pinned HF metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUITE = REPO_ROOT / "rlvr" / "evaluation" / "trace_eval" / "suite.v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "rlvr" / "evaluation" / "trace_eval" / "results.json"
# These two values are part of the byte-frozen canonical document. They name
# the producer-tree locations recorded by the original consolidation run;
# release_receipts.v1.json maps them to the curated public destinations.
CANONICAL_SUITE_PROVENANCE_PATH = "evaluation/trace_eval/suite.v1.json"
CANONICAL_BUILDER_PROVENANCE_PATH = "scripts/build_trace_eval_release_results.py"

SOURCE_BUNDLES = (
    {
        "repository_id": "maveryn/trace-eval-runs",
        "repository_revision": "4178a839b689babe16f8ac36f0de7b1b2c5ef36c",
        "local_revision_dir": "rev-4178",
        "run_id": "qwen2.5-vl-3b-comparison-temp06-seeds42-44-v1",
        "parameter_scale": "3B",
    },
    {
        "repository_id": "maveryn/trace-eval-runs",
        "repository_revision": "4178a839b689babe16f8ac36f0de7b1b2c5ef36c",
        "local_revision_dir": "rev-4178",
        "run_id": "qwen2.5-vl-7b-comparison-temp06-seeds42-44-v1",
        "parameter_scale": "7B",
    },
    {
        "repository_id": "maveryn/trace-eval-runs",
        "repository_revision": "4ca25af7a4d7daa644e6f35e070dbed1af078321",
        "local_revision_dir": "rev-4ca25",
        "run_id": "qwen2.5-vl-7b-rl-baselines-temp06-seeds42-44-v1",
        "parameter_scale": "7B",
    },
)

MODEL_ORDER = (
    "qwen2.5-vl-3b-base",
    "trace-qwen2.5-vl-3b",
    "qwen2.5-vl-7b-base",
    "trace-qwen2.5-vl-7b",
    "vero-qwen2.5-vl-7b",
    "game-rl-qwen2.5-vl-7b",
    "sphinx-qwen2.5-vl-7b",
    "pcgrpo-qwen2.5-vl-7b",
)

MODEL_SCALE = {
    "qwen2.5-vl-3b-base": "3B",
    "trace-qwen2.5-vl-3b": "3B",
    "qwen2.5-vl-7b-base": "7B",
    "trace-qwen2.5-vl-7b": "7B",
    "vero-qwen2.5-vl-7b": "7B",
    "game-rl-qwen2.5-vl-7b": "7B",
    "sphinx-qwen2.5-vl-7b": "7B",
    "pcgrpo-qwen2.5-vl-7b": "7B",
}

COMPARISONS = (
    ("trace-qwen2.5-vl-3b", "qwen2.5-vl-3b-base"),
    ("trace-qwen2.5-vl-7b", "qwen2.5-vl-7b-base"),
    ("vero-qwen2.5-vl-7b", "qwen2.5-vl-7b-base"),
    ("game-rl-qwen2.5-vl-7b", "qwen2.5-vl-7b-base"),
    ("sphinx-qwen2.5-vl-7b", "qwen2.5-vl-7b-base"),
    ("pcgrpo-qwen2.5-vl-7b", "qwen2.5-vl-7b-base"),
)

SEEDS = (42, 43, 44)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_close(actual: float, expected: float, *, context: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"{context}: {actual!r} != {expected!r}")


def _mean(values: Iterable[float]) -> float:
    collected = [float(value) for value in values]
    if not collected:
        raise ValueError("cannot average an empty collection")
    return statistics.fmean(collected)


def _sample_stddev(values: Iterable[float]) -> float:
    collected = [float(value) for value in values]
    return statistics.stdev(collected) if len(collected) > 1 else 0.0


def _verify_metadata_files(run_root: Path, manifest: dict[str, Any]) -> None:
    for record in manifest["metadata_files"]:
        relative_path = Path(record["path"])
        path = run_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["size"]):
            raise ValueError(f"size mismatch for {path}")
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}")


def _bundle_root(inputs_root: Path, source: dict[str, str]) -> Path:
    return (
        inputs_root
        / source["local_revision_dir"]
        / "runs"
        / source["run_id"]
    )


def _source_records(inputs_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for source in SOURCE_BUNDLES:
        run_root = _bundle_root(inputs_root, source)
        manifest_path = run_root / "metadata" / "manifest.json"
        manifest = _load_json(manifest_path)
        _verify_metadata_files(run_root, manifest)

        if manifest["run_ids"] != [source["run_id"]]:
            raise ValueError(f"unexpected run ids in {manifest_path}: {manifest['run_ids']}")
        if manifest["suite_id"] != "trace_eval_v1":
            raise ValueError(f"unexpected suite in {manifest_path}")

        metadata_by_path = {record["path"]: record for record in manifest["metadata_files"]}
        result_relative = "metadata/results/benchmark_scores.json"
        result_path = run_root / result_relative
        result = _load_json(result_path)
        if result.get("schema_version") != "trace_eval_results_v1":
            raise ValueError(f"unexpected result schema in {result_path}")
        if result.get("aggregation") != "unweighted_macro_mean":
            raise ValueError(f"unexpected aggregation in {result_path}")

        run_relative = f"metadata/runs/{source['run_id']}.json"
        run_record = _load_json(run_root / run_relative)
        sources.append(
            {
                "repository_id": source["repository_id"],
                "repository_revision": source["repository_revision"],
                "run_id": source["run_id"],
                "result_path": result_relative,
                "result_sha256": metadata_by_path[result_relative]["sha256"],
                "manifest_sha256": _sha256(manifest_path),
                "source_selection_sha256": manifest["source_selection_sha256"],
                "source_slice_set_sha256": manifest["source_slice_set_sha256"],
                "run_metadata": run_record,
            }
        )

        for relative, metadata_record in sorted(metadata_by_path.items()):
            if not relative.startswith("metadata/models/"):
                continue
            model = _load_json(run_root / relative)
            model["parameter_scale"] = source["parameter_scale"]
            model["source_run_id"] = source["run_id"]
            model["source_sha256"] = metadata_record["sha256"]
            models.append(model)

        for row in result["benchmark_scores"]:
            copied = dict(row)
            copied["source_run_id"] = source["run_id"]
            rows.append(copied)

    return sources, models, rows


def _validate_model_records(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for model in models:
        model_id = model["model_id"]
        if model_id in by_id:
            raise ValueError(f"duplicate model descriptor: {model_id}")
        if MODEL_SCALE.get(model_id) != model["parameter_scale"]:
            raise ValueError(f"unexpected scale for {model_id}")
        by_id[model_id] = model
    if set(by_id) != set(MODEL_ORDER):
        raise ValueError(
            f"model set mismatch: missing={sorted(set(MODEL_ORDER) - set(by_id))}, "
            f"extra={sorted(set(by_id) - set(MODEL_ORDER))}"
        )
    return [by_id[model_id] for model_id in MODEL_ORDER]


def _validate_and_index_rows(
    rows: list[dict[str, Any]], suite: dict[str, Any]
) -> dict[tuple[str, int, str], dict[str, Any]]:
    benchmark_rows = {item["key"]: int(item["rows"]) for item in suite["benchmarks"]}
    expected = {
        (model_id, seed, benchmark_id)
        for model_id in MODEL_ORDER
        for seed in SEEDS
        for benchmark_id in benchmark_rows
    }
    indexed: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        identity = (row["model_id"], int(row["seed"]), row["benchmark_id"])
        if identity in indexed:
            raise ValueError(f"duplicate score identity: {identity}")
        if identity not in expected:
            raise ValueError(f"unexpected score identity: {identity}")
        if int(row["evaluated_rows"]) != benchmark_rows[row["benchmark_id"]]:
            raise ValueError(f"row-count mismatch for {identity}")
        if row.get("scoring_scope") != "aggregate":
            raise ValueError(f"unexpected scoring scope for {identity}")
        indexed[identity] = row
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        raise ValueError(f"missing {len(missing)} score identities; first={missing[:5]}")
    return indexed


def _recompute(
    indexed: dict[tuple[str, int, str], dict[str, Any]], suite: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    benchmark_order = [item["key"] for item in suite["benchmarks"]]
    benchmark_rows = {item["key"]: int(item["rows"]) for item in suite["benchmarks"]}
    categories = suite["categories"]

    benchmark_scores: list[dict[str, Any]] = []
    benchmark_summaries: list[dict[str, Any]] = []
    category_scores: list[dict[str, Any]] = []
    category_summaries: list[dict[str, Any]] = []
    overall_scores: list[dict[str, Any]] = []
    overall_summaries: list[dict[str, Any]] = []

    for model_id in MODEL_ORDER:
        for seed in SEEDS:
            for benchmark_id in benchmark_order:
                benchmark_scores.append(dict(indexed[(model_id, seed, benchmark_id)]))

        for benchmark_id in benchmark_order:
            values = [indexed[(model_id, seed, benchmark_id)]["score"] for seed in SEEDS]
            benchmark_summaries.append(
                {
                    "model_id": model_id,
                    "benchmark_id": benchmark_id,
                    "evaluated_rows": benchmark_rows[benchmark_id],
                    "seed_count": len(SEEDS),
                    "mean": _mean(values),
                    "stddev": _sample_stddev(values),
                }
            )

        model_overalls: list[float] = []
        category_values: dict[str, list[float]] = defaultdict(list)
        for seed in SEEDS:
            seed_values = {
                benchmark_id: float(indexed[(model_id, seed, benchmark_id)]["score"])
                for benchmark_id in benchmark_order
            }
            overall = _mean(seed_values.values())
            model_overalls.append(overall)
            overall_scores.append(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "benchmark_count": len(benchmark_order),
                    "score": overall,
                }
            )
            for category_name, category_benchmarks in categories.items():
                score = _mean(seed_values[key] for key in category_benchmarks)
                category_values[category_name].append(score)
                category_scores.append(
                    {
                        "model_id": model_id,
                        "seed": seed,
                        "category_name": category_name,
                        "benchmark_count": len(category_benchmarks),
                        "score": score,
                    }
                )
            _assert_close(
                overall,
                _mean(category_values[name][-1] for name in categories),
                context=f"category/overall equivalence for {model_id} seed {seed}",
            )

        overall_summaries.append(
            {
                "model_id": model_id,
                "benchmark_count": len(benchmark_order),
                "seed_count": len(SEEDS),
                "mean": _mean(model_overalls),
                "stddev": _sample_stddev(model_overalls),
            }
        )
        for category_name, values in category_values.items():
            category_summaries.append(
                {
                    "model_id": model_id,
                    "category_name": category_name,
                    "benchmark_count": len(categories[category_name]),
                    "seed_count": len(SEEDS),
                    "mean": _mean(values),
                    "stddev": _sample_stddev(values),
                }
            )

    return {
        "benchmark_scores": benchmark_scores,
        "benchmark_summaries": benchmark_summaries,
        "category_scores": category_scores,
        "category_summaries": category_summaries,
        "overall_scores": overall_scores,
        "overall_summaries": overall_summaries,
    }


def _comparison_records(
    indexed: dict[tuple[str, int, str], dict[str, Any]], suite: dict[str, Any]
) -> list[dict[str, Any]]:
    benchmark_order = [item["key"] for item in suite["benchmarks"]]
    records: list[dict[str, Any]] = []
    for model_id, base_model_id in COMPARISONS:
        benchmark_deltas = []
        for benchmark_id in benchmark_order:
            seed_deltas = [
                float(indexed[(model_id, seed, benchmark_id)]["score"])
                - float(indexed[(base_model_id, seed, benchmark_id)]["score"])
                for seed in SEEDS
            ]
            benchmark_deltas.append(
                {
                    "benchmark_id": benchmark_id,
                    "mean": _mean(seed_deltas),
                    "stddev": _sample_stddev(seed_deltas),
                    "seed_values": [
                        {"seed": seed, "delta": delta}
                        for seed, delta in zip(SEEDS, seed_deltas, strict=True)
                    ],
                }
            )
        overall_seed_deltas = [
            _mean(
                float(indexed[(model_id, seed, benchmark_id)]["score"])
                - float(indexed[(base_model_id, seed, benchmark_id)]["score"])
                for benchmark_id in benchmark_order
            )
            for seed in SEEDS
        ]
        records.append(
            {
                "model_id": model_id,
                "base_model_id": base_model_id,
                "parameter_scale": MODEL_SCALE[model_id],
                "paired_by": "benchmark_id_and_decoding_seed",
                "benchmark_deltas": benchmark_deltas,
                "overall_delta": {
                    "mean": _mean(overall_seed_deltas),
                    "stddev": _sample_stddev(overall_seed_deltas),
                    "seed_values": [
                        {"seed": seed, "delta": delta}
                        for seed, delta in zip(SEEDS, overall_seed_deltas, strict=True)
                    ],
                },
            }
        )
    return records


def build_release_results(inputs_root: Path, suite_path: Path) -> dict[str, Any]:
    suite = _load_json(suite_path)
    if suite.get("suite_id") != "trace_eval_v1" or suite.get("status") != "canonical":
        raise ValueError(f"not the canonical trace_eval_v1 suite: {suite_path}")
    if len(suite.get("benchmarks", [])) != 24:
        raise ValueError(f"expected 24 benchmarks in {suite_path}")

    sources, models, rows = _source_records(inputs_root)
    models = _validate_model_records(models)
    indexed = _validate_and_index_rows(rows, suite)
    scores = _recompute(indexed, suite)

    return {
        "schema_version": "trace_eval_release_results_v1",
        "release_id": "trace-paper-answer-only-rlvr",
        "status": "canonical",
        "score_unit": "percent",
        "suite": {
            "suite_id": suite["suite_id"],
            "suite_schema_version": suite["schema_version"],
            "suite_path": CANONICAL_SUITE_PROVENANCE_PATH,
            "suite_sha256": _sha256(suite_path),
            "benchmark_count": len(suite["benchmarks"]),
            "rows_per_model_seed": sum(int(item["rows"]) for item in suite["benchmarks"]),
            "aggregation": suite["aggregation"],
            "decoding": suite["generation"],
            "vlmevalkit": suite["vlmevalkit"],
        },
        "source_artifacts": sources,
        "models": models,
        "seeds": list(SEEDS),
        "scores": scores,
        "comparisons": _comparison_records(indexed, suite),
        "generation": {
            "builder": CANONICAL_BUILDER_PROVENANCE_PATH,
            "policy": "recomputed_from_immutable_sanitized_score_metadata",
            "historical_final_answer_only_manifest_is_input": False,
        },
    }


def _serialized(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs-root",
        type=Path,
        required=True,
        help="Root containing rev-4178/ and rev-4ca25/ immutable HF metadata downloads.",
    )
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the generated document differs from --output.",
    )
    args = parser.parse_args()

    document = build_release_results(args.inputs_root.resolve(), args.suite.resolve())
    rendered = _serialized(document)
    if args.check:
        if not args.output.is_file():
            raise SystemExit(f"canonical output not found: {args.output}")
        if args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"canonical output is stale: {args.output}")
        print(f"Verified canonical release results: {args.output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote canonical release results: {args.output}")


if __name__ == "__main__":
    main()
