#!/usr/bin/env python3
"""Load and validate the canonical TRACE evaluation suite manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUITE_PATH = REPO_ROOT / "rlvr" / "evaluation" / "trace_eval" / "suite.v1.json"
SCHEMA_VERSION = "trace-eval-suite-v1"
SUITE_ID = "trace_eval_v1"
ROUTE_NAMES = ("direct_score", "official_vlmevalkit", "dedicated_score")


@dataclass(frozen=True)
class TraceEvalBenchmark:
    key: str
    display: str
    official_alias: str
    category: str
    route: str
    rows: int
    answer_contract: str
    score_contract: str


@dataclass(frozen=True)
class TraceEvalSuite:
    path: Path
    manifest_sha256: str
    suite_id: str
    benchmarks: tuple[TraceEvalBenchmark, ...]
    categories: Mapping[str, tuple[str, ...]]
    routes: Mapping[str, tuple[str, ...]]
    dataset_manifest_view: str

    @property
    def benchmark_keys(self) -> tuple[str, ...]:
        return tuple(item.key for item in self.benchmarks)

    @property
    def rows_by_benchmark(self) -> dict[str, int]:
        return {item.key: item.rows for item in self.benchmarks}

    @property
    def rows_per_model_seed(self) -> int:
        return sum(item.rows for item in self.benchmarks)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(value: Any, *, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field} must be a nonempty string")
    return result


def _ordered_key_map(value: Any, *, field: str) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    result: dict[str, tuple[str, ...]] = {}
    for name, keys in value.items():
        if not isinstance(keys, list) or not all(isinstance(key, str) and key for key in keys):
            raise ValueError(f"{field}.{name} must be a list of nonempty benchmark keys")
        result[str(name)] = tuple(keys)
    return result


def load_trace_eval_suite(path: Path = DEFAULT_SUITE_PATH) -> TraceEvalSuite:
    suite_path = path.expanduser().resolve()
    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported TRACE evaluation suite schema: {payload.get('schema_version')!r}"
        )
    if payload.get("suite_id") != SUITE_ID or payload.get("status") != "canonical":
        raise ValueError("TRACE evaluation suite must identify canonical trace_eval_v1")

    raw_benchmarks = payload.get("benchmarks")
    if not isinstance(raw_benchmarks, list):
        raise ValueError("benchmarks must be a list")
    benchmarks: list[TraceEvalBenchmark] = []
    for index, item in enumerate(raw_benchmarks):
        if not isinstance(item, dict):
            raise ValueError(f"benchmarks[{index}] must be an object")
        rows = item.get("rows")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
            raise ValueError(f"benchmarks[{index}].rows must be a positive integer")
        route = _required_text(item.get("route"), field=f"benchmarks[{index}].route")
        if route not in ROUTE_NAMES:
            raise ValueError(f"benchmarks[{index}].route is unknown: {route!r}")
        benchmarks.append(
            TraceEvalBenchmark(
                key=_required_text(item.get("key"), field=f"benchmarks[{index}].key"),
                display=_required_text(item.get("display"), field=f"benchmarks[{index}].display"),
                official_alias=_required_text(
                    item.get("official_alias"),
                    field=f"benchmarks[{index}].official_alias",
                ),
                category=_required_text(item.get("category"), field=f"benchmarks[{index}].category"),
                route=route,
                rows=rows,
                answer_contract=_required_text(
                    item.get("answer_contract"),
                    field=f"benchmarks[{index}].answer_contract",
                ),
                score_contract=_required_text(
                    item.get("score_contract"),
                    field=f"benchmarks[{index}].score_contract",
                ),
            )
        )

    benchmark_keys = tuple(item.key for item in benchmarks)
    if len(benchmark_keys) != len(set(benchmark_keys)) == 24:
        raise ValueError("trace_eval_v1 must contain exactly 24 unique benchmarks")

    categories = _ordered_key_map(payload.get("categories"), field="categories")
    if len(categories) != 6 or {len(keys) for keys in categories.values()} != {4}:
        raise ValueError("trace_eval_v1 must contain six categories of four benchmarks")
    categorized = tuple(key for keys in categories.values() for key in keys)
    if categorized != benchmark_keys:
        raise ValueError("category order must flatten to the canonical benchmark order")
    category_by_key = {key: category for category, keys in categories.items() for key in keys}
    for benchmark in benchmarks:
        if category_by_key.get(benchmark.key) != benchmark.category:
            raise ValueError(f"category mismatch for {benchmark.key}")

    routes = _ordered_key_map(payload.get("routes"), field="routes")
    if tuple(routes) != ROUTE_NAMES:
        raise ValueError(f"routes must be ordered as {ROUTE_NAMES!r}")
    routed = tuple(key for keys in routes.values() for key in keys)
    if len(routed) != len(set(routed)) or set(routed) != set(benchmark_keys):
        raise ValueError("route groups must partition the canonical benchmark set")
    route_by_key = {key: route for route, keys in routes.items() for key in keys}
    for benchmark in benchmarks:
        if route_by_key.get(benchmark.key) != benchmark.route:
            raise ValueError(f"route mismatch for {benchmark.key}")

    vlmevalkit = payload.get("vlmevalkit")
    if not isinstance(vlmevalkit, dict):
        raise ValueError("vlmevalkit must be an object")
    _required_text(vlmevalkit.get("repository"), field="vlmevalkit.repository")
    commit = _required_text(vlmevalkit.get("commit"), field="vlmevalkit.commit")
    if len(commit) != 40:
        raise ValueError("vlmevalkit.commit must be an immutable git commit")

    generation = payload.get("generation")
    if not isinstance(generation, dict) or generation.get("run_set") != SUITE_ID:
        raise ValueError("generation.run_set must be trace_eval_v1")
    dataset_view = _required_text(
        generation.get("dataset_manifest_view"),
        field="generation.dataset_manifest_view",
    )
    if dataset_view != SUITE_ID:
        raise ValueError("generation.dataset_manifest_view must be trace_eval_v1")
    if sum(item.rows for item in benchmarks) != 32805:
        raise ValueError("trace_eval_v1 row contract must total 32,805 rows")

    return TraceEvalSuite(
        path=suite_path,
        manifest_sha256=sha256_file(suite_path),
        suite_id=SUITE_ID,
        benchmarks=tuple(benchmarks),
        categories=categories,
        routes=routes,
        dataset_manifest_view=dataset_view,
    )


def main() -> None:
    suite = load_trace_eval_suite()
    print(
        json.dumps(
            {
                "suite_id": suite.suite_id,
                "manifest_sha256": suite.manifest_sha256,
                "benchmarks": list(suite.benchmark_keys),
                "rows_per_model_seed": suite.rows_per_model_seed,
                "routes": {key: list(value) for key, value in suite.routes.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
