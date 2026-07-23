#!/usr/bin/env python3
"""Validate the TRACE paper-evaluation inputs without network access."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_ROOT = Path(__file__).resolve().parents[1]
TRACE_EVAL_ROOT = EVALUATION_ROOT / "trace_eval"
DEFAULT_SUITE = TRACE_EVAL_ROOT / "suite.v1.json"
DEFAULT_PROVENANCE = TRACE_EVAL_ROOT / "benchmark_provenance.v1.json"
DEFAULT_RESULTS = TRACE_EVAL_ROOT / "results.json"
DEFAULT_RECEIPTS = TRACE_EVAL_ROOT / "release_receipts.v1.json"
DEFAULT_POST_RUN_PATCHES = TRACE_EVAL_ROOT / "post_run_patches.v1.json"
DEFAULT_SOURCE_MAP = REPO_ROOT / "rlvr" / "source_map.v1.json"
DEFAULT_DATASET_EQUIVALENCE = REPO_ROOT / "rlvr" / "dataset_equivalence.v1.json"
TRACE_VALIDATION_ROOT = EVALUATION_ROOT / "trace_validation"
TRACE_VALIDATION_VERIFY = TRACE_VALIDATION_ROOT / "verify.py"

HISTORICAL_DATASET_REVISION = "e317b746b258630682367cc6a9d87dedd195113c"
COMPARED_DATASET_REVISION = "78f09b5482abc8e447a0a722cdf39e7d32f483c8"
REPRODUCTION_DATASET_REVISION = "4e5b54361360296a855542b40cfd8b7f81b355fe"
DATASET_RELEASE_TAG = "dataset-v1"
DATASET_EQUIVALENCE_SHA256 = (
    "40afcb59e0c67d7b7f47b78bb673a6c152320eb0656a990910ff52c634d43ae7"
)
DATASET_EQUIVALENCE_PROGRAM_SHA256 = (
    "0e775854a1494050272258d3c0931b59d649ae1fee7ab4ab96798a1ae25e9994"
)
DATASET_COMPARED_COLUMNS = (
    "images",
    "image_sizes",
    "prompt_answer",
    "prompt_answer_and_annotation",
    "answer_gt",
    "annotation_gt",
    "reward_contract",
    "instance_id",
    "domain",
    "task",
    "scene_id",
    "query_id",
    "scene_variant",
    "trace_ref",
)
DATASET_ADVISORY_COLUMN = "trace_supervision_mode"
EXPECTED_DATASET_RELEASE = {
    "repository_id": "maveryn/trace",
    "historical_training_revision": HISTORICAL_DATASET_REVISION,
    "reproduction_dataset_revision": REPRODUCTION_DATASET_REVISION,
    "release_tag": DATASET_RELEASE_TAG,
    "equivalence_receipt": {
        "path": "rlvr/dataset_equivalence.v1.json",
        "sha256": DATASET_EQUIVALENCE_SHA256,
    },
}

EXPECTED_MODELS = (
    "qwen2.5-vl-3b-base",
    "trace-qwen2.5-vl-3b",
    "qwen2.5-vl-7b-base",
    "trace-qwen2.5-vl-7b",
    "vero-qwen2.5-vl-7b",
    "game-rl-qwen2.5-vl-7b",
    "sphinx-qwen2.5-vl-7b",
    "pcgrpo-qwen2.5-vl-7b",
)
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_RESULT_REVISIONS = {
    "4178a839b689babe16f8ac36f0de7b1b2c5ef36c",
    "4ca25af7a4d7daa644e6f35e070dbed1af078321",
}
EXPECTED_RESULT_SHA256 = {
    "3a7acbf1d9baf3f32fe7ba7b4e522cc713a648fecf438eeae0319b013d8efda7",
    "578e574d68702af0b83a9c1962bd73c36f8887787cc9ac10a8f7da3d207c10ac",
    "7af92decffc261ad70bd2925632a4aeb3aa61a14cbc5974d285a264a32082c85",
}
EXPECTED_PRODUCER_REVISION = "5cea97310204b197fdacecdd83ef938c1e3b67cd"
EXPECTED_CONSOLIDATION_REVISION = "c28b706d7be5da62ee453375c9f559e99752e843"
EXPECTED_FREEZE_REVISION = "191ce9802cf04d9fa1d5435a348871c946620753"
EXPECTED_POST_RUN_REVISION = "b7e4bcf2bae88684a442834419d41d74c58e3eac"
EXPECTED_SOURCE_MAP_REVIEW_REVISION = "191ce9802cf04d9fa1d5435a348871c946620753"
EXPECTED_INTERNAL_SOURCE_MAP_SHA256 = (
    "8c7f14fbc49c2de97c0b3e05396e1fa3407e60a22f6c0e2b2c935ec31caa5b4d"
)
EXPECTED_VLMEVALKIT_REVISION = "a8b12bf1c3737a33fc1de967c202f9c592b22e86"
EXPECTED_JUDGE_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
EXPECTED_CANONICAL_FILE_SHA256 = {
    "rlvr/evaluation/requirements.txt": (
        "6d8ae0a5a6c6ff60e6f56376a5f9b2b175f906753c2796d1a2292a02d9f89c1e"
    ),
    "rlvr/evaluation/trace_eval/RESULTS.md": (
        "601730dd3cdc55d14f43e58c7d82a48d31e7614ef521758c8799fbcabd07ca5a"
    ),
    "rlvr/evaluation/trace_eval/benchmark_provenance.v1.json": (
        "6a1a8f8e39580cfc83f5c68f5fb38d9969cd73df29058a97bf5d33982d49da11"
    ),
    "rlvr/evaluation/trace_eval/results.json": (
        "569ea2463ace4c6b505d1fb168957e558667641810e90a8d9ede80f555823c98"
    ),
    "rlvr/evaluation/trace_eval/suite.v1.json": (
        "b84262bcd2243d1d2879b2e02d9b49b17052659ea767df3d7303758f4d63de71"
    ),
}
EXPECTED_SOURCE_BUNDLES = {
    "qwen2.5-vl-3b-comparison-temp06-seeds42-44-v1": {
        "revision": "4178a839b689babe16f8ac36f0de7b1b2c5ef36c",
        "manifest_sha256": "73b2799b927526b5ff15b5b0a379d8c35305f704b90ac7722bc532aa0acb7fe8",
        "result_sha256": "3a7acbf1d9baf3f32fe7ba7b4e522cc713a648fecf438eeae0319b013d8efda7",
    },
    "qwen2.5-vl-7b-comparison-temp06-seeds42-44-v1": {
        "revision": "4178a839b689babe16f8ac36f0de7b1b2c5ef36c",
        "manifest_sha256": "0334afa738e441631591197e5e16a379bc6c0380d293122b4036dd5621d47b28",
        "result_sha256": "578e574d68702af0b83a9c1962bd73c36f8887787cc9ac10a8f7da3d207c10ac",
    },
    "qwen2.5-vl-7b-rl-baselines-temp06-seeds42-44-v1": {
        "revision": "4ca25af7a4d7daa644e6f35e070dbed1af078321",
        "manifest_sha256": "ebef0bb8b07e9ec080843545874f5c2a8ca442faf8966102d2b99b65ad36672b",
        "result_sha256": "7af92decffc261ad70bd2925632a4aeb3aa61a14cbc5974d285a264a32082c85",
    },
}
EXPECTED_TRACE_VALIDATION_MAPPINGS = {
    "rlvr/evaluation/trace_validation/README.md",
    "rlvr/evaluation/trace_validation/__init__.py",
    "rlvr/evaluation/trace_validation/answer_extraction.py",
    "rlvr/evaluation/trace_validation/generate.py",
    "rlvr/evaluation/trace_validation/judge_extract.py",
    "rlvr/evaluation/trace_validation/prepare_dataset.py",
    "rlvr/evaluation/trace_validation/release_receipt.v1.json",
    "rlvr/evaluation/trace_validation/results.v1.json",
    "rlvr/evaluation/trace_validation/score.py",
    "rlvr/evaluation/trace_validation/suite.v1.json",
    "rlvr/evaluation/trace_validation/verify.py",
}
EXPECTED_TRACE_VALIDATION_REVIEW_DESTINATIONS = EXPECTED_TRACE_VALIDATION_MAPPINGS | {
    "rlvr/evaluation/scripts/prepare_trace_eval_models.py",
    "rlvr/evaluation/scripts/validate_release_inputs.py",
    "rlvr/evaluation/tests/test_release_inputs.py",
    "rlvr/train.py",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_strict(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            value[key] = child
        return value

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_url(value: Any, context: str) -> str:
    text = _require_text(value, context)
    if not text.startswith(("https://", "http://")):
        raise ValueError(f"{context} must be an HTTP(S) URL: {text!r}")
    return text


def _assert_close(actual: Any, expected: Any, context: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"{context}: {actual!r} != {expected!r}")


def _mean(values: Iterable[float]) -> float:
    result = [float(value) for value in values]
    if not result:
        raise ValueError("cannot average an empty collection")
    return statistics.fmean(result)


def _stddev(values: Iterable[float]) -> float:
    result = [float(value) for value in values]
    return statistics.stdev(result) if len(result) > 1 else 0.0


def validate_suite(suite: dict[str, Any]) -> dict[str, Any]:
    if suite.get("schema_version") != "trace-eval-suite-v1":
        raise ValueError("unexpected trace_eval_v1 suite schema")
    if suite.get("suite_id") != "trace_eval_v1" or suite.get("status") != "canonical":
        raise ValueError("suite.v1.json is not the canonical trace_eval_v1 suite")
    if suite.get("vlmevalkit", {}).get("commit") != EXPECTED_VLMEVALKIT_REVISION:
        raise ValueError("suite has the wrong VLMEvalKit revision")

    benchmarks = suite.get("benchmarks")
    if not isinstance(benchmarks, list) or len(benchmarks) != 24:
        raise ValueError("trace_eval_v1 must contain exactly 24 benchmarks")
    benchmark_ids = [row.get("key") for row in benchmarks]
    if len(set(benchmark_ids)) != 24 or any(not item for item in benchmark_ids):
        raise ValueError("suite benchmark identities must be unique and non-empty")
    for row in benchmarks:
        benchmark_id = row["key"]
        for field in (
            "display",
            "official_alias",
            "category",
            "route",
            "answer_contract",
            "score_contract",
        ):
            _require_text(row.get(field), f"{benchmark_id}.{field}")
        if not isinstance(row.get("rows"), int) or row["rows"] <= 0:
            raise ValueError(f"{benchmark_id}.rows must be a positive integer")

    categories = suite.get("categories")
    if not isinstance(categories, dict) or len(categories) != 6:
        raise ValueError("trace_eval_v1 must contain six reporting categories")
    category_members = [item for members in categories.values() for item in members]
    if any(len(members) != 4 for members in categories.values()):
        raise ValueError("each reporting category must contain four benchmarks")
    if category_members != benchmark_ids:
        raise ValueError("category membership/order must exactly cover the suite")

    routes = suite.get("routes")
    if not isinstance(routes, dict):
        raise ValueError("suite routes must be an object")
    routed = [item for members in routes.values() for item in members]
    if set(routed) != set(benchmark_ids) or len(routed) != len(benchmark_ids):
        raise ValueError("suite routes must cover each benchmark exactly once")
    for row in benchmarks:
        if row["key"] not in routes.get(row["route"], ()):
            raise ValueError(f"route registry disagrees for {row['key']}")

    rows_per_model_seed = sum(row["rows"] for row in benchmarks)
    if rows_per_model_seed != 32_805:
        raise ValueError(f"unexpected trace_eval_v1 row total: {rows_per_model_seed}")
    return {
        "benchmarks": len(benchmarks),
        "categories": len(categories),
        "rows_per_model_seed": rows_per_model_seed,
    }


def validate_provenance(
    suite: dict[str, Any], provenance: dict[str, Any]
) -> dict[str, Any]:
    if provenance.get("schema_version") != "trace_eval_benchmark_provenance_v1":
        raise ValueError("unexpected benchmark provenance schema")
    if provenance.get("suite_id") != "trace_eval_v1":
        raise ValueError("provenance is not for trace_eval_v1")
    if provenance.get("status") != "reviewed_release_input":
        raise ValueError("benchmark provenance has not been reviewed")
    if (
        provenance.get("vlmevalkit", {}).get("revision")
        != suite["vlmevalkit"]["commit"]
    ):
        raise ValueError("VLMEvalKit revision differs between suite and provenance")

    suite_rows = suite.get("benchmarks")
    entries = provenance.get("benchmarks")
    if not isinstance(suite_rows, list) or len(suite_rows) != 24:
        raise ValueError("trace_eval_v1 must contain exactly 24 benchmarks")
    if not isinstance(entries, list) or len(entries) != 24:
        raise ValueError("provenance must contain exactly 24 benchmarks")
    suite_ids = [row["key"] for row in suite_rows]
    provenance_ids = [row.get("benchmark_id") for row in entries]
    if provenance_ids != suite_ids:
        raise ValueError(
            "provenance benchmark order/identity differs from suite.v1.json"
        )
    if len(set(provenance_ids)) != 24:
        raise ValueError("duplicate benchmark provenance identity")

    license_statuses: dict[str, int] = defaultdict(int)
    for suite_row, entry in zip(suite_rows, entries, strict=True):
        benchmark_id = suite_row["key"]
        expected = {
            "public_name": suite_row["display"],
            "official_alias": suite_row["official_alias"],
            "category": suite_row["category"],
            "row_count": int(suite_row["rows"]),
        }
        for field, value in expected.items():
            if entry.get(field) != value:
                raise ValueError(
                    f"{benchmark_id}.{field}: {entry.get(field)!r} != {value!r}"
                )

        source = entry.get("source")
        if not isinstance(source, dict):
            raise ValueError(f"{benchmark_id}.source must be an object")
        _require_url(source.get("repository"), f"{benchmark_id}.source.repository")
        revision = _require_text(
            source.get("revision"), f"{benchmark_id}.source.revision"
        )
        if len(revision) < 7:
            raise ValueError(f"{benchmark_id}.source.revision is not immutable enough")
        _require_text(
            source.get("revision_kind"), f"{benchmark_id}.source.revision_kind"
        )
        _require_text(source.get("split"), f"{benchmark_id}.source.split")
        source_rows = source.get("source_row_count")
        if not isinstance(source_rows, int) or source_rows <= 0:
            raise ValueError(f"{benchmark_id}.source.source_row_count must be positive")

        artifact = entry.get("evaluation_artifact")
        if not isinstance(artifact, dict):
            raise ValueError(f"{benchmark_id}.evaluation_artifact must be an object")
        if "url" in artifact:
            _require_url(artifact["url"], f"{benchmark_id}.evaluation_artifact.url")
        else:
            _require_url(
                artifact.get("repository"),
                f"{benchmark_id}.evaluation_artifact.repository",
            )
            _require_text(
                artifact.get("repository_revision"),
                f"{benchmark_id}.evaluation_artifact.repository_revision",
            )
            _require_text(
                artifact.get("path"), f"{benchmark_id}.evaluation_artifact.path"
            )
        checksum = artifact.get("checksum")
        if checksum is not None:
            if checksum.get("algorithm") not in {"md5", "sha256"}:
                raise ValueError(f"unsupported artifact checksum for {benchmark_id}")
            _require_text(checksum.get("value"), f"{benchmark_id}.checksum.value")

        terms = entry.get("license_or_terms")
        if not isinstance(terms, dict):
            raise ValueError(f"{benchmark_id}.license_or_terms must be an object")
        status = _require_text(terms.get("status"), f"{benchmark_id}.license.status")
        expression = terms.get("expression")
        if status == "not_declared":
            if expression is not None:
                raise ValueError(
                    f"{benchmark_id} must not invent an undeclared license"
                )
        else:
            _require_text(expression, f"{benchmark_id}.license.expression")
        _require_url(terms.get("evidence"), f"{benchmark_id}.license.evidence")
        license_statuses[status] += 1

        citations = entry.get("citations")
        if not isinstance(citations, list) or not citations:
            raise ValueError(f"{benchmark_id} must have at least one citation")
        for index, citation in enumerate(citations):
            if not isinstance(citation, dict):
                raise ValueError(f"{benchmark_id}.citations[{index}] must be an object")
            _require_text(
                citation.get("title"), f"{benchmark_id}.citations[{index}].title"
            )
            _require_url(citation.get("url"), f"{benchmark_id}.citations[{index}].url")

        _require_text(entry.get("prompt_route"), f"{benchmark_id}.prompt_route")
        _require_text(entry.get("official_scorer"), f"{benchmark_id}.official_scorer")
        adapter = entry.get("adapter")
        if not isinstance(adapter, dict):
            raise ValueError(f"{benchmark_id}.adapter must be an object")
        if adapter.get("status") not in {"none", "approved", "approved_required"}:
            raise ValueError(f"{benchmark_id}.adapter has an unreviewed status")
        _require_text(adapter.get("description"), f"{benchmark_id}.adapter.description")

    return {
        "benchmarks": 24,
        "rows_per_model_seed": sum(int(row["rows"]) for row in suite_rows),
        "license_statuses": dict(sorted(license_statuses.items())),
    }


def _index_unique(
    rows: list[dict[str, Any]], fields: tuple[str, ...], label: str
) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        identity = tuple(row[field] for field in fields)
        if identity in result:
            raise ValueError(f"duplicate {label} identity: {identity}")
        result[identity] = row
    return result


def validate_results(suite: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    if results.get("schema_version") != "trace_eval_release_results_v1":
        raise ValueError("unexpected release result schema")
    if results.get("status") != "canonical":
        raise ValueError("release results are not canonical")
    if tuple(results.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("release result seeds differ from 42/43/44")
    if (
        results.get("generation", {}).get(
            "historical_final_answer_only_manifest_is_input"
        )
        is not False
    ):
        raise ValueError("historical final_answer_only_manifest must not be an input")
    if results.get("suite", {}).get("suite_id") != "trace_eval_v1":
        raise ValueError("release results are not bound to trace_eval_v1")
    if results.get("suite", {}).get("benchmark_count") != 24:
        raise ValueError("release results do not cover exactly 24 benchmarks")
    if results.get("suite", {}).get("rows_per_model_seed") != 32_805:
        raise ValueError("unexpected trace_eval_v1 row total")

    models = results.get("models")
    if not isinstance(models, list):
        raise ValueError("results.models must be an array")
    model_ids = tuple(row.get("model_id") for row in models)
    if model_ids != EXPECTED_MODELS:
        raise ValueError(f"unexpected model order/set: {model_ids}")
    for model in models:
        for field in (
            "model_id",
            "model_revision",
            "repository_id",
            "repository_revision",
            "source_run_id",
            "source_sha256",
        ):
            _require_text(model.get(field), f"{model.get('model_id')}.{field}")

    artifacts = results.get("source_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 3:
        raise ValueError("results must identify the three immutable source bundles")
    revisions = {row.get("repository_revision") for row in artifacts}
    result_hashes = {row.get("result_sha256") for row in artifacts}
    if revisions != EXPECTED_RESULT_REVISIONS:
        raise ValueError(f"unexpected result artifact revisions: {revisions}")
    if result_hashes != EXPECTED_RESULT_SHA256:
        raise ValueError(f"unexpected result artifact hashes: {result_hashes}")

    benchmark_ids = tuple(row["key"] for row in suite["benchmarks"])
    benchmark_rows = {row["key"]: int(row["rows"]) for row in suite["benchmarks"]}
    categories = suite["categories"]
    scores = results.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("results.scores must be an object")

    benchmark_scores = scores.get("benchmark_scores")
    if not isinstance(benchmark_scores, list) or len(benchmark_scores) != 576:
        raise ValueError("expected 8 x 3 x 24 benchmark score rows")
    indexed = _index_unique(
        benchmark_scores,
        ("model_id", "seed", "benchmark_id"),
        "benchmark score",
    )
    expected_identities = {
        (model_id, seed, benchmark_id)
        for model_id in EXPECTED_MODELS
        for seed in EXPECTED_SEEDS
        for benchmark_id in benchmark_ids
    }
    if set(indexed) != expected_identities:
        raise ValueError("benchmark score identities are incomplete or contain extras")
    for identity, row in indexed.items():
        if row.get("evaluated_rows") != benchmark_rows[identity[2]]:
            raise ValueError(f"row-count mismatch for {identity}")
        score = row.get("score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError(f"non-finite score for {identity}")

    category_scores = _index_unique(
        scores.get("category_scores", []),
        ("model_id", "seed", "category_name"),
        "category score",
    )
    overall_scores = _index_unique(
        scores.get("overall_scores", []),
        ("model_id", "seed"),
        "overall score",
    )
    for model_id in EXPECTED_MODELS:
        for seed in EXPECTED_SEEDS:
            benchmark_values = {
                benchmark_id: float(indexed[(model_id, seed, benchmark_id)]["score"])
                for benchmark_id in benchmark_ids
            }
            for category, members in categories.items():
                row = category_scores[(model_id, seed, category)]
                _assert_close(
                    row["score"],
                    _mean(benchmark_values[item] for item in members),
                    f"category score {(model_id, seed, category)}",
                )
            overall = _mean(benchmark_values.values())
            _assert_close(
                overall_scores[(model_id, seed)]["score"],
                overall,
                f"overall score {(model_id, seed)}",
            )

    benchmark_summaries = _index_unique(
        scores.get("benchmark_summaries", []),
        ("model_id", "benchmark_id"),
        "benchmark summary",
    )
    category_summaries = _index_unique(
        scores.get("category_summaries", []),
        ("model_id", "category_name"),
        "category summary",
    )
    overall_summaries = _index_unique(
        scores.get("overall_summaries", []),
        ("model_id",),
        "overall summary",
    )
    for model_id in EXPECTED_MODELS:
        for benchmark_id in benchmark_ids:
            values = [
                indexed[(model_id, seed, benchmark_id)]["score"]
                for seed in EXPECTED_SEEDS
            ]
            summary = benchmark_summaries[(model_id, benchmark_id)]
            _assert_close(
                summary["mean"],
                _mean(values),
                f"benchmark mean {model_id}/{benchmark_id}",
            )
            _assert_close(
                summary["stddev"],
                _stddev(values),
                f"benchmark stddev {model_id}/{benchmark_id}",
            )
        for category in categories:
            values = [
                category_scores[(model_id, seed, category)]["score"]
                for seed in EXPECTED_SEEDS
            ]
            summary = category_summaries[(model_id, category)]
            _assert_close(
                summary["mean"], _mean(values), f"category mean {model_id}/{category}"
            )
            _assert_close(
                summary["stddev"],
                _stddev(values),
                f"category stddev {model_id}/{category}",
            )
        values = [overall_scores[(model_id, seed)]["score"] for seed in EXPECTED_SEEDS]
        summary = overall_summaries[(model_id,)]
        _assert_close(summary["mean"], _mean(values), f"overall mean {model_id}")
        _assert_close(summary["stddev"], _stddev(values), f"overall stddev {model_id}")

    comparisons = results.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 6:
        raise ValueError("expected the six paper model-to-base comparisons")
    for comparison in comparisons:
        model_id = comparison["model_id"]
        base_id = comparison["base_model_id"]
        values = [
            overall_scores[(model_id, seed)]["score"]
            - overall_scores[(base_id, seed)]["score"]
            for seed in EXPECTED_SEEDS
        ]
        _assert_close(
            comparison["overall_delta"]["mean"],
            _mean(values),
            f"overall paired delta {model_id}",
        )
        _assert_close(
            comparison["overall_delta"]["stddev"],
            _stddev(values),
            f"overall paired delta stddev {model_id}",
        )

    return {
        "models": len(models),
        "seeds": len(EXPECTED_SEEDS),
        "benchmark_scores": len(benchmark_scores),
        "overall_means": {
            model_id: overall_summaries[(model_id,)]["mean"]
            for model_id in EXPECTED_MODELS
        },
    }


def validate_release_receipts(
    results: dict[str, Any], receipts: dict[str, Any]
) -> dict[str, Any]:
    if receipts.get("schema_version") != "trace_eval_release_receipts_v1":
        raise ValueError("unexpected release receipt schema")
    if receipts.get("release_id") != "trace-paper-answer-only-rlvr":
        raise ValueError("release receipts identify the wrong release")
    if receipts.get("status") != "reviewed_release_input_freeze":
        raise ValueError("release receipts have not been reviewed")

    revisions = receipts.get("code_revisions", {})
    expected_revisions = {
        "producer": EXPECTED_PRODUCER_REVISION,
        "results_consolidation": EXPECTED_CONSOLIDATION_REVISION,
        "release_freeze": EXPECTED_FREEZE_REVISION,
    }
    if revisions != expected_revisions:
        raise ValueError(f"unexpected release code revisions: {revisions}")

    canonical_files = receipts.get("canonical_files")
    if not isinstance(canonical_files, list):
        raise ValueError("canonical file receipts must be an array")
    indexed_files = {row.get("path"): row for row in canonical_files}
    if set(indexed_files) != set(EXPECTED_CANONICAL_FILE_SHA256):
        raise ValueError("canonical file receipt set is incomplete or contains extras")
    for relative, expected_hash in EXPECTED_CANONICAL_FILE_SHA256.items():
        row = indexed_files[relative]
        path = REPO_ROOT / relative
        if row.get("sha256") != expected_hash or _sha256_path(path) != expected_hash:
            raise ValueError(f"canonical file hash mismatch: {relative}")
        source_revision = row.get("source_revision")
        if source_revision not in {
            EXPECTED_PRODUCER_REVISION,
            EXPECTED_CONSOLIDATION_REVISION,
        }:
            raise ValueError(f"unexpected source revision for {relative}")
        _require_text(row.get("source_path"), f"{relative}.source_path")

    result_sources = {
        row["run_id"]: {
            "revision": row["repository_revision"],
            "manifest_sha256": row["manifest_sha256"],
            "result_sha256": row["result_sha256"],
        }
        for row in results["source_artifacts"]
    }
    if result_sources != EXPECTED_SOURCE_BUNDLES:
        raise ValueError("results.json source artifact receipts are unexpected")

    archives = receipts.get("hugging_face_source_archives")
    if not isinstance(archives, list) or len(archives) != 3:
        raise ValueError("release receipts must bind three Hugging Face run archives")
    indexed_archives = {row.get("run_id"): row for row in archives}
    if set(indexed_archives) != set(EXPECTED_SOURCE_BUNDLES):
        raise ValueError(
            "Hugging Face archive run set is incomplete or contains extras"
        )
    repository = "maveryn/trace-eval-runs"
    base_url = f"https://huggingface.co/datasets/{repository}"
    for run_id, expected in EXPECTED_SOURCE_BUNDLES.items():
        row = indexed_archives[run_id]
        if row.get("repository_id") != repository:
            raise ValueError(f"unexpected archive repository for {run_id}")
        for field in ("revision", "manifest_sha256", "result_sha256"):
            if row.get(field) != expected[field]:
                raise ValueError(f"unexpected archive {field} for {run_id}")
        expected_url = f"{base_url}/tree/{expected['revision']}/runs/{run_id}"
        if row.get("immutable_tree_url") != expected_url:
            raise ValueError(f"archive URL is not revision-pinned for {run_id}")

    environment = receipts.get("evaluation_environment", {})
    requirements_path = environment.get("requirements_path")
    if requirements_path != "rlvr/evaluation/requirements-runtime.txt":
        raise ValueError("release receipt has the wrong runtime requirements path")
    requirements_sha256 = (
        "51e44d2f7ad1cb6a0d49b1d9b6f30c8e42cb690a3fd47125b0b28449690edfb2"
    )
    if (
        environment.get("requirements_sha256") != requirements_sha256
        or _sha256_path(REPO_ROOT / requirements_path) != requirements_sha256
    ):
        raise ValueError("runtime requirements receipt is stale")
    if environment.get("composes") != [
        "rlvr/requirements-cu128.txt",
        "rlvr/evaluation/requirements.txt",
    ]:
        raise ValueError(
            "runtime requirements do not compose the documented environments"
        )
    if (
        environment.get("vlmevalkit", {}).get("revision")
        != EXPECTED_VLMEVALKIT_REVISION
    ):
        raise ValueError("release receipt has the wrong VLMEvalKit revision")
    judge = environment.get("judge", {})
    if (
        judge.get("model_id") != "Qwen/Qwen3-32B"
        or judge.get("revision") != EXPECTED_JUDGE_REVISION
    ):
        raise ValueError("release receipt has the wrong judge revision")

    historical = receipts.get("historical_exclusions")
    if not isinstance(historical, list) or not any(
        row.get("path") == "rlvr/experiments/final_answer_only_manifest.json"
        and row.get("is_release_input") is False
        for row in historical
    ):
        raise ValueError("historical manifest exclusion is missing")
    serialized = json.dumps(receipts, sort_keys=True).lower()
    if "hf-token" in serialized or "github_pat_" in serialized:
        raise ValueError("release receipts must not reference local credential files")
    return {
        "canonical_files": len(canonical_files),
        "source_archives": len(archives),
        "producer_revision": revisions["producer"],
    }


def validate_post_run_patches(patches: dict[str, Any]) -> dict[str, Any]:
    if patches.get("schema_version") != "trace_eval_post_run_patches_v1":
        raise ValueError("unexpected post-run patch ledger schema")
    if patches.get("producer_revision") != EXPECTED_PRODUCER_REVISION:
        raise ValueError("post-run ledger silently changed the producer revision")
    if patches.get("canonical_results_recomputed") is not False:
        raise ValueError(
            "post-run patches must not rewrite canonical result provenance"
        )

    rows = patches.get("patches")
    if not isinstance(rows, list):
        raise ValueError("post-run patch ledger must contain an array")
    indexed = {row.get("patch_id"): row for row in rows}
    expected_statuses = {
        "countqa-cached-answer-normalization": "approved_for_public_application",
        "tokenless-local-archive-build": "approved_for_public_application",
        "vllm-optional-server-arguments": "approved_for_public_application",
        "vllm-truncated-image-sitecustomize": "excluded_behavior_change",
        "private-archive-migration-reemit-iid": "excluded_internal_only",
    }
    if set(indexed) != set(expected_statuses):
        raise ValueError("post-run patch ledger set is incomplete or contains extras")
    for patch_id, expected_status in expected_statuses.items():
        row = indexed[patch_id]
        if row.get("source_revision") != EXPECTED_POST_RUN_REVISION:
            raise ValueError(
                f"unexpected source revision for post-run patch {patch_id}"
            )
        if row.get("review_status") != expected_status:
            raise ValueError(f"unexpected review status for post-run patch {patch_id}")
        if row.get("changes_canonical_results") is not False:
            raise ValueError(f"post-run patch rewrites canonical results: {patch_id}")
        paths = row.get("source_paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"post-run patch has no source paths: {patch_id}")
        _require_text(row.get("rationale"), f"{patch_id}.rationale")
    return {
        "patches": len(rows),
        "approved": sum(
            row["review_status"] == "approved_for_public_application" for row in rows
        ),
        "pending": sum(
            row["review_status"] == "pending_public_runtime_review" for row in rows
        ),
        "excluded": sum(row["review_status"].startswith("excluded") for row in rows),
    }


def validate_dataset_equivalence(
    path: Path = DEFAULT_DATASET_EQUIVALENCE,
) -> dict[str, Any]:
    """Validate the offline proof relating historical and reproduction data."""

    if _sha256_path(path) != DATASET_EQUIVALENCE_SHA256:
        raise ValueError("TRACE dataset equivalence receipt hash mismatch")
    raw = path.read_text(encoding="utf-8")
    machine_path_markers = ("/home/", "/" + "dev" + "/shm", ".tmp/")
    if any(marker in raw for marker in machine_path_markers):
        raise ValueError("TRACE dataset equivalence receipt contains a machine path")
    receipt = _load_strict(path)
    expected_keys = {
        "added_advisory_column",
        "aggregate",
        "compared_columns",
        "compared_current_inventory",
        "compared_current_revision",
        "comparison_program",
        "comparison_runtime",
        "current_schema",
        "original_schema",
        "original_training_revision",
        "parquet_files",
        "receipt_version",
        "repository",
        "splits",
    }
    if set(receipt) != expected_keys:
        raise ValueError("TRACE dataset equivalence receipt fields changed")
    if (
        receipt.get("receipt_version") != "trace-dataset-equivalence-v1"
        or receipt.get("repository") != "maveryn/trace"
        or receipt.get("original_training_revision") != HISTORICAL_DATASET_REVISION
        or receipt.get("compared_current_revision") != COMPARED_DATASET_REVISION
        or receipt.get("compared_columns") != list(DATASET_COMPARED_COLUMNS)
        or receipt.get("added_advisory_column") != DATASET_ADVISORY_COLUMN
        or receipt.get("comparison_program")
        != {
            "name": "verify_equivalence.py",
            "sha256": DATASET_EQUIVALENCE_PROGRAM_SHA256,
        }
    ):
        raise ValueError("TRACE dataset equivalence identity changed")

    original_schema = receipt.get("original_schema")
    current_schema = receipt.get("current_schema")
    if not isinstance(original_schema, list) or not isinstance(current_schema, list):
        raise ValueError("TRACE dataset equivalence schemas are missing")
    if [field.get("name") for field in original_schema] != list(
        DATASET_COMPARED_COLUMNS
    ):
        raise ValueError("historical TRACE dataset schema changed")
    if current_schema[:-1] != original_schema or current_schema[-1] != {
        "metadata": {},
        "name": DATASET_ADVISORY_COLUMN,
        "nullable": True,
        "type": "string",
    }:
        raise ValueError("reproduction TRACE dataset has an unexpected schema delta")

    expected_paths = [
        "data/train/trace_rlvr_train_64000_all1000_seed42-"
        f"{index:05d}-of-00016.parquet"
        for index in range(16)
    ] + ["data/validation/trace_rlvr_validation_iid_2000_all1000_seed1042.parquet"]
    files = receipt.get("parquet_files")
    if (
        not isinstance(files, list)
        or [row.get("path") for row in files] != expected_paths
    ):
        raise ValueError("TRACE dataset equivalence file inventory changed")
    current_files: dict[str, dict[str, Any]] = {}
    file_true_fields = (
        "compared_columns_equal_in_row_order",
        "existing_field_schemas_equal",
        "image_bytes_equal_in_row_order",
        "schema_metadata_only_adds_trace_supervision_mode",
        "task_distribution_equal",
    )
    for index, row in enumerate(files):
        if not isinstance(row, dict):
            raise ValueError("invalid TRACE dataset equivalence file entry")
        expected_rows = 4_000 if index < 16 else 2_000
        expected_split = "train" if index < 16 else "validation"
        if row.get("rows") != expected_rows or row.get("split") != expected_split:
            raise ValueError("TRACE dataset equivalence split rows changed")
        if any(row.get(field) is not True for field in file_true_fields):
            raise ValueError("TRACE dataset equivalence file check did not pass")
        if row.get("column_equality") != {
            column: True for column in DATASET_COMPARED_COLUMNS
        }:
            raise ValueError("TRACE dataset equivalence column check did not pass")
        mode_counts = row.get("trace_supervision_mode_counts")
        if (
            not isinstance(mode_counts, dict)
            or set(mode_counts) != {"answer", "answer_and_annotation"}
            or sum(mode_counts.values()) != expected_rows
        ):
            raise ValueError("TRACE supervision-mode counts are invalid")
        for side in ("original", "current"):
            identity = row.get(side)
            if not isinstance(identity, dict):
                raise ValueError(
                    f"TRACE dataset equivalence {side} identity is missing"
                )
            digest = identity.get("sha256")
            if (
                not isinstance(identity.get("bytes"), int)
                or not isinstance(digest, str)
                or len(digest) != 64
                or identity.get("lfs_sha256") != digest
                or identity.get("lfs_oid_matches_sha256") is not True
            ):
                raise ValueError(f"TRACE dataset equivalence {side} hash is invalid")
        current_files[str(row["path"])] = row["current"]

    expected_splits = {"train": (64_000, 64), "validation": (2_000, 2)}
    splits = receipt.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(expected_splits):
        raise ValueError("TRACE dataset equivalence split summary changed")
    for split, (rows, rows_per_task) in expected_splits.items():
        summary = splits[split]
        if (
            summary.get("rows") != rows
            or summary.get("task_count") != 1_000
            or summary.get("rows_per_task") != rows_per_task
            or summary.get("unique_instance_ids") != rows
            or summary.get("task_distribution_equal") is not True
            or summary.get("instance_ids_equal") is not True
            or summary.get("original_task_distribution_sha256")
            != summary.get("current_task_distribution_sha256")
            or sum(summary.get("trace_supervision_mode_counts", {}).values()) != rows
        ):
            raise ValueError(f"TRACE dataset equivalence {split} summary is invalid")

    aggregate = receipt.get("aggregate")
    expected_aggregate = {
        "current_differs_only_by_added_trace_supervision_mode": True,
        "existing_columns_value_equal": True,
        "existing_field_schemas_equal": True,
        "image_bytes_equal": True,
        "instance_ids_equal_and_unique": True,
        "outcome": "pass",
        "parquet_file_count": 17,
        "row_count": 66_000,
        "row_counts_equal": True,
        "row_order_equal": True,
        "schema_metadata_only_adds_trace_supervision_mode": True,
        "task_distributions_equal": True,
        "train_validation_instance_ids_disjoint": True,
    }
    if aggregate != expected_aggregate:
        raise ValueError("TRACE dataset equivalence aggregate did not pass")

    inventory = receipt.get("compared_current_inventory")
    if (
        not isinstance(inventory, dict)
        or inventory.get("revision") != COMPARED_DATASET_REVISION
    ):
        raise ValueError("TRACE dataset comparison inventory revision changed")
    inventory_rows = inventory.get("files")
    if not isinstance(inventory_rows, list):
        raise ValueError("TRACE dataset comparison inventory is missing")
    inventory_by_path: dict[str, dict[str, Any]] = {}
    for row in inventory_rows:
        repo_path = row.get("path") if isinstance(row, dict) else None
        if (
            not isinstance(repo_path, str)
            or Path(repo_path).is_absolute()
            or ".." in Path(repo_path).parts
            or repo_path in inventory_by_path
        ):
            raise ValueError("TRACE dataset comparison inventory has an unsafe path")
        inventory_by_path[repo_path] = row
    for repo_path, identity in current_files.items():
        inventory_row = inventory_by_path.get(repo_path) or {}
        if (
            inventory_row.get("size") != identity["bytes"]
            or inventory_row.get("lfs_sha256") != identity["sha256"]
        ):
            raise ValueError(f"TRACE comparison inventory differs for {repo_path}")
    return {
        "status": "ok",
        "historical_training_revision": HISTORICAL_DATASET_REVISION,
        "compared_revision": COMPARED_DATASET_REVISION,
        "reproduction_dataset_revision": REPRODUCTION_DATASET_REVISION,
        "release_tag": DATASET_RELEASE_TAG,
        "receipt_sha256": DATASET_EQUIVALENCE_SHA256,
        "parquet_files": 17,
        "rows": 66_000,
    }


def validate_trace_validation_release() -> dict[str, Any]:
    """Run the standalone TRACE validation metadata verifier offline."""

    module_name = "_trace_public_validation_release_verify"
    repo_path = str(REPO_ROOT)
    added_repo_path = repo_path not in sys.path
    if added_repo_path:
        sys.path.insert(0, repo_path)
    spec = importlib.util.spec_from_file_location(module_name, TRACE_VALIDATION_VERIFY)
    if spec is None or spec.loader is None:
        raise ValueError("could not load the TRACE validation release verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        report = module.verify_release_metadata()
    finally:
        sys.modules.pop(module_name, None)
        if added_repo_path:
            sys.path.remove(repo_path)
    if report.get("status") != "ok":
        raise ValueError("TRACE validation release verification did not pass")
    return report


def validate_source_map(source_map: dict[str, Any]) -> dict[str, Any]:
    """Validate the reviewed internal-to-public map and runtime closure."""

    if source_map.get("schema_version") != "trace-rlvr-public-file-manifest-v1":
        raise ValueError("unexpected RLVR source-map schema")
    if source_map.get("status") != "reviewed_public_curation":
        raise ValueError("RLVR source map is not a reviewed public curation")
    if (source_map.get("review_policy") or {}).get("default") != "deny":
        raise ValueError("RLVR source map must use a default-deny review policy")

    receipt = source_map.get("source_review_receipt") or {}
    if (
        receipt.get("internal_manifest_revision") != EXPECTED_SOURCE_MAP_REVIEW_REVISION
        or receipt.get("internal_manifest_sha256")
        != EXPECTED_INTERNAL_SOURCE_MAP_SHA256
    ):
        raise ValueError("RLVR source-map review receipt is stale")

    rows = source_map.get("files")
    if not isinstance(rows, list) or len(rows) != 128:
        raise ValueError("RLVR source map must contain exactly 128 reviewed mappings")
    destinations = [row.get("destination_path") for row in rows]
    if not all(
        isinstance(path, str) and path.startswith("rlvr/") for path in destinations
    ):
        raise ValueError("RLVR source map contains an invalid destination")
    if len(set(destinations)) != len(destinations):
        raise ValueError("RLVR source map contains duplicate destinations")

    valid_statuses = {"approved_exact_copy", "approved_for_public_adaptation"}
    for row in rows:
        destination = str(row["destination_path"])
        if row.get("review_status") not in valid_statuses:
            raise ValueError(f"unreviewed source-map entry: {destination}")
        for field in ("source_path", "source_revision", "source_sha256"):
            _require_text(row.get(field), f"{destination}.{field}")
        if len(str(row["source_revision"])) != 40:
            raise ValueError(f"source revision is not immutable: {destination}")
        if len(str(row["source_sha256"])) != 64:
            raise ValueError(f"source hash is invalid: {destination}")
        public_path = REPO_ROOT / destination
        if not public_path.is_file():
            raise ValueError(f"mapped public release file is missing: {destination}")
        if (
            row.get("review_status") == "approved_exact_copy"
            and _sha256_path(public_path) != row["source_sha256"]
        ):
            raise ValueError(f"exact-copy source map entry was adapted: {destination}")

    closure_path = "rlvr/evaluation/vlmevalkit_extensions/batched_vlmevalkit_qwen3vl.py"
    closure = rows[destinations.index(closure_path)]
    if (
        closure.get("component") != "trace_eval_v1_runtime_closure"
        or closure.get("source_revision") != EXPECTED_PRODUCER_REVISION
        or closure.get("source_sha256")
        != "85479464019ec93d87af97bc2254f005d57c6a524f6f13a330b43408bd939651"
    ):
        raise ValueError("generic VLMEvalKit runner closure is not source-bound")

    trace_validation_rows = {
        row["destination_path"]: row
        for row in rows
        if row["destination_path"] in EXPECTED_TRACE_VALIDATION_MAPPINGS
    }
    if set(trace_validation_rows) != EXPECTED_TRACE_VALIDATION_MAPPINGS:
        raise ValueError("TRACE validation source mappings are incomplete")
    if any(
        row.get("review_status") != "approved_for_public_adaptation"
        for row in trace_validation_rows.values()
    ):
        raise ValueError("TRACE validation contains an unreviewed source mapping")

    validation_review = source_map.get("trace_validation_review_receipt") or {}
    reviewed_destinations = sorted(EXPECTED_TRACE_VALIDATION_REVIEW_DESTINATIONS)
    reviewed_destination_sha256 = hashlib.sha256(
        json.dumps(
            reviewed_destinations,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    reviewed_mapping_projection = [
        {
            key: row[key]
            for key in (
                "destination_path",
                "source_path",
                "source_revision",
                "source_sha256",
                "review_status",
            )
        }
        for row in sorted(rows, key=lambda item: item["destination_path"])
        if row["destination_path"] in EXPECTED_TRACE_VALIDATION_REVIEW_DESTINATIONS
    ]
    reviewed_mapping_sha256 = hashlib.sha256(
        json.dumps(
            reviewed_mapping_projection,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if validation_review != {
        "documentation_review_revision": ("43123b33e7646393ee4b2c4a4bae18fa6923d642"),
        "evaluation_harness_revision": EXPECTED_POST_RUN_REVISION,
        "immutable_archive_revision": ("cf0d14aed86db2661d397ce8b68b36171873478d"),
        "reviewed_destination_count": len(reviewed_destinations),
        "reviewed_destination_set_sha256": reviewed_destination_sha256,
        "reviewed_mapping_projection_sha256": reviewed_mapping_sha256,
        "status": "reviewed_public_adaptation",
    }:
        raise ValueError("TRACE validation review receipt is stale")

    counts = source_map.get("counts") or {}
    if counts != {"exact_copy": 68, "files": 128, "public_adaptation": 60}:
        raise ValueError("RLVR source-map counts are stale")

    release_inputs = source_map.get("release_inputs") or {}
    if release_inputs.get("trace_dataset") != EXPECTED_DATASET_RELEASE:
        raise ValueError("TRACE dataset release input is stale")
    training_contract = release_inputs.get("training_contract") or {}
    if (
        training_contract.get("dataset_revision") != HISTORICAL_DATASET_REVISION
        or training_contract.get("reproduction_dataset_revision")
        != REPRODUCTION_DATASET_REVISION
        or training_contract.get("dataset_equivalence_receipt_sha256")
        != DATASET_EQUIVALENCE_SHA256
    ):
        raise ValueError("TRACE training dataset provenance is stale")
    validation_input = release_inputs.get("trace_validation") or {}
    if validation_input != {
        "evaluation_harness_revision": "b7e4bcf2bae88684a442834419d41d74c58e3eac",
        "models": 8,
        "result_rows": 8,
        "results_sha256": "1e2b78d46dbbd606210653b584ed77dc436d16bbdc0d329c5aaf55718dc37e11",
        "rows_per_model": 2000,
        "seed": 42,
        "source_revision": "cf0d14aed86db2661d397ce8b68b36171873478d",
        "suite_sha256": "f9cccdcdddb6135c16d3a9d434f985b51e4105c07ff0c74a54a71a4dfe7c85c7",
    }:
        raise ValueError("TRACE validation release-input receipt is stale")

    expected_public_authored = {
        "rlvr/README.md",
        "rlvr/dataset_equivalence.v1.json",
        "rlvr/easyr1_backend/VENDOR_MANIFEST.v1.json",
        "rlvr/environment_receipt.v1.json",
        "rlvr/evaluation/requirements-runtime.txt",
        "rlvr/evaluation/trace_validation/reproducibility_patches.v1.json",
        "rlvr/evaluation/trace_eval/post_run_patches.v1.json",
        "rlvr/evaluation/trace_eval/release_receipts.v1.json",
        "rlvr/release_files.v1.json",
        "rlvr/requirements-cu128.txt",
        "rlvr/source_map.v1.json",
    }
    public_authored = source_map.get("public_authored_files")
    if not isinstance(public_authored, list):
        raise ValueError("RLVR source map has no public-authored file boundary")
    public_paths = {row.get("path") for row in public_authored}
    if public_paths != expected_public_authored:
        raise ValueError("RLVR public-authored file boundary is stale")
    actual_files = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "rlvr").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    if actual_files != set(destinations) | public_paths:
        raise ValueError("RLVR source/public-authored mapping does not cover the tree")
    return {
        "files": len(rows),
        "exact_copy": counts["exact_copy"],
        "public_adaptation": counts["public_adaptation"],
        "public_authored": len(public_paths),
    }


def validate_release_inputs(
    *,
    suite_path: Path = DEFAULT_SUITE,
    provenance_path: Path = DEFAULT_PROVENANCE,
    results_path: Path = DEFAULT_RESULTS,
    receipts_path: Path = DEFAULT_RECEIPTS,
    post_run_patches_path: Path = DEFAULT_POST_RUN_PATCHES,
    source_map_path: Path = DEFAULT_SOURCE_MAP,
    dataset_equivalence_path: Path = DEFAULT_DATASET_EQUIVALENCE,
) -> dict[str, Any]:
    suite = _load(suite_path)
    provenance = _load(provenance_path)
    results = _load(results_path)
    receipts = _load(receipts_path)
    post_run_patches = _load(post_run_patches_path)
    source_map = _load(source_map_path)
    return {
        "status": "ok",
        "suite_id": "trace_eval_v1",
        "suite": validate_suite(suite),
        "provenance": validate_provenance(suite, provenance),
        "results": validate_results(suite, results),
        "release_receipts": validate_release_receipts(results, receipts),
        "post_run_patches": validate_post_run_patches(post_run_patches),
        "dataset_release": validate_dataset_equivalence(dataset_equivalence_path),
        "trace_validation": validate_trace_validation_release(),
        "source_map": validate_source_map(source_map),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument(
        "--post-run-patches", type=Path, default=DEFAULT_POST_RUN_PATCHES
    )
    parser.add_argument("--source-map", type=Path, default=DEFAULT_SOURCE_MAP)
    parser.add_argument(
        "--dataset-equivalence",
        type=Path,
        default=DEFAULT_DATASET_EQUIVALENCE,
    )
    args = parser.parse_args()
    report = validate_release_inputs(
        suite_path=args.suite,
        provenance_path=args.provenance,
        results_path=args.results,
        receipts_path=args.receipts,
        post_run_patches_path=args.post_run_patches,
        source_map_path=args.source_map,
        dataset_equivalence_path=args.dataset_equivalence,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
