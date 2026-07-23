#!/usr/bin/env python3
"""Canonical scoring-route contracts for the 24-benchmark TRACE evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from trace_eval_suite import load_trace_eval_suite


@dataclass(frozen=True)
class TraceEvalContract:
    key: str
    output_type: str
    extraction: str
    scoring: str
    judge_role: str
    runner: str


_SUITE = load_trace_eval_suite()
_ROUTE_RUNNERS = {
    "direct_score": "run_external_benchmark_score_queue.py",
    "official_vlmevalkit": "run_official_vlmevalkit_saved_score.py:dataset.evaluate",
    "dedicated_score": "run_mme_reasoning_eval.py",
}


def _judge_role(route: str, answer_contract: str, score_contract: str) -> str:
    extraction_judge = "Qwen3-32B" in answer_contract
    scoring_judge = "Qwen3-32B" in score_contract
    if extraction_judge and scoring_judge:
        return "both"
    if extraction_judge:
        return "extraction"
    if scoring_judge:
        return "scoring"
    if route == "official_vlmevalkit":
        return "benchmark-defined"
    return "none"


CONTRACTS: tuple[TraceEvalContract, ...] = tuple(
    TraceEvalContract(
        key=benchmark.key,
        output_type="benchmark-defined",
        extraction=benchmark.answer_contract,
        scoring=benchmark.score_contract,
        judge_role=_judge_role(
            benchmark.route, benchmark.answer_contract, benchmark.score_contract
        ),
        runner=_ROUTE_RUNNERS[benchmark.route],
    )
    for benchmark in _SUITE.benchmarks
)
CONTRACT_BY_KEY = {contract.key: contract for contract in CONTRACTS}
CATEGORY_BY_KEY = {
    key: category for category, keys in _SUITE.categories.items() for key in keys
}

DIRECT_SCORE_KEYS = tuple(
    benchmark.key for benchmark in _SUITE.benchmarks if benchmark.route == "direct_score"
)
OFFICIAL_VLMEVAL_SCORE_KEYS = tuple(
    benchmark.key
    for benchmark in _SUITE.benchmarks
    if benchmark.route == "official_vlmevalkit"
)
DEDICATED_SCORE_KEYS = tuple(
    benchmark.key
    for benchmark in _SUITE.benchmarks
    if benchmark.route == "dedicated_score"
)

# The queue uses this name for saved workbooks that are handed to the pinned
# official evaluator. No generic answer-extraction campaign is exposed.
LLM_EXTRACT_SCORE_KEYS = OFFICIAL_VLMEVAL_SCORE_KEYS
