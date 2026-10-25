"""CPU tests for jobs/ext_eval_lib.py (trace_eval_v1 external suite helpers; DESIGN 6, 7.5 W5).

Run from the repository root:  python -m pytest experiments/certificates/tests_ext -q
The suite-file checks use $TRACE_RLVR_ROOT (any checkout of the rlvr branch at af32fe0) and are skipped without it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

pytest.importorskip("blake3")

JOBS = Path(__file__).resolve().parents[1] / "jobs"
spec = importlib.util.spec_from_file_location("ext_eval_lib", JOBS / "ext_eval_lib.py")
lib = importlib.util.module_from_spec(spec)
sys.modules["ext_eval_lib"] = lib
spec.loader.exec_module(lib)


@dataclass
class Bench:
    key: str
    rows: int
    display: str = ""


@dataclass
class Suite:
    benchmarks: tuple
    suite_id: str = "trace_eval_v1"
    manifest_sha256: str = "0" * 64
    dataset_manifest_view: str = "trace_eval_v1"
    categories: dict = field(default_factory=dict)

    @property
    def benchmark_keys(self):
        return tuple(b.key for b in self.benchmarks)


def _manifest(sizes: dict[str, int], dup: str | None = None) -> dict:
    datasets = {}
    for key, n in sizes.items():
        rows = []
        for o in range(n):
            index = 7 if (key == dup and o in (3, 5)) else 1000 + o
            rows.append({"ordinal": o, "index": index, "source_row_hash": hashlib.sha256(f"{key}{o}".encode()).hexdigest()})
        datasets[key] = {"status": "ready", "row_media": rows}
    return {"view_snapshot_sha256": {"trace_eval_v1": "a" * 64}, "datasets": datasets}


def test_rank_rule_is_the_documented_blake3_of_canonical_json():
    from blake3 import blake3

    want = blake3(b'{"benchmark":"mmstar","row_id":"12"}').hexdigest()
    assert lib.rank_hex("mmstar", "12") == want


def test_q25_and_smoke_are_deterministic_stratified_and_nested():
    sizes = {"aa": 10, "bb": 7, "cc": 1}
    suite = Suite(tuple(Bench(k, n) for k, n in sizes.items()))
    man = _manifest(sizes)
    q25 = lib.build_subset(man, suite, "q25")
    again = lib.build_subset(man, suite, "q25")
    assert q25["row_list_sha256"] == again["row_list_sha256"]
    assert {k: v["selected"] for k, v in q25["benchmarks"].items()} == {"aa": 3, "bb": 2, "cc": 1}
    assert q25["total_selected"] == 6
    smoke = lib.build_subset(man, suite, "smoke", per_bench=2)
    for key, entry in smoke["benchmarks"].items():
        q = {r["ordinal"] for r in q25["benchmarks"][key]["rows"]}
        s = {r["ordinal"] for r in entry["rows"]}
        assert s <= q, key  # smoke rows are the first q25 ranks
        ranks = sorted(lib.rank_hex(key, str(1000 + o)) for o in range(sizes[key]))
        assert sorted(r["rank"] for r in q25["benchmarks"][key]["rows"]) == ranks[: math.ceil(sizes[key] / 4)]
    lines = "".join(f"{k}\t{r['row_id']}\n" for k, e in q25["benchmarks"].items() for r in e["rows"])
    assert q25["row_list_sha256"] == hashlib.sha256(lines.encode()).hexdigest()


def test_duplicate_indices_fall_back_to_ordinal_row_ids():
    sizes = {"aa": 8}
    suite = Suite((Bench("aa", 8),))
    sub = lib.build_subset(_manifest(sizes, dup="aa"), suite, "q25")
    assert sub["benchmarks"]["aa"]["row_id_rule"] == "ordinal:index"


def test_wemath_samples_whole_problem_groups_and_strata_are_covered():
    sizes = {"wemath": 30, "puzzlevqa": 40}
    suite = Suite(tuple(Bench(k, n) for k, n in sizes.items()))
    cols = {
        "wemath": [{"ID": f"g{o // 3}"} for o in range(30)],                     # 10 groups of 3 rows
        "puzzlevqa": [{"category": "rare" if o == 39 else f"c{o % 3}"} for o in range(40)],
    }
    man = _manifest(sizes)
    q25 = lib.build_subset(man, suite, "q25", columns=cols)
    w = q25["benchmarks"]["wemath"]
    assert w["units_total"] == 10 and w["units_selected"] == 3 and w["selected"] == 9
    groups = {}
    for r in w["rows"]:
        groups.setdefault(cols["wemath"][r["ordinal"]]["ID"], []).append(r["ordinal"])
    assert all(len(v) == 3 for v in groups.values())  # complete groups only
    p = q25["benchmarks"]["puzzlevqa"]
    assert {cols["puzzlevqa"][r["ordinal"]]["category"] for r in p["rows"]} == {"c0", "c1", "c2", "rare"}
    assert p["selected"] == 10 + p["added_for_coverage"]
    smoke = lib.build_subset(man, suite, "smoke", per_bench=2, columns=cols)
    assert smoke["benchmarks"]["wemath"]["units_selected"] == 4
    with pytest.raises(ValueError):
        lib.build_subset(man, suite, "q25")  # the exceptions need their TSV columns


def test_subset_refuses_incomplete_manifest():
    suite = Suite((Bench("aa", 5),))
    man = _manifest({"aa": 4})
    with pytest.raises(ValueError):
        lib.build_subset(man, suite, "q25")


def test_judge_plan_hardware_classes():
    w = lib.QWEN3_32B_WEIGHT_GIB_FALLBACK
    h200 = lib.judge_plan(["0", "1"], [143771, 143771], w)
    assert h200["feasible"] and h200["tp"] == 1 and h200["endpoints"] == 2
    a100x2 = lib.judge_plan(["0", "1"], [81920, 81920], w)
    assert a100x2["feasible"] and a100x2["tp"] == 2 and not a100x2["tight"]
    a100x1 = lib.judge_plan(["0"], [81920], w)
    assert a100x1["feasible"] and a100x1["tight"] and a100x1["max_num_batched_tokens"] == 8192
    l40s = lib.judge_plan(["0", "1"], [46068, 46068], w)
    assert l40s["feasible"] and l40s["tp"] == 2
    rtx4000x8 = lib.judge_plan([str(i) for i in range(8)], [24467] * 8, w)
    assert rtx4000x8["feasible"] and rtx4000x8["tp"] == 4 and rtx4000x8["gpu_groups"] == "0,1,2,3 4,5,6,7"
    rtx4000x2 = lib.judge_plan(["0", "1"], [24467] * 2, w)
    assert not rtx4000x2["feasible"]
    a800x2 = lib.judge_plan(["0", "1"], [40960, 40960], w)
    assert a800x2["feasible"] and a800x2["tight"] and a800x2["tp"] == 2  # 80 GB total: only just
    rtx6000 = lib.judge_plan(["0"], [97887], w)
    assert rtx6000["feasible"] and rtx6000["tp"] == 1


def test_natural_mapping_is_frozen():
    assert len(lib.NATURAL_KEYS) == 9 and "mmstar" in lib.NATURAL_KEYS and lib.MIXED_ALONE_KEYS == ("mmstar",)


@pytest.mark.skipif(not os.environ.get("TRACE_RLVR_ROOT"), reason="needs TRACE_RLVR_ROOT (rlvr checkout)")
def test_real_suite_mapping_and_q25_size():
    suite = json.loads((Path(os.environ["TRACE_RLVR_ROOT"]) / "rlvr/evaluation/trace_eval/suite.v1.json").read_text())
    keys = [b["key"] for b in suite["benchmarks"]]
    assert len(keys) == 24 and set(lib.NATURAL_KEYS) <= set(keys)
    assert len([k for k in keys if k not in lib.NATURAL_KEYS]) == 15
    assert sum(b["rows"] for b in suite["benchmarks"]) == 32805
    assert sum(math.ceil(b["rows"] / 4) for b in suite["benchmarks"]) == 8205  # 8,207 with WeMath's 132 groups
    assert set(lib.JUDGE_EXTRACTION_KEYS) <= set(keys)


def test_evaluator_exception_count_reads_the_mme_evaluation_block():
    # post-release fix-2: the MME-Reasoning scorer counts rows whose official per-row evaluator raised
    assert lib.evaluator_exception_count({"evaluation": {"evaluator_exception_scored_incorrect": 2}}) == 2
    assert lib.evaluator_exception_count({"evaluation": {"evaluator_invalid": 0}}) == 0  # scored before fix-2
    assert lib.evaluator_exception_count({"score": 50.0}) == 0  # other routes have no such policy
    assert lib.evaluator_exception_count({"evaluation": "n/a"}) == 0


def test_summary_markdown_reports_evaluator_exceptions():
    s = {"subset": "full", "served": "m", "seed": 42, "model": "m", "model_revision": "r", "rows_expected": 1,
         "benchmarks_scored": 24, "complete": True, "benchmarks": [],
         "macro": {"overall_24": 1.0, "natural_9": 1.0, "chart_diagram_synthetic_15": 1.0, "mmstar": 1.0,
                   "categories": {}},
         "evaluator_exception_scored_incorrect": {"total": 2, "by_benchmark": {"mme_reasoning": 2, "mmstar": 0}}}
    md = lib.summary_markdown(s)
    assert "Official-evaluator exceptions scored incorrect ('Failed to evaluate', post-release fix-2): 2 (mme_reasoning 2)." in md
    s["evaluator_exception_scored_incorrect"] = {"total": 0, "by_benchmark": {"mme_reasoning": 0}}
    assert "post-release fix-2): 0." in lib.summary_markdown(s)


def test_judge_unresolved_counts_read_the_mme_and_mathverse_blocks():
    # post-release fix-3: rows whose judge output was still unusable after the official 5 attempts
    mme = {"evaluation": {"evaluator_exception_scored_incorrect": 1, "extraction_failed_scored_incorrect": 1}}
    assert lib.judge_unresolved_counts(mme) == {"extraction_failed_scored_incorrect": 1,
                                                "mathverse_all_retries_failed_scored_incorrect": 0}
    assert lib.evaluator_exception_count(mme) == 1  # fix-2's counter is separate (no double count)
    mathverse = {"artifacts": {"mathverse_all_retries_failed_scored_incorrect": 1,
                               "mathverse_all_retries_failed_scored_incorrect_indices": ["1880"]}}
    assert lib.judge_unresolved_counts(mathverse) == {"extraction_failed_scored_incorrect": 0,
                                                      "mathverse_all_retries_failed_scored_incorrect": 1}
    # scored before fix-3 (or another route): absent -> 0
    for payload in ({"evaluation": {"evaluator_invalid": 0}}, {"artifacts": {"judged_table": "x.xlsx"}},
                    {"score": 50.0}, {"evaluation": "n/a", "artifacts": None}):
        assert set(lib.judge_unresolved_counts(payload).values()) == {0}


def test_summary_markdown_reports_judge_unresolved_rows():
    s = {"subset": "full", "served": "m", "seed": 42, "model": "m", "model_revision": "r", "rows_expected": 1,
         "benchmarks_scored": 24, "complete": True, "benchmarks": [],
         "macro": {"overall_24": 1.0, "natural_9": 1.0, "chart_diagram_synthetic_15": 1.0, "mmstar": 1.0,
                   "categories": {}},
         "evaluator_exception_scored_incorrect": {"total": 0, "by_benchmark": {"mme_reasoning": 0}}}
    assert "fix-3" not in lib.summary_markdown(s)  # a summary written before fix-3
    s["extraction_failed_scored_incorrect"] = {"total": 1, "by_benchmark": {"mme_reasoning": 1, "mathverse": 0}}
    s["mathverse_all_retries_failed_scored_incorrect"] = {"total": 1, "by_benchmark": {"mme_reasoning": 0,
                                                                                       "mathverse": 1}}
    md = lib.summary_markdown(s)
    assert ("Judge output still unusable after 5 attempts, scored incorrect (MME-Reasoning extraction, MathVerse score "
            "judgement; post-release fix-3): 2 (mme_reasoning 1, mathverse 1).") in md
    assert "post-release fix-2): 0." in md
    s["extraction_failed_scored_incorrect"] = {"total": 0, "by_benchmark": {"mme_reasoning": 0, "mathverse": None}}
    s["mathverse_all_retries_failed_scored_incorrect"] = {"total": 0, "by_benchmark": {"mathverse": 0}}
    assert "post-release fix-3): 0." in lib.summary_markdown(s)


def _descriptor(root: Path, name: str, stage: str, slug: str, code: str, seed: int = 42, run_id: str = "r1") -> None:
    (root / "ready").mkdir(parents=True, exist_ok=True)
    (root / "ready" / f"{name}.ready.json").write_text(json.dumps({
        "identity": {"stage": stage, "model_slug": slug, "seed": seed, "run_id": run_id},
        "provenance": {"final25_code_hash": code}}))


def test_archive_code_hash_is_the_single_generation_hash(tmp_path, capsys):
    a, b = "a" * 64, "b" * 64
    _descriptor(tmp_path, "g1", "generation", "m", a)
    _descriptor(tmp_path, "g2", "generation", "m", a)
    _descriptor(tmp_path, "s1", "score", "m", b)  # scored after a post-release fix: not the generation's code
    _descriptor(tmp_path, "o1", "generation", "other", b)
    _descriptor(tmp_path, "x1", "generation", "m", b, seed=43)
    _descriptor(tmp_path, "y1", "generation", "m", b, run_id="r2")
    assert lib.archive_code_hashes(tmp_path, stage="generation", slug="m", seed=42, run_id="r1") == {a}
    assert lib.main(["archive-code-hash", "--archive", str(tmp_path), "--slug", "m", "--seed", "42", "--run-id", "r1"]) == 0
    assert capsys.readouterr().out.strip() == a
    # a generation spread over two code versions has no single hash: the caller keeps the protocol's own check
    assert lib.main(["archive-code-hash", "--archive", str(tmp_path), "--slug", "m", "--seed", "42"]) == 1
    assert lib.main(["archive-code-hash", "--archive", str(tmp_path / "missing"), "--slug", "m", "--seed", "42"]) == 1


def test_verify_generation_only_pins_the_generation_phase():
    assert lib.main(["verify-generation", "--code-hash", "a" * 64, "--", "--phase", "score", "--seeds", "42"]) == 2
    assert lib.main(["verify-generation", "--code-hash", "a" * 64, "--", "--seeds", "42"]) == 2
