"""``format-normalized-v2``: a diagnostic row score beside (never instead of) the official score (work package F0).

v2 supersedes v1 (frozen 2026-11-06; superseded before any confirmation output was read, after the blinded development
check of TableVQABench found two defects): T1 no longer applies on vtabfact, where the pinned scorer already reads the
label anywhere and treats a prediction holding both labels as unparsed, so a box must not override that rule; and boxed
text is un-escaped (``\\%``, ``\\$``, ...). The failure taxonomy (audit only) classifies the rewritten answer of rows a
rule touched. The MC rule is unchanged.

What it answers: how much of a score difference is the pinned scorer failing to *read* a clearly designated final
answer, as opposed to the model answering wrongly. The official score stays the primary metric everywhere.

Row rule (identical for every model, interface and seed; ``rescore.normalized_row_scores``):

* Out of scope (21 of the 24 benchmarks; reasons in ``OUT_OF_SCOPE``): normalized = official.
* ``mc_final_option`` (PuzzleVQA, VisualPuzzles). The pinned scorer parses the prediction cell with
  ``extract_unambiguous_abcd``. When that parse yields an option the row is untouched (``official_read``), whether it
  was right or wrong. Only when the pinned parse is unresolved (``Z``: no explicit candidate, or conflicting ones; the
  official row score is then 0) does ``extract.extract_final_option`` read the same cell with the row's allowed option
  labels and no gold. Resolved -> normalized = 1 if that option is the gold option (pinned ``to_choice_letter`` for
  PuzzleVQA, the licensed answer letter for VisualPuzzles) else 0. Unresolved (ambiguous, no final answer, not an
  option) -> normalized = official = 0.
* ``tvqa_final_answer`` (TableVQABench). Rows with full official credit are untouched. On the others the scored
  prediction (the exact string the pinned split scorer read) is rewritten by at most two rules, then re-scored with the
  *pinned* split scorer of the row's split (no new matcher, same numeric tolerances): T1 (fintabnetqa, vwtq,
  vwtq_syn) a single designated ``\\boxed{}`` answer (``extract.extract_final_boxed``) replaces the prediction; T2 on
  vwtq/vwtq_syn, empty items of a ``||``-separated list (``5||``) are dropped when a non-empty item remains. Each component (fintabnetqa relieved and
  exact; the 0/1 score elsewhere) becomes max(official, rewritten). No rule applies -> official.

So normalized >= official on every row, every row stays in its benchmark's denominator, and the gold answer is only
ever used to score an extraction that was made without it. Benchmark values use each benchmark's native aggregation
(TableVQABench mean of five split scores, WeMath problem-group strict score, pooled means elsewhere) through the
reviewed study's metric adapters (``trace_merge.metrics``); the macro is the unweighted mean of the 24 benchmarks.

Input (``normalized_row_scores``): one row per model_label x seed x content row, in the schema of
``aligned/predictions.parquet``. Required columns: ``REQUIRED_COLUMNS``. ``official_row_score`` is computed with the
reviewed metric adapters when absent (``official_row_scores``). The pinned scorers must be importable (prepared protocol
clone on PYTHONPATH, e.g. via the reviewed study's ``study_env.sh``).
"""

from __future__ import annotations

import ast
import contextlib
import io
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

_MERGE_STUDY = Path(__file__).resolve().parents[2] / "trace_merge"
if str(_MERGE_STUDY) not in sys.path:
    sys.path.insert(0, str(_MERGE_STUDY))

from trace_merge import metrics as mt  # noqa: E402
from trace_merge.config import StudyConfig, load_config  # noqa: E402

from . import extract as ex  # noqa: E402

METRIC_NAME = "format-normalized-v2"
RULE_MC = "mc_final_option"
RULE_TVQA = "tvqa_final_answer"
SCOPE: dict[str, str] = {"puzzlevqa": RULE_MC, "visualpuzzles": RULE_MC, "tablevqabench": RULE_TVQA}
MC_LABELS_FIXED = {"visualpuzzles": "ABCD"}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
WTQ_SPLITS = ("vwtq", "vwtq_syn")
T1_SPLITS = ("fintabnetqa", "vwtq", "vwtq_syn")   # not vtabfact: its pinned label rule stands (v2)
SUITE_BENCHMARKS = 24

# Why each other benchmark is left at its official score (development evidence in audit/scoring_changes.md).
OUT_OF_SCOPE: dict[str, str] = {
    **{b: "the official scorer reads whole responses with an LLM judge or matcher (VLMEvalKit MCQ/judge route); a "
          "0 there with a correct final option is a judgment of an inconsistent response, not an unreadable wrapper"
       for b in ("mmstar", "blink", "mmmu_pro_vision", "cvbench_3d", "wemath", "spatialvizbench_cot", "erqa",
                 "logicvista", "mme_reasoning", "mathvista", "mathvision", "mathverse", "charxivreason", "evochart")},
    "phyx_mini_mc": "the pinned option parser falls back to the last standalone letter anywhere, so it almost never "
                    "fails to read; its disagreements with the final option are misreads (e.g. a unit 'C'), and v1 "
                    "never overrides an official read",
    "treebench": "the pinned parser reads the first capital A-E of the answer text, so it almost never fails to "
                 "read; its errors are misreads of prose, which v1 does not override",
    "realworldqa": "explicit-MCQ adapter plus VLMEvalKit matching; no development row where the final option is "
                   "unread",
    "embspatial": "spatial-bench option matcher with tail/phrase fallbacks; no development row where the final option "
                  "is unread",
    "chartqapro": "free-form relaxed-accuracy scorer with its own final-answer adapter; not an option task",
    "countbenchqa": "substring match of the count anywhere in the prediction; no reading failure to normalize",
    "countqa": "numeric scorer of the prediction; not an option task",
}

REQUIRED_COLUMNS = ("model_label", "seed", "benchmark_id", "content_key", "source_ordinal", "source_index",
                    "source_row_sha256", "source_record_sha256", "archived_row_score", "official_excluded",
                    "extraction_status", "extraction_value_json", "judge_output")
OUTPUT_COLUMNS = ("official_row_score", "normalized_row_score", "fn_status", "fn_rule", "fn_extracted", "fn_detail")

# fn_status values
OUT = "out_of_scope"
OFFICIAL_READ = "official_read"                 # MC: the pinned parser produced an option; untouched
OFFICIAL_FULL = "official_full_credit"          # TableVQA: already full credit; untouched
REPAIRED = "repaired"                           # normalized > official
RESOLVED_WRONG = "resolved_still_wrong"         # rule read a final answer; it is wrong -> unchanged
UNRESOLVED = "unresolved"                       # rule found no single final answer -> unchanged (fn_detail says why)
NO_RULE = "no_rule_applies"                     # TableVQA: no boxed answer, no empty list items -> unchanged


class RescoreError(ValueError):
    pass


def _quiet() -> contextlib.AbstractContextManager:
    return contextlib.redirect_stdout(io.StringIO())


# ----------------------------------------------------------------------------------------------- metric access
def load_metrics(cfg: StudyConfig | None = None, benchmarks: Sequence[str] | None = None,
                 *, require_accepted: bool = True) -> dict[str, mt.BenchmarkMetric]:
    """The reviewed study's accepted metric adapters with licensed metadata bound.

    ``cfg`` defaults to ``load_config()`` (``TRACE_MERGE_WORK_ROOT`` = the reviewed study's work root, which holds the
    accepted metric registry; ``TRACE_MERGE_PROTOCOL_CLONE`` = the prepared clone)."""

    cfg = cfg or load_config()
    return mt.build_metrics(cfg, benchmarks, require_accepted=require_accepted)


def _require(rows: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in rows.columns]
    if missing:
        raise RescoreError(f"rows lack columns {missing}")


def _slices(rows: pd.DataFrame) -> dict[tuple[str, int, str], np.ndarray]:
    return rows.groupby(["model_label", "seed", "benchmark_id"], sort=False).indices


def _excluded_flag(rows: pd.DataFrame) -> pd.DataFrame:
    if "official_excluded" in rows.columns and rows["official_excluded"].dtype != bool:
        rows = rows.copy()
        rows["official_excluded"] = rows["official_excluded"].map(lambda v: str(v) == "True" or v is True)
    return rows


def official_row_scores(rows: pd.DataFrame, metrics: Mapping[str, mt.BenchmarkMetric]) -> pd.Series:
    """Official per-row scores with the reviewed adapters, one (label, seed, benchmark) slice at a time.

    0/1 for pooled benchmarks, ChartQAPro fractional, TableVQABench component mean (0, 0.5, 1 on fintabnetqa), WeMath
    0/1 question scores (aggregated by problem group in ``benchmark_values``)."""

    _require(rows, REQUIRED_COLUMNS)
    rows = _excluded_flag(rows)
    out = pd.Series(np.nan, index=rows.index, name="official_row_score")
    for (label, seed, bench), idx in _slices(rows).items():
        sub = rows.iloc[idx]
        if bench not in metrics:
            raise RescoreError(f"no metric adapter for {bench}")
        out.iloc[idx] = metrics[bench].row_scores(sub).to_numpy(dtype=float)
    if out.isna().any():
        raise RescoreError(f"{int(out.isna().sum())} rows without an official row score")
    return out


# ------------------------------------------------------------------------------------------------- MC rule
def _mc_slice(sub: pd.DataFrame, metric: mt.BenchmarkMetric, official: np.ndarray) -> list[dict[str, Any]]:
    bench = metric.benchmark_id
    meta = metric.aligned_meta(sub)
    cells = mt.official_route_view(sub)               # the exact prediction cell the pinned scorer read
    if bench == "puzzlevqa":
        mod = mt._pinned("vlmeval.dataset.utils.puzzlevqa")
    else:
        mod = mt._pinned("vlmeval.dataset.utils.visualpuzzles")
    out = []
    with _quiet():
        for i, cell in enumerate(cells):
            parse = mod.extract_answer(cell)
            if bench == "puzzlevqa":
                options = ast.literal_eval(meta["options"].iloc[i])
                gold = str(mod.to_choice_letter(options, meta["answer"].iloc[i]))
                labels = LETTERS[:len(options)]
            else:
                gold = str(meta["answer"].iloc[i])
                labels = MC_LABELS_FIXED[bench]
            rec = {"official_row_score": float(official[i]), "fn_rule": RULE_MC, "fn_extracted": None}
            if parse != "Z":
                rec.update(normalized_row_score=float(official[i]), fn_status=OFFICIAL_READ,
                           fn_detail=f"pinned parse {parse}")
                out.append(rec)
                continue
            if official[i] != 0:
                raise RescoreError(f"{bench}: unresolved pinned parse with official score {official[i]}")
            text = cell if isinstance(cell, str) else ""
            x = ex.extract_final_option(text, labels)
            if x.resolved:
                hit = float(x.option.lower() == gold.lower())
                rec.update(normalized_row_score=hit, fn_status=REPAIRED if hit else RESOLVED_WRONG,
                           fn_extracted=x.option, fn_detail=f"{x.region}/{x.rule}")
            else:
                rec.update(normalized_row_score=float(official[i]), fn_status=UNRESOLVED,
                           fn_detail=f"{x.status} ({x.region})")
            out.append(rec)
    return out


# ----------------------------------------------------------------------------------------------- TableVQA rule
def tvqa_candidate(prediction: str, split: str) -> tuple[str | None, list[str], str]:
    """(rewritten prediction or None, rules applied, detail) for one scored TableVQABench prediction (T1, T2)."""

    cand, applied, detail = prediction, [], ""
    x = ex.extract_final_boxed(prediction) if split in T1_SPLITS else None
    if x is not None and x.resolved:
        cand, applied = x.option, ["T1_boxed"]
    elif x is not None and x.status == ex.AMBIGUOUS:
        detail = "boxed answers disagree"
    if split in WTQ_SPLITS:
        items = cand.replace("||", "|").split("|")
        kept = [i for i in items if i.strip()]
        if kept and len(kept) < len(items):
            cand, applied = "|".join(kept), applied + ["T2_empty_items"]
    return (cand if applied else None), applied, detail


def _tvqa_components(preds: Sequence[str], answers: Sequence[Any], splits: Sequence[str]) -> np.ndarray:
    """Pinned per-row (relieved-or-primary, exact) components for arbitrary predictions, one record per call."""

    tv = mt._pinned("vlmeval.dataset.utils.tablevqabench")
    out = np.zeros((len(preds), 2))
    with _quiet():
        for i, (p, a, s) in enumerate(zip(preds, answers, splits)):
            rec = [{"prediction": p, "answer": a}]
            if s == "fintabnetqa":
                tv.evaluate_fintabnet(rec, ["accuracy"])
            elif s == "vtabfact":
                tv.evaluate_tabfact(rec, ["accuracy"])
            elif s in WTQ_SPLITS:
                tv.evaluate_wtq(rec, ["accuracy"])
            else:
                raise RescoreError(f"tablevqabench: unknown split {s}")
            sc = rec[0]["scores"]
            prim = float(sc["accuracy"] or 0)
            out[i] = (prim, float(sc.get("exact_score", sc["accuracy"]) or 0))
    return out


def _tvqa_slice(sub: pd.DataFrame, metric: mt.BenchmarkMetric, official: np.ndarray) -> list[dict[str, Any]]:
    meta = metric.aligned_meta(sub)
    preds = mt.scored_prediction_view(sub)
    comp = mt.tablevqa_components(preds, meta, {})
    off_p, off_e = comp["primary"].to_numpy(), comp["exact"].to_numpy()
    split = meta["split"].to_numpy()
    fin = split == "fintabnetqa"
    recomposed = np.where(fin, (off_p + off_e) / 2.0, off_p)
    if not np.allclose(recomposed, official):
        raise RescoreError("tablevqabench: official row scores do not match the pinned components")
    out = []
    for i, pred in enumerate(preds):
        rec = {"official_row_score": float(official[i]), "fn_rule": RULE_TVQA, "fn_extracted": None}
        full = off_p[i] == 1 and off_e[i] == 1
        if full:
            rec.update(normalized_row_score=float(official[i]), fn_status=OFFICIAL_FULL, fn_detail="")
            out.append(rec)
            continue
        cand, applied, detail = tvqa_candidate(pred, split[i])
        if cand is None:
            status = UNRESOLVED if detail else NO_RULE
            rec.update(normalized_row_score=float(official[i]), fn_status=status, fn_detail=detail)
            out.append(rec)
            continue
        p, e = _tvqa_components([cand], [meta["answer"].iloc[i]], [split[i]])[0]
        p, e = max(p, off_p[i]), max(e, off_e[i])
        new = (p + e) / 2.0 if fin[i] else p
        rec.update(normalized_row_score=float(new), fn_status=REPAIRED if new > official[i] else RESOLVED_WRONG,
                   fn_extracted=cand[:200], fn_detail="+".join(applied))
        out.append(rec)
    return out


# ---------------------------------------------------------------------------------------------------- public API
def normalized_row_scores(rows: pd.DataFrame, metrics: Mapping[str, mt.BenchmarkMetric] | None = None, *,
                          cfg: StudyConfig | None = None) -> pd.DataFrame:
    """Per-row official and ``format-normalized-v2`` scores with a status per row.

    ``rows``: schema of ``aligned/predictions.parquet`` (``REQUIRED_COLUMNS``; ``official_row_score`` optional, else
    recomputed). Each (model_label, seed, benchmark_id) slice must hold each content row once. Returns a copy of
    ``rows`` with ``OUTPUT_COLUMNS`` added (``fn_status``: out_of_scope | official_read | official_full_credit |
    repaired | resolved_still_wrong | unresolved | no_rule_applies). Metrics default to ``load_metrics(cfg)``.
    """

    _require(rows, REQUIRED_COLUMNS)
    rows = _excluded_flag(rows.reset_index(drop=True))
    benches = sorted(set(rows["benchmark_id"].astype(str)))
    metrics = metrics or load_metrics(cfg, benches)
    dup = rows.duplicated(["model_label", "seed", "content_key"])
    if dup.any():
        raise RescoreError(f"{int(dup.sum())} duplicate (model_label, seed, content_key) rows")
    official = (pd.to_numeric(rows["official_row_score"], errors="raise").astype(float)
                if "official_row_score" in rows.columns else official_row_scores(rows, metrics))
    out = rows.copy()
    out["official_row_score"] = official.to_numpy()
    out["normalized_row_score"] = official.to_numpy()
    out["fn_status"] = OUT
    out["fn_rule"] = None
    out["fn_extracted"] = None
    out["fn_detail"] = ""
    for (label, seed, bench), idx in _slices(rows).items():
        rule = SCOPE.get(bench)
        if rule is None:
            continue
        sub = rows.iloc[idx]
        off = official.to_numpy()[idx]
        recs = (_mc_slice if rule == RULE_MC else _tvqa_slice)(sub, metrics[bench], off)
        frame = pd.DataFrame(recs, index=out.index[idx])
        for col in OUTPUT_COLUMNS:
            out.loc[frame.index, col] = frame[col].to_numpy()
    out["normalized_row_score"] = out["normalized_row_score"].astype(float)
    if (out["normalized_row_score"] < out["official_row_score"] - 1e-12).any():
        raise RescoreError("a normalized row score fell below its official score")
    return out


def benchmark_values(scored: pd.DataFrame, metrics: Mapping[str, mt.BenchmarkMetric], *,
                     score_col: str) -> pd.DataFrame:
    """Benchmark value per (model_label, seed, benchmark_id) with each benchmark's native aggregation."""

    rows = []
    for (label, seed, bench), idx in _slices(scored).items():
        sub = scored.iloc[idx]
        value = metrics[bench].official_score(sub, sub[score_col].astype(float))
        rows.append({"model_label": label, "seed": int(seed), "benchmark_id": bench, "value": float(value),
                     "rows": len(sub)})
    return pd.DataFrame(rows)


def macro_values(values: pd.DataFrame, *, n_suite: int = SUITE_BENCHMARKS) -> pd.DataFrame:
    """Unweighted benchmark macro per (model_label, seed) and its mean over seeds (seed = -1 in the output).

    Refuses a macro that lacks benchmarks (rename it explicitly instead)."""

    out = []
    for (label, seed), g in values.groupby(["model_label", "seed"]):
        if g["benchmark_id"].nunique() != n_suite:
            raise RescoreError(f"{label}/{seed}: {g['benchmark_id'].nunique()} benchmarks, not {n_suite}")
        out.append({"model_label": label, "seed": int(seed), "macro": float(g["value"].mean())})
    frame = pd.DataFrame(out)
    mean = frame.groupby("model_label")["macro"].mean().reset_index().assign(seed=-1)
    return pd.concat([frame, mean[["model_label", "seed", "macro"]]], ignore_index=True)


class ScoreColumnMetric:
    """A reviewed metric adapter whose ``row_scores`` returns a stored column (``official_row_score`` or
    ``normalized_row_score``), so ``trace_merge.statistics`` can bootstrap either score with the same draws."""

    def __init__(self, base: mt.BenchmarkMetric, score_col: str) -> None:
        self.base = base
        self.score_col = score_col

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def row_scores(self, rows: pd.DataFrame, meta: pd.DataFrame | None = None) -> pd.Series:
        self.base.check_rows(rows)
        return pd.to_numeric(rows[self.score_col], errors="raise").astype(float)


def paired_contrasts(scored: pd.DataFrame, metrics: Mapping[str, mt.BenchmarkMetric], assignments: pd.DataFrame, *,
                     labels: Sequence[str], seeds: Sequence[int], comparisons: Sequence[str],
                     categories: Mapping[str, Sequence[str]] | None = None, n_replicates: int = 10_000,
                     seed: int = 314159, confidence: float = 0.95) -> dict[str, Any]:
    """Paired cluster bootstrap (reviewed ``trace_merge.statistics``) of official and normalized scores, same draws.

    ``assignments`` needs ``content_key`` and ``component_id`` (the bootstrap units). Returns
    ``{"official": BootstrapResult, "normalized": BootstrapResult}``."""

    from trace_merge import statistics as st

    benches = sorted(set(scored["benchmark_id"]))
    out = {}
    for name, col in (("official", "official_row_score"), ("normalized", "normalized_row_score")):
        wrapped = {b: ScoreColumnMetric(metrics[b], col) for b in benches}
        out[name] = st.bootstrap_from_rows(scored, wrapped, assignments, labels=list(labels), seeds=list(seeds),
                                           benchmarks=benches, categories=categories,
                                           allow_partial_categories=len(benches) < SUITE_BENCHMARKS,
                                           comparisons=list(comparisons), n_replicates=n_replicates,
                                           confidence=confidence, seed=seed)
    return out


# ======================================================================================================================
# F0 audit driver (python -m trace_followup.rescore ...): reads the reviewed study's stored outputs (never writes
# there) and writes F0 tables under the follow-up work root. Not part of the row rule above.
# ======================================================================================================================
ARCHIVE_LABELS = ("B", "T", "V@P0")
P1_LOCAL = ("V@P1", "M050@P1", "CV050@P1", "CT050@P1")
P2_LOCAL = ("B@P2", "T@P2", "M050@P2", "CV050@P2", "CT050@P2")
SEEDS = (42, 43, 44)
LEAN_COLUMNS = ("model_label", "seed", "benchmark_id", "content_key", "source_ordinal", "source_index",
                "source_row_sha256", "source_record_sha256", *OUTPUT_COLUMNS)
CONTRASTS = {
    "development_partition": ["T-B", "V@P0-B", "T-V@P0"],
    "development_screen": ["V@P1-V@P0", "T-B", "V@P0-B", "M050@P1-T", "M050@P1-V@P1", "M050@P1-CV050@P1",
                           "M050@P1-CT050@P1", "CV050@P1-B", "CT050@P1-B", "V@P1-B"],
    "development_screen_P2": ["T@P2-B@P2", "V@P0-B@P2", "M050@P2-T@P2", "M050@P2-V@P0", "M050@P2-CV050@P2",
                              "M050@P2-CT050@P2"],
    "confirmation": ["V@P1-V@P0", "T-B", "V@P0-B", "M050@P1-T", "M050@P1-V@P1", "M050@P1-CV050@P1",
                     "M050@P1-CT050@P1", "CV050@P1-B", "CT050@P1-B", "V@P1-B", "T-V@P0", "M050@P1-V@P0"],
}


def _subsets(partition: str) -> dict[str, tuple[list[str], list[int], str]]:
    """subset name -> (labels, seeds, row filter: 'partition' or 'in_dev_screen')."""

    if partition == "development":
        return {"development_partition": (list(ARCHIVE_LABELS), list(SEEDS), "partition"),
                "development_screen": ([*ARCHIVE_LABELS, *P1_LOCAL], [42], "in_dev_screen"),
                "development_screen_P2": (["V@P0", *P2_LOCAL], [42], "in_dev_screen")}
    if partition == "confirmation":
        return {"confirmation": ([*ARCHIVE_LABELS, *P1_LOCAL], list(SEEDS), "partition")}
    raise RescoreError(f"unknown partition {partition}")


def score_partition(cfg: StudyConfig, partition: str, *, metrics: Mapping[str, mt.BenchmarkMetric] | None = None,
                    log=print) -> pd.DataFrame:
    """Official and normalized row scores of every stored model x seed row of one partition (lean columns)."""

    from trace_merge import analysis as an

    ga = an.assignments(cfg)
    metrics = metrics or load_metrics(cfg)
    if partition == "development":
        labels, local, subset = [*ARCHIVE_LABELS, *P1_LOCAL, *P2_LOCAL], {l: [42] for l in (*P1_LOCAL, *P2_LOCAL)}, \
            "development_screen"
    else:
        labels, local, subset = [*ARCHIVE_LABELS, *P1_LOCAL], {l: list(SEEDS) for l in P1_LOCAL}, "confirmation"
    frames = []
    for bench in cfg.benchmark_keys():
        rows = an.load_rows(cfg, labels=labels, seeds=list(SEEDS), benchmarks=[bench], local=local, subset=subset)
        rows = an.restrict(rows, ga, partition)
        scored = normalized_row_scores(rows, metrics)
        frames.append(scored[list(LEAN_COLUMNS)])
        log(f"[f0] {partition} {bench}: {len(scored)} rows, {int((scored.fn_status == REPAIRED).sum())} repaired")
    out = pd.concat(frames, ignore_index=True)
    flags = ga.set_index("content_key")["in_dev_screen"]
    out["in_dev_screen"] = out["content_key"].map(flags).astype(bool)
    return out


def _subset_rows(scored: pd.DataFrame, labels: Sequence[str], seeds: Sequence[int], filt: str) -> pd.DataFrame:
    sub = scored[scored["model_label"].isin(labels) & scored["seed"].isin([int(s) for s in seeds])]
    if filt == "in_dev_screen":
        sub = sub[sub["in_dev_screen"]]
    return sub.reset_index(drop=True)


def values_long(scored: pd.DataFrame, metrics: Mapping[str, mt.BenchmarkMetric]) -> pd.DataFrame:
    off = benchmark_values(scored, metrics, score_col="official_row_score")
    norm = benchmark_values(scored, metrics, score_col="normalized_row_score")
    return off.merge(norm, on=["model_label", "seed", "benchmark_id", "rows"], suffixes=("_official", "_normalized"),
                     validate="one_to_one")


def reproduction_check(cfg: StudyConfig, subset: str, vals: pd.DataFrame) -> dict[str, Any]:
    """Official values recomputed here against the reviewed study's own seed tables (must agree to 1e-6 pp)."""

    t = cfg.work("tables")
    ref = {"development_partition": [t / "stage_b" / "seed_scores.development.csv"],
           "development_screen": [t / "screen" / "seed_scores.P1.csv"],
           "development_screen_P2": [t / "screen" / "seed_scores.P2.csv"],
           "confirmation": [t / "confirmation" / "seed_scores.csv"]}[subset]
    old = pd.concat([pd.read_csv(p) for p in ref], ignore_index=True)[["model_label", "seed", "benchmark_id",
                                                                         "official_score"]]
    m = vals.merge(old, on=["model_label", "seed", "benchmark_id"], how="left")
    missing = int(m["official_score"].isna().sum())
    err = float((m["value_official"] - m["official_score"]).abs().max())
    return {"subset": subset, "slices": len(m), "slices_without_reference": missing, "max_abs_error_pp": err,
            "passed": missing == 0 and err <= 1e-6, "reference": [str(p.relative_to(cfg.work())) for p in ref]}


def summary_table(vals: pd.DataFrame, categories: Mapping[str, Sequence[str]], subset: str) -> pd.DataFrame:
    """Per model: benchmark values, category macros and the 24-benchmark macro, each the mean over seeds of the
    per-seed value (benchmark) or of the per-seed benchmark means (macros); official vs normalized."""

    out = []
    for label, g in vals.groupby("model_label"):
        seeds = "+".join(str(s) for s in sorted(g["seed"].unique()))
        bench = g.groupby("benchmark_id")[["value_official", "value_normalized"]].mean()
        recs = [{"level": "benchmark", "scope": b, **r.to_dict()} for b, r in bench.iterrows()]
        for cat, members in categories.items():
            present = [m for m in members if m in bench.index]
            if present:
                recs.append({"level": "category", "scope": cat if len(present) == len(members) else
                             f"{cat} ({len(present)} of {len(members)})",
                             "value_official": bench.loc[present, "value_official"].mean(),
                             "value_normalized": bench.loc[present, "value_normalized"].mean()})
        if len(bench) == SUITE_BENCHMARKS:
            recs.append({"level": "overall", "scope": "24-benchmark macro",
                         "value_official": bench["value_official"].mean(),
                         "value_normalized": bench["value_normalized"].mean()})
        out += [{"subset": subset, "model_label": label, "seeds": seeds, **r} for r in recs]
    frame = pd.DataFrame(out)
    frame["delta"] = frame["value_normalized"] - frame["value_official"]
    return frame[["subset", "model_label", "seeds", "level", "scope", "value_official", "value_normalized", "delta"]]


def contrast_table(res: Mapping[str, Any], subset: str) -> pd.DataFrame:
    keep = ["comparison", "label_a", "label_b", "level", "scope", "estimate", "ci_low", "ci_high", "p_value"]
    o = res["official"].differences[keep]
    n = res["normalized"].differences[keep]
    m = o.merge(n, on=["comparison", "label_a", "label_b", "level", "scope"], suffixes=("_official", "_normalized"),
                validate="one_to_one")
    m["estimate_change"] = m["estimate_normalized"] - m["estimate_official"]
    m.insert(0, "subset", subset)
    return m


def contrast_reproduction(cfg: StudyConfig, subset: str, table: pd.DataFrame) -> dict[str, Any]:
    """Official contrasts recomputed here against the reviewed study's difference tables (same units, seed and
    replicates give the same draws; estimates and intervals must agree to 1e-9 pp where both exist)."""

    t = cfg.work("tables")
    ref = {"development_partition": t / "stage_b" / "differences.development.csv",
           "development_screen": t / "screen" / "differences.P1.csv",
           "development_screen_P2": t / "screen" / "differences.P2.csv",
           "confirmation": t / "confirmation" / "differences.csv"}[subset]
    old = pd.read_csv(ref)[["label_a", "label_b", "level", "scope", "estimate", "ci_low", "ci_high"]]
    m = table.merge(old, on=["label_a", "label_b", "level", "scope"], how="inner")
    if not len(m):
        return {"subset": subset, "matched": 0, "passed": None}
    err = max(float((m["estimate_official"] - m["estimate"]).abs().max()),
              float((m["ci_low_official"] - m["ci_low"]).abs().max()),
              float((m["ci_high_official"] - m["ci_high"]).abs().max()))
    return {"subset": subset, "matched": int(len(m)), "max_abs_error_pp": err, "passed": err <= 1e-9,
            "reference": str(ref.relative_to(cfg.work()))}


def change_ledger(scored: pd.DataFrame, partition: str) -> pd.DataFrame:
    examined = scored[~scored["fn_status"].isin([OUT, OFFICIAL_READ, OFFICIAL_FULL])].copy()
    examined["changed"] = examined["normalized_row_score"] > examined["official_row_score"]
    examined["reason_code"] = examined["fn_status"] + np.where(examined["fn_detail"].astype(str) != "",
                                                               ":" + examined["fn_detail"].astype(str), "")
    examined.insert(0, "partition", partition)
    cols = ["partition", "benchmark_id", "model_label", "seed", "content_key", "in_dev_screen", "official_row_score",
            "normalized_row_score", "changed", "reason_code", "fn_rule", "fn_extracted"]
    return examined[cols].sort_values(["benchmark_id", "model_label", "seed", "content_key"]).reset_index(drop=True)


def status_counts(scored: pd.DataFrame, subsets: Mapping[str, tuple[list[str], list[int], str]]) -> pd.DataFrame:
    out = []
    for name, (labels, seeds, filt) in subsets.items():
        sub = _subset_rows(scored, labels, seeds, filt)
        sub = sub[sub["benchmark_id"].isin(list(SCOPE))]
        g = sub.groupby(["benchmark_id", "model_label", "fn_status"]).size().unstack(fill_value=0)
        g.insert(0, "rows", g.sum(axis=1))
        g.insert(0, "subset", name)
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True).fillna(0)


# ------------------------------------------------------------------------------------- MC scope survey (evidence)
OPTION_SOURCES: dict[str, dict[str, Any]] = {
    "puzzlevqa": {"kind": "options_list", "column": "options", "gold": "option_text"},
    "visualpuzzles": {"kind": "fixed", "labels": "ABCD", "gold": "letter"},
    "embspatial": {"kind": "options_list", "column": "options", "gold": "letter"},
    "erqa": {"kind": "fixed", "labels": "ABCD", "gold": "letter"},
    "phyx_mini_mc": {"kind": "fixed", "labels": "ABCD", "gold": "letter"},
    "spatialvizbench_cot": {"kind": "letter_columns", "columns": "ABCD", "gold": "letter"},
    "mmstar": {"kind": "letter_columns", "columns": "ABCD", "gold": "letter"},
    "blink": {"kind": "letter_columns", "columns": "ABCD", "gold": "letter"},
    "realworldqa": {"kind": "letter_columns", "columns": "ABCD", "gold": "letter"},
    "cvbench_3d": {"kind": "letter_columns", "columns": "AB", "gold": "letter"},
    "mmmu_pro_vision": {"kind": "letter_columns", "columns": "ABCDEFGHIJ", "gold": "letter"},
    "treebench": {"kind": "letter_columns", "columns": "ABCDEFGHIJK", "gold": "letter"},
    "wemath": {"kind": "letter_columns", "columns": "ABCDEFG", "gold": "letter"},
    "logicvista": {"kind": "fixed", "labels": "ABCDEFG", "gold": "letter"},
    "mme_reasoning": {"kind": "fixed", "labels": "ABCDEFGH", "gold": "letter", "row_filter": ("question_type", "choice")},
    "mathvista": {"kind": "choices_list", "column": "choices", "gold": "answer_option",
                  "row_filter": ("question_type", "multi_choice")},
    "mathvision": {"kind": "choices_list", "column": "choices", "gold": "letter", "row_filter": ("choices", "nonempty")},
}


def _literal_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        out = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        out = re.findall(r"'([^']*)'", text)
    return list(out) if isinstance(out, (list, tuple)) else []


def load_option_meta(cfg: StudyConfig, benchmark: str) -> pd.DataFrame:
    """Per content row of an option benchmark: allowed labels and gold letter (None: not a single-option row)."""

    spec = OPTION_SOURCES[benchmark]
    entry = mt.dataset_manifest(cfg)["datasets"][benchmark]
    path, sha = mt._metadata_tsv(cfg, entry)
    if mt.sha256_file(path) != sha:
        raise RescoreError(f"{benchmark}: licensed metadata sha256 differs from the dataset manifest")
    header = pd.read_csv(path, sep="\t", nrows=0).columns
    want = ["index", "answer"]
    for c in (spec.get("column"), "answer_option" if spec["gold"] == "answer_option" else None,
              (spec.get("row_filter") or (None,))[0]):
        if c and c not in want:
            want.append(c)
    if spec["kind"] == "letter_columns":
        want += [c for c in spec["columns"] if c in header]
    table = pd.read_csv(path, sep="\t", usecols=want, keep_default_na=False, dtype=str)
    media = entry["row_media"]
    if len(table) != len(media):
        raise RescoreError(f"{benchmark}: metadata rows {len(table)} != manifest rows {len(media)}")
    labels_out, gold_out = [], []
    for i, row in table.iterrows():
        if str(row["index"]) != str(media[i]["index"]):
            raise RescoreError(f"{benchmark}: metadata row {i} does not match the manifest index")
        if spec["kind"] == "letter_columns":
            labels = "".join(c for c in spec["columns"] if c in table.columns and str(row[c]).strip()
                             and str(row[c]).strip().lower() != "nan")
        elif spec["kind"] == "fixed":
            labels = spec["labels"]
        else:
            labels = LETTERS[:len(_literal_list(row[spec["column"]]))]
        answer = str(row["answer"]).strip()
        if spec["gold"] == "option_text":
            opts = [str(o).strip() for o in _literal_list(row[spec["column"]])]
            hits = [k for k, o in enumerate(opts) if o == answer]
            gold = LETTERS[hits[0]] if len(hits) == 1 else None
        elif spec["gold"] == "answer_option":
            gold = str(row["answer_option"]).strip() or None
        else:
            gold = answer if len(answer) == 1 and answer.isalpha() else None
        rf = spec.get("row_filter")
        if rf is not None:
            col, value = rf
            ok = bool(_literal_list(row[col])) if value == "nonempty" else str(row[col]).strip() == value
            gold = gold if ok else None
        labels_out.append(labels)
        gold_out.append(gold if gold is not None and gold in labels else None)
    return pd.DataFrame({"source_ordinal": np.arange(len(table), dtype=np.int64), "option_labels": labels_out,
                         "gold_option": gold_out})


def mc_scope_survey(cfg: StudyConfig, scored: pd.DataFrame, *, log=print) -> pd.DataFrame:
    """Development evidence for the scope: for every option benchmark x model, how often the gold-blind extractor's
    final option disagrees with the official outcome (rows of the development partition; dev-screen models seed 42)."""

    from trace_merge import ingest as ig

    local = [f"{l}.development_screen.s42" for l in (*P1_LOCAL, *P2_LOCAL)]
    out = []
    for bench in OPTION_SOURCES:
        om = load_option_meta(cfg, bench).set_index("source_ordinal")
        cols = ["model_label", "seed", "content_key", "source_ordinal", "response_text"]
        frames = [ig.load_aligned(cfg, columns=cols, labels=list(ARCHIVE_LABELS), seeds=list(SEEDS),
                                  benchmarks=[bench])]
        for name in local:
            frames.append(ig.load_aligned(cfg, columns=cols, benchmarks=[bench], include_archive=False,
                                          local_runs=[name]))
        resp = pd.concat(frames, ignore_index=True)
        s = scored[scored["benchmark_id"] == bench][["model_label", "seed", "content_key", "official_row_score"]]
        df = s.merge(resp, on=["model_label", "seed", "content_key"], how="inner", validate="one_to_one")
        m = om.loc[df["source_ordinal"].astype(int)]
        df["gold"] = m["gold_option"].to_numpy()
        df["labels"] = m["option_labels"].to_numpy()
        df = df[df["gold"].notna() & (df["labels"].str.len() > 0)]
        xs = [ex.extract_final_option(t, lab) for t, lab in zip(df["response_text"], df["labels"])]
        df["resolved"] = [x.resolved for x in xs]
        df["same"] = [x.option == g for x, g in zip(xs, df["gold"])]
        for label, g in df.groupby("model_label"):
            off1 = g["official_row_score"] == 1
            out.append({"benchmark_id": bench, "in_scope": bench in SCOPE, "model_label": label, "rows": len(g),
                        "official_accuracy": 100 * off1.mean(), "extractor_resolved_pct": 100 * g["resolved"].mean(),
                        "official0_final_option_is_gold": int((~off1 & g["resolved"] & g["same"]).sum()),
                        "official1_final_option_differs": int((off1 & g["resolved"] & ~g["same"]).sum())})
        log(f"[f0] scope survey {bench}: {len(df)} option rows")
    return pd.DataFrame(out)


# -------------------------------------------------------------------------------- TableVQABench failure taxonomy
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers(text: str) -> list[float]:
    vals = []
    for tok in _NUM.findall(str(text)):
        try:
            vals.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return vals


def _plain(text: str) -> str:
    s = re.sub(r"[^\w\s]", " ", str(text).lower())
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_UNIT_WORDS = {"million", "millions", "thousand", "thousands", "billion", "billions", "percent", "dollar", "dollars",
               "usd", "m", "k", "bn", "mn", "e", "episode", "no", "number", "year", "years", "points", "pts"}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", str(text).lower()))


def _same_number(pred: str, gold: str) -> bool:
    """Same numbers (thousands separators ignored), or an accounting negative '(x)' written as '-x'."""

    p, g = _numbers(pred), _numbers(gold)
    if not p or not g:
        return False
    if sorted(p) == sorted(g):
        return True
    return bool(re.search(r"\(\s*\$?\s*[\d,.]+\s*\)", str(gold))) and sorted(p) == sorted(-abs(v) for v in g)


def _classify_answer(p: str, gold: str, split: str) -> str:
    """Failure class of a single (possibly rewritten) wrong answer string."""

    if not p.strip():
        return "malformed_empty"
    if split == "vtabfact":
        low = p.lower()
        if "true" in low and "false" in low:
            return "malformed_tabfact_both_words"
        if "true" not in low and "false" not in low:
            return "malformed_tabfact_no_label"
        return "wrong_answer"
    if split in WTQ_SPLITS and "|" in p and "\n" not in p.strip():
        tv = mt._pinned("vlmeval.dataset.utils.tablevqabench")
        items = [i for i in p.replace("||", "|").split("|") if i.strip()]
        gold_items = [i for i in str(gold).split("|") if i.strip()]
        if len(items) > len(gold_items):
            with _quiet():
                hit = any(tv.check_denotation(tv.to_value_list(tv.tsv_unescape_list(str(gold))),
                                              tv.to_value_list(tv.tsv_unescape_list(i))) for i in items)
            return "extra_values_including_gold" if hit and len(gold_items) == 1 else "extra_values"
        if len(items) < len(gold_items):
            return "missing_values"
    if len(p) > 100 or "\n" in p.strip() or re.search(r"<answer>|</?think>|\\boxed|^\s*\**answer\**\s*:", p, re.I):
        return "malformed_prose_or_wrapper"
    if split in WTQ_SPLITS:
        items = [i for i in p.replace("||", "|").split("|") if i.strip()]
        if len(items) < len([i for i in str(gold).split("|") if i.strip()]):
            return "missing_values"
    if _same_number(p, gold):
        extra = _words(gold) - _words(p) - _UNIT_WORDS
        return "normalization_numeric_format" if not extra else "partial_answer"
    if _plain(p) and _plain(p) == _plain(gold):
        return "normalization_text_format"
    return "wrong_answer"


def tvqa_failure_type(pred: str, gold: str, split: str, primary: float, exact: float, normalized: float,
                      official: float, finish_reason: Any) -> str:
    """Failure class of one TableVQABench row (``correct`` when it has full official credit). Audit only.

    A row a rule rewrote without gaining credit is classified by its rewritten answer, so a wrong boxed value counts
    as a wrong answer, not as a wrapper problem."""

    full = primary == 1 and (split != "fintabnetqa" or exact == 1)
    if full:
        return "correct"
    p = "" if pred is None else str(pred)
    cand, applied, detail = tvqa_candidate(p, split)
    if normalized > official:
        return "extraction_fixed_boxed" if "T1_boxed" in applied else "extraction_fixed_empty_list_items"
    if split == "fintabnetqa" and primary == 1:
        return "fintab_relieved_only"
    if detail:
        return "extraction_boxed_conflict"
    if str(finish_reason) == "length":
        return "malformed_length_stop"
    return _classify_answer(cand if cand is not None else p, gold, split)


TVQA_FAILURE_GROUP = {
    "extraction_fixed_boxed": "extraction failure (fixed by v1)",
    "extraction_fixed_empty_list_items": "extraction failure (fixed by v1)",
    "extraction_boxed_conflict": "extraction failure (unresolved)",
    "malformed_empty": "malformed response", "malformed_length_stop": "malformed response",
    "malformed_prose_or_wrapper": "malformed response", "malformed_tabfact_both_words": "malformed response",
    "malformed_tabfact_no_label": "malformed response",
    "normalization_numeric_format": "numeric/text normalization (not fixed)",
    "normalization_text_format": "numeric/text normalization (not fixed)",
    "fintab_relieved_only": "partial credit by design (fintabnetqa exact)",
    "extra_values_including_gold": "real error: hedged list", "extra_values": "real error: hedged list",
    "missing_values": "real error", "wrong_answer": "real error",
    "partial_answer": "real error: partial answer (gold is a longer phrase)",
}


def tvqa_rows(cfg: StudyConfig, scored: pd.DataFrame, partition: str) -> pd.DataFrame:
    """TableVQABench rows of a partition with the scored prediction, split, pinned components and failure class."""

    from trace_merge import analysis as an

    metric = load_metrics(cfg, ["tablevqabench"])["tablevqabench"]
    local = ({l: [42] for l in (*P1_LOCAL, *P2_LOCAL)} if partition == "development"
             else {l: list(SEEDS) for l in P1_LOCAL})
    subset = "development_screen" if partition == "development" else "confirmation"
    cols = list(an.ROW_COLUMNS) + ["finish_reason", "completion_tokens"]
    rows = an.load_rows(cfg, labels=[*ARCHIVE_LABELS, *local], seeds=list(SEEDS), benchmarks=["tablevqabench"],
                        local=local, subset=subset, columns=cols)
    rows = an.restrict(rows, an.assignments(cfg), partition)
    s = scored[scored["benchmark_id"] == "tablevqabench"][["model_label", "seed", "content_key", "in_dev_screen",
                                                           "official_row_score", "normalized_row_score", "fn_status"]]
    rows = rows.merge(s, on=["model_label", "seed", "content_key"], how="inner", validate="one_to_one")
    meta = metric.aligned_meta(rows)
    rows["split"] = meta["split"].to_numpy()
    rows["gold"] = meta["answer"].to_numpy()
    rows["scored_prediction"] = mt.scored_prediction_view(rows)
    comp = _tvqa_components(rows["scored_prediction"].tolist(), rows["gold"].tolist(), rows["split"].tolist())
    rows["primary"], rows["exact"] = comp[:, 0], comp[:, 1]
    rows["failure_type"] = [tvqa_failure_type(*a) for a in zip(
        rows["scored_prediction"], rows["gold"], rows["split"], rows["primary"], rows["exact"],
        rows["normalized_row_score"], rows["official_row_score"], rows["finish_reason"])]
    rows["failure_group"] = rows["failure_type"].map(TVQA_FAILURE_GROUP).fillna("correct")
    keep = ["model_label", "seed", "content_key", "in_dev_screen", "split", "gold", "scored_prediction",
            "finish_reason", "completion_tokens", "primary", "exact", "official_row_score", "normalized_row_score",
            "fn_status", "failure_type", "failure_group"]
    return rows[keep]


def tvqa_failure_table(rows: pd.DataFrame, subsets: Mapping[str, tuple[list[str], list[int], str]]) -> pd.DataFrame:
    out = []
    for name, (labels, seeds, filt) in subsets.items():
        sub = _subset_rows(rows, labels, seeds, filt)
        for (label, split), g in sub.groupby(["model_label", "split"]):
            n = len(g)
            for ftype, c in g["failure_type"].value_counts().items():
                out.append({"subset": name, "model_label": label, "split": split, "failure_type": ftype,
                            "failure_group": TVQA_FAILURE_GROUP.get(ftype, "correct"), "rows": int(c),
                            "share_of_split_rows_pct": 100.0 * c / n, "split_rows": n})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------- blinded manual check
def blind_sample(cfg: StudyConfig, scored: pd.DataFrame, tvqa: pd.DataFrame, *, seed: int = 20261105,
                 sizes: Mapping[str, int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A stratified development sample for the blinded check: (items shown to the reviewer, key kept apart).

    Items carry no model label, seed, status or extracted value: the reviewer reads the response tail (and option
    texts / gold) and records the final answer the response designates and whether its reasoning supports it; the key
    is joined afterwards."""

    from trace_merge import ingest as ig

    sizes = dict(sizes or {"mc:repaired": 24, "mc:resolved_still_wrong": 6, "mc:unresolved": 5, "mc:official_read": 5,
                           "tvqa:repaired": 10, "tvqa:resolved_still_wrong": 4, "tvqa:no_rule_applies": 6})
    rng = np.random.default_rng(seed)
    dev = scored[scored["benchmark_id"].isin(list(SCOPE))].copy()
    dev["family"] = np.where(dev["benchmark_id"] == "tablevqabench", "tvqa", "mc")
    picks = []
    for stratum, n in sizes.items():
        fam, status = stratum.split(":")
        pool = dev[(dev["family"] == fam) & (dev["fn_status"] == status)]
        if len(pool):
            picks.append(pool.iloc[rng.choice(len(pool), size=min(n, len(pool)), replace=False)].assign(stratum=stratum))
    sample = pd.concat(picks, ignore_index=True)
    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    sample["blind_id"] = [f"F0-{i + 1:03d}" for i in range(len(sample))]
    texts = []
    for bench, g in sample.groupby("benchmark_id"):
        names = {f"{l}.development_screen.s42" for l in (*P1_LOCAL, *P2_LOCAL)}
        cols = ["model_label", "seed", "content_key", "response_text", "extraction_value_json"]
        frames = [ig.load_aligned(cfg, columns=cols, labels=list(ARCHIVE_LABELS), seeds=list(SEEDS),
                                  benchmarks=[bench])]
        frames += [ig.load_aligned(cfg, columns=cols, benchmarks=[bench], include_archive=False, local_runs=[n])
                   for n in sorted(names)]
        resp = pd.concat(frames, ignore_index=True)
        texts.append(g[["blind_id", "model_label", "seed", "content_key"]].merge(
            resp, on=["model_label", "seed", "content_key"], how="left", validate="one_to_one"))
    texts = pd.concat(texts, ignore_index=True).set_index("blind_id")
    om = {b: load_option_meta(cfg, b).set_index("source_ordinal") for b in ("puzzlevqa", "visualpuzzles")}
    pz_meta = mt.load_benchmark_meta(cfg, "puzzlevqa").set_index("source_ordinal")
    tv = tvqa.set_index(["model_label", "seed", "content_key"])
    items = []
    for r in sample.itertuples():
        t = texts.loc[r.blind_id]
        item = {"blind_id": r.blind_id, "benchmark_id": r.benchmark_id}
        if r.benchmark_id == "tablevqabench":
            row = tv.loc[(r.model_label, r.seed, r.content_key)]
            item.update(split=row["split"], gold=str(row["gold"]), scored_prediction=str(row["scored_prediction"])[-900:])
        else:
            o = om[r.benchmark_id].loc[int(r.source_ordinal)]
            item.update(option_labels=o["option_labels"], gold=o["gold_option"],
                        option_texts=(str(pz_meta.loc[int(r.source_ordinal), "options"])
                                      if r.benchmark_id == "puzzlevqa" else "(options are in the image)"),
                        response_tail=str(t["response_text"])[-900:])
        items.append(item)
    key = sample[["blind_id", "stratum", "benchmark_id", "model_label", "seed", "content_key", "official_row_score",
                  "normalized_row_score", "fn_status", "fn_extracted", "fn_detail"]]
    return pd.DataFrame(items), key


# --------------------------------------------------------------------------------------------------------- driver
def run_partition(partition: str, out_root: Path, *, cfg: StudyConfig | None = None, n_replicates: int = 10_000,
                  log=print) -> dict[str, Any]:
    """Score one partition of the reviewed study's stored outputs and write the F0 tables for it."""

    from trace_merge import analysis as an

    cfg = cfg or load_config()
    metrics = load_metrics(cfg)
    work = out_root / "audit" / "f0_work"
    tables = out_root / "tables" / "f0"
    work.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    scored = score_partition(cfg, partition, metrics=metrics, log=log)
    scored.to_parquet(work / f"rows.{partition}.parquet", index=False)
    subsets = _subsets(partition)
    ga = an.assignments(cfg)
    stats_cfg = cfg.raw["statistics"]
    summaries, contrasts, checks, values = [], [], [], []
    for name, (labels, seeds, filt) in subsets.items():
        sub = _subset_rows(scored, labels, seeds, filt)
        vals = values_long(sub, metrics)
        values.append(vals.assign(subset=name))
        checks.append(reproduction_check(cfg, name, vals))
        summaries.append(summary_table(vals, cfg.categories(), name))
        res = paired_contrasts(sub, metrics, ga, labels=labels, seeds=seeds, comparisons=CONTRASTS[name],
                               categories=cfg.categories(), n_replicates=n_replicates, seed=int(stats_cfg["seed"]),
                               confidence=float(stats_cfg["confidence"]))
        contrasts.append(contrast_table(res, name))
        if n_replicates == int(stats_cfg["bootstrap_replicates"]):
            checks.append(contrast_reproduction(cfg, name, contrasts[-1]))
        log(f"[f0] {partition}/{name}: values, reproduction {checks[-1]['passed']}, contrasts done")
    pd.concat(values, ignore_index=True).to_csv(tables / f"seed_values.{partition}.csv", index=False)
    pd.concat(summaries, ignore_index=True).assign(partition=partition).to_csv(
        tables / f"official_vs_normalized.{partition}.csv", index=False)
    pd.concat(contrasts, ignore_index=True).assign(partition=partition).to_csv(
        tables / f"contrasts.{partition}.csv", index=False)
    change_ledger(scored, partition).to_csv(work / f"scoring_changes.{partition}.csv", index=False)
    status_counts(scored, subsets).to_csv(tables / f"status_counts.{partition}.csv", index=False)
    tvqa = tvqa_rows(cfg, scored, partition)
    tvqa.to_parquet(work / f"tvqa_rows.{partition}.parquet", index=False)
    tvqa_failure_table(tvqa, subsets).to_csv(tables / f"tablevqa_failures.{partition}.csv", index=False)
    summary = {"partition": partition, "metric": METRIC_NAME, "reproduction": checks,
               "rows": int(len(scored)), "repaired_rows": int((scored["fn_status"] == REPAIRED).sum()),
               "n_replicates": n_replicates}
    if partition == "development":
        mc_scope_survey(cfg, scored, log=log).to_csv(tables / "mc_scope_survey.development.csv", index=False)
        items, key = blind_sample(cfg, scored, tvqa)
        items.to_json(work / "blind_check.items.jsonl", orient="records", lines=True, force_ascii=False)
        key.to_csv(work / "blind_check.key.csv", index=False)
    (work / f"run_summary.{partition}.json").write_text(pd.Series(summary).to_json(indent=1), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import os

    ap = argparse.ArgumentParser(description=f"F0: {METRIC_NAME} on the reviewed study's stored outputs")
    ap.add_argument("partition", choices=["development", "confirmation"])
    ap.add_argument("--out-root", default=os.environ.get("TRACE_FOLLOWUP_ROOT"),
                    help="follow-up work root (default TRACE_FOLLOWUP_ROOT)")
    ap.add_argument("--replicates", type=int, default=10_000)
    args = ap.parse_args(argv)
    if not args.out_root:
        raise SystemExit("--out-root or TRACE_FOLLOWUP_ROOT is required")
    print(run_partition(args.partition, Path(args.out_root), n_replicates=args.replicates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
