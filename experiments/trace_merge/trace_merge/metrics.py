"""Metric registry and official-metric adapters for the 24 suite benchmarks (handoff 4.3, DESIGN.md section 5).

Why this module exists: the suite's headline number is a macro over 24 *official* benchmark metrics, and several of
them are not pooled row accuracy (TableVQABench averages five split scores, WeMath's "Score (Strict)" is a
problem-group statistic). A subset or bootstrap replicate must therefore recompute each official metric, not a row
mean. Every benchmark gets one ``BenchmarkMetric`` whose per-unit statistics are *additive*, so
``statistics.paired_cluster_bootstrap`` can sum multiplicity-weighted unit statistics and call ``finalize``.

Kinds (DESIGN.md section 5):

* ``binary_row`` (16 benchmarks): the archived 0/1 row scores; every pinned scorer's primary value is the pooled
  row mean (checked in the pinned code, see ``SPECS[...].code_refs``), so unit stats are ``[sum, rows]``.
* ``grouped_row`` (WeMath): archived 0/1 question scores plus the licensed ``ID``/``key`` metadata reproduce the
  pinned ``Score (Strict)``; unit stats are counts of problem-group outcomes.
* ``aggregate_only_rederived`` (7 benchmarks): the archive holds only the aggregate. Row scores are re-derived from
  the archived extraction rows (the exact cell the pinned scorer read) and the licensed answers by calling the
  pinned scorer functions from the prepared protocol clone. No judge is called; every route needed here is either
  judge-free or its judge output is the archived extraction value.
* ``unsupported``: a benchmark whose adapter failed reproduction; ``build_metrics`` returns it as such.

An adapter is accepted only when ``official_score`` on the full split reproduces the archived aggregate of all 9
model x seed slices to 1e-6 percent (``reproduce_archive``; CLI ``python -m trace_merge.metrics reproduce``). The
outcome goes to ``audit/metric_reproduction.json`` and ``manifests/metric_registry.json``.

Protocol code is imported, never copied or edited: run with the prepared clone's documented PYTHONPATH, or pass
``--protocol-clone``, which puts the same paths ahead of site-packages and refuses to continue when a pinned
dependency is already loaded at another version.

Subset semantics differ from the pinned code for one benchmark only. WeMath's pinned "Score (Strict)" divides by a
fixed 525 groups, which on a subset scores every absent group as mastered; ``finalize`` divides by the groups present
(identical on the full split) and ``campaign_value`` gives the pinned fixed-525 value, which is what a subset run's
own score campaign reports and what its cross-check compares (``crosscheck_local_rerun``).
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import functools
import gc
import importlib
import io
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import ConfigError, StudyConfig, canonical_json, load_config, sha256_bytes, sha256_file, write_json

KINDS = ("binary_row", "grouped_row", "aggregate_only_rederived", "unsupported")
REPRODUCTION_TOLERANCE_PP = 1e-6
ARCHIVE_LABELS = {"B": "B", "T": "T", "V": "V@P0"}
ROLES = ("B", "T", "V")
REGISTRY_SCHEMA = "trace_merge_metric_registry_v1"
REPRODUCTION_SCHEMA = "trace_merge_metric_reproduction_v1"

# The pinned MathVision ``is_equal`` is decorated ``timeout_decorator.timeout(30, use_signals=False)``.
MATHV_IS_EQUAL_TIMEOUT_S = 30

try:  # pandas' default ``na_values`` (read_excel / read_csv); the pinned scorers read workbooks with these defaults
    from pandas._libs.parsers import STR_NA_VALUES as _PANDAS_NA
except ImportError:  # pragma: no cover - fallback for a pandas without the private constant
    _PANDAS_NA = {"", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>",
                  "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"}
PANDAS_NA_STRINGS = frozenset(_PANDAS_NA)


class MetricError(ValueError):
    """A metric cannot be computed faithfully on these rows; the caller must not fall back to a guess."""


# ------------------------------------------------------------------------------------------------------- specs
VK = "external/VLMEvalKit/vlmeval/"
SC = "rlvr/evaluation/scripts/"
RUNNER = "external/VLMEvalKit/scripts/batched_vlmevalkit_qwen3vl.py"
MCQ_POOLED = (VK + "dataset/image_mcq.py::ImageMCQDataset.evaluate_heuristic",
              VK + "dataset/utils/multiple_choice.py::report_acc",
              VK + "dataset/image_mcq.py::ImageMCQDataset.report_primary_metric")
SPATIAL_POOLED = VK + "dataset/utils/spatial_bench/cal_scores.py::eval_mcq_score"


@dataclass(frozen=True)
class MetricSpec:
    """What the pinned scorer computes for one benchmark, with the code that shows it."""

    benchmark_id: str
    kind: str
    route: str
    official_alias: str
    official_metric_key: str              # key the pinned scorer reports as the primary value
    aggregation: str                      # row_mean | split_macro | wemath_strict_groups
    row_score_type: str                   # binary | fractional | component_mean | binary_question
    code_refs: tuple[str, ...]
    meta_columns: tuple[str, ...] = ()
    official_strata: tuple[str, ...] = ()  # columns the pinned scorer divides by (it raises if a value is absent)
    notes: str = ""


SPECS: dict[str, MetricSpec] = {}


def _spec(**kw: Any) -> None:
    spec = MetricSpec(**kw)
    SPECS[spec.benchmark_id] = spec


# binary_row: pinned primary value = pooled row mean (verified in each referenced function)
_spec(benchmark_id="charxivreason", kind="binary_row", route="direct_score", official_alias="CharXiv_reasoning_val",
      official_metric_key="Overall", aggregation="row_mean", row_score_type="binary",
      code_refs=(SC + "run_external_benchmark_score_queue.py::_run_charxiv_local_judge",
                 SC + "run_external_benchmark_score_queue.py::_charxiv_primary_score"),
      notes="scores['Overall'] = sum(row judge scores) / rows; the per-category means are not averaged")
_spec(benchmark_id="evochart", kind="binary_row", route="direct_score", official_alias="EvoChart",
      official_metric_key="Overall", aggregation="row_mean", row_score_type="binary",
      code_refs=(SC + "run_external_benchmark_score_queue.py::_run_evochart_local_score",
                 VK + "dataset/evochart.py::EvoChart.evaluate"),
      notes="Overall row: sum(scores) / len(scores) * 100")
_spec(benchmark_id="mathverse", kind="binary_row", route="direct_score",
      official_alias="MathVerse_MINI_Vision_Only_cot", official_metric_key="Overall", aggregation="row_mean",
      row_score_type="binary",
      code_refs=(RUNNER + "::run_mathverse_local_judge", VK + "dataset/utils/mathverse.py::MathVerse_acc",
                 SC + "run_external_benchmark_score_queue.py::_normalize_mathverse_binary_judgement_summary"),
      notes="one problem_version (Vision Only), so 'Overall' = np.mean(score) * 100 over all rows")
_spec(benchmark_id="logicvista", kind="binary_row", route="direct_score", official_alias="LogicVista",
      official_metric_key="Overall", aggregation="row_mean", row_score_type="binary",
      code_refs=(RUNNER + "::run_logicvista_local_judge", VK + "dataset/utils/logicvista.py::evaluate_logicvista",
                 SC + "run_external_benchmark_score_queue.py::_normalize_logicvista_option_summary"),
      notes="acc['Overall'] = hit.sum() / rows * 100; skill accuracies are reported separately")
_spec(benchmark_id="phyx_mini_mc", kind="binary_row", route="official_vlmevalkit", official_alias="PhyX_mini_MC",
      official_metric_key="Overall", aggregation="row_mean", row_score_type="binary",
      code_refs=(SC + "run_official_vlmevalkit_saved_score.py::_adapt_phyx_option_prediction",
                 VK + "dataset/image_vqa.py::PhyX.evaluate"),
      notes="valid_type STR: ret['Overall'] = np.mean(hit)")
_spec(benchmark_id="mmmu_pro_vision", kind="binary_row", route="official_vlmevalkit", official_alias="MMMU_Pro_V_COT",
      official_metric_key="Overall Acc", aggregation="row_mean", row_score_type="binary",
      code_refs=(VK + "dataset/image_mcq.py::MMMUProDataset.evaluate",) + MCQ_POOLED,
      notes="COT post-processing then report_acc: 'Overall' = np.mean(hit) over split 'none'")
_spec(benchmark_id="realworldqa", kind="binary_row", route="official_vlmevalkit", official_alias="RealWorldQA",
      official_metric_key="Overall Acc", aggregation="row_mean", row_score_type="binary",
      code_refs=(SC + "run_official_vlmevalkit_saved_score.py::_adapt_realworldqa_prediction",) + MCQ_POOLED)
_spec(benchmark_id="mmstar", kind="binary_row", route="official_vlmevalkit", official_alias="MMStar",
      official_metric_key="Overall Acc", aggregation="row_mean", row_score_type="binary", code_refs=MCQ_POOLED,
      notes="six categories of 250 rows: report_acc's 'Overall' is the pooled mean, which equals the category macro "
            "only on the balanced full split; the pooled mean is what the pinned scorer reports")
_spec(benchmark_id="embspatial", kind="binary_row", route="official_vlmevalkit", official_alias="EmbSpatialBench",
      official_metric_key="overall", aggregation="row_mean", row_score_type="binary",
      code_refs=(VK + "dataset/embspatialbench.py::EmbSpatialBench.evaluate", SPATIAL_POOLED),
      notes="summary['overall'] = mcq_scored['hit'].mean() * 100")
_spec(benchmark_id="spatialvizbench_cot", kind="binary_row", route="official_vlmevalkit",
      official_alias="SpatialVizBench_CoT", official_metric_key="overall", aggregation="row_mean",
      row_score_type="binary",
      code_refs=(VK + "dataset/spatialvizbench.py::SpatialVizBench.evaluate", SPATIAL_POOLED))
_spec(benchmark_id="cvbench_3d", kind="binary_row", route="official_vlmevalkit", official_alias="CV-Bench-3D",
      official_metric_key="split=3D|Overall", aggregation="row_mean", row_score_type="binary",
      code_refs=(VK + "dataset/image_mcq.py::CVBench.evaluate", VK + "dataset/utils/multiple_choice.py::report_acc"),
      notes="3D uses report_acc (pooled over split 3D); only CV-Bench-2D uses the per-source macro report_accuracy")
_spec(benchmark_id="erqa", kind="binary_row", route="official_vlmevalkit", official_alias="ERQA",
      official_metric_key="overall", aggregation="row_mean", row_score_type="binary",
      code_refs=(VK + "dataset/erqabench.py::ERQABench.evaluate", SPATIAL_POOLED))
_spec(benchmark_id="blink", kind="binary_row", route="official_vlmevalkit", official_alias="BLINK",
      official_metric_key="Overall Acc", aggregation="row_mean", row_score_type="binary", code_refs=MCQ_POOLED,
      notes="14 subtasks; report_acc's 'Overall' is the pooled mean over split 'none'")
_spec(benchmark_id="countqa", kind="binary_row", route="official_vlmevalkit", official_alias="CountQA",
      official_metric_key="split=Overall|acc", aggregation="row_mean", row_score_type="binary",
      code_refs=(VK + "dataset/trace_eval_local_vqa.py::CountQA.evaluate",))
_spec(benchmark_id="treebench", kind="binary_row", route="official_vlmevalkit", official_alias="TreeBench",
      official_metric_key="Overall", aggregation="row_mean", row_score_type="binary",
      code_refs=(SC + "run_official_vlmevalkit_saved_score.py::_adapt_treebench_prediction",
                 VK + "dataset/image_mcq.py::TreeBench.evaluate",
                 VK + "dataset/utils/treebench.py::get_dimension_rating"),
      notes="results['Overall'] = succ_all / sum_all, pooled over all dimensions (rows with score >= 0, i.e. all)")
_spec(benchmark_id="mme_reasoning", kind="binary_row", route="dedicated_score", official_alias="MME-Reasoning",
      official_metric_key="Overall", aggregation="row_mean", row_score_type="binary",
      code_refs=(SC + "run_mme_reasoning_eval.py",),
      notes="result['Overall'] = data['score'].mean() * 100")

# grouped_row
_spec(benchmark_id="wemath", kind="grouped_row", route="official_vlmevalkit", official_alias="WeMath_COT",
      official_metric_key="Score (Strict)", aggregation="wemath_strict_groups", row_score_type="binary_question",
      code_refs=(VK + "dataset/image_mcq.py::WeMath.evaluate", VK + "dataset/utils/wemath.py::wemath_evaluate_models",
                 VK + "dataset/utils/wemath.py::evaluate_calculate_metrics",
                 VK + "dataset/utils/wemath.py::evaluate_update_main_results_df",
                 SC + "run_trace_eval_official_score_campaign.py::_primary_metric_contract"),
      meta_columns=("ID", "key"),
      notes="per problem group (ID): CM = all steps and the multi-step question correct, IG = all steps correct but "
            "multi wrong, RM = some step wrong but multi correct, IK = some step wrong and multi wrong; pinned "
            "Score (Strict) = (525 - 0.5 IG - RM - IK) / 525 formatted '{:.2%}' (two decimals)")

# aggregate_only_rederived
_spec(benchmark_id="chartqapro", kind="aggregate_only_rederived", route="official_vlmevalkit",
      official_alias="ChartQAPro_CoT", official_metric_key="Overall Acc", aggregation="row_mean",
      row_score_type="fractional",
      code_refs=(SC + "run_official_vlmevalkit_saved_score.py::_adapt_chartqapro_prediction",
                 VK + "dataset/chartqapro.py::ChartQAPro.get_scores",
                 VK + "dataset/utils/chartqapro.py::evaluate_predictions_chartqapro"),
      meta_columns=("answer", "question_type", "year"),
      notes="final_numbers['Overall'] = sum(relaxed correctness) / rows; per-row relaxed correctness/ANLS is "
            "fractional")
_spec(benchmark_id="tablevqabench", kind="aggregate_only_rederived", route="direct_score",
      official_alias="TableVQABench", official_metric_key="Overall", aggregation="split_macro",
      row_score_type="component_mean",
      code_refs=(SC + "run_external_benchmark_score_queue.py::_run_tablevqabench_local_score",
                 VK + "dataset/utils/tablevqabench.py::evaluate_fintabnet",
                 VK + "dataset/utils/tablevqabench.py::evaluate_tabfact",
                 VK + "dataset/utils/tablevqabench.py::evaluate_wtq",
                 VK + "dataset/utils/trace_eval_answer_parsing.py::unwrap_single_answer_block"),
      meta_columns=("answer", "split"), official_strata=("split",),
      notes="Overall = mean of the five average_scores [fintabnetqa relieved, fintabnetqa exact, vtabfact, vwtq, "
            "vwtq_syn], each round((correct + 1e-9) / (rows + 1e-9), 8) * 100; not a pooled row mean")
_spec(benchmark_id="mathvision", kind="aggregate_only_rederived", route="direct_score", official_alias="MathVision",
      official_metric_key="Overall", aggregation="row_mean", row_score_type="binary",
      code_refs=(RUNNER + "::run_mathv_local_judge", VK + "dataset/utils/mathv.py::MATH_V_acc",
                 VK + "dataset/utils/mathv.py::post_check", VK + "dataset/utils/mathv.py::is_equal"),
      meta_columns=("answer", "choices", "category"),
      notes="hit = post_check(row, prefetch=False) on the judged table's 'res' (prefetch or archived judge output)")
_spec(benchmark_id="mathvista", kind="aggregate_only_rederived", route="direct_score", official_alias="MathVista_MINI",
      official_metric_key="Overall", aggregation="row_mean", row_score_type="binary",
      code_refs=(RUNNER + "::run_mathvista_local_judge", VK + "dataset/utils/mathvista.py::MathVista_acc",
                 VK + "dataset/utils/mathvista.py::post_check"),
      meta_columns=("answer", "answer_option", "choices", "question_type", "answer_type", "task"))
_spec(benchmark_id="countbenchqa", kind="aggregate_only_rederived", route="official_vlmevalkit",
      official_alias="CountBenchQA", official_metric_key="accuracy", aggregation="row_mean", row_score_type="binary",
      code_refs=(VK + "dataset/image_vqa.py::CountBenchQA.evaluate",), meta_columns=("answer",),
      notes="hit = str(answer) in str(prediction) on the raw prediction cell")
_spec(benchmark_id="puzzlevqa", kind="aggregate_only_rederived", route="official_vlmevalkit",
      official_alias="PuzzleVQA", official_metric_key="category=overall|acc", aggregation="row_mean",
      row_score_type="binary",
      code_refs=(VK + "dataset/image_mcq.py::PuzzleVQA.evaluate", VK + "dataset/utils/puzzlevqa.py::PuzzleVQA_acc",
                 VK + "dataset/utils/trace_eval_answer_parsing.py::extract_unambiguous_abcd"),
      meta_columns=("options", "answer", "category"), official_strata=("category",),
      notes="primary value is the pooled 'overall' acc; the pinned function also divides by each of its 20 listed "
            "categories and raises ZeroDivisionError on a subset that lacks one")
_spec(benchmark_id="visualpuzzles", kind="aggregate_only_rederived", route="official_vlmevalkit",
      official_alias="VisualPuzzles", official_metric_key="group=overall|type=category|acc", aggregation="row_mean",
      row_score_type="binary",
      code_refs=(VK + "dataset/image_mcq.py::VisualPuzzles.evaluate",
                 VK + "dataset/utils/visualpuzzles.py::VisulPuzzles_acc",
                 VK + "dataset/utils/trace_eval_answer_parsing.py::extract_unambiguous_abcd"),
      meta_columns=("answer", "category", "difficulty"), official_strata=("category", "difficulty"),
      notes="primary value is the pooled 'overall' acc; the pinned function divides by every listed category and "
            "difficulty and raises on a subset that lacks one")

BINARY_ROW_BENCHMARKS = tuple(k for k, s in SPECS.items() if s.kind == "binary_row")
GROUPED_ROW_BENCHMARKS = tuple(k for k, s in SPECS.items() if s.kind == "grouped_row")
AGGREGATE_ONLY_BENCHMARKS = tuple(k for k, s in SPECS.items() if s.kind == "aggregate_only_rederived")
TABLEVQA_SPLITS = ("fintabnetqa", "vtabfact", "vwtq", "vwtq_syn")
WEMATH_KEYS = {"2steps": ("2steps_1", "2steps_2", "2steps_multi"),
               "3steps": ("3steps_1", "3steps_2", "3steps_3", "3steps_multi")}
WEMATH_PINNED_GROUPS = 525


# ------------------------------------------------------------------------------------------- protocol imports
def protocol_paths(clone: Path) -> list[Path]:
    """The import roots of the prepared protocol clone, in the documented PYTHONPATH order."""

    return [clone / "rlvr/evaluation/.work/eval_deps", clone / "src", clone / "rlvr/evaluation/scripts",
            clone / "external/VLMEvalKit", clone / "external/VLMEvalKit/scripts"]


def ensure_protocol_imports(clone: Path | None) -> None:
    """Make the pinned scorers importable without copying them, with the pinned dependency versions.

    The documented PYTHONPATH puts the clone's ``eval_deps`` (pinned scorer dependencies, e.g. antlr4 4.11.1, which
    ``latex2sympy2_extended`` uses to pick its parser) ahead of site-packages. When the paths are missing they are
    inserted where PYTHONPATH would put them (right after the script directory), not appended, so this environment's
    own copies (antlr4 4.9.3 here) cannot shadow the pinned ones. A pinned package that was already imported from
    elsewhere cannot be swapped any more; unless its version equals the pinned one this fails closed.
    """

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if clone is None:
        return
    os.environ.setdefault("LMUData", str(clone / "rlvr/evaluation/.work/LMUData"))
    wanted = [str(p) for p in protocol_paths(clone) if p.is_dir()]
    real = [os.path.realpath(p) for p in wanted]
    current = [os.path.realpath(p) if p else p for p in sys.path]
    positions = [current.index(r) if r in current else -1 for r in real]
    first_site = next((i for i, p in enumerate(current) if p and "site-packages" in p), len(current))
    in_place = all(0 <= p < first_site for p in positions) and positions == sorted(positions)
    if not in_place:
        keep = [p for p, r in zip(sys.path, current) if r not in set(real)]
        at = 1 if keep else 0
        sys.path[:] = keep[:at] + wanted + keep[at:]
        importlib.invalidate_caches()
    deps = clone / "rlvr/evaluation/.work/eval_deps"
    if deps.is_dir():
        check_pinned_dependencies(deps)


def _pinned_dependency_versions(deps: Path) -> dict[str, str]:
    """Top-level import name -> pinned distribution version, read from the dist-info RECORD/METADATA in eval_deps."""

    out: dict[str, str] = {}
    for info in sorted(deps.glob("*.dist-info")):
        meta, record = info / "METADATA", info / "RECORD"
        if not meta.is_file() or not record.is_file():
            continue
        version = next((line.split(":", 1)[1].strip() for line in meta.read_text(errors="replace").splitlines()
                        if line.startswith("Version:")), None)
        if version is None:
            continue
        for line in record.read_text(errors="replace").splitlines():
            top = line.split(",", 1)[0].split("/", 1)[0]
            if not top or top.startswith(".") or top in ("bin", "__pycache__") or top.endswith((".dist-info",
                                                                                                ".libs")):
                continue
            name = top[:-3] if top.endswith(".py") else top.split(".", 1)[0]
            out.setdefault(name, version)
    return out


def check_pinned_dependencies(deps: Path) -> None:
    """Fail if a pinned scorer dependency is loaded from outside ``eval_deps`` at a different version."""

    root = os.path.realpath(deps) + os.sep
    bad: list[str] = []
    for name, version in sorted(_pinned_dependency_versions(deps).items()):
        module = sys.modules.get(name)
        if module is None:
            continue
        where = getattr(module, "__file__", None)
        if where and os.path.realpath(where).startswith(root):
            continue
        loaded = getattr(module, "__version__", None)
        if loaded is None or str(loaded) != version:
            bad.append(f"{name} {loaded} (pinned {version})")
    if bad:
        raise MetricError(f"pinned scorer dependencies already loaded from this environment at other versions: {bad}; "
                          "run with the prepared clone's documented PYTHONPATH")


def _pinned(module: str) -> Any:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return importlib.import_module(module)
    except Exception as error:  # noqa: BLE001 - any import failure means the pinned scorer is unavailable
        raise MetricError(f"pinned scorer module {module} is not importable ({type(error).__name__}: {error}); put "
                          "the prepared protocol clone on PYTHONPATH or pass --protocol-clone") from error


@contextlib.contextmanager
def _in_process_is_equal(mathv: Any, diag: dict[str, int]) -> Iterator[None]:
    """Run MathVision's ``is_equal`` in-process with the same 30 s limit.

    The pinned decorator forks one process per comparison (``use_signals=False``), ~0.2 s each, which makes 27k
    re-derived rows take hours. The same library's signal-based limit around the same undecorated function has the
    same semantics (``TimeoutError`` inside ``post_check``'s ``except Exception`` -> row incorrect). Timeouts are
    counted. Only the module attribute is swapped for the duration; no file is changed.
    """

    original = mathv.is_equal
    inner = getattr(original, "__wrapped__", None)
    if inner is None or threading.current_thread() is not threading.main_thread():
        yield
        return
    td = _pinned("timeout_decorator")
    timed = td.timeout(MATHV_IS_EQUAL_TIMEOUT_S, use_signals=True)(inner)

    def counted(*args: Any, **kwargs: Any) -> Any:
        try:
            return timed(*args, **kwargs)
        except td.TimeoutError:
            diag["is_equal_timeouts"] = diag.get("is_equal_timeouts", 0) + 1
            raise

    mathv.is_equal = counted
    try:
        yield
    finally:
        mathv.is_equal = original


# -------------------------------------------------------------------------------------------- cell views
_XLSX_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f]")
XLSX_MAX_STRING = 32767


def xlsx_cell_text(value: Any) -> Any:
    """A string as pandas reads it back after the pinned pipeline wrote it with xlsxwriter.

    xlsxwriter cuts text at Excel's 32,767 characters and writes control characters as ``_xHHHH_`` escapes that
    openpyxl does not undo; the scorers saw the escaped text (e.g. MathVision answer 2830). A string starting with
    '=' would become a formula, which no metadata cell does; it fails closed. Non-strings pass through.
    """

    if not isinstance(value, str):
        return value
    if value.startswith("=") and len(value) > 1:
        raise MetricError(f"metadata cell {value[:40]!r} would be written as an xlsx formula")
    text = value[:XLSX_MAX_STRING]
    text = _XLSX_CONTROL.sub(lambda m: f"_x{ord(m.group(0)):04X}_", text)
    return text.replace("￾", "_xFFFE_").replace("￿", "_xFFFF_")


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _decode(raw: Any) -> Any:
    return None if _is_null(raw) else json.loads(raw)


def _judge_final_output_empty(judge_output: Any) -> bool:
    """True when the last archived judge event is blank, i.e. the judged table's 'res' cell was empty."""

    if _is_null(judge_output):
        return False
    try:
        events = json.loads(judge_output)
    except (TypeError, ValueError):
        return False
    return isinstance(events, list) and bool(events) and str(events[-1]).strip() == ""


def official_route_view(rows: pd.DataFrame, diag: dict[str, int] | None = None) -> list[Any]:
    """The prediction cell the pinned ``dataset.evaluate`` read, for the official saved-workbook route.

    The archive stored exactly that cell (``pd.read_excel`` of the evaluator input, default NA handling); a cell
    pandas read as NaN, or blank text, was stored as ``invalid`` with a null value, so it is NaN again here.
    """

    out: list[Any] = []
    for status, raw in zip(rows["extraction_status"], rows["extraction_value_json"]):
        value = _decode(raw)
        if status != "resolved" or value is None:
            if diag is not None:
                diag["invalid_extraction_as_nan"] = diag.get("invalid_extraction_as_nan", 0) + 1
            out.append(math.nan)
        else:
            out.append(value)
    return out


def judged_res_view(rows: pd.DataFrame, diag: dict[str, int] | None = None) -> list[Any]:
    """The judged table's 'res' cell as MATH_V_acc / MathVista_acc read it.

    The archive read that table with ``keep_default_na=False`` and fell back to the raw prediction when 'res' was
    blank, while the scorer read it with pandas' default NA strings ('None', 'NA', 'nan', ...). So: a blank final
    judge output or an invalid extraction -> NaN, and an NA string -> NaN.
    """

    out: list[Any] = []
    d = diag if diag is not None else {}
    for status, raw, judge in zip(rows["extraction_status"], rows["extraction_value_json"], rows["judge_output"]):
        value = _decode(raw)
        if status != "resolved" or value is None:
            d["invalid_extraction_as_nan"] = d.get("invalid_extraction_as_nan", 0) + 1
            out.append(math.nan)
        elif _judge_final_output_empty(judge):
            d["blank_judge_res_as_nan"] = d.get("blank_judge_res_as_nan", 0) + 1
            out.append(math.nan)
        elif isinstance(value, str) and value in PANDAS_NA_STRINGS:
            d["na_string_res_as_nan"] = d.get("na_string_res_as_nan", 0) + 1
            out.append(math.nan)
        else:
            out.append(value)
    return out


def scored_prediction_view(rows: pd.DataFrame, diag: dict[str, int] | None = None) -> list[str]:
    """TableVQABench's in-memory prediction (after the pinned unwrap and '^Answer: ' cleanup).

    The scorer used that string directly and the archive stored it via the scored workbook; an empty string was
    stored as ``invalid``. Cells the workbook may have changed are counted, not repaired: a non-string value (a
    number, or a formula's cached result), an ``_xHHHH_`` escape, and text of exactly 32,767 characters (possibly
    cut by xlsxwriter). The full-split reproduction is exact with these counts at 0; a new run with non-zero counts
    needs its own cross-check before its TableVQABench value is trusted.
    """

    out: list[str] = []
    d = diag if diag is not None else {}
    for status, raw in zip(rows["extraction_status"], rows["extraction_value_json"]):
        value = _decode(raw)
        if status != "resolved" or value is None:
            d["invalid_extraction_as_empty"] = d.get("invalid_extraction_as_empty", 0) + 1
            out.append("")
        elif isinstance(value, str):
            if _XLSX_CONTROL.search(value) or re.search(r"_x[0-9A-Fa-f]{4}_", value):
                d["xlsx_escape_in_prediction"] = d.get("xlsx_escape_in_prediction", 0) + 1
            if len(value) == XLSX_MAX_STRING:
                d["xlsx_max_length_prediction"] = d.get("xlsx_max_length_prediction", 0) + 1
            out.append(value)
        else:
            d["non_string_prediction"] = d.get("non_string_prediction", 0) + 1
            out.append(str(value))
    return out


# ------------------------------------------------------------------------------------------- pinned scorers
RowScorer = Callable[[list, pd.DataFrame, dict], np.ndarray]


def _quiet() -> contextlib.AbstractContextManager:
    return contextlib.redirect_stdout(io.StringIO())


def score_puzzlevqa(preds: list, meta: pd.DataFrame, diag: dict) -> np.ndarray:
    """PuzzleVQA_acc's per-row test: extract_answer(prediction) == letter of the answer among the options."""

    pz = _pinned("vlmeval.dataset.utils.puzzlevqa")
    out = np.empty(len(preds))
    with _quiet():
        for i, (pred, options, answer) in enumerate(zip(preds, meta["options"], meta["answer"])):
            letter = pz.to_choice_letter(ast.literal_eval(options), answer)
            out[i] = float(pz.extract_answer(pred).lower() == letter.lower())
    return out


def score_visualpuzzles(preds: list, meta: pd.DataFrame, diag: dict) -> np.ndarray:
    """VisulPuzzles_acc's per-row test: extract_answer(prediction) == answer letter (case-insensitive)."""

    vp = _pinned("vlmeval.dataset.utils.visualpuzzles")
    out = np.empty(len(preds))
    with _quiet():
        for i, (pred, answer) in enumerate(zip(preds, meta["answer"])):
            out[i] = float(vp.extract_answer(pred).lower() == answer.lower())
    return out


def score_countbenchqa(preds: list, meta: pd.DataFrame, diag: dict) -> np.ndarray:
    """CountBenchQA.evaluate's per-row rule, ``str(answer) in str(prediction)``.

    The rule sits inside an evaluate() that needs a workbook, so it is restated here; ``reproduce_archive`` also
    runs the pinned evaluate() on the rebuilt workbook of every slice to confirm the restatement.
    """

    return np.array([float(str(a) in str(p)) for p, a in zip(preds, meta["answer"])], dtype=float)


def score_chartqapro(preds: list, meta: pd.DataFrame, diag: dict) -> np.ndarray:
    """ChartQAPro.get_scores' per-row record through the pinned evaluate_predictions_chartqapro (one row at a time,
    whose 'Overall' is that row's relaxed correctness)."""

    cq = _pinned("vlmeval.dataset.utils.chartqapro")
    out = np.empty(len(preds))
    with _quiet():
        for i, (pred, answer, qtype, year) in enumerate(zip(preds, meta["answer"], meta["question_type"],
                                                            meta["year"])):
            record = {"Answer": ast.literal_eval(answer), "Question Type": qtype, "Year": ast.literal_eval(year),
                      "prediction": pred}
            out[i] = float(cq.evaluate_predictions_chartqapro([record])["Overall"])
    return out


def score_mathvision(preds: list, meta: pd.DataFrame, diag: dict) -> np.ndarray:
    """MATH_V_acc's per-row hit: post_check(row, prefetch=False) with the judged 'res'."""

    mathv = _pinned("vlmeval.dataset.utils.mathv")
    out = np.empty(len(preds))
    with _in_process_is_equal(mathv, diag), _quiet():
        for i, (res, answer, choices) in enumerate(zip(preds, meta["answer"], meta["choices"])):
            out[i] = float(bool(mathv.post_check({"answer": answer, "choices": choices, "res": res},
                                                 prefetch=False)))
    return out


def score_mathvista(preds: list, meta: pd.DataFrame, diag: dict) -> np.ndarray:
    """MathVista_acc's per-row hit: post_check(row, prefetch=False) with the judged 'res'."""

    mv = _pinned("vlmeval.dataset.utils.mathvista")
    cols = ["answer", "answer_option", "choices", "question_type", "answer_type"]
    out = np.empty(len(preds))
    with _quiet():
        for i, (res, *vals) in enumerate(zip(preds, *(meta[c] for c in cols))):
            line = dict(zip(cols, vals))
            line["res"] = res
            out[i] = float(bool(mv.post_check(line, prefetch=False)))
    return out


def tablevqa_components(preds: list, meta: pd.DataFrame, diag: dict) -> pd.DataFrame:
    """Per-row scores of the pinned TableVQABench split scorers.

    Returns ``primary`` (the scorer's 'accuracy' key: fintabnetqa relieved accuracy, otherwise correctness; a
    vtabfact row with both 'true' and 'false' scores None and counts as incorrect, as in the pinned counter) and
    ``exact`` (fintabnetqa exact_score; equal to ``primary`` for the other splits). The pinned average_scores of
    each split are recomputed from these and must agree.
    """

    tv = _pinned("vlmeval.dataset.utils.tablevqabench")
    split = meta["split"].to_numpy()
    if set(split) - set(TABLEVQA_SPLITS):
        raise MetricError(f"tablevqabench: unknown splits {sorted(set(split) - set(TABLEVQA_SPLITS))}")
    primary = np.zeros(len(preds))
    exact = np.zeros(len(preds))
    with _quiet():
        for name in TABLEVQA_SPLITS:
            idx = np.flatnonzero(split == name)
            if not len(idx):
                continue
            records = [{"prediction": preds[i], "answer": meta["answer"].iloc[i]} for i in idx]
            if name == "fintabnetqa":
                info = tv.evaluate_fintabnet(records, ["accuracy"])
            elif name == "vtabfact":
                info = tv.evaluate_tabfact(records, ["accuracy"])
            else:
                info = tv.evaluate_wtq(records, ["accuracy"])
            p = np.array([float(r["scores"]["accuracy"] or 0) for r in records])
            e = np.array([float(r["scores"].get("exact_score", r["scores"]["accuracy"]) or 0) for r in records])
            primary[idx], exact[idx] = p, e
            expected = [_tvqa_split_value(p.sum(), len(idx))]
            if name == "fintabnetqa":
                expected.append(_tvqa_split_value(e.sum(), len(idx)))
            if [float(v) for v in info["average_scores"]] != expected:
                raise MetricError(f"tablevqabench {name}: row scores do not recompute the pinned average_scores")
            if name == "vtabfact":
                diag["vtabfact_unparsed_scored_incorrect"] = diag.get("vtabfact_unparsed_scored_incorrect", 0) + int(
                    sum(r["scores"]["accuracy"] is None for r in records))
    if np.any(exact > primary):
        raise MetricError("tablevqabench: a fintabnetqa row is exact-correct but not relieved-correct")
    return pd.DataFrame({"primary": primary, "exact": exact})


def _tvqa_split_value(correct: float, rows: float) -> float:
    """One pinned TableVQABench average score, with the same arithmetic on Python floats."""

    return round((float(correct) + 1e-9) / (float(rows) + 1e-9), 8) * 100


ROW_VIEWS: dict[str, Callable[[pd.DataFrame, dict], list]] = {
    "chartqapro": official_route_view, "countbenchqa": official_route_view, "puzzlevqa": official_route_view,
    "visualpuzzles": official_route_view, "mathvision": judged_res_view, "mathvista": judged_res_view,
    "tablevqabench": scored_prediction_view,
}
ROW_SCORERS: dict[str, RowScorer] = {
    "chartqapro": score_chartqapro, "countbenchqa": score_countbenchqa, "puzzlevqa": score_puzzlevqa,
    "visualpuzzles": score_visualpuzzles, "mathvision": score_mathvision, "mathvista": score_mathvista,
}


# ------------------------------------------------------------------------------------------------- metrics
def _require_columns(rows: pd.DataFrame, cols: Sequence[str], who: str) -> None:
    missing = [c for c in cols if c not in rows.columns]
    if missing:
        raise MetricError(f"{who}: rows lack columns {missing}")


def _sum_by_unit(contrib: np.ndarray, unit: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if unit.isna().any():
        raise MetricError(f"{int(unit.isna().sum())} rows have no unit id")
    codes, uniques = pd.factorize(unit, sort=True)
    stats = np.zeros((len(uniques), contrib.shape[1]), dtype=float)
    np.add.at(stats, codes, contrib)
    return np.asarray(uniques), stats


class BenchmarkMetric:
    """One benchmark's official metric with additive unit statistics (shared interface, see statistics.py)."""

    stat_names: tuple[str, ...] = ()

    def __init__(self, spec: MetricSpec, meta: pd.DataFrame | None = None) -> None:
        self.spec = spec
        self.benchmark_id = spec.benchmark_id
        self.kind = spec.kind
        self.official_metric_id = f"{spec.benchmark_id}:primary"
        self.meta: pd.DataFrame | None = None
        self.last_diagnostics: dict[str, int] = {}
        if meta is not None:
            self.bind_meta(meta)

    # -------------------------------------------------------------- metadata
    def bind_meta(self, meta: pd.DataFrame) -> None:
        """Attach the licensed per-content metadata (see ``load_benchmark_meta``), keyed by source ordinal."""

        _require_columns(meta, ["source_ordinal", "source_index", *self.spec.meta_columns], self.benchmark_id)
        ordinals = meta["source_ordinal"].astype(int)
        if ordinals.duplicated().any():
            raise MetricError(f"{self.benchmark_id}: metadata has duplicate source ordinals")
        self.meta = meta.assign(source_ordinal=ordinals, source_index=meta["source_index"].astype(str)
                                ).set_index("source_ordinal", drop=False)

    def aligned_meta(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Metadata rows for ``rows`` (same index), matched on source ordinal and checked on index/record hash."""

        if self.meta is None:
            raise MetricError(f"{self.benchmark_id}: licensed metadata is not bound (load_benchmark_meta)")
        _require_columns(rows, ["source_ordinal", "source_index"], self.benchmark_id)
        ordinals = pd.to_numeric(rows["source_ordinal"], errors="raise").astype(int).to_numpy()
        missing = ~np.isin(ordinals, self.meta.index.to_numpy())
        if missing.any():
            raise MetricError(f"{self.benchmark_id}: {int(missing.sum())} rows have no metadata row")
        m = self.meta.loc[ordinals]
        if (m["source_index"].to_numpy() != rows["source_index"].astype(str).to_numpy()).any():
            raise MetricError(f"{self.benchmark_id}: source_index disagrees with the licensed metadata")
        # The ordinal is the only unique join key (TableVQABench repeats 237 indices), so every row hash the rows
        # carry is checked too: archive rows carry both, a local run's spool carries the row hash.
        for col in ("source_record_sha256", "source_row_sha256"):
            if col in rows.columns and col in m.columns:
                if (m[col].to_numpy() != rows[col].astype(str).to_numpy()).any():
                    raise MetricError(f"{self.benchmark_id}: {col} disagrees with the dataset manifest")
        m = m.reset_index(drop=True)
        m.index = rows.index
        return m

    # -------------------------------------------------------------- interface
    def check_rows(self, rows: pd.DataFrame) -> None:
        """Fail unless ``rows`` is one slice of this benchmark with each content row once.

        Callers normally guarantee this (``statistics.collect_unit_stats``, ``load_archive_slice``), but a repeated
        row or a row of another benchmark would change the official value silently (one duplicated BLINK row in
        three turns 33.33 into 50.0), so the metric checks the identity columns it is given.
        """

        if "benchmark_id" in rows.columns:
            other = sorted(set(rows["benchmark_id"].astype(str)) - {self.benchmark_id})
            if other:
                raise MetricError(f"{self.benchmark_id}: rows of other benchmarks {other[:3]}")
        for col in ("model_label", "seed"):
            if col in rows.columns and rows[col].astype(str).nunique() > 1:
                raise MetricError(f"{self.benchmark_id}: rows span several {col} values; score one slice at a time")
        for col in ("content_key", "source_ordinal"):
            if col in rows.columns:
                repeated = int(rows[col].astype(str).duplicated().sum())
                if repeated:
                    raise MetricError(f"{self.benchmark_id}: {repeated} repeated {col} values; each content row "
                                      "must appear once")

    def row_scores(self, rows: pd.DataFrame, meta: pd.DataFrame | None = None) -> pd.Series:
        raise NotImplementedError

    def row_contributions(self, rows: pd.DataFrame, scores: pd.Series) -> np.ndarray:
        """Per-row additive statistics; summing them over any row set gives that set's totals."""

        raise NotImplementedError

    def unit_stats(self, rows: pd.DataFrame, scores: pd.Series, unit: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        if not scores.index.equals(rows.index) or not unit.index.equals(rows.index):
            raise MetricError(f"{self.benchmark_id}: scores/unit are not aligned to rows")
        self.check_rows(rows)
        return _sum_by_unit(self.row_contributions(rows, scores), unit)

    def finalize(self, totals: np.ndarray) -> float:
        raise NotImplementedError

    def finalize_many(self, totals: np.ndarray) -> np.ndarray:
        totals = np.asarray(totals, dtype=float)
        return np.array([self.finalize(t) for t in totals], dtype=float)

    def campaign_value(self, totals: np.ndarray) -> float:
        """The value the protocol's own score campaign reports on exactly these rows.

        Used to check a run's ``scores.json`` (DESIGN.md section 5: an adapter must reproduce the campaign's aggregate
        on every new subset run). It equals ``finalize`` except where this study's subset metric deliberately
        departs from the pinned formula (WeMath, see ``WeMathStrictMetric``).
        """

        return self.finalize(totals)

    def official_score(self, rows: pd.DataFrame, scores: pd.Series) -> float:
        if not scores.index.equals(rows.index):
            raise MetricError(f"{self.benchmark_id}: scores are not aligned to rows")
        self.check_rows(rows)
        return self.finalize(self.row_contributions(rows, scores).sum(axis=0))

    def describe(self) -> dict[str, Any]:
        s = self.spec
        return {"benchmark_id": s.benchmark_id, "kind": self.kind, "official_metric_id": self.official_metric_id,
                "official_metric_key": s.official_metric_key, "route": s.route, "official_alias": s.official_alias,
                "units": "percent", "aggregation": s.aggregation, "unit_stats": list(self.stat_names),
                "row_score_type": s.row_score_type, "meta_columns": list(s.meta_columns),
                "official_strata": list(s.official_strata), "code_refs": list(s.code_refs), "notes": s.notes}


class PooledMeanMetric(BenchmarkMetric):
    """Official value = 100 x sum(row scores) / rows; unit stats [sum, rows]."""

    stat_names = ("score_sum", "rows")

    def row_contributions(self, rows: pd.DataFrame, scores: pd.Series) -> np.ndarray:
        values = scores.to_numpy(dtype=float)
        if np.isnan(values).any():
            raise MetricError(f"{self.benchmark_id}: null row scores")
        return np.column_stack([values, np.ones(len(values))])

    def finalize(self, totals: np.ndarray) -> float:
        t = np.asarray(totals, dtype=float)
        if t.shape != (2,):
            raise MetricError(f"{self.benchmark_id}: totals must have shape (2,)")
        if not t[1] > 0:
            raise MetricError(f"{self.benchmark_id}: no rows")
        return float(100.0 * t[0] / t[1])

    def finalize_many(self, totals: np.ndarray) -> np.ndarray:
        t = np.asarray(totals, dtype=float)
        if t.ndim != 2 or t.shape[1] != 2 or not (t[:, 1] > 0).all():
            raise MetricError(f"{self.benchmark_id}: totals must be (n, 2) with rows > 0")
        return 100.0 * t[:, 0] / t[:, 1]


class BinaryRowMetric(PooledMeanMetric):
    """``binary_row``: the archived 0/1 row scores are the official per-row outcomes."""

    def row_scores(self, rows: pd.DataFrame, meta: pd.DataFrame | None = None) -> pd.Series:
        self.check_rows(rows)
        return archived_binary_scores(rows, self.benchmark_id)


def archived_binary_scores(rows: pd.DataFrame, who: str) -> pd.Series:
    """Archived 0/1 row scores, failing closed on nulls, exclusions (semantics unknown) or non-binary values."""

    _require_columns(rows, ["archived_row_score"], who)
    if "official_excluded" in rows.columns and rows["official_excluded"].fillna(False).astype(bool).any():
        raise MetricError(f"{who}: rows marked official_excluded; their official treatment is not defined here")
    values = pd.to_numeric(rows["archived_row_score"], errors="coerce").astype(float)
    if values.isna().any():
        raise MetricError(f"{who}: {int(values.isna().sum())} rows have no archived row score")
    if not values.isin([0.0, 1.0]).all():
        raise MetricError(f"{who}: archived row scores are not binary")
    return values


class RederivedPooledMetric(PooledMeanMetric):
    """``aggregate_only_rederived`` with a pooled-mean official value; row scores from the pinned scorer."""

    def __init__(self, spec: MetricSpec, meta: pd.DataFrame | None = None, *,
                 view: Callable[[pd.DataFrame, dict], list] | None = None,
                 scorer: RowScorer | None = None) -> None:
        super().__init__(spec, meta)
        self.view = view or ROW_VIEWS[spec.benchmark_id]
        self.scorer = scorer or ROW_SCORERS[spec.benchmark_id]

    def row_scores(self, rows: pd.DataFrame, meta: pd.DataFrame | None = None) -> pd.Series:
        if meta is not None:
            self.bind_meta(meta)
        self.check_rows(rows)
        _require_columns(rows, ["extraction_status", "extraction_value_json"], self.benchmark_id)
        m = self.aligned_meta(rows)
        diag: dict[str, int] = {}
        values = np.asarray(self.scorer(self.view(rows, diag), m, diag), dtype=float)
        if values.shape != (len(rows),) or np.isnan(values).any() or (values < 0).any() or (values > 1).any():
            raise MetricError(f"{self.benchmark_id}: re-derived row scores are not in [0, 1]")
        self.last_diagnostics = diag
        return pd.Series(values, index=rows.index, name="row_score")


class SplitMacroMetric(BenchmarkMetric):
    """TableVQABench: mean of the five pinned split scores, each rounded like the pinned scorer.

    ``row_scores`` returns each row's mean component ((relieved + exact) / 2 for fintabnetqa, otherwise its single
    0/1 score). Exact correctness implies relieved correctness (checked), so the components decode back from that
    value in ``row_contributions``. Row diagnostics are unsupported: the official value is not a row mean.
    """

    stat_names = ("fintabnetqa_relieved", "fintabnetqa_exact", "fintabnetqa_rows", "vtabfact_correct",
                  "vtabfact_rows", "vwtq_correct", "vwtq_rows", "vwtq_syn_correct", "vwtq_syn_rows")

    def __init__(self, spec: MetricSpec, meta: pd.DataFrame | None = None, *,
                 components: Callable[[list, pd.DataFrame, dict], pd.DataFrame] | None = None) -> None:
        super().__init__(spec, meta)
        self.components = components or tablevqa_components

    def row_scores(self, rows: pd.DataFrame, meta: pd.DataFrame | None = None) -> pd.Series:
        if meta is not None:
            self.bind_meta(meta)
        self.check_rows(rows)
        _require_columns(rows, ["extraction_status", "extraction_value_json"], self.benchmark_id)
        m = self.aligned_meta(rows)
        diag: dict[str, int] = {}
        comp = self.components(scored_prediction_view(rows, diag), m, diag)
        fin = (m["split"] == "fintabnetqa").to_numpy()
        values = np.where(fin, (comp["primary"].to_numpy() + comp["exact"].to_numpy()) / 2.0,
                          comp["primary"].to_numpy())
        self.last_diagnostics = diag
        return pd.Series(values, index=rows.index, name="row_score")

    def row_contributions(self, rows: pd.DataFrame, scores: pd.Series) -> np.ndarray:
        split = self.aligned_meta(rows)["split"].to_numpy()
        s = scores.to_numpy(dtype=float)
        out = np.zeros((len(s), len(self.stat_names)))
        fin = split == "fintabnetqa"
        if not np.isin(s[fin], (0.0, 0.5, 1.0)).all() or not np.isin(s[~fin], (0.0, 1.0)).all():
            raise MetricError("tablevqabench: row scores are not decodable split components")
        out[fin, 0] = (s[fin] > 0).astype(float)          # relieved
        out[fin, 1] = (s[fin] == 1.0).astype(float)        # exact
        out[fin, 2] = 1.0
        for j, name in ((3, "vtabfact"), (5, "vwtq"), (7, "vwtq_syn")):
            sel = split == name
            out[sel, j] = s[sel]
            out[sel, j + 1] = 1.0
        unknown = ~np.isin(split, TABLEVQA_SPLITS)
        if unknown.any():
            raise MetricError("tablevqabench: rows with an unknown split")
        return out

    def finalize(self, totals: np.ndarray) -> float:
        t = [float(v) for v in np.asarray(totals, dtype=float)]
        if len(t) != len(self.stat_names):
            raise MetricError("tablevqabench: wrong totals length")
        fr, fe, fn, tc, tn, wc, wn, sc, sn = t
        missing = [n for n, rows in zip(TABLEVQA_SPLITS, (fn, tn, wn, sn)) if not rows > 0]
        if missing:
            raise MetricError(f"tablevqabench: split(s) {missing} absent; the pinned scorer is undefined there")
        values = [_tvqa_split_value(fr, fn), _tvqa_split_value(fe, fn), _tvqa_split_value(tc, tn),
                  _tvqa_split_value(wc, wn), _tvqa_split_value(sc, sn)]
        return float(sum(values) / len(values))


class WeMathStrictMetric(BenchmarkMetric):
    """WeMath 'Score (Strict)' from 0/1 question scores and problem groups (licensed ``ID``/``key``).

    Pinned formula (utils/wemath.py): with IG, RM_strict and IK counted over the groups present,
    ``(525 - 0.5*IG - RM - IK) / 525`` formatted ``'{:.2%}'``. The 525 is the number of groups in the full split, so
    on a subset the pinned value counts every absent group as fully mastered (on 105 random groups of T seed 42 it
    reads 90.10 where the groups present score 50.48), and under resampling it would move with the number of groups
    drawn rather than with the outcomes. This study's metric (``finalize``) therefore divides by the number of groups
    present, G: ``(G - 0.5*IG - RM - IK) / G`` = (CM + 0.5*IG) / G. That is a group-preserving adapter in the sense
    of handoff section 6.1, identical to the pinned value on the full split and on the same scale for every subset;
    adopting it for subsets needs the DESIGN.md amendment recorded in the metric registry (``subset_semantics``).

    Rounding: on exactly 525 groups (the full split, the pinned scorer's own regime) ``finalize`` returns the pinned
    value with its two-decimal display rounding, so the archived aggregates reproduce exactly. On any other group
    count it returns the unrounded value, so subsets and bootstrap replicates carry no 0.005 pp display
    quantisation. ``finalize_unrounded`` is always unrounded. ``campaign_value`` (= ``pinned_fixed525_score``) is
    what the protocol's score campaign reports on the same rows; a subset run's cross-check compares that, and
    reports both values.

    Unit statistics are group-outcome counts; a group must lie wholly inside one unit and be complete.
    """

    stat_names = ("groups", "complete_mastery", "inadequate_generalization", "rote_memorization",
                  "insufficient_knowledge")

    def row_scores(self, rows: pd.DataFrame, meta: pd.DataFrame | None = None) -> pd.Series:
        if meta is not None:
            self.bind_meta(meta)
        self.check_rows(rows)
        return archived_binary_scores(rows, self.benchmark_id)

    def _groups(self, rows: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
        m = self.aligned_meta(rows)
        df = pd.DataFrame({"ID": m["ID"].astype(str).to_numpy(), "key": m["key"].astype(str).to_numpy(),
                           "hit": scores.to_numpy(dtype=float), "pos": np.arange(len(rows))})
        step_type = df["key"].str.extract(r"^([23]steps)_")[0]
        if step_type.isna().any():
            raise MetricError(f"wemath: unknown keys {sorted(set(df.loc[step_type.isna(), 'key']))[:5]}")
        df["type"] = step_type
        return df

    def row_contributions(self, rows: pd.DataFrame, scores: pd.Series) -> np.ndarray:
        df = self._groups(rows, scores)
        out = np.zeros((len(df), len(self.stat_names)))
        for gid, g in df.groupby("ID", sort=False):
            types = set(g["type"])
            if len(types) != 1:
                raise MetricError(f"wemath: group {gid} mixes step types {sorted(types)}")
            expected = WEMATH_KEYS[types.pop()]
            if sorted(g["key"]) != sorted(expected):
                raise MetricError(f"wemath: group {gid} is incomplete or duplicated (keys {sorted(g['key'])}); a "
                                  "subset must contain whole problem groups")
            hit = dict(zip(g["key"], g["hit"] == 1.0))
            multi = hit[expected[-1]]
            steps = all(hit[k] for k in expected[:-1])
            pos = int(g.loc[g["key"] == expected[-1], "pos"].iloc[0])
            out[pos] = (1.0, float(steps and multi), float(steps and not multi), float(not steps and multi),
                        float(not steps and not multi))
        return out

    def unit_stats(self, rows: pd.DataFrame, scores: pd.Series, unit: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        if not unit.index.equals(rows.index):
            raise MetricError("wemath: unit is not aligned to rows")
        ids = self.aligned_meta(rows)["ID"].astype(str)
        spread = pd.DataFrame({"ID": ids.to_numpy(), "unit": unit.to_numpy()}).groupby("ID")["unit"].nunique()
        if (spread > 1).any():
            raise MetricError(f"wemath: problem group(s) {list(spread[spread > 1].index[:5])} span several units")
        return super().unit_stats(rows, scores, unit)

    def _parts(self, totals: np.ndarray) -> tuple[float, float, float, float, float]:
        t = [float(v) for v in np.asarray(totals, dtype=float)]
        if len(t) != 5:
            raise MetricError("wemath: wrong totals length")
        g, cm, ig, rm, ik = t
        if not g > 0:
            raise MetricError("wemath: no problem groups")
        if abs(cm + ig + rm + ik - g) > 1e-6 * max(1.0, g):
            raise MetricError("wemath: group outcome counts do not add up to the group count")
        return g, cm, ig, rm, ik

    def finalize_unrounded(self, totals: np.ndarray) -> float:
        g, _, ig, rm, ik = self._parts(totals)
        return 100.0 * (g - 0.5 * ig - rm - ik) / g

    def finalize(self, totals: np.ndarray) -> float:
        g, _, ig, rm, ik = self._parts(totals)
        if g == WEMATH_PINNED_GROUPS:
            return _pinned_percent((g - 0.5 * ig - rm - ik) / g)
        return 100.0 * (g - 0.5 * ig - rm - ik) / g

    def pinned_fixed525_score(self, totals: np.ndarray) -> float:
        """The pinned scorer's value on whatever groups are present (fixed denominator 525, display rounding)."""

        _, _, ig, rm, ik = self._parts(totals)
        n = WEMATH_PINNED_GROUPS
        return _pinned_percent((n - 0.5 * ig - rm - ik) / n)

    def campaign_value(self, totals: np.ndarray) -> float:
        return self.pinned_fixed525_score(totals)


def _pinned_percent(fraction: float) -> float:
    """The pinned WeMath display value: ``'{:.2%}'.format(fraction)`` read back as a percent number."""

    return float("{:.2%}".format(fraction)[:-1])


class UnsupportedMetric(BenchmarkMetric):
    """A benchmark whose adapter did not reproduce; it stays in full-suite reporting only."""

    def __init__(self, spec: MetricSpec, reason: str) -> None:
        super().__init__(spec)
        self.kind = "unsupported"
        self.reason = reason

    def _refuse(self, *_: Any, **__: Any) -> Any:
        raise MetricError(f"{self.benchmark_id} has no accepted adapter: {self.reason}")

    row_scores = row_contributions = unit_stats = finalize = finalize_many = campaign_value = official_score = \
        _refuse  # type: ignore[assignment]


def get_metric(benchmark_id: str, *, meta: pd.DataFrame | None = None) -> BenchmarkMetric:
    """The metric object for one benchmark (licensed metadata bound when given)."""

    if benchmark_id not in SPECS:
        raise MetricError(f"no metric spec for {benchmark_id!r}")
    spec = SPECS[benchmark_id]
    if spec.kind == "binary_row":
        return BinaryRowMetric(spec)
    if spec.kind == "grouped_row":
        return WeMathStrictMetric(spec, meta)
    if spec.aggregation == "split_macro":
        return SplitMacroMetric(spec, meta)
    return RederivedPooledMetric(spec, meta)


# ------------------------------------------------------------------------------------------ licensed metadata
@functools.lru_cache(maxsize=2)
def _load_manifest(path: str, mtime_ns: int) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dataset_manifest(cfg: StudyConfig) -> dict[str, Any]:
    path = cfg.dataset_manifest
    if not path.is_file():
        raise MetricError("prepared dataset manifest not found in the protocol clone")
    return _load_manifest(str(path), path.stat().st_mtime_ns)


def _metadata_tsv(cfg: StudyConfig, entry: Mapping[str, Any]) -> tuple[Path, str]:
    files = entry.get("metadata_files") or []
    if len(files) != 1:
        raise MetricError(f"{entry.get('key')}: expected one licensed metadata file, found {len(files)}")
    recorded = Path(files[0]["path"])
    path = recorded if recorded.is_file() else cfg.dataset_manifest.parent / recorded.name
    if not path.is_file():
        raise MetricError(f"{entry.get('key')}: licensed metadata file {recorded.name} is missing")
    return path, str(files[0]["sha256"])


def load_benchmark_meta(cfg: StudyConfig, benchmark_id: str, *, verify_sha256: bool = True) -> pd.DataFrame:
    """Licensed per-row metadata as the pinned scorer saw it.

    Read from the prepared TSV with the same pandas defaults as VLMEvalKit's ``load`` (only the needed columns),
    matched to the dataset manifest's row identities (ordinal, index, record and row hashes), and passed through
    the xlsx round trip of the pipeline's prediction workbooks (``xlsx_cell_text``).
    """

    spec = SPECS[benchmark_id]
    entry = dataset_manifest(cfg)["datasets"].get(benchmark_id)
    if entry is None:
        raise MetricError(f"{benchmark_id}: not in the prepared dataset manifest")
    path, sha = _metadata_tsv(cfg, entry)
    if verify_sha256 and sha256_file(path) != sha:
        raise MetricError(f"{benchmark_id}: licensed metadata sha256 differs from the dataset manifest")
    cols = ["index", *[c for c in spec.meta_columns if c != "index"]]
    table = pd.read_csv(path, sep="\t", usecols=cols)
    media = entry["row_media"]
    if len(table) != len(media):
        raise MetricError(f"{benchmark_id}: metadata has {len(table)} rows, manifest {len(media)}")
    for i, (idx, row) in enumerate(zip(table["index"], media)):
        if int(row["ordinal"]) != i or str(idx) != str(row["index"]):
            raise MetricError(f"{benchmark_id}: metadata row {i} does not match the manifest ordinal/index")
    out = pd.DataFrame({
        "source_ordinal": np.arange(len(table), dtype=np.int64),
        "source_index": [str(r["index"]) for r in media],
        "source_row_sha256": [str(r["source_row_hash"]) for r in media],
        "source_record_sha256": [str(r["source_record_sha256"]) for r in media],
    })
    for c in spec.meta_columns:
        out[c] = [xlsx_cell_text(v) for v in table[c].tolist()]
    out.attrs["metadata_sha256"] = sha
    return out


# ------------------------------------------------------------------------------------------------ archive reader
_SCORE_COLS = ["score_scope", "source_ordinal", "source_index", "source_row_sha256", "source_record_sha256",
               "scorer_id", "metric_id", "score_unit", "score_value", "excluded", "evaluated_rows"]
_EXTRACTION_COLS = ["source_ordinal", "source_index", "source_row_sha256", "source_record_sha256",
                    "extraction_status", "extraction_value_type", "extraction_value_json", "used_judge",
                    "judge_output"]
_KEY = ["source_ordinal", "source_index", "source_row_sha256", "source_record_sha256"]


def _archive_part(cfg: StudyConfig, config_name: str, model_id: str, seed: int, benchmark: str) -> Path:
    d = (cfg.archive_run_root / "data" / config_name / f"run={cfg.raw['archive']['run_id']}" / f"model={model_id}"
         / f"seed={seed}" / f"benchmark={benchmark}")
    parts = sorted(d.glob("part-*.parquet"))
    if len(parts) != 1:
        raise MetricError(f"{config_name}/{model_id}/{seed}/{benchmark}: expected one part, found {len(parts)}")
    return parts[0]


def _read_part(path: Path, columns: Sequence[str], verify: bool) -> pd.DataFrame:
    import pyarrow.parquet as pq

    if verify:
        digest = path.name[len("part-"):-len(".parquet")]
        if sha256_file(path) != digest:
            raise MetricError(f"archive part {path.name} does not match its content digest")
    return pq.ParquetFile(path).read(columns=list(columns)).to_pandas()


def _norm_key(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(source_ordinal=pd.to_numeric(df["source_ordinal"], errors="raise").astype(np.int64),
                     source_index=df["source_index"].astype(str))


def load_archive_slice(cfg: StudyConfig, role: str, seed: int, benchmark: str, *, extractions: bool = True,
                       verify: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One archived (model, seed, benchmark) slice as metric input rows plus its archived aggregate.

    Private reader (ingest.py owns the validated aligned table); reads only the score and extraction columns a
    metric needs and joins them one-to-one on the content identity.
    """

    model_id = cfg.role(role)["archived_model_id"]
    scores = _read_part(_archive_part(cfg, "scores", model_id, seed, benchmark), _SCORE_COLS, verify)
    agg = scores[scores["score_scope"] == "aggregate"]
    if len(agg) != 1:
        raise MetricError(f"{benchmark}/{role}/{seed}: {len(agg)} aggregate rows")
    a = agg.iloc[0]
    aggregate = {"official_value": float(a["score_value"]), "evaluated_rows": int(a["evaluated_rows"]),
                 "scorer_id": str(a["scorer_id"]), "metric_id": str(a["metric_id"]), "score_unit": str(a["score_unit"])}
    if aggregate["score_unit"] != "percent" or aggregate["metric_id"] != "primary":
        raise MetricError(f"{benchmark}/{role}/{seed}: aggregate is not the primary percent metric")
    row = scores[scores["score_scope"] == "row"]
    frame: pd.DataFrame | None = None
    if len(row):
        frame = _norm_key(row[_KEY + ["score_value", "excluded"]]).rename(
            columns={"score_value": "archived_row_score", "excluded": "official_excluded"})
    if extractions or frame is None:
        ext = _norm_key(_read_part(_archive_part(cfg, "extractions", model_id, seed, benchmark), _EXTRACTION_COLS,
                                   verify))
        if frame is None:
            frame = ext.assign(archived_row_score=np.nan, official_excluded=False)
        else:
            frame = frame.merge(ext, on=_KEY, how="outer", validate="one_to_one", indicator=True)
            if (frame["_merge"] != "both").any():
                raise MetricError(f"{benchmark}/{role}/{seed}: row scores and extractions do not join one-to-one")
            frame = frame.drop(columns="_merge")
    if frame[_KEY].duplicated().any():
        raise MetricError(f"{benchmark}/{role}/{seed}: duplicate content rows")
    if len(frame) != aggregate["evaluated_rows"]:
        raise MetricError(f"{benchmark}/{role}/{seed}: {len(frame)} rows, aggregate says {aggregate['evaluated_rows']}")
    frame = frame.sort_values("source_ordinal", kind="stable").reset_index(drop=True)
    frame.insert(0, "benchmark_id", benchmark)
    frame.insert(1, "model_label", ARCHIVE_LABELS[role])
    frame.insert(2, "seed", int(seed))
    return frame, aggregate


def _verify_archive_manifest(cfg: StudyConfig) -> str:
    path = cfg.archive_run_root / "metadata" / "manifest.json"
    if not path.is_file():
        raise MetricError("archive manifest not found under the work root's archive snapshot")
    digest = sha256_file(path)
    if digest != cfg.raw["archive"]["manifest_sha256"]:
        raise MetricError("archive manifest sha256 differs from the pinned receipt")
    return digest


# ------------------------------------------------------------------------------------- pinned evaluate() checks
def _pinned_countbenchqa_value(preds: list, meta: pd.DataFrame) -> float:
    """Run the pinned CountBenchQA.evaluate on a workbook rebuilt from the re-derivation inputs."""

    image_vqa = _pinned("vlmeval.dataset.image_vqa")
    with tempfile.TemporaryDirectory(prefix="trace_merge_cbqa_") as tmp:
        book = Path(tmp) / "CountBenchQA_predictions.xlsx"
        pd.DataFrame({"index": meta["source_index"].to_numpy(), "answer": meta["answer"].to_numpy(),
                      "prediction": preds}).to_excel(book, index=False)
        with _quiet():
            result = image_vqa.CountBenchQA.evaluate(None, str(book))
    return float(result["accuracy"])


# ------------------------------------------------------------------------------------------- reproduction
def _results_json_index(cfg: StudyConfig) -> dict[tuple[str, int, str], float]:
    data = json.loads(cfg.repo_path(cfg.raw["results"]).read_text(encoding="utf-8"))
    return {(str(r["model_id"]), int(r["seed"]), str(r["benchmark_id"])): float(r["score"])
            for r in data["scores"]["benchmark_scores"]}


def _archive_results_index(cfg: StudyConfig) -> dict[tuple[str, int, str], float]:
    path = cfg.archive_run_root / "metadata" / "results" / "benchmark_scores.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {(str(r["model_id"]), int(r["seed"]), str(r["benchmark_id"])): float(r["score"])
            for r in data["benchmark_scores"]}


def bind_metric_meta(cfg: StudyConfig, metric: BenchmarkMetric, cache: dict[str, pd.DataFrame] | None = None
                     ) -> BenchmarkMetric:
    if metric.spec.meta_columns and metric.meta is None:
        bench = metric.benchmark_id
        if cache is not None and bench in cache:
            metric.bind_meta(cache[bench])
        else:
            meta = load_benchmark_meta(cfg, bench)
            if cache is not None:
                cache[bench] = meta
            metric.bind_meta(meta)
    return metric


def reproduce_benchmark(cfg: StudyConfig, benchmark: str, *, verify: bool = True,
                        results: Mapping[tuple[str, int, str], float] | None = None,
                        archive_results: Mapping[tuple[str, int, str], float] | None = None) -> dict[str, Any]:
    """Recompute one benchmark's official score on the full split of all archived slices."""

    metric = bind_metric_meta(cfg, get_metric(benchmark))
    seeds = [int(s) for s in cfg.raw["archive"]["seeds"]]
    slices: list[dict[str, Any]] = []
    diag_total: dict[str, int] = {}
    t0 = time.time()
    for role in ROLES:
        model_id = cfg.role(role)["archived_model_id"]
        for seed in seeds:
            rows, agg = load_archive_slice(cfg, role, seed, benchmark, extractions=metric.kind != "binary_row"
                                           and metric.kind != "grouped_row", verify=verify)
            scores = metric.row_scores(rows)
            value = metric.official_score(rows, scores)
            entry: dict[str, Any] = {
                "model_label": ARCHIVE_LABELS[role], "model_id": model_id, "seed": seed, "rows": int(len(rows)),
                "archived_value": agg["official_value"], "reproduced_value": value,
                "abs_error_pp": abs(value - agg["official_value"]), "scorer_id": agg["scorer_id"],
                "row_score_mean_x100": float(100.0 * scores.mean()),
            }
            if metric.kind == "grouped_row":
                totals = metric.row_contributions(rows, scores).sum(axis=0)
                entry["group_totals"] = dict(zip(metric.stat_names, [int(v) for v in totals]))
                entry["unrounded_value"] = metric.finalize_unrounded(totals)
                entry["pinned_fixed525_value"] = metric.pinned_fixed525_score(totals)
                entry["campaign_value"] = metric.campaign_value(totals)
                entry["campaign_abs_error_pp"] = abs(entry["campaign_value"] - agg["official_value"])
            if isinstance(metric, SplitMacroMetric):
                totals = metric.row_contributions(rows, scores).sum(axis=0)
                entry["split_totals"] = dict(zip(metric.stat_names, [int(v) for v in totals]))
            if benchmark == "countbenchqa":
                pinned = _pinned_countbenchqa_value(official_route_view(rows), metric.aligned_meta(rows))
                entry["pinned_evaluate_on_rebuilt_workbook"] = pinned
                entry["pinned_evaluate_abs_error_pp"] = abs(pinned - value)
            for k, v in metric.last_diagnostics.items():
                diag_total[k] = diag_total.get(k, 0) + int(v)
            key = (model_id, seed, benchmark)
            if results is not None and key in results:
                entry["results_json_value"] = results[key]
                entry["results_json_abs_error_pp"] = abs(results[key] - agg["official_value"])
            if archive_results is not None and key in archive_results:
                entry["archive_results_abs_error_pp"] = abs(archive_results[key] - agg["official_value"])
            slices.append(entry)
            del rows, scores
            gc.collect()
    max_err = max(s["abs_error_pp"] for s in slices)
    extra_ok = all(s.get(k, 0.0) <= REPRODUCTION_TOLERANCE_PP for s in slices
                   for k in ("pinned_evaluate_abs_error_pp", "campaign_abs_error_pp"))
    results_errs = [s["results_json_abs_error_pp"] for s in slices if "results_json_abs_error_pp" in s]
    metadata_sha = metric.meta.attrs.get("metadata_sha256") if metric.meta is not None else None
    return {
        "benchmark_id": benchmark, "kind": metric.kind, "aggregation": metric.spec.aggregation,
        "official_metric_key": metric.spec.official_metric_key, "slices": slices, "max_abs_error_pp": max_err,
        "accepted": bool(max_err <= REPRODUCTION_TOLERANCE_PP and extra_ok and len(slices) == 3 * len(seeds)),
        "results_json_max_abs_error_pp": max(results_errs) if results_errs else None,
        "row_diagnostics": diag_total, "seconds": round(time.time() - t0, 1),
        "metadata_sha256": metadata_sha,
        "provenance": reproduction_provenance(cfg, benchmark, verify=verify, metadata_sha256=metadata_sha),
    }


# ------------------------------------------------------------------------------------------------- provenance
METRIC_CONFIG_KEYS = ("archive", "external_suite", "results", "protocol")


def metrics_code_sha256() -> str:
    """sha256 of this module: an accepted reproduction is evidence only for the adapter code that ran it."""

    return sha256_file(Path(__file__))


def metric_config_sha256(cfg: StudyConfig) -> str:
    """Digest of the config sections the adapters read (archive pin, suite, results, protocol pins).

    Narrower than the whole-file ``config_sha256`` on purpose: editing a later stage's settings does not change
    what was reproduced, so it must not invalidate the registry.
    """

    section = {k: cfg.raw.get(k) for k in METRIC_CONFIG_KEYS}
    if isinstance(section.get("protocol"), dict):
        section["protocol"] = {k: v for k, v in section["protocol"].items() if k != "clone"}
    return sha256_bytes(canonical_json(section).encode("utf-8"))


def reproduction_provenance(cfg: StudyConfig, benchmark: str, *, verify: bool,
                            metadata_sha256: str | None = None) -> dict[str, Any]:
    """What a reproduction result is evidence for; ``build_metrics`` refuses a registry whose provenance is stale."""

    return {"metrics_code_sha256": metrics_code_sha256(),
            "protocol_code_sha256": _code_digests(cfg, SPECS[benchmark].code_refs),
            "metric_config_sha256": metric_config_sha256(cfg), "config_sha256": cfg.sha256,
            "part_digests_verified": bool(verify), "metadata_sha256": metadata_sha256}


# ------------------------------------------------------------------------------- existing local rerun check
SPOOL_STAGES = ("extraction", "score")   # the generation stage (response texts) is not needed for scoring


def _spool_index(run_dir: Path, seed: int) -> dict[tuple[str, str], tuple[Path, str]]:
    """(benchmark, stage) -> (payload, sha256) of the protocol's private archive spool of a local run.

    Only descriptors are read here; payloads are parsed one benchmark at a time (``_spool_records``), which keeps
    the generation payloads (response texts, ~100 MB per full run) out of memory altogether.
    """

    spool = run_dir / "hf_archive"
    out: dict[tuple[str, str], tuple[Path, str]] = {}
    for path in sorted((spool / "ready").glob("*.ready.json")):
        desc = json.loads(path.read_text(encoding="utf-8"))
        ident = desc["identity"]
        if int(ident["seed"]) != int(seed) or str(ident["stage"]) not in SPOOL_STAGES:
            continue
        key = (str(ident["benchmark"]), str(ident["stage"]))
        if key in out:
            raise MetricError(f"spool holds two {key[1]} slices for {key[0]}")
        out[key] = (spool / desc["payload_path"], str(desc["payload_sha256"]))
    return out


def _spool_records(index: Mapping[tuple[str, str], tuple[Path, str]], benchmark: str
                   ) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """The extraction and score payloads of one benchmark, each checked against its descriptor digest."""

    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for stage in SPOOL_STAGES:
        key = (benchmark, stage)
        if key not in index:
            continue
        payload, digest = index[key]
        if sha256_file(payload) != digest:
            raise MetricError(f"spool payload {payload.name} does not match its descriptor digest")
        out[key] = [json.loads(line) for line in payload.read_text(encoding="utf-8").splitlines() if line.strip()]
    return out


def _spool_score_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "correct"}:
            return 1.0
        if low in {"false", "incorrect"}:
            return 0.0
    return float(value)


def spool_slice_rows(records: Mapping[tuple[str, str], list[dict[str, Any]]], benchmark: str) -> pd.DataFrame:
    """Metric input rows built from a local run's spool, in the same columns the archive reader produces."""

    ext = records.get((benchmark, "extraction"))
    if ext is None:
        raise MetricError(f"{benchmark}: no extraction spool slice")
    frame = pd.DataFrame({
        "benchmark_id": benchmark,
        "source_ordinal": [int(r["source_ordinal"]) for r in ext],
        "source_index": [str(r["source_index"]) for r in ext],
        "source_row_sha256": [str(r["source_row_hash"]) for r in ext],
        "extraction_status": [str(r["normalized_extraction"]["status"]).lower() for r in ext],
        "extraction_value_json": [canonical_json(r["normalized_extraction"].get("value")) for r in ext],
        "judge_output": [None if r.get("judge_response") in (None, "") else str(r["judge_response"]) for r in ext],
    })
    score = [r for r in records.get((benchmark, "score"), []) if str(r.get("source_index")) != "__aggregate__"]
    by_key = {(int(r["source_ordinal"]), str(r["source_index"])): r for r in score}
    if len(by_key) != len(score):
        raise MetricError(f"{benchmark}: the score spool repeats content rows")
    if set(by_key) - set(zip(frame["source_ordinal"], frame["source_index"])):
        raise MetricError(f"{benchmark}: the score spool has rows the extraction spool lacks")
    values: list[float | None] = []
    excluded: list[bool] = []
    for o, i in zip(frame["source_ordinal"], frame["source_index"]):
        r = by_key.get((o, i))
        values.append(None if r is None else _spool_score_value(r.get("score")))
        excluded.append(False if r is None else str(r.get("excluded", False)).lower() == "true")
    frame["archived_row_score"] = pd.array(values, dtype="Float64").astype(float)
    frame["official_excluded"] = excluded
    return frame.sort_values("source_ordinal", kind="stable").reset_index(drop=True)


def _local_official(run_dir: Path, seed: int, benchmark: str) -> tuple[float, Path]:
    found = sorted((run_dir / "scoring" / f"seed_{seed}" / "benchmark" / benchmark).glob("*/*/scores.json"))
    if len(found) != 1:
        raise MetricError(f"{benchmark}: expected one local scores.json, found {len(found)}")
    data = json.loads(found[0].read_text(encoding="utf-8"))
    value = data.get("score")
    if value is None:
        value = data.get("accuracy")
    if value is None:
        raise MetricError(f"{benchmark}: local scores.json has no score")
    return float(value), found[0].parent


def _runs_dir(out_dir: Path, benchmark: str) -> Path:
    """scoring/seed_S/runs/<key>/<served>/<run>: where direct routes keep their judged/scored tables."""

    return out_dir.parents[3] / "runs" / benchmark / out_dir.parent.name / out_dir.name


def _xlsx_rowwise(benchmark: str, out_dir: Path, metric: BenchmarkMetric) -> pd.DataFrame | None:
    """Per-row official outcomes computed by the pinned scorer functions on the local run's own scorer inputs.

    Independent of the archive-format reconstruction: the cells come from the workbooks the pinned scorer read
    (pandas defaults, as the scorer loaded them) rather than from the spool. Returns source_ordinal + score.
    """

    diag: dict[str, int] = {}
    runs_dir = _runs_dir(out_dir, benchmark)
    if benchmark in ("mathvision", "mathvista"):
        books = sorted(runs_dir.glob("*_judged_qwen3_32b.xlsx"))
        if len(books) != 1:
            return None
        data = pd.read_excel(books[0])
        meta = data.copy()
        preds = data["res"].tolist()
    elif benchmark == "tablevqabench":
        books = sorted(runs_dir.glob("*_trace_final_answer_scored.xlsx"))
        if len(books) != 1:
            return None
        data = pd.read_excel(books[0])
        parsed = [ast.literal_eval(s) for s in data["scores"]]
        prim = np.array([float(p["accuracy"] or 0) for p in parsed])
        ex = np.array([float(p.get("exact_score", p["accuracy"]) or 0) for p in parsed])
        fin = (data["split"] == "fintabnetqa").to_numpy()
        return pd.DataFrame({"source_ordinal": data["source_ordinal"].astype(int),
                             "score": np.where(fin, (prim + ex) / 2.0, prim)})
    elif benchmark == "wemath":
        books = sorted(out_dir.glob("*_qwen3-32b-judge.xlsx"))
        if len(books) != 1:
            return None
        data = pd.read_excel(books[0])
        return pd.DataFrame({"source_ordinal": data["source_ordinal"].astype(int),
                             "score": data["hit"].astype(float)})
    elif benchmark in ("puzzlevqa", "visualpuzzles", "countbenchqa", "chartqapro"):
        books = [p for p in sorted(out_dir.glob("*_predictions.xlsx"))]
        if len(books) != 1:
            return None
        data = pd.read_excel(books[0])
        meta = data.copy()
        preds = data["prediction"].tolist()
    else:
        return None
    if "source_ordinal" not in data.columns:
        return None
    scores = ROW_SCORERS[benchmark](preds, meta.reset_index(drop=True), diag)
    return pd.DataFrame({"source_ordinal": data["source_ordinal"].astype(int), "score": scores})


_STRATA_TABLES = {   # pinned per-stratum output tables of a local run: (directory, glob, key column, meta column(s))
    "puzzlevqa": ("benchmark", "*_acc.csv", "category", ("category",)),
    "visualpuzzles": ("benchmark", "*_acc.csv", "group", ("category", "difficulty")),
    "mathvision": ("runs", "*_judged_qwen3_32b_score.csv", "Subject", ("category",)),
    "mathvista": ("runs", "*_judged_qwen3_32b_score.csv", "Task&Skill", ("task",)),
    "chartqapro": ("benchmark", "*_acc.csv", "0", ("question_type",)),
}


def _strata_check(benchmark: str, out_dir: Path, scores: pd.Series, meta: pd.DataFrame) -> dict[str, Any] | None:
    """Compare per-stratum hit counts (or ChartQAPro per-type means) with the pinned scorer's own stratum table."""

    if benchmark not in _STRATA_TABLES:
        return None
    where, pattern, key, cols = _STRATA_TABLES[benchmark]
    base = out_dir if where == "benchmark" else _runs_dir(out_dir, benchmark)
    tables = sorted(base.glob(pattern))
    if len(tables) != 1:
        return None
    ref = pd.read_csv(tables[0])
    ref[key] = ref[key].astype(str)
    s = scores.reset_index(drop=True)
    mine: dict[str, tuple[float, float]] = {"overall": (float(len(s)), float(s.sum()))}
    for c in cols:
        for value, idx in meta.reset_index(drop=True).groupby(c).groups.items():
            mine[str(value)] = (float(len(idx)), float(s.iloc[idx].sum()))
    compared = mismatched = 0
    for _, r in ref.iterrows():
        name = r[key]
        k = "overall" if name.lower() == "overall" else name
        if k not in mine:
            continue
        compared += 1
        tot, hit = mine[k]
        if benchmark == "chartqapro":
            ok = abs(hit / tot - float(r["1"])) <= 1e-12
        else:
            ok = tot == float(r["tot"]) and hit == float(r["hit"])
        mismatched += int(not ok)
    return {"strata_compared": compared, "strata_mismatches": mismatched, "table": tables[0].name}


def crosscheck_local_rerun(cfg: StudyConfig, run_dir: Path, *, label: str, seed: int = 42,
                           benchmarks: Sequence[str] | None = None) -> dict[str, Any]:
    """Apply every adapter to an existing local full-suite rerun and compare with that run's own scorer outputs.

    Evidence independent of the archive: other generations, the same pinned scorers. For each benchmark the value
    the run's own score campaign would report on the run's spool rows (``campaign_value``) must equal the run's
    scores.json, and for the re-derived and grouped benchmarks the adapter's row scores must equal the per-row
    outcomes the pinned functions give on the run's own scorer workbooks. The same check applies to a new subset
    run with this layout: there WeMath's campaign value (fixed 525 groups) and the study's value (groups present)
    differ by design, so both are reported and only the campaign value is compared.
    """

    index = _spool_index(run_dir, seed)
    out: dict[str, Any] = {}
    meta_cache: dict[str, pd.DataFrame] = {}
    for bench in benchmarks or list(SPECS):
        item: dict[str, Any] = {}
        try:
            metric = bind_metric_meta(cfg, get_metric(bench), meta_cache)
            rows = spool_slice_rows(_spool_records(index, bench), bench)
            scores = metric.row_scores(rows)
            value = metric.official_score(rows, scores)
            campaign = metric.campaign_value(metric.row_contributions(rows, scores).sum(axis=0))
            local, out_dir = _local_official(run_dir, seed, bench)
            item.update({"rows": int(len(rows)), "local_official_value": local, "adapter_value": value,
                         "campaign_value": campaign, "abs_error_pp": abs(campaign - local),
                         "row_diagnostics": dict(metric.last_diagnostics)})
            if metric.kind != "binary_row":
                ref = _xlsx_rowwise(bench, out_dir, metric)
                if ref is not None:
                    mine = pd.DataFrame({"source_ordinal": rows["source_ordinal"].to_numpy(),
                                         "mine": scores.to_numpy()})
                    joined = mine.merge(ref, on="source_ordinal", how="outer", validate="one_to_one")
                    item["rowwise_compared"] = int(joined[["mine", "score"]].notna().all(axis=1).sum())
                    item["rowwise_mismatches"] = int((~np.isclose(joined["mine"], joined["score"], atol=1e-12,
                                                                  equal_nan=False)).sum())
                    if item["rowwise_mismatches"]:
                        bad = joined[~np.isclose(joined["mine"], joined["score"], atol=1e-12)]
                        item["rowwise_mismatch_ordinals"] = [int(v) for v in bad["source_ordinal"].head(10)]
                strata = _strata_check(bench, out_dir, scores, metric.aligned_meta(rows))
                if strata is not None:
                    item.update(strata)
            item["ok"] = bool(item["abs_error_pp"] <= REPRODUCTION_TOLERANCE_PP
                              and item.get("rowwise_mismatches", 0) == 0 and item.get("strata_mismatches", 0) == 0)
        except MetricError as error:
            item.update({"ok": False, "error": str(error)})
        out[bench] = item
        gc.collect()
    return {"label": label, "seed": seed, "run_name": run_dir.name, "benchmarks": out,
            "all_ok": all(v.get("ok") for v in out.values())}


def local_meta_check(cfg: StudyConfig, run_dir: Path, seed: int = 42,
                     benchmarks: Sequence[str] | None = None) -> dict[str, Any]:
    """Compare the TSV-derived metadata with the local run's prediction workbooks (what the scorers read)."""

    out: dict[str, Any] = {}
    for bench in benchmarks or [b for b, s in SPECS.items() if s.meta_columns]:
        spec = SPECS[bench]
        books = sorted((run_dir / "campaign" / f"seed_{seed}" / "runs" / bench).glob("*/*/*_predictions.xlsx"))
        if len(books) != 1:
            out[bench] = {"ok": False, "error": f"{len(books)} prediction workbooks"}
            continue
        meta = load_benchmark_meta(cfg, bench)
        book = pd.read_excel(books[0], usecols=lambda c: c in {"source_ordinal", *spec.meta_columns})
        book = book.sort_values("source_ordinal").reset_index(drop=True)
        mism: dict[str, int] = {}
        for c in spec.meta_columns:
            a, b = meta[c].tolist(), book[c].tolist()
            mism[c] = sum(not ((x == y and type(x) is type(y)) or (_is_null(x) and _is_null(y)))
                          for x, y in zip(a, b))
        out[bench] = {"rows": len(book), "mismatches": mism, "ok": len(book) == len(meta) and not any(mism.values())}
    return out


# ------------------------------------------------------------------------------------------------- registry
def _code_digests(cfg: StudyConfig, refs: Sequence[str]) -> dict[str, str | None]:
    clone = cfg.protocol_clone
    out: dict[str, str | None] = {}
    for ref in refs:
        rel = ref.split("::")[0]
        if rel in out:
            continue
        path = clone / rel if clone is not None else None
        out[rel] = sha256_file(path) if path is not None and path.is_file() else None
    return out


def build_metric_registry(cfg: StudyConfig, reproduction: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The JSON metric registry (handoff 4.3): one entry per suite benchmark, plus reproduction status."""

    suite = {b["key"]: b for b in cfg.suite()["benchmarks"]}
    if set(suite) != set(SPECS):
        raise MetricError(f"suite and metric specs disagree: {sorted(set(suite) ^ set(SPECS))}")
    repro = (reproduction or {}).get("benchmarks", {})
    benches: dict[str, Any] = {}
    for key in suite:
        metric = get_metric(key)
        entry = metric.describe()
        entry["suite"] = {k: suite[key].get(k) for k in ("display", "category", "rows", "score_contract",
                                                          "answer_contract")}
        entry["scorer_id"] = f"{key}-primary-v1"
        entry["code_sha256"] = _code_digests(cfg, metric.spec.code_refs)
        entry["row_diagnostics"] = {
            "supported": metric.spec.row_score_type in ("binary",),
            "row_score_type": metric.spec.row_score_type,
            "basis": {"binary_row": "archived official 0/1 row scores",
                      "grouped_row": "archived 0/1 question scores; the official unit is the problem group",
                      "aggregate_only_rederived": "re-derived from archived extractions with the pinned scorer",
                      }[metric.kind],
        }
        entry["group_columns"] = list(metric.spec.meta_columns) if metric.kind == "grouped_row" else []
        entry["strata_columns"] = ["split"] if metric.spec.aggregation == "split_macro" else []
        entry["subset_semantics"] = _subset_semantics(metric.spec)
        r = repro.get(key)
        if r is None:
            entry["reproduction"] = {"status": "not_run"}
        else:
            provenance = r.get("provenance")
            entry["reproduction"] = {"status": _reproduction_status(cfg, key, r, provenance),
                                     "max_abs_error_pp": r.get("max_abs_error_pp"), "slices": len(r.get("slices", [])),
                                     "tolerance_pp": REPRODUCTION_TOLERANCE_PP,
                                     "results_json_max_abs_error_pp": r.get("results_json_max_abs_error_pp"),
                                     "provenance": provenance}
        benches[key] = entry
    proto = cfg.raw["protocol"]
    return {
        "schema_version": REGISTRY_SCHEMA, "study_id": cfg.study_id, "config_sha256": cfg.sha256,
        "protocol": {"code_revision": proto["code_revision"], "vlmevalkit_revision": proto["vlmevalkit_revision"],
                     "judge": proto["judge"]},
        "kinds": {k: sorted(b for b, e in benches.items() if e["kind"] == k) for k in KINDS},
        "accepted_benchmarks": sorted(b for b, e in benches.items() if e["reproduction"]["status"] == "accepted"),
        "benchmarks": benches,
    }


def _reproduction_status(cfg: StudyConfig, benchmark: str, result: Mapping[str, Any],
                         provenance: Mapping[str, Any] | None) -> str:
    """accepted | failed | unverified (archive part digests not checked, or no provenance) | stale.

    Only "accepted" lets ``build_metrics`` use the adapter. "stale" means the adapter code, the pinned scorer files
    or the metric-relevant config changed after the reproduction ran, so its evidence no longer applies.
    """

    if not result.get("accepted"):
        return "failed"
    if not provenance or not provenance.get("part_digests_verified"):
        return "unverified"
    return "stale" if _provenance_drift(cfg, benchmark, provenance) else "accepted"


def _provenance_drift(cfg: StudyConfig, benchmark: str, provenance: Mapping[str, Any]) -> list[str]:
    """Which recorded digests differ from the current code, pinned scorer files and metric config."""

    drift = []
    if provenance.get("metrics_code_sha256") != metrics_code_sha256():
        drift.append("metrics.py")
    if provenance.get("protocol_code_sha256") != _code_digests(cfg, SPECS[benchmark].code_refs):
        drift.append("pinned scorer files")
    if provenance.get("metric_config_sha256") != metric_config_sha256(cfg):
        drift.append("metric config sections")
    return drift


def _subset_semantics(spec: MetricSpec) -> str:
    if spec.aggregation == "row_mean":
        text = "pooled row mean of the subset's rows (same formula as the full split)"
        if spec.official_strata:
            text += (f"; the pinned function also divides by every value of {list(spec.official_strata)} and raises "
                     "on a subset lacking one, which does not change the pooled primary value")
        return text
    if spec.aggregation == "split_macro":
        return ("mean of the five split scores computed on the subset's rows of each split; undefined (error) when a "
                "split is absent, as in the pinned scorer")
    return ("(G - 0.5 IG - RM - IK) / G x 100 over the G complete problem groups present, unrounded unless G = 525 "
            "(full split: the pinned value with its two-decimal rounding). Departs from the pinned scorer on subsets, "
            "whose fixed denominator 525 scores every absent group as mastered; needs a DESIGN.md amendment. A subset "
            "run's own score campaign reports the fixed-525 value (campaign_value / pinned_fixed525_score), which is "
            "what its cross-check compares")


def build_metrics(cfg: StudyConfig, benchmarks: Sequence[str] | None = None, *, require_accepted: bool = True
                  ) -> dict[str, BenchmarkMetric]:
    """Metric objects with licensed metadata bound (DESIGN.md section 5).

    An adapter whose reproduction failed comes back ``unsupported`` (it stays in full-suite reporting, out of subset
    macros). Anything else short of a current acceptance is a process error and raises, so a benchmark never drops
    out of a macro silently: no reproduction, part digests not verified, or provenance that no longer matches the
    current metrics.py, pinned scorer files, metric config sections or licensed metadata (rerun ``reproduce``).
    """

    registry: dict[str, Any] | None = None
    path = cfg.work("manifests", "metric_registry.json") if cfg.work_root is not None else None
    if require_accepted:
        if path is None or not path.is_file():
            raise MetricError("manifests/metric_registry.json is missing; run `python -m trace_merge.metrics "
                              "reproduce` first")
        registry = json.loads(path.read_text(encoding="utf-8"))
        if registry.get("schema_version") != REGISTRY_SCHEMA:
            raise MetricError(f"metric registry schema {registry.get('schema_version')!r} is not {REGISTRY_SCHEMA}")
    out: dict[str, BenchmarkMetric] = {}
    cache: dict[str, pd.DataFrame] = {}
    for bench in benchmarks or list(SPECS):
        provenance: Mapping[str, Any] = {}
        if registry is not None:
            repro = registry["benchmarks"].get(bench, {}).get("reproduction", {})
            status = repro.get("status")
            if status == "failed":
                out[bench] = UnsupportedMetric(SPECS[bench], "the adapter did not reproduce the archived aggregates")
                continue
            if status != "accepted":
                raise MetricError(f"{bench}: reproduction status {status!r} in the metric registry; run `python -m "
                                  "trace_merge.metrics reproduce` (with part digests verified)")
            provenance = repro.get("provenance") or {}
            if any(v is None for v in (provenance.get("protocol_code_sha256") or {None: None}).values()):
                raise MetricError(f"{bench}: the registry's pinned scorer digests are incomplete")
            if cfg.protocol_clone is None:
                raise MetricError("checking the registry's pinned scorer digests needs the protocol clone "
                                  "(--protocol-clone or TRACE_MERGE_PROTOCOL_CLONE)")
            drift = _provenance_drift(cfg, bench, provenance)
            if drift:
                raise MetricError(f"{bench}: the registry is stale ({', '.join(drift)} changed since the "
                                  "reproduction); rerun `python -m trace_merge.metrics reproduce`")
        metric = bind_metric_meta(cfg, get_metric(bench), cache)
        if registry is not None and metric.meta is not None:
            if metric.meta.attrs.get("metadata_sha256") != provenance.get("metadata_sha256"):
                raise MetricError(f"{bench}: the licensed metadata differs from the one the reproduction used")
        out[bench] = metric
    return out


# ------------------------------------------------------------------------------------------------------ CLI
def _summary_table(benches: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"benchmark_id": b, "kind": r["kind"], "aggregation": r["aggregation"],
             "max_abs_error_pp": r.get("max_abs_error_pp"), "accepted": r.get("accepted", False),
             "error": r.get("error")} for b, r in benches.items()]


def reproduce_archive(cfg: StudyConfig, *, benchmarks: Sequence[str] | None = None, verify: bool = True,
                      local_reruns: Mapping[str, Path] | None = None, seed_local: int = 42,
                      log: Callable[[str], None] = print) -> dict[str, Any]:
    """Reproduce every benchmark's archived aggregates and cross-check against existing local reruns."""

    manifest_sha = _verify_archive_manifest(cfg)
    results = _results_json_index(cfg)
    archive_results = _archive_results_index(cfg)
    benches: dict[str, Any] = {}
    for bench in benchmarks or list(SPECS):
        try:
            r = reproduce_benchmark(cfg, bench, verify=verify, results=results, archive_results=archive_results)
        except MetricError as error:
            r = {"benchmark_id": bench, "kind": SPECS[bench].kind, "aggregation": SPECS[bench].aggregation,
                 "accepted": False, "error": str(error), "slices": []}
        benches[bench] = r
        log(f"[metrics] {bench:20s} {r['kind']:25s} max|err|={r.get('max_abs_error_pp')} "
            f"accepted={r['accepted']} ({r.get('seconds', 0)}s)")
        gc.collect()
    local: dict[str, Any] = {}
    for label, run_dir in (local_reruns or {}).items():
        log(f"[metrics] cross-checking existing local rerun {label}")
        local[label] = crosscheck_local_rerun(cfg, run_dir, label=label, seed=seed_local, benchmarks=benchmarks)
        local[label]["metadata_vs_prediction_workbooks"] = local_meta_check(cfg, run_dir, seed_local)
    return {
        "schema_version": REPRODUCTION_SCHEMA, "study_id": cfg.study_id, "config_sha256": cfg.sha256,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tolerance_pp": REPRODUCTION_TOLERANCE_PP,
        "archive": {k: cfg.raw["archive"][k] for k in ("repository_id", "revision", "run_id", "manifest_sha256")},
        "archive_manifest_verified": manifest_sha == cfg.raw["archive"]["manifest_sha256"],
        "part_digests_verified": bool(verify), "metrics_code_sha256": metrics_code_sha256(),
        "metric_config_sha256": metric_config_sha256(cfg),
        "benchmarks": benches, "existing_local_reruns": local,
        "summary": {"accepted": sorted(b for b, r in benches.items() if r.get("accepted")),
                    "not_accepted": sorted(b for b, r in benches.items() if not r.get("accepted")),
                    "table": _summary_table(benches)},
    }


def _parse_local(values: Sequence[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for v in values:
        if "=" not in v:
            raise ConfigError(f"--local-rerun expects LABEL=DIR, got {v!r}")
        label, path = v.split("=", 1)
        out[label] = Path(path)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m trace_merge.metrics", description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["reproduce", "registry"])
    ap.add_argument("--config")
    ap.add_argument("--work-root")
    ap.add_argument("--protocol-clone")
    ap.add_argument("--benchmarks", nargs="*")
    ap.add_argument("--local-rerun", action="append", default=[],
                    help="LABEL=DIR of an existing local full-suite rerun (e.g. B=<runs>/qwen2.5-vl-7b-instruct/full)")
    ap.add_argument("--local-seed", type=int, default=42)
    ap.add_argument("--no-verify-parts", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, work_root=args.work_root, protocol_clone=args.protocol_clone)
    ensure_protocol_imports(cfg.protocol_clone)
    audit = cfg.work("audit", "metric_reproduction.json")
    if args.command == "reproduce":
        report = reproduce_archive(cfg, benchmarks=args.benchmarks, verify=not args.no_verify_parts,
                                   local_reruns=_parse_local(args.local_rerun), seed_local=args.local_seed)
        if args.benchmarks and audit.is_file():   # partial run: merge into the existing audit
            old = json.loads(audit.read_text(encoding="utf-8"))
            old["benchmarks"].update(report["benchmarks"])
            old["existing_local_reruns"].update(report["existing_local_reruns"])
            benches = old["benchmarks"]
            old["summary"] = {"accepted": sorted(b for b, r in benches.items() if r.get("accepted")),
                              "not_accepted": sorted(b for b, r in benches.items() if not r.get("accepted")),
                              "table": _summary_table(benches)}
            for k in ("generated_at", "part_digests_verified", "metrics_code_sha256", "metric_config_sha256"):
                old[k] = report[k]   # this run's values; each benchmark's own provenance says what it ran with
            report = old
        write_json(audit, report)
        print(f"[metrics] wrote audit/{audit.name}")
    reproduction = json.loads(audit.read_text(encoding="utf-8")) if audit.is_file() else None
    registry = build_metric_registry(cfg, reproduction)
    write_json(cfg.work("manifests", "metric_registry.json"), registry)
    print(f"[metrics] wrote manifests/metric_registry.json ({len(registry['accepted_benchmarks'])} accepted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
