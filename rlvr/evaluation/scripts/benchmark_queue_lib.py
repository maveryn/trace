#!/usr/bin/env python3
"""Shared, trace_eval_v1-only benchmark and queue helpers."""

from __future__ import annotations

import json
import math
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VLMEVAL_ROOT = REPO_ROOT / "external" / "VLMEvalKit"
VLMEVAL_ROOT = Path(os.environ.get("VLMEVAL_ROOT", str(DEFAULT_VLMEVAL_ROOT)))
DEFAULT_RUN_ROOT = EVALUATION_ROOT / ".work" / "runs"
DEFAULT_BENCHMARK_ROOT = EVALUATION_ROOT / ".work" / "benchmark"
DEFAULT_QUEUE_ROOT = DEFAULT_BENCHMARK_ROOT / "queues"
BENCHMARK_RUN_SETS = ("trace_eval_v1",)

MME_REASONING_TSV_URL = (
    "https://huggingface.co/datasets/InternScience/MME-Reasoning/resolve/main/"
    "MME_Reasoning.tsv"
)
MME_REASONING_MD5 = "b243f44778782d3821523689f6b40a1e"
REALWORLDQA_TSV_URL = "http://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv"
REALWORLDQA_MD5 = "4de008f55dc4fd008ca9e15321dc44b7"


def parse_lettered_option_blob(value: Any) -> dict[str, str]:
    """Parse compact ``A. ... B. ...`` option metadata deterministically."""

    text = str(value or "").strip()
    if not text:
        return {}

    markers_by_start: dict[int, tuple[int, str]] = {}
    for marker in re.finditer(r"(?:^|\s)([A-K])[\.\):]\s+", text):
        markers_by_start[marker.start(1)] = (marker.end(), marker.group(1))
    for marker in re.finditer(r"(?m)^\s*([A-K])\s+(?=\S)", text):
        markers_by_start.setdefault(marker.start(1), (marker.end(), marker.group(1)))

    markers = sorted(
        (start, end, letter) for start, (end, letter) in markers_by_start.items()
    )
    options: dict[str, str] = {}
    for pos, (_, value_start, letter) in enumerate(markers):
        value_end = markers[pos + 1][0] if pos + 1 < len(markers) else len(text)
        option = re.sub(r"\s+", " ", text[value_start:value_end]).strip()
        if option:
            options.setdefault(letter, option)
    return options


try:
    from scripts.trace_eval_suite import load_trace_eval_suite
except ModuleNotFoundError:  # Supports direct ``python scripts/...`` imports.
    from trace_eval_suite import load_trace_eval_suite


_TRACE_EVAL_V1_SUITE = load_trace_eval_suite()
TRACE_EVAL_V1_BENCHMARK_CATEGORIES: dict[str, tuple[str, ...]] = dict(
    _TRACE_EVAL_V1_SUITE.categories
)
TRACE_EVAL_V1_BENCHMARKS = _TRACE_EVAL_V1_SUITE.benchmark_keys


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    display: str
    alias: str
    run_name: str
    eval_mode: str = "auto"
    split: str | None = None


# Filesystem run names are retained because they are part of the canonical
# campaign layout. Every key and alias is checked against suite.v1.json below.
_SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        "chartqapro", "ChartQAPro", "ChartQAPro_CoT", "vlmevalkit_faithful_cot"
    ),
    BenchmarkSpec(
        "charxivreason",
        "CharXivReason",
        "CharXiv_reasoning_val",
        "vlmevalkit_defaults_qwen32b_judge",
    ),
    BenchmarkSpec(
        "tablevqabench", "TableVQABench", "TableVQABench", "vlmevalkit_defaults"
    ),
    BenchmarkSpec(
        "evochart",
        "EvoChart",
        "EvoChart",
        "vlmevalkit_defaults_qwen32b_judge",
        "evochart_local_judge",
    ),
    BenchmarkSpec(
        "mathvision", "MathVision", "MathVision", "vlmevalkit_defaults_qwen32b_judge"
    ),
    BenchmarkSpec(
        "mathvista", "MathVista", "MathVista_MINI", "vlmevalkit_defaults_qwen32b_judge"
    ),
    BenchmarkSpec(
        "mathverse",
        "MathVerse",
        "MathVerse_MINI_Vision_Only_cot",
        "vlmevalkit_defaults_qwen32b_judge",
    ),
    BenchmarkSpec(
        "wemath",
        "WeMath",
        "WeMath_COT",
        "vlmevalkit_cot_qwen32b_judge",
        "wemath_local_judge",
    ),
    BenchmarkSpec(
        "phyx_mini_mc", "PhyX mini MC", "PhyX_mini_MC", "vlmevalkit_defaults"
    ),
    BenchmarkSpec(
        "mmmu_pro_vision", "MMMU-ProVis", "MMMU_Pro_V_COT", "vlmevalkit_cot_max2048"
    ),
    BenchmarkSpec("realworldqa", "RealWorldQA", "RealWorldQA", "vlmevalkit_defaults"),
    BenchmarkSpec("mmstar", "MMStar", "MMStar", "vlmevalkit_defaults"),
    BenchmarkSpec("embspatial", "EmbSpatial", "EmbSpatialBench", "vlmevalkit_defaults"),
    BenchmarkSpec(
        "spatialvizbench_cot",
        "SpatialVizBench COT",
        "SpatialVizBench_CoT",
        "vlmevalkit_cot",
    ),
    BenchmarkSpec("cvbench_3d", "CV-Bench 3D", "CV-Bench-3D", "vlmevalkit_defaults"),
    BenchmarkSpec("erqa", "ERQA", "ERQA", "vlmevalkit_defaults"),
    BenchmarkSpec("blink", "BLINK", "BLINK", "vlmevalkit_defaults"),
    BenchmarkSpec(
        "countbenchqa", "CountBenchQA", "CountBenchQA", "vlmevalkit_defaults"
    ),
    BenchmarkSpec("countqa", "CountQA", "CountQA", "vlmevalkit_cot_boxed"),
    BenchmarkSpec("treebench", "TreeBench", "TreeBench", "vlmevalkit_defaults"),
    BenchmarkSpec("puzzlevqa", "PuzzleVQA", "PuzzleVQA", "vlmevalkit_reasoning"),
    BenchmarkSpec(
        "visualpuzzles", "VisualPuzzles", "VisualPuzzles", "vlmevalkit_reasoning"
    ),
    BenchmarkSpec(
        "logicvista", "LogicVista", "LogicVista", "vlmevalkit_defaults_qwen32b_judge"
    ),
    BenchmarkSpec(
        "mme_reasoning",
        "MME-Reasoning",
        "MME-Reasoning",
        "vlmevalkit_defaults_qwen32b_judge",
        "mme_reasoning_local_judge",
    ),
)

_SPEC_BY_KEY = {spec.key: spec for spec in _SPECS}
_SUITE_BY_KEY = {
    benchmark.key: benchmark for benchmark in _TRACE_EVAL_V1_SUITE.benchmarks
}
if tuple(_SPEC_BY_KEY) != TRACE_EVAL_V1_BENCHMARKS:
    raise RuntimeError("trace_eval_v1 benchmark registry does not match suite order")
for _key, _spec in _SPEC_BY_KEY.items():
    _suite_benchmark = _SUITE_BY_KEY[_key]
    if _spec.alias != _suite_benchmark.official_alias:
        raise RuntimeError(
            f"trace_eval_v1 alias mismatch for {_key}: "
            f"{_spec.alias!r} != {_suite_benchmark.official_alias!r}"
        )

ALL_BENCHMARKS = _SPECS


def spec_by_key(key: str) -> BenchmarkSpec:
    try:
        return _SPEC_BY_KEY[key]
    except KeyError as error:
        raise KeyError(f"{key!r} is not a trace_eval_v1 benchmark") from error


def _normalize_treebench_options(dataset: Any) -> list[dict[str, Any]]:
    """Normalize the one upstream TreeBench delimiter defect fail-closed."""

    import pandas as pd

    def has_value(value: Any) -> bool:
        try:
            if pd.isna(value):
                return False
        except (TypeError, ValueError):
            pass
        return bool(str(value).strip())

    data = dataset.data.copy()
    changes: list[dict[str, Any]] = []
    option_letters = "ABCDEFGHIJK"
    for row_pos, row in data.iterrows():
        answer = str(row.get("answer", "")).strip().upper()
        if len(answer) != 1 or answer not in option_letters:
            continue
        present = [
            letter
            for letter in option_letters
            if letter in data and has_value(row.get(letter))
        ]
        if answer in present:
            continue

        metadata_options = parse_lettered_option_blob(row.get("multi-choice options"))
        if not present and answer in metadata_options:
            # The OCR split intentionally leaves option columns empty because
            # the choices must be read from the image.
            continue

        normalized = False
        for letter in present:
            if letter == option_letters[-1]:
                continue
            next_letter = option_letters[option_letters.index(letter) + 1]
            value = str(row.get(letter, ""))
            marker = re.search(rf"\s+{re.escape(next_letter)}\.\s+", value)
            if marker is None:
                continue

            prefix = value[: marker.start()].strip()
            embedded = value[marker.end() :].strip()
            if not prefix or not embedded:
                continue

            highest = max(option_letters.index(item) for item in present)
            next_index = option_letters.index(next_letter)
            for option_index in range(highest, next_index - 1, -1):
                source = option_letters[option_index]
                target = option_letters[option_index + 1]
                data.at[row_pos, target] = row.get(source)
            data.at[row_pos, letter] = prefix
            data.at[row_pos, next_letter] = embedded
            changes.append(
                {
                    "index": str(row.get("index", row_pos)),
                    "split_column": letter,
                    "inserted_column": next_letter,
                    "ground_truth": answer,
                }
            )
            normalized = True
            break

        refreshed = data.loc[row_pos]
        normalized_present = [
            letter
            for letter in option_letters
            if letter in data and has_value(refreshed.get(letter))
        ]
        if not normalized or answer not in normalized_present:
            raise ValueError(
                "TreeBench row has a ground-truth option missing from its prompt "
                f"choices: index={row.get('index', row_pos)!r} "
                f"answer={answer!r} choices={present!r}"
            )

    dataset.data = data
    return changes


def build_vlmeval_dataset(spec: BenchmarkSpec) -> Any:
    """Build one canonical dataset with reviewed reproducibility adaptations."""

    # Reject hand-constructed or stale specs before touching VLMEvalKit.
    if spec_by_key(spec.key) != spec:
        raise ValueError(f"benchmark spec is not canonical trace_eval_v1: {spec.key}")

    if spec.key == "mme_reasoning":
        from vlmeval.dataset.image_vqa import MMEReasoning

        MMEReasoning.DATASET_URL[spec.alias] = MME_REASONING_TSV_URL
        MMEReasoning.DATASET_MD5 = {spec.alias: MME_REASONING_MD5}

    if spec.key == "realworldqa":
        # The HTTP endpoint serves the byte-identical artifact and the upstream
        # MD5 remains mandatory.
        from vlmeval.dataset.image_mcq import ImageMCQDataset

        ImageMCQDataset.DATASET_URL[spec.alias] = REALWORLDQA_TSV_URL
        ImageMCQDataset.DATASET_MD5[spec.alias] = REALWORLDQA_MD5

    if spec.key == "erqa":
        # Two upstream classes share the ERQA alias. Select the 400-row EASI
        # leaderboard dataset explicitly.
        from vlmeval.dataset.erqabench import ERQABench

        dataset = ERQABench(dataset=spec.alias)
    else:
        from vlmeval.dataset import build_dataset

        dataset = build_dataset(spec.alias)
    if dataset is None:
        raise RuntimeError(f"VLMEvalKit could not build dataset {spec.alias}")

    normalization: dict[str, Any] = {}
    if spec.key == "treebench":
        normalization["treebench_option_normalization"] = _normalize_treebench_options(
            dataset
        )
    dataset.trace_normalization = normalization
    return dataset


def benchmark_dir(
    spec: BenchmarkSpec,
    model_slug: str,
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT,
) -> Path:
    return benchmark_root / spec.key / model_slug / spec.run_name


def run_dir(
    spec: BenchmarkSpec,
    model_slug: str,
    run_root: Path = DEFAULT_RUN_ROOT,
) -> Path:
    return run_root / spec.key / model_slug / spec.run_name


def score_path(
    spec: BenchmarkSpec,
    model_slug: str,
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT,
) -> Path:
    return benchmark_dir(spec, model_slug, benchmark_root) / "scores.json"


def benchmark_specs_for_run_set(
    run_set: str, model_slug: str = "model"
) -> list[BenchmarkSpec]:
    del model_slug
    if run_set != "trace_eval_v1":
        raise ValueError(f"Unknown run_set {run_set!r}")
    return list(ALL_BENCHMARKS)


def spec_matches_selector(spec: BenchmarkSpec, selector: str) -> bool:
    return selector in {spec.key, spec.alias, spec.display, spec.run_name}


def filter_benchmark_specs(
    specs: Iterable[BenchmarkSpec],
    *,
    only: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> list[BenchmarkSpec]:
    out = list(specs)
    keep = {str(item) for item in only if str(item)}
    if keep:
        out = [
            spec
            for spec in out
            if any(spec_matches_selector(spec, selector) for selector in keep)
        ]
    drop = {str(item) for item in exclude if str(item)}
    if drop:
        out = [
            spec
            for spec in out
            if not any(spec_matches_selector(spec, selector) for selector in drop)
        ]
    return out


def local_judge_eval_mode(spec: BenchmarkSpec) -> str | None:
    if spec.eval_mode == "evochart_local_judge":
        return spec.eval_mode
    if spec.alias.startswith("CharXiv_"):
        return "charxiv_local_judge"
    if spec.alias == "MathVision":
        return "mathv_local_judge"
    if spec.alias.startswith("MathVista"):
        return "mathvista_local_judge"
    if spec.alias.startswith("MathVerse"):
        return "mathverse_local_judge"
    if spec.alias == "LogicVista":
        return "logicvista_local_judge"
    return None


def json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.bool_):
            return bool(value)
    except Exception:
        pass
    return str(value)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


@contextmanager
def file_lock(lock_path: Path):
    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _job_done(done_path: Path | None) -> bool:
    return bool(done_path and done_path.exists())


def _pid_alive(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except Exception:
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def claim_next_job(
    *,
    queue_path: Path,
    jobs: Iterable[tuple[str, Path | None]],
    worker_id: str,
    stale_after_sec: float = 12 * 3600,
    max_attempts: int = 2,
) -> str | None:
    lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
    now = time.time()
    with file_lock(lock_path):
        state = load_json(queue_path, {"jobs": {}})
        state_jobs = state.setdefault("jobs", {})
        for job_id, done_path in jobs:
            info = state_jobs.get(job_id, {})
            if _job_done(done_path):
                state_jobs[job_id] = {
                    **info,
                    "status": "done",
                    "done_path": str(done_path),
                    "updated_at": now,
                }
                continue
            attempts = int(info.get("attempts") or 0)
            if info.get("status") == "failed" and attempts >= max_attempts:
                continue
            if info.get("status") == "running":
                if _pid_alive(info.get("pid")):
                    continue
                if now - float(info.get("updated_at", 0)) < stale_after_sec:
                    continue
            if info.get("status") == "done":
                continue
            state_jobs[job_id] = {
                "status": "running",
                "worker": worker_id,
                "pid": os.getpid(),
                "updated_at": now,
                "attempts": attempts + 1,
            }
            write_json(queue_path, state)
            return job_id
        write_json(queue_path, state)
    return None


def mark_job(queue_path: Path, job_id: str, status: str, **extra: Any) -> None:
    now = time.time()
    lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
    with file_lock(lock_path):
        state = load_json(queue_path, {"jobs": {}})
        state_jobs = state.setdefault("jobs", {})
        previous = state_jobs.get(job_id, {})
        entry = {
            "status": status,
            "worker": extra.pop("worker", None),
            "pid": os.getpid(),
            "updated_at": now,
            **extra,
        }
        if "attempts" not in entry and "attempts" in previous:
            entry["attempts"] = previous["attempts"]
        state_jobs[job_id] = entry
        write_json(queue_path, state)


def score_to_percent(value: Any) -> float | None:
    try:
        score = float(value)
    except Exception:
        return None
    if math.isnan(score):
        return None
    if abs(score) <= 1.0:
        score *= 100.0
    return score


def extract_score_and_rows(
    scores_obj: dict[str, Any],
) -> tuple[float | None, int | None]:
    """Extract the canonical scalar from a trace_eval_v1 score receipt."""

    rows = scores_obj.get("rows")
    rows_int = int(rows) if rows is not None else None
    if "Overall_Accuracy" in scores_obj:
        return float(scores_obj["Overall_Accuracy"]), rows_int
    for key in ("accuracy", "acc", "score", "Overall", "overall"):
        if key in scores_obj:
            parsed = score_to_percent(scores_obj[key])
            if parsed is not None:
                return parsed, rows_int

    scores = scores_obj.get("scores", {})
    if not isinstance(scores, dict):
        return None, rows_int
    if "Overall_Accuracy" in scores:
        return float(scores["Overall_Accuracy"]), rows_int
    for key in ("Overall", "overall", "accuracy", "acc", "val"):
        if key not in scores:
            continue
        value = scores[key]
        if isinstance(value, dict):
            for nested_key in (
                "Accuracy (%)",
                "accuracy",
                "acc",
                "score",
                "Score",
                "Overall",
                "overall",
            ):
                if nested_key in value:
                    parsed = score_to_percent(value[nested_key])
                    if parsed is not None:
                        return parsed, rows_int
        return score_to_percent(value), rows_int

    table = scores.get("table")
    if isinstance(table, list):
        average_values: list[float] = []
        for row in table:
            if not isinstance(row, dict):
                continue
            if isinstance(row.get("average_scores"), list):
                for value in row["average_scores"]:
                    parsed = score_to_percent(value)
                    if parsed is not None:
                        average_values.append(parsed)
            if any(str(value).lower() == "overall" for value in row.values()):
                for key in (
                    "acc",
                    "accuracy",
                    "Accuracy (%)",
                    "score",
                    "Score",
                    "Overall",
                    "overall",
                    "1",
                ):
                    if key in row:
                        table_rows = int(
                            row.get("tot", row.get("Samples", rows or 0)) or 0
                        )
                        return score_to_percent(row[key]), table_rows
        if average_values:
            return sum(average_values) / len(average_values), rows_int

    numeric_scores: list[float] = []
    skip_keys = {
        "rows",
        "samples",
        "num_correct",
        "num_total",
        "illformed_responses",
        "category_stats",
        "table",
        "tabulated_keys",
        "tabulated_results",
    }
    for key, value in scores.items():
        if str(key).lower() in skip_keys:
            continue
        parsed = score_to_percent(value)
        if parsed is not None:
            numeric_scores.append(parsed)
    if numeric_scores:
        return sum(numeric_scores) / len(numeric_scores), rows_int
    return None, rows_int
