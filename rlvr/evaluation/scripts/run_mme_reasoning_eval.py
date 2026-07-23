#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
VLMEVAL_ROOT = REPO_ROOT / "external" / "VLMEvalKit"
for path in (SCRIPTS_ROOT, VLMEVAL_ROOT, VLMEVAL_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_queue_lib import (  # noqa: E402
    MME_REASONING_MD5,
    MME_REASONING_TSV_URL,
    json_default,
    run_dir,
    spec_by_key,
    write_json,
)
from run_external_benchmark_score_queue import (  # noqa: E402
    PERSISTENT_JUDGE_CACHE_CONTRACT_VERSION,
    PersistentJudge,
)

MME_REASONING_SPEC = spec_by_key("mme_reasoning")
MME_EVALUATION_CONTRACT_VERSION = "trace-eval-v1-mme-evaluation-v5"
MME_EXTRACTION_ARCHIVE_CONTRACT_VERSION = "trace-eval-v1-mme-extraction-v4"
MME_SCORE_ARCHIVE_CONTRACT_VERSION = "trace-eval-v1-mme-score-v4"
MME_EXTRACTION_JUDGE_CONTRACT_VERSION = "trace-eval-v1-mme-extraction-judge-v2"
MME_OPEN_JUDGE_CONTRACT_VERSION = "trace-eval-v1-mme-open-judge-v2"
MME_FAILURE_FILENAME = "mme_reasoning_failures.json"
MME_API_FAILURE_MESSAGE = "Failed to obtain answer via API."
MME_NO_CHOICE_SENTINEL = "__MME_NO_CHOICE__"
MME_RETRY_TEMPERATURES = (0.0, 0.5, 1.0, 1.5, 2.0)


class MMEEvaluationFailure(RuntimeError):
    """Raised when an MME slice cannot produce a trustworthy score."""


@dataclass
class _PendingMMEScore:
    data: pd.DataFrame
    acc_df: pd.DataFrame
    score_dict: dict[str, float]
    scores: dict[str, Any]
    extraction_rows: dict[str, dict[str, Any]]
    score_rows: dict[str, dict[str, Any]]


def _mme_score_artifact_paths(output_dir: Path) -> tuple[Path, ...]:
    return (
        output_dir / "scores.json",
        output_dir / "scores_raw.json",
        output_dir / f"{MME_REASONING_SPEC.alias}_predictions_qwen3_32b_score.xlsx",
        output_dir / "MME-Reasoning_acc.csv",
        output_dir / "MME-Reasoning_acc.xlsx",
    )


def _clear_mme_score_artifacts(output_dir: Path) -> None:
    for path in _mme_score_artifact_paths(output_dir):
        path.unlink(missing_ok=True)


def _discard_archive_descriptor(path: Path | None) -> None:
    """Withdraw a local ready descriptor before the archive daemon can consume it."""

    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _mme_judge_contract_version(args: argparse.Namespace, route_version: str) -> str:
    base_version = str(
        getattr(
            args,
            "judge_cache_contract_version",
            PERSISTENT_JUDGE_CACHE_CONTRACT_VERSION,
        )
    )
    return f"{base_version}+{route_version}"


def _load_mme_scoring_runtime():
    from vlmeval.dataset.utils.mme_reasoning import (
        FAIL_MSG,
        mme_reasoning_eval_functions,
    )
    from vlmeval.smp import dump

    return FAIL_MSG, mme_reasoning_eval_functions, dump


def _write_mme_failure(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    stage: str,
    failures: list[dict[str, Any]],
) -> Path:
    path = output_dir / MME_FAILURE_FILENAME
    write_json(
        path,
        {
            "contract_version": MME_EVALUATION_CONTRACT_VERSION,
            "dataset": MME_REASONING_SPEC.alias,
            "model": args.model,
            "model_slug": args.model_slug,
            "seed": int(args.seed),
            "stage": stage,
            "count": len(failures),
            "failures": failures,
        },
    )
    return path


def _fail_mme_evaluation(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    stage: str,
    failures: list[dict[str, Any]],
    message: str,
) -> None:
    failure_path = _write_mme_failure(args, output_dir, stage=stage, failures=failures)
    raise MMEEvaluationFailure(f"{message}; details={failure_path}")


def _patch_mme_reasoning_dataset() -> None:
    from vlmeval.dataset.image_vqa import MMEReasoning

    MMEReasoning.DATASET_URL["MME-Reasoning"] = MME_REASONING_TSV_URL
    MMEReasoning.DATASET_MD5 = {"MME-Reasoning": MME_REASONING_MD5}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _clean_optional(value: Any) -> Any:
    return None if _is_missing(value) else value


def _safe_literal_eval(value: Any) -> Any:
    if _is_missing(value):
        return None
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return eval(
            value, {"__builtins__": {}}, {}
        )  # noqa: S307 - matches VLMEvalKit evaluator contract.


def _ensure_eval_prompt_function(line: pd.Series) -> tuple[str, str]:
    eval_prompt = _clean_optional(line.get("prompt_id"))
    eval_function = _clean_optional(line.get("function_id"))
    if eval_function is None:
        question_type = str(line.get("question_type", "")).lower()
        if question_type == "choice":
            return "choice_prompt", "choice_function"
        if question_type == "open":
            return "open_question_prompt", "open_function"
        raise NotImplementedError(
            f"Question type requires function_id: {line.get('question_type')}"
        )
    if eval_prompt is None:
        raise ValueError(
            f"Row {line.get('index')} has function_id={eval_function!r} but no prompt_id"
        )
    return str(eval_prompt), str(eval_function)


def _extract_prompt(line: pd.Series) -> tuple[str, str]:
    from vlmeval.dataset.utils.mme_reasoning import mme_reasoning_eval_prompts

    eval_prompt, eval_function = _ensure_eval_prompt_function(line)
    prompt = mme_reasoning_eval_prompts[eval_prompt].format(
        question=line["question"],
        response=line["prediction"],
    )
    return prompt, eval_prompt


def _extract_json_from_response(text: str) -> str | None:
    from vlmeval.dataset.utils.mme_reasoning import extract_json_from_response

    return extract_json_from_response(text)


def _normalize_choice_extraction(output: str) -> str | None:
    """Normalize the MME judge's single- or multi-option extraction."""

    text = str(output or "").strip()
    if not text:
        return None

    no_choice = re.sub(r"\s+", " ", text).strip().rstrip(".!").casefold()
    if (
        no_choice == "none"
        or no_choice == "none of the above"
        or no_choice.startswith("none of the given options ")
        or no_choice.startswith("none of the options ")
    ):
        return MME_NO_CHOICE_SENTINEL

    placeholder = text
    if text.startswith("[") or text.startswith("("):
        closing = "]" if text.startswith("[") else ")"
        if not text.endswith(closing) or any(token in text[1:-1] for token in "[]()"):
            placeholder = ""
        else:
            placeholder = text[1:-1].strip()
    elif any(token in text for token in "[]()"):
        placeholder = ""
    if placeholder and re.fullmatch(r"\?(?:(?:\s*,\s*|\s+)\?)*", placeholder):
        return MME_NO_CHOICE_SENTINEL

    repeated_label = re.fullmatch(
        r"([A-Ga-g])\s*(?:\r?\n)+\s*Extracted answer:\s*([A-Ga-g])",
        text,
        flags=re.IGNORECASE,
    )
    if repeated_label is not None:
        first, second = (value.upper() for value in repeated_label.groups())
        return first if first == second else text

    # MME-Reasoning labels three choice rows with numeric identifiers (2, 4,
    # and 44).  The pinned evaluator treats these exactly like letter labels.
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return text

    # The official choice extractor accepts any nonempty response before
    # choice_function compares it with the option label. Preserve a boxed
    # atomic symbol as one response, while keeping nested LaTeX/prose invalid.
    if re.fullmatch(r"\\boxed\{\\[A-Za-z]+\}", text):
        return text

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = None
    if isinstance(parsed, (list, tuple)):
        values = [str(value).strip().upper() for value in parsed]
    else:
        inner = text.strip("[]() ")
        values = [part.strip().strip("\"'").upper() for part in inner.split(",")]
        if len(values) == 1 and re.fullmatch(r"[A-G]{2,7}", values[0]):
            compact = list(values[0])
            values = compact if len(set(compact)) == len(compact) else values
    if values and all(re.fullmatch(r"[A-Z]", value) is not None for value in values):
        return ",".join(values)

    # The official choice prompt asks for a bare option, but the judge can
    # echo an unambiguous label as "B." or "B. <option text>".  Preserve the
    # official scorer's label semantics without accepting conflicting labels.
    prefixed = re.fullmatch(r"([A-Za-z])\s*[.)]\s*(.*)", text, flags=re.DOTALL)
    if prefixed is not None:
        label = prefixed.group(1).upper()
        tail = prefixed.group(2).strip()
        if re.search(
            r"(?:^|[\s,;/])(?:or|and)?\s*[A-Za-z]\s*[.)](?:\s|$)", tail, re.IGNORECASE
        ):
            return text
        return label

    # The pinned MME choice extractor accepts any nonempty response and lets
    # choice_function determine correctness. Preserve short atomic option text
    # while still rejecting wrappers, prose labels, and conflicting candidates.
    if (
        len(text) <= 128
        and "\n" not in text
        and "\r" not in text
        and len(text.split()) <= 12
        and re.search(r"[`\\<>{}\[\]]", text) is None
        and re.search(
            r"\b(?:answer|choice|extracted|none|option)\b",
            text,
            re.IGNORECASE,
        )
        is None
        and re.search(r"(?<![A-Za-z0-9])[A-G](?![A-Za-z0-9])", text) is None
        and re.fullmatch(
            r"\s*[A-Ga-g]\s*(?:(?:,|/|\bor\b|\band\b)\s*[A-Ga-g]\s*)+",
            text,
            re.IGNORECASE,
        )
        is None
        and re.fullmatch(r"[A-Z]{1,7}", text) is None
    ):
        return text
    # Pinned VLMEvalKit accepts every nonempty choice extraction and delegates
    # correctness to choice_function. Preserve its raw fallback instead of
    # turning verbose or conflicting judge text into an evaluation failure.
    return text


def _validate_extraction(eval_prompt: str, output: str) -> tuple[bool, str]:
    if not str(output or "").strip():
        return False, output
    if MME_API_FAILURE_MESSAGE in output:
        return False, output
    if eval_prompt == "choice_prompt":
        normalized = _normalize_choice_extraction(output)
        return (normalized is not None), (normalized or output)
    if eval_prompt in {"open_question_prompt", "points24_prompt"}:
        return True, output
    try:
        json.loads(output)
        return True, output
    except Exception:
        extracted = _extract_json_from_response(output)
        if extracted is None:
            return False, output
        try:
            json.loads(extracted)
            return True, extracted
        except Exception:
            return False, output


def _openeval_prompt(line: pd.Series) -> str:
    return """
Please read the following example. Then judge the answer and type it at the end of the prompt.
Below are two examples. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note:
    Different expressions of the same number should also be considered consistent, for example, \\frac{{7}}{{2}} and 3.5.
    If a conversion results in a decimal approximation, the expressions can be considered consistent if the values are equal up to two decimal places, for example, \\sqrt{{3}} and 1.73.
If they are consistent, Judgement is 1; if they are different, Judgement is 0.\n
Example 1:
    [Question]: What is the minimize length of the line?
    [Standard answer]: \\sqrt{{2}}
    [Model answer]: 1.414
    [Judgement]: 1
Example 2:
    [Question]: Given an image of a 3x3 maze. How to reach the end cell marked 'E' from the start cell is marked 'S'.
    [Standard answer]: ['Left', 'Right']
    [Model answer]: 'Left', 'Right'
    [Judgement]: 1

Now, judge the anwser for the following question:
    [Question]: {question}
    [Standard answer]: {answer}
    [Model answer]: {response}
    [Judgement]:
You should only output the judgement without any other texts.
""".format(question=line["question"], answer=line["answer"], response=line["res"])


def _load_eval_table(output_dir: Path) -> pd.DataFrame:
    eval_file = output_dir / f"{MME_REASONING_SPEC.alias}_predictions.xlsx"
    if not eval_file.exists():
        raise FileNotFoundError(f"Missing prediction table: {eval_file}")
    return pd.read_excel(eval_file).replace({float("nan"): None})


def _mme_reasoning_acc(data: pd.DataFrame) -> pd.DataFrame:
    capabilities = [
        "planning and exploring",
        "calculation",
        "spatial-temporal",
        "casual chaining analysis",
        "pattern analysis",
    ]
    reasoning_types = ["inductive", "deductive", "abductive"]

    result: dict[str, list[float]] = {"Overall": [float(data["score"].mean() * 100)]}
    for capability in capabilities:
        sub = data[data["capability"].apply(lambda value: capability in str(value))]
        result[capability] = [
            float(sub["score"].mean() * 100) if len(sub) else float("nan")
        ]
    for reasoning_type in reasoning_types:
        sub = data[
            data["reasoning_type"].apply(lambda value: reasoning_type in str(value))
        ]
        result[reasoning_type] = [
            float(sub["score"].mean() * 100) if len(sub) else float("nan")
        ]
    return pd.DataFrame(result)


def _archive_mme_common(
    row: dict[str, Any],
    ordinal: int,
    *,
    canonical_json,
    sanitize_benchmark_source_row,
) -> dict[str, Any]:
    source_hash = str(row.get("source_row_hash") or "").strip()
    if not source_hash:
        source_row = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "prediction",
                "finish_reason",
                "output_token_count",
                "prompt_token_count",
                "request_hash",
                "source_ordinal",
                "source_row_hash",
                "prompt",
                "res",
                "log",
                "score",
                "log_score",
            }
        }
        safe_source = sanitize_benchmark_source_row(source_row)
        source_hash = hashlib.sha256(
            canonical_json(safe_source).encode("utf-8")
        ).hexdigest()
    persisted_ordinal = row.get("source_ordinal")
    if persisted_ordinal is None or str(persisted_ordinal).strip() == "":
        persisted_ordinal = ordinal
    return {
        "source_index": str(row.get("index", ordinal)),
        "source_ordinal": int(persisted_ordinal),
        "source_row_hash": source_hash,
        "question": row.get("question"),
        "ground_truth": row.get("answer"),
        "metadata": {
            "question_type": row.get("question_type"),
            "function_id": row.get("function_id"),
            "prompt_id": row.get("prompt_id"),
        },
    }


def _mme_archive_identity(args: argparse.Namespace) -> dict[str, Any]:
    from trace_eval_archive_hooks import resolve_model_revision, resolve_model_source

    return {
        "model": resolve_model_source(args.model_slug, args.model),
        "model_slug": args.model_slug,
        "model_revision": resolve_model_revision(args.model_slug, args.model),
        "seed": int(args.seed),
        "benchmark": MME_REASONING_SPEC.key,
        "dataset_alias": MME_REASONING_SPEC.alias,
        "dataset_split": MME_REASONING_SPEC.split or "default",
        "dataset_revision": os.environ.get(
            "TRACE_EVAL_DATASET_REVISION",
            os.environ.get("TRACE_VLMEVALKIT_GIT_COMMIT", "unknown"),
        ),
    }


def _archive_mme_extractions(
    args: argparse.Namespace,
    data: pd.DataFrame,
    extraction_rows: dict[str, dict[str, Any]],
) -> Path | None:
    if not os.environ.get("TRACE_EVAL_HF_SPOOL_ROOT", "").strip():
        return None
    from trace_eval_archive_hooks import (
        canonical_json,
        emit_extraction_slice,
        sanitize_benchmark_source_row,
    )

    records = []
    for ordinal, row in enumerate(data.to_dict(orient="records")):
        index = str(row["index"])
        extracted = extraction_rows[index]
        request_hash = str(extracted.get("request_hash") or "").strip()
        if not request_hash:
            request_hash = hashlib.sha256(
                canonical_json(
                    {
                        "contract_version": MME_EXTRACTION_ARCHIVE_CONTRACT_VERSION,
                        "index": index,
                        "model_response": row.get("prediction", ""),
                        "judge_prompt": extracted.get("judge_prompt", ""),
                    }
                ).encode("utf-8")
            ).hexdigest()
        records.append(
            {
                **_archive_mme_common(
                    row,
                    ordinal,
                    canonical_json=canonical_json,
                    sanitize_benchmark_source_row=sanitize_benchmark_source_row,
                ),
                "request_hash": request_hash,
                "model_response": row.get("prediction", ""),
                "judge_prompt": extracted.get("judge_prompt", ""),
                "judge_response": extracted.get("judge_response", ""),
                "normalized_extraction": {
                    "status": (
                        "resolved"
                        if str(extracted.get("res", "")).strip()
                        else "invalid"
                    ),
                    "value": extracted.get("res"),
                    "method": "mme_reasoning_official_prompt",
                    "contract_version": MME_EVALUATION_CONTRACT_VERSION,
                },
                "retries": {"count": int(extracted.get("retry_count") or 0)},
            }
        )
    return emit_extraction_slice(
        records=records,
        contract_version=MME_EXTRACTION_ARCHIVE_CONTRACT_VERSION,
        aggregate={
            "rows": len(records),
            "unresolved": sum(
                not str(extraction_rows[str(row["index"])].get("res", "")).strip()
                for row in data.to_dict(orient="records")
            ),
        },
        **_mme_archive_identity(args),
    )


def _archive_mme_scores(
    args: argparse.Namespace,
    data: pd.DataFrame,
    extraction_rows: dict[str, dict[str, Any]],
    score_rows: dict[str, dict[str, Any]],
    summary: dict[str, Any],
) -> Path | None:
    if not os.environ.get("TRACE_EVAL_HF_SPOOL_ROOT", "").strip():
        return None
    from trace_eval_archive_hooks import (
        canonical_json,
        emit_score_slice,
        sanitize_benchmark_source_row,
    )

    records = []
    for ordinal, row in enumerate(data.to_dict(orient="records")):
        index = str(row["index"])
        extracted = extraction_rows[index]
        scored = score_rows[index]
        scorer = (
            "qwen3_open_correctness"
            if scored.get("judge_prompt")
            else str(row.get("function_id") or "choice_function")
        )
        score_request_hash = hashlib.sha256(
            canonical_json(
                {
                    "contract_version": MME_SCORE_ARCHIVE_CONTRACT_VERSION,
                    "extraction_request_hash": extracted.get("request_hash"),
                    "judge_request_hash": scored.get("judge_request_hash"),
                    "prediction": extracted.get("res"),
                    "score": bool(scored.get("score")),
                    "scorer": scorer,
                }
            ).encode("utf-8")
        ).hexdigest()
        common = _archive_mme_common(
            row,
            ordinal,
            canonical_json=canonical_json,
            sanitize_benchmark_source_row=sanitize_benchmark_source_row,
        )
        common["metadata"] = {
            **common["metadata"],
            "judge_prompt": scored.get("judge_prompt", ""),
            "judge_response": scored.get("judge_response", ""),
            "judge_request_hash": scored.get("judge_request_hash", ""),
            "judge_retry_count": int(scored.get("judge_retry_count") or 0),
            "log_score": scored.get("log_score", ""),
            "evaluation_status": (
                "correct" if bool(scored.get("score")) else "incorrect"
            ),
            "evaluation_contract_version": MME_EVALUATION_CONTRACT_VERSION,
        }
        records.append(
            {
                **common,
                "request_hash": score_request_hash,
                "prediction": extracted.get("res"),
                "score": bool(scored.get("score")),
                "scorer": scorer,
                "excluded": False,
            }
        )
    return emit_score_slice(
        records=records,
        contract_version=MME_SCORE_ARCHIVE_CONTRACT_VERSION,
        aggregate=summary,
        **_mme_archive_identity(args),
    )


def _mme_preflight_failures(data: pd.DataFrame, fail_msg: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if data.empty:
        return [{"kind": "dataset_contract", "message": "prediction table is empty"}]
    missing_columns = sorted({"index", "prediction"} - set(data.columns))
    if missing_columns:
        failures.append(
            {
                "kind": "dataset_contract",
                "message": f"prediction table is missing columns: {', '.join(missing_columns)}",
            }
        )
        return failures

    seen: set[str] = set()
    duplicate_indices: set[str] = set()
    for ordinal, (_, line) in enumerate(data.iterrows()):
        idx = str(line["index"])
        if idx in seen:
            duplicate_indices.add(idx)
        seen.add(idx)
        prediction = _clean_optional(line.get("prediction"))
        finish_reason = str(_clean_optional(line.get("finish_reason")) or "").lower()
        if fail_msg in str(prediction or "") or finish_reason == "error":
            failures.append(
                {
                    "kind": "generation_failure",
                    "index": idx,
                    "source_ordinal": ordinal,
                    "finish_reason": finish_reason or None,
                    "message": str(prediction or fail_msg),
                }
            )
    if duplicate_indices:
        failures.append(
            {
                "kind": "dataset_contract",
                "indices": sorted(duplicate_indices),
                "message": "source indices must be unique",
            }
        )
    return failures


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = run_dir(MME_REASONING_SPEC, args.model_slug, args.run_root)
    benchmark_output_dir = run_dir(
        MME_REASONING_SPEC, args.model_slug, args.benchmark_root
    )
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    _clear_mme_score_artifacts(benchmark_output_dir)
    failure_context: dict[str, Any] = {"stage": "initialization", "indices": []}
    judge: PersistentJudge | None = None
    cleanup_error: Exception | None = None
    pending_score: _PendingMMEScore | None = None
    dump = None
    try:
        fail_msg, eval_functions, dump = _load_mme_scoring_runtime()
        _patch_mme_reasoning_dataset()
        failure_context["stage"] = "prediction_loading"
        data = _load_eval_table(output_dir)
        preflight_failures = _mme_preflight_failures(data, fail_msg)
        if preflight_failures:
            _fail_mme_evaluation(
                args,
                benchmark_output_dir,
                stage="prediction_preflight",
                failures=preflight_failures,
                message="MME-Reasoning prediction input is not evaluable",
            )
        judge = PersistentJudge(args)
        pending_score = _run_score_impl(
            args,
            data=data,
            output_dir=output_dir,
            benchmark_output_dir=benchmark_output_dir,
            judge=judge,
            fail_msg=fail_msg,
            eval_functions=eval_functions,
            failure_context=failure_context,
        )
    except MMEEvaluationFailure:
        raise
    except Exception as exc:
        failures = [
            {
                "kind": "evaluator_exception",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "pending_indices": list(failure_context.get("indices") or []),
            }
        ]
        failure_path = _write_mme_failure(
            args,
            benchmark_output_dir,
            stage=str(failure_context.get("stage") or "unknown"),
            failures=failures,
        )
        raise MMEEvaluationFailure(
            f"MME-Reasoning evaluation failed at {failure_context.get('stage')}; details={failure_path}"
        ) from exc
    finally:
        if judge is not None:
            try:
                judge.cleanup()
            except (
                Exception
            ) as exc:  # Cleanup must run, but must not hide an active failure.
                cleanup_error = exc

    if cleanup_error is not None:
        _clear_mme_score_artifacts(benchmark_output_dir)
        failure_path = _write_mme_failure(
            args,
            benchmark_output_dir,
            stage="judge_cleanup",
            failures=[
                {
                    "kind": "evaluator_exception",
                    "exception_type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                }
            ],
        )
        raise MMEEvaluationFailure(
            f"MME-Reasoning judge cleanup failed; details={failure_path}"
        )

    if pending_score is None or dump is None:
        raise AssertionError(
            "MME-Reasoning score preparation completed without a pending result"
        )

    score_archive_path: Path | None = None
    try:
        failure_context["stage"] = "score_artifacts"
        score_file = (
            benchmark_output_dir
            / f"{MME_REASONING_SPEC.alias}_predictions_qwen3_32b_score.xlsx"
        )
        pending_score.data.to_excel(score_file, index=False)
        pending_score.acc_df.to_csv(
            benchmark_output_dir / "MME-Reasoning_acc.csv", index=False
        )
        pending_score.acc_df.to_excel(
            benchmark_output_dir / "MME-Reasoning_acc.xlsx", index=False
        )
        dump(pending_score.score_dict, str(benchmark_output_dir / "scores_raw.json"))

        failure_context["stage"] = "archive_finalization"
        extraction_archive_path = _archive_mme_extractions(
            args,
            pending_score.data,
            pending_score.extraction_rows,
        )
        score_archive_path = _archive_mme_scores(
            args,
            pending_score.data,
            pending_score.extraction_rows,
            pending_score.score_rows,
            pending_score.scores,
        )
        archive_paths = [
            str(path)
            for path in (extraction_archive_path, score_archive_path)
            if path is not None
        ]
        if archive_paths:
            pending_score.scores["archive_descriptors"] = archive_paths

        failure_context["stage"] = "score_manifest"
        (benchmark_output_dir / MME_FAILURE_FILENAME).unlink(missing_ok=True)
        write_json(benchmark_output_dir / "scores.json", pending_score.scores)
    except Exception as exc:
        _discard_archive_descriptor(score_archive_path)
        _clear_mme_score_artifacts(benchmark_output_dir)
        failure_path = _write_mme_failure(
            args,
            benchmark_output_dir,
            stage=str(failure_context.get("stage") or "score_finalization"),
            failures=[
                {
                    "kind": "evaluator_exception",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            ],
        )
        raise MMEEvaluationFailure(
            f"MME-Reasoning score finalization failed at {failure_context.get('stage')}; "
            f"details={failure_path}"
        ) from exc
    print(
        f"[mme-score:done] slug={args.model_slug} rows={pending_score.scores['rows']} "
        f"accuracy={pending_score.scores['accuracy']:.2f} output={benchmark_output_dir}"
    )
    return pending_score.scores


def _run_score_impl(
    args: argparse.Namespace,
    *,
    data: pd.DataFrame,
    output_dir: Path,
    benchmark_output_dir: Path,
    judge: PersistentJudge,
    fail_msg: str,
    eval_functions: dict[str, Any],
    failure_context: dict[str, Any],
) -> _PendingMMEScore:
    extraction_judge_contract = _mme_judge_contract_version(
        args, MME_EXTRACTION_JUDGE_CONTRACT_VERSION
    )
    open_judge_contract = _mme_judge_contract_version(
        args, MME_OPEN_JUDGE_CONTRACT_VERSION
    )
    row_by_index = {str(row["index"]): row for _, row in data.iterrows()}
    extraction_rows: dict[str, dict[str, Any]] = {}
    pending_indices = [str(idx) for idx in data["index"]]
    eval_prompt_by_idx: dict[str, str] = {}
    prompt_by_idx: dict[str, str] = {}
    failure_context.update(stage="answer_extraction_setup", indices=pending_indices)
    for _, line in data.iterrows():
        idx = str(line["index"])
        prompt, eval_prompt = _extract_prompt(line)
        prompt_by_idx[idx] = prompt
        eval_prompt_by_idx[idx] = eval_prompt

    last_invalid_extractions: dict[str, dict[str, Any]] = {}
    for attempt, temperature in enumerate(MME_RETRY_TEMPERATURES):
        pending = [(idx, prompt_by_idx[idx]) for idx in pending_indices]
        if not pending:
            break
        failure_context.update(
            stage="answer_extraction", indices=[idx for idx, _ in pending]
        )
        cache_name = f"mme_reasoning_extract_attempt{attempt}.jsonl"
        cache = judge.run_cached(
            output_dir=benchmark_output_dir,
            prompts=pending,
            cache_name=cache_name,
            max_tokens=(
                args.extract_max_tokens
                if attempt == 0
                else max(
                    args.extract_max_tokens, min(args.extract_max_tokens * 2, 2048)
                )
            ),
            temperature=temperature,
            top_p=1.0,
            no_resume=args.no_resume,
            desc=f"{args.model_slug} MME extract attempt {attempt}",
            contract_version=extraction_judge_contract,
            return_unresolved=True,
        )
        next_pending: list[str] = []
        for idx, prompt in pending:
            cache_entry = cache.get(idx, {})
            raw = str(cache_entry.get("judge_output", "")).strip()
            finish_reason = (
                str(cache_entry.get("judge_finish_reason", "")).strip().lower()
            )
            ok, res = _validate_extraction(eval_prompt_by_idx[idx], raw)
            ok = ok and finish_reason not in {"length", "max_tokens"}
            if ok:
                extraction_rows[idx] = {
                    "log": "Succeed",
                    "res": res,
                    "judge_prompt": prompt,
                    "judge_response": raw,
                    "request_hash": cache_entry.get("request_hash", ""),
                    "retry_count": attempt
                    + int(cache_entry.get("judge_retry_count") or 0),
                }
            else:
                last_invalid_extractions[idx] = {
                    "kind": "malformed_extraction",
                    "index": idx,
                    "attempt": attempt,
                    "cache": cache_name,
                    "request_hash": cache_entry.get("request_hash", ""),
                    "judge_output": raw,
                    "judge_finish_reason": finish_reason,
                }
                next_pending.append(idx)
        pending_indices = next_pending

    if pending_indices:
        _fail_mme_evaluation(
            args,
            benchmark_output_dir,
            stage="answer_extraction",
            failures=[last_invalid_extractions[idx] for idx in pending_indices],
            message=(
                "MME-Reasoning judge failed to produce parseable extractions for "
                f"{len(pending_indices)} rows"
            ),
        )

    data = data.copy()
    data["res"] = [extraction_rows[str(idx)]["res"] for idx in data["index"]]
    data["log"] = [extraction_rows[str(idx)]["log"] for idx in data["index"]]
    row_by_index = {str(row["index"]): row for _, row in data.iterrows()}
    extract_file = (
        benchmark_output_dir
        / f"{MME_REASONING_SPEC.alias}_predictions_qwen3_32b_extract.xlsx"
    )
    failure_context.update(stage="extraction_artifact", indices=[])
    data.to_excel(extract_file, index=False)

    score_rows: dict[str, dict[str, Any]] = {}
    open_prompts: list[tuple[str, str]] = []
    invalid_open_rows: list[dict[str, Any]] = []
    for _, line in data.iterrows():
        idx = str(line["index"])
        function_id = _clean_optional(line.get("function_id"))
        is_open = (
            str(line["question_type"]).lower() == "open" and function_id is None
        ) or function_id == "open_function"
        if is_open:
            prediction = _clean_optional(line.get("res"))
            if prediction is None or fail_msg in str(prediction):
                invalid_open_rows.append(
                    {
                        "kind": "invalid_extraction",
                        "index": idx,
                        "message": str(prediction),
                    }
                )
            else:
                open_prompts.append((idx, _openeval_prompt(line)))
    if invalid_open_rows:
        _fail_mme_evaluation(
            args,
            benchmark_output_dir,
            stage="open_answer_setup",
            failures=invalid_open_rows,
            message="MME-Reasoning open-answer extraction is invalid",
        )

    pending_open = [idx for idx, _ in open_prompts]
    prompt_open = dict(open_prompts)
    last_invalid_open: dict[str, dict[str, Any]] = {}
    for attempt, temperature in enumerate(MME_RETRY_TEMPERATURES):
        pending = [(idx, prompt_open[idx]) for idx in pending_open]
        if not pending:
            break
        failure_context.update(
            stage="open_answer_scoring", indices=[idx for idx, _ in pending]
        )
        cache_name = f"mme_reasoning_score_open_attempt{attempt}.jsonl"
        cache = judge.run_cached(
            output_dir=benchmark_output_dir,
            prompts=pending,
            cache_name=cache_name,
            max_tokens=args.score_max_tokens,
            temperature=temperature,
            top_p=1.0,
            no_resume=args.no_resume,
            desc=f"{args.model_slug} MME open score attempt {attempt}",
            contract_version=open_judge_contract,
            return_unresolved=True,
        )
        next_pending = []
        for idx, prompt in pending:
            cache_entry = cache.get(idx, {})
            raw = str(cache_entry.get("judge_output", "")).strip()
            finish_reason = (
                str(cache_entry.get("judge_finish_reason", "")).strip().lower()
            )
            if raw in {"0", "1"} and finish_reason not in {"length", "max_tokens"}:
                score = raw == "1"
                score_rows[idx] = {
                    "log_score": "Succeed",
                    "score": score,
                    "evaluation_status": "correct" if score else "incorrect",
                    "judge_prompt": prompt,
                    "judge_response": raw,
                    "judge_request_hash": cache_entry.get("request_hash", ""),
                    "judge_retry_count": attempt
                    + int(cache_entry.get("judge_retry_count") or 0),
                }
            else:
                last_invalid_open[idx] = {
                    "kind": "malformed_open_judgement",
                    "index": idx,
                    "attempt": attempt,
                    "cache": cache_name,
                    "request_hash": cache_entry.get("request_hash", ""),
                    "judge_output": raw,
                    "judge_finish_reason": finish_reason,
                }
                next_pending.append(idx)
        pending_open = next_pending

    if pending_open:
        _fail_mme_evaluation(
            args,
            benchmark_output_dir,
            stage="open_answer_scoring",
            failures=[last_invalid_open[idx] for idx in pending_open],
            message=(
                "MME-Reasoning judge failed to produce binary decisions for "
                f"{len(pending_open)} open-answer rows"
            ),
        )

    deterministic_failures: list[dict[str, Any]] = []
    failure_context.update(stage="deterministic_scoring", indices=[])
    for _, line in data.iterrows():
        idx = str(line["index"])
        if idx in score_rows:
            continue
        res = _clean_optional(line.get("res"))
        function_id = _clean_optional(line.get("function_id"))
        try:
            if res is None or fail_msg in str(res):
                raise ValueError(
                    "extracted response is missing or contains the API failure sentinel"
                )
            if function_id is None:
                if str(line["question_type"]).lower() != "choice":
                    raise ValueError(
                        f"unexpected non-choice deterministic row: {line['question_type']}"
                    )
                function_id = "choice_function"
            function = eval_functions[str(function_id)]
            if function_id not in {"open_function", "choice_function"}:
                response = (
                    str(res)
                    if function_id == "judge_24points_function"
                    else json.loads(str(res))
                )
                answer = _safe_literal_eval(line.get("answer"))
                if function_id in {
                    "calculate_answer_function_hashi",
                    "calculate_answer_function_skyscraper",
                    "calculate_answer_function_sudoku_4",
                    "calculate_answer_function_sudoku_6",
                    "calculate_answer_function_yinyang",
                    "judge_24points_function",
                }:
                    raw_special_info = _clean_optional(line.get("special_info"))
                    if raw_special_info is None:
                        raise ValueError(
                            f"official evaluator {function_id} requires special_info"
                        )
                    special_info = _safe_literal_eval(raw_special_info)
                else:
                    special_info = None
            else:
                response = str(res)
                answer = line.get("answer")
                special_info = None
            answer_judge = (
                function(response, answer)
                if special_info is None
                else function(response, answer, special_info)
            )
            if type(answer_judge) is not bool:
                deterministic_failures.append(
                    {
                        "kind": "non_boolean_evaluator_result",
                        "index": idx,
                        "function_id": str(function_id),
                        "result_type": type(answer_judge).__name__,
                        "result_repr": repr(answer_judge),
                    }
                )
                continue
            score_rows[idx] = {
                "log_score": "Succeed",
                "score": answer_judge,
                "evaluation_status": "correct" if answer_judge else "incorrect",
            }
        except Exception as exc:
            deterministic_failures.append(
                {
                    "kind": "official_evaluator_exception",
                    "index": idx,
                    "function_id": (
                        str(function_id) if function_id is not None else None
                    ),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    if deterministic_failures:
        _fail_mme_evaluation(
            args,
            benchmark_output_dir,
            stage="deterministic_scoring",
            failures=deterministic_failures,
            message=f"MME-Reasoning official evaluator failed for {len(deterministic_failures)} rows",
        )

    data["score"] = [score_rows[str(idx)]["score"] for idx in data["index"]]
    data["evaluation_status"] = [
        score_rows[str(idx)]["evaluation_status"] for idx in data["index"]
    ]
    data["log_score"] = [score_rows[str(idx)]["log_score"] for idx in data["index"]]
    score_file = (
        benchmark_output_dir
        / f"{MME_REASONING_SPEC.alias}_predictions_qwen3_32b_score.xlsx"
    )

    acc_df = _mme_reasoning_acc(data)
    acc_csv = benchmark_output_dir / "MME-Reasoning_acc.csv"
    acc_xlsx = benchmark_output_dir / "MME-Reasoning_acc.xlsx"
    acc_df.to_csv(acc_csv, index=False)
    acc_df.to_excel(acc_xlsx, index=False)
    score_dict = {key: float(value) for key, value in acc_df.iloc[0].to_dict().items()}
    correct = sum(bool(row["score"]) for row in score_rows.values())
    incorrect = len(score_rows) - correct
    scores = {
        "dataset": MME_REASONING_SPEC.alias,
        "display": MME_REASONING_SPEC.display,
        "model": args.model,
        "model_slug": args.model_slug,
        "rows": int(len(data)),
        "accuracy": float(score_dict.get("Overall", 0.0)),
        "scores": score_dict,
        "evaluation": {
            "contract_version": MME_EVALUATION_CONTRACT_VERSION,
            "rows_scored": len(score_rows),
            "correct": correct,
            "incorrect": incorrect,
            "evaluator_invalid": 0,
        },
        "artifacts": {
            "predictions": str(
                output_dir / f"{MME_REASONING_SPEC.alias}_predictions.xlsx"
            ),
            "extracted": str(extract_file),
            "scored": str(score_file),
            "acc_csv": str(acc_csv),
        },
        "judge": {
            "model": args.judge_model,
            "api_model": args.judge_api_model,
            "api_bases": args.judge_api_bases,
            "temperature": MME_RETRY_TEMPERATURES[0],
            "temperature_schedule": list(MME_RETRY_TEMPERATURES),
            "extraction_retries": len(MME_RETRY_TEMPERATURES),
            "unresolved_extractions": 0,
            "unresolved_open_scores": 0,
            "cache_contract_versions": {
                "extraction": extraction_judge_contract,
                "open_scoring": open_judge_contract,
            },
        },
    }
    failure_context.update(stage="score_prepared", indices=[])
    return _PendingMMEScore(
        data=data,
        acc_df=acc_df,
        score_dict=score_dict,
        scores=scores,
        extraction_rows=extraction_rows,
        score_rows=score_rows,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score canonical trace_eval_v1 MME-Reasoning predictions."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score")
    score.add_argument("--model", required=True)
    score.add_argument("--model-slug", required=True)
    score.add_argument(
        "--seed", type=int, default=int(os.environ.get("TRACE_EVAL_SEED", "42"))
    )
    score.add_argument("--run-root", type=Path, required=True)
    score.add_argument("--benchmark-root", type=Path, required=True)
    score.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    score.add_argument("--judge-model", default="Qwen/Qwen3-32B")
    score.add_argument("--judge-batch-size", type=int, default=1024)
    score.add_argument("--judge-tensor-parallel-size", type=int, default=1)
    score.add_argument("--judge-gpu-memory-utilization", type=float, default=0.90)
    score.add_argument("--judge-max-model-len", type=int, default=8192)
    score.add_argument("--judge-max-num-seqs", type=int, default=128)
    score.add_argument("--judge-max-num-batched-tokens", type=int, default=32768)
    score.add_argument("--judge-max-tokens", type=int, default=256)
    score.add_argument("--judge-api-base", action="append", dest="judge_api_bases")
    score.add_argument("--judge-api-model", default="qwen3-32b-judge")
    score.add_argument("--judge-api-tokenizer-model", default="Qwen/Qwen3-32B")
    score.add_argument("--judge-api-parallelism", type=int, default=128)
    score.add_argument("--judge-api-batch-size", type=int, default=16)
    score.add_argument("--judge-api-batches-per-endpoint", type=int, default=1)
    score.add_argument("--judge-api-max-batch-chars", type=int, default=100_000)
    score.add_argument("--judge-api-timeout", type=float, default=120.0)
    score.add_argument("--judge-api-max-retries", type=int, default=5)
    score.add_argument(
        "--judge-cache-contract-version",
        default=PERSISTENT_JUDGE_CACHE_CONTRACT_VERSION,
    )
    score.add_argument("--attention-backend", default="FLASH_ATTN")
    score.add_argument("--extract-max-tokens", type=int, default=1024)
    score.add_argument("--score-max-tokens", type=int, default=16)
    score.add_argument("--no-resume", action="store_true")
    score.set_defaults(func=run_score)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("LMUData", str(EVALUATION_ROOT / ".work" / "LMUData"))
    Path(os.environ["LMUData"]).mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
