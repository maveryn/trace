"""trace-vero-followup fix-4 (MathVerse score stage) and fix-5 (ChartQAPro adapter) on synthetic inputs (design A1).

The fixes live in a private protocol overlay: a local clone of the shared protocol clone (af32fe0 plus its post-release
fixes) with fix-4 in run_external_benchmark_score_queue.py and fix-5 in run_official_vlmevalkit_saved_score.py. Its path
is machine-specific and comes from the environment: TRACE_FOLLOWUP_PROTOCOL_CLONE, else TVM_PROTOCOL_CLONE (as
dispatch/env_fix45.sh sets it). Every test skips when neither names a clone carrying both fix markers (the shared clone
itself, for instance). The canonical behaviour each fix replaces is exercised on the same inputs: af32fe0's ChartQAPro
adapter is read from the clone's own git HEAD (the shared clone's post-release diff leaves that file untouched), and the
judge's run_cached, which fix-4 does not change, is called the way af32fe0 + fix-3 calls it. The judge is a scripted
stand-in for the /v1/completions batch call; no network, GPU or benchmark content is used. Tests marked ``pinned`` also
need the pinned VLMEvalKit (the clone's external/VLMEvalKit) importable."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

FIX4 = "trace-vero-followup fix-4"
FIX5 = "trace-vero-followup fix-5"
AF32FE0 = "af32fe0e668eb878e909daef7322ac1c4ae20f64"
SCORE_QUEUE = "rlvr/evaluation/scripts/run_external_benchmark_score_queue.py"
SAVED_SCORE = "rlvr/evaluation/scripts/run_official_vlmevalkit_saved_score.py"


def _find_clone() -> Path | None:
    for var in ("TRACE_FOLLOWUP_PROTOCOL_CLONE", "TVM_PROTOCOL_CLONE"):
        value = os.environ.get(var)
        if not value:
            continue
        root = Path(value)
        try:
            if FIX4 in (root / SCORE_QUEUE).read_text(encoding="utf-8") and FIX5 in (root / SAVED_SCORE).read_text(
                encoding="utf-8"
            ):
                return root
        except OSError:
            continue
    return None


CLONE = _find_clone()
pytestmark = pytest.mark.skipif(
    CLONE is None, reason="no protocol clone with fix-4/fix-5 (TRACE_FOLLOWUP_PROTOCOL_CLONE / TVM_PROTOCOL_CLONE)"
)


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if CLONE is not None:
    _SCRIPTS = CLONE / "rlvr/evaluation/scripts"
    _VLMEVAL = CLONE / "external/VLMEvalKit"
    for _entry in (_VLMEVAL / "scripts", _VLMEVAL, CLONE / "src", _SCRIPTS):
        sys.path.insert(0, str(_entry))
    _DEPS = CLONE / "rlvr/evaluation/.work/eval_deps"
    if _DEPS.is_dir():
        sys.path.append(str(_DEPS))  # only for packages the interpreter lacks (anls); never ahead of it
    sq = _load("tvf_fix4_score_queue", _SCRIPTS / "run_external_benchmark_score_queue.py")
    saved = _load("tvf_fix5_saved_score", _SCRIPTS / "run_official_vlmevalkit_saved_score.py")


def _pinned_available() -> bool:
    if CLONE is None:
        return False
    try:
        importlib.import_module("vlmeval.dataset.utils.chartqapro")
        importlib.import_module("vlmeval.dataset.utils.mathverse")
        return True
    except Exception:  # noqa: BLE001
        return False


pinned = pytest.mark.skipif(not _pinned_available(), reason="pinned VLMEvalKit not importable")


def _git(*args: str) -> str:
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    return subprocess.run(["git", "-C", str(CLONE), *args], check=True, capture_output=True, text=True, env=env).stdout


@pytest.fixture(scope="module")
def canonical_saved(tmp_path_factory: pytest.TempPathFactory) -> types.ModuleType:
    """af32fe0's run_official_vlmevalkit_saved_score.py (the shared clone's version: its post-release diff skips it)."""
    assert _git("rev-parse", "HEAD").strip() == AF32FE0
    source = _git("show", f"HEAD:{SAVED_SCORE}")
    assert FIX5 not in source and "def _chartqapro_final_sentence" in source
    path = tmp_path_factory.mktemp("canonical") / "canonical_saved_score.py"
    path.write_text(source, encoding="utf-8")
    return _load("tvf_canonical_saved_score", path)


# ============================================================================== fix-5: ChartQAPro adapter

# (synthetic response, af32fe0 result or IndexError, fix-5 result)
FIX5_CASES = [
    ("Reasoning first. The answer is 42.", "42", "42"),
    ("The answer is: **7**", "7", "7"),
    ("The answer is 3\nThe answer is 4", "4", "4"),
    ("no final-answer marker here", None, None),
    ("", None, None),
    ("The answer is\nmore text after a line break", None, None),
    ("The answer is\n", None, None),
    ("<answer>The answer is 5</answer>", "5", "5"),
    ("<answer>\\boxed{3}</answer>", "3", "3"),
    ("<answer>x</answer><answer>y</answer>", None, None),
    ("<answer>plain text</answer>", "plain text", "plain text"),
    # the raising rows: nothing after the last marker
    ("Some reasoning, so the answer is", IndexError, ""),
    ("The answer is: ", IndexError, ""),
    ("The answer is:\t", IndexError, ""),
    ("The answer is 1. Wait, the answer is", IndexError, ""),
    ("</think>\n<answer\nThe answer is", IndexError, ""),  # an unclosed tag is not an answer block
    ("<answer>The answer is</answer>", IndexError, ""),
    ("<answer>\nThe answer is:  \n</answer>", IndexError, ""),
]


@pytest.mark.parametrize("text, before, after", FIX5_CASES)
def test_fix5_changes_only_texts_on_which_af32fe0_raised(canonical_saved, text, before, after):
    assert saved._chartqapro_final_answer(text) == after
    if before is IndexError:
        with pytest.raises(IndexError):
            canonical_saved._chartqapro_final_answer(text)
    else:
        assert canonical_saved._chartqapro_final_answer(text) == before == after


def test_fix5_random_texts_agree_with_af32fe0_wherever_it_did_not_raise(canonical_saved):
    pieces = ["The answer is", "the answer is:", "THE ANSWER IS ", "the  answer\tis", "x", "42", "**7**", "\n", " ",
              "\t", ".", ":", "<answer>", "</answer>", "<answer", "\\boxed{3}", "\\boxed{", "}", "is", "`a`", "\r\n"]
    rng = random.Random(20261109)
    raised = 0
    for _ in range(20000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 8)))
        after = saved._chartqapro_final_answer(text)
        try:
            before = canonical_saved._chartqapro_final_answer(text)
        except IndexError:
            raised += 1
            assert after == "", repr(text)
            continue
        assert after == before, repr(text)
        assert after != "", repr(text)  # "" only ever comes from fix-5
    assert raised > 100  # the generator reaches the raising branch


def _chartqapro_table(path: Path) -> None:
    pd.DataFrame(
        {
            "index": [11, 12, 13, 14],
            "prediction": ["work... The answer is 9", "work... the answer is", "no marker", "<answer>The answer is</answer>"],
        }
    ).to_excel(path, index=False)


def test_fix5_adapter_records_empty_final_answers(tmp_path, canonical_saved):
    table = tmp_path / "ChartQAPro_CoT.xlsx"
    _chartqapro_table(table)
    with pytest.raises(IndexError):  # af32fe0 fails the whole benchmark on this table
        canonical_saved._adapt_chartqapro_prediction(table, pd.read_excel)

    _chartqapro_table(table)
    record = saved._adapt_chartqapro_prediction(table, pd.read_excel)
    assert record["contract"] == "chartqapro-official-final-answer-v2+trace-vero-followup-fix-5"
    assert record["empty_final_answer_rows"] == 2
    assert record["empty_final_answer_indices"] == ["12", "14"]
    assert record["changed_rows"] == 3 and record["unresolved_rows"] == 1 and record["rows"] == 4
    adapted = pd.read_excel(table, keep_default_na=False)
    assert adapted["prediction"].astype(str).tolist() == ["9", "", "no marker", ""]
    assert adapted["raw_prediction"].tolist()[1] == "work... the answer is"


@pinned
def test_fix5_empty_final_answer_is_scored_by_the_pinned_scorer(tmp_path):
    """The pinned scorer reads the adapted workbook back as VLMEvalKit's load does; an empty final answer is then the
    same input as an empty response (an empty cell), and it is scored, not an error."""
    from vlmeval.dataset.utils.chartqapro import evaluate_predictions_chartqapro

    table = tmp_path / "ChartQAPro_CoT.xlsx"
    _chartqapro_table(table)
    saved._adapt_chartqapro_prediction(table, pd.read_excel)
    data = pd.read_excel(table)  # vlmeval.smp.load: pd.read_excel
    gold = {11: "9", 12: "12", 13: "red", 14: "blue"}
    records = [
        {"Answer": [gold[int(row["index"])]], "Question Type": "Factoid", "Year": ["NO"], "prediction": row["prediction"]}
        for _, row in data.iterrows()
    ]
    scores = evaluate_predictions_chartqapro(records)
    assert scores["Overall"] == pytest.approx(0.25)  # only the explicit "9" is correct


# ============================================================================== fix-4: MathVerse score stage

TRUNCATED = "Let me compare the two answers step by step"


def _behaviour(marker: str, temperature: float, max_tokens: int) -> tuple[str, str]:
    """Scripted judge output for one score request of a synthetic row."""
    if marker == "ROW_OK":
        return "1", "stop"
    if marker == "ROW_LONG_THEN_OK":  # resolved inside the token-limit retries: unchanged by fix-4
        return (TRUNCATED, "length") if max_tokens < 256 else ("1", "stop")
    if marker == "ROW_TRUNC_ONCE":  # incomplete at every token limit of attempt 1, decided in attempt 2
        return (TRUNCATED, "length") if temperature == 0.0 else ("0", "stop")
    if marker == "ROW_TRUNC_ALWAYS":
        return TRUNCATED, "length"
    if marker == "ROW_EMPTY_ALWAYS":
        return "", "stop"
    if marker == "ROW_INVALID":  # complete but without a 0/1 decision: fix-3, as before
        return "Judgement: maybe", "stop"
    raise AssertionError(marker)


class ScriptedJudgeBatch:
    """Stand-in for PersistentJudge._call_api_completion_batch (one /v1/completions batch)."""

    def __init__(self, extract_outputs: dict[str, tuple[str, str]] | None = None, score_behaviour=None):
        self.extract_outputs = extract_outputs or {}
        self.score_behaviour = score_behaviour or _behaviour
        self.calls: list[tuple[str, str, float, int]] = []

    def __call__(self, start_endpoint, prompts, *, max_tokens, temperature, top_p):
        rows = []
        for prompt in prompts:
            marker = re.search(r"ROW_[A-Z_]+", prompt).group(0)
            stage = "extract" if "Model response: '" in prompt else "score"
            self.calls.append((stage, marker, float(temperature), int(max_tokens)))
            if stage == "extract":
                output, reason = self.extract_outputs.get(marker, (f"extracted {marker}", "stop"))
            else:
                output, reason = self.score_behaviour(marker, float(temperature), int(max_tokens))
            rows.append({"judge_output": output, "judge_finish_reason": reason,
                         "judge_output_token_count": max_tokens if reason == "length" else 1})
        return rows

    def score_calls(self, marker: str) -> list[tuple[float, int]]:
        return [(t, m) for stage, mk, t, m in self.calls if stage == "score" and mk == marker]


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        judge_api_bases=["http://judge.invalid/v1"], judge_api_batch_size=64, judge_api_parallelism=1,
        judge_api_batches_per_endpoint=1, judge_api_max_batch_chars=200000, judge_api_model="qwen3-32b-judge",
        judge_model="synthetic-judge", judge_max_tokens=256, no_resume=False, eval_judge_model=None, eval_nproc=1,
        gpu=None, judge_tensor_parallel_size=1, judge_gpu_memory_utilization=0.9, judge_max_model_len=8192,
        judge_max_num_seqs=1, judge_max_num_batched_tokens=1, judge_batch_size=1, attention_backend=None,
    )


def _judge(batch: ScriptedJudgeBatch):
    judge = sq.PersistentJudge(_args())
    judge.api_chat_prompt = lambda prompt, *, system_prompt=None: prompt
    judge._call_api_completion_batch = batch
    return judge


@pytest.mark.parametrize("marker", ["ROW_TRUNC_ALWAYS", "ROW_EMPTY_ALWAYS"])
def test_af32fe0_judge_call_raises_on_an_output_still_incomplete(tmp_path, marker):
    """The failure fix-4 handles: run_cached as af32fe0 + fix-3's score loop calls it (no return_unresolved)."""
    judge = _judge(ScriptedJudgeBatch())
    with pytest.raises(RuntimeError, match="incomplete or parse-invalid output after retries"):
        judge.run_cached(
            output_dir=tmp_path, prompts=[("7", f"[Model_answer] : {marker}")], cache_name="s.jsonl", max_tokens=16,
            temperature=0.0, top_p=1.0, desc="score", output_validator=sq._nonempty_judge_output, contract_version="v",
        )


SCORE_ROWS = ["ROW_OK", "ROW_LONG_THEN_OK", "ROW_TRUNC_ONCE", "ROW_TRUNC_ALWAYS", "ROW_EMPTY_ALWAYS", "ROW_INVALID"]


def _jsonl_by_index(path):
    path = Path(path)
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[str(row["index"])] = row
    return rows


def _append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _score_loop(monkeypatch, output_dir: Path, batch: ScriptedJudgeBatch) -> dict[str, dict[str, Any]]:
    """fix-4's loop inside _run_local_math_like, with a runner whose MathVerse judge only runs the score stage."""
    prompts = [(str(i), f"[Model_answer] : {marker}") for i, marker in enumerate(SCORE_ROWS)]
    runner = types.SimpleNamespace(load_jsonl_by_index=_jsonl_by_index, append_jsonl=_append_jsonl)
    runner._run_local_text_judge = None

    def run_mathverse_local_judge(ns):
        return {"scored": runner._run_local_text_judge(
            ns, prompts=prompts, cache_name="mathverse_qwen3_32b_score.jsonl", max_tokens=16)}

    runner.run_mathverse_local_judge = run_mathverse_local_judge
    monkeypatch.setattr(sq, "_import_vlmeval_runner", lambda: (runner, None))
    monkeypatch.setattr(sq, "_normalize_mathverse_binary_judgement_summary", lambda spec, model, out, summary: summary)
    monkeypatch.setattr(sq, "_validate_math_like_judged_table", lambda spec, out: None)
    spec = sq.spec_by_key("mathverse")
    return sq._run_local_math_like(_args(), spec, "synthetic-model", output_dir, _judge(batch))["scored"]


def test_fix4_incomplete_output_is_a_failed_attempt(monkeypatch, tmp_path):
    batch = ScriptedJudgeBatch()
    scored = _score_loop(monkeypatch, tmp_path, batch)
    by_marker = {marker: scored[str(i)] for i, marker in enumerate(SCORE_ROWS)}

    # complete outputs: exactly as before (one request, or the token-limit retries inside attempt 1)
    assert by_marker["ROW_OK"]["judge_output"] == "1" and batch.score_calls("ROW_OK") == [(0.0, 128)]
    assert by_marker["ROW_LONG_THEN_OK"]["judge_output"] == "1"
    assert batch.score_calls("ROW_LONG_THEN_OK") == [(0.0, 128), (0.0, 256)]
    assert by_marker["ROW_INVALID"]["judge_resolution"] == "all_retries_failed_scored_incorrect"

    # incomplete after the token-limit retries: a failed attempt; the next attempt runs at fix-3's next temperature
    limits = [128, 256, 512, 1024]
    assert batch.score_calls("ROW_TRUNC_ONCE") == [(0.0, m) for m in limits] + [(0.5, 128)]
    assert by_marker["ROW_TRUNC_ONCE"]["judge_output"] == "0" and "judge_resolution" not in by_marker["ROW_TRUNC_ONCE"]
    temperatures = [0.0, 0.5, 1.0, 1.5, 2.0]
    for marker in ("ROW_TRUNC_ALWAYS", "ROW_EMPTY_ALWAYS"):
        assert batch.score_calls(marker) == [(t, m) for t in temperatures for m in limits]
        row = by_marker[marker]  # no 0/1 decision after 5 attempts: scored incorrect by fix-3's existing record
        assert row["judge_output"] == "0" and row["judge_resolution"] == "all_retries_failed_scored_incorrect"
    assert by_marker["ROW_TRUNC_ALWAYS"]["judge_output_raw"] == TRUNCATED

    unresolved = json.loads((tmp_path / "mathverse_score_unresolved.json").read_text())
    assert unresolved["fix"] == "trace-cert post-release fix-3" and unresolved["resolution"] == "scored_incorrect"
    assert unresolved["indices"] == ["3", "4", "5"] and unresolved["count"] == 3
    assert [a["incomplete_after_token_retries"] for a in unresolved["attempts"]["3"]] == [True] * 5
    assert [a["judge_finish_reason"] for a in unresolved["attempts"]["3"]] == ["length"] * 5
    assert [a["incomplete_after_token_retries"] for a in unresolved["attempts"]["5"]] == [False] * 5

    incomplete = json.loads((tmp_path / "mathverse_score_incomplete_attempts.json").read_text())
    assert incomplete["fix"] == FIX4 and incomplete["indices"] == ["2", "3", "4"] and incomplete["count"] == 3
    assert incomplete["incomplete_attempts"] == {"2": [1], "3": [1, 2, 3, 4, 5], "4": [1, 2, 3, 4, 5]}
    assert incomplete["resolved_in_later_attempt"] == ["2"]
    assert incomplete["contract_version"].endswith("+trace-cert-fix-3+trace-vero-followup-fix-4")

    cache = _jsonl_by_index(tmp_path / "mathverse_qwen3_32b_score.jsonl")
    assert cache["3"]["judge_finish_reason"] == "length"  # the failed attempt is cached, not raised
    assert cache["0"]["contract_version"].endswith("+trace-vero-followup-fix-4-attempt1")

    # a resumed scoring makes no new requests and rebuilds the same records
    calls = len(batch.calls)
    assert _score_loop(monkeypatch, tmp_path, batch) == scored and len(batch.calls) == calls
    assert json.loads((tmp_path / "mathverse_score_incomplete_attempts.json").read_text()) == incomplete


def test_fix4_writes_no_record_when_every_attempt_completes(monkeypatch, tmp_path):
    monkeypatch.setattr(sys.modules[__name__], "SCORE_ROWS", ["ROW_OK", "ROW_LONG_THEN_OK"])
    scored = _score_loop(monkeypatch, tmp_path, ScriptedJudgeBatch())
    assert {row["judge_output"] for row in scored.values()} == {"1"}
    assert not (tmp_path / "mathverse_score_incomplete_attempts.json").exists()
    assert not (tmp_path / "mathverse_score_unresolved.json").exists()


# ------------------------------------------------------------------------------ fix-4 through the pinned runner


def _mathverse_table(output_dir: Path, markers: list[str]) -> str:
    spec = sq.spec_by_key("mathverse")
    meta = json.dumps({"subject": "synthetic subject", "subfield": "synthetic subfield"})
    pd.DataFrame(
        {
            "index": list(range(len(markers))),
            "question_for_eval": [f"synthetic question {m}" for m in markers],
            "answer": [f"gold {m}" for m in markers],
            "prediction": [f"synthetic response {m}" for m in markers],
            "metadata": [meta] * len(markers),
            "problem_version": ["synthetic version"] * len(markers),
        }
    ).to_excel(output_dir / f"{spec.alias}_predictions.xlsx", index=False)
    return spec.alias


@pinned
def test_fix4_end_to_end_through_the_pinned_mathverse_runner(tmp_path):
    markers = [*SCORE_ROWS, "ROW_PREFETCH", "ROW_LONG_EXTRACT"]
    _mathverse_table(tmp_path, markers)

    def behaviour(marker, temperature, max_tokens):
        return ("1", "stop") if marker == "ROW_LONG_EXTRACT" else _behaviour(marker, temperature, max_tokens)

    batch = ScriptedJudgeBatch(
        extract_outputs={
            "ROW_PREFETCH": ("gold ROW_PREFETCH", "stop"),  # equals the answer: prefetched, never judged
            # the official extraction route does not retry on length: a truncated extraction is used as it is
            "ROW_LONG_EXTRACT": ("extracted ROW_LONG_EXTRACT and then more", "length"),
        },
        score_behaviour=behaviour,
    )
    spec = sq.spec_by_key("mathverse")
    summary = sq._run_local_math_like(_args(), spec, "synthetic-model", tmp_path, _judge(batch))

    art = summary["artifacts"]
    assert art["mathverse_all_retries_failed_scored_incorrect_indices"] == ["3", "4", "5"]
    assert art["mathverse_incomplete_score_attempt_rows"] == 3
    assert art["mathverse_incomplete_score_attempt_indices"] == ["2", "3", "4"]
    judged = pd.read_excel(tmp_path / f"{spec.alias}_judged_qwen3_32b.xlsx", keep_default_na=False)
    log = dict(zip(judged["index"].astype(str), judged["log_score"].astype(str)))
    assert log == {"0": "Judge output: 1", "1": "Judge output: 1", "2": "Judge output: 0", "3": "Judge output: 0",
                   "4": "Judge output: 0", "5": "Judge output: 0", "6": "Prefetch succeed", "7": "Judge output: 1"}
    assert summary["score"] == pytest.approx(100 * 4 / 8)
    long_extract = [call for call in batch.calls if call[:2] == ("extract", "ROW_LONG_EXTRACT")]
    assert long_extract == [("extract", "ROW_LONG_EXTRACT", 0.0, 128)]
    scores = json.loads((tmp_path / "scores.json").read_text())
    assert scores["artifacts"]["mathverse_incomplete_score_attempt_rows"] == 3


@pinned
def test_fix4_leaves_the_extract_stage_fail_closed(tmp_path):
    """Not broadened: an extraction still empty after the token-limit retries fails the slice as in af32fe0."""
    _mathverse_table(tmp_path, ["ROW_OK"])
    batch = ScriptedJudgeBatch(extract_outputs={"ROW_OK": ("", "stop")})
    with pytest.raises(RuntimeError, match="incomplete or parse-invalid output after retries"):
        sq._run_local_math_like(_args(), sq.spec_by_key("mathverse"), "synthetic-model", tmp_path, _judge(batch))
    assert not (tmp_path / "mathverse_score_incomplete_attempts.json").exists()


@pinned
def test_fix4_normalizer_rejects_a_record_that_does_not_match(tmp_path):
    """scores.json counts the fix-4 rows only when the record names rows of the judged table (checked before scoring)."""
    spec = sq.spec_by_key("mathverse")
    meta = json.dumps({"subject": "s", "subfield": "f"})
    pd.DataFrame({"index": [0, 1], "extract": ["a", "b"], "score": [True, False],
                  "log_score": ["Judge output: 1", "Judge output: 0"], "metadata": [meta] * 2,
                  "problem_version": ["v"] * 2}).to_excel(tmp_path / f"{spec.alias}_judged_qwen3_32b.xlsx", index=False)
    (tmp_path / "mathverse_score_incomplete_attempts.json").write_text(json.dumps({"count": 1, "indices": ["9"]}))
    with pytest.raises(RuntimeError, match="does not match the judged table"):
        sq._normalize_mathverse_binary_judgement_summary(spec, "m", tmp_path, {"artifacts": {}})
    (tmp_path / "mathverse_score_incomplete_attempts.json").write_text(json.dumps({"count": 1, "indices": ["1"]}))
    summary = sq._normalize_mathverse_binary_judgement_summary(spec, "m", tmp_path, {"artifacts": {}})
    assert summary["artifacts"]["mathverse_incomplete_score_attempt_indices"] == ["1"]
    assert math.isclose(summary["score"], 50.0)


# ------------------------------------------------------------------------------------------------ fix-6 (archive)
ARCHIVE_LIB = "rlvr/evaluation/scripts/trace_eval_hf_archive_lib.py"
FIX6 = "trace-vero-followup fix-6"


def _archive_lib() -> types.ModuleType:
    text = (CLONE / ARCHIVE_LIB).read_text(encoding="utf-8")
    if FIX6 not in text:
        pytest.skip("clone without fix-6")
    return _load("_fu_archive_lib_fix6", CLONE / ARCHIVE_LIB)


def _extraction(value: Any) -> dict[str, Any]:
    return {"model_response": "reasoning ... final", "judge_prompt": "", "judge_response": "",
            "normalized_extraction": {"status": "resolved", "value": value, "method": "adapter", "candidates": []},
            "retries": {"count": 0}}


def test_fix6_extracted_answer_that_looks_like_a_file_name_is_archived():
    lib = _archive_lib()
    paths = lib.OPAQUE_TEXT_RECORD_PATHS["extraction"]
    for value in ("C.png", "option_b.jpg", "figure/answer.png"):
        assert lib._is_media_path(value)          # the canonical sanitizer raised on these
        out = lib.sanitize_archive_value(_extraction(value), path="extraction_records[0]",
                                         opaque_text_keys=lib.OPAQUE_TEXT_RECORD_KEYS["extraction"],
                                         opaque_text_paths=paths)
        assert out["normalized_extraction"]["value"] == value   # kept verbatim, nothing else changes


def test_fix6_leaves_every_other_check_strict():
    lib = _archive_lib()
    kw = {"opaque_text_keys": lib.OPAQUE_TEXT_RECORD_KEYS["extraction"],
          "opaque_text_paths": lib.OPAQUE_TEXT_RECORD_PATHS["extraction"]}
    rec = _extraction("B")
    rec["normalized_extraction"]["candidates"] = ["answer.png"]          # a nested value: still checked
    with pytest.raises(lib.ArchiveValidationError):
        lib.sanitize_archive_value(rec, path="extraction_records[0]", **kw)
    rec = _extraction({"nested": "x.png"})                                 # only a scalar string is exempt
    with pytest.raises(lib.ArchiveValidationError):
        lib.sanitize_archive_value(rec, path="extraction_records[0]", **kw)
    rec = _extraction("B")
    rec["image_path"] = "x.png"                                            # media-named fields stay forbidden
    with pytest.raises(lib.ArchiveValidationError):
        lib.sanitize_archive_value(rec, path="extraction_records[0]", **kw)
    # generation and score contracts are unchanged
    assert lib.OPAQUE_TEXT_RECORD_PATHS["generation"] == frozenset()
    assert lib.OPAQUE_TEXT_RECORD_PATHS["score"] == frozenset()
