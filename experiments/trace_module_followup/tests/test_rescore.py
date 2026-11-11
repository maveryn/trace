"""format-normalized-v1 row rule and aggregation on synthetic rows.

Portable tests stub the pinned option parser; the tests marked ``pinned`` call the real pinned PuzzleVQA and
TableVQABench scorers and run only when the prepared protocol clone is importable (``study_env.sh``)."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_followup import extract as ex  # noqa: E402
from trace_followup import rescore as rs  # noqa: E402
from trace_merge import metrics as mt  # noqa: E402


def _pinned_available() -> bool:
    try:
        with rs._quiet():
            importlib.import_module("vlmeval.dataset.utils.puzzlevqa")
            importlib.import_module("vlmeval.dataset.utils.tablevqabench")
        return True
    except Exception:  # noqa: BLE001
        return False


pinned = pytest.mark.skipif(not _pinned_available(), reason="pinned VLMEvalKit scorers not importable")


def _rows(bench: str, texts: list[str], *, label: str = "M", seed: int = 42) -> pd.DataFrame:
    n = len(texts)
    return pd.DataFrame({
        "model_label": label, "seed": seed, "benchmark_id": bench,
        "content_key": [f"{bench}-{i}" for i in range(n)], "source_ordinal": np.arange(n),
        "source_index": [str(i) for i in range(n)], "source_row_sha256": [f"r{i}" for i in range(n)],
        "source_record_sha256": [f"c{i}" for i in range(n)], "archived_row_score": np.nan,
        "official_excluded": False, "extraction_status": "resolved",
        "extraction_value_json": [json.dumps(t) for t in texts], "judge_output": None,
    })


def _meta(n: int, **cols) -> pd.DataFrame:
    return pd.DataFrame({"source_ordinal": np.arange(n), "source_index": [str(i) for i in range(n)],
                         "source_row_sha256": [f"r{i}" for i in range(n)],
                         "source_record_sha256": [f"c{i}" for i in range(n)], **cols})


# ---------------------------------------------------------------------------------------------- portable (stub)
class _StubMetric:
    """Enough of a trace_merge metric for the MC rule: aligned licensed metadata."""

    def __init__(self, bench: str, meta: pd.DataFrame) -> None:
        self.benchmark_id = bench
        self.meta = meta.set_index("source_ordinal", drop=False)

    def aligned_meta(self, rows: pd.DataFrame) -> pd.DataFrame:
        m = self.meta.loc[rows["source_ordinal"].astype(int)].reset_index(drop=True)
        m.index = rows.index
        return m


def _stub_parser(monkeypatch):
    """A stand-in for the pinned parser: reads only a bare 'Answer: X' (A-D), else 'Z'."""

    import re

    def extract_answer(value):
        found = set(re.findall(r"\banswer\s*:\s*([A-D])\b", value if isinstance(value, str) else "", re.I))
        return found.pop().upper() if len(found) == 1 else "Z"

    mod = types.SimpleNamespace(extract_answer=extract_answer,
                                to_choice_letter=lambda options, answer: "ABCD"[list(options).index(answer)])
    monkeypatch.setattr(mt, "_pinned", lambda name: mod)


TEXTS = [
    "Rows alternate.\n\nAnswer: \\( C \\)",       # unread by the parser, gold option -> repaired
    "Answer: C",                                   # read by the parser, correct -> untouched
    "Answer: B",                                   # read by the parser, wrong -> untouched
    "Rows alternate.\n\nAnswer: (B)",              # unread, explicit wrong option -> stays 0
    "Rows alternate.\n\nThe answer is \\( A \\) or \\( C \\).",   # unread, two options -> unresolved
    "I think it is green.",                        # unread, no final answer -> unresolved
    "<think>The answer is C because",              # unread, reasoning never ended -> unresolved
]
OFFICIAL = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _mc_case(monkeypatch, answers=None):
    _stub_parser(monkeypatch)
    n = len(TEXTS)
    meta = _meta(n, options=["['red', 'blue', 'green', 'yellow']"] * n, answer=answers or ["green"] * n)
    rows = _rows("puzzlevqa", TEXTS).assign(official_row_score=OFFICIAL)
    return rs.normalized_row_scores(rows, {"puzzlevqa": _StubMetric("puzzlevqa", meta)})


def test_mc_rule_touches_only_unread_rows(monkeypatch):
    out = _mc_case(monkeypatch)
    assert out["fn_status"].tolist() == [rs.REPAIRED, rs.OFFICIAL_READ, rs.OFFICIAL_READ, rs.RESOLVED_WRONG,
                                         rs.UNRESOLVED, rs.UNRESOLVED, rs.UNRESOLVED]
    assert out["normalized_row_score"].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert out["fn_extracted"].tolist()[:4] == ["C", None, None, "B"]


def test_explicit_wrong_answers_stay_wrong(monkeypatch):
    out = _mc_case(monkeypatch)
    wrong = out[out["fn_extracted"].notna() & (out["fn_status"] != rs.REPAIRED)]
    assert len(wrong) == 1 and (wrong["normalized_row_score"] == 0).all()


def test_gold_only_scores_the_extraction(monkeypatch):
    """Changing the gold answer changes which extractions are correct, never what is extracted or which rows the
    rule examines."""

    a = _mc_case(monkeypatch, answers=["green"] * len(TEXTS))
    b = _mc_case(monkeypatch, answers=["blue"] * len(TEXTS))
    assert a["fn_extracted"].tolist() == b["fn_extracted"].tolist()
    examined = ~a["fn_status"].isin([rs.OUT, rs.OFFICIAL_READ])
    assert examined.tolist() == (~b["fn_status"].isin([rs.OUT, rs.OFFICIAL_READ])).tolist()
    assert b.loc[0, "normalized_row_score"] == 0.0 and b.loc[3, "normalized_row_score"] == 1.0


def test_out_of_scope_rows_unchanged(monkeypatch):
    rows = _rows("mmstar", ["Answer: \\( B \\)", "B"]).assign(official_row_score=[0.0, 1.0])
    out = rs.normalized_row_scores(rows, {"mmstar": object()})
    assert (out["fn_status"] == rs.OUT).all()
    assert out["normalized_row_score"].tolist() == [0.0, 1.0]


def test_rows_stay_in_denominator_and_never_lose_credit(monkeypatch):
    out = _mc_case(monkeypatch)
    assert len(out) == len(TEXTS)
    assert (out["normalized_row_score"] >= out["official_row_score"]).all()


def test_duplicate_rows_refused(monkeypatch):
    rows = _rows("mmstar", ["B", "B"]).assign(official_row_score=[1.0, 1.0], content_key="same")
    with pytest.raises(rs.RescoreError):
        rs.normalized_row_scores(rows, {"mmstar": object()})


def test_missing_columns_refused():
    with pytest.raises(rs.RescoreError):
        rs.normalized_row_scores(pd.DataFrame({"model_label": ["M"]}), {})


def test_tvqa_candidate_rules():
    assert rs.tvqa_candidate("5||", "vwtq") == ("5", ["T2_empty_items"], "")
    assert rs.tvqa_candidate("||Lisbon||", "vwtq_syn") == ("Lisbon", ["T2_empty_items"], "")
    assert rs.tvqa_candidate("5||", "fintabnetqa") == (None, [], "")          # T2 only on the WTQ splits
    assert rs.tvqa_candidate("Lisbon||Porto", "vwtq") == (None, [], "")       # a hedged list is never narrowed
    assert rs.tvqa_candidate("There are 7. \\boxed{7}", "vwtq") == ("7", ["T1_boxed"], "")
    assert rs.tvqa_candidate("\\boxed{7} or \\boxed{8}", "vwtq") == (None, [], "boxed answers disagree")
    assert rs.tvqa_candidate("\\boxed{5||}", "vwtq") == ("5", ["T1_boxed", "T2_empty_items"], "")
    assert rs.tvqa_candidate("Lisbon", "vwtq") == (None, [], "")
    # v2: T1 never applies on vtabfact (the pinned label rule, including "both labels = unparsed", stands)
    assert rs.tvqa_candidate("The statement is True. \\boxed{False}", "vtabfact") == (None, [], "")
    assert rs.tvqa_candidate("\\boxed{55\\%}", "fintabnetqa") == ("55%", ["T1_boxed"], "")


def test_score_column_metric_returns_stored_scores():
    class Base:
        kind = "binary_row"

        def check_rows(self, rows):
            self.checked = True

    base = Base()
    w = rs.ScoreColumnMetric(base, "normalized_row_score")
    rows = pd.DataFrame({"normalized_row_score": [0.0, 1.0]})
    assert w.row_scores(rows).tolist() == [0.0, 1.0] and base.checked and w.kind == "binary_row"


def test_macro_requires_all_benchmarks():
    vals = pd.DataFrame({"model_label": "M", "seed": 42, "benchmark_id": ["a", "b"], "value": [10.0, 20.0]})
    with pytest.raises(rs.RescoreError):
        rs.macro_values(vals)
    out = rs.macro_values(vals, n_suite=2)
    assert out.loc[out["seed"] == -1, "macro"].item() == 15.0


# ------------------------------------------------------------------------------------------ real pinned scorers
@pinned
def test_puzzlevqa_with_pinned_parser():
    n = len(TEXTS)
    meta = _meta(n, options=["['red', 'blue', 'green', 'yellow']"] * n, answer=["green"] * n,
                 category=["colors"] * n)
    metric = mt.get_metric("puzzlevqa", meta=meta)
    rows = _rows("puzzlevqa", TEXTS)
    out = rs.normalized_row_scores(rows, {"puzzlevqa": metric})
    assert out["official_row_score"].tolist() == OFFICIAL
    assert out["normalized_row_score"].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert out.loc[0, "fn_status"] == rs.REPAIRED and out.loc[3, "fn_status"] == rs.RESOLVED_WRONG
    vals = rs.benchmark_values(out, {"puzzlevqa": metric}, score_col="normalized_row_score")
    assert vals["value"].item() == pytest.approx(100 * 2 / n)
    assert vals["rows"].item() == n                       # unresolved rows stay in the denominator


TVQA = [  # (split, gold, prediction)
    ("fintabnetqa", "$7,066", "The loss was \\boxed{7066}."),   # T1 repairs both components
    ("fintabnetqa", "12.5%", "12.5%"),                           # full credit, untouched
    ("fintabnetqa", "55%", "The target is \\boxed{55\\%}."),     # T1 with the LaTeX escape removed (v2)
    ("vtabfact", "1", "True"),                                   # full credit, untouched
    ("vtabfact", "0", "\\boxed{True}"),                          # no rule on vtabfact, stays wrong
    ("vtabfact", "0", "The statement is True. So \\boxed{False}"),  # both labels: pinned 'unparsed' stands (v2)
    ("vwtq", "5", "5||"),                                        # T2 repairs
    ("vwtq", "Lisbon", "Lisbon||Porto"),                         # hedged list: no rule, stays wrong
    ("vwtq", "7", "\\boxed{7} or \\boxed{8}"),                   # conflicting boxes: unresolved
    ("vwtq_syn", "3", "There are 3 rows. \\boxed{3}"),           # T1 repairs
]


@pinned
def test_tablevqabench_rules_with_pinned_split_scorers():
    meta = _meta(len(TVQA), split=[t[0] for t in TVQA], answer=[t[1] for t in TVQA])
    metric = mt.get_metric("tablevqabench", meta=meta)
    rows = _rows("tablevqabench", [t[2] for t in TVQA])
    out = rs.normalized_row_scores(rows, {"tablevqabench": metric})
    assert out["official_row_score"].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert out["normalized_row_score"].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    assert out["fn_status"].tolist() == [rs.REPAIRED, rs.OFFICIAL_FULL, rs.REPAIRED, rs.OFFICIAL_FULL, rs.NO_RULE,
                                         rs.NO_RULE, rs.REPAIRED, rs.NO_RULE, rs.UNRESOLVED, rs.REPAIRED]
    # native aggregation: mean of the five pinned split scores
    vals = rs.benchmark_values(out, {"tablevqabench": metric}, score_col="normalized_row_score")
    assert vals["value"].item() == pytest.approx(np.mean([100.0, 100.0, 100 / 3, 100 / 3, 100.0]), abs=1e-6)
    off = rs.benchmark_values(out, {"tablevqabench": metric}, score_col="official_row_score")
    assert off["value"].item() == pytest.approx(np.mean([100 / 3, 100 / 3, 100 / 3, 0.0, 0.0]), abs=1e-6)


@pinned
def test_tablevqabench_failure_taxonomy_uses_rewritten_answer():
    """Audit taxonomy: a wrong boxed value is a wrong answer, a hedged list is a hedged list."""

    assert rs.tvqa_failure_type("There are 3. \\boxed{158}", "154", "vwtq", 0, 0, 0, 0, "stop") == "wrong_answer"
    assert rs.tvqa_failure_type("Lisbon||Porto", "Lisbon", "vwtq", 0, 0, 0, 0, "stop") == \
        "extra_values_including_gold"
    assert rs.tvqa_failure_type("Episode 7", "7", "vwtq", 0, 0, 0, 0, "stop") == "normalization_numeric_format"
    assert rs.tvqa_failure_type("-16.7", "$(16.7) million", "fintabnetqa", 0, 0, 0, 0, "stop") == \
        "normalization_numeric_format"
    assert rs.tvqa_failure_type("2.5%", "Three month rate of 2.5%", "fintabnetqa", 0, 0, 0, 0, "stop") == \
        "partial_answer"
    assert rs.tvqa_failure_type("5||", "5", "vwtq", 0, 0, 1, 0, "stop") == "extraction_fixed_empty_list_items"
    assert rs.tvqa_failure_type("x", "5", "vwtq", 1, 1, 1, 1, "stop") == "correct"


@pinned
def test_tablevqabench_fintab_partial_credit_kept():
    """fintabnetqa components are max(official, rewritten) separately; exact never exceeds relieved."""

    meta = _meta(1, split=["fintabnetqa"], answer=["$3,852 million"])
    metric = mt.get_metric("tablevqabench", meta=meta)
    rows = _rows("tablevqabench", ["3,852"])
    out = rs.normalized_row_scores(rows, {"tablevqabench": metric})
    assert out["official_row_score"].item() in (0.0, 0.5)
    assert out["normalized_row_score"].item() == out["official_row_score"].item()
