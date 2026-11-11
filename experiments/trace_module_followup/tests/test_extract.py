"""format-normalized-v1 extractor: harmless wrappers resolve, conflicts stay unresolved, reasoning never supplies an
answer, and the gold answer is not an input. All responses are synthetic (tests/fixtures/extract_cases.json)."""

from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_followup import extract as ex  # noqa: E402

CASES = json.loads((Path(__file__).parent / "fixtures" / "extract_cases.json").read_text(encoding="utf-8"))


def _ids(group: str) -> list[str]:
    return [c["id"] for c in CASES[group]]


@pytest.mark.parametrize("case", CASES["wrapped_final_options"], ids=_ids("wrapped_final_options"))
def test_harmless_wrappers_resolve(case):
    x = ex.extract_final_option(case["response"], case["labels"])
    assert x.status == ex.RESOLVED, x
    assert x.option == case["option"]


@pytest.mark.parametrize("case", CASES["conflicting_answers"], ids=_ids("conflicting_answers"))
def test_conflicting_answers_stay_unresolved(case):
    x = ex.extract_final_option(case["response"], case["labels"])
    assert x.status == case["status"], x
    assert x.option is None


@pytest.mark.parametrize("case", CASES["no_single_final_option"], ids=_ids("no_single_final_option"))
def test_no_single_final_option(case):
    x = ex.extract_final_option(case["response"], case["labels"])
    assert x.status == case["status"], x
    assert x.option is None


@pytest.mark.parametrize("case", CASES["reasoning_does_not_leak"], ids=_ids("reasoning_does_not_leak"))
def test_reasoning_never_supplies_the_answer(case):
    x = ex.extract_final_option(case["response"], case["labels"])
    assert x.status == case["status"], x
    assert x.option == case.get("option")


@pytest.mark.parametrize("case", CASES["boxed_free_form"], ids=_ids("boxed_free_form"))
def test_boxed_free_form(case):
    x = ex.extract_final_boxed(case["response"])
    assert x.status == case["status"], x
    assert x.option == case.get("value")


def test_gold_is_not_an_input():
    """The extractors' only inputs are the response and (for options) the allowed labels; the module never names a
    gold/answer/score field."""

    assert list(inspect.signature(ex.extract_final_option).parameters) == ["response", "labels"]
    assert list(inspect.signature(ex.extract_final_boxed).parameters) == ["response"]
    tree = ast.parse(Path(ex.__file__).read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
        {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)} | \
        {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not {n for n in names if any(k in n.lower() for k in ("gold", "correct_option", "score", "truth"))}


def test_result_independent_of_any_answer_key():
    """Every letter that appears in the reasoning yields the same extraction: the result is a function of the final
    region only, never of which letter would be correct."""

    base = "Tile A has three sides, tile B four, tile C five and tile D six.\n\n"
    for final, letter in (("Answer: \\( B \\)", "B"), ("Answer: (D)", "D"), ("\\boxed{C}", "C"), ("Answer: $A$", "A")):
        x = ex.extract_final_option(base + final, "ABCD")
        assert (x.status, x.option) == (ex.RESOLVED, letter)
    assert ex.extract_final_option(base + "Hard to say.", "ABCD").status == ex.NO_FINAL_ANSWER


def test_explicit_wrong_letter_is_returned_as_written():
    """A clearly designated option is returned as written; correctness is decided later against gold, so an explicit
    wrong option can only ever score 0."""

    x = ex.extract_final_option("The missing colour is red.\n\nAnswer: \\( C \\)", "ABCD")
    assert (x.status, x.option) == (ex.RESOLVED, "C")


def test_labels_order_and_case_do_not_matter():
    r = "Rows add up.\n\nAnswer: $B$"
    assert ex.extract_final_option(r, "abcd").option == ex.extract_final_option(r, ["D", "C", "B", "A"]).option == "B"


def test_bad_labels_rejected():
    with pytest.raises(ValueError):
        ex.extract_final_option("B", ["AB"])
    with pytest.raises(ValueError):
        ex.extract_final_option("B", [])


def test_non_string_response_is_no_answer():
    assert ex.extract_final_option(None, "ABCD").status == ex.NO_FINAL_ANSWER   # type: ignore[arg-type]
    assert ex.extract_final_boxed(float("nan")).status == ex.NO_FINAL_ANSWER    # type: ignore[arg-type]


def test_regions():
    assert ex.final_regions("<think>x</think><answer>B</answer>")[0] == "answer_tag"
    assert ex.final_regions("<think>x</think>\n<answer>\nB")[0] == "answer_tag_open"
    assert ex.final_regions("<think>x</think>\nB")[0] == "after_think"
    assert ex.final_regions("<think>x")[0] == "unterminated_think"
    kind, spans = ex.final_regions("one\n\ntwo\n\n")
    assert kind == "final_paragraph" and "one\n\ntwo\n\n"[spans[0][0]:spans[0][1]].strip() == "two"
    # the WeMath-style "<Answer>:" template line is not an answer tag
    assert ex.final_regions("<Answer>: <<C>>")[0] == "final_paragraph"
