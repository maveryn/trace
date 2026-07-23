from __future__ import annotations

import inspect
import json
import math

import pytest

from rlvr.evaluation.trace_validation.answer_extraction import (
    ANSWER_EXTRACTION_CONTRACT_VERSION,
    extract_answer,
)


def test_extractor_contract_is_versioned_and_ground_truth_blind() -> None:
    signature = inspect.signature(extract_answer)
    assert list(signature.parameters) == ["response", "answer_type"]
    assert "ground" not in str(signature).lower()
    assert "score" not in str(signature).lower()

    result = extract_answer('{"answer": 4}', "integer")
    assert result.version == ANSWER_EXTRACTION_CONTRACT_VERSION
    assert result.as_dict()["candidate_provenance"]


def test_terminal_json_overrides_prior_explicit_candidates() -> None:
    result = extract_answer(
        'Answer: 3\nIntermediate guess: \\boxed{3}\n{"answer": 4}',
        "integer",
    )

    assert result.status == "found"
    assert result.route == "balanced_object"
    assert result.raw_candidate == 4
    assert result.typed_candidate == 4
    assert {candidate.priority for candidate in result.candidate_provenance} == {
        "explicit",
        "terminal_structured",
    }


def test_prose_apostrophe_does_not_hide_terminal_structured_answer() -> None:
    result = extract_answer('I don\'t trust the first count.\n{"answer": 6}', "integer")
    assert result.status == "found"
    assert result.typed_candidate == 6


@pytest.mark.parametrize(
    ("response", "answer_type", "expected", "route"),
    [
        ("Final JSON:\n{'answer': '4'}", "integer", 4, "balanced_object"),
        ('<answer>"4"</answer>', "integer", 4, "answer_tag"),
        ('```json\n{"answer": -2.5}\n```', "number", -2.5, "code_fence"),
        ("Reasoning.\n\\boxed{-3}", "integer", -3, "boxed"),
        ("Final answer: 6.25e-2", "number", 0.0625, "final_answer"),
    ],
)
def test_structured_and_explicit_routes_are_typed(
    response: str,
    answer_type: str,
    expected: object,
    route: str,
) -> None:
    result = extract_answer(response, answer_type)  # type: ignore[arg-type]
    assert result.status == "found"
    assert result.route == route
    assert result.typed_candidate == expected


def test_quoted_numeric_is_coerced_but_string_numeric_is_preserved() -> None:
    integer = extract_answer('{"answer": "4"}', "integer")
    string = extract_answer('{"answer": "4"}', "string")
    explicit_string = extract_answer("Final answer: 4", "string")

    assert integer.status == "found"
    assert integer.raw_candidate == "4"
    assert integer.typed_candidate == 4
    assert type(integer.typed_candidate) is int
    assert string.status == "found"
    assert string.raw_candidate == "4"
    assert string.typed_candidate == "4"
    assert type(string.typed_candidate) is str
    assert explicit_string.status == "found"
    assert explicit_string.typed_candidate == "4"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"answer": -12}', -12),
        ('{"answer": -0.125}', -0.125),
        ('{"answer": "1.2e3"}', 1200),
        ("Final answer: .5", 0.5),
    ],
)
def test_negative_decimal_and_scientific_numbers(
    response: str, expected: int | float
) -> None:
    result = extract_answer(response, "number")
    assert result.status == "found"
    assert result.typed_candidate == expected


def test_option_letter_accepts_terminal_period_without_reading_choices() -> None:
    result = extract_answer("After checking each option:\nAnswer: C.", "option_letter")

    assert result.status == "found"
    assert result.typed_candidate == "C"
    assert all(
        candidate.typed_candidate == "C" for candidate in result.candidate_provenance
    )

    bare = extract_answer("After checking each option, I select:\nC.", "option_letter")
    assert bare.status == "found"
    assert bare.route == "terminal_option"
    assert bare.typed_candidate == "C"


def test_conflicting_explicit_payloads_are_ambiguous() -> None:
    result = extract_answer("Answer: 4\n\\boxed{5}", "integer")

    assert result.status == "ambiguous"
    assert result.route == "ambiguous"
    assert result.raw_candidate is None
    assert result.typed_candidate is None
    assert {candidate.typed_candidate for candidate in result.candidate_provenance} == {
        4,
        5,
    }


def test_conflicting_structured_payloads_are_ambiguous_when_none_is_terminal() -> None:
    result = extract_answer(
        '{"answer": "A"}\nThen I reconsidered.\n<answer>B</answer>\nTrailing note.',
        "option_letter",
    )

    assert result.status == "ambiguous"
    assert {candidate.typed_candidate for candidate in result.candidate_provenance} == {
        "A",
        "B",
    }


def test_canonical_equal_duplicates_are_not_ambiguous() -> None:
    result = extract_answer("Answer: 4\n\\boxed{4.0}", "number")

    assert result.status == "found"
    assert result.typed_candidate == 4
    assert len(result.candidate_provenance) == 2


@pytest.mark.parametrize(
    "response",
    [
        "I counted 2 triangles, then 3 more, so there are 5 altogether.",
        "No explicit answer is supplied here.",
        "",
        None,
        '{"answer": 4',
        '```json\n{"answer": 4\n```',
        "{'answer': __import__('os').system('echo unsafe')}",
    ],
)
def test_missing_and_malformed_responses_do_not_guess(response: str | None) -> None:
    result = extract_answer(response, "integer")
    assert result.status == "missing"
    assert result.typed_candidate is None


def test_malformed_json_is_not_reinterpreted_as_a_string_answer() -> None:
    for response in (
        'Final JSON:\n{"answer": "four"',
        '<answer>{"answer": "four"</answer>',
        '```json\n{"answer": "four"\n```',
    ):
        result = extract_answer(response, "string")
        assert result.status == "missing"


@pytest.mark.parametrize(
    "response",
    [
        '{"answer": true}',
        "{'answer': False}",
        '{"answer": NaN}',
        '{"answer": Infinity}',
        '{"answer": "nan"}',
        '{"answer": "1e9999"}',
    ],
)
def test_boolean_and_nonfinite_numbers_are_rejected(response: str) -> None:
    result = extract_answer(response, "number")
    assert result.status == "missing"
    assert result.typed_candidate is None
    assert not any(
        candidate.accepted
        and isinstance(candidate.typed_candidate, float)
        and not math.isfinite(candidate.typed_candidate)
        for candidate in result.candidate_provenance
    )


def test_rejected_terminal_structured_answer_does_not_revive_an_earlier_guess() -> None:
    result = extract_answer('Answer: 4\n{"answer": true}', "integer")
    assert result.status == "missing"
    assert result.typed_candidate is None
    assert any(
        candidate.priority == "terminal_structured" and not candidate.accepted
        for candidate in result.candidate_provenance
    )


def test_option_parser_does_not_extract_letters_from_reasoning() -> None:
    result = extract_answer(
        "A is tempting, while B and C are also plausible.", "option_letter"
    )
    assert result.status == "missing"


def test_string_type_rejects_non_string_structured_values() -> None:
    result = extract_answer('{"answer": 4}', "string")
    assert result.status == "missing"
    assert result.candidate_provenance[0].rejection_reason == "string_value_required"


def test_duplicate_json_keys_are_not_silently_selected() -> None:
    result = extract_answer('{"answer": 4, "answer": 5}', "integer")
    assert result.status == "missing"


@pytest.mark.parametrize(
    "response",
    [
        "{'answer': 4, 'answer': 5}",
        "{'answer': 4, \"answer\": 5}",
        "{'answer': 4, **{'answer': 5}}",
    ],
)
def test_duplicate_python_literal_keys_are_not_silently_selected(
    response: str,
) -> None:
    result = extract_answer(response, "integer")
    assert result.status == "missing"
    assert result.typed_candidate is None


def test_rejected_literal_provenance_is_json_serializable() -> None:
    result = extract_answer("{'answer': {2, 1}}", "integer")
    assert result.status == "missing"
    payload = result.as_dict()
    assert payload["candidate_provenance"][0]["raw_candidate"] == [1, 2]
    json.dumps(payload, allow_nan=False)


def test_unknown_answer_type_fails_closed() -> None:
    with pytest.raises(ValueError, match="answer_type"):
        extract_answer('{"answer": 4}', "count")  # type: ignore[arg-type]
