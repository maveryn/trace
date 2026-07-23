from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from rlvr.evaluation.trace_validation.answer_extraction import (
    ANSWER_EXTRACTION_CONTRACT_VERSION,
)
from rlvr.evaluation.trace_validation.judge_extract import (
    CONTRACT_VERSION,
    RECEIPT_SCHEMA_VERSION,
    SYSTEM_PROMPT,
    JudgeOutputError,
    _request_identity,
    _receipt_is_reusable,
    _validate_pending_row,
    _validate_receipt,
    parse_args,
    validate_judge_output,
)
from rlvr.evaluation.trace_validation import score


def test_accepts_grounded_option_extraction() -> None:
    response = "I considered B, but the final answer is C."
    output = '{"status":"ok","answer":"C",' '"evidence":"the final answer is C."}'
    assert validate_judge_output(
        output, raw_response=response, answer_type="option_letter"
    ) == {
        "status": "ok",
        "answer": "C",
        "evidence": "the final answer is C.",
    }


def test_tokenizer_remote_code_is_disabled_unless_explicitly_enabled() -> None:
    required = [
        "--pending-jsonl",
        "pending.jsonl",
        "--output-dir",
        "output",
        "--api-model",
        "judge-served",
        "--tokenizer-model",
        "Qwen/Judge",
        "--judge-revision",
        "a" * 40,
    ]

    assert parse_args(required).trust_remote_code is False
    assert parse_args([*required, "--trust-remote-code"]).trust_remote_code is True


def test_rejects_answer_not_supported_by_evidence() -> None:
    with pytest.raises(JudgeOutputError, match="does not contain"):
        validate_judge_output(
            '{"status":"ok","answer":"B","evidence":"Final answer: C."}',
            raw_response="Final answer: C.",
            answer_type="option_letter",
        )


@pytest.mark.parametrize(
    ("answer_type", "response", "output", "expected"),
    [
        (
            "integer",
            "Answer: -2.",
            '{"status":"ok","answer":-2,"evidence":"Answer: -2."}',
            -2,
        ),
        (
            "number",
            "Answer: 2.5e-1",
            '{"status":"ok","answer":0.25,"evidence":"Answer: 2.5e-1"}',
            0.25,
        ),
        (
            "string",
            "The answer is Cerulean Blue.",
            '{"status":"ok","answer":"cerulean blue",' '"evidence":"Cerulean Blue"}',
            "cerulean blue",
        ),
    ],
)
def test_accepts_supported_typed_values(
    answer_type: str, response: str, output: str, expected: object
) -> None:
    parsed = validate_judge_output(
        output, raw_response=response, answer_type=answer_type
    )
    assert parsed["answer"] == expected


def test_accepts_final_abstention() -> None:
    assert validate_judge_output(
        '{"status":"ambiguous","answer":null,"evidence":""}',
        raw_response="Either B or C.",
        answer_type="option_letter",
    ) == {"status": "ambiguous", "answer": None, "evidence": ""}


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            "```json\n" '{"status":"ok","answer":-2,"evidence":"Answer: -2."}' "\n```",
            {"status": "ok", "answer": -2, "evidence": "Answer: -2."},
        ),
        (
            "  \n```json\r\n"
            '{"status":"missing","answer":null,"evidence":""}'
            "\r\n```\n ",
            {"status": "missing", "answer": None, "evidence": ""},
        ),
    ],
)
def test_accepts_one_exact_json_fence(output: str, expected: dict[str, object]) -> None:
    assert (
        validate_judge_output(output, raw_response="Answer: -2.", answer_type="integer")
        == expected
    )


@pytest.mark.parametrize(
    "output",
    [
        '```\n{"status":"missing","answer":null,"evidence":""}\n```',
        '```JSON\n{"status":"missing","answer":null,"evidence":""}\n```',
        'Here is JSON:\n```json\n{"status":"missing","answer":null,"evidence":""}\n```',
        '```json\n{"status":"missing","answer":null,"evidence":""}\n```\nextra',
        '```json\n{"status":"missing","answer":null,"evidence":""}\n```\n```json\n{}\n```',
    ],
)
def test_rejects_nonexact_json_fence_envelopes(output: str) -> None:
    with pytest.raises(JudgeOutputError):
        validate_judge_output(
            output, raw_response="No final answer.", answer_type="integer"
        )


@pytest.mark.parametrize(
    "output",
    [
        '{"status":"ok","answer":2,"answer":3,"evidence":"3"}',
        '{"status":"missing","answer":null,"evidence":"x"}',
        '{"status":"ok","answer":true,"evidence":"true"}',
        '{"status":"ok","answer":2,"evidence":"2","extra":1}',
    ],
)
def test_rejects_invalid_contract_outputs(output: str) -> None:
    with pytest.raises(JudgeOutputError):
        validate_judge_output(
            output, raw_response="Final answer: 3", answer_type="integer"
        )


def test_rejects_nonfinite_json_and_string_substring_false_positive() -> None:
    with pytest.raises(JudgeOutputError, match="non-finite"):
        validate_judge_output(
            '{"status":"ok","answer":NaN,"evidence":"NaN"}',
            raw_response="Final answer: NaN",
            answer_type="number",
        )
    with pytest.raises(JudgeOutputError, match="does not contain"):
        validate_judge_output(
            '{"status":"ok","answer":"red","evidence":"credit"}',
            raw_response="Final answer: credit",
            answer_type="string",
        )


def test_option_contract_matches_deterministic_a_through_z() -> None:
    assert (
        validate_judge_output(
            '{"status":"ok","answer":"M","evidence":"Final answer: M"}',
            raw_response="Final answer: M",
            answer_type="option_letter",
        )["answer"]
        == "M"
    )


def _pending_row(*, row_index: int = 7) -> dict[str, object]:
    raw_response = "The response contains reasoning but states no answer."
    return {
        "model_slug": "model-a",
        "row_index": row_index,
        "instance_id": f"instance-{row_index}",
        "answer_type": "integer",
        "raw_response": raw_response,
        "raw_response_sha256": hashlib.sha256(raw_response.encode()).hexdigest(),
        "deterministic_status": "missing",
        "deterministic_extraction_version": ANSWER_EXTRACTION_CONTRACT_VERSION,
    }


def test_pending_contract_is_gt_blind_and_recomputed() -> None:
    row = _pending_row()
    _validate_pending_row(row, position=0)

    leaked = {**row, "answer_gt": {"type": "integer", "value": 3}}
    with pytest.raises(ValueError, match="GT-blind"):
        _validate_pending_row(leaked, position=0)

    forged = dict(row)
    forged["raw_response"] = "Final answer: 3"
    forged["raw_response_sha256"] = hashlib.sha256(
        str(forged["raw_response"]).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="deterministic status mismatch"):
        _validate_pending_row(forged, position=0)


def _identity_and_receipt(
    row: dict[str, object], *, status: str = "missing"
) -> tuple[dict[str, object], dict[str, object]]:
    identity = _request_identity(
        row,
        rendered_prompt="rendered prompt",
        api_model="judge-served",
        judge_revision="b" * 40,
        tokenizer_model="Qwen/Judge",
        tokenizer_template_sha256="c" * 64,
        retry_limits=[128, 256],
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "request_hash": identity["request_hash"],
        "request_contract": identity["request_contract"],
        "raw_response_sha256": identity["raw_response_sha256"],
        "rendered_prompt_sha256": identity["rendered_prompt_sha256"],
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "tokenizer_template_sha256": "c" * 64,
        "tokenizer_model": "Qwen/Judge",
        "judge_revision": "b" * 40,
        "judge_model": "judge-served",
        "model_slug": row["model_slug"],
        "row_index": row["row_index"],
        "instance_id": row["instance_id"],
        "answer_type": row["answer_type"],
        "deterministic_status": row["deterministic_status"],
        "deterministic_extraction_version": row["deterministic_extraction_version"],
        "judge_status": status,
        "answer": None,
        "evidence": "",
        "attempts": [
            {
                "retry_number": 0,
                "max_tokens": 128,
                "endpoint": "http://judge-0/v1",
                "raw_output": ('{"status":"missing","answer":null,"evidence":""}'),
                "finish_reason": "stop",
                "elapsed_seconds": 0.5,
                "validation_error": None,
            }
        ],
    }
    return identity, receipt


def _receipt_validation(
    row: dict[str, object], identity: dict[str, object]
) -> dict[str, object]:
    return {
        "row": row,
        "identity": identity,
        "api_model": "judge-served",
        "judge_revision": "b" * 40,
        "tokenizer_model": "Qwen/Judge",
        "tokenizer_template_sha256": "c" * 64,
    }


def test_request_cache_identity_is_row_bound() -> None:
    first = _pending_row(row_index=1)
    second = _pending_row(row_index=2)
    first_identity, _ = _identity_and_receipt(first)
    second_identity, _ = _identity_and_receipt(second)
    assert first["raw_response"] == second["raw_response"]
    assert first_identity["request_hash"] != second_identity["request_hash"]


def test_failed_receipts_are_not_reused_and_valid_receipts_are_revalidated(
    tmp_path: Path,
) -> None:
    row = _pending_row()
    identity, receipt = _identity_and_receipt(row)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    validation = _receipt_validation(row, identity)
    assert _receipt_is_reusable(path, **validation)

    failed_attempts = [
        {
            "retry_number": retry_number,
            "max_tokens": max_tokens,
            "endpoint": "http://judge-0/v1",
            "raw_output": "not JSON",
            "finish_reason": "length",
            "elapsed_seconds": 0.5,
            "validation_error": "JudgeOutputError: invalid JSON",
        }
        for retry_number, max_tokens in enumerate((128, 256))
    ]
    failed = {**receipt, "judge_status": "failed", "attempts": failed_attempts}
    path.write_text(json.dumps(failed), encoding="utf-8")
    assert not _receipt_is_reusable(path, **validation)
    _validate_receipt(failed, allow_failed=True, **validation)

    # A previously failed receipt must rerun even when its raw output is now
    # accepted after removing one exact JSON fence. Its stale validation_error
    # may not silently turn it into a cached success.
    fenced_output = (
        "```json\n" '{"status":"missing","answer":null,"evidence":""}' "\n```"
    )
    fenced_failed = {
        **failed,
        "attempts": [
            {
                **attempt,
                "raw_output": fenced_output,
                "validation_error": "JudgeOutputError: judge output: invalid JSON",
            }
            for attempt in failed_attempts
        ],
    }
    path.write_text(json.dumps(fenced_failed), encoding="utf-8")
    assert not _receipt_is_reusable(path, **validation)
    with pytest.raises(JudgeOutputError, match="valid attempt as a validation failure"):
        _validate_receipt(fenced_failed, allow_failed=True, **validation)

    stale = {**receipt, "instance_id": "different-instance"}
    path.write_text(json.dumps(stale), encoding="utf-8")
    assert not _receipt_is_reusable(path, **validation)


def test_score_aggregate_keeps_unresolved_rows_in_accuracy_denominator() -> None:
    base = {
        "model_slug": "model-a",
        "historical_answer_correct": 1,
        "historical_format_correct": 1,
        "terminal_rfc_json": 1,
        "historical_reward": 1.0,
        "deterministic_semantic_correct": 0,
        "combined_semantic_correct": 1,
        "deterministic_found": 0,
        "judge_requested": 1,
        "judge_resolved": 1,
        "unresolved": 0,
        "final_extraction_route": "judge",
    }
    unresolved = {
        **base,
        "historical_answer_correct": 0,
        "historical_reward": 0.0,
        "combined_semantic_correct": 0,
        "judge_resolved": 0,
        "unresolved": 1,
        "final_extraction_route": "judge:failed",
    }
    [aggregate] = score._aggregate([base, unresolved], ("model_slug",))
    assert aggregate["accuracy_denominator_rows"] == 2
    assert aggregate["combined_semantic_accuracy"] == 0.5
    assert aggregate["judge_requested_rows"] == 2
    assert aggregate["judge_resolved_rows"] == 1
    assert aggregate["judge_resolution_rate"] == 0.5
    assert aggregate["unresolved_rows"] == 1


def test_score_json_encoding_is_strict_and_summary_is_renderable() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        score._strict_json_loads('{"value": NaN}', source="fixture")
    with pytest.raises(ValueError, match="duplicate"):
        score._strict_json_loads('{"value": 1, "value": 2}', source="fixture")
    with pytest.raises(ValueError):
        score._canonical_json({"value": float("nan")})

    aggregate = {
        "model_slug": "model-a",
        "historical_answer_accuracy": 0.5,
        "historical_format_rate": 0.5,
        "deterministic_semantic_accuracy": 0.5,
        "combined_semantic_accuracy": 0.5,
        "judge_fallback_rate": 0.5,
        "unresolved_rate": 0.5,
    }
    assert "| Model A |" in score._summary_markdown([aggregate], {"model-a": "Model A"})


def test_suite_loader_requires_exact_eight_model_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite_path = Path("rlvr/evaluation/trace_validation/suite.v1.json")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    manifest_identity = score.dataset_prep.reproduction_dataset_identity()
    _, models, equivalence = score._load_suite(
        suite_path,
        dataset_identity=manifest_identity,
        dataset_manifest_sha256=score.dataset_prep.DATASET_MANIFEST_SHA256,
        dataset_equivalence_receipt=(score.dataset_prep.DATASET_EQUIVALENCE_RECEIPT),
    )
    assert len(models) == 8
    assert equivalence["receipt_sha256"] == (
        score.dataset_prep.DATASET_EQUIVALENCE_RECEIPT_SHA256
    )

    shortened = copy.deepcopy(suite)
    shortened["models"].pop()
    shortened_path = tmp_path / "suite.json"
    shortened_path.write_text(json.dumps(shortened), encoding="utf-8")
    monkeypatch.setattr(
        score.dataset_prep,
        "HISTORICAL_SUITE_SHA256",
        hashlib.sha256(shortened_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="exactly eight"):
        score._load_suite(
            shortened_path,
            dataset_identity=manifest_identity,
            dataset_manifest_sha256=score.dataset_prep.DATASET_MANIFEST_SHA256,
            dataset_equivalence_receipt=(
                score.dataset_prep.DATASET_EQUIVALENCE_RECEIPT
            ),
        )
