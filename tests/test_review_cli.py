from __future__ import annotations

import json

from trace_tasks.review.models import VerificationIssue, VerificationReport
from trace_tasks.review_cli import _build_parser, main


def test_review_cli_exposes_portable_recipe_and_calibration_controls() -> None:
    parser = _build_parser()

    capture = parser.parse_args(
        [
            "capture",
            "--recipe",
            "recipe",
            "--workers",
            "4",
            "--all",
        ]
    )
    assert capture.workers == 4
    assert capture.requests_per_task == 25

    verify = parser.parse_args(
        [
            "verify",
            "--recipe",
            "recipe",
            "--scene",
            "symbolic/abacus",
            "--query",
            "displayed_value",
        ]
    )
    assert verify.scene == ["symbolic/abacus"]
    assert verify.query == ["displayed_value"]

    audit = parser.parse_args(
        [
            "audit",
            "--recipe",
            "recipe",
            "--output",
            "review/task-reviews",
            "--task",
            "task_symbolic__abacus__displayed_value_readout",
        ]
    )
    assert audit.task == ["task_symbolic__abacus__displayed_value_readout"]

    serve = parser.parse_args(
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--trusted-host",
            "review-host.example.org",
            "--trusted-host",
            "192.0.2.10",
        ]
    )
    assert serve.trusted_host == ["review-host.example.org", "192.0.2.10"]

    calibration = parser.parse_args(
        [
            "calibrate",
            "--endpoint",
            "http://127.0.0.1:8000/v1",
            "--model",
            "local/model",
            "--output",
            "result.json",
            "--rollouts",
            "3",
            "--max-retries",
            "1",
            "--diagnostic-accuracy-threshold",
            "0.8",
        ]
    )
    assert calibration.rollouts == 3
    assert calibration.max_retries == 1
    assert calibration.diagnostic_accuracy_threshold == 0.8


def test_verify_cli_serializes_computed_report_fields_and_fails(
    monkeypatch, capsys
) -> None:
    report = VerificationReport(
        checked_requests=1,
        issues=(
            VerificationIssue(
                severity="error",
                code="semantic_hash_mismatch",
                message="changed",
            ),
        ),
    )
    monkeypatch.setattr(
        "trace_tasks.review.materialize.verify_recipe",
        lambda *args, **kwargs: report,
    )

    assert main(["verify", "--recipe", "missing-for-mocked-test"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error_count"] == 1
    assert payload["checked_requests"] == 1
