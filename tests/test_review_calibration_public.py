from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from trace_tasks.review.calibration import (
    CalibrationConfig,
    _NoRedirectHandler,
    calibrate_review_root,
    discover_calibration_summaries,
)
from tests.test_review_app_public import _DIGEST_B, _materialize_one


def _sample(root: Path) -> str:
    task_id, _, _ = _materialize_one(root)
    return task_id


def test_portable_calibration_uses_answer_only_public_scoring_and_env_key(
    tmp_path: Path,
) -> None:
    root = tmp_path / "task-reviews"
    task_id = _sample(root)
    calls: list[tuple[str, Mapping[str, str], Mapping[str, Any], float]] = []

    def fake_transport(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        return {"choices": [{"message": {"content": '{"answer": 3}'}}]}

    output = tmp_path / "calibration" / "result.json"
    report = calibrate_review_root(
        review_root=root,
        config=CalibrationConfig(
            endpoint="https://model.example/v1",
            model="example/model",
            max_retries=0,
        ),
        output_path=output,
        task_ids=(task_id,),
        limit=1,
        transport=fake_transport,
        environ={"TRACE_REVIEW_API_KEY": "top-secret-value"},
    )

    assert report["summary"]["rollout_count"] == 1
    assert report["summary"]["successful_count"] == 1
    assert report["summary"]["error_count"] == 0
    assert report["summary"]["mean_answer_reward"] == 1.0
    assert report["summary"]["diagnostic_threshold_met"] is None
    assert report["results"][0]["answer_reward"] == 1.0
    assert report["endpoint"] == "https://model.example"
    assert calls[0][0] == "https://model.example/v1/chat/completions"
    assert calls[0][1]["Authorization"] == "Bearer top-secret-value"
    message = calls[0][2]["messages"][0]
    assert message["content"][0]["text"].startswith("Count the marked items")
    assert message["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert "top-secret-value" not in output.read_text(encoding="utf-8")
    summaries, errors = discover_calibration_summaries(root)
    assert errors == []
    assert summaries[0]["model"] == "example/model"
    assert summaries[0]["task_ids"] == [task_id]
    assert "results" not in summaries[0]
    assert "endpoint" not in summaries[0]


def test_calibration_retries_transport_and_records_errors(tmp_path: Path) -> None:
    root = tmp_path / "task-reviews"
    _sample(root)
    attempts = 0

    def failing_transport(url, headers, payload, timeout):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("offline top-secret-value /private/endpoint.sock")

    report = calibrate_review_root(
        review_root=root,
        config=CalibrationConfig(
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
            model="local/model",
            max_retries=1,
        ),
        output_path=tmp_path / "calibration" / "result.json",
        limit=1,
        transport=failing_transport,
        environ={"TRACE_REVIEW_API_KEY": "top-secret-value"},
    )
    assert attempts == 2
    assert report["summary"]["error_count"] == 1
    assert report["results"][0]["status"] == "error"
    assert "offline" in report["results"][0]["error"]
    assert "top-secret-value" not in report["results"][0]["error"]
    assert "/private/" not in report["results"][0]["error"]


def test_calibration_refuses_an_empty_materialization(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no materialized samples"):
        calibrate_review_root(
            review_root=tmp_path / "task-reviews",
            config=CalibrationConfig(
                endpoint="http://127.0.0.1:8000/v1", model="local/model"
            ),
            output_path=tmp_path / "calibration" / "result.json",
            transport=lambda *_args: {},
            environ={},
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "model.example/v1",
        "ftp://model.example/v1",
        "https://user:secret@model.example/v1",
    ],
)
def test_calibration_rejects_non_http_or_embedded_credentials(endpoint: str) -> None:
    with pytest.raises(ValueError):
        CalibrationConfig(endpoint=endpoint, model="model").validate()


def test_calibration_rejects_mixed_recipe_selection_before_network(
    tmp_path: Path,
) -> None:
    root = tmp_path / "task-reviews"
    _materialize_one(root, task_offset=0)
    _materialize_one(root, task_offset=1, recipe_digest=_DIGEST_B)
    called = False

    def transport(*_args):
        nonlocal called
        called = True
        return {}

    with pytest.raises(ValueError, match="share one canonical recipe"):
        calibrate_review_root(
            review_root=root,
            config=CalibrationConfig(endpoint="https://model.example/v1", model="m"),
            output_path=tmp_path / "calibration" / "mixed.json",
            transport=transport,
            environ={},
        )
    assert not called


def test_calibration_output_is_confined_to_review_workspace(tmp_path: Path) -> None:
    root = tmp_path / "review" / "task-reviews"
    _sample(root)
    with pytest.raises(ValueError, match="review/calibration"):
        calibrate_review_root(
            review_root=root,
            config=CalibrationConfig(endpoint="https://model.example/v1", model="m"),
            output_path=tmp_path / "outside.json",
            transport=lambda *_args: {},
            environ={},
        )


def test_endpoint_origin_is_idna_normalized_and_controls_are_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "task-reviews"
    _sample(root)
    report = calibrate_review_root(
        review_root=root,
        config=CalibrationConfig(endpoint="https://münich.example/v1", model="m"),
        output_path=tmp_path / "calibration" / "idna.json",
        limit=1,
        transport=lambda *_args: {
            "choices": [{"message": {"content": '{"answer": 3}'}}]
        },
        environ={},
    )
    assert report["endpoint"] == "https://xn--mnich-kva.example"
    with pytest.raises(ValueError, match="control"):
        CalibrationConfig(
            endpoint="https://model.example\n.evil/v1", model="m"
        ).validate()


def test_default_calibration_transport_disables_redirects() -> None:
    handler = _NoRedirectHandler()
    assert (
        handler.redirect_request(None, None, 302, "Found", {}, "https://other") is None
    )
