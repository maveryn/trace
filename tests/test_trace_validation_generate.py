from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

from rlvr.evaluation.trace_validation import generate
from rlvr.evaluation.trace_validation import prepare_dataset as prepare


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


class FakeSession:
    def __init__(self, *, fail_first: bool = False, permanent_error: bool = False):
        self.fail_first = fail_first
        self.permanent_error = permanent_error
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def post(self, url, *, headers, json, timeout):
        with self._lock:
            call_index = len(self.calls)
            self.calls.append(
                {"url": url, "headers": headers, "json": json, "timeout": timeout}
            )
        if self.permanent_error:
            return FakeResponse(400, {"error": "bad request"})
        if self.fail_first and call_index == 0:
            return FakeResponse(503, {"error": "busy"})
        return FakeResponse(
            200,
            {
                "id": f"response-{call_index}",
                "model": json["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f'reasoning {call_index}\n{{"answer": {call_index}}}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )


def _png_bytes(color: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _fixture_manifest(root: Path, count: int = 2) -> Path:
    rows = []
    for index in range(count):
        payload = _png_bytes("red" if index % 2 == 0 else "blue")
        digest = prepare.sha256_bytes(payload)
        relative_path = f"media/sha256/{digest[:2]}/{digest}.png"
        image_path = root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(payload)
        rows.append(
            {
                "row_index": index,
                "instance_id": f"instance-{index}",
                "task": "task_fixture__scene__objective",
                "domain": "fixture",
                "answer_type": "string",
                "prompt_answer": f"<image>question {index}",
                "answer_gt": {"type": "string", "value": f"SECRET_GT_{index}"},
                "images": [
                    {
                        "image_index": 0,
                        "sha256": digest,
                        "size_bytes": len(payload),
                        "width": 8,
                        "height": 6,
                        "format": "PNG",
                        "mime_type": "image/png",
                        "relative_path": relative_path,
                    }
                ],
            }
        )
    manifest = {
        "schema_version": prepare.MANIFEST_SCHEMA,
        "dataset": {
            "repo_id": "fixture/trace",
            "revision": "a" * 40,
            "config": "default",
            "split": "validation",
            "file": "data/validation.parquet",
            "file_sha256": "b" * 64,
            "file_size_bytes": 123,
            "row_count": count,
        },
        "media": {
            "storage": prepare.MEDIA_STORAGE,
            "paths_relative_to": prepare.MANIFEST_NAME,
            "reencoded": False,
            "resized": False,
        },
        "rows": rows,
    }
    path = root / prepare.MANIFEST_NAME
    prepare.atomic_write_json(path, manifest)
    return path


def _run(
    tmp_path: Path,
    session: FakeSession,
    *,
    count: int = 2,
    shard_index: int = 0,
    shard_count: int = 1,
):
    manifest_path = _fixture_manifest(tmp_path / "prepared", count=count)
    model_path = tmp_path / "model"
    model_path.mkdir(exist_ok=True)
    with mock.patch.object(generate, "_http_session", return_value=session):
        metadata = generate.run_generation(
            manifest_path=manifest_path,
            output_dir=tmp_path / "outputs",
            endpoint_url="http://127.0.0.1:8123/v1",
            served_model="fixture-served-model",
            model_slug="fixture-model",
            model_path=model_path,
            model_revision="c" * 40,
            shard_index=shard_index,
            shard_count=shard_count,
            concurrency=2,
            max_attempts=3,
            retry_backoff_seconds=0,
            request_timeout_seconds=12,
            progress_every=0,
            expected_rows=count,
            require_pinned_manifest=False,
        )
    return metadata, manifest_path


def test_generation_uses_exact_chat_contract_and_resumes(tmp_path: Path):
    session = FakeSession()
    metadata, manifest_path = _run(tmp_path, session)

    assert metadata["status"] == "complete"
    assert metadata["completed_rows"] == 2
    assert metadata["error_rows"] == 0
    assert metadata["model_revision"] == "c" * 40
    assert metadata["decoding"] == generate.DECODING_CONTRACT
    assert len(session.calls) == 2
    expected_system = generate.SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    for call in session.calls:
        request = call["json"]
        assert call["url"] == "http://127.0.0.1:8123/v1/chat/completions"
        assert "Authorization" not in call["headers"]
        assert request["messages"][0] == {"role": "system", "content": expected_system}
        assert {key: request[key] for key in generate.DECODING_CONTRACT} == (
            generate.DECODING_CONTRACT
        )
        assert request["top_k"] == -1
        assert request["repetition_penalty"] == 1.0
        assert request["frequency_penalty"] == 0.0
        assert request["mm_processor_kwargs"] == {
            "min_pixels": 262_144,
            "max_pixels": 4_194_304,
        }
        assert request["messages"][1]["content"][0]["type"] == "image_url"
        assert request["messages"][1]["content"][0]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )
        encoded_request = json.dumps(request)
        assert "answer_gt" not in encoded_request
        assert "SECRET_GT" not in encoded_request

    responses_path = Path(metadata["responses_path"])
    records = [json.loads(line) for line in responses_path.read_text().splitlines()]
    assert [record["row_index"] for record in records] == [0, 1]
    assert all(record["status"] == "complete" for record in records)
    assert all(record["model_slug"] == "fixture-model" for record in records)
    assert all(record["request_hash"] for record in records)
    assert all(record["response_sha256"] for record in records)
    assert all(record["raw_response"] for record in records)
    assert metadata["mm_processor_kwargs"] == generate.MM_PROCESSOR_CONTRACT
    assert metadata["run_contract"]["mm_processor_kwargs"] == (
        generate.MM_PROCESSOR_CONTRACT
    )

    # A second identical invocation verifies and reuses receipts without API work.
    with mock.patch.object(generate, "_http_session", return_value=session):
        second = generate.run_generation(
            manifest_path=manifest_path,
            output_dir=tmp_path / "outputs",
            endpoint_url="http://127.0.0.1:8123/v1",
            served_model="fixture-served-model",
            model_slug="fixture-model",
            model_path=tmp_path / "model",
            model_revision="c" * 40,
            concurrency=2,
            retry_backoff_seconds=0,
            progress_every=0,
            expected_rows=2,
            require_pinned_manifest=False,
        )
    assert second["status"] == "complete"
    assert len(session.calls) == 2


def test_generation_retries_transient_endpoint_failure(tmp_path: Path):
    session = FakeSession(fail_first=True)
    metadata, _ = _run(tmp_path, session, count=1)

    assert metadata["status"] == "complete"
    assert len(session.calls) == 2
    record = json.loads(Path(metadata["responses_path"]).read_text())
    assert record["attempt_count"] == 2
    assert record["attempt_history"][0]["status_code"] == 503
    assert record["attempt_history"][0]["retryable"] is True


def test_generation_keeps_failed_rows_and_marks_run_incomplete(tmp_path: Path):
    session = FakeSession(permanent_error=True)
    metadata, _ = _run(tmp_path, session, count=1)

    assert metadata["status"] == "incomplete"
    assert metadata["completed_rows"] == 0
    assert metadata["error_rows"] == 1
    records = Path(metadata["responses_path"]).read_text().splitlines()
    assert len(records) == 1
    record = json.loads(records[0])
    assert record["row_index"] == 0
    assert record["status"] == "error"
    assert record["raw_response"] is None
    assert record["request_hash"]
    assert record["response_sha256"] is None


def test_modulo_shards_use_isolated_canonical_outputs(tmp_path: Path):
    session = FakeSession()
    metadata, _ = _run(tmp_path, session, count=4, shard_index=1, shard_count=2)

    responses_path = Path(metadata["responses_path"])
    assert responses_path == (
        tmp_path.resolve() / "outputs" / "shards" / "00001-of-00002" / "responses.jsonl"
    )
    records = [json.loads(line) for line in responses_path.read_text().splitlines()]
    assert [record["row_index"] for record in records] == [1, 3]
    assert [record["instance_id"] for record in records] == ["instance-1", "instance-3"]


def test_file_url_transport_enforces_allowed_root(tmp_path: Path):
    manifest_path = _fixture_manifest(tmp_path / "prepared", count=1)
    manifest = prepare.load_manifest(
        manifest_path,
        expected_rows=1,
        require_pinned=False,
    )
    system_prompt, _ = generate._load_system_prompt()
    outside_root = tmp_path / "other"
    outside_root.mkdir()

    with pytest.raises(ValueError, match="outside allowed media root"):
        generate.build_request_payload(
            manifest["rows"][0],
            manifest_path,
            served_model="fixture",
            system_prompt=system_prompt,
            media_transport="file-url",
            allowed_local_media_root=outside_root,
        )


def test_api_key_environment_requires_explicit_opt_in(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    required = [
        "--manifest",
        "manifest.json",
        "--output-dir",
        "output",
        "--endpoint-url",
        "http://localhost:8000/v1",
        "--served-model",
        "served",
        "--model-slug",
        "model",
        "--model-path",
        "weights",
        "--model-revision",
        "a" * 40,
    ]

    default_args = generate._build_parser().parse_args(required)
    assert default_args.api_key_env is None
    assert generate._api_key_from_environment(default_args.api_key_env) is None

    opted_in = generate._build_parser().parse_args(
        [*required, "--api-key-env", "OPENAI_API_KEY"]
    )
    assert generate._api_key_from_environment(opted_in.api_key_env) == "ambient-secret"

    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ValueError, match="unset or empty"):
        generate._api_key_from_environment("OPENAI_API_KEY")


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.org/v1/chat/completions",
        "http://10.0.0.5:8000/v1/chat/completions",
    ],
)
def test_authenticated_plain_http_is_rejected_off_loopback(endpoint: str):
    with pytest.raises(ValueError, match="plain HTTP"):
        generate._validate_endpoint_auth(endpoint, "secret")


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://localhost:8000/v1/chat/completions",
        "http://127.0.0.1:8000/v1/chat/completions",
        "http://[::1]:8000/v1/chat/completions",
        "https://example.org/v1/chat/completions",
    ],
)
def test_authenticated_transport_allows_loopback_http_or_https(endpoint: str):
    generate._validate_endpoint_auth(endpoint, "secret")


def test_post_adds_authorization_only_for_explicit_key():
    session = FakeSession()
    with mock.patch.object(generate, "_http_session", return_value=session):
        generate._post_once(
            completion_url="http://127.0.0.1:8000/v1/chat/completions",
            payload={"model": "fixture"},
            api_key="secret",
            request_timeout_seconds=1,
        )
    assert session.calls[0]["headers"]["Authorization"] == "Bearer secret"
