"""run_eval -> score -> compare through the real OpenAI client against a fake vLLM endpoint."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trace_tool import compare, run_eval, score_runs

from .conftest import make_image, png_bytes

ZOOM = '<tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [0, 0, 500, 500], "label": "x", "img_idx": 0}}\n</tool_call>'
# Qwen3.5 (requests carry chat_template_kwargs): the same call in the Qwen3 XML format
ZOOM_XML = (
    "<tool_call>\n<function=image_zoom_in_tool>\n<parameter=bbox_2d>\n[0, 0, 500, 500]\n</parameter>\n"
    "<parameter=label>\nx\n</parameter>\n<parameter=img_idx>\n0\n</parameter>\n</function>\n</tool_call>"
)
# What every request sends: the OpenAI fields plus vLLM's extra sampling fields.
HERMES_REQUEST_KEYS = {"model", "messages", "max_tokens", "temperature", "top_p", "presence_penalty", "seed", "top_k", "repetition_penalty"}


class FakeVLLM(BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeVLLM.requests.append(body)
        messages = body["messages"]
        has_tools = "# Tools" in messages[0]["content"]
        answered_tool = any(
            m["role"] == "user" and isinstance(m["content"], list) and m["content"][0].get("text", "").startswith("<tool_response>")
            for m in messages
        )
        # with tools: zoom once, then answer correctly; without tools: answer wrong
        if has_tools and not answered_tool:
            text = "Let me look closer.\n" + (ZOOM_XML if "chat_template_kwargs" in body else ZOOM)
        else:
            text = 'Done.\n{"answer": %d}' % (2 if has_tools else 1)
        payload = {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLM)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    FakeVLLM.requests = []
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def _dataset(path, n=6):
    contract = json.dumps({"annotation": {"id": "bbox_soft_iou_v0", "type": "bbox"}, "answer": {"id": "answer_exact_match_v0", "type": "integer"}, "reward_contract_version": "v0"})
    image = make_image(640, 480)
    table = pa.table(
        {
            "images": [[{"bytes": png_bytes(image), "path": None}] for _ in range(n)],
            "image_sizes": [[{"width": 640, "height": 480}] for _ in range(n)],
            "prompt_answer": ['<image>How many shapes?\nExample JSON:\n{"answer":1}'] * n,
            "answer_gt": [json.dumps({"type": "integer", "value": 2})] * n,
            "annotation_gt": [json.dumps({"type": "bbox", "value": [50, 50, 150, 120]})] * n,
            "reward_contract": [contract] * n,
            "instance_id": [f"id{i}" for i in range(n)],
            "domain": ["charts" if i % 2 else "games" for i in range(n)],
            "task": [f"task{i // 2}" for i in range(n)],
            "scene_id": ["s"] * n,
            "query_id": ["q"] * n,
        }
    )
    pq.write_table(table, path)


def test_pipeline(tmp_path, endpoint):
    data = tmp_path / "val.parquet"
    _dataset(data)
    out = tmp_path / "runs"
    common = ["--endpoint", endpoint, "--model", "fake", "--family", "qwen2.5-vl", "--dataset-path", str(data), "--out", str(out), "--concurrency", "4"]
    run_eval.main(common + ["--conditions", "no_tool,zoom", "--seeds", "42,43"])

    zoom_lines = (out / "zoom" / "seed42" / "responses.jsonl").read_text().splitlines()
    assert len(zoom_lines) == 6
    record = json.loads(zoom_lines[0])
    assert record["n_tool_calls"] == 1 and record["tool_calls"][0]["ok"] and record["turns"] == 2
    # Hermes families: request bodies carry no chat_template_kwargs (or anything else new)
    assert all(set(body) == HERMES_REQUEST_KEYS for body in FakeVLLM.requests)

    # resume is a no-op, and a changed setting is refused
    n_requests = len(FakeVLLM.requests)
    run_eval.main(common + ["--conditions", "no_tool,zoom", "--seeds", "42,43"])
    assert len(FakeVLLM.requests) == n_requests
    with pytest.raises(SystemExit):
        run_eval.main(common + ["--conditions", "zoom", "--seeds", "42", "--model", "other"])

    score_runs.main([str(out), "--dataset-path", str(data), "--report", str(out / "report.md")])
    zoom_summary = json.loads((out / "zoom" / "seed42" / "summary.json").read_text())
    base_summary = json.loads((out / "no_tool" / "seed42" / "summary.json").read_text())
    assert zoom_summary["official_acc"] == 1.0 and zoom_summary["tool_use_rate"] == 1.0
    assert base_summary["official_acc"] == 0.0
    assert "| fake | zoom | 42 | 6 | 100.00" in (out / "report.md").read_text()

    compare.main([str(out), "--baseline", "no_tool", "--out", str(out / "compare.md"), "--n-boot", "200"])
    report = (out / "compare.md").read_text()
    assert "| zoom | 42 | 6 | 0.00 | 100.00 | +100.00 |" in report and "Holm" in report
    assert "| zoom | 6 | 0.00 | 100.00 | +100.00 |" in report  # pooled over seeds


def test_dry_run_prints_prompt(tmp_path, capsys):
    data = tmp_path / "val.parquet"
    _dataset(data, n=2)
    run_eval.main(["--model", "fake", "--family", "qwen2.5-vl", "--dataset-path", str(data), "--out", str(tmp_path), "--conditions", "all", "--dry-run"])
    printed = capsys.readouterr().out
    assert '"name": "image_zoom_in_tool"' in printed and '"name": "ocr"' in printed and "<image 0>" in printed


def test_torn_write_and_endpoint_errors(tmp_path, endpoint):
    data = tmp_path / "val.parquet"
    _dataset(data)
    out = tmp_path / "runs"
    common = ["--endpoint", endpoint, "--model", "fake", "--family", "qwen2.5-vl", "--dataset-path", str(data), "--out", str(out), "--conditions", "no_tool"]
    run_eval.main(common + ["--limit", "4"])
    path = out / "no_tool" / "seed42" / "responses.jsonl"
    lines = path.read_text().splitlines()
    # simulate: one row ended in an endpoint error, then the process died mid-write
    errored = json.loads(lines[1])
    errored.update(stop_reason="error", error="APIConnectionError")
    path.write_text("\n".join([lines[0], json.dumps(errored), lines[2], lines[3]]) + "\n" + lines[3][:40])

    score_runs.main([str(out), "--dataset-path", str(data)])  # tolerant of the partial line
    summary = json.loads((out / "no_tool" / "seed42" / "summary.json").read_text())
    assert summary["n"] == 3 and summary["excluded_endpoint_errors"] == 1

    n_requests = len(FakeVLLM.requests)
    run_eval.main(common + ["--limit", "4"])  # repairs the tail and retries only the errored row
    assert len(FakeVLLM.requests) == n_requests + 1
    records = [json.loads(l) for l in path.read_text().splitlines()]  # every line parses again
    assert len(records) == 5
    score_runs.main([str(out), "--dataset-path", str(data)])
    summary = json.loads((out / "no_tool" / "seed42" / "summary.json").read_text())
    assert summary["n"] == 4 and summary["excluded_endpoint_errors"] == 0


def test_success_then_error_record_keeps_success(tmp_path, endpoint):
    data = tmp_path / "val.parquet"
    _dataset(data, n=2)
    out = tmp_path / "runs"
    common = ["--endpoint", endpoint, "--model", "fake", "--family", "qwen2.5-vl", "--dataset-path", str(data), "--out", str(out), "--conditions", "no_tool"]
    run_eval.main(common)
    path = out / "no_tool" / "seed42" / "responses.jsonl"
    first = json.loads(path.read_text().splitlines()[0])
    path.write_text(path.read_text() + json.dumps({**first, "stop_reason": "error", "error": "late failure"}) + "\n")
    score_runs.main([str(out), "--dataset-path", str(data)])
    summary = json.loads((out / "no_tool" / "seed42" / "summary.json").read_text())
    assert summary["n"] == 2 and summary["excluded_endpoint_errors"] == 0


def test_resume_rejects_different_model_id_and_bad_domains(tmp_path, endpoint):
    data = tmp_path / "val.parquet"
    _dataset(data, n=2)
    out = tmp_path / "runs"
    common = ["--endpoint", endpoint, "--model", "fake", "--family", "qwen2.5-vl", "--dataset-path", str(data), "--out", str(out), "--conditions", "no_tool"]
    run_eval.main(common + ["--model-id", "org/model-8b"])
    with pytest.raises(SystemExit):
        run_eval.main(common + ["--model-id", "org/model-4b"])
    with pytest.raises(SystemExit):
        run_eval.main(common + ["--domains", "chartz"])


def test_compare_refuses_duplicate_runs(tmp_path, endpoint):
    data = tmp_path / "val.parquet"
    _dataset(data, n=2)
    for root in ("a", "b"):
        run_eval.main(["--endpoint", endpoint, "--model", "fake", "--family", "qwen2.5-vl", "--dataset-path", str(data), "--out", str(tmp_path / root), "--conditions", "no_tool"])
    score_runs.main([str(tmp_path / "a"), str(tmp_path / "b"), "--dataset-path", str(data), "--pending-out", str(tmp_path / "p.jsonl")])
    with pytest.raises(SystemExit):
        compare.main([str(tmp_path)])


def test_judge_survives_rejected_item(tmp_path, monkeypatch):
    import openai

    from trace_tool import judge

    pending = tmp_path / "pending.jsonl"
    pending.write_text("\n".join(json.dumps({"key": k, "answer_type": "integer", "raw_response": "three"}) for k in ("a", "b")) + "\n")

    class Rejecting:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    # build without the HTTP response object (its library differs across openai versions)
                    error = openai.BadRequestError.__new__(openai.BadRequestError)
                    Exception.__init__(error, "maximum context length exceeded")
                    raise error

    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kw: Rejecting())
    out = tmp_path / "results.jsonl"
    judge.main(["--pending", str(pending), "--out", str(out)])
    statuses = sorted(json.loads(l)["status"] for l in out.read_text().splitlines())
    assert statuses == ["rejected", "rejected"]


def test_broken_ocr_stops_before_any_request(tmp_path, endpoint, monkeypatch):
    from trace_tool.tools import ocr

    def broken():
        raise ImportError("libGL.so.1: cannot open shared object file")

    monkeypatch.setitem(ocr._BACKENDS, "rapidocr", broken)
    monkeypatch.setattr(ocr, "_SHARED_BACKENDS", {})
    data = tmp_path / "val.parquet"
    _dataset(data, n=2)
    with pytest.raises(SystemExit, match="libGL"):
        run_eval.main(["--endpoint", endpoint, "--model", "fake", "--family", "qwen2.5-vl", "--dataset-path", str(data), "--out", str(tmp_path / "runs"), "--conditions", "no_tool,ocr"])
    assert FakeVLLM.requests == [] and not (tmp_path / "runs").exists()


def test_changed_dataset_file_is_refused_on_resume(tmp_path, endpoint):
    data = tmp_path / "val.parquet"
    _dataset(data, n=2)
    common = ["--endpoint", endpoint, "--model", "fake", "--family", "qwen2.5-vl", "--dataset-path", str(data), "--out", str(tmp_path / "runs"), "--conditions", "no_tool"]
    run_eval.main(common)
    config = json.loads((tmp_path / "runs" / "no_tool" / "seed42" / "run_config.json").read_text())
    assert len(config["dataset"]["sha256"]) == 64 and config["dataset"]["revision"] is None  # not the pinned file
    assert len(config["code_sha256"]) == 64
    _dataset(data, n=3)  # different file, same path
    with pytest.raises(SystemExit):
        run_eval.main(common)


def test_dry_run_needs_no_out_and_judge_needs_pending(tmp_path, capsys):
    from trace_tool import judge

    data = tmp_path / "val.parquet"
    _dataset(data, n=1)
    run_eval.main(["--model", "x", "--family", "qwen2.5-vl", "--dataset-path", str(data), "--conditions", "oracle_crop", "--dry-run"])
    assert "zoomed-in view" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        run_eval.main(["--model", "x", "--family", "qwen2.5-vl", "--dataset-path", str(data), "--conditions", "no_tool"])
    with pytest.raises(SystemExit, match="does not exist"):
        judge.main(["--pending", str(tmp_path / "missing.jsonl"), "--out", str(tmp_path / "out.jsonl")])


def test_qwen35_xml_pipeline(tmp_path, endpoint):
    data = tmp_path / "val.parquet"
    _dataset(data, n=4)
    out = tmp_path / "runs"
    common = ["--endpoint", endpoint, "--model", "qwen3.5-4b", "--model-id", "Qwen/Qwen3.5-4B", "--dataset-path", str(data), "--out", str(out)]
    run_eval.main(common + ["--conditions", "no_tool,zoom,zoom_forced"])

    # every Qwen3.5 request switches thinking off; nothing else is added
    assert FakeVLLM.requests and all(body["chat_template_kwargs"] == {"enable_thinking": False} for body in FakeVLLM.requests)
    assert all(set(body) == HERMES_REQUEST_KEYS | {"chat_template_kwargs"} for body in FakeVLLM.requests)
    tool_requests = [b for b in FakeVLLM.requests if "# Tools" in b["messages"][0]["content"]]
    assert all(b["messages"][0]["content"].startswith("# Tools\n\nYou have access to the following functions:") for b in tool_requests)

    for name in ("zoom", "zoom_forced"):
        records = [json.loads(line) for line in (out / name / "seed42" / "responses.jsonl").read_text().splitlines()]
        assert len(records) == 4
        for record in records:
            (call,) = record["tool_calls"]
            assert call["ok"] and call["arguments"] == {"bbox_2d": [0, 0, 500, 500], "label": "x", "img_idx": 0}
            assert record["turns"] == 2 and record["stop_reason"] == "answer"
            assert ("tool_calls_used" in record) == (name == "zoom_forced")
    forced = json.loads((out / "zoom_forced" / "seed42" / "responses.jsonl").read_text().splitlines()[0])
    assert {k: forced[k] for k in ("tool_call_budget", "tool_calls_used", "forced_turn", "forced_reason")} == {
        "tool_call_budget": 10, "tool_calls_used": 1, "forced_turn": False, "forced_reason": None
    }
    tool_turn = forced["transcript"][3]["content"]
    assert tool_turn[-1]["text"] == "\n\n[Tool calls left: 8]\n</tool_response>"
    assert "You can make at most 10 tool calls in total" in forced["transcript"][0]["content"]

    settings = {n: json.loads((out / n / "seed42" / "run_config.json").read_text())["settings"] for n in ("no_tool", "zoom", "zoom_forced")}
    assert all(s["family"] == "qwen3.5" and s["tool_format"] == "qwen3_xml" for s in settings.values())
    assert all(s["chat_template_kwargs"] == {"enable_thinking": False} for s in settings.values())
    assert "forced_answer" not in settings["zoom"] and settings["zoom_forced"]["forced_answer"] is True

    n_requests = len(FakeVLLM.requests)
    run_eval.main(common + ["--conditions", "no_tool,zoom,zoom_forced"])  # resume is a no-op
    assert len(FakeVLLM.requests) == n_requests
    score_runs.main([str(out), "--dataset-path", str(data)])
    assert json.loads((out / "zoom_forced" / "seed42" / "summary.json").read_text())["official_acc"] == 1.0


@pytest.mark.parametrize("family", ["qwen2.5-vl", "qwen3.5"])
def test_dry_run_shows_the_budget_statement(tmp_path, capsys, family):
    data = tmp_path / "val.parquet"
    _dataset(data, n=1)
    run_eval.main(["--model", "x", "--family", family, "--dataset-path", str(data), "--conditions", "zoom,zoom_forced", "--dry-run"])
    printed = capsys.readouterr().out
    statement = "You can make at most 10 tool calls in total, and at most 2 per turn."
    assert printed.count(statement) == 1
    zoom, forced = printed.split("===== zoom_forced")
    assert statement not in zoom and statement in forced.split("<|user|>")[0]  # in the system message
