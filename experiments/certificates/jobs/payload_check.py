"""Check the exact request trace-tool sends for the certificate conditions (DESIGN 4, leakage test 5).

Runs in trace-tool's own environment (its ``.venv``; no ``trace_tasks`` import), CPU only:

    python payload_check.py --config <rendered trace_tool_conditions_v1.yaml> --model-id Qwen/Qwen3.5-4B \
        --case ans_v1=<public/e1_dev.ans.parquet>=<expected system prompt file> \
        --case ann_v1=<public/e1_dev.ann.parquet>=<cert_system_v1.txt> \
        --case cert_v1=<public/e1_dev.cert_rel1000.parquet>=<cert_system_v1.txt>

For one row of every task in each parquet it runs trace-tool's ``run_episode`` against a fake
OpenAI client that records the ``chat.completions.create`` call, then asserts that the request is
exactly: one system message (the expected file's stripped text), one user message holding the
image and the row's prompt text (``prompt_answer`` minus ``<image>``), max_tokens 8192, the TRACE
decoding preset, the per-condition seed, and ``chat_template_kwargs={"enable_thinking": false}``
for Qwen3.5 only. No tools, no other keys, no label values, no instance or extension ids.
Exit 0 when every check passes; prints one JSON summary line per condition.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pyarrow.parquet as pq
from PIL import Image

from trace_tool.agent import OpenAIChatBackend, run_episode
from trace_tool.conditions import build_episode_config, describe, load_conditions
from trace_tool.data import rows_from_parquet
from trace_tool.models import chat_template_kwargs, detect_family

SEED = 42
REQUEST_KEYS = {"model", "messages", "max_tokens", "temperature", "top_p", "presence_penalty", "seed", "extra_body"}
TRACE_DECODING = {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "repetition_penalty": 1.0, "presence_penalty": 0.0}


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        message = SimpleNamespace(content='{"answer": 0}')
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage)


def fail(msg: str) -> None:
    raise AssertionError(msg)


def check_request(call: Dict[str, Any], row: Any, system_text: str, family: Any, served: str) -> Dict[str, Any]:
    if set(call) != REQUEST_KEYS:
        fail(f"request keys {sorted(call)} != {sorted(REQUEST_KEYS)}")
    if call["model"] != served:
        fail(f"model {call['model']!r}")
    if call["max_tokens"] != 8192:
        fail(f"max_tokens {call['max_tokens']}")
    for key in ("temperature", "top_p", "presence_penalty"):
        if call[key] != TRACE_DECODING[key]:
            fail(f"{key} {call[key]!r} != {TRACE_DECODING[key]!r}")
    if call["seed"] != SEED:
        fail(f"seed {call['seed']}")
    extra = dict(call["extra_body"])
    want_extra = {"top_k": TRACE_DECODING["top_k"], "repetition_penalty": TRACE_DECODING["repetition_penalty"]}
    kwargs = chat_template_kwargs(family)
    if kwargs is not None:
        want_extra["chat_template_kwargs"] = kwargs
    if extra != want_extra:
        fail(f"extra_body {extra} != {want_extra}")
    if family.name == "qwen3.5" and extra.get("chat_template_kwargs") != {"enable_thinking": False}:
        fail("Qwen3.5 request without enable_thinking=false")
    messages = call["messages"]
    if [m.get("role") for m in messages] != ["system", "user"]:
        fail(f"roles {[m.get('role') for m in messages]}")
    if sorted(messages[0]) != ["content", "role"] or messages[0]["content"] != system_text:
        fail("system message differs from the expected system prompt file")
    parts = messages[1]["content"]
    if sorted(messages[1]) != ["content", "role"] or not isinstance(parts, list):
        fail("user message is not a content list")
    if [p.get("type") for p in parts] != ["image_url", "text"]:
        fail(f"user parts {[p.get('type') for p in parts]} != ['image_url', 'text']")
    if sorted(parts[0]) != ["image_url", "type"] or sorted(parts[0]["image_url"]) != ["url"]:
        fail("image part carries extra keys")
    url = parts[0]["image_url"]["url"]
    prefix = "data:image/png;base64,"
    if not url.startswith(prefix):
        fail("image is not an inline PNG")
    image = Image.open(io.BytesIO(base64.b64decode(url[len(prefix):])))
    if sorted(parts[1]) != ["text", "type"]:
        fail("text part carries extra keys")
    if not row.prompt.startswith("<image>") or parts[1]["text"] != row.prompt[len("<image>"):]:
        fail("user text differs from the parquet prompt_answer (minus <image>)")
    return {"shown": list(image.size), "source": list(row.image_sizes[0])}


def check_case(cond: str, parquet: Path, system_file: Path, conditions: Dict[str, Any], model_id: str, served: str) -> Dict[str, Any]:
    family = detect_family(model_id)
    spec = conditions[cond]
    cfg = build_episode_config(spec, seed=SEED, family=family)
    system_text = system_file.read_bytes().decode("utf-8").strip()
    if cfg.system_prompt != system_text:
        fail(f"{cond}: trace-tool system prompt != {system_file}")
    rows = rows_from_parquet(parquet)
    ids = pq.read_table(str(parquet), columns=["instance_id", "cert_extension_id"]).to_pylist()
    ext_ids = {r["instance_id"]: r["cert_extension_id"] for r in ids}
    first: Dict[str, Any] = {}
    for row in rows:
        first.setdefault(row.task, row)
    backend = OpenAIChatBackend("http://127.0.0.1:9/v1", served, chat_template_kwargs=chat_template_kwargs(family))
    fake = FakeCompletions()
    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))  # type: ignore[assignment]
    sizes = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for task, row in sorted(first.items()):
            if row.answer_gt.get("value") is not None or row.annotation_gt.get("value") or row.reward_contract:
                fail(f"{cond}/{task}: the public parquet carries label values")
            before = len(fake.calls)
            record = asyncio.run(run_episode(row, cfg, backend, pool))
            if len(fake.calls) != before + 1 or record["turns"] != 1 or record["stop_reason"] != "answer":
                fail(f"{cond}/{task}: expected exactly one request")
            call = fake.calls[-1]
            sizes.append({"task": task, **check_request(call, row, system_text, family, served)})
            text_only = json.dumps([m["content"] if isinstance(m["content"], str) else
                                    [p for p in m["content"] if p["type"] == "text"] for m in call["messages"]])
            for secret in (row.instance_id, ext_ids.get(row.instance_id)):
                if secret and secret in text_only:
                    fail(f"{cond}/{task}: request text contains an instance/extension id")
    settings = describe(spec, cfg)
    return {
        "condition": cond,
        "parquet": parquet.name,
        "family": family.name,
        "tasks_checked": len(first),
        "system_prompt_sha256": hashlib.sha256(system_text.encode()).hexdigest(),
        "decoding_params": settings["decoding_params"],
        "chat_template_kwargs": settings.get("chat_template_kwargs"),
        "max_tokens": cfg.max_tokens_per_turn,
        "images": sizes,
        "ok": True,
    }


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--served", default="served-model")
    p.add_argument("--case", action="append", required=True, help="COND=PARQUET=SYSTEM_PROMPT_FILE")
    args = p.parse_args(argv)
    conditions = load_conditions(args.config)
    bad = 0
    for case in args.case:
        cond, parquet, system_file = case.split("=", 2)
        try:
            out = check_case(cond, Path(parquet), Path(system_file), conditions, args.model_id, args.served)
        except AssertionError as exc:
            out = {"condition": cond, "ok": False, "error": str(exc)}
            bad += 1
        print(json.dumps(out, sort_keys=True))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
