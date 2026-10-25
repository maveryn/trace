"""Optional fallback extraction for responses the deterministic extractor could not resolve.

Replicates TRACE's judge pass (rlvr/evaluation/trace_validation/judge_extract.py):
Qwen3-32B, thinking off, temperature 0, the vendored system prompt, the user
message {"answer_type": ..., "model_response": ...} as canonical JSON, retries
at max_tokens 128 / 256 / 512, and the vendored strict output validator. The
judge never sees the image, question or reference answer.

TRACE calls /v1/completions with a locally rendered prompt; we use
/v1/chat/completions with ``chat_template_kwargs={"enable_thinking": false}``,
which renders the same template on the server.

    vllm serve Qwen/Qwen3-32B --revision 9216db5781bf21249d130ec9da846c4624c16137 \
        --served-model-name qwen3-32b-judge --port 18200 --max-model-len 16384

TRACE served the judge with an 8192-token context for 2048-token responses;
16384 fits the 8192-token responses of no_tool_budget. A response that still
does not fit is recorded as ``rejected`` and scores as unresolved (0).

    trace-tool-judge --pending runs/x/judge_pending.jsonl --out runs/x/judge_results.jsonl \
        --endpoint http://127.0.0.1:18200/v1 --model qwen3-32b-judge
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .jsonl import append_jsonl, exclusive_lock, read_jsonl, repair_tail
from .vendor.trace_rlvr.judge_contract import SYSTEM_PROMPT, JudgeOutputError, _canonical_json, validate_judge_output

RETRY_MAX_TOKENS = (128, 256, 512)


async def judge_one(client: Any, model: str, item: dict[str, Any]) -> dict[str, Any] | None:
    """Judge result for one pending row; None if the endpoint failed transiently (retried next run)."""
    import openai

    try:
        return await _judge_one(client, model, item)
    except (openai.BadRequestError, openai.UnprocessableEntityError) as exc:
        # e.g. the response does not fit the judge's context: terminal, scores as unresolved
        return {"key": item["key"], "status": "rejected", "answer": None, "evidence": "", "error": str(exc)[:500], "attempts": []}
    except (openai.APIError, OSError) as exc:
        print(f"warning: judge request failed ({type(exc).__name__}: {str(exc)[:200]}); will retry next run", file=sys.stderr)
        return None


async def _judge_one(client: Any, model: str, item: dict[str, Any]) -> dict[str, Any]:
    user = _canonical_json({"answer_type": item["answer_type"], "model_response": item["raw_response"]})
    attempts = []
    for max_tokens in RETRY_MAX_TOKENS:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            seed=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        output = (response.choices[0].message.content or "").strip()
        try:
            parsed = validate_judge_output(output, raw_response=item["raw_response"], answer_type=item["answer_type"])
            attempts.append({"max_tokens": max_tokens, "ok": True})
            return {"key": item["key"], **parsed, "raw_output": output, "attempts": attempts}
        except JudgeOutputError as exc:
            attempts.append({"max_tokens": max_tokens, "ok": False, "error": str(exc)})
    return {"key": item["key"], "status": "invalid", "answer": None, "evidence": "", "raw_output": output, "attempts": attempts}


async def run(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    if not Path(args.pending).exists():
        raise SystemExit(f"--pending {args.pending} does not exist")
    pending, _ = read_jsonl(Path(args.pending))
    out_path = Path(args.out)
    with exclusive_lock(out_path.with_name(out_path.name + ".lock")):
        repair_tail(out_path)
        done = {r["key"] for r in read_jsonl(out_path)[0] if "key" in r}
        todo = [p for p in pending if p["key"] not in done]
        client = AsyncOpenAI(base_url=args.endpoint, api_key="EMPTY", timeout=300, max_retries=2)
        semaphore = asyncio.Semaphore(args.concurrency)
        progress = tqdm(total=len(todo), desc="judge")
        failed = 0

        async def one(item: dict[str, Any]) -> None:
            nonlocal failed
            async with semaphore:
                result = await judge_one(client, args.model, item)
            if result is None:
                failed += 1
            else:
                append_jsonl(out_path, result)
            progress.update(1)

        await asyncio.gather(*(one(item) for item in todo))
        progress.close()
    if failed:
        print(f"{failed} items failed transiently; rerun the same command to retry them", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pending", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--endpoint", default="http://127.0.0.1:18200/v1")
    p.add_argument("--model", default="qwen3-32b-judge")
    p.add_argument("--concurrency", type=int, default=64)
    asyncio.run(run(p.parse_args(argv)))


if __name__ == "__main__":
    main()
