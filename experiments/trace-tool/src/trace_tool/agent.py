"""Multi-turn tool-calling episode against an OpenAI-compatible endpoint (vLLM).

One episode = one TRACE row under one condition:

    system -> user (image + question)
    -> assistant [-> user(<tool_response>...) -> assistant]*  (at most max_turns assistant turns)

The system message is the TRACE prompt (the condition's prompt files), then the
budget statement for forced conditions. When the condition has tools, the tool
block comes after both (Qwen2.5-VL, Qwen3-VL) or before both (Qwen3.5, where its
chat template puts it).

The final assistant message is the response that gets scored. With no tools
and max_turns=1 this is TRACE's single-turn validation request (same prompt,
decoding and pixel bounds; images are resized client-side by tools.base.fit_image).

Forced conditions (``forced_answer``) state a call budget in the system prompt,
end the last tool response of each tool turn (inside its ``<tool_response>`` tags)
with the number of calls left and, once the turns, calls or tokens run out, put a
final-answer notice there instead and give the model one more turn to answer
(see :func:`run_episode`).
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from PIL import Image

from .data import TraceRow
from .parsing import ParsedCall, parse_tool_calls, parse_xml_tool_calls, strip_hallucinated_responses
from .prompting import (
    FINAL_ANSWER_NOTICE,
    ImageURLCache,
    budget_statement,
    calls_left_notice,
    image_part,
    merge_text_parts,
    system_message,
    text_part,
    to_openai_messages,
    to_transcript,
    tool_response_content,
    user_prompt_content,
)
from .models import ModelFamily
from .tools.base import Tool, ToolContext, ToolResult, fit_image
from .tools.zoom import zoom_crop_pixels

DECODING_PRESETS: dict[str, dict[str, float]] = {
    # TRACE validation suite (rlvr/evaluation/trace_validation/suite.v1.json).
    "trace": {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "repetition_penalty": 1.0, "presence_penalty": 0.0},
    # Qwen3-VL Instruct recommendation, as used in the think_with_images cookbook.
    "qwen3vl": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0, "presence_penalty": 1.5},
    "greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "presence_penalty": 0.0},
}


DEFAULT_ANSWER_TOKEN_RESERVE = 1024


@dataclass
class EpisodeConfig:
    system_prompt: str
    family: ModelFamily  # patch grid and coordinate frame (trace_tool.models)
    tools: dict[str, Tool] = field(default_factory=dict)
    max_turns: int = 6
    max_tokens_per_turn: int = 2048
    max_total_tokens: int = 8192
    max_calls_per_turn: int = 2
    decoding: str = "trace"
    seed: int = 42
    # Pixel bounds for the image shown to the model (TRACE suite: 262144..4194304).
    min_pixels: int = 262144
    max_pixels: int = 4194304
    # Diagnostic: also show a crop of the ground-truth annotation region.
    oracle_crop: bool = False
    tool_timeout_s: float = 60.0
    # Forced conditions: budget notices and a forced final-answer turn (see run_episode).
    forced_answer: bool = False
    tool_call_budget: int | None = None  # None: (max_turns - 1) * max_calls_per_turn
    answer_token_reserve: int = DEFAULT_ANSWER_TOKEN_RESERVE  # tokens kept back for the final answer

    @property
    def call_budget(self) -> int:
        """Total tool calls a forced episode may execute (the resolved ``tool_call_budget``)."""
        if self.tool_call_budget is not None:
            return self.tool_call_budget
        return (self.max_turns - 1) * self.max_calls_per_turn


@dataclass
class Completion:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class RequestRejected(Exception):
    """The server refused the request (HTTP 4xx, e.g. context length exceeded).

    Terminal for the episode: rerunning the same conversation would fail again.
    Transient failures (connection, timeout, 5xx) propagate as other exceptions
    and are retried by the next ``trace-tool-eval`` run.
    """


class ChatBackend(Protocol):
    async def complete(self, messages: list[dict[str, Any]], *, max_tokens: int, sampling: dict[str, Any], seed: int) -> Completion: ...


class OpenAIChatBackend:
    """vLLM OpenAI-compatible chat endpoint.

    ``chat_template_kwargs`` (e.g. ``{"enable_thinking": False}`` for Qwen3.5, see
    ``models.chat_template_kwargs``) is sent with every request when not None.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "EMPTY",
        timeout: float = 600.0,
        max_retries: int = 3,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        self.model = model
        self.max_retries = max_retries
        self.chat_template_kwargs = chat_template_kwargs

    async def complete(self, messages: list[dict[str, Any]], *, max_tokens: int, sampling: dict[str, Any], seed: int) -> Completion:
        import openai

        extra_body = {"top_k": sampling["top_k"], "repetition_penalty": sampling["repetition_penalty"]}
        if self.chat_template_kwargs is not None:
            extra_body["chat_template_kwargs"] = self.chat_template_kwargs
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=sampling["temperature"],
                    top_p=sampling["top_p"],
                    presence_penalty=sampling["presence_penalty"],
                    seed=seed,
                    extra_body=extra_body,
                )
                break
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError, openai.RateLimitError):
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(2**attempt + random.random())
            except (openai.BadRequestError, openai.UnprocessableEntityError) as exc:
                raise RequestRejected(str(exc)) from exc
        choice = response.choices[0]
        usage = response.usage
        return Completion(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason or "",
            prompt_tokens=int(usage.prompt_tokens) if usage else 0,
            completion_tokens=int(usage.completion_tokens) if usage else 0,
        )


def annotation_region(annotation_gt: dict[str, Any]) -> list[float] | None:
    """Union box (absolute px) of every coordinate pair in a TRACE annotation."""
    xs: list[float] = []
    ys: list[float] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            if len(value) in (2, 4) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
                xs.extend(value[0::2])
                ys.extend(value[1::2])
            else:
                for item in value:
                    walk(item)

    walk(annotation_gt.get("value"))
    if not xs:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def oracle_crop_image(
    image: Image.Image, annotation_gt: dict[str, Any], family: ModelFamily, pad_frac: float = 0.1, min_side: int = 64
) -> Image.Image | None:
    region = annotation_region(annotation_gt)
    if region is None:
        return None
    width, height = image.size
    x1, y1, x2, y2 = region
    pad_x = max(pad_frac * width, (min_side - (x2 - x1)) / 2, 0)
    pad_y = max(pad_frac * height, (min_side - (y2 - y1)) / 2, 0)
    box = [max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)]
    crop, _ = zoom_crop_pixels(image, box, factor=family.patch_factor, min_side=family.zoom_min_side)
    return crop


ORACLE_NOTE = "The second image is a zoomed-in view of the region relevant to the question.\n"
CALL_BUDGET_SKIPPED = "Error: no tool calls left; this call was skipped."


def _tool_body(result: ToolResult, ctx: ToolContext) -> list[dict[str, Any]]:
    """Images first, then text: the order verl's ToolAgentLoop uses."""
    body: list[dict[str, Any]] = []
    for image, index in zip(result.images, ctx.add_images(result.images)):
        body.append(image_part(image, index))
    if result.text:
        body.append(text_part(result.text))
    if not body:
        body.append(text_part("Tool returned no output."))
    return body


class _PoolSlots:
    """One slot per worker thread, so a call's timeout only starts once a thread is free.

    The in-flight count is shared across event loops (a timed-out call keeps its
    thread until it really finishes); the asyncio waiters belong to one loop, so
    a pool must not be used by two loops at once.
    """

    def __init__(self, pool: ThreadPoolExecutor):
        self.capacity = pool._max_workers
        self.in_flight = 0
        self.lock = threading.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.semaphore: asyncio.Semaphore | None = None

    def for_loop(self, loop: asyncio.AbstractEventLoop) -> asyncio.Semaphore:
        with self.lock:
            if self.loop is not loop:
                if self.loop is not None and not self.loop.is_closed() and self.loop.is_running():
                    raise RuntimeError("a tool thread pool cannot be shared by two running event loops")
                self.loop = loop
                # threads still busy with calls from an earlier loop keep their slots
                self.semaphore = asyncio.Semaphore(max(0, self.capacity - self.in_flight))
            return self.semaphore

    def started(self) -> None:
        with self.lock:
            self.in_flight += 1

    def finished(self, loop: asyncio.AbstractEventLoop, semaphore: asyncio.Semaphore) -> None:
        with self.lock:
            self.in_flight -= 1
            current = self.loop is loop and self.semaphore is semaphore
        if current:
            try:
                loop.call_soon_threadsafe(semaphore.release)
            except RuntimeError:  # loop already closed
                pass
        else:  # a call from an earlier loop: hand its slot to the current loop
            with self.lock:
                new_loop, new_sem = self.loop, self.semaphore
            if new_loop is not None and new_sem is not None:
                try:
                    new_loop.call_soon_threadsafe(new_sem.release)
                except RuntimeError:
                    pass


_POOL_SLOTS: "weakref.WeakKeyDictionary[ThreadPoolExecutor, _PoolSlots]" = weakref.WeakKeyDictionary()
_POOL_SLOTS_LOCK = threading.Lock()


def _pool_slots(pool: ThreadPoolExecutor) -> _PoolSlots:
    with _POOL_SLOTS_LOCK:
        slots = _POOL_SLOTS.get(pool)
        if slots is None:
            slots = _POOL_SLOTS[pool] = _PoolSlots(pool)
        return slots


async def _execute(call: ParsedCall, tools: dict[str, Tool], ctx: ToolContext, pool: ThreadPoolExecutor, timeout: float) -> tuple[ToolResult, float]:
    """Run one tool call; returns (result, seconds spent executing, excluding queueing)."""
    if call.error is not None:
        return ToolResult.error(call.error), 0.0
    tool = tools.get(call.name or "")
    if tool is None:
        return ToolResult.error(f"Error: unknown function '{call.name}'. Available: {sorted(tools)}"), 0.0
    loop = asyncio.get_running_loop()
    slots = _pool_slots(pool)
    semaphore = slots.for_loop(loop)
    await semaphore.acquire()

    def work() -> tuple[ToolResult, float]:
        started = time.perf_counter()
        result = tool(call.arguments, ctx)
        return result, time.perf_counter() - started

    slots.started()
    try:
        future = pool.submit(work)
    except BaseException:
        slots.finished(loop, semaphore)
        raise
    # the slot is freed when the thread really finishes, even after a timeout
    future.add_done_callback(lambda _: slots.finished(loop, semaphore))
    try:
        return await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), timeout=timeout)
    except asyncio.TimeoutError:
        if future.done() and not future.cancelled() and future.exception() is None:
            return future.result()  # finished while the loop was busy past the deadline
        return ToolResult.error(f"Error: {call.name} timed out after {timeout:.0f}s"), timeout


def build_initial_messages(row: TraceRow, cfg: EpisodeConfig) -> tuple[list[dict[str, Any]], ToolContext, list[Image.Image]]:
    """System + first user turn exactly as sent (used by run_episode and --dry-run)."""
    full_images = row.images()
    factor = cfg.family.patch_factor
    shown = [fit_image(img, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels, factor=factor) for img in full_images]
    # tools read the full-resolution images; model coordinates refer to the shown sizes
    ctx = ToolContext(full_images, shown_sizes=[img.size for img in shown])
    content = user_prompt_content(row.prompt, [(img, i) for i, img in enumerate(shown)])
    if cfg.oracle_crop:
        crop = oracle_crop_image(full_images[0], row.annotation_gt, cfg.family)
        if crop is not None:
            (crop_index,) = ctx.add_images([crop])
            lead = [p for p in content if p["type"] == "image"]
            rest = [p for p in content if p["type"] != "image"]
            content = merge_text_parts(lead + [image_part(crop, crop_index), text_part(ORACLE_NOTE)] + rest)
    messages: list[dict[str, Any]] = []
    system_text = cfg.system_prompt
    if cfg.forced_answer:
        system_text = f"{cfg.system_prompt}\n\n{budget_statement(cfg.call_budget, cfg.max_calls_per_turn)}"
    system = system_message(system_text, [tool.schema() for tool in cfg.tools.values()], family=cfg.family)
    if system is not None:
        messages.append(system)
    messages.append({"role": "user", "content": content})
    return messages, ctx, shown


def tool_call_parser(cfg: EpisodeConfig) -> Callable[[str], list[ParsedCall]]:
    """The parser for the family's tool-call format (XML calls are typed by the tools' schemas)."""
    if cfg.family.tool_format == "qwen3_xml":
        schemas = [tool.schema()["function"] for tool in cfg.tools.values()]
        param_schemas = {schema["name"]: schema["parameters"] for schema in schemas}
        return lambda text: parse_xml_tool_calls(text, param_schemas)
    return parse_tool_calls


async def run_episode(
    row: TraceRow,
    cfg: EpisodeConfig,
    backend: ChatBackend,
    pool: ThreadPoolExecutor,
) -> dict[str, Any]:
    """One episode. Unforced: up to ``max_turns`` turns, stopping on an answer or when
    the turns or ``max_total_tokens`` run out (the last turn's calls are not executed).

    Forced (``cfg.forced_answer``): at most ``call_budget`` calls are executed in
    total (extra calls get an error, as do calls over ``max_calls_per_turn``), tool
    turns are generated with ``answer_token_reserve`` tokens held back, and the last
    tool response of each tool turn ends with the number of calls left. When the turns
    (one is left), the calls or the tokens (only the reserve is left) run out, it ends
    with FINAL_ANSWER_NOTICE instead and the next turn is the final one: calls in it
    are recorded but never executed (stop reason "forced_tool_call").
    """
    sampling = DECODING_PRESETS[cfg.decoding]
    parse_calls = tool_call_parser(cfg)
    # image decoding/resizing and PNG encoding run off the event loop thread
    messages, ctx, shown = await asyncio.to_thread(build_initial_messages, row, cfg)

    cache = ImageURLCache()
    tool_calls: list[dict[str, Any]] = []
    completion_tokens = 0
    last_prompt_tokens = 0
    turns = 0
    stop_reason = "max_turns"
    hallucinated = 0
    error: str | None = None
    started = time.perf_counter()
    forced = cfg.forced_answer
    budget = cfg.call_budget
    per_turn = cfg.max_calls_per_turn
    calls_used = 0
    forced_next = False  # the next turn is the forced final-answer turn
    forced_turn = False
    forced_reason: str | None = None

    for turn in range(cfg.max_turns):
        remaining = cfg.max_total_tokens - completion_tokens
        if forced:
            # tool turns keep the answer reserve; the forcing rule below keeps both positive
            max_tokens = min(cfg.max_tokens_per_turn, remaining if forced_next else remaining - cfg.answer_token_reserve)
            if max_tokens <= 0:  # unreachable unless the endpoint overshoots max_tokens
                stop_reason = "token_budget"
                break
        else:
            if remaining <= 0:
                stop_reason = "token_budget"
                break
            max_tokens = min(cfg.max_tokens_per_turn, remaining)
        try:
            payload = await asyncio.to_thread(to_openai_messages, messages, cache)
            completion = await backend.complete(
                payload,
                max_tokens=max_tokens,
                sampling=sampling,
                seed=cfg.seed,
            )
        except RequestRejected as exc:
            stop_reason = "rejected"
            error = str(exc)[:2000]
            break
        except Exception as exc:
            stop_reason = "error"
            error = f"{type(exc).__name__}: {exc}"[:2000]
            break
        turns += 1
        completion_tokens += completion.completion_tokens
        last_prompt_tokens = completion.prompt_tokens
        text, cut = completion.text, False
        if cfg.tools:  # without tools the text is scored verbatim, as in TRACE
            text, cut = strip_hallucinated_responses(text)
        hallucinated += int(cut)
        messages.append({"role": "assistant", "content": text})

        calls = parse_calls(text) if cfg.tools else []
        if forced_next:  # the forced final turn: calls are recorded, never executed
            forced_turn = True
            for call in calls:
                tool_calls.append(_call_record(turn, call, executed=False, reason="forced_answer"))
            if calls:
                stop_reason = "forced_tool_call"
            else:
                stop_reason = "answer" if completion.finish_reason == "stop" or cut else completion.finish_reason or "length"
            break
        if not calls:
            stop_reason = "answer" if completion.finish_reason == "stop" or cut else completion.finish_reason or "length"
            break
        if turn == cfg.max_turns - 1:  # never in forced mode: the turn before forces this one
            stop_reason = "max_turns"
            for call in calls:  # requested but never executed: no turn left to show the result
                tool_calls.append(_call_record(turn, call, executed=False, reason="max_turns"))
            break

        # Calls run in order, each seeing the images earlier calls returned, as in
        # Qwen-Agent (verl's ToolAgentLoop runs a turn's calls in parallel instead).
        # Forced conditions also stop at the budget (>= 1 here: no calls left forces the next turn).
        allowed = min(per_turn, budget - calls_used) if forced else per_turn
        bodies = []
        for call in calls[:allowed]:
            result, latency = await _execute(call, cfg.tools, ctx, pool, cfg.tool_timeout_s)
            bodies.append(_tool_body(result, ctx))
            tool_calls.append(_call_record(turn, call, executed=call.error is None and call.name in cfg.tools, result=result, latency=latency))
        for call in calls[allowed:per_turn]:
            bodies.append([text_part(CALL_BUDGET_SKIPPED)])
            tool_calls.append(_call_record(turn, call, executed=False, reason="call_budget"))
        for call in calls[per_turn:]:
            bodies.append([text_part(f"Error: at most {cfg.max_calls_per_turn} tool calls per turn; this call was skipped.")])
            tool_calls.append(_call_record(turn, call, executed=False, reason="per_turn_cap"))
        if forced:
            calls_used += min(len(calls), allowed)
            turns_left = cfg.max_turns - (turn + 1)
            if turns_left <= 1:
                forced_next, forced_reason = True, "turns"
            elif budget - calls_used <= 0:
                forced_next, forced_reason = True, "calls"
            elif cfg.max_total_tokens - completion_tokens - cfg.answer_token_reserve <= 0:
                # only the reserve is left. Known limit (EXPERIMENT_DESIGN.md): a later tool turn capped
                # below max_tokens_per_turn by the reserve and cut before any call ends the episode unforced
                forced_next, forced_reason = True, "tokens"
            notice = FINAL_ANSWER_NOTICE if forced_next else calls_left_notice(min(budget - calls_used, (turns_left - 1) * per_turn))
            # inside the last <tool_response>: text outside the tags would make the turn a new query
            bodies[-1] = bodies[-1] + [text_part("\n\n" + notice)]
        messages.append({"role": "user", "content": tool_response_content(bodies, family=cfg.family)})

    final = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")
    return {
        "row_index": row.row_index,
        "instance_id": row.instance_id,
        "domain": row.domain,
        "task": row.task,
        "answer_type": row.answer_type,
        "final_response": final,
        "stop_reason": stop_reason,
        "error": error,
        "turns": turns,
        "tool_calls": tool_calls,
        "n_tool_calls": len(tool_calls),
        **(
            {"tool_call_budget": budget, "tool_calls_used": calls_used, "forced_turn": forced_turn, "forced_reason": forced_reason}
            if forced
            else {}
        ),
        "hallucinated_tool_responses": hallucinated,
        "completion_tokens": completion_tokens,
        "final_prompt_tokens": last_prompt_tokens,
        "n_images": len(ctx.images),
        "shown_image_size": list(shown[0].size),
        "latency_s": round(time.perf_counter() - started, 3),
        "transcript": to_transcript(messages),
    }


def _call_record(
    turn: int,
    call: ParsedCall,
    *,
    executed: bool,
    result: ToolResult | None = None,
    latency: float = 0.0,
    reason: str | None = None,
) -> dict[str, Any]:
    """One entry per requested call. ``executed`` means a known tool actually ran;
    ``ok`` means it ran and succeeded."""
    record: dict[str, Any] = {
        "turn": turn,
        "name": call.name,
        "arguments": call.arguments,
        "executed": executed,
        "ok": bool(executed and result is not None and result.ok),
        "parse_error": call.error,
    }
    if result is not None:
        record.update(text=(result.text or "")[:1000], n_images=len(result.images), meta=result.meta, latency_s=round(latency, 4))
    if reason:
        record["skipped"] = reason
    return record
