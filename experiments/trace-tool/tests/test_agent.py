import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from trace_tool.agent import (
    CALL_BUDGET_SKIPPED,
    Completion,
    EpisodeConfig,
    RequestRejected,
    annotation_region,
    oracle_crop_image,
    run_episode,
)
from trace_tool.conditions import build_episode_config, load_conditions, read_system_prompt
from trace_tool.prompting import FINAL_ANSWER_NOTICE, QWEN35_TOOLS_HEADER, budget_statement
from trace_tool.scoring import score_row
from trace_tool.tools import build_tools
from trace_tool.tools.base import Tool, ToolResult
from trace_tool.models import QWEN3_VL, QWEN25_VL, QWEN35

from .conftest import make_image, make_row

ZOOM_CALL = '<tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [0, 0, 400, 400], "label": "box", "img_idx": 0}}\n</tool_call>'
CALC_CALL = '<tool_call>\n{"name": "calculator", "arguments": {"expression": "1 + 2"}}\n</tool_call>'


class ScriptedBackend:
    def __init__(self, replies, finish="stop"):
        self.replies = list(replies)
        self.requests = []
        self.finish = finish

    async def complete(self, messages, *, max_tokens, sampling, seed):
        self.requests.append({"messages": messages, "max_tokens": max_tokens, "sampling": sampling, "seed": seed})
        text = self.replies.pop(0)
        return Completion(text=text, finish_reason=self.finish, prompt_tokens=100, completion_tokens=50)


def _run(row, cfg, backend):
    with ThreadPoolExecutor(2) as pool:
        return asyncio.run(run_episode(row, cfg, backend, pool))


def test_zoom_then_calculator_then_answer(row):
    cfg = EpisodeConfig(family=QWEN3_VL, system_prompt=read_system_prompt(["prompts/trace_system_prompt.txt"]), tools=build_tools(["zoom", "calculator"], QWEN3_VL))
    backend = ScriptedBackend([ZOOM_CALL, "Two shapes.\n" + CALC_CALL, 'So 3.\n{"answer": 3}'])
    record = _run(row, cfg, backend)

    assert record["stop_reason"] == "answer"
    assert record["turns"] == 3 and record["n_tool_calls"] == 2
    assert [c["name"] for c in record["tool_calls"]] == ["image_zoom_in_tool", "calculator"]
    assert all(c["ok"] for c in record["tool_calls"])
    assert record["n_images"] == 2  # original + zoom crop
    assert score_row(record["final_response"], row.answer_gt, row.annotation_gt, row.reward_contract)["strict_correct"] == 1.0

    # request 2 replays the assistant turn and sends the crop inside <tool_response>
    msgs = backend.requests[1]["messages"]
    assert msgs[0]["role"] == "system" and "# Tools" in msgs[0]["content"]
    assert msgs[0]["content"].startswith("You are a helpful, conversational assistant")
    assert msgs[2] == {"role": "assistant", "content": ZOOM_CALL}
    tool_turn = msgs[3]
    assert tool_turn["role"] == "user"
    assert [p["type"] for p in tool_turn["content"]] == ["text", "image_url", "text"]
    assert tool_turn["content"][0]["text"] == "<tool_response>\n"
    assert tool_turn["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert backend.requests[2]["messages"][5]["content"] == [{"type": "text", "text": "<tool_response>\n3\n</tool_response>"}]
    # the transcript is JSON-serialisable
    json.dumps(record)


def test_no_tool_condition_is_single_turn(row):
    spec = load_conditions()["no_tool"]
    cfg = build_episode_config(spec, seed=42, family=QWEN3_VL)
    backend = ScriptedBackend([ZOOM_CALL])  # tool-looking text is not executed without tools
    record = _run(row, cfg, backend)
    assert record["turns"] == 1 and record["n_tool_calls"] == 0
    req = backend.requests[0]
    assert req["max_tokens"] == 2048 and req["seed"] == 42
    assert req["sampling"] == {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "repetition_penalty": 1.0, "presence_penalty": 0.0}
    assert "# Tools" not in req["messages"][0]["content"]


def test_turn_limit_and_budget(row):
    cfg = EpisodeConfig(family=QWEN3_VL, system_prompt="s", tools=build_tools(["calculator"], QWEN3_VL), max_turns=3, max_total_tokens=1000)
    backend = ScriptedBackend([CALC_CALL] * 3)
    record = _run(row, cfg, backend)
    assert record["stop_reason"] == "max_turns" and record["turns"] == 3 and record["n_tool_calls"] == 3
    # the last turn's call is recorded but was never executed
    assert [c["executed"] for c in record["tool_calls"]] == [True, True, False]
    assert record["tool_calls"][-1]["skipped"] == "max_turns"

    cfg = EpisodeConfig(family=QWEN3_VL, system_prompt="s", tools=build_tools(["calculator"], QWEN3_VL), max_turns=10, max_total_tokens=120, max_tokens_per_turn=100)
    backend = ScriptedBackend([CALC_CALL] * 10)
    record = _run(row, cfg, backend)
    assert record["stop_reason"] == "token_budget"
    assert [r["max_tokens"] for r in backend.requests] == [100, 70, 20]


def test_errors_are_returned_to_model(row):
    cfg = EpisodeConfig(family=QWEN3_VL, system_prompt="s", tools=build_tools(["calculator"], QWEN3_VL), max_calls_per_turn=1)
    bad = '<tool_call>\n{"name": "nope", "arguments": {}}\n</tool_call>\n' + CALC_CALL
    backend = ScriptedBackend([bad, '{"answer": 3}'])
    record = _run(row, cfg, backend)
    tool_text = backend.requests[1]["messages"][-1]["content"][0]["text"]
    assert "unknown function 'nope'" in tool_text and "at most 1 tool calls" in tool_text
    assert [c["ok"] for c in record["tool_calls"]] == [False, False]


def test_endpoint_error_is_recorded(row):
    class Failing:
        async def complete(self, *a, **k):
            raise RuntimeError("boom")

    record = _run(row, EpisodeConfig(family=QWEN3_VL, system_prompt="s"), Failing())
    assert record["stop_reason"] == "error" and "boom" in record["error"]


def test_oracle_crop_and_lowres(row):
    assert annotation_region(row.annotation_gt) == [50, 50, 460, 310]
    assert annotation_region({"type": "point_map", "value": {"a": [10, 20], "b": [30, 5]}}) == [10, 5, 30, 20]
    assert oracle_crop_image(make_image(), {"type": "point", "value": [300, 200]}, QWEN3_VL) is not None

    spec = load_conditions()["oracle_crop"]
    backend = ScriptedBackend(['{"answer": 3}'])
    record = _run(row, build_episode_config(spec, seed=1, family=QWEN3_VL), backend)
    content = backend.requests[0]["messages"][1]["content"]
    assert [p["type"] for p in content] == ["image_url", "image_url", "text"]
    assert record["n_images"] == 2

    big = make_row(image=make_image(1600, 1200))
    spec = load_conditions()["no_tool_lowres"]
    record = _run(big, build_episode_config(spec, seed=1, family=QWEN3_VL), ScriptedBackend(['{"answer": 3}']))
    w, h = record["shown_image_size"]
    assert w * h <= 401408 and w % 32 == 0 and h % 32 == 0


def test_all_conditions_build():
    for name, spec in load_conditions().items():
        cfg = build_episode_config(spec, seed=0, family=QWEN3_VL)
        assert cfg.max_turns >= 1, name


def test_rejected_request_is_terminal(row):
    from trace_tool.agent import RequestRejected

    class Rejecting:
        async def complete(self, *a, **k):
            raise RequestRejected("maximum context length exceeded")

    record = _run(row, EpisodeConfig(family=QWEN3_VL, system_prompt="s"), Rejecting())
    assert record["stop_reason"] == "rejected"


def test_rgba_composited_on_white():
    from PIL import Image

    from trace_tool.data import to_rgb

    rgba = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    assert to_rgb(rgba).getpixel((0, 0)) == (255, 255, 255)


def test_tool_timeout_excludes_queue_time():
    """With one worker thread, the second call waits for a slot; its timeout must not run meanwhile."""
    import time

    from trace_tool.agent import _execute
    from trace_tool.parsing import ParsedCall
    from trace_tool.tools.base import Tool, ToolContext, ToolResult

    class Slow(Tool):
        name = "slow"

        def run(self, args, ctx):
            time.sleep(0.6)
            return ToolResult(text="done")

    async def go():
        with ThreadPoolExecutor(1) as pool:
            calls = [ParsedCall("slow", {}, "") for _ in range(2)]
            return await asyncio.gather(*(_execute(c, {"slow": Slow()}, ToolContext([]), pool, 1.0) for c in calls))

    results = asyncio.run(go())
    assert [r.ok for r, _ in results] == [True, True]
    assert all(0.5 < latency < 0.9 for _, latency in results)


def test_elongated_zoom_becomes_tool_error():
    tool = build_tools(["zoom"], QWEN3_VL)["image_zoom_in_tool"]
    from trace_tool.tools.base import ToolContext

    result = tool({"bbox_2d": [0, 0, 1, 1000], "label": "x", "img_idx": 0}, ToolContext([make_image(800, 600)]))
    assert not result.ok and "elongated" in result.text


def test_calls_in_a_turn_run_in_order_and_see_earlier_images(row):
    two_zooms = (
        '<tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [0, 0, 500, 500], "label": "a", "img_idx": 0}}\n</tool_call>\n'
        '<tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [0, 0, 500, 500], "label": "b", "img_idx": 1}}\n</tool_call>'
    )
    cfg = EpisodeConfig(family=QWEN3_VL, system_prompt="s", tools=build_tools(["zoom"], QWEN3_VL))
    record = _run(row, cfg, ScriptedBackend([two_zooms, '{"answer": 3}']))
    assert [c["ok"] for c in record["tool_calls"]] == [True, True]  # call 2 zooms into call 1's output
    assert record["n_images"] == 3


def test_zoom_validates_like_qwen_agent():
    from trace_tool.tools.base import ToolContext

    tool = build_tools(["zoom"], QWEN3_VL)["image_zoom_in_tool"]
    ctx = ToolContext([make_image(800, 600), make_image(400, 300)])
    assert not tool({"bbox_2d": [0, 0, 500, 500], "label": "x"}, ctx).ok  # img_idx required
    assert not tool({"bbox_2d": [0, 0, 500, 500], "img_idx": 0}, ctx).ok  # label required
    assert not tool({"bbox_2d": ["0", 0, 500, 500], "label": "x", "img_idx": 0}, ctx).ok  # numbers only
    latest = tool({"bbox_2d": [0, 0, 1000, 1000], "label": "x", "img_idx": -1}, ctx)
    assert latest.ok and latest.meta["abs_box"] == [0, 0, 400, 300]  # -1 = latest image, as in Python


def test_no_tool_text_is_scored_verbatim(row):
    text = 'Counting.\n<tool_response>\nnone\n</tool_response>\nSo three.\n{"answer": 3}'
    record = _run(row, build_episode_config(load_conditions()["no_tool"], seed=1, family=QWEN3_VL), ScriptedBackend([text]))
    assert record["final_response"] == text and record["hallucinated_tool_responses"] == 0


def test_timeout_race_keeps_finished_result():
    """A call that finished while the loop was blocked past its deadline is not reported as timed out."""
    import time

    from trace_tool.agent import _execute
    from trace_tool.parsing import ParsedCall
    from trace_tool.tools.base import Tool, ToolContext, ToolResult

    class Quick(Tool):
        name = "quick"

        def run(self, args, ctx):
            time.sleep(0.05)
            return ToolResult(text="done")

    async def go():
        with ThreadPoolExecutor(1) as pool:
            task = asyncio.ensure_future(_execute(ParsedCall("quick", {}, ""), {"quick": Quick()}, ToolContext([]), pool, 0.1))
            await asyncio.sleep(0)
            time.sleep(0.3)  # block the loop past the deadline; the tool finishes meanwhile
            return await task

    result, _ = asyncio.run(go())
    assert result.ok and result.text == "done"


# --- forced conditions (budget notices + forced final answer) -----------------------

ANSWER = 'So 3.\n{"answer": 3}'
FORCED_KEYS = {"tool_call_budget", "tool_calls_used", "forced_turn", "forced_reason"}
RECORD_KEYS = {
    "row_index", "instance_id", "domain", "task", "answer_type", "final_response", "stop_reason", "error", "turns",
    "tool_calls", "n_tool_calls", "hallucinated_tool_responses", "completion_tokens", "final_prompt_tokens", "n_images",
    "shown_image_size", "latency_s", "transcript",
}


class TokenBackend(ScriptedBackend):
    """Spends ``tokens`` per reply (an int, or one per reply) but never more than max_tokens,
    as vLLM does; a reply clipped by max_tokens finishes with "length"."""

    def __init__(self, replies, tokens=50, finishes=None):
        super().__init__(replies)
        self.tokens = list(tokens) if isinstance(tokens, list) else None
        self.default_tokens = tokens if isinstance(tokens, int) else 50
        self.finishes = list(finishes or [])  # explicit finish reasons override the rule above

    async def complete(self, messages, *, max_tokens, sampling, seed):
        self.requests.append({"messages": messages, "max_tokens": max_tokens, "sampling": sampling, "seed": seed})
        wanted = self.tokens.pop(0) if self.tokens else self.default_tokens
        finish = self.finishes.pop(0) if self.finishes else ("length" if wanted > max_tokens else "stop")
        return Completion(text=self.replies.pop(0), finish_reason=finish, prompt_tokens=100, completion_tokens=min(wanted, max_tokens))


def test_token_backend_clips_like_an_endpoint():
    async def go(backend, *caps):
        return [await backend.complete([], max_tokens=cap, sampling={}, seed=0) for cap in caps]

    # an int and a list behave alike: a reply that wants more than max_tokens is clipped and finishes "length"
    for tokens in (300, [300, 300, 300]):
        done = asyncio.run(go(TokenBackend(["a", "b", "c"], tokens=tokens), 400, 300, 200))
        assert [(c.completion_tokens, c.finish_reason) for c in done] == [(300, "stop"), (300, "stop"), (200, "length")]


def _forced(tools=("calculator",), family=QWEN3_VL, **kw):
    return EpisodeConfig(
        family=family,
        system_prompt=read_system_prompt(["prompts/trace_system_prompt.txt"]),
        tools=build_tools(list(tools), family),
        forced_answer=True,
        **kw,
    )


def _last_turn_text(request):
    return "".join(p["text"] for p in request["messages"][-1]["content"] if p["type"] == "text")


def test_forced_countdown_text_and_position(row):
    cfg = _forced()
    assert cfg.call_budget == 10  # (max_turns - 1) * max_calls_per_turn
    backend = ScriptedBackend([CALC_CALL, CALC_CALL + "\n" + CALC_CALL, ANSWER])
    record = _run(row, cfg, backend)

    # the budget statement follows the system prompt; the tool block comes after it (Hermes layout)
    system = backend.requests[0]["messages"][0]["content"]
    assert system.startswith(cfg.system_prompt + "\n\n" + budget_statement(10, 2) + "\n\n# Tools")
    assert "You can make at most 10 tool calls in total, and at most 2 per turn." in system
    # the countdown ends the last tool response, inside its tags; left = min(budget left, calls the turns allow)
    assert backend.requests[1]["messages"][-1]["content"] == [{"type": "text", "text": "<tool_response>\n3\n\n[Tool calls left: 8]\n</tool_response>"}]
    assert _last_turn_text(backend.requests[2]) == "<tool_response>\n3\n</tool_response>\n<tool_response>\n3\n\n[Tool calls left: 6]\n</tool_response>"
    assert [r["max_tokens"] for r in backend.requests] == [2048, 2048, 2048]
    assert record["stop_reason"] == "answer" and record["turns"] == 3
    assert {k: record[k] for k in FORCED_KEYS} == {"tool_call_budget": 10, "tool_calls_used": 3, "forced_turn": False, "forced_reason": None}
    assert set(record) == RECORD_KEYS | FORCED_KEYS
    json.dumps(record)


def test_forced_by_turns_then_answer(row):
    cfg = _forced(max_turns=3)
    backend = ScriptedBackend([CALC_CALL, CALC_CALL, ANSWER])
    record = _run(row, cfg, backend)
    assert _last_turn_text(backend.requests[1]) == "<tool_response>\n3\n\n[Tool calls left: 2]\n</tool_response>"
    assert _last_turn_text(backend.requests[2]) == f"<tool_response>\n3\n\n{FINAL_ANSWER_NOTICE}\n</tool_response>"
    assert record["stop_reason"] == "answer" and record["final_response"] == ANSWER
    assert {k: record[k] for k in FORCED_KEYS} == {"tool_call_budget": 4, "tool_calls_used": 2, "forced_turn": True, "forced_reason": "turns"}
    assert [c["executed"] for c in record["tool_calls"]] == [True, True]
    # the forced turn is capped by max_tokens_per_turn too, not only by the tokens left (8,092 here)
    assert [r["max_tokens"] for r in backend.requests] == [2048, 2048, 2048]


def test_forced_turn_with_tool_calls_is_recorded_not_executed(row):
    cfg = _forced(tools=("calculator", "zoom"), max_turns=3)
    record = _run(row, cfg, ScriptedBackend([CALC_CALL, CALC_CALL, "Still zooming.\n" + ZOOM_CALL]))
    assert record["stop_reason"] == "forced_tool_call" and record["forced_turn"] and record["forced_reason"] == "turns"
    last = record["tool_calls"][-1]
    assert last["name"] == "image_zoom_in_tool" and last["executed"] is False and last["skipped"] == "forced_answer" and "text" not in last
    assert record["n_images"] == 1 and record["tool_calls_used"] == 2  # the zoom never ran
    assert record["final_response"].startswith("Still zooming.")


def test_forced_by_calls_skips_calls_over_the_budget(row):
    cfg = _forced(tool_call_budget=3)
    two_calls = CALC_CALL + "\n" + CALC_CALL
    backend = ScriptedBackend([two_calls, two_calls, ANSWER])
    record = _run(row, cfg, backend)
    assert _last_turn_text(backend.requests[1]).endswith("<tool_response>\n3\n\n[Tool calls left: 1]\n</tool_response>")
    # one call left: the first runs, the second is skipped, and the final-answer notice ends the last response
    assert _last_turn_text(backend.requests[2]) == (
        f"<tool_response>\n3\n</tool_response>\n<tool_response>\n{CALL_BUDGET_SKIPPED}\n\n{FINAL_ANSWER_NOTICE}\n</tool_response>"
    )
    assert [c["executed"] for c in record["tool_calls"]] == [True, True, True, False]
    assert [c.get("skipped") for c in record["tool_calls"]] == [None, None, None, "call_budget"]
    assert record["stop_reason"] == "answer" and record["forced_reason"] == "calls" and record["tool_calls_used"] == 3


def test_budget_skips_and_per_turn_cap_interplay(row):
    cfg = _forced(tool_call_budget=3)
    backend = ScriptedBackend(["\n".join([CALC_CALL] * 3), "\n".join([CALC_CALL] * 4), ANSWER])
    record = _run(row, cfg, backend)
    cap = "Error: at most 2 tool calls per turn; this call was skipped."
    # turn 0: the budget is not binding, only the per-turn cap is
    assert _last_turn_text(backend.requests[1]) == (
        f"<tool_response>\n3\n</tool_response>\n<tool_response>\n3\n</tool_response>\n<tool_response>\n{cap}\n\n[Tool calls left: 1]\n</tool_response>"
    )
    # turn 1: 1 call left -> run, budget-skip up to the cap, cap-skip beyond it
    assert _last_turn_text(backend.requests[2]) == (
        f"<tool_response>\n3\n</tool_response>\n<tool_response>\n{CALL_BUDGET_SKIPPED}\n</tool_response>\n"
        f"<tool_response>\n{cap}\n</tool_response>\n<tool_response>\n{cap}\n\n{FINAL_ANSWER_NOTICE}\n</tool_response>"
    )
    skipped = [(c["turn"], c.get("skipped")) for c in record["tool_calls"]]
    assert skipped == [(0, None), (0, None), (0, "per_turn_cap"), (1, None), (1, "call_budget"), (1, "per_turn_cap"), (1, "per_turn_cap")]
    assert record["tool_calls_used"] == 3 and record["forced_reason"] == "calls" and record["n_tool_calls"] == 7


def test_forced_by_tokens_keeps_the_reserve(row):
    cfg = _forced(max_total_tokens=1000, max_tokens_per_turn=400, answer_token_reserve=300)
    backend = TokenBackend([CALC_CALL, CALC_CALL, ANSWER], tokens=10_000)  # every turn uses all it may
    record = _run(row, cfg, backend)
    # tool turns hold back the reserve; the forced turn gets everything that is left
    assert [r["max_tokens"] for r in backend.requests] == [400, 300, 300]
    # the forced reply also runs into its limit, so the endpoint reports "length" and so does the episode
    assert record["forced_reason"] == "tokens" and record["forced_turn"] and record["stop_reason"] == "length"
    assert record["completion_tokens"] == 1000
    # an answer that fits in the reserve finishes "stop" and is an answer
    backend = TokenBackend([CALC_CALL, CALC_CALL, ANSWER], tokens=[10_000, 10_000, 50])
    record = _run(row, cfg, backend)
    assert [r["max_tokens"] for r in backend.requests] == [400, 300, 300]
    assert record["forced_reason"] == "tokens" and record["stop_reason"] == "answer" and record["completion_tokens"] == 750
    assert _last_turn_text(backend.requests[1]).endswith("\n\n[Tool calls left: 8]\n</tool_response>")
    assert _last_turn_text(backend.requests[2]).endswith(f"\n\n{FINAL_ANSWER_NOTICE}\n</tool_response>")


def test_forced_token_accounting_never_exceeds_the_cap(row):
    for max_turns, budget, total, per_turn, reserve in [(6, None, 8192, 2048, 1024), (4, 2, 900, 500, 200), (10, 30, 3000, 700, 700), (2, 1, 50, 40, 1)]:
        for tokens in (1, 37, 199, 400, 10_000):
            cfg = _forced(max_turns=max_turns, tool_call_budget=budget, max_total_tokens=total, max_tokens_per_turn=per_turn, answer_token_reserve=reserve)
            backend = TokenBackend([CALC_CALL] * max_turns, tokens=tokens)  # the model never stops calling
            record = _run(row, cfg, backend)
            caps = [r["max_tokens"] for r in backend.requests]
            assert record["completion_tokens"] <= total and all(c > 0 for c in caps), (max_turns, budget, tokens, caps)
            assert record["stop_reason"] == "forced_tool_call" and record["forced_turn"], (max_turns, budget, tokens)
            spent = 0
            for request in backend.requests[:-1]:  # every tool turn left room for the reserve
                assert request["max_tokens"] <= total - spent - reserve
                spent += min(tokens, request["max_tokens"])
            # the forced turn gets everything that is left, up to the per-turn cap, and at least the reserve
            assert caps[-1] == min(per_turn, total - spent) and caps[-1] >= reserve, (max_turns, budget, tokens, caps)
            assert record["tool_calls_used"] <= cfg.call_budget


def test_forced_calculator_condition_using_every_call_is_forced_by_turns(row):
    # calculator_forced: 2 calls in each of 5 turns uses the 10 calls and leaves one turn, together;
    # turns are checked first, so the reason is "turns"
    cfg = build_episode_config(load_conditions()["calculator_forced"], seed=0, family=QWEN3_VL)
    two_calls = CALC_CALL + "\n" + CALC_CALL
    backend = ScriptedBackend([two_calls] * 5 + [ANSWER])
    record = _run(row, cfg, backend)
    notices = [_last_turn_text(r).rsplit("\n\n", 1)[1] for r in backend.requests[1:]]
    assert notices == [f"{n}\n</tool_response>" for n in ("[Tool calls left: 8]", "[Tool calls left: 6]", "[Tool calls left: 4]", "[Tool calls left: 2]", FINAL_ANSWER_NOTICE)]
    assert {k: record[k] for k in FORCED_KEYS} == {"tool_call_budget": 10, "tool_calls_used": 10, "forced_turn": True, "forced_reason": "turns"}
    assert record["stop_reason"] == "answer" and record["turns"] == 6
    assert [r["max_tokens"] for r in backend.requests] == [2048] * 6


@pytest.mark.parametrize(
    "max_turns, budget, reason",
    [
        (6, 4, "calls"),  # calls and tokens run out on turn 2: calls are checked before tokens
        (3, 10, "turns"),  # turns and tokens
        (3, 4, "turns"),  # all three
        (6, 10, "tokens"),  # tokens alone
    ],
)
def test_forced_reason_order_when_limits_run_out_together(row, max_turns, budget, reason):
    # 1,000 tokens, 400 per turn, 300 reserved: the turns spend 400 + 300, so after turn 2 only the reserve is left
    cfg = _forced(max_turns=max_turns, tool_call_budget=budget, max_total_tokens=1000, max_tokens_per_turn=400, answer_token_reserve=300)
    two_calls = CALC_CALL + "\n" + CALC_CALL
    backend = TokenBackend([two_calls, two_calls, ANSWER], tokens=[10_000, 10_000, 50])  # the answer fits in the reserve
    record = _run(row, cfg, backend)
    assert [r["max_tokens"] for r in backend.requests] == [400, 300, 300]
    assert record["tool_calls_used"] == 4 and record["forced_turn"] and record["forced_reason"] == reason
    assert record["stop_reason"] == "answer"


HALLUCINATED = '\n<tool_response>\n{"answer": 9}\n</tool_response>'


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_forced_turn_strips_hallucinated_tool_responses(row, finish):
    # the forced reply invents a tool response with another answer; it is cut off before scoring,
    # and a cut reply counts as an answer even when the endpoint reports "length"
    backend = TokenBackend([CALC_CALL, ANSWER + HALLUCINATED], finishes=["stop", finish])
    record = _run(row, _forced(max_turns=2), backend)
    assert record["final_response"] == ANSWER and record["hallucinated_tool_responses"] == 1
    assert record["stop_reason"] == "answer" and record["forced_turn"] and record["forced_reason"] == "turns"
    assert score_row(record["final_response"], row.answer_gt, row.annotation_gt, row.reward_contract)["strict_correct"] == 1.0
    # with a call before the invented response, the call is recorded (not run) and the stop is forced_tool_call
    backend = TokenBackend([CALC_CALL, "Checking.\n" + CALC_CALL + HALLUCINATED], finishes=["stop", finish])
    record = _run(row, _forced(max_turns=2), backend)
    assert record["final_response"] == "Checking.\n" + CALC_CALL and record["hallucinated_tool_responses"] == 1
    assert record["stop_reason"] == "forced_tool_call" and record["tool_calls"][-1]["skipped"] == "forced_answer"
    assert record["tool_calls_used"] == 1 and record["n_tool_calls"] == 2


def test_forced_budget_counts_calls_that_fail(row):
    # unknown-function and malformed calls count against the budget like calls that run
    unknown = '<tool_call>\n{"name": "nope", "arguments": {}}\n</tool_call>'
    malformed = "<tool_call>\n{not json}\n</tool_call>"
    backend = ScriptedBackend([unknown + "\n" + CALC_CALL, malformed + "\n" + CALC_CALL, ANSWER])
    record = _run(row, _forced(tool_call_budget=3), backend)
    first = _last_turn_text(backend.requests[1])
    assert "unknown function 'nope'" in first and first.endswith("<tool_response>\n3\n\n[Tool calls left: 1]\n</tool_response>")
    # one call left: the malformed call takes it, the valid one is skipped
    second = _last_turn_text(backend.requests[2])
    assert second.startswith("<tool_response>\nError: tool call is not valid JSON")
    assert second.endswith(f"<tool_response>\n{CALL_BUDGET_SKIPPED}\n\n{FINAL_ANSWER_NOTICE}\n</tool_response>")
    assert [(c["executed"], c.get("skipped")) for c in record["tool_calls"]] == [(False, None), (True, None), (False, None), (False, "call_budget")]
    assert record["tool_calls_used"] == 3 and record["forced_reason"] == "calls" and record["stop_reason"] == "answer"


def test_forced_reserve_capped_tool_turn_can_end_without_a_forced_turn(row):
    """Known limit of the forcing rule (EXPERIMENT_DESIGN.md, "Forced final answer"): "tokens" forces
    only when no more than the reserve is left. A tool turn capped below the per-turn limit by the
    reserve that is cut off before any call ends the episode with "length", and the reserve is unused."""
    cfg = build_episode_config(load_conditions()["calculator_forced"], seed=0, family=QWEN3_VL)
    backend = TokenBackend([CALC_CALL] * 4 + ["Let me think about the next step"], tokens=[2048, 2048, 2048, 1020, 600])
    record = _run(row, cfg, backend)
    assert [r["max_tokens"] for r in backend.requests] == [2048, 2048, 2048, 1024, 4]
    assert record["stop_reason"] == "length" and not record["forced_turn"] and record["forced_reason"] is None
    assert record["completion_tokens"] == 7168  # 1,024 of the 8,192 never used


def test_forced_answer_without_calls_and_length_finish(row):
    record = _run(row, _forced(), ScriptedBackend([ANSWER]))
    assert record["stop_reason"] == "answer" and record["turns"] == 1
    assert {k: record[k] for k in FORCED_KEYS} == {"tool_call_budget": 10, "tool_calls_used": 0, "forced_turn": False, "forced_reason": None}
    # a forced turn cut at its token limit without calls reports the finish reason, as unforced turns do
    record = _run(row, _forced(max_turns=2), TokenBackend([CALC_CALL, "I think the answer"], finishes=["stop", "length"]))
    assert record["stop_reason"] == "length" and record["forced_turn"]


def test_unforced_records_have_no_forced_keys(row):
    cfg = EpisodeConfig(family=QWEN3_VL, system_prompt="s", tools=build_tools(["calculator"], QWEN3_VL))
    backend = ScriptedBackend([CALC_CALL, ANSWER])
    record = _run(row, cfg, backend)
    assert set(record) == RECORD_KEYS
    assert backend.requests[0]["messages"][0]["content"].startswith("s\n\n# Tools")
    assert "Tool calls left" not in _last_turn_text(backend.requests[1])
    assert "skipped" not in record["tool_calls"][0]


def test_forced_zoom_with_qwen35_xml(row):
    spec = load_conditions()["zoom_forced"]
    cfg = build_episode_config(spec, seed=0, family=QWEN35)
    xml_zoom = "Zooming in.\n\n<tool_call>\n<function=image_zoom_in_tool>\n<parameter=bbox_2d>\n[0, 0, 400, 400]\n</parameter>\n<parameter=label>\nbox\n</parameter>\n<parameter=img_idx>\n0\n</parameter>\n</function>\n</tool_call>"
    backend = ScriptedBackend([xml_zoom, ANSWER])
    record = _run(row, cfg, backend)
    system = backend.requests[0]["messages"][0]["content"]
    # Qwen3.5 layout: tool block first, then the system prompt with the budget statement
    assert system.startswith(QWEN35_TOOLS_HEADER) and system.endswith(cfg.system_prompt + "\n\n" + budget_statement(10, 2))
    tool_turn = backend.requests[1]["messages"][-1]["content"]
    assert [p["type"] for p in tool_turn] == ["text", "image_url", "text"]
    assert tool_turn[0]["text"] == "<tool_response>\n" and tool_turn[2]["text"] == "\n\n[Tool calls left: 8]\n</tool_response>"
    call = record["tool_calls"][0]
    assert call["ok"] and call["arguments"] == {"bbox_2d": [0, 0, 400, 400], "label": "box", "img_idx": 0}
    assert record["stop_reason"] == "answer" and record["n_images"] == 2 and record["tool_calls_used"] == 1


def test_qwen35_parses_xml_and_trims_tool_bodies(row):
    cfg = build_episode_config(load_conditions()["calculator"], seed=0, family=QWEN35)
    xml_calc = "<tool_call>\n<function=calculator>\n<parameter=expression>\n1 + 2\n</parameter>\n</function>\n</tool_call>"
    backend = ScriptedBackend([xml_calc, ANSWER])
    record = _run(row, cfg, backend)
    assert record["tool_calls"][0]["ok"] and record["tool_calls"][0]["arguments"] == {"expression": "1 + 2"}
    assert backend.requests[1]["messages"][-1]["content"] == [{"type": "text", "text": "<tool_response>\n3\n</tool_response>"}]
    assert set(record) == RECORD_KEYS
    # Hermes JSON is not a Qwen3.5 call outside <tool_call> tags, but inside them it still parses
    record = _run(row, cfg, ScriptedBackend([CALC_CALL, ANSWER]))
    assert record["tool_calls"][0]["ok"]


class EchoTool(Tool):
    """Returns its text unchanged, so a test controls the tool body's edge whitespace."""

    name = "echo"
    description = "Return the text."
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    def run(self, args, ctx):
        return ToolResult(text=args["text"])


EDGED = "  42 \n\n"  # the body the tool returns: whitespace at both ends


@pytest.mark.parametrize(
    "family, call, tool_turn",
    [
        # Qwen3.5: the template trims each role: tool message, so the agent must too (family=cfg.family)
        (QWEN35, f"<tool_call>\n<function=echo>\n<parameter=text>\n{EDGED}\n</parameter>\n</function>\n</tool_call>", "<tool_response>\n42\n</tool_response>"),
        # Hermes families send the body as it is
        (QWEN3_VL, '<tool_call>\n{"name": "echo", "arguments": {"text": "  42 \\n\\n"}}\n</tool_call>', f"<tool_response>\n{EDGED}\n</tool_response>"),
    ],
)
def test_agent_trims_tool_bodies_by_family(row, family, call, tool_turn):
    cfg = EpisodeConfig(family=family, system_prompt="s", tools={"echo": EchoTool(family)})
    backend = ScriptedBackend([call, ANSWER])
    record = _run(row, cfg, backend)
    assert record["tool_calls"][0]["ok"] and record["tool_calls"][0]["arguments"] == {"text": EDGED}
    assert backend.requests[1]["messages"][-1]["content"] == [{"type": "text", "text": tool_turn}]
    if family is QWEN35:  # the whole second request renders like the native role: tool path
        from .test_qwen35 import _native_assistant, _parts, _render

        ours = [{**m, "content": _parts(m["content"])} if isinstance(m["content"], list) else m for m in backend.requests[1]["messages"]]
        schemas = [EchoTool(family).schema()]
        native = [
            {"role": "system", "content": "s"},
            ours[1],
            _native_assistant("", [("echo", {"text": EDGED})]),
            {"role": "tool", "content": EDGED},
        ]
        assert _render(ours) == _render(native, tools=schemas)


@pytest.mark.parametrize(
    "exc, stop_reason, error",
    [
        (RequestRejected("maximum context length exceeded"), "rejected", "maximum context length exceeded"),
        (RuntimeError("boom"), "error", "RuntimeError: boom"),
    ],
)
def test_failed_forced_request_is_not_a_forced_turn(row, exc, stop_reason, error):
    """forced_turn means a forced final turn happened, not only that one was due (forced_next)."""

    class FailForcedTurn(ScriptedBackend):
        failed = None

        async def complete(self, messages, **kw):
            if self.requests:  # the forced turn: its request carries the longest context
                self.failed = {"messages": messages}
                raise exc
            return await super().complete(messages, **kw)

    backend = FailForcedTurn([CALC_CALL])
    record = _run(row, _forced(max_turns=2), backend)
    # the failed request was the forced one: its last tool response holds the final-answer notice
    assert _last_turn_text(backend.failed).endswith(f"\n\n{FINAL_ANSWER_NOTICE}\n</tool_response>")
    assert record["stop_reason"] == stop_reason and record["error"] == error
    assert {k: record[k] for k in FORCED_KEYS} == {"tool_call_budget": 2, "tool_calls_used": 1, "forced_turn": False, "forced_reason": "turns"}
    assert record["turns"] == 1 and [c["executed"] for c in record["tool_calls"]] == [True]
