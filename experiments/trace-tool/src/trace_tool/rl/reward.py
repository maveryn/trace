"""verl ``custom_reward_function`` for TRACE, usable for single- and multi-turn rollouts.

Same reward as TRACE's GRPO runs (0.95 * exact_json answer + 0.05 * format),
computed on the final assistant segment only, so a correct ``{"answer": ...}``
in an earlier turn (before more tool calls) earns nothing.

verl config (key names as of verl 0.9; older versions use a top-level
``custom_reward_function``):
    reward.custom_reward_function.path=src/trace_tool/rl/reward.py
    reward.custom_reward_function.name=compute_score

verl decodes the response with ``skip_special_tokens=True``, so turn headers
arrive as bare "user\n" / "assistant\n" lines; both forms are handled.
"""

from __future__ import annotations

import json
import re
from typing import Any

from trace_tool.scoring import strict_score

# Where the chat template hands the turn back to the model after tool results:
# "</tool_response><|im_end|>\n<|im_start|>assistant\n", or with special tokens
# stripped by verl's decode, "</tool_response>\nassistant\n".
_TURN_BOUNDARY_RE = re.compile(r"</tool_response>(?:<\|im_end\|>)?\n(?:<\|im_start\|>)?assistant\n")


def final_assistant_segment(solution_str: str, num_turns: int | None = None) -> str:
    """Text of the last assistant turn (the whole string for single-turn rollouts).

    Only a template-inserted turn boundary counts, so a literal "</tool_response>"
    written by the policy itself does not cut its answer. With ``num_turns``
    from verl's extra_info, single-turn rollouts (<= 2: user + assistant) are
    never cut.
    """
    if num_turns is not None and num_turns <= 2:
        return solution_str.replace("<|im_end|>", "").strip()
    boundaries = list(_TURN_BOUNDARY_RE.finditer(solution_str))
    if num_turns is not None and num_turns > 2 and not boundaries:
        # e.g. a Qwen2.5-VL policy without the tool chat template (rl/prepare_policy.py):
        # scoring the whole rollout would credit answers from earlier turns
        raise ValueError(
            f"multi-turn rollout (num_turns={num_turns}) has no <tool_response> turn boundary; "
            "check the policy's chat template"
        )
    tail = solution_str[boundaries[-1].end() :] if boundaries else solution_str
    return tail.replace("<|im_end|>", "").strip()


def _as_dict(value: Any) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else dict(value)


def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, float]:
    extra_info = extra_info or {}
    num_turns = extra_info.get("num_turns")
    response = final_assistant_segment(solution_str, int(num_turns) if num_turns is not None else None)
    scores = strict_score(
        response,
        _as_dict(ground_truth),
        _as_dict(extra_info["annotation_gt"]),
        _as_dict(extra_info["reward_contract"]),
    )
    n_calls = float(solution_str.count("<tool_call>"))
    return {
        "score": scores["strict_reward"],
        "acc": scores["strict_correct"],
        "format": scores["strict_format"],
        "n_tool_calls": n_calls,
        "used_tool": float(n_calls > 0),
    }
