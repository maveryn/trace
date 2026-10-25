import json

import pyarrow as pa
import pytest

from trace_tool.rl.prepare_verl_data import convert_table
from trace_tool.rl.reward import compute_score, final_assistant_segment

GT = json.dumps({"type": "integer", "value": 3})
EXTRA = {
    "annotation_gt": json.dumps({"type": "bbox", "value": [0, 0, 1, 1]}),
    "reward_contract": json.dumps({"annotation": {"id": "bbox_soft_iou_v0", "type": "bbox"}, "answer": {"id": "answer_exact_match_v0", "type": "integer"}, "reward_contract_version": "v0"}),
}
CALL = '<tool_call>\n{"name": "calculator", "arguments": {"expression": "1+2"}}\n</tool_call>'


@pytest.mark.parametrize(
    "solution",
    [
        'Three.\n{"answer": 3}',
        CALL + "<|im_end|>\n<|im_start|>user\n<tool_response>\n3\n</tool_response><|im_end|>\n<|im_start|>assistant\nSo three.\n{\"answer\": 3}<|im_end|>",
        CALL + "\nuser\n<tool_response>\n3\n</tool_response>\nassistant\nSo three.\n{\"answer\": 3}",  # special tokens stripped
    ],
)
def test_reward_correct(solution):
    out = compute_score("maveryn/trace", solution, GT, EXTRA)
    assert out["acc"] == 1.0 and out["score"] == pytest.approx(1.0)


def test_reward_ignores_answers_before_last_tool_response():
    solution = '{"answer": 3}\n' + CALL + "\nuser\n<tool_response>\n3\n</tool_response>\nassistant\n" + CALL
    out = compute_score("maveryn/trace", solution, GT, EXTRA)
    assert out["acc"] == 0.0 and out["n_tool_calls"] == 2
    assert final_assistant_segment(solution) == CALL


def test_convert_table_shape():
    table = pa.table(
        {
            "images": [[{"bytes": b"png", "path": None}]],
            "prompt_answer": ["<image>Q?"],
            "answer_gt": [GT],
            "annotation_gt": [EXTRA["annotation_gt"]],
            "reward_contract": [EXTRA["reward_contract"]],
            "instance_id": ["a"],
            "domain": ["charts"],
            "task": ["t"],
        }
    )
    (row,) = convert_table(table, system_prompt="SYS", agent="tool_agent", split="train", offset=5).to_pylist()
    assert row["prompt"][0] == {"role": "system", "content": "SYS"}
    assert row["prompt"][1]["content"] == "<image>Q?"
    assert row["reward_model"]["ground_truth"] == GT and row["extra_info"]["index"] == 5
    assert row["agent_name"] == "tool_agent"


def test_reward_literal_tool_response_text_does_not_cut_single_turn():
    solution = '{"answer": 3}\n</tool_response>\nDone. {"answer": 3}'
    assert compute_score("maveryn/trace", solution, GT, EXTRA)["acc"] == 1.0
    assert compute_score("maveryn/trace", solution, GT, {**EXTRA, "num_turns": 2})["acc"] == 1.0
