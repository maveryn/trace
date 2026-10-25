import pytest

from trace_tool.data import rows_from_parquet
from trace_tool.scoring import apply_judge, score_row

from .conftest import make_row, validation_parquet


def _score(response, answer=3, answer_type="integer"):
    row = make_row(answer=answer, answer_type=answer_type)
    return score_row(response, row.answer_gt, row.annotation_gt, row.reward_contract)


def test_correct_final_json():
    s = _score('Counting gives three.\n{"answer": 3}')
    assert s["strict_correct"] == 1.0 and s["strict_format"] == 1.0
    assert s["strict_reward"] == pytest.approx(1.0)
    assert s["deterministic_correct"] == 1.0 and not s["judge_needed"]


def test_wrong_and_missing():
    wrong = _score('{"answer": 4}')
    assert wrong["strict_correct"] == 0.0 and wrong["strict_reward"] == pytest.approx(0.05)
    missing = _score("I think it is three")
    assert missing["judge_needed"] and missing["strict_correct"] == 0.0
    assert apply_judge(missing, None, {"type": "integer", "value": 3})["official_source"] == "judge_pending"
    judged = apply_judge(missing, {"status": "ok", "answer": 3}, {"type": "integer", "value": 3})
    assert judged["official_correct"] == 1.0 and judged["official_source"] == "judge"


def test_tool_call_json_before_answer_is_ignored():
    response = '<tool_call>\n{"name": "calculator", "arguments": {"expression": "1+2"}}\n</tool_call>'
    s = _score(response)
    assert s["strict_correct"] == 0.0


def test_string_and_option_answers():
    assert _score('{"answer": "controller radar"}', answer="Controller Radar", answer_type="string")["strict_correct"] == 1.0
    assert _score('{"answer": "C"}', answer="C", answer_type="option_letter")["deterministic_correct"] == 1.0


@pytest.mark.skipif(validation_parquet() is None, reason="set TRACE_VALIDATION_PARQUET to a local copy")
def test_gold_answers_score_perfectly_on_real_rows():
    import json

    rows = rows_from_parquet(validation_parquet())
    assert len(rows) == 2000
    for row in rows:
        response = "Reasoning...\n" + json.dumps({"answer": row.answer_gt["value"]})
        s = score_row(response, row.answer_gt, row.annotation_gt, row.reward_contract)
        assert s["strict_correct"] == 1.0 and s["deterministic_correct"] == 1.0, row.instance_id
