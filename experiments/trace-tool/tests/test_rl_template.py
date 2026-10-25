"""The RL chat template must render tool turns exactly like the eval prompts."""

import json
import pathlib

import jinja2
import pytest

from trace_tool.models import QWEN25_VL
from trace_tool.prompting import system_message, text_part, tool_response_content, user_prompt_content
from trace_tool.rl.prepare_policy import TEMPLATE_PATH, install_template
from trace_tool.rl.reward import compute_score
from trace_tool.tools import build_tools

from .conftest import REPO, make_image

STOCK = (REPO / "tests" / "data" / "qwen2_5_vl_7b_instruct_chat_template.jinja").read_text()


def _env(template):
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )
    return env.from_string(template)


def _parts(ps):
    return [{"type": "text", "text": p["text"]} if p["type"] == "text" else {"type": "image"} for p in ps]


@pytest.mark.parametrize("n_responses", [1, 2])
def test_verl_style_rendering_matches_eval(n_responses):
    schemas = [t.schema() for t in build_tools(["zoom", "calculator"], QWEN25_VL).values()]
    image = make_image()
    call = '<tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [0, 0, 300, 200], "label": "box", "img_idx": 0}}\n</tool_call>'
    user = _parts(user_prompt_content("<image>Q?", [(image, 0)]))
    bodies = [[{"type": "image", "image": image, "index": 1}], [text_part("42")]][:n_responses]
    # verl: tools via tools=, results as role: tool
    verl_messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": user}, {"role": "assistant", "content": call}]
    verl_messages += [{"role": "tool", "content": _parts(b)} for b in bodies]
    ours = _env(TEMPLATE_PATH.read_text()).render(messages=verl_messages, tools=schemas, add_generation_prompt=True)
    # eval: flattened messages through the stock Qwen2.5-VL template
    eval_messages = [
        system_message("SYS", schemas),
        {"role": "user", "content": user},
        {"role": "assistant", "content": call},
        {"role": "user", "content": _parts(tool_response_content(bodies))},
    ]
    assert ours == _env(STOCK).render(messages=eval_messages, add_generation_prompt=True)


@pytest.mark.parametrize("with_system", [True, False])
def test_plain_conversations_render_like_stock_qwen25(with_system):
    messages = ([{"role": "system", "content": "SYS"}] if with_system else []) + [{"role": "user", "content": "hi"}]
    assert _env(TEMPLATE_PATH.read_text()).render(messages=messages, add_generation_prompt=True) == _env(STOCK).render(
        messages=messages, add_generation_prompt=True
    )


def test_install_template(tmp_path: pathlib.Path):
    (tmp_path / "chat_template.json").write_text(json.dumps({"chat_template": "old"}))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "old", "model_max_length": 1}))
    changed = install_template(tmp_path, "NEW")
    assert changed == ["chat_template.json", "tokenizer_config.json"]
    assert json.loads((tmp_path / "chat_template.json").read_text())["chat_template"] == "NEW"
    assert json.loads((tmp_path / "tokenizer_config.json").read_text()) == {"chat_template": "NEW", "model_max_length": 1}


def test_reward_refuses_multi_turn_without_boundary():
    gt = json.dumps({"type": "integer", "value": 3})
    extra = {
        "annotation_gt": json.dumps({"type": "bbox", "value": [0, 0, 1, 1]}),
        "reward_contract": json.dumps({"annotation": {"id": "bbox_soft_iou_v0", "type": "bbox"}, "answer": {"id": "answer_exact_match_v0", "type": "integer"}, "reward_contract_version": "v0"}),
        "num_turns": 4,
    }
    # what a Qwen2.5-VL policy without the tool template would produce
    rollout = '{"answer": 3}<tool_call>\n{"name": "calculator", "arguments": {}}\n</tool_call>\ntool\n3\nassistant\nno answer'
    with pytest.raises(ValueError, match="turn boundary"):
        compute_score("maveryn/trace", rollout, gt, extra)
