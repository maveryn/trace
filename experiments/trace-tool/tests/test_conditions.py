"""Settings recorded in run_config.json (resume keys) and the prompts of every condition.

The existing families and conditions are frozen: describe() and the first request
must stay byte-identical to the goldens captured before the Qwen3.5 / forced
changes (tests/data/golden_*_v1.json). New keys appear only where they are set.

The request goldens hold the raw payload sha256 and a host-independent one with the
images as decoded pixels (tests/golden_payload.py): PNG bytes come from the host's zlib,
so the raw hash is checked only where this host's encoder reproduces the captured bytes.
"""

import base64
import copy
import hashlib
import io
import json
import warnings

import pytest
import yaml
from PIL import Image

from trace_tool.agent import build_initial_messages
from trace_tool.conditions import DEFAULT_CONFIG, build_episode_config, describe, load_conditions
from trace_tool.models import QWEN3_VL, QWEN25_VL, QWEN35
from trace_tool.prompting import (
    BUDGET_STATEMENT,
    CALLS_LEFT_NOTICE,
    FINAL_ANSWER_NOTICE,
    FORCED_PROMPT_SHA256,
    ImageURLCache,
    budget_statement,
    to_openai_messages,
)

from .conftest import REPO, make_row
from .golden_payload import fingerprint, reencoded_png_sha256

GOLDEN_DESCRIBE = json.loads((REPO / "tests" / "data" / "golden_describe_v1.json").read_text())
GOLDEN_MESSAGES = json.loads((REPO / "tests" / "data" / "golden_initial_messages_v1.json").read_text())
FAMILIES_V1 = {f.name: f for f in (QWEN25_VL, QWEN3_VL)}
CONDITIONS_V1 = [
    "no_tool", "no_tool_budget", "zoom", "visual_ops", "calculator", "ocr", "all",
    "zoom_cookbook", "oracle_crop", "no_tool_lowres", "zoom_lowres",
]
FORCED_BASES = {"zoom_forced": "zoom", "visual_ops_forced": "visual_ops", "calculator_forced": "calculator", "ocr_forced": "ocr", "all_forced": "all"}
FORCED_SETTINGS = {"forced_answer", "tool_call_budget", "answer_token_reserve", "forced_prompt_sha256"}
FAMILY_SETTINGS = {"tool_format", "chat_template_kwargs"}


def _describe(name, family, conditions=None):
    spec = (conditions or load_conditions())[name]
    settings = describe(spec, build_episode_config(spec, seed=0, family=family))
    settings.pop("ocr_versions", None)  # environment-dependent
    return settings


# --- goldens (constraint: existing families and conditions are byte-identical) -----------


def test_goldens_cover_every_v1_condition_and_family():
    expected = {f"{family}/{name}" for family in FAMILIES_V1 for name in CONDITIONS_V1}
    assert set(GOLDEN_DESCRIBE) == expected and set(GOLDEN_MESSAGES) == expected


@pytest.mark.parametrize("key", sorted(GOLDEN_DESCRIBE))
def test_describe_matches_golden(key):
    family, name = key.split("/")
    assert _describe(name, FAMILIES_V1[family]) == GOLDEN_DESCRIBE[key]


@pytest.mark.parametrize("key", sorted(GOLDEN_MESSAGES))
def test_initial_messages_match_golden(key):
    family, name = key.split("/")
    cfg = build_episode_config(load_conditions()[name], seed=0, family=FAMILIES_V1[family])
    messages = build_initial_messages(make_row(), cfg)[0]
    payload = to_openai_messages(messages, ImageURLCache())
    system = messages[0]["content"] if messages[0]["role"] == "system" else None
    got = fingerprint(payload)
    expected = GOLDEN_MESSAGES[key]
    # on every host: every text byte, where the images are, and their decoded pixels
    assert {
        "n_messages": len(messages),
        "system_sha256": hashlib.sha256(system.encode()).hexdigest() if system is not None else None,
        "payload_pixels_sha256": got["payload_pixels_sha256"],
    } == {k: expected[k] for k in ("n_messages", "system_sha256", "payload_pixels_sha256")}
    # byte for byte (PNG bytes included) wherever this host's PNG encoder writes the capture
    # host's bytes, checked independently of the code under test
    if reencoded_png_sha256(payload) == expected["png_sha256"]:
        assert got["png_sha256"] == expected["png_sha256"] and got["payload_sha256"] == expected["payload_sha256"]
    else:
        warnings.warn(
            f"{key}: this host's zlib/Pillow encode PNGs differently from the host the goldens were captured on; "
            "checked text and decoded pixels only, not the PNG bytes",
            stacklevel=1,
        )


def test_golden_payload_fingerprint_separates_encoder_from_content():
    cfg = build_episode_config(load_conditions()["oracle_crop"], seed=0, family=QWEN25_VL)
    payload = to_openai_messages(build_initial_messages(make_row(), cfg)[0], ImageURLCache())
    got = fingerprint(payload)
    assert len(got["png_sha256"]) == 2 and reencoded_png_sha256(payload) == got["png_sha256"]

    def recoded(payload, edit=None, **save):  # every PNG decoded, optionally edited, and encoded again
        out = copy.deepcopy(payload)
        for part in (p for m in out if isinstance(m["content"], list) for p in m["content"] if p["type"] == "image_url"):
            image = Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",", 1)[1])))
            image = edit(image) if edit else image
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", **save)
            part["image_url"]["url"] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        return out

    # another deflate (as from another zlib): other bytes, same pixels, and the re-encoding check notices
    other = recoded(payload, compress_level=1)
    assert fingerprint(other)["payload_sha256"] != got["payload_sha256"]
    assert fingerprint(other)["payload_pixels_sha256"] == got["payload_pixels_sha256"]
    assert reencoded_png_sha256(other) == got["png_sha256"]  # a default re-encoding restores the bytes
    # one changed pixel or one changed text byte changes the host-independent hash
    def poke(image):
        image = image.copy()
        image.putpixel((0, 0), (0, 0, 0))
        return image

    assert fingerprint(recoded(payload, edit=poke))["payload_pixels_sha256"] != got["payload_pixels_sha256"]
    text = copy.deepcopy(payload)
    text[-1]["content"][-1]["text"] += " "
    assert fingerprint(text)["payload_pixels_sha256"] != got["payload_pixels_sha256"]


# --- new keys only where set ---------------------------------------------------------------


@pytest.mark.parametrize("family", [QWEN25_VL, QWEN3_VL, QWEN35])
def test_unforced_conditions_record_no_forced_keys(family):
    conditions = load_conditions()
    for name in conditions:
        settings = _describe(name, family, conditions)
        if name in FORCED_BASES:
            assert FORCED_SETTINGS <= set(settings), name
        else:  # not one forced key, not only not all of them
            assert not FORCED_SETTINGS & set(settings), name
        assert (FAMILY_SETTINGS <= set(settings)) == (family is QWEN35), name
        if family is not QWEN35:
            assert not FAMILY_SETTINGS & set(settings), name


@pytest.mark.parametrize("family", [QWEN25_VL, QWEN3_VL, QWEN35])
def test_forced_conditions_are_their_base_plus_forced_keys(family):
    conditions = load_conditions()
    for forced, base in FORCED_BASES.items():
        settings = _describe(forced, family, conditions)
        assert {k: v for k, v in settings.items() if k not in FORCED_SETTINGS} == _describe(base, family, conditions)
        assert {k: settings[k] for k in FORCED_SETTINGS} == {
            "forced_answer": True,
            "tool_call_budget": 10,
            "answer_token_reserve": 1024,
            "forced_prompt_sha256": FORCED_PROMPT_SHA256,
        }


def test_qwen35_settings_record_the_tool_format_and_thinking_switch():
    settings = _describe("zoom", QWEN35)
    assert settings["family"] == "qwen3.5" and settings["coords"] == "rel1000" and settings["patch_factor"] == 32
    assert settings["tool_format"] == "qwen3_xml" and settings["chat_template_kwargs"] == {"enable_thinking": False}
    # everything else is what the same condition records for Qwen3-VL (same grid and frame)
    qwen3 = _describe("zoom", QWEN3_VL)
    assert {k: v for k, v in settings.items() if k not in FAMILY_SETTINGS | {"family"}} == {k: v for k, v in qwen3.items() if k != "family"}


def test_forced_prompt_texts_are_pinned():
    assert BUDGET_STATEMENT == (
        "You can make at most {budget} tool calls in total, and at most {per_turn} per turn. Each tool response ends "
        "with the number of tool calls you have left. When none are left, give your final answer without calling any tools."
    )
    assert CALLS_LEFT_NOTICE == "[Tool calls left: {left}]"
    assert FINAL_ANSWER_NOTICE == (
        "[No tool calls left. Do not call any more tools. Give your final answer now, ending your response with a "
        'JSON object in this format: {"answer": ...}]'
    )
    joined = BUDGET_STATEMENT + "\n" + CALLS_LEFT_NOTICE + "\n" + FINAL_ANSWER_NOTICE
    assert FORCED_PROMPT_SHA256 == hashlib.sha256(joined.encode()).hexdigest()
    assert FORCED_PROMPT_SHA256 == "4e8b0b2a231d36f7a9541d9304166a5caa70ad63e2d3b41924b74f8fa258158e"


@pytest.mark.parametrize("family", [QWEN25_VL, QWEN3_VL, QWEN35])
def test_budget_statement_is_in_the_system_message_of_forced_conditions_only(family):
    conditions = load_conditions()
    statement = budget_statement(10, 2)
    for name in ("zoom", "zoom_forced", "all_forced", "no_tool"):
        cfg = build_episode_config(conditions[name], seed=0, family=family)
        system = build_initial_messages(make_row(), cfg)[0][0]["content"]
        assert system.count(statement) == (1 if name.endswith("_forced") else 0), name
        assert (cfg.system_prompt + "\n\n" + statement in system) == name.endswith("_forced"), name


# --- validation of the forced keys ------------------------------------------------------


def _load(tmp_path, conditions, defaults=None):
    base = yaml.safe_load(DEFAULT_CONFIG.read_text())["defaults"]
    path = tmp_path / "conditions.yaml"
    path.write_text(yaml.safe_dump({"defaults": {**base, **(defaults or {})}, "conditions": conditions}))
    return load_conditions(path)


@pytest.mark.parametrize(
    "defaults, condition, match",
    [
        ({"forced_answer": True}, {"tools": ["zoom"]}, "cannot be defaults"),
        ({"tool_call_budget": 4}, {"tools": ["zoom"]}, "cannot be defaults"),
        ({}, {"forced_answer": True}, "non-empty tools"),
        ({}, {"tools": [], "forced_answer": True}, "non-empty tools"),
        ({}, {"tools": ["zoom"], "forced_answer": "yes"}, "true or false"),
        ({}, {"tools": ["zoom"], "tool_call_budget": 4}, "need forced_answer: true"),
        ({}, {"tools": ["zoom"], "forced_answer": False, "answer_token_reserve": 10}, "need forced_answer: true"),
        ({}, {"tools": ["zoom"], "forced_answer": True, "tool_call_budget": 0}, "tool_call_budget"),
        ({}, {"tools": ["zoom"], "forced_answer": True, "tool_call_budget": True}, "tool_call_budget"),
        ({}, {"tools": ["zoom"], "forced_answer": True, "max_calls_per_turn": 0}, "max_calls_per_turn >= 1"),
        # an explicit budget does not hide a per-turn cap below 1
        ({}, {"tools": ["zoom"], "forced_answer": True, "tool_call_budget": 10, "max_calls_per_turn": 0}, "max_calls_per_turn >= 1"),
        ({}, {"tools": ["zoom"], "forced_answer": True, "tool_call_budget": 10, "max_calls_per_turn": -1}, "max_calls_per_turn >= 1"),
        ({}, {"tools": ["zoom"], "forced_answer": True, "tool_call_budget": 10, "max_calls_per_turn": True}, "max_calls_per_turn >= 1"),
        ({}, {"tools": ["zoom"], "forced_answer": True, "answer_token_reserve": 0}, "answer_token_reserve"),
        ({}, {"tools": ["zoom"], "forced_answer": True, "answer_token_reserve": 2049}, "answer_token_reserve"),
        ({}, {"tools": ["zoom"], "forced_answer": True, "max_tokens_per_turn": 8192, "answer_token_reserve": 8192}, "below max_total_tokens"),
        ({}, {"tools": ["zoom"], "forced_answer": True, "max_turns": 1}, "max_turns >= 2"),
        ({}, {"tools": ["zoom"], "forced": True}, "unknown keys"),
    ],
)
def test_forced_key_validation(tmp_path, defaults, condition, match):
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, {"c": condition}, defaults)


def test_forced_defaults_resolve_and_valid_edges(tmp_path):
    conditions = _load(
        tmp_path,
        {
            "default_budget": {"tools": ["zoom"], "forced_answer": True},
            "small": {"tools": ["calculator"], "forced_answer": True, "max_turns": 4, "max_calls_per_turn": 3, "answer_token_reserve": 2048},
            "off": {"tools": ["zoom"], "forced_answer": False},
        },
    )
    settings = _describe("default_budget", QWEN3_VL, conditions)
    assert settings["tool_call_budget"] == 10 and settings["answer_token_reserve"] == 1024
    small = _describe("small", QWEN3_VL, conditions)
    assert small["tool_call_budget"] == 9 and small["answer_token_reserve"] == 2048  # (4 - 1) * 3; reserve == max_tokens_per_turn
    # forced_answer: false is the unforced condition, settings included
    off = _describe("off", QWEN3_VL, conditions)
    assert not FORCED_SETTINGS & set(off) and off == _describe("zoom", QWEN3_VL)


def test_forced_conditions_in_the_config():
    conditions = load_conditions()
    for forced, base in FORCED_BASES.items():
        spec = conditions[forced]
        assert {k: v for k, v in spec.items() if k not in ("forced_answer", "tool_call_budget")} == conditions[base]
        assert spec["forced_answer"] is True and spec["tool_call_budget"] == 10
        assert build_episode_config(spec, seed=0, family=QWEN35).call_budget == 10
