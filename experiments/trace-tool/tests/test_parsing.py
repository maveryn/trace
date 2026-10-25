import math

import pytest

from trace_tool.models import QWEN35
from trace_tool.parsing import XML_CALL_FORMAT_ERROR, parse_tool_calls, parse_xml_tool_calls, strip_hallucinated_responses
from trace_tool.tools import build_tools


def test_single_call():
    text = 'Let me zoom.\n<tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [1, 2, 3, 4], "label": "a", "img_idx": 0}}\n</tool_call>'
    (call,) = parse_tool_calls(text)
    assert call.error is None and call.name == "image_zoom_in_tool"
    assert call.arguments["bbox_2d"] == [1, 2, 3, 4]


def test_multiple_and_string_arguments():
    text = (
        '<tool_call>\n{"name": "calculator", "arguments": "{\\"expression\\": \\"1+1\\"}"}\n</tool_call>\n'
        '<tool_call>\n{"name": "ocr", "arguments": {"img_idx": 0}}\n</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["calculator", "ocr"]
    assert calls[0].arguments == {"expression": "1+1"}


def test_malformed_and_unterminated():
    calls = parse_tool_calls("<tool_call>\n{not json}\n</tool_call>\nthen <tool_call>\n{\"name\": \"ocr\"")
    assert len(calls) == 2
    assert calls[0].error and "valid JSON" in calls[0].error
    assert calls[1].error == "Error: unterminated <tool_call> block"


def test_no_calls():
    assert parse_tool_calls('The answer is 3.\n{"answer": 3}') == []


def test_strip_hallucinated_response():
    text = '<tool_call>\n{"name": "ocr", "arguments": {}}\n</tool_call>\n<tool_response>\nfake\n</tool_response>'
    cut, flagged = strip_hallucinated_responses(text)
    assert flagged and cut.endswith("</tool_call>")


# --- Qwen3 XML (Qwen3.5) -----------------------------------------------------------

SCHEMAS = {t.schema()["function"]["name"]: t.schema()["function"]["parameters"] for t in build_tools(["all"], QWEN35).values()}
TYPED = {  # one parameter of every JSON-schema type
    "typed": {
        "type": "object",
        "properties": {
            "n": {"type": "number"},
            "i": {"type": "integer"},
            "b": {"type": "boolean"},
            "o": {"type": "object"},
            "a": {"type": "array"},
            "s": {"type": "string"},
            "e": {"enum": ["x", "1"]},
            "opt": {"type": ["integer", "null"]},
        },
    }
}


def _xml(name, *params, close_function=True):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params)
    return f"<tool_call>\n<function={name}>\n{body}" + ("</function>\n" if close_function else "") + "</tool_call>"


def _one(text, schemas=SCHEMAS):
    (call,) = parse_xml_tool_calls(text, schemas)
    return call


def test_xml_single_call_is_typed_by_the_schema():
    text = "Let me zoom.\n\n" + _xml("image_zoom_in_tool", ("bbox_2d", "[0, 0, 500, 500]"), ("label", "123"), ("img_idx", "0"))
    call = _one(text)
    assert call.error is None and call.name == "image_zoom_in_tool"
    assert call.arguments == {"bbox_2d": [0, 0, 500, 500], "label": "123", "img_idx": 0}
    assert isinstance(call.arguments["label"], str) and isinstance(call.arguments["img_idx"], int)


def test_xml_strings_are_never_json_decoded():
    assert _one(_xml("calculator", ("expression", "3"))).arguments == {"expression": "3"}
    assert _one(_xml("calculator", ("expression", '"quoted"'))).arguments == {"expression": '"quoted"'}
    assert _one(_xml("image_flip_tool", ("direction", "horizontal"), ("img_idx", "1"))).arguments == {"direction": "horizontal", "img_idx": 1}


@pytest.mark.parametrize(
    "key, raw, value",
    [
        ("n", "3", 3),
        ("n", "-2", -2),
        ("n", " 2.5 ", 2.5),
        ("n", "1e3", 1000.0),
        ("n", ".5", 0.5),
        ("i", "+7", 7),
        ("b", "TRUE", True),
        ("b", "false", False),
        ("o", '{"k": [1, 2]}', {"k": [1, 2]}),
        ("a", "[1, [2, 3]]", [1, [2, 3]]),
        ("s", "  keep  spaces ", "  keep  spaces "),
        ("e", "1", "1"),
        ("opt", "4", 4),
        ("unknown_key", "[1]", "[1]"),  # keys absent from the schema stay raw strings
    ],
)
def test_xml_typing(key, raw, value):
    call = _one(_xml("typed", (key, raw)), TYPED)
    assert call.error is None and call.arguments == {key: value} and type(call.arguments[key]) is type(value)


@pytest.mark.parametrize(
    "key, raw, message",
    [
        ("n", "abc", "Error: parameter 'n' must be a number"),
        ("n", "inf", "Error: parameter 'n' must be a number"),
        ("n", "1_000", "Error: parameter 'n' must be a number"),
        ("i", "2.5", "Error: parameter 'i' must be an integer"),
        ("b", "yes", "Error: parameter 'b' must be true or false"),
        ("o", "{not json", "Error: parameter 'o' must be a JSON object"),
        ("a", "0, 0, 1, 1", "Error: parameter 'a' must be a JSON array"),
    ],
)
def test_xml_typing_errors(key, raw, message):
    call = _one(_xml("typed", (key, raw)), TYPED)
    assert call.name == "typed" and call.arguments is None and call.error == message


def test_xml_bbox_error_wording():
    call = _one(_xml("image_zoom_in_tool", ("bbox_2d", "0, 0, 500, 500"), ("label", "x"), ("img_idx", "0")))
    assert call.error == "Error: parameter 'bbox_2d' must be a JSON array"


def test_xml_json5_fallback_for_arrays():
    pytest.importorskip("json5")
    assert _one(_xml("ocr", ("img_idx", "0"), ("bbox_2d", "[0, 0, 10, 10,]"))).arguments == {"img_idx": 0, "bbox_2d": [0, 0, 10, 10]}


def test_xml_unknown_function_keeps_raw_strings():
    call = _one(_xml("nope", ("img_idx", "0"), ("bbox_2d", "[1, 2]")))
    assert call.error is None and call.name == "nope" and call.arguments == {"img_idx": "0", "bbox_2d": "[1, 2]"}


def test_xml_multi_line_values_lose_only_one_wrapping_newline():
    expression = "(1 + 2)\n* 3"
    assert _one(_xml("calculator", ("expression", expression))).arguments == {"expression": expression}
    text = "<tool_call>\n<function=calculator>\n<parameter=expression>\n\n1+1\n\n</parameter>\n</function>\n</tool_call>"
    assert _one(text).arguments == {"expression": "\n1+1\n"}
    inline = "<tool_call><function=calculator><parameter=expression>2*3</parameter></function></tool_call>"
    assert _one(inline).arguments == {"expression": "2*3"}


def test_xml_missing_closing_tags():
    # </parameter> missing before the next <parameter=, before </function>, and </function> missing at block end
    text = (
        "<tool_call>\n<function=image_zoom_in_tool>\n<parameter=bbox_2d>\n[0, 0, 10, 10]\n"
        "<parameter=label>\nbox\n<parameter=img_idx>\n0\n</function>\n</tool_call>"
    )
    assert _one(text).arguments == {"bbox_2d": [0, 0, 10, 10], "label": "box", "img_idx": 0}
    no_function_end = _xml("calculator", ("expression", "1+1"), close_function=False)
    assert _one(no_function_end).arguments == {"expression": "1+1"}
    bare_param = "<tool_call>\n<function=calculator>\n<parameter=expression>\n4/2\n</tool_call>"
    assert _one(bare_param).arguments == {"expression": "4/2"}


def test_xml_tolerates_spaces_in_parameter_tags():
    text = "<tool_call>\n<function= calculator >\n< parameter = expression >\n5\n< / parameter >\n</function>\n</tool_call>"
    call = _one(text)
    assert call.error is None and call.name == "calculator" and call.arguments == {"expression": "5"}


def test_xml_multiple_calls_and_functions_in_one_block():
    text = _xml("calculator", ("expression", "1")) + "\n" + _xml("ocr", ("img_idx", "0"))
    assert [(c.name, c.arguments) for c in parse_xml_tool_calls(text, SCHEMAS)] == [("calculator", {"expression": "1"}), ("ocr", {"img_idx": 0})]
    two = "<tool_call>\n<function=calculator>\n<parameter=expression>\n1\n</parameter>\n</function>\n<function=ocr>\n<parameter=img_idx>\n0\n</parameter>\n</function>\n</tool_call>"
    assert [c.name for c in parse_xml_tool_calls(two, SCHEMAS)] == ["calculator", "ocr"]


def test_xml_json_inside_tool_call_uses_hermes_rules():
    text = '<tool_call>\n{"name": "calculator", "arguments": {"expression": "1+1"}}\n</tool_call>'
    call = _one(text)
    assert call.error is None and call.name == "calculator" and call.arguments == {"expression": "1+1"}
    # Hermes rules apply to a JSON object with a string "name"; their argument errors name no call format
    bad_args = _one('<tool_call>\n{"name": "calculator", "arguments": "{\\"expression\\": "}\n</tool_call>')
    assert bad_args.name == "calculator" and bad_args.error.startswith("Error: arguments are not valid JSON")
    assert _one('<tool_call>\n{"name": "calculator", "arguments": [1]}\n</tool_call>').error == "Error: arguments must be a JSON object"


@pytest.mark.parametrize(
    "content",
    [
        "{not json}",
        "{calculator}",
        '{"name": "calculator", "arguments": {"expression": "1+1"}',  # truncated
        '{"function": "calculator", "parameters": {"expression": "1+1"}}',  # an object, but no "name"
        '{"name": 3, "arguments": {}}',
        "{}",
    ],
)
def test_xml_json_that_is_not_a_hermes_call_gets_the_xml_error(content):
    # the model sees this error: it must show the XML format, never the Hermes JSON one
    call = _one(f"<tool_call>\n{content}\n</tool_call>")
    assert call.name is None and call.error == XML_CALL_FORMAT_ERROR


def test_xml_bare_function_without_wrapper():
    bare = "I will compute.\n<function=calculator>\n<parameter=expression>\n2+2\n</parameter>\n</function>"
    call = _one(bare)
    assert call.error is None and call.name == "calculator" and call.arguments == {"expression": "2+2"}
    assert parse_xml_tool_calls("<function=calculator>\n<parameter=expression>\n2+2", SCHEMAS) == []  # never closed
    # with a <tool_call> wrapper anywhere, bare blocks are not calls
    mixed = _xml("ocr", ("img_idx", "0")) + "\n" + bare
    assert [c.name for c in parse_xml_tool_calls(mixed, SCHEMAS)] == ["ocr"]


def test_xml_malformed_blocks_get_xml_worded_errors():
    for block in ("<tool_call>\ncalculator(1+1)\n</tool_call>", "<tool_call>\n</tool_call>", "<tool_call>\n<function=>\n</function>\n</tool_call>"):
        call = _one(block)
        assert call.name is None and call.error == XML_CALL_FORMAT_ERROR
    assert "<function=NAME>" in XML_CALL_FORMAT_ERROR and "<parameter=KEY>" in XML_CALL_FORMAT_ERROR
    unclosed_name = _one("<tool_call>\n<function=calculator\n<parameter=expression>\n1\n</parameter>\n</function>\n</tool_call>")
    assert unclosed_name.error == XML_CALL_FORMAT_ERROR


def test_xml_unterminated_blocks():
    cut = "Zooming.\n<tool_call>\n<function=image_zoom_in_tool>\n<parameter=bbox_2d>\n[0, 0,"
    (call,) = parse_xml_tool_calls(cut, SCHEMAS)
    assert call.name is None and call.error == "Error: unterminated <tool_call> block"
    calls = parse_xml_tool_calls(_xml("calculator", ("expression", "1")) + "\n<tool_call>\n<function=ocr>", SCHEMAS)
    assert [c.error for c in calls] == [None, "Error: unterminated <tool_call> block"]


CALC_FUNCTION = "<function=calculator>\n<parameter=expression>\n1+2\n</parameter>\n</function>"


@pytest.mark.parametrize(
    "text, expected",
    [
        # a complete function whose </tool_call> is missing is unterminated (vLLM runs the call)
        ("<tool_call>\n" + CALC_FUNCTION + "\n", [(None, None, "Error: unterminated <tool_call> block")]),
        # cut mid-value: unterminated (vLLM runs calculator({}))
        ("<tool_call>\n<function=calculator>\n<parameter=expression>\n1+", [(None, None, "Error: unterminated <tool_call> block")]),
        # a bare block next to a <tool_call> block is ignored (vLLM runs both)
        (_xml("ocr", ("img_idx", "0")) + "\n" + CALC_FUNCTION.replace("1+2", "3+4"), [("ocr", {"img_idx": 0}, None)]),
        # a bare block needs its </function> (vLLM returns the call)
        ("<function=calculator>\n<parameter=expression>\n1+2\n</parameter>\n", []),
        ("I will compute.\n<function=calculator>\n<parameter=expression>\n1+2\n</parameter>\n", []),
        # more lenient than vLLM: a Hermes call inside <tool_call>, and a last value with no closing tags
        ('<tool_call>\n{"name": "calculator", "arguments": {"expression": "1+1"}}\n</tool_call>', [("calculator", {"expression": "1+1"}, None)]),
        ("<tool_call>\n<function=calculator>\n<parameter=expression>\n4/2\n</tool_call>", [("calculator", {"expression": "4/2"}, None)]),
        # number-typed "90.0" and "1e3" stay floats (vLLM: 90, 1000); the tools accept integral floats
        (_xml("image_rotate_tool", ("angle", "90.0"), ("img_idx", "0")), [("image_rotate_tool", {"angle": 90.0, "img_idx": 0}, None)]),
        (_xml("image_rotate_tool", ("angle", "1e3"), ("img_idx", "0")), [("image_rotate_tool", {"angle": 1000.0, "img_idx": 0}, None)]),
        # a last parameter missing only its </parameter> keeps its value (vLLM: calculator({}))
        ("<tool_call>\n<function=calculator>\n<parameter=expression>\n6 * 7\n</function>\n</tool_call>", [("calculator", {"expression": "6 * 7"}, None)]),
        # <function=A> without </function>, then <function=B> in one <tool_call>: two calls
        # (vLLM: one call, calculator({"expression": "1+2", "img_idx": "0"}))
        (
            "<tool_call>\n<function=calculator>\n<parameter=expression>\n1+2\n</parameter>\n"
            "<function=ocr>\n<parameter=img_idx>\n0\n</parameter>\n</function>\n</tool_call>",
            [("calculator", {"expression": "1+2"}, None), ("ocr", {"img_idx": 0}, None)],
        ),
        # no usable <function=NAME> inside <tool_call>: an error that shows the format (vLLM: no call;
        # for the missing ">", a call named "calculator\n<parameter=expression")
        ("<tool_call>\nI want to compute 6*7\n</tool_call>", [(None, None, XML_CALL_FORMAT_ERROR)]),
        ("<tool_call>\n<function = calculator>\n<parameter=expression>\n1\n</parameter>\n</function>\n</tool_call>", [(None, None, XML_CALL_FORMAT_ERROR)]),
        ("<tool_call>\n<function=>\n<parameter=expression>\n1\n</parameter>\n</function>\n</tool_call>", [(None, None, XML_CALL_FORMAT_ERROR)]),
        ("<tool_call>\n<function=calculator\n<parameter=expression>\n1\n</parameter>\n</function>\n</tool_call>", [(None, None, XML_CALL_FORMAT_ERROR)]),
        # a literal </tool_call> inside a value ends the block (vLLM: expression "1 </tool_call> 2")
        (_xml("calculator", ("expression", "1 </tool_call> 2")), [("calculator", {"expression": "1 "}, None)]),
        # parameter names are stripped (vLLM keeps the key "expression ")
        ("<tool_call>\n<function=calculator>\n<parameter=expression >\n6 * 7\n</parameter>\n</function>\n</tool_call>", [("calculator", {"expression": "6 * 7"}, None)]),
        # a Hermes call next to an XML call: both count (vLLM: only the XML one)
        (
            '<tool_call>\n{"name": "calculator", "arguments": {"expression": "1"}}\n</tool_call>\n' + _xml("calculator", ("expression", "2")),
            [("calculator", {"expression": "1"}, None), ("calculator", {"expression": "2"}, None)],
        ),
        # a value that does not fit its schema type is an error (vLLM passes "90 degrees", "inf",
        # null, true, and reads "1_000" as 1000)
        (_xml("image_rotate_tool", ("angle", "90 degrees"), ("img_idx", "0")), [("image_rotate_tool", None, "Error: parameter 'angle' must be a number")]),
        (_xml("image_rotate_tool", ("angle", "inf"), ("img_idx", "0")), [("image_rotate_tool", None, "Error: parameter 'angle' must be a number")]),
        (_xml("image_rotate_tool", ("angle", "90"), ("img_idx", "null")), [("image_rotate_tool", None, "Error: parameter 'img_idx' must be a number")]),
        (_xml("image_rotate_tool", ("angle", "true"), ("img_idx", "0")), [("image_rotate_tool", None, "Error: parameter 'angle' must be a number")]),
        (_xml("image_rotate_tool", ("angle", "1_000"), ("img_idx", "0")), [("image_rotate_tool", None, "Error: parameter 'angle' must be a number")]),
        (_xml("ocr", ("img_idx", "0"), ("bbox_2d", "(0, 0, 10, 10)")), [("ocr", None, "Error: parameter 'bbox_2d' must be a JSON array")]),
        # non-finite numbers in an array are kept (vLLM: the raw string "[0, 0, 1e999, 500]")
        (_xml("ocr", ("img_idx", "0"), ("bbox_2d", "[0, 0, 1e999, 500]")), [("ocr", {"img_idx": 0, "bbox_2d": [0, 0, math.inf, 500]}, None)]),
    ],
)
def test_xml_documented_differences_from_vllm(text, expected):
    """The known differences from vLLM 0.30.0's Qwen3 parser, listed in parsing.py and
    EXPERIMENT_DESIGN.md (section 3); change those together with this test and the two after it.
    The vLLM results in the comments were produced by vLLM 0.30.0's Qwen3Parser.extract_tool_calls."""
    calls = parse_xml_tool_calls(text, SCHEMAS)
    got = [(c.name, c.arguments, c.error) for c in calls]
    assert got == expected
    assert repr(got) == repr(expected)  # also the types: 90.0 is not 90


@pytest.mark.parametrize(
    "raw, value",
    [
        ("[0, 0, 500, 500,]", [0, 0, 500, 500]),  # trailing comma
        ("['0', '0', '5', '5']", ["0", "0", "5", "5"]),  # single quotes
        ("[0x10, 0, 500, 500]", [16, 0, 500, 500]),  # hex
    ],
)
def test_xml_documented_json5_differences_from_vllm(raw, value):
    """Arrays and objects fall back to json5 here; vLLM keeps the raw string."""
    pytest.importorskip("json5")
    call = _one(_xml("ocr", ("img_idx", "0"), ("bbox_2d", raw)))
    assert call.error is None and call.arguments == {"img_idx": 0, "bbox_2d": value}


@pytest.mark.parametrize(
    "key, raw, message",
    [
        ("i", "2.5", "Error: parameter 'i' must be an integer"),  # vLLM: 2.5
        ("i", "1e0", "Error: parameter 'i' must be an integer"),  # vLLM: 1.0
        ("b", "1", "Error: parameter 'b' must be true or false"),  # vLLM: true
        ("b", "0", "Error: parameter 'b' must be true or false"),  # vLLM: false
    ],
)
def test_xml_documented_integer_boolean_differences_from_vllm(key, raw, message):
    """No shipped tool has integer or boolean parameters; pinned for completeness."""
    call = _one(_xml("typed", (key, raw)), TYPED)
    assert call.arguments is None and call.error == message


def test_xml_no_calls_and_hallucinated_responses():
    assert parse_xml_tool_calls('The answer is 3.\n{"answer": 3}', SCHEMAS) == []
    text = _xml("calculator", ("expression", "1")) + "\n<tool_response>\nfake\n</tool_response>"
    cut, flagged = strip_hallucinated_responses(text)
    assert flagged and cut.endswith("</tool_call>") and len(parse_xml_tool_calls(cut, SCHEMAS)) == 1
