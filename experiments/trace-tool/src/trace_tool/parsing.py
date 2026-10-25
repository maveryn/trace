"""Parse tool calls from model output.

Hermes-style ``<tool_call>{...}</tool_call>`` blocks (Qwen2.5-VL, Qwen3-VL):
:func:`parse_tool_calls`. Qwen3 XML blocks (Qwen3.5):
:func:`parse_xml_tool_calls`::

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    value
    </parameter>
    </function>
    </tool_call>
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

try:  # Qwen-Agent parses tool calls with json5; optional here
    import json5  # type: ignore
except ImportError:  # pragma: no cover
    json5 = None

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
HALLUCINATED_RESPONSE_MARKER = "<tool_response>"


@dataclass
class ParsedCall:
    name: str | None
    arguments: dict[str, Any] | None
    raw: str
    error: str | None = None


def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if json5 is None:
            raise
        return json5.loads(text)


def _parse_one(raw: str) -> ParsedCall:
    body = raw.strip()
    try:
        payload = _loads(body)
    except Exception as exc:
        return ParsedCall(None, None, raw, f"Error: tool call is not valid JSON ({exc})")
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        return ParsedCall(None, None, raw, 'Error: tool call must be {"name": <function-name>, "arguments": {...}}')
    arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = _loads(arguments) if arguments.strip() else {}
        except Exception as exc:
            return ParsedCall(payload["name"], None, raw, f"Error: arguments are not valid JSON ({exc})")
    if not isinstance(arguments, dict):
        return ParsedCall(payload["name"], None, raw, "Error: arguments must be a JSON object")
    return ParsedCall(payload["name"], arguments, raw)


def parse_tool_calls(text: str) -> list[ParsedCall]:
    calls = [_parse_one(m.group(1)) for m in _TOOL_CALL_RE.finditer(text)]
    tail = text[text.rfind("</tool_call>") + len("</tool_call>") :] if calls else text
    if "<tool_call>" in tail:  # opened but never closed (usually a length cut-off)
        raw = tail[tail.index("<tool_call>") + len("<tool_call>") :]
        calls.append(ParsedCall(None, None, raw, "Error: unterminated <tool_call> block"))
    return calls


def strip_hallucinated_responses(text: str) -> tuple[str, bool]:
    """Cut the text where the model starts writing its own ``<tool_response>``."""
    cut = text.find(HALLUCINATED_RESPONSE_MARKER)
    if cut < 0:
        return text, False
    return text[:cut].rstrip("\n"), True


# --- Qwen3 XML (Qwen3.5) -------------------------------------------------------
# Modelled on vLLM's Qwen3 parser (vllm/parser/qwen3.py): one wrapping newline is
# stripped from each value, and a missing </function> at the end of a <tool_call>
# block is tolerated. In vLLM a parameter value ends only at </parameter> or at the
# next <parameter=; here it also ends at the end of the function body (</function>,
# the next <function=, or the end of the block).
#
# Known differences from vLLM 0.30.0 (Qwen3Parser, non-streaming). They were found by
# running both parsers on hand-written and fuzzed inputs, so the list may not be
# exhaustive. The server runs without a tool parser, so these rules alone decide what
# is a call. tests/test_parsing.py pins each one (test_xml_documented_differences_from_vllm
# and the tests after it); EXPERIMENT_DESIGN.md section 3 repeats the list.
# Block structure:
# - A <tool_call> without its </tool_call> is "Error: unterminated <tool_call> block",
#   as in the Hermes parser, even when its <function=...>...</function> is complete,
#   and also when it was cut mid-value. vLLM returns the call (minus a cut-off value).
# - A bare <function=...>...</function> (no <tool_call> around it) counts as a call
#   only when the text holds no <tool_call> at all. Next to a <tool_call> block it is
#   ignored: no response, no error, and it is not counted. vLLM returns both.
# - A bare block needs its </function>; without one the text has no call, so the turn
#   is an answer. vLLM returns a call.
# - <function=A> without </function>, then <function=B> in the same <tool_call>: two
#   calls here. vLLM merges them into one call to A with the parameters of both (B's
#   value wins for a key both have).
# - A <tool_call> with no <function=NAME> in it (prose, "<function = NAME>") or with an
#   empty name gets XML_CALL_FORMAT_ERROR; vLLM finds no call. A <function=NAME whose
#   ">" is missing gets the same error; vLLM returns a call with a garbled name.
# - A literal </tool_call> inside a value ends the block here, so the value is cut
#   there; vLLM keeps it in the value.
# - Parameter names are stripped here; vLLM keeps trailing whitespace, so
#   "<parameter=label >" gives the key "label " (which the tool would reject).
# - More lenient than vLLM: a Hermes JSON call inside <tool_call> is parsed (vLLM finds
#   none, also when an XML call follows), and a last parameter missing its </parameter>
#   keeps its value, with or without </function> (vLLM drops that parameter).
# Typing (vLLM never fails here: a value it cannot convert stays the raw string and
# the tool rejects it):
# - A value that does not fit its schema type is a parse error here ("Error: parameter
#   'angle' must be a number"). vLLM passes the raw string ("90 degrees", "inf"), or,
#   against a number, null for "null" and true for "true".
# - Numbers: "1e3" or "90.0" are floats here (vLLM: int; the tools accept integral
#   floats). "1_000" is an error here (vLLM: 1000, as Python's int()/float() read it).
# - Arrays and objects: json5 is the fallback here (trailing commas, single quotes,
#   hex), and non-finite numbers (1e999, Infinity, NaN) are kept as float inf/nan.
#   vLLM keeps the raw string in both cases.
# - Integers and booleans (no shipped tool has one): vLLM also reads "2.5" for an
#   integer, and "1"/"0" for a boolean; both are errors here.
_FUNCTION_RE = re.compile(r"<function=([^>]*)>(.*?)(?:</function>|(?=<function=)|\Z)", re.DOTALL)
_BARE_FUNCTION_RE = re.compile(r"<function=([^>]*)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(
    r"<\s*parameter\s*=\s*([^>]*)>(.*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=)|\Z)", re.DOTALL
)
_INT_RE = re.compile(r"[+-]?\d+")
_NUMBER_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")

XML_CALL_FORMAT_ERROR = (
    "Error: a tool call must look like <tool_call>\n<function=NAME>\n<parameter=KEY>\nvalue\n</parameter>\n"
    "</function>\n</tool_call>"
)


class _ParamError(ValueError):
    pass


def _trim_wrapping_newlines(value: str) -> str:
    """One leading and one trailing newline belong to the template's markup (as in vLLM)."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _schema_type(prop: Any) -> str | None:
    if not isinstance(prop, Mapping):
        return None
    kind = prop.get("type")
    if isinstance(kind, list):  # e.g. ["number", "null"]: the first non-null type
        kind = next((k for k in kind if k != "null"), None)
    if kind is None and "enum" in prop:
        return "string"
    return kind if isinstance(kind, str) else None


def _typed_value(key: str, raw: str, prop: Any) -> Any:
    """Schema-guided typing of one parameter value; strings (and enums) stay exactly as written."""
    kind = _schema_type(prop)
    text = raw.strip()
    if kind in ("array", "object"):
        noun = "a JSON array" if kind == "array" else "a JSON object"
        try:
            return _loads(text)
        except Exception:
            raise _ParamError(f"Error: parameter '{key}' must be {noun}") from None
    if kind == "number":
        if _INT_RE.fullmatch(text):
            return int(text)
        if _NUMBER_RE.fullmatch(text) and math.isfinite(float(text)):
            return float(text)
        raise _ParamError(f"Error: parameter '{key}' must be a number")
    if kind == "integer":
        if _INT_RE.fullmatch(text):
            return int(text)
        raise _ParamError(f"Error: parameter '{key}' must be an integer")
    if kind == "boolean":
        if text.lower() in ("true", "false"):
            return text.lower() == "true"
        raise _ParamError(f"Error: parameter '{key}' must be true or false")
    return raw  # string, enum, unknown key or unknown function: the tool reports any problem


def _parse_xml_function(name: str, body: str, raw: str, param_schemas: Mapping[str, Mapping]) -> ParsedCall:
    name = name.strip()
    if not name or re.search(r"[\s<]", name):  # empty, or the ">" closing <function=NAME was missing
        return ParsedCall(None, None, raw, XML_CALL_FORMAT_ERROR)
    schema = param_schemas.get(name)
    properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
    arguments: dict[str, Any] = {}
    for match in _PARAM_RE.finditer(body):
        key = match.group(1).strip()
        value = _trim_wrapping_newlines(match.group(2))
        try:
            arguments[key] = _typed_value(key, value, properties.get(key)) if schema is not None else value
        except _ParamError as exc:
            return ParsedCall(name, None, raw, str(exc))
    return ParsedCall(name, arguments, raw)


def _is_hermes_call(raw: str) -> bool:
    """A Hermes JSON call, a JSON object with a string "name", inside the XML format's tags."""
    body = raw.strip()
    if not body.startswith("{"):
        return False
    try:
        payload = _loads(body)
    except Exception:
        return False
    return isinstance(payload, dict) and isinstance(payload.get("name"), str)


def _parse_xml_block(raw: str, param_schemas: Mapping[str, Mapping]) -> list[ParsedCall]:
    functions = list(_FUNCTION_RE.finditer(raw))
    if not functions:
        # Models sometimes mix in a Hermes call; it is parsed with the Hermes rules. Anything
        # else (invalid or truncated JSON, an object without a "name") gets the XML-worded
        # error: the model sees it, and a Hermes-worded one would point it at the wrong format.
        if _is_hermes_call(raw):
            return [_parse_one(raw)]
        return [ParsedCall(None, None, raw, XML_CALL_FORMAT_ERROR)]
    return [_parse_xml_function(m.group(1), m.group(2), raw, param_schemas) for m in functions]


def parse_xml_tool_calls(text: str, param_schemas: Mapping[str, Mapping]) -> list[ParsedCall]:
    """Qwen3 XML tool calls; ``param_schemas`` maps tool name -> its JSON-schema ``parameters``.

    Values are typed by the parameter's schema type (arrays/objects as JSON, numbers,
    integers, booleans); strings stay as written, so a label "123" is not a number.
    Unknown functions and keys keep raw strings: the agent reports unknown functions
    and the tools validate their arguments. Where this differs from vLLM's Qwen3
    parser is listed in the comment above ``_FUNCTION_RE``.
    """
    if "<tool_call>" not in text:  # complete bare <function=...> blocks, only without any wrapper (see above)
        return [_parse_xml_function(m.group(1), m.group(2), m.group(0), param_schemas) for m in _BARE_FUNCTION_RE.finditer(text)]
    calls: list[ParsedCall] = []
    for m in _TOOL_CALL_RE.finditer(text):
        calls.extend(_parse_xml_block(m.group(1), param_schemas))
    tail = text[text.rfind("</tool_call>") + len("</tool_call>") :] if calls else text
    if "<tool_call>" in tail:  # opened but never closed (usually a length cut-off)
        raw = tail[tail.index("<tool_call>") + len("<tool_call>") :]
        calls.append(ParsedCall(None, None, raw, "Error: unterminated <tool_call> block"))
    return calls
