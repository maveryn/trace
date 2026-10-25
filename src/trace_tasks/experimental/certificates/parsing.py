"""Strict JSON, response location, certificate schema validation and F (DESIGN 2.1).

- **Final object**: the last fenced block when the response ends with one (the block opened
  by the fence just before the closing one; only an empty or ``json`` info string counts), or
  else the last balanced ``{...}`` that ends the response. Unlike TRACE's legacy regex this
  never spans several fences (review EX-4: an example fence followed by the answer fence was
  read as one block).
- **Scored object** (read by A, G and M): when a final object exists it *is* the claim: it
  must strict-parse to a dict with a ``certificate`` key, otherwise the response has an
  envelope failure (review CONF-08; a draft is never scored instead). Only when there is no
  final object (trailing prose) is the last balanced ``{...}`` span that strict-parses with a
  ``certificate`` key scored. Trailing prose therefore costs F only.
- **Strict parse**: RFC JSON via ``json.loads`` with duplicate keys *marked* (last value kept,
  as ``json.loads`` does), ``NaN``/``Infinity``/overflowing floats rejected, no Python-literal
  fallback, 16 KiB cap. Nothing is ever ``eval``-ed.
- **Envelope failure** (no scored object, invalid JSON, duplicate key outside certificate
  elements, missing/non-object certificate, wrong kind, missing list, cap exceeded): every M
  check is ``valid_input=False`` and M=0. **Element failure** (mistyped, missing/extra key,
  duplicate key, ``annotation_index`` out of ``[0, n)``): that element stays in n with no credit.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from trace_tasks.core.reward_scoring import (
    _collect_json_candidates,
    _extract_final_json_object,
    _parse_json_like,
)

from .checks import read_bool, read_enum, read_int, read_point
from .geometry import as_finite_float
from .types import MAX_OBJECT_BYTES

MAX_SCAN_CANDIDATES = 64
MAX_JSON_DEPTH = 32
TOP_LEVEL_KEYS = frozenset({"annotation", "certificate", "answer"})


class StrictJSONError(ValueError):
    """Strict parse failure with a stable ``code``: invalid_json, non_finite or budget."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        super().__init__(f"{code}: {message}" if message else code)


class JSONObject(dict):
    """A JSON object from :func:`strict_loads`; remembers keys that appeared more than once."""

    duplicate_keys: Tuple[str, ...] = ()
    duplicate_values: Dict[str, List[Any]]


def _pairs_hook(pairs: List[Tuple[str, Any]]) -> JSONObject:
    obj = JSONObject()
    duplicates: List[str] = []
    values: Dict[str, List[Any]] = {}
    for key, value in pairs:
        if key in obj:
            duplicates.append(key)
            values.setdefault(key, [obj[key]]).append(value)
        obj[key] = value
    obj.duplicate_keys = tuple(duplicates)
    obj.duplicate_values = values
    return obj


def _reject_constant(name: str) -> Any:
    raise StrictJSONError("non_finite", name)


def _parse_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise StrictJSONError("non_finite", text)
    return value


def json_depth(text: str) -> int:
    """Maximum ``[``/``{`` nesting depth outside strings (a linear pre-scan)."""
    depth = deepest = 0
    in_string = escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            deepest = max(deepest, depth)
        elif char in "]}":
            depth = max(0, depth - 1)
    return deepest


def strict_loads(text: str, *, max_bytes: int = MAX_OBJECT_BYTES, max_depth: int = MAX_JSON_DEPTH) -> Any:
    """Parse RFC JSON strictly. Raises :class:`StrictJSONError`; never evaluates anything.

    Caps (deterministic, decided by input size): ``max_bytes`` UTF-8 bytes and ``max_depth``
    levels of nesting (certificates nest at most 4 levels); both raise code ``budget``.
    """
    if not isinstance(text, str):
        raise StrictJSONError("invalid_json", "not a string")
    if len(text) > max_bytes or len(text.encode("utf-8")) > max_bytes:
        raise StrictJSONError("budget", f"object exceeds {max_bytes} bytes")
    if json_depth(text) > max_depth:
        raise StrictJSONError("budget", f"object nests deeper than {max_depth}")
    try:
        return json.loads(
            text,
            object_pairs_hook=_pairs_hook,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except StrictJSONError:
        raise
    except (ValueError, RecursionError, TypeError) as exc:
        raise StrictJSONError("invalid_json", type(exc).__name__) from exc


def duplicate_key_paths(value: Any, path: str = "$") -> List[str]:
    """JSON paths of every duplicated key inside ``value`` (depth-first)."""
    found: List[str] = []
    if isinstance(value, JSONObject):
        found.extend(f"{path}.{key}" for key in dict.fromkeys(value.duplicate_keys))
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(duplicate_key_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(duplicate_key_paths(item, f"{path}[{index}]"))
    return found


def find_brace_spans(text: str) -> List[Tuple[int, int]]:
    """Top-level balanced ``{...}`` spans, same scan as the legacy extractor (string-aware)."""
    spans: List[Tuple[int, int]] = []
    depth = 0
    start: Optional[int] = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append((start, index + 1))
                start = None
    return spans


@dataclass(frozen=True)
class LocatedObject:
    """Where the final and the scored objects are, and why there is none."""

    final_found: bool
    final_text: Optional[str]
    final_value: Any
    final_error: Optional[str]
    scored: Optional[JSONObject]
    scored_text: Optional[str]
    scored_is_final: bool
    envelope_error: Optional[str]
    candidates_tried: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_found": self.final_found,
            "final_error": self.final_error,
            "scored_found": self.scored is not None,
            "scored_is_final": self.scored_is_final,
            "envelope_error": self.envelope_error,
            "candidates_tried": self.candidates_tried,
        }


_FENCE = "```"
_FENCE_TAG = re.compile(r"[A-Za-z0-9_+.#-]+")


def final_json_object(response: str) -> Tuple[Optional[str], bool]:
    """``(text, found)`` of the response's final object (DESIGN 2.1 position rule).

    - The response (right-stripped) ends with a fence: the block is the text between the
      closing fence and the fence just before it, so an earlier example or draft fence is
      never merged into it. An info string other than empty or ``json`` (for example
      ``python``) means there is no final JSON object.
    - Otherwise: the last balanced ``{...}`` span that ends the response.
    """
    stripped = (response if isinstance(response, str) else "").rstrip()
    if not stripped:
        return None, False
    if stripped.endswith(_FENCE):
        close = len(stripped) - len(_FENCE)
        opened = stripped.rfind(_FENCE, 0, close)
        if opened < 0:
            return None, False
        body = stripped[opened + len(_FENCE) : close]
        first_line, _, rest = body.partition("\n")
        tag = first_line.strip()
        if tag.lower() == "json":
            body = rest
        elif tag[:4].lower() == "json" and tag[4:5] in ("{", "["):
            body = body.lstrip()[4:]
        elif tag and _FENCE_TAG.fullmatch(tag):
            return None, False  # a non-JSON fence (```python ...) ends the response
        content = body.strip()
        return (content, True) if content else (None, False)
    for start, end in reversed(find_brace_spans(stripped)):
        if end == len(stripped):
            return stripped[start:end].strip(), True
        break
    return None, False


def locate_scored_object(
    response: str,
    *,
    max_bytes: int = MAX_OBJECT_BYTES,
    max_candidates: int = MAX_SCAN_CANDIDATES,
) -> LocatedObject:
    """Locate the final object and the scored object (DESIGN 2.1, review EX-4 / CONF-08)."""
    if not isinstance(response, str):
        response = "" if response is None else str(response)
    final_text, final_found = final_json_object(response)
    final_value: Any = None
    final_error: Optional[str] = None
    if final_found and final_text is not None:
        try:
            final_value = strict_loads(final_text, max_bytes=max_bytes)
        except StrictJSONError as exc:
            final_error = exc.code
        if final_error is None and isinstance(final_value, dict) and "certificate" in final_value:
            return LocatedObject(True, final_text, final_value, None, final_value, final_text, True, None)
        if final_error is None:
            final_error = "missing_certificate" if isinstance(final_value, dict) else "not_object"
        # The final object is the response's claim: a broken one is an envelope failure and
        # an earlier draft is never scored in its place.
        return LocatedObject(True, final_text, final_value, final_error, None, None, False, final_error)

    over_cap = False
    tried = 0
    for start, end in reversed(find_brace_spans(response)):
        if tried >= max_candidates:
            break
        text = response[start:end]
        if len(text) > max_bytes or len(text.encode("utf-8")) > max_bytes:
            over_cap = over_cap or '"certificate"' in text
            continue
        tried += 1
        try:
            value = strict_loads(text, max_bytes=max_bytes)
        except StrictJSONError:
            continue
        if isinstance(value, dict) and "certificate" in value:
            return LocatedObject(
                final_found, final_text, final_value, final_error, value, text, False, None, tried
            )
    error = "budget" if over_cap else "no_object"
    return LocatedObject(final_found, final_text, final_value, final_error, None, None, False, error, tried)


def check_envelope(obj: Any) -> Optional[str]:
    """Envelope errors of a scored object outside the certificate body, else None."""
    if not isinstance(obj, dict):
        return "not_object"
    if isinstance(obj, JSONObject) and obj.duplicate_keys:
        return "duplicate_key"
    if "certificate" not in obj:
        return "missing_certificate"
    if duplicate_key_paths(obj.get("annotation"), "$.annotation"):
        return "duplicate_key"
    return None


# ----------------------------------------------------------------------------------------
# Certificate schema


@dataclass(frozen=True)
class FieldSpec:
    """Type of one certificate field: int, index, enum, point, bool or const."""

    type: str
    allowed: Tuple[Any, ...] = ()


INT = FieldSpec("int")
INDEX = FieldSpec("index")
POINT = FieldSpec("point")
BOOL = FieldSpec("bool")


def enum(*values: str) -> FieldSpec:
    return FieldSpec("enum", tuple(values))


def const(value: Any) -> FieldSpec:
    """Exactly this JSON value (bools compare by identity, so ``true`` never equals ``1``)."""
    return FieldSpec("const", (value,))


@dataclass(frozen=True)
class CertificateSchema:
    """Declarative schema of one certificate variant.

    ``fields`` are the certificate-level keys other than ``kind`` and ``list_key``;
    ``element_fields`` are the exact keys of every list element.
    """

    kind: str
    fields: Mapping[str, FieldSpec]
    list_key: Optional[str]
    element_fields: Mapping[str, FieldSpec]
    list_cap: int
    variant: str = "default"


@dataclass(frozen=True)
class ElementView:
    """One list element. ``values`` holds the typed values of the keys that parsed."""

    position: int
    raw: Any
    valid: bool
    error_code: Optional[str]
    values: Mapping[str, Any]
    annotation_index: Optional[int] = None
    integral_float: bool = False

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


@dataclass(frozen=True)
class CertificateView:
    """Typed, validated view of a certificate. Adapters read this, never the raw JSON.

    ``answer`` is the response's top-level answer read as an integer (bool rejected, integral
    float accepted by value) or None; it is only for local checks such as ``*.answer.local``.
    """

    raw: Any
    kind: Optional[str]
    variant: str
    envelope_error: Optional[str]
    elements: Tuple[ElementView, ...] = ()
    fields: Mapping[str, Any] = field(default_factory=dict)
    field_errors: Mapping[str, str] = field(default_factory=dict)
    extra_keys: Tuple[str, ...] = ()
    integral_floats: bool = False
    duplicate_keys: Tuple[str, ...] = ()
    n_annotation: int = 0
    answer: Optional[int] = None

    @property
    def envelope_ok(self) -> bool:
        return self.envelope_error is None

    @property
    def n_claimed(self) -> int:
        return len(self.elements)

    def element_validity(self) -> List[bool]:
        return [element.valid for element in self.elements]

    @property
    def schema_ok(self) -> bool:
        """Everything F needs from the certificate body."""
        return (
            self.envelope_ok
            and all(element.valid for element in self.elements)
            and not self.field_errors
            and not self.extra_keys
            and not self.integral_floats
            and not self.duplicate_keys
        )

    def schema_failures(self) -> List[str]:
        reasons: List[str] = []
        if self.envelope_error is not None:
            reasons.append(f"certificate_{self.envelope_error}")
            return reasons
        if not all(element.valid for element in self.elements):
            reasons.append("element_invalid")
        if self.field_errors:
            reasons.append("field_invalid")
        if self.extra_keys:
            reasons.append("extra_key")
        if self.integral_floats:
            reasons.append("integral_float")
        if self.duplicate_keys:
            reasons.append("duplicate_key")
        return reasons


def read_field(value: Any, spec: FieldSpec, n_annotation: int) -> Tuple[Any, Optional[str], bool]:
    """``(typed value, error code, integral_float)`` for one field."""
    if spec.type in ("int", "index"):
        read = read_int(value)
        if read.value is None:
            return None, "bad_type", False
        if spec.type == "index" and not 0 <= read.value < n_annotation:
            return read.value, "index_out_of_range", read.integral_float
        return read.value, None, read.integral_float
    if spec.type == "enum":
        typed = read_enum(value, spec.allowed)
        return (typed, None, False) if typed is not None else (None, "bad_value", False)
    if spec.type == "point":
        typed = read_point(value)
        return (typed, None, False) if typed is not None else (None, "bad_type", False)
    if spec.type == "bool":
        typed = read_bool(value)
        return (typed, None, False) if typed is not None else (None, "bad_type", False)
    if spec.type == "const":
        expected = spec.allowed[0]
        same = type(value) is type(expected) and value == expected
        return (value, None, False) if same else (None, "bad_value", False)
    raise ValueError(f"unknown field type {spec.type!r}")


def _envelope(cert: Any, schema: CertificateSchema, error: str, n_annotation: int) -> CertificateView:
    kind = cert.get("kind") if isinstance(cert, dict) else None
    return CertificateView(
        raw=cert,
        kind=kind if isinstance(kind, str) else None,
        variant=schema.variant,
        envelope_error=error,
        n_annotation=n_annotation,
    )


def _element(position: int, raw: Any, schema: CertificateSchema, n_annotation: int) -> ElementView:
    if not isinstance(raw, dict):
        return ElementView(position, raw, False, "not_object", {})
    values: Dict[str, Any] = {}
    error: Optional[str] = None
    integral = False
    if isinstance(raw, JSONObject) and raw.duplicate_keys:
        error = "duplicate_key"
    missing = [key for key in schema.element_fields if key not in raw]
    extra = [key for key in raw if key not in schema.element_fields]
    if error is None and missing:
        error = "missing_key"
    if error is None and extra:
        error = "extra_key"
    index: Optional[int] = None
    for key, spec in schema.element_fields.items():
        if key not in raw:
            continue
        typed, field_error, was_float = read_field(raw[key], spec, n_annotation)
        integral = integral or was_float
        if field_error is None:
            values[key] = typed
            if spec.type == "index":
                index = typed
        elif error is None:
            error = field_error
    return ElementView(position, raw, error is None, error, values, index if error is None else None, integral)


def validate_certificate(cert: Any, schema: CertificateSchema, n_annotation: int) -> CertificateView:
    """Validate a certificate against ``schema``; ``n_annotation`` bounds ``annotation_index``."""
    if not isinstance(cert, dict):
        return _envelope(cert, schema, "certificate_not_object", n_annotation)
    if isinstance(cert, JSONObject) and cert.duplicate_keys:
        return _envelope(cert, schema, "duplicate_key", n_annotation)
    if cert.get("kind") != schema.kind or not isinstance(cert.get("kind"), str):
        return _envelope(cert, schema, "wrong_kind", n_annotation)
    items: List[Any] = []
    if schema.list_key is not None:
        raw_list = cert.get(schema.list_key)
        if not isinstance(raw_list, list):
            return _envelope(cert, schema, "missing_list", n_annotation)
        if len(raw_list) > schema.list_cap:
            return _envelope(cert, schema, "budget", n_annotation)
        items = raw_list

    fields: Dict[str, Any] = {}
    field_errors: Dict[str, str] = {}
    integral = False
    for key, spec in schema.fields.items():
        if key not in cert:
            field_errors[key] = "missing_key"
            continue
        typed, error, was_float = read_field(cert[key], spec, n_annotation)
        integral = integral or was_float
        if error is None:
            fields[key] = typed
        else:
            field_errors[key] = error
    known = {"kind", *schema.fields}
    if schema.list_key is not None:
        known.add(schema.list_key)
    extra = tuple(key for key in cert if key not in known)
    elements = tuple(_element(position, raw, schema, n_annotation) for position, raw in enumerate(items))
    integral = integral or any(element.integral_float for element in elements)
    duplicates = tuple(duplicate_key_paths(cert, "$.certificate"))
    return CertificateView(
        raw=cert,
        kind=schema.kind,
        variant=schema.variant,
        envelope_error=None,
        elements=elements,
        fields=fields,
        field_errors=field_errors,
        extra_keys=extra,
        integral_floats=integral,
        duplicate_keys=duplicates,
        n_annotation=n_annotation,
    )


# ----------------------------------------------------------------------------------------
# F


@dataclass(frozen=True)
class FormatResult:
    """F = 1 iff ``reasons`` is empty."""

    reasons: Tuple[str, ...]

    @property
    def score(self) -> float:
        return 0.0 if self.reasons else 1.0


def format_check(
    located: LocatedObject,
    certificate: Optional[CertificateView],
    *,
    annotation_parse_ok: bool,
    answer_ok: bool,
    annotation_error: Optional[str] = None,
) -> FormatResult:
    """DESIGN 2.1 F: final position, strict parse, exact keys, strict annotation structure
    (``annotation_error``, public schema) and legacy annotation parse, certificate
    kind/element/field schema, caps, integers only (integral floats fail)."""
    if located.scored is None:
        return FormatResult((f"envelope_{located.envelope_error or 'no_object'}",))
    reasons: List[str] = []
    obj = located.scored
    if not (located.final_found and located.scored_is_final):
        reasons.append("not_final")
    if duplicate_key_paths(obj):
        reasons.append("duplicate_key")
    if set(obj.keys()) != TOP_LEVEL_KEYS:
        reasons.append("top_level_keys")
    if annotation_error is not None:
        reasons.append(annotation_error)
    if not annotation_parse_ok:
        reasons.append("annotation_parse")
    if certificate is None:
        reasons.append("certificate_missing")
    else:
        reasons.extend(reason for reason in certificate.schema_failures() if reason != "duplicate_key")
    if not answer_ok:
        reasons.append("answer_type")
    return FormatResult(tuple(dict.fromkeys(reasons)))


def _json_value_key(item: Any) -> str:
    """Comparison key of one duplicated JSON ``answer`` value; never raises (review VER-1).

    A value with no JSON text (in practice never: ``json.loads`` already caps integers at
    ``sys.int_max_str_digits``, but a pathological nesting can still exhaust the encoder) gets
    a key of its own, so it is never taken as equal to another value: the guard fires."""
    try:
        return "json:" + json.dumps(item, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001 - ValueError, TypeError, RecursionError on model data
        return f"unserializable:{id(item)}"


def answer_duplicate_conflict(value: Any) -> bool:
    """True when a JSON object repeats ``answer`` with different values (A guard, DESIGN 5).
    Total: it never raises on model data (review VER-1)."""
    if not isinstance(value, JSONObject) or "answer" not in value.duplicate_keys:
        return False
    seen = {_json_value_key(item) for item in value.duplicate_values["answer"]}
    return len(seen) > 1


def strict_final_object(response: str) -> Any:
    """The final object (:func:`final_json_object`) parsed strictly, or None."""
    text, found = final_json_object(response if isinstance(response, str) else "")
    if not found or text is None:
        return None
    try:
        return strict_loads(text)
    except StrictJSONError:
        return None


def legacy_answer_source(response: str) -> Optional[str]:
    """The text of the object TRACE's ``extract_trace_prediction`` reads the answer from.

    Mirrors the legacy order exactly: the legacy final block (whole, else its first dict),
    when it has ``answer`` or ``annotation``; else the last JSON/Python-literal candidate of
    the response with one of those keys. None when legacy extraction finds nothing.
    """
    if not isinstance(response, str):
        return None
    text, found = _extract_final_json_object(response)
    if found and text:
        payload_text: Optional[str] = None
        payload: Any = _parse_json_like(text)
        if isinstance(payload, dict):
            payload_text = text
        else:
            for candidate in _collect_json_candidates(text):
                parsed = _parse_json_like(candidate)
                if isinstance(parsed, dict):
                    payload_text, payload = candidate, parsed
                    break
        if payload_text is not None and ("answer" in payload or "annotation" in payload):
            return payload_text
    for candidate in reversed(_collect_json_candidates(response)):
        parsed = _parse_json_like(candidate)
        if isinstance(parsed, dict) and ("answer" in parsed or "annotation" in parsed):
            return candidate
    return None


def _literal_value_key(source: str, node: ast.AST) -> str:
    """Comparison key of one literal ``answer`` value; never raises (review VER-1).

    In order: the literal value's ``repr``; else (not a literal, e.g. a name) its AST dump;
    else the value as written. The last case is an integer beyond ``sys.int_max_str_digits``
    (a hex, octal or binary literal parses without the digit limit, but neither ``repr`` nor
    ``ast.dump`` can print it), alone or inside a container. The prefixes keep the three kinds
    of key apart; ``sys.set_int_max_str_digits`` is never touched.
    """
    try:
        return "value:" + repr(ast.literal_eval(node))
    except Exception:  # noqa: BLE001 - not a literal, or a value with no repr
        pass
    try:
        return "ast:" + ast.dump(node)
    except Exception:  # noqa: BLE001 - an int constant beyond the digit limit
        pass
    try:
        return "source:" + (ast.get_source_segment(source, node) or "")
    except Exception:  # noqa: BLE001 - no source position: a key of its own (never "equal")
        return f"node:{id(node)}"


def _literal_answer_conflict(text: str) -> bool:
    """Duplicate ``answer`` keys with different values in a Python-literal dict (legacy
    ``ast.literal_eval`` fallback); keys are compared as written, values by literal value
    (:func:`_literal_value_key`). Total: any failure means no detectable duplicate."""
    source = text.strip()
    try:
        node = ast.parse(source, mode="eval").body
    except Exception:  # noqa: BLE001 - any unparsable text has no detectable duplicate
        return False
    if not isinstance(node, ast.Dict):
        return False
    try:
        values = {
            _literal_value_key(source, value)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value == "answer"
        }
    except Exception:  # noqa: BLE001 - defensive: the comparison itself must never raise
        return False
    return len(values) > 1


def text_answer_conflict(text: Optional[str]) -> bool:
    """Conflicting duplicate ``answer`` keys in the object ``text`` (JSON, NaN tolerated, or a
    Python literal): the DESIGN 5 A guard applied to whatever object A was read from.
    Total: it never raises on model text (review VER-1)."""
    if not text:
        return False
    try:
        value = json.loads(text, object_pairs_hook=_pairs_hook)
    except (ValueError, RecursionError, TypeError):
        return _literal_answer_conflict(text)
    return answer_duplicate_conflict(value)


def annotation_schema_error(
    value: Any,
    annotation_type: str,
    *,
    keys: Sequence[str] = (),
    cap: Optional[int] = None,
) -> Optional[str]:
    """Strict structure of the annotation against the public schema (F, review CONF-02).

    Independent of TRACE's repairing legacy normalizer: the value must be the JSON type the
    contract names (never a string), boxes are lists of exactly 4 float-representable
    numbers, set types are lists of boxes (no ``{"bboxes": ...}`` wrapper, no bare box),
    map types have exactly ``keys``, and set lists hold at most ``cap`` boxes.
    """

    def is_box(item: Any) -> bool:
        return isinstance(item, list) and len(item) == 4 and all(as_finite_float(x) is not None for x in item)

    def box_list_error(item: Any) -> Optional[str]:
        if not isinstance(item, list):
            return "annotation_not_list"
        if cap is not None and len(item) > cap:
            return "annotation_budget"
        return None if all(is_box(box) for box in item) else "annotation_box"

    if annotation_type == "bbox":
        return None if is_box(value) else "annotation_box"
    if annotation_type in ("bbox_set", "bbox_sequence"):
        return box_list_error(value)
    if annotation_type in ("bbox_map", "bbox_set_map"):
        if not isinstance(value, dict):
            return "annotation_not_object"
        if set(value) != set(keys):
            return "annotation_keys"
        for key in keys:
            error = box_list_error(value[key]) if annotation_type == "bbox_set_map" else (
                None if is_box(value[key]) else "annotation_box"
            )
            if error is not None:
                return error
        return None
    return "annotation_type_unsupported"


def element_schema_ok(cert: Any, schema: CertificateSchema, n_annotation: int) -> List[bool]:
    """Per-element validity (empty list on an envelope failure)."""
    return validate_certificate(cert, schema, n_annotation).element_validity()


__all__ = [
    "BOOL",
    "CertificateSchema",
    "CertificateView",
    "ElementView",
    "FieldSpec",
    "FormatResult",
    "INDEX",
    "INT",
    "JSONObject",
    "LocatedObject",
    "MAX_JSON_DEPTH",
    "MAX_SCAN_CANDIDATES",
    "POINT",
    "StrictJSONError",
    "TOP_LEVEL_KEYS",
    "annotation_schema_error",
    "answer_duplicate_conflict",
    "check_envelope",
    "const",
    "duplicate_key_paths",
    "element_schema_ok",
    "enum",
    "final_json_object",
    "find_brace_spans",
    "format_check",
    "json_depth",
    "legacy_answer_source",
    "locate_scored_object",
    "read_field",
    "strict_final_object",
    "strict_loads",
    "text_answer_conflict",
    "validate_certificate",
]
