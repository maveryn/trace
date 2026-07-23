"""Ground-truth-blind deterministic answer extraction for TRACE validation.

The extractor deliberately receives only a model response and the declared
answer type.  It never receives a reference answer and it does not score the
candidate.  Unresolved responses are intended for a separately provenance-bound
fallback extractor.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

ANSWER_EXTRACTION_CONTRACT_VERSION = "trace-validation-answer-extraction-v1"

AnswerType = Literal["integer", "number", "option_letter", "string"]
ExtractionStatus = Literal["found", "missing", "ambiguous"]
TypedCandidate = int | float | str

_ANSWER_TYPES = frozenset({"integer", "number", "option_letter", "string"})
_ANSWER_TAG_RE = re.compile(
    r"<(?P<tag>answer|final[_-]?answer)>\s*(?P<body>.*?)\s*</(?P=tag)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_CODE_FENCE_RE = re.compile(
    r"```[ \t]*(?P<language>json|python|py)?[ \t]*\r?\n(?P<body>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
_STRUCTURED_SECTION_RE = re.compile(
    r"^[ \t]*(?:final[ _-]*json|final[ _-]*answer[ _-]*json|answer[ _-]*json)"
    r"[ \t]*:[ \t]*(?P<inline>[^\r\n]*)",
    flags=re.IGNORECASE | re.MULTILINE,
)
_FINAL_ANSWER_RE = re.compile(
    r"^[ \t]*(?:the[ \t]+)?final[ \t]+answer[ \t]*(?:is|=|:|：)[ \t]*"
    r"(?P<answer>[^\r\n]+?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_ANSWER_LINE_RE = re.compile(
    r"^[ \t]*answer[ \t]*(?:is|=|:|：)[ \t]*(?P<answer>[^\r\n]+?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_BOX_START_RE = re.compile(r"\\boxed\s*\{", flags=re.IGNORECASE)
_NUMBER_RE = re.compile(r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$")
_OPTION_RE = re.compile(
    r"^(?:option[ \t]+)?(?:[\(\[]\s*)?([A-Za-z])(?:\s*[\)\]])?[.!?]?$",
    flags=re.IGNORECASE,
)


def _json_safe(value: Any) -> Any:
    """Return a deterministic JSON-safe representation for audit records."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_safe(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
        )
    return repr(value)


@dataclass(frozen=True)
class CandidateProvenance:
    """Auditable evidence for one accepted or rejected answer candidate."""

    route: str
    priority: Literal["terminal_structured", "structured", "explicit"]
    raw_candidate: Any
    typed_candidate: TypedCandidate | None
    canonical_candidate: str | None
    source_start: int
    source_end: int
    parser: str
    accepted: bool
    rejection_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["raw_candidate"] = _json_safe(self.raw_candidate)
        payload["typed_candidate"] = _json_safe(self.typed_candidate)
        return payload


@dataclass(frozen=True)
class AnswerExtractionResult:
    """Result of deterministic extraction without reference-answer access."""

    version: str
    answer_type: AnswerType
    status: ExtractionStatus
    route: str
    raw_candidate: Any | None
    typed_candidate: TypedCandidate | None
    candidate_provenance: tuple[CandidateProvenance, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["raw_candidate"] = _json_safe(self.raw_candidate)
        payload["typed_candidate"] = _json_safe(self.typed_candidate)
        payload["candidate_provenance"] = [
            candidate.as_dict() for candidate in self.candidate_provenance
        ]
        return payload


@dataclass(frozen=True)
class _ParsedValue:
    value: Any
    parser: str


class _DuplicateKeyError(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_literal_eval(text: str) -> Any:
    """Evaluate a Python literal without allowing dict-key overwrite semantics."""

    expression = ast.parse(text, mode="eval")
    for node in ast.walk(expression):
        if not isinstance(node, ast.Dict):
            continue
        seen: set[Any] = set()
        for key_node in node.keys:
            if key_node is None:
                raise ValueError("dict unpacking is not accepted")
            key = ast.literal_eval(key_node)
            try:
                if key in seen:
                    raise _DuplicateKeyError(f"duplicate Python-literal key: {key!r}")
                seen.add(key)
            except TypeError as exc:
                raise ValueError("unhashable Python-literal mapping key") from exc
    return ast.literal_eval(expression.body)


def _parse_structured_value(text: str) -> _ParsedValue | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        value = json.loads(
            stripped,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateKeyError:
        # A duplicated answer key is conflicting evidence, not a Python-literal
        # compatibility case.  Do not silently let literal_eval select the last
        # value.
        return None
    except (json.JSONDecodeError, TypeError, ValueError):
        try:
            value = _strict_literal_eval(stripped)
        except (
            SyntaxError,
            ValueError,
            TypeError,
            MemoryError,
            RecursionError,
        ):
            return None
        return _ParsedValue(value=value, parser="python_literal")
    return _ParsedValue(value=value, parser="json")


def _balanced_object_spans(text: str) -> list[tuple[int, int, str]]:
    """Return balanced outer brace objects while respecting quoted strings."""

    spans: list[tuple[int, int, str]] = []
    depth = 0
    start: int | None = None
    quote: str | None = None
    escaped = False
    for index, character in enumerate(text):
        if depth == 0:
            # Quotes and apostrophes in surrounding prose do not affect brace
            # balancing.  Quoted-string state begins only after an object has
            # opened.
            if character == "{":
                start = index
                depth = 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
            continue
        if character == "{":
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append((start, index + 1, text[start : index + 1]))
                start = None
    return spans


def _balanced_box_end(text: str, content_start: int) -> int | None:
    depth = 1
    for index in range(content_start, len(text)):
        character = text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _unwrap_scalar_text(value: str) -> Any:
    text = value.strip()
    if not text:
        return text

    boxed = _BOX_START_RE.fullmatch(text[: text.find("{") + 1]) if "{" in text else None
    if boxed is not None:
        end = _balanced_box_end(text, boxed.end())
        if end == len(text):
            text = text[boxed.end() : end - 1].strip()

    for pattern in (
        r"\\(?:text|mathrm|mathbf)\s*\{\s*(.*?)\s*\}",
        r"`\s*(.*?)\s*`",
    ):
        match = re.fullmatch(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match is not None:
            text = match.group(1).strip()
            break

    parsed = _parse_structured_value(text)
    if parsed is not None and isinstance(parsed.value, (str, int, float, bool)):
        return parsed.value
    return text


def _unwrap_string_text(value: str) -> str:
    """Remove explicit wrappers without retyping a numeric-looking string."""

    text = value.strip()
    if not text:
        return text
    boxed = _BOX_START_RE.fullmatch(text[: text.find("{") + 1]) if "{" in text else None
    if boxed is not None:
        end = _balanced_box_end(text, boxed.end())
        if end == len(text):
            text = text[boxed.end() : end - 1].strip()
    for pattern in (
        r"\\(?:text|mathrm|mathbf)\s*\{\s*(.*?)\s*\}",
        r"`\s*(.*?)\s*`",
    ):
        match = re.fullmatch(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match is not None:
            text = match.group(1).strip()
            break
    parsed = _parse_structured_value(text)
    return (
        parsed.value if parsed is not None and isinstance(parsed.value, str) else text
    )


def _decimal_from_scalar(value: Any) -> tuple[Decimal | None, str | None]:
    if isinstance(value, bool):
        return None, "boolean_not_allowed"
    if isinstance(value, int):
        text = str(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            return None, "nonfinite_not_allowed"
        text = repr(value)
    elif isinstance(value, str):
        text = value.strip()
        if not _NUMBER_RE.fullmatch(text):
            # A final sentence period after a decimal is common (for example
            # ``2.5.``); remove it only when doing so yields one atomic number.
            if text.endswith(".") and _NUMBER_RE.fullmatch(text[:-1]):
                text = text[:-1]
            else:
                return None, "not_an_atomic_number"
    else:
        return None, "numeric_scalar_required"
    try:
        decimal = Decimal(text)
    except InvalidOperation:
        return None, "not_an_atomic_number"
    try:
        finite_float = float(decimal)
    except (OverflowError, ValueError):
        return None, "nonfinite_not_allowed"
    if not decimal.is_finite() or not math.isfinite(finite_float):
        return None, "nonfinite_not_allowed"
    if decimal != 0 and finite_float == 0.0:
        return None, "number_out_of_range"
    return decimal, None


def _coerce_candidate(
    raw_candidate: Any,
    answer_type: AnswerType,
    *,
    parser: str,
) -> tuple[TypedCandidate | None, str | None, str | None]:
    # Values obtained from JSON/Python mappings are already typed.  In
    # particular, a structured string answer ``"4"`` must stay a string for a
    # string-typed task.  Text routes still unwrap quotes and narrow wrappers so
    # numeric/option types can be coerced deliberately.
    if isinstance(raw_candidate, str) and parser == "text":
        value = (
            _unwrap_string_text(raw_candidate)
            if answer_type == "string"
            else _unwrap_scalar_text(raw_candidate)
        )
    else:
        value = raw_candidate

    if answer_type in {"integer", "number"}:
        decimal, rejection = _decimal_from_scalar(value)
        if decimal is None:
            return None, None, rejection
        if answer_type == "integer" and decimal != decimal.to_integral_value():
            return None, None, "non_integral_number"
        if decimal == decimal.to_integral_value():
            typed: TypedCandidate = int(decimal)
        else:
            typed = float(decimal)
        canonical = str(decimal.normalize())
        if canonical in {"-0", "-0E+1"} or decimal == 0:
            canonical = "0"
        return typed, canonical, None

    if isinstance(value, bool):
        return None, None, "boolean_not_allowed"
    if isinstance(value, float) and not math.isfinite(value):
        return None, None, "nonfinite_not_allowed"
    if answer_type == "option_letter":
        if not isinstance(value, str):
            return None, None, "option_letter_string_required"
        match = _OPTION_RE.fullmatch(value.strip())
        if match is None:
            return None, None, "not_an_atomic_option_letter"
        typed = match.group(1).upper()
        return typed, typed, None

    if not isinstance(value, str):
        return None, None, "string_value_required"
    typed = value.strip()
    if not typed:
        return None, None, "empty_string"
    return typed, typed, None


def _mapping_answer_values(value: Any) -> list[Any]:
    if not isinstance(value, dict):
        return []
    answers: list[Any] = []
    for key, candidate in value.items():
        if isinstance(key, str) and key.strip().casefold() == "answer":
            answers.append(candidate)
    return answers


def _candidate_evidence(
    *,
    route: str,
    priority: Literal["terminal_structured", "structured", "explicit"],
    raw_candidate: Any,
    source_start: int,
    source_end: int,
    parser: str,
    answer_type: AnswerType,
) -> CandidateProvenance:
    typed, canonical, rejection = _coerce_candidate(
        raw_candidate,
        answer_type,
        parser=parser,
    )
    return CandidateProvenance(
        route=route,
        priority=priority,
        raw_candidate=raw_candidate,
        typed_candidate=typed,
        canonical_candidate=canonical,
        source_start=source_start,
        source_end=source_end,
        parser=parser,
        accepted=rejection is None,
        rejection_reason=rejection,
    )


def _structured_evidence(
    *,
    parsed: _ParsedValue | None,
    fallback_text: str | None,
    allow_scalar: bool,
    route: str,
    terminal: bool,
    source_start: int,
    source_end: int,
    answer_type: AnswerType,
) -> list[CandidateProvenance]:
    if parsed is None and not (allow_scalar and fallback_text is not None):
        return []
    if parsed is None and (fallback_text or "").lstrip().startswith(
        ("{", "[", '"', "'")
    ):
        # Do not reinterpret something that looks like malformed structured
        # data as a literal string answer.
        return []
    priority: Literal["terminal_structured", "structured"] = (
        "terminal_structured" if terminal else "structured"
    )
    if parsed is not None:
        answers = _mapping_answer_values(parsed.value)
        if answers:
            return [
                _candidate_evidence(
                    route=route,
                    priority=priority,
                    raw_candidate=answer,
                    source_start=source_start,
                    source_end=source_end,
                    parser=parsed.parser,
                    answer_type=answer_type,
                )
                for answer in answers
            ]
        if not allow_scalar or isinstance(parsed.value, (dict, list, tuple, set)):
            return []
        raw_candidate = parsed.value
        parser = parsed.parser
    else:
        raw_candidate = (fallback_text or "").strip()
        parser = "text"
    return [
        _candidate_evidence(
            route=route,
            priority=priority,
            raw_candidate=raw_candidate,
            source_start=source_start,
            source_end=source_end,
            parser=parser,
            answer_type=answer_type,
        )
    ]


def _collect_structured_candidates(
    text: str,
    answer_type: AnswerType,
) -> list[CandidateProvenance]:
    evidence: list[CandidateProvenance] = []
    terminal_end = len(text.rstrip())

    for match in _ANSWER_TAG_RE.finditer(text):
        body = match.group("body").strip()
        terminal = match.end() == terminal_end
        evidence.extend(
            _structured_evidence(
                parsed=_parse_structured_value(body),
                fallback_text=body,
                allow_scalar=True,
                route="answer_tag",
                terminal=terminal,
                source_start=match.start(),
                source_end=match.end(),
                answer_type=answer_type,
            )
        )

    for match in _CODE_FENCE_RE.finditer(text):
        body = match.group("body").strip()
        terminal = match.end() == terminal_end
        evidence.extend(
            _structured_evidence(
                parsed=_parse_structured_value(body),
                fallback_text=body,
                allow_scalar=terminal,
                route="code_fence",
                terminal=terminal,
                source_start=match.start(),
                source_end=match.end(),
                answer_type=answer_type,
            )
        )

    for match in _STRUCTURED_SECTION_RE.finditer(text):
        inline = match.group("inline")
        if inline.strip():
            source_start = match.start("inline")
            body = text[source_start:terminal_end].strip()
        else:
            source_start = match.end()
            body = text[source_start:terminal_end].strip()
        parsed = _parse_structured_value(body)
        evidence.extend(
            _structured_evidence(
                parsed=parsed,
                fallback_text=None,
                allow_scalar=True,
                route="structured_section",
                terminal=parsed is not None,
                source_start=match.start(),
                source_end=terminal_end,
                answer_type=answer_type,
            )
        )

    for start, end, body in _balanced_object_spans(text):
        parsed = _parse_structured_value(body)
        evidence.extend(
            _structured_evidence(
                parsed=parsed,
                fallback_text=None,
                allow_scalar=False,
                route="balanced_object",
                terminal=end == terminal_end,
                source_start=start,
                source_end=end,
                answer_type=answer_type,
            )
        )
    return evidence


def _collect_explicit_candidates(
    text: str,
    answer_type: AnswerType,
) -> list[CandidateProvenance]:
    evidence: list[CandidateProvenance] = []
    for route, pattern in (
        ("final_answer", _FINAL_ANSWER_RE),
        ("answer_line", _ANSWER_LINE_RE),
    ):
        for match in pattern.finditer(text):
            evidence.append(
                _candidate_evidence(
                    route=route,
                    priority="explicit",
                    raw_candidate=match.group("answer").strip(),
                    source_start=match.start(),
                    source_end=match.end(),
                    parser="text",
                    answer_type=answer_type,
                )
            )

    for match in _BOX_START_RE.finditer(text):
        end = _balanced_box_end(text, match.end())
        if end is None:
            continue
        evidence.append(
            _candidate_evidence(
                route="boxed",
                priority="explicit",
                raw_candidate=text[match.end() : end - 1].strip(),
                source_start=match.start(),
                source_end=end,
                parser="text",
                answer_type=answer_type,
            )
        )

    if answer_type == "option_letter":
        nonempty_lines = list(re.finditer(r"(?m)^\s*(\S[^\r\n]*?)\s*$", text.rstrip()))
        if nonempty_lines:
            match = nonempty_lines[-1]
            raw_candidate = match.group(1).strip()
            if _OPTION_RE.fullmatch(raw_candidate):
                evidence.append(
                    _candidate_evidence(
                        route="terminal_option",
                        priority="explicit",
                        raw_candidate=raw_candidate,
                        source_start=match.start(),
                        source_end=match.end(),
                        parser="text",
                        answer_type=answer_type,
                    )
                )
    return evidence


def _deduplicate_evidence(
    evidence: list[CandidateProvenance],
) -> tuple[CandidateProvenance, ...]:
    unique: list[CandidateProvenance] = []
    seen: set[tuple[Any, ...]] = set()
    for candidate in sorted(
        evidence,
        key=lambda item: (
            item.source_start,
            item.source_end,
            item.route,
            repr(item.raw_candidate),
        ),
    ):
        key = (
            candidate.route,
            candidate.priority,
            repr(candidate.raw_candidate),
            candidate.source_start,
            candidate.source_end,
            candidate.canonical_candidate,
            candidate.rejection_reason,
        )
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


def _missing_result(
    answer_type: AnswerType,
    evidence: tuple[CandidateProvenance, ...],
) -> AnswerExtractionResult:
    return AnswerExtractionResult(
        version=ANSWER_EXTRACTION_CONTRACT_VERSION,
        answer_type=answer_type,
        status="missing",
        route="missing",
        raw_candidate=None,
        typed_candidate=None,
        candidate_provenance=evidence,
    )


def extract_answer(
    response: str | None, answer_type: AnswerType
) -> AnswerExtractionResult:
    """Extract one typed answer without consulting the reference answer.

    A valid terminal structured candidate is authoritative and shadows earlier
    evidence.  Without one, every accepted structured and explicit candidate is
    reconciled: canonical-equal duplicates resolve, while conflicts return
    ``ambiguous`` for fallback extraction.
    """

    normalized_type = str(answer_type).strip().lower()
    if normalized_type not in _ANSWER_TYPES:
        raise ValueError(
            f"answer_type must be one of {sorted(_ANSWER_TYPES)!r}; got {answer_type!r}"
        )
    resolved_type: AnswerType = normalized_type  # type: ignore[assignment]
    if response is None or not str(response).strip():
        return _missing_result(resolved_type, ())
    if not isinstance(response, str):
        raise TypeError(
            f"response must be a string or None; got {type(response).__name__}"
        )

    evidence = _deduplicate_evidence(
        _collect_structured_candidates(response, resolved_type)
        + _collect_explicit_candidates(response, resolved_type)
    )
    terminal_evidence = [
        candidate
        for candidate in evidence
        if candidate.priority == "terminal_structured"
    ]
    considered = (
        [candidate for candidate in terminal_evidence if candidate.accepted]
        if terminal_evidence
        else [candidate for candidate in evidence if candidate.accepted]
    )
    if not considered:
        return _missing_result(resolved_type, evidence)

    canonical_values = {candidate.canonical_candidate for candidate in considered}
    if len(canonical_values) != 1:
        return AnswerExtractionResult(
            version=ANSWER_EXTRACTION_CONTRACT_VERSION,
            answer_type=resolved_type,
            status="ambiguous",
            route="ambiguous",
            raw_candidate=None,
            typed_candidate=None,
            candidate_provenance=evidence,
        )

    route_rank = {
        "balanced_object": 0,
        "code_fence": 1,
        "answer_tag": 2,
        "structured_section": 3,
        "final_answer": 4,
        "answer_line": 5,
        "boxed": 6,
        "terminal_option": 7,
    }
    selected = sorted(
        considered,
        key=lambda item: (
            0 if item.priority == "terminal_structured" else 1,
            route_rank.get(item.route, 99),
            -item.source_end,
        ),
    )[0]
    return AnswerExtractionResult(
        version=ANSWER_EXTRACTION_CONTRACT_VERSION,
        answer_type=resolved_type,
        status="found",
        route=selected.route,
        raw_candidate=selected.raw_candidate,
        typed_candidate=selected.typed_candidate,
        candidate_provenance=evidence,
    )


__all__ = [
    "ANSWER_EXTRACTION_CONTRACT_VERSION",
    "AnswerExtractionResult",
    "CandidateProvenance",
    "extract_answer",
]
