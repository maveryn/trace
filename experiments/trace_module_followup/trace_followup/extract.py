"""Narrow, gold-blind final-option extractor for ``format-normalized-v2`` (follow-up work package F0).

v2 (supersedes v1 before any confirmation output was read): ``extract_final_boxed`` also removes the LaTeX escapes
``\\%``, ``\\$``, ``\\&``, ``\\#``, ``\\_`` inside the boxed text (``\\boxed{55\\%}`` designates ``55%``).
``extract_final_option`` is unchanged from v1.

Purpose: tell an answer-*format* artifact (the model designates option D as its final answer but writes it in a form the
pinned scorer does not read, e.g. ``Answer: \\( D \\)`` or ``Answer: (D)``) apart from a capability difference (the
model chose a wrong option, or never committed to one). ``extract_final_option`` sees only the response text and the
allowed option labels. It never sees the gold answer, the question, the option texts or the official score, so it
cannot prefer whichever letter happens to be correct; scoring against gold happens afterwards, in ``rescore.py``.

Grammar (one rule for every model, interface and benchmark; precedence top to bottom):

1. Final region R (reasoning outside R is never a source of the answer).
   a. ``answer_tag``: the response has complete ``<answer>...</answer>`` blocks (any case): R = their contents.
   b. ``answer_tag_open``: no complete block, but a lower-case ``<answer>`` opener: R = text after the last opener.
   c. ``after_think``: otherwise, if ``</think>`` occurs: R = text after the last ``</think>``.
   d. ``unterminated_think``: otherwise, if ``<think>`` occurs (reasoning never ended): no R -> ``no_final_answer``.
   e. ``final_paragraph``: otherwise R = the last paragraph (text after the last blank line; the whole response when
      it has no blank line).
2. Designators inside R, all collected in text order:
   * ``boxed``: every ``\\boxed{...}`` with balanced braces; value = the option token of its content (rule 4);
   * ``answer_statement``: head (``final answer``, ``correct/best answer|option|choice``, ``answer``, ``option``,
     ``choice``) + copula (``is``, ``:``, ``=``, ``would/should/will be``, optionally followed by ``:``) + an option
     token, possibly wrapped; or ``option/choice X is (the) correct/right/best``;
   * ``answer_block``: in a tag region with no other designator, the block content itself when it is one option token;
   * ``final_line``: the last non-empty line of R when it is one option token on its own (``B``, ``(B)``, ``**B**``,
     ``\\( B \\)``, ``$B$``, ``Option B``, ``B.``); when R is a single line, ``B. text`` / ``(B) text`` also count.
3. Decision.
   * No designator in R -> ``no_final_answer``.
   * Every designator in R names the same allowed label X, and no explicit designator (``boxed`` or
     ``answer_statement``) *outside* R names a different allowed label -> ``resolved`` X.
   * Two different labels inside R, or a different label outside R -> ``ambiguous``. Text outside R can therefore
     block a resolution but never supply one.
   * Designators exist but none names a single allowed label (``\\boxed{12}``, ``answer is E`` with labels A-D,
     ``answer is B, D``) -> ``not_an_option``; a mix of one label and non-option designators -> ``ambiguous``.
   There is no "take the last one" rule.
4. Option token: one capital letter, alone after removing harmless wrappers (whitespace, ``\\( \\)``, ``\\[ \\]``,
   ``$ $``, ``$$ $$``, ``\\text{}``-style macros, ``( )``, ``[ ]``, ``{ }``, ``** **``, ``* *``, ``__ __``, quotes,
   an ``Option``/``Choice``/``Answer:`` prefix, a trailing period). Inside ``\\boxed{}`` and answer blocks a lower-case
   letter is accepted too, and so is a letter followed by a separator and option text (``\\boxed{B. 12}``,
   ``(B) 12``) unless that text names another option label (``(A) 3, (B) 5`` is a listing). A letter
   followed by ``and``/``or``/``,``/``/``/``&`` and another letter names several options (not a single option). In an
   answer statement only capitals count, and an unwrapped ``A`` or ``I`` (English words) counts only when followed by
   punctuation, a line end or the end of the text.

``extract_final_boxed`` is the free-form counterpart used for TableVQABench (same region rule 1; only ``\\boxed{}``
designators; all boxes in R must agree exactly and no box outside R may disagree).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

EXTRACTOR_VERSION = "format-normalized-v2/extract"

RESOLVED = "resolved"
AMBIGUOUS = "ambiguous"
NOT_AN_OPTION = "not_an_option"
NO_FINAL_ANSWER = "no_final_answer"
STATUSES = (RESOLVED, AMBIGUOUS, NOT_AN_OPTION, NO_FINAL_ANSWER)

NON_OPTION = "<non-option>"          # a designator whose value is not one allowed label
MULTI_OPTION = "<several-options>"   # a designator naming several labels
EXPLICIT_SOURCES = ("boxed", "answer_statement")


@dataclass(frozen=True)
class Extraction:
    status: str
    option: str | None
    region: str
    rule: str | None                                                         # source of the last designator in R
    designations: tuple[tuple[str, str], ...] = field(default_factory=tuple)  # (source, value) inside R
    outside_conflicts: tuple[str, ...] = field(default_factory=tuple)         # other labels designated outside R

    @property
    def resolved(self) -> bool:
        return self.status == RESOLVED


# ---------------------------------------------------------------------------------------------------- regions
_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)
_ANSWER_OPEN = re.compile(r"<answer>")
_THINK_CLOSE = re.compile(r"</think>", re.I)
_THINK_OPEN = re.compile(r"<think>", re.I)
_BLANK_LINE = re.compile(r"\n[ \t\r]*\n")


def final_regions(response: str) -> tuple[str, list[tuple[int, int]]]:
    """(region kind, [(start, end)] spans of R in the response) per rule 1."""

    text = response or ""
    blocks = [m.span(1) for m in _ANSWER_BLOCK.finditer(text)]
    if blocks:
        return "answer_tag", blocks
    opens = list(_ANSWER_OPEN.finditer(text))
    if opens:
        return "answer_tag_open", [(opens[-1].end(), len(text))]
    closes = list(_THINK_CLOSE.finditer(text))
    if closes:
        return "after_think", [(closes[-1].end(), len(text))]
    if _THINK_OPEN.search(text):
        return "unterminated_think", []
    stripped_end = len(text.rstrip())
    breaks = [m.end() for m in _BLANK_LINE.finditer(text, 0, stripped_end)]
    return "final_paragraph", [(breaks[-1] if breaks else 0, len(text))]


# ------------------------------------------------------------------------------------------------ option tokens
# (pattern, delimiter text the inner part must not contain, bracket pair the inner part must balance)
_WRAPPERS = (
    (re.compile(r"\\\(\s*(.*?)\s*\\\)", re.S), ("\\(", "\\)"), None),
    (re.compile(r"\\\[\s*(.*?)\s*\\\]", re.S), ("\\[", "\\]"), None),
    (re.compile(r"\$\$\s*(.*?)\s*\$\$", re.S), ("$",), None),
    (re.compile(r"\$\s*(.*?)\s*\$", re.S), ("$",), None),
    (re.compile(r"\\(?:text|textbf|textrm|mathrm|mathbf|mathit|mbox)\s*\{\s*(.*?)\s*\}", re.S), (), "{}"),
    (re.compile(r"\*\*\s*(.*?)\s*\*\*", re.S), ("*",), None),
    (re.compile(r"__\s*(.*?)\s*__", re.S), ("__",), None),
    (re.compile(r"\*\s*(.*?)\s*\*", re.S), ("*",), None),
    (re.compile(r"\(\s*(.*?)\s*\)", re.S), (), "()"),
    (re.compile(r"\[\s*(.*?)\s*\]", re.S), (), "[]"),
    (re.compile(r"\{\s*(.*?)\s*\}", re.S), (), "{}"),
    (re.compile(r"[\"'`“‘]\s*(.*?)\s*[\"'`”’]", re.S), ("\"", "'", "`", "“", "”", "‘", "’"), None),
)


def _pair_balanced(text: str, pair: str) -> bool:
    depth = 0
    for ch in text:
        depth += (ch == pair[0]) - (ch == pair[1])
        if depth < 0:
            return False
    return depth == 0


_PREFIX = re.compile(r"^(?:(?:final\s+)?answer\s*[:：]\s*|(?:option|choice)\s+)", re.I)
_LETTER_WITH_TEXT = re.compile(r"^(?:\(\s*([A-Za-z])\s*\)|\[\s*([A-Za-z])\s*\]|([A-Za-z])\s*(?:[.:：)]|\s-))\s*\S")
_OTHER_LABEL = re.compile(r"\(\s*[A-Z]\s*\)|\b[A-Z][.)](?=\s)")
_MULTI_TAIL = re.compile(r"^\s*(?:,|[Aa]nd\b|[Oo]r\b|&|/|、)\s*(?:[Oo]ption\s+|[Cc]hoice\s+)?[\\(\[{$*]*\s*[A-Z]"
                         r"(?![A-Za-z])")


def _unwrap(text: str) -> str:
    """Strip harmless wrappers (rule 4) until nothing changes."""

    s = text.strip()
    while s:
        before = s
        s = s.rstrip(".。").strip()
        s = _PREFIX.sub("", s).strip()
        for pat, forbidden, pair in _WRAPPERS:
            m = pat.fullmatch(s)
            if m and not any(f in m.group(1) for f in forbidden) and (pair is None or _pair_balanced(m.group(1), pair)):
                s = m.group(1).strip()
                break
        if s == before:
            break
    return s


def option_token(text: str, labels: Sequence[str], *, allow_lower: bool, allow_text: bool) -> str:
    """The single label ``text`` designates, ``MULTI_OPTION`` or ``NON_OPTION`` (rule 4)."""

    s = _unwrap(text)
    if len(s) == 1 and s.isalpha():
        letter = s.upper() if allow_lower else s
        return letter if letter in labels else NON_OPTION
    if re.match(r"^[A-Za-z](?![A-Za-z0-9])", s) and _MULTI_TAIL.match(s[1:]):
        return MULTI_OPTION
    if allow_text:
        m = _LETTER_WITH_TEXT.match(s)
        if m:
            letter = next(g for g in m.groups() if g)
            letter = letter.upper() if allow_lower else letter
            if _OTHER_LABEL.search(s[m.end() - 1:]):
                return MULTI_OPTION                      # "(A) 3, (B) 5": a listing of options, not a choice
            return letter if letter in labels else NON_OPTION
    return NON_OPTION


# ------------------------------------------------------------------------------------------------ designators
def boxed_spans(text: str) -> list[tuple[int, str]]:
    """(start, content) of every ``\\boxed{...}`` whose braces balance, in order."""

    out: list[tuple[int, str]] = []
    for m in re.finditer(r"\\boxed\s*\{", text):
        depth, pos = 1, m.end()
        while pos < len(text) and depth:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            out.append((m.start(), text[m.end():pos - 1]))
    return out


_TOKEN = (r"(?P<tok>(?:\\\(\s*|\\\[\s*|\$\s*|\*\*\s*|\*\s*|\(\s*|\[\s*|\{\s*|[\"'“‘]\s*|"
          r"\\(?:text|textbf|mathrm|mathbf)\s*\{\s*|(?:option|choice)\s+)*"
          r"(?P<letter>[A-Za-z])(?![A-Za-z0-9]))")
_HEAD = r"(?:final\s+answer|correct\s+(?:answer|option|choice)|best\s+(?:answer|option|choice)|answer|option|choice)"
_COPULA = r"(?:\s+is|\s*[:：=]|\s+would\s+be|\s+should\s+be|\s+will\s+be)\s*[:：]?"
_STATEMENT = re.compile(r"\b" + _HEAD + _COPULA + r"\s*" + _TOKEN, re.I)
_STATEMENT_REVERSED = re.compile(r"\b(?:option|choice)\s+" + _TOKEN + r"\s+is\s+(?:the\s+)?(?:correct|right|best)\b",
                                 re.I)
_CLOSERS = re.compile(r"^(?:\s*(?:\\\)|\\\]|\$\$?|\*\*?|[)\]}\"'”’]))+")
_STRICT_AFTER = re.compile(r"\s*(?:[.,;:!?)\]}\\$*\"'”’]|\n|$)")


def _statement_value(match: re.Match, text: str, labels: Sequence[str]) -> str | None:
    """Value of one answer statement, or None when the matched letter is prose, not a designation."""

    letter, tok = match.group("letter"), match.group("tok")
    after = text[match.end():]
    if not letter.isupper():
        return None                                   # "the answer is a ..." is prose
    if letter in ("A", "I") and tok == letter and not _STRICT_AFTER.match(after):
        return None                                   # "the answer is A pattern ..." is prose
    if _MULTI_TAIL.match(_CLOSERS.sub("", after)):
        return MULTI_OPTION                           # "the answer is \\( A \\) or \\( C \\)"
    return letter if letter in labels else NON_OPTION


def explicit_designators(text: str, labels: Sequence[str], lo: int = 0, hi: int | None = None
                         ) -> list[tuple[int, str, str]]:
    """(position, source, value) of every ``boxed`` and ``answer_statement`` designator starting in [lo, hi)."""

    hi = len(text) if hi is None else hi
    found: list[tuple[int, str, str]] = []
    for start, content in boxed_spans(text):
        if lo <= start < hi:
            found.append((start, "boxed", option_token(content, labels, allow_lower=True, allow_text=True)))
    window = text[:hi]
    for pat in (_STATEMENT, _STATEMENT_REVERSED):
        for m in pat.finditer(window, lo):
            value = _statement_value(m, text, labels)
            if value is not None:
                found.append((m.start(), "answer_statement", value))
    return sorted(found)


def _final_line(region: str, labels: Sequence[str]) -> str | None:
    lines = [ln for ln in region.splitlines() if ln.strip()]
    if not lines:
        return None
    value = option_token(lines[-1], labels, allow_lower=False, allow_text=len(lines) == 1)
    return value if value in labels else None


def extract_final_option(response: str, labels: Iterable[str]) -> Extraction:
    """Apply the grammar in the module docstring. ``labels`` are the allowed option labels (capital letters)."""

    labels = tuple(dict.fromkeys(str(l).strip().upper() for l in labels))
    if not labels or any(len(l) != 1 or not l.isalpha() for l in labels):
        raise ValueError(f"option labels must be single letters, got {labels!r}")
    text = response if isinstance(response, str) else ""
    kind, spans = final_regions(text)
    if not spans:
        return Extraction(NO_FINAL_ANSWER, None, kind, None)
    tagged = kind.startswith("answer_tag")
    inside: list[tuple[str, str]] = []
    for lo, hi in spans:
        region = text[lo:hi]
        found = [(src, val) for _, src, val in explicit_designators(text, labels, lo, hi)]
        if tagged and not found and region.strip():
            found.append(("answer_block", option_token(region, labels, allow_lower=True, allow_text=True)))
        if not tagged:
            line = _final_line(region, labels)
            if line is not None:
                found.append(("final_line", line))
        inside.extend(found)
    if not inside:
        return Extraction(NO_FINAL_ANSWER, None, kind, None)
    values = [v for _, v in inside]
    letters = sorted({v for v in values if v in labels})
    rule = inside[-1][0]
    if not letters:
        return Extraction(NOT_AN_OPTION, None, kind, rule, tuple(inside))
    if len(letters) > 1 or any(v not in labels for v in values):
        return Extraction(AMBIGUOUS, None, kind, rule, tuple(inside))
    # text outside R may block (a different label explicitly designated there) but never supplies an answer
    outside: list[str] = []
    cursor = 0
    for lo, hi in [*spans, (len(text), len(text))]:
        outside += [v for _, _, v in explicit_designators(text, labels, cursor, lo) if v in labels and v != letters[0]]
        cursor = hi
    if outside:
        return Extraction(AMBIGUOUS, None, kind, rule, tuple(inside), tuple(dict.fromkeys(outside)))
    return Extraction(RESOLVED, letters[0], kind, rule, tuple(inside))


# --------------------------------------------------------------------------------------- free-form final answer
_MACRO_WRAPPERS = (
    re.compile(r"\\(?:text|textbf|textrm|mathrm|mathbf|mbox)\s*\{\s*(.*)\s*\}", re.S),
    re.compile(r"\\\(\s*(.*?)\s*\\\)", re.S),
    re.compile(r"\$\s*([^$]*?)\s*\$", re.S),
)


_LATEX_ESCAPE = re.compile(r"\\([%$&#_])")


def _balanced(text: str) -> bool:
    depth = 0
    for ch in text:
        depth += (ch == "{") - (ch == "}")
        if depth < 0:
            return False
    return depth == 0


def _unwrap_macros(text: str) -> str:
    s = text.strip()
    while True:
        for pat in _MACRO_WRAPPERS:
            m = pat.fullmatch(s)
            if m and _balanced(m.group(1)):
                s = m.group(1).strip()
                break
        else:
            return s


def extract_final_boxed(response: str) -> Extraction:
    """Free-form variant (TableVQABench rule T1): the content of the ``\\boxed{}`` designators in the final region.

    Region R as in rule 1. Resolved when R holds at least one balanced ``\\boxed{}``, all boxes in R hold the same
    non-empty text (after stripping whole-content ``\\text{}``-style, ``\\( \\)`` and ``$ $`` wrappers and
    un-escaping ``\\% \\$ \\& \\# \\_``), and no box
    outside R holds a different text. Text outside boxes is never read. ``option`` carries the boxed text.
    """

    text = response if isinstance(response, str) else ""
    kind, spans = final_regions(text)
    if not spans:
        return Extraction(NO_FINAL_ANSWER, None, kind, None)
    boxes = [(start, _LATEX_ESCAPE.sub(r"\1", _unwrap_macros(content))) for start, content in boxed_spans(text)]
    inside = [c for start, c in boxes if any(lo <= start < hi for lo, hi in spans)]
    if not inside:
        return Extraction(NO_FINAL_ANSWER, None, kind, None)
    found = tuple(("boxed", c) for c in inside)
    distinct = list(dict.fromkeys(inside))
    if len(distinct) > 1:
        return Extraction(AMBIGUOUS, None, kind, "boxed", found)
    if not distinct[0]:
        return Extraction(NOT_AN_OPTION, None, kind, "boxed", found)
    outside = [c for start, c in boxes if not any(lo <= start < hi for lo, hi in spans) and c and c != distinct[0]]
    if outside:
        return Extraction(AMBIGUOUS, None, kind, "boxed", found, tuple(dict.fromkeys(outside)))
    return Extraction(RESOLVED, distinct[0], kind, "boxed", found)
