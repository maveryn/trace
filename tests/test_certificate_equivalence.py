"""Equivalent responses score identically (DESIGN 9 "Equivalence", 2.1).

On every fixture row of every adapter:

- JSON key order never matters (every object's keys reversed, recursively);
- a permuted annotation with re-bound ``annotation_index`` values and a shuffled element list
  score identically (index-binding adapters; table rows permute both box lists together);
- the same certificate object scores identical A, G and M whether it is the final object, sits
  in a ```json fence or is followed by prose; only F differs, and only for trailing prose.

All optimal rpg paths scoring identically is checked in ``test_certificate_roundtrip.py``.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List

import pytest

from tests.test_certificate_roundtrip import CASE_IDS, CASES, FixtureCase, extension_of, oracle_object, score, score_vector

INDEX_CASES = [case for case in CASES if case.adapter.list_key in ("members", "rows", "terms")]


def reverse_keys(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: reverse_keys(node[key]) for key in reversed(list(node))}
    if isinstance(node, list):
        return [reverse_keys(item) for item in node]
    return node


def rebound(obj: Dict[str, Any], roles: List[str], rng: random.Random) -> Dict[str, Any]:
    """Permute the annotation boxes (every role alike), re-bind the indices, shuffle elements."""
    out = json.loads(json.dumps(obj))
    key = next(k for k in ("members", "rows", "terms") if k in out["certificate"])
    n = len(out["annotation"][roles[0]] if roles[0] else out["annotation"])
    order = list(range(n))
    rng.shuffle(order)  # new position j holds old box order[j]
    new_index = {old: new for new, old in enumerate(order)}
    for role in roles:
        boxes = out["annotation"][role] if role else out["annotation"]
        permuted = [boxes[old] for old in order]
        if role:
            out["annotation"][role] = permuted
        else:
            out["annotation"] = permuted
    for element in out["certificate"][key]:
        element["annotation_index"] = new_index[element["annotation_index"]]
    rng.shuffle(out["certificate"][key])
    return out


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_key_order_never_matters(case: FixtureCase) -> None:
    ext = extension_of(case)
    obj = oracle_object(ext)
    reference = score_vector(score(obj, ext))
    assert score_vector(score(reverse_keys(obj), ext)) == reference
    compact = json.dumps(reverse_keys(obj), separators=(",", ":"))
    assert score_vector(score(compact, ext)) == reference


@pytest.mark.parametrize("case", INDEX_CASES, ids=[str(c) for c in INDEX_CASES])
def test_permuted_annotation_with_rebound_indices(case: FixtureCase) -> None:
    ext = extension_of(case)
    roles = [role for role in case.adapter.annotation_roles]
    reference = score_vector(score(oracle_object(ext), ext))
    rng = random.Random(case.record.instance_seed)
    for _ in range(3):
        assert score_vector(score(rebound(oracle_object(ext), roles, rng), ext)) == reference


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_position_changes_only_f(case: FixtureCase) -> None:
    ext = extension_of(case)
    text = json.dumps(oracle_object(ext))
    final = score(text, ext)
    assert final.format == 1.0
    leading = score("Reading the image step by step.\n" + text, ext)
    fenced = score("Here is the result.\n```json\n" + text + "\n```", ext)
    for other in (leading, fenced):
        assert score_vector(other) == score_vector(final)
    trailing = score(text + "\nThat is the answer.", ext)
    assert (trailing.answer, trailing.grounding, trailing.semantics) == (final.answer, final.grounding, final.semantics)
    assert [c.score for c in trailing.checks] == [c.score for c in final.checks]
    assert trailing.format == 0.0 and "not_final" in trailing.diagnostics["format_reasons"]
