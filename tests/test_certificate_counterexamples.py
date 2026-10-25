"""Counterexamples for every adapter (DESIGN 9 "Counterexamples", 8 go/no-go 1b).

A generic runner scores every ``Adapter.corruptions`` case of every fixture row and asserts
**every** declared expectation (check ids and the signals A/G/M/F):
``"pass"`` == 1, ``"fail"`` < 1, ``"unchanged"`` within 0.01 of the uncorrupted object's score,
or an exact float. The per-adapter files pin the adapter-specific cases (teleports, wrong
pairing, distractor sides, ...); this file pins the shared contract and the pre-registered
go/no-go criterion 1(b): at least 3 corruption types per numeric adapter leave G unchanged
(|dG| < 0.01) while lowering M.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Mapping

import pytest

from tests.test_certificate_roundtrip import ADAPTERS, CASE_IDS, CASES, FixtureCase, extension_of, oracle_object, score
from trace_tasks.experimental.certificates import registry
from trace_tasks.experimental.certificates.types import Corruption, ScoreResult

NUMERIC_ADAPTERS = ("filtered_mean", "weighted_sum", "path")  # selected_set is the control (DESIGN 8)
SIGNALS = {"A": "answer", "G": "grounding", "M": "semantics", "F": "format"}
INDEX_BINDING = ("selected_set", "filtered_mean", "weighted_sum")  # path binds by points


def corruptions(case: FixtureCase) -> List[Corruption]:
    ext = extension_of(case)
    return case.adapter.corruptions(ext.oracle, copy.deepcopy(oracle_object(ext)), case.record.instance_seed)


def value_of(result: ScoreResult, key: str) -> float:
    if key in SIGNALS:
        return float(getattr(result, SIGNALS[key]))
    found = [check.score for check in result.checks if check.check_id == key]
    if len(found) != 1:
        raise KeyError(f"check {key!r} reported {len(found)} times")
    return float(found[0])


def expectation_problems(result: ScoreResult, baseline: ScoreResult, expected: Mapping[str, Any]) -> List[str]:
    problems = []
    for key, want in expected.items():
        try:
            got = value_of(result, key)
        except KeyError as exc:
            problems.append(str(exc))
            continue
        if want == "pass":
            ok = got == 1.0
        elif want == "fail":
            ok = got < 1.0
        elif want == "unchanged":
            ok = abs(got - value_of(baseline, key)) <= 0.01
        elif isinstance(want, (int, float)) and not isinstance(want, bool):
            ok = abs(got - float(want)) <= 1e-9
        else:
            problems.append(f"{key}: unknown expectation {want!r}")
            continue
        if not ok:
            problems.append(f"{key}: expected {want!r}, got {got:.4f}")
    return problems


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_every_corruption_meets_every_expectation(case: FixtureCase) -> None:
    ext = extension_of(case)
    baseline = score(oracle_object(ext), ext)
    found = corruptions(case)
    assert found, "an adapter must declare corruptions"
    failures: Dict[str, List[str]] = {}
    for corruption in found:
        assert {"A", "G", "M", "F"} <= set(corruption.expected), corruption.name  # signals always declared
        result = score(json.dumps(corruption.response_obj), ext)
        problems = expectation_problems(result, baseline, corruption.expected)
        if problems:
            failures[corruption.name] = problems
    assert failures == {}


@pytest.mark.parametrize("name", ADAPTERS)
def test_corruptions_are_deterministic_unique_and_leave_the_input_alone(name: str) -> None:
    case = next(c for c in CASES if c.adapter_name == name)
    ext = extension_of(case)
    sft = copy.deepcopy(oracle_object(ext))
    first = case.adapter.corruptions(ext.oracle, sft, 3)
    assert sft == oracle_object(ext)
    second = case.adapter.corruptions(ext.oracle, copy.deepcopy(sft), 3)
    assert [(c.name, c.response_obj, dict(c.expected)) for c in first] == [
        (c.name, c.response_obj, dict(c.expected)) for c in second]
    names = [c.name for c in first]
    assert len(names) == len(set(names))
    if name in INDEX_BINDING:
        assert {"negative_index", "index_out_of_range"} <= set(names)


@pytest.mark.parametrize("name", NUMERIC_ADAPTERS)
def test_go_no_go_1b_corruptions_that_keep_g_and_lower_m(name: str) -> None:
    """DESIGN 8 go/no-go 1(b): >= 3 corruption types per numeric adapter with |dG| < 0.01 and
    M lowered, on every fixture row where the type applies."""
    keeps_g: Dict[str, bool] = {}
    for case in (c for c in CASES if c.adapter_name == name):
        ext = extension_of(case)
        baseline = score(oracle_object(ext), ext)
        for corruption in corruptions(case):
            result = score(json.dumps(corruption.response_obj), ext)
            ok = abs(result.grounding - baseline.grounding) < 0.01 and result.semantics < baseline.semantics
            keeps_g[corruption.name] = keeps_g.get(corruption.name, True) and ok
    passing = sorted(type_ for type_, ok in keeps_g.items() if ok)
    assert len(passing) >= 3, passing


def test_registry_never_scores_an_unsupported_task() -> None:
    resolved = registry.resolve("task_physics__lever__missing_weight_balance_value", "single")
    assert isinstance(resolved, registry.Unsupported) and "balance" in resolved.reason
    assert isinstance(registry.resolve("task_not_a_task", "single"), registry.Unsupported)
