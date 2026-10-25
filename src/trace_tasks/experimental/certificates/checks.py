"""Typed readers and check builders shared by all adapters (DESIGN 2.4, D5).

Rules encoded here:
- certificate numbers are JSON integers; booleans are never numbers; an integral float is
  read by value but flagged (F=0);
- partial-credit families use ``max(n_claimed, n_oracle)`` so claims can never raise credit;
- a gated family scores 1 iff all of its input families are 1 and the local computation on
  the model's own values is right;
- M is the mean of the adapter's fixed truth families that are applicable (oracle-only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Collection, Dict, Iterable, List, Mapping, Optional, Sequence

from .identity import RESOLVED, Resolution
from .types import ERROR_CLASSES, CheckResult


@dataclass(frozen=True)
class IntRead:
    """A typed integer read: ``value`` is None when invalid; ``integral_float`` marks ``15.0``."""

    value: Optional[int]
    integral_float: bool = False

    @property
    def ok(self) -> bool:
        return self.value is not None


def read_int(value: Any) -> IntRead:
    """JSON integer (bool rejected). Finite integral floats are accepted by value and flagged."""
    if isinstance(value, bool):
        return IntRead(None)
    if isinstance(value, int):
        return IntRead(int(value))
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return IntRead(int(value), integral_float=True)
    return IntRead(None)


def read_enum(value: Any, allowed: Collection[str]) -> Optional[str]:
    """Exact (case-sensitive) member of ``allowed``, else None."""
    return value if isinstance(value, str) and value in allowed else None


def read_bool(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def read_point(value: Any) -> Optional[tuple]:
    """``[x, y]`` of two finite non-bool numbers (any precision), else None.

    An integer too large for a float (valid JSON, e.g. ``10**400``) is not finite here: the
    element is invalid (no credit, still counted), never an exception.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        return None
    coords = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            return None
        try:
            coord = float(item)
        except (OverflowError, ValueError, TypeError):
            return None
        if not math.isfinite(coord):
            return None
        coords.append(coord)
    return (coords[0], coords[1])


def partial_credit(numerator: float, n_claimed: int, n_oracle: int) -> float:
    """``numerator / max(n_claimed, n_oracle)``; an empty claim on an empty oracle scores 1."""
    denominator = max(int(n_claimed), int(n_oracle))
    if denominator <= 0:
        return 1.0
    return min(1.0, max(0.0, float(numerator) / denominator))


def truth_check(
    check_id: str,
    score: float,
    *,
    family: Optional[str] = None,
    applicable: bool = True,
    valid_input: bool = True,
    error_code: Optional[str] = None,
    detail: Optional[Mapping[str, Any]] = None,
) -> CheckResult:
    """A truth check. Family-level results use ``family == check_id`` (the default)."""
    if not applicable:
        error_code = "not_applicable"
    return CheckResult(
        check_id=check_id,
        family=family or check_id,
        kind="truth",
        score=float(score) if applicable else 0.0,
        applicable=applicable,
        valid_input=valid_input,
        error_code=error_code,
        detail=dict(detail or {}),
    )


def local_check(
    check_id: str,
    ok: bool | float,
    *,
    family: Optional[str] = None,
    applicable: bool = True,
    valid_input: bool = True,
    error_code: Optional[str] = None,
    detail: Optional[Mapping[str, Any]] = None,
) -> CheckResult:
    """A local-consistency check (logged, never part of M)."""
    score = float(ok) if not isinstance(ok, bool) else (1.0 if ok else 0.0)
    return CheckResult(
        check_id=check_id,
        family=family or check_id,
        kind="local",
        score=score if applicable else 0.0,
        applicable=applicable,
        valid_input=valid_input,
        error_code=error_code if error_code is not None or score >= 1.0 else "inconsistent",
        detail=dict(detail or {}),
    )


def counted_truth(
    check_id: str,
    numerator: float,
    n_claimed: int,
    n_oracle: int,
    *,
    applicable: bool = True,
    error_code: Optional[str] = None,
    detail: Optional[Mapping[str, Any]] = None,
) -> CheckResult:
    """Partial-credit truth family with denominator ``max(n_claimed, n_oracle)``."""
    score = partial_credit(numerator, n_claimed, n_oracle)
    info = {"numerator": float(numerator), "n_claimed": int(n_claimed), "n_oracle": int(n_oracle)}
    info.update(detail or {})
    if error_code is None and score < 1.0:
        error_code = "partial"
    return truth_check(check_id, score, applicable=applicable, error_code=error_code, detail=info)


def _gate_value(gate: CheckResult | float) -> float:
    if isinstance(gate, CheckResult):
        return gate.score if gate.applicable else 1.0
    return float(gate)


def gated_check(
    check_id: str,
    gates: Iterable[CheckResult | float],
    value_ok: bool,
    *,
    applicable: bool = True,
    detail: Optional[Mapping[str, Any]] = None,
) -> CheckResult:
    """Gated truth family: 1 iff every gate input is exactly 1 and ``value_ok``.

    A non-applicable gate (for example an attribute family on an empty oracle set) is open.
    """
    gate_open = all(_gate_value(gate) >= 1.0 for gate in gates)
    if not gate_open:
        error_code: Optional[str] = "gate_closed"
    elif not value_ok:
        error_code = "wrong_value"
    else:
        error_code = None
    info = {"gate_open": gate_open, "value_ok": bool(value_ok)}
    info.update(detail or {})
    score = 1.0 if gate_open and value_ok else 0.0
    return truth_check(check_id, score, applicable=applicable, error_code=error_code, detail=info)


def envelope_failure_checks(
    families: Sequence[str],
    error_code: str,
    applicable: Optional[Mapping[str, bool]] = None,
) -> List[CheckResult]:
    """Every M family scores 0 with ``valid_input=False`` (DESIGN 2.1 envelope failure)."""
    applicable = applicable or {}
    return [
        CheckResult(
            check_id=family,
            family=family,
            kind="truth",
            score=0.0,
            applicable=bool(applicable.get(family, True)),
            valid_input=False,
            error_code=error_code,
        )
        for family in families
    ]


def family_results(checks: Iterable[CheckResult], families: Sequence[str]) -> Dict[str, CheckResult]:
    """The family-level result (``check_id == family``) of each listed family; missing raises."""
    by_id: Dict[str, CheckResult] = {}
    for check in checks:
        if check.check_id in families:
            if check.check_id in by_id:
                raise ValueError(f"duplicate family-level check {check.check_id!r}")
            if check.kind != "truth":
                raise ValueError(f"M family {check.check_id!r} must be a truth check")
            by_id[check.check_id] = check
    missing = [family for family in families if family not in by_id]
    if missing:
        raise ValueError(f"adapter did not emit M families {missing}")
    return by_id


def semantics_score(checks: Iterable[CheckResult], families: Sequence[str]) -> float:
    """M = mean over the applicable families of the adapter's fixed truth-family list."""
    results = family_results(checks, families)
    applicable = [results[family].score for family in families if results[family].applicable]
    if not applicable:
        raise ValueError("no applicable M family; applicability must leave at least one family")
    return float(sum(applicable) / len(applicable))


def classify_error(
    *,
    envelope_failure: bool,
    families: Mapping[str, CheckResult],
    family_stages: Mapping[str, str],
    semantics: float,
    answer: float,
) -> str:
    """First failing stage: envelope, grounding, operand, computation, answer_inconsistent, correct."""
    if envelope_failure:
        return "envelope"
    for stage in ("grounding", "operand", "computation"):
        for family, result in families.items():
            if family_stages.get(family) == stage and result.applicable and result.score < 1.0:
                return stage
    if semantics >= 1.0 and answer < 1.0:
        return "answer_inconsistent"
    if semantics >= 1.0 and answer >= 1.0:
        return "correct"
    # M < 1 with every staged family at 1 means an unstaged family failed; report computation.
    return "computation"


@dataclass(frozen=True)
class Binding:
    """One certificate element bound (through its annotation index) to a scene object.

    ``status`` is ``bound`` or the reason it is not: ``invalid_element`` (schema/index
    failure) or the identity status of its box (``unresolved``, ``ambiguous``, ``duplicate``,
    ``invalid``). ``repeated_index`` marks an index already used by an earlier element; such
    elements stay bound, so "exactly one" families give the shared object no credit.
    """

    position: int
    annotation_index: Optional[int]
    status: str
    candidate: Optional[int] = None
    repeated_index: bool = False

    @property
    def bound(self) -> bool:
        return self.status == "bound"


def bind_elements(elements: Sequence[Any], resolutions: Sequence[Resolution]) -> List[Binding]:
    """Bind elements (objects with ``valid`` and ``annotation_index``) to resolved boxes."""
    by_index = {resolution.index: resolution for resolution in resolutions}
    seen: set[int] = set()
    out: List[Binding] = []
    for position, element in enumerate(elements):
        index = getattr(element, "annotation_index", None)
        if not getattr(element, "valid", False) or index is None:
            out.append(Binding(position, index, "invalid_element"))
            continue
        repeated = index in seen
        seen.add(index)
        resolution = by_index.get(index)
        if resolution is None:
            out.append(Binding(position, index, "invalid_element", repeated_index=repeated))
        elif resolution.status == RESOLVED:
            out.append(Binding(position, index, "bound", resolution.candidate, repeated))
        else:
            out.append(Binding(position, index, resolution.status, resolution.candidate, repeated))
    return out


def claims_by_candidate(bindings: Iterable[Binding]) -> Dict[int, List[int]]:
    """Candidate -> positions of the bound elements claiming it."""
    claims: Dict[int, List[int]] = {}
    for binding in bindings:
        if binding.bound and binding.candidate is not None:
            claims.setdefault(binding.candidate, []).append(binding.position)
    return claims


def bound_exactly_once(bindings: Iterable[Binding], targets: Iterable[int]) -> Dict[int, int]:
    """Target candidate -> the single element position bound to it (targets claimed once only)."""
    claims = claims_by_candidate(bindings)
    return {target: claims[target][0] for target in targets if len(claims.get(target, [])) == 1}


def indices_bijective(elements: Sequence[Any], n_annotation: int) -> bool:
    """Local ``*.ref.local``: indices valid and distinct, and elements <-> annotation is a bijection."""
    indices = [getattr(element, "annotation_index", None) for element in elements]
    if any(index is None for index in indices) or not all(getattr(e, "valid", False) for e in elements):
        return False
    return len(indices) == n_annotation and sorted(indices) == list(range(n_annotation))


__all__ = [
    "Binding",
    "ERROR_CLASSES",
    "IntRead",
    "bind_elements",
    "bound_exactly_once",
    "claims_by_candidate",
    "classify_error",
    "counted_truth",
    "envelope_failure_checks",
    "family_results",
    "gated_check",
    "indices_bijective",
    "local_check",
    "partial_credit",
    "read_bool",
    "read_enum",
    "read_int",
    "read_point",
    "semantics_score",
    "truth_check",
]
