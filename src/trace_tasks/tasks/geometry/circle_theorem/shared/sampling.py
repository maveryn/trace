"""Candidate enumeration primitives for circle-theorem tasks."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .state import (
    DIAMETER_CHORD_MAX_CHORD,
    DIAMETER_CHORD_RADIUS_MAX,
    TANGENT_SECANT_INTERNAL_MAX,
    TANGENT_SECANT_OUTSIDE_MAX,
    TANGENT_SECANT_TANGENT_MAX,
    TANGENT_SECANT_TARGET_KINDS,
    VARIABLE_SECANT_TARGET_KINDS,
)

def _candidate_diameter_chord_values(target_answer: int) -> List[Dict[str, int]]:
    candidates: List[Dict[str, int]] = []
    for radius in range(8, DIAMETER_CHORD_RADIUS_MAX + 1):
        for offset in range(2, radius - 1):
            half_chord_sq = (radius * radius) - (offset * offset)
            half_chord = int(math.isqrt(int(half_chord_sq)))
            if int(half_chord * half_chord) != int(half_chord_sq):
                continue
            answer_value = int(radius - offset)
            if int(answer_value) != int(target_answer):
                continue
            if int(answer_value) < 4:
                continue
            chord_length = int(2 * half_chord)
            diameter_length = int(2 * radius)
            if half_chord < 4 or chord_length > DIAMETER_CHORD_MAX_CHORD:
                continue
            candidates.append(
                {
                    "radius": int(radius),
                    "offset": int(offset),
                    "half_chord": int(half_chord),
                    "diameter": int(diameter_length),
                    "chord": int(chord_length),
                    "answer": int(answer_value),
                }
            )
    return candidates

def _hard_tangent_secant_triple(*, outside: int, internal: int, tangent: int) -> bool:
    if int(outside) < 16 or int(internal) < 10:
        return False
    if len({int(outside), int(internal), int(tangent)}) != 3:
        return False
    if int(tangent) == int(2 * outside) or int(tangent) % int(outside) == 0:
        return False
    if int(internal) in {int(outside), int(2 * outside), int(3 * outside)}:
        return False
    if int(outside) % 2 == 0 and int(internal) == int(outside // 2):
        return False
    return True

def _candidate_tangent_secant_values(
    target_answer: int, *, target_kind: str | None = None
) -> List[Dict[str, int | str]]:
    """Enumerate integer tangent-secant power configurations by target answer."""
    if target_kind is not None and str(target_kind) not in TANGENT_SECANT_TARGET_KINDS:

        raise ValueError(f"unsupported tangent secant target kind: {target_kind}")
    candidates: List[Dict[str, int | str]] = []
    for outside in range(16, TANGENT_SECANT_OUTSIDE_MAX + 1):
        for internal in range(10, TANGENT_SECANT_INTERNAL_MAX + 1):
            tangent_sq = int(outside * (outside + internal))
            tangent = int(math.isqrt(tangent_sq))
            if int(tangent * tangent) != int(tangent_sq):
                continue
            if int(tangent) > TANGENT_SECANT_TANGENT_MAX:
                continue
            if not _hard_tangent_secant_triple(
                outside=int(outside), internal=int(internal), tangent=int(tangent)
            ):
                continue
            base_spec = {
                "PA": int(outside),
                "AB": int(internal),
                "PT": int(tangent),
                "PB": int(outside + internal),
            }
            target_options = {
                "outside": ("PA", int(outside)),
                "inside": ("AB", int(internal)),
                "tangent": ("PT", int(tangent)),
            }
            for option_kind, (answer_segment, answer_value) in target_options.items():
                if target_kind is not None and str(option_kind) != str(target_kind):
                    continue
                if int(answer_value) != int(target_answer):
                    continue
                candidates.append(
                    {
                        **base_spec,
                        "target_kind": str(option_kind),
                        "canonical_answer_segment": str(answer_segment),
                        "answer": int(answer_value),
                    }
                )
    return candidates

def _feasible_tangent_secant_target_kinds(target_answer: int) -> Tuple[str, ...]:
    kinds = sorted(
        {
            str(candidate["target_kind"])
            for candidate in _candidate_tangent_secant_values(int(target_answer))
        }
    )
    return tuple(kind for kind in TANGENT_SECANT_TARGET_KINDS if kind in set(kinds))

def _candidate_secant_secant_values(target_answer: int) -> List[Dict[str, int]]:
    candidates: List[Dict[str, int]] = []
    pa = int(target_answer)
    for ab in range(11, 201):
        power = int(pa * (pa + ab))
        center_x = float(pa + (0.5 * ab))
        for pc in range(3, 181):
            if int(power) % int(pc) != 0:
                continue
            pd = int(power // pc)
            cd = int(pd - pc)
            if not (3 <= int(cd) <= 260):
                continue
            if int(pc) == int(pa) and int(cd) == int(ab):
                continue
            cos_theta = float(pc + pd) / float(2.0 * center_x)
            if 0.18 <= float(cos_theta) <= 0.92:
                candidates.append(
                    {
                        "PA": int(pa),
                        "AB": int(ab),
                        "PB": int(pa + ab),
                        "PC": int(pc),
                        "CD": int(cd),
                        "PD": int(pd),
                        "power": int(power),
                    }
                )
    return candidates

def _candidate_secant_secant_variable_values(
    target_answer: int,
    *,
    target_kind: str | None = None,
) -> List[Dict[str, int | str]]:
    """Enumerate integer two-secant power configurations for all hidden segment choices."""
    if (

        target_kind is not None
        and str(target_kind) not in VARIABLE_SECANT_TARGET_KINDS
    ):
        raise ValueError(f"unsupported secant secant target kind: {target_kind}")
    candidates: List[Dict[str, int | str]] = []
    for pa in range(4, 31):
        for ab in range(8, 61):
            power = int(pa * (pa + ab))
            center_x = float(pa + (0.5 * ab))
            for pc in range(4, 31):
                if int(power) % int(pc) != 0:
                    continue
                pd = int(power // pc)
                cd = int(pd - pc)
                if not (8 <= int(cd) <= 60):
                    continue
                if int(pc) == int(pa) and int(cd) == int(ab):
                    continue
                cos_theta = float(pc + pd) / float(2.0 * center_x)
                if not (0.20 <= float(cos_theta) <= 0.88):
                    continue
                values = (int(pa), int(ab), int(pc), int(cd))
                if len(set(values)) != 4:
                    continue
                if int(ab) in {int(pa), int(2 * pa), int(3 * pa)}:
                    continue
                if int(cd) in {int(pc), int(2 * pc), int(3 * pc)}:
                    continue
                base_spec = {
                    "PA": int(pa),
                    "AB": int(ab),
                    "PB": int(pa + ab),
                    "PC": int(pc),
                    "CD": int(cd),
                    "PD": int(pd),
                    "power": int(power),
                }
                target_options = {
                    "outside_first": ("PA", int(pa)),
                    "inside_first": ("AB", int(ab)),
                    "outside_second": ("PC", int(pc)),
                    "inside_second": ("CD", int(cd)),
                }
                for option_kind, (
                    answer_segment,
                    answer_value,
                ) in target_options.items():
                    if target_kind is not None and str(option_kind) != str(target_kind):
                        continue
                    if int(answer_value) != int(target_answer):
                        continue
                    candidates.append(
                        {
                            **base_spec,
                            "target_kind": str(option_kind),
                            "canonical_answer_segment": str(answer_segment),
                            "answer": int(answer_value),
                        }
                    )
    return candidates

def _feasible_secant_secant_variable_target_kinds(
    target_answer: int,
) -> Tuple[str, ...]:
    kinds = sorted(
        {
            str(candidate["target_kind"])
            for candidate in _candidate_secant_secant_variable_values(
                int(target_answer)
            )
        }
    )
    return tuple(
        kind for kind in VARIABLE_SECANT_TARGET_KINDS if kind in set(kinds)
    )

__all__ = [
    '_candidate_diameter_chord_values',
    '_hard_tangent_secant_triple',
    '_candidate_tangent_secant_values',
    '_feasible_tangent_secant_target_kinds',
    '_candidate_secant_secant_values',
    '_candidate_secant_secant_variable_values',
    '_feasible_secant_secant_variable_target_kinds',
]
