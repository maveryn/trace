"""Sampling helpers for cone-net geometry cases."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Mapping

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng

from .measurements import base_radius_from_sector, height_from_sector
from .state import ConeNetCase

SLANT_HEIGHT_SUPPORT: tuple[int, ...] = tuple(range(8, 61))
THETA_DEGREE_SUPPORT: tuple[int, ...] = tuple(range(60, 301, 5))
CONE_NET_CASES: tuple[ConeNetCase, ...] = tuple(
    ConeNetCase(int(slant_height), int(theta_degrees))
    for slant_height in SLANT_HEIGHT_SUPPORT
    for theta_degrees in THETA_DEGREE_SUPPORT
    if base_radius_from_sector(ConeNetCase(int(slant_height), int(theta_degrees))) < float(slant_height)
)


def _answer_key(value: float) -> str:
    return f"{float(value):.1f}"


@lru_cache(maxsize=None)
def _cases_by_answer(target_measure: str) -> dict[str, tuple[ConeNetCase, ...]]:
    if str(target_measure) == "base_radius":
        answer_fn: Callable[[ConeNetCase], float] = base_radius_from_sector
    elif str(target_measure) == "height":
        answer_fn = height_from_sector
    else:
        raise ValueError(f"unsupported cone-net target_measure: {target_measure}")
    grouped: dict[str, list[ConeNetCase]] = {}
    for case in CONE_NET_CASES:
        grouped.setdefault(_answer_key(answer_fn(case)), []).append(case)
    return {key: tuple(values) for key, values in grouped.items()}


def _sort_answer_key(value: str) -> float:
    return float(value)


def resolve_cone_net_case(
    *,
    target_measure: str,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> tuple[ConeNetCase, int]:
    """Select one cone-net case by target answer first, then construction."""

    explicit_case = params.get("case_index")
    if explicit_case is not None:
        case_index = int(explicit_case)
        if case_index < 0 or case_index >= len(CONE_NET_CASES):
            raise ValueError(f"case_index must be in [0, {len(CONE_NET_CASES) - 1}]")
        selected = CONE_NET_CASES[int(case_index)]
    else:
        answer_cases = _cases_by_answer(str(target_measure))
        answer_keys = tuple(sorted(answer_cases, key=_sort_answer_key))
        explicit_answer = params.get("target_answer")
        if explicit_answer is not None:
            answer_key = _answer_key(float(explicit_answer))
            if answer_key not in answer_cases:
                raise ValueError(f"target_answer={explicit_answer} is not supported for {target_measure}")
        else:
            rng = spawn_rng(int(instance_seed), f"{namespace}.answer")
            answer_key = str(uniform_choice(rng, answer_keys))
        cases = tuple(answer_cases[str(answer_key)])
        rng = spawn_rng(int(instance_seed), f"{namespace}.case.{answer_key}")
        selected = uniform_choice(rng, cases)
        case_index = CONE_NET_CASES.index(selected)
    slant_height = int(params.get("slant_height", selected.slant_height))
    theta_degrees = int(params.get("theta_degrees", selected.theta_degrees))
    if slant_height <= 0:
        raise ValueError("cone net slant height must be positive")
    if not 0 < theta_degrees < 360:
        raise ValueError("cone net central angle must be between 0 and 360 degrees")
    return ConeNetCase(int(slant_height), int(theta_degrees)), int(case_index)


__all__ = ["CONE_NET_CASES", "SLANT_HEIGHT_SUPPORT", "THETA_DEGREE_SUPPORT", "resolve_cone_net_case"]
