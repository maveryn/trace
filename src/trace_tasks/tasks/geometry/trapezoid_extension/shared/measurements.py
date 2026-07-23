"""Identity-free trapezoid-extension measurement formulas."""

from __future__ import annotations

from collections.abc import Iterable

from trace_tasks.tasks.geometry.shared.measurement_rendering import fmt_measure, round1

from .state import TrapezoidExtensionCase


def iter_cases(
    *,
    top_bases: Iterable[int],
    extensions: Iterable[int],
    heights: Iterable[int],
    sides: Iterable[int],
) -> tuple[TrapezoidExtensionCase, ...]:
    """Build positive numeric construction cases from finite supports."""

    cases: list[TrapezoidExtensionCase] = []
    for top_base in top_bases:
        for extension in extensions:
            for height in heights:
                for side in sides:
                    if min(int(top_base), int(extension), int(height), int(side)) <= 0:
                        continue
                    cases.append(
                        TrapezoidExtensionCase(
                            top_base=int(top_base),
                            extension=int(extension),
                            height=int(height),
                            side=int(side),
                        )
                    )
    return tuple(cases)


def extension_case_pool(
    *,
    extensions: Iterable[int] = range(3, 81),
    variants_per_answer: int = 3,
) -> tuple[TrapezoidExtensionCase, ...]:
    """Build compact extension-answer cases with deterministic distractor dimensions."""

    cases: list[TrapezoidExtensionCase] = []
    for extension in extensions:
        for variant in range(max(1, int(variants_per_answer))):
            ext = int(extension)
            top_base = 6 + ((ext + variant * 7) % 18)
            height = 4 + ((ext * 3 + variant * 5) % 15)
            side = 5 + ((ext * 5 + variant * 2) % 20)
            cases.append(
                TrapezoidExtensionCase(
                    top_base=int(top_base),
                    extension=int(ext),
                    height=int(height),
                    side=int(side),
                )
            )
    return tuple(cases)


def area_case_pool(
    *,
    top_bases: Iterable[int] = range(4, 30),
    extensions: Iterable[int] = range(3, 35),
    heights: Iterable[int] = range(3, 22),
) -> tuple[TrapezoidExtensionCase, ...]:
    """Build compact area-answer cases with deterministic side distractors."""

    cases: list[TrapezoidExtensionCase] = []
    for top_base in top_bases:
        for extension in extensions:
            for height in heights:
                side = 5 + ((int(top_base) + int(extension) + int(height)) % 8)
                cases.append(
                    TrapezoidExtensionCase(
                        top_base=int(top_base),
                        extension=int(extension),
                        height=int(height),
                        side=int(side),
                    )
                )
    return tuple(cases)


def extension_length(case: TrapezoidExtensionCase) -> float:
    return float(case.extension)


def trapezoid_area(case: TrapezoidExtensionCase) -> float:
    return round1(float(case.trapezoid_area))


def completion_length_from_area(case: TrapezoidExtensionCase) -> float:
    return round1(float(case.parallelogram_area) / float(case.height) - float(case.top_base))


def completion_length_from_perimeter(case: TrapezoidExtensionCase) -> float:
    return round1(float(case.parallelogram_perimeter) / 2.0 - float(case.side) - float(case.top_base))


def trapezoid_area_from_parallelogram_perimeter(case: TrapezoidExtensionCase) -> float:
    derived_bottom_base = (float(case.parallelogram_perimeter) / 2.0) - float(case.side)
    return round1(float(case.height) * (float(case.top_base) + derived_bottom_base) / 2.0)


def case_trace_values(case: TrapezoidExtensionCase) -> dict[str, float | int]:
    return {
        "top_base": int(case.top_base),
        "extension": int(case.extension),
        "bottom_base": int(case.bottom_base),
        "height": int(case.height),
        "side": int(case.side),
        "parallelogram_area": int(case.parallelogram_area),
        "parallelogram_perimeter": int(case.parallelogram_perimeter),
        "trapezoid_area": float(trapezoid_area(case)),
    }


__all__ = [
    "case_trace_values",
    "area_case_pool",
    "completion_length_from_area",
    "completion_length_from_perimeter",
    "extension_case_pool",
    "extension_length",
    "fmt_measure",
    "iter_cases",
    "round1",
    "trapezoid_area",
]
