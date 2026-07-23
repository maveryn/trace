"""Sampling primitives for stack-stability diagrams."""

from __future__ import annotations

from typing import Tuple

from trace_tasks.core.seed import spawn_rng

from .state import (
    OPTION_LETTERS,
    STATUS_STABLE,
    STATUS_TIPPING,
    TIP_DIRECTIONS,
    SCENE_NAMESPACE,
    StackCandidateSpec,
    StackProfile,
    StackSceneSpec,
)


_STABLE_OFFSET_PATTERNS: Tuple[Tuple[float, ...], ...] = (
    (0.0, 0.10, -0.06, 0.05),
    (0.0, -0.18, -0.06, 0.08, 0.16),
    (0.0, 0.22, 0.08, -0.08),
    (0.0, -0.22, -0.10, 0.06, 0.12),
    (0.0, 0.26, 0.12, -0.02, -0.10),
    (0.0, -0.26, -0.14, 0.02, 0.08),
)
_TIPPING_RIGHT_OFFSET_PATTERNS: Tuple[Tuple[float, ...], ...] = (
    (0.0, 0.48, 0.92, 1.34),
    (0.0, 0.40, 0.84, 1.22, 1.58),
    (0.0, 0.54, 0.98, 1.38),
    (0.0, 0.36, 0.76, 1.14, 1.48),
)
_BRICK_PALETTES: Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], ...] = (
    ((198, 92, 70), (115, 50, 42)),
    ((211, 130, 76), (121, 72, 40)),
    ((177, 110, 91), (98, 60, 54)),
    ((189, 146, 88), (105, 79, 45)),
    ((158, 118, 95), (84, 64, 55)),
    ((207, 116, 102), (117, 61, 57)),
    ((180, 125, 72), (101, 66, 37)),
    ((196, 103, 65), (111, 55, 37)),
)


def profile_for_status(
    rng,
    *,
    status: str,
    forced_tip_direction: str | None = None,
) -> StackProfile:
    """Return row offsets that make a stack visibly stable or tipping."""

    if str(status) == STATUS_STABLE:
        offsets = tuple(float(value) for value in rng.choice(_STABLE_OFFSET_PATTERNS))
        return StackProfile(
            status=STATUS_STABLE,
            tip_direction=None,
            row_offsets=offsets,
        )
    direction = str(forced_tip_direction or rng.choice(TIP_DIRECTIONS))
    base_offsets = tuple(float(value) for value in rng.choice(_TIPPING_RIGHT_OFFSET_PATTERNS))
    if direction == "left":
        base_offsets = tuple(-float(value) for value in base_offsets)
    return StackProfile(
        status=STATUS_TIPPING,
        tip_direction=direction,
        row_offsets=base_offsets,
    )


def make_stack_scene_spec(
    *,
    instance_seed: int,
    target_status: str,
    correct_option_letter: str,
) -> StackSceneSpec:
    """Build six candidate stacks with exactly one candidate matching the target status."""

    target = str(target_status)
    if target not in {STATUS_STABLE, STATUS_TIPPING}:
        raise ValueError(f"unsupported target status: {target}")
    letter = str(correct_option_letter).strip().upper()
    if letter not in OPTION_LETTERS:
        raise ValueError(f"unsupported correct option letter: {letter}")

    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.scene")
    distractor_status = STATUS_TIPPING if target == STATUS_STABLE else STATUS_STABLE
    palettes = list(_BRICK_PALETTES)
    rng.shuffle(palettes)
    candidates: list[StackCandidateSpec] = []
    for index, option_letter in enumerate(OPTION_LETTERS):
        status = target if str(option_letter) == letter else distractor_status
        forced_direction = None
        if status == STATUS_TIPPING:
            forced_direction = str(rng.choice(TIP_DIRECTIONS))
        profile = profile_for_status(
            rng,
            status=status,
            forced_tip_direction=forced_direction,
        )
        fill, outline = palettes[index]
        candidates.append(
            StackCandidateSpec(
                label=str(option_letter),
                status=str(profile.status),
                tip_direction=profile.tip_direction,
                row_offsets=tuple(float(value) for value in profile.row_offsets),
                brick_fill_rgb=tuple(int(value) for value in fill),
                brick_outline_rgb=tuple(int(value) for value in outline),
            )
        )
    return StackSceneSpec(
        target_status=target,
        correct_option_letter=letter,
        candidates=tuple(candidates),
    )


__all__ = ["make_stack_scene_spec", "profile_for_status"]
