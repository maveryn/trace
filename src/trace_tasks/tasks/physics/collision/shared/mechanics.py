"""Mechanics and vector primitives for physics collision tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Tuple

from trace_tasks.core.seed import spawn_rng

from .state import (
    ANNOTATION_ENTITY_IDS,
    DIRECTION_ANGLE_DEGREES,
    DIRECTION_NAMES,
    OPTION_LETTERS,
    StickySceneSpec,
    StickyScenario,
)


def direction_unit(direction: str) -> Tuple[float, float]:
    """Return the screen-space unit vector for one named compass direction."""

    angle = math.radians(float(DIRECTION_ANGLE_DEGREES[str(direction)]))
    return (float(math.cos(angle)), float(-math.sin(angle)))


def direction_sign(value: Any, *, axis: str) -> int:
    """Resolve one named cardinal direction into a signed component."""

    text = str(value).strip().lower()
    if str(axis) == "horizontal":
        mapping = {"right": 1, "+x": 1, "east": 1, "left": -1, "-x": -1, "west": -1}
    else:
        mapping = {"up": 1, "+y": 1, "north": 1, "down": -1, "-y": -1, "south": -1}
    if text not in mapping:
        raise ValueError(f"unsupported {axis} direction: {value}")
    return int(mapping[text])


def angle_degrees(vx: float, vy: float) -> float:
    """Return physics-plane angle in degrees in [0, 360)."""

    return float((math.degrees(math.atan2(float(vy), float(vx))) + 360.0) % 360.0)


def rounded_speed_value(vx: float, vy: float) -> float:
    """Return speed magnitude rounded to one decimal place."""

    return float(round(math.hypot(float(vx), float(vy)) + 1e-9, 1))


def speed_answer_tenths(vx: float, vy: float) -> int:
    """Return one-decimal speed as an integer number of tenths."""

    return int(round(10.0 * rounded_speed_value(float(vx), float(vy))))


def angular_distance(a: float, b: float) -> float:
    """Return the smaller absolute angle difference in degrees."""

    return float(abs((float(a) - float(b) + 180.0) % 360.0 - 180.0))


def direction_label(angle_degrees_value: float) -> str:
    """Return the nearest eight-way direction label for one vector angle."""

    labels = (
        ("east", 0.0),
        ("northeast", 45.0),
        ("north", 90.0),
        ("northwest", 135.0),
        ("west", 180.0),
        ("southwest", 225.0),
        ("south", 270.0),
        ("southeast", 315.0),
    )
    return min(
        labels,
        key=lambda item: angular_distance(float(angle_degrees_value), float(item[1])),
    )[0]


def aftermath_option_directions(
    *,
    instance_seed: int,
    final_motion_direction: str,
    correct_option_letter: str,
) -> Dict[str, str]:
    """Place the correct incoming direction among separated distractors."""

    correct_index = DIRECTION_NAMES.index(str(final_motion_direction))
    distractors = [
        direction
        for index, direction in enumerate(DIRECTION_NAMES)
        if index != correct_index
        and min(abs(index - correct_index), len(DIRECTION_NAMES) - abs(index - correct_index)) != 1
    ]
    if len(distractors) != len(OPTION_LETTERS) - 1:
        raise ValueError("collision aftermath distractor construction requires five distractors")
    rng = spawn_rng(int(instance_seed), "physics_collision.aftermath.option_directions")
    shuffled = list(distractors)
    rng.shuffle(shuffled)
    out: Dict[str, str] = {str(correct_option_letter): str(final_motion_direction)}
    for letter, direction in zip(
        [value for value in OPTION_LETTERS if value != str(correct_option_letter)],
        shuffled,
    ):
        out[str(letter)] = str(direction)
    return {letter: out[letter] for letter in OPTION_LETTERS}


def feasible_sticky_scenarios(
    *,
    params: Mapping[str, Any],
    masses: Tuple[int, ...],
    speeds: Tuple[int, ...],
    component_values: Tuple[int, ...],
) -> Tuple[StickyScenario, ...]:
    """Enumerate constructively feasible integer-component sticky collisions."""

    component_value_set = set(int(value) for value in component_values)
    component_magnitudes = {abs(int(value)) for value in component_value_set}
    explicit_horizontal_mass = params.get("horizontal_mass")
    explicit_vertical_mass = params.get("vertical_mass")
    explicit_horizontal_speed = params.get("horizontal_speed")
    explicit_vertical_speed = params.get("vertical_speed")
    explicit_horizontal_direction = params.get("horizontal_direction")
    explicit_vertical_direction = params.get("vertical_direction")
    horizontal_signs = (
        (direction_sign(explicit_horizontal_direction, axis="horizontal"),)
        if explicit_horizontal_direction is not None
        else (-1, 1)
    )
    vertical_signs = (
        (direction_sign(explicit_vertical_direction, axis="vertical"),)
        if explicit_vertical_direction is not None
        else (-1, 1)
    )

    scenarios: List[StickyScenario] = []
    for horizontal_mass in masses:
        if explicit_horizontal_mass is not None and int(horizontal_mass) != int(explicit_horizontal_mass):
            continue
        for vertical_mass in masses:
            if explicit_vertical_mass is not None and int(vertical_mass) != int(explicit_vertical_mass):
                continue
            total_mass = int(horizontal_mass) + int(vertical_mass)
            if total_mass <= 0:
                continue
            for horizontal_speed in speeds:
                if explicit_horizontal_speed is not None and int(horizontal_speed) != int(explicit_horizontal_speed):
                    continue
                horizontal_momentum = int(horizontal_mass) * int(horizontal_speed)
                if horizontal_momentum % total_mass != 0:
                    continue
                final_vx_magnitude = horizontal_momentum // total_mass
                if final_vx_magnitude not in component_magnitudes:
                    continue
                for vertical_speed in speeds:
                    if explicit_vertical_speed is not None and int(vertical_speed) != int(explicit_vertical_speed):
                        continue
                    vertical_momentum = int(vertical_mass) * int(vertical_speed)
                    if vertical_momentum % total_mass != 0:
                        continue
                    final_vy_magnitude = vertical_momentum // total_mass
                    if final_vy_magnitude not in component_magnitudes:
                        continue
                    for horizontal_sign in horizontal_signs:
                        final_vx = int(horizontal_sign) * int(final_vx_magnitude)
                        if final_vx not in component_value_set:
                            continue
                        for vertical_sign in vertical_signs:
                            final_vy = int(vertical_sign) * int(final_vy_magnitude)
                            if final_vy not in component_value_set:
                                continue
                            scenarios.append(
                                StickyScenario(
                                    horizontal_mass=int(horizontal_mass),
                                    vertical_mass=int(vertical_mass),
                                    horizontal_speed=int(horizontal_speed),
                                    vertical_speed=int(vertical_speed),
                                    horizontal_sign=int(horizontal_sign),
                                    vertical_sign=int(vertical_sign),
                                    final_vx=int(final_vx),
                                    final_vy=int(final_vy),
                                    total_mass=int(total_mass),
                                )
                            )
    if not scenarios:
        raise ValueError("no feasible sticky-collision scenarios for configured supports")
    return tuple(scenarios)


def candidate_distractor_angles(correct_angle_degrees: float) -> Tuple[float, ...]:
    """Return a stable set of separated distractor angles around the correct arrow."""

    return candidate_distractor_angles_for_option_count(
        correct_angle_degrees=correct_angle_degrees,
        option_count=len(OPTION_LETTERS),
    )


def candidate_distractor_angles_for_option_count(
    *,
    correct_angle_degrees: float,
    option_count: int,
) -> Tuple[float, ...]:
    """Return separated distractor angles for a visible option set."""

    correct_angle = float(correct_angle_degrees) % 360.0
    required_count = max(0, int(option_count) - 1)
    candidates = [
        0.0,
        45.0,
        90.0,
        135.0,
        180.0,
        225.0,
        270.0,
        315.0,
        (correct_angle + 90.0) % 360.0,
        (correct_angle + 180.0) % 360.0,
        (correct_angle + 270.0) % 360.0,
        (360.0 - correct_angle) % 360.0,
        (180.0 - correct_angle) % 360.0,
    ]
    selected: List[float] = []
    for candidate in candidates:
        angle = round(float(candidate) % 360.0, 3)
        if angular_distance(angle, correct_angle) < 22.0:
            continue
        if any(angular_distance(angle, existing) < 18.0 for existing in selected):
            continue
        selected.append(float(angle))
        if len(selected) >= required_count:
            return tuple(selected)
    offset = 36.0
    while len(selected) < required_count:
        angle = round(float(correct_angle + offset) % 360.0, 3)
        if angular_distance(angle, correct_angle) >= 22.0 and all(
            angular_distance(angle, existing) >= 18.0 for existing in selected
        ):
            selected.append(float(angle))
        offset += 37.0
    return tuple(selected[:required_count])


def sample_sticky_scene_spec(
    rng,
    *,
    scene_variant: str,
    component_axis: str | None,
    target_answer: int | float | str,
    correct_option_letter: str,
    params: Mapping[str, Any],
    masses: Tuple[int, ...],
    speeds: Tuple[int, ...],
    component_values: Tuple[int, ...],
    option_letters: Tuple[str, ...] = OPTION_LETTERS,
    target_speed_tenths: int | None = None,
) -> StickySceneSpec:
    """Sample one symbolic sticky-collision scene realizing the requested target."""

    option_letters = tuple(str(letter) for letter in option_letters)
    if str(correct_option_letter) not in set(option_letters):
        raise ValueError(f"correct option {correct_option_letter} is outside visible options {option_letters}")
    scenarios = list(
        feasible_sticky_scenarios(
            params=params,
            masses=masses,
            speeds=speeds,
            component_values=component_values,
        )
    )
    if str(component_axis) == "x":
        scenarios = [scenario for scenario in scenarios if int(scenario.final_vx) == int(target_answer)]
    elif str(component_axis) == "y":
        scenarios = [scenario for scenario in scenarios if int(scenario.final_vy) == int(target_answer)]
    elif target_speed_tenths is not None:
        scenarios = [
            scenario
            for scenario in scenarios
            if speed_answer_tenths(scenario.final_vx, scenario.final_vy)
            == int(target_speed_tenths)
        ]
    if not scenarios:
        raise ValueError(
            f"no feasible sticky-collision scenario for axis {component_axis} target {target_answer}"
        )

    scenario = scenarios[int(rng.randrange(len(scenarios)))]
    correct_angle = angle_degrees(float(scenario.final_vx), float(scenario.final_vy))
    distractors = list(
        candidate_distractor_angles_for_option_count(
            correct_angle_degrees=correct_angle,
            option_count=len(option_letters),
        )
    )
    rng.shuffle(distractors)
    option_angles: Dict[str, float] = {}
    for letter in option_letters:
        if str(letter) == str(correct_option_letter):
            option_angles[str(letter)] = round(float(correct_angle), 3)
        else:
            option_angles[str(letter)] = round(float(distractors.pop()), 3)

    return StickySceneSpec(
        scene_variant=str(scene_variant),
        component_axis=component_axis,
        scenario=scenario,
        correct_option_letter=str(correct_option_letter),
        option_letters=tuple(option_letters),
        option_angles_degrees=dict(option_angles),
        direction_label=str(direction_label(correct_angle)),
        target_answer=target_answer,
        annotation_entity_ids=tuple(ANNOTATION_ENTITY_IDS),
    )


def component_axis_label(component_axis: str | None) -> str:
    """Return the prompt-facing axis label for a component query."""

    return "horizontal" if str(component_axis) == "x" else "vertical"
