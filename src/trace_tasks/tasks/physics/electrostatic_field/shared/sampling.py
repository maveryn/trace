"""Sampling helpers for electrostatic field-map scenes."""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import uniform_probability_map
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant
from trace_tasks.tasks.physics.shared.style import SUPPORTED_PHYSICS_COLOR_NAMES
from trace_tasks.tasks.physics.shared.support_sampling import resolve_integer_support

from .formulas import requested_field_direction
from .state import (
    CandidatePoint,
    Charge,
    DIRECTION_MODE_NEGATIVE_FORCE,
    DIRECTION_MODE_POSITIVE_FORCE,
    DIRECTION_VECTORS,
    DirectionAxes,
    DirectionScenario,
    OPTION_LETTERS,
    POINT_LETTERS,
    POTENTIAL_ANSWER_SUPPORT,
    POTENTIAL_CHARGE_COORDS,
    POTENTIAL_CONTRIBUTION_SUPPORT,
    POTENTIAL_DISTANCE_UNITS,
    PotentialAxes,
    PotentialCharge,
    PotentialScenario,
    SCENE_MODE_DIRECTION,
    SCENE_MODE_POTENTIAL,
    SCENE_MODE_ZERO_FIELD,
    SCENE_STYLE_VARIANTS,
    SUPPORTED_DIRECTIONS,
    SUPPORTED_DIRECTION_MODES,
    OptionLetterAxes,
    SceneAxes,
    SceneSpec,
    ZeroFieldScenario,
)


def _uniform_string_probability_map(values: Sequence[str], *, selected: str | None = None) -> Dict[str, float]:
    """Return a deterministic uniform probability map over a finite string support."""

    support = tuple(str(value) for value in values)
    if not support:
        return {}
    if selected is not None:
        return {str(selected): 1.0}
    probability = 1.0 / float(len(support))
    return {str(value): float(probability) for value in support}


def potential_answer_support(params: Mapping[str, Any], *, defaults: Mapping[str, Any]) -> Tuple[int, ...]:
    """Return configured signed electric-potential answer support."""

    return resolve_integer_support(
        params,
        gen_defaults=defaults,
        key="potential_answer_support",
        fallback=POTENTIAL_ANSWER_SUPPORT,
    )


def potential_contribution_support(params: Mapping[str, Any], *, defaults: Mapping[str, Any]) -> Tuple[int, ...]:
    """Return configured per-charge contribution support for potential scenes."""

    return resolve_integer_support(
        params,
        gen_defaults=defaults,
        key="potential_contribution_support",
        fallback=POTENTIAL_CONTRIBUTION_SUPPORT,
    )


def resolve_scene_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> SceneAxes:
    """Resolve scene style and accent color without task or public branch routing."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.scene_axes")
    scene_variant, scene_probs = resolve_variant(
        rng,
        params=params,
        gen_defaults=defaults,
        supported_variants=SCENE_STYLE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )
    scene_variant = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        selected_variant=str(scene_variant),
        variant_probabilities=scene_probs,
        supported_variants=SCENE_STYLE_VARIANTS,
        balance_flag_key="balanced_scene_variant_sampling",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        sampling_namespace=f"{namespace}.scene_variant",
    )
    accent_name, accent_probs = resolve_variant(
        rng,
        params=params,
        gen_defaults=defaults,
        supported_variants=SUPPORTED_PHYSICS_COLOR_NAMES,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
    )
    accent_name = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        selected_variant=str(accent_name),
        variant_probabilities=accent_probs,
        supported_variants=SUPPORTED_PHYSICS_COLOR_NAMES,
        balance_flag_key="balanced_accent_color_name_sampling",
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        sampling_namespace=f"{namespace}.accent_color_name",
    )
    return SceneAxes(
        scene_variant=str(scene_variant),
        accent_color_name=str(accent_name),
        scene_variant_probabilities={str(k): float(v) for k, v in sorted(scene_probs.items())},
        accent_color_name_probabilities={str(k): float(v) for k, v in sorted(accent_probs.items())},
    )


def resolve_direction_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> DirectionAxes:
    """Resolve all field/force direction axes before symbolic scene construction."""

    raw_mode = params.get("direction_mode")
    if raw_mode is None:
        raise ValueError("direction_mode must be supplied by the selected field-direction query_id")
    mode = str(raw_mode)
    if mode not in SUPPORTED_DIRECTION_MODES:
        raise ValueError(f"unsupported electrostatic direction_mode: {mode}; supported: {SUPPORTED_DIRECTION_MODES}")
    mode_probs = _uniform_string_probability_map(SUPPORTED_DIRECTION_MODES, selected=mode)
    direction, direction_probs = resolve_variant(
        spawn_rng(int(instance_seed), f"{namespace}.target_direction"),
        params=params,
        gen_defaults=defaults,
        supported_variants=SUPPORTED_DIRECTIONS,
        explicit_key="target_direction",
        weights_key="target_direction_weights",
    )
    direction = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        selected_variant=str(direction),
        variant_probabilities=direction_probs,
        supported_variants=SUPPORTED_DIRECTIONS,
        balance_flag_key="balanced_target_direction_sampling",
        explicit_key="target_direction",
        weights_key="target_direction_weights",
        sampling_namespace=f"{namespace}.target_direction",
    )
    letter_axes = resolve_option_letter_axes(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=f"{namespace}.direction_option",
        option_letters=OPTION_LETTERS,
        weights_key="direction_option_letter_weights",
        balance_key="balanced_direction_option_letter_sampling",
    )
    return DirectionAxes(
        direction_mode=str(mode),
        target_direction=str(direction),
        correct_option_letter=str(letter_axes.correct_option_letter),
        direction_mode_probabilities={str(k): float(v) for k, v in sorted(mode_probs.items())},
        target_direction_probabilities={str(k): float(v) for k, v in sorted(direction_probs.items())},
        correct_option_letter_probabilities=dict(letter_axes.correct_option_letter_probabilities),
    )


def resolve_option_letter_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
    option_letters: Sequence[str],
    weights_key: str,
    balance_key: str,
) -> OptionLetterAxes:
    """Resolve a correct option letter for a visible-option task."""

    supported = tuple(str(value) for value in option_letters)
    option_params = dict(params)
    if option_params.get("correct_option_letter") is None and option_params.get("target_answer") is not None:
        option_params["correct_option_letter"] = str(option_params["target_answer"]).strip().upper()
    selected, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), f"{namespace}.correct_option_letter"),
        params=option_params,
        gen_defaults=defaults,
        supported_variants=supported,
        explicit_key="correct_option_letter",
        weights_key=str(weights_key),
    )
    balanced_enabled = bool(
        option_params.get(
            str(balance_key),
            group_default(defaults, str(balance_key), True),
        )
    )
    has_override = any(option_params.get(str(key)) is not None for key in ("correct_option_letter", str(weights_key)))
    if bool(balanced_enabled) and not bool(has_override):
        selected = apply_balanced_variant_sampling(
            instance_seed=int(instance_seed),
            params=option_params,
            gen_defaults=defaults,
            selected_variant=str(selected),
            variant_probabilities=probabilities,
            supported_variants=supported,
            balance_flag_key=str(balance_key),
            explicit_key="correct_option_letter",
            weights_key=str(weights_key),
            sampling_namespace=f"{namespace}.correct_option_letter",
        )
    return OptionLetterAxes(
        correct_option_letter=str(selected),
        correct_option_letter_probabilities={str(k): float(v) for k, v in sorted(probabilities.items())},
    )


def feasible_potential_scenarios(params: Mapping[str, Any], *, defaults: Mapping[str, Any]) -> Tuple[PotentialScenario, ...]:
    """Enumerate exact integer potential scenarios from contribution supports."""

    answer_support = set(int(value) for value in potential_answer_support(params, defaults=defaults))
    contribution_support = tuple(int(value) for value in potential_contribution_support(params, defaults=defaults) if int(value) != 0)
    explicit_answer = params.get("target_answer")
    explicit_contributions = params.get("potential_contributions")
    scenarios: List[PotentialScenario] = []

    if explicit_contributions is not None:
        if not isinstance(explicit_contributions, Sequence) or isinstance(explicit_contributions, (str, bytes)):
            raise ValueError(f"potential_contributions must be a sequence of {len(POTENTIAL_DISTANCE_UNITS)} integers")
        contribution_rows = [tuple(int(value) for value in explicit_contributions)]
    else:
        contribution_rows = product(contribution_support, repeat=len(POTENTIAL_DISTANCE_UNITS))

    for contributions in contribution_rows:
        if len(tuple(contributions)) != len(POTENTIAL_DISTANCE_UNITS):
            raise ValueError(f"potential_contributions must contain exactly {len(POTENTIAL_DISTANCE_UNITS)} integers")
        contribution_values = tuple(int(value) for value in contributions)
        answer = int(sum(contribution_values))
        if explicit_answer is not None and int(answer) != int(explicit_answer):
            continue
        if int(answer) not in answer_support:
            continue
        charges = []
        for index, (contribution, distance, coord) in enumerate(
            zip(contribution_values, POTENTIAL_DISTANCE_UNITS, POTENTIAL_CHARGE_COORDS),
            start=1,
        ):
            charges.append(
                PotentialCharge(
                    charge_id=f"charge_{index}",
                    charge_value=int(contribution) * int(distance),
                    contribution=int(contribution),
                    distance_units=int(distance),
                    x=int(coord[0]),
                    y=int(coord[1]),
                )
            )
        scenarios.append(
            PotentialScenario(
                charges=tuple(charges),
                point_x=0,
                point_y=0,
                potential_value=int(answer),
            )
        )
    if not scenarios:
        raise ValueError("no feasible electrostatic-potential scenarios for configured supports")
    return tuple(scenarios)


def feasible_potential_answers(params: Mapping[str, Any], *, defaults: Mapping[str, Any]) -> Tuple[int, ...]:
    """Return configured potential answers that have at least one construction."""

    feasible = sorted({int(scenario.potential_value) for scenario in feasible_potential_scenarios(params, defaults=defaults)})
    if not feasible:
        raise ValueError("no feasible potential answers")
    return tuple(int(value) for value in feasible)


def resolve_potential_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> PotentialAxes:
    """Resolve an exact integer potential target."""

    support = feasible_potential_answers(params, defaults=defaults)
    explicit = params.get("target_answer")
    if explicit is not None:
        selected = int(explicit)
        if int(selected) not in set(support):
            raise ValueError(f"unsupported electrostatic potential target_answer: {selected}")
        probabilities = uniform_probability_map(support, selected=int(selected))
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_answer.potential")
        selected = int(rng.choice(support))
        probabilities = uniform_probability_map(support)
    return PotentialAxes(
        target_answer=int(selected),
        target_answer_probabilities={str(k): float(v) for k, v in sorted(probabilities.items())},
        answer_support=tuple(int(value) for value in support),
        contribution_support=potential_contribution_support(params, defaults=defaults),
    )


def _direction_options(rng, *, requested_direction: str, correct_option_letter: str) -> Dict[str, str]:
    """Assign every option letter to a unique compass direction."""

    remaining = [direction for direction in SUPPORTED_DIRECTIONS if str(direction) != str(requested_direction)]
    rng.shuffle(remaining)
    option_directions: Dict[str, str] = {}
    for letter in OPTION_LETTERS:
        if str(letter) == str(correct_option_letter):
            option_directions[str(letter)] = str(requested_direction)
        else:
            option_directions[str(letter)] = str(remaining.pop())
    return option_directions


def sample_direction_scenario(rng, *, axes: DirectionAxes) -> DirectionScenario:
    """Build a field-direction scene with a known exact compass direction."""

    requested_direction = str(axes.target_direction)
    direction_mode = str(axes.direction_mode)
    field_direction = requested_field_direction(
        requested_direction=requested_direction,
        direction_mode=direction_mode,
    )
    field_dx, field_dy = DIRECTION_VECTORS[str(field_direction)]
    perp_dx, perp_dy = -int(field_dy), int(field_dx)
    if int(perp_dx) == 0 and int(perp_dy) == 0:
        perp_dx, perp_dy = 0, 1
    use_negative_source = int(rng.randrange(2)) == 0
    if use_negative_source:
        main_charge = Charge(charge_id="charge_main", charge_value=-4, x=int(2 * field_dx), y=int(2 * field_dy))
    else:
        main_charge = Charge(charge_id="charge_main", charge_value=4, x=int(-2 * field_dx), y=int(-2 * field_dy))
    charges = (
        main_charge,
        Charge(charge_id="charge_cancel_a", charge_value=1, x=int(2 * perp_dx), y=int(2 * perp_dy)),
        Charge(charge_id="charge_cancel_b", charge_value=1, x=int(-2 * perp_dx), y=int(-2 * perp_dy)),
    )
    test_charge_sign = None
    if direction_mode == DIRECTION_MODE_POSITIVE_FORCE:
        test_charge_sign = "+"
    elif direction_mode == DIRECTION_MODE_NEGATIVE_FORCE:
        test_charge_sign = "-"
    return DirectionScenario(
        charges=charges,
        point_x=0,
        point_y=0,
        direction_mode=direction_mode,
        test_charge_sign=test_charge_sign,
        field_direction=str(field_direction),
        requested_direction=str(requested_direction),
        option_directions=_direction_options(
            rng,
            requested_direction=str(requested_direction),
            correct_option_letter=str(axes.correct_option_letter),
        ),
    )


def _inside_candidate_bounds(x: int, y: int) -> bool:
    """Return whether a candidate point is comfortably inside the board."""

    return -4 <= int(x) <= 4 and -4 <= int(y) <= 4


def sample_zero_field_scenario(rng, *, axes: OptionLetterAxes) -> ZeroFieldScenario:
    """Build an unequal-charge zero-field candidate scene."""

    symmetry_axis = "horizontal" if int(rng.randrange(2)) == 0 else "vertical"
    charge_sign = 1 if int(rng.randrange(2)) == 0 else -1
    small_charge_first = int(rng.randrange(2)) == 0
    if str(symmetry_axis) == "horizontal":
        axis_dx, axis_dy = 1, 0
        point_x = int((-1, 0, 1)[int(rng.randrange(3))])
        point_y = int((-2, -1, 0, 1, 2)[int(rng.randrange(5))])
    else:
        axis_dx, axis_dy = 0, 1
        point_x = int((-2, -1, 0, 1, 2)[int(rng.randrange(5))])
        point_y = int((-1, 0, 1)[int(rng.randrange(3))])

    if bool(small_charge_first):
        first_distance, second_distance = 1, 2
        first_magnitude, second_magnitude = 1, 4
    else:
        first_distance, second_distance = 2, 1
        first_magnitude, second_magnitude = 4, 1

    first_x = int(point_x - first_distance * axis_dx)
    first_y = int(point_y - first_distance * axis_dy)
    second_x = int(point_x + second_distance * axis_dx)
    second_y = int(point_y + second_distance * axis_dy)
    if str(symmetry_axis) == "horizontal":
        charges = (
            Charge(charge_id="charge_left", charge_value=int(first_magnitude * charge_sign), x=first_x, y=first_y),
            Charge(charge_id="charge_right", charge_value=int(second_magnitude * charge_sign), x=second_x, y=second_y),
        )
    else:
        charges = (
            Charge(charge_id="charge_bottom", charge_value=int(first_magnitude * charge_sign), x=first_x, y=first_y),
            Charge(charge_id="charge_top", charge_value=int(second_magnitude * charge_sign), x=second_x, y=second_y),
        )

    charge_coords = {(int(charge.x), int(charge.y)) for charge in charges}
    distractor_coords: List[Tuple[int, int]] = []

    def add_candidate(x: int, y: int) -> None:
        coord = (int(x), int(y))
        if coord == (int(point_x), int(point_y)):
            return
        if coord in charge_coords:
            return
        if coord in distractor_coords:
            return
        if not _inside_candidate_bounds(int(x), int(y)):
            return
        distractor_coords.append(coord)

    perp_dx, perp_dy = -int(axis_dy), int(axis_dx)
    for offset in (-3, -2, -1, 1, 2, 3):
        add_candidate(int(point_x + offset * axis_dx), int(point_y + offset * axis_dy))
    for offset in (-2, -1, 1, 2):
        add_candidate(int(point_x + offset * perp_dx), int(point_y + offset * perp_dy))
    for axis_offset, perp_offset in ((-1, -1), (-1, 1), (1, -1), (1, 1), (-2, 1), (2, -1)):
        add_candidate(int(point_x + axis_offset * axis_dx + perp_offset * perp_dx), int(point_y + axis_offset * axis_dy + perp_offset * perp_dy))
    if len(distractor_coords) < len(POINT_LETTERS) - 1:
        raise RuntimeError("not enough electrostatics zero-field distractor candidates")

    rng.shuffle(distractor_coords)
    candidates: List[CandidatePoint] = []
    distractor_index = 0
    for letter in POINT_LETTERS:
        if str(letter) == str(axes.correct_option_letter):
            x, y = int(point_x), int(point_y)
            is_correct = True
        else:
            x, y = distractor_coords[int(distractor_index)]
            distractor_index += 1
            is_correct = False
        candidates.append(CandidatePoint(letter=str(letter), x=int(x), y=int(y), is_correct=bool(is_correct)))
    return ZeroFieldScenario(
        charges=charges,
        candidate_points=tuple(candidates),
        correct_option_letter=str(axes.correct_option_letter),
        symmetry_axis=str(symmetry_axis),
    )


def sample_potential_scenario(rng, *, target_answer: int, params: Mapping[str, Any], defaults: Mapping[str, Any]) -> PotentialScenario:
    """Sample one exact electric-potential scenario for the target answer."""

    scenarios = [
        scenario
        for scenario in feasible_potential_scenarios(params, defaults=defaults)
        if int(scenario.potential_value) == int(target_answer)
    ]
    if not scenarios:
        raise ValueError(f"no electrostatic-potential scenario for target {target_answer}")
    return scenarios[int(rng.randrange(len(scenarios)))]


def make_direction_spec(*, scene_axes: SceneAxes, direction_axes: DirectionAxes, rng) -> SceneSpec:
    """Build the symbolic scene for the direction-choice task."""

    scenario = sample_direction_scenario(rng, axes=direction_axes)
    return SceneSpec(
        scene_mode=SCENE_MODE_DIRECTION,
        scene_variant=str(scene_axes.scene_variant),
        target_answer=str(direction_axes.correct_option_letter),
        direction_scenario=scenario,
        zero_field_scenario=None,
        potential_scenario=None,
        annotation_entity_ids=("charge_main", "charge_cancel_a", "charge_cancel_b", "query_point"),
    )


def make_zero_field_spec(*, scene_axes: SceneAxes, option_axes: OptionLetterAxes, rng) -> SceneSpec:
    """Build the symbolic scene for the zero-field option task."""

    scenario = sample_zero_field_scenario(rng, axes=option_axes)
    correct_entity = f"candidate_{str(scenario.correct_option_letter)}"
    return SceneSpec(
        scene_mode=SCENE_MODE_ZERO_FIELD,
        scene_variant=str(scene_axes.scene_variant),
        target_answer=str(option_axes.correct_option_letter),
        direction_scenario=None,
        zero_field_scenario=scenario,
        potential_scenario=None,
        annotation_entity_ids=tuple(str(charge.charge_id) for charge in scenario.charges) + (correct_entity,),
    )


def make_potential_spec(*, scene_axes: SceneAxes, potential_axes: PotentialAxes, rng, params: Mapping[str, Any], defaults: Mapping[str, Any]) -> SceneSpec:
    """Build the symbolic scene for the electric-potential task."""

    scenario = sample_potential_scenario(
        rng,
        target_answer=int(potential_axes.target_answer),
        params=params,
        defaults=defaults,
    )
    return SceneSpec(
        scene_mode=SCENE_MODE_POTENTIAL,
        scene_variant=str(scene_axes.scene_variant),
        target_answer=int(scenario.potential_value),
        direction_scenario=None,
        zero_field_scenario=None,
        potential_scenario=scenario,
        annotation_entity_ids=tuple([str(charge.charge_id) for charge in scenario.charges] + ["query_point"]),
    )
