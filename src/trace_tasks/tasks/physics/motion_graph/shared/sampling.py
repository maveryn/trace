"""Sampling primitives for physics motion-graph tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.variant_sampling import (
    apply_balanced_variant_sampling,
    resolve_variant,
)

from .formulas import average_speed, interval_displacement
from .state import (
    AverageSpeedGraphSpec,
    IntervalAxes,
    IntervalGraphSpec,
    MotionGraphRenderDefaults,
    OPTION_LETTERS,
    SUPPORTED_SCENE_STYLES,
    StateChoiceAxes,
    StateGraphSpec,
)


SPEED_MAGNITUDE_PROFILE = "speed_magnitude"
FLAT_VELOCITY_PROFILE = "flat_velocity"
RAMPED_VELOCITY_PROFILE = "ramped_velocity"


def _probability_map(values: Sequence[str]) -> Dict[str, float]:
    support = tuple(str(value) for value in values)
    if not support:
        return {}
    probability = 1.0 / float(len(support))
    return {value: float(probability) for value in support}


def _supported_state_tuple(supported_states: Sequence[str]) -> Tuple[str, ...]:
    support = tuple(str(value) for value in supported_states)
    if not support:
        raise ValueError("motion graph state support cannot be empty")
    return support


def resolve_scene_style(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one visual style for the graph panel."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.scene_style")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=defaults,
        supported_variants=SUPPORTED_SCENE_STYLES,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_SCENE_STYLES,
        balance_flag_key="balanced_scene_variant_sampling",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        sampling_namespace=f"{namespace}.scene_style",
    )
    return str(selected), {str(key): float(value) for key, value in probabilities.items()}


def resolve_motion_state(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
    supported_states: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    """Resolve one motion state within a task-owned operation."""

    states = _supported_state_tuple(supported_states)
    task_params = dict(params)
    explicit_state = task_params.get(
        "motion_state",
        task_params.get("target_state", task_params.get("target_answer")),
    )
    if explicit_state is not None:
        task_params["motion_state"] = str(explicit_state)
    selected, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), f"{namespace}.motion_state"),
        params=task_params,
        gen_defaults=defaults,
        supported_variants=states,
        explicit_key="motion_state",
        weights_key="motion_state_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=states,
        balance_flag_key="balanced_motion_state_sampling",
        explicit_key="motion_state",
        weights_key="motion_state_weights",
        sampling_namespace=f"{namespace}.motion_state",
    )
    return str(selected), {str(key): float(value) for key, value in probabilities.items()}


def resolve_correct_option_letter(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the visible option letter that carries the correct state."""

    selected, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), f"{namespace}.correct_option_letter"),
        params=params,
        gen_defaults=defaults,
        supported_variants=OPTION_LETTERS,
        explicit_key="correct_option_letter",
        weights_key="correct_option_letter_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=OPTION_LETTERS,
        balance_flag_key="balanced_correct_option_letter_sampling",
        explicit_key="correct_option_letter",
        weights_key="correct_option_letter_weights",
        sampling_namespace=f"{namespace}.correct_option_letter",
    )
    return str(selected), {str(key): float(value) for key, value in probabilities.items()}


def resolve_state_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
    supported_states: Sequence[str],
) -> StateChoiceAxes:
    """Resolve all sampling axes for one state-choice objective."""

    scene_style, scene_probs = resolve_scene_style(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    motion_state, state_probs = resolve_motion_state(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
        supported_states=supported_states,
    )
    correct_option_letter, option_probs = resolve_correct_option_letter(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    return StateChoiceAxes(
        scene_style=str(scene_style),
        motion_state=str(motion_state),
        correct_option_letter=str(correct_option_letter),
        scene_style_probabilities=dict(scene_probs),
        motion_state_probabilities=dict(state_probs),
        correct_option_letter_probabilities=dict(option_probs),
    )


def _target_segment_values(rng, *, state_profile: str, motion_state: str) -> Tuple[int, int]:
    if str(state_profile) == SPEED_MAGNITUDE_PROFILE:
        sign = int(rng.choice([-1, 1]))
        if str(motion_state) == "speeding_up":
            start_abs, end_abs = rng.choice([(1, 3), (1, 4), (2, 4)])
            return sign * int(start_abs), sign * int(end_abs)
        if str(motion_state) == "slowing_down":
            start_abs, end_abs = rng.choice([(4, 2), (4, 1), (3, 1)])
            return sign * int(start_abs), sign * int(end_abs)
        if str(motion_state) == "constant_speed":
            value = sign * int(rng.choice([2, 3, 4]))
            return value, value
    raise ValueError(f"unsupported motion state {motion_state} for {state_profile}")


def _extend_values(
    rng,
    *,
    values: List[int],
    fixed_index: int,
    y_min: int,
    y_max: int,
) -> Tuple[int, ...]:
    deltas = [-2, -1, 0, 1, 2]
    for index in range(int(fixed_index) - 1, -1, -1):
        current = int(values[index + 1])
        feasible = [
            delta
            for delta in deltas
            if int(y_min) <= current - int(delta) <= int(y_max)
        ]
        values[index] = int(current - int(rng.choice(feasible)))
    for index in range(int(fixed_index) + 2, len(values)):
        current = int(values[index - 1])
        feasible = [
            delta
            for delta in deltas
            if int(y_min) <= current + int(delta) <= int(y_max)
        ]
        values[index] = int(current + int(rng.choice(feasible)))
    return tuple(int(value) for value in values)


def _build_option_map(
    *,
    instance_seed: int,
    supported_states: Sequence[str],
    motion_state: str,
    correct_option_letter: str,
    namespace: str,
) -> Dict[str, str]:
    states = list(_supported_state_tuple(supported_states)) + ["changing_direction"]
    remaining = [state for state in states if str(state) != str(motion_state)]
    rng = spawn_rng(int(instance_seed), f"{namespace}.option_map")
    rng.shuffle(remaining)
    option_map: Dict[str, str] = {}
    for letter in OPTION_LETTERS:
        if str(letter) == str(correct_option_letter):
            option_map[str(letter)] = str(motion_state)
    remaining_letters = [
        letter
        for letter in OPTION_LETTERS
        if str(letter) != str(correct_option_letter)
    ]
    for letter, state in zip(remaining_letters, remaining):
        option_map[str(letter)] = str(state)
    return {str(letter): str(option_map[str(letter)]) for letter in OPTION_LETTERS}


def make_state_graph_spec(
    instance_seed: int,
    *,
    axes: StateChoiceAxes,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    operation_label: str,
    state_profile: str,
    supported_states: Sequence[str],
    graph_kind: str,
    y_axis_label: str,
    title: str,
    namespace: str,
) -> StateGraphSpec:
    """Build one symbolic graph for a state-choice objective."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.graph_spec")
    y_min = int(params.get("y_min", group_default(render_defaults, "y_min", -5)))
    y_max = int(params.get("y_max", group_default(render_defaults, "y_max", 5)))
    t_min = int(params.get("t_min", group_default(render_defaults, "t_min", 0)))
    t_max = int(params.get("t_max", group_default(render_defaults, "t_max", 6)))
    t_values = tuple(range(int(t_min), int(t_max) + 1))
    if len(t_values) < 5:
        raise ValueError("motion graph needs at least five time samples")
    segment_index = int(params.get("query_segment_index", rng.choice([1, 2, 3, 4])))
    segment_index = max(1, min(len(t_values) - 2, int(segment_index)))
    y0, y1 = _target_segment_values(
        rng,
        state_profile=str(state_profile),
        motion_state=str(axes.motion_state),
    )
    if not (int(y_min) <= int(y0) <= int(y_max) and int(y_min) <= int(y1) <= int(y_max)):
        raise ValueError("target motion segment values out of bounds")
    y_values: List[int] = [0 for _ in t_values]
    y_values[segment_index] = int(y0)
    y_values[segment_index + 1] = int(y1)
    y_values = list(
        _extend_values(
            rng,
            values=y_values,
            fixed_index=int(segment_index),
            y_min=int(y_min),
            y_max=int(y_max),
        )
    )
    return StateGraphSpec(
        scene_style=str(axes.scene_style),
        operation=str(operation_label),
        graph_kind=str(graph_kind),
        motion_state=str(axes.motion_state),
        correct_option_letter=str(axes.correct_option_letter),
        option_map=_build_option_map(
            instance_seed=int(instance_seed),
            supported_states=supported_states,
            motion_state=str(axes.motion_state),
            correct_option_letter=str(axes.correct_option_letter),
            namespace=str(namespace),
        ),
        t_values=tuple(int(value) for value in t_values),
        y_values=tuple(int(value) for value in y_values),
        target_segment_index=int(segment_index),
        y_axis_label=str(y_axis_label),
        title=str(title),
    )


def resolve_interval_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> IntervalAxes:
    """Resolve scene axes for an interval-displacement objective."""

    scene_style, scene_probs = resolve_scene_style(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    return IntervalAxes(
        scene_style=str(scene_style),
        scene_style_probabilities=dict(scene_probs),
    )


def _resolve_interval_width(
    *,
    rng,
    mode: str,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    t_min: int,
    t_max: int,
) -> int:
    support_key = (
        "constant_velocity_interval_width_support"
        if str(mode) == FLAT_VELOCITY_PROFILE
        else "constant_acceleration_interval_width_support"
    )
    widths = [int(value) for value in group_default(defaults, support_key, (2, 3, 4))]
    widths = [value for value in widths if 1 <= value <= int(t_max) - int(t_min) - 1]
    if not widths:
        raise ValueError("no feasible interval widths for motion graph")
    return int(rng.choice(widths))


def make_interval_graph_spec(
    instance_seed: int,
    *,
    axes: IntervalAxes,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    segment_profile: str,
    namespace: str,
) -> IntervalGraphSpec:
    """Build one symbolic velocity-time graph for interval displacement."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.graph_spec")
    t_min = int(params.get("t_min", render_defaults["t_min"]))
    t_max = int(params.get("t_max", render_defaults["t_max"]))
    y_min = int(params.get("y_min", render_defaults["y_min"]))
    y_max = int(params.get("y_max", render_defaults["y_max"]))
    if int(y_min) > 0:
        raise ValueError("interval displacement graph y_min must include zero velocity")
    if int(t_max) - int(t_min) < 5:
        raise ValueError("interval displacement graph needs at least six time ticks")

    explicit_t_start = params.get("t_start")
    explicit_t_end = params.get("t_end")
    if explicit_t_start is not None or explicit_t_end is not None:
        if explicit_t_start is None or explicit_t_end is None:
            raise ValueError("t_start and t_end must be provided together")
        t_start = int(explicit_t_start)
        t_end = int(explicit_t_end)
    else:
        dt = _resolve_interval_width(
            rng=rng,
            mode=str(segment_profile),
            params=params,
            defaults=defaults,
            t_min=int(t_min),
            t_max=int(t_max),
        )
        min_start = int(t_min) + 1
        max_start = int(t_max) - int(dt) - 1
        if min_start > max_start:
            min_start = int(t_min)
            max_start = int(t_max) - int(dt)
        t_start = int(rng.randint(min_start, max_start))
        t_end = int(t_start + dt)
    if int(t_start) < int(t_min) or int(t_end) > int(t_max) or int(t_start) >= int(t_end):
        raise ValueError("invalid marked interval bounds")
    dt = int(t_end) - int(t_start)

    explicit_v_start = params.get("v_start", params.get("velocity_value"))
    explicit_v_end = params.get(
        "v_end",
        explicit_v_start if str(segment_profile) == FLAT_VELOCITY_PROFILE else None,
    )
    if explicit_v_start is not None or explicit_v_end is not None:
        if explicit_v_start is None or explicit_v_end is None:
            raise ValueError("v_start and v_end must be provided together")
        v_start = int(explicit_v_start)
        v_end = int(explicit_v_end)
    elif str(segment_profile) == FLAT_VELOCITY_PROFILE:
        velocity_support = [
            int(value)
            for value in group_default(defaults, "constant_velocity_support", (1, 2, 3, 4, 5, 6, 7))
        ]
        feasible = [value for value in velocity_support if int(y_min) <= int(value) <= int(y_max)]
        if not feasible:
            raise ValueError("no feasible constant velocities")
        v_start = v_end = int(rng.choice(feasible))
    else:
        velocity_support = [
            int(value)
            for value in group_default(
                defaults,
                "acceleration_endpoint_velocity_support",
                (1, 2, 3, 4, 5, 6, 7, 8),
            )
        ]
        slope_support = [
            int(value)
            for value in group_default(defaults, "constant_acceleration_slope_support", (-2, -1, 1, 2))
        ]
        feasible_pairs: List[Tuple[int, int]] = []
        for start in velocity_support:
            for slope in slope_support:
                end = int(start) + int(slope) * int(dt)
                if int(y_min) <= int(end) <= int(y_max) and int(end) != int(start):
                    if ((int(start) + int(end)) * int(dt)) % 2 == 0:
                        feasible_pairs.append((int(start), int(end)))
        if not feasible_pairs:
            raise ValueError("no feasible acceleration endpoint velocities")
        v_start, v_end = rng.choice(feasible_pairs)

    if not (int(y_min) <= int(v_start) <= int(y_max) and int(y_min) <= int(v_end) <= int(y_max)):
        raise ValueError("velocity endpoints out of graph bounds")
    if str(segment_profile) == FLAT_VELOCITY_PROFILE and int(v_start) != int(v_end):
        raise ValueError("constant-velocity displacement requires equal endpoint velocities")
    if str(segment_profile) == RAMPED_VELOCITY_PROFILE and int(v_start) == int(v_end):
        raise ValueError("constant-acceleration displacement requires a sloped velocity segment")
    if (int(v_end) - int(v_start)) % int(dt) != 0:
        raise ValueError("interval endpoint velocities must give integer tick values")
    displacement = interval_displacement(int(v_start), int(v_end), int(dt))

    slope = int((int(v_end) - int(v_start)) // int(dt))
    velocity_by_t: Dict[int, int] = {
        int(t): int(v_start) + int(slope) * (int(t) - int(t_start))
        for t in range(int(t_start), int(t_end) + 1)
    }
    for t_value in range(int(t_start) - 1, int(t_min) - 1, -1):
        next_value = int(velocity_by_t[int(t_value) + 1])
        feasible_values = [
            value
            for value in range(
                max(int(y_min), next_value - 2),
                min(int(y_max), next_value + 2) + 1,
            )
        ]
        velocity_by_t[int(t_value)] = int(rng.choice(feasible_values))
    for t_value in range(int(t_end) + 1, int(t_max) + 1):
        prev_value = int(velocity_by_t[int(t_value) - 1])
        feasible_values = [
            value
            for value in range(
                max(int(y_min), prev_value - 2),
                min(int(y_max), prev_value + 2) + 1,
            )
        ]
        velocity_by_t[int(t_value)] = int(rng.choice(feasible_values))

    t_values = tuple(range(int(t_min), int(t_max) + 1))
    velocity_values = tuple(int(velocity_by_t[int(t)]) for t in t_values)
    return IntervalGraphSpec(
        scene_style=str(axes.scene_style),
        segment_mode=str(segment_profile),
        graph_kind="velocity_time",
        t_values=tuple(int(value) for value in t_values),
        velocity_values=tuple(int(value) for value in velocity_values),
        t_start=int(t_start),
        t_end=int(t_end),
        v_start=int(v_start),
        v_end=int(v_end),
        displacement_m=int(displacement),
        y_axis_label="v (m/s)",
        title="velocity-time graph",
    )


def _resolve_average_speed_width(
    *,
    rng,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    t_min: int,
    t_max: int,
) -> int:
    widths = [
        int(value)
        for value in group_default(defaults, "average_speed_interval_width_support", (1, 2, 3))
    ]
    widths = [value for value in widths if 1 <= value <= int(t_max) - int(t_min) - 1]
    if not widths:
        raise ValueError("no feasible average-speed interval widths")
    return int(rng.choice(widths))


def make_average_speed_graph_spec(
    instance_seed: int,
    *,
    axes: IntervalAxes,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    namespace: str,
) -> AverageSpeedGraphSpec:
    """Build one symbolic distance-time graph for average speed."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.graph_spec")
    t_min = int(params.get("t_min", render_defaults["t_min"]))
    t_max = int(params.get("t_max", render_defaults["t_max"]))
    y_min = int(params.get("y_min", render_defaults["y_min"]))
    y_max = int(params.get("y_max", render_defaults["y_max"]))
    if int(y_min) > 0:
        raise ValueError("distance-time graph y_min must include zero distance")
    if int(t_max) - int(t_min) < 4:
        raise ValueError("average-speed graph needs at least five time ticks")

    explicit_t_start = params.get("t_start")
    explicit_t_end = params.get("t_end")
    if explicit_t_start is not None or explicit_t_end is not None:
        if explicit_t_start is None or explicit_t_end is None:
            raise ValueError("t_start and t_end must be provided together")
        t_start = int(explicit_t_start)
        t_end = int(explicit_t_end)
    else:
        dt = _resolve_average_speed_width(
            rng=rng,
            params=params,
            defaults=defaults,
            t_min=int(t_min),
            t_max=int(t_max),
        )
        min_start = int(t_min) + 1
        max_start = int(t_max) - int(dt) - 1
        if min_start > max_start:
            min_start = int(t_min)
            max_start = int(t_max) - int(dt)
        t_start = int(rng.randint(min_start, max_start))
        t_end = int(t_start + dt)
    if int(t_start) < int(t_min) or int(t_end) > int(t_max) or int(t_start) >= int(t_end):
        raise ValueError("invalid marked interval bounds")
    dt = int(t_end) - int(t_start)

    explicit_d_start = params.get("d_start", params.get("distance_start_m"))
    explicit_d_end = params.get("d_end", params.get("distance_end_m"))
    speed_support = [
        int(value)
        for value in group_default(defaults, "average_speed_support", (1, 2, 3, 4, 5, 6))
    ]
    explicit_speed = params.get("average_speed_m_s", params.get("target_answer"))
    if explicit_d_start is not None or explicit_d_end is not None:
        if explicit_d_start is None or explicit_d_end is None:
            raise ValueError("d_start and d_end must be provided together")
        d_start = int(explicit_d_start)
        d_end = int(explicit_d_end)
        inferred_speed = average_speed(d_start, d_end, dt)
        if explicit_speed is not None and int(explicit_speed) != int(inferred_speed):
            raise ValueError("explicit distance endpoints do not match requested average speed")
        speed = int(inferred_speed)
        if speed not in set(speed_support):
            raise ValueError(f"unsupported average speed: {speed}")
    else:
        if explicit_speed is not None:
            speed = int(explicit_speed)
            if speed not in set(speed_support):
                raise ValueError(f"unsupported average speed: {speed}")
        else:
            feasible_speeds = [
                value
                for value in speed_support
                if int(y_min) <= int(value) * int(dt) <= int(y_max) - int(y_min)
            ]
            if not feasible_speeds:
                raise ValueError("no feasible average speeds")
            speed = int(rng.choice(feasible_speeds))
        distance_delta = int(speed) * int(dt)
        min_distance_start = int(y_min)
        max_distance_start = int(y_max) - int(distance_delta)
        if int(max_distance_start) < int(min_distance_start):
            raise ValueError("average-speed distance delta exceeds graph bounds")
        d_start = int(rng.randint(min_distance_start, max_distance_start))
        d_end = int(d_start + distance_delta)

    if not (int(y_min) <= int(d_start) <= int(y_max) and int(y_min) <= int(d_end) <= int(y_max)):
        raise ValueError("distance endpoints out of graph bounds")
    if int(d_end) <= int(d_start):
        raise ValueError("average-speed distance interval must increase")
    speed = average_speed(int(d_start), int(d_end), int(dt))

    distance_by_t: Dict[int, int] = {
        int(t): int(d_start) + int(speed) * (int(t) - int(t_start))
        for t in range(int(t_start), int(t_end) + 1)
    }
    for t_value in range(int(t_start) - 1, int(t_min) - 1, -1):
        next_value = int(distance_by_t[int(t_value) + 1])
        feasible_values = list(range(max(int(y_min), next_value - 3), next_value + 1))
        distance_by_t[int(t_value)] = int(rng.choice(feasible_values))
    for t_value in range(int(t_end) + 1, int(t_max) + 1):
        prev_value = int(distance_by_t[int(t_value) - 1])
        feasible_values = list(range(prev_value, min(int(y_max), prev_value + 3) + 1))
        distance_by_t[int(t_value)] = int(rng.choice(feasible_values))

    t_values = tuple(range(int(t_min), int(t_max) + 1))
    distance_values = tuple(int(distance_by_t[int(t)]) for t in t_values)
    return AverageSpeedGraphSpec(
        scene_style=str(axes.scene_style),
        graph_kind="distance_time",
        t_values=tuple(int(value) for value in t_values),
        distance_values=tuple(int(value) for value in distance_values),
        t_start=int(t_start),
        t_end=int(t_end),
        d_start=int(d_start),
        d_end=int(d_end),
        average_speed_m_s=int(speed),
        y_axis_label="d (m)",
        title="distance-time graph",
    )


__all__ = [
    "FLAT_VELOCITY_PROFILE",
    "RAMPED_VELOCITY_PROFILE",
    "SPEED_MAGNITUDE_PROFILE",
    "make_average_speed_graph_spec",
    "make_interval_graph_spec",
    "make_state_graph_spec",
    "resolve_correct_option_letter",
    "resolve_interval_axes",
    "resolve_motion_state",
    "resolve_scene_style",
    "resolve_state_axes",
]
