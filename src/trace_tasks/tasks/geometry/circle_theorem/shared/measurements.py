"""Shared measurement construction primitives for circle-theorem tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.fixed_query import geometry_probability_map as _probability_map

from .spatial_primitives import _circle_point
from .state import _fixed_point_label_map, _visible_angle, _visible_segment

Point = Tuple[float, float]

DEFAULT_RADIUS_SUPPORT: Tuple[int, ...] = tuple(range(6, 31))
DEFAULT_CENTRAL_ANGLE_SUPPORT: Tuple[int, ...] = tuple(range(40, 161, 2))
DEFAULT_INSCRIBED_ANGLE_SUPPORT: Tuple[int, ...] = tuple(range(20, 81))


@dataclass(frozen=True)
class ChordLengthDiagramSpec:
    """Task-bound numeric facts for one chord-length diagram."""

    radius_value: int
    angle_degrees: int
    central_angle_degrees: int
    answer_value: float
    uses_inscribed_angle: bool


@dataclass(frozen=True)
class ChordLengthBinding:
    """Selected chord inputs plus the final diagram spec."""

    spec: ChordLengthDiagramSpec
    radius_probabilities: Dict[str, float]
    angle_probabilities: Dict[str, float]


def round_tenth(value: float) -> float:
    """Round geometry decimal answers to the one decimal place."""

    return float(round(float(value) + 1e-9, 1))


def chord_length_from_radius_central_angle(
    *, radius_value: int | float, central_angle_degrees: int | float
) -> float:
    """Return the chord length from a radius and central angle."""

    return round_tenth(
        2.0
        * float(radius_value)
        * math.sin(math.radians(float(central_angle_degrees) / 2.0))
    )


def support_from_defaults(
    defaults: Mapping[str, Any],
    *,
    key: str,
    fallback: Sequence[int],
) -> Tuple[int, ...]:
    """Read an integer support list from scene defaults."""

    raw_support = defaults.get(str(key), tuple(fallback))
    support = tuple(int(value) for value in raw_support)
    if not support:
        raise ValueError(f"{key} must contain at least one value")
    return support


def select_supported_int(
    *,
    rng,
    params: Mapping[str, Any],
    explicit_keys: Sequence[str],
    support: Sequence[int],
) -> Tuple[int, Dict[str, float]]:
    """Select or validate one integer from a named support."""

    values = tuple(int(value) for value in support)
    value_set = set(values)
    for key in explicit_keys:
        if key in params:
            selected = int(params[str(key)])
            if selected not in value_set:
                raise ValueError(f"{key}={selected!r} is not in support {values!r}")
            return selected, {str(selected): 1.0}
    return int(rng.choice(values)), _probability_map(values)


def bind_chord_length_inputs(
    *,
    instance_seed: int,
    namespace: str,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    angle_support_key: str,
    angle_support_fallback: Sequence[int],
    central_angle_from_visible_angle: Callable[[int], int],
    uses_inscribed_angle: bool,
) -> ChordLengthBinding:
    """Resolve visible chord inputs from task-owned semantic constraints."""

    rng = spawn_rng(int(instance_seed), str(namespace))
    radius, radius_probabilities = select_supported_int(
        rng=rng,
        params=params,
        explicit_keys=("radius_value", "radius"),
        support=support_from_defaults(
            defaults,
            key="radius_support",
            fallback=DEFAULT_RADIUS_SUPPORT,
        ),
    )
    angle, angle_probabilities = select_supported_int(
        rng=rng,
        params=params,
        explicit_keys=("angle_degrees", "angle_value"),
        support=support_from_defaults(
            defaults,
            key=str(angle_support_key),
            fallback=angle_support_fallback,
        ),
    )
    central_angle = int(central_angle_from_visible_angle(int(angle)))
    if not (20 <= int(central_angle) <= 170):
        raise ValueError(f"unsupported central angle for chord task: {central_angle}")
    answer = chord_length_from_radius_central_angle(
        radius_value=int(radius),
        central_angle_degrees=int(central_angle),
    )
    return ChordLengthBinding(
        spec=ChordLengthDiagramSpec(
            radius_value=int(radius),
            angle_degrees=int(angle),
            central_angle_degrees=int(central_angle),
            answer_value=float(answer),
            uses_inscribed_angle=bool(uses_inscribed_angle),
        ),
        radius_probabilities=dict(radius_probabilities),
        angle_probabilities=dict(angle_probabilities),
    )


def chord_length_query_params(
    *,
    query_probabilities: Mapping[str, float],
    binding: ChordLengthBinding,
) -> Dict[str, Any]:
    """Serialize task-bound chord inputs for verifier/review traces."""

    spec = binding.spec
    return {
        "query_id_probabilities": dict(query_probabilities),
        "radius_value": int(spec.radius_value),
        "radius_probabilities": dict(binding.radius_probabilities),
        "angle_degrees": int(spec.angle_degrees),
        "angle_probabilities": dict(binding.angle_probabilities),
        "central_angle_degrees": int(spec.central_angle_degrees),
        "answer_rounding": "one_decimal",
    }


def build_chord_length_payload(rng, *, spec: ChordLengthDiagramSpec) -> Dict[str, Any]:
    """Construct a chord/radius diagram from task-bound theorem facts."""

    radius = float(spec.radius_value)
    central_angle = int(spec.central_angle_degrees)
    rotation = float(rng.uniform(15.0, 150.0))
    center = (0.0, 0.0)
    canonical_points: Dict[str, Point] = {
        "O": center,
        "A": _circle_point(radius, rotation - (0.5 * central_angle)),
        "B": _circle_point(radius, rotation + (0.5 * central_angle)),
    }

    if spec.uses_inscribed_angle:
        c_offset = float(rng.uniform(-42.0, 42.0))
        if abs(c_offset) < 10.0:
            c_offset = 10.0 if c_offset >= 0.0 else -10.0
        canonical_points["C"] = _circle_point(radius, rotation + 180.0 + c_offset)

    canonical_labels = ("O", "A", "B", "C") if spec.uses_inscribed_angle else ("O", "A", "B")
    label_map = _fixed_point_label_map(canonical_labels)
    point_model = {label_map[key]: value for key, value in canonical_points.items()}

    radius_segment = _visible_segment(label_map, "O", "A")
    chord_segment = _visible_segment(label_map, "A", "B")
    answer_token = f"{chord_segment}=?"
    radius_token = f"{radius_segment}={int(spec.radius_value)}"
    segments = {
        "OA": (label_map["O"], label_map["A"]),
        "AB": (label_map["A"], label_map["B"]),
    }
    measurement_specs = (
        (radius_token, "OA", -1.0),
        (answer_token, "AB", 1.0),
    )

    if spec.uses_inscribed_angle:
        angle_name = _visible_angle(label_map, "A", "C", "B")
        angle_token = f"{angle_name}={int(spec.angle_degrees)}°"
        segments.update(
            {
                "CA": (label_map["C"], label_map["A"]),
                "CB": (label_map["C"], label_map["B"]),
            }
        )
        angle_marker_specs = (
            {
                "token": angle_token,
                "vertex": label_map["C"],
                "arm0": label_map["A"],
                "arm1": label_map["B"],
                "radius_px": 42.0,
            },
        )
        annotation_point_labels = (
            label_map["O"],
            label_map["A"],
            label_map["B"],
            label_map["C"],
        )
        angle_prompt_slot = "inscribed angle"
    else:
        angle_name = _visible_angle(label_map, "A", "O", "B")
        angle_token = f"{angle_name}={int(spec.angle_degrees)}°"
        segments["OB"] = (label_map["O"], label_map["B"])
        angle_marker_specs = (
            {
                "token": angle_token,
                "vertex": label_map["O"],
                "arm0": label_map["A"],
                "arm1": label_map["B"],
                "radius_px": 48.0,
            },
        )
        annotation_point_labels = (label_map["O"], label_map["A"], label_map["B"])
        angle_prompt_slot = "central angle"

    theorem_trace = {
        "theorem": "chord_length_from_radius_inscribed_angle"
        if spec.uses_inscribed_angle
        else "chord_length_from_radius_central_angle",
        "label_map": dict(label_map),
        "canonical_answer_segment": "AB",
        "answer_segment": str(chord_segment),
        "answer_value": float(spec.answer_value),
        "answer_rounding": "one_decimal",
        "radius_value": int(spec.radius_value),
        "visible_radius_segment": str(radius_segment),
        "angle_degrees": int(spec.angle_degrees),
        "central_angle_degrees": int(spec.central_angle_degrees),
        "visible_angle": str(angle_name),
        "formula": "chord=2*r*sin(theta/2)",
        "distractor_tokens": [],
    }
    return {
        "point_model": point_model,
        "circle_center": center,
        "circle_radius": float(radius),
        "segments": segments,
        "measurement_specs": measurement_specs,
        "angle_marker_specs": angle_marker_specs,
        "circle_arc_specs": tuple(),
        "support_measurement_tokens": (radius_token, angle_token),
        "annotation_point_labels": tuple(annotation_point_labels),
        "annotation_values": {
            str(radius_segment): int(spec.radius_value),
            str(angle_name): int(spec.angle_degrees),
        },
        "theorem_trace": theorem_trace,
        "prompt_slots": {
            "radius_segment": str(radius_segment),
            "angle_name": str(angle_name),
            "angle_kind": str(angle_prompt_slot),
            "answer_segment": str(chord_segment),
            "chord_segment": str(chord_segment),
        },
    }


__all__ = [
    "ChordLengthDiagramSpec",
    "DEFAULT_CENTRAL_ANGLE_SUPPORT",
    "DEFAULT_INSCRIBED_ANGLE_SUPPORT",
    "DEFAULT_RADIUS_SUPPORT",
    "ChordLengthBinding",
    "bind_chord_length_inputs",
    "build_chord_length_payload",
    "chord_length_from_radius_central_angle",
    "chord_length_query_params",
    "round_tenth",
    "select_supported_int",
    "support_from_defaults",
]
