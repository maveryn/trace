"""Scene construction primitives for circle-theorem diagrams."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.pythagorean import (
    IntegerRightTriangle,
    integer_right_triangles,
    validate_integer_right_triangle,
)
from trace_tasks.tasks.shared.fixed_query import geometry_probability_map as _probability_map
from .state import (
    Point,
    CircleTheoremProblem,
    CYCLIC_QUADRILATERAL_ANGLE_SUPPORT,
    EXTERNAL_SECANT_ANGLE_ANSWER_SUPPORT,
    _fixed_point_label_map,
    _visible_segment,
    _visible_angle,
    _visible_arc,
    _line_intersection,
    _angle_degrees_at,
)
from .spatial_primitives import (
    _add_points,
    _apply_external_point_side,
    _circle_point,
    _extend_ray,
    _mirror_point_x,
    _rotated_tangent_unit,
    _sample_external_point_side,
    _split_cyclic_arc_sum,
)
from .sampling import (
    _candidate_diameter_chord_values,
    _candidate_tangent_secant_values,
    _candidate_secant_secant_values,
    _candidate_secant_secant_variable_values,
)


_DEGREE_SIGN = "\N{DEGREE SIGN}"


def _arc_measure_token(arc_name: str, value: int) -> str:
    """Format a circle-arc measure label as degrees, not arc length."""

    return f"{arc_name}={int(value)}{_DEGREE_SIGN}"


def _unknown_arc_measure_token(arc_name: str) -> str:
    """Format an unknown circle-arc measure label with an explicit degree unit."""

    return f"{arc_name}=?{_DEGREE_SIGN}"


def _build_diameter_perpendicular_chord_scene(
    rng, *, problem: CircleTheoremProblem
) -> Dict[str, Any]:
    """Build a perpendicular-diameter chord theorem scene with the missing chord segment fixed."""
    candidates = _candidate_diameter_chord_values(int(problem.target_answer))

    if not candidates:
        raise ValueError(
            f"unsupported target answer for diameter chord theorem: {problem.target_answer}"
        )
    spec = dict(candidates[int(rng.randrange(len(candidates)))])
    label_map = _fixed_point_label_map(("O", "B", "D", "E", "A", "C"))
    radius = float(spec["radius"])
    offset = float(spec["offset"])
    half_chord = float(spec["half_chord"])
    canonical_point_model = {
        "O": (0.0, 0.0),
        "B": (0.0, -radius),
        "D": (0.0, radius),
        "E": (0.0, -offset),
        "A": (-half_chord, -offset),
        "C": (half_chord, -offset),
    }
    point_model = {
        label_map[key]: value for key, value in canonical_point_model.items()
    }
    diameter_segment = _visible_segment(label_map, "D", "B")
    chord_segment = _visible_segment(label_map, "A", "C")
    answer_segment = _visible_segment(label_map, "B", "E")
    angle_token = f"{_visible_angle(label_map, 'D', 'E', 'C')}=90"
    diameter_token = f"{diameter_segment}={int(spec['diameter'])}"
    chord_token = f"{chord_segment}={int(spec['chord'])}"
    theorem_trace = {
        "theorem": "diameter_perpendicular_chord",
        "label_map": dict(label_map),
        "radius": int(spec["radius"]),
        "center_to_chord_distance": int(spec["offset"]),
        "half_chord_length": int(spec["half_chord"]),
        "diameter_length": int(spec["diameter"]),
        "chord_length": int(spec["chord"]),
        "canonical_answer_segment": "BE",
        "answer_segment": str(answer_segment),
        "answer_value": int(spec["answer"]),
        "distractor_tokens": [str(angle_token)],
    }
    return {
        "point_model": point_model,
        "circle_center": (0.0, 0.0),
        "circle_radius": radius,
        "segments": {
            "DB": (label_map["D"], label_map["B"]),
            "AC": (label_map["A"], label_map["C"]),
            "BE": (label_map["B"], label_map["E"]),
            "DE": (label_map["D"], label_map["E"]),
        },
        "measurement_specs": (
            (diameter_token, "DB", -1.0),
            (chord_token, "AC", -1.0),
            (f"{answer_segment}=?", "BE", 1.0),
        ),
        "angle_marker_specs": (
            {
                "token": angle_token,
                "vertex": label_map["E"],
                "arm0": label_map["D"],
                "arm1": label_map["C"],
                "radius_px": 34.0,
            },
        ),
        "support_measurement_tokens": (diameter_token, chord_token),
        "annotation_point_labels": (
            label_map["D"],
            label_map["B"],
            label_map["A"],
            label_map["C"],
            label_map["E"],
        ),
        "annotation_values": {
            str(diameter_segment): int(spec["diameter"]),
            str(chord_segment): int(spec["chord"]),
            str(answer_segment): int(spec["answer"]),
        },
        "theorem_trace": theorem_trace,
        "prompt_slots": {
            "diameter_segment": str(diameter_segment),
            "chord_segment": str(chord_segment),
            "intersection_label": str(label_map["E"]),
            "answer_segment": str(answer_segment),
        },
    }

def _build_tangent_secant_scene(rng, *, problem: CircleTheoremProblem) -> Dict[str, Any]:
    """Build a tangent-secant power theorem scene for the selected hidden segment."""
    candidates = _candidate_tangent_secant_values(

        int(problem.target_answer),
        target_kind=problem.tangent_secant_target_kind,
    )
    if not candidates:
        raise ValueError(
            f"unsupported target answer for tangent secant theorem: {problem.target_answer}"
        )
    spec = dict(candidates[int(rng.randrange(len(candidates)))])
    label_map = _fixed_point_label_map(("P", "A", "B", "T", "O"))
    outside = float(spec["PA"])
    internal = float(spec["AB"])
    tangent = float(spec["PT"])
    target_kind = str(spec["target_kind"])
    radius = internal / 2.0
    center_x = outside + radius
    center = (center_x, 0.0)
    tangent_x = ((center_x * center_x) - (radius * radius)) / center_x
    tangent_y = (radius * tangent) / center_x
    if abs(math.hypot(tangent_x - center_x, tangent_y) - radius) > 1e-7:
        raise ValueError("sampled tangent point is not on the circle")
    if abs((tangent_x * (tangent_x - center_x)) + (tangent_y * tangent_y)) > 1e-7:
        raise ValueError("sampled tangent point is not perpendicular to the radius")
    canonical_point_model = {
        "P": (0.0, 0.0),
        "A": (outside, 0.0),
        "B": (outside + internal, 0.0),
        "T": (float(tangent_x), float(tangent_y)),
        "O": center,
    }
    external_point_side = _sample_external_point_side(rng)
    canonical_point_model, center = _apply_external_point_side(
        canonical_point_model,
        center,
        side=external_point_side,
    )
    point_model = {
        label_map[key]: value for key, value in canonical_point_model.items()
    }
    tangent_segment = _visible_segment(label_map, "P", "T")
    outside_segment = _visible_segment(label_map, "P", "A")
    inside_segment = _visible_segment(label_map, "A", "B")
    full_secant_segment = _visible_segment(label_map, "P", "B")
    answer_segment_by_kind = {
        "outside": outside_segment,
        "inside": inside_segment,
        "tangent": tangent_segment,
    }
    answer_segment = str(answer_segment_by_kind[str(target_kind)])
    canonical_answer_segment = str(spec["canonical_answer_segment"])
    known_token_by_segment = {
        "PT": f"{tangent_segment}={int(tangent)}",
        "PA": f"{outside_segment}={int(outside)}",
        "AB": f"{inside_segment}={int(internal)}",
    }
    known_segment_ids_by_kind = {
        "outside": ("PT", "AB"),
        "inside": ("PT", "PA"),
        "tangent": ("PA", "AB"),
    }
    known_segment_ids = known_segment_ids_by_kind[str(target_kind)]
    tokens = tuple(
        str(known_token_by_segment[str(segment_id)]) for segment_id in known_segment_ids
    )
    measurement_specs_by_kind = {
        "outside": (
            (known_token_by_segment["PT"], "PT", -1.0),
            (f"{outside_segment}=?", "PA", 1.0),
            (known_token_by_segment["AB"], "AB", 1.0),
        ),
        "inside": (
            (known_token_by_segment["PT"], "PT", -1.0),
            (known_token_by_segment["PA"], "PA", 1.0),
            (f"{inside_segment}=?", "AB", 1.0),
        ),
        "tangent": (
            (f"{tangent_segment}=?", "PT", -1.0),
            (known_token_by_segment["PA"], "PA", 1.0),
            (known_token_by_segment["AB"], "AB", 1.0),
        ),
    }
    distractor_angle = _visible_angle(label_map, "T", "P", "A")
    distractor_angle_value = _angle_degrees_at(
        canonical_point_model["P"],
        canonical_point_model["T"],
        canonical_point_model["A"],
    )
    distractor_token = f"{distractor_angle}={int(distractor_angle_value)}"
    theorem_trace = {
        "theorem": "tangent_secant",
        "label_map": dict(label_map),
        "target_kind": str(target_kind),
        "PT": int(tangent),
        "PA": int(outside),
        "AB": int(internal),
        "PB": int(outside + internal),
        "canonical_answer_segment": str(canonical_answer_segment),
        "answer_segment": str(answer_segment),
        "answer_value": int(spec["answer"]),
        "power_PT_squared": int(tangent * tangent),
        "power_PA_times_PB": int(outside * (outside + internal)),
        "distractor_tokens": [str(distractor_token)],
        "external_point_side": str(external_point_side),
    }
    return {
        "point_model": point_model,
        "circle_center": center,
        "circle_radius": float(radius),
        "segments": {
            "PT": (label_map["P"], label_map["T"]),
            "PAB": (label_map["P"], label_map["B"]),
            "PA": (label_map["P"], label_map["A"]),
            "AB": (label_map["A"], label_map["B"]),
            "OB": (label_map["O"], label_map["B"]),
        },
        "measurement_specs": measurement_specs_by_kind[str(target_kind)],
        "angle_marker_specs": (
            {
                "token": distractor_token,
                "vertex": label_map["P"],
                "arm0": label_map["T"],
                "arm1": label_map["A"],
                "radius_px": 36.0,
            },
        ),
        "support_measurement_tokens": tokens,
        "annotation_point_labels": (
            label_map["P"],
            label_map["T"],
            label_map["A"],
            label_map["B"],
        ),
        "annotation_values": {
            str(tangent_segment): int(tangent),
            str(outside_segment): int(outside),
            str(inside_segment): int(internal),
            str(full_secant_segment): int(outside + internal),
        },
        "theorem_trace": theorem_trace,
        "prompt_slots": {
            "external_point": str(label_map["P"]),
            "tangent_point": str(label_map["T"]),
            "near_point": str(label_map["A"]),
            "far_point": str(label_map["B"]),
            "tangent_segment": str(tangent_segment),
            "inside_segment": str(inside_segment),
            "answer_segment": str(answer_segment),
        },
    }

def _build_secant_secant_scene(rng, *, problem: CircleTheoremProblem) -> Dict[str, Any]:
    """Build a two-secant power theorem scene with one outside segment hidden."""
    candidates = _candidate_secant_secant_values(int(problem.target_answer))

    if not candidates:
        raise ValueError(
            f"unsupported target answer for secant secant theorem: {problem.target_answer}"
        )
    spec = dict(candidates[int(rng.randrange(len(candidates)))])
    label_map = _fixed_point_label_map(("P", "A", "B", "C", "D", "O"))
    pa = float(spec["PA"])
    ab = float(spec["AB"])
    pc = float(spec["PC"])
    cd = float(spec["CD"])
    pd = float(spec["PD"])
    radius = ab / 2.0
    center_x = pa + radius
    center = (center_x, 0.0)
    cos_theta = float(pc + pd) / float(2.0 * center_x)
    sin_theta = math.sqrt(max(0.0, 1.0 - (cos_theta * cos_theta)))
    u2 = (float(cos_theta), float(sin_theta))
    canonical_point_model = {
        "P": (0.0, 0.0),
        "A": (pa, 0.0),
        "B": (pa + ab, 0.0),
        "C": (pc * u2[0], pc * u2[1]),
        "D": (pd * u2[0], pd * u2[1]),
        "O": center,
    }
    external_point_side = _sample_external_point_side(rng)
    canonical_point_model, center = _apply_external_point_side(
        canonical_point_model,
        center,
        side=external_point_side,
    )
    for label in ("A", "B", "C", "D"):
        distance = math.hypot(
            canonical_point_model[label][0] - center[0],
            canonical_point_model[label][1] - center[1],
        )
        if abs(float(distance) - float(radius)) > 1e-6:
            raise ValueError("sampled secant point is not on the circle")
    point_model = {
        label_map[key]: value for key, value in canonical_point_model.items()
    }
    outside_segment = _visible_segment(label_map, "P", "A")
    inside_segment = _visible_segment(label_map, "A", "B")
    full_first_secant = _visible_segment(label_map, "P", "B")
    outside_second_segment = _visible_segment(label_map, "P", "C")
    inside_second_segment = _visible_segment(label_map, "C", "D")
    full_second_secant = _visible_segment(label_map, "P", "D")
    distractor_angle = _visible_angle(label_map, "A", "P", "C")
    distractor_angle_value = _angle_degrees_at(
        canonical_point_model["P"],
        canonical_point_model["A"],
        canonical_point_model["C"],
    )
    tokens = (
        f"{inside_segment}={int(ab)}",
        f"{outside_second_segment}={int(pc)}",
        f"{inside_second_segment}={int(cd)}",
    )
    distractor_token = f"{distractor_angle}={int(distractor_angle_value)}"
    theorem_trace = {
        "theorem": "secant_secant",
        "label_map": dict(label_map),
        "PA": int(pa),
        "AB": int(ab),
        "PB": int(pa + ab),
        "PC": int(pc),
        "CD": int(cd),
        "PD": int(pd),
        "canonical_answer_segment": "PA",
        "answer_segment": str(outside_segment),
        "answer_value": int(pa),
        "power_PA_times_PB": int(pa * (pa + ab)),
        "power_PC_times_PD": int(pc * pd),
        "distractor_tokens": [str(distractor_token)],
        "external_point_side": str(external_point_side),
    }
    return {
        "point_model": point_model,
        "circle_center": center,
        "circle_radius": float(radius),
        "segments": {
            "PAB": (label_map["P"], label_map["B"]),
            "PCD": (label_map["P"], label_map["D"]),
            "PA": (label_map["P"], label_map["A"]),
            "AB": (label_map["A"], label_map["B"]),
            "PC": (label_map["P"], label_map["C"]),
            "CD": (label_map["C"], label_map["D"]),
        },
        "measurement_specs": (
            (f"{outside_segment}=?", "PA", 1.0),
            (tokens[0], "AB", 1.0),
            (tokens[1], "PC", -1.0),
            (tokens[2], "CD", 1.0),
        ),
        "angle_marker_specs": (
            {
                "token": distractor_token,
                "vertex": label_map["P"],
                "arm0": label_map["A"],
                "arm1": label_map["C"],
                "radius_px": 38.0,
            },
        ),
        "support_measurement_tokens": tokens,
        "annotation_point_labels": (
            label_map["P"],
            label_map["A"],
            label_map["B"],
            label_map["C"],
            label_map["D"],
        ),
        "annotation_values": {
            str(outside_segment): int(pa),
            str(inside_segment): int(ab),
            str(full_first_secant): int(pa + ab),
            str(outside_second_segment): int(pc),
            str(inside_second_segment): int(cd),
            str(full_second_secant): int(pd),
        },
        "theorem_trace": theorem_trace,
        "prompt_slots": {
            "external_point": str(label_map["P"]),
            "near_point": str(label_map["A"]),
            "far_point": str(label_map["B"]),
            "near_point_alt": str(label_map["C"]),
            "far_point_alt": str(label_map["D"]),
            "inside_segment": str(inside_segment),
            "outside_second_segment": str(outside_second_segment),
            "inside_second_segment": str(inside_second_segment),
            "answer_segment": str(outside_segment),
        },
    }

def _build_secant_secant_variable_scene(
    rng, *, problem: CircleTheoremProblem
) -> Dict[str, Any]:
    """Build a two-secant power theorem scene with a task-selected segment hidden."""
    candidates = _candidate_secant_secant_variable_values(

        int(problem.target_answer),
        target_kind=problem.secant_secant_variable_target_kind,
    )
    if not candidates:
        raise ValueError(
            f"unsupported target answer for variable secant secant theorem: {problem.target_answer}"
        )
    spec = dict(candidates[int(rng.randrange(len(candidates)))])
    label_map = _fixed_point_label_map(("P", "A", "B", "C", "D", "O"))
    pa = float(spec["PA"])
    ab = float(spec["AB"])
    pc = float(spec["PC"])
    cd = float(spec["CD"])
    pd = float(spec["PD"])
    target_kind = str(spec["target_kind"])
    canonical_answer_segment = str(spec["canonical_answer_segment"])
    radius = ab / 2.0
    center_x = pa + radius
    center = (center_x, 0.0)
    cos_theta = float(pc + pd) / float(2.0 * center_x)
    sin_theta = math.sqrt(max(0.0, 1.0 - (cos_theta * cos_theta)))
    u2 = (float(cos_theta), float(sin_theta))
    canonical_point_model = {
        "P": (0.0, 0.0),
        "A": (pa, 0.0),
        "B": (pa + ab, 0.0),
        "C": (pc * u2[0], pc * u2[1]),
        "D": (pd * u2[0], pd * u2[1]),
        "O": center,
    }
    external_point_side = _sample_external_point_side(rng)
    canonical_point_model, center = _apply_external_point_side(
        canonical_point_model,
        center,
        side=external_point_side,
    )
    for label in ("A", "B", "C", "D"):
        distance = math.hypot(
            canonical_point_model[label][0] - center[0],
            canonical_point_model[label][1] - center[1],
        )
        if abs(float(distance) - float(radius)) > 1e-6:
            raise ValueError("sampled secant point is not on the circle")
    point_model = {
        label_map[key]: value for key, value in canonical_point_model.items()
    }
    visible_by_canonical = {
        "PA": _visible_segment(label_map, "P", "A"),
        "AB": _visible_segment(label_map, "A", "B"),
        "PB": _visible_segment(label_map, "P", "B"),
        "PC": _visible_segment(label_map, "P", "C"),
        "CD": _visible_segment(label_map, "C", "D"),
        "PD": _visible_segment(label_map, "P", "D"),
    }
    value_by_canonical = {
        "PA": int(pa),
        "AB": int(ab),
        "PB": int(pa + ab),
        "PC": int(pc),
        "CD": int(cd),
        "PD": int(pd),
    }
    token_by_canonical = {
        canonical: f"{visible_by_canonical[canonical]}={value_by_canonical[canonical]}"
        for canonical in ("PA", "AB", "PC", "CD")
    }
    measurement_token_by_canonical = dict(token_by_canonical)
    measurement_token_by_canonical[str(canonical_answer_segment)] = (
        f"{visible_by_canonical[str(canonical_answer_segment)]}=?"
    )
    tokens = tuple(
        str(token_by_canonical[canonical])
        for canonical in ("PA", "AB", "PC", "CD")
        if str(canonical) != str(canonical_answer_segment)
    )
    distractor_angle = _visible_angle(label_map, "A", "P", "C")
    distractor_angle_value = _angle_degrees_at(
        canonical_point_model["P"],
        canonical_point_model["A"],
        canonical_point_model["C"],
    )
    distractor_token = f"{distractor_angle}={int(distractor_angle_value)}"
    theorem_trace = {
        "theorem": "secant_secant_variable",
        "label_map": dict(label_map),
        "target_kind": str(target_kind),
        "PA": int(pa),
        "AB": int(ab),
        "PB": int(pa + ab),
        "PC": int(pc),
        "CD": int(cd),
        "PD": int(pd),
        "canonical_answer_segment": str(canonical_answer_segment),
        "answer_segment": str(visible_by_canonical[str(canonical_answer_segment)]),
        "answer_value": int(spec["answer"]),
        "power_PA_times_PB": int(pa * (pa + ab)),
        "power_PC_times_PD": int(pc * pd),
        "distractor_tokens": [str(distractor_token)],
        "external_point_side": str(external_point_side),
    }
    return {
        "point_model": point_model,
        "circle_center": center,
        "circle_radius": float(radius),
        "segments": {
            "PAB": (label_map["P"], label_map["B"]),
            "PCD": (label_map["P"], label_map["D"]),
            "PA": (label_map["P"], label_map["A"]),
            "AB": (label_map["A"], label_map["B"]),
            "PC": (label_map["P"], label_map["C"]),
            "CD": (label_map["C"], label_map["D"]),
        },
        "measurement_specs": (
            (measurement_token_by_canonical["PA"], "PA", 1.0),
            (measurement_token_by_canonical["AB"], "AB", 1.0),
            (measurement_token_by_canonical["PC"], "PC", -1.0),
            (measurement_token_by_canonical["CD"], "CD", 1.0),
        ),
        "angle_marker_specs": (
            {
                "token": distractor_token,
                "vertex": label_map["P"],
                "arm0": label_map["A"],
                "arm1": label_map["C"],
                "radius_px": 38.0,
            },
        ),
        "support_measurement_tokens": tokens,
        "annotation_point_labels": (
            label_map["P"],
            label_map["A"],
            label_map["B"],
            label_map["C"],
            label_map["D"],
        ),
        "annotation_values": {
            str(visible_by_canonical[key]): int(value)
            for key, value in value_by_canonical.items()
        },
        "theorem_trace": theorem_trace,
        "prompt_slots": {
            "external_point": str(label_map["P"]),
            "near_point": str(label_map["A"]),
            "far_point": str(label_map["B"]),
            "near_point_alt": str(label_map["C"]),
            "far_point_alt": str(label_map["D"]),
            "inside_segment": str(visible_by_canonical["AB"]),
            "outside_second_segment": str(visible_by_canonical["PC"]),
            "inside_second_segment": str(visible_by_canonical["CD"]),
            "answer_segment": str(visible_by_canonical[str(canonical_answer_segment)]),
        },
    }

def _build_intersecting_chords_arc_scene(
    rng, *, problem: CircleTheoremProblem
) -> Dict[str, Any]:
    """Build an intersecting-chords angle scene and solve the missing intercepted arc."""
    target_arc = int(problem.target_answer)

    if not (40 <= int(target_arc) <= 180):
        raise ValueError(
            f"unsupported target answer for intersecting-chords arc theorem: {problem.target_answer}"
        )
    label_map = _fixed_point_label_map(("O", "A", "B", "C", "D", "E"))
    known_arc_candidates = [
        value
        for value in range(40, 171)
        if 35 <= int(360 - int(target_arc) - int(value) - 55)
        and int(value + target_arc) % 2 == 0
    ]
    if not known_arc_candidates:
        raise ValueError(f"no feasible known arc for target arc: {problem.target_answer}")
    known_arc = int(rng.choice(known_arc_candidates))
    gap_bc_candidates = [
        value
        for value in range(45, 131)
        if int(360 - int(known_arc) - int(target_arc) - int(value)) >= 45
    ]
    if not gap_bc_candidates:
        raise ValueError(f"no feasible arc gap for target arc: {problem.target_answer}")
    arc_bc = int(rng.choice(gap_bc_candidates))
    arc_da = int(360 - int(known_arc) - int(target_arc) - int(arc_bc))
    angle_value = int((int(known_arc) + int(target_arc)) // 2)
    radius = float(rng.choice((10, 11, 12, 13, 14)))
    rotation = float(rng.choice((35, 50, 65, 80, 95, 110, 125)))
    center = (0.0, 0.0)
    angles = {
        "A": float(rotation),
        "B": float(rotation + known_arc),
        "C": float(rotation + known_arc + arc_bc),
        "D": float(rotation + known_arc + arc_bc + target_arc),
    }
    canonical_point_model = {
        "O": center,
        "A": _circle_point(radius, angles["A"]),
        "B": _circle_point(radius, angles["B"]),
        "C": _circle_point(radius, angles["C"]),
        "D": _circle_point(radius, angles["D"]),
    }
    canonical_point_model["E"] = _line_intersection(
        canonical_point_model["A"],
        canonical_point_model["C"],
        canonical_point_model["B"],
        canonical_point_model["D"],
    )
    observed_angle = _angle_degrees_at(
        canonical_point_model["E"],
        canonical_point_model["A"],
        canonical_point_model["B"],
    )
    if abs(int(observed_angle) - int(angle_value)) > 1:
        raise ValueError("sampled intersecting-chords angle does not match arc theorem")
    point_model = {
        label_map[key]: value for key, value in canonical_point_model.items()
    }
    first_chord = _visible_segment(label_map, "A", "C")
    second_chord = _visible_segment(label_map, "B", "D")
    angle_name = _visible_angle(label_map, "A", "E", "B")
    known_arc_name = _visible_arc(label_map, "A", "B")
    answer_arc_name = _visible_arc(label_map, "C", "D")
    distractor_arc_name = _visible_arc(label_map, "B", "C")
    angle_token = f"{angle_name}={int(angle_value)}"
    known_arc_token = _arc_measure_token(known_arc_name, known_arc)
    distractor_token = _arc_measure_token(distractor_arc_name, arc_bc)
    query_arc_token = _unknown_arc_measure_token(answer_arc_name)
    theorem_trace = {
        "theorem": "intersecting_chords_angle",
        "label_map": dict(label_map),
        "angle_AEB": int(angle_value),
        "arc_AB": int(known_arc),
        "arc_BC": int(arc_bc),
        "arc_CD": int(target_arc),
        "arc_DA": int(arc_da),
        "canonical_answer_segment": "arcCD",
        "answer_segment": str(answer_arc_name),
        "answer_value": int(target_arc),
        "arc_sum_for_angle": int(known_arc + target_arc),
        "distractor_tokens": [str(distractor_token)],
    }
    return {
        "point_model": point_model,
        "circle_center": center,
        "circle_radius": float(radius),
        "segments": {
            "AC": (label_map["A"], label_map["C"]),
            "BD": (label_map["B"], label_map["D"]),
        },
        "measurement_specs": tuple(),
        "angle_marker_specs": (
            {
                "token": angle_token,
                "vertex": label_map["E"],
                "arm0": label_map["A"],
                "arm1": label_map["B"],
                "radius_px": 42.0,
            },
        ),
        "circle_arc_specs": (
            {"token": known_arc_token, "start": label_map["A"], "end": label_map["B"]},
            {"token": distractor_token, "start": label_map["B"], "end": label_map["C"]},
            {"token": query_arc_token, "start": label_map["C"], "end": label_map["D"]},
        ),
        "support_measurement_tokens": (angle_token, known_arc_token),
        "annotation_point_labels": (
            label_map["A"],
            label_map["E"],
            label_map["B"],
            label_map["C"],
            label_map["D"],
        ),
        "annotation_values": {
            str(angle_name): int(angle_value),
            str(known_arc_name): int(known_arc),
            str(distractor_arc_name): int(arc_bc),
            str(answer_arc_name): int(target_arc),
        },
        "theorem_trace": theorem_trace,
        "prompt_slots": {
            "first_chord_segment": str(first_chord),
            "second_chord_segment": str(second_chord),
            "intersection_label": str(label_map["E"]),
            "known_arc": str(known_arc_name),
            "answer_arc": str(answer_arc_name),
            "answer_segment": str(answer_arc_name),
        },
    }

def _build_multi_step_angle_scene(rng, *, problem: CircleTheoremProblem) -> Dict[str, Any]:
    """Build an intersecting-chords scene where arc sums determine the target angle."""
    target_angle = int(problem.target_answer)

    if not (45 <= int(target_angle) <= 135):
        raise ValueError(
            f"unsupported target answer for multi-step circle angle theorem: {problem.target_answer}"
        )
    label_map = _fixed_point_label_map(("O", "A", "B", "C", "D", "E"))
    arc_sum = int(2 * int(target_angle))
    known_arc_candidates = [
        value for value in range(40, 171) if 40 <= int(arc_sum - int(value)) <= 170
    ]
    if not known_arc_candidates:
        raise ValueError(
            f"no feasible known arcs for target angle: {problem.target_answer}"
        )
    first_arc = int(rng.choice(known_arc_candidates))
    opposite_arc = int(arc_sum - int(first_arc))
    remaining_arc = int(360 - int(first_arc) - int(opposite_arc))
    if int(remaining_arc) < 90:
        raise ValueError(
            f"no feasible remaining arcs for target angle: {problem.target_answer}"
        )
    gap_bc_candidates = [
        value
        for value in range(45, int(remaining_arc) - 44)
        if int(remaining_arc - int(value)) >= 45
    ]
    if not gap_bc_candidates:
        raise ValueError(f"no feasible arc gap for target angle: {problem.target_answer}")
    arc_bc = int(rng.choice(gap_bc_candidates))
    arc_da = int(remaining_arc - int(arc_bc))
    radius = float(rng.choice((10, 11, 12, 13, 14)))
    rotation = float(rng.choice((30, 45, 60, 75, 90, 105, 120)))
    center = (0.0, 0.0)
    angles = {
        "A": float(rotation),
        "B": float(rotation + first_arc),
        "C": float(rotation + first_arc + arc_bc),
        "D": float(rotation + first_arc + arc_bc + opposite_arc),
    }
    canonical_point_model = {
        "O": center,
        "A": _circle_point(radius, angles["A"]),
        "B": _circle_point(radius, angles["B"]),
        "C": _circle_point(radius, angles["C"]),
        "D": _circle_point(radius, angles["D"]),
    }
    canonical_point_model["E"] = _line_intersection(
        canonical_point_model["A"],
        canonical_point_model["C"],
        canonical_point_model["B"],
        canonical_point_model["D"],
    )
    observed_angle = _angle_degrees_at(
        canonical_point_model["E"],
        canonical_point_model["A"],
        canonical_point_model["B"],
    )
    if abs(int(observed_angle) - int(target_angle)) > 1:
        raise ValueError(
            "sampled multi-step angle does not match intersecting-chords theorem"
        )
    point_model = {
        label_map[key]: value for key, value in canonical_point_model.items()
    }
    first_chord = _visible_segment(label_map, "A", "C")
    second_chord = _visible_segment(label_map, "B", "D")
    angle_name = _visible_angle(label_map, "A", "E", "B")
    first_arc_name = _visible_arc(label_map, "A", "B")
    opposite_arc_name = _visible_arc(label_map, "C", "D")
    distractor_arc_name = _visible_arc(label_map, "B", "C")
    answer_angle_token = f"{angle_name}=?"
    first_arc_token = _arc_measure_token(first_arc_name, first_arc)
    opposite_arc_token = _arc_measure_token(opposite_arc_name, opposite_arc)
    distractor_token = _arc_measure_token(distractor_arc_name, arc_bc)
    theorem_trace = {
        "theorem": "intersecting_chords_angle_from_arcs",
        "label_map": dict(label_map),
        "angle_AEB": int(target_angle),
        "arc_AB": int(first_arc),
        "arc_BC": int(arc_bc),
        "arc_CD": int(opposite_arc),
        "arc_DA": int(arc_da),
        "canonical_answer_segment": "angleAEB",
        "answer_segment": str(angle_name),
        "answer_value": int(target_angle),
        "arc_sum_for_angle": int(first_arc + opposite_arc),
        "distractor_tokens": [str(distractor_token)],
    }
    return {
        "point_model": point_model,
        "circle_center": center,
        "circle_radius": float(radius),
        "segments": {
            "AC": (label_map["A"], label_map["C"]),
            "BD": (label_map["B"], label_map["D"]),
        },
        "measurement_specs": tuple(),
        "angle_marker_specs": (
            {
                "token": answer_angle_token,
                "vertex": label_map["E"],
                "arm0": label_map["A"],
                "arm1": label_map["B"],
                "radius_px": 42.0,
            },
        ),
        "circle_arc_specs": (
            {"token": first_arc_token, "start": label_map["A"], "end": label_map["B"]},
            {"token": distractor_token, "start": label_map["B"], "end": label_map["C"]},
            {
                "token": opposite_arc_token,
                "start": label_map["C"],
                "end": label_map["D"],
            },
        ),
        "support_measurement_tokens": (first_arc_token, opposite_arc_token),
        "annotation_point_labels": (
            label_map["A"],
            label_map["B"],
            label_map["C"],
            label_map["D"],
            label_map["E"],
        ),
        "annotation_values": {
            str(angle_name): int(target_angle),
            str(first_arc_name): int(first_arc),
            str(distractor_arc_name): int(arc_bc),
            str(opposite_arc_name): int(opposite_arc),
        },
        "theorem_trace": theorem_trace,
        "prompt_slots": {
            "first_chord_segment": str(first_chord),
            "second_chord_segment": str(second_chord),
            "intersection_label": str(label_map["E"]),
            "answer_angle": str(angle_name),
            "answer_segment": str(angle_name),
        },
    }

def _build_inscribed_angle_scene(
    rng,
    *,
    problem: CircleTheoremProblem,
    source_kind: str,
) -> Dict[str, Any]:
    """Build an inscribed-angle theorem scene from the selected semantic source measurement."""
    source = str(source_kind)

    if source == "known_inscribed":
        central_angle = int(problem.target_answer)
        if int(central_angle) % 2 != 0 or not (40 <= int(central_angle) <= 160):
            raise ValueError(f"unsupported central angle answer: {problem.target_answer}")
        inscribed_angle = int(central_angle // 2)
        answer_kind = "central"
    elif source in {"known_central", "known_arc"}:
        inscribed_angle = int(problem.target_answer)
        if not (20 <= int(inscribed_angle) <= 80):
            raise ValueError(
                f"unsupported inscribed angle answer: {problem.target_answer}"
            )
        central_angle = int(2 * int(inscribed_angle))
        answer_kind = "inscribed"
    else:
        raise ValueError(f"unsupported inscribed-angle source kind: {source_kind}")

    radius = float(rng.choice((10, 11, 12, 13, 14)))
    rotation = float(rng.choice((20, 35, 50, 65, 80, 95, 110, 125, 140)))
    c_offset = float(rng.choice((-35, -25, -15, 15, 25, 35)))
    angles = {
        "A": float(rotation - (0.5 * central_angle)),
        "B": float(rotation + (0.5 * central_angle)),
        "C": float(rotation + 180.0 + c_offset),
    }
    center = (0.0, 0.0)
    canonical_point_model = {
        "O": center,
        "A": _circle_point(radius, angles["A"]),
        "B": _circle_point(radius, angles["B"]),
        "C": _circle_point(radius, angles["C"]),
    }
    observed_inscribed = _angle_degrees_at(
        canonical_point_model["C"],
        canonical_point_model["A"],
        canonical_point_model["B"],
    )
    observed_central = _angle_degrees_at(
        canonical_point_model["O"],
        canonical_point_model["A"],
        canonical_point_model["B"],
    )
    if abs(int(observed_inscribed) - int(inscribed_angle)) > 1:
        raise ValueError("sampled inscribed angle does not match intercepted arc")
    if abs(int(observed_central) - int(central_angle)) > 1:
        raise ValueError("sampled central angle does not match intercepted arc")

    label_map = _fixed_point_label_map(("O", "A", "B", "C"))
    point_model = {
        label_map[key]: value for key, value in canonical_point_model.items()
    }
    central_angle_name = _visible_angle(label_map, "A", "O", "B")
    inscribed_angle_name = _visible_angle(label_map, "A", "C", "B")
    intercepted_arc_name = _visible_arc(label_map, "A", "B")
    distractor_arc_name = _visible_arc(label_map, "B", "C")
    raw_distractor_arc = float((angles["C"] - angles["B"]) % 360.0)
    distractor_arc = int(
        round(min(raw_distractor_arc, 360.0 - raw_distractor_arc) / 5.0) * 5
    )

    token_by_kind = {
        "central": f"{central_angle_name}={int(central_angle)}",
        "inscribed": f"{inscribed_angle_name}={int(inscribed_angle)}",
        "arc": _arc_measure_token(intercepted_arc_name, central_angle),
    }
    if source == "known_central":
        support_measurement_tokens = (token_by_kind["central"],)
        annotation_point_labels = (
            label_map["A"],
            label_map["O"],
            label_map["B"],
            label_map["C"],
        )
        angle_marker_specs = (
            {
                "token": token_by_kind["central"],
                "vertex": label_map["O"],
                "arm0": label_map["A"],
                "arm1": label_map["B"],
                "radius_px": 48.0,
            },
            {
                "token": f"{inscribed_angle_name}=?",
                "vertex": label_map["C"],
                "arm0": label_map["A"],
                "arm1": label_map["B"],
                "radius_px": 42.0,
            },
        )
        circle_arc_specs = (
            {
                "token": _arc_measure_token(distractor_arc_name, distractor_arc),
                "start": label_map["B"],
                "end": label_map["C"],
            },
        )
    elif source == "known_inscribed":
        support_measurement_tokens = (token_by_kind["inscribed"],)
        annotation_point_labels = (
            label_map["A"],
            label_map["C"],
            label_map["B"],
            label_map["O"],
        )
        angle_marker_specs = (
            {
                "token": f"{central_angle_name}=?",
                "vertex": label_map["O"],
                "arm0": label_map["A"],
                "arm1": label_map["B"],
                "radius_px": 48.0,
            },
            {
                "token": token_by_kind["inscribed"],
                "vertex": label_map["C"],
                "arm0": label_map["A"],
                "arm1": label_map["B"],
                "radius_px": 42.0,
            },
        )
        circle_arc_specs = (
            {
                "token": _arc_measure_token(distractor_arc_name, distractor_arc),
                "start": label_map["B"],
                "end": label_map["C"],
            },
        )
    else:
        support_measurement_tokens = (token_by_kind["arc"],)
        annotation_point_labels = (
            label_map["A"],
            label_map["B"],
            label_map["C"],
        )
        angle_marker_specs = (
            {
                "token": f"{inscribed_angle_name}=?",
                "vertex": label_map["C"],
                "arm0": label_map["A"],
                "arm1": label_map["B"],
                "radius_px": 42.0,
            },
        )
        circle_arc_specs = (
            {
                "token": token_by_kind["arc"],
                "start": label_map["A"],
                "end": label_map["B"],
            },
            {
                "token": _arc_measure_token(distractor_arc_name, distractor_arc),
                "start": label_map["B"],
                "end": label_map["C"],
            },
        )

    answer_name = (
        central_angle_name if answer_kind == "central" else inscribed_angle_name
    )
    answer_value = int(central_angle if answer_kind == "central" else inscribed_angle)
    distractor_token = _arc_measure_token(distractor_arc_name, distractor_arc)
    theorem_trace = {
        "theorem": "inscribed_angle",
        "label_map": dict(label_map),
        "central_angle_AOB": int(central_angle),
        "inscribed_angle_ACB": int(inscribed_angle),
        "arc_AB": int(central_angle),
        "arc_BC": int(distractor_arc),
        "canonical_answer_segment": (
            "angleAOB" if answer_kind == "central" else "angleACB"
        ),
        "answer_segment": str(answer_name),
        "answer_value": int(answer_value),
        "distractor_tokens": [str(distractor_token)],
    }
    return {
        "point_model": point_model,
        "circle_center": center,
        "circle_radius": float(radius),
        "segments": {
            "OA": (label_map["O"], label_map["A"]),
            "OB": (label_map["O"], label_map["B"]),
            "CA": (label_map["C"], label_map["A"]),
            "CB": (label_map["C"], label_map["B"]),
        },
        "measurement_specs": tuple(),
        "angle_marker_specs": angle_marker_specs,
        "circle_arc_specs": circle_arc_specs,
        "support_measurement_tokens": support_measurement_tokens,
        "annotation_point_labels": annotation_point_labels,
        "annotation_values": {
            str(central_angle_name): int(central_angle),
            str(inscribed_angle_name): int(inscribed_angle),
            str(intercepted_arc_name): int(central_angle),
            str(distractor_arc_name): int(distractor_arc),
        },
        "theorem_trace": theorem_trace,
        "prompt_slots": {
            "central_angle": str(central_angle_name),
            "inscribed_angle": str(inscribed_angle_name),
            "intercepted_arc": str(intercepted_arc_name),
            "answer_angle": str(answer_name),
            "answer_segment": str(answer_name),
        },
    }


def _build_inscribed_angle_from_central_scene(rng, *, problem: CircleTheoremProblem) -> Dict[str, Any]:
    """Build the branch where a central angle determines an inscribed angle."""

    return _build_inscribed_angle_scene(rng, problem=problem, source_kind="known_central")


def _build_central_angle_from_inscribed_scene(rng, *, problem: CircleTheoremProblem) -> Dict[str, Any]:
    """Build the branch where an inscribed angle determines a central angle."""

    return _build_inscribed_angle_scene(rng, problem=problem, source_kind="known_inscribed")


def _build_inscribed_angle_from_arc_scene(rng, *, problem: CircleTheoremProblem) -> Dict[str, Any]:
    """Build the branch where an intercepted arc determines an inscribed angle."""

    return _build_inscribed_angle_scene(rng, problem=problem, source_kind="known_arc")

def _build_tangent_chord_angle_scene(
    rng,
    *,
    problem: CircleTheoremProblem,
    source_kind: str,
) -> Dict[str, Any]:
    """Build a tangent-chord theorem scene from the selected semantic source measurement."""
    source = str(source_kind)

    tangent_chord_angle = int(problem.target_answer)
    if not (25 <= int(tangent_chord_angle) <= 75):
        raise ValueError(
            f"unsupported tangent-chord angle answer: {problem.target_answer}"
        )
    if source not in {"known_arc", "known_inscribed"}:
        raise ValueError(f"unsupported tangent-chord source kind: {source_kind}")

    central_angle = int(2 * int(tangent_chord_angle))
    radius = float(rng.choice((10, 11, 12, 13, 14)))
    rotation = float(rng.choice((10, 25, 40, 55, 70, 85, 100, 115, 130)))
    tangent_unit = _rotated_tangent_unit(rotation)
    point_t = _circle_point(radius, rotation)
    point_a = _circle_point(radius, rotation + central_angle)
    arc_ab = int(rng.choice((105, 120, 135, 150)))
    point_b = _circle_point(radius, rotation + central_angle + float(arc_ab))
    point_p = _add_points(point_t, tangent_unit, scale=float(radius * 1.05))
    center = (0.0, 0.0)
    canonical_point_model = {
        "O": center,
        "P": point_p,
        "T": point_t,
        "A": point_a,
        "B": point_b,
    }
    observed_tangent_angle = _angle_degrees_at(
        canonical_point_model["T"],
        canonical_point_model["P"],
        canonical_point_model["A"],
    )
    observed_inscribed = _angle_degrees_at(
        canonical_point_model["B"],
        canonical_point_model["T"],
        canonical_point_model["A"],
    )
    if abs(int(observed_tangent_angle) - int(tangent_chord_angle)) > 1:
        raise ValueError("sampled tangent-chord angle does not match intercepted arc")
    if abs(int(observed_inscribed) - int(tangent_chord_angle)) > 1:
        raise ValueError("sampled inscribed angle does not match tangent-chord angle")

    label_map = _fixed_point_label_map(("O", "P", "T", "A", "B"))
    point_model = {
        label_map[key]: value for key, value in canonical_point_model.items()
    }
    tangent_chord_angle_name = _visible_angle(label_map, "P", "T", "A")
    inscribed_angle_name = _visible_angle(label_map, "T", "B", "A")
    intercepted_arc_name = _visible_arc(label_map, "T", "A")
    distractor_arc_name = _visible_arc(label_map, "A", "B")
    distractor_arc = int(arc_ab)
    answer_token = f"{tangent_chord_angle_name}=?"
    arc_token = _arc_measure_token(intercepted_arc_name, central_angle)
    inscribed_token = f"{inscribed_angle_name}={int(tangent_chord_angle)}"
    distractor_token = _arc_measure_token(distractor_arc_name, distractor_arc)
    if source == "known_arc":
        support_measurement_tokens = (arc_token,)
        annotation_point_labels = (
            label_map["P"],
            label_map["T"],
            label_map["A"],
        )
        extra_angle_token: str | None = None
    else:
        support_measurement_tokens = (inscribed_token,)
        annotation_point_labels = (
            label_map["P"],
            label_map["T"],
            label_map["A"],
            label_map["B"],
        )
        extra_angle_token = inscribed_token

    angle_marker_specs: Tuple[Mapping[str, Any], ...]
    if extra_angle_token is None:
        angle_marker_specs = (
            {
                "token": answer_token,
                "vertex": label_map["T"],
                "arm0": label_map["P"],
                "arm1": label_map["A"],
                "radius_px": 42.0,
            },
        )
    else:
        angle_marker_specs = (
            {
                "token": answer_token,
                "vertex": label_map["T"],
                "arm0": label_map["P"],
                "arm1": label_map["A"],
                "radius_px": 42.0,
            },
            {
                "token": extra_angle_token,
                "vertex": label_map["B"],
                "arm0": label_map["T"],
                "arm1": label_map["A"],
                "radius_px": 38.0,
            },
        )

    circle_arc_specs = (
        {
            "token": (
                arc_token if source == "known_arc" else None
            ),
            "start": label_map["T"],
            "end": label_map["A"],
        },
        {"token": distractor_token, "start": label_map["A"], "end": label_map["B"]},
    )
    theorem_trace = {
        "theorem": "tangent_chord_angle",
        "label_map": dict(label_map),
        "angle_PTA": int(tangent_chord_angle),
        "angle_TBA": int(tangent_chord_angle),
        "arc_TA": int(central_angle),
        "arc_AB": int(distractor_arc),
        "canonical_answer_segment": "anglePTA",
        "answer_segment": str(tangent_chord_angle_name),
        "answer_value": int(tangent_chord_angle),
        "distractor_tokens": [str(distractor_token)],
    }
    return {
        "point_model": point_model,
        "circle_center": center,
        "circle_radius": float(radius),
        "segments": {
            "PT": (label_map["P"], label_map["T"]),
            "TA": (label_map["T"], label_map["A"]),
            "OT": (label_map["O"], label_map["T"]),
            "BT": (label_map["B"], label_map["T"]),
            "BA": (label_map["B"], label_map["A"]),
        },
        "measurement_specs": tuple(),
        "angle_marker_specs": angle_marker_specs,
        "circle_arc_specs": circle_arc_specs,
        "support_measurement_tokens": support_measurement_tokens,
        "annotation_point_labels": annotation_point_labels,
        "annotation_values": {
            str(tangent_chord_angle_name): int(tangent_chord_angle),
            str(inscribed_angle_name): int(tangent_chord_angle),
            str(intercepted_arc_name): int(central_angle),
            str(distractor_arc_name): int(distractor_arc),
        },
        "theorem_trace": theorem_trace,
        "prompt_slots": {
            "tangent_chord_angle": str(tangent_chord_angle_name),
            "inscribed_angle": str(inscribed_angle_name),
            "intercepted_arc": str(intercepted_arc_name),
            "answer_angle": str(tangent_chord_angle_name),
            "answer_segment": str(tangent_chord_angle_name),
        },
    }


def _build_tangent_chord_angle_from_arc_scene(rng, *, problem: CircleTheoremProblem) -> Dict[str, Any]:
    """Build the branch where the intercepted arc is the support value."""

    return _build_tangent_chord_angle_scene(rng, problem=problem, source_kind="known_arc")


def _build_tangent_chord_angle_from_inscribed_scene(rng, *, problem: CircleTheoremProblem) -> Dict[str, Any]:
    """Build the branch where the inscribed angle is the support value."""

    return _build_tangent_chord_angle_scene(rng, problem=problem, source_kind="known_inscribed")

def _build_external_secant_angle_scene(
    rng, *, problem: CircleTheoremProblem
) -> Dict[str, Any]:
    """Build an external-secant angle scene using far and near intercepted arcs."""
    target_angle = int(problem.target_answer)

    if int(target_angle) not in EXTERNAL_SECANT_ANGLE_ANSWER_SUPPORT:
        raise ValueError(
            f"unsupported target answer for external secant angle theorem: {problem.target_answer}"
        )
    beta_candidates = [
        value
        for value in range(10, 61)
        if int(value + target_angle) <= 85
        and int(180 - (value + target_angle) - value) >= 35
    ]
    if not beta_candidates:
        raise ValueError(
            f"no feasible intercepted arcs for external secant angle: {problem.target_answer}"
        )
    beta = int(rng.choice(beta_candidates))
    alpha = int(beta + target_angle)
    near_arc = int(2 * beta)
    far_arc = int(2 * alpha)
    distractor_arc = int(180 - alpha - beta)
    radius = float(rng.choice((10, 11, 12, 13, 14)))
    rotation = float(rng.choice((-15, -8, 0, 8, 15)))
    center = (0.0, 0.0)
    angles = {
        "A": float(180 - alpha + rotation),
        "B": float(beta + rotation),
        "C": float(180 + alpha + rotation),
        "D": float(360 - beta + rotation),
    }
    canonical_point_model = {
        "O": center,
        "A": _circle_point(radius, angles["A"]),
        "B": _circle_point(radius, angles["B"]),
        "C": _circle_point(radius, angles["C"]),
        "D": _circle_point(radius, angles["D"]),
    }
    canonical_point_model["P"] = _line_intersection(
        canonical_point_model["A"],
        canonical_point_model["B"],
        canonical_point_model["C"],
        canonical_point_model["D"],
    )
    observed_angle = _angle_degrees_at(
        canonical_point_model["P"],
        canonical_point_model["B"],
        canonical_point_model["D"],
    )
    if abs(int(observed_angle) - int(target_angle)) > 1:
        raise ValueError("sampled external secant angle does not match arc theorem")
    if (
        math.hypot(
            float(canonical_point_model["P"][0]) - float(center[0]),
            float(canonical_point_model["P"][1]) - float(center[1]),
        )
        <= float(radius) * 1.03
    ):
        raise ValueError("sampled external secant point is not outside the circle")

    external_point_side = _sample_external_point_side(rng)
    if str(external_point_side) == "left":
        canonical_point_model = {
            str(label): _mirror_point_x(point)
            for label, point in canonical_point_model.items()
        }
        center = _mirror_point_x(center)
    elif str(external_point_side) != "right":
        raise ValueError(f"unsupported external point side: {external_point_side!r}")

    label_map = _fixed_point_label_map(("O", "P", "A", "B", "C", "D"))
    point_model = {
        label_map[key]: value for key, value in canonical_point_model.items()
    }
    answer_angle_name = _visible_angle(label_map, "B", "P", "D")
    far_arc_name = _visible_arc(label_map, "A", "C")
    near_arc_name = _visible_arc(label_map, "B", "D")
    distractor_arc_name = _visible_arc(label_map, "A", "B")
    answer_token = f"{answer_angle_name}=?"
    far_arc_token = _arc_measure_token(far_arc_name, far_arc)
    near_arc_token = _arc_measure_token(near_arc_name, near_arc)
    distractor_token = _arc_measure_token(distractor_arc_name, distractor_arc)
    theorem_trace = {
        "theorem": "external_secant_angle_from_arcs",
        "label_map": dict(label_map),
        "angle_BPD": int(target_angle),
        "arc_AC": int(far_arc),
        "arc_BD": int(near_arc),
        "arc_AB": int(distractor_arc),
        "canonical_answer_segment": "angleBPD",
        "answer_segment": str(answer_angle_name),
        "answer_value": int(target_angle),
        "far_intercepted_arc_measure": int(far_arc),
        "near_intercepted_arc_measure": int(near_arc),
        "arc_difference_for_angle": int(far_arc - near_arc),
        "external_point_side": str(external_point_side),
        "distractor_tokens": [str(distractor_token)],
    }
    return {
        "point_model": point_model,
        "circle_center": center,
        "circle_radius": float(radius),
        "segments": {
            "first_secant": (label_map["P"], label_map["A"]),
            "second_secant": (label_map["P"], label_map["C"]),
        },
        "measurement_specs": tuple(),
        "angle_marker_specs": (
            {
                "token": answer_token,
                "vertex": label_map["P"],
                "arm0": label_map["B"],
                "arm1": label_map["D"],
                "radius_px": 42.0,
            },
        ),
        "circle_arc_specs": (
            {"token": far_arc_token, "start": label_map["A"], "end": label_map["C"]},
            {"token": near_arc_token, "start": label_map["B"], "end": label_map["D"]},
            {
                "token": distractor_token,
                "start": label_map["A"],
                "end": label_map["B"],
            },
        ),
        "support_measurement_tokens": (far_arc_token, near_arc_token),
        "annotation_point_labels": (
            label_map["P"],
            label_map["B"],
            label_map["A"],
            label_map["D"],
            label_map["C"],
        ),
        "annotation_values": {
            str(answer_angle_name): int(target_angle),
            str(far_arc_name): int(far_arc),
            str(near_arc_name): int(near_arc),
            str(distractor_arc_name): int(distractor_arc),
        },
        "theorem_trace": theorem_trace,
        "prompt_slots": {
            "external_point": str(label_map["P"]),
            "near_point": str(label_map["B"]),
            "far_point": str(label_map["A"]),
            "near_point_alt": str(label_map["D"]),
            "far_point_alt": str(label_map["C"]),
            "near_arc": str(near_arc_name),
            "far_arc": str(far_arc_name),
            "answer_angle": str(answer_angle_name),
            "answer_segment": str(answer_angle_name),
        },
    }

def _build_cyclic_quadrilateral_angle_scene(
    rng, *, problem: CircleTheoremProblem, target_kind: str
) -> Dict[str, Any]:
    """Build a cyclic-quadrilateral angle scene for opposite or exterior angle transfer."""
    target_angle = int(problem.target_answer)

    if int(target_angle) not in CYCLIC_QUADRILATERAL_ANGLE_SUPPORT:
        raise ValueError(
            f"unsupported target answer for cyclic quadrilateral angle theorem: {problem.target_answer}"
        )
    kind = str(target_kind)
    if kind not in {"opposite", "exterior"}:
        raise ValueError(f"unsupported cyclic quadrilateral target kind: {target_kind}")

    target_vertex = str(rng.choice(("B", "D")))
    if kind == "opposite":
        angle_b = int(target_angle if target_vertex == "B" else 180 - target_angle)
        known_vertex = "D" if target_vertex == "B" else "B"
        target_canonical_angle = "ABC" if target_vertex == "B" else "CDA"
        known_canonical_angle = "CDA" if target_vertex == "B" else "ABC"
        exterior_point = None
    else:
        exterior_vertex = target_vertex
        if exterior_vertex == "D":
            angle_b = int(target_angle)
            known_vertex = "B"
            target_canonical_angle = "ADE"
            known_canonical_angle = "ABC"
        else:
            angle_b = int(180 - target_angle)
            known_vertex = "D"
            target_canonical_angle = "EBC"
            known_canonical_angle = "CDA"
        exterior_point = "E"

    arc_cd_plus_da = int(2 * angle_b)
    arc_ab_plus_bc = int(360 - arc_cd_plus_da)
    arc_ab, arc_bc = _split_cyclic_arc_sum(rng, arc_ab_plus_bc)
    arc_cd, arc_da = _split_cyclic_arc_sum(rng, arc_cd_plus_da)
    radius = float(rng.choice((10, 11, 12, 13, 14)))
    rotation = float(rng.choice((12, 28, 44, 60, 76, 92, 108)))
    center = (0.0, 0.0)
    angles = {
        "A": float(rotation),
        "B": float(rotation + arc_ab),
        "C": float(rotation + arc_ab + arc_bc),
        "D": float(rotation + arc_ab + arc_bc + arc_cd),
    }
    canonical_point_model = {
        "O": center,
        "A": _circle_point(radius, angles["A"]),
        "B": _circle_point(radius, angles["B"]),
        "C": _circle_point(radius, angles["C"]),
        "D": _circle_point(radius, angles["D"]),
    }
    if exterior_point == "E":
        extension_length = float(radius) * float(rng.choice((0.34, 0.42, 0.50)))
        if target_canonical_angle == "ADE":
            canonical_point_model["E"] = _extend_ray(
                canonical_point_model["C"],
                canonical_point_model["D"],
                distance=extension_length,
            )
        else:
            canonical_point_model["E"] = _extend_ray(
                canonical_point_model["A"],
                canonical_point_model["B"],
                distance=extension_length,
            )

    angle_abc = _angle_degrees_at(
        canonical_point_model["B"],
        canonical_point_model["A"],
        canonical_point_model["C"],
    )
    angle_cda = _angle_degrees_at(
        canonical_point_model["D"],
        canonical_point_model["C"],
        canonical_point_model["A"],
    )
    if abs(int(angle_abc) + int(angle_cda) - 180) > 1:
        raise ValueError("sampled cyclic quadrilateral angles are not supplementary")
    angle_by_canonical = {
        "ABC": int(angle_abc),
        "CDA": int(angle_cda),
        "DAB": _angle_degrees_at(
            canonical_point_model["A"],
            canonical_point_model["D"],
            canonical_point_model["B"],
        ),
        "BCD": _angle_degrees_at(
            canonical_point_model["C"],
            canonical_point_model["B"],
            canonical_point_model["D"],
        ),
    }
    if kind == "exterior":
        if target_canonical_angle == "ADE":
            angle_by_canonical["ADE"] = _angle_degrees_at(
                canonical_point_model["D"],
                canonical_point_model["A"],
                canonical_point_model["E"],
            )
        else:
            angle_by_canonical["EBC"] = _angle_degrees_at(
                canonical_point_model["B"],
                canonical_point_model["E"],
                canonical_point_model["C"],
            )
    if abs(int(angle_by_canonical[target_canonical_angle]) - int(target_angle)) > 1:
        raise ValueError("sampled cyclic quadrilateral target angle does not match")

    labels = ("O", "A", "B", "C", "D", "E") if exterior_point else ("O", "A", "B", "C", "D")
    label_map = _fixed_point_label_map(labels)
    point_model = {
        label_map[key]: value for key, value in canonical_point_model.items()
    }
    visible_angle_by_canonical = {
        "ABC": _visible_angle(label_map, "A", "B", "C"),
        "CDA": _visible_angle(label_map, "C", "D", "A"),
        "DAB": _visible_angle(label_map, "D", "A", "B"),
        "BCD": _visible_angle(label_map, "B", "C", "D"),
    }
    if exterior_point:
        visible_angle_by_canonical["ADE"] = _visible_angle(label_map, "A", "D", "E")
        visible_angle_by_canonical["EBC"] = _visible_angle(label_map, "E", "B", "C")
    answer_angle_name = str(visible_angle_by_canonical[target_canonical_angle])
    known_angle_name = str(visible_angle_by_canonical[known_canonical_angle])
    distractor_canonical_angle = "DAB" if target_canonical_angle != "DAB" else "BCD"
    distractor_angle_name = str(visible_angle_by_canonical[distractor_canonical_angle])
    answer_token = f"{answer_angle_name}=?"
    known_token = f"{known_angle_name}={int(angle_by_canonical[known_canonical_angle])}"
    distractor_token = (
        f"{distractor_angle_name}={int(angle_by_canonical[distractor_canonical_angle])}"
    )
    angle_marker_specs = (
        {
            "token": known_token,
            "vertex": label_map[known_canonical_angle[1]],
            "arm0": label_map[known_canonical_angle[0]],
            "arm1": label_map[known_canonical_angle[2]],
            "radius_px": 42.0,
        },
        {
            "token": answer_token,
            "vertex": label_map[target_canonical_angle[1]],
            "arm0": label_map[target_canonical_angle[0]],
            "arm1": label_map[target_canonical_angle[2]],
            "radius_px": 42.0,
        },
        {
            "token": distractor_token,
            "vertex": label_map[distractor_canonical_angle[1]],
            "arm0": label_map[distractor_canonical_angle[0]],
            "arm1": label_map[distractor_canonical_angle[2]],
            "radius_px": 34.0,
        },
    )
    segments = {
        "AB": (label_map["A"], label_map["B"]),
        "BC": (label_map["B"], label_map["C"]),
        "CD": (label_map["C"], label_map["D"]),
        "DA": (label_map["D"], label_map["A"]),
    }
    if exterior_point:
        if target_canonical_angle == "ADE":
            segments["DE"] = (label_map["D"], label_map["E"])
        else:
            segments["BE"] = (label_map["B"], label_map["E"])
    annotation_point_labels = (
        (label_map["A"], label_map["B"], label_map["C"], label_map["D"], label_map["E"])
        if exterior_point
        else (label_map["A"], label_map["B"], label_map["C"], label_map["D"])
    )
    theorem_trace = {
        "theorem": "cyclic_quadrilateral_angle",
        "label_map": dict(label_map),
        "angle_ABC": int(angle_by_canonical["ABC"]),
        "angle_CDA": int(angle_by_canonical["CDA"]),
        "angle_DAB": int(angle_by_canonical["DAB"]),
        "angle_BCD": int(angle_by_canonical["BCD"]),
        "target_vertex": str(target_vertex),
        "known_vertex": str(known_vertex),
        "canonical_known_angle": str("angle" + known_canonical_angle),
        "known_angle": str(known_angle_name),
        "canonical_answer_segment": str("angle" + target_canonical_angle),
        "answer_segment": str(answer_angle_name),
        "answer_value": int(target_angle),
        "opposite_angle_sum": 180,
        "distractor_tokens": [str(distractor_token)],
    }
    if exterior_point:
        theorem_trace["exterior_vertex"] = str(target_vertex)
        theorem_trace[f"angle_{target_canonical_angle}"] = int(
            angle_by_canonical[target_canonical_angle]
        )
        theorem_trace["extension_point"] = str(label_map["E"])
    return {
        "point_model": point_model,
        "circle_center": center,
        "circle_radius": float(radius),
        "segments": segments,
        "measurement_specs": tuple(),
        "angle_marker_specs": angle_marker_specs,
        "circle_arc_specs": tuple(),
        "support_measurement_tokens": (known_token,),
        "annotation_point_labels": annotation_point_labels,
        "annotation_values": {
            str(answer_angle_name): int(target_angle),
            str(known_angle_name): int(angle_by_canonical[known_canonical_angle]),
            str(distractor_angle_name): int(angle_by_canonical[distractor_canonical_angle]),
        },
        "theorem_trace": theorem_trace,
        "prompt_slots": {
            "known_angle": str(known_angle_name),
            "answer_angle": str(answer_angle_name),
            "extension_point": "" if not exterior_point else str(label_map["E"]),
            "answer_segment": str(answer_angle_name),
        },
    }


def _build_cyclic_opposite_angle_scene(rng, *, problem: CircleTheoremProblem) -> Dict[str, Any]:
    """Build the branch where opposite cyclic angles are supplementary."""

    return _build_cyclic_quadrilateral_angle_scene(rng, problem=problem, target_kind="opposite")


def _build_cyclic_exterior_angle_scene(rng, *, problem: CircleTheoremProblem) -> Dict[str, Any]:
    """Build the branch where a cyclic exterior angle is transferred."""

    return _build_cyclic_quadrilateral_angle_scene(rng, problem=problem, target_kind="exterior")

__all__ = [
    "_build_central_angle_from_inscribed_scene",
    "_build_cyclic_exterior_angle_scene",
    "_build_cyclic_opposite_angle_scene",
    "_build_diameter_perpendicular_chord_scene",
    "_build_external_secant_angle_scene",
    "_build_inscribed_angle_from_arc_scene",
    "_build_inscribed_angle_from_central_scene",
    "_build_intersecting_chords_arc_scene",
    "_build_multi_step_angle_scene",
    "_build_secant_secant_scene",
    "_build_secant_secant_variable_scene",
    "_build_tangent_chord_angle_from_arc_scene",
    "_build_tangent_chord_angle_from_inscribed_scene",
    "_build_tangent_secant_scene",
]


# Tangent-radius right-triangle construction helpers.
SCENE_ID = "circle_theorem"


def _default_tangent_radius_triples() -> Tuple[Tuple[int, int, int], ...]:
    """Return radius-tangent-external triples with broad default support."""

    triples: list[Tuple[int, int, int]] = []
    used_radii: set[int] = set()
    used_tangent_lengths: set[int] = set()
    used_external_distances: set[int] = set()
    for triangle in integer_right_triangles(
        min_leg=3,
        max_leg=320,
        max_hypotenuse=420,
    ):
        radius = min(int(triangle.leg_a), int(triangle.leg_b))
        tangent = max(int(triangle.leg_a), int(triangle.leg_b))
        external = int(triangle.hypotenuse)
        if radius in used_radii or tangent in used_tangent_lengths:
            continue
        if external in used_external_distances:
            continue
        triples.append((radius, tangent, external))
        used_radii.add(radius)
        used_tangent_lengths.add(tangent)
        used_external_distances.add(external)
    if len(triples) < 80:
        raise RuntimeError("tangent-radius triple pool is unexpectedly small")
    return tuple(triples)


_DEFAULT_TRIPLES: Tuple[Tuple[int, int, int], ...] = _default_tangent_radius_triples()
_DEFAULT_EXTERNAL_DISTANCE_SUPPORT: Tuple[int, ...] = tuple(range(10, 71))
_DEFAULT_ANGLE_SUPPORT: Tuple[int, ...] = tuple(range(25, 71))


@dataclass(frozen=True)
class _ResolvedTangentRadiusQuery:
    radius_value: float
    tangent_length: float
    external_distance: float
    angle_degrees: int | None
    answer_value: float
    answer_segment_canonical: str
    support_probabilities: Dict[str, float]


def _round1(value: float) -> float:
    return float(round(float(value) + 1e-9, 1))


def _format_measurement(value: float) -> str:
    rounded = _round1(float(value))
    if math.isclose(rounded, round(rounded), abs_tol=1e-9):
        return str(int(round(rounded)))
    return f"{rounded:.1f}"


def _support_triples(*, generation_defaults: Mapping[str, Any]) -> Tuple[Tuple[int, int, int], ...]:
    raw = generation_defaults.get("tangent_radius_right_triangle_triples", _DEFAULT_TRIPLES)
    triples = tuple(tuple(int(part) for part in triple) for triple in raw)
    if not triples:
        raise ValueError(
            "tangent_radius_right_triangle_triples must contain at least one triple"
        )
    for triple in triples:
        if len(triple) != 3:
            raise ValueError(f"invalid tangent-radius triple: {triple!r}")
        radius, tangent, external = triple
        try:
            validate_integer_right_triangle(
                IntegerRightTriangle(
                    leg_a=int(radius),
                    leg_b=int(tangent),
                    hypotenuse=int(external),
                )
            )
        except ValueError as exc:
            raise ValueError(f"invalid tangent-radius triple: {triple!r}") from exc
    return triples


def _select_int(
    *,
    rng,
    params: Mapping[str, Any],
    explicit_keys: Sequence[str],
    support: Sequence[int],
) -> Tuple[int, Dict[str, float]]:
    values = tuple(int(value) for value in support)
    for key in explicit_keys:
        if key in params:
            selected = int(params[str(key)])
            if selected not in set(values):
                raise ValueError(f"{key}={selected!r} is not in support {values!r}")
            return selected, {str(selected): 1.0}
    return int(rng.choice(values)), _probability_map(values)


def resolve_tangent_length_from_radius_and_external_distance(
    instance_seed: int,
    *,
    rng_namespace: str,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> _ResolvedTangentRadiusQuery:
    """Resolve the Pythagorean tangent-length task inputs."""

    rng = spawn_rng(int(instance_seed), f"{rng_namespace}.query")
    triples = _support_triples(generation_defaults=generation_defaults)
    explicit_radius = params.get("radius_value", params.get("radius"))
    explicit_external = params.get(
        "external_distance", params.get("center_distance")
    )
    if explicit_radius is not None and explicit_external is not None:
        radius = float(explicit_radius)
        external = float(explicit_external)
        if external <= radius:
            raise ValueError("external_distance must be larger than radius_value")
        tangent = math.sqrt((external * external) - (radius * radius))
        support_key = "-".join(
            (
                _format_measurement(radius),
                _format_measurement(tangent),
                _format_measurement(external),
            )
        )
        support_probabilities = {support_key: 1.0}
    else:
        radius, tangent, external = triples[int(rng.randrange(len(triples)))]
        support_probabilities = _probability_map(
            tuple(f"{r}-{t}-{d}" for r, t, d in triples)
        )
    return _ResolvedTangentRadiusQuery(
        radius_value=float(radius),
        tangent_length=float(tangent),
        external_distance=float(external),
        angle_degrees=None,
        answer_value=_round1(float(tangent)),
        answer_segment_canonical="PT",
        support_probabilities=dict(support_probabilities),
    )


def resolve_radius_from_external_distance_and_angle(
    instance_seed: int,
    *,
    rng_namespace: str,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> _ResolvedTangentRadiusQuery:
    """Resolve the trigonometric radius task inputs."""

    rng = spawn_rng(int(instance_seed), f"{rng_namespace}.query")

    external, external_probabilities = _select_int(
        rng=rng,
        params=params,
        explicit_keys=("external_distance", "center_distance"),
        support=tuple(int(value) for value in generation_defaults.get(
            "tangent_radius_external_distance_support",
            _DEFAULT_EXTERNAL_DISTANCE_SUPPORT,
        )),
    )
    angle, angle_probabilities = _select_int(
        rng=rng,
        params=params,
        explicit_keys=("angle_degrees", "angle_value"),
        support=tuple(int(value) for value in generation_defaults.get("tangent_radius_angle_support", _DEFAULT_ANGLE_SUPPORT)),
    )
    radius = float(external) * math.sin(math.radians(float(angle)))
    tangent = float(external) * math.cos(math.radians(float(angle)))
    support_probabilities = {
        f"external_distance:{external_key}|angle_degrees:{angle_key}": float(
            external_probability
        )
        * float(angle_probability)
        for external_key, external_probability in external_probabilities.items()
        for angle_key, angle_probability in angle_probabilities.items()
    }
    return _ResolvedTangentRadiusQuery(
        radius_value=float(radius),
        tangent_length=float(tangent),
        external_distance=float(external),
        angle_degrees=int(angle),
        answer_value=_round1(radius),
        answer_segment_canonical="OT",
        support_probabilities=dict(support_probabilities),
    )


def _rotate(point: Point, degrees: float) -> Point:
    radians = math.radians(float(degrees))
    c = math.cos(radians)
    s = math.sin(radians)
    x, y = float(point[0]), float(point[1])
    return (float((x * c) - (y * s)), float((x * s) + (y * c)))


def build_tangent_radius_payload(
    rng, *, query: _ResolvedTangentRadiusQuery
) -> Dict[str, Any]:
    """Construct the tangent-radius diagram with the right angle and role witnesses aligned."""
    radius = float(query.radius_value)

    tangent = float(query.tangent_length)
    side = float(rng.choice((-1, 1)))
    rotation = float(rng.choice((15, 30, 45, 60, 75, 105, 120, 135, 150)))
    canonical_points = {
        "O": _rotate((0.0, 0.0), rotation),
        "T": _rotate((radius, 0.0), rotation),
        "P": _rotate((radius, side * tangent), rotation),
    }
    label_map = _fixed_point_label_map(("O", "P", "T"))
    point_model = {label_map[key]: value for key, value in canonical_points.items()}

    radius_segment = _visible_segment(label_map, "O", "T")
    tangent_segment = _visible_segment(label_map, "P", "T")
    external_segment = _visible_segment(label_map, "O", "P")
    answer_segment = (
        radius_segment
        if query.answer_segment_canonical == "OT"
        else tangent_segment
    )
    support_measurement_tokens = []
    measurement_specs = []
    annotation_values: Dict[str, int] = {}
    if query.answer_segment_canonical == "OT":
        external_token = f"{external_segment}={_format_measurement(query.external_distance)}"
        answer_token = f"{radius_segment}=?"
        angle_name = _visible_angle(label_map, "O", "P", "T")
        angle_token = f"{angle_name}={int(query.angle_degrees or 0)}°"
        support_measurement_tokens.extend((external_token, angle_token))
        measurement_specs.extend(
            (
                (external_token, "OP", 1.0),
                (answer_token, "OT", -1.0),
            )
        )
        angle_marker_specs = (
            {
                "token": angle_token,
                "vertex": label_map["P"],
                "arm0": label_map["O"],
                "arm1": label_map["T"],
                "radius_px": 42.0,
            },
        )
        annotation_values[str(external_segment)] = int(round(query.external_distance))
        annotation_values[str(angle_name)] = int(query.angle_degrees or 0)
    else:
        radius_token = f"{radius_segment}={_format_measurement(query.radius_value)}"
        external_token = f"{external_segment}={_format_measurement(query.external_distance)}"
        answer_token = f"{tangent_segment}=?"
        support_measurement_tokens.extend((radius_token, external_token))
        measurement_specs.extend(
            (
                (radius_token, "OT", 1.0),
                (external_token, "OP", -1.0),
                (answer_token, "PT", 1.0),
            )
        )
        angle_marker_specs = tuple()
        annotation_values[str(radius_segment)] = int(round(query.radius_value))
        annotation_values[str(external_segment)] = int(round(query.external_distance))

    segments = {
        "OT": (label_map["O"], label_map["T"]),
        "PT": (label_map["P"], label_map["T"]),
        "OP": (label_map["O"], label_map["P"]),
    }
    annotation_point_labels = (label_map["O"], label_map["T"], label_map["P"])
    theorem_trace = {
        "theorem": "tangent_radius_right_triangle",
        "label_map": dict(label_map),
        "canonical_answer_segment": str(query.answer_segment_canonical),
        "answer_segment": str(answer_segment),
        "answer_value": float(query.answer_value),
        "answer_rounding": "one_decimal",
        "radius_value": float(query.radius_value),
        "tangent_length": float(query.tangent_length),
        "external_distance": float(query.external_distance),
        "angle_degrees": None
        if query.angle_degrees is None
        else int(query.angle_degrees),
        "right_angle_vertex": str(label_map["T"]),
        "formula": "OT=OP*sin(angle_OPT)"
        if query.angle_degrees is not None
        else "OT^2+PT^2=OP^2",
        "distractor_tokens": [],
    }
    return {
        "point_model": point_model,
        "circle_center": canonical_points["O"],
        "circle_radius": float(radius),
        "segments": segments,
        "measurement_specs": tuple(measurement_specs),
        "support_measurement_tokens": tuple(support_measurement_tokens),
        "annotation_point_labels": tuple(annotation_point_labels),
        "annotation_values": dict(annotation_values),
        "theorem_trace": theorem_trace,
        "angle_marker_specs": angle_marker_specs,
        "right_angle_marker_specs": (
            {
                "vertex": label_map["T"],
                "arm0": label_map["O"],
                "arm1": label_map["P"],
                "size_px": 24.0,
            },
        ),
        "circle_arc_specs": tuple(),
        "prompt_slots": {
            "radius_segment": str(radius_segment),
            "tangent_segment": str(tangent_segment),
            "external_distance_segment": str(external_segment),
            "answer_segment": str(answer_segment),
            "angle_name": ""
            if query.angle_degrees is None
            else str(_visible_angle(label_map, "O", "P", "T")),
        },
    }
