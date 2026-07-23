"""Contract tests for the geometry circle-theorem value task."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.geometry.circle_theorem.cyclic_quadrilateral_exterior_angle_value import (
    GeometryCircleCyclicQuadrilateralExteriorAngleValueTask,
)
from trace_tasks.tasks.geometry.circle_theorem.cyclic_quadrilateral_opposite_angle_value import (
    GeometryCircleCyclicQuadrilateralOppositeAngleValueTask,
)
from trace_tasks.tasks.geometry.circle_theorem.diameter_perpendicular_chord_length_value import GeometryCircleDiameterPerpendicularChordLengthValueTask
from trace_tasks.tasks.geometry.circle_theorem.external_secant_angle_value import GeometryCircleExternalSecantAngleValueTask
from trace_tasks.tasks.geometry.circle_theorem.chord_length_from_radius_central_angle_value import GeometryCircleChordLengthFromRadiusCentralAngleValueTask
from trace_tasks.tasks.geometry.circle_theorem.chord_length_from_radius_inscribed_angle_value import GeometryCircleChordLengthFromRadiusInscribedAngleValueTask
from trace_tasks.tasks.geometry.circle_theorem.inscribed_central_angle_value import GeometryCircleInscribedCentralAngleValueTask
from trace_tasks.tasks.geometry.circle_theorem.inscribed_angle_value_inscribed_angle_from_arc import GeometryCircleInscribedAngleFromArcTask
from trace_tasks.tasks.geometry.circle_theorem.intersecting_chords_arc_measure_value import GeometryCircleIntersectingChordsArcMeasureValueTask
from trace_tasks.tasks.geometry.circle_theorem.multi_step_angle_value import GeometryCircleMultiStepAngleValueTask
from trace_tasks.tasks.geometry.circle_theorem.secant_secant_length_value import GeometryCircleSecantSecantLengthValueTask
from trace_tasks.tasks.geometry.circle_theorem.tangent_chord_angle_value_tangent_chord_angle_from_arc import GeometryCircleTangentChordAngleFromArcTask
from trace_tasks.tasks.geometry.circle_theorem.tangent_chord_angle_value_tangent_chord_angle_from_inscribed import GeometryCircleTangentChordAngleFromInscribedTask
from trace_tasks.tasks.geometry.circle_theorem.radius_from_external_distance_and_angle_value import (
    GeometryCircleRadiusFromExternalDistanceAndAngleValueTask,
)
from trace_tasks.tasks.geometry.circle_theorem.shared.construction import _DEFAULT_TRIPLES
from trace_tasks.tasks.geometry.circle_theorem.tangent_length_from_radius_and_external_distance_value import (
    GeometryCircleTangentLengthFromRadiusAndExternalDistanceValueTask,
)
from trace_tasks.tasks.geometry.circle_theorem.tangent_secant_length_value import GeometryCircleTangentSecantLengthValueTask
from trace_tasks.tasks.geometry.circle_theorem.shared.state import (
    CENTRAL_ANGLE_ANSWER_SUPPORT,
    CYCLIC_QUADRILATERAL_ANGLE_SUPPORT,
    DIAMETER_CHORD_ANSWER_SUPPORT,
    EXTERNAL_SECANT_ANGLE_ANSWER_SUPPORT,
    INSCRIBED_ANGLE_ANSWER_SUPPORT,
    INTERSECTING_CHORDS_ARC_ANSWER_SUPPORT,
    MULTI_STEP_ANGLE_ANSWER_SUPPORT,
    SECANT_SECANT_ANSWER_SUPPORT,
    TANGENT_CHORD_ANGLE_ANSWER_SUPPORT,
)
from trace_tasks.tasks.shared.config_defaults import (
    split_generation_rendering_prompt_defaults,
)


def _point_in_bbox(point: list[float] | tuple[float, float], bbox: list[float]) -> bool:
    return bool(
        float(bbox[0]) <= float(point[0]) <= float(bbox[2])
        and float(bbox[1]) <= float(point[1]) <= float(bbox[3])
    )


def _orientation(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    return bool(
        (_orientation(a, b, c) * _orientation(a, b, d) < 0.0)
        and (_orientation(c, d, a) * _orientation(c, d, b) < 0.0)
    )


def _segment_crosses_bbox(
    a: tuple[float, float], b: tuple[float, float], bbox: list[float]
) -> bool:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    if _point_in_bbox(a, bbox) or _point_in_bbox(b, bbox):
        return True
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return any(
        _segments_intersect(a, b, c, d)
        for c, d in zip(corners, corners[1:] + corners[:1])
    )


def _circle_crosses_bbox(center: list[float], radius: float, bbox: list[float]) -> bool:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    cx, cy = float(center[0]), float(center[1])
    closest = (min(max(cx, x0), x1), min(max(cy, y0), y1))
    nearest = math.hypot(float(closest[0]) - cx, float(closest[1]) - cy)
    farthest = max(
        math.hypot(float(x) - cx, float(y) - cy) for x in (x0, x1) for y in (y0, y1)
    )
    return bool(float(nearest) <= float(radius) <= float(farthest))


def _assert_arc_measure_tokens_use_degrees(out) -> None:
    render_tokens = [
        str(token)
        for token in out.trace_payload["render_map"]["measurement_token_bboxes"]
        if str(token).startswith("arc ")
    ]
    trace = out.trace_payload["execution_trace"]
    trace_tokens = [
        str(token)
        for token in [
            *trace.get("support_measurement_tokens", []),
            *trace.get("distractor_tokens", []),
        ]
        if str(token).startswith("arc ")
    ]
    arc_tokens = render_tokens + trace_tokens

    assert arc_tokens
    assert all(token.endswith("°") for token in arc_tokens)
    assert not any(token.endswith("=?") for token in arc_tokens)


QUERY_TASK_CLASSES = {
    "chord_length_from_radius_and_central_angle": GeometryCircleChordLengthFromRadiusCentralAngleValueTask,
    "chord_length_from_radius_and_inscribed_angle": GeometryCircleChordLengthFromRadiusInscribedAngleValueTask,
    "diameter_perpendicular_chord_length": GeometryCircleDiameterPerpendicularChordLengthValueTask,
    "secant_secant_variable_segment_length": GeometryCircleSecantSecantLengthValueTask,
    "tangent_secant_length": GeometryCircleTangentSecantLengthValueTask,
    "secant_secant_length": GeometryCircleSecantSecantLengthValueTask,
    "intersecting_chords_arc_measure": GeometryCircleIntersectingChordsArcMeasureValueTask,
    "multi_step_angle_value": GeometryCircleMultiStepAngleValueTask,
    "inscribed_angle_from_central": GeometryCircleInscribedCentralAngleValueTask,
    "central_angle_from_inscribed": GeometryCircleInscribedCentralAngleValueTask,
    "inscribed_angle_from_arc": GeometryCircleInscribedAngleFromArcTask,
    "tangent_chord_angle_from_arc": GeometryCircleTangentChordAngleFromArcTask,
    "tangent_chord_angle_from_inscribed": GeometryCircleTangentChordAngleFromInscribedTask,
    "external_two_secants_angle_from_arcs": GeometryCircleExternalSecantAngleValueTask,
    "opposite_angle_supplement": GeometryCircleCyclicQuadrilateralOppositeAngleValueTask,
    "exterior_angle_from_opposite_interior": GeometryCircleCyclicQuadrilateralExteriorAngleValueTask,
    "radius_from_external_distance_and_angle": GeometryCircleRadiusFromExternalDistanceAndAngleValueTask,
    "tangent_length_from_radius_and_external_distance": GeometryCircleTangentLengthFromRadiusAndExternalDistanceValueTask,
}


def _task_for_query(query_id: str):
    return QUERY_TASK_CLASSES[str(query_id)]()


def _public_query_id_for_query(query_id: str) -> str:
    task_cls = QUERY_TASK_CLASSES[str(query_id)]
    if tuple(getattr(task_cls, "supported_query_ids", ())) == ("single",):
        return "single"
    return str(query_id)


def _public_params_for_query(query_id: str, params: dict[str, object]) -> dict[str, object]:
    resolved = dict(params)
    if _public_query_id_for_query(str(query_id)) == "single":
        resolved.pop("query_id", None)
        resolved.pop("query_variant", None)
    else:
        resolved["query_id"] = str(query_id)
    return resolved


def _generate_for_query(
    query_id: str,
    seed: int,
    *,
    params: dict[str, object] | None = None,
    max_attempts: int = 40,
):
    task = _task_for_query(str(query_id))
    task_params = _public_params_for_query(
        str(query_id),
        {"query_id": str(query_id), **dict(params or {})},
    )
    return task.generate(int(seed), params=task_params, max_attempts=int(max_attempts))


def test_circle_theorem_tangent_radius_default_pool_has_broad_support() -> None:
    assert len(_DEFAULT_TRIPLES) >= 80

    radii = [radius for radius, _tangent, _external in _DEFAULT_TRIPLES]
    tangent_lengths = [tangent for _radius, tangent, _external in _DEFAULT_TRIPLES]
    external_distances = [external for _radius, _tangent, external in _DEFAULT_TRIPLES]

    assert len(set(radii)) == len(radii)
    assert len(set(tangent_lengths)) == len(tangent_lengths)
    assert len(set(external_distances)) == len(external_distances)
    for radius, tangent, external in _DEFAULT_TRIPLES:
        assert int(radius) ** 2 + int(tangent) ** 2 == int(external) ** 2


def test_circle_theorem_answer_supports_are_broad() -> None:
    """Guard against reverting theorem answers to tiny review-frequency pools."""

    assert len(DIAMETER_CHORD_ANSWER_SUPPORT) >= 50
    assert len(SECANT_SECANT_ANSWER_SUPPORT) >= 40
    assert len(INSCRIBED_ANGLE_ANSWER_SUPPORT) >= 50
    assert len(CENTRAL_ANGLE_ANSWER_SUPPORT) >= 50
    assert len(TANGENT_CHORD_ANGLE_ANSWER_SUPPORT) >= 50
    assert len(EXTERNAL_SECANT_ANGLE_ANSWER_SUPPORT) >= 50
    assert len(CYCLIC_QUADRILATERAL_ANGLE_SUPPORT) >= 80
    assert len(INTERSECTING_CHORDS_ARC_ANSWER_SUPPORT) >= 100
    assert len(MULTI_STEP_ANGLE_ANSWER_SUPPORT) >= 80


@pytest.mark.parametrize(
    "query_id,expected_keys",
    (
        (
            "chord_length_from_radius_and_central_angle",
            ("O", "A", "B"),
        ),
        (
            "chord_length_from_radius_and_inscribed_angle",
            ("O", "A", "B", "C"),
        ),
    ),
)
def test_circle_chord_length_from_radius_angle_contract(
    query_id: str, expected_keys: tuple[str, ...]
) -> None:
    out = _generate_for_query(
        query_id,
        29031,
        params={"query_id": query_id, "radius_value": 10, "angle_degrees": 60},
        max_attempts=40,
    )

    execution = out.trace_payload["execution_trace"]
    expected_central = 60 if query_id.endswith("central_angle") else 120
    expected_answer = round(
        2.0
        * float(execution["radius_value"])
        * math.sin(math.radians(float(expected_central) / 2.0)),
        1,
    )
    assert out.answer_gt.type == "number"
    assert float(out.answer_gt.value) == pytest.approx(expected_answer)
    assert execution["answer_type"] == "number"
    assert execution["answer_rounding"] == "one_decimal"
    assert execution["central_angle_degrees"] == expected_central
    assert execution["answer_value"] == pytest.approx(expected_answer)
    assert out.annotation_gt.type == "point_map"
    assert tuple(out.annotation_gt.value) == tuple(expected_keys)
    assert set(out.annotation_gt.value) == set(execution["annotation_point_labels"])
    assert out.trace_payload["projected_annotation"]["type"] == "point_map"
    assert (
        out.trace_payload["projected_annotation"]["point_map"]
        == out.annotation_gt.value
    )
    assert "visible point-label keys" in out.prompt
    for key in expected_keys:
        assert f'"{key}"' in out.prompt
    assert "task_variant" not in str(out.trace_payload)


@pytest.mark.parametrize(
    (
        "params",
        "expected_answer",
        "expected_canonical_segment",
        "expected_angle",
    ),
    (
        (
            {
                "query_id": "tangent_length_from_radius_and_external_distance",
                "radius_value": 5,
                "external_distance": 13,
            },
            12.0,
            "PT",
            None,
        ),
        (
            {
                "query_id": "radius_from_external_distance_and_angle",
                "external_distance": 10,
                "angle_degrees": 30,
            },
            5.0,
            "OT",
            30,
        ),
    ),
)
def test_circle_tangent_radius_tasks_contract(
    params: dict[str, object],
    expected_answer: float,
    expected_canonical_segment: str,
    expected_angle: int | None,
) -> None:
    out = _generate_for_query(
        str(params["query_id"]),
        29137,
        params=params,
        max_attempts=40,
    )
    execution = out.trace_payload["execution_trace"]
    point_model = out.trace_payload["render_map"]["point_model"]
    label_map = execution["label_map"]
    ox, oy = point_model[label_map["O"]]
    tx, ty = point_model[label_map["T"]]
    px, py = point_model[label_map["P"]]

    radius = math.hypot(float(tx) - float(ox), float(ty) - float(oy))
    tangent = math.hypot(float(px) - float(tx), float(py) - float(ty))
    external = math.hypot(float(px) - float(ox), float(py) - float(oy))
    dot_product = ((float(ox) - float(tx)) * (float(px) - float(tx))) + (
        (float(oy) - float(ty)) * (float(py) - float(ty))
    )

    assert out.answer_gt.type == "number"
    assert float(out.answer_gt.value) == pytest.approx(expected_answer)
    assert execution["answer_type"] == "number"
    assert execution["answer_rounding"] == "one_decimal"
    assert execution["canonical_answer_segment"] == expected_canonical_segment
    assert execution["answer_value"] == pytest.approx(expected_answer)
    assert execution["angle_degrees"] == expected_angle
    assert radius == pytest.approx(float(execution["radius_value"]))
    assert tangent == pytest.approx(float(execution["tangent_length"]))
    assert external == pytest.approx(float(execution["external_distance"]))
    assert dot_product == pytest.approx(0.0, abs=1e-7)
    assert (radius * radius) + (tangent * tangent) == pytest.approx(
        external * external
    )
    assert out.annotation_gt.type == "point_map"
    assert tuple(out.annotation_gt.value) == ("O", "T", "P")
    assert set(out.annotation_gt.value) == set(execution["annotation_point_labels"])
    assert out.trace_payload["projected_annotation"]["type"] == "point_map"
    assert (
        out.trace_payload["projected_annotation"]["point_map"]
        == out.annotation_gt.value
    )
    assert "visible point-label keys" in out.prompt
    for key in ("O", "T", "P"):
        assert f'"{key}"' in out.prompt
    assert "task_variant" not in str(out.trace_payload)


@pytest.mark.parametrize(
    (
        "params",
        "expected_answer",
        "expected_annotation_count",
        "expected_canonical_segment",
    ),
    (
        (
            {"query_id": "diameter_perpendicular_chord_length", "target_answer": 8},
            8,
            5,
            "BE",
        ),
        (
            {
                "query_id": "secant_secant_variable_segment_length",
                "target_answer": 24,
                "secant_secant_variable_target_kind": "inside_first",
            },
            24,
            5,
            "AB",
        ),
        (
            {
                "query_id": "tangent_secant_length",
                "target_answer": 24,
                "tangent_secant_target_kind": "outside",
            },
            24,
            4,
            "PA",
        ),
        (
            {"query_id": "secant_secant_length", "target_answer": 10},
            10,
            5,
            "PA",
        ),
        (
            {"query_id": "intersecting_chords_arc_measure", "target_answer": 120},
            120,
            5,
            "arcCD",
        ),
        (
            {"query_id": "multi_step_angle_value", "target_answer": 85},
            85,
            5,
            "angleAEB",
        ),
        (
            {"query_id": "inscribed_angle_from_central", "target_answer": 35},
            35,
            4,
            "angleACB",
        ),
        (
            {"query_id": "central_angle_from_inscribed", "target_answer": 70},
            70,
            4,
            "angleAOB",
        ),
        (
            {"query_id": "inscribed_angle_from_arc", "target_answer": 35},
            35,
            3,
            "angleACB",
        ),
        (
            {"query_id": "tangent_chord_angle_from_arc", "target_answer": 45},
            45,
            3,
            "anglePTA",
        ),
        (
            {"query_id": "tangent_chord_angle_from_inscribed", "target_answer": 45},
            45,
            4,
            "anglePTA",
        ),
        (
            {"query_id": "external_two_secants_angle_from_arcs", "target_answer": 50},
            50,
            5,
            "angleBPD",
        ),
        (
            {"query_id": "opposite_angle_supplement", "target_answer": 75},
            75,
            4,
            "angleCDA",
        ),
        (
            {"query_id": "exterior_angle_from_opposite_interior", "target_answer": 75},
            75,
            5,
            "angleADE",
        ),
    ),
)
def test_geometry_circle_theorem_value_emits_expected_contract(
    params: dict[str, int | str],
    expected_answer: int,
    expected_annotation_count: int,
    expected_canonical_segment: str,
) -> None:
    requested_query_id = str(params["query_id"])
    out = _generate_for_query(
        requested_query_id,
        23401,
        params=params,
        max_attempts=40,
    )

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == int(expected_answer)
    assert out.annotation_gt.type == "point_map"
    assert len(out.annotation_gt.value) == int(expected_annotation_count)
    assert (
        out.trace_payload["execution_trace"]["canonical_answer_segment"]
        == expected_canonical_segment
    )
    assert out.trace_payload["execution_trace"]["target_answer"] == int(expected_answer)
    render_spec = out.trace_payload["render_spec"]
    assert render_spec["font_bold"] is False
    assert int(render_spec["label_stroke_width"]) == 0
    assert int(render_spec["measurement_stroke_width"]) == 0
    expected_public_query_id = _public_query_id_for_query(requested_query_id)
    query_params = out.trace_payload["query_spec"]["params"]
    assert query_params["query_id"] == expected_public_query_id
    if expected_public_query_id == "single":
        assert query_params["internal_query_id"] == requested_query_id
    assert len(out.trace_payload["execution_trace"]["distractor_tokens"]) >= 1
    support_measurement_tokens = set(
        out.trace_payload["witness_symbolic"]["support_measurement_tokens"]
    )
    annotation_point_labels = set(
        out.trace_payload["witness_symbolic"]["annotation_point_labels"]
    )
    assert not (
        set(out.trace_payload["execution_trace"]["distractor_tokens"])
        & support_measurement_tokens
    )
    assert set(out.annotation_gt.value) == annotation_point_labels
    assert annotation_point_labels <= set(out.trace_payload["render_map"]["point_pixels"])
    assert "exactly these visible point-label keys" in out.prompt
    for label in annotation_point_labels:
        assert f'"{label}"' in out.prompt
    all_tokens = set(out.trace_payload["render_map"]["measurement_token_bboxes"])
    assert not any(str(token).startswith("angle") for token in all_tokens)
    assert any(str(token).startswith("∠") for token in all_tokens)
    for token in out.trace_payload["execution_trace"]["distractor_tokens"]:
        assert token in out.trace_payload["render_map"]["measurement_token_bboxes"]

    projected = out.trace_payload["projected_annotation"]
    assert projected["type"] == "point_map"
    assert projected["point_map"] == out.annotation_gt.value
    assert projected["pixel_point_map"] == out.annotation_gt.value
    assert len(projected["point_set"]) == int(expected_annotation_count)
    assert all(
        isinstance(point, list) and len(point) == 2 for point in projected["point_set"]
    )
    assert all(
        isinstance(point, list) and len(point) == 2
        for point in out.annotation_gt.value.values()
    )
    assert all("=" in str(token) for token in support_measurement_tokens)


def test_geometry_circle_theorem_value_is_deterministic() -> None:
    params = {
        "query_id": "secant_secant_variable_segment_length",
        "target_answer": 24,
        "secant_secant_variable_target_kind": "inside_first",
    }
    task = _task_for_query(str(params["query_id"]))

    out_a = task.generate(23411, params=params, max_attempts=40)
    out_b = task.generate(23411, params=params, max_attempts=40)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_geometry_circle_theorem_uses_fixed_visible_point_labels() -> None:
    variants = (
        "diameter_perpendicular_chord_length",
        "secant_secant_variable_segment_length",
        "tangent_secant_length",
        "secant_secant_length",
        "intersecting_chords_arc_measure",
        "multi_step_angle_value",
        "inscribed_angle_from_central",
        "central_angle_from_inscribed",
        "inscribed_angle_from_arc",
        "tangent_chord_angle_from_arc",
        "tangent_chord_angle_from_inscribed",
        "external_two_secants_angle_from_arcs",
        "opposite_angle_supplement",
        "exterior_angle_from_opposite_interior",
    )

    for seed in range(23450, 23460):
        for query_id in variants:
            out = _generate_for_query(query_id, seed, max_attempts=100)
            label_map = out.trace_payload["execution_trace"]["label_map"]

            assert label_map == {label: label for label in label_map}


def test_geometry_circle_theorem_value_rejects_unsupported_variant() -> None:
    with pytest.raises(ValueError):
        GeometryCircleDiameterPerpendicularChordLengthValueTask().generate(
            23421,
            params={"query_id": "inscribed_angle_measure", "target_answer": 8},
            max_attempts=20,
        )


def test_tangent_secant_variant_places_tangent_point_on_circle() -> None:
    out = _generate_for_query(
        "tangent_secant_length",
        23431,
        params={
            "query_id": "tangent_secant_length",
            "target_answer": 24,
            "tangent_secant_target_kind": "outside",
        },
        max_attempts=40,
    )
    point_model = out.trace_payload["render_map"]["point_model"]
    trace = out.trace_payload["execution_trace"]
    label_map = trace["label_map"]

    px, py = point_model[label_map["P"]]
    tx, ty = point_model[label_map["T"]]
    ox, oy = point_model[label_map["O"]]
    radius = float(out.trace_payload["render_map"]["circle_radius_model"])

    assert math.isclose(
        math.hypot(float(tx) - float(ox), float(ty) - float(oy)), radius, abs_tol=1e-7
    )
    assert math.isclose(
        math.hypot(float(tx) - float(px), float(ty) - float(py)),
        float(trace["PT"]),
        abs_tol=1e-7,
    )
    assert int(trace["PA"]) == int(out.answer_gt.value)
    tangent_vector = (float(tx) - float(px), float(ty) - float(py))
    radius_vector = (float(tx) - float(ox), float(ty) - float(oy))
    dot_product = (tangent_vector[0] * radius_vector[0]) + (
        tangent_vector[1] * radius_vector[1]
    )
    assert math.isclose(dot_product, 0.0, abs_tol=1e-7)


@pytest.mark.parametrize(
    ("target_kind", "target_answer", "canonical_answer_segment"),
    (
        ("outside", 24, "PA"),
        ("inside", 30, "AB"),
        ("tangent", 60, "PT"),
    ),
)
def test_tangent_secant_variant_supports_multiple_missing_segments(
    target_kind: str,
    target_answer: int,
    canonical_answer_segment: str,
) -> None:
    out = _generate_for_query(
        "tangent_secant_length",
        23435,
        params={
            "query_id": "tangent_secant_length",
            "target_answer": target_answer,
            "tangent_secant_target_kind": target_kind,
        },
        max_attempts=40,
    )
    trace = out.trace_payload["execution_trace"]

    assert trace["target_kind"] == target_kind
    assert trace["canonical_answer_segment"] == canonical_answer_segment
    assert int(out.answer_gt.value) == int(target_answer)
    assert int(trace["PT"]) * int(trace["PT"]) == int(trace["PA"]) * int(trace["PB"])
    assert int(trace["PB"]) == int(trace["PA"]) + int(trace["AB"])


def test_secant_secant_variant_places_intersections_on_circle_and_preserves_power() -> (
    None
):
    out = GeometryCircleSecantSecantLengthValueTask().generate(
        23441,
        params={"query_id": "secant_secant_length", "target_answer": 10},
        max_attempts=40,
    )
    point_model = out.trace_payload["render_map"]["point_model"]
    trace = out.trace_payload["execution_trace"]
    label_map = trace["label_map"]
    ox, oy = point_model[label_map["O"]]
    radius = float(out.trace_payload["render_map"]["circle_radius_model"])

    for label in ("A", "B", "C", "D"):
        x, y = point_model[label_map[label]]
        assert math.isclose(
            math.hypot(float(x) - float(ox), float(y) - float(oy)), radius, abs_tol=1e-7
        )
    assert int(trace["PA"]) * int(trace["PB"]) == int(trace["PC"]) * int(trace["PD"])
    assert int(trace["PA"]) == int(out.answer_gt.value)


@pytest.mark.parametrize(
    ("target_kind", "target_answer", "canonical_answer_segment"),
    (
        ("outside_first", 10, "PA"),
        ("inside_first", 24, "AB"),
        ("outside_second", 12, "PC"),
        ("inside_second", 20, "CD"),
    ),
)
def test_secant_secant_variable_variant_supports_multiple_missing_segments(
    target_kind: str,
    target_answer: int,
    canonical_answer_segment: str,
) -> None:
    out = GeometryCircleSecantSecantLengthValueTask().generate(
        23445,
        params={
            "query_id": "secant_secant_variable_segment_length",
            "target_answer": target_answer,
            "secant_secant_variable_target_kind": target_kind,
        },
        max_attempts=40,
    )
    trace = out.trace_payload["execution_trace"]

    assert trace["theorem"] == "secant_secant_variable"
    assert trace["target_kind"] == target_kind
    assert trace["canonical_answer_segment"] == canonical_answer_segment
    assert int(out.answer_gt.value) == int(target_answer)
    assert int(trace["PA"]) * int(trace["PB"]) == int(trace["PC"]) * int(trace["PD"])
    assert int(trace["PB"]) == int(trace["PA"]) + int(trace["AB"])
    assert int(trace["PD"]) == int(trace["PC"]) + int(trace["CD"])
    assert len(out.annotation_gt.value) == 5

@pytest.mark.parametrize(
    "query_id",
    (
        "tangent_secant_length",
        "secant_secant_length",
        "secant_secant_variable_segment_length",
        "external_two_secants_angle_from_arcs",
    ),
)
def test_secant_theorem_variants_sample_external_point_on_both_sides(
    query_id: str,
) -> None:
    observed_sides: set[str] = set()

    for seed in range(23480, 23490):
        out = _generate_for_query(query_id, seed, max_attempts=100)
        trace = out.trace_payload["execution_trace"]
        label_map = trace["label_map"]
        point_model = out.trace_payload["render_map"]["point_model"]
        external_x = float(point_model[label_map["P"]][0])
        center_x = float(point_model[label_map["O"]][0])
        side = str(trace["external_point_side"])

        observed_sides.add(side)
        if side == "left":
            assert external_x < center_x
        elif side == "right":
            assert external_x > center_x
        else:
            raise AssertionError(f"unsupported side: {side!r}")

    assert observed_sides == {"left", "right"}


def test_intersecting_chords_arc_variant_uses_angle_arc_relationship() -> None:
    out = _generate_for_query(
        "intersecting_chords_arc_measure",
        23451,
        params={
            "query_id": "intersecting_chords_arc_measure",
            "target_answer": 120,
        },
        max_attempts=40,
    )
    trace = out.trace_payload["execution_trace"]

    assert int(trace["angle_AEB"]) * 2 == int(trace["arc_AB"]) + int(trace["arc_CD"])
    assert int(trace["arc_CD"]) == int(out.answer_gt.value)
    assert "arc " in out.prompt
    assert "arcCD" not in out.prompt
    assert len(out.annotation_gt.value) == 5
    assert len(out.trace_payload["projected_annotation"]["pixel_point_map"]) == 5
    _assert_arc_measure_tokens_use_degrees(out)


def test_multi_step_angle_variant_uses_intersecting_chord_arc_sum() -> None:
    out = _generate_for_query(
        "multi_step_angle_value",
        23453,
        params={"query_id": "multi_step_angle_value", "target_answer": 85},
        max_attempts=40,
    )
    trace = out.trace_payload["execution_trace"]

    assert int(trace["angle_AEB"]) * 2 == int(trace["arc_AB"]) + int(trace["arc_CD"])
    assert int(trace["angle_AEB"]) == int(out.answer_gt.value)
    assert "∠" in out.prompt
    assert "angleAEB" not in out.prompt
    assert len(out.annotation_gt.value) == 5
    assert len(out.trace_payload["projected_annotation"]["pixel_point_map"]) == 5
    _assert_arc_measure_tokens_use_degrees(out)


def test_external_secant_angle_variant_uses_arc_difference() -> None:
    out = GeometryCircleExternalSecantAngleValueTask().generate(
        23454,
        params={"target_answer": 50},
        max_attempts=60,
    )
    trace = out.trace_payload["execution_trace"]

    assert trace["theorem"] == "external_secant_angle_from_arcs"
    assert int(trace["angle_BPD"]) == int(out.answer_gt.value)
    assert (
        int(trace["far_intercepted_arc_measure"])
        - int(trace["near_intercepted_arc_measure"])
    ) == 2 * int(out.answer_gt.value)
    assert "∠" in out.prompt
    assert "angleBPD" not in out.prompt
    assert "arc " in out.prompt
    assert len(out.annotation_gt.value) == 5
    assert set(out.annotation_gt.value) == set(
        out.trace_payload["witness_symbolic"]["annotation_point_labels"]
    )
    _assert_arc_measure_tokens_use_degrees(out)


def test_cyclic_quadrilateral_opposite_angle_variant_uses_supplement() -> None:
    out = GeometryCircleCyclicQuadrilateralOppositeAngleValueTask().generate(
        23458,
        params={"target_answer": 75},
        max_attempts=60,
    )
    trace = out.trace_payload["execution_trace"]

    assert trace["theorem"] == "cyclic_quadrilateral_angle"
    assert int(trace["angle_ABC"]) + int(trace["angle_CDA"]) == 180
    assert int(trace["answer_value"]) == int(out.answer_gt.value)
    assert int(out.answer_gt.value) == 75
    assert "∠" in out.prompt
    assert "angleABC" not in out.prompt
    assert "angleCDA" not in out.prompt
    assert len(out.annotation_gt.value) == 4
    assert set(out.annotation_gt.value) == set(
        out.trace_payload["witness_symbolic"]["annotation_point_labels"]
    )


def test_cyclic_quadrilateral_exterior_angle_variant_matches_opposite_angle() -> None:
    out = GeometryCircleCyclicQuadrilateralExteriorAngleValueTask().generate(
        23459,
        params={"target_answer": 75},
        max_attempts=60,
    )
    trace = out.trace_payload["execution_trace"]

    assert trace["theorem"] == "cyclic_quadrilateral_angle"
    assert int(trace["angle_ABC"]) + int(trace["angle_CDA"]) == 180
    assert int(trace["answer_value"]) == int(out.answer_gt.value)
    assert int(out.answer_gt.value) == 75
    assert str(trace["extension_point"]) in out.annotation_gt.value
    assert str(trace["answer_segment"]) in out.prompt
    assert "∠" in out.prompt
    assert "angleADE" not in out.prompt
    assert "angleEBC" not in out.prompt
    assert len(out.annotation_gt.value) == 5


@pytest.mark.parametrize(
    ("query_id", "target_answer"),
    (
        ("inscribed_angle_from_central", 35),
        ("central_angle_from_inscribed", 70),
        ("inscribed_angle_from_arc", 35),
    ),
)
def test_inscribed_angle_variants_use_half_arc_relationship(
    query_id: str, target_answer: int
) -> None:
    out = _generate_for_query(
        query_id,
        23455,
        params={"query_id": query_id, "target_answer": target_answer},
        max_attempts=40,
    )
    trace = out.trace_payload["execution_trace"]

    assert int(trace["central_angle_AOB"]) == int(trace["arc_AB"])
    assert int(trace["central_angle_AOB"]) == 2 * int(trace["inscribed_angle_ACB"])
    assert int(out.answer_gt.value) == int(target_answer)
    assert "∠" in out.prompt
    assert "angleAOB" not in out.prompt
    assert "angleACB" not in out.prompt
    expected_count = 3 if query_id == "inscribed_angle_from_arc" else 4
    assert len(out.annotation_gt.value) == expected_count
    _assert_arc_measure_tokens_use_degrees(out)


@pytest.mark.parametrize(
    "query_id",
    ("tangent_chord_angle_from_arc", "tangent_chord_angle_from_inscribed"),
)
def test_tangent_chord_angle_variants_use_matching_angle_or_arc(
    query_id: str,
) -> None:
    out = _generate_for_query(
        query_id,
        23457,
        params={"query_id": query_id, "target_answer": 45},
        max_attempts=40,
    )
    trace = out.trace_payload["execution_trace"]

    assert int(trace["arc_TA"]) == 2 * int(trace["angle_PTA"])
    assert int(trace["angle_TBA"]) == int(trace["angle_PTA"])
    assert int(trace["angle_PTA"]) == int(out.answer_gt.value)
    assert "∠" in out.prompt
    assert "anglePTA" not in out.prompt
    expected_count = 3 if query_id == "tangent_chord_angle_from_arc" else 4
    assert len(out.annotation_gt.value) == expected_count
    _assert_arc_measure_tokens_use_degrees(out)


def test_circle_theorem_rendered_label_boxes_avoid_lines_and_circle() -> None:
    variants = (
        "diameter_perpendicular_chord_length",
        "secant_secant_variable_segment_length",
        "tangent_secant_length",
        "secant_secant_length",
        "intersecting_chords_arc_measure",
        "multi_step_angle_value",
        "inscribed_angle_from_central",
        "central_angle_from_inscribed",
        "inscribed_angle_from_arc",
        "tangent_chord_angle_from_arc",
        "tangent_chord_angle_from_inscribed",
        "external_two_secants_angle_from_arcs",
        "opposite_angle_supplement",
        "exterior_angle_from_opposite_interior",
    )

    for seed in (50000, 50001):
        for query_id in variants:
            out = _generate_for_query(query_id, seed, max_attempts=100)
            render_map = out.trace_payload["render_map"]
            segments = [
                (
                    tuple(float(coord) for coord in endpoints[0]),
                    tuple(float(coord) for coord in endpoints[1]),
                )
                for endpoints in render_map["segment_pixels"].values()
            ]
            label_boxes = dict(render_map["measurement_token_bboxes"])
            label_boxes.update(
                {
                    f"point:{key}": value
                    for key, value in render_map["point_label_bboxes"].items()
                }
            )
            for label, bbox in label_boxes.items():
                assert not any(
                    _segment_crosses_bbox(a, b, bbox) for a, b in segments
                ), (query_id, seed, label, bbox)
                assert not _circle_crosses_bbox(
                    render_map["circle_center_pixel"],
                    float(render_map["circle_radius_px"]),
                    bbox,
                ), (query_id, seed, label, bbox)


def test_geometry_circle_scene_config_keeps_render_and_prompt_defaults_only() -> None:
    cfg = get_scene_defaults("geometry", "circle_theorem")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="geometry_circle_theorem_value_base",
    )

    assert generation == {}
    assert int(rendering["line_width"]) > 0
    assert str(prompt["bundle_id"]) == "geometry_circle_theorem_v1"

    for task_id in (
        "task_geometry__circle_theorem__chord_length_from_radius_central_angle_value",
        "task_geometry__circle_theorem__chord_length_from_radius_inscribed_angle_value",
        "task_geometry__circle_theorem__radius_from_external_distance_and_angle_value",
        "task_geometry__circle_theorem__tangent_length_from_radius_and_external_distance_value",
    ):
        task_generation, _, task_prompt = split_generation_rendering_prompt_defaults(
            cfg,
            task_id=task_id,
        )
        assert task_generation == {}
        assert str(task_prompt["bundle_id"]) == "geometry_circle_theorem_v1"


def test_diameter_chord_prompt_names_roles_without_readout_phrase() -> None:
    prompt_asset = Path("src/trace_tasks/resources/prompts/geometry/circle_theorem/geometry_circle_theorem_v1.json")
    bundle = json.loads(prompt_asset.read_text(encoding="utf-8"))
    templates = bundle["templates"]["query"]["diameter_perpendicular_chord_length"]
    required_slots = bundle["required_slots_by_key"][
        "query:diameter_perpendicular_chord_length"
    ]

    assert templates
    assert required_slots == [
        "diameter_segment",
        "chord_segment",
        "intersection_label",
        "answer_segment",
    ]
    for template in templates:
        normalized = str(template).lower()
        assert "measurement readout" not in normalized
        assert "diameter" in normalized and "{diameter_segment}" in template
        assert "chord" in normalized and "{chord_segment}" in template

    task = GeometryCircleDiameterPerpendicularChordLengthValueTask()
    out = task.generate(
        54001,
        params={"query_id": "single", "target_answer": 8},
        max_attempts=64,
    )
    prompt = str(out.prompt).lower()
    assert "measurement readout" not in prompt
    assert ("db is a diameter" in prompt) or ("diameter db" in prompt) or ("shown diameter db" in prompt)
    assert "chord ac" in prompt
