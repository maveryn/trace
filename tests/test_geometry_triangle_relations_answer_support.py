from __future__ import annotations

import math

import pytest

from trace_tasks.tasks.geometry.shared.measurement_rendering import fmt_measure
from trace_tasks.tasks.geometry.triangle_relations.shared import construction


MIN_UNIQUE_ANSWERS = 80
MIN_LABELED_SEGMENT_LENGTH = 52.0


CASE_BUILDER_NAMES = (
    "similar_side_cases",
    "parallel_section_cross_cases",
    "parallel_section_base_cases",
    "chained_rectangle_diagonal_cases",
    "rectangle_triangle_shared_height_cases",
    "angle_bisector_split_cases",
    "angle_bisector_base_cases",
    "centroid_vertex_cases",
    "centroid_whole_cases",
    "height_from_angle_ground_cases",
    "ground_from_angle_height_cases",
    "hypotenuse_from_angle_height_cases",
    "height_from_angle_hypotenuse_cases",
    "ground_from_angle_hypotenuse_cases",
    "angle_bisector_variable_cases",
    "split_triangle_angle_cases",
    "split_triangle_trig_side_cases",
    "altitude_from_two_projections_cases",
    "projection_from_altitude_cases",
    "leg_from_projection_cases",
    "projection_from_leg_cases",
)


@pytest.mark.parametrize("case_builder_name", CASE_BUILDER_NAMES)
def test_triangle_relations_construction_pools_have_broad_answer_support(case_builder_name: str) -> None:
    cases = getattr(construction, case_builder_name)()
    unique_answers = {str(fmt_measure(float(case.answer))) for case in cases}

    assert len(unique_answers) >= MIN_UNIQUE_ANSWERS


@pytest.mark.parametrize("case_builder_name", CASE_BUILDER_NAMES)
def test_triangle_relations_labeled_segments_are_large_enough_to_read(case_builder_name: str) -> None:
    cases = getattr(construction, case_builder_name)()

    for case in cases:
        for label in case.segment_labels:
            start = case.vertices[label.segment[0]]
            end = case.vertices[label.segment[1]]
            length = math.hypot(float(start[0]) - float(end[0]), float(start[1]) - float(end[1]))
            assert length >= MIN_LABELED_SEGMENT_LENGTH


@pytest.mark.parametrize(
    "case_builder_name",
    (
        "angle_bisector_base_cases",
        "angle_bisector_split_cases",
        "angle_bisector_variable_cases",
    ),
)
def test_triangle_relations_angle_bisector_split_points_are_not_endpoint_crowded(case_builder_name: str) -> None:
    cases = getattr(construction, case_builder_name)()

    for case in cases:
        values = dict(case.trace_values)
        bd = float(values["BD"])
        dc_raw = values.get("DC", values.get("derived_DC"))
        if dc_raw is None:
            dc_raw = int(values["x"]) + int(str(values["DC_expression"]).split("+")[1])
        dc = float(dc_raw)
        assert min(bd, dc) / (bd + dc) >= 0.2
