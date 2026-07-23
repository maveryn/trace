"""Contracts for direct solid-formula geometry tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.geometry.solid_formula.cylinder_cone_height_from_volume_radius import (
    SCENE_ID,
    GeometrySolidFormulaCylinderConeHeightFromVolumeRadiusTask,
)
from trace_tasks.tasks.geometry.solid_formula.cylinder_cone_radius_from_volume_heights import (
    GeometrySolidFormulaCylinderConeRadiusFromVolumeHeightsTask,
)
from trace_tasks.tasks.geometry.solid_formula.house_prism_length_from_volume import (
    GeometrySolidFormulaHousePrismLengthFromVolumeTask,
)
from trace_tasks.tasks.geometry.solid_formula.prism_pyramid_height_from_volume import (
    GeometrySolidFormulaPrismPyramidHeightFromVolumeTask,
)

TASK_CLASSES = (
    GeometrySolidFormulaCylinderConeRadiusFromVolumeHeightsTask,
    GeometrySolidFormulaCylinderConeHeightFromVolumeRadiusTask,
    GeometrySolidFormulaPrismPyramidHeightFromVolumeTask,
    GeometrySolidFormulaHousePrismLengthFromVolumeTask,
)

@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_solid_formula_tasks_emit_public_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(64001, params={}, max_attempts=20)

    assert out.scene_id == SCENE_ID
    assert out.query_id == "single"
    assert out.answer_gt.type == "number"
    assert out.annotation_gt.type == "bbox"
    assert isinstance(out.annotation_gt.value, list)
    assert len(out.annotation_gt.value) == 4
    assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    assert trace["query_spec"]["scene_id"] == SCENE_ID
    assert trace["query_spec"]["query_id"] == "single"
    assert trace["execution_trace"]["query_id"] == "single"
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["execution_trace"]["answer_rounding"] == "one_decimal"
    assert trace["query_spec"]["prompt_variant"]["prompt_schema_version"] == "v1"
    assert trace["execution_trace"]["answer_support_size"] >= 50
    assert out.annotation_gt.value == trace["render_map"]["solid"]["bbox"]
    assert out.annotation_gt.value == trace["projected_annotation"]["bbox"]
    assert "volume_label" in trace["render_map"]["label_bboxes"]
    assert trace["witness_symbolic"]["source_witness_type"] == "bbox"
    assert trace["witness_symbolic"]["original_annotation_value"] == out.annotation_gt.value
    assert trace["execution_trace"]["annotation_roles"] == ["solid"]
    assert "dimension-line segments" not in out.prompt_variants["answer_and_annotation"]
    assert "compound solid shape" in out.prompt_variants["answer_and_annotation"]


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_solid_formula_tasks_are_deterministic(task_cls) -> None:
    task = task_cls()
    params = {}
    out_a = task.generate(64011, params=params, max_attempts=20)
    out_b = task.generate(64011, params=params, max_attempts=20)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert out_a.image.tobytes() == out_b.image.tobytes()


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_solid_formula_tasks_support_single_query(task_cls) -> None:
    task = task_cls()
    out = task.generate(64021, params={"query_id": "single"}, max_attempts=20)
    trace = out.trace_payload["execution_trace"]
    assert out.query_id == "single"
    assert out.answer_gt.type == "number"
    assert out.trace_payload["query_spec"]["params"]["query_id_probabilities"] == {
        "single": 1.0
    }

    if task_cls is GeometrySolidFormulaCylinderConeRadiusFromVolumeHeightsTask:
        radius = float(trace["radius"])
        total_height = float(trace["total_height"])
        cone_height = float(trace["cone_height"])
        cylinder_height = float(trace["cylinder_height"])
        assert total_height == pytest.approx(cylinder_height + cone_height)
        assert trace["volume_pi_multiple"] == pytest.approx(
            radius**2 * (cylinder_height + (cone_height / 3.0))
        )
        assert out.answer_gt.value == pytest.approx(radius)
    elif task_cls is GeometrySolidFormulaCylinderConeHeightFromVolumeRadiusTask:
        radius = float(trace["radius"])
        cylinder_height = float(trace["cylinder_height"])
        cone_height = float(trace["cone_height"])
        assert trace["volume_pi_multiple"] == pytest.approx(
            radius**2 * (cylinder_height + (cone_height / 3.0))
        )
        assert out.answer_gt.value == pytest.approx(cylinder_height)
    elif task_cls is GeometrySolidFormulaPrismPyramidHeightFromVolumeTask:
        side_a = float(trace["side_a"])
        side_b = float(trace["side_b"])
        prism_height = float(trace["prism_height"])
        pyramid_height = float(trace["pyramid_height"])
        assert trace["volume"] == pytest.approx(
            side_a * side_b * (prism_height + (pyramid_height / 3.0))
        )
        assert out.answer_gt.value == pytest.approx(prism_height)
    elif task_cls is GeometrySolidFormulaHousePrismLengthFromVolumeTask:
        triangle_base = float(trace["triangle_base"])
        wall_height = float(trace["wall_height"])
        roof_height = float(trace["roof_height"])
        prism_length = float(trace["prism_length"])
        assert trace["volume"] == pytest.approx(
            ((triangle_base * wall_height) + (0.5 * triangle_base * roof_height))
            * prism_length
        )
        assert out.answer_gt.value == pytest.approx(prism_length)


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_solid_formula_annotation_stays_inside_canvas(task_cls) -> None:
    task = task_cls()
    out = task.generate(64041, params={"query_id": "single"}, max_attempts=20)
    width, height = out.image.size
    x0, y0, x1, y1 = out.annotation_gt.value
    assert 0.0 <= x0 < x1 <= float(width)
    assert 0.0 <= y0 < y1 <= float(height)
    assert (x1 - x0) > 80.0
    assert (y1 - y0) > 80.0


def test_solid_formula_measurement_labels_avoid_known_geometry_overlaps() -> None:
    radius_out = GeometrySolidFormulaCylinderConeRadiusFromVolumeHeightsTask().generate(
        64051,
        params={"query_id": "single"},
        max_attempts=20,
    )
    height_out = GeometrySolidFormulaCylinderConeHeightFromVolumeRadiusTask().generate(
        64052,
        params={"query_id": "single"},
        max_attempts=20,
    )
    house_out = GeometrySolidFormulaHousePrismLengthFromVolumeTask().generate(
        64053,
        params={"query_id": "single"},
        max_attempts=20,
    )

    target_radius_bbox = radius_out.trace_payload["render_map"]["label_bboxes"]["target_radius_label"]
    radius_bbox = height_out.trace_payload["render_map"]["label_bboxes"]["radius_label"]
    roof_height_bbox = house_out.trace_payload["render_map"]["label_bboxes"]["roof_height_label"]

    assert target_radius_bbox[1] > 260.0
    assert radius_bbox[1] > 260.0
    assert roof_height_bbox[0] > 438.0


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_solid_formula_numeric_labels_do_not_use_heavy_stroke(task_cls) -> None:
    default_out = task_cls().generate(
        64060,
        params={"query_id": "single"},
        max_attempts=20,
    )
    out = task_cls().generate(
        64061,
        params={"query_id": "single", "label_stroke_width": 8},
        max_attempts=20,
    )
    assert default_out.trace_payload["render_spec"]["style"]["label_stroke_width"] <= 1
    assert out.trace_payload["render_spec"]["style"]["label_stroke_width"] <= 1


def test_solid_formula_dark_treatment_uses_readable_measurement_label_ink() -> None:
    out = GeometrySolidFormulaHousePrismLengthFromVolumeTask().generate(
        930965330361450,
        params={"query_id": "single"},
        max_attempts=20,
    )
    palette = out.trace_payload["render_spec"]["style"]["palette"]
    assert palette["measurement_label_color_policy"] == "neutral_high_contrast"
    assert palette["measurement_label_min_surface_contrast"] >= 7.0
    assert min(palette["measurement_label_color"]) >= 240
    assert out.trace_payload["render_spec"]["style"]["label_stroke_width"] == 1

    records = out.trace_payload["render_spec"]["drawn_text"]["text_legibility"]["records"]
    measurement_records = [
        record
        for record in records
        if record.get("text") in {"b=10", "h=4", "t=6", "L=?"}
    ]
    assert len(measurement_records) == 4
    for record in measurement_records:
        assert record["fill_rgb"] == palette["measurement_label_color"]
        assert record["stroke_width_px"] == 1


def test_solid_formula_tasks_reject_unknown_query_id() -> None:
    task = GeometrySolidFormulaCylinderConeRadiusFromVolumeHeightsTask()
    with pytest.raises(ValueError):
        task.generate(64031, params={"query_id": "not_a_query"}, max_attempts=20)
