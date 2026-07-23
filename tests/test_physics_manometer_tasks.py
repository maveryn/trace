"""Contract tests for the migrated physics manometer scene."""

from __future__ import annotations

from trace_tasks.tasks.physics.manometer.pressure_difference_value import PhysicsManometerPressureDifferenceValueTask


def _assert_bbox_map_in_bounds(out) -> None:
    width, height = out.image.size
    for bbox in out.annotation_gt.value.values():
        x0, y0, x1, y1 = [float(value) for value in bbox]
        assert 0 <= x0 < x1 <= width
        assert 0 <= y0 < y1 <= height


def test_physics_manometer_pressure_difference_contract() -> None:
    out = PhysicsManometerPressureDifferenceValueTask().generate(
        80277,
        params={
            "height_cm": 6,
            "kpa_per_cm": 3,
            "higher_pressure_side": "A",
        },
        max_attempts=20,
    )
    execution = out.trace_payload["execution_trace"]
    render_map = out.trace_payload["render_map"]

    assert out.scene_id == "manometer"
    assert out.query_id == "single"
    assert out.trace_payload["query_spec"]["query_id"] == "single"
    assert out.trace_payload["query_spec"]["params"]["internal_query_id"] == "u_tube_pressure_difference"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 18
    assert out.answer_gt.value == int(execution["height_cm"]) * int(execution["kpa_per_cm"])
    assert render_map["pressure_difference_kpa"] == 18
    assert render_map["higher_pressure_side"] == "A"
    assert render_map["left_level_y_px"] > render_map["right_level_y_px"]
    assert set(out.annotation_gt.value) == {"height_difference", "fluid_density_label"}
    assert render_map["annotation_bbox_map_px"] == out.annotation_gt.value
    _assert_bbox_map_in_bounds(out)
    assert out.trace_payload["projected_annotation"]["bbox_map"] == out.annotation_gt.value


def test_physics_manometer_pressure_side_does_not_change_absolute_answer() -> None:
    task = PhysicsManometerPressureDifferenceValueTask()
    left_high = task.generate(80281, params={"height_cm": 5, "kpa_per_cm": 4, "higher_pressure_side": "A"}, max_attempts=20)
    right_high = task.generate(80281, params={"height_cm": 5, "kpa_per_cm": 4, "higher_pressure_side": "B"}, max_attempts=20)

    assert left_high.answer_gt.value == 20
    assert right_high.answer_gt.value == 20
    assert left_high.trace_payload["render_map"]["left_level_y_px"] > left_high.trace_payload["render_map"]["right_level_y_px"]
    assert right_high.trace_payload["render_map"]["right_level_y_px"] > right_high.trace_payload["render_map"]["left_level_y_px"]
